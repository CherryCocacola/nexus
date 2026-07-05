"""
자기 성찰(Self-Reflection) 엔진 — 3-pass 사고 처리기.

한 번의 LLM 호출로 답을 끝내지 않고, 다음 3단계(pass)로 나눠서
스스로 점검하며 답변 품질을 끌어올리는 고급 사고 엔진이다.
같은 thinking 패키지의 HiddenCoTEngine(2-pass)에 "검증(Pass 3)" 단계를
하나 더 붙인 상위 버전이라고 이해하면 된다.

3단계 흐름:
  - Pass 1 (심층 분석): 질문을 구조적으로 분해한다. 핵심 요구사항,
    숨은 가정, 접근 방법 비교, 엣지 케이스 등을 뽑아낸다.
  - Pass 2 (초기 응답): Pass 1의 분석 결과를 근거로 실제 답변을 작성한다.
  - Pass 3 (검증/수정): 초기 응답을 리뷰어 관점에서 다시 검증하고,
    논리 오류·누락·불일관성이 있으면 고쳐 최종 응답을 만든다.

왜 3-pass로 나누는가:
  복잡한 문제에서는 모델이 단번에 완벽한 답을 내기 어렵다. 답을 만든 뒤
  "따로" 검증 단계를 거치게 하면, 처음엔 놓친 논리 오류·누락·모순을
  한 번 더 걸러낼 수 있어 최종 품질이 올라간다. 대가로 LLM 호출이 3번이라
  지연 시간이 길어지므로, 복잡도가 높은 질문에만 선택적으로 쓴다.

주요 구성:
  - SelfReflectionEngine: 3-pass 전체를 실행하는 엔진 클래스.
  - DEEP_ANALYSIS_PROMPT / INITIAL_RESPONSE_PROMPT / VERIFICATION_PROMPT:
    각 pass에서 모델에 주입하는 system 프롬프트 상수.

의존/연동:
  - core.message.Message / StreamEventType 를 사용해 대화 메시지를 만들고
    모델 스트림 이벤트를 해석한다.
  - 반환 타입 ThinkingResult 와 ThinkingStrategy 는 실행 시점에 지연 import
    한다(순환 의존 방지). 실제 선택/호출은 상위 orchestrator가 담당한다.

작성자: 이현수 / 작성일: 2026-07-05
"""

# 파이썬 3.11 미만 호환을 위한 지연 어노테이션 평가.
# (반환 타입 힌트에 아직 import 안 된 클래스명을 문자열로 쓸 수 있게 해준다.)
from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from core.message import Message, StreamEventType

# TYPE_CHECKING 블록: 타입 검사기(mypy 등)에서만 보이는 import.
# 런타임에는 실행되지 않아 orchestrator ↔ self_reflection 순환 import를 피한다.
if TYPE_CHECKING:
    from core.thinking.orchestrator import ThinkingResult

# 모듈 전용 로거. 로그 필터링/설정은 "nexus.thinking.self_reflection" 이름으로 제어.
logger = logging.getLogger("nexus.thinking.self_reflection")

# ─── Pass 1: 심층 분석 프롬프트 ───
# 질문을 바로 풀지 말고, 먼저 구조적으로 "분해"만 하라고 지시하는 system 프롬프트.
# 요구사항/숨은 가정/접근 비교/엣지 케이스/제약을 뽑아내게 유도한다.
DEEP_ANALYSIS_PROMPT = (
    "당신은 고급 문제 분석 전문가입니다. "
    "다음 질문을 심층적으로 분석하세요.\n\n"
    "분석 항목:\n"
    "1. 문제의 핵심 요구사항을 구체적으로 나열\n"
    "2. 숨겨진 가정이나 암묵적 요구사항 식별\n"
    "3. 가능한 접근 방법 비교 (장단점)\n"
    "4. 엣지 케이스와 예외 상황\n"
    "5. 제약 조건과 우선순위\n\n"
    "구조화된 분석 결과만 출력하세요."
)

# ─── Pass 2: 응답 생성 프롬프트 ───
# Pass 1 분석을 근거로 "실제 답변"을 쓰게 하는 프롬프트.
# 사용자에게는 분석 과정을 노출하지 않고 최종 답변만 보이게 한다.
INITIAL_RESPONSE_PROMPT = (
    "이전 심층 분석 결과를 바탕으로 사용자의 질문에 완전하고 정확하게 답변하세요. "
    "분석 과정은 언급하지 말고, 최종 답변만 작성하세요."
)

# ─── Pass 3: 검증 + 수정 프롬프트 ───
# 초기 응답을 코드 리뷰어/검증자 관점에서 재점검하게 하는 프롬프트.
# 문제가 있으면 수정본을, 없으면 원본을 그대로 반환하도록 지시한다.
VERIFICATION_PROMPT = (
    "당신은 코드 리뷰어이자 검증 전문가입니다. "
    "다음 응답을 검증하고 필요하면 수정하세요.\n\n"
    "검증 항목:\n"
    "1. 논리적 오류가 있는가?\n"
    "2. 누락된 내용이 있는가?\n"
    "3. 원본 질문의 모든 요구사항에 답했는가?\n"
    "4. 코드가 포함된 경우 — 문법 오류, 보안 취약점, 성능 문제가 있는가?\n"
    "5. 설명이 명확하고 일관성이 있는가?\n\n"
    "문제가 있으면 수정된 최종 응답을 작성하세요. "
    "문제가 없으면 원본 응답을 그대로 반환하세요."
)


class SelfReflectionEngine:
    """
    3-pass 자기 성찰 엔진.

    이 클래스는 상태를 들고 있지 않다(인스턴스 필드 없음). run()에 필요한
    입력(질문·컨텍스트·모델)을 매번 넘겨받아 3단계를 순서대로 돌리고
    ThinkingResult 하나를 돌려주는 "실행기" 역할만 한다. 그래서 보조
    메서드들은 대부분 staticmethod로 두었다.

    동작 흐름:
      Pass 1: 심층 분석 → 구조화된 문제 분해
      Pass 2: 분석 기반 초기 응답 생성
      Pass 3: 초기 응답 검증 + 수정 → 최종 응답

    HIDDEN_COT(2-pass)와의 차이:
      - Pass 3(검증)이 추가되어 응답 품질이 더 높다
      - 대신 LLM 호출이 3회이므로 지연 시간이 더 길다
      - 복잡도 스코어 0.6~0.8 구간에서 사용한다(선택은 orchestrator가 함)
    """

    async def run(
        self,
        message: str,
        context: list[Message] | None,
        model: ModelProvider,  # noqa: F821
    ) -> ThinkingResult:
        """
        3-pass 자기 성찰을 실행한다.

        이 메서드가 엔진의 진입점이다. Pass 1 → 2 → 3 을 순차로 await 하며,
        각 pass는 내부 헬퍼(_build_*_messages + _collect_text)로 처리한다.
        pass마다 temperature를 다르게 준다: 분석은 낮게(정확/재현성),
        응답은 보통, 검증은 가장 낮게(보수적으로 흔들림 최소화).

        Args:
            message: 사용자 입력 텍스트(원본 질문)
            context: 이전 대화 컨텍스트(선택). 있으면 각 pass 맨 앞에 붙인다.
            model: LLM 프로바이더. stream()으로 토큰을 흘려보내는 객체.

        Returns:
            ThinkingResult — 최종(검증된) 응답 + 전체 사고 과정 + 메타데이터.
            score는 여기서 0.0으로 두고, 실제 값은 호출자(orchestrator)가 채운다.
        """
        # 지연 import — orchestrator/strategy 를 런타임 시점에만 불러와
        # 모듈 로드 단계의 순환 의존을 피한다.
        from core.thinking.orchestrator import ThinkingResult
        from core.thinking.strategy import ThinkingStrategy

        # 전체 소요 시간 측정 시작. monotonic()은 시스템 시계 변경에 영향받지
        # 않는 단조 증가 시계라 경과 시간 측정에 적합하다.
        start_time = time.monotonic()

        # ── Pass 1: 심층 분석 ──
        # 질문을 곧바로 풀지 않고, 먼저 구조적으로 분해만 시킨다.
        logger.debug("Self-Reflect Pass 1: 심층 분석 시작")
        analysis_messages = self._build_analysis_messages(message, context)
        analysis_text = await self._collect_text(
            model=model,
            messages=analysis_messages,
            system_prompt=DEEP_ANALYSIS_PROMPT,
            temperature=0.3,  # 분석은 최대한 정확하게
        )
        logger.debug(f"Self-Reflect Pass 1 완료: {len(analysis_text)}자")

        # ── Pass 2: 초기 응답 생성 ──
        # Pass 1 분석 결과를 근거로 첫 답변을 만든다. 이 답은 아직 "초안"이며
        # 곧바로 Pass 3에서 다시 검증받는다.
        logger.debug("Self-Reflect Pass 2: 초기 응답 생성 시작")
        response_messages = self._build_response_messages(message, context, analysis_text)
        initial_response = await self._collect_text(
            model=model,
            messages=response_messages,
            system_prompt=INITIAL_RESPONSE_PROMPT,
            temperature=0.7,
        )
        logger.debug(f"Self-Reflect Pass 2 완료: {len(initial_response)}자")

        # ── Pass 3: 검증 + 수정 ──
        # 초기 응답을 리뷰어 관점으로 재검증한다. 문제가 있으면 고친 최종본을,
        # 없으면 사실상 같은 내용을 돌려준다.
        logger.debug("Self-Reflect Pass 3: 검증 시작")
        verification_messages = self._build_verification_messages(
            message, context, initial_response
        )
        final_response = await self._collect_text(
            model=model,
            messages=verification_messages,
            system_prompt=VERIFICATION_PROMPT,
            temperature=0.2,  # 검증은 최대한 보수적으로
        )
        logger.debug(f"Self-Reflect Pass 3 완료: {len(final_response)}자")

        # 3개 pass 전체에 걸린 시간(초).
        elapsed = time.monotonic() - start_time

        # 전체 사고 과정을 사람이 읽을 수 있는 형태로 thinking_text에 모은다.
        # 디버깅/감사(어떤 판단을 거쳐 최종답이 나왔는지 추적)에 쓰인다.
        # 마지막 줄은 검증이 실제로 답을 바꿨는지(동일/다름)를 표시한다.
        thinking_text = (
            f"=== Pass 1: 심층 분석 ===\n{analysis_text}\n\n"
            f"=== Pass 2: 초기 응답 ===\n{initial_response}\n\n"
            f"=== Pass 3: 검증 결과 ===\n"
            f"(최종 응답이 초기 응답과 {'동일' if final_response == initial_response else '다름'})"
        )

        # 결과 객체 조립. strategy로 어떤 엔진이 처리했는지 표시하고,
        # passes=3 은 이 엔진이 항상 3-pass임을 뜻한다.
        return ThinkingResult(
            strategy=ThinkingStrategy.SELF_REFLECT,
            response=final_response,
            thinking_text=thinking_text,
            passes=3,
            elapsed_seconds=round(elapsed, 3),
            score=0.0,  # 호출자(Orchestrator)가 스코어를 설정
        )

    @staticmethod
    def _build_analysis_messages(
        message: str,
        context: list[Message] | None,
    ) -> list[Message]:
        """
        Pass 1용 메시지 목록을 만든다: (이전 컨텍스트) + 심층 분석 요청.

        이전 대화가 있으면 먼저 이어 붙여 맥락을 유지하고, 마지막에
        "이 질문을 분석해 달라"는 user 메시지를 추가한다.
        실제 '분석만 하라'는 지시는 DEEP_ANALYSIS_PROMPT(system)가 담당한다.
        """
        messages: list[Message] = []
        if context:
            messages.extend(context)
        messages.append(Message.user(f"다음 질문을 심층적으로 분석해 주세요:\n\n{message}"))
        return messages

    @staticmethod
    def _build_response_messages(
        message: str,
        context: list[Message] | None,
        analysis: str,
    ) -> list[Message]:
        """
        Pass 2용 메시지 목록을 만든다: (이전 컨텍스트) + 분석 결과 + 원본 질문.

        Pass 1의 분석 결과(analysis)를 assistant의 'thinking 블록'으로 끼워 넣어
        모델이 자신의 사고 흐름을 이어받게 한다. 그 뒤에 원본 질문을 다시 주어
        "이 사고를 바탕으로 실제 답을 써라"는 구도를 만든다.
        """
        messages: list[Message] = []
        if context:
            messages.extend(context)
        # 분석 결과를 thinking 블록으로 주입
        messages.append(Message.assistant(thinking=analysis))
        messages.append(Message.user(message))
        return messages

    @staticmethod
    def _build_verification_messages(
        message: str,
        context: list[Message] | None,
        initial_response: str,
    ) -> list[Message]:
        """
        Pass 3용 메시지 목록을 만든다: (이전 컨텍스트) + [원본 질문 + 초기 응답].

        검증자에게 '원본 질문'과 '초기 응답'을 한 메시지에 함께 넣어 준다.
        둘을 나란히 봐야 "요구사항을 빠뜨렸는지", "질문과 답이 어긋나는지"를
        판단할 수 있기 때문이다. 검증/수정 지침 자체는 VERIFICATION_PROMPT가 맡는다.
        """
        messages: list[Message] = []
        if context:
            messages.extend(context)
        # 원본 질문과 초기 응답을 함께 전달
        messages.append(
            Message.user(
                f"원본 질문:\n{message}\n\n"
                f"초기 응답:\n{initial_response}\n\n"
                f"위 응답을 검증하고, 문제가 있으면 수정된 최종 응답을 작성하세요."
            )
        )
        return messages

    @staticmethod
    async def _collect_text(
        model: ModelProvider,  # noqa: F821
        messages: list[Message],
        system_prompt: str,
        temperature: float,
    ) -> str:
        """
        모델 스트림을 소비해 하나의 완성 텍스트로 합쳐 반환한다.

        3개 pass가 공통으로 쓰는 헬퍼다. model.stream()이 흘려보내는
        StreamEvent들을 순회하면서:
          - TEXT_DELTA(부분 토큰)면 조각을 모으고,
          - ERROR면 경고 로그를 남기고 즉시 중단한다(부분 텍스트라도 반환).
        TEXT_DELTA/ERROR 외의 이벤트 타입은 이 단계에선 관심 밖이라 무시한다.

        Args:
            model: 토큰을 스트리밍하는 LLM 프로바이더.
            messages: _build_*_messages 가 만든 대화 메시지 목록.
            system_prompt: 해당 pass의 역할을 지정하는 system 프롬프트.
            temperature: 샘플링 온도(낮을수록 결정적/보수적).

        Returns:
            수집한 델타들을 이어 붙인 최종 문자열. 아무것도 못 받으면 빈 문자열.
        """
        # 델타 조각을 리스트에 모았다가 마지막에 한 번에 join 한다.
        # (문자열 += 반복보다 리스트+join 이 효율적)
        collected: list[str] = []
        async for event in model.stream(
            messages=messages,
            system_prompt=system_prompt,
            temperature=temperature,
        ):
            # 정상 토큰 조각: 비어 있지 않을 때만 누적.
            if event.type == StreamEventType.TEXT_DELTA and event.text:
                collected.append(event.text)
            # 스트림 에러: 조용히 삼키지 않고 로그를 남긴 뒤 루프 종료.
            elif event.type == StreamEventType.ERROR:
                logger.warning(f"Self-Reflect 스트림 에러: {event.message}")
                break
        return "".join(collected)
