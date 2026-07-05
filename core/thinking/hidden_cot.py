"""
Hidden Chain-of-Thought(숨은 사고 사슬) 엔진 — 2-pass 사고 처리 모듈.

[이 파일이 하는 일]
사용자 질문에 곧바로 답하지 않고, 모델을 두 번 호출(2-pass)해서
"먼저 생각(분석)한 뒤 답하도록" 강제하는 사고 전략을 구현한다.
    - Pass 1: 분석 전용 프롬프트로 내부 분석 텍스트를 생성한다.
              이 분석은 사용자에게 직접 노출되지 않는 "숨은 사고"다.
    - Pass 2: Pass 1의 분석 결과와 원본 질문을 함께 넣어 최종 응답을 만든다.

[왜 이렇게 하나]
사람이 어려운 문제를 풀 때 초안을 먼저 정리하고 답을 쓰듯이,
모델도 분석 단계를 분리하면 요구사항 누락·엣지 케이스 실수가 줄고
최종 응답의 품질이 올라간다. 분석과 답변을 물리적으로 나눠서
답변에 분석 과정("제가 분석해 보니...")이 새어 나오지 않게 한다.

[결과 전달]
사용자는 Pass 2의 최종 응답만 본다. 내부 분석 과정은 버려지지 않고
ThinkingResult.thinking_text 에 보관되어, 나중에 디버깅·품질 평가·
학습 데이터 수집 등에 활용할 수 있다.

[주요 구성]
    - HiddenCoTEngine: 2-pass 실행을 담당하는 엔진 클래스.
      run()이 진입점이고, 나머지는 메시지 조립/텍스트 수집용 헬퍼다.

[의존 관계]
    - core.message: Message(대화 메시지), StreamEventType(스트림 이벤트 종류)
    - core.thinking.orchestrator: ThinkingResult(반환 타입) — 순환 의존을
      피하려고 함수 안에서 지연 import 한다.
    - core.thinking.strategy: ThinkingStrategy(전략 식별용 enum)
    - model 인자로 받는 ModelProvider: 실제 LLM 추론을 수행하는 ABC.

작성자: 이현수 / 작성일: 2026-07-05
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from core.message import Message, StreamEventType

if TYPE_CHECKING:
    from core.thinking.orchestrator import ThinkingResult

logger = logging.getLogger("nexus.thinking.hidden_cot")

# ─── 분석 단계(Pass 1) 시스템 프롬프트 ───
# Pass 1에서 모델에게 "답을 쓰지 말고 분석만 하라"고 지시하는 시스템 프롬프트.
# 마지막 문장에서 "최종 응답은 아직 작성하지 마세요"라고 못 박아, 이 단계에서
# 모델이 성급히 답변을 내놓지 않고 요구사항·엣지케이스·접근법·제약을
# 정리하는 데만 집중하도록 유도한다.
ANALYSIS_SYSTEM_PROMPT = (
    "당신은 문제 분석 전문가입니다. "
    "사용자의 질문을 깊이 분석하세요.\n\n"
    "분석할 항목:\n"
    "1. 핵심 요구사항 파악\n"
    "2. 잠재적 문제점이나 엣지 케이스\n"
    "3. 최적의 접근 방법\n"
    "4. 주의해야 할 제약 조건\n\n"
    "분석 결과만 출력하세요. 최종 응답은 아직 작성하지 마세요."
)

# ─── 응답 단계(Pass 2) 시스템 프롬프트 ───
# Pass 2에서 "앞서 만든 분석을 참고해 실제 답을 쓰되, 분석 과정은 드러내지
# 말라"고 지시하는 시스템 프롬프트. 사용자에게는 깔끔한 최종 답변만 보이고
# "제가 분석한 결과에 따르면..." 같은 메타 설명이 새어 나오지 않게 한다.
RESPONSE_SYSTEM_PROMPT = (
    "이전 분석 결과를 바탕으로 사용자의 질문에 정확하고 완전하게 답변하세요. "
    "분석 과정은 언급하지 말고, 최종 답변만 작성하세요."
)


class HiddenCoTEngine:
    """
    2-pass Hidden Chain-of-Thought(숨은 사고) 엔진.

    [역할]
    한 번의 사용자 질문을 받아 모델을 두 번 호출한다. 첫 호출로 "분석"을,
    두 번째 호출로 그 분석을 근거로 한 "최종 답변"을 만들어 낸다.
    외부에서는 run() 코루틴 하나만 호출하면 되고, 내부 단계는
    private 헬퍼 메서드들이 나눠 처리한다.

    [동작 흐름]
      Pass 1: 분석 프롬프트 → 모델이 내부 분석 텍스트 생성 (사용자 비노출)
      Pass 2: 분석 + 원본 질문 → 모델이 사용자에게 보여줄 최종 응답 생성

    [왜 2-pass로 나누나]
      단일 pass로 "먼저 분석하고 그다음 답하라"고 한 번에 시키면,
      모델이 분석을 대충 건너뛰거나(추론 생략) 분석과 답변을 뒤섞어
      출력하는 경향이 있다. 호출을 물리적으로 둘로 나누면
        1) 분석 단계에 온전히 집중해 분석 품질이 올라가고,
        2) 최종 응답에는 분석 과정이 섞이지 않아 사용자 경험이 깔끔해진다.
      대가로 모델 호출이 2회라 지연시간과 토큰 비용이 늘어난다.

    [상태 없음]
      이 엔진은 인스턴스 필드에 상태를 저장하지 않는다. 모든 입력은
      run()의 인자로 들어오므로, 하나의 인스턴스를 여러 요청에서
      안전하게 재사용할 수 있다.
    """

    async def run(
        self,
        message: str,
        context: list[Message] | None,
        model: ModelProvider,  # noqa: F821 — 런타임에 import 가능
    ) -> ThinkingResult:
        """
        2-pass Hidden CoT를 실행하는 진입점(코루틴).

        전체 순서는 다음과 같다.
          1) Pass 1 메시지를 조립하고 모델을 호출해 분석 텍스트를 모은다.
          2) Pass 1 분석을 끼워 넣은 Pass 2 메시지를 조립하고 다시 모델을
             호출해 최종 응답 텍스트를 모은다.
          3) 두 결과와 소요 시간·pass 수 등을 ThinkingResult로 묶어 돌려준다.

        Args:
            message: 사용자 입력 텍스트(원본 질문).
            context: 이전 대화 컨텍스트 메시지 목록. 없으면 None.
                     있으면 두 pass 모두에 앞부분으로 끼워 맥락을 유지한다.
            model: 실제 LLM 추론을 수행하는 프로바이더(ModelProvider ABC).
                   stream()으로 StreamEvent를 비동기 산출해야 한다.

        Returns:
            ThinkingResult — 최종 응답(response), 내부 분석(thinking_text),
            pass 수, 소요 시간(초) 등을 담은 결과 객체. score는 여기서
            0.0으로 두고, 상위 호출자(Orchestrator)가 채운다.
        """
        # 순환 의존(circular import) 회피를 위한 지연 import.
        # orchestrator/strategy 모듈이 이 파일을 (간접적으로) 참조할 수 있어,
        # 모듈 최상단이 아니라 실제 사용 시점에 불러온다.
        from core.thinking.orchestrator import ThinkingResult
        from core.thinking.strategy import ThinkingStrategy

        # 전체 2-pass 소요 시간 측정 시작.
        # time.time()이 아니라 monotonic()을 쓰는 이유: 시스템 시계가
        # 도중에 조정돼도 뒤로 흐르지 않아, 경과 시간 측정에 안전하다.
        start_time = time.monotonic()

        # ── Pass 1: 내부 분석 생성 ──
        logger.debug("Hidden CoT Pass 1: 분석 시작")
        # 분석용 메시지(컨텍스트 + "분석해 주세요" 요청)를 조립한다.
        analysis_messages = self._build_analysis_messages(message, context)

        # 분석 프롬프트로 모델을 호출하고 스트림에서 텍스트를 모두 모은다.
        analysis_text = await self._collect_text(
            model=model,
            messages=analysis_messages,
            system_prompt=ANALYSIS_SYSTEM_PROMPT,
            # temperature 0.4 — 분석은 창의성보다 정확도·일관성이 중요하므로
            # 낮게 잡아 표현이 덜 튀고 안정적인 결과를 얻는다.
            temperature=0.4,
        )
        logger.debug(f"Hidden CoT Pass 1 완료: {len(analysis_text)}자")

        # ── Pass 2: 분석 기반 최종 응답 생성 ──
        logger.debug("Hidden CoT Pass 2: 응답 생성 시작")
        # Pass 1 분석 결과를 assistant 사고로 끼워 넣고 원본 질문을 다시
        # 붙인 응답용 메시지를 조립한다.
        response_messages = self._build_response_messages(message, context, analysis_text)

        # 응답 프롬프트로 모델을 다시 호출해 사용자에게 보여줄 텍스트를 모은다.
        response_text = await self._collect_text(
            model=model,
            messages=response_messages,
            system_prompt=RESPONSE_SYSTEM_PROMPT,
            # temperature 0.7 — 최종 응답은 사람이 읽기에 자연스럽고 매끄러운
            # 표현이 중요하므로, 분석보다 높여 문장의 다양성을 확보한다.
            temperature=0.7,
        )
        logger.debug(f"Hidden CoT Pass 2 완료: {len(response_text)}자")

        # 두 pass 전체에 걸린 시간(초)을 계산한다.
        elapsed = time.monotonic() - start_time

        # 결과를 불변 결과 객체로 묶어 반환한다.
        return ThinkingResult(
            strategy=ThinkingStrategy.HIDDEN_COT,
            response=response_text,  # 사용자에게 보여줄 최종 답변
            thinking_text=analysis_text,  # 숨은 분석(디버깅·평가용 보관)
            passes=2,  # 이 전략은 항상 모델을 2회 호출
            elapsed_seconds=round(elapsed, 3),  # 소수점 3자리로 반올림
            # score는 이 엔진이 스스로 매기지 않는다. 여러 전략 결과를
            # 비교·선택하는 상위 호출자(Orchestrator)가 나중에 채운다.
            score=0.0,
        )

    @staticmethod
    def _build_analysis_messages(
        message: str,
        context: list[Message] | None,
    ) -> list[Message]:
        """
        Pass 1(분석 단계)에 넣을 메시지 목록을 구성한다.

        이전 대화 컨텍스트가 있으면 앞부분에 그대로 붙여, 모델이 지금까지의
        흐름을 알고 분석하도록 한다. 그 뒤에 "이 질문을 분석해 달라"는
        사용자 메시지를 하나 덧붙인다. 실제 "답하지 말고 분석만 하라"는
        지시는 ANALYSIS_SYSTEM_PROMPT(시스템 프롬프트) 쪽이 담당한다.

        Args:
            message: 원본 사용자 질문.
            context: 이전 대화 메시지 목록(없으면 None).

        Returns:
            분석 단계에 그대로 넣을 Message 목록.
        """
        messages: list[Message] = []
        if context:
            # 이전 대화가 있으면 맥락 유지를 위해 앞에 그대로 이어 붙인다.
            messages.extend(context)
        # 분석을 요청하는 사용자 메시지. 원본 질문을 본문에 그대로 감싼다.
        messages.append(Message.user(f"다음 질문을 분석해 주세요:\n\n{message}"))
        return messages

    @staticmethod
    def _build_response_messages(
        message: str,
        context: list[Message] | None,
        analysis: str,
    ) -> list[Message]:
        """
        Pass 2(응답 단계)에 넣을 메시지 목록을 구성한다.

        핵심은 Pass 1에서 얻은 분석 텍스트를 assistant의 "사고(thinking)"로
        대화에 끼워 넣는 것이다. 이렇게 하면 모델은 마치 자기가 방금 그
        분석을 한 것처럼 이어받아, 그 내용을 근거로 최종 답변을 작성한다.
        마지막에 원본 질문을 사용자 메시지로 다시 붙여 "이제 이 질문에
        답하라"는 목표를 분명히 한다.

        Args:
            message: 원본 사용자 질문.
            context: 이전 대화 메시지 목록(없으면 None).
            analysis: Pass 1에서 생성된 내부 분석 텍스트.

        Returns:
            응답 단계에 그대로 넣을 Message 목록.
        """
        messages: list[Message] = []
        if context:
            # Pass 1과 마찬가지로 이전 대화 맥락을 앞에 이어 붙인다.
            messages.extend(context)
        # Pass 1 분석을 assistant의 사고로 삽입해, 모델이 참고하도록 한다.
        messages.append(Message.assistant(thinking=analysis))
        # 분석 뒤에 원본 질문을 다시 전달해 답변 대상을 못 박는다.
        messages.append(Message.user(message))
        return messages

    @staticmethod
    async def _collect_text(
        model: ModelProvider,  # noqa: F821
        messages: list[Message],
        system_prompt: str,
        temperature: float,
    ) -> str:
        """
        모델의 스트리밍 출력에서 텍스트 조각을 모아 하나의 문자열로 반환한다.

        모델은 답을 한 번에 주지 않고 여러 StreamEvent로 조금씩 흘려보낸다.
        이 헬퍼는 그중 실제 글자에 해당하는 TEXT_DELTA 이벤트만 골라
        순서대로 이어 붙여 완성된 텍스트를 만든다. Pass 1·Pass 2가
        모델 호출 방식이 같아서, 이 한 메서드로 두 단계를 공통 처리한다.

        [에러 처리]
        도중에 ERROR 이벤트가 오면 경고 로그를 남기고 즉시 루프를 멈춘다.
        예외를 던지지 않고, 그때까지 모은 텍스트만이라도 반환한다(부분 성공).
        빈 문자열로 끝날 수도 있으니 호출자는 결과가 비어 있을 가능성을
        염두에 둬야 한다.

        Args:
            model: 스트림을 산출하는 LLM 프로바이더.
            messages: 모델에 넣을 대화 메시지 목록.
            system_prompt: 이 호출에 적용할 시스템 프롬프트.
            temperature: 샘플링 온도(높을수록 표현이 다양해짐).

        Returns:
            수집된 전체 텍스트(에러 시 중단 지점까지의 부분 텍스트).
        """
        # 도착하는 텍스트 조각을 순서대로 담아 둘 버퍼.
        collected: list[str] = []
        # 모델 스트림을 비동기로 순회하며 이벤트를 하나씩 처리한다.
        async for event in model.stream(
            messages=messages,
            system_prompt=system_prompt,
            temperature=temperature,
        ):
            # 실제 글자가 실린 델타 이벤트만 골라 모은다.
            # event.text가 빈 값이면(예: 마커성 이벤트) 건너뛴다.
            if event.type == StreamEventType.TEXT_DELTA and event.text:
                collected.append(event.text)
            # 스트림 도중 에러가 나면 로그를 남기고 더 기다리지 않고 멈춘다.
            elif event.type == StreamEventType.ERROR:
                logger.warning(f"사고 엔진 스트림 에러: {event.message}")
                break
        # 모은 조각들을 이어 붙여 완성된 하나의 문자열로 돌려준다.
        return "".join(collected)
