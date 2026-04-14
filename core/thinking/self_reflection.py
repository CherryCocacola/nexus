"""
자기 성찰 엔진 — 3-pass 사고 처리 (분석 → 응답 → 검증/수정).

HiddenCoTEngine의 2-pass에 검증 단계를 추가한 고급 사고 엔진이다.

Pass 1: 심층 분석 — 문제를 구조적으로 분해한다
Pass 2: 초기 응답 생성 — 분석 결과를 기반으로 응답을 작성한다
Pass 3: 응답 검증 + 수정 — 초기 응답의 정확성, 완전성, 일관성을 검증하고 수정한다

왜 3-pass인가:
  복잡한 문제에서 모델이 한 번에 완벽한 응답을 생성하기 어렵다.
  검증 단계를 추가하면 논리적 오류, 누락, 불일관성을 잡아낼 수 있다.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from core.message import Message, StreamEventType

if TYPE_CHECKING:
    from core.thinking.orchestrator import ThinkingResult

logger = logging.getLogger("nexus.thinking.self_reflection")

# ─── Pass 1: 심층 분석 프롬프트 ───
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
INITIAL_RESPONSE_PROMPT = (
    "이전 심층 분석 결과를 바탕으로 사용자의 질문에 완전하고 정확하게 답변하세요. "
    "분석 과정은 언급하지 말고, 최종 답변만 작성하세요."
)

# ─── Pass 3: 검증 + 수정 프롬프트 ───
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

    동작 흐름:
      Pass 1: 심층 분석 → 구조화된 문제 분해
      Pass 2: 분석 기반 초기 응답 생성
      Pass 3: 초기 응답 검증 + 수정 → 최종 응답

    HIDDEN_COT와의 차이:
      - Pass 3(검증)이 추가되어 응답 품질이 더 높다
      - 대신 LLM 호출이 3회이므로 지연 시간이 더 길다
      - 복잡도 스코어 0.6~0.8 구간에서 사용한다
    """

    async def run(
        self,
        message: str,
        context: list[Message] | None,
        model: ModelProvider,  # noqa: F821
    ) -> ThinkingResult:
        """
        3-pass 자기 성찰을 실행한다.

        Args:
            message: 사용자 입력 텍스트
            context: 이전 대화 컨텍스트 (선택)
            model: LLM 프로바이더

        Returns:
            ThinkingResult — 최종(검증된) 응답 + 전체 사고 과정 + 메타데이터
        """
        # 지연 import — 순환 의존 방지
        from core.thinking.orchestrator import ThinkingResult
        from core.thinking.strategy import ThinkingStrategy

        start_time = time.monotonic()

        # ── Pass 1: 심층 분석 ──
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

        elapsed = time.monotonic() - start_time

        # 전체 사고 과정을 thinking_text에 기록
        thinking_text = (
            f"=== Pass 1: 심층 분석 ===\n{analysis_text}\n\n"
            f"=== Pass 2: 초기 응답 ===\n{initial_response}\n\n"
            f"=== Pass 3: 검증 결과 ===\n"
            f"(최종 응답이 초기 응답과 {'동일' if final_response == initial_response else '다름'})"
        )

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
        """Pass 1용 메시지: 컨텍스트 + 심층 분석 요청."""
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
        """Pass 2용 메시지: 컨텍스트 + 분석 결과 + 원본 질문."""
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
        Pass 3용 메시지: 원본 질문 + 초기 응답 → 검증 요청.
        검증자에게 원본 질문과 초기 응답 모두를 제공하여
        누락이나 오류를 판단할 수 있게 한다.
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
        """모델 스트림에서 텍스트를 수집한다."""
        collected: list[str] = []
        async for event in model.stream(
            messages=messages,
            system_prompt=system_prompt,
            temperature=temperature,
        ):
            if event.type == StreamEventType.TEXT_DELTA and event.text:
                collected.append(event.text)
            elif event.type == StreamEventType.ERROR:
                logger.warning(f"Self-Reflect 스트림 에러: {event.message}")
                break
        return "".join(collected)
