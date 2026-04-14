"""
Hidden Chain-of-Thought 엔진 — 2-pass 사고 처리.

Pass 1: 분석 프롬프트로 내부 분석을 생성한다 (사용자에게 노출되지 않음).
Pass 2: 분석 결과 + 원본 질문을 합쳐 최종 응답을 생성한다.

이 방식은 모델이 "먼저 생각하고 답하게" 하여 응답 품질을 높인다.
사용자는 최종 응답만 보고, 내부 분석 과정은 ThinkingResult.thinking_text에 기록된다.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from core.message import Message, StreamEventType

if TYPE_CHECKING:
    from core.thinking.orchestrator import ThinkingResult

logger = logging.getLogger("nexus.thinking.hidden_cot")

# ─── 분석 단계 시스템 프롬프트 ───
# Pass 1에서 모델에게 분석만 요청하는 지시문
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

# Pass 2에서 분석 결과를 기반으로 최종 응답을 요청하는 지시문
RESPONSE_SYSTEM_PROMPT = (
    "이전 분석 결과를 바탕으로 사용자의 질문에 정확하고 완전하게 답변하세요. "
    "분석 과정은 언급하지 말고, 최종 답변만 작성하세요."
)


class HiddenCoTEngine:
    """
    2-pass Hidden Chain-of-Thought 엔진.

    동작 흐름:
      Pass 1: 분석 프롬프트 → 모델이 내부 분석 생성
      Pass 2: 분석 + 원본 질문 → 모델이 최종 응답 생성

    왜 2-pass인가:
      단일 pass로 "먼저 분석하고 답하라"고 하면 모델이 분석을 생략하거나
      분석과 응답을 혼합하는 경향이 있다. 2-pass로 분리하면
      분석의 질이 높아지고, 최종 응답에서 분석 내용이 노출되지 않는다.
    """

    async def run(
        self,
        message: str,
        context: list[Message] | None,
        model: ModelProvider,  # noqa: F821 — 런타임에 import 가능
    ) -> ThinkingResult:
        """
        2-pass Hidden CoT를 실행한다.

        Args:
            message: 사용자 입력 텍스트
            context: 이전 대화 컨텍스트 (선택)
            model: LLM 프로바이더 (ModelProvider ABC)

        Returns:
            ThinkingResult — 최종 응답 + 내부 분석 텍스트 + 메타데이터
        """
        # 지연 import — 순환 의존 방지
        from core.thinking.orchestrator import ThinkingResult
        from core.thinking.strategy import ThinkingStrategy

        start_time = time.monotonic()

        # ── Pass 1: 내부 분석 생성 ──
        logger.debug("Hidden CoT Pass 1: 분석 시작")
        analysis_messages = self._build_analysis_messages(message, context)

        analysis_text = await self._collect_text(
            model=model,
            messages=analysis_messages,
            system_prompt=ANALYSIS_SYSTEM_PROMPT,
            temperature=0.4,  # 분석은 정확도 우선
        )
        logger.debug(f"Hidden CoT Pass 1 완료: {len(analysis_text)}자")

        # ── Pass 2: 분석 기반 최종 응답 생성 ──
        logger.debug("Hidden CoT Pass 2: 응답 생성 시작")
        response_messages = self._build_response_messages(message, context, analysis_text)

        response_text = await self._collect_text(
            model=model,
            messages=response_messages,
            system_prompt=RESPONSE_SYSTEM_PROMPT,
            temperature=0.7,  # 응답은 자연스러움 우선
        )
        logger.debug(f"Hidden CoT Pass 2 완료: {len(response_text)}자")

        elapsed = time.monotonic() - start_time

        return ThinkingResult(
            strategy=ThinkingStrategy.HIDDEN_COT,
            response=response_text,
            thinking_text=analysis_text,
            passes=2,
            elapsed_seconds=round(elapsed, 3),
            score=0.0,  # 호출자(Orchestrator)가 스코어를 설정
        )

    @staticmethod
    def _build_analysis_messages(
        message: str,
        context: list[Message] | None,
    ) -> list[Message]:
        """
        Pass 1용 메시지 목록을 구성한다.
        컨텍스트가 있으면 포함하여 맥락을 유지한다.
        """
        messages: list[Message] = []
        if context:
            messages.extend(context)
        # 분석 요청 메시지
        messages.append(Message.user(f"다음 질문을 분석해 주세요:\n\n{message}"))
        return messages

    @staticmethod
    def _build_response_messages(
        message: str,
        context: list[Message] | None,
        analysis: str,
    ) -> list[Message]:
        """
        Pass 2용 메시지 목록을 구성한다.
        분석 결과를 system 메시지로 주입하고, 원본 질문을 user 메시지로 전달한다.
        """
        messages: list[Message] = []
        if context:
            messages.extend(context)
        # 분석 결과를 assistant 메시지로 삽입 (모델이 참고하도록)
        messages.append(Message.assistant(thinking=analysis))
        # 원본 질문 재전달
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
        모델 스트림에서 텍스트 델타를 수집하여 전체 텍스트를 반환한다.

        StreamEvent 중 TEXT_DELTA 타입만 모아서 하나의 문자열로 합친다.
        에러 이벤트 발생 시 로그를 남기고 수집된 텍스트까지 반환한다.
        """
        collected: list[str] = []
        async for event in model.stream(
            messages=messages,
            system_prompt=system_prompt,
            temperature=temperature,
        ):
            if event.type == StreamEventType.TEXT_DELTA and event.text:
                collected.append(event.text)
            elif event.type == StreamEventType.ERROR:
                logger.warning(f"사고 엔진 스트림 에러: {event.message}")
                break
        return "".join(collected)
