"""
Scout 모델 프로바이더 — llama.cpp CPU 모델 연동.

v7.0 Phase 9.5 (2026-04-17 Qwen 소형 전환): Scout(4B)는 CPU에서 실행되는
작은 모델로, 파일 탐색과 계획 수립만 담당한다. GPU VRAM에 영향을 주지 않는다.

모델 변경: Gemma 4 E4B (llama.cpp tool_call 한계) → Qwen3.5-4B
  이유: Worker(Qwen3.5-27B)와 동일 패밀리 유지. 토크나이저/chat template/
  tool-call 문법이 일관되어 Scout → Worker 핸드오프 호환성이 올라간다.

llama.cpp는 OpenAI 호환 API를 제공하므로,
LocalModelProvider를 내부적으로 재사용한다.

차이점:
  - max_context_tokens: 4096 (Scout는 짧은 컨텍스트)
  - max_output_tokens: 512 (계획서만 출력)
  - 연결 실패 시 예외 대신 None 반환 (fallback 지원)
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from typing import Any

from core.message import Message, StreamEvent
from core.model.inference import LocalModelProvider

logger = logging.getLogger("nexus.model.scout_provider")


class ScoutModelProvider(LocalModelProvider):
    """
    llama.cpp OpenAI 호환 API를 통한 Scout 모델 프로바이더.

    LocalModelProvider를 상속하여 동일 인터페이스를 제공한다.
    Scout 전용 설정(짧은 컨텍스트, 짧은 출력)을 적용한다.

    Scout 전용 오버라이드:
      stream()에서 enable_thinking 인자를 무조건 None으로 강제한다.
      이유: Qwen3.5-4B가 llama.cpp에서 enable_thinking=False를 받으면 빈
      <think></think> 블록 이후 거짓 tool_call 1개를 뱉고 28 토큰 만에 조기
      종료되는 버그가 있음(α 진단, 2026-04-21, tmp_scout_thinking 재현).
      Worker(27B)와는 독립적인 이슈이므로 Scout 경로에서만 플래그를 빼낸다.
    """

    def __init__(
        self,
        base_url: str = "http://192.168.22.28:8003",
        api_key: str = "local-key",
        model_id: str = "qwen3.5-4b",
        max_context_tokens: int = 4096,
        max_output_tokens: int = 512,
        connect_timeout: float = 5.0,
        read_timeout: float = 60.0,
    ) -> None:
        """
        Scout 모델 프로바이더를 초기화한다.

        Args:
            base_url: llama.cpp 서버 URL (CPU, :8003)
            api_key: API 키 (llama.cpp --api-key와 동일)
            model_id: 모델 식별자 (로깅용)
            max_context_tokens: 최대 컨텍스트 (Scout는 4096)
            max_output_tokens: 최대 출력 토큰 (Scout는 512)
            connect_timeout: 연결 타임아웃 (짧게 — fallback 빠르게)
            read_timeout: 읽기 타임아웃 (CPU 추론이 느리므로 60초)
        """
        super().__init__(
            base_url=base_url,
            api_key=api_key,
            model_id=model_id,
            max_context_tokens=max_context_tokens,
            max_output_tokens=max_output_tokens,
            connect_timeout=connect_timeout,
            read_timeout=read_timeout,
        )
        # Scout 전용 플래그 — reasoning_content를 TEXT_DELTA로 합쳐서 내보낸다.
        # Qwen3.5-4B가 llama.cpp에서 Qwen chat template의 `<think>` 블록을
        # reasoning_content로 자동 분리해 보내는데, 실제로 유용한 "4섹션 마크다운
        # 리포트"의 상당 부분이 reasoning 쪽에 실려 온다. Worker에게 넘겨야 할
        # 내용이므로 content와 병합해 TEXT_DELTA로 노출한다.
        self._include_reasoning_as_text = True

        logger.info(
            "ScoutModelProvider 초기화: %s (ctx=%d, max_out=%d)",
            base_url,
            max_context_tokens,
            max_output_tokens,
        )

    async def stream(
        self,
        messages: list[Message],
        system_prompt: str,
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        stop_sequences: list[str] | None = None,
        model_override: str | None = None,
        enable_thinking: bool | None = False,
    ) -> AsyncGenerator[StreamEvent, None]:
        """
        Scout 전용 stream — enable_thinking을 None으로 강제한다.

        호출자가 어떤 값을 넘기든(True/False/None) 모두 None으로 변환하여
        상위 LocalModelProvider가 chat_template_kwargs를 요청에서 빼도록 한다.
        이렇게 하면 llama.cpp의 기본 Qwen3.5 chat template이 그대로 동작하여
        정상적인 길이의 응답(마크다운 4섹션 리포트)을 얻을 수 있다.
        """
        # enable_thinking을 None으로 강제 — 4B가 False에서 깨지는 α 이슈 회피
        async for ev in super().stream(
            messages=messages,
            system_prompt=system_prompt,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
            stop_sequences=stop_sequences,
            model_override=model_override,
            enable_thinking=None,
        ):
            yield ev


async def create_scout_provider_if_available(
    base_url: str = "http://192.168.22.28:8003",
    api_key: str = "local-key",
) -> ScoutModelProvider | None:
    """
    Scout 서버가 실행 중이면 ScoutModelProvider를 생성하고,
    아니면 None을 반환한다 (fallback 지원).

    왜 이 함수인가: llama.cpp가 아직 설치/실행되지 않았을 수 있다.
    Scout 없이도 Worker 단독 모드로 동작해야 하므로,
    연결 실패를 예외가 아닌 None으로 처리한다.
    """
    provider = ScoutModelProvider(base_url=base_url, api_key=api_key)
    try:
        healthy = await provider.health_check()
        if healthy:
            logger.info("Scout 서버 연결 성공: %s", base_url)
            return provider
        logger.warning("Scout 서버 응답 없음: %s", base_url)
        return None
    except Exception as e:
        logger.warning("Scout 서버 연결 실패 (Worker 단독 모드로 전환): %s", e)
        return None
