"""
추론 엔진 — ModelProvider ABC + LocalModelProvider 구현.

Claude Code의 ApiClient를 Python ABC로 재구현한다.
LocalModelProvider는 vLLM의 OpenAI 호환 API와 SSE 스트리밍으로 통신한다.

핵심 흐름:
  Nexus Message[] → OpenAI messages[] → SSE chunks → StreamEvent yield

이 모듈은 4-Tier 체인의 Tier 3~4에 해당한다:
  Tier 3: stream() — SSE 스트림 파싱
  Tier 4: httpx 클라이언트 (재시도는 별도 with_retry에서 처리)
"""

from __future__ import annotations

import json
import logging
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import Any

import httpx

from core.message import (
    Message,
    StopReason,
    StreamEvent,
    StreamEventType,
    TextBlock,
    TokenUsage,
    ToolUseBlock,
)

logger = logging.getLogger("nexus.model.inference")


# ─────────────────────────────────────────────
# 모델 설정 (단일 모델의 런타임 설정)
# ─────────────────────────────────────────────
@dataclass
class ModelConfig:
    """
    모델별 런타임 설정.
    GPU 티어에 따라 자동 결정되거나, 설정 파일에서 로드된다.
    """

    model_id: str  # vLLM에서 사용하는 모델 식별자
    max_context_tokens: int = 8192
    max_output_tokens: int = 4096
    default_temperature: float = 0.7
    supports_tool_calling: bool = True  # vLLM 네이티브 tool calling 지원 여부
    supports_streaming: bool = True
    stop_sequences: list[str] = field(default_factory=list)
    fallback_model_id: str | None = None  # OOM 시 대체 모델
    cost_per_input_token: float = 0.0  # 로컬: 0 (전기 비용은 별도)
    cost_per_output_token: float = 0.0


# ─────────────────────────────────────────────
# ModelProvider ABC (추상 인터페이스)
# ─────────────────────────────────────────────
class ModelProvider(ABC):
    """
    LLM 추상 인터페이스.
    Claude Code의 ApiClient에 대응한다.

    이 인터페이스만 구현하면 QueryEngine, Tool System, Agent System 전체가
    구체적인 모델(Qwen, ExaOne, OpenAI 등)과 독립적으로 동작한다.

    왜 ABC인가: 향후 다른 모델 백엔드(직접 torch, TGI 등)를
    추가할 때 이 인터페이스만 구현하면 된다.
    """

    @abstractmethod
    async def stream(
        self,
        messages: list[Message],
        system_prompt: str,
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        stop_sequences: list[str] | None = None,
    ) -> AsyncGenerator[StreamEvent, None]:
        """
        모델에 스트리밍 요청을 보낸다.

        Yields:
            StreamEvent — text_delta, tool_use, message_stop 등
        """
        ...
        yield  # type: ignore  # ABC에서 AsyncGenerator yield 필요

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """텍스트 임베딩을 생성한다."""
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        """모델 서버 가용성을 확인한다."""
        ...

    @abstractmethod
    async def count_tokens(self, messages: list[Message]) -> int:
        """메시지 목록의 토큰 수를 추정한다."""
        ...

    @abstractmethod
    def get_config(self) -> ModelConfig:
        """현재 모델 설정을 반환한다."""
        ...


# ─────────────────────────────────────────────
# LocalModelProvider (에어갭 전용 — vLLM OpenAI 호환)
# ─────────────────────────────────────────────
class LocalModelProvider(ModelProvider):
    """
    vLLM OpenAI 호환 API를 통한 로컬 모델 프로바이더.

    Claude Code의 Anthropic SDK 클라이언트에 대응하지만,
    에어갭 환경에서 LAN 내 vLLM 서버와 통신한다.

    기능:
      - /v1/chat/completions SSE 스트리밍
      - Nexus Message → OpenAI message 자동 변환
      - tool schema → OpenAI function_calling 변환
      - 연결 실패 시 graceful error event yield
      - vLLM 네이티브 tool calling 지원
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_key: str = "local-key",
        model_id: str = "qwen3.5-27b",
        max_context_tokens: int = 8192,
        max_output_tokens: int = 4096,
        fallback_model_id: str | None = None,
        embedding_model_id: str = "multilingual-e5-large",
        embedding_base_url: str | None = None,
        connect_timeout: float = 10.0,
        read_timeout: float = 300.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model_id = model_id
        self.embedding_model_id = embedding_model_id
        # 임베딩 서버가 별도 포트/인스턴스인 경우 분리
        # 없으면 base_url과 동일한 서버를 사용한다
        self._embedding_base_url = (
            embedding_base_url.rstrip("/") if embedding_base_url else self.base_url
        )

        self._config = ModelConfig(
            model_id=model_id,
            max_context_tokens=max_context_tokens,
            max_output_tokens=max_output_tokens,
            fallback_model_id=fallback_model_id,
            supports_tool_calling=True,
        )

        # httpx 비동기 클라이언트 — 커넥션 풀링 + 타임아웃 설정
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(
                connect=connect_timeout,
                read=read_timeout,
                write=30.0,
                pool=10.0,
            ),
            limits=httpx.Limits(
                max_connections=20,
                max_keepalive_connections=10,
            ),
        )

        # 요청 통계
        self._request_count: int = 0
        self._total_latency: float = 0.0
        self._error_count: int = 0

    # ─── 핵심: stream() ───

    async def stream(
        self,
        messages: list[Message],
        system_prompt: str,
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        stop_sequences: list[str] | None = None,
    ) -> AsyncGenerator[StreamEvent, None]:
        """
        vLLM /v1/chat/completions SSE 스트리밍.

        변환 흐름:
          Nexus Message[] → OpenAI messages[] → SSE chunks → StreamEvent yield
        """
        self._request_count += 1
        start_time = time.monotonic()

        # 메시지 변환: Nexus → OpenAI 형식
        oai_messages = self._convert_messages(messages, system_prompt)

        # 요청 페이로드 구성
        # chat_template_kwargs로 Qwen3.5 thinking 모드를 끈다.
        # 왜: Qwen3.5의 기본 chat template은 <think> 블록을 강제 삽입한다.
        # 도구 호출 흐름에서는 Worker가 내부 독백만 뱉고 실제 답변을 누락하는
        # 사례가 관찰됨 ("사용자가 X에 대해 물어보고 있습니다..."로만 끝).
        # enable_thinking=false를 넘기면 빈 <think></think>가 주입되어
        # Worker가 곧바로 답변 생성 모드로 진입한다.
        payload: dict[str, Any] = {
            "model": self.model_id,
            "messages": oai_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},  # vLLM 사용량 추적
            "chat_template_kwargs": {"enable_thinking": False},
        }

        if stop_sequences:
            payload["stop"] = stop_sequences

        # 도구 스키마 변환: Nexus → OpenAI function_calling
        if tools:
            payload["tools"] = [self._convert_tool_schema(t) for t in tools]
            payload["tool_choice"] = "auto"

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        # MESSAGE_START 이벤트
        yield StreamEvent(
            type=StreamEventType.MESSAGE_START,
            model_id=self.model_id,
        )

        # 컨텍스트 초과 시 max_tokens를 줄여서 자동 재시도
        # 왜 여기서 하는가: vLLM의 실제 토큰 카운트는 BPE 기반이라
        # 사전 추정이 부정확하다. 에러 발생 후 정확한 input_tokens 값으로 재계산.
        max_retries_for_context = 3
        current_max_tokens = payload["max_tokens"]

        for attempt in range(max_retries_for_context):
            payload["max_tokens"] = current_max_tokens

            # SSE 스트리밍 처리
            accumulated_tool_calls: dict[int, dict[str, Any]] = {}
            total_usage = TokenUsage()
            context_exceeded = False
            # finish_reason 이후 usage 청크를 기다리기 위한 변수
            # vLLM은 finish_reason 청크 → usage 청크 → [DONE] 순으로 전송한다.
            pending_stop: StopReason | None = None

            try:
                async with self._client.stream(
                    "POST",
                    f"{self.base_url}/v1/chat/completions",
                    json=payload,
                    headers=headers,
                ) as response:
                    if response.status_code != 200:
                        error_body = ""
                        async for chunk in response.aiter_bytes():
                            error_body += chunk.decode("utf-8", errors="replace")

                        # 컨텍스트 초과 에러 감지 → max_tokens 줄여서 재시도
                        if response.status_code == 400 and "context length" in error_body:
                            import re

                            m = re.search(r"contains at least (\d+) input", error_body)
                            if m:
                                actual_input = int(m.group(1))
                                new_max = self._config.max_context_tokens - actual_input - 100
                                # 입력만으로 컨텍스트를 초과하면 재시도 불가
                                if new_max < 256:
                                    yield StreamEvent(
                                        type=StreamEventType.ERROR,
                                        error_code="CONTEXT_OVERFLOW",
                                        message=(
                                            f"입력({actual_input} 토큰)이 "
                                            f"컨텍스트({self._config.max_context_tokens})를 "
                                            f"거의 다 사용하여 응답을 생성할 수 없습니다. "
                                            f"더 짧은 내용으로 다시 시도해 주세요."
                                        ),
                                    )
                                    return
                                current_max_tokens = new_max
                                logger.warning(
                                    "컨텍스트 초과 → 재시도 %d/%d: "
                                    "input=%d, max_tokens=%d→%d",
                                    attempt + 1, max_retries_for_context,
                                    actual_input, payload["max_tokens"],
                                    current_max_tokens,
                                )
                                context_exceeded = True
                                continue

                        yield StreamEvent(
                            type=StreamEventType.ERROR,
                            error_code=f"HTTP_{response.status_code}",
                            message=f"vLLM 서버 에러: {response.status_code} - {error_body[:500]}",
                        )
                        return

                    # SSE 라인 파싱 (async with 블록 안에서 수행)
                    async for line in response.aiter_lines():
                        if not line.startswith("data: "):
                            continue

                        data_str = line[6:].strip()
                        if data_str == "[DONE]":
                            # pending_stop이 없으면 (finish_reason 없이 [DONE] 도달)
                            # 폴백으로 END_TURN 사용
                            if pending_stop is None:
                                for _evt in self._finalize_tool_calls(accumulated_tool_calls):
                                    yield _evt
                            yield StreamEvent(
                                type=StreamEventType.MESSAGE_STOP,
                                stop_reason=pending_stop or StopReason.END_TURN,
                                usage=total_usage,
                            )
                            return

                        try:
                            data = json.loads(data_str)
                        except json.JSONDecodeError:
                            logger.warning("잘못된 SSE JSON: %s", data_str[:200])
                            continue

                        if "usage" in data and data["usage"]:
                            u = data["usage"]
                            total_usage = TokenUsage(
                                input_tokens=u.get("prompt_tokens", 0),
                                output_tokens=u.get("completion_tokens", 0),
                            )

                        if not data.get("choices"):
                            continue

                        choice = data["choices"][0]
                        delta = choice.get("delta", {})
                        finish_reason = choice.get("finish_reason")

                        if delta.get("content"):
                            yield StreamEvent(
                                type=StreamEventType.TEXT_DELTA,
                                text=delta["content"],
                            )

                        if "tool_calls" in delta:
                            for tc_delta in delta["tool_calls"]:
                                for _evt in self._accumulate_tool_call(
                                    accumulated_tool_calls, tc_delta
                                ):
                                    yield _evt

                        if finish_reason:
                            stop = {
                                "stop": StopReason.END_TURN,
                                "length": StopReason.MAX_TOKENS,
                                "tool_calls": StopReason.TOOL_USE,
                            }.get(finish_reason, StopReason.END_TURN)

                            if finish_reason == "tool_calls":
                                for _evt in self._finalize_tool_calls(
                                    accumulated_tool_calls
                                ):
                                    yield _evt

                            # finish_reason 이후 usage 청크가 올 수 있으므로
                            # 바로 return하지 않고 finish 정보를 저장한다.
                            # [DONE] 또는 usage 청크에서 최종 yield + return 한다.
                            pending_stop = stop

            except httpx.ConnectError as e:
                self._error_count += 1
                yield StreamEvent(
                    type=StreamEventType.ERROR,
                    error_code="CONNECT_ERROR",
                    message=(
                        f"GPU 서버(vLLM) 연결 실패: {self.base_url}\n"
                        f"상세: {e}\n"
                        f"확인: 1) GPU 서버 실행 여부 2) 네트워크 연결 3) 방화벽 설정"
                    ),
                )
            except httpx.ReadTimeout as e:
                self._error_count += 1
                yield StreamEvent(
                    type=StreamEventType.ERROR,
                    error_code="READ_TIMEOUT",
                    message=f"GPU 서버 응답 타임아웃 ({self._client.timeout.read}s): {e}",
                )
            except httpx.HTTPStatusError as e:
                self._error_count += 1
                yield StreamEvent(
                    type=StreamEventType.ERROR,
                    error_code=f"HTTP_{e.response.status_code}",
                    message=f"vLLM HTTP 에러: {e.response.status_code} - {e.response.text[:500]}",
                )
            except Exception as e:
                self._error_count += 1
                logger.exception("stream()에서 예상치 못한 에러: %s", e)
                yield StreamEvent(
                    type=StreamEventType.ERROR,
                    error_code="UNKNOWN",
                    message=f"예상치 못한 에러: {type(e).__name__}: {e}",
                )

            # 컨텍스트 초과 재시도가 아니면 루프 탈출
            if not context_exceeded:
                break
        else:
            # for 루프가 break 없이 끝남 = 재시도 모두 실패
            yield StreamEvent(
                type=StreamEventType.ERROR,
                error_code="CONTEXT_OVERFLOW",
                message=(
                    "입력 내용이 너무 길어 분석할 수 없습니다. "
                    "더 짧은 내용으로 다시 시도해 주세요."
                ),
            )

        self._total_latency += time.monotonic() - start_time

    # ─── embed() ───

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """
        별도 임베딩 서버의 /v1/embed 엔드포인트를 호출한다.

        임베딩 서버 API:
          요청: POST /v1/embed {"texts": ["문장1", "문장2"]}
          응답: {"embeddings": [[...], [...]], "dimension": 1024}
        """
        try:
            response = await self._client.post(
                f"{self._embedding_base_url}/v1/embed",
                json={"texts": texts},
                headers={"Authorization": f"Bearer {self.api_key}"},
            )
            response.raise_for_status()
            data = response.json()
            return data["embeddings"]
        except Exception as e:
            logger.error(f"임베딩 실패: {e}")
            raise

    # ─── health_check() ───

    async def health_check(self) -> bool:
        """vLLM /health 엔드포인트로 가용성을 확인한다."""
        try:
            response = await self._client.get(
                f"{self.base_url}/health",
                timeout=5.0,
            )
            return response.status_code == 200
        except Exception:
            return False

    # ─── count_tokens() ───

    async def count_tokens(self, messages: list[Message]) -> int:
        """메시지의 토큰 수를 휴리스틱으로 추정한다."""
        return sum(m.estimated_tokens() for m in messages)

    # ─── get_config() ───

    def get_config(self) -> ModelConfig:
        return self._config

    # ─── 내부: 메시지 변환 ───

    def _convert_messages(
        self, messages: list[Message], system_prompt: str
    ) -> list[dict[str, Any]]:
        """
        Nexus Message[] → OpenAI messages[] 변환.

        변환 규칙:
          - system → {"role": "system", "content": ...}
          - user → {"role": "user", "content": ...}
          - assistant → {"role": "assistant", "content": ..., "tool_calls": [...]}
          - tool_result → {"role": "tool", "content": ..., "tool_call_id": ...}
        """
        oai: list[dict[str, Any]] = []

        # 시스템 프롬프트
        if system_prompt:
            oai.append({"role": "system", "content": system_prompt})

        for msg in messages:
            role = msg.role if isinstance(msg.role, str) else msg.role.value

            if role == "system":
                oai.append({"role": "system", "content": msg.text_content})
            elif role == "user":
                oai.append({"role": "user", "content": msg.text_content})
            elif role == "tool_result":
                oai.append({
                    "role": "tool",
                    "content": str(msg.content),
                    "tool_call_id": msg.tool_use_id or "",
                })
            elif role == "assistant":
                entry: dict[str, Any] = {"role": "assistant"}

                if isinstance(msg.content, str):
                    entry["content"] = msg.content
                elif isinstance(msg.content, list):
                    text_parts: list[str] = []
                    tool_calls: list[dict[str, Any]] = []

                    for block in msg.content:
                        if isinstance(block, TextBlock):
                            text_parts.append(block.text)
                        elif isinstance(block, ToolUseBlock):
                            tool_calls.append({
                                "id": block.id,
                                "type": "function",
                                "function": {
                                    "name": block.name,
                                    "arguments": json.dumps(
                                        block.input, ensure_ascii=False
                                    ),
                                },
                            })

                    entry["content"] = " ".join(text_parts) if text_parts else None
                    if tool_calls:
                        entry["tool_calls"] = tool_calls

                oai.append(entry)

        return oai

    # ─── 내부: 도구 스키마 변환 ───

    @staticmethod
    def _convert_tool_schema(tool_schema: dict[str, Any]) -> dict[str, Any]:
        """
        Nexus 도구 스키마 → OpenAI function_calling 형식 변환.

        Nexus: {"name": "Read", "description": "...", "input_schema": {...}}
        OpenAI: {"type": "function", "function": {"name": "Read", ...}}
        """
        return {
            "type": "function",
            "function": {
                "name": tool_schema["name"],
                "description": tool_schema.get("description", ""),
                "parameters": tool_schema.get(
                    "input_schema", {"type": "object", "properties": {}}
                ),
            },
        }

    # ─── 내부: tool_calls 증분 누적 ───

    @staticmethod
    def _accumulate_tool_call(
        accumulated: dict[int, dict[str, Any]],
        tc_delta: dict[str, Any],
    ) -> list[StreamEvent]:
        """
        SSE chunk에서 온 tool_call 델타를 누적하고,
        필요한 StreamEvent를 반환한다.
        """
        events: list[StreamEvent] = []
        idx = tc_delta.get("index", 0)

        if idx not in accumulated:
            accumulated[idx] = {
                "id": tc_delta.get("id", ""),
                "function": {"name": "", "arguments": ""},
            }

        tc = accumulated[idx]
        if tc_delta.get("id"):
            tc["id"] = tc_delta["id"]

        func = tc_delta.get("function", {})
        if func.get("name"):
            tc["function"]["name"] = func["name"]
            # TOOL_USE_START 이벤트
            events.append(
                StreamEvent(
                    type=StreamEventType.TOOL_USE_START,
                    tool_use=ToolUseBlock(
                        id=tc["id"],
                        name=func["name"],
                        input={},
                    ),
                )
            )
        if func.get("arguments"):
            tc["function"]["arguments"] += func["arguments"]
            events.append(
                StreamEvent(
                    type=StreamEventType.TOOL_USE_DELTA,
                    tool_use_delta=func["arguments"],
                )
            )

        return events

    # ─── 내부: tool_calls 최종 완성 ───

    @staticmethod
    def _finalize_tool_calls(
        accumulated: dict[int, dict[str, Any]],
    ) -> list[StreamEvent]:
        """누적된 tool_calls를 최종 TOOL_USE_STOP 이벤트로 변환한다."""
        events: list[StreamEvent] = []
        for idx in sorted(accumulated.keys()):
            tc = accumulated[idx]
            if tc.get("function", {}).get("name"):
                try:
                    args = json.loads(tc["function"].get("arguments", "{}"))
                except json.JSONDecodeError:
                    args = {}
                events.append(
                    StreamEvent(
                        type=StreamEventType.TOOL_USE_STOP,
                        tool_use=ToolUseBlock(
                            id=tc.get("id", f"call_{idx}"),
                            name=tc["function"]["name"],
                            input=args,
                        ),
                    )
                )
        return events

    # ─── 통계 ───

    @property
    def stats(self) -> dict[str, Any]:
        """요청 통계를 반환한다."""
        return {
            "request_count": self._request_count,
            "error_count": self._error_count,
            "total_latency_s": round(self._total_latency, 2),
            "avg_latency_s": round(
                self._total_latency / max(self._request_count, 1), 2
            ),
        }

    async def close(self) -> None:
        """HTTP 클라이언트를 정리한다."""
        await self._client.aclose()
