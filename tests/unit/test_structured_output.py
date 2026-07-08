# 구조화 출력(structured output / vLLM guided decoding) 기능 단위 테스트 — Point 4.1
"""
구조화 출력 기능의 4-Tier 전파와 web 핸들러 변환을 격리 검증한다.

검증 범위(설계 8장 Phase 5 표):
  - Tier 3(inference): payload 주입(response_format/structured_outputs),
    tools 상호배타 ValueError, enable_thinking 강제 False.
  - Tier 2(query_loop): structured_output passthrough, 도구 비노출 분기.
  - Config: YAML 로드 + 누락 시 폴백 기본값.
  - Web(/v1/chat/completions): response_format 변환 + fail-closed 400.

실제 vLLM/GPU/Redis/PG는 호출하지 않는다. httpx payload는 monkeypatch로 캡처하고,
AsyncGenerator는 규칙대로 `async for`로 전량 소비한다.
"""
from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

from core.config import NexusConfig, StructuredOutputConfig
from core.message import Message, StreamEvent, StreamEventType
from core.model.inference import LocalModelProvider, StructuredOutputSpec

# query_loop 테스트에서 재사용하는 스크립트형 mock 프로바이더.
from tests.unit.test_query_loop import ScriptedProvider

# ─────────────────────────────────────────────
# 공통: 간단한 JSON Schema (draft 2020-12 부분집합)
# ─────────────────────────────────────────────
_SAMPLE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "composer": {"type": "string"},
        "birth_year": {"type": "integer"},
    },
    "required": ["composer", "birth_year"],
}


# ─────────────────────────────────────────────
# 헬퍼: httpx AsyncClient.stream()을 흉내내는 가짜 스트림 컨텍스트
# (test_inference_sampling_payload.py와 동일 패턴 — payload를 캡처한다)
# ─────────────────────────────────────────────
class _FakeStreamResponse:
    """vLLM SSE 응답을 최소한으로 흉내내는 가짜 응답 객체(200 정상 경로)."""

    status_code = 200

    async def aiter_lines(self):
        yield 'data: {"choices":[{"delta":{"content":"{}"},"finish_reason":null}]}'
        yield (
            'data: {"choices":[{"delta":{},"finish_reason":"stop"}],'
            '"usage":{"prompt_tokens":3,"completion_tokens":1}}'
        )
        yield "data: [DONE]"

    async def aiter_bytes(self):
        if False:
            yield b""


class _FakeStreamCtx:
    """`async with client.stream(...) as response:`를 지원하는 캡처용 컨텍스트."""

    def __init__(self, capture: dict[str, Any], **kwargs: Any) -> None:
        capture.clear()
        capture.update(kwargs)
        self._response = _FakeStreamResponse()

    async def __aenter__(self) -> _FakeStreamResponse:
        return self._response

    async def __aexit__(self, *exc: Any) -> bool:
        return False


async def _capture_payload(
    provider: LocalModelProvider, **stream_kwargs: Any
) -> dict[str, Any]:
    """stream()을 끝까지 소비하며 vLLM으로 전송되는 payload를 캡처해 반환한다."""
    captured: dict[str, Any] = {}

    def fake_stream(method: str, url: str, **kwargs: Any) -> _FakeStreamCtx:
        return _FakeStreamCtx(captured, **kwargs)

    messages = [Message.user("작곡가 정보를 JSON으로")]
    with patch.object(provider._client, "stream", side_effect=fake_stream):
        async for _event in provider.stream(
            messages=messages,
            system_prompt="너는 도우미다.",
            **stream_kwargs,
        ):
            pass

    assert "json" in captured, "stream()이 json=payload로 POST하지 않았다"
    return captured["json"]


def _make_provider(injection_mode: str = "response_format") -> LocalModelProvider:
    """LAN 주소로 LocalModelProvider를 생성한다(에어갭 준수, 실제 연결 없음)."""
    return LocalModelProvider(
        base_url="http://192.168.21.112:8000",
        structured_output_injection_mode=injection_mode,
    )


# ═════════════════════════════════════════════
# Tier 3 — inference payload 주입
# ═════════════════════════════════════════════
async def test_stream_structured_output_injects_response_format() -> None:
    """structured_output 지정 시 payload.response_format.json_schema.schema가 일치한다."""
    provider = _make_provider()
    spec = StructuredOutputSpec(json_schema=_SAMPLE_SCHEMA, name="composer_info")
    payload = await _capture_payload(provider, structured_output=spec)

    rf = payload["response_format"]
    assert rf["type"] == "json_schema"
    assert rf["json_schema"]["name"] == "composer_info"
    assert rf["json_schema"]["schema"] == _SAMPLE_SCHEMA
    assert rf["json_schema"]["strict"] is True
    # extra_body로 감싸면 안 된다(httpx raw POST — vLLM 인식 실패 방지).
    assert "extra_body" not in payload


async def test_stream_structured_output_with_tools_raises_valueerror() -> None:
    """structured_output과 tools 동시 지정은 상호 배타로 ValueError를 던진다(fail-closed)."""
    provider = _make_provider()
    spec = StructuredOutputSpec(json_schema=_SAMPLE_SCHEMA)
    tools = [{"name": "Read", "description": "", "input_schema": {"type": "object"}}]

    with pytest.raises(ValueError, match="동시에 사용할 수 없습니다"):
        async for _event in provider.stream(
            messages=[Message.user("hi")],
            system_prompt="s",
            tools=tools,
            structured_output=spec,
        ):
            pass


async def test_stream_structured_output_forces_thinking_off() -> None:
    """structured_output 지정 시 enable_thinking=True여도 payload에서 False로 강제된다."""
    provider = _make_provider()
    spec = StructuredOutputSpec(json_schema=_SAMPLE_SCHEMA)
    payload = await _capture_payload(
        provider, structured_output=spec, enable_thinking=True
    )

    assert payload["chat_template_kwargs"]["enable_thinking"] is False


async def test_stream_structured_output_injection_mode_structured_outputs() -> None:
    """injection_mode=structured_outputs 전환 시 ② 형태(top-level)로 주입된다."""
    provider = _make_provider(injection_mode="structured_outputs")
    spec = StructuredOutputSpec(json_schema=_SAMPLE_SCHEMA)
    payload = await _capture_payload(provider, structured_output=spec)

    assert payload["structured_outputs"] == {"json": _SAMPLE_SCHEMA}
    # 표준형 response_format 키는 이 모드에서 존재하지 않아야 한다.
    assert "response_format" not in payload


# ═════════════════════════════════════════════
# Tier 2 — query_loop 전파/도구 비노출
# ═════════════════════════════════════════════
async def test_query_loop_structured_output_passthrough_to_tier3(
    tool_use_context: Any,
) -> None:
    """query_loop이 structured_output을 Tier 3(stream)로 그대로 전달한다."""
    from core.orchestrator.query_loop import query_loop

    provider = ScriptedProvider([{"text": '{"composer":"Bach","birth_year":1685}'}])
    spec = StructuredOutputSpec(json_schema=_SAMPLE_SCHEMA)

    async for _event in query_loop(
        messages=[Message.user("바흐 출생년도?")],
        system_prompt="s",
        model_provider=provider,
        tools=[],
        context=tool_use_context,
        structured_output=spec,
    ):
        pass

    assert provider.last_structured_output is spec


async def test_query_loop_structured_output_omits_tool_schemas(
    tool_use_context: Any,
    basic_tools: Any,
) -> None:
    """structured_output 지정 시 도구를 노출하지 않는다(1차 방어 — tools=None 전달)."""
    from core.orchestrator.query_loop import query_loop

    provider = ScriptedProvider([{"text": "{}"}])
    spec = StructuredOutputSpec(json_schema=_SAMPLE_SCHEMA)

    async for _event in query_loop(
        messages=[Message.user("추출")],
        system_prompt="s",
        model_provider=provider,
        tools=basic_tools,  # 도구가 있어도 구조화 모드에서는 노출 금지
        context=tool_use_context,
        structured_output=spec,
    ):
        pass

    assert provider.last_tools is None


# ═════════════════════════════════════════════
# Config — YAML 로드 + 폴백
# ═════════════════════════════════════════════
def test_config_structured_output_yaml_load_and_fallback() -> None:
    """structured_output 섹션 로드값 반영 + 누락 시 기본값 폴백을 검증한다."""
    # 1) 폴백 — 섹션 미지정 시 클래스 기본값.
    default_cfg = StructuredOutputConfig()
    assert default_cfg.enabled is True
    assert default_cfg.injection_mode == "response_format"
    assert default_cfg.max_schema_bytes == 65536
    assert default_cfg.strict is True

    # NexusConfig 루트에도 기본값으로 존재한다(무회귀).
    root_default = NexusConfig()
    assert root_default.structured_output.injection_mode == "response_format"

    # 2) 로드값 반영 — yaml에서 온 dict가 그대로 파싱된다.
    root_loaded = NexusConfig(
        structured_output={
            "enabled": False,
            "injection_mode": "structured_outputs",
            "max_schema_bytes": 1024,
            "strict": False,
        }
    )
    so = root_loaded.structured_output
    assert so.enabled is False
    assert so.injection_mode == "structured_outputs"
    assert so.max_schema_bytes == 1024
    assert so.strict is False


# ═════════════════════════════════════════════
# Web — /v1/chat/completions response_format 변환 + fail-closed
# ═════════════════════════════════════════════
class _FakeOpenAIEngine:
    """chat_completions 비스트림 경로가 쓰는 최소 표면을 구현한 가짜 엔진.

    submit_message가 받은 structured_output을 기록해 변환 결과를 검증한다.
    """

    def __init__(self) -> None:
        self.system_prompt = "base-prompt"
        self._messages: list = []
        self.received_structured_output: Any = "UNSET"

    def update_system_prompt(self, prompt: str) -> None:
        self.system_prompt = prompt

    def bind_request(self, **kwargs: Any) -> None:
        pass

    def clear_messages(self) -> None:
        self._messages.clear()

    async def submit_message(
        self, message: str, structured_output: Any = None
    ):
        self.received_structured_output = structured_output
        # 텍스트를 흘려 200 응답이 구성되게 한다.
        yield StreamEvent(type=StreamEventType.TEXT_DELTA, text="{}")


@pytest.fixture
def _openai_engine_state():
    """_app_state에 가짜 엔진을 주입하고 테스트 후 원복한다(모듈 싱글톤 격리)."""
    from web.app import _app_state

    saved_engine = _app_state.get("query_engine")
    saved_parts = _app_state.get("web_engine_parts")
    saved_mm = _app_state.get("memory_manager")
    fake = _FakeOpenAIEngine()
    _app_state["query_engine"] = fake
    _app_state["web_engine_parts"] = None  # 싱글톤 경로 강제
    _app_state["memory_manager"] = None
    yield fake
    _app_state["query_engine"] = saved_engine
    _app_state["web_engine_parts"] = saved_parts
    _app_state["memory_manager"] = saved_mm


async def test_openai_endpoint_response_format_json_schema_accepted(
    _openai_engine_state: _FakeOpenAIEngine,
) -> None:
    """json_schema response_format이 StructuredOutputSpec으로 변환돼 엔진에 전달된다."""
    from web.app import OpenAIChatCompletionRequest, chat_completions

    request = OpenAIChatCompletionRequest(
        messages=[{"role": "user", "content": "바흐 출생년도?"}],
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "composer_info", "schema": _SAMPLE_SCHEMA},
        },
    )
    resp = await chat_completions(request)

    spec = _openai_engine_state.received_structured_output
    assert isinstance(spec, StructuredOutputSpec)
    assert spec.json_schema == _SAMPLE_SCHEMA
    assert spec.name == "composer_info"
    # 정상 200 응답 본문이 구성됐는지(choices 존재) 확인.
    assert resp.choices[0].message.role == "assistant"


async def test_openai_endpoint_response_format_unknown_type_400(
    _openai_engine_state: _FakeOpenAIEngine,
) -> None:
    """알 수 없는 response_format.type은 조용히 무시하지 않고 400으로 거부한다."""
    from fastapi import HTTPException

    from web.app import OpenAIChatCompletionRequest, chat_completions

    request = OpenAIChatCompletionRequest(
        messages=[{"role": "user", "content": "hi"}],
        response_format={"type": "xml_schema"},
    )
    with pytest.raises(HTTPException) as exc:
        await chat_completions(request)
    assert exc.value.status_code == 400


async def test_openai_endpoint_schema_too_large_400(
    _openai_engine_state: _FakeOpenAIEngine,
) -> None:
    """max_schema_bytes를 초과하는 스키마는 400으로 거부한다(fail-closed)."""
    from fastapi import HTTPException

    from web.app import OpenAIChatCompletionRequest, _app_state, chat_completions

    # config의 max_schema_bytes를 작게 낮춰 초과를 유발한다.
    _app_state["config"] = NexusConfig(
        structured_output={"max_schema_bytes": 64}
    )
    try:
        big_schema = {
            "type": "object",
            "properties": {
                f"field_{i}": {"type": "string"} for i in range(50)
            },
        }
        request = OpenAIChatCompletionRequest(
            messages=[{"role": "user", "content": "hi"}],
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "big", "schema": big_schema},
            },
        )
        with pytest.raises(HTTPException) as exc:
            await chat_completions(request)
        assert exc.value.status_code == 400
    finally:
        _app_state.pop("config", None)
