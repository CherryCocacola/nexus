# 모델 오류를 컨텍스트 초과로 뭉뚱그려 보고하지 않는지 검증한다.
"""
모델 오류 사유 보고 테스트 (2026-08-20).

무엇을 막는가:
    예전에는 도구를 못 건진 모델 오류가 재시도를 소진하면, 원인이 무엇이든
    "입력 내용이 너무 길어 분석할 수 없습니다"(CONTEXT_OVERFLOW)로 보고했다.
    bakeoff 첫 시도에서 실제 원인은 **HTTP 404**(코딩 서버에 앵커 모델명을 보냄)
    였는데, 사용자에게는 입력을 줄이라고 안내됐다. 원인과 무관한 처방이다.

지키는 계약:
    - 컨텍스트 초과로 단정할 수 있을 때만 CONTEXT_OVERFLOW 를 쓴다.
    - 그 외에는 MODEL_ERROR 로, 실제 사유 문구를 실어 보낸다.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.message import Message, StreamEvent, StreamEventType
from core.orchestrator.query_loop import query_loop
from core.tools.base import ToolUseContext


class _ModelConfig:
    """query_loop 이 읽는 최소 설정 스텁."""

    max_output_tokens = 1024
    max_context_tokens = 8192


class _AlwaysErrorProvider:
    """매 턴 같은 오류 이벤트만 내보내는 프로바이더."""

    def __init__(self, message: str) -> None:
        self._message = message
        self.model_id = "stub"

    def get_config(self) -> _ModelConfig:
        return _ModelConfig()

    async def stream(self, *args, **kwargs):
        yield StreamEvent(
            type=StreamEventType.ERROR,
            error_code="HTTP_404",
            message=self._message,
        )


async def _last_error(message: str, tmp_path: Path) -> StreamEvent | None:
    """오류만 내는 프로바이더로 query_loop 을 끝까지 돌리고 마지막 ERROR 를 돌려준다."""
    ctx = ToolUseContext(
        cwd=str(tmp_path), session_id="err-reason", permission_mode="bypass_permissions"
    )
    last = None
    async for event in query_loop(
        messages=[Message.user("안녕")],
        system_prompt="s",
        model_provider=_AlwaysErrorProvider(message),
        tools=[],
        context=ctx,
    ):
        if isinstance(event, StreamEvent) and event.type == StreamEventType.ERROR:
            last = event
    return last


@pytest.mark.asyncio
async def test_http_error_is_not_reported_as_context_overflow(tmp_path: Path) -> None:
    """404 는 입력 길이 문제가 아니다 — 사유를 그대로 전달해야 한다."""
    detail = "vLLM 서버 에러: 404 - {'message': 'The model `ax-4.0` does not exist.'}"
    err = await _last_error(detail, tmp_path)

    assert err is not None, "오류가 하나도 보고되지 않았다"
    assert err.error_code == "MODEL_ERROR", f"오진: {err.error_code} / {err.message}"
    assert "404" in (err.message or ""), "실제 사유가 사용자에게 전달되지 않았다"
    assert "너무 길어" not in (err.message or "")


@pytest.mark.asyncio
async def test_context_error_still_reports_context_overflow(tmp_path: Path) -> None:
    """진짜 컨텍스트 초과는 기존 문구를 유지한다(무회귀)."""
    detail = "vLLM 서버 에러: 400 - maximum context length exceeded"
    err = await _last_error(detail, tmp_path)

    assert err is not None
    assert err.error_code == "CONTEXT_OVERFLOW"
    assert "너무 길어" in (err.message or "")
