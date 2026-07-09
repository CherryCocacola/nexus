# ImageGenerateTool 단위 테스트 — 성공 시 PNG 저장+URL, 연결오류 시 tool_use_error 검증.
"""
ImageGenerateTool 단위 테스트.

이미지 서버(FLUX)는 아직 미기동이므로 httpx.AsyncClient 를 monkeypatch 로 mock 한다
(실제 GPU 서버 호출 금지 — testing.md Mock 전략). 아래 두 축을 검증한다:
  - 성공 응답 mock → PNG 파일이 exports 디렉토리에 저장되고 다운로드/미리보기 URL 반환.
  - 연결 오류 mock → <tool_use_error> 로 래핑된 ToolResult.error 반환(bare except 금지 확인).

asyncio_mode="auto"(pyproject) 라 async def test_* 는 데코레이터 없이 동작한다.
"""

from __future__ import annotations

import base64
from pathlib import Path

import httpx

from core.tools.base import ToolUseContext
from core.tools.implementations.image_generate_tool import ImageGenerateTool

# 최소 유효 PNG(1x1) 바이트 — 테스트에서 서버 응답 image_base64 로 돌려준다.
_PNG_1X1 = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4"
    "890000000a49444154789c6360000002000154a24f5f0000000049454e44ae426082"
)
_PNG_B64 = base64.b64encode(_PNG_1X1).decode("ascii")


class _FakeResponse:
    """httpx.Response 를 흉내내는 최소 mock — json()/raise_for_status()만 제공."""

    def __init__(self, payload: dict):
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


class _FakeClient:
    """성공 응답을 돌려주는 httpx.AsyncClient mock (async context manager)."""

    def __init__(self, *args, **kwargs):
        self.captured: dict = {}

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def post(self, url: str, json: dict):  # noqa: A002 — httpx 인자명 그대로
        # 요청 URL·페이로드를 저장해 계약(경로/필드)을 검증할 수 있게 한다.
        self.captured["url"] = url
        self.captured["json"] = json
        return _FakeResponse(
            {
                "image_base64": _PNG_B64,
                "width": json["width"],
                "height": json["height"],
                "seed": json.get("seed") or 12345,
                "model": "flux-schnell",
            }
        )


class _ConnErrorClient:
    """연결 오류를 던지는 httpx.AsyncClient mock."""

    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def post(self, url: str, json: dict):  # noqa: A002
        raise httpx.ConnectError("connection refused")


def _ctx(tmp_path: Path) -> ToolUseContext:
    """exports_dir·image_url 을 주입한 도구 컨텍스트."""
    return ToolUseContext(
        cwd=str(tmp_path),
        options={"exports_dir": str(tmp_path), "image_url": "http://127.0.0.1:8003"},
    )


# ─────────────────────────────────────────────
# 성공 경로
# ─────────────────────────────────────────────
async def test_generate_saves_png_and_returns_urls(tmp_path: Path, monkeypatch):
    """성공 응답 mock 시 PNG 저장 + download/preview URL + 메타데이터 반환."""
    monkeypatch.setattr(httpx, "AsyncClient", _FakeClient)

    tool = ImageGenerateTool()
    result = await tool.call(
        {"prompt": "a red apple on a table", "size": "1024x1536", "seed": 7},
        _ctx(tmp_path),
    )

    assert not result.is_error, result.error_message
    meta = result.metadata
    # 메타데이터 계약 — DocumentExport와 대칭 + 이미지 전용 필드.
    assert meta["download_url"].startswith("/v1/download/")
    assert meta["preview_url"] == meta["download_url"]
    assert meta["width"] == 1024 and meta["height"] == 1536
    assert meta["seed"] == 7
    assert meta["model"] == "flux-schnell"
    assert meta["bytes"] > 0

    # 실제 PNG 파일이 exports 디렉토리(tmp_path) 안에 생성됐는지.
    out = tmp_path / meta["filename"]
    assert out.is_file()
    assert out.read_bytes()[:8] == _PNG_1X1[:8]  # PNG 시그니처
    assert out.parent == tmp_path


async def test_generate_requests_correct_contract(tmp_path: Path, monkeypatch):
    """이미지 서버 계약(경로/필드)대로 요청하는지 확인한다."""
    fake = _FakeClient()
    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **k: fake)

    tool = ImageGenerateTool()
    await tool.call({"prompt": "cat", "size": "1536x1024"}, _ctx(tmp_path))

    assert fake.captured["url"] == "http://127.0.0.1:8003/v1/images/generate"
    body = fake.captured["json"]
    assert body["prompt"] == "cat"
    assert body["width"] == 1536 and body["height"] == 1024
    assert body["steps"] == 4  # DEFAULT_STEPS(FLUX schnell)
    assert body["seed"] is None  # 미지정 → null


# ─────────────────────────────────────────────
# 오류 경로
# ─────────────────────────────────────────────
async def test_generate_connection_error_returns_tool_error(tmp_path: Path, monkeypatch):
    """연결 오류 mock 시 ToolResult.error + <tool_use_error> 래핑."""
    monkeypatch.setattr(httpx, "AsyncClient", _ConnErrorClient)

    tool = ImageGenerateTool()
    result = await tool.call({"prompt": "dog"}, _ctx(tmp_path))

    assert result.is_error
    assert "연결" in result.error_message
    # map_result 가 에러를 <tool_use_error> 로 감싸는지(계약).
    assert result.data == result.error_message
    assert tool.map_result(result).startswith("<tool_use_error>")


# ─────────────────────────────────────────────
# 입력 검증
# ─────────────────────────────────────────────
def test_validate_rejects_empty_prompt():
    """빈 prompt 는 거부한다."""
    tool = ImageGenerateTool()
    assert tool.validate_input({"prompt": "   "}) is not None


def test_validate_rejects_unknown_size():
    """지원하지 않는 size 는 거부한다."""
    tool = ImageGenerateTool()
    err = tool.validate_input({"prompt": "x", "size": "999x999"})
    assert err and "지원하지 않는" in err


def test_schema_and_flags():
    """스키마 필수 필드 + fail-closed behavior flag 확인."""
    tool = ImageGenerateTool()
    schema = tool.input_schema
    assert schema["required"] == ["prompt"]
    assert set(schema["properties"]["size"]["enum"]) == {
        "1024x1024",
        "1024x1536",
        "1536x1024",
    }
    # fail-closed 기본값 — 쓰기 도구 + 순차 실행.
    assert tool.is_read_only is False
    assert tool.is_concurrency_safe is False
    assert tool.name == "ImageGenerate"
