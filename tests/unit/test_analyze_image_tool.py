# AnalyzeImageTool 단위 테스트 — 성공 시 분석텍스트, 연결오류/잘못된 경로 시 error 검증.
"""
AnalyzeImageTool 단위 테스트.

VLM 서버(Gemma 4 12B)는 아직 미기동이므로 httpx.AsyncClient 를 monkeypatch 로 mock 한다
(실제 GPU 서버 호출 금지 — testing.md Mock 전략). 아래 축을 검증한다:
  - 성공 응답 mock → choices[0].message.content(분석 텍스트) 반환 + 요청 계약(OpenAI 비전 형식).
  - 연결 오류 mock → <tool_use_error> 로 래핑된 ToolResult.error(bare except 금지 확인).
  - 잘못된 경로(업로드 밖/미존재) → ToolResult.error + check_permissions DENY.
  - 스키마 필수 필드 + fail-closed behavior flag(RO=True, CS=False).

asyncio_mode="auto"(pyproject) 라 async def test_* 는 데코레이터 없이 동작한다.
"""

from __future__ import annotations

from pathlib import Path

import httpx

from core.tools.base import PermissionBehavior, ToolUseContext
from core.tools.implementations.analyze_image_tool import AnalyzeImageTool

# 최소 유효 PNG(1x1) 바이트 — 테스트에서 업로드 이미지 파일로 저장한다.
_PNG_1X1 = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4"
    "890000000a49444154789c6360000002000154a24f5f0000000049454e44ae426082"
)


class _FakeResponse:
    """httpx.Response 를 흉내내는 최소 mock — json()/raise_for_status()만 제공."""

    def __init__(self, payload: dict):
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


class _FakeClient:
    """OpenAI 비전 응답을 돌려주는 httpx.AsyncClient mock (async context manager)."""

    def __init__(self, *args, **kwargs):
        self.captured: dict = {}

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def post(self, url: str, json: dict):  # noqa: A002 — httpx 인자명 그대로
        self.captured["url"] = url
        self.captured["json"] = json
        return _FakeResponse(
            {"choices": [{"message": {"content": "빨간 사과가 테이블 위에 있습니다."}}]}
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


def _make_image(tmp_path: Path, name: str = "photo.png") -> Path:
    """업로드 디렉토리(tmp_path) 안에 유효한 PNG 파일을 만들어 경로를 돌려준다."""
    p = tmp_path / name
    p.write_bytes(_PNG_1X1)
    return p


def _ctx(tmp_path: Path) -> ToolUseContext:
    """uploads_dir·vision_url·vision_model 을 주입한 도구 컨텍스트."""
    return ToolUseContext(
        cwd=str(tmp_path),
        options={
            "uploads_dir": str(tmp_path),
            "vision_url": "http://127.0.0.1:8004",
            "vision_model": "gemma-4-12b",
        },
    )


# ─────────────────────────────────────────────
# 성공 경로
# ─────────────────────────────────────────────
async def test_analyze_returns_text_and_metadata(tmp_path: Path, monkeypatch):
    """성공 응답 mock 시 분석 텍스트 + 메타데이터(model/image_path/question) 반환."""
    monkeypatch.setattr(httpx, "AsyncClient", _FakeClient)
    img = _make_image(tmp_path)

    tool = AnalyzeImageTool()
    result = await tool.call(
        {"image_path": str(img), "question": "무엇이 보이나요?"},
        _ctx(tmp_path),
    )

    assert not result.is_error, result.error_message
    assert result.data == "빨간 사과가 테이블 위에 있습니다."
    meta = result.metadata
    assert meta["model"] == "gemma-4-12b"
    assert meta["question"] == "무엇이 보이나요?"
    assert Path(meta["image_path"]).name == "photo.png"


async def test_analyze_requests_openai_vision_contract(tmp_path: Path, monkeypatch):
    """VLM 서버 계약(경로/필드 = OpenAI 비전 형식)대로 요청하는지 확인한다."""
    fake = _FakeClient()
    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **k: fake)
    img = _make_image(tmp_path)

    tool = AnalyzeImageTool()
    await tool.call({"image_path": str(img)}, _ctx(tmp_path))

    assert fake.captured["url"] == "http://127.0.0.1:8004/v1/chat/completions"
    body = fake.captured["json"]
    assert body["model"] == "gemma-4-12b"
    assert body["temperature"] == 0.2
    assert body["max_tokens"] == 1024  # DEFAULT_MAX_TOKENS
    content = body["messages"][0]["content"]
    # 기본 질문 + 이미지(data URL) 두 파트가 순서대로 들어간다.
    assert content[0] == {"type": "text", "text": "이 이미지를 자세히 설명해줘."}
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")


# ─────────────────────────────────────────────
# 오류 경로
# ─────────────────────────────────────────────
async def test_analyze_connection_error_returns_tool_error(tmp_path: Path, monkeypatch):
    """연결 오류 mock 시 ToolResult.error + <tool_use_error> 래핑."""
    monkeypatch.setattr(httpx, "AsyncClient", _ConnErrorClient)
    img = _make_image(tmp_path)

    tool = AnalyzeImageTool()
    result = await tool.call({"image_path": str(img)}, _ctx(tmp_path))

    assert result.is_error
    assert "연결" in result.error_message
    assert tool.map_result(result).startswith("<tool_use_error>")


async def test_analyze_path_outside_uploads_rejected(tmp_path: Path):
    """업로드 디렉토리 밖 경로는 거부한다(경로 순회/임의 파일 읽기 차단)."""
    # 업로드 디렉토리는 tmp_path/uploads 로, 이미지는 그 바깥(tmp_path)에 둔다.
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    outside = tmp_path / "secret.png"
    outside.write_bytes(_PNG_1X1)
    ctx = ToolUseContext(cwd=str(tmp_path), options={"uploads_dir": str(uploads)})

    tool = AnalyzeImageTool()
    # call() 은 error 반환.
    result = await tool.call({"image_path": str(outside)}, ctx)
    assert result.is_error
    assert "업로드 디렉토리" in result.error_message
    # check_permissions 도 DENY.
    perm = await tool.check_permissions({"image_path": str(outside)}, ctx)
    assert perm.behavior == PermissionBehavior.DENY


async def test_analyze_missing_file_rejected(tmp_path: Path):
    """업로드 디렉토리 하위지만 존재하지 않는 파일은 거부한다."""
    ctx = _ctx(tmp_path)
    tool = AnalyzeImageTool()
    result = await tool.call({"image_path": str(tmp_path / "nope.png")}, ctx)
    assert result.is_error
    assert "찾을 수 없습니다" in result.error_message


async def test_analyze_unsupported_extension_rejected(tmp_path: Path):
    """지원하지 않는 확장자(.gif 등)는 거부한다."""
    bad = tmp_path / "anim.gif"
    bad.write_bytes(_PNG_1X1)
    tool = AnalyzeImageTool()
    result = await tool.call({"image_path": str(bad)}, _ctx(tmp_path))
    assert result.is_error
    assert "지원하지 않는" in result.error_message


async def test_analyze_oversized_image_rejected(tmp_path: Path):
    """크기 상한(기본 10MB)을 넘는 이미지는 거부한다(전송 지연·토큰 폭증 방지)."""
    big = tmp_path / "huge.png"
    # options 로 상한을 1MB 로 낮추고, 그보다 큰 더미 파일(2MB)을 만든다.
    big.write_bytes(b"\x00" * (2 * 1024 * 1024))
    ctx = ToolUseContext(
        cwd=str(tmp_path),
        options={"uploads_dir": str(tmp_path), "vision_max_image_mb": 1},
    )
    tool = AnalyzeImageTool()
    result = await tool.call({"image_path": str(big)}, ctx)
    assert result.is_error
    assert "너무 큽니다" in result.error_message
    # check_permissions 도 DENY(권한 레이어에서 fail-closed).
    perm = await tool.check_permissions({"image_path": str(big)}, ctx)
    assert perm.behavior == PermissionBehavior.DENY


# ─────────────────────────────────────────────
# 입력 검증
# ─────────────────────────────────────────────
def test_validate_rejects_empty_path():
    """빈 image_path 는 거부한다."""
    tool = AnalyzeImageTool()
    assert tool.validate_input({"image_path": "   "}) is not None


def test_schema_and_flags():
    """스키마 필수 필드 + fail-closed behavior flag 확인."""
    tool = AnalyzeImageTool()
    schema = tool.input_schema
    assert schema["required"] == ["image_path"]
    assert "question" in schema["properties"]
    # 읽기 전용(부작용 없음) + 순차 실행(GPU 무거움).
    assert tool.is_read_only is True
    assert tool.is_concurrency_safe is False
    assert tool.name == "AnalyzeImage"
    assert "DescribeImage" in tool.aliases
