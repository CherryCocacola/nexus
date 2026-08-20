"""
DocumentExportTool 단위 테스트 — 포맷 5종 생성 + 보안(파일명 정화) 검증.

asyncio_mode="auto"(pyproject) 라 async def test_* 는 데코레이터 없이 동작한다.
실제 라이브러리(python-docx/pptx/hwpx)로 파일을 만들어 유효성을 확인한다(mock 금지).
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from core.tools.base import ToolUseContext
from core.tools.implementations.document_export_renderers import parse_blocks
from core.tools.implementations.document_export_tool import (
    DocumentExportTool,
    _safe_filename,
    resolve_exports_dir,
)

SAMPLE = """# 강원대 AI 도입 요약

## 핵심 서비스
- 시간표 자동 작성
- **동적 데이터 분석** (Text-to-SQL)

본문 문단입니다.
"""


def _ctx(tmp_path: Path) -> ToolUseContext:
    """exports_dir 을 tmp_path 로 주입한 도구 컨텍스트."""
    return ToolUseContext(cwd=str(tmp_path), options={"exports_dir": str(tmp_path)})


# ─────────────────────────────────────────────
# 포맷별 생성
# ─────────────────────────────────────────────
@pytest.mark.parametrize("fmt", ["md", "txt", "docx", "pptx", "hwpx"])
async def test_export_generates_valid_file(fmt: str, tmp_path: Path):
    """5개 포맷 각각 실제 파일이 생성되고 유효한지 검증한다."""
    tool = DocumentExportTool()
    result = await tool.call(
        {"content": SAMPLE, "format": fmt, "filename": "요약", "title": "테스트"},
        _ctx(tmp_path),
    )
    assert not result.is_error, result.error_message

    # 메타데이터 계약 확인
    meta = result.metadata
    assert meta["format"] == fmt
    assert meta["download_url"].startswith("/v1/download/")
    assert meta["bytes"] > 0

    # 실제 파일 존재 + 유효성
    out = tmp_path / meta["filename"]
    assert out.is_file()
    if fmt in ("docx", "pptx", "hwpx"):
        assert zipfile.is_zipfile(out)  # OOXML/OWPML 은 ZIP 컨테이너
    else:
        assert "동적 데이터 분석" in out.read_text(encoding="utf-8")


# ─────────────────────────────────────────────
# 입력 검증
# ─────────────────────────────────────────────
def test_validate_rejects_unknown_format():
    """미지원 포맷은 validate_input 이 거부한다."""
    tool = DocumentExportTool()
    err = tool.validate_input({"content": "x", "format": "pdf"})
    assert err and "지원하지 않는" in err


def test_validate_rejects_empty_content():
    """빈 content 는 거부한다."""
    tool = DocumentExportTool()
    err = tool.validate_input({"content": "   ", "format": "md"})
    assert err is not None


# ─────────────────────────────────────────────
# 보안: 파일명 정화 / 경로 순회 차단
# ─────────────────────────────────────────────
def test_safe_filename_strips_traversal():
    """../ 등 경로 성분이 제거되고 확장자·uuid 가 붙는지 확인한다."""
    name = _safe_filename("../../etc/passwd", "docx")
    assert "/" not in name and ".." not in name
    assert name.endswith(".docx")


async def test_export_never_escapes_exports_dir(tmp_path: Path):
    """악의적 filename 을 줘도 파일은 exports 디렉토리 안에만 생성된다."""
    tool = DocumentExportTool()
    result = await tool.call(
        {"content": SAMPLE, "format": "txt", "filename": "../../evil"},
        _ctx(tmp_path),
    )
    assert not result.is_error
    out = tmp_path / result.metadata["filename"]
    # exports 디렉토리(tmp_path) 밖에는 아무 것도 안 생겼는지 — out 은 tmp_path 하위여야.
    assert out.parent == tmp_path
    assert out.exists()
    assert "/v1/download/" in result.metadata["download_url"]
    assert ".." not in result.metadata["download_url"]


# ─────────────────────────────────────────────
# 헬퍼
# ─────────────────────────────────────────────
def test_resolve_exports_dir_default_fallback():
    """설정 미제공 시 tempdir/nexus_exports 로 폴백하고 디렉토리를 만든다."""
    p = resolve_exports_dir(None)
    assert p.is_dir()
    assert p.name == "nexus_exports"


def test_parse_blocks_structure():
    """마크다운 파싱이 제목/글머리표/문단을 올바르게 분류하는지."""
    blocks = parse_blocks(SAMPLE)
    kinds = [b[0] for b in blocks]
    assert "h" in kinds and "bullet" in kinds and "p" in kinds
    # 글머리표 묶음에 2개 항목
    bullet = next(b for b in blocks if b[0] == "bullet")
    assert len(bullet[1]) == 2


# ─────────────────────────────────────────────
# 표면별 결과 문구 (2026-08-19)
#   웹은 서버가 다운로드 버튼을 붙여 주므로 URL·파일명 재현을 금지하고(긴 UUID 재현이
#   degeneration 의 방아쇠였다), CLI 는 붙여 줄 UI 가 없으므로 저장 경로를 알려야 한다.
# ─────────────────────────────────────────────
async def test_export_message_hides_url_when_download_ui(tmp_path: Path):
    """download_ui 미지정(=웹 기본)이면 URL 재현 금지 문구가 붙고 경로는 노출하지 않는다."""
    tool = DocumentExportTool()
    result = await tool.call(
        {"content": SAMPLE, "format": "md", "filename": "요약"},
        _ctx(tmp_path),  # download_ui 미주입 → 기본 True
    )
    assert not result.is_error, result.error_message
    body = result.data
    assert "노출 금지" in body
    assert "저장 위치" not in body
    assert str(tmp_path) not in body


async def test_export_message_shows_path_without_download_ui(tmp_path: Path):
    """download_ui=False(=CLI)면 절대경로를 알려주고 URL 은 문구에 넣지 않는다."""
    tool = DocumentExportTool()
    ctx = ToolUseContext(
        cwd=str(tmp_path),
        options={"exports_dir": str(tmp_path), "download_ui": False},
    )
    result = await tool.call(
        {"content": SAMPLE, "format": "md", "filename": "요약"}, ctx
    )
    assert not result.is_error, result.error_message
    body = result.data
    assert "저장 위치" in body
    assert str(tmp_path) in body
    assert "/v1/download/" not in body  # CLI 에는 그 링크가 무의미하다
    # 메타데이터에는 표면과 무관하게 둘 다 남는다(웹 추출·CLI 안내 양쪽 계약).
    assert result.metadata["download_url"].startswith("/v1/download/")
    assert result.metadata["path"].endswith(result.metadata["filename"])
