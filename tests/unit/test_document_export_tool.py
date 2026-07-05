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
