# RenderPreview 도구 검증 — 입력검증·브라우저 부재 폴백·실렌더(로컬 크롬 있을 때만).
"""
RenderPreviewTool의 계약을 고정한다 (2026-08-04 웹 자가 검증 루프).

  - validate_input: html/htm 외 확장자·빈 경로 거부.
  - check_permissions: ALLOW(자가 검증 루프가 승인에 막히지 않아야 함 — 명시적 완화).
  - 브라우저 부재: 명확한 한글 오류로 비활성 안내(fail-soft).
  - 실렌더: 로컬에 Chrome/Edge가 있으면 실제 스크린샷 PNG 생성까지 확인
    (없는 환경에서는 skip — CI/컨테이너 안전).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from core.tools.base import PermissionBehavior, ToolUseContext
from core.tools.implementations.render_preview_tool import (
    RenderPreviewTool,
    find_browser,
)


def _ctx(tmp_path: Path) -> ToolUseContext:
    # uploads_dir를 tmp로 돌려 실제 업로드 샌드박스를 더럽히지 않는다.
    return ToolUseContext(cwd=".", options={"uploads_dir": str(tmp_path / "uploads")})


def test_validate_rejects_non_html() -> None:
    """html/htm 이외 확장자는 렌더 대상이 아니다(fail-closed)."""
    tool = RenderPreviewTool()
    assert tool.validate_input({"file_path": "a.txt"}) is not None
    assert tool.validate_input({"file_path": ""}) is not None
    assert tool.validate_input({"file_path": "page.html"}) is None


@pytest.mark.asyncio
async def test_permissions_allow(tmp_path) -> None:
    """승인 프롬프트 없이 실행되도록 ALLOW를 반환한다(명시적 완화)."""
    result = await RenderPreviewTool().check_permissions(
        {"file_path": "page.html"}, _ctx(tmp_path)
    )
    assert result.behavior == PermissionBehavior.ALLOW


@pytest.mark.asyncio
async def test_missing_file_errors(tmp_path) -> None:
    """존재하지 않는 html은 명확한 오류로 실패한다."""
    result = await RenderPreviewTool().call(
        {"file_path": str(tmp_path / "no.html")}, _ctx(tmp_path)
    )
    assert result.is_error
    assert "찾을 수 없습니다" in result.error_message


@pytest.mark.asyncio
async def test_no_browser_graceful_error(tmp_path, monkeypatch) -> None:
    """브라우저가 없으면 도구 비활성을 안내한다(fail-soft, 다른 작업 방해 금지)."""
    f = tmp_path / "page.html"
    f.write_text("<html><body>hi</body></html>", encoding="utf-8")
    import core.tools.implementations.render_preview_tool as mod

    monkeypatch.setattr(mod, "find_browser", lambda override=None: None)
    result = await RenderPreviewTool().call({"file_path": str(f)}, _ctx(tmp_path))
    assert result.is_error
    assert "헤드리스 브라우저" in result.error_message


@pytest.mark.asyncio
async def test_broken_browser_no_screenshot_errors(tmp_path, monkeypatch) -> None:
    """브라우저가 스크린샷을 못 만들면(출력 없음) 오류로 처리한다."""
    f = tmp_path / "page.html"
    f.write_text("<html><body>hi</body></html>", encoding="utf-8")
    import core.tools.implementations.render_preview_tool as mod

    # 크롬 플래그를 이해하지 못하는 가짜 브라우저(python) — 파일을 만들지 않는다.
    monkeypatch.setattr(mod, "find_browser", lambda override=None: sys.executable)
    result = await RenderPreviewTool().call({"file_path": str(f)}, _ctx(tmp_path))
    assert result.is_error
    assert "스크린샷 생성에 실패" in result.error_message


@pytest.mark.asyncio
@pytest.mark.skipif(find_browser() is None, reason="로컬 브라우저 없음 — 실렌더 생략")
async def test_real_render_produces_png(tmp_path) -> None:
    """로컬 브라우저가 있으면 실제 렌더로 PNG가 생성되고 안내 문구가 담긴다."""
    f = tmp_path / "page.html"
    f.write_text(
        "<html><body><h1 style='color:red'>RENDER TEST</h1></body></html>",
        encoding="utf-8",
    )
    result = await RenderPreviewTool().call({"file_path": str(f)}, _ctx(tmp_path))

    assert not result.is_error, result.error_message
    shot = Path(result.metadata["screenshot_path"])
    assert shot.is_file() and shot.stat().st_size > 0
    assert shot.suffix == ".png"
    assert "AnalyzeImage" in result.data  # 다음 단계(비전 검토) 안내 포함


# ─── AnalyzeImage 파일명 단독 해석 (RenderPreview 연계, 2026-08-04) ───


@pytest.mark.asyncio
async def test_analyze_image_bare_filename_resolves_to_uploads(tmp_path) -> None:
    """디렉토리 없는 파일명은 업로드 샌드박스 기준으로 해석된다(긴 경로 환각 우회)."""
    from core.tools.implementations.analyze_image_tool import AnalyzeImageTool

    uploads = tmp_path / "uploads"
    uploads.mkdir()
    (uploads / "shot.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 10)
    ctx = ToolUseContext(cwd=".", options={"uploads_dir": str(uploads)})

    path, err = AnalyzeImageTool()._resolve_uploaded_image("shot.png", ctx)
    assert err is None
    assert path == (uploads / "shot.png").resolve()


@pytest.mark.asyncio
async def test_analyze_image_traversal_still_blocked(tmp_path) -> None:
    """상대 성분(../)이나 샌드박스 밖 절대 경로는 종전대로 차단된다(보안 경계 불변)."""
    from core.tools.implementations.analyze_image_tool import AnalyzeImageTool

    uploads = tmp_path / "uploads"
    uploads.mkdir()
    ctx = ToolUseContext(cwd=".", options={"uploads_dir": str(uploads)})
    tool = AnalyzeImageTool()

    path, err = tool._resolve_uploaded_image("../evil.png", ctx)
    assert path is None and "허용된 업로드" in err
    path2, _ = tool._resolve_uploaded_image(r"C:\Windows\evil.png", ctx)
    assert path2 is None
