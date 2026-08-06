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


def _ctx(tmp_path: Path, session_id: str = "") -> ToolUseContext:
    # uploads_dir를 tmp로 돌려 실제 업로드 샌드박스를 더럽히지 않는다.
    return ToolUseContext(
        cwd=".",
        session_id=session_id,
        options={"uploads_dir": str(tmp_path / "uploads")},
    )


def test_validate_rejects_non_html() -> None:
    """html/htm 이외 확장자는 렌더 대상이 아니다(fail-closed)."""
    tool = RenderPreviewTool()
    assert tool.validate_input({"file_path": "a.txt"}) is not None
    assert tool.validate_input({"file_path": ""}) is not None
    assert tool.validate_input({"file_path": "page.html"}) is None


@pytest.mark.asyncio
async def test_permissions_allow(tmp_path) -> None:
    """승인 프롬프트 없이 실행되도록 ALLOW를 반환한다(명시적 완화)."""
    result = await RenderPreviewTool().check_permissions({"file_path": "page.html"}, _ctx(tmp_path))
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


# ─── 자가 검증 루프 종료 조건 (2026-08-07) ───
#
# 왜 도구가 세는가: 시스템 프롬프트에 "최대 2회"라고 써 뒀는데도 실측에서 모델이
# 4회를 돌았다. 지시문은 상한이 아니라 권고로 읽힌다. 아래 테스트들은 "프롬프트를
# 무시해도 못 넘는다"는 성질을 고정한다.


def _stub_browser(monkeypatch) -> None:
    """스크린샷 파일을 실제로 만들어 주는 가짜 브라우저를 심는다."""
    import core.tools.implementations.render_preview_tool as mod

    monkeypatch.setattr(mod, "find_browser", lambda override=None: "fake-browser")

    class _Proc:
        async def wait(self) -> int:
            return 0

        def kill(self) -> None:  # pragma: no cover - 타임아웃 경로 전용
            pass

    async def _fake_exec(*cmd, **_kw):
        # 크롬처럼 --screenshot=<경로> 인자를 보고 그 자리에 PNG를 써 준다.
        for arg in cmd:
            if isinstance(arg, str) and arg.startswith("--screenshot="):
                out = Path(arg.split("=", 1)[1])
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 32)
        return _Proc()

    monkeypatch.setattr("asyncio.create_subprocess_exec", _fake_exec)


def _page(tmp_path: Path, name: str = "page.html") -> Path:
    f = tmp_path / name
    f.write_text("<html><body>hi</body></html>", encoding="utf-8")
    return f


@pytest.mark.asyncio
async def test_render_stops_after_max_rounds(tmp_path, monkeypatch) -> None:
    """★같은 파일은 3회까지만 렌더되고, 4회차는 렌더 없이 막힌다."""
    from core.tools.implementations.render_preview_tool import MAX_RENDER_ROUNDS

    _stub_browser(monkeypatch)
    tool = RenderPreviewTool()
    f = _page(tmp_path)
    ctx = _ctx(tmp_path, session_id="s1")

    for expected in range(1, MAX_RENDER_ROUNDS + 1):
        result = await tool.call({"file_path": str(f)}, ctx)
        assert not result.is_error, result.error_message
        assert result.metadata["render_round"] == expected

    blocked = await tool.call({"file_path": str(f)}, ctx)
    assert blocked.is_error
    assert "상한" in blocked.error_message
    # 막힌 호출은 스크린샷을 주지 않는다 — 줄 게 있으면 모델이 또 검토한다.
    assert "screenshot_path" not in blocked.metadata


@pytest.mark.asyncio
async def test_last_round_result_says_it_is_last(tmp_path, monkeypatch) -> None:
    """마지막 회차 결과는 '다시 렌더할 수 없다'고 미리 알린다."""
    from core.tools.implementations.render_preview_tool import MAX_RENDER_ROUNDS

    _stub_browser(monkeypatch)
    tool = RenderPreviewTool()
    f = _page(tmp_path)
    ctx = _ctx(tmp_path, session_id="s1")

    last = None
    for _ in range(MAX_RENDER_ROUNDS):
        last = await tool.call({"file_path": str(f)}, ctx)

    assert "마지막 렌더" in last.data


@pytest.mark.asyncio
async def test_round_counter_is_per_file(tmp_path, monkeypatch) -> None:
    """다른 파일은 별도로 센다 — 한 파일을 소진해도 다음 작업이 막히면 안 된다."""
    from core.tools.implementations.render_preview_tool import MAX_RENDER_ROUNDS

    _stub_browser(monkeypatch)
    tool = RenderPreviewTool()
    ctx = _ctx(tmp_path, session_id="s1")
    a, b = _page(tmp_path, "a.html"), _page(tmp_path, "b.html")

    for _ in range(MAX_RENDER_ROUNDS):
        await tool.call({"file_path": str(a)}, ctx)

    result = await tool.call({"file_path": str(b)}, ctx)
    assert not result.is_error
    assert result.metadata["render_round"] == 1


@pytest.mark.asyncio
async def test_round_counter_is_per_session(tmp_path, monkeypatch) -> None:
    """세션이 다르면 같은 파일이라도 처음부터 센다(도구 인스턴스는 상주 공유)."""
    from core.tools.implementations.render_preview_tool import MAX_RENDER_ROUNDS

    _stub_browser(monkeypatch)
    tool = RenderPreviewTool()
    f = _page(tmp_path)

    for _ in range(MAX_RENDER_ROUNDS):
        await tool.call({"file_path": str(f)}, _ctx(tmp_path, session_id="s1"))

    result = await tool.call({"file_path": str(f)}, _ctx(tmp_path, session_id="s2"))
    assert not result.is_error
    assert result.metadata["render_round"] == 1


@pytest.mark.asyncio
async def test_missing_file_does_not_consume_round(tmp_path, monkeypatch) -> None:
    """렌더가 일어나지 않은 실패는 회차를 깎지 않는다(오타로 예산을 잃지 않게)."""
    _stub_browser(monkeypatch)
    tool = RenderPreviewTool()
    ctx = _ctx(tmp_path, session_id="s1")

    for _ in range(5):
        await tool.call({"file_path": str(tmp_path / "nope.html")}, ctx)

    result = await tool.call({"file_path": str(_page(tmp_path))}, ctx)
    assert not result.is_error
    assert result.metadata["render_round"] == 1


def test_round_cache_is_bounded() -> None:
    """카운터가 무한히 쌓이지 않는다 — 도구는 프로세스 내내 상주한다."""
    from core.tools.implementations.render_preview_tool import _ROUND_CACHE_MAX

    tool = RenderPreviewTool()
    for i in range(_ROUND_CACHE_MAX * 2):
        tool._next_round(f"s{i}", f"/tmp/p{i}.html")

    assert len(tool._round_counts) <= _ROUND_CACHE_MAX


def test_verify_question_asks_for_description_not_verdict() -> None:
    """VLM에는 판정이 아니라 묘사를 요구한다.

    실측 근거: 이 등급 VLM은 판정 질문에 긍정 편향을 보인다(파손 화면을 정상이라
    답함). 반대로 평가를 빼고 묘사만 시키면 정상/파손을 정확히 구별해 묘사했다.
    """
    from core.tools.implementations.render_preview_tool import VERIFY_QUESTION

    assert "묘사" in VERIFY_QUESTION
    assert "판단이나 평가" in VERIFY_QUESTION  # 판정을 명시적으로 금지


# ─── 자산 점검 — CSS 미적용을 모델 없이 확정한다 ───
#
# 자가 검증 루프를 만들게 한 최초 사고가 "스타일 미적용"이었는데, 그건 눈으로
# 볼 필요가 없다. HTML이 가리키는 파일이 있는지 보면 끝난다. 비전 모델의 의견과
# 달리 이 판정은 틀리지 않는다.


def test_asset_check_passes_when_files_exist(tmp_path) -> None:
    """참조한 css/js가 실제로 있으면 누락 없음."""
    from core.tools.implementations.render_preview_tool import check_local_assets

    (tmp_path / "styles.css").write_text("body{}", encoding="utf-8")
    (tmp_path / "app.js").write_text("//", encoding="utf-8")
    page = tmp_path / "index.html"
    page.write_text(
        '<link rel="stylesheet" href="styles.css"><script src="app.js"></script>',
        encoding="utf-8",
    )

    missing, sheets = check_local_assets(page)
    assert missing == []
    assert sheets == 1


def test_asset_check_flags_missing_stylesheet_file(tmp_path) -> None:
    """★css를 만들지 않고 링크만 걸었을 때 — 실제로 겪은 '스타일 미적용'."""
    from core.tools.implementations.render_preview_tool import check_local_assets

    page = tmp_path / "index.html"
    page.write_text('<link rel="stylesheet" href="styles.css">', encoding="utf-8")

    missing, sheets = check_local_assets(page)
    assert missing == ["styles.css"]
    assert sheets == 1


def test_asset_check_counts_zero_when_no_stylesheet(tmp_path) -> None:
    """스타일시트 링크 자체가 없으면 개수 0 — 기본 스타일로 렌더된다는 뜻."""
    from core.tools.implementations.render_preview_tool import check_local_assets

    page = tmp_path / "index.html"
    page.write_text("<html><body>hi</body></html>", encoding="utf-8")

    missing, sheets = check_local_assets(page)
    assert missing == []
    assert sheets == 0


def test_asset_check_ignores_external_and_data_urls(tmp_path) -> None:
    """외부 URL·data URI는 존재를 물을 대상이 아니다(에어갭에선 어차피 안 뜬다)."""
    from core.tools.implementations.render_preview_tool import check_local_assets

    page = tmp_path / "index.html"
    page.write_text(
        '<link rel="stylesheet" href="https://cdn.example.com/a.css">'
        '<img src="data:image/png;base64,AAA">'
        '<script src="//cdn.example.com/b.js"></script>',
        encoding="utf-8",
    )

    missing, sheets = check_local_assets(page)
    assert missing == []
    assert sheets == 1  # 링크는 세지만 파일 존재는 묻지 않는다


def test_asset_check_strips_query_string(tmp_path) -> None:
    """캐시 버스터(styles.css?v=2)가 붙어도 같은 파일로 본다."""
    from core.tools.implementations.render_preview_tool import check_local_assets

    (tmp_path / "styles.css").write_text("body{}", encoding="utf-8")
    page = tmp_path / "index.html"
    page.write_text('<link rel="stylesheet" href="styles.css?v=2">', encoding="utf-8")

    assert check_local_assets(page)[0] == []


def test_asset_check_ignores_non_stylesheet_links(tmp_path) -> None:
    """favicon 같은 link는 스타일시트로 세지 않는다."""
    from core.tools.implementations.render_preview_tool import check_local_assets

    page = tmp_path / "index.html"
    page.write_text('<link rel="icon" href="favicon.ico">', encoding="utf-8")

    missing, sheets = check_local_assets(page)
    assert sheets == 0
    assert missing == []  # 스타일시트가 아닌 link의 href는 존재를 묻지 않는다


@pytest.mark.asyncio
async def test_render_result_reports_missing_assets(tmp_path, monkeypatch) -> None:
    """렌더 결과에 자산 누락이 '확정' 사실로 실린다 — 화면 인상보다 우선한다."""
    _stub_browser(monkeypatch)
    page = tmp_path / "index.html"
    page.write_text('<link rel="stylesheet" href="styles.css">', encoding="utf-8")

    result = await RenderPreviewTool().call({"file_path": str(page)}, _ctx(tmp_path))

    assert not result.is_error
    assert result.metadata["missing_assets"] == ["styles.css"]
    assert "자산 누락(확정)" in result.data


@pytest.mark.asyncio
async def test_render_result_reports_no_stylesheet(tmp_path, monkeypatch) -> None:
    """스타일시트 링크가 아예 없을 때도 확정 사실로 알린다."""
    _stub_browser(monkeypatch)
    result = await RenderPreviewTool().call({"file_path": str(_page(tmp_path))}, _ctx(tmp_path))

    assert "스타일시트 참조 없음(확정)" in result.data
