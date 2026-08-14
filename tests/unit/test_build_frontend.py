# 프런트 분리 배포 빌드 — 경로 치환과 산출물 구성을 고정한다
"""
2026-08-14. 프런트만 떼어 IIS 에 올리는 빌드.

★이 파일이 존재하는 이유는 실제로 낸 사고 때문이다.
  절대경로(`/static/...`)를 상대경로로 바꾸는 정규식에서 **치환 그룹 `\\1` 을
  빠뜨렸다.** 그러면 여는 따옴표가 통째로 사라져

      src="/static/x.js"   →   src=static/x.js"

  로 HTML 이 조용히 깨진다. 게다가 "남은 절대경로" 검사마저 통과해 버려
  (`"/static/` 패턴이 사라졌으므로) **배포한 뒤에야** 404 로 드러난다.
  빌드 산출물은 눈으로 보기 어려우니 이런 것은 테스트로 못 박아야 한다.
"""

from __future__ import annotations

import json

import pytest

from deployment.build_frontend import _to_relative_paths, build


# ─────────────────────────────────────────────
# 경로 치환 — 따옴표를 잃지 않는다
# ─────────────────────────────────────────────
@pytest.mark.parametrize(
    ("src", "expected"),
    [
        ('<script src="/static/a.js"></script>', '<script src="static/a.js"></script>'),
        ("<link href='/static/b.css'>", "<link href='static/b.css'>"),
        ("s.src = '/static/vendor/mermaid/mermaid.min.js';", "s.src = 'static/vendor/mermaid/mermaid.min.js';"),
        ("const u = `/static/c.png`;", "const u = `static/c.png`;"),
    ],
)
def test_relative_rewrite_keeps_quotes(src: str, expected: str) -> None:
    """★따옴표가 남아야 한다. 치환 그룹을 빠뜨리면 여기서 걸린다."""
    assert _to_relative_paths(src) == expected


def test_api_paths_are_untouched() -> None:
    """`/v1/...` 는 fetch 래퍼가 apiBase 를 붙일 자리다 — 여기서 건드리면 안 된다."""
    src = 'fetch("/v1/chat"); fetch("/health");'
    assert _to_relative_paths(src) == src


def test_js_string_paths_are_rewritten() -> None:
    """HTML 속성만 처리하면 Mermaid 동적 로드가 남는다(실제로 남았다)."""
    src = "s.src = '/static/vendor/mermaid/mermaid.min.js';"
    assert "'static/vendor" in _to_relative_paths(src)
    assert "'/static/" not in _to_relative_paths(src)


# ─────────────────────────────────────────────
# 빌드 산출물 구성
# ─────────────────────────────────────────────
@pytest.fixture()
def built(tmp_path):
    out = tmp_path / "front"
    info = build("http://192.168.21.112:8600", "test-key-abc", out)
    return out, info


def test_output_has_required_files(built) -> None:
    out, _ = built
    for rel in ("index.html", "web.config", "README-배포.md", "static/config.js"):
        assert (out / rel).exists(), f"{rel} 이 없다"


def test_config_js_carries_api_base_and_key(built) -> None:
    out, _ = built
    cfg = (out / "static" / "config.js").read_text(encoding="utf-8")
    assert '"http://192.168.21.112:8600"' in cfg
    assert '"test-key-abc"' in cfg


def test_no_absolute_static_paths_remain(built) -> None:
    """하위 경로(/nova/)에 배포해도 동작해야 한다."""
    out, info = built
    html = (out / "index.html").read_text(encoding="utf-8")

    assert info["leftover_abs_paths"] == []
    assert '="/static/' not in html
    assert "'/static/" not in html


def test_html_is_not_corrupted(built) -> None:
    """치환이 따옴표를 먹지 않았는지 — 속성이 정상적으로 닫혀 있어야 한다."""
    out, _ = built
    html = (out / "index.html").read_text(encoding="utf-8")

    # 따옴표를 잃으면 `src=static/...` 형태가 생긴다.
    assert "src=static/" not in html
    assert "href=static/" not in html
    assert 'src="static/vendor/highlight/highlight.min.js"' in html


def test_manifest_paths_are_relative(built) -> None:
    out, _ = built
    data = json.loads((out / "static" / "manifest.json").read_text(encoding="utf-8"))

    assert data["start_url"] == "./"
    for icon in data["icons"]:
        assert not icon["src"].startswith("/")


def test_dev_only_page_is_excluded(built) -> None:
    """chrome.html 은 개발용이라 배포에 넣지 않는다."""
    out, _ = built
    assert not (out / "static" / "chrome.html").exists()


def test_vendor_bundles_are_copied(built) -> None:
    """에어갭이라 CDN 을 못 쓴다 — vendor 가 빠지면 화면이 깨진다."""
    out, _ = built
    for rel in (
        "static/vendor/bootstrap-icons/bootstrap-icons.css",
        "static/vendor/katex/katex.min.js",
        "static/vendor/mermaid/mermaid.min.js",
        "static/vendor/highlight/highlight.min.js",
    ):
        assert (out / rel).exists(), f"{rel} 이 없다"


def test_web_config_registers_font_mime(built) -> None:
    """IIS 는 woff2 MIME 이 빠져 있는 경우가 많다 — 아이콘이 통째로 깨진다."""
    out, _ = built
    cfg = (out / "web.config").read_text(encoding="utf-8")

    assert 'fileExtension=".woff2"' in cfg
    assert "index.html" in cfg          # 기본 문서
    assert "DisableCache" in cfg        # 배포 후 옛 화면 방지
