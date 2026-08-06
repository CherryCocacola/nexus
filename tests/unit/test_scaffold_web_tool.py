# ScaffoldWeb 도구 검증 — 카탈로그 조회·결정적 복사·충돌/순회 차단.
"""
ScaffoldWebTool 계약 고정 (2026-08-05 템플릿 그라운딩).

  - 인자 없음 → 카탈로그 목록(ALLOW, 읽기 전용).
  - 스캐폴드 → 파일이 결정적으로 복사되고, 커스터마이징 안내가 결과에 담긴다.
  - 충돌: 기존 파일 존재 시 overwrite=true 없이는 아무것도 복사하지 않는다.
  - 보안: target_dir이 작업 디렉토리 밖이면 거부(fail-closed).
  - 실자산 검증: 리포의 landing-vue 템플릿이 카탈로그와 일치하는지.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.tools.base import PermissionBehavior, ToolUseContext
from core.tools.implementations.scaffold_web_tool import (
    _DEFAULT_TEMPLATES_DIR,
    ScaffoldWebTool,
)


def _make_templates(root: Path) -> Path:
    """테스트용 템플릿 루트(mini 카탈로그 + 파일)를 만든다."""
    tdir = root / "templates"
    (tdir / "mini" / "vendor").mkdir(parents=True)
    (tdir / "mini" / "index.html").write_text("<html>MINI</html>", encoding="utf-8")
    (tdir / "mini" / "app.js").write_text("const SITE = {};", encoding="utf-8")
    (tdir / "mini" / "vendor" / "lib.js").write_text("// lib", encoding="utf-8")
    (tdir / "catalog.json").write_text(
        json.dumps(
            {
                "templates": [
                    {
                        "name": "mini",
                        "description": "테스트 템플릿",
                        "files": ["index.html", "app.js", "vendor/lib.js"],
                        "customize": "app.js SITE만 수정",
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return tdir


def _ctx(cwd: Path, templates: Path) -> ToolUseContext:
    return ToolUseContext(
        cwd=str(cwd), options={"frontend_templates_dir": str(templates)}
    )


@pytest.mark.asyncio
async def test_list_returns_catalog(tmp_path) -> None:
    """인자 없이 호출하면 템플릿 목록과 다음 단계 안내를 돌려준다."""
    tdir = _make_templates(tmp_path)
    result = await ScaffoldWebTool().call({}, _ctx(tmp_path, tdir))

    assert not result.is_error
    assert "mini" in result.data
    assert "ScaffoldWeb(template=" in result.data


@pytest.mark.asyncio
async def test_list_permission_allow_scaffold_ask(tmp_path) -> None:
    """조회는 ALLOW, 스캐폴드는 ASK(파일 쓰기 정책)."""
    tdir = _make_templates(tmp_path)
    tool = ScaffoldWebTool()
    r1 = await tool.check_permissions({}, _ctx(tmp_path, tdir))
    r2 = await tool.check_permissions(
        {"template": "mini", "target_dir": "out"}, _ctx(tmp_path, tdir)
    )
    assert r1.behavior == PermissionBehavior.ALLOW
    assert r2.behavior == PermissionBehavior.ASK


@pytest.mark.asyncio
async def test_scaffold_copies_files(tmp_path) -> None:
    """스캐폴드는 카탈로그의 파일들을 그대로 복사하고 안내를 담는다."""
    tdir = _make_templates(tmp_path)
    result = await ScaffoldWebTool().call(
        {"template": "mini", "target_dir": "site"}, _ctx(tmp_path, tdir)
    )

    assert not result.is_error, result.error_message
    out = tmp_path / "site"
    assert (out / "index.html").read_text(encoding="utf-8") == "<html>MINI</html>"
    assert (out / "vendor" / "lib.js").is_file()
    assert "SITE" in result.data  # 커스터마이징 안내 포함
    assert "RenderPreview" in result.data  # 검증 다음 단계 안내


@pytest.mark.asyncio
async def test_scaffold_collision_blocked_without_overwrite(tmp_path) -> None:
    """기존 파일이 있으면 overwrite=true 없이 아무것도 복사하지 않는다."""
    tdir = _make_templates(tmp_path)
    out = tmp_path / "site"
    out.mkdir()
    (out / "index.html").write_text("KEEP", encoding="utf-8")

    result = await ScaffoldWebTool().call(
        {"template": "mini", "target_dir": "site"}, _ctx(tmp_path, tdir)
    )

    assert result.is_error
    assert "overwrite" in result.error_message
    assert (out / "index.html").read_text(encoding="utf-8") == "KEEP"  # 보존
    assert not (out / "app.js").exists()  # 부분 복사도 없음


@pytest.mark.asyncio
async def test_scaffold_outside_cwd_rejected(tmp_path) -> None:
    """작업 디렉토리 밖 target_dir은 거부한다(경로 순회 차단)."""
    tdir = _make_templates(tmp_path)
    cwd = tmp_path / "work"
    cwd.mkdir()
    result = await ScaffoldWebTool().call(
        {"template": "mini", "target_dir": str(tmp_path / "escape")},
        _ctx(cwd, tdir),
    )
    assert result.is_error
    assert "작업 디렉토리 하위" in result.error_message


@pytest.mark.asyncio
async def test_unknown_template_lists_available(tmp_path) -> None:
    """없는 템플릿 이름이면 사용 가능한 목록을 알려준다."""
    tdir = _make_templates(tmp_path)
    result = await ScaffoldWebTool().call(
        {"template": "nope", "target_dir": "site"}, _ctx(tmp_path, tdir)
    )
    assert result.is_error
    assert "mini" in result.error_message


def _repo_catalog() -> list[dict]:
    """리포의 실제 카탈로그 항목들."""
    catalog = json.loads(
        (_DEFAULT_TEMPLATES_DIR / "catalog.json").read_text(encoding="utf-8")
    )
    return catalog["templates"]


def test_repo_catalog_files_exist() -> None:
    """카탈로그에 적힌 파일이 리포에 실제로 있어야 한다(모든 템플릿).

    카탈로그만 고치고 자산을 안 넣으면 스캐폴드가 런타임에 실패한다 — 그 불일치를 막는다.
    """
    entries = _repo_catalog()
    assert len(entries) >= 4, "템플릿이 줄었는지 확인 필요"
    for entry in entries:
        for rel in entry["files"]:
            f = _DEFAULT_TEMPLATES_DIR / entry["name"] / rel
            assert f.is_file() and f.stat().st_size > 0, f"{entry['name']}: 누락/빈 파일 {rel}"


def test_repo_templates_expose_single_edit_slot() -> None:
    """모든 템플릿이 'app.js의 SITE 하나만 고치면 된다'는 규약을 지켜야 한다.

    이 규약이 곧 품질이다. 모델이 여러 파일을 헤집기 시작하면 결과가 무너진다.
    색상까지 SITE.colors 로 옮긴 이유는 styles.css 의 :root 를 '교체'하는 대신
    새 블록을 덧붙이는 실수가 잦았기 때문이다(CSS는 뒤 규칙이 이겨 옛 색이 남았다).
    """
    for entry in _repo_catalog():
        app_js = (_DEFAULT_TEMPLATES_DIR / entry["name"] / "app.js").read_text(
            encoding="utf-8"
        )
        name = entry["name"]
        assert "const SITE" in app_js, f"{name}: SITE 슬롯 없음"
        assert "colors:" in app_js, f"{name}: 색상이 SITE 안에 없음"
        # 색을 실제로 CSS 변수에 주입해야 styles.css 를 건드릴 필요가 사라진다.
        assert "setProperty" in app_js, f"{name}: 색상 주입 코드 없음"


def test_repo_catalog_entries_guide_selection() -> None:
    """카탈로그 항목이 '언제 쓰는지'를 알려줘야 모델이 유형을 고를 수 있다."""
    for entry in _repo_catalog():
        assert entry.get("description"), f"{entry['name']}: description 없음"
        assert entry.get("use_when"), f"{entry['name']}: use_when 없음"
        assert entry.get("customize"), f"{entry['name']}: customize 없음"
