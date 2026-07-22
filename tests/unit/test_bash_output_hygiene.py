# Bash 나열 명령 출력 위생(P0-a) 검증 — node_modules 등 노이즈 경로 프루닝.
"""
`bash_tool`의 출력 위생 로직을 검증한다.

배경(FABLE5 진단, 2026-07-22): 재귀 나열 명령(ls -R·find)이 node_modules를
통째로 뱉어 37,099토큰이 컨텍스트를 오염시켰고, 모델이 .js 파일명만 보고
백엔드를 "Node.js"로 오판했다(실제 FastAPI). 나열 명령의 노이즈 경로 줄을
제거하고 작은 상한을 적용하는 로직이 이 오염을 막는지 확인한다.
"""

from __future__ import annotations

from core.tools.implementations.bash_tool import (
    _LISTING_MAX_SIZE,
    _first_token,
    _prune_listing_output,
    _truncate_output,
)

# ── _first_token — 나열 명령 판별 ────────────────────────────────────────


def test_first_token_plain_commands():
    assert _first_token("ls -R") == "ls"
    assert _first_token("find . -name '*.py'") == "find"
    assert _first_token("tree -L 2") == "tree"


def test_first_token_strips_path_and_case():
    assert _first_token("/usr/bin/ls -la") == "ls"
    assert _first_token("Get-ChildItem -Recurse") == "get-childitem"


def test_first_token_empty():
    assert _first_token("") == ""
    assert _first_token("   ") == ""


# ── _prune_listing_output — 노이즈 경로 제거 ─────────────────────────────


def test_prune_removes_node_modules_lines():
    out = "\n".join([
        "backend/app/main.py",
        "node_modules/react/index.js",
        "node_modules/pg/lib/client.js",
        "frontend/src/App.tsx",
    ])
    pruned = _prune_listing_output(out)
    assert "node_modules" not in pruned.split("[출력 위생")[0]
    assert "backend/app/main.py" in pruned
    assert "frontend/src/App.tsx" in pruned


def test_prune_reports_dropped_count():
    out = "src/a.py\nnode_modules/x/y.js\n.git/config\n__pycache__/z.pyc\nsrc/b.py"
    pruned = _prune_listing_output(out)
    # node_modules·.git·__pycache__ 3줄 제거
    assert "3줄 제외" in pruned
    assert "src/a.py" in pruned and "src/b.py" in pruned


def test_prune_handles_windows_separators():
    out = "backend\\app\\main.py\nnode_modules\\react\\index.js\nfrontend\\App.tsx"
    pruned = _prune_listing_output(out)
    body = pruned.split("[출력 위생")[0]
    assert "node_modules" not in body
    assert "backend\\app\\main.py" in body


def test_prune_noop_when_no_noise():
    """노이즈가 없으면 원본 그대로 — 위생 안내도 붙지 않는다."""
    out = "backend/app/main.py\nfrontend/src/App.tsx\npyproject.toml"
    assert _prune_listing_output(out) == out


def test_prune_covers_build_and_venv_dirs():
    out = "\n".join([
        "src/main.py",
        "dist/bundle.js",
        ".venv/lib/site-packages/foo.py",
        ".next/static/chunk.js",
        "keep/real.py",
    ])
    body = _prune_listing_output(out).split("[출력 위생")[0]
    assert "dist/bundle.js" not in body
    assert ".venv" not in body
    assert ".next" not in body
    assert "src/main.py" in body and "keep/real.py" in body


# ── _truncate_output — 나열 명령 축소 상한 ───────────────────────────────


def test_truncate_respects_custom_max():
    big = "x" * 20_000
    out = _truncate_output(big, _LISTING_MAX_SIZE)
    assert len(out) <= _LISTING_MAX_SIZE + 100  # 안내 문구 여유
    assert "앞부분 생략" in out


def test_truncate_default_unchanged():
    """기존 계약: 기본 상한(50k) 이하는 원본 그대로."""
    assert _truncate_output("short output") == "short output"


def test_listing_pipeline_shrinks_node_modules_flood():
    """통합: node_modules 대량 나열이 실제로 크게 줄어드는지."""
    # 소스 3줄 + node_modules 5000줄을 섞은 재귀 나열을 흉내낸다.
    lines = ["backend/app/main.py", "backend/pyproject.toml", "frontend/App.tsx"]
    lines += [f"node_modules/pkg{i}/index.js" for i in range(5000)]
    raw = "\n".join(lines)
    result = _truncate_output(_prune_listing_output(raw), _LISTING_MAX_SIZE)
    assert "backend/pyproject.toml" in result
    assert len(result) < len(raw) // 10  # 10분의 1 미만으로 축소
