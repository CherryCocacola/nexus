# 폴더 드래그·다중선택 배선을 고정한다 (A5).
"""
2026-08-08 사용자 눈검증 A5 — "Ctrl 로 여럿 골라 옮기거나 폴더로 끌어다 넣는 게 안 됨".
확인해 보니 고장이 아니라 **미구현**이었다(index.html 에 draggable·dragstart·drop·
ctrlKey 가 아예 없었다).

프런트엔드라 단위 테스트로 실제 드래그를 재현할 수는 없다. 대신 기능이 조용히
사라지지 않도록 **배선과 필수 방어**를 못 박는다. 특히 두 가지가 빠지기 쉽다.

  · `dragover` 의 `preventDefault()` — 없으면 `drop` 이 아예 발생하지 않는다.
    빠져도 콘솔에 아무 것도 안 뜨고 그냥 "드래그가 안 되네"로 끝난다.
  · 평범한 클릭과 Ctrl/Shift 클릭의 분기 — 안 갈라 주면 여러 개를 고르는 동안
    대화가 계속 열려서 선택 자체가 불가능해진다.

실제 조작감은 사람이 봐야 한다 — `user_mig/TEST_사용자수동_20260808.md` A5.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

INDEX = Path(__file__).resolve().parents[2] / "web" / "static" / "index.html"


@pytest.fixture(scope="module")
def html() -> str:
    return INDEX.read_text(encoding="utf-8")


# ─────────────────────────────────────────────
# 드래그 앤 드롭
# ─────────────────────────────────────────────
def test_session_items_are_draggable(html: str) -> None:
    """끌 수 있어야 시작이 된다."""
    item = re.search(r'<div class="session-item \$\{cls\}"[^>]*>', html)
    assert item is not None, "session-item 마크업을 찾지 못했다"
    assert 'draggable="true"' in item.group(0)


def test_dnd_handlers_are_registered(html: str) -> None:
    body = html[html.index("function _initSidebarDnd("):]
    for ev in ("dragstart", "dragend", "dragover", "dragleave", "drop"):
        assert f"'{ev}'" in body, f"{ev} 핸들러가 없다"


def test_dragover_calls_prevent_default(html: str) -> None:
    """★없으면 drop 이 발생하지 않는다 — 조용히 죽는 대표적 실수."""
    body = html[html.index("list.addEventListener('dragover'"):]
    body = body[: body.index("list.addEventListener('dragleave'")]
    assert "ev.preventDefault()" in body


def test_drop_reads_payload_and_moves(html: str) -> None:
    body = html[html.index("list.addEventListener('drop'"):]
    body = body[: body.index("\n  });") + 6]
    assert "getData('text/plain')" in body
    assert "_moveSessionsToFolder" in body


def test_folder_header_and_ungrouped_are_drop_targets(html: str) -> None:
    """폴더로 넣는 것과 폴더에서 빼는 것 둘 다 있어야 왕복이 된다."""
    body = html[html.index("function _initSidebarDnd("):]
    assert ".folder-header" in body
    assert ".drop-zone-ungrouped" in body
    assert 'class="drop-zone-ungrouped" data-folder=""' in html


# ─────────────────────────────────────────────
# 다중선택
# ─────────────────────────────────────────────
def test_modifier_click_selects_instead_of_opening(html: str) -> None:
    """★Ctrl/⌘/Shift 는 고르기다 — 안 갈라 주면 선택이 불가능해진다."""
    idx = html.index("const item = target.closest && target.closest('.session-item');")
    body = html[idx : idx + 900]
    assert "ev.ctrlKey || ev.metaKey || ev.shiftKey" in body
    assert "_handleSelectionClick(sid, ev)" in body
    # 분기 안에서 열기(switchSession)로 흘러가면 안 된다.
    branch = body[body.index("ev.ctrlKey") : body.index("_clearSelection(false)")]
    assert "switchSession" not in branch


def test_plain_click_clears_selection_and_opens(html: str) -> None:
    idx = html.index("const item = target.closest && target.closest('.session-item');")
    body = html[idx : idx + 900]
    assert "_clearSelection(false)" in body
    assert "switchSession(sid)" in body


def test_shift_range_uses_visible_order(html: str) -> None:
    """범위 선택은 **화면에 보이는 순서** 기준이어야 한다.

    정렬(핀 우선·최근순)과 폴더 분류를 거친 뒤의 순서라, 원본 배열 순서로 잡으면
    사용자가 본 것과 다른 구간이 선택된다.
    """
    body = html[html.index("function _handleSelectionClick("):]
    body = body[: body.index("\n}\n") + 3]
    assert "_visibleSidsInOrder()" in body
    assert "shiftKey" in body


def test_selection_is_not_persisted(html: str) -> None:
    """선택은 화면 상태다 — 서버나 localStorage 로 새어 나가면 안 된다."""
    body = html[html.index("const _selectedSids = new Set();") :]
    body = body[: body.index("function _initSidebarDnd(")]
    assert "localStorage" not in body
    assert "patchSessionMeta(sid, { folder" in body or "patchSessionMeta" in body


def test_drag_moves_whole_selection_when_dragging_a_selected_item(html: str) -> None:
    """탐색기 관례 — 선택된 항목을 끌면 선택 전체가 따라간다."""
    body = html[html.index("function _dragTargets("):]
    body = body[: body.index("\n}\n") + 3]
    assert "_selectedSids.has(sid)" in body


def test_folder_button_applies_to_selection(html: str) -> None:
    """여러 개 골라 놓고 📁 를 눌렀는데 하나만 옮겨지면 고른 의미가 없다."""
    body = html[html.index("async function setFolder(id)"):]
    body = body[: body.index("\n}\n") + 3]
    assert "_selectedSids.has(id)" in body
    assert "_moveSessionsToFolder(targets" in body


def test_move_skips_no_op_patches(html: str) -> None:
    """이미 그 폴더인 대화는 건너뛴다 — 서버 왕복을 낭비하지 않는다."""
    body = html[html.index("async function _moveSessionsToFolder("):]
    body = body[: body.index("\n}\n") + 3]
    assert "=== clean) continue" in body.replace(" ", " ")
