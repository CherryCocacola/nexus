# 브라우저 기본 대화상자가 다시 들어오지 않게 고정한다 — 렌더러를 멈추기 때문이다.
"""
2026-08-08 실측. 메시지 편집 버튼을 누르는 순간 페이지가 통째로 멈춰 어떤 검증도
이어갈 수 없었다. 원인은 `editMessage()` 의 `window.prompt()` 였다 — alert/confirm/
prompt 는 렌더러를 **동기 차단**한다.

사용자에게는 정상 동작이라 늦게 발견된다는 점이 나쁘다. 그래서 16곳을 Promise 기반
공용 대화상자로 바꿨고, 이 테스트는 그것이 되돌아오지 않게 막는다.

★가장 중요한 것은 "네이티브 호출이 하나도 남지 않았는지"다. 하나만 남아도 그 경로에서
같은 정지가 재현되고, 그 지점부터 자동화 검증이 통째로 막힌다.
"""

from __future__ import annotations

import re
from pathlib import Path

INDEX = Path(__file__).resolve().parents[2] / "web" / "static" / "index.html"
SRC = INDEX.read_text(encoding="utf-8")

# JS 주석은 제외하고 **실제 호출**만 본다(설명문에 이름이 나오는 것은 괜찮다).
_CODE = "\n".join(ln for ln in SRC.splitlines() if not ln.lstrip().startswith("//"))


# ─────────────────────────────────────────────
# ★핵심 — 차단 대화상자가 남아 있으면 안 된다
# ─────────────────────────────────────────────
def test_no_native_blocking_dialogs_remain() -> None:
    calls = re.findall(r"(?<![.\w])(?:window\.)?(alert|confirm|prompt)\s*\(", _CODE)
    assert not calls, f"네이티브 대화상자 호출이 남아 있다: {calls}"


def test_fallback_does_not_reintroduce_native_dialogs() -> None:
    """요소를 못 찾았을 때 기본 대화상자로 되돌리면 정지가 그대로 살아난다."""
    body = _CODE[_CODE.index("function _openDialog(") :]
    body = body[: body.index("\nfunction novaAlert")]
    assert "window.prompt" not in body
    assert "window.confirm" not in body
    assert "window.alert" not in body


def test_missing_markup_fails_closed() -> None:
    """확인을 못 받았으면 진행하지 않는다 — 삭제가 조용히 실행되면 안 된다."""
    body = _CODE[_CODE.index("function _openDialog(") :]
    body = body[: body.index("\nfunction novaAlert")]
    assert "withInput ? null : (withCancel ? false : undefined)" in body


# ─────────────────────────────────────────────
# 세 진입점이 모두 있고 호출부가 실제로 쓰는가
# ─────────────────────────────────────────────
def test_three_entry_points_exist() -> None:
    for fn in ("function novaAlert(", "function novaConfirm(", "function novaPrompt("):
        assert fn in SRC, fn


def test_all_former_dialog_sites_now_use_nova() -> None:
    """원래 16곳이었다 — 교체 과정에서 몇 곳이 조용히 사라지지 않았는지 센다."""
    n = len(re.findall(r"nova(?:Alert|Confirm|Prompt)\s*\(", _CODE))
    assert n >= 16, f"호출부가 {n}곳뿐이다(원래 16곳)"


def _call_sites(name: str) -> list[str]:
    """정의부(`function novaX(`)를 뺀 **호출부**의 앞 30자를 돌려준다."""
    out = []
    for m in re.finditer(rf"(?<!function ){name}\s*\(", _CODE):
        out.append(_CODE[max(0, m.start() - 30) : m.start()])
    return out


def test_prompt_sites_are_awaited() -> None:
    """입력값을 쓰는 자리에서 await 를 빠뜨리면 Promise 객체가 제목이 된다."""
    sites = _call_sites("novaPrompt")
    assert len(sites) == 3, f"prompt 호출부가 3곳이어야 한다: {len(sites)}"
    for head in sites:
        assert "await" in head, f"await 없는 novaPrompt: ...{head[-40:]}"


def test_confirm_sites_are_awaited() -> None:
    """await 를 빼면 Promise 가 항상 truthy 라 **확인 없이 삭제가 진행된다**."""
    sites = _call_sites("novaConfirm")
    assert len(sites) == 2, f"confirm 호출부가 2곳이어야 한다: {len(sites)}"
    for head in sites:
        assert "await" in head, f"await 없는 novaConfirm: ...{head[-40:]}"


# ─────────────────────────────────────────────
# 마크업 · 되돌아갈 길
# ─────────────────────────────────────────────
def test_dialog_markup_present() -> None:
    for el in (
        'id="dialogOverlay"',
        'id="dialogMessage"',
        'id="dialogInput"',
        'id="dialogOk"',
        'id="dialogCancel"',
    ):
        assert el in SRC, el


def test_dialog_can_be_dismissed_without_the_mouse() -> None:
    """Escape 로 못 빠져나오는 모달은 기본 대화상자보다 나쁘다."""
    body = SRC[SRC.index("function _openDialog(") :]
    body = body[: body.index("\nfunction novaAlert")]
    assert "'Escape'" in body
    assert "'Enter'" in body


def test_message_preserves_newlines() -> None:
    """기존 문구가 \\n 으로 줄을 나눈다 — 한 줄로 뭉치면 읽을 수 없다."""
    assert "white-space: pre-wrap" in SRC[SRC.index(".dialog-message") : SRC.index(".dialog-input")]


def test_message_is_set_as_text_not_html() -> None:
    """파일명·에러 문구가 그대로 들어온다 — innerHTML 이면 주입 통로가 된다."""
    body = SRC[SRC.index("function _openDialog(") :]
    body = body[: body.index("\nfunction novaAlert")]
    assert "elMsg.textContent = message" in body
    assert "elMsg.innerHTML" not in body
