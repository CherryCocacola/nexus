# cli/repl.py /model 표시 전용 전환 검증 — 장식용 모델 변경 기능 회귀 방지.
"""
NexusREPL._cmd_model 이 라우팅 config의 실제 모델을 "표시만" 하는지 검증한다.

배경(B-5/B-6): 과거 `/model <name>`은 모델을 바꾸는 것처럼 보였지만, self._model은
배너·표시에만 쓰일 뿐 엔진에 배선되지 않아 실제로는 아무것도 바꾸지 못했다(장식용).
이제 변경 분기를 없애고 배너와 동일 소스(config.routing)에서 CHAT/KNOWLEDGE/TOOL
실모델명을 표로 보여주는 표시 전용으로 전환했다. 이 테스트가 회귀를 막는다.
"""

from __future__ import annotations

import asyncio
import io
from types import SimpleNamespace

from rich.console import Console

from cli.repl import NexusREPL


def _repl_with_state(state) -> tuple[NexusREPL, io.StringIO]:
    """REPL 껍데기에 가짜 _state를 심고, 출력 캡처용 콘솔로 교체한다.

    NexusREPL.__init__ 은 PromptSession(prompt-toolkit)을 만드는데, 이는 실제
    Windows 콘솔이 없는 테스트 환경에서 실패한다. _cmd_model 하나만 검증하면
    되므로 __new__ 로 생성자를 우회하고 이 메서드가 참조하는 필드만 직접 심는다.
    """
    repl = NexusREPL.__new__(NexusREPL)
    repl._state = state
    repl._model = "primary"
    buf = io.StringIO()
    # 캡처용 콘솔 — force_terminal=False + 넉넉한 폭으로 표가 잘리지 않게 한다.
    repl.console = Console(file=buf, force_terminal=False, width=200)
    return repl, buf


def test_cmd_model_shows_routing_models():
    """라우팅 활성 시 CHAT/KNOWLEDGE/TOOL 실모델명을 표시한다."""
    routing = SimpleNamespace(
        enabled=True,
        chat_mode=SimpleNamespace(model="ax4-chat"),
        knowledge_mode=SimpleNamespace(model="ax4-know"),
        tool_mode=SimpleNamespace(model="ax4-tool"),
    )
    state = SimpleNamespace(config=SimpleNamespace(routing=routing))
    repl, buf = _repl_with_state(state)

    asyncio.run(repl._cmd_model([]))

    out = buf.getvalue()
    assert "ax4-chat" in out
    assert "ax4-know" in out
    assert "ax4-tool" in out


def test_cmd_model_ignores_change_argument():
    """인자를 줘도 self._model을 바꾸지 않는다(변경 기능 제거 확인 — 표시 전용)."""
    routing = SimpleNamespace(
        enabled=True,
        chat_mode=SimpleNamespace(model="ax4-chat"),
        knowledge_mode=SimpleNamespace(model="ax4-know"),
        tool_mode=SimpleNamespace(model="ax4-tool"),
    )
    state = SimpleNamespace(config=SimpleNamespace(routing=routing))
    repl, _ = _repl_with_state(state)

    before = repl._model
    asyncio.run(repl._cmd_model(["auxiliary"]))  # 과거엔 이걸로 모델이 바뀌었다
    assert repl._model == before  # 이제는 무시 — 아무것도 바뀌지 않는다


def test_cmd_model_routing_disabled_shows_primary_auxiliary():
    """라우팅 비활성 시 primary/auxiliary 두 모델을 표시한다."""
    model_cfg = SimpleNamespace(primary_model="qwen-primary", auxiliary_model="exaone-aux")
    routing = SimpleNamespace(enabled=False)
    state = SimpleNamespace(config=SimpleNamespace(routing=routing, model=model_cfg))
    repl, buf = _repl_with_state(state)

    asyncio.run(repl._cmd_model([]))

    out = buf.getvalue()
    assert "qwen-primary" in out
    assert "exaone-aux" in out
