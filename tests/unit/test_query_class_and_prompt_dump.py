# 질의 클래스 고정과 프롬프트 덤프 — 요청 단위 제어를 고정한다
"""
2026-08-16. 두 가지를 만들었다. 둘 다 "무엇이 모델에 들어갔는가"를 호출자가
통제·확인할 수 있게 하는 장치다.

■ ① 질의 클래스 고정
    짧고 키워드 없는 코드 질문이 KNOWLEDGE 로 분류돼 **사내 문서 RAG 가 ~5,000자
    주입되는 것이 실측됐다.** 코드 분석에는 방해인데, 지금까지는 "우연히 TOOL 로
    분류되기를 기대하는" 상태였다(질문에 '파일' 같은 단어를 넣으면 TOOL 이 된다).
    호출자가 의도를 밝힐 수 있어야 한다.

    RAG 만 끄지 않고 클래스를 고정하는 이유 — RAG 만 끄면 KNOWLEDGE 프로필
    (temp 0.2 / max 4096)인데 근거는 없는 어정쩡한 상태가 된다.

■ ② 프롬프트 덤프
    무상태 엔드포인트라 최종 프롬프트가 어디에도 남지 않았다. 다만 덤프에는 사내
    문서 본문이 그대로 들어가므로 **환경변수를 준 경우에만** 켜진다.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from fastapi import HTTPException

from core.orchestrator import prompt_dump
from core.orchestrator.routing import QUERY_CLASSES
from web.app import _resolve_query_class


def _msg_user(text: str):
    """실제 Message 팩토리 — 덤프가 role/content 를 제대로 읽는지 보려면 진짜 객체여야 한다."""
    from core.message import Message

    return Message.user(text)


class _Msg:
    def __init__(self, role: str, content: str) -> None:
        self.role = role
        self.content = content


# ─────────────────────────────────────────────
# ① 질의 클래스 고정 — 값 해석
# ─────────────────────────────────────────────
@pytest.mark.parametrize("value", ["TOOL", "tool", " Tool "])
def test_class_is_normalized(value: str) -> None:
    assert _resolve_query_class(value, None) == "TOOL"


@pytest.mark.parametrize("cls", QUERY_CLASSES)
def test_all_documented_classes_are_accepted(cls: str) -> None:
    """문서에 적은 값이 실제로 통과해야 한다 — 목록이 두 곳에 있으면 어긋난다."""
    assert _resolve_query_class(cls, None) == cls


def test_body_wins_over_header() -> None:
    """요청 본문이 그 요청의 의도를 더 직접 담는다."""
    assert _resolve_query_class("TOOL", "KNOWLEDGE") == "TOOL"


def test_header_is_used_when_body_absent() -> None:
    """body 를 못 건드리는 클라이언트용 폴백."""
    assert _resolve_query_class(None, "KNOWLEDGE") == "KNOWLEDGE"


@pytest.mark.parametrize("body", [None, "", "   "])
def test_absent_means_auto(body: Any) -> None:
    """지정이 없으면 종전대로 서버가 분류한다(무회귀)."""
    assert _resolve_query_class(body, None) is None


def test_unknown_value_is_rejected_not_ignored() -> None:
    """★조용히 무시하면 '왜 여전히 RAG 가 붙지'의 원인을 영영 못 찾는다."""
    with pytest.raises(HTTPException) as exc:
        _resolve_query_class("RAG_OFF", None)

    assert exc.value.status_code == 400
    assert "RAG_OFF" in str(exc.value.detail)
    for cls in QUERY_CLASSES:
        assert cls in str(exc.value.detail)      # 허용값을 함께 알려 준다


def test_non_string_header_is_treated_as_absent() -> None:
    """엔드포인트를 직접 부르는 테스트에서는 헤더 자리에 FastAPI 표식 객체가 온다."""
    assert _resolve_query_class(None, object()) is None


# ─────────────────────────────────────────────
# ① 질의 클래스 고정 — 라우팅 반영
# ─────────────────────────────────────────────
def _resolver():
    from core.config import RoutingConfig
    from core.orchestrator.routing import RoutingResolver

    return RoutingResolver(RoutingConfig())


def test_forced_class_skips_the_classifier() -> None:
    """분류기가 KNOWLEDGE 로 볼 문장을 TOOL 로 고정한다."""
    r = _resolver()
    text = "이 함수의 시간복잡도는?"          # 짧고 tool 키워드 없음 → 원래 KNOWLEDGE

    assert r.resolve(text).query_class == "KNOWLEDGE"
    assert r.resolve(text, forced_class="TOOL").query_class == "TOOL"


def test_forced_tool_disables_knowledge_rag() -> None:
    """★이것이 이 기능의 목적이다 — 사내 문서가 코드 질문에 섞이지 않게."""
    r = _resolver()
    text = "이 함수의 시간복잡도는?"

    assert r.resolve(text).inject_knowledge_rag is True
    assert r.resolve(text, forced_class="TOOL").inject_knowledge_rag is False


def test_forced_class_also_selects_the_profile() -> None:
    """클래스를 고정하면 샘플링 프로필까지 함께 결정돼 결과가 예측 가능해진다."""
    r = _resolver()
    text = "이 함수의 시간복잡도는?"

    knowledge = r.resolve(text, forced_class="KNOWLEDGE")
    tool = r.resolve(text, forced_class="TOOL")

    assert knowledge.temperature != tool.temperature


# ─────────────────────────────────────────────
# ② 프롬프트 덤프
# ─────────────────────────────────────────────
def test_disabled_by_default(monkeypatch) -> None:
    """★민감 내용이라 기본은 꺼져 있어야 한다."""
    monkeypatch.delenv(prompt_dump.DUMP_DIR_ENV, raising=False)

    assert prompt_dump.is_enabled() is False
    assert prompt_dump.dump_prompt(
        request_id="r1", session_id="s1", turn=1,
        system_prompt="x", messages=[], tool_names=[], routing={}, sampling={},
    ) is None


def test_dump_writes_final_prompt(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv(prompt_dump.DUMP_DIR_ENV, str(tmp_path))

    path = prompt_dump.dump_prompt(
        request_id="req-abc", session_id="sess-1", turn=2,
        system_prompt="시스템 프롬프트 본문",
        messages=[_Msg("user", "안녕")],
        tool_names=["Read", "Edit"],
        routing={"model_override": "ax-4.0"},
        sampling={"temperature": 0.3},
    )

    assert path is not None and path.exists()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["request_id"] == "req-abc"
    assert data["turn"] == 2
    assert data["system_prompt"] == "시스템 프롬프트 본문"
    assert data["messages"] == [{"role": "user", "content": "안녕"}]
    assert data["tools"] == ["Read", "Edit"]


def test_turns_are_separate_files(tmp_path, monkeypatch) -> None:
    """도구 루프는 한 요청이 여러 턴을 돈다 — 덮어쓰면 앞 턴을 잃는다."""
    monkeypatch.setenv(prompt_dump.DUMP_DIR_ENV, str(tmp_path))
    for turn in (1, 2):
        prompt_dump.dump_prompt(
            request_id="req-x", session_id="s", turn=turn,
            system_prompt=f"turn{turn}", messages=[], tool_names=[],
            routing={}, sampling={},
        )

    assert len(list(tmp_path.glob("req-x_turn*.json"))) == 2


def test_falls_back_to_session_id(tmp_path, monkeypatch) -> None:
    """웹 UI·CLI 는 요청 ID 가 없다."""
    monkeypatch.setenv(prompt_dump.DUMP_DIR_ENV, str(tmp_path))

    path = prompt_dump.dump_prompt(
        request_id=None, session_id="sess-9", turn=1,
        system_prompt="x", messages=[], tool_names=[], routing={}, sampling={},
    )

    assert path is not None and path.name.startswith("sess-9_turn1")


@pytest.mark.parametrize("evil", ["../../etc/passwd", "a/b/c", "..\\win"])
def test_path_traversal_is_neutralized(tmp_path, monkeypatch, evil: str) -> None:
    """요청 ID 는 클라이언트가 정하는 값이다 — 경로 조작이 섞일 수 있다."""
    monkeypatch.setenv(prompt_dump.DUMP_DIR_ENV, str(tmp_path))

    path = prompt_dump.dump_prompt(
        request_id=evil, session_id="s", turn=1,
        system_prompt="x", messages=[], tool_names=[], routing={}, sampling={},
    )

    assert path is not None
    assert path.parent == tmp_path          # 디렉토리를 벗어나지 않았다


def test_write_failure_is_fail_soft(tmp_path, monkeypatch) -> None:
    """★진단 장치가 본 요청을 막으면 본말이 전도된다."""
    monkeypatch.setenv(prompt_dump.DUMP_DIR_ENV, str(tmp_path / "x"))
    monkeypatch.setattr(
        prompt_dump.Path, "write_text",
        lambda *a, **k: (_ for _ in ()).throw(OSError("디스크 가득")),
    )

    assert prompt_dump.dump_prompt(
        request_id="r", session_id="s", turn=1,
        system_prompt="x", messages=[], tool_names=[], routing={}, sampling={},
    ) is None


def test_old_dumps_are_cleaned_up(tmp_path, monkeypatch) -> None:
    """민감 내용을 오래 남기지 않는다."""
    import os
    import time

    monkeypatch.setenv(prompt_dump.DUMP_DIR_ENV, str(tmp_path))
    monkeypatch.setenv(prompt_dump.RETENTION_ENV, "1")     # 1시간

    stale = tmp_path / "old_turn1.json"
    stale.write_text("{}", encoding="utf-8")
    old = time.time() - 3 * 3600
    os.utime(stale, (old, old))

    prompt_dump.dump_prompt(
        request_id="new", session_id="s", turn=1,
        system_prompt="x", messages=[], tool_names=[], routing={}, sampling={},
    )

    assert not stale.exists()
    assert (tmp_path / "new_turn1.json").exists()


# ─────────────────────────────────────────────
# ③ 두 실행 경로의 인자 정합성
# ─────────────────────────────────────────────
def _kwargs_passed_to(callee: str) -> set[str]:
    """query_engine 소스에서 특정 호출에 넘기는 키워드 인자 이름을 뽑는다."""
    import re
    from pathlib import Path

    import core.orchestrator.query_engine as qe

    src = Path(qe.__file__).read_text(encoding="utf-8")
    best: set[str] = set()
    for m in re.finditer(re.escape(callee) + r"\(", src):
        i = src.index("(", m.start())
        depth = 0
        for j in range(i, len(src)):
            if src[j] == "(":
                depth += 1
            elif src[j] == ")":
                depth -= 1
                if depth == 0:
                    break
        names = set(re.findall(r"(\w+)\s*=", src[i:j]))
        if len(names) > len(best):
            best = names
    return best


def test_dispatcher_accepts_every_kwarg_the_engine_sends() -> None:
    """★이 테스트가 없어서 운영이 깨졌다.

    QueryEngine 은 실행 경로가 둘이다 — query_loop 을 직접 부르거나
    ModelDispatcher.route() 를 거친다. 인자를 늘릴 때 한쪽만 고치면 그 경로만
    TypeError 로 500 이 난다. 단위 테스트는 dispatcher 경로를 타지 않아
    **전부 통과하는데 실서버는 죽었다**(2026-08-16, request_id 추가 시 실제 발생).
    """
    import inspect

    from core.orchestrator.model_dispatcher import ModelDispatcher

    sent = _kwargs_passed_to("self._model_dispatcher.route")
    accepted = set(inspect.signature(ModelDispatcher.route).parameters)

    assert sent, "route() 호출을 못 찾았다 — 이 테스트가 헛돌고 있다"
    missing = sorted(sent - accepted)
    assert not missing, f"route() 가 못 받는 인자를 엔진이 보낸다: {missing}"


def test_dispatcher_path_has_no_context_manager() -> None:
    """★기존 결함을 못 박아 둔다(2026-08-16 발견, 이번 작업 범위 밖).

    엔진은 query_loop 직접 호출에는 context_manager 를 넘기지만 dispatcher 경로에는
    넘기지 않는다. query_loop 은 `context_manager is None` 이면 **긴급 압축을 건너뛴다**.
    즉 웹/OpenAI 엔드포인트는 컨텍스트 초과 시 압축으로 복구하지 못한다
    (예산 기반 truncate 는 별개로 동작한다).

    고치면 이 테스트가 실패한다 — 그때 의도적으로 지우면 된다.
    """
    direct = _kwargs_passed_to("query_loop")
    via_dispatcher = _kwargs_passed_to("self._model_dispatcher.route")

    assert "context_manager" in direct
    assert "context_manager" not in via_dispatcher


# ─────────────────────────────────────────────
# ④ query_loop 통합 — 실제로 최종 프롬프트가 남는가
# ─────────────────────────────────────────────
@pytest.mark.asyncio
async def test_query_loop_writes_the_final_prompt(tmp_path, monkeypatch) -> None:
    """모듈 단위가 아니라 **실제 턴을 돌려** 파일이 남는지 본다.

    덤프의 값은 "모델이 무엇을 봤는가"이므로, 호출부가 엉뚱한 값을 넘기면 파일은
    생기는데 내용이 틀린다. 그래서 시스템 프롬프트·도구 이름까지 확인한다.
    """
    from core.orchestrator.query_loop import query_loop
    from core.tools.base import ToolUseContext
    from tests.unit.test_query_loop import ScriptedProvider

    monkeypatch.setenv(prompt_dump.DUMP_DIR_ENV, str(tmp_path))

    provider = ScriptedProvider([{"text": "안녕하세요"}])
    ctx = ToolUseContext(session_id="s-1", cwd=str(tmp_path))

    async for _ in query_loop(
        messages=[_msg_user("이 함수 설명해줘")],
        system_prompt="시스템 프롬프트 전문",
        model_provider=provider,
        tools=[],
        context=ctx,
        request_id="dump-req-1",
        session_id="s-1",
    ):
        pass

    files = list(tmp_path.glob("dump-req-1_turn*.json"))
    assert files, f"덤프가 없다: {list(tmp_path.iterdir())}"

    data = json.loads(files[0].read_text(encoding="utf-8"))
    assert data["request_id"] == "dump-req-1"
    assert data["system_prompt"] == "시스템 프롬프트 전문"
    assert data["messages"][-1]["content"] == "이 함수 설명해줘"
    assert "temperature" in data["sampling"]


@pytest.mark.asyncio
async def test_query_loop_writes_nothing_when_disabled(tmp_path, monkeypatch) -> None:
    """★기본 비활성이 진짜로 지켜지는지 — 실수로 켜져 있으면 사내 문서가 쌓인다."""
    from core.orchestrator.query_loop import query_loop
    from core.tools.base import ToolUseContext
    from tests.unit.test_query_loop import ScriptedProvider

    monkeypatch.delenv(prompt_dump.DUMP_DIR_ENV, raising=False)

    async for _ in query_loop(
        messages=[_msg_user("안녕")],
        system_prompt="x",
        model_provider=ScriptedProvider([{"text": "네"}]),
        tools=[],
        context=ToolUseContext(session_id="s-2", cwd=str(tmp_path)),
        request_id="dump-req-2",
        session_id="s-2",
    ):
        pass

    assert list(tmp_path.iterdir()) == []
