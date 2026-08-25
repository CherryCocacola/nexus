# payload·usage 덤프가 실제로 나가는 값을 남기고, 동시 요청끼리 섞이지 않는지 검증한다.
"""
왜 이 테스트가 필요한가 (2026-08-25).

08-24 에 같은 요청의 prompt_tokens 가 138 늘어난 사건을 조사했는데, 기존
`dump_prompt` 는 `system_prompt`·`messages` 만 남겨서 **차이가 어디서 왔는지 볼 수
없었다.** prompt_tokens 를 좌우하는 `tools`·`tool_choice`·`response_format`·
`chat_template_kwargs` 는 Tier 3 가 그 뒤에 조립하기 때문이다.

그래서 payload 전문과 usage 를 같은 키로 남기게 했다. 이 테스트가 고정하는 것은 셋이다.
  1) 비활성(환경변수 없음)이면 아무 파일도 만들지 않는다 — 기본은 꺼져 있어야 한다.
  2) payload 전문이 그대로 남는다 — 필드를 골라 담지 않는다.
  3) 동시 요청의 키가 섞이지 않는다 — ContextVar 를 쓴 유일한 이유다.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from core.orchestrator import prompt_dump


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    """각 테스트가 자기 환경변수만 보게 한다(테스트 간 누수 차단)."""
    monkeypatch.delenv(prompt_dump.DUMP_DIR_ENV, raising=False)
    monkeypatch.delenv(prompt_dump.RETENTION_ENV, raising=False)


def test_payload_dump_disabled_writes_nothing(tmp_path, monkeypatch):
    """환경변수가 없으면 파일을 만들지 않는다 — 민감 내용이 조용히 쌓이면 안 된다."""
    monkeypatch.chdir(tmp_path)
    prompt_dump.set_context("req-1", "sess-1", 1)

    assert prompt_dump.dump_payload({"model": "ax-4.0"}) is None
    assert prompt_dump.dump_usage(100, 10) is None
    assert list(tmp_path.iterdir()) == []


def test_payload_dump_keeps_every_field(tmp_path, monkeypatch):
    """payload 를 통째로 남긴다 — 골라 담으면 다음 조사에서 또 같은 사각지대가 생긴다."""
    monkeypatch.setenv(prompt_dump.DUMP_DIR_ENV, str(tmp_path))
    prompt_dump.set_context("req-42", "sess-x", 3)

    payload = {
        "model": "ax-4.0",
        "messages": [{"role": "system", "content": "S"}],
        "tools": [{"type": "function", "function": {"name": "Edit"}}],
        "tool_choice": "auto",
        "chat_template_kwargs": {"enable_thinking": False},
        "response_format": {"type": "json_schema"},
        "stop": ["</s>"],
        "n": 1,
    }
    path = prompt_dump.dump_payload(payload, base_url="http://127.0.0.1:8001")

    assert path is not None
    assert path.name == "req-42_turn3_payload.json"
    saved = json.loads(path.read_text(encoding="utf-8"))
    for key, value in payload.items():
        assert saved[key] == value, f"{key} 가 덤프에서 누락/변형됐다"
    assert saved["_meta"]["base_url"] == "http://127.0.0.1:8001"
    assert saved["_meta"]["turn"] == 3


def test_usage_dump_shares_the_payload_key(tmp_path, monkeypatch):
    """usage 가 같은 키로 남아야 '어느 덤프가 문제의 요청인지' 를 판별할 수 있다."""
    monkeypatch.setenv(prompt_dump.DUMP_DIR_ENV, str(tmp_path))
    prompt_dump.set_context("req-7", "sess-y", 2)

    prompt_dump.dump_payload({"model": "ax-4.0"})
    usage_path = prompt_dump.dump_usage(11009, 110)

    assert usage_path is not None
    assert usage_path.name == "req-7_turn2_usage.json"
    saved = json.loads(usage_path.read_text(encoding="utf-8"))
    assert saved["prompt_tokens"] == 11009
    assert saved["completion_tokens"] == 110
    assert (tmp_path / "req-7_turn2_payload.json").is_file()


def test_request_id_falls_back_to_session_id(tmp_path, monkeypatch):
    """request_id 가 없는 클라이언트도 세션으로 추적할 수 있어야 한다."""
    monkeypatch.setenv(prompt_dump.DUMP_DIR_ENV, str(tmp_path))
    prompt_dump.set_context(None, "sess-only", 1)

    path = prompt_dump.dump_payload({"model": "ax-4.0"})

    assert path is not None
    assert path.name == "sess-only_turn1_payload.json"


def test_dump_prompt_sets_context_for_downstream(tmp_path, monkeypatch):
    """★핵심★ `dump_prompt` 가 스스로 키를 걸어야 호출부를 고칠 필요가 없다.

    배포된 컨테이너의 `query_loop.py` 는 리포 브랜치보다 수백 줄 뒤처져 있다. 거기에
    브랜치본을 복사했다가 구버전 `core/message.py` 에 없는 심볼을 import 해 부트스트랩이
    깨지고 전 요청이 401 이 된 사고가 있었다(2026-08-25). 그래서 이 모듈만 단독으로
    교체해도 payload·usage 가 같은 키로 남아야 한다.
    """
    monkeypatch.setenv(prompt_dump.DUMP_DIR_ENV, str(tmp_path))
    prompt_dump.set_context("STALE", "STALE", 99)  # 이전 턴의 잔재

    class _Msg:
        role = "user"
        content = "hello"

    prompt_dump.dump_prompt(
        request_id="req-ctx",
        session_id="sess-ctx",
        turn=5,
        system_prompt="S",
        messages=[_Msg()],
        tool_names=[],
        routing={},
        sampling={},
    )
    # 호출부가 set_context 를 부르지 않았는데도 키가 갱신돼 있어야 한다.
    prompt_dump.dump_payload({"model": "ax-4.0"})
    prompt_dump.dump_usage(10871, 5)

    assert (tmp_path / "req-ctx_turn5.json").is_file()
    assert (tmp_path / "req-ctx_turn5_payload.json").is_file()
    assert (tmp_path / "req-ctx_turn5_usage.json").is_file()
    assert not (tmp_path / "STALE_turn99_payload.json").exists()


def test_concurrent_requests_do_not_share_keys(tmp_path, monkeypatch):
    """★핵심★ 동시 요청의 키가 섞이지 않는다.

    이 서버는 요청마다 세션 전용 엔진을 만들어 병렬로 돈다. 키를 모듈 전역에 두면
    A 요청의 payload 가 B 요청 이름으로 저장돼 조사 자체가 오염된다. ContextVar 를
    쓴 유일한 이유가 이것이므로 테스트로 못박는다.
    """
    monkeypatch.setenv(prompt_dump.DUMP_DIR_ENV, str(tmp_path))

    async def one(name: str, tokens: int) -> None:
        prompt_dump.set_context(name, f"sess-{name}", 1)
        # 다른 태스크가 그 사이에 자기 컨텍스트를 걸도록 실제로 양보한다.
        await asyncio.sleep(0)
        prompt_dump.dump_payload({"model": "ax-4.0", "marker": name})
        await asyncio.sleep(0)
        prompt_dump.dump_usage(tokens, 1)

    async def main() -> None:
        await asyncio.gather(one("reqA", 10871), one("reqB", 11009))

    asyncio.run(main())

    a = json.loads((tmp_path / "reqA_turn1_payload.json").read_text(encoding="utf-8"))
    b = json.loads((tmp_path / "reqB_turn1_payload.json").read_text(encoding="utf-8"))
    assert a["marker"] == "reqA"
    assert b["marker"] == "reqB"

    ua = json.loads((tmp_path / "reqA_turn1_usage.json").read_text(encoding="utf-8"))
    ub = json.loads((tmp_path / "reqB_turn1_usage.json").read_text(encoding="utf-8"))
    assert ua["prompt_tokens"] == 10871
    assert ub["prompt_tokens"] == 11009
