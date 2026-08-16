# OpenAI 규격 클라이언트 도구 루프 — 웹 계층의 변환·응답 계약을 고정한다.
"""
`/v1/chat/completions` 의 `tools` / `tool_calls` 배선을 검증한다 (2026-08-08).

[왜 생겼나]
    코딩용 API(VSCode 플러그인)가 "CLI 와 같은 능력"을 가지려면 모델이 파일을 읽고
    고치는 판단을 해야 한다. 그 실행은 **클라이언트가** 한다 — 서버가 하면
    ① 서버 파일시스템이 열리고(테넌트 키 유출 실측), ② 개발자 코드는 개발자 PC 에
    있어서 애초에 쓸모가 없다.

    그래서 서버는 tool_calls 까지만 정하고 돌려준다. 이 파일은 그 왕복(요청 히스토리
    재현 → 응답 형식)이 규격에서 벗어나지 않게 고정한다.
"""

from __future__ import annotations

import json

import pytest
from fastapi import HTTPException

from web.app import (
    OpenAIChatMessage,
    _openai_tool_calls_to_uses,
    _split_openai_messages,
    _tool_uses_to_openai_tool_calls,
)


def _msg(role: str, content: str | None = None, **kw) -> OpenAIChatMessage:
    return OpenAIChatMessage(role=role, content=content, **kw)


READ_CALL = {
    "id": "call_abc123",
    "type": "function",
    "function": {"name": "read_file", "arguments": '{"path": "src/main.py"}'},
}


# ─────────────────────────────────────────────
# 요청 방향 — OpenAI tool_calls → 내부 형식
# ─────────────────────────────────────────────


def test_arguments_json_string_is_parsed():
    """OpenAI 는 인자를 JSON '문자열'로 싣지만 내부 계약은 dict 다."""
    uses = _openai_tool_calls_to_uses([READ_CALL])

    assert uses == [{"id": "call_abc123", "name": "read_file", "input": {"path": "src/main.py"}}]


def test_broken_arguments_keep_the_call_with_empty_input():
    """★인자가 깨져도 호출 사실은 남긴다 — 버리면 모델이 같은 도구를 또 부른다."""
    broken = {"id": "c1", "function": {"name": "read_file", "arguments": "{path:"}}

    uses = _openai_tool_calls_to_uses([broken])

    assert len(uses) == 1
    assert uses[0]["input"] == {}


def test_malformed_entries_are_skipped():
    assert _openai_tool_calls_to_uses([None, "문자열", {"function": {}}]) == []
    assert _openai_tool_calls_to_uses(None) == []


# ─────────────────────────────────────────────
# 응답 방향 — 내부 형식 → OpenAI tool_calls
# ─────────────────────────────────────────────


def test_response_serializes_arguments_as_json_string():
    calls = _tool_uses_to_openai_tool_calls(
        [{"id": "c1", "name": "read_file", "input": {"path": "a.py"}}]
    )

    assert calls[0]["type"] == "function"
    assert calls[0]["function"]["name"] == "read_file"
    # 규격상 문자열이어야 한다(dict 로 주면 표준 클라이언트가 파싱에 실패한다).
    assert json.loads(calls[0]["function"]["arguments"]) == {"path": "a.py"}


def test_korean_arguments_are_not_escaped():
    """클라이언트 로그에 그대로 읽혀야 한다."""
    calls = _tool_uses_to_openai_tool_calls(
        [{"id": "c1", "name": "write", "input": {"내용": "한글"}}]
    )

    assert "한글" in calls[0]["function"]["arguments"]


def test_roundtrip_preserves_call_identity():
    """요청 → 내부 → 응답을 돌아도 id·이름·인자가 그대로여야 짝이 맞는다."""
    back = _tool_uses_to_openai_tool_calls(_openai_tool_calls_to_uses([READ_CALL]))

    assert back[0]["id"] == "call_abc123"
    assert json.loads(back[0]["function"]["arguments"]) == {"path": "src/main.py"}


# ─────────────────────────────────────────────
# 히스토리 재현 — 도구 루프의 대화 모양
# ─────────────────────────────────────────────


def test_conversation_ending_with_tool_result_is_accepted():
    """★표준 도구 루프는 user 가 아니라 tool 로 끝난다 — 400 이 나면 루프가 막힌다."""
    _sys, prior, last_user, is_cont = _split_openai_messages(
        [
            _msg("user", "버그 고쳐줘"),
            _msg("assistant", None, tool_calls=[READ_CALL]),
            _msg("tool", "print(1/0)", tool_call_id="call_abc123"),
        ]
    )

    assert is_cont is True
    # 라우팅 판정용으로 원래 요청 문구를 넘긴다(히스토리에 다시 넣지는 않는다).
    assert last_user == "버그 고쳐줘"
    assert len(prior) == 3


def test_assistant_turn_without_text_is_preserved():
    """★도구만 부른 턴은 content 가 비어 있다 — 건너뛰면 모델이 자기 요청을 잊는다."""
    from core.message import Role

    _sys, prior, _last, _cont = _split_openai_messages(
        [
            _msg("user", "버그 고쳐줘"),
            _msg("assistant", None, tool_calls=[READ_CALL]),
            _msg("tool", "print(1/0)", tool_call_id="call_abc123"),
        ]
    )

    assistant_msg = prior[1]
    assert assistant_msg.role == Role.ASSISTANT
    names = [getattr(b, "name", None) for b in assistant_msg.content]
    assert "read_file" in names


def test_tool_result_is_linked_to_its_call():
    """tool_call_id 로 짝을 맞춰야 모델이 어느 호출의 결과인지 안다."""
    _sys, prior, _last, _cont = _split_openai_messages(
        [
            _msg("user", "버그 고쳐줘"),
            _msg("assistant", None, tool_calls=[READ_CALL]),
            _msg("tool", "print(1/0)", tool_call_id="call_abc123"),
        ]
    )

    assert prior[2].tool_use_id == "call_abc123"
    assert prior[2].content == "print(1/0)"


def test_tool_result_without_id_is_dropped():
    """짝을 못 맞추는 결과는 넣지 않는다 — 모델이 맥락 없는 텍스트로 읽는다."""
    _sys, prior, _last, _cont = _split_openai_messages(
        [_msg("user", "안녕"), _msg("tool", "결과"), _msg("user", "계속")]
    )

    assert len(prior) == 1  # 첫 user 만 남는다


# ─────────────────────────────────────────────
# 무회귀 — 도구를 안 쓰는 기존 요청
# ─────────────────────────────────────────────


def test_plain_conversation_is_unchanged():
    system, prior, last_user, is_cont = _split_openai_messages(
        [
            _msg("system", "너는 도우미다"),
            _msg("user", "안녕"),
            _msg("assistant", "네"),
            _msg("user", "질문"),
        ]
    )

    assert system == "너는 도우미다"
    assert last_user == "질문"
    assert is_cont is False
    assert len(prior) == 2


def test_last_message_must_still_be_user_or_tool():
    """assistant 로 끝나는 요청은 종전대로 400 이다(규격 위반)."""
    with pytest.raises(HTTPException) as exc:
        _split_openai_messages([_msg("user", "안녕"), _msg("assistant", "네")])

    assert exc.value.status_code == 400


def test_empty_messages_rejected():
    with pytest.raises(HTTPException):
        _split_openai_messages([])


# ─────────────────────────────────────────────
# 엔드포인트 — 실제 응답 모양
# ─────────────────────────────────────────────


class _ToolCallingEngine:
    """도구를 한 번 부르고 끝나는 가짜 엔진(비스트림 경로가 쓰는 최소 표면)."""

    def __init__(self) -> None:
        self.system_prompt = "base"
        self._messages: list = []
        self.received_append_user_message: bool | None = None

    def update_system_prompt(self, prompt: str) -> None:
        self.system_prompt = prompt

    def bind_request(self, **kwargs) -> None:
        pass

    def clear_messages(self) -> None:
        self._messages.clear()

    async def submit_message(
        self,
        message: str,
        structured_output=None,
        max_tokens_override=None,
        append_user_message: bool = True,
        # 실제 QueryEngine 이 인자를 늘려도 더미가 깨지지 않게 받아 둔다
        # (요청 클래스 고정·요청 ID 등). 검증은 각 전용 테스트가 한다.
        **_kwargs,
    ):
        from core.message import StreamEvent, StreamEventType, ToolUseBlock

        self.received_append_user_message = append_user_message
        yield StreamEvent(
            type=StreamEventType.TOOL_USE_STOP,
            tool_use=ToolUseBlock(
                id="call_abc123", name="read_file", input={"path": "src/main.py"}
            ),
        )


@pytest.fixture
def _tool_engine():
    """_app_state 에 가짜 엔진을 주입하고 원복한다(모듈 싱글톤 격리)."""
    from web.app import _app_state

    saved = {k: _app_state.get(k) for k in ("query_engine", "web_engine_parts", "memory_manager")}
    fake = _ToolCallingEngine()
    _app_state["query_engine"] = fake
    _app_state["web_engine_parts"] = None  # 싱글톤 경로 강제
    _app_state["memory_manager"] = None
    yield fake
    _app_state.update(saved)


def test_prompt_is_rewritten_to_the_actual_tool_list():
    """★프롬프트↔도구 불일치 방지.

    기본 웹 프롬프트는 서버 도구(Read/Write/Edit)를 안내한다. 도구를 교체했는데
    그 안내가 남아 있으면 모델이 없는 도구를 부른다 — 이 리포에서 이미 같은 원인으로
    `알 수 없는 도구: 'Agent'` 버그가 났었다.
    """
    from core.tools.implementations.client_tool import build_client_tools
    from web.app import _client_tools_instruction

    note = _client_tools_instruction(
        build_client_tools([{"function": {"name": "read_file", "parameters": {}}}])[0]
    )

    assert "read_file" in note
    assert "overrides" in note  # 앞의 도구 안내를 덮는다고 명시한다
    assert _client_tools_instruction([]) == ""  # 도구가 없으면 아무 말도 하지 않는다


async def test_endpoint_returns_tool_calls_and_finish_reason(_tool_engine):
    """★클라이언트는 finish_reason='tool_calls' 를 보고 도구를 실행한다."""
    from web.app import OpenAIChatCompletionRequest, chat_completions

    resp = await chat_completions(
        OpenAIChatCompletionRequest(
            messages=[{"role": "user", "content": "버그 고쳐줘"}],
            tools=[{"type": "function", "function": {"name": "read_file", "parameters": {}}}],
        )
    )

    choice = resp.choices[0]
    assert choice.finish_reason == "tool_calls"
    assert choice.message.tool_calls[0]["function"]["name"] == "read_file"
    assert json.loads(choice.message.tool_calls[0]["function"]["arguments"]) == {
        "path": "src/main.py"
    }


async def test_tool_choice_none_disables_client_tools(_tool_engine):
    """도구를 쓰지 말라고 했으면 tool_calls 를 돌려주지 않는다."""
    from web.app import OpenAIChatCompletionRequest, chat_completions

    resp = await chat_completions(
        OpenAIChatCompletionRequest(
            messages=[{"role": "user", "content": "버그 고쳐줘"}],
            tools=[{"type": "function", "function": {"name": "read_file", "parameters": {}}}],
            tool_choice="none",
        )
    )

    assert resp.choices[0].message.tool_calls is None
    assert resp.choices[0].finish_reason != "tool_calls"


async def test_no_tools_response_is_unchanged(_tool_engine):
    """무회귀 — 도구를 안 보낸 기존 소비자에게는 tool_calls 가 없다."""
    from web.app import OpenAIChatCompletionRequest, chat_completions

    resp = await chat_completions(
        OpenAIChatCompletionRequest(messages=[{"role": "user", "content": "안녕"}])
    )

    assert resp.choices[0].message.tool_calls is None


# ─────────────────────────────────────────────
# ★버려진 도구를 응답으로 알린다 (2026-08-08)
# ─────────────────────────────────────────────


async def test_dropped_tools_are_reported_in_response(_tool_engine):
    """상한을 넘겨 빠진 도구가 있으면 응답에 그 사실이 실린다.

    예전에는 조용히 잘라서, 플러그인 개발자가 "왜 내 도구를 안 쓰지?" 만 보고
    원인을 찾을 방법이 없었다. 도구 결과가 잘릴 때와 달리 도구 목록에는
    `…[중략]…` 같은 표식을 남길 자리가 없어 더더욱 알 수 없었다.
    """
    from core.tools.implementations.client_tool import MAX_CLIENT_TOOLS
    from web.app import OpenAIChatCompletionRequest, chat_completions

    resp = await chat_completions(
        OpenAIChatCompletionRequest(
            messages=[{"role": "user", "content": "버그 고쳐줘"}],
            tools=[
                {"type": "function", "function": {"name": f"t{i}", "parameters": {}}}
                for i in range(MAX_CLIENT_TOOLS + 3)
            ],
        )
    )

    assert resp.warnings, "무엇이 빠졌는지 응답에 남아야 한다"
    assert any("3건" in w for w in resp.warnings)


async def test_clean_request_has_no_warnings(_tool_engine):
    """무회귀 — 버린 것이 없으면 경고 필드는 빈 목록이다."""
    from web.app import OpenAIChatCompletionRequest, chat_completions

    resp = await chat_completions(
        OpenAIChatCompletionRequest(
            messages=[{"role": "user", "content": "버그 고쳐줘"}],
            tools=[{"type": "function", "function": {"name": "read_file", "parameters": {}}}],
        )
    )

    assert resp.warnings == []


async def test_all_tools_invalid_is_rejected_not_silently_ignored(_tool_engine):
    """★`tools` 를 보냈는데 유효 0개면 400 — 조용히 도구 없이 돌지 않는다.

    도구 없이 돌면 모델이 "파일을 읽을 수 없다"는 엉뚱한 답을 내고, 클라이언트는
    자기가 보낸 스키마가 통째로 무시됐다는 사실조차 모른다.
    """
    from web.app import OpenAIChatCompletionRequest, chat_completions

    with pytest.raises(HTTPException) as ei:
        await chat_completions(
            OpenAIChatCompletionRequest(
                messages=[{"role": "user", "content": "버그 고쳐줘"}],
                tools=[{"function": {"name": ""}}, {"nope": 1}],
            )
        )

    assert ei.value.status_code == 400
    assert "형식이 올바르지 않아" in str(ei.value.detail)


def test_stream_puts_warnings_on_the_first_frame():
    """스트림은 되돌릴 수 없다 — 경고는 **첫 프레임**에 실려야 의미가 있다.

    끝에 붙이면 클라이언트가 이미 tool_calls 를 처리한 뒤라 손쓸 수 없다.
    """
    import inspect

    from web import app as webapp

    src = inspect.getsource(webapp._openai_stream_generate)
    assert '_chunk({"role": "assistant"}, warnings=tool_warnings)' in src
    assert "tool_warnings" in inspect.signature(webapp._openai_stream_generate).parameters


async def test_continuation_does_not_add_a_user_message(_tool_engine):
    """★도구 결과로 이어 도는 호출은 사용자 발화를 새로 만들지 않는다."""
    from web.app import OpenAIChatCompletionRequest, chat_completions

    await chat_completions(
        OpenAIChatCompletionRequest(
            messages=[
                {"role": "user", "content": "버그 고쳐줘"},
                {"role": "assistant", "tool_calls": [READ_CALL]},
                {"role": "tool", "content": "print(1/0)", "tool_call_id": "call_abc123"},
            ],
            tools=[{"type": "function", "function": {"name": "read_file", "parameters": {}}}],
        )
    )

    assert _tool_engine.received_append_user_message is False
