"""
web/app.py 의 /v1/chat 비스트리밍 핸들러 — tool_calls 응답 보강 단위 테스트.

검증 대상 (후속 구현 A):
  비스트리밍 chat() 핸들러가 submit_message 가 yield 하는 StreamEvent 를 소비해
  ChatResponse.tool_calls 를 채우는 동작.
    - TOOL_USE_STOP(event.tool_use=ToolUseBlock) → 호출 1건 등록(name/input/id).
    - TOOL_RESULT(event.tool_result=ToolResultBlock) → tool_use_id 매칭으로
      result/is_error 채움.
    - 결과 본문 500자 절단(…(truncated) 접미).
    - STOP 없이 RESULT 만 온 방어 케이스(결과만으로 항목 생성).
    - 호출당 ToolCallInfo 1개, 등장 순서 유지.

테스트 격리 (운영 의존성 없음):
  실제 QueryEngine/Redis/PG/vLLM 미사용. _app_state["query_engine"] 를 가짜
  엔진으로 갈아끼우고, memory_manager 는 None 으로 두어 세션 복원 분기를 건너뛴다.
  매 테스트 후 _app_state 를 원복해 테스트 간 독립성을 보장한다(testing.md).

비스트리밍 핸들러는 async 함수이므로 asyncio_mode=auto(pytest.ini)에 따라
async 테스트 함수에서 직접 await 한다.
"""

from __future__ import annotations

import pytest

from core.message import (
    KnowledgeCitation,
    StreamEvent,
    StreamEventType,
    ToolResultBlock,
    ToolUseBlock,
)
from web.app import (
    ChatRequest,
    _app_state,
    _strip_invalid_citation_labels,
    chat,
)


# ─────────────────────────────────────────────
# 가짜 QueryEngine — submit_message 가 미리 지정한 StreamEvent 들을 yield
# ─────────────────────────────────────────────
class _FakeEngine:
    """
    chat() 핸들러가 호출하는 최소 인터페이스만 구현한 가짜 엔진.

    chat() 가 사용하는 표면:
      - bind_request(session_id, tenant, transcript): 세션/테넌트 주입(여기선 무시).
      - clear_messages(): 메시지 초기화(memory_manager 가 None 이면 호출 안 됨).
      - _messages: 메시지 리스트(복원 분기에서만 접근).
      - session_id: 응답에 담길 세션 ID.
      - submit_message(text): StreamEvent 를 yield 하는 AsyncGenerator.

    핵심: 생성자에 "yield 할 이벤트 목록"을 받아 그대로 흘려보낸다. 이로써
    실제 모델/도구 실행 없이 tool_calls 수집 로직만 격리 검증한다.
    """

    def __init__(self, events: list[StreamEvent], session_id: str = "sess-A") -> None:
        self._events = events
        self.session_id = session_id
        self._messages: list = []
        # 검증용 — 마지막으로 받은 사용자 메시지를 기록한다.
        self.last_message: str | None = None

    def bind_request(self, **kwargs) -> None:  # noqa: D401 — 가짜 주입(동작 없음)
        """세션/테넌트/트랜스크립트 주입. 가짜에서는 기록하지 않고 무시한다."""

    def clear_messages(self) -> None:
        """메시지 초기화(가짜)."""
        self._messages.clear()

    async def submit_message(self, message: str):
        """미리 지정한 StreamEvent 목록을 순서대로 yield 한다."""
        self.last_message = message
        for ev in self._events:
            yield ev


@pytest.fixture
def _restore_app_state():
    """
    _app_state 의 query_engine/memory_manager 를 테스트 후 원복하는 픽스처.

    web.app 은 모듈 레벨 _app_state 싱글톤을 공유하므로, 테스트가 끝나면
    반드시 원래 값으로 되돌려 다른 테스트에 영향을 주지 않게 한다.
    """
    saved_engine = _app_state.get("query_engine")
    saved_mm = _app_state.get("memory_manager")
    # 세션 복원 분기를 건너뛰도록 memory_manager 는 명시적으로 None 으로 둔다.
    _app_state["memory_manager"] = None
    yield
    _app_state["query_engine"] = saved_engine
    _app_state["memory_manager"] = saved_mm


# 이벤트 생성 헬퍼 — 가독성을 위해 짧은 팩토리로 묶는다.
def _stop(name: str, tool_input: dict, tu_id: str) -> StreamEvent:
    """TOOL_USE_STOP 이벤트(도구 호출 확정)를 만든다."""
    return StreamEvent(
        type=StreamEventType.TOOL_USE_STOP,
        tool_use=ToolUseBlock(id=tu_id, name=name, input=tool_input),
    )


def _result(tu_id: str, content: str, is_error: bool = False) -> StreamEvent:
    """TOOL_RESULT 이벤트(도구 결과)를 만든다."""
    return StreamEvent(
        type=StreamEventType.TOOL_RESULT,
        tool_result=ToolResultBlock(tool_use_id=tu_id, content=content, is_error=is_error),
    )


def _text(text: str) -> StreamEvent:
    """TEXT_DELTA 이벤트(응답 텍스트 조각)를 만든다."""
    return StreamEvent(type=StreamEventType.TEXT_DELTA, text=text)


# ─────────────────────────────────────────────
# tool_calls 수집 — 정상 경로
# ─────────────────────────────────────────────
class TestChatToolCallsHappyPath:
    """STOP→RESULT 매칭으로 tool_calls 가 올바르게 채워지는지 검증한다."""

    async def test_chat_single_tool_call_populates_name_input_result(self, _restore_app_state):
        """STOP+RESULT 한 쌍이 오면 name/input/result/is_error 가 모두 채워진다."""
        events = [
            _text("작업을 시작합니다. "),
            _stop("Read", {"file_path": "/tmp/x.txt"}, "toolu_1"),
            _result("toolu_1", "파일 내용입니다"),
            _text("완료했습니다."),
        ]
        _app_state["query_engine"] = _FakeEngine(events)

        resp = await chat(ChatRequest(message="파일 읽어줘", session_id="sess-A"))

        # 응답 텍스트는 TEXT_DELTA 누적
        assert resp.response == "작업을 시작합니다. 완료했습니다."
        # tool_calls 1건
        assert len(resp.tool_calls) == 1
        tc = resp.tool_calls[0]
        assert tc.name == "Read"
        assert tc.input_data == {"file_path": "/tmp/x.txt"}
        assert tc.result == "파일 내용입니다"
        assert tc.is_error is False

    async def test_chat_tool_result_error_flag_propagates(self, _restore_app_state):
        """TOOL_RESULT 의 is_error=True 가 ToolCallInfo.is_error 로 전파된다."""
        events = [
            _stop("Bash", {"command": "ls /nope"}, "toolu_err"),
            _result("toolu_err", "<tool_use_error>경로 없음</tool_use_error>", is_error=True),
        ]
        _app_state["query_engine"] = _FakeEngine(events)

        resp = await chat(ChatRequest(message="실행", session_id="sess-A"))

        assert len(resp.tool_calls) == 1
        assert resp.tool_calls[0].is_error is True
        assert "경로 없음" in resp.tool_calls[0].result

    async def test_chat_session_id_comes_from_engine(self, _restore_app_state):
        """응답 session_id 는 engine.session_id 를 그대로 사용한다."""
        _app_state["query_engine"] = _FakeEngine([], session_id="engine-sess-99")

        resp = await chat(ChatRequest(message="hi", session_id="req-sess"))

        assert resp.session_id == "engine-sess-99"


# ─────────────────────────────────────────────
# tool_use_id 매칭 — 여러 호출의 순서/매칭
# ─────────────────────────────────────────────
class TestChatToolCallsMatching:
    """여러 도구 호출이 tool_use_id 로 정확히 매칭되고 순서가 유지되는지 검증한다."""

    async def test_chat_multiple_tools_match_by_id_preserve_order(self, _restore_app_state):
        """
        3개의 호출/결과가 섞여 와도 tool_use_id 로 짝지어지고,
        STOP 등장 순서대로 결과 목록이 정렬된다.
        """
        # 결과가 호출 순서와 다르게 도착해도(2번 결과가 먼저) id 매칭으로 올바르게 채워야 한다.
        events = [
            _stop("Read", {"p": "a"}, "id-a"),
            _stop("Grep", {"q": "b"}, "id-b"),
            _result("id-b", "grep 결과 b"),  # b 결과가 먼저 도착
            _stop("Write", {"p": "c"}, "id-c"),
            _result("id-a", "read 결과 a"),
            _result("id-c", "write 결과 c"),
        ]
        _app_state["query_engine"] = _FakeEngine(events)

        resp = await chat(ChatRequest(message="다중 도구", session_id="sess-A"))

        # 호출당 1개씩, 총 3건
        assert len(resp.tool_calls) == 3
        # 순서는 STOP 등장 순서(a → b → c)를 유지한다.
        assert [tc.name for tc in resp.tool_calls] == ["Read", "Grep", "Write"]
        # id 매칭으로 결과가 올바른 호출에 들어갔는지(b가 먼저 와도)
        by_name = {tc.name: tc for tc in resp.tool_calls}
        assert by_name["Read"].result == "read 결과 a"
        assert by_name["Grep"].result == "grep 결과 b"
        assert by_name["Write"].result == "write 결과 c"

    async def test_chat_duplicate_stop_same_id_registers_once(self, _restore_app_state):
        """같은 tool_use_id 의 STOP 이 두 번 와도 호출은 1건만 등록된다(중복 방지)."""
        events = [
            _stop("Read", {"p": "a"}, "dup-id"),
            _stop("Read", {"p": "a"}, "dup-id"),  # 중복 STOP
            _result("dup-id", "내용"),
        ]
        _app_state["query_engine"] = _FakeEngine(events)

        resp = await chat(ChatRequest(message="중복", session_id="sess-A"))

        assert len(resp.tool_calls) == 1
        assert resp.tool_calls[0].result == "내용"


# ─────────────────────────────────────────────
# 방어/경계 케이스
# ─────────────────────────────────────────────
class TestChatToolCallsDefensive:
    """STOP 누락, 결과 절단, 결과 없는 호출 등 경계 동작을 검증한다."""

    async def test_chat_result_without_stop_creates_orphan_entry(self, _restore_app_state):
        """
        STOP 없이 RESULT 만 도착한 방어 케이스 — 결과만으로 항목을 만든다.

        이 경우 name 은 알 수 없으므로 빈 문자열, input 은 빈 dict 이지만
        result/is_error 는 채워져야 한다(결과를 잃지 않는다).
        """
        events = [
            _result("orphan-id", "고아 결과", is_error=True),
        ]
        _app_state["query_engine"] = _FakeEngine(events)

        resp = await chat(ChatRequest(message="고아", session_id="sess-A"))

        assert len(resp.tool_calls) == 1
        tc = resp.tool_calls[0]
        assert tc.name == ""
        assert tc.input_data == {}
        assert tc.result == "고아 결과"
        assert tc.is_error is True

    async def test_chat_long_result_is_truncated_at_500_chars(self, _restore_app_state):
        """500자를 초과하는 결과는 500자 + '…(truncated)' 로 절단된다."""
        long_content = "가" * 600  # 600자
        events = [
            _stop("Read", {"p": "big"}, "big-id"),
            _result("big-id", long_content),
        ]
        _app_state["query_engine"] = _FakeEngine(events)

        resp = await chat(ChatRequest(message="긴결과", session_id="sess-A"))

        tc = resp.tool_calls[0]
        # 앞 500자는 보존되고 절단 접미가 붙는다.
        assert tc.result == "가" * 500 + "…(truncated)"
        assert tc.result.endswith("…(truncated)")

    async def test_chat_short_result_not_truncated(self, _restore_app_state):
        """500자 이하의 결과는 그대로 보존된다(절단 접미 없음)."""
        content = "나" * 500  # 정확히 500자 — 경계값(절단 안 함)
        events = [
            _stop("Read", {"p": "edge"}, "edge-id"),
            _result("edge-id", content),
        ]
        _app_state["query_engine"] = _FakeEngine(events)

        resp = await chat(ChatRequest(message="경계", session_id="sess-A"))

        tc = resp.tool_calls[0]
        assert tc.result == content
        assert "(truncated)" not in tc.result

    async def test_chat_stop_without_result_keeps_result_none(self, _restore_app_state):
        """RESULT 없이 STOP 만 온 호출은 name/input 만 채워지고 result 는 None 으로 남는다."""
        events = [
            _stop("Task", {"prompt": "go"}, "no-result-id"),
        ]
        _app_state["query_engine"] = _FakeEngine(events)

        resp = await chat(ChatRequest(message="결과없음", session_id="sess-A"))

        assert len(resp.tool_calls) == 1
        tc = resp.tool_calls[0]
        assert tc.name == "Task"
        assert tc.input_data == {"prompt": "go"}
        assert tc.result is None
        assert tc.is_error is False

    async def test_chat_no_tool_events_yields_empty_tool_calls(self, _restore_app_state):
        """도구 이벤트가 전혀 없으면 tool_calls 는 빈 목록이다(회귀 방지)."""
        events = [_text("그냥 답변입니다.")]
        _app_state["query_engine"] = _FakeEngine(events)

        resp = await chat(ChatRequest(message="인사", session_id="sess-A"))

        assert resp.tool_calls == []
        assert resp.response == "그냥 답변입니다."


# ─────────────────────────────────────────────
# 엔진 미초기화 — placeholder 응답(회귀)
# ─────────────────────────────────────────────
class TestChatEngineUninitialized:
    """query_engine 이 없을 때 placeholder 응답을 돌려주는 기존 동작이 유지되는지."""

    async def test_chat_no_engine_returns_placeholder(self, _restore_app_state):
        """query_engine 이 None 이면 초기화 안내 메시지를 빈 tool_calls 와 함께 반환한다."""
        _app_state["query_engine"] = None

        resp = await chat(ChatRequest(message="hi", session_id="given-sess"))

        # 엔진이 없으면 요청 세션 ID 를 그대로 사용한다.
        assert resp.session_id == "given-sess"
        assert "초기화" in resp.response
        assert resp.tool_calls == []


# ─────────────────────────────────────────────
# 지식 RAG 출처 인용(Point 4-2) — /v1/chat sources 필드 + 허위 라벨 strip
# ─────────────────────────────────────────────
def _sources(cites: list[KnowledgeCitation]) -> StreamEvent:
    """KNOWLEDGE_SOURCES 이벤트(지식 RAG 출처 목록)를 만든다."""
    return StreamEvent(
        type=StreamEventType.KNOWLEDGE_SOURCES, knowledge_sources=cites
    )


class TestChatCitationSources:
    """KNOWLEDGE_SOURCES 이벤트가 ChatResponse.sources로 노출되고 허위 라벨이 제거되는지."""

    async def test_chat_sources_populated_and_invalid_label_stripped(
        self, _restore_app_state
    ):
        """(c) sources 필드 렌더 + (b) 주입 범위 밖 [출처9] 마커 제거를 함께 검증한다.

        config 미주입 시 CitationConfig 기본값(expose=True/strip=True/label='출처')으로
        폴백하므로, 유효 번호(1)는 유지되고 지어낸 번호(9)는 응답에서 사라진다.
        """
        cite = KnowledgeCitation(
            index=1, source="kowiki", title="바흐", section="생애", score=0.95
        )
        events = [
            _sources([cite]),
            _text("바흐는 1750년에 사망했다 [출처1]. 근거 없는 문장 [출처9]."),
        ]
        _app_state["query_engine"] = _FakeEngine(events)

        resp = await chat(ChatRequest(message="바흐", session_id="sess-A"))

        # (c) sources 필드가 서버 진실(retriever 메타)로 채워진다.
        assert len(resp.sources) == 1
        assert resp.sources[0].index == 1
        assert resp.sources[0].title == "바흐"
        assert resp.sources[0].section == "생애"
        # (b) 유효 라벨은 유지, 주입 범위 밖 라벨은 제거된다.
        assert "[출처1]" in resp.response
        assert "[출처9]" not in resp.response

    async def test_chat_no_knowledge_sources_leaves_sources_empty(
        self, _restore_app_state
    ):
        """(d) KNOWLEDGE_SOURCES 이벤트가 없으면 sources는 빈 리스트(기존 동작 무회귀)."""
        events = [_text("그냥 답변입니다.")]
        _app_state["query_engine"] = _FakeEngine(events)

        resp = await chat(ChatRequest(message="hi", session_id="sess-A"))

        assert resp.sources == []
        assert resp.response == "그냥 답변입니다."


class TestStripInvalidCitationLabels:
    """_strip_invalid_citation_labels 순수 함수 — 허위 라벨 제거 규칙 격리 검증."""

    def test_keeps_valid_removes_invalid(self):
        """유효 인덱스 라벨은 유지, 그 외([출처9])는 제거한다."""
        text = "문장A [출처1]. 문장B [출처9]. 문장C [출처2]."
        out = _strip_invalid_citation_labels(text, valid_indices={1, 2}, label="출처")
        assert "[출처1]" in out
        assert "[출처2]" in out
        assert "[출처9]" not in out

    def test_respects_custom_label(self):
        """label이 'Source'면 [Source1] 형식을 대상으로 동작한다."""
        text = "fact [Source1]. bad [Source5]."
        out = _strip_invalid_citation_labels(
            text, valid_indices={1}, label="Source"
        )
        assert "[Source1]" in out
        assert "[Source5]" not in out
