"""
core/orchestrator/turn_state.py 단위 테스트.

TurnState 불변 객체, TurnStateStore 저장/조회, extract_turn_state() 추출 로직을 검증한다.
v7.0에서 추가된 턴 상태 외부화 모듈의 핵심 동작을 테스트한다.
"""

from __future__ import annotations

import dataclasses

import pytest

from core.orchestrator.turn_state import TurnState, TurnStateStore, extract_turn_state


# ─────────────────────────────────────────────
# TurnState 불변 객체 테스트
# ─────────────────────────────────────────────
class TestTurnState:
    """TurnState frozen dataclass의 불변성과 직렬화를 검증한다."""

    def test_turn_state_frozen_immutable(self):
        """frozen=True이므로 필드를 수정하면 FrozenInstanceError가 발생한다."""
        state = TurnState(facts=("file exists",), turn_number=1)
        with pytest.raises(dataclasses.FrozenInstanceError):
            state.facts = ("modified",)  # type: ignore[misc]

    def test_turn_state_frozen_turn_number_immutable(self):
        """turn_number도 수정 불가능하다."""
        state = TurnState(turn_number=5)
        with pytest.raises(dataclasses.FrozenInstanceError):
            state.turn_number = 10  # type: ignore[misc]

    def test_turn_state_to_context_string_with_facts(self):
        """facts가 있으면 'Facts:' 섹션이 포함된다."""
        state = TurnState(
            facts=("Read /tmp/a.py", "Edited /tmp/b.py"),
            turn_number=1,
        )
        ctx = state.to_context_string()
        assert "Facts:" in ctx
        assert "- Read /tmp/a.py" in ctx
        assert "- Edited /tmp/b.py" in ctx

    def test_turn_state_to_context_string_with_all_fields(self):
        """모든 필드가 채워지면 각 섹션이 모두 포함된다."""
        state = TurnState(
            facts=("f1",),
            todo=("t1",),
            touched_files=("/a.py", "/b.py"),
            unresolved_issues=("issue1",),
            last_tool_results=("result1",),
            scout_plan="plan text",
            turn_number=3,
        )
        ctx = state.to_context_string()
        # 각 섹션 키워드가 모두 존재하는지 확인
        assert "Facts:" in ctx
        assert "TODO:" in ctx
        assert "Files:" in ctx
        assert "Issues:" in ctx
        assert "Last results:" in ctx
        assert "Scout plan:" in ctx

    def test_turn_state_to_context_string_empty(self):
        """빈 상태에서는 빈 문자열을 반환한다."""
        state = TurnState()
        assert state.to_context_string() == ""

    def test_turn_state_estimated_tokens_calculation(self):
        """estimated_tokens()는 문자수 / 3으로 추정한다."""
        state = TurnState(facts=("a" * 90,))  # "Facts:\n- " + 90자 = 100자 정도
        ctx_len = len(state.to_context_string())
        expected = ctx_len // 3
        assert state.estimated_tokens() == expected

    def test_turn_state_estimated_tokens_empty(self):
        """빈 상태의 토큰 추정은 0이다."""
        state = TurnState()
        assert state.estimated_tokens() == 0

    def test_turn_state_default_values(self):
        """기본값이 올바르게 설정되는지 확인한다."""
        state = TurnState()
        assert state.facts == ()
        assert state.todo == ()
        assert state.touched_files == ()
        assert state.unresolved_issues == ()
        assert state.last_tool_results == ()
        assert state.scout_plan is None
        assert state.turn_number == 0
        assert state.user_request == ""


# ─────────────────────────────────────────────
# TurnStateStore 저장/조회 테스트
# ─────────────────────────────────────────────
class TestTurnStateStore:
    """TurnStateStore의 인메모리 저장/조회/삭제를 검증한다."""

    @pytest.fixture
    def store(self) -> TurnStateStore:
        """매 테스트마다 새로운 스토어를 생성한다."""
        return TurnStateStore()

    def test_save_and_get_latest(self, store: TurnStateStore):
        """save() 후 get_latest()로 가장 최근 상태를 조회할 수 있다."""
        state1 = TurnState(facts=("first",), turn_number=1)
        state2 = TurnState(facts=("second",), turn_number=2)
        store.save("sess-1", state1)
        store.save("sess-1", state2)

        latest = store.get_latest("sess-1")
        assert latest is not None
        assert latest.turn_number == 2
        assert latest.facts == ("second",)

    def test_get_latest_empty_session(self, store: TurnStateStore):
        """존재하지 않는 세션의 get_latest()는 None을 반환한다."""
        assert store.get_latest("nonexistent") is None

    def test_get_all_returns_copy(self, store: TurnStateStore):
        """get_all()은 원본이 아닌 복사본을 반환한다."""
        state = TurnState(turn_number=1)
        store.save("sess-1", state)

        all_states = store.get_all("sess-1")
        assert len(all_states) == 1
        assert all_states[0].turn_number == 1

        # 반환된 리스트를 수정해도 스토어에 영향을 주지 않는다
        all_states.clear()
        assert len(store.get_all("sess-1")) == 1

    def test_get_all_empty_session(self, store: TurnStateStore):
        """존재하지 않는 세션의 get_all()은 빈 리스트를 반환한다."""
        assert store.get_all("nonexistent") == []

    def test_get_context_within_budget(self, store: TurnStateStore):
        """토큰 예산 내의 상태들은 모두 컨텍스트에 포함된다."""
        # 짧은 상태 3개 저장 (각각 토큰이 적음)
        for i in range(3):
            state = TurnState(facts=(f"fact{i}",), turn_number=i)
            store.save("sess-1", state)

        ctx = store.get_context("sess-1", max_tokens=5000)
        # 3개 턴 모두 포함되어야 한다
        assert "[Turn 0]" in ctx
        assert "[Turn 1]" in ctx
        assert "[Turn 2]" in ctx

    def test_get_context_exceeds_budget_drops_old(self, store: TurnStateStore):
        """토큰 예산을 초과하면 오래된 턴부터 제외된다."""
        # 긴 facts를 가진 상태를 만들어 예산 초과를 유발한다
        long_fact = "x" * 300  # 약 100토큰 (300자 / 3)
        for i in range(5):
            state = TurnState(facts=(long_fact,), turn_number=i)
            store.save("sess-1", state)

        # 예산을 200토큰으로 제한 → 최대 2개 턴만 포함 가능
        ctx = store.get_context("sess-1", max_tokens=200)
        # 가장 최신 턴(4)은 반드시 포함
        assert "[Turn 4]" in ctx
        # 가장 오래된 턴(0)은 제외될 수 있다
        assert "[Turn 0]" not in ctx

    def test_get_context_empty_session(self, store: TurnStateStore):
        """빈 세션의 get_context()는 빈 문자열을 반환한다."""
        assert store.get_context("nonexistent") == ""

    def test_clear_removes_session(self, store: TurnStateStore):
        """clear()로 세션을 삭제하면 데이터가 사라진다."""
        state = TurnState(turn_number=1)
        store.save("sess-1", state)
        assert store.get_latest("sess-1") is not None

        store.clear("sess-1")
        assert store.get_latest("sess-1") is None
        assert store.get_all("sess-1") == []

    def test_clear_nonexistent_session_no_error(self, store: TurnStateStore):
        """존재하지 않는 세션을 clear()해도 에러가 발생하지 않는다."""
        store.clear("nonexistent")  # 예외 없이 통과해야 한다

    def test_session_count(self, store: TurnStateStore):
        """session_count가 저장된 세션 수를 정확히 반환한다."""
        assert store.session_count == 0
        store.save("sess-1", TurnState(turn_number=1))
        store.save("sess-2", TurnState(turn_number=1))
        assert store.session_count == 2

    def test_get_context_order_is_chronological(self, store: TurnStateStore):
        """get_context()는 시간순(오래된 → 최신)으로 출력한다."""
        for i in range(3):
            store.save("sess-1", TurnState(facts=(f"f{i}",), turn_number=i))

        ctx = store.get_context("sess-1", max_tokens=5000)
        # Turn 0이 Turn 2보다 먼저 나와야 한다
        pos0 = ctx.index("[Turn 0]")
        pos2 = ctx.index("[Turn 2]")
        assert pos0 < pos2


# ─────────────────────────────────────────────
# extract_turn_state() 추출 로직 테스트
# ─────────────────────────────────────────────
class TestExtractTurnState:
    """extract_turn_state()의 규칙 기반 정보 추출을 검증한다."""

    def test_extract_read_tool_creates_fact(self):
        """Read 도구 호출 시 'Read {path}' fact가 생성된다."""
        blocks = [{"name": "Read", "input": {"file_path": "/tmp/test.py"}}]
        state = extract_turn_state(
            turn_number=1,
            user_request="파일 읽어줘",
            assistant_text="",
            tool_use_blocks=blocks,
        )
        assert "Read /tmp/test.py" in state.facts
        assert "/tmp/test.py" in state.touched_files

    def test_extract_bash_tool_summarizes_command(self):
        """Bash 도구 호출 시 'Ran: {command}' fact가 생성된다."""
        blocks = [{"name": "Bash", "input": {"command": "ls -la /home"}}]
        state = extract_turn_state(
            turn_number=2,
            user_request="디렉토리 확인",
            assistant_text="",
            tool_use_blocks=blocks,
        )
        assert any("Ran: ls -la /home" in f for f in state.facts)

    def test_extract_bash_long_command_truncated(self):
        """80자를 초과하는 명령어는 잘린다."""
        long_cmd = "echo " + "a" * 200
        blocks = [{"name": "Bash", "input": {"command": long_cmd}}]
        state = extract_turn_state(
            turn_number=1,
            user_request="test",
            assistant_text="",
            tool_use_blocks=blocks,
        )
        # Bash 명령어는 80자로 잘린다
        bash_fact = [f for f in state.facts if "Ran:" in f][0]
        # "Ran: " (5자) + 80자 = 85자 이하
        cmd_part = bash_fact.replace("Ran: ", "")
        assert len(cmd_part) <= 80

    def test_extract_empty_input(self):
        """빈 입력이면 빈 TurnState를 반환한다."""
        state = extract_turn_state(
            turn_number=0,
            user_request="",
            assistant_text="",
            tool_use_blocks=[],
        )
        assert state.facts == ()
        assert state.todo == ()
        assert state.touched_files == ()
        assert state.turn_number == 0

    def test_extract_multiple_tools_deduplicates_files(self):
        """같은 파일에 대한 여러 도구 호출 시 touched_files에서 중복이 제거된다."""
        blocks = [
            {"name": "Read", "input": {"file_path": "/tmp/a.py"}},
            {"name": "Edit", "input": {"file_path": "/tmp/a.py"}},
            {"name": "Read", "input": {"file_path": "/tmp/b.py"}},
        ]
        state = extract_turn_state(
            turn_number=1,
            user_request="파일 수정",
            assistant_text="",
            tool_use_blocks=blocks,
        )
        # 중복 제거 확인: /tmp/a.py는 한 번만 나와야 한다
        assert state.touched_files.count("/tmp/a.py") == 1
        assert "/tmp/b.py" in state.touched_files
        assert len(state.touched_files) == 2

    def test_extract_todo_from_assistant_text(self):
        """assistant_text에서 TODO/해야/필요 등의 패턴이 todo로 추출된다."""
        text = "파일을 확인했습니다.\n다음에 테스트를 실행해야 합니다.\nTODO: 리팩토링 필요합니다."
        state = extract_turn_state(
            turn_number=1,
            user_request="분석해줘",
            assistant_text=text,
            tool_use_blocks=[],
        )
        # "해야"와 "TODO" 패턴이 모두 추출된다
        assert len(state.todo) >= 2

    def test_extract_todo_skips_short_lines(self):
        """10자 이하의 짧은 TODO 라인은 무시된다."""
        text = "TODO: ok\nTODO: 이것은 충분히 긴 할 일 항목입니다"
        state = extract_turn_state(
            turn_number=1,
            user_request="test",
            assistant_text=text,
            tool_use_blocks=[],
        )
        # "TODO: ok" (8자)는 무시, 긴 항목만 추출
        assert len(state.todo) == 1

    def test_extract_short_response_added_as_fact(self):
        """200자 미만의 짧은 응답은 fact로 추가된다."""
        short_text = "파일이 존재합니다."
        state = extract_turn_state(
            turn_number=1,
            user_request="확인",
            assistant_text=short_text,
            tool_use_blocks=[],
        )
        assert any("Response:" in f for f in state.facts)

    def test_extract_long_response_not_added_as_fact(self):
        """200자 이상의 긴 응답은 fact에 추가되지 않는다."""
        long_text = "내용 " * 100  # 300자 이상
        state = extract_turn_state(
            turn_number=1,
            user_request="분석",
            assistant_text=long_text,
            tool_use_blocks=[],
        )
        assert not any("Response:" in f for f in state.facts)

    def test_extract_tool_results_summary(self):
        """tool_results가 있으면 last_tool_results에 요약이 포함된다."""
        results = ["파일 읽기 성공: /tmp/a.py 내용은...", "검색 결과: 3건 발견"]
        state = extract_turn_state(
            turn_number=1,
            user_request="test",
            assistant_text="",
            tool_use_blocks=[],
            tool_results=results,
        )
        assert len(state.last_tool_results) == 2

    def test_extract_tool_results_truncated(self):
        """100자를 초과하는 결과는 '...'으로 잘린다."""
        long_result = "x" * 200
        state = extract_turn_state(
            turn_number=1,
            user_request="test",
            assistant_text="",
            tool_use_blocks=[],
            tool_results=[long_result],
        )
        assert state.last_tool_results[0].endswith("...")
        assert len(state.last_tool_results[0]) == 103  # 100자 + "..."

    def test_extract_user_request_truncated(self):
        """user_request가 200자를 초과하면 잘린다."""
        long_request = "요청 " * 200  # 600자
        state = extract_turn_state(
            turn_number=1,
            user_request=long_request,
            assistant_text="",
            tool_use_blocks=[],
        )
        assert len(state.user_request) <= 200

    def test_extract_write_edit_glob_grep_tools(self):
        """Write, Edit, Glob, Grep 도구가 각각 올바른 fact를 생성한다."""
        blocks = [
            {"name": "Write", "input": {"file_path": "/tmp/w.py"}},
            {"name": "Edit", "input": {"file_path": "/tmp/e.py"}},
            {"name": "Glob", "input": {"pattern": "**/*.py"}},
            {"name": "Grep", "input": {"pattern": "import"}},
        ]
        state = extract_turn_state(
            turn_number=1,
            user_request="test",
            assistant_text="",
            tool_use_blocks=blocks,
        )
        assert "Wrote /tmp/w.py" in state.facts
        assert "Edited /tmp/e.py" in state.facts
        assert "Glob: **/*.py" in state.facts
        assert "Grep: import" in state.facts

    def test_extract_unknown_tool_generic_fact(self):
        """알 수 없는 도구는 'Used {name}' fact를 생성한다."""
        blocks = [{"name": "CustomTool", "input": {}}]
        state = extract_turn_state(
            turn_number=1,
            user_request="test",
            assistant_text="",
            tool_use_blocks=blocks,
        )
        assert "Used CustomTool" in state.facts

    def test_extract_max_tool_results_is_five(self):
        """last_tool_results는 최대 5개로 제한된다."""
        results = [f"result-{i}" for i in range(10)]
        state = extract_turn_state(
            turn_number=1,
            user_request="test",
            assistant_text="",
            tool_use_blocks=[],
            tool_results=results,
        )
        assert len(state.last_tool_results) == 5
