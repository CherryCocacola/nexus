"""
core/state.py 단위 테스트.

GlobalState 싱글톤의 초기화, 토큰 카운터, 턴 관리를 검증한다.
"""

from __future__ import annotations

import threading

import pytest

from core.state import (
    PermissionModeValue,
    get_initial_state,
    get_state,
    reset_state_for_testing,
)


# 매 테스트 전에 싱글톤을 리셋한다
@pytest.fixture(autouse=True)
def _reset_state():
    """테스트 간 싱글톤 상태를 격리한다."""
    reset_state_for_testing()
    yield
    reset_state_for_testing()


class TestGlobalStateInit:
    """GlobalState 초기화 테스트."""

    def test_get_initial_state_creates_singleton(self):
        """get_initial_state()가 싱글톤을 생성하는지 확인한다."""
        state1 = get_initial_state()
        state2 = get_initial_state()
        assert state1 is state2

    def test_get_initial_state_sets_cwd(self, tmp_path):
        """지정한 cwd가 올바르게 설정되는지 확인한다."""
        state = get_initial_state(cwd=str(tmp_path))
        # realpath로 해석되므로 정규화된 경로와 비교
        assert tmp_path.resolve().as_posix() in state.cwd.replace("\\", "/")

    def test_get_initial_state_generates_session_id(self):
        """세션 ID가 자동 생성되는지 확인한다."""
        state = get_initial_state()
        assert state.session_id is not None
        assert len(state.session_id) > 0

    def test_get_state_before_init_raises(self):
        """초기화 전에 get_state()를 호출하면 RuntimeError가 발생한다."""
        with pytest.raises(RuntimeError, match="초기화되지 않았습니다"):
            get_state()

    def test_get_state_after_init_returns_same(self):
        """초기화 후 get_state()가 동일 객체를 반환한다."""
        state = get_initial_state()
        assert get_state() is state

    def test_default_permission_mode(self):
        """기본 권한 모드가 DEFAULT인지 확인한다."""
        state = get_initial_state()
        assert state.permission_mode == PermissionModeValue.DEFAULT


class TestTokenTracking:
    """토큰 카운터 스레드 안전성 테스트."""

    def test_increment_tokens(self):
        """토큰 카운터가 정확히 증가하는지 확인한다."""
        state = get_initial_state()
        state.increment_tokens(input_tokens=100, output_tokens=50)

        assert state.total_input_tokens == 100
        assert state.total_output_tokens == 50
        assert state.total_api_calls == 1

    def test_increment_tokens_thread_safety(self):
        """여러 스레드에서 동시에 토큰을 증가시켜도 정확한지 확인한다."""
        state = get_initial_state()
        iterations = 1000

        def increment():
            for _ in range(iterations):
                state.increment_tokens(input_tokens=1, output_tokens=1)

        threads = [threading.Thread(target=increment) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        expected = iterations * 4
        assert state.total_input_tokens == expected
        assert state.total_output_tokens == expected
        assert state.total_api_calls == expected

    def test_increment_tool_calls(self):
        """도구 호출 카운터가 정확히 증가하는지 확인한다."""
        state = get_initial_state()
        state.increment_tool_calls(3)
        assert state.total_tool_calls == 3
        assert state.current_turn_tool_calls == 3


class TestTurnManagement:
    """턴 관리 테스트."""

    def test_start_new_turn_resets_counters(self):
        """새 턴 시작 시 턴별 카운터가 리셋되는지 확인한다."""
        state = get_initial_state()

        # 첫 번째 턴에서 토큰 사용
        state.increment_tokens(input_tokens=100, output_tokens=50)
        assert state.current_turn_input_tokens == 100

        # 새 턴 시작
        state.start_new_turn()
        assert state.current_turn_input_tokens == 0
        assert state.current_turn_output_tokens == 0
        assert state.current_turn_tool_calls == 0

        # 전체 누적은 유지
        assert state.total_input_tokens == 100
        assert state.total_turns == 1

    def test_session_summary(self):
        """세션 요약이 올바른 형식으로 반환되는지 확인한다."""
        state = get_initial_state()
        state.increment_tokens(input_tokens=500, output_tokens=200)
        state.increment_tool_calls(3)

        summary = state.get_session_summary()
        assert summary["total_input_tokens"] == 500
        assert summary["total_output_tokens"] == 200
        assert summary["total_tool_calls"] == 3
        assert summary["permission_mode"] == "default"
        assert "session_id" in summary


class TestMcpVisibilityFields:
    """MCP 가시성 필드(mcp_servers/mcp_connected)의 기본값과 인스턴스 격리를 검증한다.

    Phase 2 부트스트랩이 채우기 전에는 빈 컨테이너여야 하고, default_factory 로
    선언되어 인스턴스마다 별개의 객체를 가져야 한다(가변 기본값 공유 버그 방지)."""

    def test_mcp_fields_default_empty(self):
        """기본값: mcp_servers=={} 이고 mcp_connected==set() 이어야 한다."""
        state = get_initial_state()
        assert state.mcp_servers == {}
        assert state.mcp_connected == set()

    def test_mcp_fields_isolated_per_instance(self):
        """두 GlobalState 인스턴스가 mcp_servers/mcp_connected 객체를 공유하면 안 된다.

        dataclass 의 default_factory(dict/set) 가 인스턴스마다 새 객체를 만드는지
        확인한다. 한 인스턴스에 추가해도 다른 인스턴스에는 영향이 없어야 한다."""
        # 싱글톤을 우회해 직접 두 인스턴스를 만든다(격리 검증 목적).
        from core.state import GlobalState

        a = GlobalState()
        b = GlobalState()
        a.mcp_servers["kowiki"] = {"tool_count": 1}
        a.mcp_connected.add("kowiki")

        assert b.mcp_servers == {}
        assert b.mcp_connected == set()
        # 같은 객체를 공유하지 않아야 한다(가변 기본값 공유 안티패턴 방지).
        assert a.mcp_servers is not b.mcp_servers
        assert a.mcp_connected is not b.mcp_connected
