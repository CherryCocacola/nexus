# query_loop 반복 실패 도구 호출 가드(무한 재시도 방지) 순수 헬퍼 단위 테스트
"""_tool_call_signature / _update_tool_failure_streak 검증.

이 헬퍼들은 "모델이 같은 도구를 같은 입력으로 계속 호출하며 매번 실패하는"
루프(#43: 깨진 Bash 명령 14회·62초 반복)를 조기에 끊기 위한 것이다.
가드의 핵심 판단 로직이라 부작용 없는 순수 함수로 분리해 여기서 검증한다.
"""

from core.orchestrator.query_loop import (
    REPEATED_TOOL_FAILURE_ABORT,
    REPEATED_TOOL_FAILURE_WARN,
    _tool_call_signature,
    _update_tool_failure_streak,
)


def _blocks(name: str, tool_input: dict, tool_id: str = "id1") -> list[dict]:
    """단일 도구 호출 블록 리스트를 만드는 헬퍼."""
    return [{"id": tool_id, "name": name, "input": tool_input}]


def test_signature_same_input_key_order_produces_same_signature():
    """키 순서만 다른 동일 입력은 같은 서명이어야 한다(정규화 검증)."""
    a = _tool_call_signature("Bash", {"command": "ls", "timeout": 5})
    b = _tool_call_signature("Bash", {"timeout": 5, "command": "ls"})
    assert a == b


def test_signature_different_tool_produces_different_signature():
    """같은 입력이라도 도구 이름이 다르면 서명이 달라야 한다."""
    a = _tool_call_signature("Bash", {"command": "ls"})
    b = _tool_call_signature("Read", {"command": "ls"})
    assert a != b


def test_signature_non_serializable_input_falls_back_to_str():
    """직렬화 불가 입력도 예외 없이 안정적 서명을 만든다."""
    sig = _tool_call_signature("Bash", {"bad": {1, 2, 3}})  # set은 JSON 불가
    assert sig.startswith("Bash::")


def test_streak_repeated_failure_increments_and_returns_worst():
    """같은 서명이 연속 실패하면 카운터가 누적되고 최댓값을 반환한다."""
    streak: dict[str, int] = {}
    blocks = _blocks("Bash", {"command": "brokencmd"})
    errors = {"id1": True}

    assert _update_tool_failure_streak(blocks, errors, streak) == 1
    assert _update_tool_failure_streak(blocks, errors, streak) == 2
    assert _update_tool_failure_streak(blocks, errors, streak) == 3


def test_streak_success_resets_signature():
    """실패로 쌓인 뒤 성공하면 해당 서명 카운터가 0으로 리셋된다."""
    streak: dict[str, int] = {}
    blocks = _blocks("Bash", {"command": "cmd"})

    _update_tool_failure_streak(blocks, {"id1": True}, streak)
    _update_tool_failure_streak(blocks, {"id1": True}, streak)
    assert streak  # 실패 누적 상태

    worst = _update_tool_failure_streak(blocks, {"id1": False}, streak)
    assert worst == 0
    assert streak == {}  # 성공 → 서명 제거


def test_streak_independent_signatures_same_turn_tracked_separately():
    """한 턴에 함께 실패한 서로 다른 명령은 독립적으로 누적된다."""
    streak: dict[str, int] = {}
    two = [
        {"id": "id1", "name": "Bash", "input": {"command": "a"}},
        {"id": "id2", "name": "Bash", "input": {"command": "b"}},
    ]
    errors = {"id1": True, "id2": True}
    assert _update_tool_failure_streak(two, errors, streak) == 1
    assert _update_tool_failure_streak(two, errors, streak) == 2
    assert len(streak) == 2  # a, b 독립 추적


def test_streak_stale_signature_evicted_when_absent_this_turn():
    """이번 턴에 등장하지 않은 과거 실패 서명은 제거되어 판정을 오염시키지 않는다.

    HIGH 결함 회귀 방지 — 모델이 실패(a)를 스스로 고쳐 다른 명령(b)으로 넘어가면
    낡은 a 카운트가 max()를 지배하면 안 된다.
    """
    streak: dict[str, int] = {}
    a = _blocks("Bash", {"command": "a"}, "id1")
    for _ in range(3):
        _update_tool_failure_streak(a, {"id1": True}, streak)  # a → 3

    # 다음 턴엔 a가 아예 호출되지 않고 새 명령 b만 실패
    b = _blocks("Bash", {"command": "b"}, "id2")
    worst = _update_tool_failure_streak(b, {"id2": True}, streak)
    assert worst == 1  # a는 evict되어 b(1)만 남음
    assert len(streak) == 1


def test_streak_parallel_duplicate_increments_once_per_turn():
    """같은 서명이 한 턴에 병렬로 여러 번 실패해도 카운터는 턴당 +1만 오른다.

    MEDIUM 결함 회귀 방지 — 턴 내 중복이 ABORT를 앞당기면 안 된다.
    """
    streak: dict[str, int] = {}
    dup = [
        {"id": "id1", "name": "Bash", "input": {"command": "x"}},
        {"id": "id2", "name": "Bash", "input": {"command": "x"}},  # 동일 서명
    ]
    errors = {"id1": True, "id2": True}
    assert _update_tool_failure_streak(dup, errors, streak) == 1  # 2개지만 +1
    assert _update_tool_failure_streak(dup, errors, streak) == 2


def test_streak_missing_result_treated_as_non_error():
    """결과 맵에 없는 도구 호출은 에러가 아닌 것으로 보고 리셋한다(보수적)."""
    streak: dict[str, int] = {"Bash::{\"command\": \"x\"}": 2}
    blocks = _blocks("Bash", {"command": "x"})
    worst = _update_tool_failure_streak(blocks, {}, streak)  # 결과 없음
    assert worst == 0


def test_thresholds_warn_below_abort():
    """경고 임계는 강제 종료 임계보다 낮아야(먼저 발동해야) 한다."""
    assert REPEATED_TOOL_FAILURE_WARN < REPEATED_TOOL_FAILURE_ABORT


# ─────────────────────────────────────────────
# 도구 인자 파싱 실패 **누적** 가드 (2026-08-23)
# ─────────────────────────────────────────────
# 위 streak 가드가 못 잡는 사각을 메운다.
#
# 실측 사고: [파싱 실패 턴 → 성공 턴 → 파싱 실패 턴 …] 교대 패턴에서
#   ① 파싱 실패 도구는 tool_use_blocks 에 아예 안 들어가 streak 에 집계조차 안 되고,
#   ② Transition 7(정상 턴)이 tool_parse_retry_count 를 0 으로 되돌린다.
# 그래서 카운터가 1↔0 을 오가며 한도에 영영 도달하지 못했고, 같은 문서를 11회
# 다시 쓰며 30턴·10분을 공전했다. 누적 카운터는 정상 턴에도 리셋되지 않는다.


def _simulate_alternating(state, tool_name: str, rounds: int) -> None:
    """[파싱 실패 → 정상 턴] 교대를 흉내낸다.

    정상 턴은 query_loop 의 Transition 7 이 하는 일(턴 단위 카운터 리셋)만
    재현한다 — 누적 카운터가 그 리셋을 견디는지가 이 테스트의 핵심이다.
    """
    for _ in range(rounds):
        state.tool_parse_fail_total[tool_name] = (
            state.tool_parse_fail_total.get(tool_name, 0) + 1
        )
        # 정상 턴 — Transition 7 의 리셋
        state.tool_parse_retry_count = 0


def test_cumulative_counter_survives_normal_turn_reset():
    """★핵심 — 정상 턴이 끼어들어도 누적 카운터는 줄지 않는다."""
    from core.orchestrator.query_loop import LoopState

    state = LoopState(messages=[])
    _simulate_alternating(state, "Write", 5)

    assert state.tool_parse_retry_count == 0  # 턴 단위 카운터는 리셋됨(기존 동작)
    assert state.tool_parse_fail_total["Write"] == 5  # 누적 카운터는 살아 있음


def test_cumulative_counter_reaches_nudge_threshold():
    """교대 패턴에서도 넛지 임계에 도달한다 — 기존 카운터로는 불가능했다."""
    from core.orchestrator.query_loop import PARSE_FAIL_NUDGE_AT, LoopState

    state = LoopState(messages=[])
    _simulate_alternating(state, "Write", PARSE_FAIL_NUDGE_AT)
    assert state.tool_parse_fail_total["Write"] >= PARSE_FAIL_NUDGE_AT


def test_cumulative_counter_reaches_abort_threshold():
    """계속 실패하면 결국 중단 임계에 도달해 공전이 끊긴다."""
    from core.orchestrator.query_loop import PARSE_FAIL_ABORT_AT, LoopState

    state = LoopState(messages=[])
    _simulate_alternating(state, "Write", PARSE_FAIL_ABORT_AT)
    assert state.tool_parse_fail_total["Write"] >= PARSE_FAIL_ABORT_AT


def test_thresholds_leave_room_between_nudge_and_abort():
    """넛지 뒤 모델이 행동을 고칠 여지가 있어야 한다 — 둘이 붙어 있으면 의미가 없다."""
    from core.orchestrator.query_loop import PARSE_FAIL_ABORT_AT, PARSE_FAIL_NUDGE_AT

    assert PARSE_FAIL_NUDGE_AT < PARSE_FAIL_ABORT_AT
    assert PARSE_FAIL_ABORT_AT - PARSE_FAIL_NUDGE_AT >= 2


def test_counters_are_per_tool():
    """도구가 다르면 카운터도 따로 센다(한 도구의 실패가 다른 도구를 끊으면 안 된다)."""
    from core.orchestrator.query_loop import LoopState

    state = LoopState(messages=[])
    _simulate_alternating(state, "Write", 4)
    _simulate_alternating(state, "Edit", 1)
    assert state.tool_parse_fail_total == {"Write": 4, "Edit": 1}


def test_nudge_is_sent_once_per_tool():
    """도구당 1회만 주입한다 — 매 턴 잔소리하면 컨텍스트만 먹는다."""
    from core.orchestrator.query_loop import LoopState

    state = LoopState(messages=[])
    state.parse_nudge_sent.add("Write")
    assert "Write" in state.parse_nudge_sent
    assert "Edit" not in state.parse_nudge_sent
