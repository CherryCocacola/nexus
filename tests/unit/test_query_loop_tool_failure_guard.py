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
