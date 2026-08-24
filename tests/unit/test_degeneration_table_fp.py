# 붕괴 감지기(DegenerationMonitor)가 정상 마크다운 표를 오탐하던 문제의 회귀 테스트.
"""
2026-08-23 실측 사고: NOVA CLI 로 테스트계획서를 생성하다 8열 표를 쓰던 중
스트림이 두 번 잘렸다(1849자·1756자). 붕괴가 아니라 **멀쩡한 표**였다.

원인은 B 신호(문자 4-gram 최빈 비율)다. 검사가 공백을 제거하고 세는데,
표 구분선 `|--------|--------|…` 이 `----` 를 수백 개 만들어 낸다.
실측 최빈 4-gram `'----'` 457회 / 비율 0.609 > 임계 0.45.

★이 파일의 테스트는 반드시 **소청크 스트리밍**으로 먹인다.
  `is_degenerate()` 는 `check_every` 케이던스와 `_last_check_len` 상태에 의존해
  **판정이 feed 입도에 따라 달라진다**. 같은 텍스트를 한 번에 넣으면 통과하는데
  100자씩 넣으면 발화하는 일이 실제로 있었다. 단발 feed 테스트는 이 클래스의
  버그를 구조적으로 못 잡는다.
"""

from __future__ import annotations

from core.orchestrator.stream_watchdog import DegenerationMonitor, _is_table_rule


def feed_streaming(monitor: DegenerationMonitor, text: str, chunk: int = 30) -> bool:
    """텍스트를 소청크로 흘려 넣으며 **모든 검사 시점**에서 붕괴 판정을 확인한다.

    실제 스트리밍은 델타가 잘게 도착하므로 검사가 여러 번 일어난다. 한 번이라도
    True 가 나오면 프로덕션에서는 그 시점에 스트림이 잘린다 — 그래서 마지막
    판정이 아니라 **누적 OR** 을 돌려준다.

    Args:
        monitor: 검사 대상 감시기.
        text: 흘려보낼 전체 텍스트.
        chunk: 한 번에 넣을 글자 수(실제 델타 크기와 비슷하게 작게).

    Returns:
        어느 시점에든 붕괴로 판정됐으면 True.
    """
    fired = False
    for i in range(0, len(text), chunk):
        monitor.feed(text[i : i + chunk])
        if monitor.is_degenerate():
            fired = True
    return fired


def _wide_table(rows: int = 14) -> str:
    """실제 사고를 낸 것과 같은 형태의 8열 정렬 표를 만든다(열 폭이 넓다)."""
    cols = [
        ("TC-ID", 7),
        ("구분", 10),
        ("대상 모듈", 20),
        ("사전조건", 30),
        ("입력", 26),
        ("실행 절차", 36),
        ("기대 결과", 40),
        ("판정 기준", 38),
    ]
    head = "| " + " | ".join(name.ljust(w) for name, w in cols) + " |"
    rule = "|" + "|".join("-" * (w + 2) for _, w in cols) + "|"
    body = []
    for i in range(rows):
        cells = [
            f"TC{i:02d}".ljust(7),
            ("단위" if i % 2 else "통합").ljust(10),
            f"data_loader_{i}".ljust(20),
            f"유효한 CSV 파일 {i} 존재".ljust(30),
            f"orders_{i}.csv".ljust(26),
            f"load_orders() 실행 후 {i}건 확인".ljust(36),
            f"Order 객체 {i}건으로 파싱됨".ljust(40),
            f"반환 리스트 길이 == {i}".ljust(38),
        ]
        body.append("| " + " | ".join(cells) + " |")
    return "\n".join([head, rule, *body])


class TestTableRulePredicate:
    """구분선 판정 술어 자체의 경계."""

    def test_complete_rule_detected(self) -> None:
        assert _is_table_rule("|--------|--------|--------|")

    def test_tail_incomplete_rule_detected(self) -> None:
        """생성 도중 window 끝에서 잘린 구분선 — 가장 흔한 형태다."""
        assert _is_table_rule("|-----|--")

    def test_head_truncated_rule_detected(self) -> None:
        """window 앞에서 잘린 구분선."""
        assert _is_table_rule("----|--------|")

    def test_alignment_colons_allowed(self) -> None:
        assert _is_table_rule("|:-------|-------:|:------:|")

    def test_pure_dash_flood_not_excluded(self) -> None:
        """★핵심 — 파이프가 없으면 구분선이 아니다. 진짜 대시 폭주를 지켜야 한다."""
        assert not _is_table_rule("-" * 200)

    def test_content_row_not_excluded(self) -> None:
        assert not _is_table_rule("| TC01 | 단위 | data_loader |")

    def test_blank_line_not_excluded(self) -> None:
        assert not _is_table_rule("   ")


class TestWideTableNoLongerFalsePositive:
    """정상 표가 더는 잘리지 않는다."""

    def test_wide_table_survives_streaming(self) -> None:
        """실제 사고 형태 — 8열 넓은 표를 30자씩 흘려도 한 번도 발화하지 않아야 한다."""
        assert feed_streaming(DegenerationMonitor(), _wide_table()) is False

    def test_wide_table_survives_various_chunk_sizes(self) -> None:
        """입도가 달라져도 판정이 뒤집히면 안 된다(케이던스 의존성 회귀 방지)."""
        table = _wide_table()
        for chunk in (7, 20, 50, 137, 400):
            assert feed_streaming(DegenerationMonitor(), table, chunk) is False, (
                f"chunk={chunk} 에서 오탐"
            )

    def test_narrow_table_still_fine(self) -> None:
        """좁은 표는 원래도 통과했다 — 무회귀 확인."""
        narrow = "| A | B | C |\n|---|---|---|\n" + "\n".join(
            f"| a{i} | b{i} | c{i} |" for i in range(40)
        )
        assert feed_streaming(DegenerationMonitor(), narrow) is False


class TestTruePositivesPreserved:
    """진짜 붕괴는 계속 잡아야 한다 — 오탐을 고치다 탐지를 죽이면 안 된다."""

    def test_pure_dash_flood_still_detected(self) -> None:
        """B 신호가 유일하게 잡는 형태다. 여기가 죽으면 수정이 실패한 것이다."""
        assert feed_streaming(DegenerationMonitor(), "설명입니다.\n" + "-" * 1500) is True

    def test_repeated_identical_line_still_detected(self) -> None:
        """A 신호 — 같은 긴 줄 반복."""
        line = "| 상천배합 | JB10001 | 220kg | 자동라인 1호 | 정상 |"
        assert feed_streaming(DegenerationMonitor(), (line + "\n") * 40) is True

    def test_emoji_flood_still_detected(self) -> None:
        """C 신호 — 이모지 폭주."""
        assert feed_streaming(DegenerationMonitor(), "설명. " + "🎉🔥✨" * 400) is True

    def test_global_phrase_repeat_still_detected(self) -> None:
        """D 신호 — 전체에 분산된 같은 구절 반복."""
        text = "".join(
            f"항목 {i} 를 설명한다. 생산지시서를 자동으로 계산하는 시스템이다. "
            for i in range(30)
        )
        assert feed_streaming(DegenerationMonitor(), text) is True
