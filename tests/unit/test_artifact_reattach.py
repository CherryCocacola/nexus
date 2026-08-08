# 새로고침 복원 시 생성물이 "제 턴"에 붙는지 고정한다 (C2 회귀 방지).
"""
2026-08-08 사용자 눈검증에서 나온 결함.

  기대: 사과 요청 → 사과그림 → 문서 요청 → WORD
  실제: 사과 요청 → 문서 요청 → (사과그림 + WORD 가 같은 줄)

원인은 매칭 키였다. 되붙이기가 `turn` 을 봤는데 **turn 이 항상 1** 이다 — 웹은 요청마다
엔진을 새로 만들어 턴 카운터가 리셋된다. 그래서 매칭이 한 번도 성립하지 않고 전부
"실패 → 마지막 메시지에 몰아 붙이기" 경로로 떨어졌다.

이 테스트는 두 가지를 못 박는다.
  ① 시각 기준으로 각자 제 자리에 붙는다
  ② turn 이 전부 같아도(=실제 상황) 흩어지지 않는다  ← 이게 핵심 회귀 방지
"""

from __future__ import annotations

from datetime import UTC, datetime

from web.app import _index_for_created_at, _parse_ts


# ─────────────────────────────────────────────
# _parse_ts — 두 기록의 시각 표현을 한 종류로 맞춘다
# ─────────────────────────────────────────────
def test_parse_ts_accepts_iso_string_with_offset() -> None:
    """트랜스크립트는 ISO 문자열로 남긴다."""
    got = _parse_ts("2026-08-08T10:50:18.522200+00:00")
    assert got is not None
    assert got.tzinfo is not None
    assert got.year == 2026 and got.minute == 50


def test_parse_ts_accepts_datetime_from_db() -> None:
    """tb_artifacts 는 asyncpg 가 준 datetime 이다."""
    src = datetime(2026, 8, 8, 10, 50, 18, tzinfo=UTC)
    assert _parse_ts(src) == src


def test_parse_ts_treats_naive_as_utc() -> None:
    """tz 없는 값은 UTC 로 본다 — 두 기록 모두 UTC 로 남긴다."""
    got = _parse_ts(datetime(2026, 8, 8, 10, 50, 18))
    assert got is not None and got.tzinfo is not None


def test_parse_ts_returns_none_for_garbage() -> None:
    """망가진 값은 None — 비교에서 빼고 레거시 경로로 보낸다."""
    assert _parse_ts(None) is None
    assert _parse_ts("어제") is None


# ─────────────────────────────────────────────
# _index_for_created_at — 실제 관측 시각으로 검증
# ─────────────────────────────────────────────
def test_artifact_attaches_to_the_turn_that_produced_it() -> None:
    """실측 시각 그대로 — 사과는 1번 답변, 문서는 2번 답변에 붙어야 한다.

    transcript  assistant#0 10:50:18.522200
                assistant#1 10:50:20.677289
    artifacts   사과.png    10:50:18.523213   → #0
                document    10:50:20.677795   → #1
    """
    ts_list = [
        _parse_ts("2026-08-08T10:50:18.522200+00:00"),
        _parse_ts("2026-08-08T10:50:20.677289+00:00"),
    ]
    apple = _parse_ts("2026-08-08T10:50:18.523213+00:00")
    doc = _parse_ts("2026-08-08T10:50:20.677795+00:00")

    assert _index_for_created_at(ts_list, apple) == 0
    assert _index_for_created_at(ts_list, doc) == 1


def test_multiple_artifacts_in_one_turn_share_the_same_index() -> None:
    """한 턴에서 둘을 만들면 둘 다 그 턴에 붙는다."""
    ts_list = [_parse_ts("2026-08-08T10:50:18.000000+00:00")]
    a = _parse_ts("2026-08-08T10:50:18.100000+00:00")
    b = _parse_ts("2026-08-08T10:50:18.900000+00:00")
    assert _index_for_created_at(ts_list, a) == 0
    assert _index_for_created_at(ts_list, b) == 0


def test_artifact_before_any_message_falls_back() -> None:
    """첫 메시지보다 이른 생성물은 자리를 못 정한다 → 레거시 경로(None)."""
    ts_list = [_parse_ts("2026-08-08T10:50:18+00:00")]
    early = _parse_ts("2026-08-08T10:49:00+00:00")
    assert _index_for_created_at(ts_list, early) is None


def test_missing_created_at_falls_back() -> None:
    """fork 상속분은 created_at 이 없다 → 레거시 경로로 보내 링크를 잃지 않는다."""
    ts_list = [_parse_ts("2026-08-08T10:50:18+00:00")]
    assert _index_for_created_at(ts_list, None) is None


def test_messages_without_ts_are_skipped_not_crashed() -> None:
    """ts 가 비어 있는 메시지가 섞여도 죽지 않고 있는 것만 본다."""
    ts_list = [None, _parse_ts("2026-08-08T10:50:20+00:00"), None]
    got = _index_for_created_at(ts_list, _parse_ts("2026-08-08T10:50:21+00:00"))
    assert got == 1


def test_last_matching_message_wins_not_first() -> None:
    """여러 후보 중 **가장 늦은** 것을 고른다 — 생성물은 직전 답변에 속한다."""
    ts_list = [
        _parse_ts("2026-08-08T10:50:10+00:00"),
        _parse_ts("2026-08-08T10:50:20+00:00"),
        _parse_ts("2026-08-08T10:50:30+00:00"),
    ]
    got = _index_for_created_at(ts_list, _parse_ts("2026-08-08T10:50:25+00:00"))
    assert got == 1


def test_identical_turn_values_no_longer_collapse_artifacts() -> None:
    """★핵심 회귀 방지 — turn 이 전부 1 이어도 시각으로 갈린다.

    이 상황(turn 전부 동일)이 실제 운영 상태였고, 예전 turn 매칭은 여기서
    전부 마지막 메시지에 뭉쳤다. 시각 기준에서는 흩어지지 않는다.
    """
    ts_list = [
        _parse_ts("2026-08-08T10:50:18+00:00"),
        _parse_ts("2026-08-08T10:50:20+00:00"),
        _parse_ts("2026-08-08T10:50:22+00:00"),
    ]
    created = [
        _parse_ts("2026-08-08T10:50:18.5+00:00"),
        _parse_ts("2026-08-08T10:50:20.5+00:00"),
        _parse_ts("2026-08-08T10:50:22.5+00:00"),
    ]
    got = [_index_for_created_at(ts_list, c) for c in created]
    assert got == [0, 1, 2], f"생성물이 제 턴에 안 붙었다: {got}"
