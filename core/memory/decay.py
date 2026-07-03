"""
메모리 감쇠 및 통합 — 시간에 따른 메모리 정리.

지수 감쇠(exponential decay)로 오래된 메모리의 유효 중요도가 자동 감소한다:
  유효 중요도 = importance * 2^(-days_since_access / half_life)

각 메모리 타입마다 반감기(half-life)가 다르다:
  - EPISODIC: 7일 (빠르게 잊혀짐)
  - SEMANTIC: 90일 (오래 유지됨)
  - PROCEDURAL: 30일
  - USER_PROFILE: 365일 (거의 잊혀지지 않음)
  - FEEDBACK: 14일

중요한 설계 원칙 — importance는 "불변 base(원본 중요도)"다:
  - importance 필드는 저장된 원본 중요도를 그대로 유지한다(감쇠값을 되쓰지 않는다).
  - "유효 중요도"는 항상 calculate_decay(entry)로 읽는 시점에 계산한다.
  - 왜? 감쇠값을 importance에 되쓰면서 last_accessed는 그대로 두면, 다음
    사이클이 "이미 감쇠된 값"을 base로 "같은 경과일"로 또 감쇠시켜 지수 감쇠가
    복리(중복) 적용된다. 그 결과 EPISODIC(반감기 7일) 기억이 며칠 만에 임계치
    밑으로 떨어져 부당하게 삭제된다. base를 불변으로 두면 이 복리 버그가 없다.

감쇠 사이클(run_decay_cycle):
  - 유효 중요도가 임계치(0.05) 이하인 메모리를 "삭제"만 한다 (importance는 안 건드림)
  - 자주 접근된 메모리는 감쇠가 느려진다 (access boost)
  - 멱등(idempotent): 같은 상태에서 N번 반복 실행해도 결과가 1번 실행과 동일하다.

통합(consolidate):
  - 동일 키(key)를 가진 메모리를 하나로 합친다
  - 가장 최신이고 중요도가 높은 메모리를 유지
"""

from __future__ import annotations

import logging
import math
from datetime import UTC, datetime
from typing import Any

from core.memory.types import DECAY_HALF_LIFE, MemoryEntry, MemoryType

logger = logging.getLogger("nexus.memory.decay")

# 유효 중요도가 이 값 이하이면 삭제 대상
DECAY_THRESHOLD: float = 0.05

# 접근 횟수에 따른 감쇠 완화 계수
# access_count가 높으면 감쇠가 느려진다
# 실제 반감기 = half_life * (1 + access_count * ACCESS_BOOST_FACTOR)
ACCESS_BOOST_FACTOR: float = 0.1


class MemoryDecayManager:
    """
    시간 경과에 따른 메모리 감쇠 + 주기적 정리를 수행한다.

    지수 감쇠 모델:
      effective_importance = importance * 2^(-days / adjusted_half_life)
      adjusted_half_life = half_life * (1 + access_count * 0.1)
    """

    def calculate_decay(self, entry: MemoryEntry) -> float:
        """
        현재 유효 중요도를 계산한다.

        지수 감쇠 공식:
          effective = importance * 2^(-days_since_access / adjusted_half_life)

        접근 횟수(access_count)가 높을수록 감쇠가 느려진다:
          adjusted_half_life = half_life * (1 + access_count * 0.1)

        Args:
            entry: 감쇠를 계산할 MemoryEntry

        Returns:
            현재 유효 중요도 (0.0 ~ 1.0)
        """
        # 마지막 접근 이후 경과 일수
        now = datetime.now(UTC)
        # last_accessed가 timezone-aware인지 확인
        last_accessed = entry.last_accessed
        if last_accessed.tzinfo is None:
            # naive datetime이면 UTC로 간주
            last_accessed = last_accessed.replace(tzinfo=UTC)

        days_since_access = (now - last_accessed).total_seconds() / 86400.0

        # 경과일이 0 이하면 감쇠 없음
        if days_since_access <= 0:
            return entry.importance

        # 타입별 반감기 조회 (기본값 30일)
        memory_type = entry.memory_type
        if isinstance(memory_type, str):
            try:
                memory_type = MemoryType(memory_type)
            except ValueError:
                memory_type = MemoryType.EPISODIC
        half_life = DECAY_HALF_LIFE.get(memory_type, 30.0)

        # 접근 횟수에 따른 반감기 조정
        # 자주 접근되는 메모리는 더 오래 유지된다
        adjusted_half_life = half_life * (1.0 + entry.access_count * ACCESS_BOOST_FACTOR)

        # 지수 감쇠: importance * 2^(-days / half_life)
        decay_factor = math.pow(2.0, -days_since_access / adjusted_half_life)
        effective_importance = entry.importance * decay_factor

        return round(max(0.0, min(1.0, effective_importance)), 4)

    async def run_decay_cycle(self, long_term: Any) -> dict:
        """
        감쇠 사이클을 실행한다 — "삭제 전용(deletion-only)".

        모든 장기 메모리를 순회하며:
          1. calculate_decay로 유효 중요도를 계산(읽기 전용, 상태 변경 없음)
          2. 임계치(0.05) 이하인 메모리만 "삭제"한다

        왜 importance를 갱신하지 않는가 (복리 감쇠 버그 수정):
          과거 구현은 유효 중요도를 importance 필드에 되썼지만 last_accessed는
          갱신하지 않았다. 그러면 다음 사이클이 "이미 감쇠된 importance"를 base로
          "여전히 옛 last_accessed 기준 경과일"로 또 감쇠시켜, 지수 감쇠가 복리로
          중복 적용됐다(며칠 만에 임계치 붕괴 → 조기 삭제).
          importance를 불변 base로 두고 유효 중요도는 읽는 시점에 calculate_decay로
          계산하면, 이 사이클은 순수 함수처럼 동작한다.

        멱등성(idempotent) 불변식:
          같은 상태에서 이 사이클을 N번 반복 실행해도 결과가 1번 실행과 동일하다.
          (base importance/last_accessed를 건드리지 않으므로 반복 실행이 추가 감쇠를
           만들지 않는다.)

        Args:
            long_term: LongTermMemory 인스턴스

        Returns:
            실행 결과 통계:
              - total_checked: 검사한 메모리 수
              - deleted: 삭제된 메모리 수
              - updated: (deprecated) 항상 0 — 이제 importance를 갱신하지 않는다.
                         호출측 호환성을 위해 키는 유지한다.
        """
        # updated 키는 하위 호환을 위해 남기지만, 삭제 전용 정책상 항상 0이다.
        stats = {"total_checked": 0, "deleted": 0, "updated": 0}

        # 전체 메모리 조회 (최대 1000개씩 처리)
        all_entries = await long_term.get_all(limit=1000)
        stats["total_checked"] = len(all_entries)

        for entry in all_entries:
            # 읽기 전용 계산 — entry(특히 importance/last_accessed)를 변경하지 않는다.
            effective_importance = self.calculate_decay(entry)

            # 임계치 이하면 삭제 (그 외에는 아무 것도 하지 않는다)
            if effective_importance < DECAY_THRESHOLD:
                deleted = await long_term.delete(entry.id)
                if deleted:
                    stats["deleted"] += 1
                    logger.debug(
                        "감쇠 삭제: id=%s, effective=%.4f < %.2f",
                        entry.id,
                        effective_importance,
                        DECAY_THRESHOLD,
                    )

        logger.info(
            "감쇠 사이클 완료(삭제 전용): checked=%d, deleted=%d",
            stats["total_checked"],
            stats["deleted"],
        )
        return stats

    async def consolidate(self, long_term: Any) -> dict:
        """
        유사한 메모리를 통합한다.

        같은 key를 가진 메모리들을 그룹화하고,
        각 그룹에서 가장 중요하고 최신인 메모리만 유지한다.
        나머지는 삭제하되, 태그는 병합한다.

        Args:
            long_term: LongTermMemory 인스턴스

        Returns:
            통합 결과 통계:
              - groups_found: 발견된 중복 그룹 수
              - entries_merged: 병합된 (삭제된) 메모리 수
        """
        stats = {"groups_found": 0, "entries_merged": 0}

        all_entries = await long_term.get_all(limit=1000)

        # key가 비어있지 않은 메모리들을 key별로 그룹화
        key_groups: dict[str, list[MemoryEntry]] = {}
        for entry in all_entries:
            if entry.key:
                key_groups.setdefault(entry.key, []).append(entry)

        # 2개 이상인 그룹만 통합
        for key, entries in key_groups.items():
            if len(entries) < 2:
                continue

            stats["groups_found"] += 1

            # 가장 중요하고 최신인 메모리를 대표로 선정
            # 정렬 기준: importance 내림차순 → created_at 내림차순
            entries.sort(
                key=lambda e: (e.importance, e.created_at),
                reverse=True,
            )
            winner = entries[0]
            losers = entries[1:]

            # 패배자들의 태그를 승자에게 병합
            merged_tags = set(winner.tags)
            for loser in losers:
                merged_tags.update(loser.tags)

            # 승자 태그 업데이트 (변경이 있을 때만)
            if merged_tags != set(winner.tags):
                await long_term.update(winner.id, tags=sorted(merged_tags))

            # 패배자들 삭제
            for loser in losers:
                deleted = await long_term.delete(loser.id)
                if deleted:
                    stats["entries_merged"] += 1
                    logger.debug(
                        "메모리 통합: key='%s', 삭제=%s (보존=%s)",
                        key,
                        loser.id,
                        winner.id,
                    )

        logger.info(
            "메모리 통합 완료: groups=%d, merged=%d",
            stats["groups_found"],
            stats["entries_merged"],
        )
        return stats
