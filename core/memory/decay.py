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

감쇠 사이클(run_decay_cycle):
  - 유효 중요도가 임계치(0.05) 이하인 메모리를 삭제
  - 자주 접근된 메모리는 감쇠가 느려진다 (access boost)

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
        감쇠 사이클을 실행한다.

        모든 장기 메모리를 순회하며:
          1. 유효 중요도를 계산
          2. 임계치(0.05) 이하인 메모리를 삭제
          3. 나머지 메모리의 importance를 유효 중요도로 갱신

        Args:
            long_term: LongTermMemory 인스턴스

        Returns:
            실행 결과 통계:
              - total_checked: 검사한 메모리 수
              - deleted: 삭제된 메모리 수
              - updated: 중요도가 갱신된 메모리 수
        """
        stats = {"total_checked": 0, "deleted": 0, "updated": 0}

        # 전체 메모리 조회 (최대 1000개씩 처리)
        all_entries = await long_term.get_all(limit=1000)
        stats["total_checked"] = len(all_entries)

        for entry in all_entries:
            effective_importance = self.calculate_decay(entry)

            # 임계치 이하면 삭제
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
                continue

            # 유효 중요도가 원래 값과 크게 다르면 갱신
            # (차이가 0.01 이상일 때만 DB 업데이트하여 I/O 최소화)
            if abs(effective_importance - entry.importance) >= 0.01:
                updated = await long_term.update(entry.id, importance=effective_importance)
                if updated:
                    stats["updated"] += 1

        logger.info(
            "감쇠 사이클 완료: checked=%d, deleted=%d, updated=%d",
            stats["total_checked"],
            stats["deleted"],
            stats["updated"],
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
