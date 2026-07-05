"""
메모리 감쇠(decay) 및 통합(consolidate) — 시간이 지난 장기 메모리를 자동 정리.

이 파일이 하는 일 (한 줄 요약):
  "오래되고 안 쓰는 기억은 서서히 잊고, 같은 주제의 중복 기억은 하나로 합친다."
  사람의 기억이 시간이 지나면 희미해지는 것을 흉내 낸 모듈이다.

핵심 개념 — 지수 감쇠(exponential decay):
  오래된 메모리일수록 "유효 중요도(effective importance)"가 자동으로 낮아진다.
      유효 중요도 = importance * 2^(-days_since_access / half_life)
  즉 마지막 접근 이후 "반감기(half_life)"만큼 시간이 지날 때마다 중요도가 절반이 된다.

각 메모리 타입마다 반감기(half-life)가 다르다 (실제 값은 types.DECAY_HALF_LIFE):
  - EPISODIC: 7일 (대화 같은 일시적 기억 — 빠르게 잊혀짐)
  - SEMANTIC: 90일 (지식/사실 — 오래 유지됨)
  - PROCEDURAL: 30일 (절차/도구 사용 패턴)
  - USER_PROFILE: 365일 (사용자 선호/스타일 — 거의 잊혀지지 않음)
  - FEEDBACK: 14일 (피드백 기록)

중요한 설계 원칙 — importance는 "불변 base(원본 중요도)"다:
  - importance 필드는 저장된 원본 중요도를 그대로 유지한다(감쇠값을 되쓰지 않는다).
  - "유효 중요도"는 항상 calculate_decay(entry)로 읽는 시점에 계산한다.
  - 왜? 감쇠값을 importance에 되쓰면서 last_accessed는 그대로 두면, 다음
    사이클이 "이미 감쇠된 값"을 base로 "같은 경과일"로 또 감쇠시켜 지수 감쇠가
    복리(중복) 적용된다. 그 결과 EPISODIC(반감기 7일) 기억이 며칠 만에 임계치
    밑으로 떨어져 부당하게 삭제된다. base를 불변으로 두면 이 복리 버그가 없다.
  - 요약: "얼마나 감쇠했나"는 저장하지 않고, 필요할 때마다 매번 새로 계산한다.

감쇠 사이클(run_decay_cycle):
  - 유효 중요도가 임계치(0.05) 이하인 메모리를 "삭제"만 한다 (importance는 안 건드림)
  - 자주 접근된 메모리는 감쇠가 느려진다 (access boost)
  - 멱등(idempotent): 같은 상태에서 N번 반복 실행해도 결과가 1번 실행과 동일하다.

통합(consolidate):
  - 동일 키(key)를 가진 메모리를 하나로 합친다
  - 가장 최신이고 중요도가 높은 메모리를 유지하고, 나머지는 삭제(태그는 병합)

이 모듈의 위치와 의존 관계 (온보딩 참고):
  - 제공 클래스: MemoryDecayManager (아래 정의). core.memory 패키지로 노출된다.
  - 의존: core.memory.types 의 DECAY_HALF_LIFE / MemoryEntry / MemoryType.
  - 협력 대상: long_term(LongTermMemory) — PostgreSQL + pgvector 기반 장기 저장소.
    이 모듈은 long_term.get_all / delete / update 만 호출하고, DB 세부는 알지 못한다.
  - 호출 시점: 주기적인 유지보수(백그라운드 정리) 흐름에서 run_decay_cycle /
    consolidate 가 호출되는 것을 전제로 설계됐다.

작성자: 이현수 / 작성일: 2026-07-05
"""

from __future__ import annotations

import logging
import math
from datetime import UTC, datetime
from typing import Any

from core.memory.types import DECAY_HALF_LIFE, MemoryEntry, MemoryType

logger = logging.getLogger("nexus.memory.decay")

# 삭제 임계치 — 유효 중요도가 이 값 이하로 떨어진 메모리는 감쇠 사이클에서 삭제된다.
# 0.05는 "원본 중요도 대비 5% 미만으로 희미해진 기억"이라는 의미로 보면 된다.
DECAY_THRESHOLD: float = 0.05

# 접근 횟수에 따른 감쇠 완화 계수(access boost).
# 자주 꺼내 쓰는 기억일수록(access_count가 클수록) 더 오래 유지되도록 반감기를 늘린다.
# 실제(조정된) 반감기 = half_life * (1 + access_count * ACCESS_BOOST_FACTOR)
# 예: access_count=10이면 반감기가 (1 + 10*0.1) = 2배로 늘어나 절반 속도로 잊혀진다.
ACCESS_BOOST_FACTOR: float = 0.1


class MemoryDecayManager:
    """
    시간 경과에 따른 메모리 감쇠 + 주기적 정리를 수행하는 관리자 클래스.

    이 클래스는 상태(state)를 갖지 않는다 — 필드가 없고 메서드만 있다.
    필요한 데이터(장기 저장소 long_term)는 매 호출 시 인자로 주입받는다.
    그래서 인스턴스 하나를 만들어 재사용해도 되고, 매번 새로 만들어도 안전하다.

    제공 메서드:
      - calculate_decay(entry): 특정 메모리의 "현재 유효 중요도"를 읽기 전용으로 계산.
      - run_decay_cycle(long_term): 전체 메모리를 훑어 임계치 이하만 삭제(삭제 전용).
      - consolidate(long_term): 같은 key의 중복 메모리를 하나로 병합.

    지수 감쇠 모델(핵심 수식):
      effective_importance = importance * 2^(-days / adjusted_half_life)
      adjusted_half_life = half_life * (1 + access_count * 0.1)
    """

    def calculate_decay(self, entry: MemoryEntry) -> float:
        """
        하나의 메모리에 대해 "지금 이 순간의 유효 중요도"를 계산한다.

        이 메서드는 순수 계산 함수다 — entry를 읽기만 하고 절대 바꾸지 않는다.
        따라서 몇 번을 호출하든 저장된 상태(importance, last_accessed 등)는 그대로다.

        지수 감쇠 공식:
          effective = importance * 2^(-days_since_access / adjusted_half_life)

        접근 횟수(access_count)가 높을수록 감쇠가 느려진다:
          adjusted_half_life = half_life * (1 + access_count * 0.1)

        Args:
            entry: 감쇠를 계산할 MemoryEntry (원본 importance, last_accessed,
                   access_count, memory_type 필드를 참조한다)

        Returns:
            현재 유효 중요도 (0.0 ~ 1.0 범위로 clamp 후 소수 넷째 자리 반올림)
        """
        # 현재 시각(UTC 기준)을 기준점으로 잡는다.
        now = datetime.now(UTC)
        # last_accessed(마지막 접근 시각)가 timezone 정보를 가진(aware) 값인지 확인한다.
        last_accessed = entry.last_accessed
        if last_accessed.tzinfo is None:
            # tzinfo가 없는 naive datetime이면 UTC로 간주한다.
            # (naive와 aware를 그대로 빼면 TypeError가 나므로 먼저 맞춰준다.)
            last_accessed = last_accessed.replace(tzinfo=UTC)

        # 마지막 접근 이후 경과 시간을 "일(day)" 단위로 환산한다. (86400초 = 하루)
        days_since_access = (now - last_accessed).total_seconds() / 86400.0

        # 경과일이 0 이하면(방금 접근했거나 시계 오차 등) 감쇠가 없으므로 원본 값 그대로 반환.
        if days_since_access <= 0:
            return entry.importance

        # 이 메모리 타입의 반감기를 조회한다 (표에 없으면 기본값 30일).
        memory_type = entry.memory_type
        # memory_type이 문자열로 저장돼 있을 수 있어 Enum으로 정규화한다.
        if isinstance(memory_type, str):
            try:
                memory_type = MemoryType(memory_type)
            except ValueError:
                # 알 수 없는 값이면 가장 빨리 잊는 EPISODIC로 안전하게 폴백한다.
                memory_type = MemoryType.EPISODIC
        half_life = DECAY_HALF_LIFE.get(memory_type, 30.0)

        # 접근 횟수를 반영해 반감기를 늘린다(access boost).
        # 자주 접근되는 메모리는 반감기가 길어져 더 천천히 잊혀진다.
        adjusted_half_life = half_life * (1.0 + entry.access_count * ACCESS_BOOST_FACTOR)

        # 지수 감쇠 계산: 감쇠 계수 = 2^(-경과일 / 조정된 반감기).
        # 경과일이 반감기와 같으면 계수는 정확히 0.5(중요도 절반)가 된다.
        decay_factor = math.pow(2.0, -days_since_access / adjusted_half_life)
        effective_importance = entry.importance * decay_factor

        # 부동소수 오차를 감안해 [0.0, 1.0] 범위로 자르고, 넷째 자리까지 반올림해 반환.
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
        # 반환 통계를 0으로 초기화한다.
        # updated 키는 과거 API와의 하위 호환을 위해 남기지만, 삭제 전용 정책상 항상 0이다.
        stats = {"total_checked": 0, "deleted": 0, "updated": 0}

        # 장기 저장소에서 메모리를 한 번에 최대 1000개까지 가져온다.
        all_entries = await long_term.get_all(limit=1000)
        stats["total_checked"] = len(all_entries)

        # 가져온 메모리를 하나씩 검사한다.
        for entry in all_entries:
            # 읽기 전용 계산 — entry(특히 importance/last_accessed)를 변경하지 않는다.
            # 이 순수성 덕분에 사이클이 멱등(반복 실행해도 결과 동일)해진다.
            effective_importance = self.calculate_decay(entry)

            # 유효 중요도가 임계치 밑이면 삭제한다. (그 외에는 아무 것도 하지 않는다.)
            if effective_importance < DECAY_THRESHOLD:
                # long_term.delete는 실제 삭제됐으면 True를 돌려준다.
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
        같은 주제(key)의 중복 메모리들을 하나로 합쳐 저장소를 정돈한다.

        동작 개요:
          1. 전체 메모리를 key별로 묶는다.
          2. 같은 key가 2개 이상인 그룹만 대상으로 삼는다.
          3. 각 그룹에서 "가장 중요하고 가장 최신"인 하나(winner)만 남긴다.
          4. 나머지(losers)는 삭제하되, 그들이 갖고 있던 태그는 winner로 병합한다.
             (정보를 완전히 잃지 않도록 태그만은 살려두는 것이 포인트다.)

        Args:
            long_term: LongTermMemory 인스턴스

        Returns:
            통합 결과 통계:
              - groups_found: 통합 대상이 된(2개 이상 중복) key 그룹 수
              - entries_merged: 병합되어 삭제된 메모리 수
        """
        stats = {"groups_found": 0, "entries_merged": 0}

        # 장기 저장소에서 메모리를 최대 1000개 가져온다.
        all_entries = await long_term.get_all(limit=1000)

        # key가 비어있지 않은 메모리들을 key별 리스트로 그룹화한다.
        # (key가 없는 메모리는 "중복"을 판단할 기준이 없으므로 건너뛴다.)
        key_groups: dict[str, list[MemoryEntry]] = {}
        for entry in all_entries:
            if entry.key:
                key_groups.setdefault(entry.key, []).append(entry)

        # 각 그룹을 순회하며 통합한다.
        for key, entries in key_groups.items():
            # 항목이 1개뿐이면 중복이 아니므로 통합할 것이 없다.
            if len(entries) < 2:
                continue

            stats["groups_found"] += 1

            # 그룹 안에서 대표(winner)를 뽑기 위해 정렬한다.
            # 정렬 기준: importance 내림차순 → (동률이면) created_at 내림차순.
            # reverse=True라서 가장 중요하고 가장 최신인 항목이 맨 앞으로 온다.
            entries.sort(
                key=lambda e: (e.importance, e.created_at),
                reverse=True,
            )
            winner = entries[0]  # 남길 대표 메모리
            losers = entries[1:]  # 삭제 대상들

            # 삭제될 메모리들의 태그를 모아 winner의 태그에 합친다(집합으로 중복 제거).
            merged_tags = set(winner.tags)
            for loser in losers:
                merged_tags.update(loser.tags)

            # 태그가 실제로 늘어났을 때만 저장소에 업데이트한다(불필요한 쓰기 방지).
            if merged_tags != set(winner.tags):
                await long_term.update(winner.id, tags=sorted(merged_tags))

            # 대표를 제외한 나머지 메모리들을 삭제한다.
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
