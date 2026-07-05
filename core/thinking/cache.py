"""
사고(Thinking) 캐시 — 동일 문제를 두 번 사고하지 않도록 막는 인메모리 LRU 캐시.

[이 파일이 하는 일 — 한눈에 보기]
Nexus의 사고 엔진(core/thinking/orchestrator)은 하나의 질문을 놓고 여러 단계로
숙고(reasoning)를 돌리는데, 이 과정은 LLM 호출을 동반해 느리고 비용이 크다.
그래서 "똑같은 입력이 또 들어오면, 지난번 사고 결과를 그대로 돌려주자"는 것이
이 캐시의 존재 이유다. 결과적으로 중복 LLM 호출을 줄여 응답 속도와 비용을 아낀다.

[핵심 동작 요약]
  - 키(key): 입력 메시지를 SHA-256으로 해시한 값. 같은 텍스트 → 항상 같은 키.
  - 히트(hit): 키가 있고 아직 만료되지 않았으면 저장된 결과를 즉시 반환.
  - TTL: 항목마다 저장 시각을 기록해 두고, ttl_seconds가 지나면 만료로 간주해 버린다.
  - LRU: 용량(max_size)이 꽉 차면 "가장 오래 안 쓰인 항목"부터 밀어낸다.

[주요 구성]
  - ThinkingCache 클래스: get/put/clear 및 내부 정리(_evict_if_needed, _purge_expired),
    키 생성(make_key), 통계(stats)/크기(size) 프로퍼티를 제공한다.

[의존/제약]
  - ThinkingResult 타입은 core.thinking.orchestrator에 있으나, 순환 import를 피하려고
    TYPE_CHECKING 블록에서 타입 힌트 용도로만 가져온다(런타임 import 아님).
  - 인메모리 전용: 프로세스가 재시작되면 캐시는 사라진다.
  - 의미적 유사도는 보지 않는다. 글자 하나만 달라도 다른 키 → 캐시 미스.
  - 선택적 Redis 연동(프로세스 간 공유/영속화)은 Phase 5.0b에서 구현 예정.

작성자: 이현수 / 작성일: 2026-07-05
"""

# from __future__ import: 타입 힌트를 문자열로 지연 평가하게 해준다.
# 덕분에 아직 import되지 않은 타입(ThinkingResult 등)도 힌트에 자유롭게 쓸 수 있다.
from __future__ import annotations

import hashlib  # 메시지 → SHA-256 해시 키 생성에 사용
import logging  # 캐시 히트/미스/제거 등의 디버그 로그
import time  # TTL 계산용 단조 시계(time.monotonic) 사용
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # 순환 import 방지: ThinkingResult는 타입 힌트에서만 사용
    # (런타임에는 실제로 import하지 않으므로 orchestrator ↔ cache 간 순환이 생기지 않는다)
    from core.thinking.orchestrator import ThinkingResult

# 모듈 전용 로거. 로그 계층 이름을 "nexus.thinking.cache"로 고정해
# 다른 모듈 로그와 구분하고, 설정에서 이 계층만 따로 레벨 조정할 수 있게 한다.
logger = logging.getLogger("nexus.thinking.cache")


class ThinkingCache:
    """
    사고 결과를 담아 두는 인메모리 LRU 캐시.

    왜 필요한가:
      사고 엔진 한 번 실행은 LLM 호출을 여러 번 유발할 수 있어 느리고 비싸다.
      같은 질문이 반복될 때 지난 결과를 재사용하면 그 비용을 통째로 아낄 수 있다.

    동작 원리:
      1. 입력 메시지의 SHA-256 해시를 키로 사용한다(make_key 참조).
      2. 캐시 히트 시 이전 사고 결과를 즉시 반환한다(LLM 호출 자체를 건너뜀).
      3. TTL이 지난 항목은 조회 시점에 발견되면 그 자리에서 제거한다.
      4. max_size를 넘기면 가장 오래 안 쓰인 항목(LRU)부터 밀어낸다.

    내부 자료구조:
      - _cache: key → (ThinkingResult, 저장시각) 매핑. 실제 데이터 저장소.
      - _access_order: 접근 순서를 앞→뒤로 기록하는 리스트. 맨 앞이 "가장 오래됨".
        get/put이 일어날 때마다 해당 키를 맨 뒤로 옮겨 최신으로 표시한다.

    제약 사항:
      - 인메모리 전용: 프로세스 재시작 시 캐시 소멸.
      - 동일 텍스트만 매칭: 의미적 유사도는 고려하지 않음.
      - 선택적 Redis 연동은 Phase 5.0b에서 구현 예정.
    """

    def __init__(
        self,
        max_size: int = 100,
        ttl_seconds: float = 3600.0,
    ) -> None:
        """
        캐시를 초기화한다.

        Args:
            max_size: 보관할 최대 항목 수. 이 값에 도달하면 LRU 제거가 시작된다(기본 100).
            ttl_seconds: 항목 유효 시간(초). 저장 후 이 시간이 지나면 만료(기본 3600초=1시간).
        """
        # 용량/수명 정책 값. 생성 이후 바뀌지 않는 설정으로 취급한다.
        self._max_size = max_size
        self._ttl_seconds = ttl_seconds
        # key → (ThinkingResult, 저장 시각 timestamp)
        # 저장 시각은 time.monotonic() 기준값이며 TTL 만료 판정에 쓰인다.
        self._cache: dict[str, tuple[Any, float]] = {}
        # 접근 순서 추적 (LRU 구현용). 리스트 앞쪽일수록 오래된 키.
        self._access_order: list[str] = []
        # 통계용 카운터: 히트/미스 누적 횟수(stats 프로퍼티에서 히트율 계산에 사용)
        self._hits: int = 0
        self._misses: int = 0

    def get(self, key: str) -> ThinkingResult | None:
        """
        캐시에서 사고 결과를 조회한다.

        흐름:
          1) 키가 아예 없으면 미스 처리 후 None.
          2) 키는 있지만 TTL이 지났으면 즉시 제거하고 미스 처리 후 None.
          3) 유효하면 LRU 순서를 갱신(맨 뒤로)하고 히트 처리 후 결과 반환.

        Args:
            key: 메시지 해시 키 (make_key로 생성한 값).

        Returns:
            유효한 캐시가 있으면 ThinkingResult, 없거나 만료면 None.
        """
        # (1) 애초에 저장된 적 없는 키 → 미스
        if key not in self._cache:
            self._misses += 1
            return None

        # 저장된 (결과, 저장시각) 튜플을 꺼낸다.
        result, stored_at = self._cache[key]

        # (2) TTL 만료 확인: 저장 후 경과 시간이 수명을 넘겼는지 검사
        if time.monotonic() - stored_at > self._ttl_seconds:
            # 만료된 항목 제거 — 데이터와 접근 순서 양쪽에서 함께 지운다.
            del self._cache[key]
            if key in self._access_order:
                self._access_order.remove(key)
            self._misses += 1
            # 키 전체는 길고 로그 노이즈가 크므로 앞 16글자만 남긴다.
            logger.debug(f"캐시 TTL 만료: key={key[:16]}...")
            return None

        # (3) LRU 갱신: 방금 접근한 키를 리스트 맨 뒤(=가장 최근)로 이동시킨다.
        # 기존 위치에서 빼고 뒤에 다시 붙이는 방식으로 "최근 사용" 표시를 한다.
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)

        self._hits += 1
        logger.debug(f"캐시 히트: key={key[:16]}...")
        return result

    def put(self, key: str, result: ThinkingResult) -> None:
        """
        사고 결과를 캐시에 저장한다.

        같은 키가 이미 있으면 값을 갱신하고 LRU 순서만 새로 매긴다.
        새 키인데 용량이 꽉 찼다면, 먼저 자리를 비운 뒤 저장한다.

        Args:
            key: 메시지 해시 키.
            result: 저장할 ThinkingResult.
        """
        # 이미 존재하는 키면 갱신: 기존 접근 순서 항목만 떼어낸다(아래에서 다시 뒤에 붙임).
        # 이 경우엔 항목 수가 늘지 않으므로 용량 정리를 할 필요가 없다.
        if key in self._cache:
            if key in self._access_order:
                self._access_order.remove(key)
        else:
            # 새 키를 넣기 전에 용량 초과 여부를 확인하고 필요하면 LRU 제거로 자리를 만든다.
            self._evict_if_needed()

        # 값과 "현재 시각"을 함께 저장한다. 이 시각이 이후 TTL 판정의 기준이 된다.
        self._cache[key] = (result, time.monotonic())
        # 방금 저장한 키를 가장 최근으로 표시(리스트 맨 뒤).
        self._access_order.append(key)
        logger.debug(f"캐시 저장: key={key[:16]}..., 현재 크기={len(self._cache)}")

    def _evict_if_needed(self) -> None:
        """
        새 항목을 넣기 전에 용량을 확보하는 내부 정리 루틴.

        순서가 중요하다:
          1) 먼저 만료된 항목을 싹 정리한다. 이것만으로 자리가 나면 LRU 제거는 불필요.
          2) 그래도 여전히 max_size 이상이면, 가장 오래된 키부터 하나씩 제거한다.
        """
        # (1) 만료된 항목 일괄 정리 — 죽은 항목부터 비우는 게 살아있는 항목 보호에 유리.
        self._purge_expired()

        # (2) 여전히 용량 초과면 LRU 제거.
        # _access_order의 맨 앞(pop(0))이 가장 오래된 키다. 자리가 날 때까지 반복한다.
        while len(self._cache) >= self._max_size and self._access_order:
            oldest_key = self._access_order.pop(0)
            # 접근 순서와 실제 저장소가 어긋나 있을 수 있으므로 존재 확인 후 삭제.
            if oldest_key in self._cache:
                del self._cache[oldest_key]
                logger.debug(f"LRU 제거: key={oldest_key[:16]}...")

    def _purge_expired(self) -> None:
        """
        현재 시각 기준으로 TTL이 지난 항목을 한꺼번에 골라내 제거한다.

        조회(get)는 그 키를 건드릴 때만 만료를 발견하므로, 아무도 안 찾는 만료 항목이
        메모리에 계속 남을 수 있다. 이 메서드는 그런 항목들을 일괄 청소하는 역할이다.
        """
        now = time.monotonic()
        # 반복 중 딕셔너리를 바로 수정하면 위험하므로, 먼저 만료 키 목록을 만들어 둔다.
        expired_keys = [
            k for k, (_, stored_at) in self._cache.items() if now - stored_at > self._ttl_seconds
        ]
        # 모아둔 만료 키들을 데이터/접근순서 양쪽에서 제거한다.
        for key in expired_keys:
            del self._cache[key]
            if key in self._access_order:
                self._access_order.remove(key)

    @staticmethod
    def make_key(message: str) -> str:
        """
        메시지 텍스트로부터 캐시 키(SHA-256 16진 문자열)를 만든다.

        같은 메시지는 언제나 같은 키를 반환한다(결정적). 앞뒤 공백만 다른 입력이
        서로 다른 키로 갈라지지 않도록 strip()으로 정규화한 뒤 해시한다.

        Args:
            message: 캐시 키로 삼을 원본 메시지 문자열.

        Returns:
            64자리 SHA-256 16진 다이제스트 문자열.
        """
        # 앞뒤 공백 제거로 사소한 차이를 흡수 → 캐시 적중률을 조금이라도 높인다.
        normalized = message.strip()
        # UTF-8로 인코딩해 바이트로 만든 뒤 해시. 한글 등 비ASCII도 안전하게 처리된다.
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def clear(self) -> None:
        """캐시를 완전히 비운다(데이터와 접근 순서 모두 초기화). 통계 카운터는 유지된다."""
        self._cache.clear()
        self._access_order.clear()
        logger.debug("캐시 전체 초기화")

    @property
    def size(self) -> int:
        """현재 캐시에 들어 있는 항목 수를 반환한다."""
        return len(self._cache)

    @property
    def stats(self) -> dict[str, int]:
        """
        캐시 운영 통계를 딕셔너리로 반환한다(모니터링/튜닝용).

        포함 항목: 현재 크기, 최대 크기, 누적 히트/미스, 히트율(%).
        히트율은 (히트 / (히트+미스)) × 100이며, 조회가 한 번도 없으면 0.0으로 처리한다.
        """
        total = self._hits + self._misses
        return {
            "size": len(self._cache),
            "max_size": self._max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate_percent": round((self._hits / total * 100) if total > 0 else 0.0, 1),
        }
