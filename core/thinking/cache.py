"""
사고 캐시 — 동일 문제 재사고 방지를 위한 인메모리 LRU 캐시.

동일한 입력에 대해 반복적으로 사고 엔진을 실행하는 것을 방지한다.
해시 기반 키로 메시지를 식별하고, TTL 만료 시 자동 제거한다.

향후 Redis 연동은 Phase 5.0b에서 구현한다.
"""

from __future__ import annotations

import hashlib
import logging
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # 순환 import 방지: ThinkingResult는 타입 힌트에서만 사용
    from core.thinking.orchestrator import ThinkingResult

logger = logging.getLogger("nexus.thinking.cache")


class ThinkingCache:
    """
    사고 결과를 캐싱하는 인메모리 LRU 캐시.

    동작 원리:
      1. 메시지의 SHA-256 해시를 키로 사용
      2. 캐시 히트 시 이전 사고 결과를 즉시 반환 (LLM 호출 절약)
      3. TTL 만료된 항목은 조회 시 자동 제거
      4. max_size 초과 시 가장 오래된 항목부터 제거 (LRU)

    제약 사항:
      - 인메모리 전용: 프로세스 재시작 시 캐시 소멸
      - 동일 텍스트만 매칭: 의미적 유사도는 고려하지 않음
      - 선택적 Redis 연동은 Phase 5.0b에서 구현 예정
    """

    def __init__(
        self,
        max_size: int = 100,
        ttl_seconds: float = 3600.0,
    ) -> None:
        """
        캐시를 초기화한다.

        Args:
            max_size: 최대 캐시 항목 수 (기본 100)
            ttl_seconds: 항목 만료 시간(초) (기본 1시간)
        """
        self._max_size = max_size
        self._ttl_seconds = ttl_seconds
        # key → (ThinkingResult, 저장 시각 timestamp)
        self._cache: dict[str, tuple[Any, float]] = {}
        # 접근 순서 추적 (LRU 구현용)
        self._access_order: list[str] = []
        # 통계
        self._hits: int = 0
        self._misses: int = 0

    def get(self, key: str) -> ThinkingResult | None:
        """
        캐시에서 사고 결과를 조회한다.

        Args:
            key: 메시지 해시 키 (_make_key로 생성)

        Returns:
            캐시된 ThinkingResult 또는 None (미스/만료)
        """
        if key not in self._cache:
            self._misses += 1
            return None

        result, stored_at = self._cache[key]

        # TTL 만료 확인
        if time.monotonic() - stored_at > self._ttl_seconds:
            # 만료된 항목 제거
            del self._cache[key]
            if key in self._access_order:
                self._access_order.remove(key)
            self._misses += 1
            logger.debug(f"캐시 TTL 만료: key={key[:16]}...")
            return None

        # LRU: 접근 순서 갱신 (가장 최근 접근을 맨 뒤로)
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)

        self._hits += 1
        logger.debug(f"캐시 히트: key={key[:16]}...")
        return result

    def put(self, key: str, result: ThinkingResult) -> None:
        """
        사고 결과를 캐시에 저장한다.

        Args:
            key: 메시지 해시 키
            result: 저장할 ThinkingResult

        max_size 초과 시 가장 오래전에 접근된 항목(LRU)을 제거한다.
        """
        # 이미 존재하면 갱신
        if key in self._cache:
            if key in self._access_order:
                self._access_order.remove(key)
        else:
            # 용량 초과 시 LRU 제거
            self._evict_if_needed()

        self._cache[key] = (result, time.monotonic())
        self._access_order.append(key)
        logger.debug(f"캐시 저장: key={key[:16]}..., 현재 크기={len(self._cache)}")

    def _evict_if_needed(self) -> None:
        """
        캐시 용량이 max_size에 도달하면 LRU 항목을 제거한다.
        먼저 만료된 항목을 정리하고, 그래도 부족하면 가장 오래된 항목을 제거한다.
        """
        # 먼저 만료된 항목 일괄 정리
        self._purge_expired()

        # 여전히 용량 초과면 LRU 제거
        while len(self._cache) >= self._max_size and self._access_order:
            oldest_key = self._access_order.pop(0)
            if oldest_key in self._cache:
                del self._cache[oldest_key]
                logger.debug(f"LRU 제거: key={oldest_key[:16]}...")

    def _purge_expired(self) -> None:
        """만료된 캐시 항목을 일괄 제거한다."""
        now = time.monotonic()
        expired_keys = [
            k for k, (_, stored_at) in self._cache.items() if now - stored_at > self._ttl_seconds
        ]
        for key in expired_keys:
            del self._cache[key]
            if key in self._access_order:
                self._access_order.remove(key)

    @staticmethod
    def make_key(message: str) -> str:
        """
        메시지의 SHA-256 해시를 캐시 키로 생성한다.

        동일한 메시지 텍스트는 항상 같은 키를 반환한다.
        공백 정규화(strip)를 적용하여 의미없는 차이를 무시한다.
        """
        normalized = message.strip()
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def clear(self) -> None:
        """캐시를 완전히 비운다."""
        self._cache.clear()
        self._access_order.clear()
        logger.debug("캐시 전체 초기화")

    @property
    def size(self) -> int:
        """현재 캐시 항목 수를 반환한다."""
        return len(self._cache)

    @property
    def stats(self) -> dict[str, int]:
        """캐시 통계를 반환한다."""
        total = self._hits + self._misses
        return {
            "size": len(self._cache),
            "max_size": self._max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate_percent": round((self._hits / total * 100) if total > 0 else 0.0, 1),
        }
