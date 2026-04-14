"""
단기 메모리 — Redis 기반 세션 컨텍스트 + 도구 결과 캐시.

세션 내에서 빠르게 접근해야 하는 데이터를 저장한다:
  - 대화 컨텍스트 (session:{session_id}:context)
  - 도구 결과 캐시 (tool_cache:{tool_name}:{input_hash})
  - 임시 키-값 데이터

Redis 클라이언트가 없으면 인메모리 딕셔너리로 폴백한다.
에어갭 환경에서 Redis는 LAN 내 서버(192.168.x.x)에 위치한다.

설계 결정:
  - TTL 기본값: 일반 키 1시간, 대화 컨텍스트 24시간, 도구 캐시 30분
  - 인메모리 폴백: Redis 없이도 개발/테스트 가능
  - JSON 직렬화: 대화 컨텍스트는 JSON으로 직렬화하여 저장
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

logger = logging.getLogger("nexus.memory.short_term")


class ShortTermMemory:
    """
    Redis 기반 단기 메모리.

    세션 컨텍스트, 도구 결과 캐시 등을 저장한다.
    Redis가 없으면 인메모리 딕셔너리로 폴백한다.
    """

    def __init__(self, redis_client: Any | None = None):
        """
        단기 메모리를 초기화한다.

        Args:
            redis_client: redis.asyncio.Redis 인스턴스 (None이면 인메모리 폴백)
        """
        # 인메모리 폴백 저장소: {key: {"value": str, "expires_at": float | None}}
        self._store: dict[str, dict[str, Any]] = {}
        self._redis = redis_client

        if self._redis is None:
            logger.info("Redis 클라이언트 없음 — 인메모리 폴백 모드로 동작")

    # ─── 기본 CRUD ───

    async def get(self, key: str) -> str | None:
        """
        키에 해당하는 값을 조회한다.
        만료된 키는 None을 반환하고 자동 삭제한다.

        Args:
            key: 조회할 키

        Returns:
            저장된 문자열 값 또는 None
        """
        if self._redis is not None:
            try:
                value = await self._redis.get(key)
                # Redis는 bytes를 반환하므로 문자열로 변환
                return value.decode("utf-8") if isinstance(value, bytes) else value
            except Exception as e:
                logger.warning("Redis get 실패 (key=%s): %s — 폴백 사용", key, e)

        # 인메모리 폴백
        entry = self._store.get(key)
        if entry is None:
            return None

        # TTL 만료 확인
        expires_at = entry.get("expires_at")
        if expires_at is not None and time.time() > expires_at:
            del self._store[key]
            return None

        return entry["value"]

    async def set(self, key: str, value: str, ttl: int = 3600) -> None:
        """
        키-값 쌍을 저장한다.

        Args:
            key: 저장 키
            value: 저장할 문자열 값
            ttl: 만료 시간(초). 기본 1시간(3600초)
        """
        if self._redis is not None:
            try:
                await self._redis.set(key, value, ex=ttl)
                return
            except Exception as e:
                logger.warning("Redis set 실패 (key=%s): %s — 폴백 사용", key, e)

        # 인메모리 폴백
        self._store[key] = {
            "value": value,
            "expires_at": time.time() + ttl if ttl > 0 else None,
        }

    async def delete(self, key: str) -> None:
        """
        키를 삭제한다.

        Args:
            key: 삭제할 키
        """
        if self._redis is not None:
            try:
                await self._redis.delete(key)
                return
            except Exception as e:
                logger.warning("Redis delete 실패 (key=%s): %s — 폴백 사용", key, e)

        # 인메모리 폴백
        self._store.pop(key, None)

    # ─── 대화 컨텍스트 ───

    async def get_conversation_context(self, session_id: str) -> list[dict]:
        """
        세션의 대화 컨텍스트를 복원한다.

        Redis에 JSON 직렬화된 메시지 목록을 저장하고,
        세션 재접속 시 이전 대화를 복원하는 데 사용한다.

        Args:
            session_id: 세션 식별자

        Returns:
            메시지 딕셔너리 목록 (비어 있으면 빈 리스트)
        """
        key = f"session:{session_id}:context"
        raw = await self.get(key)

        if raw is None:
            return []

        try:
            messages = json.loads(raw)
            if not isinstance(messages, list):
                logger.warning("대화 컨텍스트 형식 오류 (session=%s): list가 아님", session_id)
                return []
            return messages
        except json.JSONDecodeError as e:
            logger.error("대화 컨텍스트 JSON 파싱 실패 (session=%s): %s", session_id, e)
            return []

    async def save_conversation_context(
        self,
        session_id: str,
        messages: list[dict],
        ttl: int = 86400,
    ) -> None:
        """
        세션의 대화 컨텍스트를 저장한다.

        Args:
            session_id: 세션 식별자
            messages: 저장할 메시지 딕셔너리 목록
            ttl: 만료 시간(초). 기본 24시간(86400초)
        """
        key = f"session:{session_id}:context"
        try:
            raw = json.dumps(messages, ensure_ascii=False, default=str)
            await self.set(key, raw, ttl=ttl)
            logger.debug("대화 컨텍스트 저장 (session=%s): %d개 메시지", session_id, len(messages))
        except (TypeError, ValueError) as e:
            logger.error("대화 컨텍스트 직렬화 실패 (session=%s): %s", session_id, e)

    # ─── 도구 결과 캐시 ───

    async def get_tool_result_cache(self, tool_name: str, input_hash: str) -> str | None:
        """
        도구 실행 결과 캐시를 조회한다.

        동일한 입력으로 도구를 반복 호출할 때 캐시를 활용하여
        불필요한 재실행을 방지한다. 읽기 전용 도구(Read, Glob 등)에 유용하다.

        Args:
            tool_name: 도구 이름 (예: "Read", "Glob")
            input_hash: 도구 입력의 해시값

        Returns:
            캐시된 결과 문자열 또는 None
        """
        key = f"tool_cache:{tool_name}:{input_hash}"
        return await self.get(key)

    async def cache_tool_result(
        self,
        tool_name: str,
        input_hash: str,
        result: str,
        ttl: int = 1800,
    ) -> None:
        """
        도구 실행 결과를 캐시에 저장한다.

        Args:
            tool_name: 도구 이름
            input_hash: 도구 입력의 해시값
            result: 캐시할 결과 문자열
            ttl: 만료 시간(초). 기본 30분(1800초)
        """
        key = f"tool_cache:{tool_name}:{input_hash}"
        await self.set(key, result, ttl=ttl)
        logger.debug("도구 결과 캐시 저장: %s (hash=%s)", tool_name, input_hash[:8])

    # ─── 유틸리티 ───

    async def clear_session(self, session_id: str) -> None:
        """
        세션 관련 모든 데이터를 삭제한다.

        Args:
            session_id: 정리할 세션 식별자
        """
        context_key = f"session:{session_id}:context"
        await self.delete(context_key)
        logger.info("세션 데이터 정리 완료: %s", session_id)

    def _cleanup_expired(self) -> int:
        """
        만료된 인메모리 항목을 정리한다 (폴백 모드 전용).
        주기적으로 호출하여 메모리 누수를 방지한다.

        Returns:
            삭제된 항목 수
        """
        now = time.time()
        expired_keys = [
            k
            for k, v in self._store.items()
            if v.get("expires_at") is not None and now > v["expires_at"]
        ]
        for key in expired_keys:
            del self._store[key]

        if expired_keys:
            logger.debug("만료된 인메모리 항목 %d개 정리", len(expired_keys))
        return len(expired_keys)
