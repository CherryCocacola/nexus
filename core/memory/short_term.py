"""
단기 메모리(Short-Term Memory) — Redis 기반 세션 컨텍스트 + 도구 결과 캐시.

[이 파일이 하는 일]
Nexus의 "단기 기억"을 담당하는 모듈이다. 사람으로 치면 방금 나눈 대화나
잠깐 기억해 두면 되는 메모지 같은, 오래 보관할 필요는 없지만 세션 중에는
아주 빠르게 꺼내 써야 하는 데이터를 저장한다. 반대로 오래 보관해야 하는
"장기 기억"은 PostgreSQL + pgvector를 쓰는 long_term.py가 담당한다.

[무엇을 저장하나]
  - 대화 컨텍스트   : session:{session_id}:context   (세션 재접속 시 대화 복원용)
  - 도구 결과 캐시  : tool_cache:{tool_name}:{input_hash} (같은 입력 재실행 방지)
  - 임시 키-값 데이터: get/set/delete 로 다루는 범용 저장

[핵심 클래스]
  - ShortTermMemory : 위 세 종류 데이터를 다루는 유일한 클래스. 아래 CRUD와
    대화 컨텍스트/도구 캐시 헬퍼, 세션 정리·조회 유틸을 제공한다.

[동작 방식 — 폴백 이중화]
Redis 클라이언트가 주어지면 Redis를 쓰고, 없거나 호출이 실패하면 자동으로
프로세스 내부 딕셔너리(self._store)로 폴백한다. 덕분에 Redis 서버 없이도
개발·테스트가 가능하고, 운영 중 일시적 Redis 장애에도 죽지 않는다.
에어갭(폐쇄망) 환경에서 Redis는 외부가 아닌 LAN 내부 서버(192.168.x.x)에 둔다.

[설계 결정 — TTL(만료 시간) 기본값]
  - 일반 키       : 1시간(3600초)   — 잠깐 쓰는 임시 데이터
  - 대화 컨텍스트 : 24시간(86400초) — 하루 안에 재접속하면 대화가 이어지도록
  - 도구 결과 캐시: 30분(1800초)    — 파일이 바뀔 수 있으니 짧게 유지
직렬화는 대화 컨텍스트를 JSON 문자열로 변환해 저장한다(한글 보존 위해
ensure_ascii=False 사용).

작성자: 이현수 / 작성일: 2026-07-05
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

logger = logging.getLogger("nexus.memory.short_term")


class ShortTermMemory:
    """
    Redis 기반 단기 메모리 저장소.

    세션 컨텍스트, 도구 결과 캐시, 임시 키-값 데이터를 저장한다.
    모든 메서드는 async 이며, 내부적으로 Redis(우선)와 인메모리 딕셔너리(폴백)
    두 경로를 동일한 인터페이스로 감싼다. 그래서 이 클래스를 쓰는 쪽은
    Redis 유무를 신경 쓰지 않고 get/set/delete 만 호출하면 된다.

    [상태 필드]
      - self._redis : 주입받은 redis.asyncio.Redis 인스턴스 (없으면 None)
      - self._store : 인메모리 폴백 저장소. Redis가 없거나 실패할 때만 쓴다.
    """

    def __init__(self, redis_client: Any | None = None):
        """
        단기 메모리를 초기화한다.

        Redis 클라이언트는 밖에서 만들어 주입(의존성 주입)받는다. 이렇게 하면
        테스트할 때 fakeredis 나 None 을 넘겨 실제 서버 없이도 검증할 수 있다.

        Args:
            redis_client: redis.asyncio.Redis 인스턴스.
                None 이면 인메모리 폴백 모드로 동작한다.
        """
        # 인메모리 폴백 저장소. 구조는 {key: {"value": str, "expires_at": float|None}}
        # expires_at 이 None 이면 만료 없음(영구), 값이 있으면 그 시각(epoch초) 이후 만료.
        self._store: dict[str, dict[str, Any]] = {}
        # 주입받은 Redis 클라이언트를 보관. None 이면 아래에서 폴백 모드 안내 로그를 남긴다.
        self._redis = redis_client

        # Redis가 없으면 개발자가 상황을 바로 알 수 있도록 정보 로그를 남긴다.
        if self._redis is None:
            logger.info("Redis 클라이언트 없음 — 인메모리 폴백 모드로 동작")

    # ─── 기본 CRUD ───

    async def get(self, key: str) -> str | None:
        """
        키에 해당하는 값을 조회한다.

        모든 키 조회의 공통 진입점이다. get_conversation_context 나
        get_tool_result_cache 같은 헬퍼도 결국 내부에서 이 메서드를 부른다.
        Redis가 있으면 Redis에서 읽고, 없거나 실패하면 인메모리 폴백에서 읽는다.
        인메모리 폴백은 Redis와 달리 스스로 만료시키지 못하므로, 여기서 직접
        TTL을 확인해 만료된 키는 None을 돌려주고 즉시 삭제한다(지연 만료).

        Args:
            key: 조회할 키

        Returns:
            저장된 문자열 값. 없거나 만료됐으면 None.
        """
        # 1순위: Redis 경로. 실패하면 아래 폴백으로 자연스럽게 흘러간다.
        if self._redis is not None:
            try:
                value = await self._redis.get(key)
                # Redis는 응답을 bytes로 줄 수 있으므로 문자열로 통일해서 반환한다.
                return value.decode("utf-8") if isinstance(value, bytes) else value
            except Exception as e:
                # Redis 장애 시 예외를 삼키지 않고 경고만 남긴 뒤 폴백으로 진행한다.
                logger.warning("Redis get 실패 (key=%s): %s — 폴백 사용", key, e)

        # 2순위: 인메모리 폴백에서 조회.
        entry = self._store.get(key)
        if entry is None:
            return None

        # TTL 만료 확인 — Redis처럼 자동 삭제가 안 되므로 읽는 시점에 직접 검사한다.
        expires_at = entry.get("expires_at")
        if expires_at is not None and time.time() > expires_at:
            # 이미 만료된 항목이면 정리하고 없는 것처럼 취급한다.
            del self._store[key]
            return None

        return entry["value"]

    async def set(self, key: str, value: str, ttl: int = 3600) -> None:
        """
        키-값 쌍을 TTL과 함께 저장한다.

        모든 키 저장의 공통 진입점이다. Redis에서는 ex 인자로 만료를 맡기고,
        인메모리 폴백에서는 만료 시각(expires_at)을 직접 계산해 함께 보관한다.
        ttl 이 0 이하이면 만료 없이(영구) 저장한다.

        Args:
            key: 저장 키
            value: 저장할 문자열 값
            ttl: 만료 시간(초). 기본 1시간(3600초)
        """
        # 1순위: Redis 경로. 성공하면 바로 반환하고 폴백은 건드리지 않는다.
        if self._redis is not None:
            try:
                await self._redis.set(key, value, ex=ttl)
                return
            except Exception as e:
                # Redis 저장 실패 시 경고만 남기고 폴백에 저장해 데이터 유실을 막는다.
                logger.warning("Redis set 실패 (key=%s): %s — 폴백 사용", key, e)

        # 2순위: 인메모리 폴백에 저장. ttl>0 이면 만료 시각을 지금+ttl 로 계산한다.
        self._store[key] = {
            "value": value,
            "expires_at": time.time() + ttl if ttl > 0 else None,
        }

    async def delete(self, key: str) -> None:
        """
        키를 삭제한다.

        존재하지 않는 키를 지워도 오류가 아니다(멱등). Redis/인메모리 어느
        경로든 "없으면 조용히 넘어간다"로 통일한다.

        Args:
            key: 삭제할 키
        """
        # 1순위: Redis 경로.
        if self._redis is not None:
            try:
                await self._redis.delete(key)
                return
            except Exception as e:
                # 삭제 실패 시에도 폴백 저장소에서 마저 지워 상태를 맞춘다.
                logger.warning("Redis delete 실패 (key=%s): %s — 폴백 사용", key, e)

        # 2순위: 인메모리 폴백. pop(key, None) 은 키가 없어도 예외를 내지 않는다.
        self._store.pop(key, None)

    # ─── 대화 컨텍스트 ───

    async def get_conversation_context(self, session_id: str) -> list[dict]:
        """
        세션의 대화 컨텍스트(메시지 목록)를 복원한다.

        save_conversation_context 가 JSON 문자열로 저장해 둔 것을 다시 파이썬
        list[dict] 로 되돌린다. 세션에 재접속했을 때 이전 대화를 이어가기 위한
        용도다. 저장된 값이 없거나, 형식이 깨졌거나, JSON 파싱이 실패하면
        예외를 밖으로 던지지 않고 안전하게 빈 리스트를 돌려준다(호출자 보호).

        Args:
            session_id: 세션 식별자

        Returns:
            메시지 딕셔너리 목록. 데이터가 없거나 손상됐으면 빈 리스트.
        """
        # 저장 시 사용한 것과 동일한 키 규칙으로 조회한다.
        key = f"session:{session_id}:context"
        raw = await self.get(key)

        # 저장된 컨텍스트가 아예 없는 신규/만료 세션이면 빈 대화로 시작한다.
        if raw is None:
            return []

        try:
            messages = json.loads(raw)
            # 정상이라면 list 여야 한다. 아니면 손상된 데이터로 보고 무시한다.
            if not isinstance(messages, list):
                logger.warning("대화 컨텍스트 형식 오류 (session=%s): list가 아님", session_id)
                return []
            return messages
        except json.JSONDecodeError as e:
            # JSON이 깨졌어도 세션 전체를 죽이지 않고 빈 대화로 복구한다.
            logger.error("대화 컨텍스트 JSON 파싱 실패 (session=%s): %s", session_id, e)
            return []

    async def save_conversation_context(
        self,
        session_id: str,
        messages: list[dict],
        ttl: int = 86400,
    ) -> None:
        """
        세션의 대화 컨텍스트(메시지 목록)를 JSON으로 직렬화해 저장한다.

        get_conversation_context 와 짝이 되는 메서드다. 한글이 유니코드
        이스케이프로 깨지지 않도록 ensure_ascii=False 를, JSON으로 바로 못
        바꾸는 객체(예: datetime)를 만나면 문자열로라도 저장되도록
        default=str 를 준다. 직렬화 실패 시에도 예외를 던지지 않고 로그만
        남겨 대화 루프가 멈추지 않게 한다.

        Args:
            session_id: 세션 식별자
            messages: 저장할 메시지 딕셔너리 목록
            ttl: 만료 시간(초). 기본 24시간(86400초)
        """
        # 복원 때와 동일한 키 규칙을 사용한다.
        key = f"session:{session_id}:context"
        try:
            # 한글 보존(ensure_ascii=False) + 비직렬화 객체 방어(default=str).
            raw = json.dumps(messages, ensure_ascii=False, default=str)
            await self.set(key, raw, ttl=ttl)
            logger.debug("대화 컨텍스트 저장 (session=%s): %d개 메시지", session_id, len(messages))
        except (TypeError, ValueError) as e:
            # 직렬화가 실패해도 저장만 건너뛰고 상위 흐름은 계속 진행한다.
            logger.error("대화 컨텍스트 직렬화 실패 (session=%s): %s", session_id, e)

    # ─── 도구 결과 캐시 ───

    async def get_tool_result_cache(self, tool_name: str, input_hash: str) -> str | None:
        """
        도구 실행 결과 캐시를 조회한다.

        같은 입력으로 같은 도구를 또 부를 때, 실제 실행 대신 캐시된 결과를
        재사용해 낭비를 막는다. 특히 부작용이 없는 읽기 전용 도구(Read, Glob
        등)에 유용하다. 캐시가 없으면(미스) None 을 돌려주고, 호출자는 그때
        도구를 실제로 실행한 뒤 cache_tool_result 로 결과를 채워 넣으면 된다.

        Args:
            tool_name: 도구 이름 (예: "Read", "Glob")
            input_hash: 도구 입력을 해시한 값 (동일 입력 식별용 지문)

        Returns:
            캐시된 결과 문자열. 캐시가 없거나 만료됐으면 None.
        """
        # 캐시 키는 도구 이름 + 입력 해시 조합이라 입력이 다르면 다른 슬롯에 저장된다.
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

        get_tool_result_cache 와 짝이 된다. 파일 등 원본이 바뀔 수 있으므로
        기본 TTL을 30분으로 짧게 잡아 오래된 결과를 계속 내주지 않도록 한다.

        Args:
            tool_name: 도구 이름
            input_hash: 도구 입력을 해시한 값 (조회 때와 동일해야 매칭됨)
            result: 캐시할 결과 문자열
            ttl: 만료 시간(초). 기본 30분(1800초)
        """
        # 조회(get_tool_result_cache)와 동일한 키 규칙으로 저장해야 나중에 찾을 수 있다.
        key = f"tool_cache:{tool_name}:{input_hash}"
        await self.set(key, result, ttl=ttl)
        # 로그에는 해시 앞 8자리만 남겨 가독성을 확보한다(전체는 너무 길다).
        logger.debug("도구 결과 캐시 저장: %s (hash=%s)", tool_name, input_hash[:8])

    # ─── 유틸리티 ───

    async def clear_session(self, session_id: str) -> None:
        """
        세션 관련 데이터를 삭제한다.

        현재는 대화 컨텍스트 키만 지운다. 세션 종료나 초기화 시 남은
        데이터를 정리하는 용도다. (TODO 성격의 참고: 세션에 종속된 다른
        키 종류가 생기면 여기서 함께 지우도록 확장하면 된다.)

        Args:
            session_id: 정리할 세션 식별자
        """
        # 대화 컨텍스트 키를 저장 때와 같은 규칙으로 만들어 삭제한다.
        context_key = f"session:{session_id}:context"
        await self.delete(context_key)
        logger.info("세션 데이터 정리 완료: %s", session_id)

    async def list_sessions(self, limit: int = 100) -> list[str]:
        """
        저장된 세션 ID 목록을 반환한다.

        Redis에서는 SCAN 커서로 `session:*:context` 패턴을 순회하여 키를 모으고,
        접두/접미를 잘라 session_id를 돌려준다. 인메모리 폴백 모드에서는 내부
        딕셔너리 키를 같은 방식으로 필터링한다.

        Args:
            limit: 최대 반환 건수 (Redis SCAN의 페이지 크기와 별개의 상한)

        Returns:
            세션 ID 문자열 리스트 (정렬되지 않음 — 호출자가 필요 시 정렬)
        """
        # 세션 컨텍스트 키는 "session:{id}:context" 형태다. 앞뒤 고정 부분을
        # 상수로 두고, 그 사이의 실제 session_id 만 잘라내기 위한 기준으로 쓴다.
        prefix = "session:"
        suffix = ":context"

        def _extract(key: str) -> str | None:
            # 키가 정확히 session:...:context 패턴일 때만 가운데 id를 잘라 반환한다.
            # 그 외 형태(다른 종류의 키)는 세션이 아니므로 None 을 돌려 걸러낸다.
            if key.startswith(prefix) and key.endswith(suffix):
                return key[len(prefix): -len(suffix)]
            return None

        sessions: list[str] = []

        # 1순위: Redis. 전체 키를 한 번에 읽지 않고 SCAN 커서로 나눠 순회해
        # 서버 부하를 줄인다(KEYS 명령의 블로킹 문제 회피).
        if self._redis is not None:
            try:
                # redis.asyncio의 scan_iter는 async generator를 반환한다.
                async for raw_key in self._redis.scan_iter(match="session:*:context"):
                    # Redis 키가 bytes로 올 수 있으므로 문자열로 통일한다.
                    key = (
                        raw_key.decode("utf-8")
                        if isinstance(raw_key, bytes)
                        else raw_key
                    )
                    sid = _extract(key)
                    if sid is not None:
                        sessions.append(sid)
                        # limit 만큼 모았으면 더 순회하지 않고 즉시 멈춘다.
                        if len(sessions) >= limit:
                            break
                return sessions
            except Exception as e:
                # SCAN 실패 시 예외 대신 폴백으로 넘어가 최소한의 결과라도 준다.
                logger.warning("Redis scan_iter 실패: %s — 인메모리 폴백 사용", e)

        # 2순위: 인메모리 폴백. 내부 딕셔너리 키를 같은 방식으로 필터링한다.
        # 앞에서 limit 개만 잘라 보므로 실제 세션 수는 limit보다 적을 수 있다.
        for key in list(self._store.keys())[:limit]:
            sid = _extract(key)
            if sid is not None:
                sessions.append(sid)
        return sessions

    def _cleanup_expired(self) -> int:
        """
        만료된 인메모리 항목을 한꺼번에 정리한다 (폴백 모드 전용).

        get()은 "읽는 시점"에만 만료를 처리하므로, 한 번 넣고 다시 읽지 않는
        키는 만료돼도 딕셔너리에 계속 남아 메모리를 잠식할 수 있다. 이 메서드를
        주기적으로 호출해 그런 유령 항목을 쓸어 담는다. Redis 모드에서는 서버가
        알아서 만료시키므로 이 함수가 필요 없다(그래서 폴백 전용).

        Returns:
            삭제된 항목 수
        """
        now = time.time()
        # 먼저 만료된 키만 골라 리스트로 모은다. 순회 중 딕셔너리를 바로 지우면
        # "반복 도중 크기 변경" 오류가 나므로, 수집과 삭제를 두 단계로 나눈다.
        expired_keys = [
            k
            for k, v in self._store.items()
            if v.get("expires_at") is not None and now > v["expires_at"]
        ]
        # 모아둔 만료 키를 실제로 삭제한다.
        for key in expired_keys:
            del self._store[key]

        # 정리한 게 있을 때만 로그를 남겨 불필요한 로그 소음을 줄인다.
        if expired_keys:
            logger.debug("만료된 인메모리 항목 %d개 정리", len(expired_keys))
        return len(expired_keys)
