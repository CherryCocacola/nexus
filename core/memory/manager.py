"""
메모리 매니저 — 단기+장기 메모리를 통합 관리한다.

QueryEngine의 턴(turn) 생명주기에 맞춰 메모리를 관리한다:
  - on_turn_start(): 관련 메모리를 검색하여 컨텍스트에 주입할 목록 반환
  - on_turn_end(): 단기 저장 + 중요도 평가 + 장기 승격

외부 의존성:
  - ShortTermMemory (Redis): 빠른 세션 데이터 접근
  - LongTermMemory (PostgreSQL + pgvector): 영구 저장 + 벡터 검색
  - ModelProvider (optional): 임베딩 생성용 (e5-large)

설계 결정:
  - ModelProvider가 없어도 텍스트 기반 검색은 동작한다
  - 임베딩이 있으면 벡터 검색 우선, 없으면 텍스트 검색 폴백
  - 턴 종료 시 자동으로 중요도 평가 → 승격 여부 판단
"""

from __future__ import annotations

import hashlib
import logging
from typing import TYPE_CHECKING

from core.memory.importance import ImportanceAssessor
from core.memory.long_term import LongTermMemory
from core.memory.short_term import ShortTermMemory
from core.memory.types import MemoryEntry, MemoryType

if TYPE_CHECKING:
    from core.message import Message
    from core.model.inference import ModelProvider

logger = logging.getLogger("nexus.memory.manager")


class MemoryManager:
    """
    단기+장기 메모리를 통합 관리한다.

    QueryEngine과 Tool System이 이 클래스를 통해 메모리에 접근한다.
    턴(turn) 생명주기에 맞춰 자동으로 메모리를 관리한다.
    """

    def __init__(
        self,
        short_term: ShortTermMemory,
        long_term: LongTermMemory,
        model_provider: ModelProvider | None = None,
    ):
        """
        메모리 매니저를 초기화한다.

        Args:
            short_term: 단기 메모리 (Redis 또는 인메모리 폴백)
            long_term: 장기 메모리 (PostgreSQL 또는 인메모리 폴백)
            model_provider: 임베딩 생성용 모델 프로바이더 (없으면 텍스트 검색만 가능)
        """
        self._short_term = short_term
        self._long_term = long_term
        self._model_provider = model_provider
        self._importance_assessor = ImportanceAssessor()

        logger.info(
            "메모리 매니저 초기화: embedding=%s",
            "enabled" if model_provider else "disabled",
        )

    # ─── 턴 생명주기 ───

    async def on_turn_start(self, session_id: str, user_message: str) -> list[MemoryEntry]:
        """
        턴 시작 시 호출된다.
        사용자 메시지와 관련된 메모리를 검색하여 컨텍스트에 주입할 목록을 반환한다.

        처리 순서:
          1. 벡터 검색 시도 (ModelProvider가 있으면)
          2. 텍스트 검색 폴백
          3. 결과를 중요도순으로 정렬하여 반환

        Args:
            session_id: 현재 세션 ID
            user_message: 사용자 입력 텍스트

        Returns:
            관련 MemoryEntry 목록 (최대 10개)
        """
        results: list[MemoryEntry] = []

        # 1. 벡터 검색 시도
        if self._model_provider is not None:
            try:
                embeddings = await self._model_provider.embed([user_message])
                if embeddings and len(embeddings) > 0:
                    vector_results = await self._long_term.search_by_vector(
                        embedding=embeddings[0], top_k=5
                    )
                    results.extend(vector_results)
            except Exception as e:
                logger.warning("벡터 검색 실패: %s — 텍스트 검색으로 폴백", e)

        # 2. 텍스트 검색 (벡터 검색 결과가 부족하면 보충)
        if len(results) < 5:
            text_results = await self._long_term.search_by_text(
                query=user_message, top_k=10 - len(results)
            )
            # 중복 제거 (ID 기준)
            existing_ids = {e.id for e in results}
            for entry in text_results:
                if entry.id not in existing_ids:
                    results.append(entry)

        # 3. 중요도순 정렬 (높을수록 우선)
        results.sort(key=lambda e: e.importance, reverse=True)

        # 최대 10개로 제한
        results = results[:10]

        if results:
            logger.debug(
                "턴 시작 메모리 검색: session=%s, query='%s...', found=%d",
                session_id,
                user_message[:30],
                len(results),
            )

        return results

    async def on_turn_end(
        self,
        session_id: str,
        messages: list[Message],
        tool_results: list[str] | None = None,
    ) -> None:
        """
        턴 종료 시 호출된다.
        대화 내용을 단기 메모리에 저장하고, 중요한 내용은 장기 메모리로 승격한다.

        처리 순서:
          1. 대화 컨텍스트를 단기 메모리(Redis)에 저장
          2. assistant 메시지에서 중요도를 평가
          3. 중요도가 높으면 장기 메모리로 승격
          4. 도구 결과도 중요하면 장기 저장

        Args:
            session_id: 현재 세션 ID
            messages: 이번 턴의 메시지 목록
            tool_results: 도구 실행 결과 목록 (있으면)
        """
        # 1. 대화 컨텍스트를 단기 메모리에 저장
        serialized = [self._serialize_message(m) for m in messages]
        await self._short_term.save_conversation_context(session_id, serialized)

        # 2. assistant 메시지에서 중요 내용 추출 및 평가
        for msg in messages:
            role = msg.role if isinstance(msg.role, str) else msg.role.value
            if role != "assistant":
                continue

            content = msg.text_content
            if not content or len(content) < 20:
                continue

            # 중요도 평가
            importance = self._importance_assessor.assess(content, MemoryType.EPISODIC)

            # EPISODIC 메모리 엔트리 생성
            entry = MemoryEntry(
                memory_type=MemoryType.EPISODIC,
                content=content[:2000],  # 최대 2000자로 제한
                key=f"turn:{session_id}",
                tags=["conversation", session_id],
                importance=importance,
                metadata={"session_id": session_id, "role": role},
            )

            # 임베딩 생성 (ModelProvider가 있으면)
            if self._model_provider is not None:
                try:
                    embeddings = await self._model_provider.embed([content[:500]])
                    if embeddings and len(embeddings) > 0:
                        entry = entry.model_copy(update={"embedding": embeddings[0]})
                except Exception as e:
                    logger.warning("임베딩 생성 실패: %s", e)

            # 승격 판단 → 장기 메모리에 저장
            if self._importance_assessor.should_promote(entry):
                await self._long_term.add(entry)
                logger.debug(
                    "장기 메모리 승격: id=%s, importance=%.2f",
                    entry.id,
                    entry.importance,
                )

        # 3. 도구 결과 중 중요한 것도 장기 저장
        if tool_results:
            for result_text in tool_results:
                if not result_text or len(result_text) < 30:
                    continue

                importance = self._importance_assessor.assess(result_text, MemoryType.PROCEDURAL)
                if importance > 0.6:
                    entry = MemoryEntry(
                        memory_type=MemoryType.PROCEDURAL,
                        content=result_text[:2000],
                        key=f"tool_result:{session_id}",
                        tags=["tool_result", session_id],
                        importance=importance,
                        metadata={"session_id": session_id},
                    )
                    await self._long_term.add(entry)

    # ─── 검색 ───

    async def search_relevant(self, query: str, top_k: int = 10) -> list[MemoryEntry]:
        """
        질문과 관련된 메모리를 검색한다.

        벡터 검색과 텍스트 검색을 병합하여 최적의 결과를 반환한다.
        MemoryReadTool에서 호출된다.

        Args:
            query: 검색 쿼리
            top_k: 최대 반환 건수

        Returns:
            관련 MemoryEntry 목록
        """
        results: list[MemoryEntry] = []
        existing_ids: set[str] = set()

        # 벡터 검색 시도
        if self._model_provider is not None:
            try:
                embeddings = await self._model_provider.embed([query])
                if embeddings and len(embeddings) > 0:
                    vector_results = await self._long_term.search_by_vector(
                        embedding=embeddings[0], top_k=top_k
                    )
                    for entry in vector_results:
                        if entry.id not in existing_ids:
                            results.append(entry)
                            existing_ids.add(entry.id)
            except Exception as e:
                logger.warning("벡터 검색 실패: %s", e)

        # 텍스트 검색으로 보충
        remaining = top_k - len(results)
        if remaining > 0:
            text_results = await self._long_term.search_by_text(query=query, top_k=remaining)
            for entry in text_results:
                if entry.id not in existing_ids:
                    results.append(entry)
                    existing_ids.add(entry.id)

        return results[:top_k]

    # ─── 직접 추가 ───

    async def add_semantic(
        self,
        key: str,
        value: str,
        tags: list[str] | None = None,
    ) -> str:
        """
        시맨틱 메모리를 직접 추가한다.

        MemoryWriteTool이나 외부 모듈에서 호출하여
        명시적으로 지식/사실을 저장할 때 사용한다.

        Args:
            key: 메모리 키 (예: "project_architecture", "user_preference_theme")
            value: 저장할 텍스트 내용
            tags: 분류 태그 목록

        Returns:
            저장된 메모리 ID
        """
        importance = self._importance_assessor.assess(value, MemoryType.SEMANTIC)

        entry = MemoryEntry(
            memory_type=MemoryType.SEMANTIC,
            content=value,
            key=key,
            tags=tags or [],
            importance=max(importance, 0.5),  # SEMANTIC은 최소 0.5
        )

        # 임베딩 생성
        if self._model_provider is not None:
            try:
                embeddings = await self._model_provider.embed([value[:500]])
                if embeddings and len(embeddings) > 0:
                    entry = entry.model_copy(update={"embedding": embeddings[0]})
            except Exception as e:
                logger.warning("임베딩 생성 실패: %s", e)

        memory_id = await self._long_term.add(entry)
        logger.info("시맨틱 메모리 추가: id=%s, key='%s'", memory_id, key)
        return memory_id

    async def add_feedback(self, content: str, tags: list[str] | None = None) -> str:
        """
        피드백 메모리를 추가한다.

        사용자의 긍정/부정 피드백을 저장하여 향후 응답 개선에 활용한다.

        Args:
            content: 피드백 내용
            tags: 분류 태그 목록

        Returns:
            저장된 메모리 ID
        """
        importance = self._importance_assessor.assess(content, MemoryType.FEEDBACK)

        entry = MemoryEntry(
            memory_type=MemoryType.FEEDBACK,
            content=content,
            key="feedback",
            tags=tags or ["feedback"],
            importance=max(importance, 0.5),
        )

        memory_id = await self._long_term.add(entry)
        logger.info("피드백 메모리 추가: id=%s", memory_id)
        return memory_id

    async def add_user_profile(self, key: str, value: str) -> str:
        """
        사용자 프로필 메모리를 추가한다.

        사용자의 선호, 스타일, 설정을 장기 저장한다.
        USER_PROFILE 타입은 감쇠가 매우 느리다 (반감기 365일).

        Args:
            key: 프로필 키 (예: "preferred_language", "coding_style")
            value: 프로필 값

        Returns:
            저장된 메모리 ID
        """
        entry = MemoryEntry(
            memory_type=MemoryType.USER_PROFILE,
            content=value,
            key=f"user_profile:{key}",
            tags=["user_profile", key],
            importance=0.8,  # 사용자 프로필은 높은 중요도
        )

        memory_id = await self._long_term.add(entry)
        logger.info("사용자 프로필 추가: id=%s, key='%s'", memory_id, key)
        return memory_id

    # ─── 도구 결과 캐시 ───

    async def get_cached_tool_result(self, tool_name: str, input_data: dict) -> str | None:
        """
        도구 결과 캐시를 조회한다.

        Args:
            tool_name: 도구 이름
            input_data: 도구 입력 데이터

        Returns:
            캐시된 결과 또는 None
        """
        input_hash = self._hash_input(input_data)
        return await self._short_term.get_tool_result_cache(tool_name, input_hash)

    async def cache_tool_result(
        self, tool_name: str, input_data: dict, result: str, ttl: int = 1800
    ) -> None:
        """
        도구 결과를 캐시에 저장한다.

        Args:
            tool_name: 도구 이름
            input_data: 도구 입력 데이터
            result: 캐시할 결과
            ttl: 만료 시간(초)
        """
        input_hash = self._hash_input(input_data)
        await self._short_term.cache_tool_result(tool_name, input_hash, result, ttl)

    # ─── 내부 유틸리티 ───

    @staticmethod
    def _serialize_message(message: Message) -> dict:
        """Message를 직렬화 가능한 dict로 변환한다."""
        try:
            return message.model_dump(mode="json")
        except Exception:
            # 최소한의 폴백 직렬화
            return {
                "role": str(message.role),
                "content": message.text_content,
            }

    @staticmethod
    def _hash_input(input_data: dict) -> str:
        """도구 입력 데이터의 해시를 생성한다."""
        import json

        serialized = json.dumps(input_data, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]

    # ─── 프로퍼티 ───

    @property
    def short_term(self) -> ShortTermMemory:
        """단기 메모리 인스턴스를 반환한다."""
        return self._short_term

    @property
    def long_term(self) -> LongTermMemory:
        """장기 메모리 인스턴스를 반환한다."""
        return self._long_term
