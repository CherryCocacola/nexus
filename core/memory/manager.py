"""
메모리 매니저 — 단기 메모리와 장기 메모리를 하나의 진입점으로 통합 관리한다.

[이 파일이 하는 일]
  Nexus의 "기억" 담당 모듈이다. 사람이 대화하며 방금 한 말은 짧게, 중요한 말은
  오래 기억하듯, 이 매니저는 대화를 단기 저장소(Redis)에 넣고, 그중 중요한 것만
  골라 장기 저장소(PostgreSQL+pgvector)로 옮긴다. 또한 새 사용자 입력이 들어오면
  과거 기억 중 관련된 것을 찾아 모델 컨텍스트에 붙여줄 수 있게 반환한다.

[QueryEngine 턴(turn) 생명주기와의 연결]
  - on_turn_start(): 턴이 시작될 때 호출. 사용자 메시지와 관련된 과거 기억을
    검색해 "컨텍스트에 주입할 후보" 목록으로 돌려준다.
  - on_turn_end(): 턴이 끝날 때 호출. 이번 턴 대화를 단기 저장하고, 중요도를
    평가해 기준을 넘는 내용만 장기 메모리로 승격(promote)한다.

[주요 클래스]
  - MemoryManager: 아래 정의된 유일한 공개 클래스. 매니저 자체.

[외부 의존성]
  - ShortTermMemory (Redis): 세션 대화 이력·도구결과 캐시 등 빠른 접근용 단기 저장소.
  - LongTermMemory (PostgreSQL + pgvector): 영구 저장 + 벡터/텍스트 검색용 장기 저장소.
  - ModelProvider (선택): 임베딩(e5-large) 생성용. 없으면 벡터 검색을 건너뛴다.
  - ImportanceAssessor: 텍스트의 중요도 점수와 승격 여부를 판정하는 헬퍼.

[핵심 설계 결정]
  - ModelProvider가 없어도(임베딩 불가) 텍스트 기반 검색만으로 동작하도록 설계.
  - 임베딩이 가능하면 벡터 검색을 우선하고, 결과가 모자라면 텍스트 검색으로 보충한다.
  - 턴 종료 시 자동으로 중요도를 평가해 승격 여부를 판단하므로, 호출 측은 저장
    로직을 신경 쓸 필요 없이 on_turn_end()만 부르면 된다.

작성자: 이현수 / 작성일: 2026-07-05
"""

from __future__ import annotations

import hashlib
import logging
from typing import TYPE_CHECKING

from core.memory.importance import ImportanceAssessor
from core.memory.long_term import LongTermMemory
from core.memory.short_term import ShortTermMemory
from core.memory.types import MemoryEntry, MemoryType, is_rag_chunk

if TYPE_CHECKING:
    from core.message import Message
    from core.model.inference import ModelProvider

logger = logging.getLogger("nexus.memory.manager")


class MemoryManager:
    """
    단기+장기 메모리를 통합 관리하는 매니저 클래스.

    [역할]
      QueryEngine과 Tool System이 메모리를 다룰 때 반드시 이 클래스를 통한다.
      즉 "기억"에 대한 단일 창구(facade) 역할을 하여, 호출 측이 Redis나
      PostgreSQL 같은 저장소 세부 구현을 몰라도 되게 감춰준다.

    [주요 메서드 그룹]
      - 턴 생명주기: on_turn_start(검색 주입), on_turn_end(저장+승격)
      - 검색: search_relevant(도구에서 명시적 검색)
      - 직접 추가: add_semantic / add_feedback / add_user_profile
      - 도구 결과 캐시: get_cached_tool_result / cache_tool_result
      - 내부 유틸: 직렬화·해시 헬퍼(스태틱)

    [보관 상태(인스턴스 필드)]
      - _short_term / _long_term: 두 저장소 인스턴스
      - _model_provider: 임베딩 생성기(없으면 None)
      - _importance_assessor: 중요도 평가 헬퍼(항상 존재)
    """

    def __init__(
        self,
        short_term: ShortTermMemory,
        long_term: LongTermMemory,
        model_provider: ModelProvider | None = None,
    ):
        """
        메모리 매니저를 초기화한다.

        저장소 인스턴스들을 주입(의존성 주입)받아 필드에 보관하고, 중요도 평가
        헬퍼는 여기서 직접 생성한다. 초기화 시점에 임베딩 사용 가능 여부를
        로그로 남겨, 운영 중 벡터 검색이 켜졌는지 한눈에 확인할 수 있게 한다.

        Args:
            short_term: 단기 메모리 (Redis, 연결 불가 시 인메모리 폴백 구현이 올 수 있음)
            long_term: 장기 메모리 (PostgreSQL+pgvector, 인메모리 폴백 가능)
            model_provider: 임베딩 생성용 모델 프로바이더.
                None이면 임베딩을 만들지 못하므로 벡터 검색은 생략되고
                텍스트 검색만 사용된다.
        """
        # 주입받은 저장소/프로바이더를 그대로 필드에 보관한다(소유권은 호출 측에 있음).
        self._short_term = short_term
        self._long_term = long_term
        self._model_provider = model_provider
        # 중요도 평가기는 상태가 가벼우므로 매니저가 직접 소유·생성한다.
        self._importance_assessor = ImportanceAssessor()

        logger.info(
            "메모리 매니저 초기화: embedding=%s",
            "enabled" if model_provider else "disabled",
        )

    # ─── 턴 생명주기 ───

    async def on_turn_start(self, session_id: str, user_message: str) -> list[MemoryEntry]:
        """
        턴 시작 시 호출된다. 사용자 메시지와 관련된 과거 기억을 찾아, 이번 턴의
        모델 컨텍스트에 주입할 후보 목록을 반환한다.

        [왜 필요한가]
          모델은 자기 컨텍스트 창 밖의 과거를 스스로 기억하지 못한다. 그래서 매턴
          시작 시 관련 기억을 미리 뽑아 넣어줘야 "이전에 말한 내용을 아는" 것처럼
          동작한다. 이 메서드가 그 "관련 기억 선별" 단계를 담당한다.

        [처리 순서]
          1. 벡터 검색 시도 — ModelProvider가 있으면 사용자 메시지를 임베딩해
             의미가 비슷한 기억을 top_k=5로 찾는다.
          2. 텍스트 검색으로 보충 — 벡터 결과가 5건 미만이면 부족분을 텍스트
             검색으로 채운다(ID 기준 중복 제거).
          3. 중요도 내림차순 정렬 후 최대 10건으로 잘라 반환.

        Args:
            session_id: 현재 세션 ID(로깅용)
            user_message: 사용자 입력 텍스트(검색 쿼리로 사용)

        Returns:
            관련 MemoryEntry 목록 (최대 10개, 중요도 높은 순)
        """
        # 결과를 누적할 리스트. 벡터 검색분과 텍스트 검색분을 여기에 합친다.
        results: list[MemoryEntry] = []

        # 1. 벡터 검색 시도 (임베딩 가능할 때만)
        #    임베딩 실패는 치명적이지 않으므로 경고만 남기고 텍스트 검색으로 넘어간다.
        if self._model_provider is not None:
            try:
                # 사용자 메시지 1건을 임베딩 → 벡터 하나를 얻는다.
                embeddings = await self._model_provider.embed([user_message])
                if embeddings and len(embeddings) > 0:
                    vector_results = await self._long_term.search_by_vector(
                        embedding=embeddings[0], top_k=5
                    )
                    results.extend(vector_results)
            except Exception as e:
                logger.warning("벡터 검색 실패: %s — 텍스트 검색으로 폴백", e)

        # 2. 텍스트 검색으로 보충 — 벡터 결과가 5건 미만일 때만 부족분을 채운다.
        #    (전체 상한 10건 - 지금까지 모은 건수)만큼 추가로 요청한다.
        if len(results) < 5:
            text_results = await self._long_term.search_by_text(
                query=user_message, top_k=10 - len(results)
            )
            # 이미 벡터 검색으로 담긴 항목과 겹치지 않게 ID 기준으로 중복 제거한다.
            existing_ids = {e.id for e in results}
            for entry in text_results:
                if entry.id not in existing_ids:
                    results.append(entry)

        # 2.5 코드 RAG 청크를 걸러낸다 — 대화 회상에 소스코드가 섞이면 안 된다.
        #     같은 tb_memories 에 코드 인덱스(113만 건)와 대화 기억(수백 건)이 함께
        #     사는데, RAG 청크는 importance=0.8 로 높아 아래 중요도 정렬에서 앞자리를
        #     독차지한다. 코드 검색은 SymbolSearch/RAG 리트리버가 따로 담당한다.
        results = [e for e in results if not is_rag_chunk(e)]

        # 3. 중요도 내림차순 정렬 (importance가 클수록 앞으로) — 주입 우선순위 결정.
        results.sort(key=lambda e: e.importance, reverse=True)

        # 컨텍스트 과다 주입을 막기 위해 최대 10건으로 자른다.
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
        channel: str | None = None,
    ) -> None:
        """
        턴 종료 시 호출된다. 이번 턴의 대화를 단기 메모리에 저장하고, 그중 중요한
        내용만 골라 장기 메모리로 승격(promote)한다.

        [왜 필요한가]
          모든 대화를 무조건 장기 저장하면 저장소가 잡음으로 가득 차 검색 품질이
          떨어진다. 그래서 "일단 단기 저장 → 중요도 평가 → 기준 통과분만 장기 승격"
          이라는 필터링 파이프라인을 둔다. 사람이 중요한 일만 오래 기억하는 것과 같다.

        [처리 순서]
          1. 이번 턴 메시지 전체를 직렬화해 단기 메모리(Redis)에 대화 컨텍스트로 저장.
          2. assistant 메시지만 골라 본문 중요도를 평가(너무 짧은 건 건너뜀).
          3. 승격 기준(should_promote)을 통과하면 EPISODIC 엔트리로 장기 저장.
          4. 도구 실행 결과 중 중요도가 높은 것은 PROCEDURAL 엔트리로 장기 저장.

        Args:
            session_id: 현재 세션 ID(저장 키·태그·메타데이터에 사용)
            messages: 이번 턴에서 오간 Message 목록
            tool_results: 이번 턴의 도구 실행 결과 텍스트 목록(없으면 None)
            channel: 진입점 채널(web/cli/api). 단기 메모리 키를 채널별로 격리한다.
                None이면 flat 키(하위호환). 장기 메모리(tb_memories)는 채널 무관 글로벌 유지.
        """
        # 1. 이번 턴 메시지를 dict로 직렬화해 단기 메모리에 통째로 저장한다.
        #    (직렬화 규칙은 _serialize_message 참고 — content를 평문으로 통일한다.)
        #    channel을 넘겨 web/cli/api 히스토리를 Redis 키 네임스페이스로 분리한다.
        serialized = [self._serialize_message(m) for m in messages]
        await self._short_term.save_conversation_context(
            session_id, serialized, channel=channel
        )

        # 2. assistant 메시지만 골라 중요 내용을 추출·평가한다.
        for msg in messages:
            # role은 문자열일 수도, Enum일 수도 있어 양쪽 모두 문자열로 정규화한다.
            role = msg.role if isinstance(msg.role, str) else msg.role.value
            # 사용자/도구 메시지는 여기서 승격 대상이 아니므로 건너뛴다.
            if role != "assistant":
                continue

            content = msg.text_content
            # 내용이 비었거나 너무 짧으면(20자 미만) 저장 가치가 낮아 건너뛴다.
            if not content or len(content) < 20:
                continue

            # 중요도 평가 — EPISODIC(일화 기억) 타입 기준으로 점수를 매긴다.
            importance = self._importance_assessor.assess(content, MemoryType.EPISODIC)

            # 저장 본문은 과도한 길이를 막기 위해 최대 2000자로 자른다.
            stored_content = content[:2000]

            # EPISODIC 메모리 엔트리 생성
            #
            # key에 content 해시를 붙이는 이유 (consolidate 붕괴 버그 수정):
            #   과거에는 key=f"turn:{session_id}" 로 "한 세션의 모든 assistant 턴"이
            #   동일 key를 가졌다. consolidate()는 같은 key를 한 그룹으로 묶어 대표
            #   1건만 남기고 나머지를 삭제하므로, 한 세션의 서로 다른 턴 기억 전체가
            #   1건으로 붕괴됐다.
            #   key에 본문 해시를 포함하면 서로 다른 내용의 턴은 서로 다른 key를 가져
            #   보존되고, "완전히 동일한 내용"의 턴만 같은 key로 묶여 정상적으로
            #   중복 제거(dedup)된다. (key는 조회/조인에 쓰이지 않으므로 형식 변경 안전)
            key = f"turn:{session_id}:{self._content_hash(stored_content)}"

            entry = MemoryEntry(
                memory_type=MemoryType.EPISODIC,
                content=stored_content,
                key=key,
                tags=["conversation", session_id],
                importance=importance,
                metadata={"session_id": session_id, "role": role},
            )

            # 임베딩 생성 (ModelProvider가 있을 때만) — 나중에 벡터 검색이 되도록.
            #   본문 앞 500자만 임베딩한다(임베딩 모델 입력 길이·비용 절감).
            #   MemoryEntry는 불변에 가깝게 다루므로 model_copy로 embedding을 채운 새
            #   객체를 만들어 교체한다(원본 수정이 아니라 복사본 갱신).
            if self._model_provider is not None:
                try:
                    embeddings = await self._model_provider.embed([content[:500]])
                    if embeddings and len(embeddings) > 0:
                        entry = entry.model_copy(update={"embedding": embeddings[0]})
                except Exception as e:
                    logger.warning("임베딩 생성 실패: %s", e)

            # 승격 판단 → 기준 통과 시에만 장기 메모리에 저장한다.
            if self._importance_assessor.should_promote(entry):
                await self._long_term.add(entry)
                logger.debug(
                    "장기 메모리 승격: id=%s, importance=%.2f",
                    entry.id,
                    entry.importance,
                )

        # 3. 도구 실행 결과 중 중요한 것도 장기 저장한다(절차적 지식 축적).
        if tool_results:
            for result_text in tool_results:
                # 비었거나 너무 짧은(30자 미만) 결과는 저장 가치가 낮아 건너뛴다.
                if not result_text or len(result_text) < 30:
                    continue

                # 도구 결과는 PROCEDURAL(절차 기억) 타입으로 중요도를 평가한다.
                importance = self._importance_assessor.assess(result_text, MemoryType.PROCEDURAL)
                # 여기서는 승격 헬퍼 대신 임계값 0.6을 직접 써서 명시적으로 걸러낸다.
                if importance > 0.6:
                    stored_result = result_text[:2000]
                    # turn 키와 동일한 이유로 content 해시를 붙인다:
                    # 서로 다른 도구 결과가 consolidate로 붕괴되는 것을 막고,
                    # 완전히 동일한 결과만 중복 제거되게 한다.
                    key = f"tool_result:{session_id}:{self._content_hash(stored_result)}"
                    entry = MemoryEntry(
                        memory_type=MemoryType.PROCEDURAL,
                        content=stored_result,
                        key=key,
                        tags=["tool_result", session_id],
                        importance=importance,
                        metadata={"session_id": session_id},
                    )
                    await self._long_term.add(entry)

    # ─── 검색 ───

    async def search_relevant(self, query: str, top_k: int = 10) -> list[MemoryEntry]:
        """
        질문(query)과 관련된 메모리를 검색한다.

        [on_turn_start와 무엇이 다른가]
          on_turn_start는 턴 자동 파이프라인 내부에서 쓰는 검색이고, 이 메서드는
          MemoryReadTool 등 도구/외부 모듈이 "명시적으로" 기억을 조회할 때 쓴다.
          동작 원리(벡터 우선 + 텍스트 보충 + ID 중복 제거)는 사실상 동일하다.

        Args:
            query: 검색 쿼리 텍스트
            top_k: 최대 반환 건수(기본 10)

        Returns:
            관련 MemoryEntry 목록 (최대 top_k건)
        """
        results: list[MemoryEntry] = []
        # 벡터/텍스트 결과 사이의 중복을 걸러내기 위한 이미-담긴 ID 집합.
        existing_ids: set[str] = set()

        # 벡터 검색 시도 (임베딩 가능할 때만)
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

        # 남은 자리(top_k - 지금까지 모은 건수)만큼 텍스트 검색으로 보충한다.
        remaining = top_k - len(results)
        if remaining > 0:
            text_results = await self._long_term.search_by_text(query=query, top_k=remaining)
            for entry in text_results:
                if entry.id not in existing_ids:
                    results.append(entry)
                    existing_ids.add(entry.id)

        # 코드 RAG 청크는 '기억 조회' 결과가 아니다 — 걸러낸다.
        # (MemoryRead 도구가 소스코드 조각을 돌려주면 사용자에게 잡음이다.
        #  코드 검색은 SymbolSearch/RAG 리트리버가 담당한다.)
        return [e for e in results if not is_rag_chunk(e)][:top_k]

    # ─── 직접 추가 ───

    async def add_semantic(
        self,
        key: str,
        value: str,
        tags: list[str] | None = None,
    ) -> str:
        """
        시맨틱 메모리(사실·지식)를 직접 추가한다.

        [언제 쓰나]
          on_turn_end의 자동 승격과 달리, MemoryWriteTool이나 외부 모듈이 "이건
          확실히 기억해 둬" 하고 명시적으로 지식/사실을 저장할 때 쓴다. 중요도
          자동 평가를 거치되, 최소 0.5를 보장해 명시 저장분이 쉽게 사라지지 않게 한다.

        Args:
            key: 메모리 키 (예: "project_architecture", "user_preference_theme")
            value: 저장할 텍스트 내용
            tags: 분류 태그 목록(없으면 빈 목록)

        Returns:
            저장된 메모리 ID
        """
        # 자동 평가 점수를 얻되, 명시 저장이므로 아래에서 하한 0.5를 적용한다.
        importance = self._importance_assessor.assess(value, MemoryType.SEMANTIC)

        entry = MemoryEntry(
            memory_type=MemoryType.SEMANTIC,
            content=value,
            key=key,
            tags=tags or [],
            importance=max(importance, 0.5),  # SEMANTIC은 최소 중요도 0.5 보장
        )

        # 임베딩 생성 (가능할 때만) — 앞 500자만 임베딩해 벡터 검색 대상이 되게 한다.
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

        사용자의 긍정/부정 피드백을 FEEDBACK 타입으로 저장해, 향후 응답 품질
        개선(예: 학습 데이터, 프롬프트 튜닝 참고)에 활용한다. 시맨틱과 마찬가지로
        중요도 하한 0.5를 적용해 피드백이 쉽게 소멸하지 않게 한다.

        Args:
            content: 피드백 내용
            tags: 분류 태그 목록(없으면 기본 ["feedback"])

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

        사용자의 선호·스타일·설정을 장기 저장한다. USER_PROFILE 타입은 시간에 따른
        중요도 감쇠가 매우 느려(반감기 약 365일) 오래 유지된다. 그래서 고정 중요도
        0.8로 넉넉히 부여한다. 키에는 "user_profile:" 접두사를 붙여 네임스페이스를 나눈다.

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
            importance=0.8,  # 사용자 프로필은 오래 유지해야 하므로 높은 중요도 고정
        )

        memory_id = await self._long_term.add(entry)
        logger.info("사용자 프로필 추가: id=%s, key='%s'", memory_id, key)
        return memory_id

    # ─── 도구 결과 캐시 ───

    async def get_cached_tool_result(self, tool_name: str, input_data: dict) -> str | None:
        """
        도구 결과 캐시를 조회한다.

        같은 도구를 같은 입력으로 다시 호출할 때, 이전 결과를 재사용해 불필요한
        재실행(느린 검색·외부 조회 등)을 피하기 위한 캐시 조회다. 입력 dict를
        해시로 바꿔 캐시 키의 일부로 사용한다.

        Args:
            tool_name: 도구 이름
            input_data: 도구 입력 데이터(해시하여 키로 사용)

        Returns:
            캐시된 결과 문자열, 없으면 None
        """
        input_hash = self._hash_input(input_data)
        return await self._short_term.get_tool_result_cache(tool_name, input_hash)

    async def cache_tool_result(
        self, tool_name: str, input_data: dict, result: str, ttl: int = 1800
    ) -> None:
        """
        도구 결과를 캐시에 저장한다.

        get_cached_tool_result와 짝을 이룬다. 동일한 입력 해시를 키로 써서 저장하며,
        ttl(초) 후 자동 만료된다(기본 1800초=30분). 만료를 두는 이유는 오래된
        결과가 무한정 재사용되어 최신성이 깨지는 것을 막기 위함이다.

        Args:
            tool_name: 도구 이름
            input_data: 도구 입력 데이터(해시하여 키로 사용)
            result: 캐시할 결과 문자열
            ttl: 만료 시간(초, 기본 1800)
        """
        input_hash = self._hash_input(input_data)
        await self._short_term.cache_tool_result(tool_name, input_hash, result, ttl)

    # ─── 내부 유틸리티 ───

    @staticmethod
    def _serialize_message(message: Message) -> dict:
        """Message를 직렬화 가능한 dict로 변환한다.

        왜 content를 평문(text_content)으로 통일하는가:
          model_dump(mode="json")은 user의 content는 평문 str로, assistant의
          content는 ContentBlock 리스트로 직렬화한다. 이 "형식 비대칭" 때문에
          복원 측(web/app.py)에서 Message.assistant(content)에 리스트가 들어가
          ValidationError가 발생했고, 그 예외가 복원 루프를 중단시켜 대화 이력
          일부(assistant 응답)가 유실됐다.
          단기 대화 이력 복원에는 텍스트만 필요하므로, 저장 단계에서 user/assistant
          모두 평문으로 맞춰 저장↔복원 계약을 일치시킨다.
        """
        try:
            # role 등 다른 메타데이터는 그대로 보존하고 content만 평문으로 덮어쓴다.
            data = message.model_dump(mode="json")
            data["content"] = message.text_content
            return data
        except Exception:
            # model_dump가 실패해도 대화 이력 저장이 통째로 깨지지 않도록,
            # role과 평문 content만 담은 최소 형태로 폴백 직렬화한다.
            return {
                "role": str(message.role),
                "content": message.text_content,
            }

    @staticmethod
    def _hash_input(input_data: dict) -> str:
        """도구 입력 데이터를 안정적인 16자 해시 문자열로 변환한다(캐시 키용).

        sort_keys=True로 키 순서를 고정해, 논리적으로 같은 입력이면 dict 순서가
        달라도 항상 같은 해시가 나오게 한다. ensure_ascii=False로 한글 등도
        원문 그대로 직렬화한 뒤 sha256 앞 16자만 사용한다.
        """
        import json

        serialized = json.dumps(input_data, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def _content_hash(content: str) -> str:
        """
        메모리 본문의 내용 해시를 생성한다 (자동 생성 key의 고유성 확보용).

        왜 필요한가:
          turn/tool_result 메모리의 key에 이 해시를 붙여, 서로 다른 내용의 엔트리가
          consolidate()에서 같은 그룹으로 묶여 삭제되는 것을 방지한다. 반대로 완전히
          동일한 내용은 같은 해시 → 같은 key를 가져 정상적으로 중복 제거된다.
          16자 hex(64비트)면 세션 단위 턴 수 규모에서 충돌 확률이 사실상 0이다.
        """
        return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]

    # ─── 프로퍼티 ───

    @property
    def short_term(self) -> ShortTermMemory:
        """내부 단기 메모리 인스턴스를 읽기 전용으로 노출한다.

        매니저를 거치지 않고 저수준 단기 저장소에 직접 접근해야 하는 특수
        상황(진단·마이그레이션 등)을 위한 접근자다. 일반 흐름에서는 매니저의
        고수준 메서드 사용을 권장한다.
        """
        return self._short_term

    @property
    def long_term(self) -> LongTermMemory:
        """내부 장기 메모리 인스턴스를 읽기 전용으로 노출한다.

        short_term 프로퍼티와 같은 취지의 저수준 접근자다.
        """
        return self._long_term
