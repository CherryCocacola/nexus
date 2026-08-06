"""
core/memory/ 단위 테스트.

메모리 시스템의 6개 모듈을 검증한다:
  - types: MemoryType, MemoryEntry, DECAY_HALF_LIFE
  - short_term: ShortTermMemory (인메모리 폴백)
  - long_term: LongTermMemory (인메모리 폴백)
  - importance: ImportanceAssessor
  - decay: MemoryDecayManager
  - manager: MemoryManager

외부 서비스(Redis, PostgreSQL)는 사용하지 않고 인메모리 폴백으로 테스트한다.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from core.memory.decay import MemoryDecayManager
from core.memory.importance import ImportanceAssessor
from core.memory.long_term import LongTermMemory
from core.memory.manager import MemoryManager
from core.memory.short_term import ShortTermMemory
from core.memory.types import DECAY_HALF_LIFE, MemoryEntry, MemorySearchResult, MemoryType
from core.message import Message


# ─────────────────────────────────────────────
# MemoryType & MemoryEntry 테스트
# ─────────────────────────────────────────────
class TestMemoryTypes:
    """메모리 타입 및 엔트리 테스트."""

    def test_memory_type_values(self):
        """MemoryType 열거형 값이 올바른지 확인한다."""
        assert MemoryType.EPISODIC == "episodic"
        assert MemoryType.SEMANTIC == "semantic"
        assert MemoryType.PROCEDURAL == "procedural"
        assert MemoryType.USER_PROFILE == "user_profile"
        assert MemoryType.FEEDBACK == "feedback"

    def test_memory_entry_defaults(self):
        """MemoryEntry 기본값이 올바르게 설정되는지 확인한다."""
        entry = MemoryEntry(
            memory_type=MemoryType.EPISODIC,
            content="test content",
        )
        assert len(entry.id) == 12
        assert entry.memory_type == MemoryType.EPISODIC
        assert entry.content == "test content"
        assert entry.key == ""
        assert entry.tags == []
        assert entry.importance == 0.5
        assert entry.access_count == 0
        assert entry.embedding is None
        assert entry.metadata == {}

    def test_memory_entry_custom_values(self):
        """MemoryEntry에 커스텀 값을 설정할 수 있는지 확인한다."""
        entry = MemoryEntry(
            id="custom_id_01",
            memory_type=MemoryType.SEMANTIC,
            content="architecture decision",
            key="arch_decision_01",
            tags=["architecture", "decision"],
            importance=0.9,
            access_count=5,
            metadata={"source": "conversation"},
        )
        assert entry.id == "custom_id_01"
        assert entry.importance == 0.9
        assert "architecture" in entry.tags

    def test_memory_entry_importance_bounds(self):
        """importance가 0.0~1.0 범위를 벗어나면 검증 에러가 발생하는지 확인한다."""
        with pytest.raises(ValueError):
            MemoryEntry(
                memory_type=MemoryType.EPISODIC,
                content="test",
                importance=1.5,
            )
        with pytest.raises(ValueError):
            MemoryEntry(
                memory_type=MemoryType.EPISODIC,
                content="test",
                importance=-0.1,
            )

    def test_decay_half_life_all_types(self):
        """모든 MemoryType에 대한 반감기가 정의되어 있는지 확인한다."""
        for mt in MemoryType:
            assert mt in DECAY_HALF_LIFE
            assert DECAY_HALF_LIFE[mt] > 0

    def test_memory_search_result(self):
        """MemorySearchResult 생성을 확인한다."""
        entry = MemoryEntry(memory_type=MemoryType.EPISODIC, content="test")
        result = MemorySearchResult(entry=entry, score=0.95)
        assert result.score == 0.95
        assert result.entry.content == "test"


# ─────────────────────────────────────────────
# ShortTermMemory 테스트 (인메모리 폴백)
# ─────────────────────────────────────────────
class TestShortTermMemory:
    """단기 메모리 인메모리 폴백 테스트."""

    @pytest.fixture
    def stm(self):
        """인메모리 ShortTermMemory 인스턴스를 생성한다."""
        return ShortTermMemory(redis_client=None)

    @pytest.mark.asyncio
    async def test_set_and_get(self, stm):
        """키-값 저장 및 조회가 동작하는지 확인한다."""
        await stm.set("key1", "value1")
        result = await stm.get("key1")
        assert result == "value1"

    @pytest.mark.asyncio
    async def test_get_nonexistent_key(self, stm):
        """존재하지 않는 키 조회 시 None을 반환하는지 확인한다."""
        result = await stm.get("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_delete_key(self, stm):
        """키 삭제가 동작하는지 확인한다."""
        await stm.set("key1", "value1")
        await stm.delete("key1")
        result = await stm.get("key1")
        assert result is None

    @pytest.mark.asyncio
    async def test_ttl_expiration(self, stm):
        """TTL이 0이면 즉시 만료되는지 확인한다 (폴백 모드에서 음수 TTL)."""
        # TTL=0은 만료 없음, 직접 expires_at을 과거로 설정
        await stm.set("key1", "value1", ttl=1)
        # 인메모리 저장소의 expires_at을 과거로 강제 설정
        stm._store["key1"]["expires_at"] = 0.0
        result = await stm.get("key1")
        assert result is None

    @pytest.mark.asyncio
    async def test_conversation_context_save_and_load(self, stm):
        """대화 컨텍스트 저장/복원이 동작하는지 확인한다."""
        messages = [
            {"role": "user", "content": "안녕하세요"},
            {"role": "assistant", "content": "안녕하세요! 무엇을 도와드릴까요?"},
        ]
        await stm.save_conversation_context("session-001", messages)
        result = await stm.get_conversation_context("session-001")
        assert len(result) == 2
        assert result[0]["role"] == "user"

    @pytest.mark.asyncio
    async def test_conversation_context_empty_session(self, stm):
        """존재하지 않는 세션의 컨텍스트 조회 시 빈 리스트를 반환하는지 확인한다."""
        result = await stm.get_conversation_context("nonexistent")
        assert result == []

    @pytest.mark.asyncio
    async def test_tool_result_cache(self, stm):
        """도구 결과 캐시 저장/조회가 동작하는지 확인한다."""
        await stm.cache_tool_result("Read", "abc123", "file content here")
        result = await stm.get_tool_result_cache("Read", "abc123")
        assert result == "file content here"

    @pytest.mark.asyncio
    async def test_tool_result_cache_miss(self, stm):
        """캐시 미스 시 None을 반환하는지 확인한다."""
        result = await stm.get_tool_result_cache("Read", "nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_clear_session(self, stm):
        """세션 데이터 정리가 동작하는지 확인한다."""
        await stm.save_conversation_context("session-001", [{"role": "user", "content": "hi"}])
        await stm.clear_session("session-001")
        result = await stm.get_conversation_context("session-001")
        assert result == []

    def test_cleanup_expired(self, stm):
        """만료된 인메모리 항목 정리가 동작하는지 확인한다."""
        # 이미 만료된 항목 추가
        stm._store["expired1"] = {"value": "old", "expires_at": 0.0}
        stm._store["valid1"] = {"value": "new", "expires_at": None}
        count = stm._cleanup_expired()
        assert count == 1
        assert "expired1" not in stm._store
        assert "valid1" in stm._store


# ─────────────────────────────────────────────
# LongTermMemory 테스트 (인메모리 폴백)
# ─────────────────────────────────────────────
class TestLongTermMemory:
    """장기 메모리 인메모리 폴백 테스트."""

    @pytest.fixture
    def ltm(self):
        """인메모리 LongTermMemory 인스턴스를 생성한다."""
        return LongTermMemory(pg_pool=None)

    @pytest.mark.asyncio
    async def test_add_and_get(self, ltm):
        """메모리 추가 및 조회가 동작하는지 확인한다."""
        entry = MemoryEntry(
            memory_type=MemoryType.SEMANTIC,
            content="Python은 프로그래밍 언어이다",
        )
        memory_id = await ltm.add(entry)
        assert memory_id == entry.id

        result = await ltm.get(memory_id)
        assert result is not None
        assert result.content == "Python은 프로그래밍 언어이다"
        # access_count가 1 증가해야 함
        assert result.access_count == 1

    @pytest.mark.asyncio
    async def test_get_nonexistent(self, ltm):
        """존재하지 않는 ID 조회 시 None을 반환하는지 확인한다."""
        result = await ltm.get("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_search_by_text(self, ltm):
        """텍스트 검색이 동작하는지 확인한다."""
        await ltm.add(MemoryEntry(
            memory_type=MemoryType.SEMANTIC,
            content="Python asyncio는 비동기 프로그래밍 라이브러리이다",
            importance=0.8,
        ))
        await ltm.add(MemoryEntry(
            memory_type=MemoryType.SEMANTIC,
            content="Redis는 인메모리 데이터 저장소이다",
            importance=0.7,
        ))

        results = await ltm.search_by_text("asyncio")
        assert len(results) == 1
        assert "asyncio" in results[0].content

    @pytest.mark.asyncio
    async def test_search_by_text_with_type_filter(self, ltm):
        """타입 필터링이 적용된 텍스트 검색을 확인한다."""
        await ltm.add(MemoryEntry(
            memory_type=MemoryType.SEMANTIC,
            content="error handling pattern",
        ))
        await ltm.add(MemoryEntry(
            memory_type=MemoryType.EPISODIC,
            content="error occurred in session",
        ))

        results = await ltm.search_by_text("error", memory_type=MemoryType.SEMANTIC)
        assert len(results) == 1
        assert results[0].memory_type == MemoryType.SEMANTIC

    @pytest.mark.asyncio
    async def test_search_by_vector(self, ltm):
        """벡터 검색이 동작하는지 확인한다."""
        await ltm.add(MemoryEntry(
            memory_type=MemoryType.SEMANTIC,
            content="similar content",
            embedding=[1.0, 0.0, 0.0],
        ))
        await ltm.add(MemoryEntry(
            memory_type=MemoryType.SEMANTIC,
            content="different content",
            embedding=[0.0, 1.0, 0.0],
        ))

        results = await ltm.search_by_vector([0.9, 0.1, 0.0])
        assert len(results) == 2
        # 첫 번째 결과가 더 유사해야 함
        assert results[0].content == "similar content"

    @pytest.mark.asyncio
    async def test_search_by_vector_skips_no_embedding(self, ltm):
        """임베딩이 없는 메모리는 벡터 검색에서 제외되는지 확인한다."""
        await ltm.add(MemoryEntry(
            memory_type=MemoryType.SEMANTIC,
            content="no embedding",
            embedding=None,
        ))
        results = await ltm.search_by_vector([1.0, 0.0, 0.0])
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_update(self, ltm):
        """메모리 업데이트가 동작하는지 확인한다."""
        entry = MemoryEntry(
            memory_type=MemoryType.SEMANTIC,
            content="original",
            importance=0.5,
        )
        await ltm.add(entry)

        success = await ltm.update(entry.id, importance=0.9, tags=["updated"])
        assert success

        result = await ltm.get(entry.id)
        assert result is not None
        assert result.importance == 0.9
        assert "updated" in result.tags

    @pytest.mark.asyncio
    async def test_update_nonexistent(self, ltm):
        """존재하지 않는 메모리 업데이트 시 False를 반환하는지 확인한다."""
        success = await ltm.update("nonexistent", importance=0.9)
        assert not success

    @pytest.mark.asyncio
    async def test_update_invalid_field(self, ltm):
        """허용되지 않은 필드 업데이트 시 False를 반환하는지 확인한다."""
        entry = MemoryEntry(memory_type=MemoryType.SEMANTIC, content="test")
        await ltm.add(entry)
        success = await ltm.update(entry.id, id="new_id")
        assert not success

    @pytest.mark.asyncio
    async def test_delete(self, ltm):
        """메모리 삭제가 동작하는지 확인한다."""
        entry = MemoryEntry(memory_type=MemoryType.SEMANTIC, content="to delete")
        await ltm.add(entry)

        success = await ltm.delete(entry.id)
        assert success

        result = await ltm.get(entry.id)
        assert result is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self, ltm):
        """존재하지 않는 메모리 삭제 시 False를 반환하는지 확인한다."""
        success = await ltm.delete("nonexistent")
        assert not success

    @pytest.mark.asyncio
    async def test_get_by_type(self, ltm):
        """타입별 메모리 조회가 동작하는지 확인한다."""
        await ltm.add(MemoryEntry(memory_type=MemoryType.SEMANTIC, content="fact 1"))
        await ltm.add(MemoryEntry(memory_type=MemoryType.SEMANTIC, content="fact 2"))
        await ltm.add(MemoryEntry(memory_type=MemoryType.EPISODIC, content="event 1"))

        results = await ltm.get_by_type(MemoryType.SEMANTIC)
        assert len(results) == 2
        assert all(e.memory_type == MemoryType.SEMANTIC for e in results)

    @pytest.mark.asyncio
    async def test_get_all(self, ltm):
        """전체 메모리 조회가 동작하는지 확인한다."""
        await ltm.add(MemoryEntry(memory_type=MemoryType.SEMANTIC, content="a"))
        await ltm.add(MemoryEntry(memory_type=MemoryType.EPISODIC, content="b"))

        results = await ltm.get_all()
        assert len(results) == 2

    def test_cosine_similarity_identical(self):
        """동일 벡터의 코사인 유사도가 1.0인지 확인한다."""
        sim = LongTermMemory._cosine_similarity([1.0, 0.0], [1.0, 0.0])
        assert abs(sim - 1.0) < 1e-6

    def test_cosine_similarity_orthogonal(self):
        """직교 벡터의 코사인 유사도가 0.0인지 확인한다."""
        sim = LongTermMemory._cosine_similarity([1.0, 0.0], [0.0, 1.0])
        assert abs(sim) < 1e-6

    def test_cosine_similarity_different_length(self):
        """길이가 다른 벡터의 유사도가 0.0인지 확인한다."""
        sim = LongTermMemory._cosine_similarity([1.0, 0.0], [1.0, 0.0, 0.0])
        assert sim == 0.0


# ─────────────────────────────────────────────
# ensure_schema 테스트 — 운영 정의(2026-06-01)를 멱등 보장
# ─────────────────────────────────────────────
class TestLongTermMemoryEnsureSchema:
    """ensure_schema()가 pg_pool 유무에 따라 올바르게 동작하는지 확인한다."""

    @pytest.mark.asyncio
    async def test_ensure_schema_inmemory_noop(self):
        """pg_pool=None이면 ensure_schema는 아무 작업도 하지 않는다."""
        # 인메모리 폴백 모드 — DDL 실행할 수 없으므로 조용히 no-op
        ltm = LongTermMemory(pg_pool=None)
        # 예외 없이 통과하면 성공 (no-op 검증)
        await ltm.ensure_schema()

    @pytest.mark.asyncio
    async def test_ensure_schema_executes_ddl_with_pool(self):
        """pg_pool이 있으면 vector 확장 + tb_memories 테이블 + 인덱스 5종 DDL을
        모두 실행한다. asyncpg 호출을 mock해서 실행 SQL을 검증한다."""
        from unittest.mock import AsyncMock, MagicMock

        # asyncpg 풀/커넥션 mock — async with self._pg.acquire() as conn 패턴 모사
        executed: list[str] = []

        conn = MagicMock()

        async def fake_execute(sql: str) -> None:
            # 실행된 SQL 본문을 기록해서 어떤 DDL이 흘러갔는지 검증한다.
            executed.append(sql)

        conn.execute = fake_execute

        acquire_cm = MagicMock()
        acquire_cm.__aenter__ = AsyncMock(return_value=conn)
        acquire_cm.__aexit__ = AsyncMock(return_value=False)

        pool = MagicMock()
        pool.acquire = MagicMock(return_value=acquire_cm)

        ltm = LongTermMemory(pg_pool=pool)
        await ltm.ensure_schema()

        # 최소한 다음 DDL이 실행되어야 한다 (순서·갯수까지 검증):
        #   1) CREATE EXTENSION vector
        #   2) CREATE TABLE tb_memories
        #   3) 인덱스 6종 (type/tags/created_at/importance/embedding hnsw/owner)
        #      owner 인덱스는 2026-08-06 추가 — 회상이 metadata->>'owner' 로 좁힌 뒤
        #      거리 계산을 하므로 그 조회를 받쳐 준다.
        assert len(executed) == 8, f"실행된 SQL 수 불일치: {len(executed)} (기대 8)"
        assert "CREATE EXTENSION" in executed[0]
        assert "tb_memories" in executed[1]
        assert "varchar(12) PRIMARY KEY" in executed[1]
        assert "vector(1024)" in executed[1]
        assert "tb_memories_importance_check" in executed[1]
        # 인덱스 — 순서는 보장하나 핵심 키워드 매칭으로 검증
        idx_sqls = "\n".join(executed[2:])
        assert "idx_memories_type" in idx_sqls
        assert "idx_memories_tags" in idx_sqls and "USING gin" in idx_sqls
        assert "idx_memories_created_at" in idx_sqls and "DESC" in idx_sqls
        assert "idx_memories_importance" in idx_sqls
        # hnsw 선택 검증 — 운영 DB와 동일한 인덱스 종류
        assert "idx_memories_embedding" in idx_sqls
        assert "USING hnsw" in idx_sqls
        assert "vector_cosine_ops" in idx_sqls
        # 소유자 인덱스 — 소유자 없는 행(코드 RAG 113만 건)은 제외하는 부분 인덱스여야
        # 인덱스 크기가 불필요하게 커지지 않는다.
        assert "idx_memories_owner" in idx_sqls
        assert "metadata->>'owner'" in idx_sqls
        assert "WHERE metadata ? 'owner'" in idx_sqls


# ─────────────────────────────────────────────
# ImportanceAssessor 테스트
# ─────────────────────────────────────────────
class TestImportanceAssessor:
    """중요도 평가기 테스트."""

    @pytest.fixture
    def assessor(self):
        return ImportanceAssessor()

    def test_high_importance_error_keyword(self, assessor):
        """error 키워드가 포함되면 중요도가 높아지는지 확인한다."""
        score = assessor.assess("Critical error in authentication module", MemoryType.EPISODIC)
        assert score > 0.5

    def test_high_importance_architecture(self, assessor):
        """architecture 키워드가 포함되면 중요도가 높아지는지 확인한다."""
        score = assessor.assess(
            "Architecture decision: use event-driven design", MemoryType.SEMANTIC
        )
        assert score > 0.5

    def test_low_importance_routine(self, assessor):
        """루틴 명령어만 있으면 중요도가 낮은지 확인한다."""
        score = assessor.assess("ls -la, cat file, status check", MemoryType.EPISODIC)
        assert score < 0.4

    def test_medium_importance_config(self, assessor):
        """config 키워드가 포함되면 중간 중요도인지 확인한다."""
        score = assessor.assess("Updated configuration settings", MemoryType.SEMANTIC)
        assert 0.3 <= score <= 0.7

    def test_importance_range(self, assessor):
        """중요도가 항상 0.0~1.0 범위인지 확인한다."""
        # 많은 높은 키워드
        score = assessor.assess(
            "error exception bug fix security vulnerability critical",
            MemoryType.EPISODIC,
        )
        assert 0.0 <= score <= 1.0

        # 많은 낮은 키워드
        score = assessor.assess(
            "ls cat head tail pwd cd echo status",
            MemoryType.EPISODIC,
        )
        assert 0.0 <= score <= 1.0

    def test_content_length_bonus(self, assessor):
        """긴 콘텐츠에 보너스가 적용되는지 확인한다."""
        short_score = assessor.assess("test", MemoryType.EPISODIC)
        long_score = assessor.assess("a" * 600, MemoryType.EPISODIC)
        assert long_score >= short_score

    def test_type_bias_user_profile(self, assessor):
        """USER_PROFILE 타입에 보정이 적용되는지 확인한다."""
        ep_score = assessor.assess("preference", MemoryType.EPISODIC)
        up_score = assessor.assess("preference", MemoryType.USER_PROFILE)
        assert up_score > ep_score

    def test_should_promote_high_importance(self, assessor):
        """importance > 0.6이면 승격되는지 확인한다."""
        entry = MemoryEntry(
            memory_type=MemoryType.EPISODIC,
            content="important",
            importance=0.7,
        )
        assert assessor.should_promote(entry)

    def test_should_promote_low_importance(self, assessor):
        """importance <= 0.6이면 승격되지 않는지 확인한다."""
        entry = MemoryEntry(
            memory_type=MemoryType.EPISODIC,
            content="routine",
            importance=0.3,
            access_count=0,
        )
        assert not assessor.should_promote(entry)

    def test_should_promote_high_access_count(self, assessor):
        """access_count >= 3이면 승격되는지 확인한다."""
        entry = MemoryEntry(
            memory_type=MemoryType.EPISODIC,
            content="routine",
            importance=0.3,
            access_count=3,
        )
        assert assessor.should_promote(entry)

    def test_should_promote_user_profile_always(self, assessor):
        """USER_PROFILE 타입은 항상 승격되는지 확인한다."""
        entry = MemoryEntry(
            memory_type=MemoryType.USER_PROFILE,
            content="user prefers dark mode",
            importance=0.2,
        )
        assert assessor.should_promote(entry)


# ─────────────────────────────────────────────
# MemoryDecayManager 테스트
# ─────────────────────────────────────────────
class TestMemoryDecayManager:
    """메모리 감쇠 매니저 테스트."""

    @pytest.fixture
    def decay_mgr(self):
        return MemoryDecayManager()

    def test_no_decay_recent_access(self, decay_mgr):
        """최근 접근된 메모리는 감쇠가 없는지 확인한다."""
        entry = MemoryEntry(
            memory_type=MemoryType.SEMANTIC,
            content="recent",
            importance=0.8,
            last_accessed=datetime.now(UTC),
        )
        effective = decay_mgr.calculate_decay(entry)
        assert abs(effective - 0.8) < 0.01

    def test_decay_after_half_life(self, decay_mgr):
        """반감기 경과 후 중요도가 절반으로 감쇠되는지 확인한다."""
        half_life = DECAY_HALF_LIFE[MemoryType.EPISODIC]  # 7일
        entry = MemoryEntry(
            memory_type=MemoryType.EPISODIC,
            content="old event",
            importance=0.8,
            last_accessed=datetime.now(UTC) - timedelta(days=half_life),
        )
        effective = decay_mgr.calculate_decay(entry)
        # 반감기 후 약 0.4 (0.8 * 0.5)
        assert 0.35 <= effective <= 0.45

    def test_decay_access_boost(self, decay_mgr):
        """접근 횟수가 높으면 감쇠가 느려지는지 확인한다."""
        past = datetime.now(UTC) - timedelta(days=7)

        # 접근 횟수 0
        entry_low = MemoryEntry(
            memory_type=MemoryType.EPISODIC,
            content="test",
            importance=0.8,
            access_count=0,
            last_accessed=past,
        )
        # 접근 횟수 10
        entry_high = MemoryEntry(
            memory_type=MemoryType.EPISODIC,
            content="test",
            importance=0.8,
            access_count=10,
            last_accessed=past,
        )

        effective_low = decay_mgr.calculate_decay(entry_low)
        effective_high = decay_mgr.calculate_decay(entry_high)
        # 접근 횟수가 높은 쪽이 더 높은 유효 중요도를 가져야 함
        assert effective_high > effective_low

    def test_decay_user_profile_slow(self, decay_mgr):
        """USER_PROFILE 타입은 감쇠가 매우 느린지 확인한다."""
        past_30d = datetime.now(UTC) - timedelta(days=30)

        entry = MemoryEntry(
            memory_type=MemoryType.USER_PROFILE,
            content="user pref",
            importance=0.8,
            last_accessed=past_30d,
        )
        effective = decay_mgr.calculate_decay(entry)
        # USER_PROFILE 반감기 365일 → 30일 후에도 거의 감쇠 없음
        assert effective > 0.7

    @pytest.mark.asyncio
    async def test_run_decay_cycle(self, decay_mgr):
        """감쇠 사이클이 만료된 메모리를 삭제하는지 확인한다."""
        ltm = LongTermMemory(pg_pool=None)

        # 매우 오래된 메모리 (EPISODIC, 100일 전, 중요도 낮음)
        old_entry = MemoryEntry(
            memory_type=MemoryType.EPISODIC,
            content="very old",
            importance=0.1,
            last_accessed=datetime.now(UTC) - timedelta(days=100),
        )
        await ltm.add(old_entry)

        # 최근 메모리
        recent_entry = MemoryEntry(
            memory_type=MemoryType.SEMANTIC,
            content="recent fact",
            importance=0.8,
            last_accessed=datetime.now(UTC),
        )
        await ltm.add(recent_entry)

        stats = await decay_mgr.run_decay_cycle(ltm)
        assert stats["total_checked"] == 2
        assert stats["deleted"] >= 1  # 오래된 메모리 삭제됨

        # 오래된 메모리가 삭제되었는지 확인
        result = await ltm.get(old_entry.id)
        assert result is None

        # 최근 메모리는 유지되는지 확인
        result = await ltm.get(recent_entry.id)
        assert result is not None

    @pytest.mark.asyncio
    async def test_consolidate_same_key(self, decay_mgr):
        """같은 key를 가진 메모리들이 통합되는지 확인한다."""
        ltm = LongTermMemory(pg_pool=None)

        # 같은 key를 가진 3개 메모리
        for i, imp in enumerate([0.3, 0.9, 0.5]):
            await ltm.add(MemoryEntry(
                memory_type=MemoryType.SEMANTIC,
                content=f"knowledge v{i}",
                key="shared_key",
                tags=[f"v{i}"],
                importance=imp,
            ))

        stats = await decay_mgr.consolidate(ltm)
        assert stats["groups_found"] == 1
        assert stats["entries_merged"] == 2  # 3개 중 2개 삭제

        # 가장 중요한 것(imp=0.9)만 남아야 함
        remaining = await ltm.get_by_type(MemoryType.SEMANTIC)
        assert len(remaining) == 1
        assert remaining[0].importance == 0.9

    @pytest.mark.asyncio
    async def test_consolidate_different_keys(self, decay_mgr):
        """다른 key를 가진 메모리는 통합되지 않는지 확인한다."""
        ltm = LongTermMemory(pg_pool=None)

        await ltm.add(MemoryEntry(
            memory_type=MemoryType.SEMANTIC, content="a", key="key_a",
        ))
        await ltm.add(MemoryEntry(
            memory_type=MemoryType.SEMANTIC, content="b", key="key_b",
        ))

        stats = await decay_mgr.consolidate(ltm)
        assert stats["groups_found"] == 0
        assert stats["entries_merged"] == 0

    # ─────────────────────────────────────────────
    # 감사 Critical #6 회귀 방지 — 복리 감쇠(compounding decay) 버그
    # ─────────────────────────────────────────────
    # 배경(버그):
    #   과거 run_decay_cycle은 유효 중요도를 importance 필드에 되썼지만
    #   last_accessed는 갱신하지 않았다. 그러면 다음 사이클이 "이미 감쇠된
    #   importance"를 base로 "여전히 옛 경과일"로 또 감쇠시켜, 지수 감쇠가 복리로
    #   중복 적용됐다(며칠 만에 임계치 붕괴 → 조기 삭제).
    # 수정: importance는 불변 base로 두고, 사이클은 삭제만 한다.
    @pytest.mark.asyncio
    async def test_run_decay_cycle_idempotent(self, decay_mgr):
        """같은 상태로 감쇠 사이클을 여러 번 실행해도 결과가 1회 실행과 동일한지 확인한다.

        핵심 불변식: base importance가 사이클로 변하지 않아야 한다(복리 없음).
        """
        ltm = LongTermMemory(pg_pool=None)

        # 3일 지난 EPISODIC 메모리(반감기 7일) — 유효 중요도는 임계치보다 훨씬 높다.
        # 0.8 * 2^(-3/7) ≈ 0.59 >> 0.05 이므로 절대 삭제되면 안 된다.
        entry = MemoryEntry(
            memory_type=MemoryType.EPISODIC,
            content="important architecture decision",
            importance=0.8,
            last_accessed=datetime.now(UTC) - timedelta(days=3),
        )
        await ltm.add(entry)

        # 사이클을 3회 반복 실행한다.
        s1 = await decay_mgr.run_decay_cycle(ltm)
        s2 = await decay_mgr.run_decay_cycle(ltm)
        s3 = await decay_mgr.run_decay_cycle(ltm)

        # 매 실행 통계가 동일해야 한다(멱등).
        assert s1 == s2 == s3
        # 삭제 전용 정책: 아무 것도 삭제되지 않고, updated는 항상 0이다.
        assert s1["deleted"] == 0
        assert s1["updated"] == 0
        assert s1["total_checked"] == 1

        # base importance가 원본 그대로여야 한다(감쇠값이 되쓰이지 않음 = 복리 없음).
        # get()은 access_count/last_accessed를 변경하므로 get_all로 부작용 없이 읽는다.
        stored = {e.id: e for e in await ltm.get_all()}
        assert entry.id in stored
        assert stored[entry.id].importance == 0.8

    @pytest.mark.asyncio
    async def test_run_decay_cycle_fresh_memory_survives_repeated_cycles(self, decay_mgr):
        """신선한 EPISODIC 메모리가 사이클 반복만으로 조기 삭제되지 않는지 확인한다.

        과거 버그에선 복리 감쇠 때문에 며칠 안 된 기억이 여러 사이클 만에 임계치
        아래로 떨어져 삭제됐다. 수정 후에는 시간이 실제로 흐르지 않는 한(같은 상태
        반복 실행) 삭제되지 않아야 한다.
        """
        ltm = LongTermMemory(pg_pool=None)

        # 2일 지난 EPISODIC 메모리. 유효 중요도 = 0.6 * 2^(-2/7) ≈ 0.49 >> 0.05.
        entry = MemoryEntry(
            memory_type=MemoryType.EPISODIC,
            content="a recent conversation turn worth keeping",
            importance=0.6,
            last_accessed=datetime.now(UTC) - timedelta(days=2),
        )
        await ltm.add(entry)

        # 사이클을 5회 반복 실행한다(옛 버그라면 여기서 importance가 복리로 붕괴).
        for _ in range(5):
            await decay_mgr.run_decay_cycle(ltm)

        # 여전히 존재하고 base importance가 그대로여야 한다.
        stored = {e.id: e for e in await ltm.get_all()}
        assert entry.id in stored, "신선한 메모리가 반복 사이클로 부당하게 삭제됨(복리 감쇠 회귀)"
        assert stored[entry.id].importance == 0.6


# ─────────────────────────────────────────────
# MemoryManager 테스트
# ─────────────────────────────────────────────
class TestMemoryManager:
    """메모리 매니저 통합 테스트."""

    @pytest.fixture
    def manager(self):
        """인메모리 폴백 기반 MemoryManager를 생성한다."""
        stm = ShortTermMemory(redis_client=None)
        ltm = LongTermMemory(pg_pool=None)
        return MemoryManager(short_term=stm, long_term=ltm, model_provider=None)

    @pytest.mark.asyncio
    async def test_add_semantic(self, manager):
        """시맨틱 메모리 추가가 동작하는지 확인한다."""
        mem_id = await manager.add_semantic(
            key="test_key",
            value="Python is a programming language",
            tags=["programming"],
        )
        assert mem_id is not None

        # 검색으로 확인 (텍스트 검색은 부분 문자열 매칭)
        results = await manager.search_relevant("Python is a programming")
        assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_add_feedback(self, manager):
        """피드백 메모리 추가가 동작하는지 확인한다."""
        mem_id = await manager.add_feedback(
            content="User prefers concise answers",
            tags=["style"],
        )
        assert mem_id is not None

    @pytest.mark.asyncio
    async def test_add_user_profile(self, manager):
        """사용자 프로필 추가가 동작하는지 확인한다."""
        mem_id = await manager.add_user_profile(
            key="language",
            value="Korean",
        )
        assert mem_id is not None

        # 타입별 조회로 확인
        profiles = await manager.long_term.get_by_type(MemoryType.USER_PROFILE)
        assert len(profiles) == 1
        assert profiles[0].content == "Korean"

    @pytest.mark.asyncio
    async def test_search_relevant_text_only(self, manager):
        """ModelProvider 없이 텍스트 검색이 동작하는지 확인한다."""
        await manager.add_semantic("arch", "4-Tier AsyncGenerator architecture pattern")
        await manager.add_semantic("tool", "BaseTool ABC implementation")

        results = await manager.search_relevant("AsyncGenerator")
        assert len(results) >= 1
        assert "AsyncGenerator" in results[0].content

    @pytest.mark.asyncio
    async def test_on_turn_start_empty(self, manager):
        """메모리가 비어있을 때 on_turn_start가 빈 리스트를 반환하는지 확인한다."""
        results = await manager.on_turn_start("session-001", "hello world")
        assert results == []

    @pytest.mark.asyncio
    async def test_on_turn_start_with_memories(self, manager):
        """관련 메모리가 있으면 on_turn_start가 결과를 반환하는지 확인한다."""
        await manager.add_semantic("test", "asyncio event loop implementation details")
        results = await manager.on_turn_start("session-001", "asyncio event loop")
        assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_on_turn_end_stores_context(self, manager):
        """on_turn_end가 대화 컨텍스트를 저장하는지 확인한다."""
        messages = [
            Message.user("architecture decision about error handling"),
            Message.assistant("We should use the error wrapping pattern for all tool calls"),
        ]
        await manager.on_turn_end("session-001", messages)

        # 단기 메모리에 컨텍스트가 저장되었는지 확인
        ctx = await manager.short_term.get_conversation_context("session-001")
        assert len(ctx) == 2

    @pytest.mark.asyncio
    async def test_on_turn_end_promotes_important(self, manager):
        """on_turn_end가 중요한 메시지를 장기 메모리로 승격하는지 확인한다."""
        messages = [
            Message.user("what happened?"),
            Message.assistant(
                "Critical error in authentication: the security vulnerability "
                "was caused by a missing permission check in the auth middleware. "
                "This is an architecture decision to fix."
            ),
        ]
        await manager.on_turn_end("session-001", messages)

        # 장기 메모리에 승격되었는지 확인
        all_memories = await manager.long_term.get_all()
        assert len(all_memories) >= 1

    # ─────────────────────────────────────────────
    # 감사 Critical #7 회귀 방지 — consolidate가 세션 턴 기억을 파괴하던 버그
    # ─────────────────────────────────────────────
    # 배경(버그):
    #   과거 on_turn_end는 EPISODIC 엔트리에 key=f"turn:{session_id}"를 부여해
    #   한 세션의 모든 assistant 턴이 동일 key를 가졌다. consolidate()는 같은 key를
    #   한 그룹으로 묶어 대표 1건만 남기므로, 한 세션의 서로 다른 턴 기억 전체가
    #   1건으로 붕괴됐다.
    # 수정: key에 content 해시를 붙여 서로 다른 내용의 턴은 서로 다른 key를 갖게 한다.
    #   완전히 동일한 내용만 같은 key로 묶여 정상 dedup된다.
    @pytest.mark.asyncio
    async def test_on_turn_end_distinct_turns_survive_consolidate(self, manager):
        """서로 다른 내용의 세션 턴들이 consolidate 후에도 모두 보존되는지 확인한다."""
        # 각기 다른 내용의 assistant 턴 3건 — 모두 승격되도록 고중요도 키워드 포함.
        messages = [
            Message.user("질문"),
            Message.assistant(
                "Critical architecture decision: we adopted the 4-Tier AsyncGenerator "
                "chain to avoid the deprecated streaming bug in the old design."
            ),
            Message.assistant(
                "Security hotfix: resolved the authentication permission vulnerability "
                "and the critical auth bug in the layer 2 middleware."
            ),
            Message.assistant(
                "Migration decision documented: the breaking change requires a careful "
                "tradeoff between the two critical designs before deployment."
            ),
        ]
        await manager.on_turn_end("session-777", messages)

        # 3건이 각기 다른 key로 장기 메모리에 승격되어야 한다.
        before = await manager.long_term.get_by_type(MemoryType.EPISODIC)
        assert len(before) == 3
        assert len({e.key for e in before}) == 3, "서로 다른 턴은 서로 다른 key를 가져야 함"

        # consolidate 실행 — 서로 다른 내용이므로 아무 것도 병합되지 않아야 한다.
        decay_mgr = MemoryDecayManager()
        stats = await decay_mgr.consolidate(manager.long_term)
        assert stats["entries_merged"] == 0

        # 3건 모두 보존되어야 한다(세션 턴 붕괴 회귀 방지).
        after = await manager.long_term.get_by_type(MemoryType.EPISODIC)
        assert len(after) == 3

    @pytest.mark.asyncio
    async def test_on_turn_end_true_duplicate_turns_are_deduped(self, manager):
        """완전히 동일한 내용의 턴은 consolidate로 정상 중복 제거되는지 확인한다.

        dedup 기능 자체는 유지되어야 한다 — 서로 다른 턴만 보존, 진짜 중복은 병합.
        """
        dup_text = (
            "Critical security decision about the authentication architecture "
            "and the permission bug fix in the middleware."
        )
        # 동일 세션에서 완전히 동일한 assistant 응답이 두 번 저장되는 상황.
        await manager.on_turn_end("session-888", [Message.assistant(dup_text)])
        await manager.on_turn_end("session-888", [Message.assistant(dup_text)])

        # 동일 내용 → 동일 key → 2건 저장됨.
        before = await manager.long_term.get_by_type(MemoryType.EPISODIC)
        assert len(before) == 2
        assert before[0].key == before[1].key, "동일 내용은 동일 key여야 dedup 대상이 됨"

        # consolidate 실행 — 진짜 중복이므로 1건으로 병합되어야 한다.
        decay_mgr = MemoryDecayManager()
        stats = await decay_mgr.consolidate(manager.long_term)
        assert stats["groups_found"] == 1
        assert stats["entries_merged"] == 1

        after = await manager.long_term.get_by_type(MemoryType.EPISODIC)
        assert len(after) == 1

    @pytest.mark.asyncio
    async def test_tool_result_cache(self, manager):
        """도구 결과 캐시가 동작하는지 확인한다."""
        input_data = {"path": "/tmp/test.py"}
        await manager.cache_tool_result("Read", input_data, "file content")

        result = await manager.get_cached_tool_result("Read", input_data)
        assert result == "file content"

    @pytest.mark.asyncio
    async def test_tool_result_cache_miss(self, manager):
        """캐시 미스 시 None을 반환하는지 확인한다."""
        result = await manager.get_cached_tool_result("Read", {"path": "/nonexistent"})
        assert result is None

    def test_hash_input_deterministic(self):
        """같은 입력에 대해 같은 해시가 생성되는지 확인한다."""
        h1 = MemoryManager._hash_input({"a": 1, "b": 2})
        h2 = MemoryManager._hash_input({"b": 2, "a": 1})
        assert h1 == h2

    def test_hash_input_different(self):
        """다른 입력에 대해 다른 해시가 생성되는지 확인한다."""
        h1 = MemoryManager._hash_input({"a": 1})
        h2 = MemoryManager._hash_input({"a": 2})
        assert h1 != h2


# ─────────────────────────────────────────────
# _serialize_message 직렬화 계약 회귀 테스트
# ─────────────────────────────────────────────
# 배경(버그):
#   웹 비스트리밍 /v1/chat 멀티턴에서 1턴 assistant 응답이 Redis 대화 이력에서
#   유실됐다. 근본 원인은 저장 직렬화가 model_dump(mode="json")을 그대로 써서
#   user content는 평문 str, assistant content는 ContentBlock 리스트로 "비대칭"
#   저장됐기 때문이다. 복원 측이 Message.assistant(리스트)에 리스트를 넘겨
#   ValidationError가 났고, 그 예외가 복원 루프 전체를 중단시켜 이력이 유실됐다.
#
# 수정:
#   _serialize_message가 content를 항상 평문(text_content)으로 통일해 저장한다.
#   아래 테스트는 user/assistant 모두 content가 "평문 str"로 직렬화되는지를
#   못박아(회귀 방지) 검증한다. 특히 assistant content가 리스트가 아님이 핵심.
class TestSerializeMessageContract:
    """MemoryManager._serialize_message의 저장 직렬화 계약을 검증한다."""

    def test_serialize_message_user_content_is_plain_string(self):
        """user 메시지 직렬화 시 content가 평문 str로 보존되는지 확인한다."""
        # Message.user는 content를 평문 str로 보관한다.
        msg = Message.user("안녕")
        # staticmethod라 인스턴스 없이 호출한다.
        data = MemoryManager._serialize_message(msg)

        assert isinstance(data["content"], str)
        assert data["content"] == "안녕"

    def test_serialize_message_assistant_content_is_plain_string_not_list(self):
        """assistant 메시지 직렬화 시 content가 (리스트가 아닌) 평문 str인지 확인한다.

        이것이 버그 회귀 방지의 핵심이다.
        Message.assistant("...")는 내부적으로 content를 list[ContentBlock]으로
        정규화하지만, 저장 단계에서는 평문 str로 통일되어야 한다.
        """
        msg = Message.assistant("네, 반갑습니다")
        # 사전 조건: 메모리 상의 content는 실제로 리스트(ContentBlock)다.
        assert isinstance(msg.content, list)

        data = MemoryManager._serialize_message(msg)

        # 저장 직렬화 결과의 content는 반드시 평문 str여야 한다(리스트가 아님).
        assert isinstance(data["content"], str), (
            f"assistant content가 평문이어야 하는데 {type(data['content'])} 이다 — "
            "버그 회귀: model_dump의 리스트가 그대로 저장됨"
        )
        assert not isinstance(data["content"], list)
        assert data["content"] == "네, 반갑습니다"

    def test_serialize_message_preserves_role_field(self):
        """직렬화 결과에 role 필드(user/assistant)가 보존되는지 확인한다."""
        user_data = MemoryManager._serialize_message(Message.user("질문"))
        asst_data = MemoryManager._serialize_message(Message.assistant("답변"))

        # role은 문자열로 직렬화된다(use_enum_values=True). 값만 비교한다.
        assert str(user_data["role"]) == "user"
        assert str(asst_data["role"]) == "assistant"
