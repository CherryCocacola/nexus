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
