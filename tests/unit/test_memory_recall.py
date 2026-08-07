# 장기 기억 회상 배선과 코드 RAG 분리 동작을 검증하는 단위 테스트.
"""
회상(recall) 배선 + 코드 RAG 청크 분리 검증.

배경 (2026-08-06 실측):
    MemoryManager.on_turn_start(회상 훅)은 구현돼 있었지만 프로덕션 어디서도 호출되지
    않아 장기 기억이 **쓰기 전용**이었다. 새 세션에서 이전 대화를 전혀 회상하지 못했다.
    동시에 tb_memories 에는 코드 RAG 청크가 113만 건(전체의 99.98%) 쌓여 있고 이들의
    importance 가 0.8 로 높아, 필터 없이 회상을 켜면 소스코드가 컨텍스트를 덮는다.

무엇을 지키나:
    - is_rag_chunk 가 RAG 청크(태그/메타데이터)만 정확히 골라낸다.
    - 회상 검색 결과에서 RAG 청크가 제외된다(on_turn_start / search_relevant 양쪽).
    - 회상 주입은 **기본 비활성**이다 — 설정을 켜지 않으면 아무 것도 주입되지 않는다.
    - 켜면 주입 블록이 만들어지고, 건수·길이 상한이 지켜진다.
    - 회상 실패는 대화를 막지 않는다(fail-soft).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from core.memory.types import MemoryEntry, MemoryType, is_rag_chunk
from core.orchestrator.query_engine import QueryEngine


def _entry(content: str, *, rag: bool = False, importance: float = 0.5) -> MemoryEntry:
    """대화 기억 또는 코드 RAG 청크 한 건을 만든다."""
    if rag:
        return MemoryEntry(
            memory_type=MemoryType.SEMANTIC,
            content=content,
            key="rag:core/bootstrap.py:chunk_3",
            tags=["rag", "py", "bootstrap.py"],
            importance=0.8,
            metadata={"source": "rag_indexer"},
        )
    return MemoryEntry(
        memory_type=MemoryType.EPISODIC,
        content=content,
        key="turn:s1:abc",
        tags=["conversation", "s1"],
        importance=importance,
        metadata={"session_id": "s1"},
    )


# ─────────────────────────────────────────────
# RAG 청크 판별
# ─────────────────────────────────────────────


def test_is_rag_chunk_detects_indexer_metadata():
    """metadata.source 가 rag_indexer 면 RAG 청크다."""
    assert is_rag_chunk(_entry("def foo(): ...", rag=True)) is True


def test_is_rag_chunk_detects_tag_only():
    """metadata 가 비어도 tags 에 'rag' 가 있으면 RAG 청크다(리트리버와 동일 기준)."""
    entry = MemoryEntry(
        memory_type=MemoryType.SEMANTIC, content="x", key="k", tags=["rag"], metadata={}
    )
    assert is_rag_chunk(entry) is True


def test_is_rag_chunk_keeps_conversation_memory():
    """대화 기억은 RAG 청크가 아니다."""
    assert is_rag_chunk(_entry("사용자 사번은 NX-8842")) is False


# ─────────────────────────────────────────────
# 회상 주입 (QueryEngine._recall_memories)
# ─────────────────────────────────────────────


class _StubManager:
    """on_turn_start 만 흉내 내는 MemoryManager 스텁."""

    def __init__(self, entries: list[MemoryEntry] | None = None, fail: bool = False) -> None:
        self._entries = entries or []
        self._fail = fail
        self.called = False

    async def on_turn_start(
        self, session_id: str, user_message: str, owner: str | None = None
    ) -> list[MemoryEntry]:
        self.called = True
        self.owner = owner  # 회상이 소유자를 넘겨주는지 확인용
        if self._fail:
            raise RuntimeError("검색 실패")
        return self._entries


def _engine(manager: Any, options: dict | None = None) -> QueryEngine:
    """_recall_memories 만 호출할 수 있는 최소 엔진을 만든다.

    QueryEngine 전체를 세우면 모델·도구까지 필요하므로, 실제로 읽는 속성만 갖춘
    가벼운 객체에 메서드를 빌려 붙인다(이 테스트의 관심사는 회상 로직 하나다).
    """
    engine = SimpleNamespace(
        _memory_manager=manager,
        _session_id="s1",
        _context=SimpleNamespace(options=options or {}),
    )
    # 회상이 소유자 해석을 함께 쓰므로 두 메서드를 같이 빌려 붙인다.
    engine._recall_memories = QueryEngine._recall_memories.__get__(engine, QueryEngine)
    engine._memory_owner = QueryEngine._memory_owner.__get__(engine, QueryEngine)
    return engine  # type: ignore[return-value]


async def test_recall_disabled_by_default():
    """설정을 켜지 않으면 회상 조회 자체를 하지 않는다(무회귀 보장)."""
    manager = _StubManager([_entry("과거 기억")])

    block = await _engine(manager, options={})._recall_memories("질문")

    assert block == ""
    assert manager.called is False, "꺼져 있는데 검색을 수행했다"


async def test_recall_enabled_builds_block():
    """켜면 과거 기억이 주입 블록으로 만들어진다."""
    manager = _StubManager([_entry("사용자는 정보전산원 소속이다")])
    engine = _engine(manager, {"memory_recall": {"enabled": True}})

    block = await engine._recall_memories("제 소속이 어디죠?")

    assert "사용자는 정보전산원 소속이다" in block
    # 모델이 이걸 사용자 발화로 오해하지 않도록 성격을 명시해야 한다.
    assert "참고용" in block
    assert manager.called is True


async def test_recall_respects_item_limit():
    """주입 건수 상한을 지킨다(컨텍스트 보호)."""
    entries = [_entry(f"기억{i}") for i in range(10)]
    engine = _engine(_StubManager(entries), {"memory_recall": {"enabled": True, "max_items": 3}})

    block = await engine._recall_memories("질문")

    assert block.count("- 기억") == 3


async def test_recall_truncates_long_entries():
    """기억 1건이 길어도 지정 길이로 잘라 넣는다."""
    engine = _engine(
        _StubManager([_entry("가" * 900)]),
        {"memory_recall": {"enabled": True, "max_chars": 50}},
    )

    block = await engine._recall_memories("질문")

    assert "가" * 50 in block
    assert "가" * 51 not in block


async def test_recall_skips_empty_entries():
    """내용이 빈 기억만 있으면 블록을 만들지 않는다."""
    engine = _engine(_StubManager([_entry("   ")]), {"memory_recall": {"enabled": True}})

    assert await engine._recall_memories("질문") == ""


async def test_recall_failure_is_fail_soft():
    """검색이 실패해도 예외를 올리지 않고 빈 블록으로 넘어간다."""
    engine = _engine(_StubManager(fail=True), {"memory_recall": {"enabled": True}})

    assert await engine._recall_memories("질문") == ""


async def test_recall_without_manager_is_noop():
    """MemoryManager 가 없으면(테스트/경량 경로) 아무 것도 하지 않는다."""
    engine = _engine(None, {"memory_recall": {"enabled": True}})

    assert await engine._recall_memories("질문") == ""


# ─────────────────────────────────────────────
# 소유자 격리 — 개인정보가 남의 대화로 새지 않아야 한다
# ─────────────────────────────────────────────
#
# 왜 중요한가: 사용자 발화까지 저장하게 되면서 사번·소속 같은 개인정보가 기억에
# 들어온다. tb_memories 에는 테넌트 컬럼이 없어 metadata.owner 로 구분하므로,
# 이 필터가 뚫리면 A 테넌트의 개인정보가 B 테넌트 대화에 주입된다.


class _StubLongTerm:
    """search_by_vector/search_by_text 의 owner 필터를 실제로 수행하는 스텁."""

    def __init__(self, entries: list[MemoryEntry]) -> None:
        self._entries = entries

    def _filtered(self, owner: str | None) -> list[MemoryEntry]:
        if owner is None:
            return list(self._entries)
        return [e for e in self._entries if e.metadata.get("owner") == owner]

    async def search_by_vector(self, embedding, memory_type=None, top_k=5, owner=None):
        return self._filtered(owner)[:top_k]

    async def search_by_text(self, query, memory_type=None, top_k=10, owner=None):
        return self._filtered(owner)[:top_k]


def _owned(content: str, owner: str | None) -> MemoryEntry:
    meta: dict[str, Any] = {"session_id": "s"}
    if owner:
        meta["owner"] = owner
    return MemoryEntry(
        memory_type=MemoryType.EPISODIC,
        content=content,
        key="turn:s:x",
        tags=["conversation"],
        importance=0.9,
        metadata=meta,
    )


def _manager_with(entries: list[MemoryEntry]):
    """LongTermMemory 만 스텁으로 갈아끼운 실제 MemoryManager."""
    from core.memory.manager import MemoryManager

    manager = MemoryManager.__new__(MemoryManager)
    manager._long_term = _StubLongTerm(entries)
    manager._model_provider = None  # 임베딩 없이 텍스트 검색 경로만 탄다
    return manager


async def test_recall_returns_only_own_tenant_memories():
    """다른 테넌트의 기억은 회상되지 않는다(개인정보 격리)."""
    entries = [
        _owned("A사 직원 사번은 NX-8842", owner="tenant-a"),
        _owned("B사 직원 사번은 QQ-1111", owner="tenant-b"),
    ]

    got = await _manager_with(entries).on_turn_start("s", "사번", owner="tenant-a")

    assert [e.content for e in got] == ["A사 직원 사번은 NX-8842"]


async def test_recall_excludes_ownerless_legacy_memories():
    """소유자 표식이 없는 레거시 기억은 회상되지 않는다(fail-closed)."""
    entries = [_owned("소유자 미상 기억", owner=None)]

    got = await _manager_with(entries).on_turn_start("s", "기억", owner="tenant-a")

    assert got == []


async def test_search_relevant_is_also_scoped():
    """MemoryRead 경로(search_relevant)도 같은 격리를 적용한다."""
    entries = [
        _owned("A사 기억", owner="tenant-a"),
        _owned("B사 기억", owner="tenant-b"),
    ]

    got = await _manager_with(entries).search_relevant("기억", owner="tenant-b")

    assert [e.content for e in got] == ["B사 기억"]


def test_memory_owner_reads_tenant_id():
    """테넌트 객체에서 소유자 식별자를 뽑는다."""
    engine = _engine(None, {"tenant": SimpleNamespace(id="tenant-a")})

    assert engine._memory_owner() == "tenant-a"


def test_memory_owner_none_without_tenant():
    """테넌트를 모르면 소유자도 None — 그 기억은 회상에서 제외된다."""
    assert _engine(None, {})._memory_owner() is None


# ─────────────────────────────────────────────
# 명시적 기억 요청 — 사용자가 말할 때만 저장한다
# ─────────────────────────────────────────────
#
# 기존 중요도 키워드는 전부 에러·수정·아키텍처 같은 '코딩 조수' 어휘라, 사용자가 말한
# 사실은 0.30점에 그쳐 승격 임계(0.6)를 넘지 못했다. 그래서 "기억해 줘"라고 해도
# 아무 것도 저장되지 않았다. 반대로 모든 발화를 저장하면 개인정보가 무분별하게 쌓인다.
# 그래서 **사용자가 명시적으로 요청했을 때만** 승격시킨다 — 저장 여부를 사용자가 정한다.


def _assessed(content: str):
    from core.memory.importance import ImportanceAssessor

    assessor = ImportanceAssessor()
    score = assessor.assess(content, MemoryType.EPISODIC)
    entry = MemoryEntry(memory_type=MemoryType.EPISODIC, content=content, importance=score)
    return score, assessor.should_promote(entry)


def test_explicit_memory_request_is_promoted():
    """'기억해 줘'가 붙은 사용자 발화는 승격된다."""
    score, promoted = _assessed(
        "제 소속은 정보전산원이고 담당 업무는 서버 운영입니다. 기억해 주세요."
    )

    assert promoted is True
    assert score > 0.6


def test_casual_utterance_is_not_promoted():
    """평범한 발화는 저장하지 않는다 — 개인정보가 무분별하게 쌓이지 않게."""
    for text in ("안녕하세요", "오늘 날씨 어때?", "제 사번은 NX-8842입니다."):
        _score, promoted = _assessed(text)
        assert promoted is False, f"저장돼선 안 되는 발화가 승격됨: {text}"


def test_english_memory_request_is_promoted():
    """영어 요청도 인식한다."""
    _score, promoted = _assessed("My team standup is at 10am. Please remember this.")

    assert promoted is True


# ─────────────────────────────────────────────
# 기계 봉투 제외 — 클라이언트가 씌운 형식은 기억이 아니다
# ─────────────────────────────────────────────
#
# 실측(2026-08-07): 소유자 있는 대화 기억 14건 중 7건(50%)이 VSCode 플러그인이 씌운
# "[COMPANY_CODING_AGENT_REQUEST] …" 지시문 봉투와 "[AGENT_TOOL_RESULTS] …" 였다.
# 길고 코딩 어휘가 많아 중요도 평가를 쉽게 통과한다. 플러그인은 요청마다 이걸 만들어
# 보내므로, 막지 않으면 그 테넌트의 기억이 통째로 스캐폴딩이 되고 회상이 그것을
# 다음 프롬프트에 도로 주입한다.


def test_plugin_request_envelope_is_not_memory():
    """VSCode 플러그인 지시문 봉투는 기억 대상이 아니다."""
    from core.memory.types import is_machine_envelope

    assert is_machine_envelope(
        "[COMPANY_CODING_AGENT_REQUEST]\n당신은 VSCode 안에서 동작하는 코딩 Agent입니다."
    )


def test_agent_tool_results_envelope_is_not_memory():
    """도구 결과 봉투도 마찬가지다."""
    from core.memory.types import is_machine_envelope

    assert is_machine_envelope("[AGENT_TOOL_RESULTS]\n아래 결과는 참고 데이터입니다.")


def test_plugin_response_json_is_not_memory():
    """`type` 키를 가진 응답 JSON도 제외한다."""
    from core.memory.types import is_machine_envelope

    assert is_machine_envelope('{"type": "chat_response", "content": "개선사항을 적용"}')


def test_user_utterance_is_still_memory():
    """★사람이 말한 것은 그대로 통과해야 한다(과잉 차단 방지)."""
    from core.memory.types import is_machine_envelope

    for text in (
        "제 소속은 정보전산원입니다. 기억해 주세요.",
        "이 함수를 리팩터링해줘",
        "[중요] 내일 회의는 10시입니다. 기억해 주세요.",  # 대괄호로 시작해도 봉투는 아니다
    ):
        assert is_machine_envelope(text) is False, f"정상 발화가 차단됨: {text}"


def test_pasted_json_without_type_is_still_memory():
    """사용자가 붙여넣은 JSON은 막지 않는다 — 조건을 좁게 뒀다."""
    from core.memory.types import is_machine_envelope

    assert is_machine_envelope('{"name": "홍길동", "dept": "정보전산원"}') is False
    assert is_machine_envelope("{ 이건 JSON이 아니라 그냥 중괄호") is False
