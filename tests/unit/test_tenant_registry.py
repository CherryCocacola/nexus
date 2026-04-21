"""
멀티테넌시 단위 테스트 (Part 5 Ch 15, 2026-04-21).

검증:
  1. TenantRegistry.get / resolve / resolve_by_api_key
  2. KnowledgeStore.search_by_vector 인메모리 폴백 — allowed_sources 필터
  3. KnowledgeRetriever.get_context — tenant allowed 전달 경로
  4. QueryEngine submit_message — tenant.model_override가 라우팅에 반영
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from core.config import RoutingConfig, TenantConfig, TenantRegistry
from core.rag.knowledge_retriever import KnowledgeRetriever
from core.rag.knowledge_store import KnowledgeEntry, KnowledgeStore


# ─────────────────────────────────────────────
# TenantRegistry
# ─────────────────────────────────────────────
def _make_registry() -> TenantRegistry:
    return TenantRegistry(
        default_tenant="default",
        tenants=[
            TenantConfig(
                id="default",
                name="Default",
                allowed_knowledge_sources=["kowiki"],
            ),
            TenantConfig(
                id="school-a",
                name="A학교",
                model_override="nexus-school-a",
                allowed_knowledge_sources=["kowiki", "school-a-textbook"],
                api_keys=["sk-school-a"],
            ),
            TenantConfig(
                id="corp-b",
                name="B기업",
                allowed_knowledge_sources=["corp-b-manual"],
                api_keys=["sk-corp-b"],
            ),
        ],
    )


def test_registry_get_returns_exact_match() -> None:
    r = _make_registry()
    t = r.get("school-a")
    assert t is not None
    assert t.model_override == "nexus-school-a"


def test_registry_get_unknown_returns_none() -> None:
    assert _make_registry().get("unknown") is None


def test_registry_resolve_falls_back_to_default() -> None:
    r = _make_registry()
    assert r.resolve(None).id == "default"
    assert r.resolve("unknown").id == "default"
    assert r.resolve("school-a").id == "school-a"


def test_registry_resolve_by_api_key() -> None:
    r = _make_registry()
    assert r.resolve_by_api_key("sk-school-a").id == "school-a"
    assert r.resolve_by_api_key("sk-corp-b").id == "corp-b"
    assert r.resolve_by_api_key("invalid") is None
    assert r.resolve_by_api_key("") is None


# ─────────────────────────────────────────────
# KnowledgeStore — allowed_sources 필터
# ─────────────────────────────────────────────
@pytest.mark.asyncio
async def test_store_vector_search_respects_allowed_sources() -> None:
    """인메모리 폴백에서도 allowed_sources가 cross-tenant 데이터를 차단한다."""
    store = KnowledgeStore(pg_pool=None)
    await store.add(KnowledgeEntry(
        source="kowiki", title="A", content="공개",
        embedding=(1.0, 0.0, 0.0),
    ))
    await store.add(KnowledgeEntry(
        source="school-a-textbook", title="S", content="A학교 교재",
        embedding=(1.0, 0.0, 0.0),
    ))
    await store.add(KnowledgeEntry(
        source="corp-b-manual", title="C", content="B기업 매뉴얼",
        embedding=(1.0, 0.0, 0.0),
    ))

    # A학교 테넌트는 kowiki + school-a-textbook만
    a = await store.search_by_vector(
        embedding=[1.0, 0.0, 0.0], top_k=10,
        allowed_sources=["kowiki", "school-a-textbook"],
        min_similarity=0.5,
    )
    sources = {r["source"] for r in a}
    assert sources == {"kowiki", "school-a-textbook"}
    assert all(r["source"] != "corp-b-manual" for r in a)


@pytest.mark.asyncio
async def test_store_vector_search_single_source_back_compat() -> None:
    """기존 단일 source 인자는 allowed_sources=[source]로 정규화된다."""
    store = KnowledgeStore(pg_pool=None)
    await store.add(KnowledgeEntry(
        source="kowiki", title="A", content="x", embedding=(1.0, 0.0, 0.0),
    ))
    await store.add(KnowledgeEntry(
        source="corp-b", title="B", content="y", embedding=(1.0, 0.0, 0.0),
    ))
    r = await store.search_by_vector(
        embedding=[1.0, 0.0, 0.0], top_k=10, source="kowiki", min_similarity=0.5,
    )
    assert all(x["source"] == "kowiki" for x in r)


# ─────────────────────────────────────────────
# KnowledgeRetriever — tenant 필터 전달
# ─────────────────────────────────────────────
@pytest.mark.asyncio
async def test_retriever_passes_allowed_sources_to_store() -> None:
    """get_context(allowed_sources=[...])가 store.search_by_vector에 그대로 전파."""
    store = MagicMock()
    store.search_by_vector = AsyncMock(return_value=[])
    store.search_by_text = AsyncMock(return_value=[])
    emb = MagicMock()
    emb.embed = AsyncMock(return_value=[[0.1] * 3])

    retr = KnowledgeRetriever(store=store, embedding_provider=emb)
    await retr.get_context("query", allowed_sources=["kowiki", "school-a"])

    store.search_by_vector.assert_awaited_once()
    call = store.search_by_vector.await_args
    assert call.kwargs["allowed_sources"] == ["kowiki", "school-a"]


@pytest.mark.asyncio
async def test_retriever_filters_text_fallback_by_allowed_sources() -> None:
    """텍스트 검색 폴백도 allowed_sources로 Python-side 필터."""
    store = MagicMock()
    store.search_by_vector = AsyncMock(return_value=[])  # 벡터 실패
    row_kowiki = {
        "source": "kowiki", "title": "A", "content": "x",
        "tags": [], "similarity": 0.5, "metadata": {},
    }
    row_leak = {
        "source": "leak", "title": "B", "content": "y",
        "tags": [], "similarity": 0.5, "metadata": {},
    }
    store.search_by_text = AsyncMock(return_value=[row_kowiki, row_leak])
    emb = MagicMock()
    emb.embed = AsyncMock(return_value=[[0.1] * 3])

    retr = KnowledgeRetriever(store=store, embedding_provider=emb)
    ctx = await retr.get_context("q", allowed_sources=["kowiki"])
    assert "leak" not in ctx


# ─────────────────────────────────────────────
# QueryEngine — tenant.model_override 라우팅
# ─────────────────────────────────────────────
@pytest.mark.asyncio
async def test_query_engine_applies_tenant_model_override_on_knowledge() -> None:
    """KNOWLEDGE 분류 질의에서 tenant.model_override가 라우팅 프로필 model을 덮어쓴다."""
    from core.orchestrator.query_engine import QueryEngine
    from core.tools.base import ToolUseContext

    tenant = TenantConfig(
        id="school-a",
        model_override="nexus-school-a",
        allowed_knowledge_sources=["kowiki"],
    )

    dispatcher_stream = AsyncMock()
    dispatcher_stream.__aiter__ = lambda self: iter([])
    dispatcher = MagicMock()

    # route()는 async generator를 돌려줘야 하므로 직접 구현
    captured: dict = {}
    async def fake_route(**kwargs):
        captured.update(kwargs)
        if False:
            yield None
        return
    dispatcher.route = fake_route

    ctx = ToolUseContext(cwd=".", session_id="t-sa", options={"tenant": tenant})
    eng = QueryEngine(
        model_provider=MagicMock(),
        tools=[],
        context=ctx,
        system_prompt="",
        model_dispatcher=dispatcher,
        routing_config=RoutingConfig(),  # 기본: enabled=True, KNOWLEDGE → qwen3.5-27b
    )

    async for _ in eng.submit_message("니체 철학 설명해줘"):
        pass

    # 라우팅 결과 — tenant override가 적용되어 model="nexus-school-a"
    assert captured.get("model_override") == "nexus-school-a"


@pytest.mark.asyncio
async def test_query_engine_does_not_apply_tenant_override_on_tool_mode() -> None:
    """TOOL 모드에서는 tenant.model_override 적용 안 함 (Phase LoRA 유지)."""
    from core.orchestrator.query_engine import QueryEngine
    from core.tools.base import ToolUseContext

    tenant = TenantConfig(id="school-a", model_override="nexus-school-a")

    captured: dict = {}
    async def fake_route(**kwargs):
        captured.update(kwargs)
        if False:
            yield None
        return
    dispatcher = MagicMock()
    dispatcher.route = fake_route

    ctx = ToolUseContext(cwd=".", session_id="t", options={"tenant": tenant})
    eng = QueryEngine(
        model_provider=MagicMock(),
        tools=[],
        context=ctx,
        system_prompt="",
        model_dispatcher=dispatcher,
        routing_config=RoutingConfig(),
    )

    # "이 파일 읽어줘"는 tool_keywords 매칭으로 TOOL로 분류
    async for _ in eng.submit_message("이 파일 읽어줘"):
        pass

    # TOOL 경로 — tenant override 무시, 원래 tool_mode.model 사용
    assert captured.get("model_override") == "nexus-phase3"
