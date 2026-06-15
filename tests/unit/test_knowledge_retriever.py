"""
KnowledgeRetriever.get_context() ILIKE 폴백 분기 단위 테스트 (2026-06-03).

배경:
  기존 폴백 로직은 벡터 검색 결과가 0건이면 무조건 search_by_text()
  (ILIKE 1M행 풀스캔, 콜드 시 150초)를 호출했다. 이는
  "임베딩 서버 불능"과 "관련 지식 0건"을 구분하지 못하는 버그였다.

수정:
  vector_search_done 플래그를 도입해, 임베딩이 성공하고 search_by_vector를
  끝까지 수행했으면(True) 결과가 0건이어도 ILIKE 폴백을 건너뛰고 ""를 반환한다.
  임베딩 서버가 실제로 불능일 때만(provider None / 빈 벡터 / embed 예외)
  폴백을 허용한다. 핵심 분기: `if not results and not vector_search_done:`

검증 전략:
  - KnowledgeStore 는 MagicMock 으로 만들고, search_by_vector / search_by_text 를
    AsyncMock 으로 교체한다. 이렇게 하면 "폴백이 호출됐는가"를
    assert_awaited / assert_not_awaited 로 정밀하게 감시할 수 있다.
  - 임베딩 provider(ModelProvider) 도 MagicMock + AsyncMock(embed) 로 대체한다.
  - 실제 DB / 임베딩 서버는 절대 호출하지 않는다 (전부 mock).
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from core.rag.knowledge_retriever import KnowledgeRetriever


# ─────────────────────────────────────────────
# 테스트 헬퍼 — mock store / mock 임베딩 provider 조립
# ─────────────────────────────────────────────
def _make_mock_store(
    vector_results: list[dict] | None = None,
    text_results: list[dict] | None = None,
) -> MagicMock:
    """
    KnowledgeStore 를 흉내내는 MagicMock 을 만든다.

    - search_by_vector: 벡터 검색 결과(기본 0건)를 돌려주는 AsyncMock
    - search_by_text:   ILIKE 폴백 결과(기본 0건)를 돌려주는 AsyncMock
    두 메서드 모두 AsyncMock 이라 호출(await) 여부를 추적할 수 있다.
    """
    store = MagicMock()
    store.search_by_vector = AsyncMock(return_value=vector_results or [])
    store.search_by_text = AsyncMock(return_value=text_results or [])
    return store


def _make_mock_embedding(*, return_value=None, side_effect=None) -> MagicMock:
    """
    임베딩 provider(ModelProvider) 를 흉내내는 MagicMock 을 만든다.

    embed() 은 AsyncMock 이며 호출 시:
      - return_value 가 주어지면 그 벡터 리스트를 반환
      - side_effect 가 주어지면 예외를 발생
    """
    emb = MagicMock()
    if side_effect is not None:
        emb.embed = AsyncMock(side_effect=side_effect)
    else:
        emb.embed = AsyncMock(return_value=return_value)
    return emb


# 벡터 검색이 실제로 무언가를 찾았을 때 돌려줄 표준 결과 한 건.
# 헤더 조립(3단계)이 정상 동작하는지 회귀 검증할 때 쓴다.
_SAMPLE_CHUNK = {
    "source": "kowiki",
    "title": "니체",
    "section": "생애",
    "similarity": 0.91,
    "content": "독일 철학자, 1844~1900",
}


# ─────────────────────────────────────────────
# 시나리오 1 (핵심) — 임베딩 성공 + 벡터검색 0건 → 폴백 차단
# ─────────────────────────────────────────────
@pytest.mark.asyncio
async def test_get_context_vector_search_zero_results_skips_text_fallback() -> None:
    """
    이번 수정의 핵심:
    임베딩이 정상이고 search_by_vector 를 끝까지 수행했는데 0건이면,
    "관련 지식 없음"이므로 ILIKE 폴백(search_by_text)을 호출하면 안 된다.
    → search_by_text 는 await 되지 않아야 하고, 반환값은 "" 여야 한다.
    """
    # 벡터 검색은 0건, (만약 잘못 호출되면) 텍스트 검색은 1건 주도록 세팅.
    # 이렇게 해두면 폴백이 잘못 호출됐을 때 결과가 ""가 아니게 되어 곧장 드러난다.
    store = _make_mock_store(vector_results=[], text_results=[_SAMPLE_CHUNK])
    emb = _make_mock_embedding(return_value=[[1.0, 0.0, 0.0]])
    retr = KnowledgeRetriever(store=store, embedding_provider=emb)

    ctx = await retr.get_context("관련 지식 없는 질의", max_tokens=500)

    # 폴백은 절대 호출되면 안 된다 — 이게 이번 수정의 본질
    store.search_by_text.assert_not_awaited()
    # 벡터 검색은 정상적으로 한 번 수행됐어야 한다
    store.search_by_vector.assert_awaited_once()
    # 결과는 빈 문자열
    assert ctx == ""


# ─────────────────────────────────────────────
# 시나리오 2 — 임베딩 provider 가 None → 폴백 진입
# ─────────────────────────────────────────────
@pytest.mark.asyncio
async def test_get_context_embedding_provider_none_uses_text_fallback() -> None:
    """
    임베딩 provider 자체가 없으면(None) 벡터 검색을 아예 수행하지 못하므로,
    vector_search_done 이 False 로 남아 ILIKE 폴백이 허용되어야 한다.
    """
    store = _make_mock_store(vector_results=[], text_results=[_SAMPLE_CHUNK])
    retr = KnowledgeRetriever(store=store, embedding_provider=None)

    ctx = await retr.get_context("니체", max_tokens=500)

    # 임베딩이 없으니 벡터 검색은 호출되지 않는다
    store.search_by_vector.assert_not_awaited()
    # 폴백은 반드시 호출되어야 한다
    store.search_by_text.assert_awaited_once()
    # 폴백 결과가 조립되어 비어 있지 않은 문자열이 나온다
    assert "니체" in ctx


# ─────────────────────────────────────────────
# 시나리오 3 — embed() 가 예외 → 폴백 진입
# ─────────────────────────────────────────────
@pytest.mark.asyncio
async def test_get_context_embedding_raises_uses_text_fallback() -> None:
    """
    embed() 호출이 예외를 던지면(임베딩 서버 다운 등) try 블록이 깨져
    vector_search_done 이 False 로 남으므로 ILIKE 폴백이 허용되어야 한다.
    """
    store = _make_mock_store(vector_results=[], text_results=[_SAMPLE_CHUNK])
    emb = _make_mock_embedding(side_effect=RuntimeError("embedding server down"))
    retr = KnowledgeRetriever(store=store, embedding_provider=emb)

    ctx = await retr.get_context("니체", max_tokens=500)

    # 예외가 나도 벡터 검색(search_by_vector)까지는 도달하지 못한다
    store.search_by_vector.assert_not_awaited()
    # 폴백은 반드시 호출되어야 한다
    store.search_by_text.assert_awaited_once()
    assert "니체" in ctx


# ─────────────────────────────────────────────
# 시나리오 4 — embed() 가 빈 벡터 반환 → 폴백 진입
# ─────────────────────────────────────────────
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "empty_vecs",
    [
        [[]],  # 벡터 리스트는 있으나 첫 벡터가 빈 리스트 → vecs[0] 이 falsy
        [],    # 벡터 리스트 자체가 빈 경우 → vecs 가 falsy
    ],
)
async def test_get_context_embedding_empty_vector_uses_text_fallback(
    empty_vecs: list,
) -> None:
    """
    embed() 가 빈 벡터([[]] 또는 [])를 반환하면 `if vecs and vecs[0]` 가
    False 가 되어 search_by_vector 를 호출하지 못한다.
    → vector_search_done 이 False 로 남아 ILIKE 폴백이 허용되어야 한다.
    """
    store = _make_mock_store(vector_results=[], text_results=[_SAMPLE_CHUNK])
    emb = _make_mock_embedding(return_value=empty_vecs)
    retr = KnowledgeRetriever(store=store, embedding_provider=emb)

    ctx = await retr.get_context("니체", max_tokens=500)

    # embed 는 호출됐지만 벡터가 비어 search_by_vector 까지 가지 못했다
    emb.embed.assert_awaited_once()
    store.search_by_vector.assert_not_awaited()
    # 폴백은 반드시 호출되어야 한다
    store.search_by_text.assert_awaited_once()
    assert "니체" in ctx


# ─────────────────────────────────────────────
# 회귀 — 임베딩 성공 + 벡터검색 N건 → 정상 조립
# ─────────────────────────────────────────────
@pytest.mark.asyncio
async def test_get_context_vector_search_with_results_assembles_block() -> None:
    """
    회귀 방지:
    임베딩이 정상이고 벡터 검색이 N건을 돌려주면, 폴백 없이 청크가
    헤더와 함께 조립되어 주입 문자열이 반환되어야 한다.
    """
    second = {
        "source": "kowiki",
        "title": "쇼펜하우어",
        "section": "",
        "similarity": 0.77,
        "content": "의지와 표상으로서의 세계",
    }
    store = _make_mock_store(vector_results=[_SAMPLE_CHUNK, second])
    emb = _make_mock_embedding(return_value=[[1.0, 0.0, 0.0]])
    retr = KnowledgeRetriever(store=store, embedding_provider=emb, top_k=5)

    ctx = await retr.get_context("철학자들", max_tokens=1500)

    # 폴백은 호출되지 않고, 벡터 검색만 수행된다
    store.search_by_text.assert_not_awaited()
    store.search_by_vector.assert_awaited_once()
    # 두 청크의 핵심 내용과 헤더 메타데이터가 모두 들어가야 한다
    assert "니체" in ctx
    assert "쇼펜하우어" in ctx
    assert "독일 철학자" in ctx
    assert "의지와 표상" in ctx
    assert "sim=0.91" in ctx  # 헤더에 유사도가 소수점 2자리로 포맷된다
    assert "kowiki" in ctx
