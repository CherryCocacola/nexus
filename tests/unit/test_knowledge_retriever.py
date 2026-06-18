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
# 엔티티(식별자) 매칭 게이팅 (옵션 A) — 부분 관련 함정 차단
# ─────────────────────────────────────────────
@pytest.mark.asyncio
async def test_entity_gating_drops_chunk_missing_identifier() -> None:
    """질의에 식별자(다자리 숫자)가 있는데 청크에 없으면 드롭 → 빈 반환.

    'BWV 543' 질의에 '바흐 일반' 청크(543 미포함)가 잡혀도, 식별자 543이
    없으므로 드롭되어 ""를 반환한다(→ 호출자가 '관련 자료 없음' 주입).
    """
    chunk = {
        "source": "kowiki", "title": "바흐", "section": "", "similarity": 0.6,
        "content": "요한 제바스티안 바흐는 독일의 작곡가이자 오르가니스트다.",
    }
    store = _make_mock_store(vector_results=[chunk])
    emb = _make_mock_embedding(return_value=[[0.1] * 4])
    kr = KnowledgeRetriever(store=store, embedding_provider=emb)

    out = await kr.get_context("바흐 작품 BWV 543 알려줘")
    assert out == ""


@pytest.mark.asyncio
async def test_entity_gating_keeps_chunk_with_identifier() -> None:
    """식별자(543)가 청크에 실제로 등장하면 정상 주입한다."""
    chunk = {
        "source": "kowiki", "title": "BWV 543", "section": "", "similarity": 0.8,
        "content": "전주곡과 푸가 BWV 543은 바흐의 오르간 작품이다.",
    }
    store = _make_mock_store(vector_results=[chunk])
    emb = _make_mock_embedding(return_value=[[0.1] * 4])
    kr = KnowledgeRetriever(store=store, embedding_provider=emb)

    out = await kr.get_context("BWV 543 알려줘")
    assert "BWV 543" in out


@pytest.mark.asyncio
async def test_entity_gating_skipped_for_conceptual_query() -> None:
    """다자리 숫자 식별자가 없는 개념 질의는 게이팅을 적용하지 않는다(회귀 방지)."""
    store = _make_mock_store(vector_results=[_SAMPLE_CHUNK])
    emb = _make_mock_embedding(return_value=[[0.1] * 4])
    kr = KnowledgeRetriever(store=store, embedding_provider=emb)

    out = await kr.get_context("광합성 원리를 설명해줘")
    assert "니체" in out


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


# ─────────────────────────────────────────────
# 유사도 2단 게이팅 (2026-06-18) — 무관 청크 주입 차단(할루시네이션 저감)
#   1단계: 절대 임계(abs_threshold) — top1조차 미만이면 전체 드롭
#   2단계: 상대 마진(relevance_margin) — top_sim - margin 미만 청크 절단
# 운영 게이팅 값: abs_threshold=0.84, relevance_margin=0.03 (config 기본값)
# ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_abs_threshold_top1_below_threshold_drops_all_returns_empty() -> None:
    """top1 유사도가 abs_threshold 미만이면 전체 드롭되어 빈 문자열을 반환한다.

    실측 사례: 메타질문 "rag에 이런 정보가 있었어?"의 top1=0.824는
    abs_threshold=0.84 미만이므로 "관련 자료 없음"으로 보고 전부 드롭한다.
    엔티티 게이팅과 섞이지 않도록 질의에 다자리 숫자를 넣지 않는다.
    """
    # top1=0.824 < 0.84 → 절대 임계에서 전체 드롭되어야 한다.
    chunk = {
        "source": "kowiki", "title": "RAG", "section": "", "similarity": 0.824,
        "content": "검색 증강 생성에 관한 일반 설명 청크.",
    }
    store = _make_mock_store(vector_results=[chunk])
    emb = _make_mock_embedding(return_value=[[0.1] * 4])
    kr = KnowledgeRetriever(
        store=store,
        embedding_provider=emb,
        abs_threshold=0.84,
        relevance_margin=0.03,
    )

    out = await kr.get_context("rag에 이런 정보가 있었어")
    assert out == ""


@pytest.mark.asyncio
async def test_relevance_margin_keeps_chunks_within_margin_drops_outliers() -> None:
    """top1 >= abs_threshold일 때, top_sim - margin 미만 청크만 절단한다.

    top1=0.857, margin=0.03이면 컷오프는 0.827이다.
      - 0.857(top), 0.83(>= 0.827)  → 유지
      - 0.80(< 0.827)               → 노이즈로 절단
    엔티티 게이팅 회피를 위해 질의에 다자리 숫자를 넣지 않는다.
    """
    top = {
        "source": "kowiki", "title": "바흐", "section": "", "similarity": 0.857,
        "content": "요한 제바스티안 바흐는 독일의 작곡가다.",
    }
    near = {
        "source": "kowiki", "title": "헨델", "section": "", "similarity": 0.83,
        "content": "게오르크 프리드리히 헨델도 바로크 작곡가다.",
    }
    far = {
        "source": "kowiki", "title": "잡음청크", "section": "", "similarity": 0.80,
        "content": "마진 밖 노이즈 청크 본문 텍스트.",
    }
    store = _make_mock_store(vector_results=[top, near, far])
    emb = _make_mock_embedding(return_value=[[0.1] * 4])
    kr = KnowledgeRetriever(
        store=store,
        embedding_provider=emb,
        abs_threshold=0.84,
        relevance_margin=0.03,
    )

    out = await kr.get_context("바흐와 바로크 음악 알려줘")
    # 컷오프(0.827) 이상인 top/near는 유지, 미만인 far는 절단되어야 한다.
    assert "바흐" in out
    assert "헨델" in out
    assert "잡음청크" not in out


@pytest.mark.asyncio
async def test_gating_disabled_by_default_passes_all_results() -> None:
    """게이팅 인자를 주지 않으면(기본 abs=0.0 / margin=1.0) 종전과 동일하게 통과.

    하위 호환 검증: 낮은 유사도(0.55, 0.50)의 청크라도 기본값에서는
    절대 임계(0.0)와 상대 마진(1.0) 어느 쪽에도 걸리지 않아 모두 주입된다.
    """
    low1 = {
        "source": "kowiki", "title": "청크A", "section": "", "similarity": 0.55,
        "content": "유사도가 낮은 첫 번째 청크.",
    }
    low2 = {
        "source": "kowiki", "title": "청크B", "section": "", "similarity": 0.50,
        "content": "유사도가 더 낮은 두 번째 청크.",
    }
    store = _make_mock_store(vector_results=[low1, low2])
    emb = _make_mock_embedding(return_value=[[0.1] * 4])
    # 게이팅 인자 미지정 → abs_threshold=0.0, relevance_margin=1.0 (비활성)
    kr = KnowledgeRetriever(store=store, embedding_provider=emb)

    out = await kr.get_context("두 청크 모두 보여줘")
    assert "청크A" in out
    assert "청크B" in out


@pytest.mark.asyncio
async def test_similarity_gating_then_entity_gating_cooperate_returns_empty() -> None:
    """유사도 게이팅을 통과해도 엔티티(식별자)가 청크에 없으면 전부 드롭.

    질의에 식별자 "543"이 있고 청크는 유사도(0.857)로 게이팅을 통과하지만,
    content에 543이 없으므로 엔티티 게이팅이 전부 드롭 → 빈 문자열.
    두 게이팅이 순서대로(유사도 → 엔티티) 협동함을 검증한다.
    """
    chunk = {
        "source": "kowiki", "title": "바흐", "section": "", "similarity": 0.857,
        "content": "요한 제바스티안 바흐는 독일의 작곡가이자 오르가니스트다.",
    }
    store = _make_mock_store(vector_results=[chunk])
    emb = _make_mock_embedding(return_value=[[0.1] * 4])
    kr = KnowledgeRetriever(
        store=store,
        embedding_provider=emb,
        abs_threshold=0.84,
        relevance_margin=0.03,
    )

    out = await kr.get_context("바흐 작품 BWV 543 알려줘")
    # 유사도 게이팅은 통과(0.857 >= 0.84)했지만 식별자 543 부재로 전부 드롭.
    assert out == ""


@pytest.mark.asyncio
async def test_abs_threshold_exact_boundary_passes_strict_less_than() -> None:
    """경계값: top1 == abs_threshold(0.84 == 0.84)는 `<` 비교라 통과해야 한다.

    절대 임계는 `top_sim < abs_threshold`로 비교하므로, 정확히 같은 값은
    드롭되지 않고 통과한다(엄격한 미만 비교의 경계 동작 회귀 가드).
    엔티티 게이팅 회피를 위해 질의에 다자리 숫자를 넣지 않는다.
    """
    chunk = {
        "source": "kowiki", "title": "경계청크", "section": "", "similarity": 0.84,
        "content": "정확히 임계값과 같은 유사도를 가진 청크.",
    }
    store = _make_mock_store(vector_results=[chunk])
    emb = _make_mock_embedding(return_value=[[0.1] * 4])
    kr = KnowledgeRetriever(
        store=store,
        embedding_provider=emb,
        abs_threshold=0.84,
        relevance_margin=0.03,
    )

    out = await kr.get_context("경계값 청크 알려줘")
    # 0.84 == 0.84 → `<` 비교에서 False이므로 드롭되지 않고 통과한다.
    assert "경계청크" in out
