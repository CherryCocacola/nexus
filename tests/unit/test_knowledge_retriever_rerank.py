"""
KnowledgeRetriever 크로스인코더 리랭커 통합 단위 테스트 (2026-07-05).

배경:
  B200 임베딩 서버에 /v1/rerank 엔드포인트가 추가돼, (질의, 각 청크)의 관련도를
  크로스인코더로 직접 점수화(0~1)한다. KnowledgeRetriever는 rerank_enabled일 때만
  이 점수로 후보를 재정렬·게이팅하고, e5 코사인 분포 전용 2단 유사도 게이팅
  (abs_threshold/relevance_margin)은 rerank 점수 게이팅(min_score)으로 대체한다.
  엔티티(식별자) 게이팅은 유지된다. 기본 OFF라 켜기 전 동작은 종전과 100% 동일.

검증 전략(기존 test_knowledge_retriever.py와 동일):
  - KnowledgeStore / 임베딩 provider는 전부 mock. 실제 DB/서버는 호출하지 않는다.
  - provider.rerank는 AsyncMock으로 대체해 "호출 여부/인자/점수"를 정밀 제어한다.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from core.rag.knowledge_retriever import KnowledgeRetriever


# ─────────────────────────────────────────────
# 테스트 헬퍼 — mock store / mock provider(embed + rerank)
# ─────────────────────────────────────────────
def _make_mock_store(
    vector_results: list[dict] | None = None,
    text_results: list[dict] | None = None,
) -> MagicMock:
    store = MagicMock()
    store.search_by_vector = AsyncMock(return_value=vector_results or [])
    store.search_by_text = AsyncMock(return_value=text_results or [])
    return store


def _make_mock_provider(
    *,
    embed_value=None,
    rerank_scores: list[float] | None = None,
    rerank_side_effect=None,
) -> MagicMock:
    """embed()·rerank() 를 모두 AsyncMock으로 갖춘 provider mock."""
    p = MagicMock()
    p.embed = AsyncMock(return_value=embed_value or [[0.1] * 4])
    if rerank_side_effect is not None:
        p.rerank = AsyncMock(side_effect=rerank_side_effect)
    else:
        p.rerank = AsyncMock(return_value=rerank_scores or [])
    return p


def _chunk(title: str, sim: float, content: str, section: str = "") -> dict:
    return {
        "source": "kowiki",
        "title": title,
        "section": section,
        "similarity": sim,
        "content": content,
    }


# ─────────────────────────────────────────────
# 1) rerank 비활성(기본) — 벡터검색 인자·게이팅 현행과 동일, rerank 미호출
# ─────────────────────────────────────────────
@pytest.mark.asyncio
async def test_rerank_disabled_keeps_legacy_vector_search_args() -> None:
    """rerank_enabled=False면 벡터검색 인자가 종전과 동일하고 rerank는 호출되지 않는다."""
    chunk = _chunk("니체", 0.91, "독일 철학자, 1844~1900")
    store = _make_mock_store(vector_results=[chunk])
    prov = _make_mock_provider()  # rerank는 준비돼 있지만 호출되면 안 됨
    kr = KnowledgeRetriever(store=store, embedding_provider=prov)  # rerank 인자 미지정

    out = await kr.get_context("철학자 알려줘")

    # rerank는 절대 호출되지 않는다(하위 호환)
    prov.rerank.assert_not_awaited()
    # 벡터검색은 종전 인자: top_k=기본 top_k(5), min_similarity=기본(0.5), 임베딩 미포함
    kwargs = store.search_by_vector.call_args.kwargs
    assert kwargs["top_k"] == 5
    assert kwargs["min_similarity"] == 0.5
    assert kwargs["with_embedding"] is False
    # 헤더는 종전대로 sim= 표기, 청크는 정상 조립
    assert "니체" in out
    assert "sim=0.91" in out


# ─────────────────────────────────────────────
# 2) rerank 활성 — 넓은 리콜 그물 + rerank 점수로 재정렬
# ─────────────────────────────────────────────
@pytest.mark.asyncio
async def test_rerank_enabled_uses_wide_recall_and_reorders() -> None:
    """rerank ON: fetch_k·완화된 컷오프로 검색하고 rerank 점수 내림차순으로 재정렬한다."""
    a = _chunk("A", 0.90, "청크 A 본문")
    b = _chunk("B", 0.80, "청크 B 본문")
    c = _chunk("C", 0.70, "청크 C 본문")
    store = _make_mock_store(vector_results=[a, b, c])
    # 벡터순은 A>B>C지만 rerank는 B>C>A로 뒤집는다(순서는 vector_results 정렬 기준).
    prov = _make_mock_provider(rerank_scores=[0.40, 0.95, 0.60])
    kr = KnowledgeRetriever(
        store=store,
        embedding_provider=prov,
        rerank_enabled=True,
        rerank_fetch_k=20,
        rerank_top_k=5,
        rerank_min_score=0.3,
        rerank_min_similarity=0.6,
    )

    out = await kr.get_context("세 청크 재정렬 확인")

    # 벡터검색은 리랭킹용 파라미터로 호출된다
    kwargs = store.search_by_vector.call_args.kwargs
    assert kwargs["top_k"] == 20  # rerank_fetch_k
    assert kwargs["min_similarity"] == 0.6  # rerank_min_similarity
    assert kwargs["with_embedding"] is False
    # rerank는 벡터순 그대로의 content 리스트로 1회 호출된다
    prov.rerank.assert_awaited_once()
    call = prov.rerank.await_args
    assert call.args[1] == ["청크 A 본문", "청크 B 본문", "청크 C 본문"]
    # 출력 순서는 rerank 내림차순 B > C > A
    assert out.index("청크 B") < out.index("청크 C") < out.index("청크 A")
    # 헤더는 rr= 표기(최고 점수 B=0.95)
    assert "rr=0.95" in out


# ─────────────────────────────────────────────
# 3) rerank min_score 절대 게이팅 — 최고점 < min_score면 전체 드롭
# ─────────────────────────────────────────────
@pytest.mark.asyncio
async def test_rerank_top_score_below_min_score_drops_all() -> None:
    """최고 rerank 점수조차 min_score 미만이면 전부 드롭 → 빈 문자열."""
    a = _chunk("A", 0.90, "청크 A 본문")
    b = _chunk("B", 0.80, "청크 B 본문")
    store = _make_mock_store(vector_results=[a, b])
    prov = _make_mock_provider(rerank_scores=[0.20, 0.10])  # 최고 0.20 < 0.3
    kr = KnowledgeRetriever(
        store=store, embedding_provider=prov, rerank_enabled=True, rerank_min_score=0.3
    )

    out = await kr.get_context("관련 낮은 질의")
    assert out == ""


@pytest.mark.asyncio
async def test_rerank_keeps_only_scores_at_or_above_min_score() -> None:
    """min_score 이상만 유지하고 미만 청크는 절단한다."""
    a = _chunk("A", 0.90, "청크 A 본문")
    b = _chunk("B", 0.80, "청크 B 본문")
    c = _chunk("C", 0.70, "청크 C 본문")
    store = _make_mock_store(vector_results=[a, b, c])
    prov = _make_mock_provider(rerank_scores=[0.90, 0.20, 0.50])  # B=0.20 < 0.3 절단
    kr = KnowledgeRetriever(
        store=store, embedding_provider=prov, rerank_enabled=True, rerank_min_score=0.3
    )

    out = await kr.get_context("min_score 게이팅 확인")
    assert "청크 A" in out
    assert "청크 C" in out
    assert "청크 B" not in out


# ─────────────────────────────────────────────
# 4) 엔티티(식별자) 게이팅 — rerank 뒤에도 유지된다
# ─────────────────────────────────────────────
@pytest.mark.asyncio
async def test_rerank_then_entity_gating_still_applies() -> None:
    """rerank 점수가 높아도 질의 식별자가 청크에 없으면 엔티티 게이팅이 드롭한다."""
    # 질의에 '543' 식별자가 있으나 청크 content에는 543이 없다.
    chunk = _chunk("바흐", 0.85, "요한 제바스티안 바흐는 독일의 작곡가다.")
    store = _make_mock_store(vector_results=[chunk])
    prov = _make_mock_provider(rerank_scores=[0.99])  # rerank는 매우 높게 준다
    kr = KnowledgeRetriever(
        store=store, embedding_provider=prov, rerank_enabled=True, rerank_min_score=0.3
    )

    out = await kr.get_context("바흐 작품 BWV 543 알려줘")
    # rerank 게이팅은 통과(0.99)했지만 식별자 543 부재로 엔티티 게이팅이 전부 드롭.
    assert out == ""


# ─────────────────────────────────────────────
# 5) fail-safe — rerank 예외 시 벡터순 + 기존 게이팅으로 폴백
# ─────────────────────────────────────────────
@pytest.mark.asyncio
async def test_rerank_exception_falls_back_to_vector_order() -> None:
    """rerank 호출이 예외를 던지면 리랭킹을 건너뛰고 벡터순+기존 게이팅으로 폴백한다."""
    a = _chunk("A", 0.90, "청크 A 본문")
    b = _chunk("B", 0.80, "청크 B 본문")
    store = _make_mock_store(vector_results=[a, b])
    prov = _make_mock_provider(rerank_side_effect=RuntimeError("rerank server down"))
    # 기존 게이팅은 기본값(abs=0.0/margin=1.0)이라 전부 통과 → 폴백 시 둘 다 조립돼야 함.
    kr = KnowledgeRetriever(
        store=store, embedding_provider=prov, rerank_enabled=True
    )

    out = await kr.get_context("폴백 확인 질의")

    # rerank는 시도됐지만(1회 await) 실패 → 벡터순 폴백
    prov.rerank.assert_awaited_once()
    assert out.index("청크 A") < out.index("청크 B")  # 벡터순 유지
    assert "sim=" in out  # rr= 아님(리랭킹 미적용)
    assert "rr=" not in out


# ─────────────────────────────────────────────
# 6) e5 2단 유사도 게이팅이 rerank ON에서 스킵됨
# ─────────────────────────────────────────────
@pytest.mark.asyncio
async def test_e5_two_stage_gating_skipped_when_rerank_on() -> None:
    """rerank ON이면 e5 abs_threshold 게이팅을 건너뛴다(rerank 점수가 대체).

    청크 유사도(0.50)는 abs_threshold(0.84) 미만이라 e5 게이팅이 켜져 있었다면
    전체 드롭됐을 것이다. 그러나 rerank ON이면 e5 게이팅을 스킵하므로, 높은
    rerank 점수(0.95)를 받은 청크가 살아남아 주입된다.
    """
    chunk = _chunk("낮은유사도", 0.50, "e5 유사도는 낮지만 rerank 점수는 높은 청크")
    store = _make_mock_store(vector_results=[chunk])
    prov = _make_mock_provider(rerank_scores=[0.95])
    kr = KnowledgeRetriever(
        store=store,
        embedding_provider=prov,
        abs_threshold=0.84,  # e5 게이팅이 켜져 있었다면 0.50을 드롭했을 값
        relevance_margin=0.03,
        rerank_enabled=True,
        rerank_min_score=0.3,
    )

    out = await kr.get_context("e5 게이팅 스킵 확인")
    # e5 게이팅이 스킵됐으므로 청크가 살아남는다.
    assert "낮은유사도" in out
    assert "rr=0.95" in out


# ─────────────────────────────────────────────
# 7) rerank_top_k 컷 — 최종 상위 N개만 조립
# ─────────────────────────────────────────────
@pytest.mark.asyncio
async def test_rerank_top_k_limits_final_chunks() -> None:
    """게이팅을 통과한 후보가 많아도 상위 rerank_top_k개만 조립한다."""
    chunks = [
        _chunk("A", 0.90, "청크 A 본문"),
        _chunk("B", 0.80, "청크 B 본문"),
        _chunk("C", 0.70, "청크 C 본문"),
        _chunk("D", 0.60, "청크 D 본문"),
    ]
    store = _make_mock_store(vector_results=chunks)
    # 모두 min_score 이상. rerank 내림차순: A(0.99)>B(0.80)>C(0.70)>D(0.60)
    prov = _make_mock_provider(rerank_scores=[0.99, 0.80, 0.70, 0.60])
    kr = KnowledgeRetriever(
        store=store,
        embedding_provider=prov,
        rerank_enabled=True,
        rerank_top_k=2,  # 상위 2개만
        rerank_min_score=0.3,
    )

    out = await kr.get_context("top_k 컷 확인", max_tokens=5000)  # 예산 넉넉히
    assert "청크 A" in out
    assert "청크 B" in out
    assert "청크 C" not in out  # top_k=2 컷으로 제외
    assert "청크 D" not in out


# ─────────────────────────────────────────────
# 8) rerank + MMR 동시 활성 → 리랭커 우선(MMR 스킵)
# ─────────────────────────────────────────────
@pytest.mark.asyncio
async def test_rerank_priority_over_mmr() -> None:
    """rerank·MMR이 동시에 켜지면 리랭커가 우선하고 MMR은 스킵된다.

    검증: 벡터검색이 리랭커 파라미터(with_embedding=False)로 호출되는지로 확인한다.
    MMR이 우선이었다면 with_embedding=True로 임베딩을 함께 가져왔을 것이다.
    """
    a = _chunk("A", 0.90, "청크 A 본문")
    b = _chunk("B", 0.80, "청크 B 본문")
    store = _make_mock_store(vector_results=[a, b])
    prov = _make_mock_provider(rerank_scores=[0.90, 0.80])
    kr = KnowledgeRetriever(
        store=store,
        embedding_provider=prov,
        rerank_enabled=True,
        rerank_min_score=0.3,
        mmr_enabled=True,  # 동시 활성 — 그러나 리랭커가 우선
        mmr_fetch_k=30,
    )

    out = await kr.get_context("리랭커 우선 확인")

    kwargs = store.search_by_vector.call_args.kwargs
    # 리랭커 파라미터 사용(임베딩 미포함) — MMR 경로였다면 with_embedding=True였을 것.
    assert kwargs["with_embedding"] is False
    assert kwargs["top_k"] == 20  # rerank_fetch_k(=20), mmr_fetch_k(30) 아님
    prov.rerank.assert_awaited_once()
    assert "청크 A" in out and "청크 B" in out
