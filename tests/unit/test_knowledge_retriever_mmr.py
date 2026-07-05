"""
KnowledgeRetriever MMR(Maximal Marginal Relevance) 리랭킹 단위 테스트 (2026-07-05).

검증 목표:
  1. mmr_enabled=False(기본) → 종전과 100% 동일: search_by_vector가 top_k /
     with_embedding=False로 호출되고, 결과 순서·조립이 변하지 않는다(무회귀).
  2. mmr_enabled=True → 게이팅을 통과한 survivors 중에서 중복(임베딩 거의 동일)
     후보는 하나만, 서로 다른(다양한) 후보가 선택된다.
  3. survivors 수가 top_k 이하이면 MMR을 건너뛴다(선별 불필요).
  4. MMR은 게이팅 '이후' 단계다 — 유사도 2단 게이팅(abs_threshold)과 엔티티
     매칭 게이팅이 MMR 앞에서 그대로 작동한다(할루시네이션 저감 로직 불훼손).

검증 전략:
  - KnowledgeStore는 MagicMock, search_by_vector/search_by_text는 AsyncMock.
    임베딩 벡터를 fixture로 직접 넣어 후보간 코사인을 결정론적으로 만든다.
  - 실제 DB / 임베딩 서버는 절대 호출하지 않는다(전부 mock).
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from core.rag.knowledge_retriever import KnowledgeRetriever


# ─────────────────────────────────────────────
# 헬퍼 — mock store / mock 임베딩 provider
# ─────────────────────────────────────────────
def _make_mock_store(vector_results: list[dict] | None = None) -> MagicMock:
    """search_by_vector가 주어진 결과를 돌려주는 mock store."""
    store = MagicMock()
    store.search_by_vector = AsyncMock(return_value=vector_results or [])
    store.search_by_text = AsyncMock(return_value=[])
    return store


def _make_mock_embedding(vec: list[float] | None = None) -> MagicMock:
    """embed()가 질의 임베딩 한 건을 돌려주는 mock provider."""
    emb = MagicMock()
    emb.embed = AsyncMock(return_value=[vec if vec is not None else [0.1, 0.1, 0.1]])
    return emb


# ─────────────────────────────────────────────
# 1) 무회귀 — mmr_enabled=False면 종전과 동일하게 호출·조립
# ─────────────────────────────────────────────
@pytest.mark.asyncio
async def test_mmr_disabled_calls_vector_search_with_topk_no_embedding() -> None:
    """MMR 비활성이면 search_by_vector가 top_k / with_embedding=False로 호출된다.

    하위 호환의 핵심: 넓은 후보 풀(fetch_k)도, 임베딩 반환도 요구하지 않아
    전송/파싱 오버헤드가 0이고 동작이 현재와 100% 동일하다.
    """
    a = {
        "source": "kowiki", "title": "청크A", "section": "", "similarity": 0.90,
        "content": "첫 번째 청크 본문.",
    }
    b = {
        "source": "kowiki", "title": "청크B", "section": "", "similarity": 0.80,
        "content": "두 번째 청크 본문.",
    }
    store = _make_mock_store(vector_results=[a, b])
    emb = _make_mock_embedding([1.0, 0.0, 0.0])
    # mmr_enabled 기본값(False) — 인자를 주지 않는다.
    kr = KnowledgeRetriever(store=store, embedding_provider=emb, top_k=5)

    out = await kr.get_context("두 청크 보여줘", max_tokens=1500)

    # search_by_vector는 top_k=self._top_k(5), with_embedding=False로 불려야 한다.
    kwargs = store.search_by_vector.await_args.kwargs
    assert kwargs["top_k"] == 5
    assert kwargs["with_embedding"] is False
    # 결과 순서/내용 불변 — 유사도 내림차순 그대로 조립된다.
    assert out.index("청크A") < out.index("청크B")


# ─────────────────────────────────────────────
# 2) MMR 활성 — 중복 후보 억제, 다양한 후보 선택
# ─────────────────────────────────────────────
@pytest.mark.asyncio
async def test_mmr_enabled_prefers_diverse_over_near_duplicate() -> None:
    """MMR 활성 시 중복(임베딩 거의 동일) 후보 대신 다양한 후보를 고른다.

    구성(top_k=2, λ=0.7):
      A: sim=0.90, emb=[1,0,0]  (관련도 최대 → 첫 선택)
      B: sim=0.89, emb=[1,0,0]  (A와 임베딩 동일 = 중복)
      C: sim=0.88, emb=[0,1,0]  (A와 직교 = 다양)
    2번째 선택 점수:
      B = 0.7*0.89 - 0.3*cos(B,A)=0.623-0.30 = 0.323
      C = 0.7*0.88 - 0.3*cos(C,A)=0.616-0.00 = 0.616  ← 승
    → 최종 [A, C], 중복 B는 탈락한다.
    """
    a = {
        "source": "kowiki", "title": "알파", "section": "", "similarity": 0.90,
        "content": "알파 청크 본문.", "embedding": [1.0, 0.0, 0.0],
    }
    b = {
        "source": "kowiki", "title": "베타", "section": "", "similarity": 0.89,
        "content": "베타 청크 본문(알파와 사실상 중복).", "embedding": [1.0, 0.0, 0.0],
    }
    c = {
        "source": "kowiki", "title": "감마", "section": "", "similarity": 0.88,
        "content": "감마 청크 본문(다양).", "embedding": [0.0, 1.0, 0.0],
    }
    store = _make_mock_store(vector_results=[a, b, c])
    emb = _make_mock_embedding([1.0, 0.0, 0.0])
    kr = KnowledgeRetriever(
        store=store,
        embedding_provider=emb,
        top_k=2,
        mmr_enabled=True,
        mmr_fetch_k=20,
        mmr_lambda=0.7,
    )

    out = await kr.get_context("알파와 감마 주제", max_tokens=1500)

    # 넓은 후보 풀 + 임베딩 반환으로 호출됐는지 확인.
    kwargs = store.search_by_vector.await_args.kwargs
    assert kwargs["top_k"] == 20
    assert kwargs["with_embedding"] is True
    # 관련도 최대 A와 다양한 C는 선택, 중복 B는 탈락.
    assert "알파" in out
    assert "감마" in out
    assert "베타" not in out
    # 임베딩 키가 주입 텍스트로 새어나가지 않아야 한다(오염 방지).
    assert "embedding" not in out
    assert "1.0" not in out


# ─────────────────────────────────────────────
# 3) survivors <= top_k → MMR 스킵
# ─────────────────────────────────────────────
@pytest.mark.asyncio
async def test_mmr_skipped_when_survivors_not_exceeding_topk() -> None:
    """게이팅 후 survivors가 top_k 이하이면 MMR 선별을 건너뛴다(그대로 통과)."""
    a = {
        "source": "kowiki", "title": "하나", "section": "", "similarity": 0.90,
        "content": "첫 청크.", "embedding": [1.0, 0.0, 0.0],
    }
    b = {
        "source": "kowiki", "title": "둘", "section": "", "similarity": 0.85,
        "content": "둘째 청크.", "embedding": [0.0, 1.0, 0.0],
    }
    store = _make_mock_store(vector_results=[a, b])
    emb = _make_mock_embedding([1.0, 0.0, 0.0])
    # top_k=5인데 survivors 2건 → 선별할 것이 없어 둘 다 유지.
    kr = KnowledgeRetriever(
        store=store, embedding_provider=emb, top_k=5, mmr_enabled=True,
    )

    out = await kr.get_context("두 청크 모두", max_tokens=1500)
    assert "하나" in out
    assert "둘" in out
    # 스킵 경로에서도 임베딩 키는 조립 전 제거된다.
    assert "embedding" not in out


# ─────────────────────────────────────────────
# 4) 게이팅은 MMR '앞'에서 그대로 작동
# ─────────────────────────────────────────────
@pytest.mark.asyncio
async def test_abs_threshold_gating_still_drops_before_mmr() -> None:
    """MMR 활성이라도 절대 임계 게이팅이 MMR 앞에서 전체 드롭한다.

    top1=0.824 < abs_threshold=0.84 → survivors 0건 → MMR에 도달하기 전에
    빈 문자열. 할루시네이션 게이팅이 MMR로 인해 훼손되지 않음을 검증한다.
    """
    chunk = {
        "source": "kowiki", "title": "무관", "section": "", "similarity": 0.824,
        "content": "무관한 일반 청크.", "embedding": [1.0, 0.0, 0.0],
    }
    store = _make_mock_store(vector_results=[chunk])
    emb = _make_mock_embedding([1.0, 0.0, 0.0])
    kr = KnowledgeRetriever(
        store=store,
        embedding_provider=emb,
        top_k=2,
        abs_threshold=0.84,
        relevance_margin=0.03,
        mmr_enabled=True,
    )

    out = await kr.get_context("무관 메타질문")
    assert out == ""


@pytest.mark.asyncio
async def test_entity_gating_still_drops_before_mmr() -> None:
    """MMR 활성이라도 엔티티(식별자) 게이팅이 MMR 앞에서 전체 드롭한다.

    질의 식별자 543이 어느 후보 content에도 없으면 survivors 0건 → 빈 문자열.
    """
    a = {
        "source": "kowiki", "title": "바흐", "section": "", "similarity": 0.90,
        "content": "요한 제바스티안 바흐 일반 설명.", "embedding": [1.0, 0.0, 0.0],
    }
    b = {
        "source": "kowiki", "title": "헨델", "section": "", "similarity": 0.88,
        "content": "게오르크 프리드리히 헨델 일반 설명.", "embedding": [0.0, 1.0, 0.0],
    }
    store = _make_mock_store(vector_results=[a, b])
    emb = _make_mock_embedding([1.0, 0.0, 0.0])
    kr = KnowledgeRetriever(
        store=store, embedding_provider=emb, top_k=1, mmr_enabled=True,
    )

    out = await kr.get_context("BWV 543 알려줘")
    assert out == ""
