# -*- coding: utf-8 -*-
"""multilingual-e5-large 임베딩 + bge-reranker-v2-m3-ko 리랭커 서버.

Nexus 계약:
  - POST /v1/embed   {"texts": [...]}            -> {"embeddings": [[...]], "dimension": 1024}
  - POST /v1/rerank  {"query": ..., "documents": [...]} -> {"scores": [...]}  # documents 순서 유지

왜 한 서버에 합치나(설계):
  vLLM 네이티브 rerank는 별도 인스턴스 + GPU 메모리 선점이라 비효율(util 0.92 이미 점유).
  기존 임베딩 FastAPI에 CrossEncoder를 추가 로드해 재사용 — 운영 단순·메모리 효율.
  리랭커 로드 실패는 임베딩 기능을 막지 않는다(fail-open: _reranker=None → /v1/rerank만 비활성).
"""
import os

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

MODEL_ID = os.environ.get("EMBED_MODEL", "intfloat/multilingual-e5-large")
DEVICE = os.environ.get("EMBED_DEVICE", "cuda")

print(f"[embed] '{MODEL_ID}' 로드 시작 (device={DEVICE}) ...", flush=True)
_model = SentenceTransformer(MODEL_ID, device=DEVICE)
_dim = _model.get_sentence_embedding_dimension()
print(f"[embed] 로드 완료 — 차원={_dim}", flush=True)

# ── 리랭커(크로스인코더) — 지식 RAG 정밀도 향상용 ─────────────────────────────
# 왜 fp16인가: 0.6B 리랭커를 fp16으로 올려 GPU 메모리를 아낀다(품질 영향 미미,
# B200 잔여 여유가 넉넉하지 않아 안전 마진 확보). 로드 실패 시 임베딩은 그대로 동작.
RERANK_MODEL = os.environ.get("RERANK_MODEL", "dragonkue/bge-reranker-v2-m3-ko")
RERANK_ENABLED = os.environ.get("RERANK_ENABLED", "1") == "1"
_reranker = None
if RERANK_ENABLED:
    try:
        from sentence_transformers import CrossEncoder

        print(f"[rerank] '{RERANK_MODEL}' 로드 시작 (fp16) ...", flush=True)
        _reranker = CrossEncoder(
            RERANK_MODEL, device=DEVICE, model_kwargs={"torch_dtype": "float16"}
        )
        print("[rerank] 로드 완료", flush=True)
    except Exception as e:  # noqa: BLE001 — 리랭커 실패가 임베딩을 막으면 안 됨(fail-open)
        print(f"[rerank] 로드 실패(임베딩은 정상 동작): {e}", flush=True)
        _reranker = None

app = FastAPI(title="nexus-embed-e5")


class EmbedRequest(BaseModel):
    texts: list[str]


class RerankRequest(BaseModel):
    # (query, 각 document) 쌍의 관련도를 채점한다. documents 순서를 유지해 응답.
    query: str
    documents: list[str]


@app.post("/v1/embed")
def embed(req: EmbedRequest) -> dict:
    embs = _model.encode(
        req.texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        batch_size=32,
    )
    return {"embeddings": embs.tolist(), "dimension": int(embs.shape[1])}


@app.post("/v1/rerank")
def rerank(req: RerankRequest) -> dict:
    # 리랭커 미로드 시 빈 점수 + 에러표시(호출자가 폴백: 벡터순 유지).
    if _reranker is None:
        return {"scores": [], "error": "reranker not loaded"}
    if not req.documents:
        return {"scores": []}
    pairs = [[req.query, d] for d in req.documents]
    # predict 출력 스케일(로짓/시그모이드)은 모델 설정에 따르며, Nexus 측 min_score
    # 게이팅을 실측 분포로 캘리브레이션한다(여기선 원값 그대로 반환 — 단조성 보존).
    scores = _reranker.predict(pairs, convert_to_numpy=True, batch_size=32)
    return {"scores": [float(s) for s in scores]}


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model": MODEL_ID,
        "dimension": _dim,
        "device": DEVICE,
        "reranker": RERANK_MODEL if _reranker is not None else None,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8002, log_level="warning")
