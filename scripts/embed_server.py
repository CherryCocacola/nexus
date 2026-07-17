"""multilingual-e5-large 임베딩 + bge-reranker-v2-m3-ko 리랭커 서버.

이 파일은 Nexus의 "지식 RAG"가 사용하는 경량 추론 서버다. 두 가지 일을 한다.
  1) 임베딩: 입력 문장들을 1024차원 벡터로 변환(pgvector 검색용).
  2) 리랭킹: (질의, 문서) 쌍의 관련도를 다시 채점해 검색 정밀도를 끌어올린다.
FastAPI로 아주 얇은 HTTP API만 노출하며, 127.0.0.1:8002에서 로컬로만 뜬다(에어갭 준수).

Nexus 계약(호출자와 약속한 요청/응답 형태 — 절대 깨지 않는다):
  - POST /v1/embed   {"texts": [...]}            -> {"embeddings": [[...]], "dimension": 1024}
  - POST /v1/rerank  {"query": ..., "documents": [...]} -> {"scores": [...]}  # documents 순서 유지

왜 한 서버에 합치나(설계):
  vLLM 네이티브 rerank는 별도 인스턴스 + GPU 메모리 선점이라 비효율(util 0.92 이미 점유).
  기존 임베딩 FastAPI에 CrossEncoder를 추가 로드해 재사용 — 운영 단순·메모리 효율.
  리랭커 로드 실패는 임베딩 기능을 막지 않는다(fail-open: _reranker=None → /v1/rerank만 비활성).

구성 요소 한눈에:
  - EmbedRequest / RerankRequest : 요청 본문 스키마(Pydantic).
  - embed() / rerank()           : 실제 추론 엔드포인트.
  - health()                     : 헬스체크(모델·차원·디바이스·리랭커 상태 반환).
  - 모듈 로드 시점에 _model(임베더)과 _reranker(선택)를 미리 GPU에 올려둔다.

작성자: 이현수 / 작성일: 2026-07-05
"""
import os

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

# 환경변수로 모델 ID와 디바이스를 주입받는다(하드코딩 회피). 기본값은 e5-large + CUDA.
MODEL_ID = os.environ.get("EMBED_MODEL", "intfloat/multilingual-e5-large")
DEVICE = os.environ.get("EMBED_DEVICE", "cuda")

# 임베딩 모델은 서버 기동 시 딱 한 번 GPU로 로드한다(요청마다 로드하면 너무 느림).
print(f"[embed] '{MODEL_ID}' 로드 시작 (device={DEVICE}) ...", flush=True)
_model = SentenceTransformer(MODEL_ID, device=DEVICE)
# 벡터 차원(=1024)을 모델에서 직접 조회해 응답/헬스체크에 그대로 실어 보낸다.
_dim = _model.get_sentence_embedding_dimension()
print(f"[embed] 로드 완료 — 차원={_dim}", flush=True)

# ── 리랭커(크로스인코더) — 지식 RAG 정밀도 향상용 ─────────────────────────────
# 왜 device별 dtype인가: GPU(cuda)에서는 0.6B 리랭커를 fp16으로 올려 메모리를 아끼지만
# (품질 영향 미미), CPU에서는 fp16이 비효율/비호환(대부분 연산이 fp32로 폴백되거나
# 예외)이라 fp32로 로드한다. 로드 실패 시 임베딩은 그대로 동작(fail-open).
RERANK_MODEL = os.environ.get("RERANK_MODEL", "dragonkue/bge-reranker-v2-m3-ko")
# 리랭커 디바이스는 임베딩과 별도로 지정할 수 있다(기본 = 임베딩 디바이스).
# 예: 112는 GPU를 vLLM이 점유하므로 임베딩·리랭커 모두 CPU로 둔다(EMBED/RERANK_DEVICE=cpu).
RERANK_DEVICE = os.environ.get("RERANK_DEVICE", DEVICE)
# RERANK_ENABLED="0"으로 두면 리랭커 자체를 로드하지 않는다(임베딩 전용 모드로 가볍게 운영).
RERANK_ENABLED = os.environ.get("RERANK_ENABLED", "1") == "1"
# _reranker는 "미로드" 상태를 None으로 표현한다. 아래에서 성공 시에만 실제 객체로 채운다.
_reranker = None
if RERANK_ENABLED:
    try:
        # CrossEncoder는 임베딩과 달리 (질의,문서)를 함께 넣어 단일 관련도 점수를 낸다.
        from sentence_transformers import CrossEncoder

        # CPU면 fp32, CUDA면 fp16(위 설계 주석 참조). fp16 강제는 CPU에서 오류·저속의 원인.
        _rr_kwargs = {"torch_dtype": "float16"} if RERANK_DEVICE == "cuda" else {}
        _rr_dtype = "fp16" if RERANK_DEVICE == "cuda" else "fp32"
        print(f"[rerank] '{RERANK_MODEL}' 로드 시작 ({RERANK_DEVICE}, {_rr_dtype}) ...", flush=True)
        _reranker = CrossEncoder(
            RERANK_MODEL, device=RERANK_DEVICE, model_kwargs=_rr_kwargs
        )
        print("[rerank] 로드 완료", flush=True)
    except Exception as e:  # noqa: BLE001 — 리랭커 실패가 임베딩을 막으면 안 됨(fail-open)
        # 어떤 이유로든 리랭커 로드가 깨지면, 로그만 남기고 _reranker=None으로 되돌린다.
        # 그 결과 /v1/embed는 계속 정상, /v1/rerank만 "미로드" 응답을 주게 된다.
        print(f"[rerank] 로드 실패(임베딩은 정상 동작): {e}", flush=True)
        _reranker = None

# FastAPI 앱 인스턴스. 아래 데코레이터들이 이 app에 라우트를 등록한다.
app = FastAPI(title="nexus-embed-e5")


class EmbedRequest(BaseModel):
    """POST /v1/embed 요청 본문. 임베딩할 문장 리스트를 담는다."""

    texts: list[str]


class RerankRequest(BaseModel):
    """POST /v1/rerank 요청 본문.

    query 하나에 대해 여러 documents의 관련도를 채점한다. 응답 scores는
    반드시 입력 documents와 같은 순서·같은 길이로 돌려주는 것이 호출자와의 계약이다.
    """

    # (query, 각 document) 쌍의 관련도를 채점한다. documents 순서를 유지해 응답.
    query: str
    documents: list[str]


@app.post("/v1/embed")
def embed(req: EmbedRequest) -> dict:
    """입력 문장들을 정규화된 1024차원 벡터로 변환해 반환한다.

    흐름: texts를 배치(32개씩) 인코딩 → L2 정규화 → numpy → 리스트로 직렬화.
    normalize_embeddings=True로 코사인 유사도를 내적으로 계산할 수 있게 맞춘다.
    반환: {"embeddings": 2차원 리스트, "dimension": 벡터 차원(int)}.
    """
    embs = _model.encode(
        req.texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        batch_size=32,
    )
    return {"embeddings": embs.tolist(), "dimension": int(embs.shape[1])}


@app.post("/v1/rerank")
def rerank(req: RerankRequest) -> dict:
    """(query, 각 document) 쌍을 리랭커로 채점해 관련도 점수 리스트를 반환한다.

    가장자리 케이스를 먼저 처리한 뒤 실제 채점을 수행한다.
      - 리랭커 미로드: 빈 scores + error 표시(호출자는 벡터 검색 순서로 폴백).
      - documents가 비었으면: 빈 scores.
    반환 scores는 documents와 동일한 순서/길이를 보장한다.
    """
    # 리랭커 미로드 시 빈 점수 + 에러표시(호출자가 폴백: 벡터순 유지).
    if _reranker is None:
        return {"scores": [], "error": "reranker not loaded"}
    # 문서가 하나도 없으면 채점할 게 없으니 빈 리스트로 즉시 반환.
    if not req.documents:
        return {"scores": []}
    # CrossEncoder 입력 형식: [[query, doc1], [query, doc2], ...] 쌍의 리스트.
    pairs = [[req.query, d] for d in req.documents]
    # predict 출력 스케일(로짓/시그모이드)은 모델 설정에 따르며, Nexus 측 min_score
    # 게이팅을 실측 분포로 캘리브레이션한다(여기선 원값 그대로 반환 — 단조성 보존).
    scores = _reranker.predict(pairs, convert_to_numpy=True, batch_size=32)
    # numpy 스칼라를 파이썬 float으로 바꿔 JSON 직렬화가 깨지지 않게 한다.
    return {"scores": [float(s) for s in scores]}


@app.get("/health")
def health() -> dict:
    """헬스체크. 로드된 모델·차원·디바이스와 리랭커 활성 여부를 알려준다.

    운영/기동 스크립트가 이 엔드포인트로 서버 준비 상태를 확인한다.
    reranker는 로드 성공 시 모델 이름, 미로드 시 None으로 내려간다.
    """
    return {
        "status": "ok",
        "model": MODEL_ID,
        "dimension": _dim,
        "device": DEVICE,
        "reranker": RERANK_MODEL if _reranker is not None else None,
    }


# 스크립트를 직접 실행하면(예: python embed_server.py) uvicorn으로 서버를 띄운다.
# 바인드 호스트: 기본 127.0.0.1(로컬 전용·에어갭 안전 기본값). 임베딩 서버를 LAN에서
# 호출하는 배포(예: 112 — 웹 컨테이너·오케스트레이터가 LAN IP 192.168.x로 접근)에서는
# EMBED_HOST=0.0.0.0으로 연다(LAN 바인딩은 에어갭 위반 아님 — 외부망 호출만 금지).
# 포트 8002·로그는 warning 이상만 출력.
if __name__ == "__main__":
    HOST = os.environ.get("EMBED_HOST", "127.0.0.1")
    uvicorn.run(app, host=HOST, port=8002, log_level="warning")
