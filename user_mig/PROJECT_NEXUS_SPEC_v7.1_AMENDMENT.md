# Project Nexus — 기술 사양서 v7.1 개정안

## 운영 정합화 + 인프라 운영 가이드

**버전**: 7.1 (Operational Reconciliation Amendment)
**기준 문서**: PROJECT_NEXUS_SPEC_v7.0_AMENDMENT.md
**개정일**: 2026-06-01
**상태**: 운영 발견 사항 반영본

---

## 개정 개요

### 왜 v7.1인가

v7.0은 RTX 5090 환경에 맞춰 4-Tier 체인을 유지하면서 멀티모델/티어/라우팅을
도입했다. 약 6주 운영 후 발견된 다음 항목들은 **사양서와 코드/운영의 분기점**
이라 별도 개정안으로 묶어 정리한다.

v7.1은 **신기능 도입이 아니다**. 다음 세 가지만 한다:
1. v6.1/v7.0 사양서와 실 운영이 어긋난 지점을 사실(현 운영)에 맞춰 갱신
2. 운영 인프라(컨테이너 이미지, 외부 진단) 권장 사항을 사양서에 명문화
3. v0.14.x 진행 동안 추가된 운영성 개선(임베딩 워밍업 등) 절차화

### 3가지 불변 전제 (v7.0 그대로 유지)

1. Claude Code 설계안 그대로 (4-Tier 체인, 24개 도구, 5계층 권한, Hook,
   Thinking, Memory 전부 유지)
2. GPU 업그레이드 시 성능 향상 (강한 GPU일수록 분리가 합쳐지며 원래 구조 수렴)
3. Claude Code 형태로 AI 작동 (도구 자율 선택, 스트리밍 응답, 멀티턴)

### 변경 범위 요약

| 구분 | 챕터/Part | 변경 수준 | 비고 |
|---|---|---|---|
| **갱신** | Ch 12 (Memory) | 중간 | 4테이블 분리 → 단일 `tb_memories`로 사실 갱신 |
| **갱신** | Ch 4.5 (Embedding Server) | 중간 | 커스텀 계약 명문화 (OpenAI 호환 아님) |
| **갱신** | v7.0 Part 4 (Scout) | 경미 | 모델 `gemma-4-4b-it` → `Qwen3.5-4B-Q4_K_M` |
| **신규** | Part 11 (인프라 운영 가이드) | 신규 | docutil-postgres 이미지 정책, NVML 등 |
| **신규** | Part 12 (운영성 절차) | 신규 | 임베딩 워밍업/keep-warm (v0.14.8) |
| **변경 없음** | 그 외 모든 챕터 | — | 4-Tier 체인, 권한, Hook, Thinking, Training 전부 불변 |

---

## Part 0: 변경 없는 챕터 (재확인)

v7.0과 동일하게 다음 챕터는 **코드 변경이 전혀 없다**.

| 챕터 | 모듈 | 이유 |
|---|---|---|
| Ch 1 | Executive Summary | 목표·제약·원칙 불변 |
| Ch 8 | 5계층 Permission Pipeline | 도구 실행 전 검증, 모델 무관 |
| Ch 9 | Security System | PathGuard·CommandFilter, 모델 무관 |
| Ch 10 | Hook System | 12+3 이벤트, 도구 실행 파이프라인에 연결 |
| Ch 11 | Thinking Engine | 5전략, Worker 모델에서만 사용 |
| Ch 13 | 24 Tools (구현체) | BaseTool 인터페이스, 각 도구 로직 |
| Ch 18 | Training Pipeline | 5-Phase 전략, LoRA/QLoRA |
| Ch 19 | Air-Gap Strategy | 오프라인 패키지, 무결성 검증 |
| Ch 21 | Project Structure | 디렉토리 구조 (신규 파일만 추가) |

---

## Part 1: 갱신 — Ch 12 Memory (4테이블 → 단일 `tb_memories`)

### 1.1 사실 갱신

v6.1 Ch 12는 메모리를 4개 테이블로 분리하여 정의했다:

```sql
-- v6.1 원안 (Ch 12, 라인 19149~19296)
CREATE TABLE episodic_memories  (... embedding vector(1024) ...);
CREATE TABLE semantic_memories  (... embedding vector(1024) ...);
CREATE TABLE procedural_memories(... embedding vector(1024) ...);
CREATE TABLE feedback_history   (... embedding vector(1024) ...);
```

운영 환경(2026-06-01 기준)은 단일 `tb_memories` 테이블에 `memory_type` 컬럼으로
4가지 종류를 구분하여 통합 운영한다.

```sql
-- 현 운영 정의 (2026-06-01 추출)
CREATE TABLE tb_memories (
    id            varchar(12) PRIMARY KEY,
    memory_type   varchar(50) NOT NULL,    -- 'episodic'|'semantic'|'procedural'|'feedback' 등
    content       text NOT NULL,
    key           varchar(255),
    tags          text[],
    importance    double precision,
    access_count  integer DEFAULT 0,
    created_at    timestamptz DEFAULT now(),
    last_accessed timestamptz DEFAULT now(),
    embedding     vector(1024),
    metadata      jsonb DEFAULT '{}'::jsonb,
    CONSTRAINT tb_memories_importance_check CHECK (importance BETWEEN 0.0 AND 1.0)
);
```

### 1.2 단일화 결정 근거

| 측면 | 4테이블 분리 (v6.1 원안) | 단일 tb_memories (현 운영) |
|---|---|---|
| 쿼리 복잡도 | UNION ALL 4건 + 정렬 | 단일 인덱스 스캔 |
| 인덱스 관리 | 4 × 6개 = 24개 | 6개 |
| 신규 메모리 타입 추가 | DDL 마이그레이션 | enum-like 컬럼값 추가만 |
| 운영 가시성 | 4개 따로 모니터링 | 단일 테이블 + `memory_type` 그룹화 |
| MemoryType 확장 (5종+α) | 테이블 증가 | 코드만 |

운영 200K+ 행 규모에서 단일 테이블 + `idx_memories_type` (btree)로 충분히
빠르며, 향후 새 MemoryType이 추가될 때마다 DDL 없이 코드만 수정 가능.

### 1.3 인덱스 정책

v6.1은 모든 메모리 테이블에 **ivfflat** 인덱스를 권장하고 hnsw는 주석으로만
표기했다. v7.1에서는 다음 운영 정책을 공식화한다.

| 조건 | 인덱스 선택 | 근거 |
|---|---|---|
| 행수 < 1,000 | **벡터 인덱스 없음** | seq scan이 더 빠름 (사양서 Ch 12 권장 그대로) |
| 1,000 ≤ 행수 < 100,000 | `ivfflat (lists=100)` | 빌드 비용 낮음, 일반 RAG 충분 |
| 행수 ≥ 100,000 | `hnsw` | 재현율·지연 모두 ivfflat 대비 우수 (운영 검증) |

운영 적용 현황:

| 테이블 | 행수 (2026-06-01) | 인덱스 |
|---|---|---|
| `tb_knowledge` | 1,067,978 | `idx_knowledge_embed` ivfflat lists=100 |
| `tb_memories` | 203,991 | `idx_memories_embedding` **hnsw** |
| `tb_symbols` | 2,118 | `idx_symbols_embed` ivfflat lists=100 (2026-06-01 추가) |

### 1.4 스키마 책임 — `LongTermMemory.ensure_schema()`

v7.0까지 `core/memory/long_term.py`에는 `CREATE TABLE`/`CREATE INDEX` 코드가
없었다 (운영 DB는 외부 수동 생성). v7.1부터는 다음을 코드가 보장한다.

```python
# core/memory/long_term.py — LongTermMemory
async def ensure_schema(self) -> None:
    """tb_memories 스키마·인덱스를 멱등 보장 (pg_pool=None이면 no-op)."""
    if self._pg is None:
        return
    async with self._pg.acquire() as conn:
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        await conn.execute(_DDL_TB_MEMORIES)
        for ddl in _DDL_TB_MEMORIES_INDEXES:
            await conn.execute(ddl)
```

호출 위치: `core/bootstrap.py` Phase 2 ③ MemoryManager 초기화 직후.
`KnowledgeStore.ensure_schema()` / `SymbolStore.ensure_schema()`와 동일 패턴.

이로써 **새 PG 인스턴스(컨테이너 교체·재해 복구·CI)** 에서도 동일 정의가
자동 재현된다.

---

## Part 2: 갱신 — Ch 4.5 Embedding Server (커스텀 계약)

v6.1 Ch 4.5는 e5-large 임베딩 서버를 always-on으로 명시했지만, OpenAI 호환
경로(`/v1/embeddings`)를 가정하는 듯한 표현이 있다. 실 구현은 **커스텀 경량
계약**이며 v7.1에서 명문화한다.

### 2.1 실 가동 사양

**파일**: `/opt/nexus-gpu/embedding_server.py` (FastAPI, sentence-transformers)
**프로세스**: PID 1087 (2026-06-01 기준 46일+ 무중단)
**위치**: 192.168.21.112:8002
**디바이스**: CPU (`device="cpu"` — GPU VRAM 절약)

### 2.2 API 계약 (확정)

```http
POST /v1/embed
Content-Type: application/json

{"texts": ["query: 검색하려는 문장", "passage: 적재되는 문장"]}
```

응답:
```json
{
  "embeddings": [[0.012, -0.034, ...], [...]],
  "dimension": 1024
}
```

헬스체크:
```http
GET /health
→ {"status": "healthy", "model": "multilingual-e5-large", "dimension": 1024}
```

### 2.3 호출 코드 — `ModelProvider.embed()`

```python
# core/model/inference.py
async def embed(self, texts: list[str]) -> list[list[float]]:
    response = await self._client.post(
        f"{self._embedding_base_url}/v1/embed",
        json={"texts": texts},
        headers={"Authorization": f"Bearer {self.api_key}"},
    )
    ...
```

### 2.4 왜 OpenAI 호환이 아닌가

- 임베딩 서버는 단일 모델(e5-large) 전용이라 모델 라우팅 불필요
- e5-large는 입력에 `"query: "` 또는 `"passage: "` 접두가 권장 — 텍스트
  배열 그대로 받는 게 일관됨
- 외부 SDK 호환성보다 사내 코드 일관성 우선 (에어갭 원칙)
- vLLM 임베딩 모드(OpenAI 호환)는 GPU 사용 — CPU 전용 sentence-transformers
  대비 VRAM 비용·복잡도 더 큼

### 2.5 진단 도구

`scripts/_verify_pgvector.py --search "<질의>"` 가 임베딩 서버 호출 + 벡터
검색까지 e2e 검증한다.

---

## Part 3: 갱신 — Scout 모델 (Qwen3.5-4B로 진화)

### 3.1 사실 갱신

v7.0 Part 4는 Scout 모델로 `gemma-4-4b-it`를 명시했다. 실 운영은 **`Qwen3.5-4B`** (Q4_K_M GGUF)를 사용한다.

```bash
# 현 운영 (192.168.21.112, PID 26253)
/opt/nexus-gpu/llama.cpp/llama-b8808/llama-server \
  --model /opt/nexus-gpu/models/qwen3.5-4b-gguf/Qwen3.5-4B-Q4_K_M.gguf \
  --host 0.0.0.0 --port 8003 \
  --ctx-size 4096 --threads 8 --batch-size 512 \
  --api-key local-key --jinja
```

### 3.2 진화 근거

| 항목 | gemma-4-4b-it (v7.0 원안) | Qwen3.5-4B-Q4_K_M (현 운영) |
|---|---|---|
| 한국어 성능 | 보통 | 우수 (Worker(Qwen3.5-27B)와 같은 패밀리) |
| Worker와 토크나이저 | 다름 → 변환 필요 | **동일** → tool_calls 형식 호환 |
| Phase 3 LoRA 정합성 | 별도 토크나이저 | Worker LoRA와 동일 베이스 모델 패밀리 |
| 메모리 footprint (Q4) | ~2.5GB | ~2.7GB |

Worker(Qwen3.5-27B Phase 3 LoRA)와 같은 모델 패밀리로 통일하여 토크나이저·
프롬프트 템플릿·도구 호출 형식을 일치시킨 결정.

### 3.3 `scout:` 섹션 yaml 표기 정책

v7.0 Part 4는 `nexus_config.yaml`에 명시적 `scout:` 섹션을 둘 것을 권장했다.
현 운영은 `core/config.py::ScoutConfig`의 기본값(`default_factory=ScoutConfig`)
으로 동작하며 yaml에 명시 섹션이 없다.

v7.1 정책: **yaml 명시는 운영자 옵션** — 기본 동작은 코드 기본값으로 충분하며,
override가 필요할 때만 yaml에 섹션 추가한다.

```yaml
# config/nexus_config.yaml (선택)
scout:
  enabled: true
  base_url: "http://192.168.21.112:8003"
  model_id: "qwen3.5-4b"
  max_context_tokens: 4096
  max_output_tokens: 512
```

---

## Part 4: 신규 — Part 11 인프라 운영 가이드

### 4.1 docutil-postgres 이미지 정책

**현 운영**: `pgvector/pgvector:pg17` (Debian 기반, pgvector v0.8.0 포함)

**과거 사건 (2026-05-27 추정)**: docutil-postgres가 순정 `postgres:17-alpine`
이미지로 교체되어 pgvector 동적 라이브러리(.so)가 사라졌고, 5일간 모든
vector 연산이 `$libdir/vector: No such file or directory`로 실패. RAG/장기
메모리/심볼 검색 전부 정지.

**복구 (2026-06-01)**: `pgvector/pgvector:pg17`로 교체, 데이터 볼륨 그대로
유지. 5초 다운타임으로 완전 정상화. 데이터 무결성 100%.

**정책**:
1. PG 컨테이너 이미지는 **`pgvector/pgvector:pg17` 또는 동등한 pgvector 사전
   설치 이미지만 허용** — 순정 `postgres:*` 이미지로 절대 교체 금지
2. compose 파일 변경 시 review에서 image 라인 점검 필수
3. 정기 점검: `python scripts/_verify_pgvector.py` 로 vector 연산 정상 확인

### 4.2 nexus DB 자격 정책

`config/nexus_config.yaml`의 `postgresql.password = "idino@12"` 가 정. URL
인코딩 시 `%40` 1회만 사용(`idino%4012`).

과거 오타 (`prepare_kowiki.py` 기본값 `idino%40%4012` = `idino@@12`)는 2026-06-01 수정 완료.

### 4.3 NVML driver/library mismatch (GPU 서버)

**현 상태 (2026-06-01)**: `192.168.21.112`에서 `nvidia-smi` 실행 시
`Failed to initialize NVML: Driver/library version mismatch` (NVML library v580.159).

**영향**:
- 추론(vLLM, llama.cpp) 자체는 정상 동작 중 — CUDA 런타임은 별도
- 운영 모니터링(GPU 메모리·사용률 추적) 불가
- `_diag_all_services.py` 의 GPU 메모리 표시 불가

**복구 방향**: nvidia driver 또는 nvidia-utils 패키지 버전 정합 (운영자 작업).
긴급도는 낮음 (추론 영향 없음). 별도 정기 작업 항목.

### 4.4 진단 스크립트 목록

| 스크립트 | 용도 |
|---|---|
| `scripts/_verify_pgvector.py` | pgvector + 벡터 테이블 + 실 KNOWLEDGE e2e |
| `scripts/_diag_all_services.py` | 웹/GPU/DB 전체 도달성 |
| `scripts/_diag_rag_latency*.py` | tb_memories 검색 EXPLAIN ANALYZE |
| `scripts/_ssh_kowiki_status.py` | kowiki tmux 적재 상태 |

전부 SSH(paramiko) 기반 LAN-only, 에어갭 준수.

---

## Part 5: 신규 — Part 12 운영성 절차 (v0.14.8 임베딩 워밍업)

### 5.1 배경

e5-large 임베딩 서버(:8002)는 idle 시 다음 호출에서 모델·CUDA 컨텍스트
워밍업으로 ~60초 cold start. 사용자의 첫 KNOWLEDGE 질의가 이 비용을 그대로
떠안는 일이 발견됨 (v0.14.6 운영 보고).

### 5.2 도입 (v0.14.8, 2026-04-27)

**A. 부트스트랩 워밍업** — `_warmup_embedding(provider, retries=2)`
- 부트스트랩 직후 fire-and-forget 1회 호출 + 짧은 backoff 2회 재시도
- 실패는 WARNING만 (본류 응답 무영향)

**B. 주기적 keep-warm** — `_embedding_keepalive(provider, interval_sec=300)`
- 5분 간격 ping → cold 상태로 떨어지지 않게 유지
- `asyncio.CancelledError`로 깨끗 종료
- 핸들은 `components["embedding_keepalive_task"]`로 노출, lifespan 종료 시 cancel + await

### 5.3 사용자 가시화

`cli/repl.py::_process_message`에 Rich `Console.status()` 통합:
- `STREAM_REQUEST_START` → "모델 추론 중..."
- `MESSAGE_START` → "응답 생성 중..."
- `THINKING_START` → "생각 정리 중..."
- `TOOL_USE_START` → "{tool_name} 실행 중..."
- `TEXT_DELTA` 첫 도착 → spinner 닫고 본문 출력

설사 keep-warm가 실패해서 idle로 떨어진 경우에도 사용자는 "지식 검색 중..."
스피너로 진행 인지.

---

## Part 6: 검증 — 설계 일관성 (v7.0 Part 8 갱신)

v7.0 Part 8의 무결성 검증(4-Tier 체인/도구/권한/Hook/Thinking·Memory/Training)은
모두 v7.1에서도 그대로 유지된다. v7.1 변경은 **운영 데이터 정의·인프라 운영
정책**에 국한되며 다음 어떤 무결성도 깨지 않는다.

| 영역 | 영향 |
|---|---|
| 4-Tier AsyncGenerator 체인 | 영향 없음 |
| 24개 도구 | 영향 없음 |
| 5계층 권한 | 영향 없음 |
| Hook 시스템 | 영향 없음 |
| Thinking Engine | 영향 없음 |
| Memory (Ch 12) | DDL 책임만 코드로 이동 — 호출 인터페이스 불변 |
| Training | 영향 없음 |
| Air-gap | 영향 없음 (운영 권장 이미지 pgvector는 LAN 내 docker hub 사전 캐싱 가능) |

---

## 부록 A: 06-01 운영 점검에서 정정된 사항

| # | 항목 | 사실 | 위치 |
|---|---|---|---|
| 1 | tb_memories 테이블 수 | 단일 (4개 분리 아님) | Part 1 |
| 2 | tb_memories 인덱스 종류 | hnsw (ivfflat 아님) | Part 1.3 |
| 3 | LongTermMemory DDL | 코드가 보장 (이전엔 외부 수동) | Part 1.4 |
| 4 | tb_symbols 벡터 인덱스 | 06-01 추가 (이전엔 부재) | Part 1.3 |
| 5 | Scout 모델 | Qwen3.5-4B (gemma 아님) | Part 3 |
| 6 | 임베딩 서버 계약 | 커스텀 `/v1/embed` | Part 2 |
| 7 | scout: yaml 섹션 | 코드 기본값 우선 | Part 3.3 |
| 8 | docutil-postgres 이미지 | `pgvector/pgvector:pg17` 정책 | Part 4.1 |
| 9 | NVML mismatch | 추적 중 (추론 무영향) | Part 4.3 |
| 10 | 임베딩 워밍업/keep-warm | 도입 절차화 | Part 5 |

---

## 부록 B: 코드 위치 매트릭스 (v7.1 신규 ↔ 갱신)

| 사양서 항목 | 코드 위치 | 비고 |
|---|---|---|
| Part 1.4 `LongTermMemory.ensure_schema` | `core/memory/long_term.py` | 2026-06-01 추가 |
| Part 1.4 bootstrap 호출 | `core/bootstrap.py` Phase 2 ③ | 2026-06-01 추가 |
| Part 1.3 tb_symbols 조건부 인덱스 | `core/rag/symbol_indexer.py::background_index` | 2026-06-01 추가 |
| Part 2 임베딩 호출 | `core/model/inference.py::ModelProvider.embed` | 이미 존재 (계약만 명문화) |
| Part 3 Scout 모델 ID | `core/config.py::ScoutConfig` | 코드 기본값 |
| Part 5 워밍업/keepalive | `core/bootstrap.py::_warmup_embedding/_embedding_keepalive` | v0.14.8 |
| Part 4.4 진단 스크립트 | `scripts/_verify_pgvector.py` 등 | 2026-06-01 신규/기존 |

---

*작성일: 2026-06-01*
*기준 운영 점검: tb_knowledge 1,067,978행 / tb_memories 203,991행 / tb_symbols 2,118행*
