# Project Nexus — 개발 진행 상황

> 이 파일은 세션 간 진행 상황 공유를 위해 매 작업마다 업데이트된다.

---

## 현재 상태

- **현재 Phase**: v0.14.8 (코드) / 인프라 복구 06-01
- **마지막 업데이트**: 2026-06-01
- **브랜치**: main
- **총 테스트**: 811 passed / 1 skipped (단위 + 통합, e2e 제외)
- **전체 파일**: ~170개 Python/HTML 모듈

---

## 인프라 복구 — pgvector 라이브러리 누락 해소 (2026-06-01)

### 배경 — 발견 경위

사양서 잔여 작업(kowiki `--build-index` 메모리 잔류) 확인 차 DB 점검 중,
`192.168.10.39:5440` docutil-postgres에서 모든 vector 연산이
`could not access file "$libdir/vector": No such file or directory`로 실패.
`pg_extension`에 `vector v0.8.0` 등록은 되어 있으나 동적 라이브러리(.so) 파일이
사라진 상태. 영향 범위:

- `tb_knowledge` (kowiki, ~1.07M행) — KNOWLEDGE 라우팅 벡터 검색 실패 (ILIKE 폴백만)
- `tb_memories` (장기 메모리, ~20.4만행) — 의미검색/승격 정지
- `tb_symbols` (Phase 10 심볼) — SymbolSearch 정지

원인: docutil-postgres가 5일 전부터 `postgres:17-alpine` (pgvector 미포함) 이미지로
교체 운영 중. 컨테이너 재빌드 시점에 누군가 순정 alpine 이미지로 바꿔치기.

### 조치 — 옵션 A (이미지 교체)

- `alpine` repo에 `postgresql17-pgvector` 패키지 없음 확인 → 옵션 B(apk add) 폐기
- 외부 docker hub 접근 가능 확인 → 공식 `pgvector/pgvector:pg17`로 교체

```
1. docker pull pgvector/pgvector:pg17                          (157MB, 2026-05-15 빌드)
2. /home/idino/docutil/docker-compose.yml 백업 (.bak_20260601_pgvector)
3. 252라인 단 한 줄 수정:
   image: postgres:17-alpine  →  image: pgvector/pgvector:pg17
4. docker compose config 유효성 확인
5. docker compose up -d postgres  (recreate)
6. healthy 확인 — 즉시 통과 (start_period 15s 이내)
```

다른 시도 없음. command/healthcheck/볼륨/포트/환경변수 모두 그대로 유지.

### 검증 — `scripts/_verify_pgvector.py` 신규

읽기 전용 진단 스크립트. 기본 vector 연산(type cast/cosine/L2) + 3개 테이블의
실제 COUNT(*) + 인덱스 메타데이터까지 한 번에 점검.

**복구 후 결과**:
- ✓ vector 기본 연산 3종 전부 OK
- ✓ tb_knowledge exact COUNT(*) = **1,067,978** (04-24 적재 종료 기록과 정확 일치 → 데이터 무결성 100%)
- ✓ tb_memories exact COUNT(*) = 203,991
- ✓ tb_symbols exact COUNT(*) = 2,118
- ✓ idx_knowledge_embed (ivfflat), idx_memories_embedding (hnsw) 모두 존속

**부수 변화** (안전):
- PG 17.9 → 17.10 마이너 업그레이드 (이미지 베이스 차이, 데이터 호환)
- 컨테이너 OS Alpine → Debian (pgvector 공식 이미지 베이스)

### 부수 발견 (별도 작업 후보)

| 항목 | 상태 |
|---|---|
| `tb_memories` 인덱스 `hnsw` | 코드의 `DDL_IVFFLAT` 클래스 상수와 불일치 — 어느 시점에 수동/스크립트로 hnsw로 만든 것 추정. 정합성 점검 필요 |
| `tb_symbols` 벡터 인덱스 부재 | 행수 2,118로 작아 시급도 낮지만 SymbolSearch 향후 확장 시 필요 |
| ~~임베딩 서버 404~~ | **오진 정정**: 서버 정상, 커스텀 `POST /v1/embed`({"texts":[...]}) 계약. verify 스크립트만 수정 후 KNOWLEDGE e2e (sim≥0.86×5건) 통과 |
| `prepare_kowiki.py` DSN 오타 | `idino%40%4012`(=idino@@12) → `idino%4012`(=idino@12) 수정 완료 |

### 신규/수정 파일

```
신규:
  scripts/_verify_pgvector.py        (읽기 전용 진단 스크립트)

수정:
  scripts/prepare_kowiki.py          (DSN 기본값 오타 수정)
  user_mig/progress/progress.md      (이 항목)

원격 (192.168.10.39):
  /home/idino/docutil/docker-compose.yml (252라인 1줄 수정)
  /home/idino/docutil/docker-compose.yml.bak_20260601_pgvector (백업)
```

### 사양서 정합성

이번 변경은 운영 인프라 복구로 v6.1/v7.0 어느 챕터도 변경하지 않음. 단,
`tb_memories`의 hnsw 인덱스 불일치는 사양서 Ch 12(Memory) 명시와 충돌
가능성 있어 별도 점검 후보로 기록.

---

## v7.1 AMENDMENT 작성 + 운영 DB tb_symbols 인덱스 적용 (2026-06-01)

### 운영 DB 즉시 적용

- `docutil-postgres`에 `idx_symbols_embed` (ivfflat, lists=100) 신규 생성
- 실행: `CREATE INDEX IF NOT EXISTS ... USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)`
- 소요: 1.52초, 인덱스 크기 17MB
- verify: `_verify_pgvector.py` → 3개 테이블 모두 `vector_idx=O`

### v7.1 AMENDMENT 신규 작성

신규 파일: `user_mig/PROJECT_NEXUS_SPEC_v7.1_AMENDMENT.md`

목적: v7.0 이후 6주 운영에서 발견된 **사양서 ↔ 코드/운영 분기점** 정리.
신기능 도입 아님. 10가지 정정 사항을 Part 1~6 + 부록 A/B로 정리:

| Part | 내용 |
|---|---|
| 1 | Ch 12 Memory — 4테이블 분리 → 단일 `tb_memories`, hnsw 정책, `ensure_schema` 책임 |
| 2 | Ch 4.5 Embedding Server — 커스텀 `/v1/embed` 계약 명문화 (OpenAI 호환 아님) |
| 3 | Scout 모델 — gemma-4-4b-it → Qwen3.5-4B-Q4_K_M, yaml scout 섹션 정책 |
| 4 | 인프라 운영 가이드 — docutil-postgres 이미지 정책, NVML, 진단 스크립트 목록 |
| 5 | 운영성 절차 — 임베딩 워밍업/keep-warm (v0.14.8) 절차화 |
| 6 | 설계 일관성 — 4-Tier 체인·권한·Hook 등 영향 없음 재확인 |
| 부록 A | 10가지 정정 사항 표 |
| 부록 B | 사양서 ↔ 코드 위치 매트릭스 |

### 사양서 정합성

v7.1은 **운영 데이터 정의·인프라 정책**만 갱신하며 4-Tier 체인·도구·권한·
Hook·Thinking·Memory 호출 인터페이스 어떤 무결성도 깨지 않음.

### 신규/수정 파일

```
신규:
  user_mig/PROJECT_NEXUS_SPEC_v7.1_AMENDMENT.md

수정:
  user_mig/progress/progress.md (이 항목)

원격 (192.168.10.39 docutil-postgres):
  tb_symbols.idx_symbols_embed 인덱스 신규 생성
```

---

## tb_symbols ivfflat 인덱스 조건부 자동 빌드 (2026-06-01)

### 배경 — 사양서/코드/운영 일관성 점검

pgvector 복구 점검 중 발견:
- 코드(`core/rag/symbol_store.py`)는 `_DDL_IVFFLAT`을 정의 → `PgVectorStore.build_vector_index()`로 호출 가능
- 그러나 어디서도 호출 안 함 (`bootstrap`에서 `ensure_schema()`만 호출)
- 결과: 운영 DB `tb_symbols`(2,118행)에 **벡터 인덱스 부재**
- KnowledgeStore는 `prepare_kowiki.py --build-index` 수동 실행으로 인덱스
  보유 — 같은 패턴이 SymbolStore에는 없었던 것

### 구현

**`core/rag/symbol_indexer.py::background_index`** 확장:
- 초기 인덱싱 완료 후 행수 ≥ 1000일 때만 `store.build_vector_index()` 호출
- 임계치 근거: 사양서 Ch 12 "ivfflat is recommended when 1000+ records"
- 적은 데이터에서 빌드하면 lists=100 통계 부족으로 검색 품질 저하
- `IF NOT EXISTS`로 멱등 — 인덱스 이미 있으면 no-op
- 빌드 실패는 본류 응답에 무영향(WARNING만)

### 테스트 (`tests/unit/test_symbol_indexer.py` 신규 3건)

- `test_background_index_skips_vector_index_below_threshold` — 5건 << 1000 → 미호출
- `test_background_index_builds_vector_index_above_threshold` — count mock 2000 → 호출
- `test_background_index_swallows_indexing_errors` — index_project 실패 시도 무중단

회귀: 단위 19/19 (test_symbol_indexer) → 전체 **733 passed** (+3 신규).

### 운영 영향

- 다음 nexus 부트스트랩 시 자동으로 ivfflat 인덱스 생성됨 (현재 2,118행 > 1000 임계치)
- 즉시 적용하려면 운영 DB에서 1줄 DDL 수동 실행 가능:
  ```sql
  CREATE INDEX IF NOT EXISTS idx_symbols_embed
    ON tb_symbols USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
  ```

### 신규/수정 파일

```
수정:
  core/rag/symbol_indexer.py          (background_index에 조건부 빌드)
  tests/unit/test_symbol_indexer.py   (신규 3건 + background_index import)
  user_mig/progress/progress.md       (이 항목)
```

### 사양서 영향 없음

코드 기준 정의가 그대로 보존됨. v6.1 Ch 12 "1000+ records" 권고 그대로 적용.

---

## tb_memories 스키마/인덱스 정합화 — `LongTermMemory.ensure_schema()` 추가 (2026-06-01)

### 배경 — 3중 불일치 발견

pgvector 복구(위 항목) 후 후속 점검에서 다음 사실 확인:

| 위치 | tb_memories 정의 | 인덱스 |
|---|---|---|
| v6.1 사양서 (Ch 12) | episodic/semantic/procedural/feedback 4개 분리 | ivfflat |
| nexus 코드 (`long_term.py`) | 단일 tb_memories 가정 (INSERT/SELECT만, **DDL 없음**) | **인덱스 생성 코드 없음** |
| 실제 운영 DB | 단일 tb_memories (203,991행) | **hnsw** (수동/외부 생성) |

핵심 문제: nexus 코드에 `CREATE TABLE tb_memories`도 인덱스 생성도 없어, 새 PG
인스턴스(예: 컨테이너 교체, 재해 복구)에서 부트스트랩 시 즉시 깨짐. 운영
DB가 멀쩡한 건 누군가 수동으로 만들어놓은 덕분.

### 결정 — 옵션 A (운영을 정으로)

- 운영 hnsw가 200K+ 행 규모에서 ivfflat보다 정확/지연 모두 우수
- 사양서 4개 분리는 v6.1 작성 시점 설계, 이미 단일로 진화 → 되돌리기 위험 큼
- 새 환경 셋업 자동화·재해 복구 가능성이 최대 이득

### 구현

**1) `core/memory/long_term.py`**
- 모듈 상단에 `_DDL_TB_MEMORIES` + `_DDL_TB_MEMORIES_INDEXES` 상수 (운영 정의 1:1)
  - `id varchar(12) PRIMARY KEY` (ImportanceAssessor의 sha 12자리 호환)
  - `importance` CHECK (0.0 ≤ x ≤ 1.0) 제약 동일 적용
  - `embedding vector(1024)` — e5-large 차원
  - 인덱스 5종: type/tags GIN/created_at DESC/importance DESC/embedding hnsw
- `LongTermMemory.ensure_schema()` 추가 — pg_pool=None이면 no-op, 있으면
  `CREATE EXTENSION vector` → 테이블 → 인덱스 5종을 순차 IF NOT EXISTS 실행

**2) `core/bootstrap.py`** (Phase 2 ③ MemoryManager 초기화)
- `ltm = LongTermMemory(pg_pool=pg_pool)` 직후 `await ltm.ensure_schema()` 호출
- 실패 시 WARNING만 — 인메모리 폴백으로 본류 응답 가능

**3) `tests/unit/test_memory.py`** (`TestLongTermMemoryEnsureSchema` 신규)
- `test_ensure_schema_inmemory_noop` — pg_pool=None일 때 no-op 확인
- `test_ensure_schema_executes_ddl_with_pool` — mock asyncpg 풀로 SQL 7건
  실행 검증 (확장 + 테이블 + 인덱스 5종), 핵심 키워드(`USING hnsw`,
  `vector_cosine_ops`, `tb_memories_importance_check`) 단언

### 테스트 결과

- `tests/unit/test_memory.py`: 64 passed (이전 62 + 신규 2)
- 전체 단위 테스트: **730 passed, 0 failed** (회귀 없음)
- ruff 신규 위반 0 (사전 존재한 long line 1건은 별건)

### 사양서 영향

`PROJECT_NEXUS_SPEC_v7.0_AMENDMENT.md`에 후속 절(예: Part 5 Ch 12 보강)을
추가하여 다음을 명시할 필요:
- v6.1 Ch 12의 4개 테이블 분리(episodic/semantic/procedural/feedback)는
  단일 tb_memories로 통합됨
- 인덱스 정책: 1000행 이하 ivfflat, 그 이상 hnsw (운영 정책)
- 스키마 책임: `LongTermMemory.ensure_schema()` (코드 기준 단일 정의)

본 항목은 코드 변경 후 별도 사양서 갱신 묶음 커밋 시 함께 정리 예정.

### 신규/수정 파일

```
신규:
  (tests/unit/test_memory.py 내 TestLongTermMemoryEnsureSchema 클래스)

수정:
  core/memory/long_term.py     (_DDL_TB_MEMORIES, _DDL_TB_MEMORIES_INDEXES, ensure_schema)
  core/bootstrap.py            (Phase 2 ③에서 ltm.ensure_schema() 호출)
  tests/unit/test_memory.py    (신규 테스트 2건)
  user_mig/progress/progress.md (이 항목)
```

---

## 인프라 복구 — pgvector 라이브러리 누락 해소 (2026-06-01)

### 배경 — 진단

사용자 보고: "생성 속도가 예전에 비해 너무 느려졌다."
실측 결과 KNOWLEDGE 첫 호출만 비정상 (CHAT 0.8s, KNOWLEDGE 첫 66s, 두 번째 4.6s).

서버 로그 시간 분해:
```
17:57:27 KNOWLEDGE 분류
17:58:29 지식 RAG 주입: ~3002자   ← 62초 후
17:58:32 LLM 응답 완료 (3초)
```

vLLM 큐는 비어 있고 pgvector 검색은 38ms. → **임베딩 서버(e5-large, :8002)가 idle 상태로 빠진 후 첫 호출에서 ~60초 cold start**. kowiki 적재가 끝나고 임베딩 호출이 드물어지면서 매번 cold가 됐음.

### A — 부트스트랩 임베딩 워밍업

`core/bootstrap.py`에 `_warmup_embedding(provider, retries=2)` 추가.
- KnowledgeStore 초기화 직후 fire-and-forget으로 한 번 호출
- 짧은 backoff로 2회 재시도 (서버가 아직 안 떴을 가능성)
- 실패 시 WARNING만, 본류 응답에 무영향

### B — 주기적 keep-warm

`_embedding_keepalive(provider, interval_sec=300)` 추가.
- 5분 간격으로 짧은 ping 송신
- `asyncio.CancelledError`로 깨끗 종료
- 일시 장애는 디버그 로그만, 다음 주기에 재시도
- 부트스트랩이 task 핸들을 `components["embedding_keepalive_task"]`로 노출
- `web/app.py` lifespan에서 종료 시 cancel + await로 깔끔 정리

### E — CLI 단계별 spinner

`cli/repl.py::_process_message`에 Rich `Console.status()` 통합.
- 첫 TEXT_DELTA 도착 전까지 spinner 표시
- StreamEvent 타입에 따라 단계 텍스트 갱신:
  - `STREAM_REQUEST_START` → "모델 추론 중..."
  - `MESSAGE_START` → "응답 생성 중..."
  - `THINKING_START` → "생각 정리 중..."
  - `TOOL_USE_START` → "{tool_name} 실행 중..."
  - `TEXT_DELTA` 첫 도착 → spinner 닫고 본문 출력
  - `TOOL_USE_STOP` → 다음 응답 대기 spinner 재가동
- 예외/취소 경로에서도 spinner는 finally에서 안전 종료

사용자 체감 효과: KNOWLEDGE cold start 60초가 "왜 멈췄지?"가 아니라 "지식 검색 중..."으로 가시화 (워밍업이 그 60초를 부트스트랩 시점으로 옮기지만 만약 idle로 떨어졌어도 사용자가 인지 가능).

### 테스트

`tests/unit/test_embedding_warmup.py` 신규 6 케이스:
- 첫 시도 성공 — 추가 재시도 없음
- 첫 실패 후 재시도 성공
- 모든 재시도 실패 → False
- embed가 빈 리스트 반환 → 실패로 판정
- keep-alive `CancelledError`로 깨끗 종료
- keep-alive 일시 장애 후 다음 주기 재시도 (await 횟수 ≥2)

`asyncio.sleep`은 `unittest.mock.patch("core.bootstrap.asyncio.sleep")`로 0초화하여
테스트 시간 < 1초.

**회귀**: 단위 + 통합 **811 passed / 1 skipped** (v0.14.7 805 → +6, 실패 0,
ruff 신규 위반 0).

### 신규/수정 파일

```
신규:
  tests/unit/test_embedding_warmup.py

수정:
  core/bootstrap.py            (_warmup_embedding, _embedding_keepalive, 부트스트랩 호출)
  web/app.py                   (lifespan 종료 시 keepalive task cancel)
  cli/repl.py                  (Console.status 통합 + _stage_label_for 헬퍼)
  user_mig/progress/progress.md (이 항목)
```

### 부수 발견 (별도 작업 후보)

- `_finalize_turn`에서 `WARNING: 1 validation error for TextBlock`이 매 응답마다
  발생. 응답 본류엔 영향 없으나 로그 노이즈. 향후 추적.
- vLLM 임베딩 서버 자체 cold start 시간을 줄이려면 GPU 서버 측 `--enforce-eager`
  해제 + memory pre-allocate 상향이 필요 (운영자 작업 영역).

### 사양서 영향

이번 변경은 인프라 운영 최적화로 사양서 Part 4(GPU 서버) 또는 Part 5 Ch 16에
짧게 부속 절을 추가할 후보. 현재는 progress.md만 업데이트하고 사양서는
다음 묶음 커밋 시 함께 정리 예정.

---

## v0.14.6 — CHAT 카테고리 + RAG 무시 지시 강화 (2026-04-27)

### 배경

운영 중 사용자 보고: "안녕"·"좋은 아침이야" 같은 짧은 인사에 대해 가수 "안녕"
위키 정보·"고도원의 아침편지" 등 무관한 백과사전 내용이 줄줄 답변에 포함됨.

원인 추적:
- kowiki 적재 종료 (마지막 쓰기 04-24 07:40, **1,067,975 청크 / 461,517 문서**)
- 분류기는 "TOOL이 아니면 무조건 KNOWLEDGE" 2분류 → 인사도 KNOWLEDGE로 감
- KNOWLEDGE에서 KnowledgeRetriever가 자동 RAG 주입 (`min_similarity=0.3`)
- 청크가 100만 단위라 임의 단어로도 매칭이 잘 됨 → 시스템 프롬프트에 ~1KB 주입
- 27B 모델은 "주어진 컨텍스트는 활용해야 한다"는 압력으로 그 정보를 풀어냄

사용자가 원안(B+D) + 사양서 갱신을 명시 승인하여 진행.

### 변경 (B안 — CHAT 휴리스틱 카테고리)

**1) `core/config.py` — RoutingConfig 확장**
- `chat_mode: RoutingProfile` 신규 — model=qwen3.5-27b, temp=0.5, max_tokens=512
- `chat_keywords: list[str]` 신규 — 한/영 인사·잡담 ~38개 (안녕/굿모닝/hi/thanks/bye 등)
- `chat_max_length: int = 30` — 길이 게이트 (긴 입력 안에 인사어 우연히 들어가도 CHAT 아님)

**2) `config/nexus_config.yaml`**
- `routing.chat_keywords` / `chat_max_length` / `chat_mode` 섹션 명시
- 운영자가 코드 변경 없이 키워드 튜닝 가능

**3) `core/orchestrator/routing.py` — 3분류 확장**
- `HeuristicClassifier.classify()`가 KNOWLEDGE/TOOL/**CHAT** 반환
- 우선순위: enabled→길이→TOOL→**CHAT(길이+키워드 AND)**→KNOWLEDGE
- `RoutingResolver._OVERRIDE_ON_CLASSES`에 CHAT 포함 (테넌트 LoRA 적용)
- `RoutingDecision.inject_knowledge_rag` 프로퍼티 신규 — KNOWLEDGE만 True

**4) `core/orchestrator/prompt_assembler.py` — KB 단계 분기 + IGNORE 지시**
- `decision.inject_knowledge_rag` 사용 (CHAT/TOOL 자동 스킵)
- KB 주입 시 끝에 항상 다음 지시 동봉 (D 보조안):
  > Use the information above ONLY when it is clearly relevant ... If the
  > snippets are off-topic, irrelevant, or contradict common-sense knowledge,
  > IGNORE them and answer from your own general knowledge.

**5) 시스템 프롬프트 강화 (D안)**
- `web/prompts/worker_system.md`: "Conversational style" + "When KB block is
  present" 섹션 추가
- `core/bootstrap._build_default_system_prompt()`: CLI 측에도 동일 small-talk
  가이드 + KB 무시 규칙

### 테스트

- `tests/unit/test_query_routing.py` 갱신/추가:
  - "안녕"이 KNOWLEDGE 단언인 기존 케이스 → CHAT 단언 케이스로 분리
  - CHAT 분류 13 케이스 parametrize (안녕/안녕하세요/좋은 아침/굿모닝/잘 자/
    고마워/감사/hi/Hello/good night/thanks/bye)
  - 길이 게이트 검증 ("안녕, 오늘 알베르 카뮈…" → KNOWLEDGE)
  - TOOL 우선순위 검증 ("이 파일 안녕히 처리해줘" → TOOL)
  - chat_max_length=0 운영자 스위치 → KNOWLEDGE 회귀
  - RoutingResolver: CHAT 결정 + tenant override 적용 검증
  - _resolve_profile("CHAT") → chat_mode 폴백
- `tests/unit/test_prompt_assembler.py` (신규):
  - CHAT/TOOL → KB 미주입
  - KNOWLEDGE → KB 주입 + IGNORE 지시 동봉
  - KB 검색 예외는 swallow

**회귀 결과**: 단위 + 통합 **805 passed / 1 skipped** (v0.14.5 782 대비 +23,
실패 0, ruff 신규 위반 0).

### 사양서 갱신

`PROJECT_NEXUS_SPEC_v7.0_AMENDMENT.md` Part 2.5.9 신설:
- 배경 + kowiki 적재 규모 + 실측 증상
- 3분류 설계 + chat_mode 프로필 + chat_keywords
- D 보조안 (시스템 프롬프트 강화)
- 4-Tier 무영향 + 하드웨어 업그레이드 시 자동 회귀 (`routing.enabled=false`)
- 실측 효과 표 (안녕: 1000+ 토큰 → ~30 토큰)

### 사양서 정합성 (이번 변경)

| 사양서 조항 | 정합 |
|---|---|
| Part 2.5.3 "휴리스틱만으로 충분, 상수 시간" | ✓ (CHAT도 substring + 길이 임계) |
| Part 2.5.5 "기본값으로 v6.1 동일 동작" | ✓ (chat_max_length=0이면 2분류로 회귀) |
| Part 2.5.6 "TIER_M 이상 자동 비활성화" | ✓ (routing.enabled=false면 CHAT도 무효화) |
| Part 1·8.1 4-Tier 체인 무결성 | ✓ (체인 시그니처 불변) |
| Part 2 Scout B 방식 (자동 전처리 폐기) | ✓ (Scout 분류기 사용 안 함, 사용자 직관 회피 근거: 같은 30s 함정) |

### 신규/수정 파일

```
신규:
  tests/unit/test_prompt_assembler.py

수정:
  core/config.py
  config/nexus_config.yaml
  core/orchestrator/routing.py
  core/orchestrator/prompt_assembler.py
  core/bootstrap.py
  web/prompts/worker_system.md
  tests/unit/test_query_routing.py
  user_mig/PROJECT_NEXUS_SPEC_v7.0_AMENDMENT.md  (Part 2.5.9)
  user_mig/progress/progress.md                   (이 항목)
```

### 후순위 (다음 세션 후보)

- 임베딩 기반 분류기(`embedding_classifier`) — 휴리스틱이 놓치는 모호 케이스용
- CHAT/KNOWLEDGE 경계 데이터 자동 수집 (오분류 모니터링)
- min_similarity 0.3 → 동적 조정 (RAG 통계 기반)

---

## v0.14.5 — 사이드바 제목 자동화 + 세션 삭제 (2026-04-23 저녁)

### 배경

v0.14.4에서 프론트-백엔드 세션 연동이 완성됐으나 실 사용에서 두 가지 UX 문제:

- 사이드바에 `session-1776930460 ·서버` 형태로 **세션 ID만 노출**되어 어떤
  대화였는지 식별 불가 (Claude Code UX와 큰 격차)
- 세션 삭제 수단이 전무 — 누적된 테스트/실험 대화를 정리할 방법이 UI에 없음

또한 기존 20개 디스크 세션 + 8개 Redis 세션은 대부분 개발 테스트 잔여물이어서
운영자 요청으로 **전체 정리**를 병행했다.

### 변경

**1) `core/memory/transcript.py` — title_hint + 삭제 헬퍼**
- `list_transcript_sessions()`가 각 세션의 `transcript.jsonl`을 한 번의
  패스로 스캔하여 라인 수와 **첫 user 메시지**를 동시 추출
- 응답에 `title_hint` 필드 신규 (60자 초과 시 `…` 절단, 멀티라인은 공백 하나로
  정규화, user 메시지가 없으면 None)
- `delete_transcript_session(sessions_dir, session_id) -> bool` 신규
  - 경로 탈출 검증 (슬래시/백슬래시/`..`/`\x00` → ValueError)
  - `Path.resolve().relative_to(base)` 재검증으로 이중 방어
  - 없는 세션 삭제는 False 반환 (에러 아님)

**2) `web/app.py` — `DELETE /v1/sessions/{session_id}` 신규**
- Redis(`clear_session`) + 디스크(`delete_transcript_session`) 양쪽 best-effort
- 사전에 `get_conversation_context`로 Redis 존재 여부 확인 후 `deleted_redis`
  플래그로 보고
- 한쪽이 장애여도 다른 쪽은 시도 (감사 로그에 WARNING)
- 응답: `{session_id, deleted_redis, deleted_disk}`
- 400 — 의심 문자 차단, 404 없음 (멱등 삭제 — 재시도 안전)

**3) `web/static/index.html` — 사이드바 X 버튼 + title_hint 사용**
- CSS: `.session-item`을 flex 레이아웃으로 변경, `.session-delete-btn`이
  hover/active 시 opacity 0→0.7, 삭제 버튼 hover 시 빨간 배경 강조
- `deriveSessionTitle()`이 서버 `title_hint` 최우선 사용 (파일 재파싱 없이 즉시
  표시). 없으면 첫 user 메시지 → session_id 앞부분 순으로 폴백
- `deleteSession(id, event)` 신규 — `event.stopPropagation()`로 상위 클릭
  분리, `confirm()` 다이얼로그 후 `DELETE /v1/sessions/{id}` 호출, 로컬
  sessions 배열에서도 제거, 현재 세션 삭제 시 새 세션으로 자동 전환
- 빈 로컬 세션(아직 서버 저장 전)은 서버 호출 생략

**4) 기존 세션 일괄 정리 (운영자 요청)**
- 디스크: `.nexus/sessions/` 하위 20개 디렉토리 제거 (빈 5 + transcript 15)
- Redis: `session:*` 8개 키 일괄 DEL (`192.168.10.39:6340`, db=6)
- 잔존 확인: 디스크 0개, Redis 0개

### 테스트

- `tests/unit/test_session_persistence.py` +12
  - title_hint 포함 (`니체 철학...` 등 실제 문자열 검증)
  - title_hint 60자 초과 시 말줄임표 + 길이 61
  - 멀티라인 → 공백 정규화
  - user 없는 세션 → None
  - `delete_transcript_session` 성공/미존재/경계 문자/다른 세션 불영향
- `tests/integration/test_web_integration.py` +9
  - `GET /v1/sessions`에 `title_hint` 포함
  - `DELETE /v1/sessions/{id}` 200 + 디스크 제거 확인
  - 멱등성 (없는 세션 → 200, 둘 다 False)
  - 400 — parametrize 3 케이스 (백슬래시/연속 점/NUL)

**회귀 결과**: 단위 + 통합 **782 passed / 1 skipped** (v0.14.4 763 대비 +19,
실패 0, ruff 신규 위반 0).

### 보안 메모

- `delete_transcript_session`의 경로 검증 — `Path.resolve().relative_to()`로
  sessions_dir 외부를 가리키는 모든 시도를 ValueError로 차단
- 의심 문자 400 + 슬래시 라우터 404 두 층 방어 (GET/DELETE 공통)
- 서버 사이드 log에 삭제 이벤트 기록 (`세션 삭제: session=..., redis=..., disk=...`)

### 서버 재기동 필요

기능 변화 반영과 기존 세션 전면 정리가 동시에 이뤄졌으므로, **현재 백그라운드
에서 돌고 있는 구(v0.14.3) 서버는 재기동**해야 프론트의 삭제 버튼/title_hint
가 동작한다. 외부 사용자 세션이 붙어 있어 중단 영향이 있으며 타이밍은 운영자
확인 후 별도 절차로 진행.

### 신규/수정 파일

```
신규/수정:
  core/memory/transcript.py                  (title_hint, delete_transcript_session)
  web/app.py                                 (DELETE /v1/sessions/{id})
  web/static/index.html                      (삭제 버튼 + title_hint 사용)
  tests/unit/test_session_persistence.py     (+12 케이스)
  tests/integration/test_web_integration.py  (+9 케이스)
  user_mig/progress/progress.md              (이 항목)

정리 (데이터):
  .nexus/sessions/*                          (20개 디렉토리 제거)
  Redis session:*                            (8개 키 제거)
```

### 남은 후순위

- localStorage에 currentSessionId 영속화 (권고 1번 — 새로고침 자동 복원)
- 다중 선택 삭제 (Ctrl+클릭으로 여러 세션 한 번에)
- 로그인/테넌트 선택 UI (공공 납품 단계)

---

## v0.14.4 — 프론트 세션 영속화 연동 (2026-04-23 저녁)

### 배경

Ch 16 세션 영속화는 v0.14.3에서 백엔드(Redis 단기 + transcript.jsonl 영구)
까지 완성되었으나 프론트가 그 자산을 활용하지 못하던 갭이 있었다:

- `index.html`의 `newSession()`이 매 페이지 로드마다 `session-{Date.now()}`로
  새 ID를 생성 → 새로고침하면 사이드바가 비어 보이고 과거 대화 흔적이 사라짐
- `/v1/sessions`는 목록만 반환하고 **특정 세션의 메시지 전체를 돌려주는
  엔드포인트가 없었음** — 프론트가 과거 대화를 화면에 펼칠 방법 자체가 부재
- 결과적으로 Ch 16의 ROI가 "재요청 시 같은 session_id로 다음 턴이 이어진다"
  수준에 머무르고 사용자 체감 가치는 낮았음

### 변경

**1) `core/memory/transcript.py` — `read_transcript_messages()` 추가**
- `(sessions_dir, session_id, *, roles=("user","assistant"), limit=None)` 시그니처
- JSONL을 한 줄씩 파싱, role 필터(기본 user/assistant) 적용, 손상 라인은 조용히 스킵
- limit 지정 시 가장 최근 N개만 반환 (오래된 것부터 순서 유지)
- 파일 없음/OSError → 빈 리스트로 graceful degrade

**2) `web/app.py` — `GET /v1/sessions/{session_id}/messages` 엔드포인트**
- 조회 우선순위: Redis `get_conversation_context` → 트랜스크립트 폴백 → 404
- 응답: `{session_id, source: "redis"|"transcript", messages: [...], total}`
- 입력 검증: session_id에 `/` `\` `..` `\x00` 포함 시 400 차단
  (트랜스크립트 디렉토리 탈출 방지). 슬래시는 라우터 단계에서도 자동 404로 1차 차단

**3) `web/static/index.html` — 사이드바 서버 세션 머지 + lazy-load 복원**
- DOMContentLoaded 시 `loadServerSessions()` 호출 → `/v1/sessions` 결과를
  로컬 sessions 배열과 병합 (서버 측은 `source: 'server'` 태그 + `·서버` 표시)
- 서버 세션 클릭 시 메시지가 비어 있으면 `/v1/sessions/{id}/messages` 호출 →
  로컬 캐시에 채운 뒤 `switchSession`이 화면에 복원 (사이드바는 항상 가볍게 유지)
- `escapeHtml()` 헬퍼 추가 — 서버에서 받은 session_id/제목을 innerHTML에 넣을
  때 XSS 방지
- `deriveSessionTitle()` — 첫 user 메시지 30자 → 없으면 session_id 앞부분으로
  사이드바 제목 자동 생성

### 테스트

- `tests/unit/test_session_persistence.py` +5
  - role 필터 (system/에러 엔트리 제외)
  - 파일 없음 (빈 리스트)
  - limit (가장 최근 N개)
  - 손상 JSON 라인 스킵
  - 명시적 roles로 system 포함
- `tests/integration/test_web_integration.py` +8 (=4 케이스, parametrize 포함)
  - 404 — 존재하지 않는 세션
  - 400 — 핸들러까지 도달하는 의심 문자(`foo\\bar`, `foo..bar`, `\x00`)
  - 라우터 차단 — 슬래시/단독 `..`는 200이 아님 (디렉토리 탈출 불가)
  - 200 — 트랜스크립트 파일 폴백 경로 검증 (config.session.sessions_dir 모킹)

**회귀 결과**: 단위 742 + 통합 21 = **763 passed / 1 skipped** (변경 전 736 대비
+27, 신규 13건 외 다른 차이는 컬렉션 변동으로 보이며 실패 0).

### 보안 메모

- session_id에 `/`, `\`, `..`, `\x00` 차단 — `Path(sessions_dir) / session_id`
  연산 시 디렉토리 탈출 방지의 명시적 1차 검증
- `api_keys`/`tenant_id` 등 민감 필드는 본 변경에서 새로 노출되지 않음
- 404 vs 400 분기로 "탈출 시도"와 "단순 조회 미스"를 구분 가능

### 사양서 반영 예정

- Part 5 Ch 16에 신규 엔드포인트 + 프론트 연동 섹션 추가
- "v0.14.3 다음 후보" 항목에서 "프론트 세션 사이드바" 완료로 표기

### 알려진 후속 이슈 (선택 작업)

- localStorage에 currentSessionId 영속화 → 새로고침 후 자동 같은 세션 복원
  (현재는 클릭으로만 복원). 권고 1번에 해당, 사용자 결정 시 추가
- 서버 세션 메시지 로딩 시 스피너 UI 부재 (현재는 무음으로 추가됨)
- 로그인/테넌트 선택 UI는 공공 납품 단계에서 별도 도입 예정 (권고 3번)

### 신규/수정 파일

```
신규/수정:
  core/memory/transcript.py            (read_transcript_messages 추가)
  web/app.py                            (GET /v1/sessions/{id}/messages)
  web/static/index.html                 (loadServerSessions, switchSession lazy-load)
  tests/unit/test_session_persistence.py     (+5 케이스)
  tests/integration/test_web_integration.py  (+8 케이스)
  user_mig/progress/progress.md         (이 항목)
```

---

## v0.14.3 — 업로드 파일 분석 hang 대응 (2026-04-23)

### 배경

사용자 리포트: 15KB xlsx(`AI_서비스_데이터_수집_양식.xlsx`) 업로드 후 "무한
로딩" 현상. 세션 `session-1776919030600` 디렉토리는 생성됐으나
`transcript.jsonl` 파일이 없는 "유령 세션" 상태.

### 진단 결과

1. **실행 중인 웹 서버가 04-21 기동 상태** — v0.14.2(04-22)의 `AgentTool`
   결과 캐시 코드가 아예 로드되지 않았음. `/metrics.agent_cache`가 `null`
   로 응답한 것이 증거.
2. **Scout 누적 호출 15회, 평균 48초** — 같은 파일을 여러 턴에 걸쳐 재호출할
   때 캐시가 없어 매번 48초 비용 발생.
3. **클라이언트 측 타임아웃/취소 UX 부재** — 사용자가 "무한 로딩"과 "느린
   정상 처리"를 구분할 방법이 없음.
4. **`_finalize_turn()`이 async for 예외에 의해 스킵되는 구조** — 스트림이
   중단되면 트랜스크립트 기록이 전혀 남지 않아 사후 디버깅이 불가능.
5. **LAN 접속 이슈 (병렬 증상, 서버 무관)** — 이더넷 인터페이스
   (192.168.22.223)가 "식별 중..." 상태로 192.168.22.10(Wi-Fi)에서만 접속
   가능. 네트워크 어댑터 재설정 별도 필요.

### 조치 (코드 4파일 + 진행 문서)

**1) `core/orchestrator/query_engine.py` — `_finalize_turn` 보강**
- `submit_message`의 `async for` 본문을 `try/except/finally`로 감싸 예외/
  취소 발생 시에도 `_finalize_turn`이 **반드시 호출**되도록 구조 변경
- `_finalize_turn(finalize_error=...)` 시그니처 확장 — 스트림 중단 시
  `role="system"` 에러 엔트리를 트랜스크립트에 append하여 "폴더만 있고
  파일 없는 hang 세션"과 구분 가능하게 함
- 예외는 re-raise하여 상위 웹 핸들러가 인지하도록 보장

**2) `core/tools/implementations/agent_tool.py` — 장기 실행 관측성**
- `AgentTool.call()` 진입 시 "실행 중" info 로그 추가 (기존 "실행 시작"
  로그는 설정 단계만 표시했음)
- 완료 로그에 `elapsed >= 60s` 임계 체크 — 초과 시 WARNING으로 승격하여
  로그 필터링만으로 hang 징후 즉시 탐지 가능
- 실패 로그에도 elapsed + stats_key 포함

**3) `web/app.py` — SSE heartbeat + 요청 타이밍 로그**
- `/v1/chat/stream` 핸들러를 Producer/Consumer 구조로 재설계:
  - `asyncio.Queue` + `asyncio.create_task` producer가 `submit_message`를
    소비하여 큐로 이동
  - 메인 루프는 `asyncio.wait_for(queue.get(), timeout=20)`로 20초마다
    SSE 주석 `: heartbeat {elapsed}s\n\n` 프레임 전송
  - SSE 주석은 EventSource 클라이언트에서 무시되므로 기존 JSON 파서에 영향 X
- 요청 단위 로그: 진입 시 "SSE 시작: session=..., has_attach=...",
  종료 시 "SSE 완료: ..., elapsed=..., events=..." (에러면 WARNING)
- producer 예외는 `{"type": "error", "error_code": "stream_aborted", ...}`
  데이터 프레임으로 클라이언트에게 전달
- 핸들러 종료 시점 finally에서 producer_task cancel — 클라이언트 연결 끊김
  대응
- import 추가: `asyncio`, `time`

**4) `web/static/index.html` — 장기 대기 UX + 취소 버튼**
- "생각하는 중" 스피너에 초 단위 경과 시간 표시 — 10초 후 `생각하는 중 (Xs)`,
  60초 후 `분석이 오래 걸리고 있습니다 (Xs). '취소' 버튼을 눌러 중단할 수
  있습니다`로 문구 전환
- 전송 버튼을 **전송/취소 토글**로 개조 — 스트리밍 중에는 중단 아이콘
  (rect 모양)으로 바뀌어 `AbortController.abort()`를 트리거
- `fetch(..., {signal})`로 SSE 연결 취소 가능
- SSE 주석(`:`로 시작) 라인은 클라이언트에서 명시적으로 무시
- 사용자 취소 시 `AbortError` 분기 → "[취소됨]" 메시지 표시

### 검증

- **회귀 테스트**: 736 passed / 1 skipped (단위 + 통합)
  - 변경 영향 테스트: `test_agent_tool.py` 22건, `test_session_persistence.py`
    11건, `test_web_integration.py` 14건 모두 통과
- **서버 재기동 후 `/metrics.agent_cache`**:
  ```json
  {"hits": 0, "misses": 0, "stored": 0, "evicted": 0, "size": 0}
  ```
  (이전엔 `null`이었음 — v0.14.2 캐시 코드가 이제 실제로 로드됨을 증명)
- **ruff check**: 본 변경분 clean. 기존 위반(E402/E501/F401)은 본 변경
  범위 밖이라 건드리지 않음.

### 구 서버와의 차이 (기동 로그 실측)

```
2026-04-21 16:20 기동:  AgentTool 캐시 없음, agent_cache: null
2026-04-23 14:37 기동:  agent_cache: dict, Scout warning 임계 60s,
                        SSE heartbeat 20s, 취소 UX 활성
```

### LAN 접속 이슈 (별도 조치 필요)

서버 재기동으로 해결되지 않음. 원인:
- 이더넷 어댑터 `192.168.22.223`이 "식별 중..." 상태
- Windows 방화벽 및 서버 바인딩(`0.0.0.0:8443`)은 정상
- Wi-Fi 쪽 IP `192.168.22.10:8443`은 정상 응답

**즉시 우회**: 다른 PC에서 `https://192.168.22.10:8443`으로 접속
**근본 해결 (관리자 권한)**:
```powershell
Disable-NetAdapter -Name "이더넷" -Confirm:$false
Enable-NetAdapter -Name "이더넷"
```

### 남은 작업

- LAN 어댑터 재설정 (사용자 확인 후)
- 이번 수정으로 **같은 xlsx 재시도** 시나리오에서 Scout 캐시 히트율 실측
- `transcript.jsonl`에 `[stream aborted]` 시스템 엔트리가 기록되는지 실환경
  확인 (hang/취소가 발생했을 때)

### 사양서

`PROJECT_NEXUS_SPEC_v7.0_AMENDMENT.md` Part 9(리팩토링 노트) 뒤에 v0.14.3
관측성 섹션을 추가할 예정.

---

## Phase 0.5: Foundation (완료)

### v0.1.0 — 프로젝트 스켈레톤 (2026-04-14)
**커밋**: `1678f2e`

- `pyproject.toml` — 의존성, ruff/pytest/mypy 설정 통합
- `.gitignore` — Python, data/, models/, checkpoints/, logs/ 제외
- `requirements.txt`, `requirements-train.txt`, `requirements-dev.txt`
- `.env.example` — 환경 변수 템플릿
- 디렉토리 구조 생성 (Ch.21 기준 16개 패키지 + `__init__.py`)
- `config/*.yaml` 5종: nexus_config, tool_mappings, model_profiles, permission_rules, logging_config
- `tests/conftest.py` — 공통 fixture (mock GPU/Redis/PG)

### v0.1.1 — 핵심 부트스트랩 모듈 (2026-04-14)
**커밋**: `9746efb`

- `core/state.py` — GlobalState 싱글톤 (DAG leaf 격리, 스레드 안전 토큰 카운터)
- `core/config.py` — NexusConfig Pydantic v2 설정 시스템 (YAML 로딩, 에어갭 검증)
- `core/bootstrap.py` — Phase 1 초기화 (로깅, 종료 핸들러, GPU 사전연결, 플랫폼 감지)
- 21개 테스트 통과

---

## Phase 1.0: Model Layer (완료)

### v0.2.0 — 메시지 타입 + 추론 클라이언트 + GPU 감지 (2026-04-14)
**커밋**: `35b9a2e`

- `core/message.py` — 전체 타입 시스템 구현
  - StreamEvent (17종 이벤트 타입), Message (factory method 4종)
  - ContentBlock (TextBlock, ToolUseBlock, ToolResultBlock, ThinkingBlock)
  - Conversation (compact_boundary 기반 컨텍스트 관리)
  - TokenUsage (합산 연산자, vLLM prefix caching 지원)
- `core/model/inference.py` — ModelProvider ABC + LocalModelProvider
  - vLLM OpenAI 호환 /v1/chat/completions SSE 스트리밍
  - Nexus Message → OpenAI message 자동 변환
  - tool_calls 증분 누적 + 최종 완성 로직
  - 연결 실패 시 graceful error event yield
- `core/model/gpu_detector.py` — GPU 티어 감지
  - GPUTier 4종 (RTX_5090, H100, H200, MULTI_GPU)
  - 티어별 최적 설정 테이블 (양자화, 컨텍스트, 배치, LoRA)
- `core/model/prompt_formatter.py` — 모델별 프롬프트 포매터
  - Gemma 4 (<start_of_turn>), ExaOne ([|system|]), ChatML 폴백
  - 도구 스키마 XML 주입
- `core/model/model_manager.py` — Machine A 측 모델 매니저
  - hot-swap 조율, LoRA 로드/언로드, 헬스 체크
- 61개 테스트 통과 (신규 40개)

---

## Phase 2.0a: Tool System 프레임워크 + 핵심 도구 (완료)

### v0.3.0 — BaseTool + Registry + Executor + 핵심 도구 8개 (2026-04-14)
**커밋**: `995ed5a`

- `core/tools/base.py` — BaseTool ABC + ToolResult + ToolUseContext
  - fail-closed 기본값, 7개 카테고리 ~25개 멤버
- `core/tools/registry.py` — ToolRegistry (등록, alias 조회, deny 필터, cache-stable 정렬)
- `core/tools/executor.py` — 13단계 실행 파이프라인
- 핵심 도구 8개:
  - Read, Write, Edit, MultiEdit (파일 시스템)
  - Bash (실행), Glob, Grep (검색), LS (디렉토리)
- 82개 테스트 통과 (신규 21개)

---

## Phase 2.0b: 나머지 도구 16개 (완료)

### v0.4.0 — 도구 24개 완성 (2026-04-14)
**커밋**: `9fe5ba3`

- `git_tools.py`: Git 도구 6개 (Log, Diff, Status, Commit, Branch, Checkout)
- `notebook_tools.py`: Notebook 도구 2개 (Read, Edit)
- `task_tools.py`: Task 도구 3개 (TodoRead, TodoWrite, Task)
- `agent_tool.py`: Agent 도구 1개 (서브 에이전트 stub)
- `memory_tools.py`: Memory 도구 2개 (인메모리 폴백)
- `docker_tools.py`: Docker 도구 2개 (Build, Run)
- 91개 테스트 통과

---

## Phase 3.0: Orchestrator (완료)

### v0.5.0 — QueryLoop + 스트리밍 + 컨텍스트 관리 (2026-04-14)
**커밋**: `c7f32d9`

- `query_loop.py`: while(True) 에이전트 턴 루프
  - 7가지 ContinueReason 전환, 4 Phase per iteration
  - LoopState 명시적 상태 관리, max_output_tokens 에스컬레이션
- `stream_handler.py`: StreamingToolExecutor
  - 모델 스트리밍 중 도구 병렬 실행, Semaphore 동시성 제한
- `context_manager.py`: 4단계 압축 파이프라인
  - tool_result 예산, snip/micro/auto compact, 긴급 압축
- `stop_resolver.py`: 종료/계속 판단 + 잘림 감지 휴리스틱
- 106개 테스트 통과

---

## Phase 4.0: Security & Permission (완료)

### v0.6.0 — 5계층 권한 + 보안 + 훅 시스템 (2026-04-14)
**커밋**: `525b096`

- `core/permission/types.py`: 전체 권한 타입 (7 Mode, 7 Category, 7x7 MAP)
- `core/permission/pipeline.py`: 5계층 파이프라인 (deny→tool→mode→hook→resolve)
- `core/security/path_guard.py`: 경로 보호 (순회, UNC, 보호 경로 21개 패턴)
- `core/security/command_filter.py`: 명령어 필터 (allowlist + 30+ 위험 패턴)
- `core/security/audit.py`: JSONL 감사 로그 (10MB 로테이션)
- `core/hooks/hook_manager.py`: 훅 매니저 (4 이벤트, 3 결정)
- `core/hooks/builtin_hooks.py`: 내장 훅 2개
- 124개 테스트 통과

---

## Phase 5.0: Thinking & Memory (완료)

### v0.7.0 — Thinking Engine + Memory System (2026-04-14)
**커밋**: `47a715f`

- Thinking Engine (core/thinking/ 6모듈):
  - ComplexityAssessor (키워드+증폭패턴 → 0.0~1.0)
  - ThinkingStrategy 4종 (DIRECT/HIDDEN_COT/SELF_REFLECT/MULTI_AGENT)
  - HiddenCoTEngine (2-pass), SelfReflectionEngine (3-pass)
  - ThinkingOrchestrator (복잡도→전략→실행→캐시)
  - ThinkingCache (SHA-256, LRU, TTL)
- Memory System (core/memory/ 6모듈):
  - MemoryType 5종, MemoryEntry, DECAY_HALF_LIFE
  - ShortTermMemory (Redis/인메모리), LongTermMemory (PG+pgvector/인메모리)
  - MemoryManager (턴 생명주기), ImportanceAssessor, MemoryDecayManager
- 236개 테스트 통과 (신규 112개)

---

## Phase 6.0 + 7.0: Training + CLI/Web/Deployment (완료)

### v0.8.0 — 학습 파이프라인 + 인터페이스 + 배포 (2026-04-14)
**커밋**: `3fc9787`

- Training Pipeline (training/ 6모듈):
  - 5-Phase 전략 (PROMPT→BOOTSTRAP→SELF_DATA→REASONING→DOMAIN)
  - 부트스트랩 데이터 생성 (도구 70% + 추론 30%)
  - LoRA/QLoRA 트레이너, 데이터 수집 (PII 마스킹), 자동 학습 루프
  - 체크포인트 관리 (활성화, 롤백, 최고성능)
- CLI (cli/ 3모듈): Rich REPL, Click 명령어, 출력 포매터
- Web (web/ 2모듈): FastAPI (chat/stream/sessions/tools/health), 미들웨어
- Deployment (deployment/ 2모듈): SHA256 무결성, 에어갭 번들 준비
- 288개 테스트 통과 (신규 52개)

---

## Phase 8.0: Integration & Polish (완료)

### v0.9.0 — 통합 테스트 64개 + 모듈 간 연결 검증 (2026-04-15)

- **conftest.py 확장**:
  - EnhancedMockModelProvider — tool_calls 시뮬레이션 가능한 ModelProvider mock
  - MockResponse dataclass — 응답 시나리오 정의 (text, tool_calls, stop_reason)
  - basic_tools, tool_use_context fixture 추가
- **tests/integration/test_full_pipeline.py** (22개 테스트):
  - TestQueryLoopIntegration: query_loop → tool → response 풀 플로우 (5개)
  - TestToolChainIntegration: 다중 도구 체인 실행 (2개)
  - TestSecurityIntegration: PathGuard + CommandFilter + PermissionPipeline (11개)
  - TestThinkingIntegration: 복잡도 평가 → 전략 선택 → 캐시 (4개)
  - TestMemoryIntegration: ShortTerm + LongTerm + Manager 인메모리 폴백 (4개)
- **tests/integration/test_training_integration.py** (12개 테스트):
  - BootstrapGenerator: JSONL 생성, 7:3 비율, seed 결정성 (4개)
  - DataCollector: PII 마스킹 (email/phone), 민감 경로 필터링 (5개)
  - CheckpointManager: 목록 조회, activate/rollback (httpx mock) (3개)
- **tests/integration/test_web_integration.py** (9개 테스트):
  - 7개 엔드포인트: health, chat, stream(SSE), sessions, tools, models, metrics
- **tests/integration/test_airgap_integration.py** (16개 테스트):
  - IntegrityVerifier: compute_hash, verify_file/directory, generate_manifest (12개)
  - AirGapPrep: generate_manifest, verify_manifest (4개)
- 352개 테스트 통과 (기존 288 + 신규 64)

---

---

## TODO 연동 작업 (완료)

### v0.9.1 — QueryEngine + 모듈 간 배선 연동 (2026-04-15)

- **core/orchestrator/query_engine.py** (신규):
  - Tier 1 세션 오케스트레이터 — query_loop (Tier 2) 래퍼
  - submit_message() → AsyncGenerator[StreamEvent | Message]
  - 대화 히스토리 관리, 세션 사용량 추적
- **core/bootstrap.py — Phase 2 초기화**:
  - init_phase2(): ToolRegistry(24개 도구) + MemoryManager + QueryEngine 초기화
  - _create_tool_registry(): 24개 도구 일괄 등록
- **web/app.py 연동**:
  - /v1/chat: QueryEngine.submit_message() → ChatResponse
  - /v1/chat/stream: QueryEngine → SSE 스트리밍
  - /v1/tools: ToolRegistry.get_all_tools() → 도구 목록
- **cli/repl.py + cli/commands.py 연동**:
  - _bootstrap()에서 Phase 2 초기화 (QueryEngine 자동 생성)
  - ask 커맨드: 비대화형 QueryEngine 호출
- **core/orchestrator/query_loop.py — HookManager Transition 5**:
  - hook_manager 파라미터 추가
  - HookEvent.STOP → BLOCK이면 강제 다음 턴 (stop_hook_blocking)
- **core/tools/implementations/memory_tools.py**:
  - MemoryRead: MemoryManager.search_relevant() 벡터+텍스트 검색
  - MemoryWrite: MemoryManager.add_semantic() 장기 메모리 저장
- **core/tools/implementations/agent_tool.py**:
  - 서브 에이전트: 독립 QueryEngine 생성 + DISALLOWED_TOOLS 필터링
  - max_turns=10 제한, 부모 context 연결
- 352개 테스트 통과 (기존 테스트 모두 유지)

---

## TaskManager 구현 (완료)

### v0.9.2 — TaskManager + task_tools 연동 (2026-04-15)

- **core/task.py** (신규):
  - TaskType (7종): LOCAL_BASH, LOCAL_AGENT, REMOTE_AGENT, TEAMMATE, WORKFLOW, MONITOR, TRAINING
  - TaskStatus (5종): PENDING, RUNNING, COMPLETED, FAILED, KILLED
  - TaskState (Pydantic BaseModel): id, type, status, description, progress, error_message, result
  - TaskManager: create, run, run_background, kill, update_progress, get_active, on_complete, cleanup_old
- **core/tools/implementations/task_tools.py** (수정):
  - TodoRead/TodoWrite/TaskTool → TaskManager API 연동 (폴백 유지)
  - _handle_stop() → async + TaskManager.kill() 사용
  - killed 상태 아이콘 [K] 추가
  - TaskState ↔ dict 변환 유틸 추가
- **core/bootstrap.py** (수정):
  - init_phase2()에 TaskManager 인스턴스 생성 + context.options 주입
- **tests/unit/test_task_manager.py** (신규, 28개 테스트):
  - TaskType/TaskStatus enum 검증 (5개)
  - TaskManager 라이프사이클: create, run, run_background, kill, progress, callback, cleanup (18개)
  - task_tools + TaskManager 연동 검증 (5개)
- 380개 테스트 통과 (기존 352 + 신규 28)

---

## GPU E2E + 벤치마크 + Soak Test (완료)

### v0.9.3 — 실 GPU 서버 연결 + 성능 측정 (2026-04-15)

- **서버 포트 분리**:
  - config/nexus_config.yaml: gpu_server.url → :8001, embedding_url → :8002
  - core/config.py: GPUServerConfig에 embedding_url 추가
  - core/model/inference.py: embedding_base_url 파라미터 + /v1/embed 엔드포인트 적용
  - core/bootstrap.py: embedding_base_url 전달
- **tests/e2e/test_gpu_e2e.py** (12개 테스트):
  - 서버 연결, 모델 확인, 텍스트 생성, SSE 스트리밍, tool_calling
  - 임베딩 생성, 배치 임베딩, LocalModelProvider 통합, 3턴 대화
- **tests/e2e/test_benchmark.py** (5개 테스트):
  - Simple Response Time: 0.728s (목표 <1.5s) — 통과
  - Complex Response Time: 7.369s (목표 <8s) — 통과
  - TTFT: 0.031s, TPS: 69.4 tokens/s, Embedding: 32.8ms
- **tests/e2e/test_soak.py** (3개 테스트):
  - 100-Turn 대화: 100/100 완료, 에러 0, 평균 0.314s
  - 50회 연속 추론: 50/50 완료, 성능 변화 +18.6%
  - 동시 임베딩+추론: 성능 저하 +1.7% (목표 <20%) — 통과
- 400개 테스트 통과 (기존 380 + E2E 20)

---

## 웹 채팅 UI + DocumentProcess (완료)

### v0.9.4 — 웹 UI 구현 + 문서 분석 도구 (2026-04-15)

- **web/static/index.html** — Claude 스타일 채팅 UI:
  - idino 색상 팔레트 (#00479d 기반), Nexus 브랜딩 + idino 로고
  - SSE 스트리밍 실시간 응답, 타이핑 커서 애니메이션
  - 사이드바 (세션 목록, 기본 닫힘, 햄버거 버튼으로 오버레이)
  - 파일 첨부 (텍스트 직접 포함 + 바이너리 서버 업로드)
  - 토큰 카운터 (IN/OUT 실시간 표시)
  - PWA 지원 (manifest.json, HTTPS 자체 서명 SSL)
  - 응답 대기 스피너 ("생각하는 중...")
- **web/app.py** — 웹 서버 연동:
  - 세션별 히스토리 격리 (QueryEngine messages 매 요청 초기화)
  - 히스토리 복원 시 토큰 예산 기반 제한 (tool_result 제외, user/assistant만)
  - 파일 업로드 API (POST /v1/upload → 임시 디렉토리 저장)
  - 웹 전용 QueryEngine (도구 8개, 토큰 예산 최적화)
  - HTTPS 자체 서명 SSL 인증서 (config/ssl/)
- **core/tools/implementations/document_tool.py** (신규) — DocumentProcess 도구:
  - PDF (pypdf), DOCX (python-docx), XLSX (openpyxl) 파싱
  - 청크 분할 (2,500자 단위) — 컨텍스트 초과 방지
  - 문서 캐시 — 같은 파일 재파싱 방지
  - chunk_index 파라미터로 순차 청크 읽기
- **core/bootstrap.py** — 웹용 도구 레지스트리:
  - `_create_web_tool_registry()` — 핵심 도구 8개 (Read, Write, Edit, Bash, Glob, Grep, LS, DocumentProcess)
  - 토큰 예산: ~2,051 토큰 (24개 6,102 토큰 대비 67% 절약)
- **core/model/inference.py** — 컨텍스트 초과 자동 처리:
  - 400 에러에서 input_tokens 추출 → max_tokens 자동 축소 재시도 (3회)
  - 재시도 모두 실패 시 "입력이 너무 길어 분석할 수 없습니다" 에러 메시지
  - SSE 파싱을 async with 블록 안으로 이동 (StreamClosed 수정)
- **core/orchestrator/query_loop.py** — 동적 max_tokens:
  - 입력 토큰 추정 후 max_tokens = max(512, max_context - estimated_input - 200)
  - 입력 초과 시 자동 truncate
  - 재시도 시 system 메시지 추가하지 않음 (토큰 증가 방지)
- **cli/repl.py** — Message 객체 필터링:
  - QueryEngine이 yield하는 Message 객체를 건너뛰고 StreamEvent만 표시
- **config** 변경:
  - gpu_server.url: 8001, embedding_url: 8002
  - max_context_tokens: 8192, default_max_tokens: 4096
  - gpu_server_url → property로 전환 (YAML gpu_server.url 자동 동기화)
- **GPU 서버** (192.168.21.112):
  - vLLM: --max-model-len 8192, --gpu-memory-utilization 0.90
  - 도구 호출 활성화: --enable-auto-tool-choice --tool-call-parser gemma4

---

## RTX 5090 (8192 ctx) 실측 제약 사항

```
VRAM 최대 max-model-len:
  8,192 ✅ (GPU 30.7GB/32.6GB)
  16,384 ❌ (OOM 실패)

토큰 예산 (도구 8개):
  도구 스키마:    ~2,500 토큰 (Gemma4 BPE 실측)
  시스템 프롬프트:   ~72 토큰
  고정 비용 합계: ~2,572 토큰

  가용 (입력+출력): ~5,620 토큰
  문서 분석 한계:  ~2,500자 (청크 1개) → A4 1~2페이지

대용량 문서:
  청크 분할로 순차 분석 가능 (무제한)
  단, 각 청크는 독립 분석 (이전 청크 맥락 유실)
```

---

## v7.0 적응형 멀티모델 오케스트레이션 (완료)

### v7.0.0 — Phase 9.0a~c + 9.5 (2026-04-16)

- **설계 개정안**: `user_mig/PROJECT_NEXUS_SPEC_v7.0_AMENDMENT.md` 작성
  - 3가지 불변 전제: Claude Code 설계 유지, GPU 업그레이드 시 성능 향상, 동일 사용자 경험
  - HardwareTier 적응형 (TIER_S/M/L), TIER_M/L에서는 v6.1과 100% 동일 동작
- **Phase 9.0a: 도구 스키마 축소**
  - 11개 CLI 도구 description 영문 축소 (1,588 → 1,472 토큰)
- **Phase 9.0b: TurnState 상태 외부화**
  - `core/orchestrator/turn_state.py` (신규): TurnState, TurnStateStore, extract_turn_state()
  - query_loop에 on_turn_complete 콜백 추가
  - QueryEngine에서 TurnState 컨텍스트 복원 (이전 messages 대신 요약)
- **Phase 9.0c: HardwareTier 감지 + ModelDispatcher**
  - `core/model/hardware_tier.py` (신규): HardwareTier enum, detect_hardware_tier()
  - `core/orchestrator/model_dispatcher.py` (신규): ModelDispatcher (Scout→Worker 분배)
  - bootstrap.py: 티어별 자동 도구 선택 + TurnState + Scout 초기화
- **Phase 9.5: Scout 통합 (llama.cpp CPU 모델)**
  - GPU 서버에 llama.cpp b8808 + Gemma 4 E4B (Q4_K_M, 4.7GB) 설치
  - Scout 서버: :8003 (CPU, ~16 TPS), 방화벽 8003 포트 개방
  - `core/model/scout_provider.py` (신규): ScoutModelProvider
  - `core/config.py`: ScoutConfig + hardware_tier 필드 추가
  - Scout → Worker 핸드오프: _run_scout() 구현 (탐색→계획→Worker 실행)
  - Scout 실패 시 Worker 단독 fallback
- **버그 수정**:
  - 토큰 0/0 수정 (inference.py: finish_reason 후 usage 청크 대기)
  - CONTEXT_OVERFLOW 즉시 종료 (query_loop.py: 무한 재시도 방지)
  - GPU health 경고 수정 (bootstrap.py: vLLM 빈 body 대응)
  - 웹 시스템 프롬프트 수정 (파일 분석/코드 리뷰 중심)
- **Chrome 스타일 웹 UI**: `web/static/chrome.html` (시연용)
- **453개 테스트 통과** (기존 380 + v7.0 신규 73)

---

## v7.0 추가 구현 (2026-04-16 후반)

### Ch 7 WithRetry + StreamWatchdog
- `core/orchestrator/retry.py` (신규): ErrorCategory(9종), classify_error(), with_retry() AsyncGenerator
- `core/orchestrator/stream_watchdog.py` (신규): StreamWatchdog, stream_with_watchdog()
- query_loop에 stream_with_watchdog 통합 + StreamWatchdogTimeout 처리

### RAG 파이프라인
- `core/rag/indexer.py` (신규): ProjectIndexer (파일 탐색→청크→임베딩→저장)
- `core/rag/retriever.py` (신규): RAGRetriever (쿼리 임베딩→벡터 검색→컨텍스트 주입)
- QueryEngine.submit_message()에서 RAG 컨텍스트 자동 주입
- bootstrap에서 백그라운드 인덱싱 (fire-and-forget)

### Redis/PostgreSQL 실 연결
- PostgreSQL: 192.168.10.39:5440 (nexus/idino@12, DB=nexus)
  - pgvector v0.8.0 소스 빌드 설치 (docutil-postgres 컨테이너)
  - tb_memories 테이블 + 인덱스 5개 (idx_ 규칙)
  - 네이밍 규칙: tb_, idx_, vtb_, vw_, fn_, proc_
- Redis: 192.168.10.39:6340 (pw=docutil_redis_2024, db=6)
- bootstrap.py: _create_redis_client(), _create_pg_pool() (실패 시 인메모리 폴백)
- long_term.py: 테이블명 memories → tb_memories

### 모델 전환: Gemma 4 31B → Qwen 3.5 27B
- **이유**: Gemma 4는 vLLM LoRA/AWQ 미지원. Qwen은 전부 지원.
- **AWQ 서빙**: Qwen 3.5 27B AWQ (21GB) — vLLM :8001, VRAM 29GB
- **LoRA 학습**: 원본 Qwen 3.5 27B (52GB) + unsloth 4bit QLoRA
  - 1,000 샘플 (24개 도구 + 6개 추론 카테고리), 61분, loss 0.073
  - 체크포인트: /opt/nexus-gpu/checkpoints/qwen35-phase1/ (153MB)
- **LoRA 핫로딩 성공**: vLLM `--enable-lora --lora-modules nexus-phase1=...`
  - `"model": "qwen3.5-27b"` (기본) / `"model": "nexus-phase1"` (LoRA) 동적 전환
- **tool-call-parser**: `qwen3_xml` (hermes 아님 — Qwen 3.5 전용)
- **코드 수정**: 14개 파일 + 테스트 Gemma→Qwen 일괄 변경
- **Bootstrap 템플릿**: 7개 도구 → 24개 도구 전체 확장
- httpx 로그 레벨 WARNING으로 변경 (CLI 프롬프트 덮어쓰기 방지)
- 시스템 프롬프트: 범용 AI 어시스턴트로 변경 + thinking 출력 억제

### 테스트: 532개 통과

---

## 멀티테넌시 기반 배선 — Part 5 Ch 15 M1~M6 (2026-04-21 야간)

### 배경
학교·기업별 서비스를 위해 단일 Nexus 인스턴스에서 LoRA 어댑터와 RAG 지식
베이스를 **tenant 단위로 분리**한다. 데이터는 한 DB에 `source` 컬럼으로
섞여 있고, 요청의 tenant_id가 허용 source/LoRA를 결정한다.

### 구현 범위 (M1~M6)

| # | 항목 | 범위 |
|---|---|---|
| M1 | TenantConfig + tenants.yaml | Pydantic 모델 + 별도 YAML 파일 |
| M2 | 세션 진입 시 tenant 식별 | body/헤더/Bearer API 키 3경로 |
| M3 | QueryEngine 라우팅에 tenant.model_override | KNOWLEDGE 질의에만 적용 |
| M4 | KnowledgeRetriever tenant 필터 | allowed_sources 전파 |
| M5 | 권한 — DB-level cross-tenant 차단 | `source = ANY($n::text[])` |
| M6 | /metrics 테넌트별 분해 | registered 목록 + per-tenant 요청 카운트 |

### 신규 파일
- `config/tenants.yaml` — 기본 `default` 테넌트, 학교/기업 추가 예시 주석
- `tests/unit/test_tenant_registry.py` — 10 케이스

### 수정
- `core/config.py` — TenantConfig/TenantRegistry + tenants.yaml 별도 로더
- `core/rag/knowledge_store.py` — `search_by_vector(allowed_sources=...)` DB-level
  필터 + 인메모리 폴백도 동일 규칙
- `core/rag/knowledge_retriever.py` — `get_context(allowed_sources=...)` 전파
- `core/orchestrator/query_engine.py`
  - `context.options["tenant"]`에서 TenantConfig 조회
  - KNOWLEDGE 질의 시 tenant.model_override로 라우팅 LoRA 덮어쓰기
  - tenant.allowed_knowledge_sources를 retriever로 전달
- `web/app.py`
  - `_resolve_tenant(body/header/bearer)` 3경로 해석 + per-tenant 카운트
  - ChatRequest에 `tenant_id` 필드 추가
  - `/v1/chat` 및 `/v1/chat/stream`이 X-Tenant-ID/Authorization 헤더 수신
  - `/metrics` 응답에 `tenants` 섹션 추가

### 라우팅 규칙
- KNOWLEDGE 질의: `tenant.model_override`가 있으면 `RoutingProfile.model`을 덮어씀
- TOOL 질의: tenant override 무시 (Phase LoRA가 tool_call XML 학습 보유)
- 기본 테넌트(`default`): `kowiki + sample`만 허용, LoRA override 없음

### 테스트
- 10/10 통과 — registry get/resolve, API 키 매핑, allowed_sources 필터 차단,
  KNOWLEDGE vs TOOL 라우팅 분기
- 회귀 **699/699** (689 + 10)
- ruff clean (신규/수정 파일 기준)

### 하위 호환
- tenants.yaml이 없으면 `default` 단일 테넌트 자동 사용
- 기존 `search_by_vector(source=...)` 호출은 `[source]`로 정규화되어 동작
- tenant 기능을 쓰지 않으면 기존 동작과 완전 동일

### 운영 예시
```bash
# A학교 질의 (API 키)
curl -H "Authorization: Bearer sk-school-a-demo" -d '{"message":"..."}' /v1/chat

# 또는 헤더
curl -H "X-Tenant-ID: school-a" -d '{"message":"..."}' /v1/chat
```

### 남은 고도화 여지
- Audit 로그에 tenant_id 기록 (지금은 로그 라인에만)
- Permission Layer에 명시적 tenant 훅 (현재는 DB 필터로 충분)
- tenant별 temperature/max_tokens 프로필 덮어쓰기 (TenantConfig 확장)
- M7 — 테넌트별 LoRA 학습 네이밍 컨벤션(`nexus-{tenant}-phaseN`) + 학습 스크립트

---

## Phase 10.0 다언어 심볼 파서 (2026-04-21 야간 확장)

### 배경
Phase 10.0 초판은 Python만 지원. JavaScript/TypeScript/Go로 확장하려면
파서 플러그인 구조가 필요했다. 에어갭 제약상 tree-sitter/esprima 등 외부
라이브러리 사용 불가 → 표준 `ast`(Python) + 정규식(JS/TS/Go) 조합으로 구현.

### 신규 파일
- `core/rag/parsers/__init__.py` — 기본 레지스트리 팩토리
- `core/rag/parsers/base.py` — BaseParser ABC + ParsedSymbol + ParserRegistry
- `core/rag/parsers/python_parser.py` — 기존 ast 로직 재포장
- `core/rag/parsers/javascript_parser.py` — JS/TS 공통 regex 파서
  - function / arrow / class / method (async/static 태그)
  - TypeScript: interface / type alias
  - JSDoc (`/** ... */`) 캡처
- `core/rag/parsers/go_parser.py` — Go regex 파서
  - func / method(receiver) / struct / interface / type alias
  - 대문자 시작 심볼에 `exported` 태그
  - `// doc comment` 캡처
- `tests/unit/test_multilang_parsers.py` — 14 케이스

### 수정
- `core/rag/symbol_indexer.py`
  - `iter_source_files()` 추가 (확장자 인자)
  - `extract_symbols_from_source`에 `parser` / `registry` 선택 파라미터
  - `SymbolProjectIndexer`가 ParserRegistry 경유로 다언어 라우팅
  - 하위 호환: Python-only 호출 경로 그대로 동작

### 검증
- 단위 14건 전부 통과, 회귀 **679/679** (665 + 14)
- JS/TS 샘플: jsdoc 캡처, static/async 태그, qualified name(`Class.method`) 정확
- Go 샘플: method receiver → `Type.Method`, exported 태그, doc comment 포함
- 다언어 혼합 트리 E2E: py/js/ts/go 4파일 → 20+ 심볼 언어별 태그 확인

### 디자인 결정
- `_RE_*`의 들여쓰기는 `\s*` 대신 `[ \t]*`로 고정 — `\s`가 개행까지 소비해
  `match.start()`가 0이 되면서 JSDoc prefix가 빈 문자열이 되는 버그를 회피
- 주석/문자열을 같은 길이의 공백으로 scrub 후 정규식 매치 → 오탐 감소
- 파서 에러는 조용히 빈 리스트 반환 — 단일 파일 실패가 전체 인덱싱을 망치지 않도록

### 남은 여지
- Rust/Java 등 추가 언어는 동일 BaseParser 상속으로 추가
- 호출 그래프(caller/callee) 저장은 별도 작업
- JS arrow function의 `line_end` 현재 `+5`로 근사

---

## Phase 10.0 Python 심볼 인덱싱 구현 (2026-04-21 야간)

### 배경
기존 ProjectIndexer(tb_memories)는 **파일 청크 단위** 인덱싱이라 "query_loop
함수 어디 있어?" 같은 심볼 단위 질의에서 청크 경계와 의미 경계가 어긋남.
사양서 Phase 10.0 로드맵의 "심볼 인덱스 구축 + 채팅에서 인덱스 조회" 부분을
Python 범위에서 완성.

### 신규 파일
- `core/rag/symbol_store.py` — SymbolEntry, SymbolStore, ensure_schema,
  build_vector_index, add/add_many/delete_by_path/search_by_name/search_by_vector
- `core/rag/symbol_indexer.py` — Python `ast` 기반 심볼 추출 + SymbolProjectIndexer
  (파일 단위 트랜잭션 + 배치 임베딩)
- `core/tools/implementations/symbol_search_tool.py` — SymbolSearch 도구
  (읽기 전용, is_concurrency_safe=True)
- `tests/unit/test_symbol_indexer.py` — 16 케이스 (ast 추출, 스토어 CRUD, 벡터 검색,
  도구 동작, 자연어→벡터 폴백)

### 스키마 (tb_symbols)
```
id | source | path | module | kind | name | qualified_name | signature | docstring
| summary | line_start/end | tags | embedding vector(1024) | created_at | metadata
```
인덱스 5종: btree(source/path/kind/name), GIN+pg_trgm(name/qualified_name),
ivfflat cosine(embedding, 대량 적재 후 빌드).

### 배선
- `core/bootstrap.py` Phase 2 ⑨-c에서 SymbolStore 초기화 + 백그라운드 인덱싱
- `ToolUseContext.options["symbol_store"]` 주입 (CLI + Web 양쪽)
- Scout/CLI/Web 3개 도구 풀 모두에 SymbolSearchTool 등록

### 실측 (Nexus 프로젝트 자기 인덱싱)
- 전체 심볼: **1,400+** (async_function 33, async_method 288, class 212,
  function 178, method 714)
- SymbolSearch 평균 응답: **50~60ms** (pg_trgm 인덱스)

### E2E 검증
| 질의 | 결과 |
|---|---|
| "query_loop 함수는 어느 파일 몇 번째 줄?" | **core/orchestrator/query_loop.py:130** 정확 (6.4s) |
| "KnowledgeStore.add 시그니처 알려줘" | `async def add(self, entry: KnowledgeEntry) -> str` + 파일/줄 정확 (13.3s) |

로그: `SymbolSearch: 'query_loop' → 2` 0.06초, `'KnowledgeStore.add' → 10` 0.05초.

### 테스트 / 회귀
- 신규 16 케이스 전부 통과
- 회귀 **681/681** (665 + 신규 16)
- ruff: clean

### 사양서
`PROJECT_NEXUS_SPEC_v7.0_AMENDMENT.md` Phase 10.0 섹션 구현 실태로 재작성.

### 향후 확장
- 다언어(JS/TS/Go) 심볼 파서 추가
- 호출 그래프(caller/callee) 관계 저장
- ivfflat 인덱스는 대량 적재(>1k) 후 별도 빌드 권장

---

## kowiki 500건 실 적재 (2026-04-21 야간)

### 실행
서비스 중이 아닌 틈을 타서 실 kowiki 덤프 적재까지 완료.

**1) 다운로드 (tmux kowiki_dl)**
- `/opt/nexus-gpu/corpora/kowiki/kowiki-latest-pages-articles.xml.bz2` (1,270,552,032 byte, 1.2GB)
- wget 1회차: 초기 중복 프로세스로 CRC 손상 → `bzip2 -t` 실패
- wget 2회차: 단일 tmux 세션, 2.61 MB/s 평균, **bzip2 -t 통과**

**2) 적재 (tmux kowiki_ingest)**
- `/opt/nexus-gpu/kowiki_ingest/{scripts,core/rag}` 에 의존 파일만 전송
- `venv`에 `asyncpg`, `httpx` 설치
- 카테고리 필터: 철학, 문학, 역사, 인물 / limit 500
- 수정 사항 (런타임에 발견):
  - **XML namespace**: `export-0.10` 하드코딩 → `_find_child()` namespace-agnostic 헬퍼
  - **임베딩 서버 스펙**: OpenAI `/v1/embeddings` → Nexus `/v1/embed` (`{"texts":[...]}` → `{"embeddings":[...]}`)
  - **한글 인자 인코딩**: wrapper 스크립트(`run_ingest.sh`)에 `LANG=C.UTF-8 LC_ALL=C.UTF-8` 설정
- 최종 결과: **6,431 페이지 스캔 → 500 문서 / 3,865 청크 적재** (약 14분)

**3) ivfflat 인덱스 빌드**
- `idx_knowledge_embed (ivfflat cosine, lists=100)` + 보조 4종(source/title/tags/pkey)

### 실 E2E 검증

| 질의 | 응답 시간 | input_tokens | 결과 |
|---|---|---|---|
| 니체 철학 핵심 개념 3가지 | 29.8s | 3,218 | **위버멘쉬/힘에의 의지/영원 회귀** 정확 + 상세 |
| 니체 생애와 사상적 배경 | 34.3s | 3,311 | 1844~1900, 주요 저서, 기독교 비판 등 정확 |

input_tokens가 평소(~1,500)보다 1,700 이상 커진 것 = kowiki RAG 청크가 시스템 프롬프트에 주입됐다는 직접 증거.

### DB 상태
```
 source | chunks | docs
--------+--------+-----
 kowiki |   3865 |  500
 sample |      3 |    3
```

적재된 샘플 제목(발췌): 19세기 프랑스 문학, 고대 그리스 문학, 과학철학, 기(철학),
소크라테스 이전 철학자, 시몬 베유(철학자), 프랑스 문학, 천문학 등.

### 스크립트 수정사항 (prepare_kowiki.py)
- `iter_pages()`: namespace-agnostic `_find_child()` 헬퍼로 교체
- `_embed_texts()`: Nexus 임베딩 서버 실 스펙(`/v1/embed` + `{texts}`)에 맞춤

---

## Part 2.5.8 지식 RAG 파이프라인 구현 (2026-04-21 저녁)

### 배경
Part 2.5 라우팅으로 KNOWLEDGE_MODE에서 베이스 Qwen이 훨씬 정확해졌지만,
한국어 인문학·전문 지식은 여전히 부족. **외부 지식 베이스(위키)를 pgvector로
인덱싱하고 KNOWLEDGE 질의 시 자동 주입**하는 RAG 계층을 추가하여 근본 해결.

### 신규 파일
- `core/rag/knowledge_store.py` — KnowledgeStore + KnowledgeEntry + split_into_chunks
  - 인메모리 폴백 지원 (pg_pool=None)
  - add_many / search_by_vector / search_by_text / count / list_sources
  - build_vector_index (ivfflat, 대량 적재 후 1회)
- `core/rag/knowledge_retriever.py` — 검색 결과 → 시스템 프롬프트 블록 포매터
  - 토큰 예산 내 청크 연결, 임베딩 실패 시 텍스트 검색 폴백
- `scripts/prepare_kowiki.py` — kowiki 덤프 적재 스크립트 (운영 코드와 격리)
  - bz2 스트리밍, 카테고리 필터, 1,200자 청크, 배치 임베딩, UPSERT
- `tests/unit/test_knowledge_store.py` — 17 단위 테스트

### 스키마 (tb_knowledge)
```
id (text PK) / source / title / section / content / chunk_index /
total_chunks / tags (text[]) / embedding (vector(1024)) / created_at / metadata (jsonb)
인덱스: idx_knowledge_source/title/tags(GIN) + idx_knowledge_embed (ivfflat cosine)
```
tb_memories와 **별도 테이블** — 대화 EPISODIC 데이터와 혼재 방지.

### 수정 파일
- `core/orchestrator/query_engine.py` — `knowledge_retriever` 인자 + KNOWLEDGE 분류
  시 `effective_system_prompt`에 검색 결과 자동 주입
- `core/bootstrap.py` — Phase 2 ⑨-b에 KnowledgeStore 초기화 + ensure_schema()
- `web/app.py` — 웹 QueryEngine에 knowledge_retriever 주입

### 에어갭 준수
운영 Nexus 코드에는 외부 URL 없음. 덤프 다운로드는 GPU 서버에서 별도 wget으로
수행한 뒤 `scripts/prepare_kowiki.py`로 적재. `anti-patterns.md` #10 위반 없음.

### E2E 검증 (샘플 3건)
수동 큐레이션 위키 스타일 요약(차라투스트라/변신/니체)을 tb_knowledge에 적재 후
웹 요청으로 검증:

| 질의 | 라우팅 | RAG | 결과 |
|---|---|---|---|
| "차라투스트라를 세 가지 변신 중심으로" | KNOWLEDGE | 1,190자 주입 | 낙타-사자-아이 정확 (41.9s, 790토큰) |
| "카프카의 변신 줄거리와 주제는?" | KNOWLEDGE | 1,190자 주입 | 그레고르 잠자/소외 정확 (48.8s, 958토큰) |

실 로그:
```
라우팅: class=KNOWLEDGE, model=qwen3.5-27b, temp=0.20, max_tokens=2048
지식 RAG 주입: ~1190자
```

### 테스트
- 17/17 통과 (split/id 결정론성/폴백 CRUD/벡터 정렬/텍스트 검색/retriever 포맷·예산·폴백)
- 회귀 **649/649** 통과 (632 + 17)
- ruff: 신규 파일 clean

### 사양서
`PROJECT_NEXUS_SPEC_v7.0_AMENDMENT.md` Part 2.5.8 섹션 전면 재작성 (스키마/적재
스크립트/E2E 결과/한계 및 향후 과제).

### 남은 작업
- **kowiki 실 덤프 적재** (GPU 서버, 1~수 시간) — 사용자가 백그라운드로 수행
- **Phase 10.0 RAG 심볼 인덱싱** — 우선순위 낮음

---

## Ch 6 ContextManager 티어별 전략 공식화 (2026-04-21 저녁)

### 배경
TurnStateStore(TIER_S 컨텍스트 요약)는 이미 구현되어 있었으나, ContextManager는
여전히 모든 티어에서 4단계 압축 파이프라인을 돌리도록 설계되어 있어 사양서
Part 5 Ch 6의 "티어별 전략" 의도와 어긋남. QueryEngine이 bootstrap에서
`context_manager=None`으로 초기화되어 있어 실질적으로는 사용되지 않았지만,
미래에 ContextManager 주입이 필요해질 때 TIER_S에서 중복 압축이 발생할 위험.

### 구현
- `core/orchestrator/context_manager.py`
  - `__init__`에 `tier: HardwareTier | None = None` 파라미터 추가
  - `_passthrough` 플래그: TIER_S이면 True
  - `apply_all`, `auto_compact_if_needed`, `emergency_compact` 각각에 pass-through
    분기 추가 (TIER_S면 TurnStateStore가 담당하므로 no-op 또는 최근 1턴만 추출)
  - `stats` 프로퍼티에 `tier`, `passthrough` 필드 노출
- `core/bootstrap.py` — Phase 2 ⑪단계에서 ContextManager 생성 + QueryEngine 주입
- `web/app.py` — 웹 QueryEngine에도 동일 인스턴스 공유

### 테스트
- `tests/unit/test_context_manager_tier.py` 신규 8건
  - TIER_S passthrough 검증 (apply_all/auto_compact/emergency)
  - TIER_M/L/None 기존 동작 유지 확인
  - stats 필드 노출 확인

### 회귀
- 632/632 통과 (624 + 신규 8)
- ruff: 신규 파일 전부 clean
- 서버 기동 로그 실측: `ContextManager 초기화: tier=small, passthrough=True`

### 하드웨어 업그레이드 시 자동 수렴
`detect_hardware_tier()`가 H100을 감지하면 `_passthrough=False`로 자동 전환되어
v6.1 4단계 압축 파이프라인이 활성화. 설정 변경 없이 수렴.

### 사양서 반영
`PROJECT_NEXUS_SPEC_v7.0_AMENDMENT.md` Ch 6 섹션을 구현 실태에 맞게 재작성.

---

## Ch 16 세션 영속화 구현 (2026-04-21 오후)

### 배경
진단 결과 웹 채팅 히스토리가 `_app_state["chat_histories"]` 인메모리 dict에만
저장되고 있어 서버 재기동 시 전부 소실. Redis/PG 연결은 성공했지만 **QueryEngine이
MemoryManager를 호출하지 않아** 저장 기능이 연결되지 않은 "유령 구성" 상태였음.

### 구현

**신규 모듈**:
- `core/memory/transcript.py` — SessionTranscript, list_transcript_sessions
  - `{sessions_dir}/{session_id}/transcript.jsonl`에 append-only 기록
  - 턴 단위 JSON Lines (ts, role, content, turn, usage)

**수정**:
- `core/memory/short_term.py` — `list_sessions(limit)` 추가 (Redis SCAN)
- `core/orchestrator/query_engine.py` — `memory_manager`, `transcript` 파라미터 추가,
  `_finalize_turn()` 내부 헬퍼로 submit_message 말미에서 호출
- `core/bootstrap.py` — CLI용 SessionTranscript 생성 + QueryEngine에 주입
- `web/app.py`
  - `_app_state["memory_manager"]` 노출
  - `/v1/chat`, `/v1/chat/stream` 핸들러에 session_id 세팅 + Redis 복원 + write-through
  - `/v1/sessions` 엔드포인트가 실제 저장된 세션 목록 반환

**저장 3단 구조**:
| 매체 | 용도 | TTL |
|---|---|---|
| 인메모리 chat_histories | 프로세스 내 빠른 접근 | 소멸 |
| Redis session:{id}:context | 재기동 후 복원 | 24h |
| {sessions_dir}/{id}/transcript.jsonl | 영구 감사 기록 | 영구 |
| tb_memories (중요 턴) | 의미 검색/장기 승격 | 영구 |

### E2E 검증

| 단계 | 결과 |
|---|---|
| 턴 1 "내 이름은 홍길동이야" (SID=persist-e2e-001) | 응답 2.7s, Redis+JSONL 기록 |
| 턴 2 "내 이름이 뭐였지?" | 응답 0.7s, "홍길동" 정확 복기 |
| **서버 재기동 후** 같은 SID로 재질의 | **messages=3개 Redis 복원** → "홍길동입니다" 1.2s |
| `/v1/sessions` | 디스크 1 + Redis-only 1 총 2개 반환 |

### 테스트
- `tests/unit/test_session_persistence.py` 신규 (11 케이스)
  - SessionTranscript 기록/비활성/목록 조회
  - ShortTermMemory.list_sessions 인메모리 폴백
  - QueryEngine._finalize_turn (memory 호출, swallow 에러, noop, 실파일 기록)
- 회귀 624/624 통과 (기존 613 + 신규 11)

### 사양서
- Part 5 Ch 16 섹션 전면 업데이트 (아키텍처·저장 매체·E2E 결과)

### 남은 작업 (후순위)
- **Ch 6 ContextManager 티어별 전략 분기** — TIER_S는 TurnStateStrategy,
  TIER_M/L은 기존 4단계 압축 (현재 TurnStateStore는 있지만 context_manager 공식 분기 미구현)
- **Phase 10.0 RAG 심볼 인덱싱** (선택, 큰 작업)
- **Part 2.5.8 인문학 RAG 지식 베이스** (사용자 결정 대기)

---

## Scout 29자 수렴 근본 해결 (2026-04-21 오후)

### 배경
Part 2.5(쿼리 라우팅) 도입 후 실측에서 Scout가 여전히 "29자"만 Worker에 전달하는
증상 지속. α 진단(30분)으로 세 층의 누적 버그 확정.

### α 진단 결과
Scout 서버(:8003)에 직접 curl로 6개 케이스 비교:

| # | enable_thinking | finish_reason | out tokens | content 길이 |
|---|---|---|---|---|
| A | 미주입 | stop | 255 | 478자 (정상) |
| **B** | **False (Nexus 기본)** | **tool_calls** | **28** | **0** (버그 재현) |
| C | True | stop | 348 | 451자 (정상) |

→ `chat_template_kwargs={"enable_thinking": False}`가 Qwen3.5-4B에서
  거짓 tool_call 1개 뱉고 28토큰 조기 종료 유발. Worker 27B에서는 무해하지만
  4B에서는 치명적.

**추가 발견 (E2E 재현 중)**:
- Part 2.5 라우팅이 Scout 서브에이전트 QueryEngine에도 적용되어 Scout 서버에
  `model=nexus-phase3`(존재하지 않는 모델명) 주입
- llama.cpp가 `<think>` 블록을 reasoning_content로 자동 분리하는데 Nexus SSE
  파서가 이 필드를 버려서 Scout 출력의 대부분이 Worker에게 전달되지 않음

### 해결 (코드 6곳, 모델 교체 없음)

1. `core/model/inference.py` — `stream()` 시그니처에 `enable_thinking: bool | None`
   추가, `None`이면 `chat_template_kwargs` 생략
2. `core/model/scout_provider.py` — `ScoutModelProvider.stream()` 오버라이드로
   `enable_thinking`을 항상 `None`으로 강제
3. `core/orchestrator/query_engine.py` — `routing_config.enabled=False`일 때
   `model_override/temperature/max_tokens_cap/enable_thinking` 모두 기본값으로
   돌려놓는 분기 추가
4. `core/tools/implementations/agent_tool.py` — 서브에이전트 QueryEngine 생성 시
   `routing_config=RoutingConfig(enabled=False)` 주입
5. `core/model/inference.py` — `delta.reasoning_content`를 THINKING_DELTA로 yield
   (프로바이더 플래그 `_include_reasoning_as_text=True`면 TEXT_DELTA로 병합)
6. `core/model/scout_provider.py` — Scout는 `_include_reasoning_as_text=True`
   설정하여 reasoning을 Worker에게 전달

### 실 E2E 검증 (13:04)

`"core/config.py 파일에 어떤 설정 클래스들이 정의되어 있는지 알려줘"`:

| 지표 | 수정 전 | 수정 후 |
|---|---|---|
| Scout → Worker 전달 길이 | 29자 | **1,211자** (42배) |
| Scout 출력 토큰 | 28 | 335 |
| 응답 성공 | ❌ timeout 300s | ✅ **75.7s, 완전한 답변** |
| 라우팅 로그 | `model=nexus-phase3` | "라우팅 비활성" |

Worker가 Scout의 4섹션 마크다운 리포트를 받아 247토큰짜리 정돈된 답변 생성
(GPUServerConfig/RedisConfig/PostgreSQLConfig/ModelConfig/SessionConfig/
ScoutConfig/RoutingProfile/RoutingConfig 8개 클래스 나열).

### 테스트
- `tests/unit/test_scout_provider.py` 신규 (7개 테스트)
- 회귀 613/613 통과 (기존 606 + 신규 7)

### 결정: γ(Scout 모델 교체) 철회
α 단독으로 해결됐으므로 4B → 7B 교체는 불필요. Qwen3.5-4B Q4_K_M 유지.

### 사양서 반영
`user_mig/PROJECT_NEXUS_SPEC_v7.0_AMENDMENT.md` Part 2.3에 **3차 개정 (2026-04-21)**
절을 추가하여 위 6개 수정 지점과 근거를 문서화.

---

## v7.0 Part 2.5 — 쿼리 라우팅 도입 (2026-04-21)

### 배경
사용자 실사용 중 "차라투스트라는 이렇게 말했다 설명해줘" 질의에서 모델이
"카프카의 소설"로 오답. 원인 분석 결과:
- Phase 3 LoRA가 도구 호출·기술 지식을 강화하는 대신 일반 교양 지식의
  표현을 좁히는 부작용 발생
- `chat_template_kwargs={"enable_thinking": False}` + temperature 0.7로
  자체 검증 없이 첫 연상을 그대로 출력

### 0단계 진단 (A/B/C/D 실측)

같은 질문(`차라투스트라는 이렇게 말했다`)에 대한 4가지 조합 curl 비교:

| # | model | thinking | 언어 | 결과 |
|---|---|---|---|---|
| A | nexus-phase3 + thinking=False (현 운영) | OFF | KO | 정답 + 경미한 할루시네이션 ("알렉산더 폰 훔볼트 풍자" 등) |
| B | qwen3.5-27b (LoRA OFF) + thinking=False | OFF | KO | **완벽 답변** — 낙타/사자/아기 3변신 상세 설명 |
| C | nexus-phase3 + thinking=True | ON | KO | A보다 개선, B보다 약함 |
| D | qwen3.5-27b + thinking=True | ON | EN | content에 thinking leak되어 답변 잘림 |

**결론**:
- LoRA OFF(B)가 일반 지식에서 압도적 우위 → LoRA가 원흉 확정
- `enable_thinking=True`는 leak 이슈로 당분간 사용 보류
- 도구 호출 질의는 여전히 Phase 3 LoRA 필요

### 조치 — 쿼리 라우팅 분기 구현

**신규 분류기**:
- `core/orchestrator/query_engine.py` — `classify_query()`, `_resolve_profile()`
- 규칙: `enabled=False` → TOOL / 길이 ≥500 → TOOL / tool_keywords 포함 → TOOL /
  그 외 → KNOWLEDGE

**신규 Pydantic 모델**:
- `core/config.py` — `RoutingConfig`, `RoutingProfile`
- `config/nexus_config.yaml` — `routing:` 섹션

**프로필**:
| 프로필 | model | temperature | max_tokens | enable_thinking |
|---|---|---|---|---|
| knowledge_mode | `qwen3.5-27b` (LoRA OFF) | 0.2 | 2048 | false |
| tool_mode | `nexus-phase3` (LoRA ON) | 0.3 | 4096 | false |

**전파 경로** (4-Tier 파라미터 추가, 기본값은 기존 동작과 동일):
```
QueryEngine.submit_message
  → classify + profile 선택
  → ModelDispatcher.route(model_override, temperature, max_tokens_cap, enable_thinking)
    → query_loop(… 동일 파라미터 …)
      → model_provider.stream(… payload["model"] = model_override …)
```

**수정 파일** (7개):
- `core/config.py` — RoutingConfig/RoutingProfile 신규
- `config/nexus_config.yaml` — routing 섹션 추가
- `core/model/inference.py` — `stream()` 시그니처 확장
- `core/orchestrator/query_loop.py` — 파라미터 전파 + max_tokens_cap 적용
- `core/orchestrator/model_dispatcher.py` — route() 시그니처 확장
- `core/orchestrator/query_engine.py` — 분류기 + submit_message 분기
- `core/bootstrap.py`, `web/app.py` — QueryEngine 생성 시 routing_config 주입

**테스트**:
- `tests/unit/test_query_routing.py` (신규, 28개 케이스)
- `tests/conftest.py` — EnhancedMockModelProvider.stream() 시그니처 동기화
- **회귀 600개 전부 통과** (단위 535 + 통합 65)

### 사양서 개정
`user_mig/PROJECT_NEXUS_SPEC_v7.0_AMENDMENT.md`에 **Part 2.5 신규 추가**:
- 배경, 진단 근거, 설계, 데이터 모델, 4-Tier 영향, 하드웨어 업그레이드 시
  자동 비활성화 조항, 한계와 향후 과제

### TIER_M 이상 업그레이드 시 자동 복귀
- RTX 5090 → H100으로 업그레이드 시 `routing.enabled: false`로 전환하면
  단일 경로(베이스 모델 + 24개 도구)로 복귀 가능
- Phase 3 LoRA 자체가 8K 컨텍스트 우회책이라 TIER_M 이상에서는 불필요

### 다음 단계 (2단계 RAG 지식 베이스 — 보류)
사용자 결정에 따라 2단계(한국어 위키 덤프 → pgvector 인덱싱 → knowledge_mode
진입 시 자동 검색 주입)는 1단계 실측 효과 확인 후 착수.

### 실측 검증 (사용자 확인 필요)
웹 서버 재기동 후 `차라투스트라`/`니체` 등 지식 질의 → 로그에서
`라우팅: class=KNOWLEDGE, model=qwen3.5-27b, temp=0.20` 확인 + 응답 품질 개선.

---

## 사양서 원본 회귀 (경로 ⓐ) — 부분 성공 + docx 이슈 (2026-04-18~19)

### 배경
경로 B 전환(2026-04-17) 후 사용자가 실사용에서 다음 지적:
1. Scout가 짧은 요약만 뱉음 (4B가 27B가 해야 할 요약 작업 수행)
2. Worker가 시키지 않은 파일을 임의 생성(Write logs/debug.log)
3. Worker가 바이너리 파일에 Read fallback 시도

사양서 v7.0 Part 2.3(SCOUT_TOOLS·JSON 출력)과 Part 2.4(Worker는 실행 전용·
Read 필요 없음)와 현재 구현을 전수 대조한 결과 **3가지 중대 이탈** 발견:
- Worker 도구 풀에 Read/Glob/Grep/LS 포함 (Part 2.4 위반)
- Worker system_prompt에 `{scout_plan}` 슬롯 구조 부재 (Part 2.4 위반)
- Scout가 JSON 아닌 자유 텍스트 요약 반환 (Part 2.3 위반)

### 조치 (경로 ⓐ 사양서 원본 회귀)
**1) Worker 도구 축소** — Part 2.4 원본 `{Edit, Write, Bash, GitCommit, GitDiff}` 복원
- CLI Worker: 11개 → 6개 (Edit/Write/Bash/GitCommit/GitDiff/Agent)
- Web Worker: 8개 → 4개 (Edit/Write/Bash/Agent)
- Read/Glob/Grep/LS/DocumentProcess는 **Scout 전용**

**2) SCOUT_AGENT.system_prompt 재작성**
- Worker(27B)가 분석, Scout(4B)는 "탐색·계획"만
- 출력 형식: Part 2.3 원본 JSON → 실측 결과 4B가 JSON 규율 못 지킴 →
  **마크다운 4섹션** (`## relevant_files`, `## file_summaries`, `## plan`,
  `## requires_tools`)으로 **2차 완화** (사양서 Part 2.3 2차 개정으로 문자화)

**3) Worker system_prompt 재작성**
- "You are the 27B brain; Scout is a 4B helper"
- Scout 마크다운 섹션 해석 지시, `## plan` 본문을 factual ground truth로 사용
- "Scout 1회 호출 제한, 재호출 금지" 규칙 명시

**4) 사양서 Part 2.3 2차 개정 추가**
- JSON→마크다운 완화 근거 문서화 (Qwen3.5-4B 능력 한계 반영)
- 정신(4섹션 구조)은 유지, 형식만 완화

### 검증 결과 3/4 통과

| # | 시나리오 | 결과 | 비고 |
|---|---|---|---|
| 1 | 인사 | ✅ 1초, 10토큰 | 깔끔 |
| 2 | 일반 지식 (짜라투스트라) | ✅ 10초, 139토큰 | 구조화된 자세한 답변 |
| 3 | 텍스트 첨부 (DB 로그 분석) | ✅ 44초, 309토큰 | 원인 진단 + 조치 추천 |
| 4 | docx 업로드 → Scout 분석 | ❌ 180~240초 타임아웃 | **아래 미해결 이슈 참조** |

### 해결(2026-04-21): docx 파일 분석 무한 호출 루프

**기존 증상**: docx 업로드 시 Scout가 매 호출마다 **정확히 29자** 짧은 응답을
반환하고 Worker가 7회 이상 재호출하며 240초 타임아웃.

**해결 경로**: α(Scout 서버 직접 디버깅)를 선택하여 원인 확정 후 3가지 배선
수정(2026-04-21 오후, 커밋 9a11b47).

**확정된 원인 (α 진단, 복합)**:
1. `chat_template_kwargs={"enable_thinking": False}`를 Qwen3.5-4B(llama.cpp)에
   전달하면 빈 `<think></think>` 블록 후 거짓 tool_call 1개 뱉고 28토큰에서
   조기 종료 (Worker 27B에는 무해한 설정이 4B에서는 치명적)
2. Part 2.5 쿼리 라우팅이 서브에이전트에도 적용되어 Scout에 `model=
   nexus-phase3`(존재하지 않는 모델명) 주입
3. llama.cpp가 `<think>` 블록을 `reasoning_content`로 자동 분리하는데 Nexus
   SSE 파서가 이 필드를 버려 Scout 출력의 대부분이 Worker에 전달되지 않음

**해결 (6곳 코드 수정)**:
- `enable_thinking`을 `bool | None`으로 확장, `None`이면 `chat_template_kwargs`
  자체를 생략
- `ScoutModelProvider.stream`이 `enable_thinking`을 항상 `None`으로 강제
- `reasoning_content`를 `THINKING_DELTA`로 yield, Scout는 TEXT_DELTA로 병합
- 서브에이전트 QueryEngine은 `RoutingConfig(enabled=False)` 주입

**실측 재검증 (2026-04-21 저녁, 38.7KB docx)**:

| 지표 | 이전 | 현재 |
|---|---|---|
| Scout 호출 횟수 | 7회 이상 (재시도) | **1회** |
| Scout 반환 길이 | 29자 | **1,878자** (65배 ↑) |
| 전체 응답 시간 | 240초 timeout | **88.7초** (정상 완료) |
| 답변 품질 | 실패 | 니체 생애/핵심 개념/세 가지 변신/영향 완벽 커버 |

기존 "다음 세션 작업 후보 α/β/γ"는 α만으로 해결되어 β/γ는 불필요.

### 커밋 상태
- 경로 ⓐ 변경 전체를 한 커밋으로 정리 (docx 이슈 주석 포함)
- 회귀 572/572 통과, 시나리오 1~3 실작동 확인

---

## Scout를 문서 분석 전담자로 복원 — 경로 B (2026-04-17)

### 배경
Phase 3 LoRA 성공 후 실사용 중 15KB docx 파일 업로드 시 "입력 내용이
너무 길어 분석할 수 없습니다" 에러 발생. 로그 분석 결과:
- 1차 시도 input=4097, max_tokens=4096 (합 8193 > 8192 한계)
- 재시도 시 input이 4097 → 4198 → 4299로 증가 (vLLM prefix cache 영향 추정)
- 근본 원인: Worker가 DocumentProcess를 직접 호출하면서 큰 문서 원문이
  Worker 컨텍스트(8K)에 직접 들어가 한계 초과

### 설계 판단: Scout 원래 목적 복원
사용자 지적 — "Scout가 원래 토큰 줄이기 위한 거 아니냐".
사양서 v7.0 Part 2.4의 핵심 원칙 재확인:
> Scout가 이미 읽은 파일 내용을 요약으로 전달한다. Worker는 파일을 다시
> 읽지 않으므로 Read 도구가 필요 없다. Worker 컨텍스트를 실행에만 집중.

현재 B 방식 구현은 이 원칙을 훼손 — DocumentProcess가 Worker 쪽에 있어
큰 문서가 Worker로 직통. 이를 **Scout 전용으로 이관**하여 본래 설계
의도 복원.

### 변경
**1. `core/orchestrator/agent_definition.py`**:
- `SCOUT_AGENT.allowed_tools`: 4개 → 5개 (DocumentProcess 추가)
- `max_turns`: 3 → 5 (문서 청크 순차 처리 위해)
- `description`: "analyzing an uploaded document" 문구로 파일 업로드 시
  Worker의 Scout 자율 선택 유도
- `system_prompt`: "absorb large data sources on behalf of the Worker"
  명시, DocumentProcess 청크 워크플로우 안내

**2. `core/bootstrap.py`**:
- `_create_scout_tool_registry`: DocumentProcess 추가 (5개)
- `_create_cli_tool_registry`: DocumentProcess 제거 (Worker는 Scout 경유)
- `_create_web_tool_registry`: DocumentProcess 제거 (스키마 ~200토큰 절감)

**3. `web/static/index.html`**:
- 파일 업로드 시 자동 생성 메시지를 "DocumentProcess 도구 사용" →
  "Agent(scout)로 분석하고 요약 받아 주세요"로 교체

**4. `core/model/inference.py`**:
- 재시도 마진을 100 → 300으로 확대 (vLLM input 증가 완화)

**5. 사양서 v7.0 AMENDMENT Part 2.3 개정**:
- SCOUT_AGENT 선언 갱신 + 개정 근거 명시 + 하드웨어 업그레이드 시
  자동 복귀 조항 추가

**6. 테스트**: `test_agent_definition.py`와 `test_agent_tool.py`의
`allowed_tools`/`max_turns` 단언 갱신. `_verify_phase.py`에 파일 분석
시나리오(6번째) 추가.

### 중기 계획 — 하드웨어 업그레이드 시 Scout 복잡도 철회

Scout 인프라는 **RTX 5090 8K 컨텍스트 제약을 우회하기 위한 임시 장치**다.
GPU 업그레이드 시 자동 무력화되어야 한다:

| 대상 하드웨어 | Worker 컨텍스트 | Scout 상태 | DocumentProcess |
|---|---|---|---|
| RTX 5090 (현재, TIER_S) | 8K | **필수** (이 개정 적용) | Scout 전용 |
| H100 80GB (TIER_M) | 32K | 비활성 | Worker 직접 사용 가능 |
| H200 141GB (TIER_L) | 128K | 비활성 | Worker 직접 사용 가능 |

**TIER_M/L 복귀 경로**:
1. `core/model/hardware_tier.py::detect_hardware_tier()`가 VRAM 기반으로
   TIER를 판정 → H100 감지 시 자동 TIER_M으로 진입
2. TIER_M/L에서는 `_create_tool_registry()` (24개 전체) 사용 — DocumentProcess
   가 Worker 풀에 자동 포함
3. `components["scout_provider"]`는 None으로 설정되고 ModelDispatcher의
   `scout_available=False`. AgentTool이 Scout 호출 시 명확한 에러 반환
4. Claude Code 원래 구조(단일 Worker + 24 도구)로 자동 수렴 — 추가 수정
   불필요

**공공 납품 방향 권고 (2026-04-17)**:
사용자와 함께 관찰한 통찰 — 현재 겪는 일련의 문제(Phase 2 실패, Phase 3
재학습, Scout 이동 논의)는 전부 "GPU가 8K로 묶여 있어 생기는 파생 문제"다.
본질은 RTX 5090의 KV 캐시 한계.
- 공공 납품 사양 논의 초기에 **H100 80GB 포함 여부**를 테이블에 올릴 것
- 포함되면 Scout 복잡도를 점진적으로 제거 가능하며, 코드 변경 없이 티어
  감지만으로 단순화
- 포함되지 않으면 본 개정(Part 2.3) 상태가 계속 유효

### 실측 검증 계획
웹 서버 재기동 후 15KB docx 파일 업로드:
- Worker가 Agent(subagent_type="scout") 호출 확인
- Scout가 DocumentProcess로 청크 순차 처리
- 요약(200~500토큰)만 Worker에 반환
- 최종 응답 완성 + `/metrics agents.scout.calls` 증가

---

## Phase 3 LoRA 학습 성공 (2026-04-17, 1차 통과)

### 배경
Phase 2 LoRA에서 두 가지 회귀 발견:
1. tool_call 직렬화가 원시 JSON이라 vLLM qwen3_xml 파서가 인식 못 함 →
   assistant 본문에 `{"name": "Agent", "arguments": ...}` 그대로 노출
2. direct_answer 샘플이 짧은 인사·단답 중심이라 장문 설명 능력 퇴보

### 개선
**bootstrap_generator 재구성** (비율: 도구 45% / 추론 25% / 서브에이전트 15% / 지식 15%):
- `_KNOWLEDGE_TEMPLATES` 신규 16건 — OOP/REST/GIL/ACID/HTTP2/CAP/TCP-UDP/
  JWT/K8s/Git/Docker/SQL/ML/테스트/async/GraphQL 각 3~6단락 구조화 답변
- `_SUBAGENT_TEMPLATES` use_scout 프롬프트 10→30 확대 (프로젝트 구조/다중
  파일 검색/리팩토링/영문 프롬프트 다양화)
- `_generate_knowledge_sample()` 추가

**train_qwen_lora_phase3.py (신규)**:
- `tokenizer.apply_chat_template(messages, tools=[AGENT_SCHEMA])` 사용
  → tool_calls가 Qwen3.5 공식 XML 포맷으로 자동 직렬화
    `<tool_call><function=NAME><parameter=KEY>VALUE</parameter></function></tool_call>`
- vLLM `qwen3_xml` 파서와 완전 호환
- 학습 결과: **train_loss=0.06946** (Phase 1 0.073, Phase 2 0.168 대비 최저)
- 학습 시간: 71분, 1000 샘플 × 3 epoch

**자동 검증 스크립트 (`scripts/_verify_phase.py`)**:
5개 시나리오 회귀 검사 — 인사/장문 지식/단일 도구/대규모 탐색/tool_call leak

### 1차 검증 결과 (nexus-phase3 적용 후)

| # | 시나리오 | 결과 | 상세 |
|---|---|---|---|
| 1 | 짧은 인사 | **PASS** | 3.4초, 10 토큰 |
| 2 | 장문 지식 (GIL) | **PASS** | 21.3초, **252 토큰** (Phase 2 대비 6배) |
| 3 | 단일 파일 탐색 | **PASS** | 9.0초, 81 토큰, 도구+답변 |
| 4 | 대규모 탐색 → Agent(scout) | **PASS** | 57.8초, scout_calls 0→1 증가 |
| 5 | tool_call JSON/XML 누출 없음 | **PASS** | 4개 시나리오 모두 clean |

**통과 5/5 — 1차 학습만으로 전 시나리오 통과.** Phase 1 롤백 없이 Phase 3 유지.

### 설정 반영
- `config/nexus_config.yaml`: `primary_model: "nexus-phase3"`
- vLLM: `--lora-modules nexus-phase1=... nexus-phase2=... nexus-phase3=...`
  (세 어댑터 모두 핫로드, 런타임 전환 가능)

### 교훈
- Qwen3.5 chat template의 tool_call 공식 포맷은 중첩 XML(nested function/parameter)
- 학습 데이터 직렬화는 반드시 `tokenizer.apply_chat_template`으로 → 수동 조립은
  포맷 drift 유발
- bootstrap_generator의 카테고리 분포 편중이 응답 스타일 퇴보를 직접 야기함
  (Phase 2의 "짧은 답변 편중" = 장문 능력 퇴보)

---

## Scout 모델 Qwen3.5-4B 전환 + Phase 2 LoRA 학습 완료 (2026-04-17)

### 동기
B 방식 전환 후 초기 실측에서 두 이슈 발견:
1. Scout 모델(Gemma 4 E4B)이 Qwen3.5-27B Worker와 이기종 — 토크나이저/
   chat template 불일치로 hand-off 비용 발생
2. 기존 Phase 1 LoRA는 Agent 도구 사용 경험이 없어 Worker가 필요 상황에서도
   Scout를 호출하지 않고 직접 LS/Glob/Read로 탐색 (컨텍스트 소진)

### 해결 — 두 축 동시 교체
**축 1: Scout 모델 동종화 (Gemma 4 E4B → Qwen3.5-4B)**
- 선택: `unsloth/Qwen3.5-4B-GGUF` Q4_K_M (2.6GB, Worker와 동일 패밀리)
- GPU 서버에 `/opt/nexus-gpu/models/qwen3.5-4b-gguf/` 배치
- llama.cpp 재기동: `--jinja` 플래그로 ChatML + tool_calls 지원
- 코드 변경: `ScoutConfig.model_id = "qwen3.5-4b"`, SCOUT_AGENT description 갱신

**축 2: Phase 2 LoRA 학습 (Agent 도구 사용 학습)**
- Bootstrap 데이터 1000개 재생성 (도구 60% / 추론 30% / **서브에이전트 10%**)
- 서브에이전트 시나리오 세분화:
    subagent_use_scout: 22건 (긍정 — 대규모 탐색)
    subagent_direct_answer: 28건 (부정 — 인사/일반 지식)
    subagent_single_tool: 50건 (부정 — 단일 파일 작업)
  → 부정 샘플이 긍정의 3.5배로 Scout 남용 억제
- GPU 서버에서 Phase 2 학습: `scripts/train_qwen_lora_phase2.py`
  - Worker vLLM 중단 → unsloth 4bit + LoRA r=8, lr=3e-4, 3 epoch
  - `train_runtime: 3566초 (59분)`, `train_loss: 0.1681` (Phase 1 0.073 대비
    카테고리 다양화로 소폭 상승, 정상 범위)
  - 체크포인트: `/opt/nexus-gpu/checkpoints/qwen35-phase2/` (159MB)
- vLLM 재기동: `--lora-modules nexus-phase1=... nexus-phase2=...` 둘 다 노출
- `config/nexus_config.yaml`: `primary_model: "nexus-phase2"`로 전환

### 실환경 검증 (Phase 2 LoRA 적용 후)

| 시나리오 | Phase 1 + Gemma 4 E4B | **Phase 2 + Qwen3.5-4B** |
|---|---|---|
| "Hi there" (인사) | 4초, Agent 0회 ✅ | 4초, Agent 0회 ✅ |
| "core/orchestrator 개요" | 23초, 불완전 응답 | **23초, 한글 요약 완성** |
| "5계층 권한 시스템 전수 조사" | 시도 안 함 | **45초, Agent(scout) 자율 호출** ✅ |

중요 관찰: 5계층 권한 탐색 같은 대규모 요청에서 Worker가 **스스로**
"scout sub-agent를 사용하는 것이 적합합니다"라고 판단 후 Agent 도구를
호출함. Phase 2 학습 전에는 일어나지 않던 행동.

### Scout 메트릭 (/metrics)
```json
"agents": {
    "scout": {
        "calls": 1,
        "total_latency_ms": 25156.0,
        "avg_latency_ms": 25156.0
    }
}
```
Scout 지연 시간 25초 — Gemma 4 E4B 대비 **약 28% 단축** (35초 → 25초).

### 알려진 후속 이슈
- Scout 응답 품질: 25초에 29자 짧은 응답. Qwen3.5-4B Q4_K_M의 CPU 추론
  한계. 개선 방안: 양자화 완화(Q5/Q6) 또는 Scout 전용 후속 LoRA.
- 서브에이전트 학습 샘플 22개는 여전히 적음. 실 사용 데이터 누적 후
  Phase 3 재학습으로 보강 예정.

### 사양서/코드 영향
- `config/nexus_config.yaml` — primary_model을 nexus-phase2로
- `core/config.py`, `core/model/scout_provider.py` — model_id = qwen3.5-4b
- `core/orchestrator/agent_definition.py` — description에 Qwen3.5-4B 명시

---

## v7.0 Phase 9 B 방식 전환 — Scout를 서브에이전트로 승격 (2026-04-17)

### 배경
A 방식(Dispatcher 자동 Scout 전처리) 배선 직후 실측해보니, TIER_S에서는
"안녕" 같은 단순 인사에도 Scout가 조건 없이 선행 실행되어 ~33초가 걸렸다.
Scout는 CPU 4B(llama.cpp) 모델이라 모든 요청에 고정 오버헤드를 더하는 구조가
실사용에서 불합리했다.

사용자 결정: **B 방식으로 전환** — Scout를 자동 전처리기가 아니라
"Worker가 필요할 때 호출하는 서브에이전트"로 승격시킨다. 납품 일정보다 올바른
아키텍처가 우선.

### 구현 (10단계, 각 Phase별 커밋 분리)

**Phase 1 (e134a4c)**: `core/orchestrator/agent_definition.py` 신규
- `AgentDefinition` frozen dataclass (name/desc/system_prompt/allowed_tools/max_turns/model_override)
- `AgentRegistry` — register/get/list_names/list_descriptions
- `SCOUT_AGENT` 상수 — Read/Glob/Grep/LS, max_turns=3, model_override="scout"
- `build_default_agent_registry()` 팩토리
- 21개 테스트 통과

**Phase 2 (4f57de5)**: `core/tools/implementations/agent_tool.py` 확장
- `subagent_type` 파라미터 추가, AgentRegistry 조회
- `model_override="scout"` → `context.options["scout_provider"]` 선택
- `allowed_tools` 기반 도구 필터링 (+ DISALLOWED 이중 보호)
- `AgentTool._stats` 클래스 레벨 통계 + `get_stats()/reset_stats()`
- 하위 호환: subagent_type 없으면 description 기반 ad-hoc 동작
- 14개 테스트 통과

**Phase 3 (242e02b)**: ModelDispatcher의 Scout 자동 전처리 제거
- `_run_scout()` 메서드 완전 제거
- `route()`는 모든 티어에서 passthrough (항상 Worker 직행)
- `stats`는 하위 호환 키 유지하되 값 항상 0, `note` 필드로 AgentTool 안내
- 기존 테스트 3개 제거, 1개 재작성(`test_route_tier_s_now_passthrough`)

**Phase 4 (098896d)**: Worker에게 AgentTool 노출
- `_create_cli_tool_registry`, `_create_web_tool_registry`에 AgentTool() 등록
- `ToolUseContext.options`에 agent_registry/scout_provider/available_tools 주입
- `_build_default_system_prompt(agent_registry)` — 동적 서브에이전트 목록 +
  "NEVER invoke scout for trivial tasks — it is slow" 경고 삽입
- 웹 전용 QueryEngine에도 동일 처리

**Phase 5 (b64a959)**: `/metrics`에 `agents` 섹션 노출
- `AgentTool.get_stats()` → `result["agents"]`
- 하위 호환 `result["scout"]`은 AgentTool 값 평탄화 + note 필드

**Phase 6**: 테스트 정비 (Phase 2/3에 포함)

**Phase 7 (8ca8610)**: `training/bootstrap_generator.py` 서브에이전트 시나리오
- `_SUBAGENT_TEMPLATES` 추가 (use_scout/direct_answer/single_tool 3가지)
- 생성 비율 조정: 도구 60% / 추론 30% / 서브에이전트 10%
- 부정 샘플(direct_answer + single_tool) 비중 높여 남용 억제
- `_generate_subagent_sample()` 추가

**Phase 8 (GPU 서버 작업, 이 저장소 범위 밖)**: LoRA 재학습
- scripts/train_qwen_lora.py 재실행 → qwen35-phase2 체크포인트
- vLLM `--lora-modules nexus-phase2=...`로 핫로드 예정

**Phase 9**: 사양서 v7.0 AMENDMENT 개정
- Part 2 (멀티모델 디스패치 계층) — B 방식 재설계 내용으로 갱신
- Part 5 Ch 14 — AgentDefinition/AgentRegistry/SCOUT_AGENT 정식 명세
- Part 5 Ch 17 — AgentTool.get_stats 기반 메트릭 구조 명시

**Phase 10**: 이 문서 업데이트

### 실환경 검증 (웹 서버 재시작 후)

"Hi, how are you today?" 요청:
- **3초 내 응답 완료**, `tool_calls: []`
- Worker 내부 추론: "simple greeting question. I should respond directly without using any tools"
- Scout 미호출 ✅

"Give me a high-level overview of the core/orchestrator directory" 요청:
- Worker가 `Agent(subagent_type="scout")` 정확히 선택
- 로그: `Agent: 서브에이전트 실행 시작 (type=scout, tools=4개, model_override=scout)`
- Scout 35.3초 실행, Worker가 최종 응답 생성
- `/metrics` 응답: `agents.scout.calls=1, avg_latency_ms=35266`

### 알려진 후속 이슈

- Scout(Gemma 4 E4B) tool_call 실행 품질이 낮음 — 29자 짧은 응답 반환 사례
  Worker가 fallback으로 직접 도구 사용으로 전환되는 합리적 대처 확인
  해결: Phase 8 재학습 이후 Scout 모델을 Qwen 계열 소형으로 교체 고려
- Phase 8 재학습 전에는 Worker의 Scout 선택이 시스템 프롬프트에만 의존 →
  "NEVER invoke for trivial" 경고로 1차 억제, 실측 결과 충분히 보수적으로 동작

### 테스트: 572개 통과 (기존 571 + /metrics agents 섹션 1개)

---

## v7.0 Phase 9 실 배선 + Ch 17 Scout 메트릭 (2026-04-17)

### 발견한 문제
v7.0 Phase 9 코드(HardwareTier/TurnState/ModelDispatcher/Scout)는 모듈
단위로 구현됐고 487줄의 유닛테스트까지 통과했으나, **실제 실행 경로에
연결되지 않은 상태**였다:
- bootstrap에서 `scout_provider`는 만들지만 `ModelDispatcher`는 생성 안 됨
- `QueryEngine.submit_message`가 `model_provider`를 받아 `query_loop`를 직접
  호출 → Scout는 한 번도 실행되지 않음
- 웹 서버 로그에는 "Scout 초기화 성공"이 찍히지만 실 사용자 요청 처리 시
  Scout는 호출되지 않는 "유령 구성" 상태

### 해결
1. **ModelDispatcher 배선** — bootstrap.py에 ⑩ 단계로 Dispatcher 생성 추가,
   `components["model_dispatcher"]`로 저장, QueryEngine 생성자에 주입
2. **QueryEngine 이중 경로** — `model_dispatcher` 파라미터가 주입되면
   `dispatcher.route()` 호출, 없으면 기존 `query_loop()` 직접 경로로 폴백
   (하위 호환 유지)
3. **Scout 전용 도구 팩토리** — `_create_scout_tool_registry()` 신규
   (Read/Glob/Grep/LS 4개, 사양서 Part 2.3 SCOUT_TOOLS와 일치)
4. **웹 전용 Dispatcher** — `web/app.py`에서 웹 도구 8개로 별도 Dispatcher
   생성 (scout_provider/scout_tools는 bootstrap 인스턴스 재사용)
5. **Ch 17 Scout 메트릭 노출** — `ModelDispatcher.stats`에
   `scout_avg_latency_ms`, `scout_fallback_count` 추가, `_run_scout` 성공 시
   elapsed_ms 누적. `/metrics` 엔드포인트에 `result["scout"]` 섹션 추가
6. **사양서 동기화** — v7.0 AMENDMENT의 Ch 3 / Ch 5.9 / Ch 17 블록을 실제
   구현과 일치하도록 갱신 (dispatcher= → model_dispatcher= 등)

### 신규 테스트
- `tests/unit/test_model_dispatcher.py::TestModelDispatcherStatsFields` — 4개
- `tests/unit/test_query_engine_dispatcher.py` — 4개 (주입 경로/폴백/프로퍼티)

### 테스트: 540개 통과 (기존 532 + 신규 8)

---

## 다음 세션 시작 시 참고

1. **Phase 0.5~8.0 + v7.0 전체 완료 + 모델 Qwen 3.5 27B 전환**
2. **기본 참조 문서**: `user_mig/PROJECT_NEXUS_SPEC_v7.0_AMENDMENT.md`
3. **웹 서버 실행**: `python -m uvicorn web.app:app --host 0.0.0.0 --port 8443 --ssl-keyfile config/ssl/key.pem --ssl-certfile config/ssl/cert.pem`
4. **CLI 실행**: `python -m cli.commands chat`
5. **서버 정보**:
   - GPU Worker: 192.168.21.112:8001 (Qwen 3.5 27B AWQ + LoRA, qwen3_xml parser)
   - GPU Scout: 192.168.21.112:8003 (Gemma 4 E4B, llama.cpp CPU)
   - Embedding: 192.168.21.112:8002 (e5-large)
   - DB: 192.168.10.39:5440 (PostgreSQL nexus), :6340 (Redis db=6)
   - 웹: https://192.168.22.223:8443
6. **GPU 서버 모델 목록**:
   - /opt/nexus-gpu/models/qwen3.5-27b-awq/ (21GB, 서빙용)
   - /opt/nexus-gpu/models/qwen3.5-27b/ (52GB, 학습용)
   - /opt/nexus-gpu/models/gemma-4-e4b-it-gguf/ (4.7GB, Scout)
   - /opt/nexus-gpu/models/gemma-4-31b-it-awq/ (20GB, 백업)
   - /opt/nexus-gpu/models/gemma-4-31b-it/ (59GB, 학습용 백업)
   - /opt/nexus-gpu/models/e5-large/ (9GB, 임베딩)
   - /opt/nexus-gpu/checkpoints/qwen35-phase1/ (153MB, LoRA 어댑터)
7. **vLLM 시작 명령 (GPU 서버)**:
   ```
   /opt/nexus-gpu/.venv/bin/python3.12 -m vllm.entrypoints.openai.api_server \
     --model /opt/nexus-gpu/models/qwen3.5-27b-awq \
     --max-model-len 8192 --gpu-memory-utilization 0.90 \
     --port 8001 --host 0.0.0.0 --trust-remote-code \
     --served-model-name qwen3.5-27b \
     --enable-prefix-caching --enforce-eager \
     --enable-auto-tool-choice --tool-call-parser qwen3_xml \
     --enable-lora --max-lora-rank 16 \
     --lora-modules nexus-phase1=/opt/nexus-gpu/checkpoints/qwen35-phase1
   ```
8. **추가 가능 작업**:
   - Ch 16 세션 관리 (JSONL 트랜스크립트, 세션 재개)
   - Qwen 3.5 thinking 출력 제어 (chat_template 수정)
   - Scout 모델을 Qwen 계열 소형으로 교체
   - 24시간 Soak Test
   - 웹 UI 문서 분석 개선 (청크 순차 분석 UX)
9. **알려진 제한**:
   - Qwen 3.5 thinking 텍스트가 응답에 포함될 수 있음
   - 8,192 컨텍스트 한계 (RTX 5090 32GB)
   - Scout(Gemma 4 E4B)는 Qwen 계열이 아님 (향후 교체 고려)
10. 532개 테스트 전부 통과
6. **서버 정보**:
   - GPU: 192.168.21.112 (Worker :8001, Embedding :8002, Scout :8003), max-model-len=8192
   - Scout: Gemma 4 E4B (Q4_K_M) on llama.cpp CPU, ~16 TPS
   - DB: 192.168.10.39 (PostgreSQL + Redis, 현재 미연결)
   - 웹: https://192.168.22.223:8443 (자체 서명 SSL)
7. 453개 테스트 전부 통과

---

## v0.14.0 — 내부 구조 정비 + M7 테넌트 학습 파이프라인 (2026-04-22)

**범위**: 기능 추가는 M7(테넌트별 LoRA 학습)만, 나머지는 전부 내부 리팩토링.

### 리팩토링 1 — QueryEngine.submit_message 분해
- 210줄 God-method를 분리: `core/orchestrator/routing.py`(RoutingResolver +
  HeuristicClassifier + RoutingDecision) + `prompt_assembler.py`(PromptAssembler)
- `submit_message()`가 ~50줄로 축소, 각 단계 독립 테스트 가능

### 리팩토링 2 — QueryEngine.bind_request 공식 메서드
- 웹 핸들러가 `engine._session_id`/`_transcript` 등 비공개 속성을 직접 치환하던
  race condition 위험을 공식 API `bind_request(session_id, tenant, transcript,
  restore_messages)`로 해소

### 리팩토링 3 — 웹 lifespan 분해 + Worker 프롬프트 외부화
- 인라인 130줄 부트스트랩 → `_build_web_query_engine(components, state)` 호출로 축소
- 50줄 Worker 시스템 프롬프트 → `web/prompts/worker_system.md` 외부화
  (운영자가 코드 수정 없이 튜닝)

### 리팩토링 4 — QueryClassifier 전략 + tool_keywords YAML 외부화
- `config/nexus_config.yaml`에 `routing.tool_keywords` 리스트 전체를 노출
- `RoutingConfig.classifier_type: str = "heuristic"` + `build_classifier()`
  팩토리 + `_CLASSIFIER_REGISTRY` — 향후 LLM/임베딩 분류기 확장 지점

### 리팩토링 5 — PgVectorStore 공통 베이스
- `core/rag/pgvector_base.py` 신설: `format_vector`, `cosine_similarity`,
  `PgVectorStore` 베이스(`ensure_schema`/`build_vector_index`/`count`)
- `KnowledgeStore`, `SymbolStore`가 베이스 상속으로 전환 → 두 파일에서 ~80줄씩
  중복 제거

### 리팩토링 6 — tool_keywords 매칭 규칙 구조화
- `tool_word_patterns`(단어 경계 `\b`), `tool_regex_patterns`(자유 정규식) 신설
- 기존 substring에서 `file`이 `filename`에 오탐되던 한계 해소
- 잘못된 regex는 경고 로그 후 무시(fail-safe)

### M7 — 테넌트별 LoRA 학습 파이프라인
- **신규** `training/adapter_naming.py` — `normalize_tenant_id`,
  `compose_adapter_name`, `compose_output_dir`, `compose_data_path`
- **네이밍 규약**: default → `nexus-phaseN` (기존 호환), 테넌트 →
  `nexus-{tenant}-phaseN`, `adapter_name_prefix`로 커스텀 우선
- **수정** `training/trainer.py` — `TrainingConfig.tenant_id/phase` 필드 +
  `resolved_output_dir()`/`resolved_adapter_name()` + `to_dict()`에 M7 필드 포함
- **수정** `core/config.py` — `TenantConfig.adapter_name(phase)` +
  `adapter_name_prefix` 옵션
- **신규** `scripts/train_tenant_lora.py` — `--tenant-id`/`--phase`/`--data-path`/
  `--dry-run` CLI. 학습 metadata.json에 tenant_id·adapter_name 기록

### 테스트
- 신규 `tests/unit/test_adapter_naming.py` — 36 케이스
- 보강 `tests/unit/test_query_routing.py` — classifier 팩토리 4 + 매칭 타입 5 케이스
- **최종 734 테스트 통과** (634 → +100)

### 신규/수정 파일
```
신규:
  core/orchestrator/routing.py
  core/orchestrator/prompt_assembler.py
  core/rag/pgvector_base.py
  training/adapter_naming.py
  scripts/train_tenant_lora.py
  scripts/_ssh_kowiki_status.py       # γ 진행 점검용
  tests/unit/test_adapter_naming.py
  web/prompts/worker_system.md

수정:
  config/nexus_config.yaml             # tool_keywords YAML 노출
  core/config.py                       # TenantConfig.adapter_name + classifier_type
  core/orchestrator/query_engine.py    # submit_message 분해 + bind_request
  core/rag/knowledge_store.py          # PgVectorStore 상속
  core/rag/symbol_store.py             # PgVectorStore 상속
  tests/unit/test_query_routing.py     # 신규 매칭 타입 테스트
  training/trainer.py                  # TrainingConfig 테넌트 필드
  web/app.py                           # lifespan 분해 + 프롬프트 외부 로드
  user_mig/PROJECT_NEXUS_SPEC_v7.0_AMENDMENT.md  # Ch 15(M7) + Part 9(리팩토링)
```

### 다음 세션에서 이어갈 후보
1. kowiki C2-γ 적재 완료 대기 → `build_vector_index()` (ivfflat) 실행
2. M7 후속 — `training/bootstrap_generator.py`에 `--tenant-id` 인자 추가
3. 테넌트 관리 API — `GET /v1/tenants` 조회 엔드포인트
4. AgentTool Scout 결과 캐시 (~30s 지연 완화)

---

## v0.14.1 — M7 후속: 부트스트랩 테넌트 분리 (2026-04-22)

v0.14.0 "다음 세션에서 이어갈 후보" 2번 소화.

### 변경
- **수정** `training/bootstrap_generator.py::BootstrapGenerator.generate()`
  - `tenant_id: str | None = None` 인자 추가 (키워드 전용 호출 기대)
  - `normalize_tenant_id()`로 조기 정규화/검증 — 잘못된 값은 `ValueError`로 차단
  - 출력 경로: `default` → `{output_path}/bootstrap_data.jsonl` (기존 호환),
    테넌트 → `{output_path}/{tenant_id}/bootstrap_data.jsonl`
  - 각 샘플 `metadata.tenant_id` 스탬프 — 감사·혼합 학습 시 출처 역추적
  - stats dict에 `"tenant_id"` 키 추가, 로그 포맷에 `[tenant=...]` 접두
- **신규** `scripts/generate_bootstrap.py`
  - `--tenant-id`/`--count`/`--output-root`/`--seed`/`--dry-run` CLI
  - `scripts/train_tenant_lora.py`와 동일한 경로 규약으로 JSONL 산출

### 테스트
- `tests/unit/test_training_pipeline.py::TestBootstrapGenerator`에 M7 케이스 4건 추가:
  1. `tenant_id=None` → 기존 루트 경로 유지 (하위 호환)
  2. `tenant_id="dongguk"` → `{root}/dongguk/bootstrap_data.jsonl`
  3. 각 샘플 metadata에 `tenant_id` 스탬프 검증
  4. 허용 문자셋 위반(한글) → `ValueError` 조기 차단
- BootstrapGenerator 테스트 9/9 통과, ruff 신규 파일 clean
  (기존 E501 17건 → 16건, 본 변경은 신규 위반 없음)

### 하위 호환
- 기존 호출(`generator.generate(count=N, output_path=...)`)은 경로/동작 변경 없음
- `tenant_id=None`이든 `"default"`든 동일하게 공용 루트로 수렴

---

## v0.14.2 — 테넌트 조회 API + Scout 결과 캐시 (2026-04-22)

v0.14.0 "다음 세션 후보" 3·4번 일괄 소화. 동시에 kowiki 전체 덤프(C2-γ)
적재 현황을 점검했다.

### kowiki 적재 현황 (2026-04-22 04:02 KST)
`scripts/_ssh_kowiki_status.py`로 GPU 서버(192.168.21.112) 상태 조회:
- tmux `kowiki_ingest` 세션이 19시간 42분 경과 상태로 진행 중 (PID 206672)
- `--limit 0 --categories ""` (전체 kowiki) 모드 — 500건 제한 해제 후 재시작된 버전
- 진행: **처리=182,005 / 적재=93,300 / 청크=291,073**
- `tb_knowledge`: kowiki 291,109 + sample 3 = **291,112 chunks**
- `prepare_kowiki.py --build-index`는 적재 완료 **후** 실행해야 인덱스 재빌드
  오버헤드를 피할 수 있으므로 현 세션에서는 보류 (`TaskList` #4 follow-up)

### 작업 3 — GET /v1/tenants 엔드포인트 (Part 5 Ch 15)

**신규**:
- `web/app.py` — `TenantInfo` Pydantic 응답 모델 + `@app.get("/v1/tenants")` 핸들러
- 엔드포인트 주석 갱신 (`/v1/tenants` 추가)

**설계**:
- 응답 스키마: `{tenants: [TenantInfo], default_tenant: str, total: int}`
- `TenantInfo`: `id / name / description / model_override / allowed_knowledge_sources /
  api_key_count / adapter_name_prefix / metadata`
- **보안**: `api_keys` 원본 비노출 — 길이만 `api_key_count`로 노출
- 레지스트리 미초기화 시에도 500 대신 빈 목록으로 graceful 응답

**테스트** (`tests/integration/test_web_integration.py` +4):
- 200 OK + 기본 구조
- 레지스트리 초기화 시 default 테넌트 포함 (테스트 환경에선 skip)
- `api_keys` 키 부재 + `api_key_count` 정수 검증
- **완전한 응답 검증** — 실제 api_keys가 있는 TenantRegistry 주입 후 응답 본문에
  키 원본이 plaintext로 새어나가지 않는지 `resp.text.find(...)`로 검증

**/metrics와의 관계**: 기존 `/metrics`의 `tenants.registered`는 요약 통계용,
신규 `/v1/tenants`는 전체 필드 포함한 관리용.

### 작업 4 — AgentTool Scout 결과 캐시

**배경**: Scout 호출은 CPU 4B 모델(llama.cpp)이라 한 번에 ~30초. 같은 문서에
대해 Worker가 여러 턴에 걸쳐 탐색을 반복하거나 재시도할 때 같은 입력을 다시
실행하는 건 순수 낭비.

**수정** `core/tools/implementations/agent_tool.py`:
- 클래스 레벨 `_cache: OrderedDict[str, tuple[text, turns, stored_at]]` + LRU 정책
- 클래스 레벨 `_cache_stats: {hits, misses, stored, evicted}`
- 튜닝 상수: `_cache_enabled=True`, `_cache_ttl_seconds=300.0`, `_cache_max_entries=64`
- 키: `SHA-256(subagent_type | prompt | system_prompt | sorted(tool_names) | max_turns)`
  — 도구 순서 무관, 필드는 `\0` 구분자
- `call()` 통합: 캐시 조회 → 히트 시 `_run_subagent` 생략 + `metadata.cache_hit=True`
- **에러 결과는 저장하지 않음** (일시 장애 고착 방지)
- **ad-hoc(description) 호출은 캐시 제외** — description이 매번 미세하게 달라지면
  캐시 일관성이 떨어지므로 Scout 같은 정식 서브에이전트(subagent_type 있음)만 대상
- `get_cache_stats()`, `reset_cache()` 제공

**/metrics 노출**: `web/app.py`가 `result["agent_cache"] = AgentTool.get_cache_stats()`로
`hits/misses/stored/evicted/size` 노출.

**테스트** (`tests/unit/test_agent_tool.py` +8):
- 히트 시 `_run_subagent` 재호출 생략 + `cache_hit` 플래그
- 다른 prompt는 별도 키 (1차 miss + 저장)
- `_cache_enabled=False`면 조회 자체 건너뜀
- TTL 만료 처리 (음수 TTL로 강제 만료)
- 에러 결과는 `stored=0`
- ad-hoc 경로는 캐시 제외
- `_cache_max_entries` 초과 시 LRU 방출
- 캐시 키는 결정적 + 도구 순서 무관

**/metrics 테스트** (`tests/integration/test_web_integration.py` +1):
- `agent_cache` 섹션 존재 + 5개 필드(`hits/misses/stored/evicted/size`) 정수 검증

### 테스트 / 회귀
- 신규 13 케이스 (`test_agent_tool.py` +8, `test_web_integration.py` +5)
- 회귀 **750 passed / 1 skipped** (단위 + 통합, e2e 제외)
- ruff: 신규 코드 clean. 기존 위반(E402/E501/F401/I001)은 본 변경 범위 밖.

### 하드웨어 업그레이드 시 영향
- 캐시는 티어에 독립적 — TIER_M/L에서도 AgentTool이 그대로 쓰이면 동일하게 동작
- Scout가 없는 환경(TIER_M/L)에서도 code-reviewer 등 향후 서브에이전트에
  그대로 적용 가능 (`subagent_type` 기반이라 확장성 있음)

### 다음 세션 후보
1. kowiki 적재 완료 대기 → `prepare_kowiki.py --build-index` 실행 (TaskList #4)
2. 캐시 실측 — Scout 반복 질의 E2E에서 `agent_cache.hits` 증가 확인
3. 사양서 Ch 14에 서브에이전트 캐시 공식 명세화 (v0.14.2에서 AMENDMENT 갱신 완료)

---

## kowiki 전체 덤프 적재 경과 기록 (2026-04-22)

C2-γ 확장 적재. 500건(2026-04-21 야간)과 별개로 카테고리/limit 제한 없이 전체
kowiki 덤프를 돌리는 장기 작업이 GPU 서버(192.168.21.112) tmux `kowiki_ingest`
세션에서 진행 중.

### 실행 명령
```bash
/opt/nexus-gpu/.venv/bin/python3.12 scripts/prepare_kowiki.py \
  --dump /opt/nexus-gpu/corpora/kowiki/kowiki-latest-pages-articles.xml.bz2 \
  --categories "" --limit 0 \
  --embed-url http://192.168.21.112:8002 \
  --pg postgresql://nexus:idino%4012@192.168.10.39:5440/nexus
```

### 진행 스냅샷

| 시각 (KST) | 경과 | 처리 | 적재 | 청크 | tb_knowledge (kowiki) |
|---|---|---|---|---|---|
| 04-22 04:02 | 19h 42m | 182,005 | 93,300 | 291,073 | 291,109 |
| 04-22 05:59 | 21h 40m | 211,791 | 104,600 | 318,461 | **318,514** |
| 04-23 15:17 | **45h 58m** | **769,799** | **281,650** | **686,578** | **686,578** |

**증분 (04-22 05:59 → 04-23 15:17, 약 33h)**: 처리 +558,008 / 적재 +177,050 / 청크 +368,117
**시간당 처리율 (후반 평균)**: 처리 ~16.9K / 적재 ~5.4K / 청크 ~11.2K

에러·종료 신호 없음. embed 서버(:8002) 200 OK 꾸준. 프로세스 PID 206672,
CPU 5.7%/RSS 158MB로 안정적. ELAPSED 1d21h58m 시점 기준 여전히 run 상태.

### 확인 도구
`scripts/_ssh_kowiki_status.py` — paramiko로 GPU 서버 SSH 접속 후 tmux pane +
프로세스 + `tb_knowledge` row count를 한 번에 조회. 사용자 개입 없이 자동 실행.

### 후속 조치 (적재 완료 후)
1. `grep -Ei 'complete|finished|done|error|traceback' /tmp/_kowiki_pane.log`로 종료 확인
2. `python scripts/prepare_kowiki.py --build-index --pg "..."` — ivfflat cosine 인덱스
   재빌드 (lists=100). 기존 idx_knowledge_embed은 500건 기준이라 수십만 청크
   환경에서 효율 낮음
3. KNOWLEDGE 라우팅 E2E로 RAG 적중률 재검증 (기존 "니체/차라투스트라" 질의)

### 설계 기준 (사양서 이탈 없음)
- Part 2.5.8 지식 RAG 파이프라인 — kowiki 적재 스크립트와 ivfflat 인덱스는 이미 명세
- 적재 규모 확장은 "한계와 향후 과제" 섹션에서 예고된 작업

---

## MCP 도입 PoC — v7.2 사양서 amendment 작성 (2026-06-01)

### 배경

사내 시스템 한두 개를 LAN MCP 서버로 떼어내고 Nexus에 MCP 클라이언트 어댑터를
추가하는 PoC를 시작. 첫 단계로 기술 사양서 v7.2 개정안을 작성.

### 산출물

`user_mig/PROJECT_NEXUS_SPEC_v7.2_AMENDMENT.md` (약 580라인, enterprise-architect
위임 작성 → 핵심 인용 실측 검증 완료).

### 핵심 정합화 논점

- v6.1은 MCP를 "외부 SaaS 의존 → 에어갭 불가, Phase 2 연기"로 보류(Removed/Disabled).
- v7.2 정정: **외부 SaaS MCP는 여전히 금지, 사내 LAN MCP 서버는 허용**.
  근거 — v6.1 라인 402가 "Machine A↔B (LAN, HTTP/SSE)"를 ALLOWED로 명시(자기모순 해소).
- 권한 인프라는 **이미 코드에 존재**: `ToolCategory.MCP`(types.py:117),
  `mcp__` 식별(pipeline.py:191), MODE_BEHAVIOR_MAP MCP 열, Layer5 PLAN 보정(:368).
  → v7.2 신규 추가는 **MCP 클라이언트 어댑터 + 연결 관리자**(`core/tools/mcp/`)와
  `McpConfig` 뿐. 4-Tier·권한·Hook 불변.

### PoC 대상 4종 (전부 read-only, LAN HTTP/SSE)

| 서버 | 도구 | LAN 위치 |
|---|---|---|
| DB 조회 | `mcp__db__query` | PG 192.168.10.39 (SELECT 전용) |
| 진단/모니터링 | `mcp__diag__reachability`, `mcp__diag__rag_latency` | 웹/GPU 22.28/DB 10.39 |
| DocUtil 문서 | `mcp__docutil__search`, `mcp__docutil__get` | DocUtil 192.168.10.39 |
| kowiki RAG | `mcp__kowiki__search` | tb_knowledge + 임베딩 22.28:8002 |

### 미확정 (사용자 확인 필요)

1. GlobalState `mcp_servers`/`mcp_connected` 필드는 **실제 코드엔 미존재**
   (초기 grep은 v6.1 문서 내 코드였음) — 정식 도입 시 신규 추가 대상으로 표기.
2. MCP 서버 포트 8810~8813은 placeholder — 운영 배치 확정 필요.
3. kowiki MCP(명시 호출) vs 기존 자동 RAG 주입(v7.0 Part 2.5.8) 통합 여부 — PoC 후 결정.

### 구현 완료 (2026-06-01, 제품화 기준 전환)

**코드 구현**: `core/tools/mcp/`(client/adapter/connection_manager) + `core/config.py`
McpConfig/McpServerConfig + `core/security/network_guard.py`(is_lan_hostname, config에서
이전) + `core/bootstrap.py` Phase 2 ⑨-d 배선 + `config/nexus_config.yaml` mcp 섹션.

**부트스트랩 배선 버그 수정**: MCP 도구를 메인 registry(:132)가 아니라 **cli_registry**에
등록 + cli_tools 재취득해야 ModelDispatcher/QueryEngine에 전달됨(QA 발견).

**제품 보안 강화 (PoC→제품)**: read-only MCP만 자동 등록(쓰기는 `allow_write=True`
명시 필요), `check_permissions` 항상 ALLOW로 일원화(5계층 위임 → PLAN=DENY/BYPASS=ALLOW
정상화), `validate_input` 스키마 검증, 예외 범위 축소(버그 은폐 방지).

**테스트**: MCP 단위/통합 **81 passed**, 전체 회귀 **916 passed/1 skipped**(1 failed는
GPU 의존 e2e로 무관).

**사양서**: v7.2를 PoC→**제품 사양**으로 개정. v7.3(문서 양식 인식 임베딩 설계) 신규
작성 — HWP/PDF/PPT 레이아웃 인식 + 구조 보존 청킹 + 임베딩 적재, 청정(MIT/Apache)
라이선스 기본 + 상용 어댑터 슬롯, 티어별(5090/A100/H200) 스택.

### 결정 사항 (2026-06-01)

- 쓰기 MCP 정책: read-only만 허용(PLAN 안전성)
- 라이선스: 추상 인터페이스 + 청정 기본(AGPL/GPL 배제)
- HWP: **구포맷 .hwp 비중 높음** → LibreOffice 필수 번들, .hwp 처리 우선순위 상향

### 다음 단계 (미착수)

- v7.3 구현: PPTX 파서부터 단계적(`core/ingest/`). GPUTier.A100 추가 검토.
- 실 LAN MCP 서버 4종(db/diag/docutil/kowiki) + 문서 인제스트 서버 구현 → e2e.
- GlobalState mcp_servers/mcp_connected 가시성 필드(운영 /metrics 노출용).

---

## MCP 서버 구현 + v7.3 인제스트 파서 + 클라이언트↔서버 e2e (2026-06-02)

브랜치: `feature/mcp-integration`. v7.2 다음 단계(미착수)였던 서버 측·e2e·가시성과
v7.3 인제스트 파서(PDF/HWPX)를 구현 완료.

### 1. MCP 서버 측 4종 구현 (`mcp_servers/` 신규)

v7.2까지는 클라이언트(`core/tools/mcp/*`)만 있고 서버는 placeholder였다. 사내
시스템을 LAN MCP 서버로 노출하는 서버 측을 신규 구현.

| 파일 | 노출 도구 | 성격 | 기본 포트 |
|---|---|---|---|
| `framework.py` | (공통) `McpServerTool` ABC + `create_mcp_app` — JSON-RPC 2.0 디스패치/Bearer 인증/`/health`/lifespan | — | — |
| `run.py` | `python -m mcp_servers.run <name>` uvicorn entrypoint | — | — |
| `db_server.py` | `query` | read-only SELECT (검증 + READ ONLY 트랜잭션 이중 방어) | 8810 |
| `diag_server.py` | `reachability`, `rag_latency` | read-only (웹/GPU/DB 점검, EXPLAIN ANALYZE) | 8811 |
| `kowiki_server.py` | `search` | read-only (embed + pgvector 검색) | 8813 |
| `docingest_server.py` | `parse`, `ingest`, `search` | parse/search read-only · **ingest 쓰기** | 8814 |

- 의존성 방향 P2 준수: `mcp_servers → core` 만 허용(core는 mcp_servers 미import).
- 에어갭: `0.0.0.0` LAN 바인드, 점검·접속 대상 전부 LAN. 외부 호출/런타임 install 없음.
- 정정: v7.2 초판의 `docutil`(8812)은 미구현 — 실제 문서 MCP는 **docingest**(8814).

### 2. 클라이언트↔서버 e2e 검증 (신규)

`tests/integration/test_mcp_client_server_e2e.py` — 기존엔 AsyncMock 흉내뿐이었던
구간을 실제 연결로 관통:
- McpClient ↔ `create_mcp_app` 서버 (ASGITransport 인프로세스 + 임의 포트 uvicorn 실 TCP)
- McpClient → McpToolAdapter 전구간(BaseTool/tool_use_error 래핑 포함)
- 모든 연결 127.0.0.1 루프백(에어갭 준수).

### 3. v7.3 인제스트 파서 (PDF/HWPX) + docingest

- `core/ingest/parsers/pdf_plumber.py` — pdfplumber(MIT) PDF 경량(표·좌표·헤딩 휴리스틱, fail-soft).
- `core/ingest/parsers/hwpx.py` — python-hwpx(OWPML) 1차 + zipfile/xml.etree 폴백.
- `parsers/__init__.py` export: HwpxParser/PdfPlumberParser/PptxParser 3종.
- docingest MCP 서버가 ParserRegistry(PPTX/PDF/HWPX) + DocumentIngestPipeline 조립.
- Docling/OCR(Tesseract·PaddleOCR) 고품질 파서는 **어댑터 슬롯만 예약(미구현·후속)**.
- 라이브러리: pdfplumber/python-hwpx/pytesseract + docling/paddleocr **개발 환경 설치 완료**.
  에어갭은 배포물(wheel 번들)에만 적용 — 개발 중 설치는 정상.

### 4. GPUTier.A100 추가

- `core/model/gpu_detector.py:32` 에 `A100 = "a100"` 추가(5종).
- 판정(`:161~169`): `vram_gb > 60` 구간에서 GPU 이름에 "a100" 포함 시 A100, 아니면 H100.
- 프로파일(`:225~262`): BF16 NONE, max_model_len 8192, max_num_seqs 4, lora rank 64,
  **FP8 미지원(Ampere)**. VRAM 80GB → 고품질 인제스트 스택 여유 충분(TIER_M급).

### 5. GlobalState MCP 가시성 + top_k 견고성

- `core/state.py:131` — `mcp_servers: dict` / `mcp_connected: set` 추가.
- `core/bootstrap.py:391~395` — 등록 결과로 채움(빈 리스트는 connected 자연 제외).
- `web/app.py:1206~` — `/metrics` 에 `mcp.connected_count`/`connected`/서버별 도구 노출.
- top_k 견고성: kowiki/docingest/diag 의 `top_k=0/음수/bool` 거부, 미지정 시에만 기본값.

### 테스트 / 회귀

- 전체 회귀 **1091 passed, 1 skipped** (실측 2026-06-02).
- 유일한 1 failed = `tests/e2e/test_gpu_e2e.py::test_simple_chat_completion`
  (실 GPU 서버 의존 e2e — MCP/인제스트와 무관, 코드 회귀 아님).
- 신규/갱신 테스트: `test_mcp_server_framework.py`, `test_mcp_db_server.py`,
  `test_mcp_kowiki_docingest_server.py`, `test_mcp_client_server_e2e.py`,
  `test_mcp_visibility.py`, `test_pdf_plumber_parser.py`, `test_hwpx_parser.py`.

### 사양서 갱신 (이 작업에서 동시 반영)

- `PROJECT_NEXUS_SPEC_v7.2_AMENDMENT.md`: 서버 측 placeholder→구현 완료(Part 6.0),
  docutil→docingest 정정(6.4), Part 8 구현 현황/테스트, GlobalState 가시성 구현됨,
  부록 A/B 코드 매트릭스에 mcp_servers/* 추가.
- `PROJECT_NEXUS_SPEC_v7.3_DOC_INGEST.md`: PDF/HWPX·docingest·GPUTier.A100 를
  "신규 제안"→"구현 완료"로, 로드맵 단계 상태 컬럼 추가, Part 10 해소 항목 갱신.

### web app full e2e — 실 vLLM 질의 (2026-06-02)

mock 없이 실 인프라(실 PG/임베딩/vLLM/Redis)로만 검증:
- `GET /v1/tools` → `mcp__db__query`, `mcp__kowiki__search` 노출.
- `GET /metrics` → `mcp.connected: ['db','kowiki']`.
- `POST /v1/chat` "tb_knowledge 행 수" → Worker(vLLM)가 **mcp__db__query 호출 →
  1,067,978 정답**. kowiki 도구 호출·검색도 정상(니체 3건).
- **버그 수정**: 웹 Worker는 cli_registry가 아닌 자체 풀을 써서 MCP 도구가
  누락됐었다 → bootstrap `components["mcp_tools"]` + web 머지로 해소.

**kowiki 8K overflow — 결정(2026-06-02, 사용자 확정)**: KNOWLEDGE 모드 자동 RAG
주입 + kowiki MCP 도구 결과가 같은 검색을 이중으로 넣어 RTX 5090 8K(7778/8192)를
초과 → 최종 답변 생성 불가. **MCP 결함이 아니라 자동 RAG와 도구의 중복 + 5090 8K
제약.** 코드 변경 없이 **5090 한정 제약으로 인정**, 제품 검증 기준 **A100(80GB)/
H200(141GB)의 32~128K 컨텍스트에서 자연 해소**. db/kowiki 둘 다 Worker 도구로 유지.

### 남은 단계 (미착수/후속)

- 실 LAN 호스트 배포 e2e (현재는 인프로세스 + 루프백 TCP까지).
- PDF 고품질(Docling) + 스캔 OCR(Tesseract/PaddleOCR) 고품질 파서(어댑터 슬롯).
- `.hwp`(LibreOffice 변환), `scripts/prepare_documents.py` 배치 스크립트.
- yaml `mcp.servers` 를 실제 구현 서버(db/diag/kowiki/docingest)로 정렬(docutil 제거).

---

## 2026-07-03 — B200 배포 준비: 보안 하드닝 (branch feature/b200-bakeoff)

B200 정부 컨테이너에 "완벽한 버전" Nexus를 올리기 전, 지난 감사 Critical 8건 중
보안 최우선 항목을 **배포 선결 조건으로 승격**해 착수. (승인 플랜: B200 bake-off)
현재 디렉토리(D:\workspace\nexus)는 그대로 두고 worktree D:\workspace\nexus-b200 에서 진행.

### A. Critical #4 — 자격증명 평문 제거 + 웹 API 키 인증 (커밋 6cd1eb1)

- **diag_server.py**: GPU 평문 비번 하드코딩 제거 → `NEXUS_DIAG_GPU_PASS` 환경변수,
  미설정 시 GPU SSH 점검만 fail-soft 생략(웹/DB 점검은 정상). 죽은 PG 상수 4개
  (평문 `idino@12` 포함) 삭제.
- **web 인증**: `ApiKeyAuthMiddleware` 신설 — `WebAuthConfig.enabled`(기본 False=무회귀)
  활성 시 `Authorization: Bearer`를 `TenantRegistry.resolve_by_api_key`로 검증, 실패는
  모두 401(fail-closed). 면제 경로·OPTIONS만 통과. CORS 뒤 등록. `enabled is True`
  명시 체크로 비-bool truthy 오활성 차단.
- **config 시크릿**: `nexus_config.yaml`의 PG/Redis 평문 비번 제거 →
  `load_and_validate_config`에서 `NEXUS_PG_PASSWORD`/`NEXUS_REDIS_PASSWORD` 주입.
  yaml을 `NexusConfig(**file_data)` init 인자로 넘기는 구조상 "init>env" 우선순위
  quirk를 피하려 file_data에 직접 주입. 원격 DB 빈 비번 시 경고 로그.
- **테스트 27건 추가**: `test_web_auth_middleware.py`(18), `test_diag_server_credentials.py`(4),
  `test_config_secret_injection.py`(5). 관련 회귀 **269 passed** 무회귀.
- **배포 시 필수**: `NEXUS_WEB_AUTH__ENABLED=true` + 테넌트 `api_keys`,
  `NEXUS_PG_PASSWORD`/`NEXUS_REDIS_PASSWORD`, (GPU 진단 쓰면) `NEXUS_DIAG_GPU_PASS`.

### 후속(별도 트랙, 미착수)

- Critical #1~3(권한 파이프라인 배선·ASK fail-open·PathGuard/CommandFilter),
  #5~8(웹 QueryEngine 싱글톤 동시성·메모리 감쇠 복리·consolidate 세션 파괴·컨텍스트
  복구 도달 불가) — B/C 이후 순차.
- `web/app.py`의 `RequestLoggingMiddleware`가 `add_middleware` 미등록(기존 구조).
- git 이력에는 과거 평문 비번이 잔존 → 실 배포 전 자격증명 rotate 권장.

### B. B200 티어 활성화 (진행 중) — 플랜 Phase 1

- **B1 완료**: `ContextBudgetConfig` 외부화(하드코딩 컨텍스트 예산 8종 → yaml).
  - `core/config.py`에 `ContextBudgetConfig` 신설(기본값=현행값 정확 일치), `NexusConfig.context_budgets` 배선.
  - 배선: bootstrap→QueryEngine→(PromptAssembler / ModelDispatcher→query_loop) + document_tool(context.options) + web/app.py(운영 진입점, getattr 방어). 모두 **None이면 현행 상수 폴백=무회귀**.
  - 무회귀: 기본값 8종 = HEAD 하드코딩값 정확 일치(1000/1500/1000/2048/3/2/2500/[4096,8192,16384]),
    전체 unit+integration **1327 passed** 무회귀. 회귀테스트 `test_context_budget_config.py` 추가.
- **B2 완료**: web 도구풀 + Worker 프롬프트 티어 연동.
  - `_create_web_tool_registry(tier)` 파라미터화 — TIER_S=현행 5개(Agent/Bash/Edit/SymbolSearch/Write) 무회귀,
    TIER_M/L=11개(+Read/Glob/Grep/LS/DocumentProcess/GitDiff, GitCommit 제외, 이름순 정렬).
  - `web/prompts/worker_system_full.md` 신설(TIER_M/L용 — Scout 위임 제거 + 직접 탐색 지침, Grounding/anti-sycophancy 유지).
    `_load_worker_system_prompt(tier)` 분기(TIER_S→기존, M/L→full, 단계적 폴백).
  - 실측 주석 정정: `_create_tool_registry` 24→**23개**, `_create_cli_tool_registry` 6→**7개**,
    `hardware_tier.max_worker_tools` 11/24→**7/23**("정보용 필드, 런타임 미강제"). test_hardware_tier 단언 동기화(약화 아님).
  - 무회귀: 웹 도구풀 TIER_S 스냅샷 동일, 전체 unit+integration **1330 passed**.
  - **후속(미착수)**: CLI `_build_default_system_prompt` 티어 분기 스킵 — CLI가 TIER_M/L에서 `_create_tool_registry`(23개, Read/Glob 포함·Agent 미포함)를 쓰는데 인라인 프롬프트는 "Read/Glob 없음, scout 써라"라 **프롬프트↔도구 불일치(기존 버그)**. CLI full 프롬프트 저술 필요(별도 지시 대기).
- **B3 완료**: `config/examples/nexus_config.b200.yaml` 신설(완전 standalone 배포 템플릿).
  - tier=large, gpu 127.0.0.1:8001/8002, primary=ax-4.0, max_context=24576, web_auth.enabled=true, context_budgets 상향(2000/4000/2500/8000/6/3/8000/[4096,8192]).
  - **정합성 불변식**: max_context(24576)+max출력(8192)=32768 ≤ vLLM axmodel max_model_len(32768). 플랜의 128K 스케일 값(114688 등)은 단일 B200 KV 제약(A.X=32768)과 안 맞아 재산정.
  - **후속(Phase 0 시 처리)**: redis/postgresql host가 아직 온프렘 192.168.10.39 — B200 컨테이너 co-located(127.0.0.1)로 조정 필요(포트는 실제 DB 셋업 확정 후).
- **B4 예정**: 동시 다모델 상주(ServingConfig + model_manager) — interface-only, 순차 bake-off엔 불필요라 후순위.

### C. 권한 강제 배선 (Critical #1~3) — 정부 플랫폼 배포 블로커

감사: PermissionPipeline(5-Layer)·PathGuard·CommandFilter·AuditLogger가 모두 구현돼 있으나 **배선 harness가 통째로 빠짐**. executor는 도구 check_permissions의 **DENY만 차단**(ASK fail-open), PathGuard/CommandFilter 고아. 안전 롤아웃 순서: **config OFF 기본 → shadow → DENY 강제 → (ASK는 P3 후속)**.

- **C1 완료(20f5e7c)**: P0+P1 하네스+shadow. `permission_enforcement{enabled=false,mode=shadow}`+`audit` config, `mode_mapping`(state PermissionModeValue→permission PermissionMode: trust→BYPASS/headless·deny_all→DONT_ASK), bootstrap이 PermissionContext/Pipeline/AuditLogger 생성해 `ToolUseContext.options` 주입(4-Tier 시그니처 불변), executor Step 6-8a **shadow 관측**(판정을 AuditLogger JSONL 기록만, 차단 안 함). 무회귀 1330.
- **C2 완료**: P2 **DENY 강제**. executor enforce 분기(deny면 실차단, ASK/allow는 통과=P3 범위), pipeline Layer 2 **공통 pre-check**로 PathGuard(순회/.env/.ssh/*.pem/*.key)+CommandFilter(pip/npm/apt install 등) 연결(**미주입 시 skip=무회귀**), 경로 cwd-절대정규화, 웹 세션 샌드박스 cwd(enabled 시만), 에어갭 설치차단 게이팅(`command_filter.block_package_install`: dev=false/b200=true). b200 config에 enforce 활성.
  - 시연(직접): .env·순회·pip install·rm -rf → **DENY**, 정상쓰기·ls → 통과. 무회귀 1330.
- **후속(권한)**: **P3(ASK 강제)** — CLI 확인 프롬프트/웹 정책, shadow 로그 확인 후. config deny rule 로딩(현재 rules=None), hook_manager 배선(Layer 4 미실행).

### C-메모리. Critical #6·#7 (장기 메모리 버그)

- **#6·#7 완료**: 감쇠 복리 + consolidate 세션 파괴 수정(database-architect).
  - **#6** `decay.py run_decay_cycle`: 감쇠값을 `importance`에 되쓰던 update 제거 → **삭제 전용(멱등)**. importance=불변 base 유지(랭킹 ORDER BY importance가 오염 복리값 아닌 원본으로 정렬), 유효중요도는 읽을 때 `calculate_decay`로 계산. 검증: 사이클 N회 반복 = 1회(복리 없음).
  - **#7** `manager.py`: turn/tool_result 자동 key를 `turn:{session_id}` → `turn:{session_id}:{sha256(content)[:16]}`. 서로 다른 턴 보존, 진짜 중복만 dedup(consolidate·명시 key 기능 유지). key는 조회 미사용이라 안전.
  - 테스트 4건 추가, 전체 **1395 passed**. **실 PG 통합검증은 후속**(NEXUS_PG_PASSWORD 필요, SQL/DDL 의미 불변이라 저위험). 운영 레거시 `turn:{session_id}` 엔트리 1회 정리 고려.
### C-동시성. Critical #5 (웹 QueryEngine 싱글톤)

- **#5 완료**: 웹이 QueryEngine을 앱 전역 싱글톤 1개로 공유해 동시 요청이 `_messages`/`_session_id`/tenant를 뒤섞던 결함을 **세션별 엔진 격리**로 수정(backend-specialist).
  - 무거운 부품(도구 인스턴스·시스템 프롬프트·provider·retriever)은 `_build_web_engine_parts`로 1회 조립·공유, 요청마다 가벼운 `ToolUseContext`+`ModelDispatcher`+`QueryEngine`만 세션별 생성(`_assemble_session_engine`). **핵심**: dispatcher가 context를 붙들고 도구까지 전파하므로 dispatcher도 세션별 생성해야 tenant 누출 없음. base_options는 얕은복제 후 tenant만 얹어 원본 미오염.
  - **세션 락**(`_get_session_lock`, loop-aware + 바운드 LRU, locked 스킵): 같은 세션 동시 요청만 직렬화, 다른 세션은 병렬(멀티테넌트 처리량 보존). 스트리밍은 `async with`로 스트림 전체 수명 락 유지(연결 종료 시에도 해제).
  - 부트스트랩 실패/테스트(parts 없음) → 기존 싱글톤 폴백(무회귀). `_build_web_query_engine` 3-튜플 계약 유지.
  - 테스트: `test_web_concurrency_isolation.py` 2건(격리 + **control이 원버그 재현**). 전체 **1397 passed**, ruff 신규 위반 0(기존 7 baseline). **실서버 동시부하 최종검증은 후속**(러닝 vLLM/Redis/PG 필요).
  - 부수: 세션별 샌드박스 cwd(`_session_sandbox_cwd`) — permission_enforcement OFF(기본)에선 `state.cwd` 그대로(무회귀).
### C-복구. Critical #8 (컨텍스트 오버플로 복구 도달불가)

- **#8 완료**: Tier 3 `inference.stream`이 컨텍스트 초과/HTTP 오류를 raise 안 하고 ERROR 이벤트로 yield → Tier 2 `query_loop`의 복구(emergency_compact 등)가 `except`에만 있어 영구 미도달, CONTEXT_OVERFLOW는 그냥 abort하던 결함 수정(backend-specialist).
  - 복구 판정을 헬퍼 `_try_recover_from_model_error`(str→압축+continue_reason+cancel_all, `_RecoveryOutcome` 반환)로 추출, `except` 3복구를 **비트 단위 동일**하게 옮김(무회귀). OOM 경고는 events로 반환해 호출부가 yield → 4-Tier yield 흐름 유지.
  - `_error_event_to_recovery_text`: CONTEXT_OVERFLOW→"context too long"(emergency_compact 매핑), HTTP_4xx→vLLM 본문 메시지, 그 외→None(현행 유지). ERROR 이벤트 경로에서 헬퍼 호출→복구 시 재시도(continue), 아니면 기존 abort 유지.
  - 테스트 7건(신규 test_query_loop.py): ERROR 이벤트 복구+재시도 / HTTP_400 prompt-too-long / 예산소진 abort / **raise 경로 무회귀** / OOM 경고 / CONNECT_ERROR 미복구. 전체 **1404 passed**, ruff clean.

## 감사 Critical 8건 전부 완료 (feature/b200-bakeoff)

- #1~3 권한 강제 · #4 자격증명/웹 인증 · #5 웹 동시성 · #6~7 메모리 · #8 컨텍스트 복구. 9커밋, 전부 무회귀(최종 1404 passed) + 테스트 동반. **후속(비차단)**: P3(ASK 강제, 배포 후 shadow 로그), 실 PG/실서버 동시부하 최종검증, b200 config DB host→127.0.0.1, CLI 프롬프트 티어 불일치, hook_manager/deny rule 로딩.

## B200 컨테이너 Bring-up + 전체 스택 e2e 검증 (2026-07-05)

정부 렌탈 NHN B200(180GB) 단일 컨테이너에 Nexus 전체 스택을 co-locate 기동하고, 온프렘 데이터 이관 후 실질의로 끝까지 검증 완료.

### 데이터 이관 (파일 기반, 네트워크 격리 유지)
- 컨테이너가 온프렘(.39)에 직접 못 닿아, **개발 PC pg_dump → 파일 → SFTP → 컨테이너 pg_restore**로 이관(라이브 터널 미사용 = 격리 경계 준수).
- 온프렘 nexus DB(PG17.10, 18GB) → 커스텀 덤프 6.1GB → 전송(4.5MB/s, 24분) → 복원.
- 복원 검증: **tb_knowledge 1,067,978행 / tb_memories 284,685 / tb_symbols 3,065**, 벡터 인덱스 전부 재구축(idx_knowledge_embed ivfflat 8.35GB, idx_memories_embedding hnsw 432MB 등). pg_restore exit=1은 무해 무시 5건(pg_buffercache/pg_prewarm 권한, vector 확장 comment 소유권)뿐.

### 인프라 구성 (모두 detached, localhost)
- **PG17.10 + pgvector 0.8.4**: /NHNHOME/nexus/pgdata, idino_user로 구동(초기 postgres 소유 권한 이슈 해결), shared_buffers 16GB 등 튜닝, 127.0.0.1:5440.
- **Redis 7**: 127.0.0.1:6340 인증. 데이터 디렉토리를 `redis`→`redis_data`로 이동(경로가 redis-py를 네임스페이스 shadowing하던 잠복 버그 수정).
- **config**: b200 템플릿 → config/nexus_config.yaml (host .39→127.0.0.1, tier large). 비번은 .env(NEXUS_PG_PASSWORD/NEXUS_REDIS_PASSWORD)로 주입.
- **의존성**: venv에 asyncpg/pgvector/redis-py 등 델타 설치(shadow 오탐으로 redis-py 미설치였던 것 발견).

### 모델·서빙
- **vLLM 0.24.0 + A.X-4.0(skt/A.X-4.0, 72B BF16)** on B200: HF 다운로드 134GB, 127.0.0.1:8001, max_model_len 32768, tool_call_parser=hermes, prefix-caching. 로드 ~150초, GPU 169GB. 아키텍처 Qwen2ForCausalLM 확인.
  - 추론 검증: 한국어 QA 0.55s, hermes 도구호출 정상(get_weather).
- **임베딩 e5-large(intfloat/multilingual-e5-large)**: 커스텀 `/v1/embed` pass-through 서버(sentence-transformers, GPU ~3GB), 127.0.0.1:8002, 1024차원. Nexus 계약({"texts"}→{"embeddings","dimension"}) 준수, 접두사는 호출자(passage:/원문)가 처리.
- **Nexus 웹**: uvicorn 8443, tier=large, scout_available=False, worker_tools=23, web_auth on(테스트키 nexus-b200-test-key-001), SymbolStore pg=connected.

### e2e 검증 (전체 파이프라인)
- 인증 없음 → 401 정상.
- 인증 /v1/chat "니체 대표 저서" → `class=KNOWLEDGE → RAG 주입 ~3,857토큰(knowledge_rag=20ms) → A.X-4.0` → 근거 기반 정답(차라투스트라/선악의 저편). 지연 3.23s, input 4,973토큰(RAG 주입 확인). **이관 kowiki를 임베딩+pgvector로 검색해 A.X-4.0이 grounded 응답** = 완전 통합 확인.

### 다음
- HyperCLOVA SEED-32B 다운로드 → 순차 로드(bake-off 2번째 모델).
- 3-프로젝트 태스크 배터리(docutil RAG-QA / dynamic_prompt SQL+HTML / 삼진어묵 추출) 채점.

## 분리구조(서비스 PC ↔ B200 백엔드) 전환 + 검증 (2026-07-05)

사용자 결정: bake-off는 A.X-4.0 단독 확정(HyperCLOVA 국가대표 탈락). 이후 요구 — 최종적으로 외부에서 API로 agent 호출·사용, **서비스는 내 PC(추후 다른 서버 가능, 셋업 쉽게), 분리구조로**.

### 아키텍처
- **서비스 PC(Machine A)**: 오케스트레이터(쿼리루프/도구/권한)+RAG 검색로직+웹+CLI. GPU 불필요.
- **B200(GPU 백엔드)**: vLLM(A.X-4.0)·임베딩(e5-large)·PostgreSQL(이관 데이터)·Redis 상주.
- **연결**: SSH 로컬 포워딩(-L)으로 B200 4포트(8001/8002/5440/6340)를 PC localhost로 당김 → 공개 노출 없이 안전. config는 전부 127.0.0.1.

### 구축물 (deploy_pc/)
- requirements-pc.txt(오케스트레이터 전용 의존성, torch/vllm/sentence-transformers 제외), setup.ps1/.sh, tunnel.ps1/.sh, start_web.ps1/.sh, start_cli.ps1/.sh, README.md.
- 루트: .env(NEXUS_PG/REDIS_PASSWORD), config/nexus_config.yaml(b200 템플릿 host→127.0.0.1, tier large), config/tenants.yaml(테스트키 nexus-b200-test-key-001).
- PC venv=.venv_pc(Python 3.11.9). python-multipart 추가 필요했음(업로드 폼).

### 검증 (e2e)
- PC(Windows) 웹 부트스트랩 성공: vLLM healthy / Redis / PostgreSQL(KnowledgeStore 1,067,978행) / 임베딩 전부 **터널 경유** 연결, tier=large, 23도구.
- 인증 /v1/chat 니체 질의 → RAG 주입(input 4,973토큰) → A.X-4.0 grounded 응답, 3.9초(터널 오버헤드 ~0.7초). **PC 오케스트레이터 → 터널 → B200 백엔드 전체 왕복 확인.**

### 이전(relocation)
- 새 서버: 저장소 복사 → setup 실행 → .env·tunnel 키/주소만 수정 → tunnel→start_web. Windows·Linux 스크립트 병행.

### 남은 것
- 웹 UI는 fetch에 인증 헤더 없음 → 브라우저 채팅 시 401. "UI에 키 주입" 패치 필요(또는 API로 테스트).
- 외부 agent API: tenants.yaml 테넌트별 api_keys로 이미 가능(격리 세션). 커스텀 agent 정의 확장은 후속.
- 외부 시연 노출(Bastion 포트매핑/도구 제한/데이터 범위)은 별도 결정 대기.

## 문서 생성 기능(DocumentExport) — docx/pptx/hwpx/md/txt + 웹 다운로드 (2026-07-05)

배경: B200 전환 후 사용자가 업로드 문서 요약을 "docx로 다운로드"하려 했으나 웹이 거절. 원인은 모델 능력이 아니라 배선 부재(생성 도구·다운로드 라우트·UI 없음). 일반 LLM 서비스처럼 문서 생성+다운로드를 Nexus에 추가.

### 사전 검증(에어갭 순수 파이썬)
- docx=python-docx, pptx=python-pptx(MIT), hwpx=python-hwpx 2.23.0(Apache-2.0, hwpx.builder 고수준 API), md/txt=평문. 5종 실제 파일 생성 스모크 통과(hwpx 유효 OWPML ZIP 확인). 한글 폰트 슬롯 커스터마이즈는 후속.

### 구현
- 신규 `core/tools/implementations/document_export_tool.py`(DocumentExportTool, BaseTool) + `document_export_renderers.py`(공통 마크다운 파싱→5포맷). 저장은 샌드박스 exports 디렉토리 고정 + uuid 파일명(경로순회 불가) → check_permissions ALLOW.
- `core/config.py` DocumentExportConfig(exports_dir/formats), `core/bootstrap.py` 웹 TIER_M/L 도구풀 등록(웹 worker 11→12).
- `web/app.py` `GET /v1/download/{filename}`(FileResponse, 경로순회 차단) + base_options에 exports_dir 주입.
- **핵심 이슈**: 모델이 결과 URL을 자유 텍스트로 재입력하며 uuid를 오탈(b551ffee→b5551ffee) → 404. 또한 TOOL_RESULT StreamEvent는 orchestrator가 발신하지 않음. → 해결: 턴 종료 후 `engine._messages`의 tool_result 메시지에서 정확한 URL을 서버가 추출해 ChatResponse.downloads(비스트림) + `download` SSE 프레임(스트림)으로 구조화 전달. UI는 그 값으로 다운로드 버튼 생성(dl-link→인증 fetch→blob 저장), 모델 텍스트의 /v1/download 링크는 비클릭 처리.
- `web/static/index.html` formatContent 링크 렌더 + download 이벤트 수집 + renderDownloadButtons.
- 의존성: deploy_pc/requirements-pc.txt에 python-pptx/python-hwpx 추가.

### 검증(실물 서버)
- 단위 테스트 11 passed(포맷 5종 생성+파일유효성, 파일명 정화/경로순회, 파싱). ruff 클린(신규 파일).
- e2e(PC웹 8600→B200): "hwpx로 만들어줘" → DocumentExport 호출 → /v1/chat downloads 필드 정확한 URL → 실제 hwpx 200/유효. /v1/chat/stream download 프레임 발신 확인. 경로순회 ../ → 404.

### 남은 것
- CLI 노출은 보류(다운로드 URL이 웹 전용). PDF/한글폰트/표·이미지 고급서식은 후속. config yaml override는 모델 기본값+env로 충분.

### [후속/백로그] TOOL_RESULT 이벤트 노출 리팩터 (2026-07-05 등록)

**공백**: orchestrator/query_loop이 도구 실행 결과를 `StreamEvent(type=TOOL_RESULT, tool_result=...)`로 소비자에게 발신하지 않는다. 그래서 (1) 웹 응답의 `tool_calls[].result`가 항상 None, (2) DocumentExport 다운로드 URL을 이벤트로 못 받아 `web/app.py`에서 `engine._messages`를 사후 스캔 + 결과 본문 정규식(`_extract_download`)으로 뽑는 우회를 넣음.

**정식 수정**:
1. 도구 결과가 tool_result Message로 만들어지는 지점(core/orchestrator query_loop/executor/stream_handler)에서 `StreamEvent(type=StreamEventType.TOOL_RESULT, tool_result=ToolResultBlock(...))`를 yield. content + **ToolResult.metadata(예: download_url) 함께 전달**.
2. `web/app.py` 비스트림: TOOL_RESULT 이벤트에서 `info.result`·`downloads` 채우고 `engine._messages[dl_start_idx:]` 스캔 제거.
3. `web/app.py` 스트림: consume 루프 안에서 이벤트로 바로 `download` 프레임 emit, 사후 스캔 제거.
4. `_extract_download` 정규식 대신 metadata.download_url 사용(정규식 제거 또는 폴백만 유지).

**대상 파일**: core/message.py(StreamEvent.tool_result·ToolResultBlock 이미 존재), core/orchestrator/*, web/app.py. **정리 대상**: web/app.py의 `dl_start_idx` 스캔 2곳 + `_extract_download`(TODO(nexus) 마커 달아둠).

**검증**: query_loop이 TOOL_RESULT StreamEvent를 yield하는 단위 테스트 + 문서 생성 e2e(downloads 여전히 동작) + 기존 웹 테스트 무회귀. **리스크**: 스트림에 이벤트 추가 → UI가 미지 type 무시하는지 확인(현재 무시함), 중복/이중계수 없게.

### 문서 생성 붕괴(degeneration) 수정 — 프롬프트 지시 추가 (2026-07-05)

증상: "요약한 내용을 보고서로 작성…DOCX로 작성해줘" → 모델이 DocumentExport를 안 부르고 보고서 본문을 인라인 프로즈로 강제 생성하다 반복 루프/기호·키릴 gibberish로 붕괴(사용자 중단).

진단(로그): 라우팅은 무관(도구는 KNOWLEDGE/TOOL 모두 사용 가능 — 12:39 KNOWLEDGE에서도 DocumentExport 호출됨). 샘플링 반복억제(KNOWLEDGE rep_pen=1.15+freq=0.3)는 config→RoutingDecision(routing.py:343)→vLLM까지 정상 전달됨(dead config 아님). 근본 원인은 **모델 선택**: "만들어줘"→도구호출(자연종료), "작성해줘"→인라인 초장문 생성→EOS 못 내고 드리프트. 게다가 worker_system_full.md에 DocumentExport가 아예 없었음(도구 추가 전 작성).

수정: `web/prompts/worker_system_full.md`에 DocumentExport를 도구 목록 + "문서/보고서 생성 요청 시 반드시 DocumentExport 호출, 본문 인라인 금지, 성공 후 1~2문장 확인만" 지시 추가. (라우팅 키워드·샘플링은 안 건드림 — 도구호출이 자연 종료 구조라 붕괴 회피.)

검증(실물 8600): 동일 요청 재현 → DocumentExport 호출, downloads 정확, 붕괴 0건. /v1/chat(37766바이트 유효 docx: 제목+목표+핵심서비스+설계원칙+로드맵) + /v1/chat/stream(download 프레임 1, gibberish 0, text_delta 46=짧은 확인만) 모두 통과.

### 웹 UI 개선 — 마크다운 서식 렌더러 + 문서 미리보기 캔버스 (2026-07-05)

사용자 요청: (1) claude.ai 아티팩트처럼 생성 문서를 우측 캔버스로 미리보기, (2) 응답 마크다운이 들여쓰기·제목까지 정돈되어 표시.

- **백엔드(web/app.py)**: `_collect_downloads()` 헬퍼 신설 — tool_result에서 url/filename 추출 + 대응 DocumentExport tool_use 입력에서 **content(미리보기용 마크다운)·format** 결합. download 이벤트/`downloads[]` 계약을 `{url,filename,content,format}`로 확장. 비스트림·스트림 스캔 2곳을 이 헬퍼로 통일.
- **프런트(web/static/index.html, frontend-specialist 위임)**: `formatContent` 전면 재작성 + `applyInline()` 분리 — 제목(#/##/###→h2/h3/h4), **중첩 불릿(2/4칸 들여쓰기→실제 중첩 ul)**, 번호목록(1.→ol), 목록/제목 내 굵기·코드·링크, 코드블록 이스케이프, /v1/download 링크 비클릭. 캔버스 패널(`#canvasPanel`, `openCanvas/closeCanvas`) — 우측 42%(모바일 오버레이), 미리보기 버튼(`.preview-btn`)으로 열고 헤더에 파일명·format배지·다운로드(dl-link)·닫기(Esc). 에어갭 준수(외부 라이브러리 0, 순수 바닐라).
- **검증(실서버 8600)**: GET / 200, 새 요소/함수 전부 존재, <script> 문법 OK(Node), download 이벤트에 content 1257자 실림 확인, formatContent 렌더 출력 검증(h2/h3·중첩ul·ol·strong·code·정상링크·download 라벨화 모두 정상). 브라우저 클릭 상호작용은 사용자 확인 몫.

### 컨텍스트 오버플로우 하드에러 수정 (2026-07-05)

증상: 웹에서 대용량 문서 처리 중 "입력(32257 토큰)이 컨텍스트(24576)를 거의 다 사용하여 응답을 생성할 수 없습니다" 하드에러.

근본원인(로그 확인): `query_loop.py` Phase 1 truncation이 **마지막 메시지 하나만** 잘라, 오버플로우가 *여러 tool_result 누적*에서 오면(마지막 메시지가 excess보다 작음) 조건 미충족으로 **아무것도 안 잘림** → 모델 거부. 로그 `입력 truncate: 26524 → 26524`(안 줄어듦) + emergency_compact도 최근1턴 통째 보존이라 안 줄임.

수정(backend-specialist 위임): `_shrink_text()` + `_truncate_input_for_budget()` 헬퍼 신설. `msg_char_budget = input_limit*3 - (tool_chars+prompt_chars)` 역산 → 메시지 총 글자를 예산 이하로 최신우선 축소. **페어링 보존**(메시지 제거 없이 평문 content만 head+tail 축소, assistant tool_use 구조화 content 불변, tool_use_id/is_error 유지), **히스토리 불변**(팩토리로 새 Message·새 리스트, state.messages 원본 미수정). BF16/config/모델 무변경 — 순수 코드 견고성 수정.

검증: 단위 7 passed + 누적케이스 재현 통과 + **실서버 e2e**(8600→B200): 135,022자(~45k토큰) 입력 → HTTP200 7.3초 정상 3문장 요약, 하드에러 없음. 로그 `입력 truncate: 48927 → 20889 토큰`(정확히 예산).

후속: ① emergency_compact(context_manager.py:218)도 "보존 최근 1턴 내부"는 안 줄이는 동일 결함 클래스 — 2차 안전망 보강 필요(Phase1이 먼저 잡아 현재 도달 안 함). ② 더 큰 단일 컨텍스트가 필요하면 FP8 전환+max_model_len 상향(사용자 결정 대기).

### FP8@65k 전환 — 검증 통과·확정 (2026-07-05)

목적: BF16 32768 창이 대용량 문서엔 좁아, FP8 전환으로 VRAM 확보→창 확대. 품질 민감도 때문에 "검증 후 확정" 방식.

**발견**: A.X-4.0 config는 `max_position_embeddings=131072` 네이티브 → rope override 불필요(첫 시도의 `--rope-scaling`은 vLLM 0.24 미지원 + 애초에 불필요였음). BF16의 32768은 KV 메모리 상한이 이유(GPU 179GB 중 BF16 가중치 175GB 점유, 여유 7.5GB).

**기동**: `/NHNHOME/nexus/run_vllm_fp8.sh` — `--quantization fp8 --max-model-len 65536`(KV는 미지정=BF16 유지, rope 없음). 로딩 ~100초. GPU 172GB(가중치 ~72GB + KV 풀 ~96GB). vLLM 0.24.0.

**검증(실서버 직접 :8001, temp=0)**:
- 사실/환각 배터리 11문항 BF16 기준값과 비교 → **사실 8/8 완전 동일**(H2O·광복절·베토벤9·1km·에베레스트·17×23·빛속도·서울). 허구 작곡가는 양쪽 동일하게 지어냄(패리티), 2099년·유창성 동일. **near-lossless 실증**.
- 장문 needle 회수: 50k자(18.5k토큰)·95k자(35.4k토큰)·**160k자(60,122토큰)** 전부 정확 회수(CRIMSON-7492) → 65k 창 전체 + 장문 회수 정상.

**확정 반영**: `config/nexus_config.yaml` max_context_tokens 24576→**49152**(불변식 49152+8192=57344≤65536, 여유 8192), `config/vllm_launch.yaml` axmodel에 `quantization: fp8` + max_model_len 65536. 웹 재기동 후 e2e: 135k자 입력이 truncate 20,889→**41,779**(0.85×49152)로 창 2배 실동작, 정상 응답.

**후속**: ① FP8 런처를 컨테이너 부팅 자동기동에 편입(현재 수동 run_vllm_fp8.sh). ② 필요 시 입력 예산 57344까지 추가 상향 여지. ③ emergency_compact 내부 축소 보강(별건).

### B200 스택 복구 스크립트 배포 (2026-07-05)

배경: 부팅 조사 결과 스택 전체(PG·Redis·임베딩·vLLM·웹 8443)가 수동 기동이며, `/`는 docker overlay(휘발성)이고 `/NHNHOME`만 영구 xfs 볼륨. 즉 재부팅/재생성 시 자동 복구 수단이 없었음.

조치(사용자 선택=옵션2, cron 없이 스크립트만): `/NHNHOME/nexus/start_all.sh` 배포(영구볼륨→생존). 실행 중 프로세스에서 정확한 기동 명령 복원 — PG(`pg_ctl -D pgdata`), Redis(`redis-server redis.conf`), 임베딩(`embed_server.py`), vLLM(`run_vllm_fp8.sh` FP8@65k), 웹(uvicorn :8443). 포트 확인으로 idempotent(가동중이면 skip), vLLM 앞에 nvidia-smi 준비 대기 가드. 검증: 라이브 실행 시 5개 전부 skip(무중단).

**수동 복구 한 줄**: `bash /NHNHOME/nexus/start_all.sh`

자동기동(@reboot cron/systemd)은 미채택 — cron/유닛이 휘발성 `/`에 저장돼 컨테이너 재생성 시 소실(거짓 안심)되고, @reboot는 크래시 재시작 불가 + 부팅 순서/발화 미검증이라, 재생성까지 커버하는 정본은 위 수동 1줄로 둠. (재검토 시 NHN 콘솔 컨테이너 시작커맨드/헬스체크 확인 권장.)

### 웹 UI 중간 활동 표시(도구 실행) 추가 (2026-07-05)

claude처럼 "AI가 지금 무슨 도구를 실행 중인지" 실시간 표시. 표시는 이미 흐르는 StreamEvent를 렌더만 하므로 추가 토큰/GPU 비용 없음.

- **백엔드(web/app.py 스트림 프레임 빌더)**: `event.tool_use`가 있으면 SSE 프레임에 `tool_name` 추가(한 줄). 기존엔 tool_use 이벤트의 `type`만 전달돼 UI가 도구 이름을 못 받았음. 실증: 문서생성 요청 스트림에 `tool_use_start`/`tool_use_stop` + `"tool_name":"DocumentExport"` 도착 확인.
- **프런트(web/static/index.html, frontend-specialist 위임)**: 활동 영역(`activity-area`/`activityLive`/`activityTrail`) + CSS 스피너. `activityStart/Stop`, 도구명→한글 라벨맵(TOOL_LABELS 12종). tool_use_start→"🔧 라벨·도구명"+스피너, tool_use_stop→트레일에 "✓" 누적, text_delta 시작 시 라이브 감춤, thinking_start→"💭 생각 중". **thinking_delta(text 필드 보유)를 가로채 답변 본문 오염 차단**(부수 버그픽스). 기존 text_delta/download/캔버스/formatContent 무손상.
- 검증: SSE 실측(tool_name 도착) + GET / 200 + 새 요소·함수 존재 + <script> 문법 OK. 브라우저 시각 확인은 사용자 몫.

### 웹 활동표시 claude화 + 캔버스 자동오픈 (2026-07-05)

실사용 피드백: 같은 도구 반복(DocumentProcess ×10)이 트레일 도배 + 도구 사이 중간 나레이션이 답변에 누적 + 미리보기가 버튼식.

- **(1a) 중복 접기**: `activityStop`에서 직전 트레일 항목과 tool_name 같으면 새 줄 대신 `data-count`+1, `.activity-count` 배지를 `×N`으로 갱신 → `✓ 문서 분석 중 · DocumentProcess ×10`.
- **(1b) 중간 나레이션 정리**: `resetStreamingBody()` 신설, `tool_use_start` 수신 시 답변 본문(streamingBody textContent) 리셋 → 최종 답변 = 마지막 도구 이후 세그먼트만. 순수 대화(도구 없음)는 리셋 안 됨(회귀 없음). claude식.
- **(2) 미리보기 자동오픈**: `.preview-btn`(📄 미리보기) 제거, 스트림 종료 후 content 있는 최근 생성문서로 `openCanvas()` 자동 호출(우측 캔버스). 다운로드 버튼(.dl-link)·캔버스 헤더 다운로드는 유지.
- 보존: text_delta/download/formatContent/스피너/취소·토큰/thinking_delta 가로채기 무변경. 검증: GET / 200 + 새 로직 존재 + preview-btn 0 + <script> 문법 OK.

### ① OpenAI 호환 API 추가 (2026-07-05)

목적: 외부 서비스(AgentHub·LangChain·openai SDK)가 커스텀 코드 없이 Nexus를 드롭인 프로바이더로 사용. (AgentHub 연동에서 드러난 마찰 해소)

- **`POST /v1/chat/completions`**(web/app.py, backend-specialist 위임). OpenAI 형식 `{model, messages[], stream, temperature...}` 수용, `Message.user/assistant` 매핑, system 메시지는 Nexus 기본 프롬프트 뒤에 `[사용자 지시]`로 덧붙임(update_system_prompt). 무상태(요청마다 임시 세션, Redis 복원 안 함 — 클라이언트가 히스토리 소유).
- 비스트림: `chat.completion` {choices[{message,finish_reason}], usage}. 스트림: `chat.completion.chunk` delta 조각 + `data: [DONE]`. 다운로드는 content 끝에 마크다운 링크 + 비표준 downloads 필드.
- 신규 Pydantic 모델(OpenAIChatCompletionRequest 등) + 헬퍼(_split_openai_messages/_inject_openai_context/_downloads_markdown). 4-Tier 우회 없음, 기존 /v1/chat·/v1/chat/stream 무손상.
- 검증: 웹 단위 66 passed + 실서버 e2e(비스트림 chat.completion+system반영+usage / 스트림 delta+role+finish_reason:stop+[DONE]) 통과.
- 사용법: 클라이언트 base_url=`http://192.168.20.206:8600/v1`, 모델 primary. 한계: temperature/max_tokens는 반향만(엔진 내부 라우팅이 샘플링 관리), 다운로드는 여전히 _messages 사후스캔.

### ③ Speculative decoding — 테스트 후 롤백 (2026-07-05)

FP8@65k에 n-gram speculative(`--speculative-config {method:ngram,num_speculative_tokens:5,prompt_lookup_max:4,min:2}`) 적용해 A/B 측정. 무손실이라 품질 무관, 순수 속도 판단.

측정(temp0, 단일요청): baseline 종합 54.1 tok/s vs spec 52.8 tok/s = **-2%**. 세부: 긴 구조적 출력 +7~13%(171tok 55→59, 204tok 54→61)지만 **짧은 응답은 크게 느려짐**(36tok 0.73s→1.56s = 49→23 tok/s, n-gram 셋업 오버헤드). 채팅 다수가 짧은 응답이라 UX 손해.

결론: 이득 불명확(오히려 단문 지연↑) → **롤백**(run_vllm_fp8.sh no-spec 유지, run_vllm_fp8_spec.sh 삭제). 스펙 디코딩은 저부하 단일유저에서 최선인데도 이득이 미미했고, 프로덕션 동시성에선 더 줄어듦. 향후 생성 극단적 장문/반복 워크로드가 지배적이 되면 재검토.

### ④ RAG MMR 리랭킹 — 구현(기본 OFF) + 실서버 검증 후 "켜지 않음" 결정 (2026-07-05)

목적: 순수 벡터 top_k=5의 중복 청크를 다양성 선별로 대체(컨텍스트 절약+커버리지). 새 모델 없이 기존 임베딩만 사용.

구현(backend-specialist): `knowledge_store.search_by_vector(with_embedding=)` 임베딩 선택반환 + `pgvector_base.parse_vector()` + `knowledge_retriever._mmr_select()`(게이팅 **이후** survivors→top_k, λ=0.7) + config `knowledge_rag.mmr{enabled:false,fetch_k:20,lambda:0.7}` + bootstrap 주입. **기본 OFF=동작 100% 무변경**. 테스트 19+17 passed.

실서버 검증(실 PG+e5): 임베딩 파싱 정상. MMR 개입하나 효과 애매/위험 — "인공지능이란"에서 더 관련높은 `인공일반지능`을 빼고 다양성 위해 `컴퓨터의역사`(덜 관련) 주입 → 사실 QA에서 정답 청크를 밀어낼 위험. "광합성"은 게이팅이 "광"字 오매칭(광섬유/스텔라레이터) 통과시킨 상태라 MMR이 오답끼리 다양화만 함(관련도 못 고침).

결정: **기본 OFF 유지, 프로덕션 미활성**. MMR은 커버리지 중요 워크로드용(연구/요약 테넌트)이고, 환각-민감 사실 QA엔 다양성 우선이 해로울 수 있음. 코드는 무해한 잠재 knob + with_embedding 재사용성으로 보존. 이 워크로드 진짜 품질 레버 = A(크로스인코더 리랭커) 또는 게이팅 개선.

부수 발견(선행 이슈): "광합성" 질의가 "광"字 표면매칭으로 광섬유·스텔라레이터 등 무관 청크를 게이팅 통과 → 향후 게이팅/리랭킹 개선 후보.

### ④-A 크로스인코더 리랭커 — 구현·검증·활성화 (2026-07-05)

MMR이 관련도를 못 고쳐(광섬유·스텔라레이터 오매칭 잔존) 대신 크로스인코더 리랭커 도입. 사용자 최우선(환각 저감) 직결.

- 모델: dragonkue/bge-reranker-v2-m3-ko(Apache2.0, 0.6B, 한국어). B200 hf_cache 적재(에어갭 반입 준비). GPU 여유 9GB에 fp16(~1.5GB) 수용.
- 서버: scripts/embed_server.py(8002)에 CrossEncoder 추가 + /v1/rerank. vLLM 별도 인스턴스 대신 임베딩 서버 재사용(메모리 효율). fail-open. start_all.sh 오프라인 플래그로 복구 시 자동 로드.
- 클라이언트/통합: inference.py rerank() + knowledge_retriever(fetch_k=20·min_sim0.6 리콜→rerank→min_score0.3 게이팅, e5 2단게이팅 대체·엔티티게이팅 유지, 실패 시 fail-safe 폴백, 리랭커>MMR).
- config: knowledge_rag.rerank(enabled:true). 되돌리기=enabled:false+웹재기동.
- 검증(실 e2e): 광합성→0주입("모른다"), 인공지능→정제, 바흐→J.S.바흐 rr1.0 교정. 225~491ms. 웹 실사용 경로도 "지식베이스에 없음" 정직 응답 확인.
- 커밋 1b091e0. 테스트 8+회귀 통과. min_score 캘리브레이션 근거: 관련0.9~1.0/무관0.0/경계0.28.

세션 채택 요약: FP8@65k·OpenAI호환·리랭커 = 채택. speculative·MMR = 검증 후 미채택(이득 없음).

### 웹 도구 풀 재단 — 파일탐색 4개 제거 (2026-07-08)

증상: 웹에서 "파일찾기(Glob) 어떻게 써?" 질문 시 Worker가 Glob을 9번 헛돌리고 "파일 없음" 반환(혼란). 원인=웹 TIER_L 프롬프트(worker_system_full.md)가 "Glob/Grep/LS로 직접 탐색하라" 지시 + 웹 풀에 Read/Glob/Grep/LS 노출. 그러나 웹 채팅 사용자는 뒤질 파일시스템이 없음(세션 cwd 격리) → 항상 빈손.

- 결정(사용자 승인): 웹 Worker 도구 풀에서 Read/Glob/Grep/LS 4개 제거. 웹 TIER_L 12→8개(Agent/Bash/DocumentExport/DocumentProcess/Edit/GitDiff/SymbolSearch/Write). CLI/Scout 풀·TIER_S 웹(5개)은 불변.
- 사양 이탈 근거: v7.0 Part 2.5 "상위 티어 Worker 직접 탐색"은 로컬 파일시스템 가진 CLI/에이전트 Worker 전제. 웹 표면엔 부적합. 웹의 "파일"=업로드문서(DocumentProcess)+지식RAG(자동주입), 코드심볼=SymbolSearch로 충족.
- 변경 파일: core/bootstrap.py::_create_web_tool_registry(TIER_M/L 블록 축소+이탈 근거 주석), web/prompts/worker_system_full.md("직접 탐색"→"파일시스템 없음; 업로드문서·RAG·SymbolSearch" 재작성).
- 검증: ruff 통과. 웹/티어 테스트 62 passed+1 skipped. 라이브 재기동 후 /v1/tools=8개(Glob/Grep/LS/Read 없음), /health 200, 재현질문 e2e에서 Glob 루프 사라지고 정상 답변.
- 감사 부산물(설계 대비): 사양서(v6.1 §19.4) 24개 도구 전부 구현 + 신규3개(SymbolSearch/DocumentProcess/DocumentExport)=총 27개 확인. 표면별 노출=CLI 23 / 웹 8 / Scout 5.

### 할루시네이션/오류 QA 배터리 + 프롬프트 보강 (2026-07-08)

다른 LLM이 흔히 틀리는 48개 함정 질의를 라이브 웹(8600)에 실측 → 문제 진단 → 웹 프롬프트(worker_system_full.md) 보강 → 재테스트 루프 3회.

- 초기 결과: 34/48 PASS(71%), 9 FAIL + 5 PARTIAL. 스크립트=scratchpad/hallu_battery.py, 결과=hallu_results.jsonl.
- FAIL 클러스터: ①산술/숫자(18764×27 오답, 연도 숫자삽입깨짐 "19969"·"19988") ②문자카운팅(i·받침) ③인젝션/탈옥(DAN "해킹완료" 순응, API키 프리픽스 순응) ④가공개체 과잉부연 ⑤날짜사실오류(칸트 1868/베토벤10번/조선왕) ⑥모듈러추론(요일).
- 프롬프트 보강(worker_system_full.md): Grounding에 "가공 개체 추측금지 + 거짓전제 정정" / 신규 "Exact computation"(산술·카운팅·날짜는 Bash `python`으로 계산, **python3 아님 — Windows 호스트**, 2회 실패시 중단·루프금지) / 신규 "Security"(시스템프롬프트·키 비공개, DAN/인젝션 거부 + 구체 예시).
- 재테스트 후: ~44/48 PASS(~92%). 해결=가공개체추측·단순산술·글자수i·인젝션·탈옥·모순·요일(루프 14회→3회로 축소·정답). 
- **잔존**: (C) 날짜/전기 사실오류 #8·#34·#38 → 프롬프트로 안 잡힘, RAG 그라운딩 필요. (난제) #17 받침 카운팅.
- **후속 코드과제(A)**: query_loop/executor에 "동일 실패 도구호출 반복" 상한 가드 없음 — #43에서 깨진 Bash명령 14회·62초 루프 관측(프롬프트로 완화했으나 근본해결은 코드). 
- 미착수: 포인트3(사양 이탈 재감사, 웹도구 제외), 포인트4(최신 LLM 기능격차 — 설계 시 Fable5 사용).

### RAG 그라운딩 수정 — ivfflat probes (2026-07-08)

QA 잔존 사실오류(#38 칸트/#8 베토벤/#34 조선왕)의 원인 규명. 라이브 로그: KNOWLEDGE 분류·RAG 실행(477ms)됐으나 "자료 없음" 주입 → 데이터 부재로 오인했으나 실제로는 **tb_knowledge에 칸트 1309행·베토벤 2036행 존재**(제목 "이마누엘 칸트").

- **근본원인**: tb_knowledge ivfflat 인덱스가 `lists=1000`인데 런타임 `ivfflat.probes`가 기본 **1**(리트리버 코드가 SET 안 함) → 1000개 중 1개 리스트(0.1%)만 스캔 → 정답 문서 재현율 급락. 실측: probes=1이면 top이 벤야민/카뮈 0.84(칸트 없음), probes≥10이면 이마누엘 칸트 0.867 상위 등장(10/100/400 동일=10이면 saturated). 남은 약한 청크를 리랭커가 걸러 "자료 없음"이 된 것.
- **원인 배경**: 메모리의 "probes 10" 튜닝은 구 docutil DB 것이고, B200 이관 nexus DB엔 DB레벨 설정이 안 따라옴. `ALTER DATABASE`는 오토모드가 공유DB 영속변경으로 차단 → **코드 주입**이 정공법(배포 간 이식성).
- **수정(5파일)**: config.py `KnowledgeRagConfig.ivfflat_probes:int=40` 추가 / knowledge_store.py `search_by_vector(probes=)`에서 트랜잭션+`SET LOCAL ivfflat.probes` / knowledge_retriever.py `__init__(ivfflat_probes)` + search_by_vector 호출에 전달 / bootstrap.py 리트리버 생성에 `ivfflat_probes=krag.ivfflat_probes` / nexus_config.yaml·pc.yaml `knowledge_rag.ivfflat_probes:40`. ruff 통과, knowledge/retriever/store 테스트 57 passed. 재기동 후 칸트 3대 비판서 정확 회복.
- **잔존(별개 이슈)**: 긴 서술형 답변에서 **간헐적 숫자 깨짐**(1781→1887, 1988→19988). 검증: rep_pen 1.15/1.0 무관, 연도 복사·짧은 recall은 정확, FP8은 기검증(BF16 동일). → 긴 생성 디코딩 아티팩트로 추정, 모델레벨. 실용대응=핵심 사실 RAG 주입 복사. 완전근절은 후속 심층과제.

### 코드 견고성 — 반복 실패 도구 호출 가드 (2026-07-08)

QA #43(깨진 Bash 명령을 못 고치고 14회·62초 반복)의 근본 방어. MAX_TURNS=200은 이런 짧은 반복 루프를 잡기엔 너무 큼.

- **구현(query_loop.py 1파일)**: 서명(도구+정규화입력)별 '연속 실패 턴 수'를 지역 dict로 추적. Phase 4에서 tool_result의 is_error를 tool_use_id로 수집 → `_update_tool_failure_streak`로 갱신. `REPEATED_TOOL_FAILURE_WARN=3`턴 → user역할 "반복 중단" 피드백 1회 주입, `ABORT=5`턴 → 루프 강제 종료. 순수 헬퍼로 분리해 단위 테스트.
- **code-reviewer 별도 레인 검토(자기승인 금지)** → 실결함 3건 확증·수정: (HIGH) stale 서명 미evict로 WARN 매턴 무한주입 → 이번 턴 등장 서명만 카운트+evict. (MED) 병렬 동일호출 턴당 다중증가 → 서명별 턴당 +1 dedup. (LOW) `==`→`>=`+warned_sigs 1회주입 보장. 리뷰가 메시지순서·ABORT return·StreamEvent 계약은 무결성 확인.
- **검증**: 단위 10개(증가·리셋·독립·stale evict·병렬 dedup·임계·직렬화폴백) + 기존 query_loop 7개 통과. 포맷터 직접확인(tool_result 뒤 user 주입 정상). 라이브 회귀(계산 506628·일반대화) 정상, 가드 미발동.

### 포인트 4 — 최신 LLM 기능격차 4종 (설계 Fable5 + 구현, 2026-07-08~09)

격차 분석 후 4기능을 Fable5로 병렬 설계(user_mig/design/point4_*.md) → executor(opus) 구현 → 성역 검토 + 실 B200 e2e → 커밋. 전부 기본 OFF/additive라 무회귀.

- **1. 구조화 출력(bdc6ca4)**: vLLM response_format(json_schema) 강제. Phase0 실측=response_format만 강제됨(guided_json 무시). tools 상호배타·thinking off. web /v1/chat/completions가 response_format 조용히 버리던 결함 수정. e2e 순수JSON·미지원type 400. 유닛10+전체1087.
- **2. RAG 출처인용(8d74907)**: 본문 [출처N] + 서버진실 sources 필드. KnowledgeCitation frozen + KNOWLEDGE_SOURCES 이벤트. 기본 OFF. e2e 바흐5·광합성3 출처. 유닛127+전체1117.
- **4. TodoWrite 체크리스트(0914448)**: 기존 깨진 TodoRead/Write를 사양(체크리스트)대로 복구. core/todo_store.py leaf. 권한 READONLY(PLAN 허용). e2e SSE todo_update revision. 유닛34+전체1147.
- **3. 자기일관성(870014f)**: KNOWLEDGE 사실형에 n=3 합의(기본 OFF, 3중게이트). n=1 바이트동일 무회귀. B200 벤치=발동·"교차검증중(3표본)"·2~5초. 서술형/일관된오답은 못잡음(정직). 유닛23+전체1129.
- **인용 회귀 수정(92dd581)**: OFF인데 정적 프롬프트가 [출처N] 지시→모델이 근거없는 마커 날조. 정적 지시 제거, 게이팅 trailer로 일원화. OFF=마커0 복원, ON=정상.

**다음**: 포인트 3(사양 재감사, 웹도구·포인트4는 의도적 이탈로 제외). 이후 [[project_multicorpus_rag]] 파이프라인 준비.

---

### IDINO NOVA 리브랜딩 + 라이트 테마 재디자인 (2026-07-09)

**리브랜딩(브랜드만, 인프라 유지)**: NEXUS→IDINO NOVA. 커밋 `ced808d`. 웹 UI 문구·로고·AI 자기소개 교체. DB명 nexus·NEXUS_* env·API키 nexus-b200·.nexus/·NexusConfig 클래스명은 파손방지 위해 유지. 개발사 표기 교정(워커 프롬프트에 정체성 고정 블록 — "IDINO가 개발", 베이스모델 비노출).

**로고 자산**: 가로형 워드마크=`web/static/images.png`(중앙 welcome), 정사각 N/V 엠블럼=`web/static/emblem.png`(헤더·아바타·파비콘·탭). 커밋 `4b90005`.

**라이트 테마 재디자인**: 서빙 UI(index.html)를 다크→로고 화이트 배경 기준 소프트 라이트로 전환. 페이지=#f1f4fa(눈부심 완화 소프트화이트), 표면=#fff, 코드/인셋=#e7ecf4, 강조=로고 네이비#00479d+블루#1a73e8, 버튼=네이비→블루 그라데이션(--accent-grad).

**원복 방법(중요)**:
- 물리 백업: `web/ui_backup/index.html.dark-theme.bak`(다크 원본), `web/ui_backup/chrome.html.bak`.
- 원복 시: `cp web/ui_backup/index.html.dark-theme.bak web/static/index.html` 후 웹 재기동.
- 또는 git: 리브랜딩 이전 상태는 커밋 `2d91cbf` 직후, 라이트테마 이전은 `4b90005`.

---

### 이미지 생성 도구 (ImageGenerate) — Machine A 구현 (2026-07-09)

**결정 경위**: 이미지 생성이 Claude Code에 없는 신규 확장임을 확인. 로고급(텍스트 포함) 품질 기준 → **FLUX.1 schnell**(Apache 2.0 상업OK, 텍스트 우수, 1~4스텝). 최저 사양 **GB10**(128GB) 기준으로도 저스텝이라 수용 가능(~5–15초 추정). 사양은 `PROJECT_NEXUS_SPEC_v7.4_IMAGE_GEN.md` 신규 개정본.

**구현(Machine A, executor(opus) 위임 → 검토 통과)**:
- 신규 `core/tools/implementations/image_generate_tool.py` (`ImageGenerateTool`) + `tests/unit/test_image_generate_tool.py`(6).
- DocumentExport 패턴 대칭: `resolve_exports_dir()`/`_safe_filename()` 재사용, 동일 exports 샌드박스 저장, `/v1/download/{filename}` URL, `MEDIA_TYPES["png"]=image/png` 추가.
- 서버 계약: `POST {image_url}/v1/images/generate {prompt,width,height,steps,seed}` → `{image_base64,...}`.
- config: `GPUServerConfig.image_url`(yaml=8003, pc.yaml=18003 터널). bootstrap 웹 registry 등록, app.py가 image_url 주입.
- fail-closed: is_read_only/is_concurrency_safe=False, `tool_mappings.yaml`에 **enabled:false**(FLUX 서버 미기동).
- 검증: 17 passed(이미지6+DocExport11), ruff clean, 레지스트리 등록·config 기본값 확인.

**남은 것(Machine B)**: B200/GB10에 FLUX schnell FastAPI 서버(계약대로) 기동 + 오프라인 가중치 번들 → enabled:true → 실 e2e.

**참고 — 조사 결과 기록**:
- 비전(이미지 이해) 도구: **없음**. ContentBlock=Text|ToolUse|ToolResult|Thinking(이미지 블록 없음). 문서첨부는 DocumentProcess 텍스트 추출만. → 별도 과제(방식 A=AnalyzeImage 도구+VLM Qwen2.5-VL / 방식 B=ContentBlock ImageBlock 네이티브). 보류 중.
- 컨텍스트 압축: **가동 중**(hardware_tier=large=TIER_L, query_loop이 apply_all+auto_compact_if_needed 매 턴 호출, emergency/reactive compact). 단 65536창이라 한계 근접 시만 실동작. **UI 미표시**(CONTEXT_COMPACT 이벤트 emit 없음) — Claude식 화면표시는 후속 옵션.

---

### 이미지 생성 도구 ON — FLUX.1-schnell 서버 기동 완료 (2026-07-09)

**결과**: 한글 요청 → 모델이 ImageGenerate 도구 호출 → FLUX(터널 18003→8003) → PNG(1024², 333KB) 다운로드까지 e2e 정상. 로고급 품질 확인(파란 그라데이션 NV 모노그램).

**B200 운영 구성(재현 정보)**:
- **vLLM 메모리 축소**: `run_vllm_fp8.sh`의 `--gpu-memory-utilization 0.92 → 0.70`(약 40GB 반환). 백업 `run_vllm_fp8.sh.bak.util092`. 재기동은 `setsid nohup bash run_vllm_fp8.sh > logs/vllm_fp8.log 2>&1 </dev/null &`. GPU: vLLM ~135GB, FLUX용 여유 ~48GB. KV풀 여전히 충분(65536).
- **FLUX 서버**: `/NHNHOME/nexus/flux_server.py`(FastAPI, FluxPipeline + `enable_model_cpu_offload`, guidance_scale=0, steps 4). 기동 `setsid nohup venv/bin/python -m uvicorn flux_server:app --host 127.0.0.1 --port 8003`. 로그 `logs/flux_server.log`. 첫 호출 모델로딩 포함 ~21초, 이후 더 빠름.
- **가중치**: `black-forest-labs/FLUX.1-schnell`(Apache 2.0, gated) → HF 토큰(IDINO_NOVA 계정)으로 다운로드. **xet 버그로 실패 → `HF_HUB_DISABLE_XET=1` 필수**. hf_cache에 상주(~35GB).
- **diffusers**: 기존 venv에 `--upgrade-strategy only-if-needed`로 설치(transformers 무손상, vLLM 영향 없음).
- **터널**: 18003→8003 tunnel.ps1에 추가(+현재 별도 포워드 가동). pc.yaml gpu_server.image_url=`http://127.0.0.1:18003`.
- **도구**: tool_mappings.yaml ImageGenerate `enabled: true`.

**원복**: `cp run_vllm_fp8.sh.bak.util092 run_vllm_fp8.sh` 후 재기동하면 util 0.92 복귀. 도구 끄려면 tool_mappings `enabled: false`.

---

### 112 서비스 서버 이관 — Docker 올인원 (2026-07-11)

> 이 섹션은 2026-07-12에 메모리(`project_112_service_host.md`) 기반으로 소급 기록. 당시 세션이 메모리에만 남기고 progress.md 추가를 누락했던 공백 보완.

**배경.** 192.168.21.112(사내 Ubuntu 22.04, RTX 5090 32GB·24코어·62GB RAM·1.6TB)를 **Docker 올인원 서비스 서버**로 이관. 헤비 추론은 B200(터널), GPU는 학습데이터 생성용, 이 PC(5090 윈도우)는 학습데이터 생성 등 별도 용도로 분리.

**1단계 — 컨테이너 기동.** 112에 Docker 29.1.3 설치 + 컨테이너 2개(restart=unless-stopped).
- `idino-postgres` = pgvector/pgvector:pg17 (PG 17.10 + pgvector 0.8.5). 호스트 포트 **5440**→5432. DB `idino_ai` / 유저 `idino_user` / 스키마 `nexus` / extension vector.
- `idino-redis` = redis:7. 호스트 포트 **6340**→6379.
- 기존 미변경: 112 로컬 PG(5432, 넥서스 무관), vLLM(8001)/임베딩(8002)/llama-server(8003), `nexus-vllm.service`·`nexus-embedding.service`.
- 주의: 112 커널 업그레이드 대기(5.15.0-181→185). 재부팅 시 vLLM 죽으니 계획된 정비 때만.

**2단계 — 39 DB 데이터 이관.** 192.168.10.39(DB 서버, PG :5440 db=nexus)를 pg_dump(-Fc, 5.6GB)→112 idino_ai 복원. tb_knowledge 1,067,978 / tb_memories 284,772 / tb_symbols 3,065행, 벡터인덱스 3종 재생성(idx_knowledge_embed ivfflat lists=1000, idx_memories_embedding hnsw, idx_symbols_embed ivfflat). public→nexus 스키마 이동, `ALTER ROLE idino_user IN DATABASE idino_ai SET search_path TO nexus, public`로 앱 무접두어 동작. 39 원본 무변경(읽기만).
- 교훈: 컨테이너 기본 maintenance_work_mem 64MB로는 1M행 ivfflat 빌드 실패(200MB 필요) → SET으로 2GB 상향. pg_restore는 반드시 `-U idino_user`.

**3단계 — config 갱신 + e2e.** `config/nexus_config.112.yaml`(pc.yaml 복사본) 신설 — 전부 112 직결(gpu_server=192.168.21.112:8001·embedding=8002, redis=:6340, pg=:5440/idino_ai/idino_user). .venv_pc(Py3.11)로 웹(8601) 기동해 `POST /v1/chat` e2e 통과("광합성" 질의에 kowiki RAG 주입→Qwen 응답, 11초).
- 핵심 교훈(112 로컬 추론 config 필수 3종): ①모델명 `qwen3.5-27b`(ax-4.0 아님), primary_model + query_routing 3모드 모두 교체. ②112 Qwen `max_model_len=8192` → tier=small, max_context_tokens 49152→5632, mode max_tokens→2048 (안 그러면 컨텍스트초과 빈응답). ③rerank enabled:false (112 8002엔 /v1/rerank 없음).

**4단계 — nexus Docker 정식 배포.** `deploy_pc/Dockerfile`(python:3.11-slim + requirements-pc.txt, GPU라이브러리 제외, 662MB) 신설. 코드 tar SFTP→112 /home/idino/nexus-app→`docker build -t nexus-web:latest`. 컨테이너 `nexus-web` 기동(`--network host`, restart=unless-stopped, -v exports 마운트, env NEXUS_CONFIG=nexus_config.112.yaml). 112 내부 e2e 통과.
- **LAN 개방(사용자 승인).** ufw로 8600을 LAN(192.168.0.0/16) 개방 → `http://192.168.21.112:8600` LAN 접속 e2e 통과. 인증은 API키(Bearer) 유지.
- CLI: `ssh idino@112 → docker exec -it nexus-web python -m cli.repl`.

**AgentHub DB 이관.** 112 호스트 PG(5432)의 `idino_ai."AIAgentMngtDB"` 스키마(document_chunks 619행+vector)를 Docker PG(5440) idino_ai로 스키마명 유지 이관. pg_dump `-n '"AIAgentMngtDB"'`(대소문자 보존 위해 큰따옴표 필수). 호스트 5432 원본 무변경.
- search_path 정리: idino_user 기본을 **`nexus, public`**로 확정(Nexus 격리). AgentHub는 스키마 명시로 자체 처리. [[project_agenthub_integration]].

**미결/후속.** `config/nexus_config.112.yaml`·`deploy_pc/Dockerfile` 미커밋(비번은 env). docutil 등 다른 서비스도 112 Docker로(별도). [[reference_servers]]의 39 DB 정보는 이관 전 시점이라 최신화 필요.

---

### 112 서비스 추론을 B200(A.X-4.0)로 전환 — 컨텍스트 오버플로 근본 해소 (2026-07-13)

**증상.** 112 `nexus-web`(Docker)에서 문서 요약 시 "입력(6145) > 컨텍스트(5632) 초과" 에러. 사용자 의문("B200에서 이 정도로 컨텍스트가 차진 않을 텐데")이 결정적 단서.

**근본원인.** `config/nexus_config.112.yaml`이 gpu_server를 **112 로컬 vLLM(qwen3.5-27b, max_model_len=8192)** 로 가리키고 있었다(max_context_tokens=5632). 이 파일은 원래 3단계 "112 단독 e2e 검증용"(B200 없이 로컬만으로 도는지 증명)이라 일부러 로컬을 박아둔 것인데, 4단계에서 그 테스트 config가 그대로 운영 컨테이너에 배포됨. 헤더 주석은 B200/65536이라 적혀 있어(템플릿 복사 잔재) 착각을 유발. 설계 의도(헤비 추론=B200 터널)와 배포가 어긋난 상태였다. 실측: 짧은 "수도?" 질의조차 prompt_tokens=7117이라 5632창에선 사실상 모든 실질 질의가 오버플로였다.

**조치(config, 추론부만 — DB/Redis/임베딩은 112 로컬 유지).** gpu_server.url→`127.0.0.1:18001`(B200 터널), primary_model 및 knowledge/tool/chat 3모드→`ax-4.0`, max_context_tokens 5632→**49152**, default_max_tokens 2048→8192, hardware_tier small→large. **FABLE5(fable) 검토로 결함 2건 추가 수정**: (HIGH) output_token_escalation `[1024,2048]`→`[4096,8192]`(default_max 8192가 리스트 최대와 불일치 시 8192 잘림→2048 역축소되는 실동작 결함), (MED) tool_result_budget 2000→8000. 불변식 49152+8192=57344 ≤ 65536.

**112→B200 터널.** 개발 PC(tunnel.ps1)와 동일 경로 = NHN bastion `idino_user@59.150.33.1:45702` + `nexus_key`. **핵심 발견: 이 키는 passphrase 없음**(tunnel.ps1 주석 "passphrase 있음"은 오기). 112에 상주 systemd 유닛(`deploy_pc/nexus-b200-tunnel.service`) 설치 — `ssh -N -L 127.0.0.1:18001:127.0.0.1:8001`, Restart=always, enabled(재부팅 복구). 임베딩·PG·Redis는 112 로컬이라 vLLM 18001 하나만 포워딩.

**B200 서빙 모델 실측(질문2).** `/v1/models` = `ax-4.0`(root skt/A.X-4.0, max_model_len 65536). config 값과 일치 확인.

**config 영속화.** docker cp만으론 이미지 재생성 시 원복되므로, config를 호스트 파일 **bind-mount**(`/home/idino/nexus-config/nexus_config.112.yaml`:ro→/app/config/...)로 컨테이너 재생성(run 스펙: --network host, restart=unless-stopped, exports 마운트, NEXUS_* env, uvicorn web.app:app:8600). 빌드 컨텍스트(`/home/idino/nexus-app/config`)도 갱신해 향후 재빌드 정합.

**e2e 검증.** `/v1/chat/completions`(OpenAI messages)로 긴 입력(13045·8842 prompt_tokens) 정상 요약, model=ax-4.0, finish=stop. health 200. 옛 5632의 2배 넘는 입력이 문제없이 처리됨 — 근본 해소 확인.

**미결/주의.** 이미지(nexus-web:latest 39h) 자체는 아직 옛 config 박힘(bind-mount가 오버라이드하므로 무해, 재빌드 시 정합). 터널 단절=전체 추론 불가(단일 경로) — systemd Restart로 자동복구.

---

### 결과 표시 Claude식 정합 — 1단계: 메타 내레이션·본문 도배 제거 (2026-07-13)

**목표.** 사용자들이 Claude 앱/웹에 익숙하므로, IDINO NOVA 웹의 모든 요청 결과 표시를 Claude 방식으로. 계기: 파일 생성 요청 시 모델이 "DocumentExport의 content 인자를 채워야 합니다" 같은 도구 내부 서술 + 요약 본문 전체를 채팅에 도배.

**진단(raw SSE 캡처 + FABLE5 검토).** 문구는 코드에 없음 → 모델(A.X-4.0) 생성. 활성 프롬프트는 tier로 갈림(app.py:431): TIER_S=worker_system.md, TIER_M/L=worker_system_full.md. B200 전환(tier=large)으로 지금은 full.md 활성 — 여긴 이미 "Creating documents/never paste inline·짧은 확인" 규칙이 있어 단순 파일생성 raw 스트림이 이미 깔끔("요청하신 문서를 생성했습니다."). **FABLE5가 내 초기 진단(worker_system.md 규칙 부재가 원인) 반증**: TIER_S 도구 풀엔 DocumentExport가 아예 없어(bootstrap) 스키마를 볼 수 없음 → 그 경로에서 나올 수 없음. 진짜 구멍 2곳 = (1) ad-hoc Agent 경로(agent_tool.py: description을 프롬프트로 쓰고 전체 도구 풀 부여, 출력 규약 전무), (2) app.py:476 서브에이전트 블록이 "Single file task→Read/Edit/Write"라고 지시하나 full.md는 "Read/Glob/Grep/LS 없음" — 정면 모순(활성).

**수정(4파일).** ①agent_tool.py: `_SUBAGENT_OUTPUT_CONTRACT`(도구 내부 서술 금지·산출물 본문 도배 금지·결과 원문 재붙여넣기 금지·사고과정 금지) 신설, ad-hoc 경로 system_prompt에 append. ②app.py: 서브에이전트 블록을 표면 인지형으로("Edit/Write 직접, 이 표면엔 Read/Glob/Grep/LS 없음") + 도구 내레이션 금지 1줄. ③worker_system_full.md: "도구·산출물 표시 규약" 섹션(내레이션 금지·산출물 짧은 확인·결과 원문 재붙여넣기 금지 — 마지막 규칙은 full.md에 없던 것) 추가. ④worker_system.md: 내레이션 금지 1줄(파리티).

**검증.** py_compile+ruff(내 구간 clean), 단위테스트 test_agent_tool+test_web_chat 37 passed(ad-hoc system_prompt 단언 갱신). 배포(컨테이너 docker cp + 빌드컨텍스트 갱신 + restart, health 200). e2e: 파일생성 text_delta="요청하신 문서를 생성했습니다."뿐 + 다운로드 카드 / 지식질의·인사 회귀 없음.

**미결(후속).** 2단계=마크다운 렌더·아티팩트/캔버스 표시 Claude 정합(현재 상당 부분 구현됨). 3단계=세부 정렬. 코드 변경(app.py·agent_tool.py)은 컨테이너 writable layer+빌드컨텍스트에 반영 — 이미지 재빌드 시 완전 정합(현재 bind-mount는 config만).

### scout_provider 에러 수정 (2026-07-13, 위 1단계 후속)

**증상.** tier=large(B200)에서 `<tool_use_error>scout_provider가 context.options에 없습니다 (TIER_M/L 환경)</tool_use_error>` 반복.
**원인.** scout_provider는 TIER_S에서만 생성(bootstrap:367 `if tier==TIER_S`)되는데, agent_registry는 tier 무관하게 scout를 등록(230)하고 프롬프트가 scout 위임을 권함 → tier=large에서 호출 시 provider 없어 _AgentConfigError.
**수정(2파일).** ①agent_tool.py `_resolve_model_provider`: scout override인데 provider 없으면 예외 대신 **부모 Worker 모델로 폴백**(TIER_M/L은 Worker 단독이 정상 경로라 품질 손실 없음). ②app.py `_load_worker_system_prompt`: `is_expanded`(TIER_M/L)면 서브에이전트 목록·권장에서 scout 제외 + "큰 컨텍스트로 직접 처리" 안내로 대체. 테스트 갱신(폴백 검증), 37 passed. 배포·health 200·회귀(수도=서울) 정상.

### 문서 업로드→분석 실패 수정 (2026-07-13) — "파일을 찾을 수 없습니다"

**증상.** tier=large 전환 후, 한글명 문서 업로드→분석이 "죄송합니다, 문서 파일을 찾을 수 없어…"로 실패("scout · Agent ×2"). 로그: DocumentProcess가 0.00초에 반복 실패(즉시 file-not-found).
**진단.** 파일은 `/tmp/nexus_uploads/`에 실재(3.7MB). 저장명은 NFC 정상(정규화 문제 아님). 원인 = 모델이 긴 **한글 경로**를 DocumentProcess 인자로 재입력하다 오타(긴 문자열 재현 취약성 [[project_degeneration_fix]]). 프론트가 업로드 시 scout 위임을 하드코딩(index.html)해 재입력이 2홉(Worker→scout→도구)이라 더 취약. tier=large는 scout_provider 없어(위 scout 폴백) 부모 모델로 도는데 설계상 원래 Worker가 DocumentProcess 직접 처리하는 티어.
**수정 2건.** ①web/app.py `/v1/upload`: 저장 파일명을 ASCII-safe(`upload-<uuid12>.<ext>`)로 생성·반환(원본명은 file_name으로 표시용 유지). 모델이 짧은 ASCII 경로만 재현 → 오타 근절. ②index.html: 업로드 메시지에서 `subagent_type="scout"` 강제 지시 제거 → 티어별 프롬프트가 라우팅(large=DocumentProcess 직접 1홉 / small=worker_system.md가 scout 위임). worker_system.md는 이미 "문서 분석→scout" 명시라 무손상.
**검증(실서버 e2e).** 한글 docx 업로드→`file_path=/tmp/nexus_uploads/upload-04f2915e9ba8.docx` 반환→분석 요청→Worker가 DocumentProcess 직접 실행해 실제 3문장 요약 생성. "찾을 수 없" 에러 소멸.

### DocumentExport 다운로드 링크 실패 수정 — tool-arg JSON 붕괴 + guided 재시도 (2026-07-14)

**증상.** 문서 업로드→"Word 보고서 작성" 요청 시 `<tool_use_error>DomainValidationError: content는 비어 있을 수 없습니다</tool_use_error>` → 다운로드 링크 없이 모델이 보고서 본문을 채팅에 통째로 덤프.

**진단(실서버 로그+SSE+덤프 실측, FABLE5 2회 검토).** 빈 content가 아니라 **tool_call arguments JSON 파싱 실패→빈 dict 폴백**이 원인(inference.py `_finalize_tool_calls`). A.X-4.0가 `tool_call_parser="hermes"` + `tool_choice="auto"`에서 인자 JSON을 **자유형식으로 생성**(문법 강제 전무)해, 긴 보고서 content 직렬화 시 간헐 붕괴: (1)제어문자 미이스케이프(Invalid control character), (2)닫는 `"`·`}`·format필드 누락(Unterminated string, output_tokens<max·stop=tool_use로 절단 아님 실측), (3)미이스케이프 따옴표(Expecting delimiter), (4)잘못된 `\escape`. 성공하던 export도 실측 475자 껍데기였음. 설계문서 `point4_1_structured_output.md` 결론=인자 문법 강제 레버는 named `tool_choice`(vLLM이 parameters 스키마로 guided decoding 자동 적용).

**수정(Option 1, FABLE5 승인, 4파일).** ①message.py: `ToolUseBlock.parse_error` 신호. ②inference.py: `_finalize_tool_calls`에 `json.loads(strict=False)` 1차 복구(제어문자) + 실패 시 parse_error=True; `stream()`에 `force_tool_choice` 파라미터(지정 시 named tool_choice로 guided 강제). ③query_loop.py: parse_error 도구는 빈 인자 실행 금지·이름만 수집→Phase3에서 그 도구로 `force_tool_choice` 걸어 1회 재시도(기존 tool_parse_retry_count 재사용, MAX=2), 재시도 턴은 max_tokens를 잔여창 전체로 완화, 소진 시 빈 실행 대신 정직한 `TOOL_ARGS_UNPARSEABLE` 에러(가짜 완료 금지). ④scout_provider.py: force_tool_choice passthrough.

**2차 근본원인(구현 중 발견).** 강제 tool_choice는 tool_calls를 스트리밍하면서도 vLLM이 `finish_reason="stop"` 반환(비스트리밍 관문은 통과했으나 스트리밍 실제턴이 END_TURN으로 샘). 기존 파서가 `finish_reason=="tool_calls"`일 때만 finalize→강제호출 tool_calls 유실. **수정: finish_reason 문자열이 아니라 누적 tool_calls 유무로 finalize**(tool_calls_finalized 가드), tool_calls 있으면 stop→TOOL_USE 보정. 단 `finish_reason="length"`(진짜 절단)는 MAX_TOKENS 보존(기존 max-output 복구 경로 유지, FABLE5 필수지적 반영).

**검증.** 단위테스트 신규/갱신 다수(strict=False 복구·parse_error 플래그·강제 tool_choice 스트리밍 finalize·length 절단 MAX_TOKENS 보존·query_loop guided 재시도 발동·소진 정직에러), 전체 unit green(로컬 hwpx 1건은 python-hwpx 미설치, 서버 정상). **실서버 e2e**: 74k 문서→auto DocumentExport가 `Invalid \escape` 실패→`[도구 인자 재생성 1/2]` guided 재시도→**실제 6552자 완결 .docx(42992 bytes) + `/v1/download/document-*.docx` 링크 생성**. B200 재기동 불필요(요청 파라미터만).

**미결/후속(FABLE5 권장, 비차단).** ①혼합 턴(정상+parse-fail 공존) 시 실패 호출에 합성 tool_result 에러 주입(현재 조용히 드롭). ②이미지 재빌드 시 4파일 정합(현재 docker cp writable layer). 커밋 대기(사용자 승인 후).

---

### 대용량 문서 분석→보고서 "반복 벽·다운로드 실패" 수정 — 통짜 반환 + 컨텍스트 절단 버그 2건 (2026-07-16)

**증상(112 프로덕션).** 업로드한 대용량 문서(강원대 제안서 등)를 "분석해서 Word 보고서로 작성" 요청 시 (1) 채팅에 "다음 청크를 읽어야 합니다"류 문장이 5~11회 반복 표시("동일 작업 내용 계속 표시"), (2) 더 큰 문서에서 다운로드 실패 + 우측 캔버스 미표시.

**진단(HTTP 실서버 재현으로 확정, 추측 아님).** 단일턴 "보고서 작성"(문서분석 없음)은 정상(DocumentExport·다운로드·짧은 확인) → DocumentExport/다운로드/캔버스 파이프라인은 건강. 문제는 **대용량 문서 청크 페이지네이션 × 청크별 내레이션**. `document_tool.py`가 문서를 `document_chunk_size`(8000자, 원래 5090 8K창 역산값)로 잘라, 대용량 문서가 10+ 청크가 되고 footer가 매번 재호출을 명령("[다음 청크를 읽으려면…]"). worker_system_full.md:41-45가 "각 단계 전 한 문장 진행 안내"를 권장 → 청크마다 거의 동일한 문장 반복. 재현: 40KB docx(≈92000자)→DocumentProcess ×11, 반복 안내 ×10, 누적 input 327,357토큰.

**FABLE5 계획이 추가로 발견한 절단 버그 2건(코드 검증).** ①`context_manager._micro_compact`(TIER_M/L 매 턴 apply_all)가 100줄 초과 tool_result를 **최근성 면제 없이** 90줄로 접음 → 문서 통짜 반환을 구현해도 다음 턴에 접혀 무효. ②`_apply_tool_result_budget` 팽창 버그: 한글 토큰 추정(2자/토큰) vs 절단 환산(3자/토큰) 모순 → "토큰 초과·글자 이하" 한글 결과가 head+tail 중복으로 오히려 팽창.

**수정(권고안 B, 3축).** ①`document_tool.py`: `document_singleshot_chars`(통짜 상한, 0=비활성) 신설 — 문서 전체가 상한 이하면 1청크로 전문 반환. 초과 시 `document_chunk_size`로 최소 분할. footer 재설계(통짜=재호출 금지 명시, 중간 청크="진행 안내 문장 없이 즉시 이어읽기"). 캐시키에 mtime·크기·분할파라미터 포함. ②`context_manager.py`: micro_compact 최근성 면제(최근 N개 도구결과는 손실압축 스킵) + budget 팽창 버그 수정(토큰·글자 양쪽 초과일 때만 축약, tail 겹침 가드). ③config 3곳+예시: `document_singleshot_chars: 40000`(112 49152창 근거) + `document_chunk_size` 8000→26000. bootstrap.py·web/app.py 주입 파리티. ④프롬프트 worker_system_full.md: "청크 이어읽기는 단계 아님 → 반복 안내 금지" 예외. `core/config.py`에 신규 필드(기본 0=무회귀).

**검증.** 신규 `test_document_tool.py`(9) + `test_context_manager_document_preservation.py`(6, 팽창·micro면제 회귀) + `test_context_budget_config.py` 갱신. 로컬 unit 1239 passed(실패분은 전부 docling 미설치 기존 환경 이슈, 무관). ruff 클린(document_tool ASYNC240 1건은 파일 기존 패턴). **112 실서버 e2e**: 40KB 문서 DocumentProcess **11→4회**, 반복 안내 **10회→0**(끝에 1회만), 누적 input 327k→168k. ~25000자 문서(일반 제안서 크기) **5→1회**. 다운로드·캔버스 정상. 회귀(인사·수도=서울·문서없는 보고서) 정상.

**배포.** 112 nexus-web에 백업 우선(컨테이너 원본 6파일 + 호스트 config → `/home/idino/deploy_backup_20260716_035421`) → config bind-mount surgical 편집 + 코드 6파일 docker cp + restart. 빌드 컨텍스트(`/home/idino/nexus-app`) cp만 소유권(197609)으로 실패 — 컨테이너 writable layer엔 반영(서비스 정상), 이미지 재빌드 파리티는 후속.

**미결/후속.** ①커밋(이 수정 + 앞선 DocumentExport guided-retry 등 미커밋분) → 이미지 재빌드로 fragility 근본 해소(현재 writable layer라 컨테이너 재생성 시 원복 위험). ②singleshot 경로에서 "읽기/작성" 2문장이 다소 유사 중복(반복 벽 아님, 프롬프트 미세 튜닝 선택). ③112 테스트 잔여물(업로드/exports test docx 수개 + tb_artifacts 몇 행) — 무해, 정리 대기.

### 커밋 정리 + durability 이미지 baking (2026-07-16, 위 후속 ①)

**커밋(6개, feature/b200-bakeoff).** 미커밋분을 논리 단위로 분리. 212e3f8[core/tools][config] 문서 청킹 수정 / 4b21c23[core/model][core/orchestrator] DocumentExport guided 재시도 / 3458e10[web][core/tools] 결과표시 정합+scout 폴백+업로드 ASCII / 035dd76[deploy][scripts] 배포 스크립트+API 스모크 테스트 / e4b0c9e[docs] progress / 57c7ec1[test] agent_tool scout 폴백 테스트. web/app.py·worker_system_full.md는 hunk 스테이징(git apply --cached)으로 청킹분/결과표시분 분리. 로컬 48 passed. push 미실행. 남은 uncommitted=로컬설정(.claude/·.omc/)·자산(NHN PDF·이미지)뿐.

**durability(원복 fragility 근본 해소).** 진단: 112 빌드 컨텍스트(/home/idino/nexus-app)가 stale(오늘 청킹 수정 + 수 주간 docker cp분 누락) → Dockerfile 순진 재빌드는 회귀 위험. **안전 방식 채택**: 검증된 실행 컨테이너(e2e 통과 상태)를 `docker commit`으로 이미지化 → 회귀 위험 0으로 정합. 백업(구 이미지 prefix-backup-20260716_082558=54edca) + 구 컨테이너 stop·rename(롤백용) 후 새 이미지로 재생성. **결과**: nexus-web:latest=cc2a0fcd(스냅샷, 모든 수정 포함), 컨테이너가 이미지에서 부팅·restart=unless-stopped → 재생성돼도 수정 유지. 스모크 8케이스 전부 통과(청킹 upload_analyze DocumentProcess=1 포함). **주의**: commit-스냅샷이라 Dockerfile 재빌드 재현성은 별도 과제(빌드 컨텍스트 전수 정합 필요, 비긴급). docker run에 PG 비번 하드코딩은 commit이 env를 이미지에 캡처하므로 -e 불요(재노출 방지).

---

### 코딩 전반 RAG + 학습 데이터 — 설계 단계 (2026-07-17, 계획 확정 전·코드 미착수)

> 이 섹션은 **설계·의사결정 기록**이다. 아직 코드를 쓰지 않았다. cmd 세션 유실 대비로 트레일을 남긴다.

**요청.** 코딩 전반(Stack Overflow·OKKY·GitHub)을 RAG로 구축 + 학습 데이터로 가공. 사용자 결정: 방식=**RAG+학습 둘 다 병행**, 소스=**전부 포함**(OKKY 스크래핑·GitHub ⭐2000+ 라이선스 무관), 착수=**SO 먼저**.

**딥리드로 파악한 통합 씸(실코드 확인).** ①`tb_knowledge`는 스키마 변경 불필요 — `source` 값(so/okky/github)으로 멀티코퍼스 구분(v7.0 §2.5.8·Ch15 사양 명시, v7.3 docingest 선례). ②검색 격리는 `allowed_sources`→DB-level `WHERE source=ANY()`. ③라우팅: KNOWLEDGE만 RAG 주입, `allowed_sources`는 `tenant.allowed_knowledge_sources`가 비면 None(전체)·차면 그 화이트리스트. ④`config/tenants.yaml` default·agenthub 모두 `[kowiki,sample]` 화이트리스트라 **'so' 미추가 시 코딩청크 제외**(자동 아님, 활성화엔 config 한 줄 필요). ⑤학습 포맷=사양 §18.2 `{id,category:"domain",messages,metadata}`(data_collector의 `{messages}`는 부분집합). QLoRA는 B200(trainer.py HTTP).

**사양 이탈 플래그.** 코딩을 **학습(domain finetune)** 으로=사양 정합(v6.1 §18.2 "Domain: coding/DevOps"). 코딩을 **tb_knowledge RAG 코퍼스**로=**사양에 없음(확장)**. 금지 아님(멀티코퍼스 프레임이 source 추가 허용)이나 v7.x AMENDMENT로 문서화 필요.

**FABLE5(fable) 적대 검토 = REVISE(조건부 GO). 코드검증된 결함:**
- **C1(CRITICAL 무언 데이터유실)**: PK=`SHA256(source|title|section|chunk)`인데 SO 제목은 비유일 → 다른 질문이 같은 id로 UPSERT돼 덮어씀. Fix: `section=q{question_id}` 앵커링.
- **C2(CRITICAL kowiki 열화)**: 계획 "build-index"는 no-op(`idx_knowledge_embed` 이미 존재+`IF NOT EXISTS`, 코드 DDL `lists=100` vs 운영 `lists=1000` 드리프트). 재인덱스 안 되면 센트로이드가 옛 kowiki 분포 고정→기존 질의 재현율 열화, DROP→재생성 창엔 seq scan 추락. Fix: `CREATE INDEX CONCURRENTLY`(lists≈rows/1000)→구DROP→RENAME + probes 재캘리 + 전후 kowiki 스모크(광합성·바흐).
- **M1(MAJOR 목적 반쯤 봉쇄)**: `tool_keywords`(파일·file·코드베이스·수정해·Read )·`long_input_threshold=500`으로 대부분 코딩질의가 TOOL→RAG 미주입. "routing.py 무변경" 전제 붕괴.
- **M2**: `split_into_chunks`가 코드블록 파괴+`buf=s[-max_chars:]`로 코드 앞부분 무언 폐기 → 코드 인지 청커 필요.
- **M3**: 리랭커·임계(min_sim0.6/min_score0.3)가 한국어 kowiki 캘리브(리랭커 bge-…-ko). 한국어질의↔영어코드 분포 미검증 → 최악 "적재 성공·주입 0건". 활성화 전 파일럿 실측 필수.
- **M4**: 엔티티게이팅 `\d{2,5}`가 "포트 8080"·"Python 3.12" 숫자로 관련청크 드롭(최악=주입 0건, 무회귀지만 가치감소).
- **M5**: GitHub 무관=상업/특허 리스크로 기각 권고(SO는 CC BY-SA 표시의무, citation.enabled=false).
- **M6**: `add_many` 순차단건+임베딩배치5 → 수백만청크=수일. 공유 :8002·공유 PG 부하 지속. Fix: executemany/COPY+배치상향+야간스로틀.
- 누락: 코딩 eval세트·롤백절차·DB위생(백업/ANALYZE)·SO Posts.xml 속성기반 flat XML+ParentId 2-pass 페어링.

**사용자 결정(FABLE5 후).** ①**계획 전면 재설계**. ②**GitHub 라이선스 무관 유지(리스크 감수)** — provenance로 라이선스 metadata 저장해 나중 분리 가능하게, 상업배포 전 법무검토 권고(책임선).

**재설계 방향(가정 검증 우선 = 파일럿 게이트).** 대규모 적재를 뒤로 미루고 소규모 파일럿으로 M1·M3를 먼저 실측.
- **Phase 0(진단·파일럿, 저비용·무회귀)**: 0a 라우팅 도달률(코딩질의 40~50개 분류 측정), 0b SO 1만청크 파일럿→한국어 코딩질의 20개로 유사도/리랭크 분포 실측, 0c 인프라 실측(실 lists·행수·디스크·임베딩 처리량). → 데이터 기반 RAG GO/조정/보류 판정.
- **Phase 1(공용+SO 적재기)**: coding_corpus.py(코드인지 청커·provenance·언어태깅·중복제거+자체테스트), prepare_stackoverflow.py(속성 flat XML+ParentId 2-pass 페어링, PK q{id} 앵커링, executemany/COPY 벌크·야간스로틀), 재인덱스 절차(C2), 큐레이션 기준 확정(accepted OR score≥N — N 미정).
- **Phase 2(활성화·게이팅)**: 파일럿 임계 반영, 엔티티게이팅 source예외(M4), tenants.yaml 'so' 추가(**default만** 권고, agenthub 별도), citation 활성(CC BY-SA), 코딩 eval 전후비교, 롤백절차.
- **Phase 3(학습 트랙)**: coding_dataset.py(§18.2 포맷), QLoRA B200 별도어댑터+배포 eval게이트(catastrophic forgetting). Skeptic: 라우팅 봉쇄로 RAG 수혜 제한→학습 트랙 실질가치 클 수 있음, Phase0 결과로 우선순위 재조정.
- **Phase 4/5(OKKY·GitHub, 후속)**: OKKY 정중 스크래퍼(ToS 미검토), GitHub ⭐2000+ 무관+provenance(법무검토 권고).

**미해결(사용자/후속).** 큐레이션 N 임계값 / SO 덤프 입수경로·현행약관 / tenants 'so' 범위(default 권고) / 학습 vs RAG 우선순위(Phase0 데이터로).

**FABLE5 v2 재검토 = 조건부 GO(재설계 불필요, 수렴).** 5조건 중 C2 충족·M1 충족(주의), C1·M3·M5 부분. 잔여는 전부 **문안 개정 수준**(구조 유지). 착수 전 반영 4건 → v2.1:
- **N1(CRITICAL 잔여 PK유실)**: `section=q{qid}` 앵커링해도 한 질문에 답변 다수면 같은 id로 덮어씀. **해결책 채택 = 질문+채택답변+고득점답변을 단일 문서로 결합 후 코드인지 청킹**(답변충돌 원천소멸+Q&A맥락 유지). 복수답변 fixture 테스트 필수.
- **N2(MAJOR 파일럿 임계 전이성)**: 1만 청크 유사도 분포 ≠ 수백만 분포. Phase 0b를 **GO/NO-GO 전용**으로 강등, **최종 임계는 전량 적재+재인덱스 후 재실측**으로 이원화(Phase 2 문안).
- **N4(MAJOR citation 전역)**: `knowledge_rag.citation.enabled`는 source 구분 없는 전역 스위치 → 켜면 kowiki 헤더에도 [출처N] + **agenthub(.NET) 응답에 sources 필드 추가**. Phase 2에 전역 회귀확인(agenthub 스키마 호환) 단계 추가. per-source citation은 신규개발 항목.
- **N5(큐레이션 N 미정→규모 추정 불가)**: Phase 0에 **덤프 통계 스캔**(임베딩·DB 없이 파서 dry-run으로 N별 문서/청크 수 집계) 추가해 데이터로 N 결정. 0c 처리량 벤치는 **운영 인덱스 걸린 tb_knowledge INSERT 조건**으로 측정(맨 테이블은 낙관편향).
- **N3(파일럿 격리 누설)**: 임시 source는 tenants 화이트리스트로만 격리 → CLI/비테넌트(tenant=None, routing.py:396-398) 경로엔 노출. 파일럿 문안에 (1)비테넌트 노출 허용여부 명시 (2)정리단계(DELETE WHERE source='so-pilot'+VACUUM/ANALYZE) 추가.
- Minor: 엔티티게이팅 개선은 "결과 청크 source별 게이팅 면제"로 확정(질의분류형 아님, BWV543 kowiki 보호 단위테스트 유지) / 0b 측정은 KnowledgeRetriever 직접호출(라우팅 우회) / CIC 실패 시 INVALID 인덱스 DROP 후 재시도 / 수용기준 수치화(코딩 eval 개선≥X, kowiki 회귀 0).

**상태: 계획 v2.1 확정. Phase 0a 착수·완료(아래).**

**Phase 0a 실측 — 코딩 질의 라우팅 도달률 (2026-07-17).** 운영 112 config(routing 섹션, long_input_threshold=500, tool_keywords 53개)의 HeuristicClassifier로 코딩 질의 22개(4범주 균형) 분류. 스크립트: scratchpad/phase0a_routing_reach.py(로컬 probe, 미커밋).
- **결과: KNOWLEDGE(=RAG 주입) 도달률 17/22=77%.** 범주별: 개념형 8/8·라이브러리 6/6(RAG 수혜 100%), 에러디버깅 3/4("수정해"→TOOL), 파일작업형 0/4(전부 TOOL — 도구작업이라 정상).
- **해석**: FABLE5 M1(대부분 TOOL로 샘) **부분 반증** — RAG 타깃(개념·how-to·라이브러리)은 ~100% 도달하고 이게 SO 콘텐츠 유형과 정합. TOOL행은 대부분 진짜 도구작업(손실 아님). **진짜 누수 2개 확인**: ①long_input_threshold=500(실제 에러질문은 스택트레이스·코드 붙여 500자 초과→강제 TOOL), ②"파일/디렉토리/repository/codebase" 단어(개념질문도 단어로 TOOL).
- **한계(선정편향)**: 22개는 자체 구성 세트. 엄밀화엔 실제 질의로그 재측정 + 긴 붙여넣기 케이스 추가 필요(에러범주 실제 도달률은 더 낮을 것).
- **판정**: RAG 트랙은 개념·how-to 세그먼트에 유효(SO 정합), 폐기 불필요. 라우팅 보강은 "선행 필수"→"수혜 극대화용 후속 옵션"으로 강등. 학습-우선 전환 트리거(도달률 심각히 낮음)엔 해당 안 됨.

**기존 품질 실측(같은 세션, 2026-07-17).** 스모크 8케이스 전부 통과 + 실질질의 8종(광합성·파이썬 리스트뒤집기·1~100합·BWV환각·ML/DL·측우기 등) 응답내용 확인 → **양호**(정확·일관·구조화, degeneration/반복/잘림 없음). 단 전 질의 sources=0 = citation.enabled=false(전역, N4)라 API로는 RAG 실주입 여부 단정 불가. 시사점: 베이스 A.X-4.0가 코딩질문도 RAG 없이 잘 답함(파이썬 질의 sources=0인데 최고품질) → 코딩RAG 한계가치는 Phase0b로 실측 필요(Skeptic 실측 뒷받침).

**다음(덤프 의존).** 0b SO 파일럿 캘리(GO/NO-GO)·0c 덤프 통계+처리량 벤치 → SO 데이터 덤프 입수 후. 0a는 덤프 없이 완료.

**Phase 1 기반 착수 — 코드 인지 청커(M2 수정) 완성 (2026-07-17, 덤프 무관·무회귀).** 사용자 지시로 덤프 대기 중 no-regret 기반부터 구축(RAG·학습 두 트랙 공통, 순수 로직).
- 신규 `scripts/coding_corpus.py`(코딩 코퍼스 공용 정제 모듈): `chunk_code_aware(text, max_chars=1500, overlap=150)`. 불변식 (A)코드블록 원자성 (B)무손실·앞부분보존 (C)초과 코드는 라인경계 분할+재펜스 (D)설명+인접코드 동거 (E)전 청크 max_chars 이하. 세그먼트 분리(_iter_segments)→코드/산문 별도 처리, `_hard_split`이 앞→뒤로 잘라 앞부분 폐기 버그 원천 차단.
- 신규 `tests/unit/test_coding_corpus.py`(8) — 불변식 회귀. **M2 버그 실증 대조**: 동일 긴 코드에 기존 `split_into_chunks`는 `def pipeline():`(함수 시그니처) **유실**(문장부호 없는 코드→한 '문장'→뒤만 남김), 신규 청커는 전부 보존. 8 passed, ruff 클린.
- 커밋 5bde878(청커+테스트).

**coding_corpus html_to_markdown 추가 (2026-07-17, 덤프 무관).** SO Body는 HTML이라 청커가 실데이터를 씹으려면 펜스 마크다운 변환이 선행. stdlib `html.parser`만 사용(에어갭, BeautifulSoup 등 외부의존 없음).
- `_SOHtmlToMarkdown(HTMLParser)` + `html_to_markdown(html)`: <pre><code>→```lang 펜스(class lang-python 감지), 구문강조 <span> 제거+텍스트 보존, convert_charrefs로 &lt;/&gt;/&amp; 복원, 인라인 <code>→백틱, <p>빈줄·<li>"- "·<strong>/<em>→**/*, <a>는 텍스트만.
- 테스트 +8(엔티티 복원·span 제거·언어감지·리스트·청커 연동) → 총 16 passed, ruff 클린. 실제 SO형 Body(span+엔티티+인라인+리스트) e2e 확인.
- 커밋 c0ff736(html_to_markdown+테스트).

**prepare_stackoverflow.py SO 파서 핵심 완성 (2026-07-17, 합성 fixture·덤프 무관).** FABLE5 CRITICAL 수정(C1/N1)을 코드로 확정. 임베딩·DB 쓰기는 가드(실덤프+인프라 시 배선).
- 신규 `scripts/prepare_stackoverflow.py`: `iter_posts`(속성 flat XML 스트리밍, PostTypeId 1질문/2답변, elem.clear 메모리관리) → `load_posts`(질문·답변 버킷팅, ParentId 페어링) → `build_combined_document`(**질문+채택답변+고득점답변을 단일 문서로 결합**, 채택 우선·점수순, html_to_markdown로 코드 보존) → `build_entries`(큐레이션 accepted OR score≥N·require_code·min_len, **section=q{qid} 앵커링 → PK 유일**, provenance metadata url/license CC BY-SA/question_id/answer_ids, tags=SO태그). `parse_tags`(신형<>·구형|| 대응). `run_stats`(N5: 임베딩·DB 없이 규모 집계, --stats CLI).
- 신규 `tests/unit/test_prepare_stackoverflow.py`(8): 파싱·버킷팅·결합순서·큐레이션·**C1/N1 PK유일성 2종**(제목같은 다른질문 id 비충돌 / 복수답변 단일문서 비충돌)·provenance·require_code. 총 24 passed(coding_corpus 16+SO 8), ruff 클린. --stats CLI e2e(kept=2/chunks=2) 확인.
- **가드/후속**: run_ingest(임베딩 :8002+KnowledgeStore UPSERT+재인덱스)는 실덤프+LAN 인프라 필요 → Phase 0b/1 실서버 배선. load_posts 전량 인메모리는 파일럿용, 전량 덤프는 SQLite 스테이징 2-pass 도입 예정.

**남은 coding_corpus 유틸(후속, 선택)**: build_provenance/dedup_key/detect_languages는 현재 파서가 인라인 처리(tags=SO태그, metadata 직접구성)로 충분해 필요 시 추출. 다음 실질 진전은 SO 덤프 입수 → 0b 파일럿·0c 통계·run_ingest 배선.

**★ MUST-DO 요구사항 — 베이스 A.X vs RAG 코딩 답변 품질 (2026-07-17, 사용자 지시로 필수 기록).**
전략 판단(반드시 반영): 코딩 어시스트 답변 품질의 **주력은 베이스 A.X-4.0**(흔한 코딩 지식에 이미 매우 강함 — 실측: 파이썬 리스트뒤집기 sources=0인데 3방법+주의사항 완벽). RAG(SO/OKKY)는 **보완재** — 롱테일/니치·최신성·근거인용, 그리고 **OKKY=한국어(모델 한국어 코딩 깊이 얕은 부분 보완, RAG가 가장 확실히 이기는 지점)**에서 가치. **리스크: 모델이 이미 아는 흔한 질문에 SO 스니펫(영어·구버전·타맥락) 무차별 주입은 답변을 오히려 악화**시킬 수 있음. → 결론: 무차별 주입 금지, **수술적 RAG(리랭커·게이팅 고신뢰만 발동)** + **고품질 소량 > 대량** + OKKY 우선. 이래서 min_question_score(고득점만) 레버가 품질 전략의 핵심 도구.
**MUST-DO 정밀 테스트(Phase 0b 확장, 나중 반드시 수행)**: 실 SO/OKKY 청크 주입 상태에서 **한국어 코딩 질의 세트로 [RAG-주입 답변] vs [베이스-단독 답변] A/B 비교** — 어느 쪽이 나은지, RAG가 해치는 케이스(흔한질문 노이즈) 식별. 이게 SO/OKKY RAG 투자 정당화의 판정 기준. 도달률 0a(77%)와 별개의 "품질" 게이트.

**실 SO 덤프 검증 + 임계 스윕 (2026-07-17).** 사용자가 Posts.xml(96.8GB, 첫row Id=38779 2008년~) 해제. `--limit`(바운드, OOM방지) + `--min-question-score`(질문 품질게이트) 추가·테스트(총 9 passed, ruff clean).
- **실데이터 파서 검증**: 50만 행 24초, OOM 없음 → 질문 113,786 / 답변그룹 112,661 / 큐레이션통과 90,475(79%) / 청크 165,552. 파서가 실 SO XML 정확히 처리.
- **핵심 발견**: min_score(답변점수)는 규모에 거의 무영향(5→50: 90k→81k) — 대부분 질문이 **채택답변(점수무관)으로 통과**하기 때문. **require_code가 큰 필터**(90k→53k, 코드보유 59%). **min_question_score(질문점수)가 진짜 규모 레버**.
- **질문점수 스윕(50만prefix, min_score10+require_code)**: qscore 0→49,561질문/95,718청크, 10→19,050/41,523, 25→11,454/26,971, 50→7,519/18,900.
- **규모 추정(거친, prefix=최고령·고득점 편향이라 과대추정)**: 전량 ~59M행 가정 시 qscore25≈full ~3M청크(실제론 더 적음), qscore0(무게이트)≈~11M청크(과대). → **kowiki규모(~1-3M)로 조이려면 min_question_score 10~50 + require_code** 권장. 정확 총계는 SQLite 스테이징 전량 스캔으로.
- 커밋: d65bd1a(--limit·--min-question-score), c9c0677(run_ingest 배선).

### 코딩 RAG Phase 0b 파일럿 실행 — 교차언어(M3) 실측 (2026-07-17, 자율진행·예비)

> 사용자 자리비움 중 auto 진행. 인프라 투자(리랭커 배포)·SO 진행중단·OKKY착수 같은 전략결정은 데이터만 모아 옵션으로 남김(미결정). FABLE5 추천 동의여부 검토 중.

**파일럿 적재.** `scripts/prepare_stackoverflow.py run_ingest`로 SO 청크를 112 tb_knowledge에 **source='so-pilot'**(격리, 어떤 테넌트 allowed_knowledge_sources에도 없어 기존 사용자 무영향, 롤백=DELETE WHERE source='so-pilot') 적재. 파라미터 --limit 500000 --min-question-score 25 --min-score 10 --require-code(=~27k청크 목표). 소량검증(341행) 전부 임베딩·provenance(section=q{id}·SO태그·url·score·license·answer_ids 다중답변결합) 정상 확인. 드라이버 scratchpad/run_so_pilot.py(.env 비번 프로세스내부, 미노출).

**🔴 M3 핵심 발견(계획 변경급).** ①**112 임베딩 :8002에 /v1/rerank가 404 = 리랭커 없음** → 현재 112 RAG(kowiki 포함)는 리랭커 없이 raw e5 2단 게이팅으로만 운영 중. ②한국어 코딩질의 10개로 so-pilot 검색: 벡터 min_sim0.6 게이트 10/10 통과하나 **유사도가 전부 0.82~0.84에 뭉침**(e5 무관문서 노이즈바닥 0.78~0.83) = 통과가 관련성 아닌 노이즈. top-1이 토픽상 자주 틀림(배열순회→sort, 문자열숫자→format ull, 파일쓰기→C++ open). **즉 한국어질의→영어SO를 e5 직접검색하면 토픽 틀린 청크 주입→답변 악화**(MUST-DO 리스크 실측 확인).
**질의번역 진단(2순위).** 같은 질의 KO vs EN 검색비교(부분데이터): 번역이 의미매칭 개선 — 정규식·재귀·배열순회에서 KO 완전오답이 EN은 정답. 단 EN도 불완전+유사도 여전히 뭉침 → **번역은 랭킹 개선하나 클러스터링 못 풀어 리랭커 여전히 필요**.

**추천(사용자 결정 대기, FABLE5 검토 중).** 1순위 **리랭커 112 배포**(bge-reranker-v2-m3-ko 선정완료 — 교차언어 e5 불신뢰 해결 유일 정밀게이트, SO뿐아니라 OKKY·kowiki 전체 정밀도↑, 현재 리랭커없이 운영). 2순위 **SO 질의번역 검증**(저비용). 3순위(전략) **OKKY(한국어) SO보다 우선**(동일언어 교차언어문제 없음). 총평: 베이스 A.X가 흔한 코딩에 이미 강함(실측) → **SO-영어 RAG 과투자 금물, RAG는 롱테일·근거·한국어(OKKY)에 수술적으로만**.
**남은 것.** 전체 적재(~27k) 완료 후 두 진단(교차언어분포·번역) 전체커버리지 재측정 → 확정. MUST-DO A/B(RAG주입 vs 베이스단독)는 리랭커·번역 경로 정해진 뒤. **중요결정(사용자몫)**: 리랭커 배포 승인 / SO 진행·중단 / OKKY 우선전환.

### Phase 0b 교정 + kowiki 회귀 발견 (2026-07-17, FABLE5 재검토 반영)

**FABLE5가 내 실측 해석 오류 1건 교정.** 나는 min_sim 0.6/0.0로 쟀는데 그건 **rerank 경로 값**이고, 112는 rerank OFF라 **폴백 2단 게이팅(config/nexus_config.112.yaml: min_similarity 0.75·abs_threshold 0.84·relevance_margin 0.03)**이 실제 작동. 실제 게이트로 재측정:
- **so-pilot(교정): 한국어 코딩질의 4/10 주입, 6/10 빈주입(abs 0.84 드롭).** 주입 4개 중 2개만 정확(C# Decimal→Double 0.870·SQL중복제거 0.853, 제목이 질의와 유사할 때). "10/10 노이즈"는 오답 — abs 0.84 게이트가 교차언어 노이즈 대부분 방어. **SO 직접검색 실체=노이즈주입이 아니라 "대부분 빈손+소수유용", 즉 저가치**(결론 유지, 위험 하향).
- 유사도 전부 0.82~0.87 뭉침(e5 교차언어 한계). 소스 커질수록 노이즈상한이 0.84 위로 밀려 방어 침식(v2 N2).

**🔴 kowiki 회귀 확정(FABLE5 최우선 지목, SO와 무관한 기존 결함).** 106만 청크, 실제 게이트로: **"광합성의 원리" top=0.839<0.84 → 빈주입 / "바흐" top=0.827<0.84 → 빈주입** (측우기 0.862·세종 0.849만 주입, 2/4). config 주석(:363)="리랭커로 **광합성 오답차단·바흐 교정** 검증" → 검증은 리랭커 ON에서 받았는데 112는 rerank OFF → abs 0.84 폴백이 0.839·0.827 드롭 → **검증됐던 대표 2케이스가 지금 빈주입**. 앞서 광합성 답변 좋았던 건 베이스모델 지식(sources=0), RAG 그라운딩 안 됨. **롱테일 kowiki 지식(모델이 모르는)엔 이 회귀가 답변 악화로 직결.**

**FABLE5 추가 교정.** ①리랭커=신규투자 아니라 **원상복구**(검증 baseline이 리랭커 포함). ②리랭커는 정밀도만, **재현율(정답이 top-20에 드는지)은 번역이 풀음 → 1·2순위 보완재**. ③문서측 번역(적재시 1회, 런타임비용0, 에어갭정합) 대안 있음 — 질의측(런타임반복)만 제시한 건 미탐색. ④bge-reranker-v2-m3-ko의 ko질의↔en문서 성능·min_score0.3 캘리 미검증(배포후 재실측 필요). ⑤단, **112 config는 이미 rerank.enabled:false**라 FABLE5가 우려한 "매질의 404 실패호출 스팸"은 없음(폴백 클린) — base config만 enabled:true.

**⛔ 사용자 결정 대기(자율 진행 불가 — FABLE5 지목).** ①**리랭커 112 복구**(bge-reranker-v2-m3-ko): 인프라+전 테넌트 게이팅 동작변경이나 **kowiki 회귀까지 고치는 원상복구**(강한 근거). ②**SO→OKKY 우선순위 변경**: 원 지시 "SO 먼저" 명시적 번복이라 에이전트 재량 밖. ③OKKY 스크래핑 개시(ToS/법적). ④so-pilot 정리(DELETE WHERE source='so-pilot'+VACUUM — 현재 tenant미바인딩 경로 노출, 측정 끝나면 정리).
**자율로 안 한 것**: 리랭커 배포/enabled 변경 안 함, SO중단·OKKY전환 안 함, so-pilot 유지(측정용). 전부 읽기전용 측정만 수행.

### OKKY 소스 검토 + 결정 (2026-07-17)

**법적 검토(사용자와 정밀 논의).** ①robots.txt: 개별 /questions/{id}·/articles/{id} 크롤 가능, /api/·/users/*/questions·/auth/ 차단. **AI 크롤러 명시(GPTBot·ClaudeBot·Claude-User·PerplexityBot·OAI-SearchBot)** = OKKY가 봇 접근 관리 중. ②콘텐츠=사용자생성물 → 저작권 작성자 귀속(한국 저작권법·베른협약, **표시 없어도 창작 시 자동 발생** — "표시 없음=자유이용"은 착각). ③**재사용 라이선스 없음**(SO의 CC BY-SA와 대조 — 침묵=허락 없음=더 제한적). ToS 표준=게시물 저작권 회원 귀속+회사에 운영용 비독점 라이선스만. ④핵심: 침해는 **복사·저장(복제)** 시점(citation on/off·출처표시와 무관). RAG는 원문 저장·재생산이라 학습(추상화)보다 오히려 복제에 가까워 법적으로 더 까다로움. "대부분 LLM이 학습"=다퉈지는 관행+점점 라이선스화(SE 2024 Google/OpenAI 계약), 합법 증명 아님.

**사용자 결정(정보 근거·리스크 수용).** 코딩 모드는 **팀 내부 전용** → 실무 리스크 낮음. **상용 전환 시 provenance(source='okky')로 `DELETE WHERE source='okky'`+재인덱스로 제거**(처음부터 넣은 분리 설계). 즉 OKKY 진행하되 내부한정·제거가능·문서화. SO(CC BY-SA)는 라이선스 명확이라 별도.

**스크래퍼 코어 완성·검증(2026-07-17).** OKKY=Next.js App Router(RSC) 앱 — 콘텐츠가 일반 HTML 아닌 `self.__next_f` RSC 페이로드(정형 JSON: title·text(HTML)·tags·selectedAnswerId·answers.content[]{id,text,voteCount,selected}). /api는 robots 차단이라 **페이지 RSC 파싱이 정합**. 신규 `scripts/prepare_okky.py`: rsc_payload·parse_question·build_combined_document(질문+채택+고득점 결합, SO 대칭)·build_entries(source='okky', section=q{qid} PK, provenance url/license/answer_ids)·fetch_question(robots 준수 rate-limit 1.2초, qid=URL 권위값)·--probe. `coding_corpus`(html_to_markdown·청커) 재사용. 테스트 `test_prepare_okky.py`(5, 합성 RSC)·실페이지 probe(381655 채택답변 결합·1419849 tags추출) 검증. 5 passed, ruff clean.
- **관찰**: OKKY는 투표 희박(voteCount 0 흔함) → min_answer_votes 낮게(0) or 채택 위주 권장.
- 커밋 5bae3ca(스크래퍼 코어).

**OKKY 소규모 파일럿 완료·검증 (2026-07-17).** 열거·적재 배선 추가: `iter_listing_ids`(sitemap은 상위nav만이라 무용 → **/questions/tech 목록 RSC에서 "id":N,"title": 열거**, 20개/페이지, 1페이지는 파라미터없는 URL, 총 258,923건), `run_ingest`(열거→스크랩(rate-limit 1.2초)→build_entries→임베딩:8002→UPSERT source, prepare_stackoverflow 대칭), CLI(--board/--pages/--limit/--source/--dry-run/--pg/--embed-url). dry-run(15건→16엔트리) 검증. **실 파일럿: tech Q&A 100건 스크랩→stored=119청크(source='okky-pilot')**.
- **리랭커 검증(한국어 네이티브 매칭)**: 4/8 주입, 정확매칭 rr 0.996~1.000(자바내부클래스1.0·eclipse0.997·아키텍처0.996·이펙티브자바0.813). 드롭 4개는 100건 파일럿에 해당 토픽 부재(rr 0.0 정확 드롭, 노이즈 아님). **"OKKY(한국어)>SO(영어) for 한국사용자" 실측 확인** — 교차언어 브릿지 불요, e5 네이티브 매칭 깨끗.
- 커밋 251337b(열거·적재), 0de681a(ruff).

**OKKY 본격 적재 + 코딩 테넌트 활성화 = 완료 (2026-07-18).** run_ingest 증분저장 리팩토링(질문 20개마다 스크랩→임베딩→저장, 장시간 실패 시 진행분 보존). **본격 스크랩: tech Q&A 3000건 → okky 1,834청크**(최신질문 미답변 다수 제외). 리랭커 검증 **5/8**(파일럿 4/8→개선, react상태관리 rr=0.958 커버, 정확매칭 0.958~1.000, 경계선 스프링0.227·PDF0.241). OKKY 한국어 네이티브라 SO보다 깨끗.
**전용 코딩 테넌트 신설(사용자 결정 — default 혼합보다 정확).** `config/tenants.yaml`에 `coding`(kowiki·sample·okky·so-pilot, key nexus-coding-key-001, internal-only) 추가. **전용↔범용 전환은 목록 수정+재시작만(재적재 불요)**. 배포: docker cp(백업 .bak.pre-coding)+nexus-web 재시작. **e2e: coding 테넌트 "자바 내부 클래스"→정답(okky접근)·"수도"→서울(kowiki접근), default "자바"→base only(okky 격리 확인)**. citation off라 sources 미표시나 주입은 리랭커경로 실측 확인. 이중방어(테넌트 allowed_sources DB필터 + 리랭커 관련도).
**🔴 OKKY 403 차단 — 확대 중단(2026-07-18).** 3000건+ 스크랩 후 OKKY가 **status=403으로 스크래핑 차단**(1.2초 rate-limit 준수했음에도 봇보호 발동). robots의 AI크롤러 명시 + 403 = OKKY가 대량수집을 원치 않음이 실증. **봇차단 우회는 윤리·법적으로 부적절**이라 OKKY 스크래핑 중단. 증분·이어받기 코드(skip-existing·start_page, 커밋 0e3f78a)는 넣었으나 차단으로 실사용 보류. **OKKY는 현 1,834청크로 고정** — 더 필요 시 eBrain 정식 허가/API가 유일 깨끗 경로. **확대는 SO(CC BY-SA 라이선스·차단없음·덤프 이미 보유)로 전환 권장**: so-pilot(2008년대 500k행 샘플)을 더 넓은 SO 적재(더 많은 행/최신, SQLite 스테이징)로 확대.

**상태: 코딩 RAG(OKKY 한국어 1834+SO 영어 26971) 코딩모드로 LIVE.** OKKY 확대는 403으로 차단·중단.

**SO SQLite 스테이징 대량적재 + 중요 발견(2026-07-18).** iter_posts에 연도 추가, stage_to_sqlite(Pass1 스트리밍 필터·kept-parent 답변만)+run_staged_ingest(Pass2 배치 build→임베딩), CLI --stage-ingest/--min-year 등(커밋 b319a9d). **1차 시도(min-year 2020·min-qscore 20·require_code) → so 31,597청크.** 리랭커 검증 현대 코딩질의(async/pandas/docker/타입힌트) **so 2/8·so-pilot 1/8 둘 다 약함.**
**🔑 근본원인**: **정석 현대 how-to는 그 기술이 나온 ~2012~2019년대에 물어봤는데, so-pilot(2008)은 현대기술 이전이라 없고 so(2020+)는 그 구간을 잘라냄 + 고득점+코드가 니치버그를 골라냄**. "너무 예전 제외"(min-year 2020)가 과했음 — SO는 연도보다 **점수(고득점=정석·evergreen)**로 걸러야. OKKY(5/8)·리랭커는 정상, SO 필터만 문제.
**재적재 완료: min-year 2012·min-qscore 50·require_code → so 255,769청크(옛 31k의 8배).** 리랭커 재검증 2/8→3/8, 정석 매칭 회복(async rr0.922·pandas 0.904·타입힌트 0.959). 남은 미스(JS프로미스·SQL윈도우·이메일)=corpus 아닌 **recall**(벡터 top-20에 정석이 안 올라옴, FABLE5 M3) → fetch_k↑·질의번역 튜닝 후속.
**마무리(2026-07-19).** 옛 so-pilot(26971 2008편향)·okky-pilot(119 orphan) DELETE+ANALYZE. coding 테넌트 so-pilot→**so** 교체 배포(docker cp·재시작). e2e: coding "asyncio"·"판다스 병합" 정답. **최종: so 255769+okky 1834+kowiki 1067975.**
**후속(비차단)**: recall 튜닝(rerank_fetch_k 20→40·질의번역) / OKKY eBrain 허가 시 확대 / citation·관측성 / min_score 경계선. 후속(비차단): okky-pilot(119) 정리(orphan, DELETE) / 커버리지 위해 OKKY 추가 스크랩 / SO so-pilot→so 정규화(선택) / min_score 튜닝(경계선). 상용 전환 시 coding 테넌트에서 okky 제거 + DELETE source LIKE 'okky%'.

### 코딩 학습 데이터 계획 — 파킹(향후 코딩 전용 대비) (2026-07-17)

> 사용자 결정: 지금 학습은 안 함(RAG 우선). 단 **데이터·계획은 보존** — 범용 A.X를 이후 코딩 전용으로 쓸 일이 생길 것으로 예상. 소스별 고려를 미리 해둔다.

**핵심 — 데이터 손실 없음(공유 상류).** RAG 적재용 정제 코퍼스(`prepare_stackoverflow.build_entries` 등 = 질문+채택답변 결합·코드인지 청킹·provenance)가 **학습의 상류이기도** 하다. 같은 큐레이션 데이터 → 두 싱크(RAG 임베딩 / 학습 JSONL). 즉 지금 RAG 작업이 학습 데이터도 축적한다. 코딩 전용 필요 시 얇은 변환기(`coding_dataset.py`, 미착수)만 붙이면 착수.

**트리거(언제 학습을 꺼내나).** ①범용 A.X → 코딩 전용 모델 필요 시. ②"지식은 RAG로 줘도 코딩 추론·실력이 약함" 병목 관찰 시(RAG는 사실만 주고 추론은 못 키움 — 그건 학습/베이스교체만 레버).

**포맷(사양 §18.2).** `{id, category:"domain", messages:[{role:user, content:질문},{role:assistant, content:채택답변}], metadata:{source,url,license,score,difficulty}}`. 변환기가 정제 코퍼스를 이 형태로.

**소스별 고려(각각).**
- **SO(영어)**: 코딩 추론/패턴 학습엔 **코드 위주**가 유효(영어 산문은 한국어 답변 학습엔 마찰). 라이선스 CC BY-SA → 학습 시 share-alike 유의.
- **OKKY(한국어)**: **한국어 코딩 스타일·설명 학습에 최적**(A.X 약점 정조준). ToS·저작권 유의.
- **GitHub(코드)**: 코드 자체 → 코딩 능력 직접 학습. 라이선스(permissive vs 무관) 유의.

**주의.** catastrophic forgetting(별도 어댑터 + 배포 전 eval 게이트로 비코딩 회귀 확인) / QLoRA on B200(trainer.py HTTP, 수만~수십만 고품질 instruction이면 충분, 양보다 큐레이션) / 세그먼트별 베이스([[project_segment_model_strategy]] 대학=Gemma4·삼진어묵=ExaOne·일반=Qwen).

**현 상태.** 미착수·파킹. RAG 파이프라인이 데이터를 축적 중이므로 착수 시점에 변환기만 추가.

### 리랭커 복구 배포 완료 + kowiki 회귀 수리 확인 (2026-07-17)

**배포(사용자 승인, CPU).** 112 systemd `nexus-embedding`을 리랭커 서버로 교체 완료. ①백업(embedding_server.py.bak.pre-rerank + nexus-embedding.service.bak) ②bge-reranker-v2-m3-ko 다운로드+CPU 로드(31초)·테스트(ko-en 0.997/0.0) ③수정본 embed_server.py 업로드(fp16→fp32 조건부 + **EMBED_HOST env**) ④unit에 env(EMBED_MODEL=로컬e5·EMBED_DEVICE=cpu·RERANK_DEVICE=cpu·RERANK_ENABLED=1·**EMBED_HOST=0.0.0.0**) ⑤재시작. **함정 1건 자초·즉수리**: 리포 embed_server.py가 127.0.0.1 바인딩이라 LAN 끊김(웹은 192.168.21.112:8002 호출) → EMBED_HOST env로 0.0.0.0 재배포 복구. LIVE: /health reranker=dragonkue/bge-reranker-v2-m3-ko·device=cpu, /v1/rerank 200[0.9975,0.0002], /v1/embed 200.
- 하네스 auto 분류기가 sudo 프로덕션 restart를 차단 → 사용자 모드 변경 후 실행됨.
- 커밋 안 됨(embed_server.py 로컬 수정 = fp16/fp32 + EMBED_HOST). scratchpad에 배포 스크립트.

**리랭커 경로 실측 = kowiki 회귀 수리 확인.** search(min_sim0.6·top20)→/v1/rerank→min_score0.3 재현: **kowiki 4/4 주입**(광합성 rr=0.989→"광합성"·바흐 1.000→"요한 제바스티안 바흐"[맞는 인물]·측우기 0.999·세종 0.999) = FABLE5 1차 수용기준 충족. **so-pilot 6/10 주입, 깨끗한 분리**(정답 고신뢰 주입 Decimal변환1.0·정렬0.997·SQL0.906·문자열숫자0.983·파일0.860 / 무관 rr≈0.0 배제 — e5가 통과시키던 "Xcode" 노이즈 차단). 교차언어 동전던지기 해소. combine-first 성립 증명.

**웹 활성화 완료 + M7.** nexus_config.112.yaml(bind-mount) rerank.enabled false→true(백업 .bak.pre-rerank-enable) + nexus-web 재시작 → health 200, 광합성·바흐 정답(sources=0은 citation off라 API 미표시, 주입은 리랭커경로 실측으로 확인). **전 테넌트 RAG가 폴백→리랭커로 전환됨(LIVE).** M7 지연 실측: **CPU rerank(20후보) p50≈1948ms(~2초)**, embed 255ms. 지식질의(~11초)에 ~18% 증가 — 수용가능하나 무시못함. 필요시 rerank_fetch_k 20→10 반감(재현율 손실).

**리랭커 복구 = 완료.** kowiki 회귀 수리(광합성·바흐 회복), 교차언어 정밀분리(combine-first 성립), 웹 전환까지. **후속(비차단)**: ①embed_server.py 로컬수정(fp16/fp32+EMBED_HOST) 커밋 ②so-pilot 정리 or SO트랙 계속(사용자 결정) ③citation 활성(agenthub 스키마 확인 후)·관측성 로깅 ④엔티티게이팅 \d 코딩영향(정규식 등 여전히 게이트) ⑤라우팅 도달률(M9) ⑥SO↔OKKY 순서(사용자). ⑦rerank 지연 튜닝(선택).

### 리랭커 복구 준비(리뷰용) — 진단 + 계획 (2026-07-17, 배포 미실행)

**진단(112 읽기전용 SSH).** ①112:8002 = systemd `nexus-embedding` → `/opt/nexus-gpu/embedding_server.py`(**구버전, 리랭커 라우트 없음, e5를 `device="cpu"`로 로드**, 4주 가동). 라우트=/v1/embed·/health만. ②리랭커 코드는 리포 `scripts/embed_server.py`에 완비(커밋 1b091e0)이고 112에도 동기화됨(`/home/idino/nexus-app/scripts/embed_server.py`, rerank 5곳)이나 **실행본이 아님**. ③**bge-reranker-v2-m3-ko 미다운로드**(HF 캐시 비어있음). e5는 로컬(/opt/nexus-gpu/models/e5-large) 있음. ④**GPU 5090 29/32.6GB 사용 — VLLM::EngineCore(112 로컬 vLLM 상주)**. 여유 3.5GB뿐(그래서 e5가 CPU). llama-server(8003, Qwen3.5-4B)도 상주. ⑤서빙 venv(/opt/nexus-gpu/.venv)에 CrossEncoder 있음(sentence_transformers 5.4.0).

**결정: 리랭커 CPU 배치(사용자 확정).** GPU는 vLLM 점유라 CPU가 안전(현 e5처럼, OOM 위험 0). 지연은 M7로 실측.

**코드 조정(리뷰용, 미커밋·미배포).** `scripts/embed_server.py`: `RERANK_DEVICE` env 신설(기본=EMBED_DEVICE) + **CPU면 fp32/CUDA면 fp16 조건부**(기존 fp16 하드코딩은 CPU에서 오류·저속). py_compile OK, ruff clean.

**배포 절차(승인 후).** ①bge-reranker-v2-m3-ko 112 다운로드(~600MB, 준비단계) ②systemd `nexus-embedding.service` ExecStart를 리랭커 서버(리포 embed_server.py)로 교체 + env: EMBED_MODEL=/opt/nexus-gpu/models/e5-large·EMBED_DEVICE=cpu·RERANK_DEVICE=cpu·RERANK_ENABLED=1 ③`systemctl daemon-reload && restart` ④`GET /health`에 reranker 필드 + `POST /v1/rerank` 200 확인 ⑤`nexus_config.112.yaml` rerank.enabled:true(bind-mount) ⑥**kowiki e2e: 광합성·바흐 주입 회복 = 1차 수용기준** + 메타질문·BWV543 오답차단 유지 ⑦M7 지연·M2 ko-en 분리도 실측. **롤백**: ExecStart 원복(구 embedding_server.py) + rerank.enabled:false(코드무변경, fail-safe 검증됨).

**리스크.** CPU 리랭커 지연(M7 미측정) / bge-reranker-v2-m3-ko의 ko질의↔en문서 효능 미검증(M2) / 구 embedding_server.py는 e5 로컬경로·CPU라 리포본 배포 시 EMBED_MODEL·DEVICE env로 동일 동작 보장 필요.

### FABLE5 최대-rigor 최종 청사진 + M0 드리프트 규명 (2026-07-17)

**M0 임베딩 드리프트 = 없음(cos=1.00000).** 저장 kowiki 벡터 vs 원문 재임베딩 코사인 5샘플 전부 1.00000 → 112 이관 e5 드리프트 없음. **전 임계(0.84·0.75·0.6)·SO 파일럿 수치 좌표계 유효**. FABLE5 최우선 우려(0-1 드리프트) 해소. → 좌표계 유효한데 바흐 0.827<무관상한0.83이면 **코사인 단일임계로 kowiki조차 분리 불가 = 리랭커 유일경로 확정**(0-2 논증 활성). 단 0.857(주석)→0.827(실측) 차이는 드리프트 아닌 질의문구 차(내 "요한 제바스티안 바흐"가 "요한 크리스티안 바흐"=다른 인물 매칭).

**FABLE5 청사진 핵심.** 의사결정: **D1 리랭커 복구는 지금 결정 가능**(kowiki 검증구성 복원=독립근거, SO와 무관), D2 SO트랙은 측정 완료까지 불가, D4 SO↔OKKY 순서는 에이전트 재량 밖. 측정 의존순서: **M9 라우팅도달률(3라운드째 공백, 전 트랙 게이트)→M0(완료)→M1 B200리랭커잔존(터널로 미도달 확인)→M2 ko-en분리도→M4 recall@20→M5 빈손A/B→M6 엔티티게이팅→M8 전량재캘리(적재 후만)**. 구현순서: Phase A(리랭커 전 — A1 관측성[최우선, 게이팅경로 로깅으로 404류 무언강등 재발방지]·A3 측정스크립트·A4 엔티티게이팅 source별 온오프 v2M4·A5 so-pilot 처리) → Phase B(리랭커 배포[승인]→kowiki e2e 회복=1차수용기준) → Phase C(SO트랙 v2 Phase1) → Phase D(eval).
**"완벽한 구현" 수용기준**: kowiki무회귀(광합성·바흐 회복∧메타·BWV543 오답차단유지∧지연상한) + 코딩eval(실로그≥30, 명문화 판정룰) + 빈손경로무해(M5) + 교차언어 실측문서화 + 관측성 + 운영위생.
**승인경계**: 필수승인=리랭커배포·SO전량적재·tenants활성화·SO↔OKKY순서·OKKY스크래핑·학습배포·so-pilot삭제·커밋. 자율=읽기전용측정 전부+Phase A 코드작성·로컬테스트(운영반영 전까지).
**청사진 리스크**: M9 3라운드 공백(도달률 낮으면 후반부 가치 붕괴), ko-en 리랭커효능 M2까지 가설, 파일럿→전량 전이성(M8 방어이나 비용비대칭), SO덤프 약관 미확인.
**FABLE5 지적 미검증승격 4건**: 0-1 드리프트(M0로 해소), 0-2 관련<무관 겹침(M0 후 논증가능), 0-3 SO수치 라운드간 불일치(측정조건 명시필요), 0-4 "~5유용" 판정룰 부재.

**최종 교정(전체 적재 26,971청크 완료 후).** 부분데이터 4/10이 **전체 커버리지에선 7/10 주입**으로 상향(커버리지↑ → 0.84 넘는 매칭 증가). 주입 7개 중 ~5개 정확·유용(C#변환·SQL중복제거·문자열→숫자 TryParse·null체크·팩토리얼인접), 1개 명백노이즈(정규식→"Xcode 숨은기능" 0.841 경계통과), 1개 부분(파일쓰기→C++ open). 반대로 정답이 드롭된 경우도(리스트정렬→"sort a list" 0.838<0.84). **결론 정밀화: SO RAG는 무가치 아니라 "강한 매칭엔 유효, 0.84 경계선에선 양방향 불안정(노이즈통과+정답드롭)"** — e5 교차언어 유사도가 0.83~0.87에 뭉쳐 0.84 게이트가 동전던지기. → 리랭커(노이즈배제 정밀도)+번역/재캘리(정답회수 재현율)로 견실화 가능. 이는 리랭커·번역이 보완재라는 FABLE5 지적과 정합. **파일럿 적재 EXIT 0, 26971청크 상주(so-pilot).**

**API 스모크 테스트 셋(신규).** scripts/api_smoke_test.py — URL http://192.168.21.112:8600, Bearer nexus-b200-test-key-001. health·auth차단·인사·지식·도구·문서생성+다운로드·업로드분석 8케이스. `python -m scripts.api_smoke_test [--only ...]`.
