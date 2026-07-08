<!-- 지식 RAG 출처 인용(source citation) 기능 상세 설계 문서 — 할루시네이션 저감·검증가능성 향상 (Point 4-2) -->

# 지식 RAG 출처 인용(Source Citation) 설계

- 작성일: 2026-07-08
- 상태: 설계(구현 전)
- 대상 코드베이스: `D:\workspace\nexus-b200` (branch: feature/b200-bakeoff)
- 관련 사양: v7.1 Part 2.5.8 (Knowledge RAG), v7.3 DOC_INGEST Part(메타데이터 기반 출처 표시 예고)

---

## 1. 개요·동기 / 현재 한계

### 1.1 목표

KNOWLEDGE 질의에서 지식 RAG가 답변에 사용한 kowiki(및 향후 docingest) 청크의
출처(제목/source/섹션)를 (a) 모델 답변 본문에 `[출처N]` 형식으로 표기하게 하고,
(b) 웹 응답에 서버가 검증한 구조화 `sources` 필드로 함께 노출한다.

목적은 두 가지다.

1. **검증가능성**: 사용자가 "이 문장이 어느 문서에서 왔는지"를 즉시 확인할 수 있다.
2. **할루시네이션 억제**: "모든 사실 서술은 출처 라벨을 달아야 한다"는 제약 자체가
   모델이 근거 없는 문장을 쓰는 것을 어렵게 만든다(citation-forcing 효과).
   기존 게이팅(★1 grounding + ★2 자료없음 마커 + 리랭커 min_score)은 "무관 청크가
   주입되는 것"을 막지만, "주입된 청크 밖의 내용을 지어내는 것"은 프롬프트 지시에만
   의존한다. 인용 의무는 이 마지막 구멍을 좁힌다.

### 1.2 현재 구조 (실제 코드 기준)

현재 주입 텍스트에 출처 정보가 "들어가긴 하지만, 인용에 쓸 수 없는 형태"다.

**(1) 청크 조립 — `core/rag/knowledge_retriever.py` `get_context()` (라인 405~441)**

```python
header = f"[{source} · {title}{section_part} · {score_part}]"
block = f"{header}\n{content}"
...
return "\n\n".join(lines)
```

즉 각 청크 앞에 `[kowiki · 요한 제바스티안 바흐 · 생애 · rr=0.95]` 같은 헤더가
이미 붙는다. 그러나:

- **번호 라벨이 없다** — 모델이 "두 번째 청크"를 참조할 안정적인 키가 없어
  본문 인용이 불가능하다(제목을 통째로 복사하는 것은 장황하고 불안정).
- **인용 지시가 없다** — `_attach_knowledge_base()`의 trailer(★1 grounding)와
  `web/prompts/worker_system*.md`의 Grounding 절 모두 "근거 없으면 단정하지 말라"
  까지만 지시하고, "사용한 근거를 표기하라"는 지시는 없다.
- **반환 타입이 평문 str** — `get_context()`가 조립된 문자열 하나만 반환하므로,
  상위 계층(prompt_assembler → QueryEngine → web)은 "어떤 청크가 주입됐는지"를
  구조화된 형태로 알 수 없다. 웹 UI에 출처 목록을 노출할 방법이 없다.

**(2) 프롬프트 주입 — `core/orchestrator/prompt_assembler.py` `_attach_knowledge_base()` (라인 230~305)**

```python
return (
    prompt
    + "\n\n--- Knowledge base ---\n"
    + kb_ctx
    + "\n--- End of knowledge base ---\n"
    + "Use the information above ONLY when it is clearly relevant to the ..."
)
```

KNOWLEDGE 질의(`decision.inject_knowledge_rag == True`, `core/orchestrator/routing.py`
라인 262~264)에서만 주입되며, trailer는 grounding 지시만 담는다.

**(3) 웹 응답 — `web/app.py` `ChatResponse` (라인 598~616)**

`session_id / response / tool_calls / usage / downloads` 필드만 있고 출처 필드가 없다.
참고로 `downloads`는 "모델 텍스트가 아니라 서버가 tool_result에서 추출한 정확한
URL"이라는 서버측 추출 패턴(`_collect_downloads`, 라인 136)을 이미 확립해 두었다 —
출처 노출도 같은 원칙(서버가 아는 진실을 별도 필드로)을 따르는 것이 자연스럽다.

**(4) 시스템 프롬프트 디렉토리 현황**

사양서상 `core/system_prompt/`(builder + Jinja2 templates)가 지시문 위치지만, 실제
저장소에는 `core/system_prompt/__init__.py`만 존재한다. 현재 grounding 지시의 실
소스는 (a) `prompt_assembler.py` 내 inline trailer 문자열, (b) `web/prompts/worker_system.md`
·`worker_system_full.md`(웹 워커 base prompt, `_load_worker_system_prompt()` 라인
292~324가 티어별 로드)다. 따라서 인용 지시도 이 두 곳에 넣는 것이 현행 구조와 정합적이다.

---

## 2. 청크 → 출처 메타데이터 매핑

### 2.1 사용 가능한 컬럼 (실측)

`core/rag/knowledge_store.py` `search_by_vector()`의 SELECT(라인 298~305)와 반환
dict(라인 320~336):

| 반환 키 | tb_knowledge 컬럼 | 인용 사용 | 비고 |
|---|---|---|---|
| `source` | source | ✅ 출처 계열 (`kowiki`, `docingest` 등) | 테넌트 필터 키와 동일 |
| `title` | title | ✅ 문서 제목 — 인용 표기의 핵심 | |
| `section` | section | ✅ 섹션명(nullable) | |
| `content` | content | ✗ (본문 자체) | |
| `similarity` | 1−distance 계산값 | ✅ 신뢰도 표시(보조) | |
| `rerank_score` | (retriever가 부착) | ✅ 신뢰도 표시(우선) | rerank 활성 시에만 존재 |
| `tags` | tags | ✗ | |
| `metadata` | metadata(jsonb) | △ 확장용 | v7.3 docingest는 `heading_path`/`page`/`doc_format`을 여기에 담음 |
| (없음) | **id** | △ 추적용 — SELECT에 추가 권장 | 아래 2.2 |

### 2.2 Citation 데이터 모델 (Pydantic v2, 신규)

`core/rag/knowledge_retriever.py`에 정의(도메인 규칙 P5 준수 — frozen 불변):

```python
class KnowledgeCitation(BaseModel):
    """주입된 지식 청크 1건의 출처 메타데이터 — 답변 인용·UI 노출용."""
    model_config = {"frozen": True}

    index: int                     # 1부터 시작하는 출처 번호 ([출처N]의 N)
    source: str                    # 'kowiki', 'docingest' 등
    title: str                     # 문서 제목
    section: str | None = None     # 섹션명(없으면 None)
    score: float = 0.0             # rerank_score 우선, 없으면 similarity
    chunk_id: str | None = None    # tb_knowledge.id (감사·추적용, 선택)


class KnowledgeContext(BaseModel):
    """get_context_with_citations()의 반환 — 주입 텍스트 + 출처 목록."""
    model_config = {"frozen": True}

    text: str = ""                            # 프롬프트 주입용 조립 문자열
    citations: tuple[KnowledgeCitation, ...] = ()   # 주입 순서 = index 순서
```

- `index`는 **조립 시점의 주입 순서**로 부여한다(토큰 예산에서 잘려 실제로 주입된
  청크에만 번호가 붙음 — "번호는 있는데 본문이 없는" 불일치를 원천 차단).
- `chunk_id`를 위해 `search_by_vector()` SELECT에 `id` 컬럼을 추가한다. 반환 dict에
  `"id"` 키 하나가 늘어날 뿐이라 하위 호환 파괴가 없다(기존 소비자는 키 존재를
  가정하지 않음). 감사 로그·재현 검증에 유용하므로 포함을 권장하되, 최소 구현에서는
  생략 가능(None 유지).
- docingest 소스 확장: `metadata.heading_path`/`page`가 있으면 `section` 표기를
  `" > ".join(heading_path)` + `p.{page}`로 보강할 수 있다 — v7.3 Part의
  "호출측이 heading_path/page로 출처를 표시·재구성 가능"(실측 라인 312) 예고와 정합.
  1차 구현 범위에서는 kowiki(title/section)만 다루고 확장 포인트로 주석만 남긴다.

---

## 3. 주입 형식 설계

### 3.1 청크 헤더에 출처 번호 라벨 부착

`get_context()` 조립부(라인 421)의 헤더를 인용 활성 시 다음과 같이 확장한다.

현행:

```
[kowiki · 요한 제바스티안 바흐 · 생애 · rr=0.95]
본문...
```

변경(citation.enabled=true일 때만):

```
[출처1 | kowiki · 요한 제바스티안 바흐 · 생애 · rr=0.95]
본문...

[출처2 | kowiki · 바로크 음악 · rr=0.87]
본문...
```

- 라벨 형식은 `[출처{N} | {기존 헤더 내용}]` — 기존 헤더를 대체하지 않고 앞에
  번호만 덧붙여, 기존 e2e 검증(광합성·바흐)과의 시각적 차이를 최소화한다.
- 라벨 접두어 "출처"는 config(`citation.label`)로 뺀다(하드코딩 금지 규칙 준수,
  영어 테넌트 대응 "Source" 등).
- `citation.enabled=false`(기본값)이면 헤더·반환 모두 **현행과 100% 동일** —
  이 저장소의 기존 관례(게이팅/MMR/리랭커 모두 "기본 OFF → yaml로 활성화" 무회귀
  패턴, `KnowledgeRetriever.__init__` 주석 참조)를 그대로 따른다.

### 3.2 인용 지시 trailer

`_attach_knowledge_base()`의 성공 분기(c) trailer 뒤에, citation.enabled일 때만
인용 지시(5장 초안)를 덧붙인다. "자료 없음" 분기(b)에는 인용할 대상이 없으므로
지시를 넣지 않는다(현행 유지).

---

## 4. 인용 노출 방식 2안 비교

### A안 — 본문 인라인 각주만

모델이 본문에 `[출처1]`을 달고, 답변 말미에 "출처: [출처1] 바흐 — 생애 (kowiki)"
목록을 **모델이 직접** 작성.

| 장점 | 단점 |
|---|---|
| 서버·응답 스키마 변경 0 (프롬프트만) | **출처 목록 자체를 모델이 생성 → 제목 왜곡·존재하지 않는 출처 날조 가능** (할루시네이션을 줄이려는 기능이 새 할루시네이션 표면을 만든다) |
| CLI·OpenAI 호환 엔드포인트에도 즉시 적용 | UI가 구조화 데이터로 활용 불가(링크·팝오버 등) |
| | 목록 작성에 출력 토큰 소모 |

### B안 — 응답 별도 `sources` 필드만

서버가 retriever의 citations를 `ChatResponse.sources`로 노출. 본문 마커 없음.

| 장점 | 단점 |
|---|---|
| **출처 데이터가 서버 진실(retriever 메타)** — 모델이 왜곡 불가. `downloads` 필드와 동일 원칙 | "어느 문장이 어느 출처인지" 문장 단위 매핑이 없음 — 검증가능성 절반 |
| 모델 지시 불필요 → 지시 무시 리스크 0 | 주입만 되고 실제로는 답변에 안 쓴 청크도 출처로 표시될 수 있음(과잉 표시) |
| 출력 토큰 비용 0 | 인용 의무가 없어 **할루시네이션 억제 효과가 없음**(핵심 목적 미달) |

### 권장 — A+B 하이브리드 (역할 분리)

- **본문 마커는 모델이** (`[출처N]` 번호만 — 제목/메타는 절대 본문에 재기술하지 않음),
- **출처 목록은 서버가** (`sources` 필드 — retriever가 반환한 citations 그대로).

모델이 생성하는 것은 "번호 참조"뿐이고, 번호가 가리키는 실체(제목·source·섹션)는
서버 데이터이므로 **날조 가능한 표면이 번호 오참조 하나로 축소**된다. 번호 오참조는
서버가 정규식으로 검증 가능하다(8장). 사용자 핵심 관심사(할루시네이션 저감)는 A의
인용 의무에서, 검증가능성·UI 노출은 B의 서버 진실에서 얻는다.

### 출처 데이터의 전달 경로 (4-Tier 체인 준수)

citations는 Tier 1(`QueryEngine.submit_message()`)에서 프롬프트 조립 직후 확보되며,
웹 핸들러까지는 **StreamEvent로만** 전달한다(체인의 유일한 데이터 전달 단위 —
architecture.md P1). 구체적으로:

1. `PromptAssembler._attach_knowledge_base()`가 `KnowledgeContext.citations`를
   `self.last_knowledge_citations`(조립 1회마다 갱신, 다음 조립 시 초기화)에 보관.
2. `submit_message()`가 `assemble()` 직후, citations가 있으면
   `StreamEvent(type=KNOWLEDGE_SOURCES, knowledge_sources=[...])` 1건을 yield.
3. `/v1/chat`은 이벤트 수집 중 해당 타입을 만나면 `ChatResponse.sources`에 채우고,
   `/v1/chat/stream`은 SSE로 그대로 흘려보내 UI가 실시간 표시.

`StreamEventType`(`core/message.py` 라인 89~114)에 `KNOWLEDGE_SOURCES =
"knowledge_sources"` 멤버를, `StreamEvent`에 `knowledge_sources:
list[KnowledgeCitation] | None = None` 필드를 추가한다. StreamEvent는 "type에 맞는
필드만 채워진다" 규칙의 union 통합 모델이므로 Optional 필드 추가는 설계 관례에
부합하며, 기존 이벤트 소비자는 모르는 type을 무시하므로 하위 호환이다.
(anti-patterns #3: 기존 이벤트 수정이 아니라 **새 이벤트 생성**이므로 위반 아님.
단, 웹 UI(static)와 CLI 이벤트 스위치가 미지 타입을 조용히 무시하는지 구현 시 확인.)

대안으로 "핸들러가 스트림 종료 후 `engine._prompt_assembler.last_knowledge_citations`를
직접 읽는" 방식도 가능하지만, 내부 속성 직참조는 체인 우회 냄새가 있고 스트리밍
경로에서 실시간 노출이 안 되므로 채택하지 않는다.

OpenAI 호환 `/v1/chat/completions`는 비표준 필드를 못 읽는 클라이언트가 대부분이므로
`_downloads_markdown()`(라인 1619) 패턴을 미러링해 content 말미에 출처 마크다운을
덧붙이는 것을 **후속 과제**로 남긴다(1차 범위 제외).

---

## 5. 프롬프트 지시문 초안 (한글)

### 5.1 `prompt_assembler._attach_knowledge_base()` trailer 추가분

기존 grounding trailer(★1) 뒤에 citation.enabled일 때만 이어 붙인다.

```
Citation rule (출처 표기): Each snippet above is labeled [출처N].
- When you state a fact taken from a snippet, append its label like [출처1] at the
  end of that sentence. Multiple labels are allowed: [출처1][출처3].
- Use ONLY the labels that actually appear above. Never invent labels or numbers.
- Do NOT restate snippet titles or metadata in the body — the label alone is enough.
- Statements from your own general knowledge get NO label; if a verifiable fact has
  no supporting snippet and you are not confident, say you are not sure (Grounding rule).
```

(주입 프롬프트의 기존 trailer들이 영어(모델 지시 효율)이므로 형식을 맞추되,
"출처N" 라벨 자체는 한글 그대로 사용한다. label을 config로 바꾸면 이 문자열도
같은 값으로 치환해 조립한다.)

### 5.2 `web/prompts/worker_system_full.md` / `worker_system.md`

`## When a --- Knowledge base --- block is present` 절(worker_system.md 라인 40~45,
full도 동일 절) 말미에 추가:

```markdown
- If snippets are labeled [출처N], cite the label at the end of each sentence that
  uses that snippet (e.g. "... 1750년에 사망했다 [출처1]."). Use only labels that
  exist in the block; never invent one. Do not add a separate source list at the
  end — the server renders it.
```

- 여기(정적 프롬프트)에는 "라벨이 있으면 인용하라"는 **조건부** 규칙만 넣는다.
  citation off이면 라벨이 주입되지 않으므로 이 규칙은 자연히 휴면한다 →
  프롬프트 파일을 config와 동기화할 필요가 없다.
- 사양서상 `core/system_prompt/` 템플릿은 미구현이므로(1.2-(4)) 대상에서 제외.
  CLI 경로는 trailer(5.1)만으로 커버된다.

---

## 6. Config 추가안 + Pydantic 변경

### 6.1 `config/nexus_config.yaml` (pc/b200 변형 3개 파일 동일 반영)

`knowledge_rag:` 섹션에 하위 블록 추가 — 기존 mmr/rerank와 동일한 "기본 OFF" 관례:

```yaml
knowledge_rag:
  # ... 기존 키 유지 ...
  # 출처 인용 — 주입 청크에 [출처N] 라벨을 붙이고 모델이 본문에 인용하게 한다.
  # 서버는 검색 메타데이터를 응답 sources 필드로 노출한다(모델 텍스트 아님).
  citation:
    enabled: false          # 기본 OFF — 켜기 전 주입 텍스트·응답 스키마 동작 100% 동일
    label: "출처"           # 라벨 접두어 ([출처1], [Source1] 등 테넌트별 언어 대응)
    max_sources: 5          # 응답 sources 필드에 노출할 최대 출처 수(주입 top_k 이하)
    expose_in_response: true  # false면 프롬프트 인용만 하고 sources 필드는 비움
    strip_invalid_labels: true  # 주입 범위 밖 번호([출처9] 등)를 응답 텍스트에서 제거
```

### 6.2 `core/config.py`

```python
class CitationConfig(BaseModel):
    """지식 RAG 출처 인용 설정 — 기본 OFF(무회귀), yaml 단일 소스."""
    enabled: bool = False
    label: str = "출처"
    max_sources: int = 5
    expose_in_response: bool = True
    strip_invalid_labels: bool = True


class KnowledgeRagConfig(BaseModel):
    # ... 기존 필드 유지 ...
    citation: CitationConfig = Field(default_factory=lambda: CitationConfig())
```

### 6.3 주입(bootstrap) 변경

`core/bootstrap.py` 라인 440~462의 `KnowledgeRetriever(...)` 생성 인자에
`citation_enabled=krag.citation.enabled, citation_label=krag.citation.label` 추가
(retriever 생성자 기본값은 `citation_enabled=False` — 기존 파라미터들과 동일한
"생성자 기본 무효 + bootstrap이 실값 주입" 패턴). `expose_in_response`/
`strip_invalid_labels`/`max_sources`는 웹 계층 관심사이므로 `web/app.py`의
엔진 조립부(라인 518 부근 components 경유)로 전달한다.

---

## 7. 구현 단계 (파일별 체크리스트) + 테스트 계획

### 7.1 구현 순서

의존성 방향(core/rag → orchestrator → web)을 따라 아래 순서로 진행한다.
3개 이상 파일 수정이므로 이 문서가 그 계획서를 겸한다.

**Step 1 — 데이터 모델·retriever (core/rag)**
- [ ] `core/rag/knowledge_retriever.py`
  - `KnowledgeCitation` / `KnowledgeContext` Pydantic 모델 추가(frozen, 한글 주석)
  - `get_context_with_citations(query, max_tokens, allowed_sources) -> KnowledgeContext`
    신설 — 기존 조립 로직을 내부 헬퍼로 추출해 재사용, citation_enabled일 때 헤더에
    `[{label}{N} | ...]` 부착 + 주입된 청크만 citations 생성
  - 기존 `get_context()`는 `(await get_context_with_citations(...)).text` 를 반환하는
    얇은 래퍼로 전환(시그니처·반환 불변 → 기존 테스트 무회귀)
  - `__init__`에 `citation_enabled: bool = False, citation_label: str = "출처"` 추가
- [ ] `core/rag/knowledge_store.py` — `search_by_vector()` SELECT에 `id` 추가,
  반환 dict에 `"id"` 키 추가(인메모리 폴백 `_inmemory_search`도 `e.id` 동봉)

**Step 2 — 이벤트·조립 (core)**
- [ ] `core/message.py` — `StreamEventType.KNOWLEDGE_SOURCES` 멤버,
  `StreamEvent.knowledge_sources: list[KnowledgeCitation] | None = None` 필드
  (lazy import 불필요 — message.py가 rag를 import하면 역방향이므로, citation 모델은
  `core/message.py`에 두거나 dict 직렬화로 담는다. **결정: 모델 정의를
  `core/message.py`에 두고 rag가 이를 import** — 의존성 방향 P2 준수:
  core/rag → core/message는 순방향)
- [ ] `core/orchestrator/prompt_assembler.py`
  - `_attach_knowledge_base()`가 `get_context_with_citations()`를 호출하도록 변경,
    citations를 `self.last_knowledge_citations`에 보관(매 assemble 시작 시 `()`로 리셋)
  - citation 활성 + 청크 존재 시 5.1 인용 trailer를 grounding trailer 뒤에 추가
- [ ] `core/orchestrator/query_engine.py` — `submit_message()`에서 `assemble()` 직후
  citations 존재 시 `KNOWLEDGE_SOURCES` StreamEvent 1건 yield
- [ ] `core/config.py` — `CitationConfig` + `KnowledgeRagConfig.citation`
- [ ] `core/bootstrap.py` — retriever 생성 인자 주입
- [ ] `config/nexus_config.yaml`, `config/nexus_config.pc.yaml`,
  `config/examples/nexus_config.b200.yaml` — `knowledge_rag.citation` 블록

**Step 3 — 웹 (web)**
- [ ] `web/app.py`
  - `SourceInfo`(또는 KnowledgeCitation 재사용) + `ChatResponse.sources:
    list[KnowledgeCitation] = Field(default_factory=list)` (기본 빈 리스트 —
    기존 클라이언트 무영향)
  - `/v1/chat`: 이벤트 수집 루프에서 KNOWLEDGE_SOURCES 수신 시 sources 채움,
    `strip_invalid_labels`면 응답 텍스트의 범위 밖 `[출처N]` 마커 정규식 제거
  - `/v1/chat/stream`: 이벤트를 SSE로 통과(직렬화가 model_dump 기반이면 무변경 확인)
  - UI(static)가 미지 이벤트 타입을 무시하는지 확인, sources 렌더링은 후속 UI 작업
- [ ] `web/prompts/worker_system.md`, `worker_system_full.md` — 5.2 인용 규칙 추가

**Step 4 — 검증**
- [ ] `ruff check . && ruff format`(수정 파일 한정) + `pytest tests/ -x`
- [ ] 실서버 e2e(mock 금지 원칙): B200에서 citation on으로 "요한 제바스티안 바흐"
  질의 → 본문 `[출처N]` 마커 + sources 필드 확인, "BWV 543"(자료 없음 경로) →
  sources 빈 리스트 + 자료없음 응답 유지 확인

### 7.2 테스트 계획

| 파일 | 테스트 (mock 청크 기반 단위) |
|---|---|
| `tests/unit/test_knowledge_retriever.py` | `test_citation_disabled_output_identical_to_legacy` — off 시 get_context 결과가 종전 문자열과 완전 동일(골든) |
| | `test_citation_enabled_headers_numbered_sequentially` — 청크 3건 주입 시 `[출처1|`~`[출처3|` 순번 부여 |
| | `test_citation_budget_cut_excludes_dropped_chunks` — 토큰 예산으로 잘린 청크는 citations에 없음 |
| | `test_citation_empty_results_returns_empty_context` — 게이팅 전체 드롭 시 text=""·citations=() |
| | `test_citation_score_prefers_rerank_over_similarity` |
| `tests/unit/test_knowledge_store.py` | `test_search_by_vector_returns_id_key` (인메모리 폴백 경로) |
| `tests/unit/test_prompt_assembler.py` | `test_attach_knowledge_citation_trailer_present_when_enabled` / `..._absent_when_disabled` / `test_last_knowledge_citations_reset_per_assemble` |
| `tests/unit/test_query_engine.py` (AsyncGenerator 패턴 — `async for` 전체 소비) | `test_submit_message_yields_knowledge_sources_event_when_citations_exist` / `..._no_event_when_chat_query` |
| `tests/unit/test_config.py` | `test_citation_config_defaults_off` / `test_citation_yaml_load` (기존 KnowledgeRagConfig 테스트 클래스 관례 따름) |
| `tests/integration/test_web_chat.py`(또는 기존 웹 테스트 파일) | `/v1/chat` mock 엔진으로 sources 필드 채움·strip_invalid_labels로 `[출처9]` 제거 검증 |

외부 서비스(vLLM/임베딩/PG)는 단위에서 mock, 최종 판정은 실서버 e2e(Step 4)로 한다.

---

## 8. 리스크와 완화

| 리스크 | 내용 | 완화 |
|---|---|---|
| 모델이 지시 무시 | 27B 로컬 모델이 `[출처N]` 표기를 빠뜨림 | (1) 지시를 주입 블록 직후 trailer + 정적 프롬프트 이중 배치, (2) sources 필드는 서버 진실이라 마커 누락과 무관하게 노출됨 — 기능이 부분 실패해도 B안 가치는 유지 |
| **허위 인용(가장 중요)** | 실제로는 청크가 뒷받침하지 않는 문장에 `[출처1]`을 붙여 "근거 있어 보이는 할루시네이션"이 됨 | 완전 차단은 불가(문장-근거 함의 검증은 별도 NLI 필요 — 범위 밖). 완화: (1) 출처 실체는 서버 데이터라 날조 불가, (2) 범위 밖 번호는 `strip_invalid_labels`로 제거, (3) 기존 게이팅이 무관 청크 자체를 차단하므로 인용 대상 풀이 이미 정제됨, (4) e2e에서 허위 인용 샘플 관찰 후 지시문 강화 반복 |
| 라벨이 본문 가독성 훼손 | CHAT스러운 짧은 답에도 라벨 남발 | 지시문에 "snippet에서 가져온 사실 문장에만" 한정 + CHAT/TOOL 질의는 주입 자체가 없어 무영향(routing 게이트 유지) |
| 토큰 비용 | 라벨 헤더 + 지시 trailer로 주입 ~150자, 출력 소폭 증가 | knowledge_rag_tokens(2500) 예산 내 흡수, max_sources=top_k=5라 상한 고정 |
| 스키마 하위 호환 | 기존 UI/AgentHub가 새 필드·이벤트에 놀람 | sources는 default 빈 리스트, KNOWLEDGE_SOURCES는 신규 타입(미지 타입 무시 확인). OpenAI 호환 경로는 1차 범위 제외 |
| 프롬프트 캐시 | 시스템 프롬프트 뒤쪽(주입부)만 변하므로 vLLM prefix cache 앞부분 유지 — 영향 미미 | 지시 trailer를 주입 블록과 함께 뒤쪽에 배치(현행과 동일 위치) |

---

## 9. 사양 대조 결과

- `user_mig/PROJECT_NEXUS_SPEC_v7.1_AMENDMENT.md`, `v7.2_AMENDMENT.md`를
  `인용|citation|출처|cite`로 grep한 결과, **RAG 출처 인용 기능에 대한 기존 계획은
  없음**(v7.2의 매치는 모두 "코드/원문 인용"이라는 일반 어휘 용례). 따라서 본 설계는
  사양 "이탈"이 아니라 **신규 확장**이며, 개정 근거는 다음과 같다.
  - v7.1 Part 2.5.8(Knowledge RAG)의 명시 목적이 할루시네이션 저감이고, 본 기능은
    동일 목적의 연장선(grounding → grounding + 검증가능성)이다.
  - v7.3 DOC_INGEST가 이미 "검색 반환 metadata의 heading_path/page로 **호출측이
    출처를 표시·재구성 가능**"(라인 312)을 예고 — 출처 표시를 소비하는 표준 통로를
    본 설계가 마련하므로 방향이 일치한다.
- 프로젝트 규칙 대조: 4-Tier 체인은 StreamEvent 전달로 준수(P1), 의존성은
  core/message ← core/rag ← orchestrator ← web 순방향(P2), 설정은 YAML
  단일 소스 + 기본 OFF fail-safe(P6·P7), 데이터 모델은 Pydantic v2 frozen(P5),
  에어갭 영향 없음(외부 호출 0, DB 컬럼은 기존 스키마 활용).
- 반영 필요 후속: 구현 완료 시 v7.1 계열 사양서에 "2.5.8.x 출처 인용" 절 추가 및
  `progress.md` 기록.

---

## 10. 권장안 요약

1. **A+B 하이브리드**: 본문 `[출처N]` 마커(모델) + 응답 `sources` 필드(서버 진실).
2. **주입 형식**: 기존 청크 헤더 앞에 `[출처N | ...]` 번호만 덧붙임 — 최소 변경.
3. **전달 경로**: 신규 `KNOWLEDGE_SOURCES` StreamEvent로 Tier 1 → 웹 (체인 준수).
4. **무회귀**: `citation.enabled` 기본 false, `get_context()` 시그니처 불변(래퍼화),
   `sources` 기본 빈 리스트 — 기존 mmr/rerank와 동일한 "기본 OFF → yaml 활성" 관례.
5. **범위**: 1차는 kowiki + `/v1/chat`(+stream). OpenAI 호환 마크다운 출처·docingest
   heading_path/page 표기·문장-근거 NLI 검증은 후속 과제.
