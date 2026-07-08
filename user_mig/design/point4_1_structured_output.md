<!-- 구조화 출력(structured output / vLLM guided decoding) 기능 추가를 위한 상세 설계 문서 — 구현 전 설계 전용 -->

# Point 4.1 — 구조화 출력(Structured Output / Guided Decoding) 상세 설계

- 작성일: 2026-07-08
- 상태: **설계(Design-only)** — 구현 전. 본 문서 승인 후 Phase별 구현.
- 대상 브랜치: `feature/b200-bakeoff`
- 근거 코드 리비전: `d1041bd` 시점의 working tree (아래 라인 번호는 이 시점 기준)

---

## 1. 개요 · 동기 / 현재 한계

### 1.1 목표

vLLM의 guided decoding(문법 제약 디코딩)을 활용해, 모델 출력이 호출자가 지정한
JSON Schema를 **토큰 생성 단계에서 강제로** 따르게 한다. 사후 검증(생성 후
jsonschema로 확인)과 달리, 디코딩 시점에 스키마 위반 토큰의 logit을 차단하므로
"거의 맞는 JSON"이 아니라 "항상 파싱 가능한 JSON"을 얻는다.

용도 3가지.

- **(a) 도구 호출 인자의 신뢰성 향상** — 현재 tool_calls 인자는
  `tool_call_parser=hermes`(vllm_launch.yaml, progress.md 2717행 확인)가 자유
  생성 텍스트에서 파싱하는 구조라, 인자 JSON이 깨지면
  `core/tools/executor.py`의 Step 3(`_validate_json_schema`, 185~196행)에서
  사후에 걸러 재시도 턴을 소모한다.
- **(b) AgentHub API 연동의 구조화 응답** — 외부 .NET AgentHub가
  `POST /v1/chat/completions`(web/app.py 619행~)로 붙는데, 현재
  `OpenAIChatCompletionRequest`는 `extra="ignore"`(650행)라서 클라이언트가
  보낸 `response_format`을 **조용히 버린다**. OpenAI 클라이언트 관점에서는
  규격 필드가 무시되는 호환성 결함이다.
- **(c) 사실 필드 형식 안정화** — 연도·수치·enum 등 형식이 고정된 답변에서
  "1685년" / "1685" / "약 1685년경" 같은 표기 흔들림을 스키마로 제거한다.

### 1.2 현재 한계 (코드 근거)

| 한계 | 근거 위치 |
|---|---|
| `ModelProvider.stream()`에 구조화 출력 파라미터가 전혀 없음 | `core/model/inference.py:100~114` (ABC), `258~272` (구현) |
| payload에 `response_format`/`guided_*` 미주입 | `core/model/inference.py:307~339` (payload 조립부) |
| Tier 2 `query_loop()`에도 전달 경로 없음 | `core/orchestrator/query_loop.py:506~529` (시그니처), `737~751` (stream 호출) |
| 외부 API가 `response_format`을 무시 | `web/app.py:639~661` (`extra="ignore"`) |
| 도구 인자 검증은 전부 **사후**(생성 완료 후) | `core/tools/executor.py:185~201`, `420~440` |

### 1.3 사후 검증과의 관계 (guided_json vs schema_validator)

기존 `executor._validate_json_schema()`(사후 검증)는 **제거하지 않는다**.
guided decoding은 Tier 3(모델 생성) 시점의 예방책이고, executor 검증은
방어선(defense-in-depth)이다. 특히 (a) 용도는 vLLM이 hermes 파서 경로에서
인자 문법을 완전히 보장하지 못할 수 있으므로 사후 검증이 계속 필요하다.
Fail-closed 원칙(P6)과도 부합한다.

---

## 2. vLLM 표면 조사 — payload에 무엇을 넣는가

### 2.1 전제: raw payload 방식

`core/model/inference.py:314~319` 주석이 명시하듯, 이 코드는 openai SDK가
아니라 **httpx raw POST**(`json=payload`)로 `/v1/chat/completions`에 직접
보낸다. 따라서 vLLM 확장 파라미터도 `extra_body` 래핑 없이 **payload 최상위**에
그대로 넣으면 된다 (repetition_penalty와 동일한 규칙, 324~327행 선례).

### 2.2 vLLM의 구조화 출력 인터페이스 3종 (버전별 변천)

| 인터페이스 | 형태 | 상태 |
|---|---|---|
| ① `guided_json` (레거시 vLLM 확장) | `{"guided_json": {…schema…}}` top-level | vLLM 0.10.x에서 deprecated 고지. **0.24(현 B200 배포 버전, progress.md 2717·2828행)에서는 제거되었을 가능성이 높음** |
| ② `structured_outputs` (vLLM 신형 확장) | `{"structured_outputs": {"json": {…schema…}}}` | 0.10+에서 ①의 후속으로 도입 |
| ③ `response_format` (OpenAI 표준) | `{"response_format": {"type": "json_schema", "json_schema": {"name": "...", "schema": {…}, "strict": true}}}` | OpenAI Chat Completions 표준. vLLM OpenAI 호환 서버가 장기 지원 |

**권장: ③ `response_format`을 1순위(primary)로 채택**한다. 이유는 다음과 같다.

1. OpenAI 표준이므로 (b) AgentHub 경로에서 외부 클라이언트가 보내는 필드를
   **그대로 통과**시키면 되어 변환 계층이 필요 없다.
2. vLLM 자체 확장(①②)은 버전마다 이름이 바뀌어 온 이력이 있고, 우리는
   vLLM 0.24라는 매우 최신 버전을 쓰므로 표준 필드가 가장 안전하다.
3. 프로젝트 규칙 P3("표준 내부 계약 = OpenAI 형식")의 정신과 일치한다.

단, **에어갭 환경에서 문서만으로 vLLM 0.24의 정확한 지원 표면을 단정할 수
없으므로**, 구현 Phase 0에 "B200 vLLM에 ③/②를 각각 curl로 보내 어떤 형태가
수용되는지 실측"하는 검증 단계를 필수로 둔다(8장 체크리스트 참조). 실측 결과에
따라 주입 형태를 config(`injection_mode`)로 전환할 수 있게 설계해 코드 수정
없이 대응한다(안티패턴 #4 하드코딩 금지의 응용).

### 2.3 guided decoding 백엔드

vLLM v1 엔진의 기본 구조화 출력 백엔드는 xgrammar(스키마→문법 컴파일, 컴파일
결과 캐시)다. 서버 기동 플래그 변경은 **불필요**(요청 단위 활성)하지만, 만약
실측에서 백엔드 관련 400 에러가 나오면 `config/vllm_launch.yaml`에 기동 옵션
추가를 검토한다(Machine B 측 변경 — Machine A 코드는 불변).

---

## 3. API 설계

### 3.1 데이터 모델 — `StructuredOutputSpec` (신규)

P5(Pydantic v2 + 불변 객체) 준수. `dict`를 그대로 넘기는 대안(9.3 참조)보다
이름·strict 여부를 함께 운반할 수 있어 채택한다. 위치는
`core/model/inference.py`의 `ModelConfig`(64행 부근) 옆 — stream()의 인자
타입이므로 같은 모듈에 두면 순환 import가 없다.

```python
# core/model/inference.py (신규 — ModelConfig 아래에 추가)
class StructuredOutputSpec(BaseModel):
    """구조화 출력(guided decoding) 요청 스펙 — 불변 객체.

    stream() 호출자가 이 객체를 넘기면 Tier 3가 vLLM payload에
    response_format(json_schema)으로 주입해 출력 문법을 강제한다.
    """
    model_config = ConfigDict(frozen=True)

    json_schema: dict[str, Any]        # JSON Schema (draft 2020-12 부분집합)
    name: str = "nexus_structured"     # OpenAI json_schema.name 필드
    strict: bool = True                # vLLM strict 모드 (스키마 완전 준수)
```

### 3.2 `stream()` 파라미터 추가

**변경 지점 1 — ABC** (`core/model/inference.py:100~114`).
기존 파라미터 나열의 맨 끝에 keyword 파라미터로 추가한다. 기본값 `None`이라
기존 호출부는 전부 무변경 동작(무회귀).

```python
    async def stream(
        self,
        messages: list[Message],
        system_prompt: str,
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        stop_sequences: list[str] | None = None,
        model_override: str | None = None,
        enable_thinking: bool | None = False,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        structured_output: StructuredOutputSpec | None = None,  # ← 신규
    ) -> AsyncGenerator[StreamEvent, None]:
```

**동일 시그니처를 유지해야 하는 구현/mock 전체 목록** (grep 실측):

| 파일 | 클래스 | 라인 |
|---|---|---|
| `core/model/inference.py` | `LocalModelProvider.stream` | 258 |
| `core/model/scout_provider.py` | `ScoutModelProvider.stream` (passthrough) | 138 |
| `tests/conftest.py` | `EnhancedMockModelProvider.stream` | 207 |
| `tests/unit/test_query_loop.py` | `ScriptedProvider.stream` | 55 |
| `tests/unit/test_thinking.py` | `MockModelProvider.stream` | 55 |

ScoutModelProvider는 enable_thinking 강제 사례(scout_provider.py:180~194)처럼
`structured_output=structured_output`으로 부모에 그대로 전달(passthrough)한다.
Scout(4B/llama.cpp)는 구조화 출력 대상이 아니지만 시그니처 정합성을 위해
파라미터는 받는다 (docstring에 "Scout 경로에서는 미사용" 명기).

### 3.3 payload 주입 위치 (정확한 코드 근거)

`core/model/inference.py`의 payload 조립부 — 구체적으로 **328행
(`"presence_penalty": presence_penalty,` 직후, `enable_thinking` 분기 329행
이전)이 아니라, 도구 스키마 주입 블록(337~339행) 직후**에 조건 분기로 넣는다.
이유: 도구와의 상호 배타 검사를 같은 자리에서 수행하기 위해서다.

```python
        # 도구 스키마 변환: Nexus → OpenAI function_calling   (기존 337~339행)
        if tools:
            payload["tools"] = [self._convert_tool_schema(t) for t in tools]
            payload["tool_choice"] = "auto"

        # ── 구조화 출력 (guided decoding) 주입 ── (신규)
        # 왜 여기인가: tools와 response_format의 동시 사용은 vLLM에서 의미가
        # 충돌한다(도구 호출 문법 vs 응답 본문 문법). fail-closed로 즉시 거부해
        # 조용한 오동작을 막는다.
        if structured_output is not None:
            if tools:
                raise ValueError(
                    "structured_output과 tools는 동시에 사용할 수 없습니다 "
                    "(guided decoding은 응답 본문 문법을 강제하므로 tool_calls "
                    "생성과 충돌). 호출자가 둘 중 하나만 지정해야 합니다."
                )
            payload["response_format"] = self._build_response_format(structured_output)
```

`_build_response_format()`은 config의 `injection_mode`에 따라 ③ 또는 ②
형태의 dict를 만든다(2.2 참조). raw payload이므로 top-level 주입 —
314~319행의 repetition_penalty 주석에 이미 문서화된 규칙과 동일하다.

**컨텍스트 초과 재시도 루프(362행 `for attempt in …`)와의 관계**: 재시도는
같은 `payload` dict를 재사용하고 `max_tokens`만 갱신하므로(363행),
`response_format` 키는 자동으로 보존된다. **추가 코드 불필요** — 설계상
확인만 해 둔다.

**enable_thinking과의 상호작용**: thinking 블록(`<think>…</think>`)은 JSON
문법을 위반한다. `structured_output` 지정 시 Tier 3에서 `enable_thinking`
값을 무시하고 `chat_template_kwargs.enable_thinking=False`로 강제한다
(ScoutModelProvider가 None을 강제하는 기존 패턴 scout_provider.py:159~163의
역방향 응용). 강제 시 DEBUG 로그 1줄을 남긴다.

### 3.4 Prompt cache(prefix caching) 영향 분석

B200 vLLM은 prefix-caching 활성 상태다(progress.md 2717행). 분석 결과.

1. **`response_format`은 프롬프트 토큰을 바꾸지 않는다.** prefix cache는
   입력 토큰 시퀀스의 prefix 해시 기반이므로, 샘플링/디코딩 파라미터
   (temperature, repetition_penalty, response_format 등)는 캐시 적중에 영향이
   없다. → 주입 자체는 캐시 중립.
2. **단, 구조화 출력 모드는 `tools`를 생략하므로**(3.3 상호 배타) 도구
   스키마가 chat template을 통해 프롬프트 앞부분에 들어가는 일반 모드와
   프롬프트 prefix가 달라진다 → 일반 대화 세션의 캐시와는 별개 라인. 이는
   "구조화 응답 전용 호출"(AgentHub 단발 호출, 내부 추출 파이프라인)이라는
   사용 패턴상 허용 가능한 비용이다. 같은 스키마로 반복 호출하면 그 라인
   자체는 캐시 적중한다.
3. **안티패턴 #5(도구 스키마 순서 고정)와의 정합**: 도구를 아예 보내지
   않으므로 순서 문제 자체가 발생하지 않는다. 스키마를 시스템 프롬프트에
   텍스트로 삽입하는 방식(프롬프트 오염 + 캐시 파편화)은 채택하지 않는다.
4. **xgrammar 문법 컴파일 캐시**: vLLM은 동일 스키마 문자열에 대해 컴파일
   결과를 캐시한다. 호출자는 **스키마 dict를 매 호출 재생성하지 말고 모듈
   상수/설정으로 고정**해 직렬화 결과가 바이트 단위로 동일하게 유지되도록
   한다(키 순서 흔들림 방지). 이를 8장 구현 체크리스트에 명기한다.

---

## 4. 4-Tier 체인 통과 경로

안티패턴 #1(체인 우회 금지) 준수 — 어떤 Tier도 건너뛰지 않고, 파라미터는
기존 `top_p`/`repetition_penalty`가 지나간 것과 **완전히 동일한 경로**로
내려보낸다(passthrough 선례: query_loop.py:745~750, model_dispatcher.py:158~161).

```
[호출자: web /v1/chat/completions 핸들러 · 내부 파이프라인]
        │  StructuredOutputSpec 생성
        ▼
Tier 1  QueryEngine.submit_message(user_input, structured_output=…)
        │  · query_engine.py:310~330 폴백 경로와 ~305 dispatcher 경로 양쪽에
        │    structured_output=… 한 줄씩 추가 (라우팅 결정과 무관 — Resolver 불변)
        ▼
(접착)  ModelDispatcher.route(…, structured_output=…)     ← model_dispatcher.py:149~165
        │  passthrough만 — 기존 output_token_escalation과 동일한 취급
        ▼
Tier 2  query_loop(…, structured_output=…)                ← query_loop.py:506~529 시그니처
        │  · Phase 1(1b, 620행): structured_output 지정 시 tool_schemas를 만들지
        │    않음(도구 비노출 — Tier 3 ValueError를 애초에 유발하지 않는 1차 방어)
        │  · Phase 2(737~751행): model_provider.stream(…, structured_output=…) 전달
        ▼
Tier 3  LocalModelProvider.stream(…, structured_output=…)  ← inference.py:258~
        │  · payload["response_format"] 주입 (3.3)
        │  · enable_thinking 강제 False (3.3)
        ▼
Tier 4  httpx AsyncClient.stream(POST /v1/chat/completions, json=payload)
        │  변경 없음 — payload에 키 하나 늘어날 뿐
        ▼
        vLLM(Machine B, LAN) — xgrammar가 logit 마스킹으로 스키마 강제
```

이벤트 역방향(vLLM→Tier1)은 **완전 무변경**: 구조화 출력도 일반 텍스트와
동일하게 `TEXT_DELTA` 스트림으로 올라온다. StreamEvent 타입 추가 없음 —
frozen dataclass 계약(P1) 불변.

**query_loop 내부에서 structured_output이 켜졌을 때의 턴 동작**: 도구가
비노출이므로 tool_use가 발생하지 않고, 모델이 JSON을 완성하면
`StopReason.END_TURN`으로 1턴 종료(`ContinueReason.COMPLETED`)가 정상
경로다. while 루프·7가지 ContinueReason 전환 로직은 손대지 않는다.

---

## 5. 호출자 인터페이스 — 누가 언제 스키마를 지정하는가

### 5.1 (b) AgentHub / 외부 OpenAI 클라이언트 — **1차 구현 대상**

`web/app.py`의 `OpenAIChatCompletionRequest`(639~661행)에 표준 필드를 추가한다.

```python
    # OpenAI 표준 response_format — {"type":"json_schema","json_schema":{…}} 또는
    # {"type":"json_object"}. 지정 시 엔진이 vLLM guided decoding으로 강제한다.
    response_format: dict[str, Any] | None = Field(default=None, description="…")
```

핸들러에서 `response_format.type == "json_schema"`이면 내부
`StructuredOutputSpec`으로 변환해 `submit_message()`에 넘긴다.
`"json_object"`(스키마 없는 JSON 강제)는 `{"type": "object"}` 스키마로
정규화한다. 알 수 없는 type은 400 응답(fail-closed — 조용한 무시 금지,
현재의 extra=ignore 동작이 바로 이 결함이었다).

**주의**: 이 경로의 요청은 stateless 단발 호출이므로, `submit_message()`의
파라미터는 세션 상태가 아니라 **호출 단위 인자**로 설계한다(세션에 스키마를
붙이면 다음 턴까지 오염됨).

### 5.2 (a) 도구 호출 인자 신뢰성 — **범위 분리 (Phase 2, 별도 레버)**

일반 에이전트 턴은 `tools + tool_choice="auto"`가 필수라 3.3의 상호 배타와
충돌한다. 도구 인자에 문법 강제를 거는 올바른 레버는 `guided_json`이 아니라
**vLLM의 named `tool_choice`**(`{"type":"function","function":{"name":…}}`
지정 시 vLLM이 해당 도구의 parameters 스키마로 guided decoding을 자동 적용)다.
이는 별도 파라미터(`tool_choice_override`)가 필요한 독립 기능이므로 본 설계의
구현 범위에서 **제외**하고, 확장 경로로만 문서화한다. 당장의 도구 인자
신뢰성은 기존 executor 사후 검증(Step 3~4) + 모델 자가교정 턴이 담당한다.

### 5.3 (c) 내부 파이프라인 (사실 필드·요약 구조화)

`ModelProvider.stream()`을 직접 소비하는 내부 호출자(예:
`context_manager.py:553`의 요약 생성, `core/ingest` 파이프라인)는 필요 시
`structured_output=`을 직접 넘긴다. Tier 2를 거치지 않는 이 호출들은
원래부터 Tier 3 직접 소비가 허용된 보조 경로다(체인 우회 아님 — 에이전트
턴이 아닌 단발 생성). **본 설계에서는 인터페이스만 열어 두고 내부 호출자
전환은 하지 않는다**(과설계 방지).

### 5.4 설정 기반 강제는 하지 않는다

"config에 스키마를 넣고 전 요청에 적용" 같은 전역 스위치는 두지 않는다.
스키마는 항상 **호출 단위**로 지정한다 — 용도 (a)(b)(c) 모두 요청마다
스키마가 다르기 때문이며, 전역 강제는 일반 대화를 파괴한다.

---

## 6. Config 추가안

### 6.1 YAML (`config/nexus_config.yaml`)

`model:` 섹션(83~95행)과 나란히 최상위 섹션으로 추가한다. 스키마 자체가
아니라 **기능 동작 방식**만 설정한다(6.3 참조).

```yaml
# ── 구조화 출력 (vLLM guided decoding) ────────────────────────────
# 외부 OpenAI 클라이언트(AgentHub 등)의 response_format 수용 및 내부
# 구조화 생성에 사용. 스키마는 항상 요청 단위로 지정한다(전역 스키마 없음).
structured_output:
  enabled: true               # 마스터 스위치. false면 response_format을 400으로 거부
  injection_mode: "response_format"  # vLLM payload 주입 형태:
                              #   "response_format"    — OpenAI 표준 (권장, 기본)
                              #   "structured_outputs" — vLLM 신형 확장 (표준형 미지원 시)
                              # Phase 0 실측 후 B200 vLLM 0.24에 맞는 값으로 확정.
  max_schema_bytes: 65536     # 스키마 직렬화 크기 상한 — 초과 시 요청 거부.
                              # 거대/재귀 스키마의 xgrammar 컴파일 지연(워치독
                              # idle 30초와 충돌 위험)을 사전 차단한다.
  strict: true                # json_schema.strict 기본값
```

### 6.2 Pydantic 모델 (`core/config.py`)

`KnowledgeRagConfig`(978행) 패턴을 그대로 따른다 — "yaml이 단일 소스,
클래스 기본값은 yaml 누락 시 폴백" 주석 관례 포함.

```python
class StructuredOutputConfig(BaseModel):
    """구조화 출력(guided decoding) 동작 설정.

    단일 소스는 config/nexus_config.yaml#structured_output 이며,
    이 기본값은 yaml 누락 시(테스트/경량 실행) 폴백으로만 쓰인다.
    """
    enabled: bool = True
    injection_mode: str = "response_format"   # "response_format" | "structured_outputs"
    max_schema_bytes: int = 65536
    strict: bool = True
```

`NexusConfig`(루트 설정 모델)에 `structured_output: StructuredOutputConfig =
Field(default_factory=StructuredOutputConfig)` 필드를 추가하고, 로더의 섹션
매핑에 `"structured_output"` 키를 등록한다(기존 `knowledge_rag` 등록부와
동일 위치).

### 6.3 왜 스키마를 YAML에 넣지 않는가

안티패턴 #4는 "설정값 하드코딩 금지"이지 "모든 데이터의 YAML화"가 아니다.
JSON Schema는 요청별 가변 데이터(AgentHub가 보내는 값)이므로 설정이 아니다.
반복 사용되는 내부 스키마가 생기면 그때 `config/structured_schemas/*.yaml`
분리를 검토한다(현 시점 과설계).

---

## 7. 스트리밍 호환성 — guided decoding + SSE 동시 동작

**결론: 동시 동작 가능. 코드 변경 불필요.** 근거와 유의점.

1. vLLM의 구조화 출력은 **logit 마스킹** 방식이다 — 각 디코딩 스텝에서
   문법상 불가능한 토큰의 확률을 0으로 만든 뒤 정상 샘플링한다. 토큰은
   여전히 한 개씩 생성되므로 SSE 스트리밍(`stream: true`)과 직교한다.
   기존 SSE 파서(`inference.py`의 청크 누적 로직)는 무변경.
2. **TTFT(첫 토큰 지연) 증가**: 새 스키마의 첫 요청은 xgrammar 문법 컴파일
   비용이 든다(단순 스키마 수십 ms ~ 복잡 스키마 수 초). StreamWatchdog
   (query_loop.py:752~756, idle 30초)의 idle 타이머는 MESSAGE_START 이벤트
   (Tier 3가 요청 직후 자체 yield, inference.py:349~352)로 이미 1회
   갱신되므로, 컴파일이 30초를 넘지 않는 한 오탐이 없다. 30초급 컴파일은
   `max_schema_bytes` 상한(6.1)으로 사전 차단하는 것이 설계 방침이다.
3. **부분 JSON 문제(호출자 책임)**: 스트림 중간 시점의 누적 텍스트는 아직
   닫히지 않은 JSON이다. 서버가 보장하는 것은 "정상 종료 시 최종 누적본이
   유효"라는 것뿐이다. AgentHub 비스트림 경로(전량 수집 후 반환)는 문제
   없고, 스트림 경로 문서에 "완료 후 파싱" 주의를 명기한다.
4. **finish_reason=length 함정**: `max_tokens` 소진으로 잘리면 guided라도
   JSON이 미완성이다. 이는 9.1 리스크에서 처리 방침을 정한다.

---

## 8. 구현 단계 (파일별 체크리스트) + 단위테스트 계획

### Phase 0 — 실측 검증 (코드 변경 없음)

- [ ] B200 vLLM 0.24에 curl로 ③ `response_format(json_schema)` / ②
      `structured_outputs` 요청을 각각 보내 수용 여부·에러 메시지 실측
      (paramiko → 192.168.21.112 컨테이너 내부에서 127.0.0.1:8001 대상)
- [ ] `stream: true` + 구조화 출력 동시 요청의 SSE 정상 수신 확인
- [ ] 실측 결과로 `injection_mode` 기본값 확정, 본 문서 2.2 갱신

### Phase 1 — Tier 3 (모델 계층)

- [ ] `core/model/inference.py` — `StructuredOutputSpec` 모델 추가 (3.1)
- [ ] `core/model/inference.py` — ABC `stream()` 시그니처에
      `structured_output: StructuredOutputSpec | None = None` 추가 (100행)
- [ ] `core/model/inference.py` — `LocalModelProvider.stream()` 구현 (258행):
      payload 주입 + tools 상호 배타 ValueError + enable_thinking 강제 False
      + `_build_response_format()` 헬퍼 (injection_mode 분기)
- [ ] `core/model/scout_provider.py:138` — passthrough 파라미터 추가
- [ ] 전 코드 한글 주석 (왜 상호 배타인지, 왜 top-level 주입인지)

### Phase 2 — Tier 2·접착층·Tier 1

- [ ] `core/orchestrator/query_loop.py:506` — 파라미터 추가 + Phase 1b(620행)
      도구 비노출 분기 + Phase 2(737행) stream 전달
- [ ] `core/orchestrator/model_dispatcher.py:149` — `route()` passthrough
- [ ] `core/orchestrator/query_engine.py` — `submit_message()` 호출 단위 인자
      추가, dispatcher 경로(~305행)/폴백 경로(310행) 양쪽 전달

### Phase 3 — Config

- [ ] `core/config.py` — `StructuredOutputConfig` + `NexusConfig` 필드 + 로더 등록
- [ ] `config/nexus_config.yaml` — `structured_output:` 섹션 (6.1)

### Phase 4 — Web (AgentHub 경로)

- [ ] `web/app.py` — `OpenAIChatCompletionRequest.response_format` 필드 추가
- [ ] `web/app.py` — 핸들러에서 스펙 변환 + `enabled=false`/미지원 type/크기
      초과 시 400 (OpenAI 규격 에러 body)

### Phase 5 — 테스트 갱신·신규

**기존 mock 갱신 (시그니처 정합):**

- [ ] `tests/conftest.py:207` — `EnhancedMockModelProvider.stream`
- [ ] `tests/unit/test_query_loop.py:55` — `ScriptedProvider.stream`
      (+ 마지막 수신 `structured_output`을 `self.last_structured_output`에
      기록해 passthrough 검증 가능하게)
- [ ] `tests/unit/test_thinking.py:55` — `MockModelProvider.stream`

**신규 단위 테스트** (`tests/` 네이밍 규칙 `test_{feature}_{scenario}_{expected}`):

| 테스트 | 검증 내용 |
|---|---|
| `test_stream_structured_output_injects_response_format` | httpx 요청 payload를 monkeypatch로 캡처해 `response_format.json_schema.schema` 일치 확인 |
| `test_stream_structured_output_with_tools_raises_valueerror` | 상호 배타 fail-closed |
| `test_stream_structured_output_forces_thinking_off` | payload의 `chat_template_kwargs.enable_thinking is False` |
| `test_stream_structured_output_injection_mode_structured_outputs` | config 전환 시 ② 형태로 주입 |
| `test_query_loop_structured_output_passthrough_to_tier3` | ScriptedProvider.last_structured_output으로 Tier2→3 전달 확인 |
| `test_query_loop_structured_output_omits_tool_schemas` | 도구 비노출 분기 |
| `test_config_structured_output_yaml_load_and_fallback` | YAML 로드 + 누락 시 기본값 |
| `test_openai_endpoint_response_format_json_schema_accepted` | web 핸들러 변환 (FastAPI TestClient) |
| `test_openai_endpoint_response_format_unknown_type_400` | fail-closed 거부 |
| `test_openai_endpoint_schema_too_large_400` | `max_schema_bytes` 상한 |

- [ ] AsyncGenerator 테스트는 규칙대로 `async for` 전량 소비로 검증
- [ ] `ruff check . && pytest tests/ -x` (기존 스위트 무회귀 확인)

### Phase 6 — E2E (실서버, mock 금지 원칙)

- [ ] B200 실 vLLM 대상: 연도 추출 스키마(`{"composer": str, "birth_year": int}`)
      질의 → 유효 JSON 수신 + jsonschema 사후 검증 통과
- [ ] AgentHub 시나리오: OpenAI 클라이언트로 `response_format` 포함 요청 e2e

---

## 9. 리스크 · 엣지케이스 · 사양 대조

### 9.1 리스크와 대응

| # | 리스크 | 대응 |
|---|---|---|
| R1 | vLLM 0.24가 ③ 형태를 거부/부분지원 | Phase 0 실측 + `injection_mode` config 전환 (코드 무수정 대응) |
| R2 | `finish_reason=length`로 JSON 미완성 | Tier 3는 기존대로 `StopReason.MAX_TOKENS`를 올림. web 핸들러가 이 경우 파싱을 시도하지 않고 OpenAI 규격대로 `finish_reason:"length"` 반환(클라이언트 책임 경계 명확화). 내부 호출자는 stop_reason 검사 후 재시도 |
| R3 | 거대/재귀 스키마의 문법 컴파일 지연 → 워치독 오탐/서버 부하 | `max_schema_bytes` 상한 + 400 거부 (fail-closed) |
| R4 | 스키마 dict 재생성으로 xgrammar 캐시 미스 | 내부 호출자는 스키마를 모듈 상수로 고정 (체크리스트 명기) |
| R5 | tools와 동시 지정 시 vLLM의 미정의 동작 | Tier 2 도구 비노출(1차) + Tier 3 ValueError(2차) 이중 방어 |
| R6 | thinking 블록이 JSON 오염 | Tier 3에서 enable_thinking 강제 False |
| R7 | guided라도 **내용**은 보장 안 됨(형식만) — 잘못된 연도를 형식만 맞게 답할 수 있음 | 문서·주석에 명기. 사실성은 기존 KB RAG 게이팅의 책임 범위로 유지 |
| R8 | 세션 오염(스키마가 다음 턴에 잔류) | 스펙을 세션 상태가 아닌 호출 단위 인자로만 설계 (5.1) |

### 9.2 사양 대조 결과 (필수 항목)

- `user_mig/PROJECT_NEXUS_SPEC_v7.1_AMENDMENT.md`,
  `PROJECT_NEXUS_SPEC_v7.2_AMENDMENT.md`(및 v7.0)를
  `guided|structured|response_format|json_schema|outlines|구조화` 패턴으로
  grep한 결과 **일치 항목 없음**. 유일한 근접 항목은 v7.2 195행의 MCP
  어댑터 `validate_input()` 설명(사후 JSON Schema 검증 — 본 기능과 무관).
- 따라서 본 기능은 **사양서에 계획되지 않은 신규 추가 = 사양 이탈**이며,
  개정 근거는 다음과 같다.
  1. AgentHub 연동(승인된 진행 항목)에서 OpenAI 표준 필드 `response_format`
     이 조용히 무시되는 것은 "드롭인 LLM 프로바이더"(web/app.py:622~623
     설계 의도)의 규격 위반이다.
  2. 기존 아키텍처 원칙(P1 체인, P3 OpenAI 계약, P5 Pydantic, P6
     fail-closed)을 전부 유지하는 순수 추가(additive)로, 기본값 None/미지정
     시 동작이 100% 불변이다.
- **후속 조치**: 구현 완료 시 v7.3 AMENDMENT(또는 v7.2 추보)에 본 문서를
  요약 반영한다. 그전까지 본 문서가 단일 설계 근거다.

### 9.3 검토했으나 채택하지 않은 대안

| 대안 | 기각 사유 |
|---|---|
| A. 프롬프트 지시("JSON으로만 답하라") + 사후 jsonschema 검증/재시도 | 보장 없음(현 상태의 한계 그 자체). 재시도 턴 낭비. 단, guided 미지원 판명 시 최후 폴백으로 재부상 가능 |
| B. `stream()`에 raw `dict` 파라미터(`guided_json: dict | None`) | 이름/strict를 별도 파라미터로 또 늘려야 함. P5(Pydantic 우선)에 따라 스펙 객체 채택. 다만 구현 난이도 차이는 미미하므로 구현자 재량 절충 허용 |
| C. 레거시 `guided_json` top-level 주입을 기본으로 | vLLM 0.10+에서 deprecated, 0.24 제거 가능성. injection_mode 후보에도 넣지 않음(실측에서 필요 판명 시 추가) |
| D. 스키마를 시스템 프롬프트에 삽입 | prefix cache 파편화 + 보장 없음. 기각 |
| E. tool 1개짜리 가짜 도구로 우회(function calling 강제) | hermes 파서 의존 + P3 계약 혼탁. named tool_choice 확장(5.2)이 정도(正道) |

### 9.4 권장안 요약

**`StructuredOutputSpec`(frozen Pydantic) 신규 + `stream()` 말단 keyword
파라미터 + `response_format(json_schema)` top-level 주입(injection_mode로
전환 가능) + tools 상호 배타 fail-closed + web `/v1/chat/completions`
`response_format` 수용**이 최소 변경·무회귀·규칙 정합(P1/P3/P5/P6,
안티패턴 #1/#4/#5 위반 없음)의 권장 조합이다. 구현 착수 전 Phase 0 실측이
선행 조건이다.
