# 설계서 — 자기일관성(Self-Consistency) 사실 검증 기능 (Point 4.3)

> 작성일: 2026-07-08 / 상태: 설계 초안 (구현 전)
> 대상: Project Nexus (D:\workspace\nexus-b200)
> 원칙: **기본 비활성(fail-closed)** · KNOWLEDGE 클래스 한정 · N=3 소규모 · 토큰 비용 정직 분석

---

## 1. 개요·동기 및 게이팅

### 1.1 동기

사실형(KNOWLEDGE) 질의에서 로컬 중형 모델(A.X-4.0 / Qwen 27B급)은 다음 오류를 낸다.

- **수치 깨짐**: 연도·인구·수치가 샘플링 노이즈로 자릿수/값이 틀어짐
- **고유명사 혼동**: 유사 엔티티(인물/지명) 치환
- **일관성 없는 오답**: 같은 질문에 매번 다른 답 (temperature > 0인 이상 불가피)

Self-Consistency(Wang et al. 2022)는 같은 질의를 N번 샘플링해 **다수결(majority)로 답을
확정**하는 기법이다. 핵심 관찰: *오답은 흩어지고 정답은 수렴한다*. 단일 샘플 오류율이
p일 때 독립 3표본 다수결의 오류율은 대략 3p²−2p³ (p=0.3이면 0.216 → 약 28% 저감).
단, 이는 **오답이 서로 다른 방향으로 흩어질 때**만 성립한다. 모델이 체계적으로 같은
오답을 내면(학습 데이터 오류) SC는 무력하다 — 이 한계를 §8에 명시한다.

### 1.2 게이팅 — 언제 발동하는가 (3중 게이트)

토큰 예산이 빠듯하므로(§4) 발동 조건을 3중으로 좁힌다. **모두 통과해야만** SC 경로 진입.

| 게이트 | 판정 주체 | 조건 |
|---|---|---|
| G1. 설정 | config | `routing.self_consistency.enabled == true` (기본 false) |
| G2. 질의 클래스 | RoutingResolver | `RoutingDecision.query_class ∈ apply_classes` (기본 `["KNOWLEDGE"]`) — TOOL/CHAT 제외 |
| G3. 사실형 패턴 | 신규 휴리스틱 | 질문이 **짧은 사실 답**을 기대하는 패턴일 때만 (선택, 기본 on) |

G3 사실형 패턴 휴리스틱(`core/orchestrator/routing.py`의 HeuristicClassifier와 동일한
사전 컴파일 정규식 방식).

- 매치 예: `언제|몇\s*(년|명|개|살|km|%)|누구|어디|얼마|무슨\s*(년도|색)|수도는|이름은` 등
- 질문 길이 상한(예: 120자) 병행 — 긴 복합 질문은 서술형 가능성이 높아 제외
- **왜 필요한가**: 서술형 답변은 majority 합의가 본질적으로 어렵다(§3.3). 서술형까지
  SC를 태우면 토큰만 3배 쓰고 품질 이득이 없다. G3가 실질적 비용 방어선이다.

게이팅 판정은 **RoutingResolver.resolve() 내부**에서 수행하고 결과를 RoutingDecision에
싣는다(§2.3). 분류가 이미 이 지점에 집중돼 있고 순수 함수라 테스트가 쉽기 때문이다.

### 1.3 기존 thinking 모듈과의 관계 (중복 검토)

`core/thinking/`에는 이미 SelfReflectionEngine(3-pass 검증)과 HiddenCoTEngine(2-pass)이
있다. Self-Consistency는 이들과 **직교**한다.

- Self-Reflection = **직렬** 재검토 (같은 표본을 다시 비평) → 논리 오류 교정에 강함
- Self-Consistency = **병렬** 독립 표본 + 투표 → 샘플링 노이즈(수치 깨짐)에 강함

ThinkingStrategy의 6번째 전략(`SELF_CONSISTENCY`)으로 넣는 안을 검토했으나 **채택하지
않는다**. 근거: grep 결과 ThinkingOrchestrator는 현재 **프로덕션 쿼리 경로에 배선되어
있지 않다**(참조처가 `tests/integration/test_full_pipeline.py`뿐). 미배선 모듈에 얹으면
기능이 실행되지 않는다. 장기적으로 thinking 모듈이 배선되면 전략 통합을 재검토한다
(§8 사양 대조 참조).

---

## 2. 아키텍처 배치 — 4-Tier 체인 무결성 유지

### 2.1 배치 후보 비교

| 후보 | 방법 | 판정 |
|---|---|---|
| A. Tier 1 감싸기 | QueryEngine이 query_loop을 N회 호출 | **기각** |
| B. Tier 2 내부 + vLLM `n` 파라미터 | query_loop 1턴에서 stream(n=N) 1요청 | **채택** |
| C. Tier 3 병렬 반복 호출 | stream()을 asyncio.gather로 N회 | 기각 |

**A 기각 근거**: query_loop 1회 = 시스템 프롬프트 조립 + KB RAG 주입 프롬프트 전체의
prefill 1회다. N회 호출이면 **prefill이 N배**(KB RAG 포함 3~6K 토큰 × N). 게다가
`_finalize_turn`(메모리/트랜스크립트 영속화)과 messages 히스토리 부작용이 N번 발생해
격리 비용이 크다. Tier 1은 "조율만 한다"는 현행 설계 철학(query_engine.py 도입부
주석)과도 어긋난다.

**C 기각 근거**: A와 동일한 prefill N배 문제. vLLM이 요청 3건을 배치해 주긴 하지만
prefix cache 적중을 보장할 수 없고(요청 간 타이밍), 프롬프트 토큰 회계상 3배로 계상된다.
구현은 단순하나 5090 예산에서 정당화 불가.

**B 채택 근거**: vLLM OpenAI 호환 API의 `n` 파라미터는 **한 요청 안에서 프롬프트
KV cache를 N개 시퀀스가 공유**한다. 즉 prefill 1회 + 디코드만 N배. 디코드도 같은
스텝에 배치되므로 벽시계 지연은 단일 대비 소폭 증가에 그친다(§4). 이것이 5090
제약에서 유일하게 현실적인 방법이다.

### 2.2 체인 통과 경로 (P1 준수 확인)

```
Tier 1: QueryEngine.submit_message()
          └ RoutingResolver.resolve() → RoutingDecision(sc 필드 포함)   ← 게이팅만
Tier 2: query_loop()
          ├ Phase 2: model_provider.stream(..., n=decision.sc_n)        ← 기존 호출에 인자 1개 추가
          ├ (신규) 후보 텍스트 N개 버퍼 수집
          ├ (신규) ConsensusResolver.resolve(candidates) → 승자 확정     ← 순수 함수, 모델 호출 없음
          └ 승자 텍스트를 TEXT_DELTA로 yield (의사 스트림, §6)
Tier 3: LocalModelProvider.stream()
          └ payload["n"] = n, SSE choices를 index로 디멀티플렉스        ← 파서 확장
Tier 4: httpx (변경 없음)
```

- Tier 1은 여전히 Tier 2만 소비하고, Tier 2는 Tier 3만 소비한다 — **체인 우회 없음**
  (anti-patterns #1 준수).
- 합의(ConsensusResolver)는 **추가 모델 호출이 없는 순수 로직**이므로 체인 밖의
  헬퍼 모듈(`core/orchestrator/self_consistency.py`)로 둔다. 임베딩 폴백(§3.3)만
  `model_provider.embed()`를 쓰는데, 이는 stream 체인이 아니라 기존 RAG와 동일한
  보조 API 사용 패턴이다(LAN 내부, 에어갭 준수).
- **StreamEvent는 frozen** — Tier 3는 후보별 델타를 수정하지 않고 새 이벤트로 생성한다
  (anti-patterns #3 준수).

### 2.3 Tier 3 확장 — SSE choices 디멀티플렉스

현행 파서는 `choices[0]`만 읽는다(`core/model/inference.py:472`,
"스트리밍에서는 한 청크당 choice 1개" 주석). n>1이면 vLLM은 청크마다
`choices[i].index`로 시퀀스를 구분해 보낸다. 확장 방식.

- `stream(..., n: int = 1)` 파라미터 추가. **n=1이면 기존 코드 경로 그대로**(무회귀).
- n>1이면: index별 텍스트 버퍼 dict를 유지하고, **라이브 TEXT_DELTA를 내보내지 않는다**
  (§6에서 UX 논의). 모든 choice가 finish되면 후보 텍스트를 `SYSTEM_INFO` 이벤트의
  구조화 필드로 올리는 대신, **Tier 2가 소비할 수 있도록 choice 0의 완성 텍스트를
  기존 이벤트 계약대로 내보내고, 나머지 후보는 MESSAGE_STOP 직전에 신규 이벤트
  `SC_CANDIDATE`(StreamEventType 추가, text + sample_index=metadata)로 1건씩 yield**한다.
- 신규 이벤트 타입 추가는 StreamEventType enum에 값 1개(`SC_CANDIDATE`)와 StreamEvent에
  선택 필드(`sample_index: int | None = None`)를 더하는 것이다. 기존 소비자는 모르는
  타입을 무시하므로(웹/CLI 이벤트 스위치 방식) 하위 호환이 유지된다.
- **tool_calls가 섞이면 SC 중단**: 어떤 choice든 TOOL_USE_START가 나오면 SC를 포기하고
  choice 0만으로 기존 단일 스트림처럼 동작(폴백). KNOWLEDGE 질의는 통상 도구를 안
  쓰지만 tools 스키마는 여전히 전달되므로 이 방어가 필요하다. N개 표본이 서로 다른
  도구 경로를 타면 합의가 정의 불가능하기 때문 — 이 한계를 숨기지 않는다.

### 2.4 ModelDispatcher 경로

dispatcher가 주입된 배포 구성(B200)에서는 `dispatcher.route()`에도 `sc_n` 등 파라미터를
passthrough한다(기존 top_p 등과 동일한 전달 관례, routing.py 주석 "passthrough" 참조).
dispatcher 내부 Worker 호출이 결국 같은 `stream(n=)`으로 수렴하므로 로직 중복은 없다.

---

## 3. 합의(Consensus) 로직

신규 모듈 `core/orchestrator/self_consistency.py` — 전부 순수 함수 + frozen dataclass.

### 3.1 답 추출·정규화

투표 전에 표면형 차이를 제거한다. `normalize_answer(text) -> str`.

1. 앞뒤 공백·마크다운 장식(`**`, 리스트 마커) 제거
2. 문장부호·조사 꼬리 제거("~입니다." → 핵심어), 라틴 문자 소문자화
3. **숫자 정규화**: 천단위 콤마 제거, 전각→반각, "약 5,100만" → `51000000` 계열의
   canonical 폼. 한국어 수사("오천만")까지 파싱하는 건 과설계 — 1차 구현 범위 제외를
   명시하고 실패 시 문자열 비교로 폴백
4. 단위 표준화는 화이트리스트(년/명/개/%/km)만. 그 외는 원문 유지

### 3.2 짧은 사실형 답 — 정규화 exact majority

- 후보 전원이 `short_answer_max_chars`(기본 80자) 이하 → 정규화 후 동일 문자열 그룹핑
- 최다 그룹의 표 수 ≥ `min_agreement`(기본 2, N=3 기준 과반) → 그 그룹의 **원문 중
  가장 긴 후보**를 승자로 채택(정규화 폼이 아닌 원문을 보여줘야 자연스러움)
- 숫자 답은 정규화 값이 일치하면 같은 표로 집계 ("5,100만 명" = "약 5100만명")

### 3.3 서술형 긴 답 — 정직한 한계와 차선책

**서술형은 self-consistency가 원리적으로 어렵다.** 표면형이 전부 다르므로 exact 투표가
불가능하고, "의미적으로 다수"를 뽑아도 문장 안의 개별 사실 오류(연도 하나 틀림)는
문장 전체 유사도에 묻혀 걸러지지 않는다. 이 설계는 서술형에 대해 **오답 교정을
보장하지 않으며**, 다음 차선책만 제공한다.

- 임베딩 클러스터링: 기존 `model_provider.embed()`(e5-large, LAN 임베딩 서버 8002,
  에어갭 준수)로 후보 N개를 임베딩 → 코사인 유사도 ≥ `similarity_threshold`(기본 0.90)
  쌍을 묶어 최대 클러스터의 **medoid**(다른 후보들과 평균 유사도 최대)를 채택
- 효과의 정직한 기술: 이 방식이 잡는 것은 "한 표본만 완전히 다른 주제로 샌 경우"
  (탈선 표본 배제)까지다. 세부 수치 오류는 못 잡는다.
- **그래서 기본 정책은 G3 게이트(§1.2)로 서술형을 아예 SC 대상에서 빼는 것**이다.
  임베딩 클러스터링은 G3를 끄고 실험할 때의 폴백 경로로만 둔다.

### 3.4 합의 실패 시

최다 표가 `min_agreement` 미만(3표가 전부 다름) → **후보 0번을 그대로 채택**하고
`SYSTEM_WARNING` 이벤트 + JSONL 로그(`nexus.orchestrator.self_consistency`)로
"합의 실패(0/3)"를 기록한다. 재샘플링(N 추가)은 하지 않는다 — 토큰 예산 원칙 위반.
합의 실패율 자체가 "이 질의는 모델이 모른다"는 신호이므로 로그를 남겨 관측한다
(후속: 실패율이 높으면 답변에 불확실성 고지 문구를 붙이는 확장 가능 — 본 설계 범위 외).

### 3.5 SC 샘플링 온도 — 반드시 별도 값

knowledge_mode의 현행 temperature 0.2(config.py 기본) / yaml 0.2로는 3표본이 사실상
동일하게 나와 **투표가 무의미**하다(전 표본이 같은 오답으로 수렴). SC는 다양성이
전제이므로 SC 전용 `temperature`(기본 0.7)·`top_p`(0.95)를 쓴다. 개별 표본 오류율은
올라가지만 투표가 상쇄한다는 것이 원 논문의 핵심이다. repetition_penalty 1.15 등
degeneration 방지 파라미터는 knowledge_mode 값을 그대로 상속한다.

---

## 4. 비용 분석 (정직)

> 전제 수치 출처: `user_mig/rtx5090_token_limits.md`(5090 실측),
> `config/nexus_config.yaml`(B200 현행: max_context_tokens 49152, knowledge max_tokens 4096).
> 아래 디코드 속도는 **추정치**이며 실측으로 대체해야 한다(§7 Phase 6).

### 4.1 토큰 비용

| 항목 | 단일 샘플 | SC (vLLM n=3, 채택안) | 반복 호출 3회 (기각안) |
|---|---|---|---|
| prefill (프롬프트+KB RAG ≈ 3~6K) | 1회 | **1회 (공유)** | 3회 |
| 디코드 출력 토큰 | T | **3T** (T는 sc max_tokens 512로 캡) | 3T |
| 사실형 짧은답 T≈150~300 기준 추가 토큰 | — | +300~600 | +6K~12K(프리필 포함) |

- vLLM `n` 방식의 실비용은 **출력 토큰 2배 추가**가 전부다. 사실형 짧은답 기준
  턴당 +300~600 토큰 — 이것은 감당 가능한 수준이다.
- 단, `sc_max_tokens=512` 캡이 전제다. knowledge_mode의 4096을 그대로 3배 하면
  최악 +8K 토큰으로, 과대 소비가 된다. **G3 게이트(짧은답 기대 질문만)와 512 캡이
  세트**로 있어야 이 분석이 성립한다.

### 4.2 지연(latency)

- vLLM은 n개 시퀀스를 같은 디코드 스텝에 배치하므로 벽시계 시간은 "가장 긴 표본
  1개 생성 시간 + 배치 오버헤드". 단일 대비 **+10~30% 수준으로 추정**(실측 필요).
- 실제 체감 지연의 주범은 따로 있다. **스트리밍 버퍼링**(§6) — 합의 전에는 출력을
  못 보여주므로, TTFT(첫 토큰까지 시간)가 "전체 생성 완료 시간"으로 늘어난다.
  300토큰 답변이 디코드 20~30 tok/s(5090 27B INT4 추정)라면 **10~15초 무출력** 후
  답이 나온다. 이것이 SC의 가장 큰 UX 비용이다.

### 4.3 VRAM (5090)

- 5090 실측: max-model-len 8192에서 잔여 VRAM ~1GB. n=3의 추가 부담은 **출력 구간
  KV cache × 2 시퀀스**뿐이다(프롬프트 KV는 공유). 512토큰 × 2면 수십 MB 수준으로
  OOM 위험은 낮다 — 그러나 동시 사용자가 있으면 그만큼 배치 슬롯을 뺏는다.
- B200(49K 컨텍스트, VRAM 여유 큼)에서는 이 항목이 제약이 아니다.

### 4.4 언제 켤 가치가 있나 / 기본 off 근거

**켤 가치가 있는 조건** (모두 해당할 때):
1. 수치·연도·고유명사 정답률이 중요한 KB QA 용도(평가/데모/시험 대비형 서비스)
2. 동시 사용자 수가 적어 처리량 손실을 감내 가능
3. B200급 여유 하드웨어이거나, 5090이라도 단일 사용자

**기본 off 권장 근거** (정직한 판단):
1. 토큰 과대추정 금지 원칙 — 상시 3배 출력은 5090 예산에서 정당화 불가
2. 실사용 KNOWLEDGE 질의의 상당수가 서술형이라 SC 이득이 없는데 게이트 오탐 시
   비용만 발생
3. TTFT 10초+ 는 대화형 UX에 치명적 — 사용자가 hang으로 오인할 수 있음
4. 다중 사용자 서빙 처리량 저하 (n배 디코드 슬롯 점유)
5. 체계적 오답(모델이 일관되게 틀리는 지식)은 SC로 못 잡는다 — 지식 정확도의
   근본 해법은 KB RAG 품질(이미 리랭커 등으로 투자 중)이며 SC는 보조 수단이다

---

## 5. Config 추가안 + Pydantic 변경

### 5.1 config/nexus_config.yaml (routing 섹션 하위)

```yaml
routing:
  # ... 기존 항목 유지 ...
  self_consistency:
    enabled: false            # 기본 비활성 (fail-closed 관례 — P6)
    n: 3                      # 표본 수. 2는 동률 불가피, 5는 비용 과다 → 3 고정 권장
    temperature: 0.7          # SC 전용 온도 (knowledge_mode 0.2로는 표본 다양성 없음)
    top_p: 0.95
    max_tokens: 512           # 표본당 출력 캡 — 비용 분석(§4.1)의 전제
    apply_classes: ["KNOWLEDGE"]   # 적용 질의 클래스 (TOOL/CHAT 금지)
    factual_gate: true        # G3 사실형 패턴 게이트 on/off
    factual_max_question_chars: 120
    short_answer_max_chars: 80     # 이 이하면 exact majority, 초과 시 임베딩 클러스터
    min_agreement: 2               # 최소 합의 표 (N=3의 과반)
    similarity_threshold: 0.90     # 서술형 임베딩 클러스터 임계 (폴백 경로)
```

하드코딩 금지(anti-patterns #4): 위 값 전부 yaml이 단일 소스이고 Pydantic 기본값은
yaml 누락 시 폴백(KnowledgeRagConfig와 동일 관례, config.py:978 주석 참조).

### 5.2 core/config.py

```python
class SelfConsistencyConfig(BaseModel):
    """자기일관성(Self-Consistency) 사실 검증 설정. 기본 비활성(fail-closed)."""
    enabled: bool = False
    n: int = Field(default=3, ge=2, le=5)   # 상한 5 — 예산 보호 밸리데이션
    temperature: float = 0.7
    top_p: float = 0.95
    max_tokens: int = Field(default=512, le=2048)
    apply_classes: list[str] = Field(default_factory=lambda: ["KNOWLEDGE"])
    factual_gate: bool = True
    factual_max_question_chars: int = 120
    short_answer_max_chars: int = 80
    min_agreement: int = 2
    similarity_threshold: float = 0.90

class RoutingConfig(BaseModel):
    # ... 기존 필드 ...
    self_consistency: SelfConsistencyConfig = Field(
        default_factory=SelfConsistencyConfig
    )
```

### 5.3 RoutingDecision (frozen dataclass) 필드 추가

기존 관례(top_p 등 "기본값=비활성, 뒤쪽 배치") 그대로.

```python
@dataclass(frozen=True)
class RoutingDecision:
    # ... 기존 필드 ...
    sc_n: int = 1                    # 1 = SC 비활성 (하위 호환 기본값)
    sc_min_agreement: int = 2
    sc_short_answer_max_chars: int = 80
    sc_similarity_threshold: float = 0.90
```

`RoutingResolver.resolve()`가 G1~G3 통과 시 `sc_n=config.n`으로 채우고, SC 적용 턴은
temperature/top_p/max_tokens_cap도 SC 값으로 치환한다(프로필 치환과 동일 지점에서 처리
— 파라미터 결정이 한 곳에 모이는 현행 구조 유지).

### 5.4 시그니처 전파 (passthrough)

`query_loop(..., sc_decision: ...)` / `dispatcher.route(...)` / `stream(..., n: int = 1)` —
전부 기본값이 "비활성"이므로 기존 호출부·테스트는 무수정 동작(무회귀). 이는 top_p 등
샘플링 파라미터 추가 때 쓴 것과 동일한 전파 패턴이다.

---

## 6. 스트리밍 UX

### 6.1 문제

N표본 중 어느 것도 "확정 답"이 아니므로 라이브 스트림할 대상이 없다. 선택지 비교.

| 방식 | 내용 | 판정 |
|---|---|---|
| (a) 표본 0을 라이브 스트림, 합의가 다르면 교체 | TTFT 최선 | 기각 — 이미 보여준 텍스트가 바뀌는 UX는 신뢰 파괴적. 웹 프로토콜에 "텍스트 철회" 이벤트도 없음 |
| (b) 전량 버퍼 → 합의 → 승자 의사-스트림 | 구현 단순, 추가 토큰 0 | **채택** |
| (c) 합의 후 승자 기반 재생성 | 스트림 자연스러움 | 기각 — 생성 1회 추가(+T 토큰), 재생성이 승자와 달라질 위험. 예산 원칙 위반 |

### 6.2 채택안 (b) 상세

1. SC 턴 시작 시 `SYSTEM_INFO` 이벤트("사실 교차 검증 중 (3표본)")를 즉시 yield —
   웹 UI가 활동 표시로 쓸 수 있고, 무출력 구간의 hang 오인을 막는다.
   (신규 이벤트 타입 불필요 — 기존 SYSTEM_INFO 재사용, message.py:105)
2. Tier 3가 전 표본 완료까지 버퍼링(§2.3), Tier 2가 합의 확정.
3. 승자 텍스트를 적당한 크기(예: 40자)로 잘라 TEXT_DELTA 연속 yield(의사-스트림) —
   상위 소비자(웹 SSE/CLI Rich)는 일반 턴과 구분 없이 렌더링한다. 체인 계약 불변.
4. usage 이벤트에는 **3표본 전체 출력 토큰을 정직하게 합산** 보고한다(과소 계상 금지).

---

## 7. 구현 단계 + 테스트 계획

기존 규칙(agent-collaboration.md) 준수: 한글 주석, 단계마다 ruff + pytest.

| Phase | 작업 | 검증 |
|---|---|---|
| 1 | `SelfConsistencyConfig` + yaml + RoutingDecision 필드 + RoutingResolver 게이팅(G1~G3) | 단위: 게이트 3종 조합, TOOL/CHAT 미적용, enabled=false 무영향 |
| 2 | `core/orchestrator/self_consistency.py` — normalize_answer / majority_vote / cluster_by_embedding (순수 함수) | 단위: 숫자 콤마·단위·마크다운 정규화, 2:1 다수결, 3분열 합의실패, 임베딩 medoid |
| 3 | Tier 3 `stream(n=)` — choices index 디멀티플렉스 + 버퍼링 + SC_CANDIDATE 이벤트 | 단위: n>1 SSE fixture 파싱. **n=1 기존 fixture 전체 무회귀** |
| 4 | Tier 2 query_loop 통합 — 후보 수집→합의→의사 스트림, tool_calls 발생 시 SC 포기 폴백 | 통합: mock provider로 아래 시나리오 |
| 5 | dispatcher.route() passthrough + 웹 e2e | e2e: 웹 SSE로 SYSTEM_INFO→TEXT_DELTA 순서 확인 |
| 6 | **실측 벤치(B200)** — kowiki 사실형 QA 셋으로 on/off 정확도·지연·토큰 비교 | §4 추정치를 실측으로 교체, 결과를 progress.md에 기록 |

**mock provider 합의 검증 시나리오** (testing.md의 AsyncGenerator 소비 패턴 준수,
`async for`로 전 이벤트 수집).

```
test_sc_majority_two_of_three_wins        # ["1443년","1443년","1446년"] → "1443년"
test_sc_numeric_normalization_same_vote   # ["5,100만 명","약 5100만명","4800만"] → 5100만 그룹 승
test_sc_all_disagree_falls_back_first     # 3분열 → 후보0 + SYSTEM_WARNING
test_sc_tool_call_aborts_to_single        # 표본에 TOOL_USE_START → SC 포기, 정상 루프 계속
test_sc_disabled_no_behavior_change       # enabled=false → stream(n=1), 이벤트 시퀀스 기존과 동일
test_sc_tool_class_never_applies          # TOOL 분류 질의는 enabled=true여도 n=1
test_stream_n1_regression                 # Tier3 n=1 경로 기존 fixture 전체 통과
```

외부 서비스는 전부 mock(vLLM SSE fixture에 multi-choice 청크 추가) — 단, 최종 검증은
사용자 원칙에 따라 실서버(B200 vLLM)에서 Phase 6으로 수행한다.

---

## 8. 리스크 및 사양 대조 결과

### 8.1 리스크 (정직 목록)

| 리스크 | 심각도 | 완화 |
|---|---|---|
| 토큰 3배 소비가 게이트 오탐으로 서술형에 발동 | 중 | G3 사실형 게이트 + max_tokens 512 캡 + 기본 off |
| TTFT 10초+ (버퍼링) → hang 오인 | 중 | SYSTEM_INFO 즉시 표출, 사실형 짧은답 한정으로 절대 시간 축소 |
| 체계적 오답(3표본이 같은 오답) — SC 무력 | 높음(원리적 한계) | 완화 불가. KB RAG가 1차 방어선임을 명시, SC는 보조 수단 |
| 서술형 합의 부적합 | 높음(원리적 한계) | 기본 정책은 서술형 제외(G3). 임베딩 클러스터는 실험 폴백 |
| temperature 0.7 상향으로 개별 표본 품질 저하 | 중 | 투표가 상쇄(원 논문 근거). 합의 실패 시 로그 관측으로 튜닝 |
| vLLM `n`+stream 파서 회귀 | 중 | n=1 경로 코드 분기 보존 + 회귀 테스트(Phase 3) |
| 다중 사용자 처리량 저하 | 중 | 기본 off. 활성 시 운영 문서에 동시성 영향 명시 |
| SC_CANDIDATE 이벤트 타입 추가로 이벤트 계약 확장 | 낮음 | 소비자는 미지 타입 무시(현행 스위치 방식), frozen 규칙 준수 |

### 8.2 사양 대조 (필수 항목)

`user_mig/PROJECT_NEXUS_SPEC_v7.1_AMENDMENT.md`, `v7.2_AMENDMENT.md`를
self-consistency / majority / voting / 투표 / 합의 / 다수결 / 앙상블 / multi-sample
키워드로 grep한 결과 **일치 항목 없음** (2026-07-08 확인).

- v7.1은 Thinking Engine을 "5전략, Worker 모델에서만 사용"으로 기술(Ch 11 요약,
  v7.1:55)하며 두 사양 모두 "Thinking 챕터 변경 없음"을 명시한다. Self-Consistency는
  이 5전략에 포함되지 않은 **신규 기능**이다.
- 따라서 본 설계는 **사양 이탈(신규 확장)**이며, 구현 착수 전 차기 AMENDMENT
  (v7.4 또는 해당 시점 버전)에 다음 근거로 개정 항목을 추가해야 한다.
  - 근거 1: 사실형 QA 수치 오답은 KB RAG만으로 못 막는 디코딩 노이즈 문제
  - 근거 2: 4-Tier 체인·기존 5전략·권한/도구 파이프라인에 구조 변경 없음
    (passthrough 파라미터 + 이벤트 타입 1개 추가가 전부)
  - 근거 3: 기본 비활성 fail-closed로 기존 배포 동작 100% 불변
- 기존 코드와의 정합: knowledge_mode 프로필 관례(config.py), 샘플링 파라미터
  passthrough 관례(routing.py→inference.py), fail-closed 기본값(P6), 에어갭
  (임베딩도 LAN 8002만 사용) 모두 준수.

---

## 9. 권장안 (요약)

1. **기본 off** (`enabled: false`) — 5090 토큰 예산과 TTFT 비용상 상시 활성은 정당화
   불가. 평가/데모/고정밀 KB QA 시나리오에서만 명시적으로 켠다.
2. **N=3 고정 권장** (config상 2~5 허용하되 3 외 값은 비권장 주석) — 2는 동률,
   5는 비용 과다.
3. **KNOWLEDGE 한정 + 사실형 패턴 게이트(G3)** — 서술형·TOOL·CHAT은 원리적으로
   부적합하므로 발동 자체를 차단한다.
4. **vLLM `n` 파라미터 1요청 방식** — prefill 공유로 실비용을 "출력 토큰 2배 추가"로
   억제. 반복 호출·Tier 1 감싸기는 prefill N배라 기각.
5. **배치는 Tier 2(query_loop) + Tier 3(stream n 확장)** — 4-Tier 체인을 우회하지 않고
   기존 passthrough 관례로 파라미터만 흘린다. 합의는 순수 함수 모듈로 분리.
6. 구현 전 **차기 사양 AMENDMENT에 개정 항목 등재**(§8.2), 구현 후 **B200 실측
   벤치로 §4 추정치를 교체**하는 것을 완료 조건으로 한다.
