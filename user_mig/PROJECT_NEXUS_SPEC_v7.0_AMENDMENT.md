# Project Nexus — 기술 사양서 v7.0 설계 개정안

## 적응형 멀티모델 오케스트레이션 아키텍처

**버전**: 7.0 (Adaptive Multi-Model Amendment)
**기준 문서**: PROJECT_NEXUS_SPEC_v6.1_EN.md (22개 챕터, ~510KB)
**개정일**: 2026-04-16
**상태**: 설계 검토 중

---

## 개정 개요

### 왜 v7.0인가

v6.1은 Claude Code(200K~1M 컨텍스트, 클라우드 API)의 구조를 그대로 복제한다.
그러나 Nexus의 실행 환경(RTX 5090, 8,192 토큰)에서는 이 구조가 물리적 한계에 부딪힌다.

v7.0은 **v6.1의 설계 원칙을 100% 유지**하면서, 실행 방식만 하드웨어에 적응시킨다.
GPU가 업그레이드되면 자동으로 v6.1 원래 동작에 수렴한다.

### 3가지 불변 전제

1. **Claude Code의 설계안을 그대로 가져온다** — 4-Tier 체인, 24개 도구, 5계층 권한, Hook, Thinking, Memory 전부 유지
2. **GPU 업그레이드 시 성능이 더 좋아진다** — 강한 GPU일수록 분리가 합쳐지며 원래 구조에 수렴
3. **Claude Code의 형태로 AI가 작동한다** — 사용자 체험은 동일 (도구 자율 선택, 스트리밍 응답, 멀티턴 대화)

### 변경 범위 요약

| 구분 | 챕터 | 변경 수준 |
|---|---|---|
| **신규** | — | Ch 2A (적응형 오케스트레이션), Ch 6A (상태 외부화), Ch 4A (CPU 모델 서빙) |
| **수정** | Ch 2, 3, 4, 5, 14, 22 | 멀티모델 디스패치, 부트스트랩, 로드맵 |
| **경미 수정** | Ch 6, 7, 15, 16, 17, 20 | 인터페이스 조정 |
| **변경 없음** | Ch 1, 8, 9, 10, 11, 12, 13, 18, 19, 21 | 그대로 유지 |

---

## Part 0: 변경 없는 모듈 (확인)

아래 챕터는 v7.0에서 **코드 변경이 전혀 없다**.

| 챕터 | 모듈 | 이유 |
|---|---|---|
| Ch 1 | Executive Summary | 목표·제약·원칙 불변 |
| Ch 8 | 5계층 Permission Pipeline | 도구 실행 전 검증. 어떤 모델이 호출하든 동일하게 적용 |
| Ch 9 | Security System | PathGuard, CommandFilter, 30+ 패턴. 모델과 무관 |
| Ch 10 | Hook System | 12+3 이벤트, HookDecision. 도구 실행 파이프라인에 연결. 모델과 무관 |
| Ch 11 | Thinking Engine | 5전략 (DIRECT~MULTI_AGENT). Worker 모델에서만 사용. 인터페이스 불변 |
| Ch 12 | Memory System | 5종 메모리, Redis/pgvector. 모델과 독립적 |
| Ch 13 | 24 Tools (구현체) | BaseTool 인터페이스, 각 도구의 call() 로직. 모델과 무관 |
| Ch 18 | Training Pipeline | 5-Phase 전략, LoRA/QLoRA. 학습 대상 모델이 바뀌지 않음 |
| Ch 19 | Air-Gap Strategy | 오프라인 패키지, 무결성 검증. 인프라 계층 |
| Ch 21 | Project Structure | 디렉토리 구조 유지 (신규 파일만 추가) |

---

## Part 1: 신규 — 적응형 하드웨어 티어 시스템

### 1.1 하드웨어 티어 정의

v6.1의 GPU Tier(Ch 4.1)를 확장하여, 오케스트레이션 모드를 자동 결정한다.

```python
class HardwareTier(str, Enum):
    """하드웨어 티어별 오케스트레이션 모드를 결정한다."""

    TIER_S = "small"     # RTX 5090 (32GB), 8K 컨텍스트
    TIER_M = "medium"    # H100 (80GB), 32K 컨텍스트
    TIER_L = "large"     # H200 (141GB) / GB10 (128GB), 128K 컨텍스트


# 티어별 오케스트레이션 모드 자동 매핑
TIER_ORCHESTRATION = {
    HardwareTier.TIER_S: "multi_model",   # Scout(CPU) + Worker(GPU) 분리
    HardwareTier.TIER_M: "single_model",  # Worker(GPU) 단독, 도구 16~24개
    HardwareTier.TIER_L: "single_model",  # Worker(GPU) 단독, 도구 24개 + 긴 히스토리
}
```

### 1.2 티어별 자원 배분

```
TIER_S (RTX 5090 32GB + CPU 64GB RAM):
  ┌─ GPU ─────────────────────────────┐
  │ Worker: Gemma 4 31B (INT4, 20GB)  │
  │ Embedding: e5-large (0.7GB)       │
  │ vLLM 오버헤드 (~11GB)             │
  └───────────────────────────────────┘
  ┌─ CPU ─────────────────────────────┐
  │ Scout: Gemma 4 4B (INT4, 3.5GB)  │  ← 신규
  │ llama.cpp 서빙 (:8003)           │
  └───────────────────────────────────┘

  Worker 컨텍스트: 8,192 토큰
  Worker 도구: 실행 도구만 (Edit, Write, Bash 등 5~7개, ~1,500 토큰)
  Scout 컨텍스트: 4,096 토큰 (CPU 모델이므로 작게)
  Scout 도구: 읽기 전용 (Read, Glob, Grep, LS, 4개, ~800 토큰)

TIER_M (H100 80GB):
  ┌─ GPU ─────────────────────────────┐
  │ Worker: Gemma 4 31B (BF16, 62GB) │
  │ Embedding: e5-large (0.7GB)       │
  └───────────────────────────────────┘
  Scout 불필요 — Worker가 전부 처리
  Worker 컨텍스트: 32,768 토큰
  Worker 도구: 24개 전체 (~6,102 토큰, 컨텍스트의 19%)

TIER_L (H200 141GB):
  ┌─ GPU ─────────────────────────────┐
  │ Worker: Gemma 4 31B (BF16, 62GB) │
  │ Auxiliary: ExaOne 32B (동시 로드) │
  │ Embedding: e5-large (0.7GB)       │
  └───────────────────────────────────┘
  Worker 컨텍스트: 131,072 토큰
  Worker 도구: 24개 전체 (~6,102 토큰, 컨텍스트의 5%)
  → v6.1 원래 설계와 동일하게 동작
```

### 1.3 핵심 원칙: 상위 티어는 하위 티어의 상위집합

```
TIER_L ⊃ TIER_M ⊃ TIER_S

TIER_S에서 동작하는 모든 것은 TIER_M, TIER_L에서도 동작한다.
TIER_M/L에서는 Scout 분리가 불필요하므로 Worker 단독 모드로 전환된다.
코드 분기 없이 설정(config)만으로 전환된다.
```

---

## Part 2: 신규 — 멀티모델 디스패치 계층

### 2.1 아키텍처 변경: 4-Tier 체인 내부에 디스패치 삽입

v6.1의 4-Tier 체인 구조를 **그대로 유지**하되, Tier 1(QueryEngine) 내부에 모델 디스패치 로직을 추가한다.

```
v6.1 (기존):
  Tier 1: QueryEngine.submit_message()
    └─ Tier 2: query_loop() ─── 단일 ModelProvider
        └─ Tier 3: model_provider.stream()
            └─ Tier 4: httpx (GPU 서버)

v7.0 (TIER_S):
  Tier 1: QueryEngine.submit_message()
    └─ ModelDispatcher.route()          ← 신규 계층
        ├─ Scout용 query_loop()  ─── ScoutModelProvider (CPU :8003)
        │   └─ 도구 4개, 읽기 전용
        │   └─ 결과: TurnState (탐색 결과 + 계획)
        │
        └─ Worker용 query_loop() ─── WorkerModelProvider (GPU :8001)
            └─ 도구 5~7개, 실행
            └─ 입력: TurnState에서 필요한 정보만

v7.0 (TIER_M/L):
  Tier 1: QueryEngine.submit_message()
    └─ ModelDispatcher.route()          ← 동일 계층, 단순 통과
        └─ Worker용 query_loop() ─── WorkerModelProvider (GPU :8001)
            └─ 도구 24개 전체
            └─ v6.1과 동일하게 동작
```

**핵심**: ModelDispatcher는 TIER_M/L에서는 **단순 통과(passthrough)**다.
Scout를 거치지 않고 바로 Worker의 query_loop()을 호출한다.
즉 TIER_M/L에서는 v6.1과 100% 동일한 코드 경로를 탄다.

### 2.2 ModelDispatcher 설계

```python
class ModelDispatcher:
    """
    하드웨어 티어에 따라 Scout/Worker를 분배하는 디스패처.

    TIER_S: Scout(탐색) → Worker(실행) 2단계
    TIER_M/L: Worker 단독 (v6.1 동일)
    """

    def __init__(
        self,
        tier: HardwareTier,
        worker_provider: ModelProvider,      # GPU 31B
        scout_provider: ModelProvider | None, # CPU 4B (TIER_S만)
        worker_tools: list[BaseTool],
        scout_tools: list[BaseTool] | None,
        turn_state_store: TurnStateStore,
    ):
        ...

    async def route(
        self,
        user_input: str,
        messages: list[Message],
        context: ToolUseContext,
    ) -> AsyncGenerator[StreamEvent | Message, None]:
        """
        사용자 입력을 적절한 모델에 라우팅한다.

        TIER_S 흐름:
          1. Scout에게 탐색/계획 요청 (도구 4개)
          2. Scout 결과를 TurnState에 저장
          3. Worker에게 TurnState 기반 실행 요청 (도구 5~7개)
          4. Worker 결과를 TurnState에 업데이트

        TIER_M/L 흐름:
          1. Worker에게 직접 전달 (v6.1 동일)
        """
        if self._tier == HardwareTier.TIER_S and self._scout_provider:
            # Phase A: Scout 탐색
            scout_state = await self._run_scout(user_input, context)

            # Phase B: Worker 실행 (Scout 결과 기반)
            async for event in self._run_worker(user_input, scout_state, context):
                yield event
        else:
            # TIER_M/L: 단일 모델 직행 (v6.1 동일 경로)
            async for event in self._run_worker_direct(user_input, messages, context):
                yield event
```

### 2.3 Scout 역할 상세

Scout는 **읽기 전용 탐색 에이전트**다. 절대 파일을 수정하지 않는다.

```python
# Scout에 할당되는 도구 (TIER_S 전용)
SCOUT_TOOLS = ["Read", "Glob", "Grep", "LS"]

# Scout의 시스템 프롬프트
SCOUT_SYSTEM_PROMPT = """
You are a Scout agent. Your job is to explore and plan, NOT to execute.

Given the user's request:
1. Use Read/Glob/Grep/LS to find relevant files
2. Identify which files need to be modified
3. Output a structured plan in JSON:

{
  "relevant_files": ["path1", "path2"],
  "file_summaries": {"path1": "짧은 요약", "path2": "짧은 요약"},
  "plan": "What the Worker should do",
  "requires_tools": ["Edit", "Bash"]
}

Do NOT attempt to edit or create files. Only read and plan.
Respond in the user's language.
"""
```

Scout의 출력은 **구조화된 JSON**이다. 자유 텍스트가 아니라 정형화된 계획서를 생성한다.
이렇게 하면 Worker에게 전달할 때 토큰을 최소화할 수 있다.

### 2.4 Worker 역할 (TIER_S)

Worker는 Scout의 계획을 받아 **실행만 담당**한다.

```python
# Worker에 할당되는 도구 (TIER_S)
WORKER_TOOLS_TIER_S = ["Edit", "Write", "Bash", "GitCommit", "GitDiff"]

# Worker의 시스템 프롬프트 (TIER_S)
WORKER_SYSTEM_PROMPT_TIER_S = """
You are a Worker agent. Execute the plan provided by Scout.

Context from Scout:
{scout_plan}

Relevant file contents (already read by Scout):
{file_summaries}

Execute the plan using your tools. Do not re-read files that Scout already summarized.
Respond in the user's language.
"""
```

**핵심 최적화**: Scout가 이미 읽은 파일 내용을 요약으로 전달한다.
Worker는 파일을 다시 읽지 않으므로 Read 도구가 필요 없다.
이로써 Worker의 컨텍스트를 **실행에만 집중**시킨다.

### 2.5 Worker 역할 (TIER_M/L)

TIER_M/L에서 Worker는 **Scout 없이 단독**으로 동작한다.
v6.1의 query_loop()과 완전히 동일하다.

```python
# Worker에 할당되는 도구 (TIER_M)
WORKER_TOOLS_TIER_M = ALL_24_TOOLS  # 전체 24개

# Worker에 할당되는 도구 (TIER_L)
WORKER_TOOLS_TIER_L = ALL_24_TOOLS  # 전체 24개

# Worker의 시스템 프롬프트 (TIER_M/L)
WORKER_SYSTEM_PROMPT_TIER_ML = _build_default_system_prompt()  # v6.1 동일
```

---

## Part 3: 신규 — 상태 외부화 (TurnState)

### 3.1 문제: raw messages 누적

v6.1에서 query_loop()은 messages[] 배열에 모든 턴의 메시지를 누적한다.
8,192 토큰 환경에서 2~3턴 만에 컨텍스트가 꽉 찬다.

### 3.2 해결: TurnState 외부 저장소

매 턴이 끝나면 핵심 정보만 추출하여 외부에 저장하고,
다음 턴에는 raw messages 대신 TurnState 요약만 컨텍스트에 넣는다.

```python
@dataclass(frozen=True)
class TurnState:
    """
    한 턴의 핵심 정보를 요약한 불변 객체.

    raw messages 대신 이 요약을 다음 턴에 전달한다.
    """

    # 확인된 사실 (파일 존재, 내용 요약 등)
    facts: list[str]

    # 남은 할 일
    todo: list[str]

    # 접근한 파일 목록
    touched_files: list[str]

    # 미해결 문제
    unresolved_issues: list[str]

    # Scout의 계획 (TIER_S에서만 사용)
    scout_plan: str | None = None

    # 직전 도구 실행 결과 요약
    last_tool_results: list[str] | None = None

    # 턴 번호
    turn_number: int = 0

    def to_context_string(self) -> str:
        """
        TurnState를 컨텍스트 문자열로 변환한다.
        이 문자열이 시스템 프롬프트에 주입된다.
        """
        parts = []
        if self.facts:
            parts.append("확인된 사실:\n" + "\n".join(f"- {f}" for f in self.facts))
        if self.todo:
            parts.append("남은 할 일:\n" + "\n".join(f"- {t}" for t in self.todo))
        if self.touched_files:
            parts.append("접근한 파일: " + ", ".join(self.touched_files))
        if self.unresolved_issues:
            parts.append("미해결: " + ", ".join(self.unresolved_issues))
        if self.last_tool_results:
            parts.append("직전 결과:\n" + "\n".join(f"- {r}" for r in self.last_tool_results))
        return "\n\n".join(parts)

    def estimated_tokens(self) -> int:
        """요약의 토큰 수를 추정한다."""
        return len(self.to_context_string()) // 3
```

### 3.3 TurnStateStore

```python
class TurnStateStore:
    """
    TurnState를 세션 단위로 저장/조회한다.

    인메모리 딕셔너리로 구현 (Redis 확장 가능).
    """

    def __init__(self):
        self._states: dict[str, list[TurnState]] = {}  # session_id → 턴별 상태

    def save(self, session_id: str, state: TurnState) -> None:
        """턴 상태를 저장한다."""
        if session_id not in self._states:
            self._states[session_id] = []
        self._states[session_id].append(state)

    def get_latest(self, session_id: str) -> TurnState | None:
        """가장 최근 턴 상태를 반환한다."""
        states = self._states.get(session_id, [])
        return states[-1] if states else None

    def get_context(self, session_id: str, max_tokens: int = 1000) -> str:
        """
        토큰 예산 내에서 최근 턴 상태들을 컨텍스트 문자열로 반환한다.
        최신 것부터 역순으로 예산이 허용하는 만큼 포함한다.
        """
        states = self._states.get(session_id, [])
        result_parts = []
        used_tokens = 0

        for state in reversed(states):
            text = state.to_context_string()
            tokens = state.estimated_tokens()
            if used_tokens + tokens > max_tokens:
                break
            result_parts.append(f"[턴 {state.turn_number}]\n{text}")
            used_tokens += tokens

        result_parts.reverse()
        return "\n---\n".join(result_parts)
```

### 3.4 적용: TIER_S vs TIER_M/L

```
TIER_S:
  매 턴 → TurnState 생성 → 저장
  다음 턴 → TurnState 요약만 컨텍스트에 포함 (~200~500 토큰)
  → raw messages 누적 없음
  → 수십 턴 대화 가능

TIER_M (32K):
  v6.1 방식 (raw messages 누적) 그대로 사용 가능
  TurnState는 보조적으로 생성 (세션 메타데이터용)
  → 10~20턴에서 컨텍스트 관리(Ch 6) 압축 파이프라인 사용

TIER_L (128K):
  v6.1 방식 그대로
  TurnState는 메타데이터 목적만
  → 수백 턴까지 raw messages로 충분
```

---

## Part 4: 신규 — CPU 모델 서빙 (llama.cpp)

### 4.1 GPU 서버 구성 변경

v6.1의 2-Machine 토폴로지(Ch 2, P4)에 CPU 모델 서빙을 추가한다.

```
Machine B (GPU 서버, 192.168.22.28):
  CPU: Intel Core Ultra 9 285K (24코어) + 64GB DDR5
  GPU: RTX 5090 (32GB VRAM)

  [GPU] vLLM :8001 — Gemma 4 31B (INT4)           ← 기존 (Worker)
  [GPU] vLLM :8002 — e5-large 임베딩               ← 기존
  [CPU] llama.cpp :8003 — Gemma 4 4B (INT4)        ← 신규 (Scout)
```

### 4.2 ScoutModelProvider

```python
class ScoutModelProvider(ModelProvider):
    """
    llama.cpp OpenAI 호환 API를 통한 Scout 모델 프로바이더.

    CPU에서 실행되는 작은 모델(4B)로, 탐색과 계획만 담당한다.
    GPU VRAM에 영향을 주지 않는다.

    llama.cpp의 --host, --port, --api-key 옵션으로 서빙한다.
    OpenAI 호환 API이므로 LocalModelProvider와 동일한 인터페이스를 사용한다.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8003",
        model_id: str = "gemma-4-4b-it",
        max_context_tokens: int = 4096,
        max_output_tokens: int = 512,  # Scout는 짧은 출력
    ):
        # LocalModelProvider를 내부적으로 재사용
        # llama.cpp도 OpenAI 호환 API를 제공하므로 동일한 코드
        self._inner = LocalModelProvider(
            base_url=base_url,
            model_id=model_id,
            max_context_tokens=max_context_tokens,
            max_output_tokens=max_output_tokens,
        )
```

### 4.3 llama.cpp 서빙 설정

```bash
# GPU 서버에서 llama.cpp 실행 (TIER_S 전용)
./llama-server \
  --model ./models/gemma-4-4b-it-Q4_K_M.gguf \
  --host 0.0.0.0 \
  --port 8003 \
  --ctx-size 4096 \
  --threads 8 \
  --batch-size 512 \
  --api-key local-key

# 예상 리소스 사용:
#   RAM: ~3.5GB
#   CPU: 8 스레드 (Ultra 9 285K의 24코어 중 1/3)
#   성능: ~15~30 tokens/s (Scout 응답 50~100토큰 → 2~3초)
```

### 4.4 TIER_M/L에서의 Scout

TIER_M/L에서는 Scout가 **존재하지 않는다**.
llama.cpp 서버를 시작하지 않고, ModelDispatcher가 Worker로 직접 라우팅한다.

```python
# config/nexus_config.yaml
hardware:
  tier: "auto"  # auto: GPU VRAM 기반 자동 감지

scout:
  enabled: true           # TIER_S에서만 자동 활성화
  base_url: "http://192.168.22.28:8003"
  model_id: "gemma-4-4b-it"
  max_context_tokens: 4096
  max_output_tokens: 512
```

---

## Part 5: 수정 — 기존 챕터 변경 사항

### Ch 2: System Architecture (수정)

**변경**: 4-Tier 체인 내부에 ModelDispatcher 추가 (Part 2 참조)
**유지**: 4-Tier 구조 자체, StreamEvent 기반 스트리밍, 의존성 방향

### Ch 3: Bootstrap & Initialization (수정)

**Phase 2 초기화에 추가**:

```python
async def init_phase2(state: GlobalState) -> dict:
    # ① ~ ④: 기존과 동일 (ModelProvider, ToolRegistry, Memory, TaskManager)

    # ⑤ 하드웨어 티어 감지 (신규)
    tier = detect_hardware_tier(state.config)

    # ⑥ Scout 초기화 (TIER_S 전용, 신규)
    scout_provider = None
    scout_tools = None
    if tier == HardwareTier.TIER_S and state.config.scout.enabled:
        scout_provider = ScoutModelProvider(
            base_url=state.config.scout.base_url,
            model_id=state.config.scout.model_id,
        )
        scout_tools = _create_scout_tool_registry().get_all_tools()

    # ⑦ TurnStateStore 생성 (신규)
    turn_state_store = TurnStateStore()

    # ⑧ ModelDispatcher 생성 (신규)
    dispatcher = ModelDispatcher(
        tier=tier,
        worker_provider=provider,
        worker_tools=_get_worker_tools(tier, registry),
        context=context,
        scout_provider=scout_provider,
        scout_tools=scout_tools,
        max_turns=200,
    )
    components["model_dispatcher"] = dispatcher

    # ⑨ QueryEngine (수정 — model_dispatcher 주입)
    # 하위 호환: model_provider와 model_dispatcher를 둘 다 받는다.
    # dispatcher가 주입되면 submit_message()는 dispatcher.route()를 호출하고,
    # dispatcher가 None이면 기존 query_loop() 직접 호출 경로로 폴백한다.
    # 이렇게 하면 기존 코드(구 QueryEngine(model_provider=...))를 깨지 않는다.
    engine = QueryEngine(
        model_provider=provider,              # 폴백 경로용
        tools=registry.get_all_tools(),
        context=context,
        system_prompt=_build_default_system_prompt(),
        max_turns=200,
        turn_state_store=turn_state_store,
        rag_retriever=rag_retriever,
        model_dispatcher=dispatcher,          # 신규: 실제 라우팅은 dispatcher가 담당
    )
```

### Ch 4: Model Configuration & GPU Server (수정)

**추가 사항**:
- llama.cpp 서빙 설정 (Part 4 참조)
- GPU Tier 감지 로직에 CPU RAM 확인 추가
- config/nexus_config.yaml에 scout 섹션 추가

**변경 없음**:
- vLLM 설정, 모델 프로필, LoRA hot-loading, 헬스 모니터링

### Ch 5: Core Framework (수정)

**5.8 query_loop (수정)**:
- TIER_S: Scout용 query_loop와 Worker용 query_loop를 분리 실행
- TIER_M/L: 기존 query_loop 그대로 (변경 없음)
- query_loop 함수 자체의 인터페이스는 변경 없음 (ModelProvider를 받는 구조 유지)

**5.9 QueryEngine (수정)**:
- 생성자에 `model_dispatcher: ModelDispatcher | None = None` 파라미터 추가
  (기존 `model_provider` 파라미터는 **폴백 경로용으로 유지**)
- submit_message()의 분기:
  - `model_dispatcher`가 주입되면 → `dispatcher.route(messages, system_prompt, on_turn_complete)` 호출
  - `model_dispatcher`가 None이면 → 기존 `query_loop(...)` 직접 호출 (하위 호환)
- TIER_M/L에서는 ModelDispatcher가 passthrough이므로 v6.1과 동일 동작
- `model_dispatcher` 프로퍼티를 노출하여 `/metrics` 엔드포인트가 Scout 통계에 접근한다

**변경 없음**: Message 타입, BaseTool, ToolRegistry, Executor, StreamingToolExecutor

### Ch 6: Context Management (수정)

**TIER_S**: 4단계 압축 파이프라인 대신 TurnState 기반 상태 외부화 (Part 3 참조)
**TIER_M/L**: v6.1 4단계 압축 파이프라인 그대로 유지

```python
# ContextManager에 티어별 전략 추가
class ContextManager:
    def __init__(self, tier: HardwareTier, ...):
        if tier == HardwareTier.TIER_S:
            self._strategy = TurnStateStrategy(...)   # 상태 외부화
        else:
            self._strategy = CompressionStrategy(...)  # v6.1 4단계 압축
```

### Ch 7: Retry & Error Recovery (수정)

**추가**: Scout 연결 실패 시 Worker 단독 모드로 fallback
**변경 없음**: WithRetry, StreamWatchdog 로직 자체

```python
# Scout 실패 시 자동 fallback
class ModelDispatcher:
    async def _run_scout(self, ...):
        try:
            # Scout 실행
            ...
        except (httpx.ConnectError, httpx.ReadTimeout):
            # Scout 불가 → Worker 단독 모드로 전환
            logger.warning("Scout 연결 실패, Worker 단독 모드로 전환")
            self._tier = HardwareTier.TIER_M  # 런타임 티어 변경
            return None  # Scout 결과 없이 Worker 직행
```

### Ch 13: 도구 스키마 최적화 (경미 수정)

도구 구현체(call, check_permissions)는 변경 없음.
**스키마(description)만 축소**하여 토큰을 절약한다.

```python
# 기존 (v6.1)
@property
def description(self) -> str:
    return "디렉토리의 파일/폴더 목록을 표시합니다. 크기와 수정 시간을 포함합니다."

# v7.0 (축소)
@property
def description(self) -> str:
    return "List directory contents with size and modification time."

# 효과: 도구당 평균 ~50토큰 절약 → 11개 기준 ~550토큰 절약
```

### Ch 14: Agent & Task System (수정)

**Scout/Worker를 내장 에이전트로 정의**:

```python
# Scout 에이전트 정의 (TIER_S 전용)
SCOUT_AGENT = AgentDefinition(
    name="scout",
    description="읽기 전용 탐색 에이전트. 파일 검색과 계획 수립만 수행.",
    system_prompt=SCOUT_SYSTEM_PROMPT,
    tools=["Read", "Glob", "Grep", "LS"],
    max_turns=3,  # Scout는 최대 3턴
    model_override="scout",  # ScoutModelProvider 사용
)

# 기존 AgentRunner, TaskManager, Coordinator는 변경 없음
# DISALLOWED_TOOLS_FOR_AGENTS도 변경 없음
```

### Ch 15: State Management (경미 수정)

**GlobalState에 필드 추가**:

```python
# 신규 필드
hardware_tier: HardwareTier = HardwareTier.TIER_S
scout_enabled: bool = False
orchestration_mode: str = "multi_model"  # "multi_model" | "single_model"
```

### Ch 16: Session Management (경미 수정)

**TurnState를 세션 메타데이터에 포함**:

```python
# metadata.json에 추가
{
    "turn_states": [
        {"turn": 1, "facts": [...], "todo": [...], "touched_files": [...]},
        {"turn": 2, ...}
    ]
}
```

### Ch 17: Monitoring & Metrics (경미 수정)

**Scout 메트릭 추가**:

ModelDispatcher.stats 프로퍼티가 아래 필드를 노출한다. `web/app.py`의
`/metrics` 엔드포인트가 `_app_state["model_dispatcher"].stats`를 읽어
`result["scout"]`로 직렬화한다.

```python
# ModelDispatcher.stats 반환 필드
{
    "tier": "small" | "medium" | "large",
    "scout_enabled": bool,
    "scout_calls": int,              # Scout 호출 총 횟수
    "scout_fallback_count": int,     # Scout 실패 → Worker 폴백 횟수
    "scout_fallbacks": int,          # 하위 호환 alias (동일 값)
    "scout_avg_latency_ms": float,   # Scout 호출 1회당 평균 지연 (ms)
}
```

누적 지연 시간은 `_scout_total_latency_ms` 필드에 `_run_scout()` 성공 시
elapsed × 1000을 더하는 방식으로 저장하며, `stats` 계산 시점에 호출 수로
나누어 평균을 낸다 (호출 0회면 0.0).

### Ch 20: CLI & Web Interface (경미 수정)

**CLI/Web의 API 계층은 변경 없음**.
QueryEngine.submit_message()의 인터페이스가 동일하므로,
CLI와 Web은 내부 오케스트레이션 변경을 인지하지 못한다.

유일한 변경: 부트스트랩에서 scout_provider 초기화 추가.

---

## Part 6: 수정 — 개발 로드맵 (Ch 22)

### 기존 Phase 0.5~8.0 이후 추가 Phase

```
기존 (완료):
  Phase 0.5~8.0: v6.1 핵심 구현 (400개 테스트 통과)

v7.0 추가:
  Phase 9.0: 적응형 오케스트레이션 (1~2주)
    - HardwareTier 감지 + ModelDispatcher
    - TurnState + TurnStateStore
    - 도구 스키마 축소
    - vLLM 설정 최적화 (--limit-mm-per-prompt)

  Phase 9.5: Scout 통합 (1주)
    - llama.cpp 서빙 설정
    - ScoutModelProvider 구현
    - Scout 시스템 프롬프트 최적화
    - Scout ↔ Worker 핸드오프 로직
    - Scout 실패 시 fallback

  Phase 10.0: 배치 인덱싱 (1~2주, 선택)
    - 백그라운드 파일 인덱싱
    - 심볼 인덱스 구축
    - 채팅에서 인덱스 조회
```

### Phase 9.0 상세 (즉시 시작 가능)

도구 스키마 축소와 TurnState는 **Scout 없이도 즉시 효과**가 있다.

```
Phase 9.0a: 도구 스키마 축소 + vLLM 최적화
  → 같은 코드로 토큰 20~30% 절약
  → 소요: 1~2일

Phase 9.0b: TurnState 구현 + query_loop 연동
  → raw messages 누적 → 요약 기반으로 전환
  → 소요: 3~5일

Phase 9.0c: ModelDispatcher + 티어 감지
  → TIER_S/M/L 자동 전환
  → 소요: 2~3일
```

---

## Part 7: 디렉토리 구조 변경

### 신규 파일 목록

```
core/
  orchestrator/
    model_dispatcher.py     # 신규: 멀티모델 디스패치
    turn_state.py           # 신규: TurnState + TurnStateStore
  model/
    scout_provider.py       # 신규: llama.cpp 프로바이더 (LocalModelProvider 래핑)
    hardware_tier.py        # 신규: HardwareTier 감지

config/
  nexus_config.yaml         # 수정: scout 섹션 추가
```

### 수정 파일 목록

```
core/
  bootstrap.py              # 수정: Phase 2에 Scout/Dispatcher 초기화 추가
  orchestrator/
    query_engine.py          # 수정: ModelDispatcher 주입
    query_loop.py            # 경미 수정: TurnState 생성 훅 추가
    context_manager.py       # 수정: 티어별 전략 분기
  tools/
    implementations/*.py     # 경미 수정: 도구 description 축소
  state.py                   # 경미 수정: GlobalState 필드 추가
```

### 변경 없는 파일 (전부)

```
core/
  message.py                 # 불변
  tools/base.py              # 불변
  tools/registry.py          # 불변
  tools/executor.py          # 불변
  permission/                # 전체 불변
  security/                  # 전체 불변
  hooks/                     # 전체 불변
  thinking/                  # 전체 불변
  memory/                    # 전체 불변
  task.py                    # 불변
training/                    # 전체 불변
deployment/                  # 전체 불변
cli/repl.py                  # 불변 (QueryEngine 인터페이스 동일)
web/app.py                   # 불변 (QueryEngine 인터페이스 동일)
tests/                       # 기존 테스트 전부 유지 + 신규 테스트 추가
```

---

## Part 8: 검증 — 설계 일관성 확인

### 8.1 4-Tier 체인 무결성

```
v7.0에서도 4-Tier 체인 구조는 동일하다:
  Tier 1: QueryEngine → ModelDispatcher → query_loop 호출
  Tier 2: query_loop() → while(True) 에이전트 턴 루프
  Tier 3: model_provider.stream() → SSE 파싱
  Tier 4: httpx → GPU 서버 (또는 CPU 서버)

ModelDispatcher는 Tier 1 내부의 라우팅 로직이다.
Tier 2~4는 어떤 ModelProvider를 받느냐만 다르고, 코드는 동일하다.
```

### 8.2 도구 시스템 무결성

```
24개 도구의 BaseTool 구현체: 변경 없음
13단계 실행 파이프라인: 변경 없음
ToolRegistry: 변경 없음
StreamingToolExecutor: 변경 없음

차이: 어떤 도구가 어떤 모델에 할당되느냐 (TIER_S에서만)
→ 이것은 ToolRegistry.get_all_tools() 대신 필터링된 리스트를 전달하는 것뿐
→ 기존 deny_patterns 필터링과 동일한 메커니즘
```

### 8.3 권한 시스템 무결성

```
5계층 Permission Pipeline: Scout든 Worker든 동일하게 적용
Scout의 도구(Read, Glob, Grep, LS)는 모두 is_read_only=True
→ Layer 3에서 DEFAULT 모드로 자동 허용
→ 권한 프롬프트가 뜨지 않음

Worker의 도구(Edit, Write, Bash)는 기존과 동일한 권한 검증
→ 모델이 바뀌어도 권한 로직은 동일
```

### 8.4 Hook 시스템 무결성

```
PreToolUse/PostToolUse: Scout의 도구 호출에도 동일하게 적용
HookDecision: 모델과 무관
→ Scout가 Read를 호출하면 PreToolUse 훅이 실행됨
→ 기존 훅 설정이 Scout에도 자동 적용
```

### 8.5 Thinking/Memory 무결성

```
ThinkingEngine: Worker에서만 사용 (Scout는 단순 탐색이라 불필요)
MemorySystem: Worker의 on_turn_start/on_turn_end에서 동작
→ Scout는 메모리를 읽지 않음 (탐색 전용)
→ Worker는 기존대로 메모리 사용
```

### 8.6 Training 무결성

```
학습 대상: Worker(31B) — 기존과 동일
학습 데이터: Worker의 도구 호출 패턴 수집 — 기존과 동일
Scout(4B)는 학습 대상이 아님 (범용 탐색만 수행)
```

### 8.7 GPU 업그레이드 시 자동 수렴

```
RTX 5090 → H100:
  detect_hardware_tier() → TIER_M
  ModelDispatcher → passthrough (Scout 비활성)
  query_loop → v6.1 동일 경로
  TurnState → 보조 메타데이터만 (압축 파이프라인 주력)
  → v6.1과 100% 동일 동작

H100 → H200:
  detect_hardware_tier() → TIER_L
  동시 모델 로드 (Primary + Auxiliary)
  128K 컨텍스트
  → v6.1 최적 시나리오
```

---

## 부록 A: v6.1 → v7.0 변경 영향 매트릭스

| v6.1 모듈 | v7.0 영향 | 테스트 영향 |
|---|---|---|
| core/message.py | 없음 | 기존 테스트 100% 유지 |
| core/model/inference.py | 없음 (Scout는 별도 Provider) | 기존 테스트 유지 + Scout 테스트 추가 |
| core/tools/base.py | 없음 | 기존 테스트 유지 |
| core/tools/registry.py | 없음 | 기존 테스트 유지 |
| core/tools/executor.py | 없음 | 기존 테스트 유지 |
| core/tools/implementations/ | description 축소만 | 기존 테스트 유지 |
| core/orchestrator/query_loop.py | TurnState 생성 훅 추가 | 기존 테스트 유지 + TurnState 테스트 |
| core/orchestrator/query_engine.py | Dispatcher 주입 | 기존 테스트 수정 (Provider → Dispatcher) |
| core/orchestrator/context_manager.py | 티어별 전략 분기 | 기존 테스트 유지 + TIER_S 테스트 |
| core/permission/ | 없음 | 기존 테스트 유지 |
| core/security/ | 없음 | 기존 테스트 유지 |
| core/hooks/ | 없음 | 기존 테스트 유지 |
| core/thinking/ | 없음 | 기존 테스트 유지 |
| core/memory/ | 없음 | 기존 테스트 유지 |
| core/state.py | 필드 3개 추가 | 기존 테스트 유지 |
| core/bootstrap.py | Phase 2 확장 | 기존 테스트 수정 + 신규 |
| training/ | 없음 | 기존 테스트 유지 |
| deployment/ | 없음 | 기존 테스트 유지 |
| cli/ | 없음 | 기존 테스트 유지 |
| web/ | 없음 | 기존 테스트 유지 |

**기존 380개 테스트 중 수정 필요**: ~5~10개 (QueryEngine, Bootstrap 관련)
**신규 테스트 추가**: ~30~50개 (ModelDispatcher, TurnState, Scout, 티어 감지)

---

*작성일: 2026-04-16*
*Project Nexus — IDINO*
*기준 문서: PROJECT_NEXUS_SPEC_v6.1_EN.md*
