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

## Part 2: 수정 — 멀티모델 디스패치 계층 (B 방식 재설계, 2026-04-17)

> **재설계 경고 (2026-04-17):** 초기 v7.0 설계는 ModelDispatcher가 TIER_S에서
> **모든 사용자 요청마다 Scout를 자동 전처리**하도록 했다. 실측 결과 CPU 기반
> Scout(Gemma 4 E4B, ~16 TPS)가 모든 요청에 ~30초 오버헤드를 추가하여 "안녕"
> 같은 단순 질문도 33초가 걸렸다. 이에 **B 방식으로 전환**한다: Scout를 자동
> 전처리기가 아닌 "Worker가 필요할 때 호출하는 서브에이전트"로 승격시킨다.
> 이 재설계는 사양서 Ch 14(Agent & Task System)의 원래 AgentDefinition 의도와
> 더 일치한다.

### 2.1 아키텍처 (재설계 후)

v6.1의 4-Tier 체인 구조를 **그대로 유지**하되, Worker가 실행 중 필요 시 서브
에이전트를 호출할 수 있도록 Agent 도구를 노출한다.

```
v7.0 B 방식 (모든 TIER 공통):
  Tier 1: QueryEngine.submit_message()
    └─ ModelDispatcher.route()          ← 단순 Worker 래퍼
        └─ Worker용 query_loop() ─── WorkerModelProvider (GPU :8001)
            ├─ 일반 도구 (Read/Edit/Bash/…)
            └─ Agent 도구 ← Worker가 필요할 때만 호출
                  └─ subagent_type="scout" → Scout query_loop
                        └─ ScoutModelProvider (CPU :8003)
                        └─ 도구 4개(Read/Glob/Grep/LS), max_turns=3
```

**핵심 변경**:
- ModelDispatcher는 이제 **모든 티어에서 passthrough**다. Worker query_loop을
  한 번 호출할 뿐이다.
- Scout 실행 여부는 **Worker가 AgentTool 호출로 스스로 결정**한다.
- TIER_S에서도 단순 요청은 Worker 직행(1~3초), 대규모 탐색만 Scout 경유(+30s).
- Dispatcher 클래스 자체는 유지하여 TIER_L 병렬 Worker 등 향후 확장을 위한
  삽입점을 보존한다.

### 2.2 ModelDispatcher 설계 (재설계)

```python
class ModelDispatcher:
    """
    Worker query_loop 실행 래퍼.

    초기 설계의 자동 Scout 전처리는 제거되었다.
    Scout는 이제 AgentTool + AgentDefinition(SCOUT_AGENT)로 처리된다.
    """

    def __init__(
        self,
        tier: HardwareTier,
        worker_provider: ModelProvider,
        worker_tools: list[BaseTool],       # AgentTool 포함
        context: ToolUseContext,
        scout_provider: ModelProvider | None = None,  # 보관만 — AgentTool이
        scout_tools: list[BaseTool] | None = None,    # context.options로 꺼내감
        max_turns: int = 200,
    ):
        ...

    async def route(
        self,
        messages: list[Message],
        system_prompt: str,
        on_turn_complete: Any | None = None,
    ) -> AsyncGenerator[StreamEvent | Message, None]:
        """Worker query_loop으로 직행 (passthrough). Scout 자동 호출 없음."""
        async for event in query_loop(
            messages=messages,
            system_prompt=system_prompt,
            model_provider=self._worker_provider,
            tools=self._worker_tools,
            context=self._context,
            max_turns=self._max_turns,
            on_turn_complete=on_turn_complete,
        ):
            yield event
```

### 2.3 Scout는 AgentDefinition으로 선언된다

Scout는 **`core/orchestrator/agent_definition.py`의 SCOUT_AGENT 상수**로
선언된다 (Ch 14 참조). AgentRegistry에 등록되어 AgentTool이 조회한다.

```python
# v7.0 Part 2.3 초판 (2026-04-16): Read/Glob/Grep/LS 4개
# v7.0 Part 2.3 개정 (2026-04-17): DocumentProcess 추가 (총 5개)
SCOUT_AGENT = AgentDefinition(
    name="scout",
    description=(
        "Read-only file/document explorer running on CPU (Qwen3.5-4B, slow ~15-30s). "
        "Use when the user asks for broad project exploration, multi-file search, "
        "codebase understanding, OR when analyzing an uploaded document "
        "(PDF/DOCX/XLSX). Do NOT use for simple questions, greetings, or "
        "single-line file edits."
    ),
    system_prompt=(
        "You are Scout, a read-only exploration and document-analysis agent.\n"
        "Your job is to absorb large data sources on behalf of the Worker so the "
        "Worker's context stays small. You must NOT modify any files.\n"
        "Tools: Read, Glob, Grep, LS, DocumentProcess (chunked PDF/DOCX/XLSX).\n"
        "Workflow: pick the right tool → for big docs call DocumentProcess with "
        "chunk_index repeatedly → return a SHORT summary (200~500 tokens)."
    ),
    allowed_tools=("Read", "Glob", "Grep", "LS", "DocumentProcess"),
    max_turns=5,        # 문서 청크 순차 처리 위해 3→5 증가
    model_override="scout",
)
```

#### Part 2.3 2차 개정 (2026-04-18) — 출력 형식 JSON → 마크다운 완화

**사유**: 실측 결과 Qwen3.5-4B(CPU, llama.cpp)가 구조화된 JSON 출력을 안정적으로
생성하지 못함. Worker가 파싱 실패로 같은 Scout를 4회 반복 호출하여 240초
타임아웃. JSON 중괄호·이스케이프·따옴표 규율은 작은 모델에겐 과도한 부담.

**변경**: Scout 출력을 JSON 대신 **4개 섹션 마크다운**으로 완화. 의미 구조는
동일 (relevant_files / file_summaries / plan / requires_tools).

```
## relevant_files
- path/to/file1

## file_summaries
- path/to/file1: one-line description

## plan
- bullet of key facts the Worker needs
- more facts
...

## requires_tools
- Edit
```

**Worker 측**: JSON 파싱 대신 마크다운 섹션 헤더로 `## plan` 본문을 추출해
답변 생성 재료로 사용. LLM이 자연어/마크다운 처리에 특화되어 있어 형식
교환 비용은 거의 없음.

**사양서 정신 준수**: Scout의 역할(탐색·계획), 4개 정보 슬롯, Worker와의
계약 모두 그대로. 단지 wire format만 더 관대한 것으로 교체. 하드웨어 업그레이드
시 Scout를 더 큰 모델로 교체하면 JSON으로 되돌려도 무방.

---

#### Part 2.3 개정 근거 (2026-04-17)

- DocumentProcess는 사양서 v7.0 AMENDMENT **초판 작성 시점에 존재하지
  않았던 도구**다. Phase 9.4에서 웹 채팅 PDF/DOCX/XLSX 분석용으로 신설.
- 초판에 포함되지 못한 이유는 "설계자가 제외했다"가 아니라 "아직 존재하지
  않았다"이다. 따라서 초판의 4개 도구 목록은 완결된 집합이 아니라 그 시점
  최선이었던 집합이다.
- **Part 2.4의 핵심 최적화 원칙**은 "Scout가 이미 흡수한 것을 Worker가
  다시 읽지 않도록 한다"이다. DocumentProcess는 바로 이 원칙에 가장 잘
  부합하는 도구다 — 대용량 PDF/DOCX는 Worker 컨텍스트(8K)에 직접 들어가면
  치명적이지만, Scout가 청크 단위로 흡수 후 200~500토큰 요약만 돌려주면
  Worker가 여유롭게 동작할 수 있다.
- 실측 근거: Worker 풀에서 DocumentProcess 제거 시 도구 스키마 ~200토큰
  절감 + 대용량 문서 처리 시 원문 직접 적재 방지로 수천 토큰 절감.

description은 Worker에게 노출되어 "언제 Scout를 호출할지" 판단하는 힌트가
된다. 특히 "Do NOT use for simple questions" 문구가 남용을 억제하고, 신규
문구 "analyzing an uploaded document"가 파일 업로드 시나리오에서 Worker가
Scout를 자발적으로 선택하도록 유도한다.

#### 하드웨어 업그레이드 시 자동 복귀

이 개정은 **TIER_S 전용 최적화**다. H100/H200 이상에서는:
- Worker 컨텍스트가 32K/128K로 커져 DocumentProcess를 Worker가 직접 호출해도
  문제 없음
- `_create_tool_registry()` (24개 전체)를 쓰면 DocumentProcess가 자동으로
  Worker 풀에 포함
- Scout 자체가 TIER_M/L에서 비활성이므로 SCOUT_AGENT의 도구 목록 확장은
  영향 없음 (사실상 dead code)

따라서 이 개정은 **"TIER_S 운영 중에만 효력이 있고 상위 티어로 가면 저절로
무력화되는 임시 장치"**다. 사양서 Part 1.3(상위 티어는 하위 티어의
상위집합) 원칙과 일치한다.

### 2.4 Worker 도구 (모든 TIER 공통)

Worker는 일반 도구 + `AgentTool`을 받는다. Agent 도구 스키마는 다음과 같다:

```python
{
    "type": "object",
    "properties": {
        "prompt": {"type": "string"},
        "subagent_type": {"type": "string"},  # "scout" 등
        "description": {"type": "string"},     # 하위 호환 ad-hoc
    },
    "required": ["prompt"],
}
```

Worker의 시스템 프롬프트에는 등록된 서브에이전트 목록이 동적으로 주입된다
(`_build_default_system_prompt(agent_registry)` 참조).

### 2.5 TIER별 차이 (재설계 후)

| TIER | Worker 도구 수 | Scout 서버 | Agent 도구 동작 |
|---|---|---|---|
| TIER_S | ~12개 (CLI) / ~9개 (웹) | llama.cpp CPU :8003 | subagent_type="scout" 호출 가능 |
| TIER_M | 24개 전체 | 없음 | subagent_type="scout" 호출 시 에러 반환(Fail-closed) |
| TIER_L | 24개 전체 + 병렬 | 없음 | 동일 |

TIER_M/L에서는 `context.options["scout_provider"]`가 None이므로 Worker가
잘못 Scout를 호출해도 AgentTool이 명확한 에러를 돌려준다 ("scout_provider
not configured"). Worker는 이 에러를 보고 직접 도구로 재시도한다.

### 2.6 왜 이 재설계가 v7.0 원칙과 일치하는가

v7.0의 "3대 불변 전제"를 더 잘 만족한다:

1. **Claude Code 설계 유지** — Claude Code의 Task/Agent 도구 패턴과 정확히 동일
2. **GPU 업그레이드 시 자동 수렴** — TIER_M/L에서 Scout 없이 Worker만 동작
   (기존 B 설계도 같은 속성을 가짐)
3. **동일한 사용자 경험** — 단순 요청은 빠르게, 복잡한 탐색은 자동 위임

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

**Scout는 서브에이전트로 정식 승격** (2026-04-17 B 방식 재설계).

구현 위치: `core/orchestrator/agent_definition.py`

#### AgentDefinition 정식 명세

```python
@dataclass(frozen=True)
class AgentDefinition:
    """서브 에이전트의 불변 명세."""

    name: str                           # subagent_type 식별자
    description: str                    # Worker에게 보이는 사용 가이드 힌트
    system_prompt: str                  # 서브 에이전트 시스템 프롬프트
    allowed_tools: tuple[str, ...]      # 사용 가능한 도구 이름 (튜플, 불변)
    max_turns: int = 10                 # 최대 턴 수 (무한 루프 방지)
    model_override: str | None = None   # "scout" 또는 None(부모 Worker 재사용)
```

#### AgentRegistry

```python
class AgentRegistry:
    """선언된 서브에이전트를 관리하는 경량 레지스트리."""

    def register(self, agent: AgentDefinition) -> None: ...
    def register_many(self, agents: list[AgentDefinition]) -> None: ...
    def get(self, name: str) -> AgentDefinition | None: ...
    def list_names(self) -> list[str]: ...  # 이름순 정렬
    def list_descriptions(self) -> dict[str, str]: ...  # Worker 프롬프트 주입용
```

부트스트랩이 `build_default_agent_registry()` 팩토리로 기본 레지스트리를 만들고
`ToolUseContext.options["agent_registry"]`로 주입한다.

#### SCOUT_AGENT 선언 (Part 2.3 개정 반영, 2026-04-17)

```python
SCOUT_AGENT = AgentDefinition(
    name="scout",
    description=(
        "Read-only file/document explorer running on CPU (Qwen3.5-4B, slow ~15-30s). "
        "Use when the user asks for broad project exploration, multi-file search, "
        "codebase understanding, OR when analyzing an uploaded document "
        "(PDF/DOCX/XLSX). Do NOT use for simple questions, greetings, or "
        "single-line file edits."
    ),
    system_prompt=(
        "You are Scout, a read-only exploration and document-analysis agent.\n"
        "Your job is to absorb large data sources on behalf of the Worker so the "
        "Worker's context stays small. You must NOT modify any files.\n"
        "Tools: Read, Glob, Grep, LS, DocumentProcess (chunked PDF/DOCX/XLSX).\n"
        "Workflow: pick the right tool → for big docs call DocumentProcess with "
        "chunk_index repeatedly → return a SHORT summary (200~500 tokens)."
    ),
    allowed_tools=("Read", "Glob", "Grep", "LS", "DocumentProcess"),
    max_turns=5,
    model_override="scout",
)
```

> 개정 근거 및 하드웨어 업그레이드 시 자동 복귀 설명은 Part 2.3 참조.

#### AgentTool의 subagent_type 처리

Worker가 Agent 도구를 호출할 때 경로:

1. `context.options["agent_registry"]`에서 `subagent_type`으로 AgentDefinition 조회
2. `model_override == "scout"`이면 `context.options["scout_provider"]` 선택;
   그렇지 않으면 부모 Worker 프로바이더 재사용
3. `allowed_tools`에 명시된 도구만 `context.options["available_tools"]`에서 필터링
4. 독립 `QueryEngine`을 생성해 서브 세션으로 실행 (messages[] 격리)
5. 실행 시간 통계를 `AgentTool._stats[name]`에 누적

에러 처리:
- `agent_registry`가 없거나 `subagent_type`이 미등록 → `ToolResult.error(...)`
- `model_override="scout"`인데 `scout_provider`가 없음 (TIER_M/L) → 명확한 에러

#### DISALLOWED_TOOLS_FOR_AGENTS (변경 없음)

서브에이전트는 `{Agent, TaskCreate, TaskStop, TrainingTool, CheckpointTool}`을
사용할 수 없다. 재귀 호출과 위험 도구를 차단하여 제어 흐름을 단순하게 유지한다.

#### 기존 AgentRunner/TaskManager/Coordinator (변경 없음)

이들은 TaskManager 아래에서 장기 실행 태스크(LOCAL_AGENT 등)를 관리하며,
지금 재설계는 그 상위의 AgentDefinition 정의만 건드린다.

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

### Ch 17: Monitoring & Metrics (재설계, 2026-04-17)

**서브에이전트 메트릭** (B 방식):

`/metrics` 엔드포인트는 `AgentTool.get_stats()`를 참조하여 서브에이전트별
호출 통계를 노출한다. Scout 자동 전처리가 제거됐으므로 `ModelDispatcher.stats`
는 더 이상 실제 수치를 누적하지 않는다 (하위 호환 0 유지).

```python
# GET /metrics 응답 구조
{
    "http": {...},
    "session": {...},
    "agents": {
        "scout": {
            "calls": int,                # AgentTool로 Scout가 호출된 횟수
            "total_latency_ms": float,   # 누적 지연 시간
            "avg_latency_ms": float,     # 평균 지연 (calls로 나눔)
        },
        # 향후 추가 서브에이전트: code-reviewer, sql-explorer 등
    },
    "scout": {                           # 하위 호환 — 대시보드 유지용
        "tier": "small" | "medium" | "large",
        "scout_enabled": bool,
        "scout_calls": int,              # = agents.scout.calls
        "scout_avg_latency_ms": float,   # = agents.scout.avg_latency_ms
        "scout_fallback_count": 0,       # B 방식에선 의미 없음 (항상 0)
        "note": "sourced from AgentTool.get_stats()",
    },
}
```

**구현**:
- `AgentTool._stats` (ClassVar dict) — subagent_type별 calls/total_latency_ms
- `AgentTool._record_stats(key, elapsed_ms)` — call() 내에서 성공·실패 모두 누적
- `AgentTool.get_stats()` — 읽기 전용 복사본 반환, avg_latency_ms 즉석 계산
- `AgentTool.reset_stats()` — 테스트 격리용

**ModelDispatcher.stats** (하위 호환 전용):
```python
{
    "tier": str,
    "scout_enabled": bool,  # 서버 연결 여부 (자동 호출 의미 아님)
    "scout_calls": 0,
    "scout_fallback_count": 0,
    "scout_fallbacks": 0,
    "scout_avg_latency_ms": 0.0,
    "note": "Scout is now invoked by the Worker via AgentTool",
}
```

기존 대시보드가 `scout_calls` 같은 평탄한 키를 참조하는 경우를 위해 `web/app.py`
가 AgentTool 통계를 꺼내 `result["scout"]`에도 평탄화해 넣는다.

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
