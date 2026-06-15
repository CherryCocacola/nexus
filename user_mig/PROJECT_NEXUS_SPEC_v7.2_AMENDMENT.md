# Project Nexus — 기술 사양서 v7.2 개정안

## MCP(Model Context Protocol) 통합 — 제품 사양

**버전**: 7.2 (MCP Integration Amendment)
**기준 문서**: PROJECT_NEXUS_SPEC_v7.1_AMENDMENT.md
**개정일**: 2026-06-01
**상태**: 제품 구현 반영본 (MCP 클라이언트 어댑터 구현·강화·테스트 완료)

---

## 개정 개요

### 제품화 전제

Nexus는 PoC가 아니라 **실제 제품화 대상**이다. 검증 기준 환경은 **고객 A100
80GB**다. MCP는 모델·티어와 직교(orthogonal)하므로 GPU와 무관하게 동작하지만,
제품 검증의 기준 GPU는 A100 80GB임을 명시한다.

> **A100 티어 (구현 완료, 2026-06-02 — 초판 대비 변경)**: v7.2 초판은 "A100 enum
> 이 없어 H100 으로 자동 매핑되며 별도 추가 불필요"로 기술했으나, 이후
> **`GPUTier.A100`(`gpu_detector.py:32`)이 별도 enum 으로 추가**되었다. A100 80GB
> 와 H100 80GB 는 VRAM 이 같아 용량만으로는 구분되지 않으므로, 감지 로직
> (`gpu_detector.py:161~169`)이 `vram_gb > 60` 구간에서 **GPU 이름에 "a100"이
> 포함되면 `GPUTier.A100`**, 아니면 `H100` 으로 판정한다. A100 프로파일은
> H100 과 거의 동일하되 **Ampere 라 FP8 미지원**(`notes`에 명시)이라는 차이를
> 반영한다. 상세는 v7.3 Part 4.4 참조. (MCP 동작에는 무관 — GPU 와 직교.)

### 왜 v7.2인가

v6.1은 MCP(Model Context Protocol)를 "에어갭에서 불가능"으로 보고 보류했다
(8장 차이표: "MCP server permission management → Planned for Phase 2 expansion").
이 판단은 **외부 SaaS MCP 서버**(예: 클라우드 호스팅 MCP)를 전제로 한 것이며
당시에는 정확했다.

그러나 운영 6주 동안 사내 시스템(PostgreSQL/pgvector, 임베딩 서버, 진단
스크립트, DocUtil 문서 시스템)이 **LAN 내부 서비스**로 누적되었고, 이들을
표준 프로토콜로 떼어내 재사용할 필요가 분명해졌다. v6.1 원본조차 에어갭
경계표(라인 400~407)에서 **"Machine A <-> Machine B (LAN, HTTP/SSE)"를 ALLOWED**
로 명시한다. 즉 LAN 내부 HTTP/SSE 통신은 처음부터 에어갭이 허용하는 영역이다.

v7.2는 다음 네 가지 성격을 가진다:

1. v6.1의 "MCP 보류" 판단을 **LAN 한정 활성화**로 정정 (에어갭 원칙은 불변)
2. **이미 예비된 인프라 활성화** — `ToolCategory.MCP`, `MODE_BEHAVIOR_MAP`의
   MCP 열, `pipeline.py`의 `mcp__` prefix 식별 로직은 v6.1부터 이미 코드에 존재
3. **MCP 클라이언트 어댑터 + 연결 관리자 구현 완료** — `core/tools/mcp/`
   (`client.py`/`adapter.py`/`connection_manager.py`/`__init__.py`)와
   `McpConfig`/`McpServerConfig`(`core/config.py`)가 **실제로 구현·강화·
   테스트 완료**되었다. 4-Tier·권한·Hook은 손대지 않는다
4. **제품 보안 기준 강화** — read-only-only 등록 정책 확정,
   `allow_write` 명시 플래그, LAN 판정 헬퍼의 보안 모듈 이전
   (`core/security/network_guard.is_lan_hostname`), 부트스트랩 배선 정정

### 핵심 정합화 논점 (한 문장)

> 외부 SaaS MCP는 **여전히 금지**한다. 사내 시스템을 **LAN 내부 MCP 서버**로
> 떼어내 Nexus가 그 도구를 호출하는 것은 **에어갭 위반이 아니다.** v7.2는
> v6.1의 보류 판단을 깨는 것이 아니라 LAN 경계 안으로 한정해 갱신한다.

### 불변 전제 유지 선언 (v7.0/v7.1 그대로)

1. Claude Code 설계안 그대로 — 4-Tier 체인, 24개 도구, 5계층 권한, Hook,
   Thinking, Memory 전부 유지
2. GPU 업그레이드 시 성능 향상 — MCP는 모델·티어와 직교(orthogonal)
3. 표준 내부 계약 불변 — query_loop은 OpenAI `tool_calls` 형식만 본다.
   MCP JSON-RPC는 어댑터 내부에 캡슐화

### 변경 범위 요약

| 구분 | Part/대상 | 변경 수준 | 비고 |
|---|---|---|---|
| **정정** | v6.1 8장 MCP 보류 판단 | 중간 | "불가능" → "LAN 한정 활성화" |
| **구현됨** | Part 2 MCP 클라이언트 어댑터 | 구현 완료 | `McpToolAdapter(BaseTool)`, `McpClient`, `McpConnectionManager` (`core/tools/mcp/`) |
| **구현됨** | Part 3 Transport/에어갭 경계 | 구현 완료 | LAN HTTP/SSE 확정, stdio 비채택. LAN 판정 `core/security/network_guard`로 이전 |
| **활성화** | Part 4 권한 통합 | 경미 | 이미 존재하는 MCP 권한 인프라 인용·활용 + read-only-only 등록 정책 확정 |
| **구현됨** | Part 5 YAML `mcp:` 섹션 | 구현 완료 | `McpConfig`/`McpServerConfig`(+`allow_write`), 기본 비활성 |
| **구현됨** | Part 6 초기 검증 대상 4종 | 구현 완료 | DB/진단/DocUtil/kowiki RAG |
| **변경 없음** | 4-Tier 체인, Hook, Thinking, Memory, Training | — | 영향 없음 (Part 7 검증) |

---

## Part 0: 변경 없는 챕터 (재확인)

v7.0/v7.1과 동일하게 다음은 **코드 변경이 전혀 없다.** MCP 도입은 도구 풀에
원격 도구 N개를 추가하는 것일 뿐, 그 도구가 거치는 파이프라인은 기존 도구와
완전히 동일하다.

| 챕터/시스템 | 이유 |
|---|---|
| 4-Tier AsyncGenerator 체인 (Tier1~4) | MCP 도구도 일반 도구처럼 query_loop을 통과. 체인 무관 |
| Ch 8 5계층 Permission Pipeline | MCP 카테고리·식별 로직이 **이미 존재** (Part 4) |
| Ch 9 Security System | PathGuard/CommandFilter — MCP는 파일·명령 도구가 아니므로 무관 |
| Ch 10 Hook System | Hook은 도구 실행 파이프라인에 연결. MCP 도구도 동일하게 Hook 적용 |
| Ch 11 Thinking Engine | Worker 모델 추론. MCP와 직교 |
| Ch 12 Memory (tb_memories) | MCP는 도구 계층. 메모리 스키마 무관 |
| Ch 18 Training Pipeline | 학습 대상 모델 불변 |
| Ch 19 Air-Gap Strategy | **강화됨** — LAN 경계 이중 강제 구현 + `network_guard.is_lan_hostname` 일원화 (Part 3) |

---

## Part 1: 정정 — v6.1 MCP 보류 판단 (LAN 한정 활성화)

### 1.1 v6.1 원안 (정정 대상, 원문 인용)

v6.1은 MCP를 세 곳에서 보류·제거했다:

```
# v6.1 Ch 1 (라인 125): 도구 집계
[x] Tool system (42 → 24 air-gap compatible tools)
# → 42개 중 MCP 관련 도구가 air-gap 비호환으로 제외되어 24개로 축소

# v6.1 Ch 8 (라인 10402): Claude Code 대비 차이표
| MCP server permission management | Planned for Phase 2 expansion |

# v6.1 Ch 8 (라인 10578~10579): ToolCategory 정의
MCP = "mcp"
"""MCP server tools: Phase 2."""
```

v6.1 작성 시 MCP의 정신적 모델은 **"외부 인터넷에 있는 SaaS 도구 서버"**였다.
그 전제에서 "에어갭 불가"는 옳다. 외부 도메인 연결은 v6.1 라인 393~395
(OpenAI/Bing/Web Fetch BLOCKED)와 동일하게 금지된다.

### 1.2 정정 근거 — v6.1 자신이 LAN HTTP/SSE를 허용한다

v6.1 에어갭 경계표(라인 400~407)는 LAN 내부 통신을 명시적으로 허용한다:

```
ALLOWED (permitted inside air-gap):
  [o] Machine A <-> Machine B  (LAN, HTTP/SSE)     ← MCP가 정확히 이 형태
  [o] Machine A <-> PostgreSQL (localhost or LAN)
  [o] Machine A <-> Redis      (localhost or LAN)
```

MCP over LAN HTTP/SSE는 라인 402의 패턴과 동일한 통신이다. 따라서 **LAN MCP
서버 연결은 v6.1 에어갭 원칙을 위반하지 않는다.** v7.2는 이 자기모순(MCP는
보류했으나 LAN HTTP/SSE는 허용)을 LAN 경계 안에서 해소한다.

### 1.3 정정 후 입장

| 구분 | v6.1 판단 | v7.2 정정 |
|---|---|---|
| 외부 SaaS MCP (인터넷) | 불가 | **여전히 불가** (변화 없음) |
| LAN 내부 MCP 서버 | (구분 없이) 보류 | **허용** — 192.168.x/10.x/localhost 한정 |
| MCP 권한 모델 | "Phase 2 계획" | **이미 코드에 존재**, 활성화만 (Part 4) |
| `mcp__` 도구 식별 | — | **이미 코드에 존재** (`pipeline.py:191`) |

이 정정은 에어갭 원칙의 **약화가 아니라 정밀화**다. "외부 네트워크 금지"는
유지하되, "LAN 내부 서비스 호출은 허용"이라는 v6.1 자신의 규정을 MCP에도
일관 적용한다.

---

## Part 2: MCP 클라이언트 어댑터 아키텍처 (구현 완료)

> **표기 규약**: 이 Part의 `core/tools/mcp/*` 파일(`client.py`/`adapter.py`/
> `connection_manager.py`/`__init__.py`)은 **모두 구현·테스트 완료**되었다.
> 인용하는 `BaseTool`/`ToolRegistry`/`ToolUseContext`/`pipeline.py`도 **이미
> 존재**한다 (부록 B 참조). 아래 코드 발췌는 실제 구현에서 인용한 것이다.

### 2.1 설계 목표

원격 MCP 서버의 도구 1개를 **`BaseTool` 서브클래스 1개**로 래핑하여, 기존
도구와 구별 없이 registry에 등록·실행한다. query_loop은 MCP의 존재를 전혀
모른다 — 오직 OpenAI `tool_calls` 형식만 본다 (P3 계약).

### 2.2 컴포넌트 구성 (구현 완료)

```
core/tools/mcp/                       ← 구현된 디렉토리
  __init__.py          — 공개 심볼 노출(McpClient/McpToolAdapter/McpConnectionManager)
  client.py            — McpClient: JSON-RPC over LAN HTTP/SSE 저수준 클라이언트
  adapter.py           — McpToolAdapter(BaseTool): 원격 도구 1개를 BaseTool로 래핑
  connection_manager.py — McpConnectionManager: 부트스트랩 시 연결·발견·등록·정리
```

의존성 방향(P2 준수, 실제 import 확인됨): `core/tools/mcp/` →
`core/tools/base.py`, `core/tools/registry.py`, **`core/security/network_guard`**
(LAN 판정 헬퍼 — 보안 관심사), `httpx`(inference.py와 동일 스타일). 역방향
import 없음.

### 2.3 McpToolAdapter — 원격 도구를 BaseTool로 래핑 (구현 완료)

MCP `tools/list`로 받은 스키마를 `input_schema`로, `tools/call`을 `call()`로
매핑한다. 도구 이름은 `mcp__{server}__{tool}` 규칙을 따라 **이미 존재하는**
권한 식별(`pipeline.py:191`, `name.startswith("mcp__")`)과 자동 정합한다.

구현(`core/tools/mcp/adapter.py`)에는 PoC 인터페이스 대비 제품 견고성을 위한
두 가지 강화가 들어 있다:

- **`validate_input()`**: 원격 호출 전 어댑터 단에서 최소 검증을 수행한다
  (최상위 `type=="object"` 검사 + `required` 필드 존재 검사). 잘못된 입력이
  LAN을 건너 원격 서버까지 나가기 전에 차단해, 원격 호출 1회를 아끼고 모델의
  자가 교정을 돕는다. 깊은 JSON Schema 검증(중첩 type/format/enum)은
  과설계 금지 원칙에 따라 원격 서버 책임으로 둔다.
- **`check_permissions()`는 항상 ALLOW** (read-only/쓰기 구분 없이) — 핵심
  설계 결정이며 Part 4에서 상술한다.

```python
# core/tools/mcp/adapter.py (구현 발췌)
class McpToolAdapter(BaseTool):
    """원격 MCP 서버의 tool 1개를 Nexus BaseTool로 래핑하는 어댑터."""

    def __init__(
        self,
        server_name: str,            # 예: "db", "diag", "docutil", "kowiki"
        remote_tool_name: str,       # MCP 서버가 보고한 tool 이름
        remote_schema: dict[str, Any],  # tools/list가 준 inputSchema
        client: McpClient,           # LAN HTTP/SSE 클라이언트
        description: str,
        is_read_only: bool = False,  # fail-closed 기본값 (명시적 완화만 허용)
    ):
        self._server = server_name
        self._remote = remote_tool_name
        self._schema = remote_schema
        self._client = client
        self._description = description
        self._read_only = is_read_only

    @property
    def name(self) -> str:
        # 권한 식별 규칙과 정합: mcp__{server}__{tool}
        return f"mcp__{self._server}__{self._remote}"

    @property
    def group(self) -> str:
        # UI 카테고리 — MCP 도구는 서버별로 묶어 보여준다
        return f"mcp:{self._server}"

    @property
    def input_schema(self) -> dict[str, Any]:
        # MCP tools/list의 inputSchema를 그대로 노출 (변환 없음)
        return self._schema

    @property
    def is_read_only(self) -> bool:
        # fail-closed: 기본 False. 조회 전용 서버면 생성 시 True로 완화됨
        return self._read_only

    def validate_input(self, input_data: dict[str, Any]) -> str | None:
        # 원격 호출 전 최소 견고성 검증: type/required만. None=유효, str=에러
        schema = self._schema or {}
        if schema.get("type") == "object" and not isinstance(input_data, dict):
            return f"MCP 도구 '{self.name}' 입력은 객체(object)여야 합니다."
        required = schema.get("required")
        if isinstance(required, list):
            missing = [k for k in required if isinstance(k, str) and k not in (input_data or {})]
            if missing:
                return f"MCP 도구 '{self.name}' 필수 입력 누락: {', '.join(missing)}"
        return None

    async def check_permissions(self, input_data, context) -> PermissionResult:
        # 핵심 결정: read-only/쓰기 구분 없이 항상 ALLOW. 신뢰 통제는 등록
        # 단계에서 끝났고, 최종 권한 판단은 전적으로 5계층 파이프라인에 위임한다
        # (Part 4 참조). details에 감사 추적용 메타를 실어 둔다.
        return PermissionResult(
            behavior=PermissionBehavior.ALLOW,
            message=f"MCP 도구 (서버 '{self._server}', read_only={self._read_only}) — 5계층 위임",
            details={"mcp_server": self._server, "remote_tool": self._remote,
                     "is_read_only": self._read_only},
        )

    async def call(self, input_data, context) -> ToolResult:
        # MCP tools/call 1회 호출. JSON-RPC는 client 내부에 캡슐화.
        # 연결 실패·타임아웃·LAN 검증 실패만 좁혀 tool_use_error로 래핑(P8).
        # 그 외 예외는 상위 executor 13단계가 일괄 래핑하도록 의도적으로 전파.
        try:
            result = await self._client.call_tool(
                self._remote, input_data, timeout=self.timeout_seconds
            )
            return ToolResult.success(result)
        except (TimeoutError, ConnectionError, ValueError) as e:
            return ToolResult.error(f"MCP 서버 '{self._server}' 호출 실패: {e}")
```

### 2.4 McpConnectionManager — 발견·등록 (구현 완료, read-only-only 정책)

부트스트랩 시 설정된 LAN MCP 서버에 연결 → `tools/list`로 도구 발견 →
서버별로 `McpToolAdapter`를 생성 → registry에 등록한다. 제품 보안 기준으로
다음 두 가지가 강화되었다(구현 확인됨):

1. **read-only-only 등록 정책(신뢰 통제를 등록 단계로 이동)**:
   `trust.read_only=True`로 신뢰된 서버의 도구만 자동 등록한다. 쓰기 가능
   서버(`trust.read_only=False`)는 운영자가 **`allow_write=True`를 명시했을
   때만** 등록한다(fail-closed). 신뢰되지 않은 쓰기 서버의 도구는 애초에 도구
   풀에 들어오지 못하므로 모델이 호출 자체를 할 수 없다.
2. **좁은 except + 실패 격리**: PoC의 broad `except Exception` 대신
   `(ConnectionError, TimeoutError, ValueError, OSError)`만 좁혀 포착한다.
   예상 가능한 운영 실패(연결/타임아웃/비-LAN/소켓)는 그 서버만 스킵하고,
   `AttributeError`/`TypeError` 같은 예상치 못한 버그는 전파시켜 상위
   bootstrap의 최후 방어선이 흡수하되 로그로 드러나게 한다(결함 가시성).

```python
# core/tools/mcp/connection_manager.py (구현 발췌 — 핵심 로직)
class McpConnectionManager:
    async def connect_and_register(self, registry: ToolRegistry) -> dict[str, list[str]]:
        registered: dict[str, list[str]] = {}
        for server in self._config.servers:
            if not server.enabled:
                continue  # 비활성(비-LAN도 config 검증에서 이미 강등됨)

            # ── read-only-only 등록 정책 (신뢰 경계는 등록 단계에서 통제) ──
            read_only = bool(server.trust.get("read_only", False))
            if not read_only and not server.allow_write:
                logger.warning(
                    "쓰기 가능 MCP 서버 '%s'은 초기 제품 정책상 등록하지 않음 "
                    "(read-only만 허용, 쓰기는 명시적 allow_write 필요)", server.name)
                registered[server.name] = []
                continue

            try:
                client = McpClient(  # 생성 단계에서 LAN URL 재검증(2단계)
                    base_url=server.base_url, api_key=server.api_key,
                    timeout=self._config.connect_timeout_sec)
                remote_tools = await client.list_tools()  # MCP tools/list
                names: list[str] = []
                for t in remote_tools:
                    adapter = McpToolAdapter(
                        server_name=server.name, remote_tool_name=t["name"],
                        remote_schema=t["inputSchema"], client=client,
                        description=t.get("description", ""),
                        is_read_only=read_only)  # allow_write 서버면 False(쓰기 표시)
                    registry.register(adapter)  # 이름순 정렬은 registry가 보장
                    names.append(adapter.name)
                self._clients.append(client)  # 종료 시 aclose_all로 정리
                registered[server.name] = names
            except (ConnectionError, TimeoutError, ValueError, OSError) as e:
                # 예상 가능한 운영 실패만 좁혀 포착 → 이 서버만 격리
                logger.warning("MCP 서버 '%s' 연결 실패 (스킵): %s", server.name, e)
                registered[server.name] = []
        return registered
```

`aclose_all()`로 보유한 모든 `McpClient`의 커넥션 풀을 정리하며, 개별 정리
실패도 격리한다(`OSError`/`RuntimeError`만 좁혀 무시).

### 2.5 tool_calls 계약 유지 다이어그램

```
[Worker 모델]
   │ OpenAI tool_calls: {"name": "mcp__db__query", "arguments": {...}}
   ▼
Tier 2: query_loop  ─── OpenAI tool_calls만 본다 (MCP를 모름)
   ▼
ToolExecutor (13단계) ─── PermissionPipeline 5계층 통과
   ▼
McpToolAdapter.call()  ◄── 여기서부터만 MCP 세계
   │ JSON-RPC: {"method": "tools/call", "params": {...}}
   ▼
McpClient ─── LAN HTTP/SSE ───► [원격 MCP 서버 (192.168.x.x)]
   │ JSON-RPC result
   ▼
ToolResult.success(...) ─── 다시 OpenAI tool_result로 정규화
   ▲
query_loop ─── tool_result 메시지로 수신 (MCP 흔적 없음)
```

**경계선**: MCP JSON-RPC는 `McpToolAdapter.call()` 아래에만 존재한다.
그 위(executor·query_loop·QueryEngine)는 v6.1과 한 글자도 다르지 않다.

---

## Part 3: 신규 — Transport(LAN HTTP/SSE) 및 에어갭 경계

### 3.1 Transport 결정: LAN HTTP/SSE 확정, stdio 비채택

MCP 표준은 stdio와 HTTP/SSE 두 transport를 정의한다. v7.2는 **HTTP/SSE만
채택**한다.

| 항목 | stdio | LAN HTTP/SSE (채택) |
|---|---|---|
| 통신 방식 | 로컬 자식 프로세스 stdin/stdout | LAN HTTP/SSE |
| 2-Machine 토폴로지 정합 | ✗ Machine A에서 MCP 서버 프로세스를 띄워야 함 | ✓ 원격 서버를 LAN으로 호출 (P4 정합) |
| 사내 시스템 외부화 | ✗ 각 시스템을 A 로컬로 끌어와야 함 | ✓ DB/임베딩 서버는 이미 원격 LAN |
| v6.1 ALLOWED 패턴 | 해당 없음 | ✓ 라인 402 "Machine A<->B LAN HTTP/SSE" |
| 에어갭 검증 용이성 | 프로세스 권한·샌드박스 추가 필요 | URL이 LAN인지만 검사하면 됨 (단순) |

**stdio 비채택 이유**: 사내 MCP 대상(PostgreSQL 192.168.10.39, 임베딩 서버
192.168.21.112, DocUtil 등)은 이미 **원격 LAN 서비스**다. stdio는 이들을
Machine A 로컬 프로세스로 끌어와야 하므로 2-Machine 토폴로지(P4)와 어긋나고
샌드박스 부담만 늘린다. HTTP/SSE는 v6.1이 이미 허용한 통신 형태를 그대로 쓴다.

### 3.2 에어갭 경계 명문화

| 구분 | 판정 | 예시 |
|---|---|---|
| **허용** | LAN MCP 서버 | `http://192.168.10.39:*`, `http://192.168.21.112:*`, `http://10.x.x.x:*`, `http://localhost:*` |
| **금지** | 외부 SaaS MCP | 공인 도메인, 인터넷 IP, TLS to public CA |

URL 검증은 **이중 강제**한다(구현 확인됨). 두 단계 모두 보안 모듈의 공용
헬퍼 **`core/security/network_guard.is_lan_hostname`** 을 사용한다:

1. **설정 로드 단계** (`McpConfig.validate_lan_urls`, Pydantic
   `@model_validator(mode="after")`, `core/config.py:191~`): `base_url`의
   hostname이 LAN 대역(192.168.x/10.x/172.16~31.x/localhost/127.x)이 아니면
   해당 서버를 강제로 `enabled=False`로 **강등** + WARNING. 외부 도메인을
   예외로 거부하면 설정 로드 전체가 멈춰 본류까지 죽으므로, 강등으로 본류를
   보호하면서 외부 연결만 구조적으로 차단한다(fail-closed + 본류 무영향).
2. **연결 단계** (`McpClient.__init__`, `core/tools/mcp/client.py:67~`):
   클라이언트 생성 시 `is_lan_hostname`으로 재검증하고, 비-LAN이면 `ValueError`를
   던져 연결 자체를 막는다. 코드에서 직접 `McpClient`를 만드는 경로가 있더라도
   어떤 경로로도 외부 연결이 불가능하다.

> **LAN 판정 헬퍼 위치 이전(보안 관심사)**: LAN/에어갭 판정은 본질적으로
> 보안 관심사이므로, 이전의 `core/config._is_lan_hostname`(private)에서
> **`core/security/network_guard.is_lan_hostname`(public)** 으로 이전했다.
> `core/config._is_lan_hostname`은 **하위호환 별칭**으로 남아 있다
> (`core/config.py:29~31`, `_is_lan_hostname = is_lan_hostname`). 이로써
> `core/tools/mcp/client.py`가 config의 private 이름을 패키지 경계 넘어
> import하던 구조적 결함이 해소되고, 의존성 방향이 명확해진다
> (config → security, tools/mcp → security 모두 상위 → 하위 단방향).

추가로 Layer 1(DenyRuleFilter)에 `mcp__*` 전역 deny 규칙을 두면 MCP 전체를
한 번에 끌 수 있다(비상 스위치). 이는 v6.1 라인 393~395(외부 BLOCKED)와
동일한 정신의 LAN 한정판이다.

### 3.3 compose/설정 review 체크 항목

| # | 점검 항목 | 통과 기준 |
|---|---|---|
| 1 | `mcp.servers[].base_url` | 전부 LAN 대역(192.168/10/172.16-31/localhost) |
| 2 | 외부 도메인·공인 IP | 0건 |
| 3 | TLS to public CA | 없음 (LAN 내부는 평문 HTTP 또는 사설 인증서) |
| 4 | `mcp.enabled` 기본값 | 코드 기본 `false` (에어갭 fail-closed) |
| 5 | Layer 1 deny `mcp__*` | 비상 스위치로 즉시 차단 가능 확인 |

---

## Part 4: 권한 통합 (5계층 + MCP) — 이미 존재하는 인프라 활용

### 4.1 MCP 권한 인프라는 이미 코드에 있다

v7.2의 권한 부분은 **신규 코드가 거의 없다.** 다음은 모두 현재 코드에 존재한다:

| 요소 | 위치 (확인됨) | 내용 |
|---|---|---|
| `ToolCategory.MCP` | `core/permission/types.py:117` | `MCP = "mcp"` |
| `mcp__` 도구 식별 | `core/permission/pipeline.py:191` | `if name.startswith("mcp__"): return ToolCategory.MCP` |
| MODE_BEHAVIOR_MAP MCP 열 | `core/permission/types.py:128~199` | 7개 모드 × MCP 카테고리 |
| Layer 5 write 보정 | `core/permission/pipeline.py:368` | MCP가 write 카테고리 집합에 포함 (PLAN에서 ASK→DENY) |

### 4.2 MODE_BEHAVIOR_MAP — MCP 열 (코드 인용, fail-closed)

`core/permission/types.py`에서 인용한 MCP 열의 실제 값:

| PermissionMode | MCP behavior | 의미 |
|---|---|---|
| DEFAULT | **ASK** | 기본은 사용자 확인 (fail-closed) |
| ACCEPT_EDITS | **ASK** | 파일 쓰기만 자동, MCP는 여전히 확인 |
| BUBBLE | **ASK** | 서브에이전트 — 상위로 결정 전달 |
| PLAN | **DENY** | 계획 모드는 부수효과 가능한 MCP 차단 |
| DONT_ASK | **DENY** | CI/CD — 물어볼 것은 거부 |
| BYPASS_PERMISSIONS | **ALLOW** | 개발/테스트 우회 |
| AUTO | **ALLOW** | 자동 모드 |

이 분포는 정확히 브리핑이 요구한 fail-closed 형태다: **DEFAULT/ACCEPT_EDITS/
BUBBLE=ASK, PLAN/DONT_ASK=DENY, BYPASS/AUTO=ALLOW.** v7.2는 이 표를 **수정하지
않는다.** 다만 추가로, Layer 5 보정(`pipeline.py:368`)에서 MCP는 write 카테고리
집합에 들어 있어 PLAN 모드의 ASK가 DENY로 강화된다.

### 4.3 5계층 통과 흐름 (MCP 도구 기준)

```
mcp__db__query 호출
  │
Layer 1 DenyRuleFilter   ─ mcp__* deny 규칙 있으면 즉시 제거 (비상 스위치)
  │
Layer 2 도구 고유 검사    ─ McpToolAdapter.check_permissions() → ALLOW
  │                         (경로·명령 검사 없음 — MCP는 파일·bash 도구 아님)
Layer 3 CanUseTool       ─ _categorize_tool() → ToolCategory.MCP
  │                         → MODE_BEHAVIOR_MAP[mode][MCP] 적용
Layer 4 HookPermission   ─ Hook이 BLOCK 가능 (서버 신뢰도 기반 정책 삽입점)
  │
Layer 5 context 보정     ─ PLAN: ASK→DENY, BYPASS: ASK→ALLOW
  │
ALLOW면 McpToolAdapter.call() 실행
```

5계층 전부 통과해야 실행된다(P11 위반 없음). MCP 도구 추가는 파이프라인
**구조를 바꾸지 않고**, 도구 1개를 더 흘려보낼 뿐이다.

### 4.4 신뢰 경계와 deny/allow 규칙 예시

각 MCP 서버를 **독립 신뢰 경계**로 본다. read-only 조회 서버는 완화, 쓰기
가능 서버는 보수적으로 둔다.

```yaml
# config/permission_rules.yaml (예시 — 제안)
rules:
  # DB 조회 MCP: read-only이므로 세션 허용 완화 후보
  - source: project_config
    behavior: allow
    tool_name: "mcp__db__query"        # SELECT 전용 서버
    rule_content: "사내 PG read-only 조회는 허용 (신뢰 경계: DB MCP)"

  # 진단 MCP: 도달성 점검만 — allow
  - source: project_config
    behavior: allow
    tool_name: "mcp__diag__*"
    rule_content: "운영 진단 도구는 부수효과 없음"

  # 비상 차단: MCP 전체 끄기
  - source: cli_flag
    behavior: deny
    tool_name: "mcp__*"
    rule_content: "MCP 전역 비상 차단 스위치 (필요 시 활성)"
```

### 4.5 핵심 설계 결정 — read-only-only 등록 + check_permissions 일원화

제품 보안 기준으로 두 가지가 확정되었다. **신뢰 통제는 "등록 단계"에서,
권한 판단은 "표준 5계층"에서** — 관심사를 분리한 것이 핵심이다.

**(1) 신뢰 통제를 등록 단계로 이동 (read-only-only 정책)**

`McpConnectionManager.connect_and_register`가 `trust.read_only=True`인 서버만
등록한다. 쓰기 서버는 `allow_write=True`를 운영자가 명시했을 때만 등록한다
(fail-closed, 기본 False). 신뢰되지 않은 쓰기 도구는 **도구 풀에 들어오지
못하므로 모델이 호출 자체를 할 수 없다**(Layer 1 deny와 동급의 차단을
등록 시점에서 달성).

**(2) `check_permissions`는 항상 ALLOW로 일원화**

`McpToolAdapter.check_permissions`는 read-only/쓰기 구분 없이 **항상 ALLOW**를
반환하고, 최종 판단을 전적으로 5계층 파이프라인
(`MODE_BEHAVIOR_MAP` + Layer 5 보정)에 위임한다.

**왜 이렇게 하는가 (직전 강화에서 발견한 버그)**: 한 단계 이전 강화에서는
쓰기 도구(`is_read_only=False`)를 Layer 2(`check_permissions`)에서 ASK로
반환했다. 그런데 그 ASK가 **권한 파이프라인을 조기 단락(short-circuit)** 시켜,
PLAN 모드의 최종 보정(ASK→DENY)을 무력화하는 문제가 있었다. Layer 2가
read-only/쓰기 무관하게 ALLOW로 통과시키면 다음이 정상 동작한다:

- DEFAULT 모드 → ASK
- PLAN 모드 → **DENY** (쓰기 차단 보정 정상 복구)
- BYPASS 모드 → ALLOW

즉 `allow_write`로 의도적으로 켠 쓰기 도구조차 표준 권한 정책을 그대로 받는다.

**anti-pattern #11(권한 레이어 건너뛰기) 위반 아님**: 파이프라인을 우회·단축
하지 않는다. Layer 2가 ALLOW를 내더라도 Layer 3~5가 그대로 이어져 모드별 최종
결정을 내린다. 신뢰 통제는 등록 단계로, 권한 판단은 표준 5계층으로 관심사를
분리한 것이다.

`McpServerConfig.allow_write: bool = False`(`core/config.py:171`) 필드가
이 정책을 뒷받침한다.

---

## Part 5: YAML 설정 (`mcp:` 섹션) — 구현 완료

### 5.1 McpConfig 데이터 모델 (구현 완료)

`core/config.py`에 `McpConfig`/`McpServerConfig`가 **구현되어 있다**
(`core/config.py:151~`). `ScoutConfig`·`RoutingConfig`와 동일한 Pydantic v2
패턴을 따르며, `McpServerConfig`에는 제품 보안용 **`allow_write`** 필드가,
`McpConfig`에는 LAN URL 검증용 **`@model_validator`** 가 추가되어 있다.

```python
# core/config.py (구현 발췌)
class McpServerConfig(BaseModel):
    """LAN MCP 서버 1개의 연결 설정."""
    name: str                                  # mcp__{name}__{tool}의 {name}
    transport: str = "http_sse"                # LAN HTTP/SSE만 채택 (stdio 비채택)
    base_url: str                              # 반드시 LAN 대역 — 비-LAN이면 강등
    api_key: str = "local-key"                 # LAN 내부 인증 키(기본 placeholder)
    enabled: bool = False                      # fail-closed: 명시 활성만
    trust: dict = Field(default_factory=dict)  # {"read_only": true} 등 신뢰 메타
    allow_write: bool = False                  # 쓰기 MCP 등록 명시 허용(기본 False)


class McpConfig(BaseModel):
    """v7.2 MCP 통합 설정."""
    enabled: bool = False                       # 전역 마스터 스위치 (기본 OFF)
    servers: list[McpServerConfig] = Field(default_factory=list)
    connect_timeout_sec: float = 5.0            # 연결 실패 시 fail-closed

    @model_validator(mode="after")
    def validate_lan_urls(self) -> "McpConfig":
        # base_url이 비-LAN이면 그 서버만 enabled=False로 강등(본류 무영향)
        for server in self.servers:
            if not is_lan_hostname(urlparse(server.base_url).hostname or ""):
                if server.enabled:
                    server.enabled = False
                    # warnings.warn(...) + logger.warning(...)
        return self
```

`NexusConfig`에 `mcp: McpConfig = Field(default_factory=McpConfig)`가 등록되어
있다(현재 `scout`/`routing`/`tenants`와 동일 방식).

### 5.2 YAML 섹션 (구현 완료)

```yaml
# config/nexus_config.yaml (구현됨 — 기본 비활성)
mcp:
  enabled: false              # 전역 마스터 스위치 (에어갭 fail-closed)
  connect_timeout_sec: 5.0    # 연결/요청 타임아웃 (실패 시 fail-closed)
  servers:
    # 포트 8810~8813은 모두 placeholder (Part 8 미확정). 운영 확정 전 임시값.
    - name: "db"
      transport: "http_sse"
      base_url: "http://192.168.10.39:8810"   # 사내 DB MCP (LAN, 포트 placeholder)
      enabled: false
      trust: { read_only: true }
    - name: "diag"
      transport: "http_sse"
      base_url: "http://192.168.21.112:8811"   # 운영 진단 MCP (LAN, 포트 placeholder)
      enabled: false
      trust: { read_only: true }
    - name: "docutil"
      transport: "http_sse"
      base_url: "http://192.168.10.39:8812"   # DocUtil 문서 MCP (LAN, 포트 placeholder)
      enabled: false
      trust: { read_only: true }
    - name: "kowiki"
      transport: "http_sse"
      base_url: "http://192.168.21.112:8813"   # kowiki RAG MCP (LAN, 포트 placeholder)
      enabled: false
      trust: { read_only: true }
```

> **포트 확정(서버 구현 완료, 2026-06-02)**: 서버 측 구현(`mcp_servers/`)이
> 완료되며 각 서버의 **기본 포트가 확정**되었다 — `db 8810 / diag 8811 /
> kowiki 8813 / docingest 8814`(`mcp_servers/run.py::_SERVERS`). 즉 db/diag/
> kowiki 의 placeholder 포트는 그대로 확정값이 되었고, **`docutil`(8812)은
> 서버 미구현**이며 실제 구현된 문서 MCP 는 **`docingest`(8814)** 다(Part 6.4
> 참조). 코드 기본값은 `enabled: false` — yaml에서 명시 활성해야만 연결을
> 시도한다. 모두 `trust.read_only: true`(docingest 의 `ingest` 쓰기 도구만 예외 —
> Part 6.4)이므로 read-only-only 등록 정책(Part 4.5)에 의해 등록된다.
>
> **주의(설정-구현 불일치)**: 위 yaml 의 `docutil`(8812) 항목은 서버가 구현되지
> 않았으므로 활성화해도 연결되지 않는다. 운영 전 yaml 을 실제 구현 서버
> (db/diag/kowiki/docingest)로 정렬해야 한다.

### 5.3 부트스트랩 초기화 위치 (구현 완료 — cli_registry 배선)

`core/bootstrap.py` Phase 2의 KnowledgeStore/SymbolStore 초기화 **직후**
(블록 ⑨-d, `core/bootstrap.py:367~405`)에 MCP 연결 관리자 초기화가 구현되어
있다. v7.1 `_warmup_embedding`/RAG 초기화와 동일한 **fire-and-forget + 실패
격리** 패턴이다.

> **배선 정정(구현 중 발견·수정한 버그)**: MCP 도구는 `registry`(:132)가 아니라
> **`cli_registry`에 등록한 뒤 `cli_tools`를 재취득**해야 한다. 이유:
> 블록 ⑩ `ModelDispatcher(worker_tools=cli_tools)`와 블록 ⑫
> `QueryEngine(tools=cli_tools)`에 실제로 전달되는 도구 풀은 ⑧에서
> `cli_registry.get_all_tools()`로 만든 `cli_tools`다(`:240`). 따라서
> `:132`의 `registry`에 등록하면 MCP 도구가 ModelDispatcher/QueryEngine까지
> 전달되지 않아 모델이 호출할 수 없다. 등록 후 `cli_tools`를 다시 취득해야
> worker/QueryEngine 도구 풀에 포함된다.

```python
# core/bootstrap.py Phase 2 블록 ⑨-d (구현 발췌)
if config.mcp.enabled:
    try:
        from core.tools.mcp import McpConnectionManager   # lazy import

        mcp_manager = McpConnectionManager(config.mcp)
        # ⚠ registry(:132)가 아니라 cli_registry에 등록해야 모델에 노출됨
        registered = await mcp_manager.connect_and_register(cli_registry)
        components["mcp_manager"] = mcp_manager
        # 등록된 MCP 도구가 worker/QueryEngine 풀에 포함되도록 재취득
        # (get_all_tools는 이름순 정렬 보장 → prompt cache 안정성 P5 유지)
        cli_tools = cli_registry.get_all_tools()
        logger.info("[Phase 2] MCP 연결: %s (cli_tools=%d개)",
                    {s: len(t) for s, t in registered.items()}, len(cli_tools))
    except Exception as e:  # 최후 방어선(intentional broad) — 본류 보호
        logger.warning("[Phase 2] MCP 초기화 실패 (무시): %s", e)
        components["mcp_manager"] = None
else:
    components["mcp_manager"] = None
```

여기서 `except Exception`(broad)은 의도적이다: MCP는 선택적 보조 기능이므로
어떤 예외가 나더라도 부트스트랩 본류(채팅·도구 풀)는 멈추면 안 된다. 한편
`connect_and_register`는 "예상 가능한 운영 실패"만 좁혀 서버별로 격리하고
예상치 못한 버그는 전파시키므로(Part 2.4), 결함 가시성과 본류 보호가
양립한다. registry의 이름순 정렬(`get_all_tools`)로 prompt cache 안정성(P5)도
유지된다.

---

## Part 6: 초기 검증 대상 4종 상세 (서버 측 구현 완료)

각 대상은 "사내 시스템을 LAN MCP 서버로 떼어내 재사용"의 구체 사례다.
read-only 여부는 `McpToolAdapter`의 `is_read_only` 명시적 완화 대상이며,
권한 카테고리는 전부 `ToolCategory.MCP`(이름이 `mcp__`로 시작하므로 자동).

### 6.0 MCP 서버 측 구현 (`mcp_servers/` — 구현 완료, 2026-06-02)

v7.2 초판은 서버 측(사내 시스템을 감싸 MCP 도구로 노출하는 LAN 호스트)을
"미확정/placeholder"로 두고 클라이언트 측(`core/tools/mcp/*`)만 구현 완료로
기술했다. 이제 **서버 측도 `mcp_servers/` 패키지로 구현 완료**되었으며,
클라이언트와 JSON-RPC 2.0 와이어 프로토콜 수준에서 **e2e 정합**이 검증되었다.

**구성 (`mcp_servers/`)**:

| 파일 | 역할 |
|---|---|
| `framework.py` | `McpServerTool(ABC)` + `create_mcp_app()` — JSON-RPC 2.0 단일 POST `/` 디스패치(tools/list·tools/call), Bearer 인증(fail-closed), `/health`, lifespan shutdown 훅 |
| `run.py` | `python -m mcp_servers.run <name>` uvicorn entrypoint. `_SERVERS` 포트 매핑: db 8810 / diag 8811 / kowiki 8813 / docingest 8814 |
| `db_server.py` | `query` — read-only SELECT. `_validate_read_only`(주석 제거·다중문 금지·SELECT/WITH 만 허용·DDL/DML 키워드 차단) + asyncpg `transaction(readonly=True)` 이중 방어. 최대 1000행 |
| `diag_server.py` | `reachability`(웹 HTTP/GPU SSH/DB TCP, `asyncio.to_thread` 병렬), `rag_latency`(임베딩 지연 + tb_knowledge EXPLAIN ANALYZE; pg 없으면 임베딩만 — fail-soft) |
| `kowiki_server.py` | `search` — `LocalModelProvider.embed("query: …")` → `KnowledgeStore.search_by_vector(top_k)`. pg 실패 시 인메모리 폴백 |
| `docingest_server.py` | `parse`(dry_run, read-only) / `ingest`(쓰기) / `search`(source='docingest' 필터, read-only). `DocumentIngestPipeline` + ParserRegistry(PPTX/PDF/HWPX) 조립 |

**의존성 방향 (P2 준수)**: `mcp_servers/ → core/`(코어 재사용)만 허용. `core/`는
`mcp_servers/`를 절대 import 하지 않는다(서버는 오케스트레이터가 아니라 사내
시스템을 감싸는 독립 서비스). 각 서버 모듈은 `async def build_app(api_key) ->
FastAPI`를 노출하며 db 풀 등 비동기 자원을 lifespan 에서 정리한다.

**에어갭**: 기본 host `0.0.0.0`(LAN 바인드), 모든 점검·접속 대상은 LAN
주소(192.168.x/localhost). 외부 도메인 호출·런타임 install 코드 없음.

**클라이언트↔서버 e2e 검증 (구현 완료)**: 기존 client/adapter 테스트는
`AsyncMock` 으로 서버를 흉내냈을 뿐이었다. 신규 `tests/integration/
test_mcp_client_server_e2e.py`가 **우리 `McpClient` ↔ `create_mcp_app` 서버**를
실제 연결해 전 구간을 관통 검증한다:
- 방식 a: `httpx.ASGITransport(app=app)` — 인프로세스 결정론적 연결(McpClient
  의 `_rpc`/`_parse_response`/에러 정규화가 진짜 서버 응답에 동작).
- 방식 b: 임의 포트 uvicorn 실 TCP 소켓(127.0.0.1 루프백) — 진짜 소켓 경로도 입증.
- `McpClient → McpToolAdapter` 전구간(BaseTool 래핑·tool_use_error 래핑 포함) 검증.

### 6.1 초기 검증 대상 4종 요약표

| # | 서버 | 노출 도구 이름 | read-only | 기본 포트 | 대상 LAN 위치 |
|---|---|---|---|---|---|
| 1 | DB 조회 (`db_server.py`) | `mcp__db__query` | ✓ | 8810 | PG 192.168.10.39 |
| 2 | 진단/모니터링 (`diag_server.py`) | `mcp__diag__reachability`, `mcp__diag__rag_latency` | ✓ | 8811 | 웹/GPU 22.28/DB 10.39 |
| 3 | 문서 인제스트 (`docingest_server.py`) | `mcp__docingest__parse`, `mcp__docingest__ingest`(쓰기), `mcp__docingest__search` | parse/search ✓ · ingest ✗ | 8814 | tb_knowledge + 임베딩 22.28:8002 |
| 4 | kowiki RAG (`kowiki_server.py`) | `mcp__kowiki__search` | ✓ | 8813 | tb_knowledge + 임베딩 22.28:8002 |

> **대상 3 변경**: v7.2 초판의 "DocUtil 문서 MCP(`mcp__docutil__*`, 8812)"는
> 서버가 구현되지 않았다. 실제 구현된 문서 MCP 는 **docingest**(8814)이며,
> parse/search 는 read-only, **ingest 만 쓰기 도구**다(상세 6.4).

### 6.2 대상 1 — DB 조회 MCP (`mcp__db__query`)

- **무엇**: 사내 PostgreSQL/pgvector(192.168.10.39:5440, `nexus` DB) read-only
  조회. 기존 내부 DBQueryTool/VectorSearch 로직을 **MCP 서버로 외부화**하는 경로.
- **read-only**: ✓ — SELECT 전용. MCP 서버 측에서 read-only 트랜잭션·계정
  분리로 강제. 어댑터는 `is_read_only=True`로 완화.
- **권한 카테고리**: `ToolCategory.MCP` (DEFAULT 모드에서 ASK).
- **트러스트 경계**: DB MCP 서버 = 신뢰 경계 1. DDL/DML이 서버 단에서 불가능
  해야 함(읽기 전용 DB 롤).
- **검증 포인트**:
  - (a) `tools/list`가 `mcp__db__query` 1개 노출 확인
  - (b) `SELECT count(*) FROM tb_knowledge` → 1,067,978행 반환 (v7.1 점검값과 일치)
  - (c) `DELETE`/`UPDATE` 시도 → 서버 단 거부 + tool_use_error 래핑 확인
  - (d) Layer 3에서 DEFAULT 모드 ASK 동작 확인

### 6.3 대상 2 — 운영 모니터링/진단 MCP (`mcp__diag__*`)

- **무엇**: 현재 `scripts/_diag_all_services.py`(웹/GPU/DB 도달성),
  `scripts/_diag_rag_latency*.py`(tb_memories 검색 EXPLAIN ANALYZE,
  확인됨: 3개 파일)가 하는 점검을 MCP 도구로 노출.
- **read-only**: ✓ — 도달성 점검·지연 측정만, 시스템 변경 없음.
- **노출 도구**: `mcp__diag__reachability`(웹/GPU 22.28/DB 10.39 ping),
  `mcp__diag__rag_latency`(RAG 검색 지연).
- **트러스트 경계**: 진단 MCP = 신뢰 경계 2. 읽기 전용이라 allow rule 후보.
- **검증 포인트**:
  - (a) `mcp__diag__reachability` → 3개 서비스 상태 JSON 반환
  - (b) NVML mismatch(v7.1 Part 4.3) 상황에서도 추론 도달성은 정상 보고
  - (c) 진단 MCP 서버 다운 시 fail-closed — 해당 도구만 미등록, 본류 무영향

### 6.4 대상 3 — 문서 인제스트 MCP (`mcp__docingest__*`) — 구현 완료

> v7.2 초판의 "DocUtil 문서 MCP(`mcp__docutil__*`)"를 대체한다. 실제 구현은
> 문서 파일(PPTX/PDF/HWPX)을 파싱·청킹·임베딩하여 tb_knowledge 에 적재/검색하는
> **docingest** 서버다(`mcp_servers/docingest_server.py`, 기본 포트 8814).

- **무엇**: 로컬 문서 파일을 v7.3 인제스트 파이프라인(`core/ingest/*`)으로
  파싱→청킹→임베딩→`tb_knowledge`(source='docingest') 적재·검색하는 MCP.
- **노출 도구 3개**:
  - `mcp__docingest__parse` (**read-only**) — 파싱·청킹만 하고 적재하지 않음
    (dry_run). 제목/포맷/청크 수/구조 경고 요약 반환(적재 전 점검).
  - `mcp__docingest__ingest` (**쓰기**) — 파일을 실제 적재. 동일 파일 재적재는
    멱등(UPSERT). 적재 행수·오류 목록 반환.
  - `mcp__docingest__search` (**read-only**) — source='docingest' 청크만 필터해
    의미 기반 검색(다른 소스 누설 방지).
- **read-only 구분**: parse/search 는 부작용 없음, ingest 만 DB 에 쓴다. 도구
  설명에 `[read-only]`/`[쓰기]`를 명시해 권한 파이프라인이 구분하게 한다. 쓰기
  도구(ingest)는 read-only-only 등록 정책상 `allow_write=True` 명시가 필요하다
  (Part 4.5).
- **트러스트 경계**: docingest MCP = 신뢰 경계 3.
- **검증 포인트**:
  - (a) `mcp__docingest__parse(path)` → 청크 수/경고 요약 반환(적재 없음)
  - (b) `mcp__docingest__ingest(path)` → tb_knowledge 적재 행수 반환, 재적재 멱등
  - (c) 대용량 본문은 어댑터가 `max_result_size`(BaseTool 기본 100,000자)로
    제한 — Worker 8K 컨텍스트 폭주 방지
  - (d) ingest 쓰기 도구가 `allow_write` 없이는 미등록(read-only-only 정책)임을 확인

### 6.5 대상 4 — kowiki 지식 RAG MCP (`mcp__kowiki__search`) — 대표 사례

- **무엇**: v7.0 Part 2.5.8 지식 RAG(tb_knowledge ~105만행 + 임베딩 서버
  192.168.21.112:8002의 커스텀 `/v1/embed` 계약, v7.1 Part 2 참조)를 **표준 MCP
  도구로 노출**. "사내 시스템을 MCP로 떼어내 재사용"의 가장 명확한 사례.
- **read-only**: ✓ — 벡터 검색 조회 전용.
- **노출 도구**: `mcp__kowiki__search`(질의→유사 청크 top-k).
- **트러스트 경계**: kowiki RAG MCP = 신뢰 경계 4. 임베딩 서버 호출은 MCP
  서버 내부에 캡슐화되어 Nexus는 `/v1/embed` 계약을 알 필요 없음.
- **기존 RAG와의 관계**: 현재 `core/rag/knowledge_retriever.py`는 KNOWLEDGE
  분류 질의에 **자동 주입**된다(v7.0 Part 2.5.8). MCP 버전은 **Worker가 명시적
  도구로 호출**하는 별도 경로다. 둘은 공존 가능하며 초기 검증은 후자의 타당성을 확인한다.
- **검증 포인트**:
  - (a) `mcp__kowiki__search("니체 위버멘쉬")` → 유사 청크 반환
  - (b) 내부 자동 RAG 주입 결과와 품질 비교(동등 이상이어야 외부화 정당)
  - (c) ivfflat 인덱스(v7.1 Part 1.3, tb_knowledge lists=100) 활용 지연 측정
  - (d) 임베딩 서버 cold start(v7.1 Part 5) 시 timeout → fail-closed 동작

---

## Part 7: 검증 — 설계 일관성 (v7.1 Part 6 스타일)

MCP 도입은 도구 풀에 원격 도구를 추가하는 것뿐이며, 다음 어떤 무결성도
깨지 않는다.

| 영역 | 영향 | 근거 |
|---|---|---|
| 4-Tier AsyncGenerator 체인 | **영향 없음** | MCP 도구도 query_loop→executor 경로를 그대로 통과 |
| 표준 내부 계약 (OpenAI tool_calls) | **영향 없음** | JSON-RPC는 `McpToolAdapter.call()` 아래에만 존재 (Part 2.5) |
| 24개 도구 (BaseTool) | **영향 없음** | MCP 도구는 추가 도구. 기존 24개 인터페이스 불변 |
| 5계층 권한 | **영향 없음** | MCP 카테고리·식별 로직 **이미 존재**, 활성화만 (Part 4). 어댑터 `check_permissions`는 항상 ALLOW로 5계층에 위임 (Part 4.5) |
| Hook 시스템 | **영향 없음** | MCP 도구도 동일한 Hook 파이프라인 적용 (Layer 4 삽입점) |
| Thinking Engine | **영향 없음** | Worker 추론과 직교 |
| Memory (tb_memories) | **영향 없음** | MCP는 도구 계층 |
| Training | **영향 없음** | 학습 대상 모델 불변 |
| Air-gap | **강화됨** | LAN 경계 이중 강제(config `@model_validator` + `McpClient.__init__`) + `network_guard.is_lan_hostname` 일원화 + deny 비상 스위치 (Part 3) |
| prompt cache 안정성 (P5) | **영향 없음** | registry 이름순 정렬이 MCP 도구도 포함해 보장 (bootstrap에서 cli_tools 재취득) |
| 모델·티어 | **영향 없음** | MCP는 GPU와 직교. (참고: `GPUTier.A100`이 별도 enum으로 추가됨 — v7.3 Part 4.4. MCP 동작에는 무관) |

---

## Part 8: 구현 현황 + 남은 검증 + 향후 확장 경로

### 8.1 구현 현황 (development-workflow.md 준수)

| 단계 | 작업 | 상태 | 검증 기준 |
|---|---|---|---|
| 1 | `McpConfig`/`McpServerConfig`(+`allow_write`) + yaml 파싱 + LAN URL 검증 | **구현 완료** | 비-LAN URL이 fail-closed로 강등되는 단위 테스트 (`test_mcp_config.py`) |
| 2 | `McpClient` (JSON-RPC over LAN HTTP/SSE, httpx 재사용) | **구현 완료** | mock SSE/JSON fixture로 tools/list·tools/call 파싱 검증 |
| 3 | `McpToolAdapter(BaseTool)` (+`validate_input`) | **구현 완료** | name 규칙, input_schema 매핑, tool_use_error 래핑 (`test_mcp_adapter.py`) |
| 4 | `McpConnectionManager`(read-only-only) + bootstrap ⑨-d 배선 | **구현 완료** | 1개 서버 연결 실패 시 격리. cli_registry 등록 + cli_tools 재취득 |
| 5 | 권한 통합 검증 | **구현 완료** | `mcp__db__query`가 DEFAULT=ASK / PLAN=DENY / BYPASS=ALLOW (`test_mcp_pipeline.py`) |
| 6 | **MCP 서버 측 4종**(`mcp_servers/` db/diag/kowiki/docingest) + 공통 프레임워크 | **구현 완료** | `test_mcp_server_framework.py`, `test_mcp_db_server.py`, `test_mcp_kowiki_docingest_server.py` |
| 7 | **클라이언트↔서버 e2e**(McpClient↔create_mcp_app, ASGITransport + 실 TCP, McpClient→McpToolAdapter 전구간) | **구현 완료** | `tests/integration/test_mcp_client_server_e2e.py` |
| 8 | GlobalState `mcp_servers`/`mcp_connected` 가시성 + `/metrics` 노출 | **구현 완료** | `core/state.py:131`, `core/bootstrap.py:391~395`, `web/app.py:1206~`, `test_mcp_visibility.py` |
| 9 | 검증 대상 e2e — **실 LAN 서버 배포 후** 6.x 시나리오 | **미착수(후속)** | 6.x 검증 포인트 — 실 LAN MCP 서버를 별도 호스트에 배포해 수행 |

### 8.2 테스트 전략 (testing.md 준수) — 구현 완료

- 클라이언트 측: mock(`httpx.AsyncClient` mock + JSON-RPC fixture)로 tools/list·
  tools/call 파싱 검증
- 서버 측(신규): `create_mcp_app` 을 `httpx.ASGITransport` 로 인프로세스 호출해
  JSON-RPC 디스패치/Bearer 인증(fail-closed)/도구 실행/에러 코드 매핑 검증
- 클라이언트↔서버 e2e(신규): 실 McpClient ↔ 실 서버(ASGITransport + 임의 포트
  uvicorn 실 TCP), McpClient→McpToolAdapter 전구간(127.0.0.1 루프백 — 에어갭 준수)
- 권한: Layer 1 deny(`mcp__*`)/Layer 3 MODE_BEHAVIOR_MAP/Layer 5 PLAN 보정 시나리오
- fail-closed: 연결 실패·비-LAN URL·timeout 3종이 모두 본류 무영향임을 검증
- read-only-only 등록 정책: 쓰기 서버(docingest ingest)가 `allow_write` 없이는 미등록됨을 검증
- top_k 견고성: kowiki/docingest/diag search 의 `top_k=0/음수/bool` 거부, 미지정 시에만 기본값 적용
- 파일(구현됨): 클라이언트 — `test_mcp_config.py`, `test_mcp_adapter.py`,
  `test_mcp_pipeline.py`; 서버 — `test_mcp_server_framework.py`,
  `test_mcp_db_server.py`, `test_mcp_kowiki_docingest_server.py`; e2e —
  `test_mcp_client_server_e2e.py`; 가시성 — `test_mcp_visibility.py`
- **테스트 결과(실측, 2026-06-02)**: 전체 회귀 **1091 passed, 1 skipped**.
  유일한 1 failed 는 `tests/e2e/test_gpu_e2e.py::test_simple_chat_completion`
  — 실 GPU 서버 의존 e2e 로 MCP/인제스트와 무관(코드 회귀 아님).

### 8.3 향후 정식 운영 시 확장 경로

| 확장 | 내용 |
|---|---|
| GlobalState 가시성 | **구현 완료** — `mcp_servers: dict`/`mcp_connected: set` 가 `core/state.py:131`에 추가됨. bootstrap ⑨-d(`core/bootstrap.py:391~395`)가 등록 결과로 채우고, `web/app.py:1206~`의 `/metrics`에 `mcp.connected_count`/`connected`/서버별 도구 목록으로 노출된다 |
| MCP 서버 측 표준화 | **구현 완료** — `mcp_servers/`(framework + db/diag/kowiki/docingest)로 사내 시스템을 LAN MCP 서버로 노출. 외부 MCP 호환 클라이언트도 동일 JSON-RPC 2.0 프로토콜로 재사용 가능 |
| 실 LAN 서버 배포 e2e | **후속(미착수)** — 현재 e2e 는 인프로세스(ASGITransport) + 루프백 TCP 검증까지다. 별도 호스트에 서버를 배포해 Part 6.x 운영 시나리오를 수행하는 것은 다음 단계 |
| 쓰기 가능 MCP | 초기 제품 정책은 read-only 4종 한정. 쓰기 MCP는 `allow_write=True` 명시 등록 + 표준 5계층 권한(DEFAULT=ASK/PLAN=DENY) 적용 |
| 동시성 | 현재 MCP 어댑터는 `is_concurrency_safe=False`(기본). read-only 검증 후 병렬 완화 가능 (P12 concurrency partitioning) |
| TIER/GPU 무관성 | MCP는 모델·티어와 직교 — TIER_S/M/L, A100/H100/H200 어디서나 동일 동작. Scout/라우팅과 달리 티어별 비활성화 불필요 |

---

## 부록 A: v7.2에서 정정/명문화한 사항

| # | 항목 | v7.2 입장 | 위치 | 코드 상태 |
|---|---|---|---|---|
| 1 | v6.1 "MCP 에어갭 불가" | LAN 한정 활성화로 정정 | Part 1 | — |
| 2 | LAN MCP vs 외부 SaaS MCP | LAN 허용 / 외부 금지 명문화 | Part 1.3, 3.2 | — |
| 3 | `ToolCategory.MCP` | **이미 존재** (`types.py:117`) | Part 4.1 | 존재 |
| 4 | `mcp__` 도구 식별 | **이미 존재** (`pipeline.py:191`) | Part 4.1 | 존재 |
| 5 | MODE_BEHAVIOR_MAP MCP 열 | **이미 존재**, fail-closed | Part 4.2 | 존재 |
| 6 | `McpToolAdapter`/`McpClient`/연결 관리자 | **구현됨** (`core/tools/mcp/*`) | Part 2 | 구현됨 |
| 7 | `McpConfig`/`McpServerConfig`/yaml `mcp:` 섹션 | **구현됨** | Part 5 | 구현됨 |
| 8 | Transport | LAN HTTP/SSE 확정, stdio 비채택 | Part 3.1 | — |
| 9 | read-only-only 등록 정책 + `allow_write` | **구현됨** (등록 단계 신뢰 통제) | Part 4.5 | 구현됨 |
| 10 | `check_permissions` 항상 ALLOW (5계층 위임) | **구현됨** (ASK 단락 버그 해소) | Part 4.5 | 구현됨 |
| 11 | LAN 판정 `network_guard.is_lan_hostname` 이전 | **구현됨** (`_is_lan_hostname` 별칭 유지) | Part 3.2 | 구현됨 |
| 12 | LAN URL 이중 강제 | **구현됨** (config validator + McpClient) | Part 3.2 | 구현됨 |
| 13 | bootstrap cli_registry 배선 | **구현됨** (배선 버그 정정) | Part 5.3 | 구현됨 |
| 14 | GPU 티어 A100 | v7.2 초판은 "A100→H100 매핑(별도 enum 불필요)"였으나, **v7.3에서 `GPUTier.A100` 별도 enum 추가**(이름 기반 판정). MCP 동작과는 무관 | v7.3 Part 4.4 | 구현됨 |
| 15 | GlobalState `mcp_servers`/`mcp_connected` | **구현됨** — `/metrics` 노출 포함 | Part 8.3 | 구현됨 |
| 16 | MCP 서버 측(`mcp_servers/` db/diag/kowiki/docingest) | **구현됨** — 클라이언트↔서버 e2e 검증 | Part 6.0 | 구현됨 |
| 17 | 검증 대상 3: `docutil` → `docingest` | **정정** — docutil 미구현, docingest(parse/ingest/search) 구현 | Part 6.4 | 구현됨 |

> **GlobalState 가시성(구현 완료, 2026-06-02)**: `mcp_servers: dict`/
> `mcp_connected: set`가 `core/state.py:131`에 추가되었다. bootstrap ⑨-d
> (`core/bootstrap.py:391~395`)가 `connect_and_register` 결과로 채우며(빈
> 리스트는 connected 에서 자연 제외 — fail-closed 요약), `web/app.py`의
> `/metrics`(`:1206~`)가 `mcp.connected_count`/`connected`/서버별 도구 목록을
> 노출한다. v7.2 초판 시점에 이미 존재하던 인프라는 `ToolCategory.MCP`(types.py)
> 와 `mcp__` 식별(pipeline.py)이었고, v7.2에서 클라이언트(어댑터·설정·연결
> 관리자)가, 2026-06-02 갱신에서 **서버 측(`mcp_servers/`)·e2e·GlobalState
> 가시성**이 추가 구현되었다.

---

## 부록 B: 코드 위치 매트릭스 (이미 존재 vs 구현됨 vs 미존재)

| 사양서 항목 | 코드 위치 | 상태 |
|---|---|---|
| `ToolCategory.MCP` | `core/permission/types.py:117` | **이미 존재** |
| MODE_BEHAVIOR_MAP MCP 열 (7모드) | `core/permission/types.py:128~199` | **이미 존재** |
| `mcp__` prefix → MCP 분류 | `core/permission/pipeline.py:191` | **이미 존재** |
| Layer 5 write 보정에 MCP 포함 | `core/permission/pipeline.py:368` | **이미 존재** |
| `BaseTool` ABC (어댑터 상속 대상) | `core/tools/base.py:106` | **이미 존재** |
| `ToolRegistry.register`/이름순 정렬 | `core/tools/registry.py:40, 76` | **이미 존재** |
| 13단계 ToolExecutor | `core/tools/executor.py` | **이미 존재** |
| Phase 2 초기화 패턴 (ensure_schema 등) | `core/bootstrap.py:90~360` | **이미 존재** |
| `ScoutConfig`/`RoutingConfig` 패턴 | `core/config.py:117, 209` | **이미 존재** (McpConfig 참조 모델) |
| 진단 스크립트 (diag MCP 원천) | `scripts/_diag_all_services.py`, `_diag_rag_latency*.py` | **이미 존재** |
| 임베딩 `/v1/embed` (kowiki MCP 원천) | `core/model/inference.py::embed` (v7.1 Part 2) | **이미 존재** |
| tb_knowledge (kowiki MCP 원천) | `core/rag/knowledge_store.py` | **이미 존재** |
| `McpClient` (JSON-RPC over LAN HTTP/SSE) | `core/tools/mcp/client.py` | **구현됨** |
| `McpToolAdapter(BaseTool)` (+`validate_input`) | `core/tools/mcp/adapter.py` | **구현됨** |
| `McpConnectionManager` (read-only-only, aclose_all) | `core/tools/mcp/connection_manager.py` | **구현됨** |
| 패키지 공개 심볼 | `core/tools/mcp/__init__.py` | **구현됨** |
| `McpConfig`/`McpServerConfig`(+`allow_write`) | `core/config.py:151, 174` | **구현됨** |
| `McpConfig.validate_lan_urls` (LAN 강등 validator) | `core/config.py:191~` | **구현됨** |
| LAN 판정 헬퍼 (public) | `core/security/network_guard.py::is_lan_hostname` | **구현됨** |
| `_is_lan_hostname` 하위호환 별칭 | `core/config.py:29~31` | **구현됨** |
| yaml `mcp:` 섹션 | `config/nexus_config.yaml:238~` | **구현됨** |
| bootstrap MCP 연결 배선 (cli_registry) | `core/bootstrap.py:367~405` (블록 ⑨-d) | **구현됨** |
| MCP 단위 테스트 (config/adapter) | `tests/unit/test_mcp_config.py`, `test_mcp_adapter.py` | **구현됨** |
| MCP 통합 테스트 (5계층 파이프라인) | `tests/integration/test_mcp_pipeline.py` | **구현됨** |
| **MCP 서버 공통 프레임워크** | `mcp_servers/framework.py` (`McpServerTool`/`create_mcp_app`) | **구현됨** |
| **MCP 서버 entrypoint** | `mcp_servers/run.py` (`python -m mcp_servers.run <name>`) | **구현됨** |
| **db MCP 서버** (read-only SELECT) | `mcp_servers/db_server.py` (포트 8810) | **구현됨** |
| **diag MCP 서버** (reachability/rag_latency) | `mcp_servers/diag_server.py` (포트 8811) | **구현됨** |
| **kowiki MCP 서버** (embed+pgvector search) | `mcp_servers/kowiki_server.py` (포트 8813) | **구현됨** |
| **docingest MCP 서버** (parse/ingest/search) | `mcp_servers/docingest_server.py` (포트 8814) | **구현됨** |
| **MCP 서버 단위 테스트** | `tests/unit/test_mcp_server_framework.py`, `test_mcp_db_server.py`, `test_mcp_kowiki_docingest_server.py` | **구현됨** |
| **클라이언트↔서버 e2e 테스트** | `tests/integration/test_mcp_client_server_e2e.py` | **구현됨** |
| GlobalState `mcp_servers`/`mcp_connected` | `core/state.py:131` | **구현됨** |
| MCP 가시성 bootstrap 배선 + `/metrics` | `core/bootstrap.py:391~395`, `web/app.py:1206~` | **구현됨** |
| MCP 가시성 테스트 | `tests/unit/test_mcp_visibility.py` | **구현됨** |

---

*작성일: 2026-06-01 (서버 측·e2e·가시성 갱신: 2026-06-02)*
*기준 운영 점검: tb_knowledge 1,067,978행 (DB MCP 검증값)*
*테스트 실측(2026-06-02): 전체 회귀 1091 passed·1 skipped*
*(1 failed = test_gpu_e2e 실 GPU 의존 e2e, MCP/인제스트 무관)*
*코드 확인 기준(2026-06-02): mcp_servers/{framework,run,db_server,diag_server,*
*kowiki_server,docingest_server}.py, core/tools/mcp/*, core/config.py,*
*core/security/network_guard.py, core/bootstrap.py(블록 ⑨-d), core/state.py,*
*core/model/gpu_detector.py, core/permission/{types,pipeline}.py, web/app.py(/metrics),*
*config/nexus_config.yaml, tests/unit/test_mcp_{config,adapter,server_framework,*
*db_server,kowiki_docingest_server,visibility}.py,*
*tests/integration/test_mcp_{pipeline,client_server_e2e}.py*
