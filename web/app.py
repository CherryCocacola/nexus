"""
FastAPI 웹 인터페이스 — HTTP API 서버.

CLI 외에 HTTP API로도 Nexus를 사용할 수 있게 한다.
SSE 스트리밍, 세션 관리, 도구/모델 조회 등을 제공한다.

의존성 방향: web/ → core/ (단방향)

엔드포인트:
  POST /v1/chat          — 비스트리밍 채팅
  POST /v1/chat/stream   — SSE 스트리밍 채팅
  GET  /v1/sessions      — 세션 목록 조회 (title_hint 포함)
  GET  /v1/sessions/{session_id}/messages — 특정 세션 대화 복원 (Ch 16)
  DELETE /v1/sessions/{session_id} — 특정 세션 삭제 (Redis + 트랜스크립트)
  GET  /v1/tools         — 도구 목록 조회
  GET  /v1/models        — 모델 목록 조회
  GET  /v1/tenants       — 테넌트 목록 조회 (멀티테넌시, Part 5 Ch 15)
  GET  /health           — 헬스체크
  GET  /metrics          — 메트릭스 조회
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
import uuid
from collections import OrderedDict
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

# ─────────────────────────────────────────────
# thinking 태그/찌꺼기 정제 헬퍼
# ─────────────────────────────────────────────
# Qwen3.5 chat template이 <think>...</think> 블록을 삽입할 수 있다.
# enable_thinking=false로 대부분 예방되지만, 과거 세션이나 예외 상황에서
# 찌꺼기가 들어오면 다음 턴에서 Worker가 그 스타일을 모방할 수 있다.
_THINK_BLOCK = re.compile(r"<think>.*?</think>", flags=re.DOTALL)
_DANGLING_THINK_TAIL = re.compile(r"^.*?</think>\s*", flags=re.DOTALL)


def _strip_thinking(text: str) -> str:
    """<think>...</think> 블록 + 비정상 잘린 </think> 접두를 제거한다."""
    if not text:
        return text
    cleaned = _THINK_BLOCK.sub("", text)
    # 여는 <think> 없이 닫는 </think>만 남은 경우(스트리밍 중단 등)
    if "</think>" in cleaned and "<think>" not in cleaned:
        cleaned = _DANGLING_THINK_TAIL.sub("", cleaned)
    return cleaned.strip()


def _sanitize_history_inplace(history: list) -> None:
    """히스토리에 저장된 Message 중 thinking 찌꺼기가 있으면 정제된 Message로 교체."""
    from core.message import Message

    for i, msg in enumerate(history):
        role = msg.role if isinstance(msg.role, str) else msg.role.value
        if role not in ("user", "assistant"):
            continue
        content = msg.text_content if hasattr(msg, "text_content") else str(msg.content)
        if content and ("<think>" in content or "</think>" in content):
            cleaned = _strip_thinking(content)
            if cleaned:
                history[i] = (
                    Message.assistant(cleaned) if role == "assistant" else Message.user(cleaned)
                )


from fastapi import FastAPI, Header, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from web.middleware import ApiKeyAuthMiddleware, CORSConfig, RequestLoggingMiddleware

logger = logging.getLogger("nexus.web.app")


# ─────────────────────────────────────────────
# 멀티테넌시 해석 헬퍼 (Part 5 Ch 15, 2026-04-21)
# ─────────────────────────────────────────────
# 우선순위: body.tenant_id > X-Tenant-ID 헤더 > Authorization Bearer(API 키)
#          > 레지스트리의 default_tenant
def _resolve_tenant(
    body_tenant_id: str | None,
    header_tenant_id: str | None,
    authorization: str | None,
) -> Any:
    """요청 컨텍스트에서 TenantConfig를 해석한다. 항상 유효한 객체 반환."""
    registry = _app_state.get("tenant_registry")
    if registry is None:
        return None

    found = None
    # 1) body
    if body_tenant_id:
        found = registry.get(body_tenant_id)
    # 2) X-Tenant-ID 헤더
    if found is None and header_tenant_id:
        found = registry.get(header_tenant_id)
    # 3) Authorization Bearer — API key 기반
    if found is None and authorization and authorization.lower().startswith("bearer "):
        api_key = authorization[7:].strip()
        if api_key:
            found = registry.resolve_by_api_key(api_key)
    # 4) 기본 테넌트 폴백
    if found is None:
        found = registry.resolve(None)

    # per-tenant 카운트 누적 (M6)
    stats = _app_state.setdefault("tenant_stats", {})
    entry = stats.setdefault(found.id, {"requests": 0})
    entry["requests"] += 1
    return found


def _build_transcript(session_id: str) -> Any:
    """세션 ID별 SessionTranscript 인스턴스를 만든다. config 실패 시 None 반환.

    /v1/chat과 /v1/chat/stream 두 핸들러에서 중복되던 로직을 한 지점으로 모음
    (2026-04-21 리팩토링).
    """
    try:
        from core.memory.transcript import SessionTranscript as _Trans

        cfg = _app_state.get("config")
        sessions_dir = cfg.session.sessions_dir if cfg else ".nexus/sessions"
        transcript_enabled = cfg.session.transcript_enabled if cfg else True
        return _Trans(
            sessions_dir=sessions_dir,
            session_id=session_id,
            enabled=transcript_enabled,
        )
    except Exception as e:
        logger.warning("트랜스크립트 생성 실패 (%s): %s", session_id, e)
        return None


# ─────────────────────────────────────────────
# 웹 QueryEngine 조립 헬퍼 (2026-04-21 리팩토링 3)
# ─────────────────────────────────────────────
# lifespan() 한 함수에 뭉쳐 있던 Scout 풀 합치기·시스템 프롬프트 빌드·ModelDispatcher
# 구성을 독립 함수로 분리해 가독성과 테스트 용이성을 확보한다.
def _combine_scout_pool(web_tools: list, scout_tools: list) -> list:
    """웹 도구 + Scout 도구를 name 중복 제거 후 하나의 풀로 합친다."""
    combined: list = []
    seen: set[str] = set()
    for t in [*web_tools, *scout_tools]:
        if t.name not in seen:
            combined.append(t)
            seen.add(t.name)
    return combined


def _load_worker_system_prompt(agent_registry: Any | None, tier: Any = None) -> str:
    """
    Worker 시스템 프롬프트를 하드웨어 티어에 맞춰 로드하고,
    AgentRegistry로부터 서브에이전트 가이드를 동적으로 추가한다(B200 Phase 2).

    티어별 프롬프트 파일:
      - TIER_S       → `web/prompts/worker_system.md`      (현행: Scout 위임)
      - TIER_M/L     → `web/prompts/worker_system_full.md` (탐색 도구 직접 사용)

    tier가 None이거나 알 수 없는 값이면 TIER_S(worker_system.md)로 폴백한다
    = fail-closed(현행 동작 유지). TIER_M/L 파일이 없으면 worker_system.md로,
    그것도 없으면 하드코딩 문자열로 단계적 폴백해 항상 유효한 프롬프트를 보장한다.
    """
    from core.model.hardware_tier import HardwareTier

    # enum이면 .value, 문자열이면 그대로 비교(둘 다 허용). 미매칭 시 TIER_S.
    tier_val = getattr(tier, "value", tier)
    is_expanded = tier_val in (HardwareTier.TIER_M.value, HardwareTier.TIER_L.value)
    fname = "worker_system_full.md" if is_expanded else "worker_system.md"

    prompts_dir = Path(__file__).parent / "prompts"
    prompt_path = prompts_dir / fname
    base: str | None = None
    try:
        base = prompt_path.read_text(encoding="utf-8")
    except OSError as e:
        logger.warning("Worker 프롬프트 파일 읽기 실패 (%s): %s", prompt_path, e)
        # TIER_M/L 전용 파일이 없으면 현행 TIER_S 프롬프트로 폴백.
        if fname != "worker_system.md":
            fallback_path = prompts_dir / "worker_system.md"
            try:
                base = fallback_path.read_text(encoding="utf-8")
                logger.warning("worker_system.md로 폴백: %s", fallback_path)
            except OSError:
                base = None
    if base is None:
        base = (
            "You are Nexus, the Worker agent developed by IDINO.\n"
            "Respond in the user's language. Be helpful and detailed."
        )

    if agent_registry is not None and len(agent_registry) > 0:
        agent_lines = [
            f"  - {name}: {desc}" for name, desc in agent_registry.list_descriptions().items()
        ]
        base += (
            "\n\n## Sub-agents (Agent tool)\n"
            "Delegate specialized tasks to sub-agents via the Agent tool.\n"
            "Available sub-agents:\n" + "\n".join(agent_lines) + "\n\nWhen to use sub-agents:\n"
            "  - Simple questions or greetings → answer directly, NO tools\n"
            "  - Single file task → use Read/Edit/Write directly\n"
            '  - Broad project exploration → Agent(subagent_type="scout")\n'
            "NEVER invoke scout for trivial tasks — it is slow (~30s on CPU)."
        )
    return base


def _build_web_engine_parts(components: dict, state: Any) -> dict:
    """
    요청과 무관한(상태 없는) 무거운 부품을 '한 번만' 조립해 모아 반환한다.

    왜 분리하는가 (동시성 결함 수정 — 감사 Critical #5, 2026-07-03):
      기존에는 QueryEngine을 앱 전역 싱글톤 1개로 두고, 모든 HTTP 요청이 그 하나의
      `_messages`/`_session_id`/tenant 상태를 락 없이 덮어써서 두 사용자의 대화·
      테넌트가 뒤섞였다(멀티테넌트 프로덕션 최대 블로커). 이를 고치려면 요청/세션
      별로 **독립된 QueryEngine 상태**를 써야 한다. 다만 도구 레지스트리 생성·시스템
      프롬프트 조립·MCP 머지는 무겁고 '요청과 무관'하므로, 여기서 딱 한 번 만들고
      모든 세션이 공유 재사용한다. 세션별로 격리하는 건 오직 _messages/_session_id/
      tenant/context 뿐이다(→ _assemble_session_engine).

    공유해도 안전한 이유(내부 mutable 상태 없음):
      model_provider·tool 인스턴스·knowledge_retriever·memory_manager 등은 요청별
      가변 상태를 인스턴스 필드에 담지 않고, 호출 시점에 넘어오는 context/인자로
      동작한다. 따라서 여러 세션이 동시에 참조해도 서로를 오염시키지 않는다.
      (반대로 tenant/cwd는 요청마다 다르므로 반드시 세션 context로 격리한다.)
    """
    from core.bootstrap import _create_web_tool_registry

    # 하드웨어 티어를 넘겨 웹 도구 풀을 티어별로 구성한다(B200 Phase 2).
    # TIER_S: 현행 5개, TIER_M/L: +Read/Glob/Grep/LS/DocumentProcess/GitDiff.
    # components에 hardware_tier가 없으면 None → _create_web_tool_registry가
    # TIER_S(5개)로 폴백 = fail-closed.
    web_registry = _create_web_tool_registry(components.get("hardware_tier"))
    web_tools = web_registry.get_all_tools()
    # v7.2 MCP — 부트스트랩이 LAN MCP 서버에서 등록한 도구(mcp__db__query 등)를
    # 웹 Worker 풀에도 흡수한다. 웹은 cli_registry가 아니라 _create_web_tool_registry
    # 로 자체 풀을 만들기 때문에, 이 머지가 없으면 모델이 MCP 도구를 볼 수 없다
    # (실측: Worker가 mcp__db__query를 못 보고 npx 셸 명령을 환각 호출하던 버그).
    # _combine_scout_pool과 동일한 name 중복 제거 규칙으로 합친다(P5 이름순 정렬은
    # registry/get_all_tools가 이미 보장 → prompt cache 안정).
    mcp_tools = components.get("mcp_tools") or []
    if mcp_tools:
        web_tools = _combine_scout_pool(web_tools, mcp_tools)
    scout_tools = components.get("scout_tools") or []
    combined_pool = _combine_scout_pool(web_tools, scout_tools)

    # 컨텍스트 예산(하드코딩 외부화, 2026-07-03). 실 NexusConfig에는 항상 존재하나,
    # 부분 config/테스트 더블에는 없을 수 있어 getattr로 방어(없으면 None → 하위가
    # 현행 상수로 폴백 = 무회귀). fail-closed가 아니라 fail-safe: 예산 미제공이
    # 곧 "현행 기본값 사용"을 의미하므로 엔진 조립을 막지 않는다.
    _web_budgets = getattr(state.config, "context_budgets", None)

    # ── 세션 샌드박스 '입력값'만 보관 ──
    # 실제 cwd 결정은 세션 ID가 정해지는 요청 시점(_session_sandbox_cwd)에서 한다.
    # 예전엔 부트스트랩 시점의 state.session_id 하나로 cwd를 고정해, 모든 세션이
    # 같은 샌드박스를 공유하는 잠재 결함이 있었다(이번 격리로 함께 해소).
    _pe_cfg = getattr(state.config, "permission_enforcement", None)
    _pe_enabled = bool(getattr(_pe_cfg, "enabled", False))
    _sessions_dir = getattr(getattr(state.config, "session", None), "sessions_dir", None)

    # AgentTool·SymbolSearchTool이 해석할 의존성 일체 — tenant를 제외한 '공용' 옵션.
    # 세션별 assemble에서 {**base_options, "tenant": tenant}로 얕은 복제해 격리한다.
    base_options = {
        "memory_manager": components.get("memory_manager"),
        "task_manager": components.get("task_manager"),
        "agent_registry": components.get("agent_registry"),
        "model_provider": components["model_provider"],
        "scout_provider": components.get("scout_provider"),
        "available_tools": combined_pool,
        "symbol_store": components.get("symbol_store"),  # Phase 10.0
        # 문서 청크 크기 — 하드코딩 외부화(2026-07-03). DocumentProcess가 읽음.
        # 예산 미제공(None)이면 도구가 CHUNK_SIZE(2500)로 폴백.
        "document_chunk_size": (
            _web_budgets.document_chunk_size if _web_budgets else None
        ),
    }

    # 시스템 프롬프트는 파일 읽기 + 서브에이전트 가이드 조립이라 비교적 무겁다 →
    # 한 번만 만들어 문자열로 공유한다(요청마다 다시 읽지 않는다).
    system_prompt = _load_worker_system_prompt(
        components.get("agent_registry"), components.get("hardware_tier")
    )

    return {
        "tier": components["hardware_tier"],
        "worker_provider": components["model_provider"],
        "scout_provider": components.get("scout_provider"),
        "web_tools": web_tools,
        "scout_tools": scout_tools,
        "combined_pool": combined_pool,
        "context_manager": components.get("context_manager"),
        "memory_manager": components.get("memory_manager"),
        "knowledge_retriever": components.get("knowledge_retriever"),
        "system_prompt": system_prompt,
        "routing_config": state.config.routing,
        "budgets": _web_budgets,
        "base_options": base_options,
        "permission_mode": state.permission_mode.value,
        "base_cwd": state.cwd or ".",
        "pe_enabled": _pe_enabled,
        "sessions_dir": _sessions_dir,
    }


def _session_sandbox_cwd(parts: dict, session_id: str) -> str:
    """세션별 샌드박스 작업 디렉토리를 결정한다.

    ★무회귀★: permission_enforcement.enabled=False(기본)면 항상 base_cwd(프로젝트
    루트)를 그대로 쓴다 — 파일 도구의 상대경로 해석이 현행과 100% 동일. 강제가
    켜진 경우에만 세션별 격리 디렉토리({sessions_dir}/{session_id}/workspace)로
    전환한다. PathGuard의 cwd-scope 순회 검사가 이 디렉토리를 기준으로 동작하므로,
    세션 밖(상위/시스템 경로) 쓰기가 자연히 차단된다.
    """
    base_cwd = parts["base_cwd"]
    if parts["pe_enabled"] and parts["sessions_dir"] and session_id:
        candidate = os.path.join(parts["sessions_dir"], str(session_id), "workspace")
        try:
            os.makedirs(candidate, exist_ok=True)
            return candidate
        except OSError as e:
            # 디렉토리 생성 실패 시 기존 cwd로 폴백(본류를 막지 않는다).
            logger.warning("[web] 세션 샌드박스 생성 실패, 기본 cwd 사용: %s", e)
    return base_cwd


def _assemble_session_engine(parts: dict, session_id: str, tenant: Any) -> tuple[Any, Any]:
    """
    공유 부품(parts)으로 '세션 전용' QueryEngine을 가볍게 조립한다.

    반환: (engine, dispatcher). dispatcher를 engine.model_dispatcher로 되꺼내지 않고
    직접 돌려주는 이유는, engine을 mock으로 대체하는 단위 테스트에서도 실제 조립된
    dispatcher를 검증할 수 있게 하기 위함이다.

    세션마다 새로 만드는 것(격리 대상):
      - ToolUseContext: session_id / cwd(샌드박스) / options["tenant"]
      - ModelDispatcher: 위 격리 context를 바인딩해 도구 실행이 올바른 tenant를 봄
        (dispatcher.route → query_loop → 도구가 context.options["tenant"]를 읽는다.
         만약 dispatcher를 공유하면 tenant가 세션 간 새어나간다 → 반드시 세션별 생성)
      - QueryEngine: _messages / _session_id / _cumulative_usage 등 요청별 가변 상태

    무거운 부품(도구 인스턴스·시스템 프롬프트·프로바이더·retriever)은 parts에서
    공유 재사용한다. 이 조립은 객체 참조 저장 + 로그 몇 줄 수준이라 매우 가벼워
    (수 마이크로초), 모델 추론(초 단위) 대비 무시할 만한 비용이다 → 요청/세션별
    생성이 정당하다.
    """
    from core.orchestrator.model_dispatcher import ModelDispatcher
    from core.orchestrator.query_engine import QueryEngine
    from core.tools.base import ToolUseContext

    context = ToolUseContext(
        cwd=_session_sandbox_cwd(parts, session_id),
        session_id=session_id,
        permission_mode=parts["permission_mode"],
        # 공용 base_options를 '새 dict'로 얕은 복제한 뒤 이 세션의 tenant만 얹는다.
        # (원본 base_options를 mutate하지 않아야 다른 세션이 오염되지 않는다.)
        options={**parts["base_options"], "tenant": tenant},
    )

    dispatcher = ModelDispatcher(
        tier=parts["tier"],
        worker_provider=parts["worker_provider"],
        worker_tools=parts["web_tools"],
        context=context,
        scout_provider=parts["scout_provider"],
        scout_tools=parts["scout_tools"],
        max_turns=200,
    )

    engine = QueryEngine(
        model_provider=parts["worker_provider"],
        tools=parts["web_tools"],
        context=context,
        model_dispatcher=dispatcher,
        context_manager=parts["context_manager"],
        memory_manager=parts["memory_manager"],
        knowledge_retriever=parts["knowledge_retriever"],
        system_prompt=parts["system_prompt"],
        max_turns=200,
        routing_config=parts["routing_config"],
        # 컨텍스트 예산(하드코딩 외부화, 2026-07-03) — RAG 주입 예산 + 출력
        # 토큰 에스컬레이션. 실 config는 항상 존재, 없으면 None → 현행 상수 폴백.
        context_budgets=parts["budgets"],
    )
    return engine, dispatcher


def _build_web_query_engine(components: dict, state: Any) -> Any:
    """
    (하위 호환) Phase 2 부트스트랩 결과로 웹 QueryEngine 1개를 조립해
    (query_engine, dispatcher, web_tools) 3-튜플로 반환한다.

    동시성 수정(2026-07-03) 이후 실제 요청 처리는 세션별 엔진
    (_acquire_session_engine → _assemble_session_engine)을 쓴다. 이 함수는 부품
    조립 + MCP 머지 결과를 검증하는 기존 단위 테스트(tests/unit/test_mcp_web_pool.py)
    와의 계약(3-튜플 반환, web_tools에 MCP 머지)을 유지하기 위해 남긴다.

    web_tools(= MCP 머지 후 실제 Worker 도구 풀)를 반환한다 — /v1/tools가 모델이
    실제로 보는 도구와 일치하도록.
    """
    parts = _build_web_engine_parts(components, state)
    engine, dispatcher = _assemble_session_engine(parts, state.session_id or "", tenant=None)
    return engine, dispatcher, parts["web_tools"]


# ─────────────────────────────────────────────
# 요청/응답 모델 (Pydantic v2)
# ─────────────────────────────────────────────
class ChatRequest(BaseModel):
    """채팅 요청 모델."""

    message: str = Field(..., description="사용자 메시지")
    session_id: str | None = Field(default=None, description="세션 ID (없으면 새 세션 생성)")
    model: str = Field(
        default="primary",
        description="사용할 모델 (primary: Qwen 3.5, auxiliary: ExaOne)",
    )
    # 멀티테넌시 (Part 5 Ch 15) — body로도 지정 가능 (헤더/API 키와 병행)
    tenant_id: str | None = Field(
        default=None,
        description="테넌트 ID. 헤더 X-Tenant-ID / Authorization Bearer와 같은 우선순위 중 하나.",
    )


class ToolCallInfo(BaseModel):
    """도구 호출 정보."""

    name: str
    input_data: dict[str, Any] = Field(default_factory=dict, alias="input")
    result: str | None = None
    is_error: bool = False


class UsageInfo(BaseModel):
    """토큰 사용량 정보."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


class ChatResponse(BaseModel):
    """채팅 응답 모델."""

    session_id: str = Field(description="세션 ID")
    response: str = Field(description="assistant 응답 텍스트")
    tool_calls: list[ToolCallInfo] = Field(
        default_factory=list, description="실행된 도구 호출 목록"
    )
    usage: UsageInfo = Field(default_factory=UsageInfo, description="토큰 사용량")


class ToolInfo(BaseModel):
    """도구 정보 (목록 조회용)."""

    name: str
    description: str
    group: str = ""
    is_read_only: bool = False


class ModelInfo(BaseModel):
    """모델 정보 (목록 조회용)."""

    id: str
    name: str
    role: str  # primary, auxiliary, embedding


class TenantInfo(BaseModel):
    """테넌트 정보 (목록 조회용).

    Part 5 Ch 15 멀티테넌시 등록부를 외부에 노출한다.
    보안상 `api_keys` 원본은 절대 반환하지 않고, 개수(`api_key_count`)만 노출한다.
    """

    # Pydantic의 model_ 접두사 경고를 막는다 (model_override 필드명 때문).
    model_config = {"protected_namespaces": ()}

    id: str
    name: str = ""
    description: str = ""
    model_override: str | None = None
    allowed_knowledge_sources: list[str] = Field(default_factory=list)
    api_key_count: int = 0  # api_keys 원본은 비노출
    adapter_name_prefix: str | None = None  # M7
    metadata: dict[str, Any] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    """헬스체크 응답."""

    status: str = "ok"
    version: str = "0.1.0"
    gpu_server: str = "unknown"


# ─────────────────────────────────────────────
# 앱 상태 (모듈 레벨)
# ─────────────────────────────────────────────
_app_state: dict[str, Any] = {
    "state": None,  # GlobalState
    "config": None,  # NexusConfig
    "logging_middleware": None,  # RequestLoggingMiddleware 인스턴스
    "query_engine": None,  # QueryEngine (Phase 2에서 초기화)
    "tool_registry": None,  # ToolRegistry (Phase 2에서 초기화)
}


# ─────────────────────────────────────────────
# Lifespan (앱 시작/종료 이벤트)
# ─────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    앱 시작 시 Phase 1 부트스트랩을 수행하고,
    종료 시 리소스를 정리한다.
    """
    # 시작: Phase 1 + Phase 2 부트스트랩
    try:
        from core.bootstrap import init, init_phase2

        # Phase 1: 환경 비의존 초기화
        state = await init()
        _app_state["state"] = state
        _app_state["config"] = state.config

        # Phase 2: ToolRegistry, MemoryManager, QueryEngine 초기화
        components = await init_phase2(state)
        _app_state["tool_registry"] = components.get("tool_registry")
        _app_state["model_provider"] = components.get("model_provider")
        _app_state["memory_manager"] = components.get("memory_manager")  # Ch 16
        _app_state["tenant_registry"] = state.config.tenants  # M2 — 헤더 해석용
        # v0.14.8: 임베딩 keepalive task — 종료 시 cancel하기 위해 보관
        _app_state["embedding_keepalive_task"] = components.get("embedding_keepalive_task")

        # 웹 전용 QueryEngine — 도구 8개로 축소 (토큰 예산 관리)
        # RTX 5090 (8192 ctx)에서 도구 24개(~6,102토큰)는 컨텍스트 초과.
        # 핵심 도구 8개(~1,851토큰)만 사용하여 입력+출력 공간 확보.
        # (2026-04-21 리팩토링 3: 인라인 조립 로직을 _build_web_query_engine으로 분리)
        # (2026-07-03 동시성 수정 #5: 공유 부품(parts)을 한 번만 만들어 저장하고,
        #  실제 요청은 세션별 격리 엔진을 조립해 쓴다. web_engine은 하위 호환/메타데이터
        #  용 템플릿으로만 남긴다 — 채팅 핸들러는 이걸 공유 mutable 상태로 쓰지 않는다.)
        web_parts = _build_web_engine_parts(components, state)
        web_engine, web_dispatcher = _assemble_session_engine(
            web_parts, state.session_id or "", tenant=None
        )
        web_tools = web_parts["web_tools"]
        # 세션별 엔진 팩토리가 참조할 공유 부품. 이 키가 존재하면 채팅 핸들러는
        # 요청마다 격리된 QueryEngine을 새로 조립한다(_acquire_session_engine).
        _app_state["web_engine_parts"] = web_parts
        _app_state["model_dispatcher"] = web_dispatcher
        _app_state["query_engine"] = web_engine
        # /v1/tools가 모델이 실제로 보는 웹 Worker 도구 풀(MCP 포함)을 노출하도록 저장.
        # 기존 _app_state["tool_registry"]는 부트스트랩의 23개 base 레지스트리라
        # 웹 Worker 실풀과 다르다(MCP 누락) — 웹 엔드포인트는 web_tools를 우선한다.
        _app_state["web_tools"] = web_tools
        logger.info(
            "웹 서버 부트스트랩 완료 (Phase 1 + 2, 웹 도구 %d개)",
            len(web_tools),
        )
    except Exception as e:
        logger.warning(f"부트스트랩 실패, 기본 설정으로 시작: {e}")

    yield

    # 종료: 리소스 정리
    # v0.14.8 — 임베딩 keepalive task를 깨끗이 취소
    keepalive = _app_state.get("embedding_keepalive_task")
    if keepalive is not None and not keepalive.done():
        keepalive.cancel()
        try:
            await keepalive
        except (asyncio.CancelledError, Exception):
            # CancelledError 또는 task 내부 예외 모두 swallow — 종료 경로
            pass
    if _app_state["state"]:
        summary = _app_state["state"].get_session_summary()
        logger.info(f"웹 서버 종료. 세션 요약: {summary}")


# ─────────────────────────────────────────────
# FastAPI 앱 생성
# ─────────────────────────────────────────────
app = FastAPI(
    title="Project Nexus",
    description="에어갭 로컬 LLM 오케스트레이션 플랫폼 API",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS 미들웨어 적용 — 로컬/LAN만 허용
app.add_middleware(CORSMiddleware, **CORSConfig.get_cors_kwargs())


# API 키 인증 미들웨어 (Security Critical #4) — CORS 뒤에 등록한다.
# 설정·테넌트 레지스트리는 lifespan 기동 후에야 _app_state에 채워지므로,
# 미들웨어가 dispatch 시점에 지연 조회할 수 있도록 무인자 함수로 넘긴다
# (생성 시점 조회 금지 — 순환 import 및 기동 순서 문제 방지).
def _get_web_auth_config() -> Any:
    """현재 로드된 WebAuthConfig를 반환한다(없으면 None → 인증 비활성 취급)."""
    cfg = _app_state.get("config")
    return getattr(cfg, "web_auth", None) if cfg is not None else None


def _get_tenant_registry() -> Any:
    """현재 TenantRegistry를 반환한다(없으면 None → fail-closed 차단)."""
    return _app_state.get("tenant_registry")


app.add_middleware(
    ApiKeyAuthMiddleware,
    get_auth_config=_get_web_auth_config,
    get_tenant_registry=_get_tenant_registry,
)

# 요청 로깅 미들웨어 적용
_logging_middleware = RequestLoggingMiddleware(app)
_app_state["logging_middleware"] = _logging_middleware

# 정적 파일 서빙 — 채팅 UI (HTML/CSS/JS)
_static_dir = Path(__file__).parent / "static"
if _static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")


# ─────────────────────────────────────────────
# 채팅 엔드포인트
# ─────────────────────────────────────────────
def _restore_messages_from_saved(saved: list[dict] | None, session_id: str) -> list:
    """Redis에서 가져온 직렬화 메시지(dict 목록)를 Message 객체 리스트로 복원한다.

    비스트리밍/스트리밍 두 핸들러가 동일하게 쓰던 복원 로직을 한 곳으로 모은다.

    왜 항목 단위로 예외를 격리하는가:
      과거 버전이 assistant content를 ContentBlock 리스트 형식으로 저장한
      "오염 데이터"가 Redis에 남아 있을 수 있다. 예전 코드는 복원 루프 전체를
      하나의 try/except로 감싸서, 한 항목(리스트 content)이 깨지면 그 뒤 항목까지
      전부 복원이 중단됐다(이력 유실의 직접 원인).
      따라서 항목마다 예외를 격리하고, content가 리스트면 텍스트만 추출해
      평문으로 되돌려 견고하게 복원한다.
    """
    from core.message import Message as _Msg

    restored: list[_Msg] = []
    for item in saved or []:
        try:
            role = item.get("role")
            content = item.get("content", "")
            # 과거 오염 데이터 호환: content가 ContentBlock 리스트면 텍스트만 추출한다.
            if isinstance(content, list):
                content = "".join(
                    block.get("text", "")
                    for block in content
                    if isinstance(block, dict) and block.get("type") == "text"
                )
            if not content:
                continue
            if role == "user":
                restored.append(_Msg.user(content))
            elif role == "assistant":
                restored.append(_Msg.assistant(content))
        except Exception as e:
            # 한 항목이 깨져도 나머지는 복원되도록 건너뛴다(부분 복원 보장).
            logger.warning("세션 복원 항목 건너뜀 (%s): %s", session_id, e)
            continue
    return restored


# ─────────────────────────────────────────────
# 세션별 엔진 격리 + 세션 락 (동시성 결함 수정 — 감사 Critical #5, 2026-07-03)
# ─────────────────────────────────────────────
# 기존: QueryEngine 싱글톤 1개를 모든 요청이 공유 → 동시 요청이 _messages/tenant를
# 락 없이 뒤섞음(멀티테넌트 최대 블로커). 수정: (1) 요청/세션별로 격리된 QueryEngine을
# 조립하고(_acquire_session_engine), (2) '같은 세션'의 동시 요청만 세션별 asyncio.Lock
# 으로 직렬화한다. 서로 다른 세션은 병렬을 유지하므로 멀티테넌트 처리량이 죽지 않는다.

# 세션 락 상한 — 무한 증가를 막는 바운드 LRU. 도달 시 '사용 중이 아닌' 가장 오래된
# 락부터 축출한다(사용 중 락은 절대 축출하지 않음).
_SESSION_LOCK_MAX = 4096


def _get_session_lock(session_id: str) -> asyncio.Lock:
    """세션 ID별 asyncio.Lock을 얻는다(없으면 생성). 바운드 LRU로 개수를 제한한다.

    반드시 실행 중인 이벤트 루프 안(async 핸들러)에서 호출한다.

    루프 인지(loop-aware)로 저장하는 이유: asyncio.Lock은 최초 acquire 시 특정
    이벤트 루프에 바인딩된다. 그런데 _app_state는 모듈 싱글톤이라 프로세스 수명 동안
    유지되고, pytest는 테스트마다 새 이벤트 루프를 쓴다(asyncio_default_fixture_
    loop_scope=function). 만약 락을 루프와 무관하게 캐시하면, 이전 테스트 루프에
    바인딩된 락을 다음 테스트 루프에서 acquire하다 'attached to a different loop'
    오류가 난다. 그래서 현재 실행 루프가 바뀌면 락 저장소를 새로 만든다. 프로덕션은
    단일 장수명 루프라 저장소가 유지되어 LRU가 정상 동작한다.

    동기 함수인 이유: dict 접근/삽입 사이에 await가 없어 단일 이벤트 루프에서 원자적
    이다(별도 async 가드 락이 필요 없다).
    """
    loop = asyncio.get_running_loop()
    store = _app_state.get("session_locks_store")
    if store is None or store[0] is not loop:
        # 최초 호출이거나 이벤트 루프가 교체됨(주로 테스트) → 저장소를 새로 시작.
        store = (loop, OrderedDict())
        _app_state["session_locks_store"] = store
    locks: OrderedDict[str, asyncio.Lock] = store[1]
    lock = locks.get(session_id)
    if lock is None:
        lock = asyncio.Lock()
        locks[session_id] = lock
    else:
        locks.move_to_end(session_id)  # 최근 사용 표시(LRU)
    # 바운드 축출 — 가장 오래된 것부터. 단, 현재 잠겨 있는(사용 중) 락이나 방금
    # 만든 이 세션의 락은 건드리지 않는다(사용 중 락 축출 시 직렬화 무력화).
    while len(locks) > _SESSION_LOCK_MAX:
        old_sid, old_lock = next(iter(locks.items()))
        if old_lock.locked() or old_sid == session_id:
            break
        locks.popitem(last=False)
    return lock


def _acquire_session_engine(session_id: str, tenant: Any) -> Any:
    """요청/세션별로 격리된 QueryEngine을 반환한다.

    - 프로덕션(부트스트랩 성공 → web_engine_parts 존재): 세션 전용 엔진을 새로
      조립한다. 공유 mutable 상태(_messages/_session_id/tenant)를 원천 제거한다.
    - 부트스트랩 미완/테스트(parts 없음): 기존 _app_state['query_engine'] 싱글톤을
      그대로 반환한다(무회귀 — 기존 웹 테스트가 주입한 fake 엔진/placeholder 경로 유지).
    """
    parts = _app_state.get("web_engine_parts")
    if parts is not None:
        engine, _dispatcher = _assemble_session_engine(parts, session_id, tenant)
        return engine
    return _app_state.get("query_engine")


@app.post("/v1/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    x_tenant_id: str | None = Header(default=None, alias="X-Tenant-ID"),
    authorization: str | None = Header(default=None),
) -> ChatResponse:
    """
    비스트리밍 채팅.

    사용자 메시지를 QueryEngine에 전달하고,
    모든 StreamEvent를 수집하여 최종 응답으로 반환한다.
    """
    # 세션 ID 생성 또는 재사용
    session_id = request.session_id or str(uuid.uuid4())
    tenant = _resolve_tenant(request.tenant_id, x_tenant_id, authorization)

    # 동시성 수정(#5): 요청/세션별 격리 엔진을 얻는다(프로덕션). parts가 없으면
    # 기존 싱글톤/placeholder 경로로 폴백(무회귀).
    engine = _acquire_session_engine(session_id, tenant)
    if engine is None:
        # QueryEngine이 초기화되지 않은 경우 placeholder 응답
        return ChatResponse(
            session_id=session_id,
            response="QueryEngine이 아직 초기화되지 않았습니다.",
            tool_calls=[],
            usage=UsageInfo(),
        )

    from core.message import StreamEvent, StreamEventType

    response_text_parts: list[str] = []
    usage = UsageInfo()
    # ─── 도구 호출 수집 (tool_calls 응답 필드 보강) ─────────────────
    # 4-Tier 체인을 우회하지 않고 submit_message가 yield하는 StreamEvent만 소비하여
    # 누적한다. TOOL_USE_STOP에서 input이 완성되므로 STOP 기준 등록, TOOL_RESULT로
    # 같은 tool_use_id를 매칭해 요약/에러를 채운다(한 호출당 ToolCallInfo 하나).
    tool_calls_by_id: dict[str, ToolCallInfo] = {}
    tool_calls_order: list[str] = []
    # 결과 본문이 과도하게 길면 응답이 비대해지므로 요약 길이를 제한한다(과설계 금지).
    result_summary_max = 500
    response_session_id = session_id

    # 같은 세션의 동시 요청만 직렬화(다른 세션은 병렬 유지). 프로덕션에서는 engine
    # 자체가 세션 전용이라 _messages는 이미 격리되지만, 공유 히스토리(Redis)·트랜스
    # 크립트 기록 순서를 안정화하기 위해 세션 단위로 감싼다(전역 처리량은 안 죽음).
    session_lock = _get_session_lock(session_id)
    async with session_lock:
        # Ch 16 + 리팩토링 2: 세션/tenant/transcript를 공식 bind_request로 한 번에 주입
        transcript = _build_transcript(session_id)
        engine.bind_request(
            session_id=session_id,
            tenant=tenant,
            transcript=transcript,
        )
        if tenant is not None:
            logger.info(
                "tenant 해석: %s (sources=%s)",
                tenant.id,
                tenant.allowed_knowledge_sources,
            )

        # Ch 16: Redis에서 해당 세션의 이전 히스토리 복원
        memory_manager = _app_state.get("memory_manager")
        if memory_manager is not None:
            try:
                engine.clear_messages()
                saved = await memory_manager.short_term.get_conversation_context(session_id)
                engine._messages.extend(_restore_messages_from_saved(saved, session_id))
            except Exception as e:
                logger.warning("비스트리밍 세션 복원 실패 (%s): %s", session_id, e)

        async for event in engine.submit_message(request.message):
            if not isinstance(event, StreamEvent):
                continue

            if event.type == StreamEventType.TEXT_DELTA and event.text:
                response_text_parts.append(event.text)

            elif event.type == StreamEventType.USAGE_UPDATE and event.usage:
                usage = UsageInfo(
                    input_tokens=event.usage.input_tokens,
                    output_tokens=event.usage.output_tokens,
                    total_tokens=event.usage.total_tokens,
                )

            elif event.type == StreamEventType.TOOL_USE_STOP and event.tool_use:
                # 도구 호출 확정 — 이름/입력/tool_use_id를 등록한다.
                tu = event.tool_use
                tu_id = tu.id
                if tu_id not in tool_calls_by_id:
                    tool_calls_by_id[tu_id] = ToolCallInfo(
                        name=tu.name,
                        input=tu.input,
                    )
                    tool_calls_order.append(tu_id)

            elif event.type == StreamEventType.TOOL_RESULT and event.tool_result:
                # 도구 결과 — 같은 tool_use_id의 호출 정보에 요약/에러 여부를 채운다.
                tr = event.tool_result
                info = tool_calls_by_id.get(tr.tool_use_id)
                if info is None:
                    # STOP 이벤트를 못 본 경우(방어적): 결과만으로 항목을 만든다.
                    info = ToolCallInfo(name="", input={})
                    tool_calls_by_id[tr.tool_use_id] = info
                    tool_calls_order.append(tr.tool_use_id)
                summary = tr.content or ""
                if len(summary) > result_summary_max:
                    summary = summary[:result_summary_max] + "…(truncated)"
                info.result = summary
                info.is_error = tr.is_error

        # 응답에 실을 세션 ID는 엔진이 확정한 값을 쓴다(fake 엔진 테스트 호환).
        response_session_id = engine.session_id

    # 등장 순서대로 ToolCallInfo 목록을 만든다.
    tool_calls_info: list[ToolCallInfo] = [tool_calls_by_id[tid] for tid in tool_calls_order]

    return ChatResponse(
        session_id=response_session_id,
        response="".join(response_text_parts),
        tool_calls=tool_calls_info,
        usage=usage,
    )


@app.post("/v1/chat/stream")
async def chat_stream(
    request: ChatRequest,
    x_tenant_id: str | None = Header(default=None, alias="X-Tenant-ID"),
    authorization: str | None = Header(default=None),
) -> StreamingResponse:
    """
    SSE 스트리밍 채팅.

    QueryEngine의 AsyncGenerator에서 yield되는 StreamEvent를
    Server-Sent Events 형식으로 실시간 전송한다.
    """
    session_id = request.session_id or str(uuid.uuid4())
    tenant = _resolve_tenant(request.tenant_id, x_tenant_id, authorization)

    async def _locked_generate() -> AsyncGenerator[str, None]:
        """세션별 격리 엔진 획득 + 세션 락으로 감싼 뒤 실제 스트림을 위임한다.

        동시성 수정(#5): (1) 요청/세션별 격리 QueryEngine을 얻고(_acquire_session_engine),
        (2) 같은 세션의 동시 요청만 세션 락으로 직렬화한다(다른 세션은 병렬 유지 →
        멀티테넌트 처리량 보존). 락은 스트림 전체 수명 동안 유지되며, 클라이언트 연결
        종료로 제너레이터가 닫혀도 async with가 반드시 해제한다.
        """
        engine = _acquire_session_engine(session_id, tenant)
        if engine is None:
            placeholder = {
                "type": "text_delta",
                "text": "QueryEngine이 아직 초기화되지 않았습니다.",
                "session_id": session_id,
            }
            yield f"data: {json.dumps(placeholder, ensure_ascii=False)}\n\n"
            return
        session_lock = _get_session_lock(session_id)
        async with session_lock:
            async for _frame in generate(engine):
                yield _frame

    async def generate(engine: Any) -> AsyncGenerator[str, None]:
        """
        SSE 이벤트를 생성하는 AsyncGenerator.

        QueryEngine의 StreamEvent를 수신하여
        'data: {json}\n\n' 형식으로 실시간 전송한다. engine은 요청/세션별로 격리된
        인스턴스다(호출부 _locked_generate가 세션 락을 잡은 채 소비한다).
        """
        from core.message import StreamEvent

        # 세션별 대화 히스토리 관리 — 요청마다 엔진 messages를 해당 세션 이력으로
        # 복원한다. histories 자체는 세션 키로 분리돼 있고, 같은 세션의 동시 접근은
        # 상위 _locked_generate의 세션 락으로 직렬화된다.
        histories = _app_state.setdefault("chat_histories", {})

        # Ch 16: 세션 영속화 — 메모리 매니저가 있으면 Redis에서 복원
        # 첫 요청(인메모리 비어 있음)일 때만 Redis에서 이전 히스토리를 가져온다.
        # 이후 턴은 인메모리 + Redis 양쪽을 유지한다(write-through).
        memory_manager = _app_state.get("memory_manager")
        if session_id not in histories:
            histories[session_id] = []
            if memory_manager is not None:
                try:
                    saved = await memory_manager.short_term.get_conversation_context(session_id)
                    restored = _restore_messages_from_saved(saved, session_id)
                    histories[session_id].extend(restored)
                    if restored:
                        logger.info(
                            "세션 %s Redis 복원: %d개 메시지",
                            session_id,
                            len(restored),
                        )
                except Exception as e:
                    logger.warning("세션 Redis 복원 실패 (%s): %s", session_id, e)

        # Qwen3.5 thinking 찌꺼기가 들어있는 과거 메시지를 1회성 정제
        # (세션이 enable_thinking=false 전의 오염된 상태일 수 있다)
        _sanitize_history_inplace(histories[session_id])

        # Ch 16: 세션별 JSONL 트랜스크립트 주입 (요청/세션별 격리 엔진에 동적 세팅)
        try:
            from core.memory.transcript import SessionTranscript as _Trans

            cfg = _app_state.get("config")
            sessions_dir = cfg.session.sessions_dir if cfg else ".nexus/sessions"
            transcript_enabled = cfg.session.transcript_enabled if cfg else True
            engine._transcript = _Trans(
                sessions_dir=sessions_dir,
                session_id=session_id,
                enabled=transcript_enabled,
            )
        except Exception as e:
            logger.warning("트랜스크립트 주입 실패 (%s): %s", session_id, e)
            engine._transcript = None

        # 리팩토링 2: 세션/tenant/transcript를 공식 bind_request로 주입
        # (이전엔 engine._session_id 등 비공개 필드를 직접 치환 — race condition 위험)
        transcript = _build_transcript(session_id)
        engine.bind_request(
            session_id=session_id,
            tenant=tenant,
            transcript=transcript,
        )
        if tenant is not None:
            logger.info("tenant 해석: %s (sources=%s)", tenant.id, tenant.allowed_knowledge_sources)

        # QueryEngine의 messages를 해당 세션의 히스토리로 교체
        # 도구 호출/결과 메시지는 토큰을 많이 차지하므로 제외하고,
        # user/assistant 텍스트 메시지만 예산 내에서 복원한다.
        engine.clear_messages()

        # user/assistant 텍스트 메시지만 필터링

        text_messages = []
        for msg in histories[session_id]:
            role = msg.role if isinstance(msg.role, str) else msg.role.value
            # tool_result, tool_use 메시지는 건너뛰고 user/assistant만
            if role in ("user", "assistant"):
                # 도구 호출이 포함된 assistant 메시지도 텍스트만 추출
                text = msg.text_content if hasattr(msg, "text_content") else str(msg.content)
                if text and len(text) > 5:  # 빈 메시지 제외
                    text_messages.append(msg)

        # 예산 내에서 최근 메시지만 복원 (2,000 토큰 = ~6,000자)
        budget = 6000
        used = len(request.message)
        restored = []
        for msg in reversed(text_messages):
            content = msg.text_content if hasattr(msg, "text_content") else str(msg.content)
            if used + len(content) > budget:
                break
            restored.append(msg)
            used += len(content)
        restored.reverse()
        for msg in restored:
            engine._messages.append(msg)

        # ─── 요청 단위 타이밍/관측 로그 ───────────────────
        # 첨부 파일 경로가 메시지에 포함되면 업로드 케이스로 표시
        has_attach = "서버 경로:" in request.message or "[첨부파일:" in request.message
        req_start_mono = time.monotonic()
        event_count = 0
        stream_abort_error: BaseException | None = None

        logger.info(
            "SSE 시작: session=%s, message_len=%d, has_attach=%s",
            session_id,
            len(request.message),
            has_attach,
        )

        # ─── Heartbeat/Producer 분리 구조 ──────────────────
        # submit_message의 이벤트 yield 사이에 긴 공백(Scout 호출 등)이 있으면
        # 브라우저/프록시가 연결을 끊거나 사용자가 "무한 로딩"으로 느낀다.
        # 이벤트는 Queue로 수거하고, 메인 루프는 get에 타임아웃을 걸어 일정 주기
        # 마다 SSE 주석(`: ping`) 프레임을 전송한다. SSE 주석은 EventSource 클라이언트
        # 에서 무시되므로 기존 JS 파서에 영향을 주지 않는다.
        sse_sentinel: tuple[str, Any] = ("done", None)
        sse_heartbeat_seconds = 20.0  # 20s마다 keep-alive

        event_queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()

        async def _producer() -> None:
            """submit_message 스트림을 큐로 옮긴다 (에러까지 포함)."""
            try:
                async for ev in engine.submit_message(request.message):
                    await event_queue.put(("event", ev))
            except BaseException as e:  # noqa: BLE001 — 모든 예외를 에러 프레임으로
                await event_queue.put(("error", e))
            finally:
                await event_queue.put(sse_sentinel)

        producer_task = asyncio.create_task(_producer())
        try:
            while True:
                try:
                    kind, payload = await asyncio.wait_for(
                        event_queue.get(), timeout=sse_heartbeat_seconds
                    )
                except TimeoutError:
                    # 이벤트 공백 → heartbeat. SSE 주석은 data 프레임이 아니므로
                    # 클라이언트 JSON 파서가 건드리지 않는다.
                    # (Python 3.11+에서 asyncio.TimeoutError는 builtin TimeoutError의 별칭)
                    elapsed = time.monotonic() - req_start_mono
                    yield f": heartbeat {elapsed:.0f}s\n\n"
                    continue

                if kind == "done":
                    break
                if kind == "error":
                    stream_abort_error = payload
                    # 에러를 클라이언트가 이해할 수 있는 형태로 변환
                    err_frame = {
                        "type": "error",
                        "session_id": session_id,
                        "error_code": "stream_aborted",
                        "message": (f"{type(payload).__name__}: {payload}"),
                    }
                    yield ("data: " + json.dumps(err_frame, ensure_ascii=False) + "\n\n")
                    break

                event = payload
                if isinstance(event, StreamEvent):
                    sse_data: dict[str, Any] = {
                        "type": event.type if isinstance(event.type, str) else event.type.value,
                        "session_id": engine.session_id,
                    }
                    if event.text:
                        sse_data["text"] = event.text
                    if event.message:
                        sse_data["message"] = event.message
                    if event.error_code:
                        sse_data["error_code"] = event.error_code
                    if event.usage:
                        sse_data["usage"] = {
                            "input_tokens": event.usage.input_tokens,
                            "output_tokens": event.usage.output_tokens,
                        }
                    if event.stop_reason:
                        stop_val = event.stop_reason
                        sse_data["stop_reason"] = (
                            stop_val if isinstance(stop_val, str) else stop_val.value
                        )
                    yield f"data: {json.dumps(sse_data, ensure_ascii=False)}\n\n"
                    event_count += 1
        finally:
            # producer가 아직 살아 있으면 취소 (클라이언트가 연결을 끊은 경우 등)
            if not producer_task.done():
                producer_task.cancel()
                try:
                    await producer_task
                except (asyncio.CancelledError, Exception):  # noqa: BLE001, S110
                    # producer 취소 시 예외는 이미 stream_abort_error 경로에서 처리됨
                    pass
            elapsed_total = time.monotonic() - req_start_mono
            if stream_abort_error is not None:
                logger.warning(
                    "SSE 중단: session=%s, elapsed=%.1fs, events=%d, error=%s",
                    session_id,
                    elapsed_total,
                    event_count,
                    type(stream_abort_error).__name__,
                )
            else:
                logger.info(
                    "SSE 완료: session=%s, elapsed=%.1fs, events=%d",
                    session_id,
                    elapsed_total,
                    event_count,
                )

        # 이번 턴의 user/assistant 텍스트 메시지만 히스토리에 저장
        # tool_result/tool_use 메시지는 토큰이 크므로 저장하지 않는다
        # Qwen3.5의 <think>…</think> 블록이 혹여 섞여 들어오면 다음 턴의
        # in-context 모방을 유발하므로 저장 전에 제거한다 (safeguard).
        for msg in engine._messages:
            role = msg.role if isinstance(msg.role, str) else msg.role.value
            if role in ("user", "assistant"):
                content = msg.text_content if hasattr(msg, "text_content") else str(msg.content)
                if content and len(content) > 5 and msg not in histories[session_id]:
                    content_clean = _strip_thinking(content)
                    if content_clean and content_clean != content:
                        # content가 정제됐다면 원본 Message는 그대로 두되 저장용
                        # 얕은 복사본을 만들어 히스토리에 넣는다. 원본 Message는
                        # Pydantic frozen이므로 text를 바꿀 수 없다 → 새 Message 생성.
                        from core.message import Message

                        new_msg = (
                            Message.assistant(content_clean)
                            if role == "assistant"
                            else Message.user(content_clean)
                        )
                        histories[session_id].append(new_msg)
                    elif content_clean:
                        histories[session_id].append(msg)

        # Ch 16: Redis에 write-through — 인메모리 히스토리를 JSON 직렬화하여 저장
        # QueryEngine.on_turn_end도 자체적으로 save_conversation_context를 호출하지만,
        # 웹의 경우 histories[session_id]가 "여러 턴 누적된 완전한 대화"이므로
        # 여기서도 한 번 더 저장하여 서버 재기동 시 UI에 보이는 대화 그대로 복원.
        if memory_manager is not None:
            try:
                serialized: list[dict[str, Any]] = []
                for m in histories[session_id]:
                    role = m.role if isinstance(m.role, str) else m.role.value
                    if role not in ("user", "assistant"):
                        continue
                    text = m.text_content if hasattr(m, "text_content") else str(m.content)
                    if text:
                        serialized.append({"role": role, "content": text})
                await memory_manager.short_term.save_conversation_context(
                    session_id, serialized, ttl=86400
                )
            except Exception as e:
                logger.warning("세션 Redis 저장 실패 (%s): %s", session_id, e)

    return StreamingResponse(
        _locked_generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Session-ID": session_id,
        },
    )


# ─────────────────────────────────────────────
# 세션 엔드포인트
# ─────────────────────────────────────────────
@app.get("/v1/sessions")
async def list_sessions() -> dict[str, Any]:
    """
    저장된 세션 목록을 반환한다 (Ch 16).

    두 소스를 병합:
      1. Redis (단기, TTL 24h) — 최근 활성 세션 (session_id만)
      2. JSONL 트랜스크립트 (영구 기록) — 파일 시스템에 남아있는 모든 세션
    트랜스크립트가 상세 메타데이터(라인 수, 최종 수정 시각)를 갖고 있으므로
    이를 기본으로 삼고, Redis-only 세션은 이후에 머지한다.
    """
    from core.memory.transcript import list_transcript_sessions

    cfg = _app_state.get("config")
    sessions_dir = cfg.session.sessions_dir if cfg else ".nexus/sessions"

    # 1) 파일 트랜스크립트 기반 세션 (상세 정보 포함)
    disk_sessions = list_transcript_sessions(sessions_dir, limit=100)

    # 2) Redis 단기 캐시 기반 세션 (session_id만) — 트랜스크립트에 없는 것만 추가
    memory_manager = _app_state.get("memory_manager")
    known_ids = {s["session_id"] for s in disk_sessions}
    redis_only: list[dict[str, Any]] = []
    if memory_manager is not None:
        try:
            for sid in await memory_manager.short_term.list_sessions(limit=100):
                if sid not in known_ids:
                    redis_only.append(
                        {
                            "session_id": sid,
                            "source": "redis_only",
                            "last_modified": None,
                            "entries": None,
                        }
                    )
        except Exception as e:
            # Redis 조회 실패는 치명적이지 않음 — disk 결과만으로 응답
            logger.debug("list_sessions Redis 조회 실패 (무시): %s", e)

    return {
        "sessions": disk_sessions + redis_only,
        "total": len(disk_sessions) + len(redis_only),
        "sessions_dir": sessions_dir,
    }


@app.get("/v1/sessions/{session_id}/messages")
async def get_session_messages(session_id: str) -> dict[str, Any]:
    """
    특정 세션의 대화 히스토리를 반환한다 (Ch 16 프론트 복원용).

    조회 우선순위:
      1) Redis 단기 캐시 (TTL 24h) — 가장 최신, JSON 직렬화된 user/assistant 페어
      2) JSONL 트랜스크립트 (영구 기록) — Redis 만료/없음 시 폴백

    어느 쪽에도 기록이 없으면 404 — UI가 "세션 없음" 분기로 전환할 수 있도록.

    응답:
      {
        "session_id": "...",
        "source": "redis" | "transcript",
        "messages": [{"role": "user"|"assistant", "content": "...",
                      "turn": N|None, "ts": ISO-8601|None}, ...],
        "total": N,
      }

    경로 파라미터 검증: session_id에 경로 분리자(슬래시/백슬래시/..)가 들어오면
    거부 — 트랜스크립트 파일 시스템 접근 시 디렉토리 탈출을 막는다.
    """
    from fastapi import HTTPException

    # 입력 검증 — 경로 탈출 차단 (파일 시스템 폴백 경로에서만 의미가 있지만
    # Redis 키 오염 방지 차원에서도 동일하게 적용)
    if not session_id or any(ch in session_id for ch in ("/", "\\", "..", "\x00")):
        raise HTTPException(status_code=400, detail="invalid session_id")

    # 1) Redis 우선 (가장 최신 상태)
    memory_manager = _app_state.get("memory_manager")
    if memory_manager is not None:
        try:
            redis_msgs = await memory_manager.short_term.get_conversation_context(session_id)
            if redis_msgs:
                normalized: list[dict[str, Any]] = []
                for m in redis_msgs:
                    role = m.get("role")
                    content = m.get("content")
                    if role in ("user", "assistant") and content:
                        normalized.append(
                            {
                                "role": role,
                                "content": content,
                                "turn": m.get("turn"),
                                "ts": m.get("ts"),
                            }
                        )
                if normalized:
                    return {
                        "session_id": session_id,
                        "source": "redis",
                        "messages": normalized,
                        "total": len(normalized),
                    }
        except Exception as e:
            # Redis 장애는 치명적 아님 — 트랜스크립트 폴백 시도
            logger.debug("get_session_messages Redis 조회 실패 (%s): %s", session_id, e)

    # 2) JSONL 트랜스크립트 폴백
    from core.memory.transcript import read_transcript_messages

    cfg = _app_state.get("config")
    sessions_dir = cfg.session.sessions_dir if cfg else ".nexus/sessions"
    disk_msgs = read_transcript_messages(sessions_dir, session_id)
    if disk_msgs:
        # ts/turn 포함 그대로 반환 (헬퍼가 이미 user/assistant만 필터)
        return {
            "session_id": session_id,
            "source": "transcript",
            "messages": [
                {
                    "role": m["role"],
                    "content": m["content"],
                    "turn": m.get("turn"),
                    "ts": m.get("ts"),
                }
                for m in disk_msgs
            ],
            "total": len(disk_msgs),
        }

    # 어느 쪽에도 없음
    raise HTTPException(status_code=404, detail=f"session not found: {session_id}")


@app.delete("/v1/sessions/{session_id}")
async def delete_session(session_id: str) -> dict[str, Any]:
    """
    특정 세션을 Redis(단기) + 트랜스크립트(영구) 양쪽에서 삭제한다 (Ch 16).

    프론트 사이드바의 세션 삭제 버튼이 호출하는 엔드포인트. 각 저장소는
    독립적이므로 한쪽만 성공해도 응답한다 (best-effort).

    응답:
      {
        "session_id": "...",
        "deleted_redis": bool,     # Redis 키가 실제로 있었고 삭제됐는지
        "deleted_disk":  bool,     # 트랜스크립트 디렉토리가 있었고 삭제됐는지
      }

    둘 다 False여도 200 — 이미 없었을 뿐 에러는 아님. 다만 session_id 자체가
    부적합(슬래시/백슬래시/'..')하면 400.
    """
    from fastapi import HTTPException

    if not session_id or any(ch in session_id for ch in ("/", "\\", "..", "\x00")):
        raise HTTPException(status_code=400, detail="invalid session_id")

    deleted_redis = False
    deleted_disk = False

    # 1) Redis — clear_session은 존재 여부와 무관하게 DEL을 호출하므로,
    #    실제 삭제 여부는 사전 존재 조회로 판정한다.
    memory_manager = _app_state.get("memory_manager")
    if memory_manager is not None:
        try:
            existing = await memory_manager.short_term.get_conversation_context(session_id)
            if existing:
                await memory_manager.short_term.clear_session(session_id)
                deleted_redis = True
        except Exception as e:
            # Redis 장애는 치명적 아님 — 디스크 삭제는 독립적으로 시도
            logger.warning("Redis 세션 삭제 실패 (%s): %s", session_id, e)

    # 2) 디스크 — 트랜스크립트 디렉토리 통째로 제거
    from core.memory.transcript import delete_transcript_session

    cfg = _app_state.get("config")
    sessions_dir = cfg.session.sessions_dir if cfg else ".nexus/sessions"
    try:
        deleted_disk = delete_transcript_session(sessions_dir, session_id)
    except ValueError:
        # delete_transcript_session의 경로 검증 실패 — 이미 위에서 400 처리했지만
        # 방어적으로 한 번 더
        raise HTTPException(status_code=400, detail="invalid session_id") from None
    except OSError as e:
        logger.warning("트랜스크립트 삭제 실패 (%s): %s", session_id, e)

    logger.info(
        "세션 삭제: session=%s, redis=%s, disk=%s",
        session_id,
        deleted_redis,
        deleted_disk,
    )
    return {
        "session_id": session_id,
        "deleted_redis": deleted_redis,
        "deleted_disk": deleted_disk,
    }


# ─────────────────────────────────────────────
# 도구 엔드포인트
# ─────────────────────────────────────────────
@app.get("/v1/tools")
async def list_tools() -> dict[str, Any]:
    """
    등록된 도구 목록을 반환한다.

    ToolRegistry에서 등록된 모든 도구의 이름, 설명, 그룹을 반환한다.
    """
    # 웹 Worker가 실제로 보는 도구 풀을 우선 노출한다(MCP 도구 포함).
    # _app_state["web_tools"]는 _build_web_query_engine이 MCP 머지 후 저장한 실풀.
    # 부재 시(부트스트랩 실패 등) base tool_registry로 폴백.
    tools = _app_state.get("web_tools")
    if not tools:
        registry = _app_state.get("tool_registry")
        if registry is None:
            return {"tools": [], "total": 0}
        tools = registry.get_all_tools()

    tool_list = [
        ToolInfo(
            name=t.name,
            description=t.description,
            group=t.group,
            is_read_only=t.is_read_only,
        ).model_dump()
        for t in tools
    ]
    return {"tools": tool_list, "total": len(tool_list)}


# ─────────────────────────────────────────────
# 모델 엔드포인트
# ─────────────────────────────────────────────
@app.get("/v1/models")
async def list_models() -> dict[str, Any]:
    """
    사용 가능한 모델 목록을 반환한다.

    설정에서 정의된 모델 정보를 반환한다.
    """
    config = _app_state.get("config")
    if config:
        models = [
            ModelInfo(
                id=config.model.primary_model,
                name="Qwen 3.5 27B",
                role="primary",
            ),
            ModelInfo(
                id=config.model.auxiliary_model,
                name="ExaOne 7.8B",
                role="auxiliary",
            ),
            ModelInfo(
                id=config.model.embedding_model,
                name="Multilingual E5 Large",
                role="embedding",
            ),
        ]
        return {
            "models": [m.model_dump() for m in models],
            "total": len(models),
        }

    # 설정이 없으면 기본 모델 정보를 반환한다
    return {
        "models": [
            {"id": "qwen3.5-27b", "name": "Qwen 3.5 27B", "role": "primary"},
            {"id": "exaone-7.8b", "name": "ExaOne 7.8B", "role": "auxiliary"},
        ],
        "total": 2,
    }


# ─────────────────────────────────────────────
# 테넌트 목록 엔드포인트 (Part 5 Ch 15)
# ─────────────────────────────────────────────
@app.get("/v1/tenants")
async def list_tenants() -> dict[str, Any]:
    """등록된 테넌트 목록을 반환한다.

    보안:
      - `api_keys` 원본은 응답에 포함하지 않는다. 개수(`api_key_count`)만 노출.
      - 이 엔드포인트는 내부 LAN 관리 용도. 외부 노출 시에는 프록시 단에서
        인증을 걸어야 한다 (현재 /metrics와 동일한 정책).

    응답 형식:
        {
            "tenants": [TenantInfo, ...],
            "default_tenant": "default",
            "total": N,
        }
    """
    registry = _app_state.get("tenant_registry")
    if registry is None:
        # 레지스트리가 아직 초기화되지 않은 경우 — 빈 목록 반환 (500 대신)
        return {"tenants": [], "default_tenant": "default", "total": 0}

    tenants = [
        TenantInfo(
            id=t.id,
            name=t.name,
            description=t.description,
            model_override=t.model_override,
            allowed_knowledge_sources=list(t.allowed_knowledge_sources),
            api_key_count=len(t.api_keys),
            adapter_name_prefix=t.adapter_name_prefix,
            metadata=dict(t.metadata),
        )
        for t in registry.tenants
    ]
    return {
        "tenants": [t.model_dump() for t in tenants],
        "default_tenant": registry.default_tenant,
        "total": len(tenants),
    }


# ─────────────────────────────────────────────
# 헬스체크 엔드포인트
# ─────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    서버 상태를 확인한다.

    Nexus 오케스트레이터와 GPU 서버 양쪽의 상태를 반환한다.
    """
    config = _app_state.get("config")
    gpu_status = "unknown"

    if config:
        try:
            import httpx

            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{config.gpu_server_url}/health")
                gpu_status = "healthy" if resp.status_code == 200 else "unhealthy"
        except Exception:
            gpu_status = "unreachable"

    return HealthResponse(
        status="ok",
        version="0.1.0",
        gpu_server=gpu_status,
    )


# ─────────────────────────────────────────────
# 메트릭스 엔드포인트
# ─────────────────────────────────────────────
@app.get("/metrics")
async def metrics() -> dict[str, Any]:
    """
    서버 메트릭스를 반환한다.

    요청 로깅 미들웨어에서 수집한 메트릭스와
    GlobalState에서 추적하는 세션 메트릭스를 반환한다.
    """
    result: dict[str, Any] = {}

    # 요청 메트릭스
    middleware = _app_state.get("logging_middleware")
    if middleware and hasattr(middleware, "metrics"):
        result["http"] = middleware.metrics

    # 세션 메트릭스
    state = _app_state.get("state")
    if state:
        result["session"] = state.get_session_summary()

        # MCP 가시성 — Phase 2 부트스트랩이 GlobalState 에 채운 등록 결과를
        # 간단히 노출한다(연결 서버 수 + 서버별 도구 개수). 과설계 없이
        # "몇 개 서버가 살아 있고 각자 도구가 몇 개인가" 만 보여준다.
        mcp_servers = getattr(state, "mcp_servers", {}) or {}
        mcp_connected = getattr(state, "mcp_connected", set()) or set()
        result["mcp"] = {
            "connected_count": len(mcp_connected),
            "connected": sorted(mcp_connected),
            "tool_counts": {name: info.get("tool_count", 0) for name, info in mcp_servers.items()},
        }

    # 서브에이전트 메트릭스 — Ch 17 (v7.0 Phase 9 재설계)
    # AgentTool.get_stats()가 subagent_type별 호출 통계를 집계한다.
    # 예: {"scout": {"calls": 3, "total_latency_ms": 99000, "avg_latency_ms": 33000}}
    from core.tools.implementations.agent_tool import AgentTool

    result["agents"] = AgentTool.get_stats()
    # v0.14.2: Scout 결과 캐시 통계. Scout 반복 호출 회피 효과를 관측한다.
    result["agent_cache"] = AgentTool.get_cache_stats()

    # 하위 호환: 기존 대시보드가 result["scout"]을 참조할 수 있으므로 alias 유지.
    # Scout 자동 전처리가 제거됐으므로 Dispatcher.stats는 0만 반환하지만,
    # AgentTool의 "scout" 항목을 평탄화해서 함께 노출한다.
    dispatcher = _app_state.get("model_dispatcher")
    scout_agent_stats = result["agents"].get("scout", {})
    result["scout"] = {
        "tier": dispatcher.tier.value if dispatcher is not None else "unknown",
        "scout_enabled": (dispatcher.scout_enabled if dispatcher is not None else False),
        "scout_calls": scout_agent_stats.get("calls", 0),
        "scout_avg_latency_ms": scout_agent_stats.get("avg_latency_ms", 0.0),
        "scout_fallback_count": 0,  # fallback 개념은 AgentTool 이관 후 의미 없음
        "note": "scout_calls/avg_latency_ms are sourced from AgentTool.get_stats().",
    }

    # 멀티테넌시 — 등록된 테넌트 목록과 테넌트별 호출 통계 (Part 5 Ch 15)
    registry = _app_state.get("tenant_registry")
    tenant_stats = _app_state.get("tenant_stats") or {}
    if registry is not None:
        result["tenants"] = {
            "registered": [
                {
                    "id": t.id,
                    "name": t.name,
                    "has_model_override": bool(t.model_override),
                    "allowed_source_count": len(t.allowed_knowledge_sources),
                    "api_key_count": len(t.api_keys),
                }
                for t in registry.tenants
            ],
            "default_tenant": registry.default_tenant,
            "per_tenant_stats": tenant_stats,
        }

    return result


# ─────────────────────────────────────────────
# 파일 업로드 (문서 분석용)
# ─────────────────────────────────────────────
@app.post("/v1/upload")
async def upload_file(file: UploadFile) -> dict[str, Any]:
    """
    파일을 서버 임시 디렉토리에 저장하고 경로를 반환한다.
    반환된 경로를 DocumentProcess 도구로 분석할 수 있다.
    """
    import tempfile

    upload_dir = Path(tempfile.gettempdir()) / "nexus_uploads"
    upload_dir.mkdir(exist_ok=True)

    file_path = upload_dir / file.filename
    content = await file.read()
    file_path.write_bytes(content)

    return {
        "status": "ok",
        "file_path": str(file_path),
        "file_name": file.filename,
        "size_bytes": len(content),
    }


# ─────────────────────────────────────────────
# 채팅 UI (루트 경로)
# ─────────────────────────────────────────────
@app.get("/")
async def root():
    """루트 경로에서 채팅 UI를 반환한다."""
    index_path = Path(__file__).parent / "static" / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {"message": "Nexus API", "docs": "/docs"}
