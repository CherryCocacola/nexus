"""
FastAPI 웹 인터페이스 — HTTP API 서버.

CLI 외에 HTTP API로도 Nexus를 사용할 수 있게 한다.
SSE 스트리밍, 세션 관리, 도구/모델 조회 등을 제공한다.

의존성 방향: web/ → core/ (단방향)

엔드포인트:
  POST /v1/chat          — 비스트리밍 채팅
  POST /v1/chat/stream   — SSE 스트리밍 채팅
  GET  /v1/sessions      — 세션 목록 조회
  GET  /v1/tools         — 도구 목록 조회
  GET  /v1/models        — 모델 목록 조회
  GET  /health           — 헬스체크
  GET  /metrics          — 메트릭스 조회
"""

from __future__ import annotations

import json
import logging
import re
import uuid
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

from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from web.middleware import CORSConfig, RequestLoggingMiddleware

logger = logging.getLogger("nexus.web.app")


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

        # 웹 전용 QueryEngine — 도구 8개로 축소 (토큰 예산 관리)
        # RTX 5090 (8192 ctx)에서 도구 24개(~6,102토큰)는 컨텍스트 초과.
        # 핵심 도구 8개(~1,851토큰)만 사용하여 입력+출력 공간 확보.
        from core.bootstrap import _create_web_tool_registry
        from core.orchestrator.model_dispatcher import ModelDispatcher
        from core.orchestrator.query_engine import QueryEngine
        from core.tools.base import ToolUseContext

        web_registry = _create_web_tool_registry()
        web_tools = web_registry.get_all_tools()

        # Scout 후보 도구 풀 — 웹 도구 + Scout 도구 (중복 제거, name 기준)
        scout_tools = components.get("scout_tools") or []
        combined_pool: list = []
        seen_names: set[str] = set()
        for t in [*web_tools, *scout_tools]:
            if t.name not in seen_names:
                combined_pool.append(t)
                seen_names.add(t.name)

        # AgentTool이 해석할 의존성을 모두 options에 주입한다
        web_context = ToolUseContext(
            cwd=state.cwd or ".",
            session_id=state.session_id,
            permission_mode=state.permission_mode.value,
            options={
                "memory_manager": components.get("memory_manager"),
                "task_manager": components.get("task_manager"),
                "agent_registry": components.get("agent_registry"),
                "model_provider": components["model_provider"],
                "scout_provider": components.get("scout_provider"),
                "available_tools": combined_pool,
            },
        )
        # 웹 전용 ModelDispatcher — Scout 자동 전처리는 Phase 3에서 제거됨.
        # 웹 경로에서도 Worker는 필요 시 AgentTool로 Scout를 호출한다.
        web_dispatcher = ModelDispatcher(
            tier=components["hardware_tier"],
            worker_provider=components["model_provider"],
            worker_tools=web_tools,
            context=web_context,
            scout_provider=components.get("scout_provider"),
            scout_tools=scout_tools,
            max_turns=200,
        )
        _app_state["model_dispatcher"] = web_dispatcher

        # 웹 시스템 프롬프트 — 서브에이전트 가이드를 동적 주입
        agent_registry = components.get("agent_registry")
        agent_guide = ""
        if agent_registry is not None and len(agent_registry) > 0:
            agent_lines = [
                f"  - {name}: {desc}"
                for name, desc in agent_registry.list_descriptions().items()
            ]
            agent_guide = (
                "\n\n## Sub-agents (Agent tool)\n"
                "Delegate specialized tasks to sub-agents via the Agent tool.\n"
                "Available sub-agents:\n"
                + "\n".join(agent_lines)
                + "\n\nWhen to use sub-agents:\n"
                "  - Simple questions or greetings → answer directly, NO tools\n"
                "  - Single file task → use Read/Edit/Write directly\n"
                "  - Broad project exploration → Agent(subagent_type=\"scout\")\n"
                "NEVER invoke scout for trivial tasks — it is slow (~30s on CPU)."
            )

        web_engine = QueryEngine(
            model_provider=components["model_provider"],
            tools=web_tools,
            context=web_context,
            model_dispatcher=web_dispatcher,
            system_prompt=(
                "You are Nexus, the Worker agent developed by IDINO.\n"
                "You are a 27B model — the brain of the system. Scout (a 4B "
                "assistant) handles all file exploration for you.\n\n"
                "## Your tools (execution only)\n"
                "- Edit: edit an existing file\n"
                "- Write: create a new file (ONLY when the user explicitly asks)\n"
                "- Bash: run a shell command\n"
                "- Agent: delegate exploration to Scout (subagent_type='scout')\n\n"
                "You do NOT have Read/Glob/Grep/LS/DocumentProcess. Scout does.\n"
                "When you need ANY file information — reading, searching, listing, "
                "analyzing documents (.pdf/.docx/.xlsx/.hwp/.pptx) — delegate to Scout:\n"
                "  Agent(prompt='<what you need>', subagent_type='scout')\n\n"
                "## Handling Scout's response (markdown sections)\n"
                "Scout returns a markdown report with 4 sections:\n"
                "  ## relevant_files — list of file paths\n"
                "  ## file_summaries — one-liner per file\n"
                "  ## plan — bullet list of the key facts you need\n"
                "  ## requires_tools — tools you may need to execute\n\n"
                "Read the ## plan section carefully — those bullets are the factual "
                "ground truth extracted from the file. Use them as source material. "
                "Then write a detailed, natural-language answer in the user's "
                "language (Korean if the user wrote Korean). You have 27B "
                "intelligence — turn Scout's raw facts into a rich, well-structured "
                "response.\n\n"
                "## CRITICAL — Scout invocation limit\n"
                "You may call Agent(subagent_type='scout') AT MOST ONCE per user "
                "turn. After Scout returns, you MUST answer the user with whatever "
                "information Scout provided, even if the plan is sparse. NEVER "
                "call Scout a second time in the same turn — this creates a loop.\n"
                "If Scout's plan looks incomplete, work with what you have and tell "
                "the user in Korean what you found plus any caveats (e.g. '문서의 "
                "일부만 요약됐을 수 있습니다'). Asking Scout again will not help.\n\n"
                "## When NOT to use tools\n"
                "- Greetings, general knowledge, conversational — answer directly\n"
                "- Questions you already have full context for — answer directly\n\n"
                "## Hard rules\n"
                "- NEVER create a file the user didn't ask for (no fake logs, no "
                "placeholder files)\n"
                "- NEVER try to Read/Glob/Grep/LS — you don't have those tools, "
                "those calls will fail. Delegate to Scout instead.\n"
                "- If the user attached a text file (content inline in user message "
                "as `[첨부파일: NAME]`), the file content is ALREADY in your context. "
                "Answer from that inline content directly — do NOT delegate to Scout.\n\n"
                "Respond in the user's language. Be helpful and detailed.\n"
                "Do NOT output your thinking process."
                + agent_guide
            ),
            max_turns=200,
        )
        _app_state["query_engine"] = web_engine
        logger.info(
            "웹 서버 부트스트랩 완료 (Phase 1 + 2, 웹 도구 %d개)",
            len(web_registry.get_all_tools()),
        )
    except Exception as e:
        logger.warning(f"부트스트랩 실패, 기본 설정으로 시작: {e}")

    yield

    # 종료: 리소스 정리
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
@app.post("/v1/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    비스트리밍 채팅.

    사용자 메시지를 QueryEngine에 전달하고,
    모든 StreamEvent를 수집하여 최종 응답으로 반환한다.
    """
    # 세션 ID 생성 또는 재사용
    session_id = request.session_id or str(uuid.uuid4())

    engine = _app_state.get("query_engine")
    if engine is None:
        # QueryEngine이 초기화되지 않은 경우 placeholder 응답
        return ChatResponse(
            session_id=session_id,
            response="QueryEngine이 아직 초기화되지 않았습니다.",
            tool_calls=[],
            usage=UsageInfo(),
        )

    # QueryEngine으로 메시지를 처리하고 모든 이벤트를 수집한다
    from core.message import StreamEvent, StreamEventType

    response_text_parts: list[str] = []
    tool_calls_info: list[ToolCallInfo] = []
    usage = UsageInfo()

    async for event in engine.submit_message(request.message):
        if isinstance(event, StreamEvent):
            if event.type == StreamEventType.TEXT_DELTA and event.text:
                response_text_parts.append(event.text)
            elif event.type == StreamEventType.USAGE_UPDATE and event.usage:
                usage = UsageInfo(
                    input_tokens=event.usage.input_tokens,
                    output_tokens=event.usage.output_tokens,
                    total_tokens=event.usage.total_tokens,
                )

    return ChatResponse(
        session_id=engine.session_id,
        response="".join(response_text_parts),
        tool_calls=tool_calls_info,
        usage=usage,
    )


@app.post("/v1/chat/stream")
async def chat_stream(request: ChatRequest) -> StreamingResponse:
    """
    SSE 스트리밍 채팅.

    QueryEngine의 AsyncGenerator에서 yield되는 StreamEvent를
    Server-Sent Events 형식으로 실시간 전송한다.
    """
    session_id = request.session_id or str(uuid.uuid4())

    async def generate() -> AsyncGenerator[str, None]:
        """
        SSE 이벤트를 생성하는 AsyncGenerator.

        QueryEngine의 StreamEvent를 수신하여
        'data: {json}\n\n' 형식으로 실시간 전송한다.
        """
        from core.message import StreamEvent

        engine = _app_state.get("query_engine")
        if engine is None:
            placeholder = {
                "type": "text_delta",
                "text": "QueryEngine이 아직 초기화되지 않았습니다.",
                "session_id": session_id,
            }
            yield f"data: {json.dumps(placeholder, ensure_ascii=False)}\n\n"
            return

        # 세션별 대화 히스토리 관리
        # QueryEngine._messages는 모든 세션이 공유하므로,
        # 매 요청마다 초기화하고 해당 세션의 히스토리만 복원한다.
        if "chat_histories" not in _app_state:
            _app_state["chat_histories"] = {}
        histories = _app_state["chat_histories"]
        if session_id not in histories:
            histories[session_id] = []

        # Qwen3.5 thinking 찌꺼기가 들어있는 과거 메시지를 1회성 정제
        # (세션이 enable_thinking=false 전의 오염된 상태일 수 있다)
        _sanitize_history_inplace(histories[session_id])

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

        # QueryEngine을 통한 전체 에이전트 루프 (도구 사용 포함)
        async for event in engine.submit_message(request.message):
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
                        new_msg = Message.assistant(content_clean) if role == "assistant" else Message.user(content_clean)
                        histories[session_id].append(new_msg)
                    elif content_clean:
                        histories[session_id].append(msg)

    return StreamingResponse(
        generate(),
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
    활성 세션 목록을 반환한다.

    TODO(nexus): 세션 관리자(Phase 5)가 완성되면 실제 세션 목록을 반환한다.
    """
    state = _app_state.get("state")
    if state:
        return {
            "sessions": [
                {
                    "session_id": state.session_id,
                    "active_model": state.active_model,
                    "total_turns": state.total_turns,
                }
            ]
        }
    return {"sessions": []}


# ─────────────────────────────────────────────
# 도구 엔드포인트
# ─────────────────────────────────────────────
@app.get("/v1/tools")
async def list_tools() -> dict[str, Any]:
    """
    등록된 도구 목록을 반환한다.

    ToolRegistry에서 등록된 모든 도구의 이름, 설명, 그룹을 반환한다.
    """
    registry = _app_state.get("tool_registry")
    if registry is None:
        return {"tools": [], "total": 0}

    # ToolRegistry에서 등록된 모든 도구의 정보를 반환한다
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

    # 서브에이전트 메트릭스 — Ch 17 (v7.0 Phase 9 재설계)
    # AgentTool.get_stats()가 subagent_type별 호출 통계를 집계한다.
    # 예: {"scout": {"calls": 3, "total_latency_ms": 99000, "avg_latency_ms": 33000}}
    from core.tools.implementations.agent_tool import AgentTool

    result["agents"] = AgentTool.get_stats()

    # 하위 호환: 기존 대시보드가 result["scout"]을 참조할 수 있으므로 alias 유지.
    # Scout 자동 전처리가 제거됐으므로 Dispatcher.stats는 0만 반환하지만,
    # AgentTool의 "scout" 항목을 평탄화해서 함께 노출한다.
    dispatcher = _app_state.get("model_dispatcher")
    scout_agent_stats = result["agents"].get("scout", {})
    result["scout"] = {
        "tier": dispatcher.tier.value if dispatcher is not None else "unknown",
        "scout_enabled": (
            dispatcher.scout_enabled if dispatcher is not None else False
        ),
        "scout_calls": scout_agent_stats.get("calls", 0),
        "scout_avg_latency_ms": scout_agent_stats.get("avg_latency_ms", 0.0),
        "scout_fallback_count": 0,  # fallback 개념은 AgentTool 이관 후 의미 없음
        "note": "scout_calls/avg_latency_ms are sourced from AgentTool.get_stats().",
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
