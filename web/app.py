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

import logging
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI
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
        description="사용할 모델 (primary: Gemma 4, auxiliary: ExaOne)",
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
        _app_state["query_engine"] = components.get("query_engine")
        _app_state["tool_registry"] = components.get("tool_registry")
        _app_state["model_provider"] = components.get("model_provider")
        logger.info("웹 서버 부트스트랩 완료 (Phase 1 + 2)")
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
        import json

        from core.message import Message, StreamEvent

        engine = _app_state.get("query_engine")
        if engine is None:
            placeholder = {
                "type": "text_delta",
                "text": "QueryEngine이 아직 초기화되지 않았습니다.",
                "session_id": session_id,
            }
            yield f"data: {json.dumps(placeholder, ensure_ascii=False)}\n\n"
            return

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
                if event.stop_reason:
                    stop_val = event.stop_reason
                    sse_data["stop_reason"] = (
                        stop_val if isinstance(stop_val, str) else stop_val.value
                    )
                yield f"data: {json.dumps(sse_data, ensure_ascii=False)}\n\n"

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
                name="Gemma 4 31B",
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
            {"id": "gemma-4-31b-it", "name": "Gemma 4 31B", "role": "primary"},
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

    return result


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
