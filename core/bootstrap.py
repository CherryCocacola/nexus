"""
부트스트랩 — 2-Phase 초기화 시스템.

Claude Code의 entrypoints/init.ts + setup.ts를 Python으로 재구현한다.

Phase 1 (이 파일): 환경 비의존 초기화
  - GlobalState 생성
  - 설정 로딩 + 검증
  - 로깅 구성
  - 정상 종료 핸들러 등록
  - GPU 서버 사전 연결 (fire-and-forget)
  - 플랫폼 감지

Phase 2 (init_phase2): 환경 의존 초기화
  - ToolRegistry (24개 도구 등록)
  - MemoryManager (단기+장기 메모리)
  - QueryEngine (Tier 1 세션 오케스트레이터)

왜 2-Phase인가: Claude Code가 이 패턴을 사용하는 이유는
Phase 1이 src/ 내부 모듈을 전혀 import하지 않아서
순환 의존이 원천 차단되기 때문이다 (DAG leaf 격리).
"""

from __future__ import annotations

import asyncio
import logging
import os
import platform
import signal
import sys
from pathlib import Path

from core.config import load_and_validate_config
from core.state import GlobalState, get_initial_state

logger = logging.getLogger("nexus.bootstrap")


# ─────────────────────────────────────────────
# Phase 1: 환경 비의존 초기화
# ─────────────────────────────────────────────
async def init(
    config_path: str | None = None,
    cwd: str | None = None,
) -> GlobalState:
    """
    Phase 1 초기화. CLI, SDK, 웹 서버 등 어떤 진입점이든 이 함수를 먼저 호출한다.
    bootstrap 모듈만 사용하며, core/ 내부의 다른 모듈은 import하지 않는다.

    Args:
        config_path: 설정 파일 경로. None이면 자동 탐색한다.
        cwd: 작업 디렉토리. None이면 현재 디렉토리를 사용한다.

    Returns:
        초기화된 GlobalState 싱글톤
    """
    # ① GlobalState 초기화
    state = get_initial_state(cwd=cwd)
    logger.info(f"[Phase 1] GlobalState 초기화 완료: session={state.session_id}")

    # ② 설정 로딩 + 검증
    config = load_and_validate_config(config_path)
    state.config = config
    logger.info(f"[Phase 1] 설정 로드 완료: gpu_server={config.gpu_server_url}")

    # ③ 로깅 구성
    _configure_logging(config.log_level, config.log_file)

    # ④ 정상 종료 핸들러 등록
    _setup_graceful_shutdown(state)
    logger.info("[Phase 1] 종료 핸들러 등록 완료")

    # ⑤ GPU 서버 사전 연결 (fire-and-forget)
    # 왜 fire-and-forget인가: GPU 서버가 아직 안 떠 있어도 부트스트랩은 진행해야 한다.
    # 첫 추론 요청 전까지 연결이 되면 100-200ms를 절약할 수 있다.
    asyncio.create_task(_preconnect_gpu_server(config.gpu_server_url))

    # ⑥ 플랫폼 감지
    state.platform = _detect_platform()
    logger.info(f"[Phase 1] 플랫폼: {state.platform.get('os', 'unknown')}")

    return state


# ─────────────────────────────────────────────
# Phase 2: 환경 의존 초기화
# ─────────────────────────────────────────────
async def init_phase2(state: GlobalState) -> dict:
    """
    Phase 2 초기화. Phase 1(init) 이후 호출한다.
    core/ 내부 모듈(ToolRegistry, MemoryManager, QueryEngine)을 초기화한다.

    Args:
        state: Phase 1에서 생성된 GlobalState

    Returns:
        초기화된 컴포넌트 딕셔너리:
          - tool_registry: ToolRegistry (24개 도구 등록됨)
          - memory_manager: MemoryManager (인메모리 폴백)
          - model_provider: LocalModelProvider
          - query_engine: QueryEngine (Tier 1)
    """
    # lazy import — Phase 2 모듈은 Phase 1에서 import하지 않는다
    from core.memory.long_term import LongTermMemory
    from core.memory.manager import MemoryManager
    from core.memory.short_term import ShortTermMemory
    from core.model.inference import LocalModelProvider
    from core.orchestrator.query_engine import QueryEngine
    from core.task import TaskManager
    from core.tools.base import ToolUseContext

    components: dict = {}

    # ① ModelProvider 생성
    config = state.config
    provider = LocalModelProvider(
        base_url=config.gpu_server_url,
        model_id=config.model.primary_model,
        max_context_tokens=config.model.max_model_len,
        max_output_tokens=config.model.max_output_tokens,
    )
    components["model_provider"] = provider
    logger.info("[Phase 2] ModelProvider 초기화: %s", config.gpu_server_url)

    # ② ToolRegistry — 24개 도구 등록
    registry = _create_tool_registry()
    components["tool_registry"] = registry
    logger.info("[Phase 2] ToolRegistry 초기화: %d개 도구", registry.tool_count)

    # ③ MemoryManager — 인메모리 폴백 (Redis/PG 없이 동작)
    stm = ShortTermMemory(redis_client=None)
    ltm = LongTermMemory(pg_pool=None)
    memory_manager = MemoryManager(
        short_term=stm,
        long_term=ltm,
        model_provider=provider,
    )
    components["memory_manager"] = memory_manager
    logger.info("[Phase 2] MemoryManager 초기화: 인메모리 폴백")

    # ③-b TaskManager — 비동기 태스크 라이프사이클 관리
    task_manager = TaskManager()
    components["task_manager"] = task_manager
    logger.info("[Phase 2] TaskManager 초기화")

    # ④ ToolUseContext 생성
    context = ToolUseContext(
        cwd=state.cwd or os.getcwd(),
        session_id=state.session_id,
        permission_mode=state.permission_mode.value,
        options={
            "memory_manager": memory_manager,
            "task_manager": task_manager,
        },
    )
    components["tool_use_context"] = context

    # ⑤ QueryEngine — Tier 1 세션 오케스트레이터
    # 시스템 프롬프트는 기본값 사용 (CLI/Web에서 커스터마이즈 가능)
    tools = registry.get_all_tools()
    engine = QueryEngine(
        model_provider=provider,
        tools=tools,
        context=context,
        system_prompt=_build_default_system_prompt(),
        max_turns=200,
    )
    components["query_engine"] = engine
    logger.info("[Phase 2] QueryEngine 초기화: session=%s", engine.session_id)

    return components


def _create_tool_registry():  # noqa: ANN202 — ToolRegistry는 함수 내부에서 import
    """24개 도구를 등록한 ToolRegistry를 생성한다."""
    from core.tools.implementations.bash_tool import BashTool
    from core.tools.implementations.docker_tools import DockerBuildTool, DockerRunTool
    from core.tools.implementations.edit_tool import EditTool
    from core.tools.implementations.git_tools import (
        GitBranchTool,
        GitCheckoutTool,
        GitCommitTool,
        GitDiffTool,
        GitLogTool,
        GitStatusTool,
    )
    from core.tools.implementations.glob_tool import GlobTool
    from core.tools.implementations.grep_tool import GrepTool
    from core.tools.implementations.ls_tool import LSTool
    from core.tools.implementations.memory_tools import MemoryReadTool, MemoryWriteTool
    from core.tools.implementations.multi_edit_tool import MultiEditTool
    from core.tools.implementations.notebook_tools import NotebookEditTool, NotebookReadTool
    from core.tools.implementations.read_tool import ReadTool
    from core.tools.implementations.task_tools import TaskTool, TodoReadTool, TodoWriteTool
    from core.tools.implementations.write_tool import WriteTool
    from core.tools.registry import ToolRegistry

    registry = ToolRegistry()
    registry.register_many([
        # 파일 시스템 (4개)
        ReadTool(),
        WriteTool(),
        EditTool(),
        MultiEditTool(),
        # 실행 (1개)
        BashTool(),
        # 검색 (3개)
        GlobTool(),
        GrepTool(),
        LSTool(),
        # Git (6개)
        GitLogTool(),
        GitDiffTool(),
        GitStatusTool(),
        GitCommitTool(),
        GitBranchTool(),
        GitCheckoutTool(),
        # 노트북 (2개)
        NotebookReadTool(),
        NotebookEditTool(),
        # 태스크 (3개)
        TodoReadTool(),
        TodoWriteTool(),
        TaskTool(),
        # 메모리 (2개)
        MemoryReadTool(),
        MemoryWriteTool(),
        # Docker (2개)
        DockerBuildTool(),
        DockerRunTool(),
    ])

    return registry


def _build_default_system_prompt() -> str:
    """기본 시스템 프롬프트를 생성한다."""
    return (
        "You are Nexus, an AI coding assistant running in an air-gapped environment.\n"
        "You help users with software engineering tasks using available tools.\n"
        "Always respond in the user's language.\n"
        "When you need to read, write, or modify files, use the appropriate tools.\n"
        "Think step by step before taking actions."
    )


# ─────────────────────────────────────────────
# 로깅 구성
# ─────────────────────────────────────────────
def _configure_logging(level: str, log_file: str | None) -> None:
    """
    로깅을 구성한다.
    콘솔(stderr) + 파일(선택) 핸들러를 설정한다.
    """
    log_level = getattr(logging, level.upper(), logging.INFO)
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]

    if log_file:
        # 로그 디렉토리가 없으면 생성
        log_dir = Path(log_file).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,
    )


# ─────────────────────────────────────────────
# 정상 종료 핸들러
# ─────────────────────────────────────────────
# 종료 시 실행할 클린업 함수 목록
_cleanup_handlers: list = []


def register_cleanup(handler) -> None:
    """종료 시 실행할 클린업 함수를 등록한다."""
    _cleanup_handlers.append(handler)


def _setup_graceful_shutdown(state: GlobalState) -> None:
    """
    정상 종료 핸들러를 등록한다.
    Claude Code의 setupGracefulShutdown()에 대응한다.

    SIGINT/SIGTERM 수신 시:
      1. 진행 중인 도구 실행을 취소한다
      2. 세션 요약을 로깅한다
      3. 등록된 클린업 함수를 실행한다
    """

    def _shutdown_handler(signum, frame):
        """시그널 핸들러: 정상 종료를 수행한다."""
        sig_name = signal.Signals(signum).name
        logger.info(f"종료 시그널 수신: {sig_name}")

        # 세션 요약 로깅
        summary = state.get_session_summary()
        logger.info(f"세션 요약: {summary}")

        # 등록된 클린업 함수 실행
        for handler in _cleanup_handlers:
            try:
                handler()
            except Exception as e:
                logger.error(f"클린업 핸들러 실행 실패: {e}")

        sys.exit(0)

    # SIGINT (Ctrl+C), SIGTERM 핸들러 등록
    signal.signal(signal.SIGINT, _shutdown_handler)
    # Windows에서는 SIGTERM이 지원되지 않을 수 있다
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _shutdown_handler)


# ─────────────────────────────────────────────
# GPU 서버 사전 연결
# ─────────────────────────────────────────────
async def _preconnect_gpu_server(gpu_server_url: str) -> None:
    """
    GPU 서버에 사전 연결한다.
    Claude Code의 preconnectAnthropicApi()에 대응한다.
    HTTP 커넥션 풀을 워밍업하여 첫 추론 요청에서 100-200ms를 절약한다.
    """
    try:
        import httpx

        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{gpu_server_url}/health")
            if resp.status_code == 200:
                data = resp.json()
                logger.info(
                    f"[Phase 1] GPU 서버 사전 연결 성공: "
                    f"gpu={data.get('gpu', 'unknown')}, "
                    f"tier={data.get('gpu_tier', 'unknown')}"
                )
            else:
                logger.warning(
                    f"[Phase 1] GPU 서버 상태 이상: status={resp.status_code}"
                )
    except ImportError:
        logger.warning("[Phase 1] httpx가 설치되지 않아 사전 연결을 건너뜁니다")
    except Exception as e:
        # fire-and-forget이므로 실패해도 계속 진행한다
        logger.warning(f"[Phase 1] GPU 서버 사전 연결 실패: {e}")


# ─────────────────────────────────────────────
# 플랫폼 감지
# ─────────────────────────────────────────────
def _detect_platform() -> dict:
    """
    플랫폼 정보를 감지한다.
    Claude Code의 setShellIfWindows()에 대응한다.
    """
    info = {
        "os": platform.system(),
        "os_version": platform.version(),
        "python_version": platform.python_version(),
        "arch": platform.machine(),
    }

    # Windows에서 git-bash 감지
    if info["os"] == "Windows":
        git_bash = Path("C:/Program Files/Git/bin/bash.exe")
        if git_bash.exists():
            info["shell"] = str(git_bash)
        else:
            info["shell"] = os.environ.get("COMSPEC", "cmd.exe")
    else:
        info["shell"] = os.environ.get("SHELL", "/bin/bash")

    return info
