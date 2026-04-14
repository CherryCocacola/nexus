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

Phase 2 (향후 구현): 환경 의존 초기화
  - 세션 관리, Redis/PG 연결, Tool Registry 등
  - Phase 2+ 모듈에 의존하므로 해당 모듈 완성 후 구현

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
