# 기동 판정을 고정한다 — 종료 traceback 을 실패로 읽으면 재시작마다 오탐이 난다.
"""
2026-08-08 실측 오탐. 배포 스크립트가 `docker logs --tail 60 | grep -i traceback` 으로
로그를 통째로 훑어서, **직전 프로세스의 정상 종료 경로**를 기동 실패로 읽었다.
기능 e2e 7/7 을 통과한 배포가 "부트스트랩 예외"로 FAIL 이 떴다.

아래 픽스처는 그때 실제로 나온 로그를 줄인 것이다. 종료 traceback 이 로그 끝에
남는 것은 **재시작할 때마다 일어나는 일**이라, 이 케이스를 고정해 두지 않으면
같은 오탐이 계속 돌아온다.

오탐이 반복되면 사람이 그 항목을 무시하게 되고, 그때는 진짜 실패도 넘어간다.
"""

from __future__ import annotations

from deployment.startup_check import check_startup, startup_segment

# ── 실측 로그(축약) — 정상 재시작 ────────────────────────
# 앞: 옛 프로세스의 SIGTERM 종료(= traceback 이 남는다)
# 뒤: 새 프로세스의 정상 기동
RESTART_LOG = """\
2026-08-08 15:25:40 [nexus.bootstrap] INFO: 종료 시그널 수신: SIGTERM
ERROR:    Traceback (most recent call last):
  File "uvloop/loop.pyx", line 379, in uvloop.loop.Loop._ceval_process_signals
  File "/app/core/bootstrap.py", line 1799, in _shutdown_handler
    sys.exit(0)
SystemExit: 0

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/usr/local/lib/python3.11/site-packages/starlette/routing.py", line 645, in lifespan
    await receive()
asyncio.exceptions.CancelledError

2026-08-08 15:25:40 [nexus.bootstrap] INFO: [Phase 2] 임베딩 keepalive 종료
INFO:     Started server process [1]
INFO:     Waiting for application startup.
2026-08-08 15:25:40 [nexus.bootstrap] INFO: [Phase 2] ToolRegistry 초기화: 26개 도구
2026-08-08 15:25:41 [nexus.bootstrap] INFO: [Phase 2] Redis 연결 성공
INFO:     Application startup complete.
"""

# ── 진짜 기동 실패 ────────────────────────
BROKEN_LOG = """\
INFO:     Started server process [1]
INFO:     Waiting for application startup.
ERROR:    Traceback (most recent call last):
  File "/app/web/app.py", line 12, in <module>
    from core.tools.implementations.client_tool import build_client_tools
ModuleNotFoundError: No module named 'core.tools.implementations.client_tool'
"""


# ─────────────────────────────────────────────
# ★핵심 — 종료 traceback 은 실패가 아니다
# ─────────────────────────────────────────────
def test_shutdown_traceback_is_not_a_startup_failure() -> None:
    """이것이 이 모듈이 존재하는 이유다.

    SIGTERM → sys.exit(0) → CancelledError 는 **종료가 잘 됐다**는 흔적이다.
    재시작하면 항상 로그에 남으므로, 이걸 실패로 읽으면 배포마다 오탐이 난다.
    """
    result = check_startup(RESTART_LOG)

    assert result.ok, result.reason
    assert result.errors == []


def test_real_startup_failure_is_caught() -> None:
    """오탐을 없애자고 진짜 실패까지 놓치면 안 된다 — import 실패는 잡아야 한다."""
    result = check_startup(BROKEN_LOG)

    assert not result.ok
    assert any("ModuleNotFoundError" in ln for ln in result.errors)


def test_segment_starts_at_the_last_startup_marker() -> None:
    """구간이 정확히 잘리는지 — 종료 로그가 한 줄도 섞이면 안 된다."""
    segment = startup_segment(RESTART_LOG)

    assert segment[0].startswith("INFO:     Started server process")
    assert not any("SIGTERM" in ln for ln in segment)
    assert not any("CancelledError" in ln for ln in segment)


# ─────────────────────────────────────────────
# 판정을 신뢰할 수 없는 경우 — 통과시키지 않는다
# ─────────────────────────────────────────────
def test_missing_complete_marker_is_not_ok() -> None:
    """오류가 없어도 완료 표식이 없으면 통과가 아니다(기동 중이거나 멈춘 상태)."""
    partial = "INFO:     Started server process [1]\nINFO:     Waiting for application startup.\n"

    result = check_startup(partial)

    assert not result.ok
    assert "Application startup complete" in result.reason


def test_secondary_marker_is_only_a_fallback() -> None:
    """`Started server process` 가 잘려 나갔을 때만 대체 마커를 쓴다.

    둘을 동등하게 보면 구간이 한 줄 늦게 시작해 기동 초반 로그를 놓친다.
    """
    cut = """\
INFO:     Waiting for application startup.
ERROR:    Traceback (most recent call last):
ImportError: boom
"""

    result = check_startup(cut)

    assert not result.ok
    assert any("ImportError" in ln for ln in result.errors)


def test_no_marker_falls_back_to_whole_log() -> None:
    """마커가 없으면 전체를 본다 — 빈 목록을 주면 '오류 없음'으로 잘못 읽힌다."""
    truncated = "ERROR:    Traceback (most recent call last):\nModuleNotFoundError: nope\n"

    result = check_startup(truncated)

    assert not result.ok
    assert result.errors


def test_empty_log_is_not_ok() -> None:
    assert not check_startup("").ok
