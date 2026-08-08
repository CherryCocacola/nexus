# 재시작한 프로세스가 깨끗하게 떴는지 판정한다 — 종료 traceback 과 섞이지 않게.
"""
기동 판정 — `docker logs` 에서 **이번에 뜬 프로세스** 구간만 잘라서 본다.

[왜 만들었나 — 2026-08-08 오탐]
    배포 스크립트가 로그를 `--tail 60 | grep -i traceback` 으로 통째로 훑어서,
    **직전 프로세스의 정상 종료 경로**를 기동 실패로 읽었다.

        SIGTERM → bootstrap._shutdown_handler 의 sys.exit(0)
                → uvloop 신호 처리 중 SystemExit
                → lifespan receive 의 CancelledError

    이건 종료가 잘 된 흔적이지 실패가 아니다. 그런데 재시작하면 **항상** 로그 끝에
    남으므로, 안 자르면 배포할 때마다 같은 오탐이 난다. 실제로 이번 배포는 기능
    e2e 7/7 을 통과하고도 "부트스트랩 예외"로 FAIL 이 떴다.

    오탐이 반복되면 더 나쁜 일이 생긴다 — 사람이 그 항목을 무시하기 시작해서,
    **진짜 기동 실패가 났을 때도 그냥 넘기게 된다.** 그래서 고친다.

[왜 이 파일에 두나]
    배포마다 임시 스크립트에 grep 을 새로 쓰는 것이 원인이었다. 판정 규칙을 한 곳에
    두고 테스트로 고정해야 다음 배포에서 같은 실수를 반복하지 않는다.

[사용]
    from deployment.startup_check import check_startup

    logs = ssh("docker logs --tail 500 nexus-web 2>&1")
    result = check_startup(logs)
    if not result.ok:
        print(result.reason)
        for ln in result.errors:
            print("  ", ln)

    SSH 는 호출부가 한다 — 이 모듈은 문자열만 다루므로 서버 없이 테스트된다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# uvicorn 이 워커 프로세스를 띄울 때 남기는 줄. 이 줄 **이후**가 이번 프로세스다.
# 앞의 것을 우선한다 — 뒤엣것은 앞 줄이 잘려 나갔을 때만 쓰는 대체 마커다.
# (둘을 동등하게 보면 구간이 한 줄 늦게 시작해 기동 초반 로그를 놓친다.)
_START_MARKERS = (
    "Started server process",
    "Waiting for application startup",
)
# 기동이 끝까지 갔다는 표식.
_READY_MARKER = "Application startup complete"
# 실패로 볼 표식. 종료 경로에서 나오는 CancelledError/SystemExit 는 여기에 없다 —
# 그것들은 정상 종료의 일부라 기동 구간에서만 봐도 걸리지 않는다.
_ERROR_MARKERS = (
    "Traceback (most recent call last)",
    "ImportError",
    "ModuleNotFoundError",
    "SyntaxError",
)


@dataclass(frozen=True)
class StartupResult:
    """기동 판정 결과."""

    ok: bool
    reason: str
    #: 판정 근거가 된 줄들(실패했을 때만 채운다).
    errors: list[str] = field(default_factory=list)
    #: 판정에 실제로 본 줄 수(기동 구간). 마커를 못 찾으면 전체 줄 수가 들어온다.
    segment_lines: int = 0


def startup_segment(logs: str) -> list[str]:
    """마지막 기동 마커 이후 구간만 돌려준다.

    마커가 없으면 **전체**를 돌려준다. 로그가 잘려 마커가 안 보이는 경우인데,
    이때 빈 목록을 주면 "오류 없음"으로 읽혀 진짜 실패를 놓친다. 넓게 보는 쪽이
    안전하다(fail-closed).
    """
    lines = logs.splitlines()
    # 우선순위대로 훑고, 먼저 걸리는 마커의 **마지막** 등장 위치를 쓴다.
    for marker in _START_MARKERS:
        last = -1
        for i, line in enumerate(lines):
            if marker in line:
                last = i
        if last >= 0:
            return lines[last:]
    return lines


def check_startup(logs: str) -> StartupResult:
    """이번에 뜬 프로세스가 깨끗하게 기동했는지 판정한다.

    Args:
        logs: `docker logs` 원문. 종료된 이전 프로세스의 로그가 섞여 있어도 된다.

    Returns:
        StartupResult. `ok=False` 면 `errors` 에 근거가 된 줄이 담긴다.
    """
    segment = startup_segment(logs)
    if not segment:
        return StartupResult(ok=False, reason="로그가 비어 있어 기동을 확인할 수 없다")

    errors = [ln for ln in segment if any(m in ln for m in _ERROR_MARKERS)]
    if errors:
        return StartupResult(
            ok=False,
            reason=f"기동 구간에 오류 {len(errors)}건",
            errors=errors[:10],
            segment_lines=len(segment),
        )

    if not any(_READY_MARKER in ln for ln in segment):
        # 오류는 없는데 완료 표식도 없다 = 아직 뜨는 중이거나 중간에 멈췄다.
        # "오류 없음"만 보고 통과시키면 멈춘 서버를 정상으로 읽는다.
        return StartupResult(
            ok=False,
            reason=f"'{_READY_MARKER}' 를 찾지 못했다(기동 중이거나 중단됨)",
            segment_lines=len(segment),
        )

    return StartupResult(ok=True, reason="기동 정상", segment_lines=len(segment))
