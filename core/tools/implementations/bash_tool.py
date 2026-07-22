"""
Bash 도구 — 셸(shell) 명령어 실행 구현 모듈.

이 파일은 Nexus의 24개 도구 중 "Bash" 도구를 구현한다.
LLM(모델)이 명령어 실행을 요청하면, 이 도구가 실제 운영체제 셸에서
그 명령어를 돌린 뒤 종료 코드(exit_code)·표준출력(stdout)·표준에러(stderr)를
모아서 모델에게 돌려준다. 즉, 모델이 파일 목록 조회·빌드·테스트 실행 같은
시스템 작업을 수행할 수 있게 해주는 "손과 발" 역할이다.

이 도구가 특별히 조심스러운 이유:
  - 셸 명령어는 파일 삭제·시스템 변경 등 되돌릴 수 없는 일을 할 수 있다.
    그래서 is_destructive=True, requires_confirmation=True로 두어
    실행 전에 항상 사용자 확인(ASK)을 거치도록 만든다 (fail-closed 원칙).
  - 에어갭(폐쇄망) 환경이므로 curl·wget 같은 외부 네트워크 호출은
    이 도구가 아니라 그 앞단의 권한 파이프라인(Layer 2)에서 차단된다.

구성 요소:
  - BashTool          : BaseTool을 상속한 도구 본체 클래스.
  - _run_command()    : 실제 subprocess.run 호출을 감싼 블로킹 헬퍼.
  - _truncate_output(): 출력이 너무 길 때 뒷부분만 남기는 헬퍼.

의존:
  - core.tools.base 의 BaseTool ABC 및 결과/권한 타입.
  - 상위 executor가 이 도구의 call()을 호출하고, 결과를 4-Tier 체인으로 전달한다.

작성자: 이현수 / 작성일: 2026-07-05
"""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
from typing import Any

from core.tools.base import (
    BaseTool,
    PermissionBehavior,
    PermissionResult,
    ToolResult,
    ToolUseContext,
)

# 모듈 전용 로거. "nexus.tools.bash" 이름 규칙을 따라 다른 모듈 로그와
# 구분되게 한다. 명령어 실행/종료 코드 등을 여기에 기록한다.
logger = logging.getLogger("nexus.tools.bash")

# stdout/stderr를 붙잡아 둘 최대 글자 수. 이 크기를 넘으면 앞부분을 버리고
# 뒷부분(최근 출력)만 남긴다. 모델 컨텍스트가 거대한 로그로 넘치는 것을 막기 위함.
_MAX_OUTPUT_SIZE = 50_000

# ── P0-a 출력 위생 (2026-07-22, FABLE5 진단) ──────────────────────────────
# 디렉토리 나열 명령(ls -R·find·tree·dir·Get-ChildItem)이 node_modules 같은
# 대형 의존성 트리를 통째로 뱉으면 컨텍스트가 수만 토큰의 파일명으로 오염된다.
# 실측: prompt_flow 분석에서 ls 1회가 37,099 토큰을 주입 → 모델이 node_modules의
# .js 파일명만 보고 백엔드를 "Node.js"로 오판(실제 FastAPI). 아래 두 장치로 막는다.
#   1) 나열 명령의 출력에서 노이즈 경로가 포함된 줄을 제거한다.
#   2) 나열 명령은 상한을 별도로(작게) 적용해 남은 대량 목록도 잘라낸다.
# 일반 명령(빌드·테스트 로그 등)은 꼬리가 중요하므로 이 프루닝을 적용하지 않는다.

# 나열 명령으로 간주할 첫 토큰(파이프·리다이렉트 앞의 실행 파일명 기준).
_LISTING_COMMANDS = frozenset({"ls", "find", "tree", "dir", "get-childitem", "gci"})

# 지식이 아니라 노이즈인 디렉토리 이름. 경로에 이 이름이 "한 컴포넌트"로 들어가면
# (맨 앞이든 중간이든) 그 줄을 제거한다. 예: node_modules/react·./dist/x·a/.venv/b 모두 매칭.
_NOISE_DIR_NAMES = frozenset({
    "node_modules", ".git", "dist", "build", ".venv", "venv",
    "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    ".next", ".turbo", "site-packages",
})

# 나열 명령에 적용하는 축소 상한(일반 상한 50k보다 훨씬 작게).
# 노이즈 제거 후에도 대형 저장소는 파일이 많으므로 앞부분(대개 최상위 구조)만 남긴다.
_LISTING_MAX_SIZE = 8_000


def _first_token(command: str) -> str:
    """명령 문자열의 첫 실행 토큰을 소문자로 돌려준다(나열 명령 판별용).

    파이프/리다이렉트/인자 앞의 실행 파일명만 본다. 완벽한 셸 파서가 아니라
    ls·find·tree 같은 나열 명령을 값싸게 식별하기 위한 최소 로직이다.
    """
    stripped = command.strip()
    if not stripped:
        return ""
    # 첫 단어만 취해 경로(/usr/bin/ls)·확장자를 떼고 소문자화한다.
    first = stripped.split()[0]
    base = first.replace("\\", "/").rsplit("/", 1)[-1]
    return base.lower()


def _prune_listing_output(output: str) -> str:
    """나열 명령의 출력에서 노이즈 경로가 포함된 줄을 제거한다.

    node_modules 등 의존성 트리의 파일명이 컨텍스트를 오염시키는 것을 막는다.
    제거한 줄 수를 요약으로 덧붙여, 모델이 "숨겨진 게 있다"는 사실은 알게 한다
    (조용한 절단은 "다 봤다"는 착각을 부르므로 명시한다).
    """
    lines = output.splitlines()
    kept: list[str] = []
    dropped = 0
    for line in lines:
        # 경로 구분자를 /로 통일하고 컴포넌트로 쪼갠다. 노이즈 디렉토리 이름이
        # 한 컴포넌트로 들어 있으면(위치 무관) 그 줄을 버린다. .egg-info는 접미사라
        # 별도로 검사한다(디렉토리 이름이 <pkg>.egg-info 형태).
        comps = line.replace("\\", "/").lower().split("/")
        if _NOISE_DIR_NAMES.intersection(comps) or any(c.endswith(".egg-info") for c in comps):
            dropped += 1
            continue
        kept.append(line)
    if dropped == 0:
        return output
    result = "\n".join(kept)
    return (
        f"{result}\n"
        f"[출력 위생: node_modules 등 의존성/빌드 경로 {dropped}줄 제외. "
        f"이 목록은 프로젝트 소스 구조만 담는다.]"
    )


class BashTool(BaseTool):
    """
    셸 명령어 실행 도구 (BaseTool 구현체).

    모델이 넘긴 command 문자열을 운영체제 셸에서 실행하고, 그 결과를
    ToolResult로 감싸 돌려준다. Nexus 도구 중 가장 강력하면서 위험하기에
    설계상 "가장 제한적인" 기본값을 그대로 유지한다:
      - requires_confirmation=True : 실행 전 항상 사용자 확인을 요구.
      - is_destructive=True        : 되돌릴 수 없는 변경을 일으킬 수 있음을 표시.

    BaseTool 규약에 따라 name/description/input_schema 등의 프로퍼티와
    validate_input / check_permissions / call 라이프사이클 메서드를 구현한다.
    실제 실행 로직은 아래 모듈 수준 헬퍼(_run_command)에 위임한다.
    """

    # ═══ 1. Identity(정체성) ═══
    # 레지스트리 등록·조회·프롬프트 노출에 쓰이는 도구의 이름표들.

    @property
    def name(self) -> str:
        # 모델과 레지스트리가 이 도구를 부를 때 쓰는 고유 이름.
        return "Bash"

    @property
    def description(self) -> str:
        # 모델에게 보여줄 한 줄 설명. 도구 스키마의 일부로 프롬프트에 실린다.
        return "Execute a shell command."

    @property
    def group(self) -> str:
        # 도구 분류(카테고리). "execution"은 실행 계열 도구를 뜻한다.
        return "execution"

    # ═══ 2. Schema(입력 스키마) ═══
    # 모델이 이 도구를 호출할 때 넘겨야 하는 인자의 JSON Schema.
    # 이 스키마로 입력을 검증하고, 모델에게 어떤 필드가 있는지 알려준다.

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                # 실행할 셸 명령어 문자열. 유일한 필수 필드다.
                "command": {
                    "type": "string",
                    "description": "Shell command to execute",
                },
                # 타임아웃(초). 지정하지 않으면 아래 default(120)가 쓰인다.
                # 1초~600초 범위로 제한해 무한 대기·과도한 값 입력을 막는다.
                # (설명 문구의 "default: 30"은 옛 표기이며 실제 기본값은 120)
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in seconds (default: 30)",
                    "default": 120,
                    "minimum": 1,
                    "maximum": 600,
                },
                # 명령어가 무슨 일을 하는지 사람이 읽을 짧은 설명.
                # 실행에는 영향이 없고 UI 진행 표시(progress label)에만 쓰인다.
                "description": {
                    "type": "string",
                    "description": "명령어에 대한 간단한 설명 (UI 표시용)",
                },
            },
            # command만 필수. timeout·description은 선택 항목이다.
            "required": ["command"],
        }

    # ═══ 3. Behavior Flags(동작 플래그) ═══
    # 이 도구의 위험 특성을 권한/실행 시스템에 알리는 스위치들.
    # Bash는 가장 위험한 도구라 두 플래그 모두 True로 켜 둔다.

    @property
    def requires_confirmation(self) -> bool:
        # True → 실행 전 반드시 사용자 확인을 받는다. 절대 자동 실행되지 않는다.
        return True

    @property
    def is_destructive(self) -> bool:
        # True → 파일 삭제 등 되돌릴 수 없는 변경을 일으킬 수 있음을 표시.
        return True

    # ═══ 4. Limits(제한) ═══

    @property
    def timeout_seconds(self) -> float:
        """
        도구 실행에 걸리는 기본 타임아웃(초).

        여기서는 120초를 기본으로 쓴다. 다만 실제 call()에서는 입력의
        timeout 필드가 있으면 그 값이 우선하므로, 이 프로퍼티는 상위
        실행기(executor)가 참고하는 기본 상한선 역할을 한다.
        """
        return 120.0

    # ═══ 5. Lifecycle(라이프사이클) ═══
    # BaseTool 규약의 실행 단계: validate_input → check_permissions → call.

    def validate_input(self, input_data: dict[str, Any]) -> str | None:
        """
        입력값을 실행 전에 가볍게 검증한다.

        command가 아예 없거나 공백뿐이면 실행할 것이 없으므로 에러 메시지
        문자열을 돌려준다. 문제가 없으면 None을 반환해 "통과"를 알린다.

        Args:
            input_data: 모델이 넘긴 입력 딕셔너리(command/timeout/description).
        Returns:
            문제가 있으면 한글 에러 메시지, 없으면 None.
        """
        command = input_data.get("command", "")
        # 빈 문자열이거나 공백만 있는 경우 → 실행 불가로 판정.
        if not command or not command.strip():
            return "command는 비어 있을 수 없습니다."
        return None

    async def check_permissions(
        self,
        input_data: dict[str, Any],
        context: ToolUseContext,
    ) -> PermissionResult:
        """
        이 도구를 실행해도 되는지 판단한다.

        Bash는 위험도가 높아 무조건 ASK(사용자에게 물어보기)를 반환한다.
        즉 스스로 ALLOW하지 않고, 최종 결정은 사용자 확인에 맡긴다.
        message에는 어떤 명령어를 돌릴지 담아 확인 창에 그대로 노출한다.

        Args:
            input_data: command 등 입력 딕셔너리.
            context: 실행 맥락(현재 디렉토리 등)을 담은 컨텍스트.
        Returns:
            behavior=ASK 로 채운 PermissionResult.
        """
        command = input_data.get("command", "")
        return PermissionResult(
            behavior=PermissionBehavior.ASK,
            message=f"Run: {command}",
        )

    async def call(
        self,
        input_data: dict[str, Any],
        context: ToolUseContext,
    ) -> ToolResult:
        """
        셸 명령어를 실제로 실행하고 그 결과를 ToolResult로 반환한다.

        이 메서드가 도구의 심장부다. 권한 확인을 통과한 뒤 호출되며,
        블로킹 방식의 subprocess를 asyncio 스레드로 넘겨 이벤트 루프를
        막지 않게 처리한다.

        처리 순서:
          1. 입력에서 command와 timeout을 꺼낸다.
          2. 작업 디렉토리를 context.cwd로 정한다.
          3. _run_command를 별도 스레드에서 실행한다(타임아웃 적용).
          4. 예외(타임아웃/디렉토리 없음/기타 OS 오류)는 error 결과로 감싼다.
          5. stdout/stderr를 잘라내고, 종료 코드와 함께 문자열로 조합한다.
          6. 결과를 ToolResult.success로 반환한다.

        Args:
            input_data: command(필수)·timeout(선택) 등을 담은 입력.
            context: 현재 작업 디렉토리(cwd) 등 실행 맥락.
        Returns:
            성공/실패 상관없이 출력과 exit_code를 담은 ToolResult.
        """
        command = input_data["command"]
        # timeout이 입력에 있으면 그 값, 없으면 기본 120초를 쓴다.
        timeout = input_data.get("timeout", 120)

        # 명령어를 어느 디렉토리에서 실행할지: 실행 맥락의 현재 디렉토리를 그대로 사용.
        cwd = context.cwd

        logger.info("Bash: %s (cwd=%s, timeout=%ds)", command, cwd, timeout)

        try:
            # subprocess.run은 끝날 때까지 블로킹된다. 그대로 await하면
            # 이벤트 루프 전체가 멈추므로, to_thread로 워커 스레드에 떠넘긴다.
            result = await asyncio.to_thread(_run_command, command, cwd, timeout)
        except subprocess.TimeoutExpired:
            # 제한 시간 안에 끝나지 못한 경우 → 타임아웃 에러로 보고.
            return ToolResult.error(
                f"명령어가 {timeout}초 타임아웃을 초과했습니다.",
                command=command,
                timeout=timeout,
            )
        except FileNotFoundError:
            # cwd 경로 자체가 존재하지 않을 때 발생.
            return ToolResult.error(
                f"작업 디렉토리를 찾을 수 없습니다: {cwd}",
                command=command,
            )
        except OSError as e:
            # 그 밖의 운영체제 수준 실행 실패(권한·리소스 등)를 포괄 처리.
            return ToolResult.error(
                f"명령어 실행에 실패했습니다: {e}",
                command=command,
            )

        # ── 여기부터는 명령어가 (성공이든 실패든) 끝까지 실행된 경우 ──
        # 종료 코드 추출. 0이면 정상, 그 외는 명령어 자체의 실패를 뜻한다.
        exit_code = result.returncode
        # 나열 명령이면 노이즈 경로를 먼저 걸러내고 작은 상한을 적용한다(P0-a).
        # 그 외 일반 명령은 기존대로 뒷부분(최근 로그)을 보존한다.
        raw_stdout = result.stdout or ""
        if _first_token(command) in _LISTING_COMMANDS:
            stdout = _truncate_output(_prune_listing_output(raw_stdout), _LISTING_MAX_SIZE)
        else:
            stdout = _truncate_output(raw_stdout)
        stderr = _truncate_output(result.stderr or "")

        # 모델에게 보여줄 최종 텍스트를 부분별로 조립한다.
        parts: list[str] = []
        if stdout:
            parts.append(stdout)
        if stderr:
            # 에러 출력은 표준출력과 구분되도록 "STDERR:" 머리표를 붙인다.
            parts.append(f"STDERR:\n{stderr}")
        # 종료 코드는 항상 마지막 줄에 명시해 성공/실패를 분명히 한다.
        parts.append(f"Exit code: {exit_code}")

        output_text = "\n".join(parts)

        logger.debug(
            "Bash exit=%d, stdout=%d chars, stderr=%d chars", exit_code, len(stdout), len(stderr)
        )

        # 참고: exit_code가 0이 아니어도 "도구 자체는 정상 동작"한 것이므로
        # ToolResult.error가 아니라 success로 감싼다. 명령어의 실패 여부는
        # 종료 코드로 전달하고, 판단은 모델에게 맡긴다.
        if exit_code != 0:
            return ToolResult.success(
                output_text,
                exit_code=exit_code,
                command=command,
            )

        return ToolResult.success(
            output_text,
            exit_code=exit_code,
            command=command,
        )

    # ═══ 7. UI Hints(UI 표시 힌트) ═══
    # 터미널/웹 UI가 진행 상황과 입력 요약을 보여줄 때 쓰는 보조 메서드.

    def get_progress_label(self, input_data: dict[str, Any]) -> str:
        """
        실행 중 UI에 띄울 짧은 진행 라벨을 만든다.

        사람이 쓴 description이 있으면 그것을 우선 쓰고, 없으면 명령어
        자체를 보여준다. 다만 명령어가 너무 길면 앞 57자만 잘라 "..."을
        붙여 한 줄에 깔끔히 들어가게 한다.
        """
        desc = input_data.get("description", "")
        if desc:
            return desc
        command = input_data.get("command", "")
        # 명령어가 60자를 넘으면 앞부분만 잘라서 표시(뒤에 생략 표시 붙임).
        if len(command) > 60:
            return command[:57] + "..."
        return command

    def get_input_summary(self, input_data: dict[str, Any]) -> str:
        """로그·목록 표시용으로 입력의 핵심(=명령어 문자열)만 돌려준다."""
        return input_data.get("command", "")


# ─────────────────────────────────────────────
# 유틸리티 함수 — call()에서 떼어낸 순수 헬퍼들
# ─────────────────────────────────────────────
def _run_command(command: str, cwd: str, timeout: int) -> subprocess.CompletedProcess:
    """
    실제로 subprocess.run을 호출해 셸 명령어를 실행하는 저수준 헬퍼.

    이 함수는 명령어가 끝날 때까지 스레드를 붙잡는 "블로킹" 함수다.
    그래서 call()에서 직접 부르지 않고 반드시 asyncio.to_thread를 통해
    워커 스레드에서 실행해 이벤트 루프가 멈추지 않도록 한다.

    Args:
        command: 실행할 셸 명령어 문자열.
        cwd: 명령어를 실행할 작업 디렉토리.
        timeout: 초 단위 제한 시간. 초과 시 TimeoutExpired가 발생한다.
    Returns:
        종료 코드·stdout·stderr를 담은 CompletedProcess 객체.
    """
    # 현재 프로세스의 환경 변수를 복사해 자식 셸에 물려준다(원본은 건드리지 않음).
    env = os.environ.copy()
    # 터미널 종류를 "dumb"로 강제해 ANSI 색상 코드 등 제어문자가 출력에
    # 섞이지 않게 한다(로그·모델 입력을 깨끗하게 유지).
    env["TERM"] = "dumb"  # 색상 코드 비활성화

    # shell=True로 셸을 거쳐 실행한다. 임의 명령 실행이라 보안 린터(bandit)가
    # 경고(S602)를 내지만, Bash 도구는 셸 실행이 곧 목적이므로 의도된 동작이다.
    return subprocess.run(  # noqa: S602 — Bash 도구는 shell=True가 의도된 동작
        command,
        shell=True,
        cwd=cwd,
        capture_output=True,  # stdout/stderr를 문자열로 캡처
        text=True,  # 바이트가 아닌 문자열(str)로 디코딩해 받음
        timeout=timeout,
        env=env,
    )


def _truncate_output(output: str, max_size: int = _MAX_OUTPUT_SIZE) -> str:
    """
    출력 문자열이 너무 길면 앞부분을 버리고 뒷부분만 남긴다.

    로그·명령 출력은 대개 마지막 부분(에러 메시지·최종 결과)이 더 중요하다.
    그래서 상한(max_size)을 넘으면 앞쪽을 잘라내고 얼마나 지웠는지 안내 문구를
    붙여 뒷부분만 보존한다. 상한 이하면 그대로 돌려준다.

    Args:
        output: 원본 출력 문자열.
        max_size: 보존할 최대 글자 수. 나열 명령은 작은 값을 넘긴다(P0-a).
    Returns:
        상한 이하 원본, 또는 앞부분을 생략 안내로 대체한 잘린 문자열.
    """
    if len(output) <= max_size:
        return output
    # 잘라낸(=제거한) 앞부분의 글자 수를 계산해 안내에 표시한다.
    removed = len(output) - max_size
    # 최근 max_size 글자만 남기고, 맨 앞에 생략 사실을 알린다.
    return f"... (앞부분 생략, {removed}자 제거)\n{output[-max_size:]}"
