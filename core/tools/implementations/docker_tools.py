"""
Docker 도구 모음 — Docker 이미지 빌드 및 컨테이너 실행 도구.

[이 파일이 하는 일]
Nexus의 도구 시스템(core/tools)에 등록되는 두 개의 Docker 관련 도구를 정의한다.
모델(에이전트)이 도구 호출로 Docker 작업을 요청하면, 이 파일의 도구들이
호스트의 `docker` CLI를 subprocess로 실행하고 그 결과를 ToolResult로 돌려준다.

[제공하는 도구 2개]
  - DockerBuildTool (name="DockerBuild"): Dockerfile로 이미지 빌드 (docker build)
  - DockerRunTool   (name="DockerRun")  : 이미지로 컨테이너 실행 (docker run)

[공통 헬퍼]
  - _run_docker()       : docker CLI를 실제로 실행하는 블로킹 함수 (subprocess.run)
  - _run_docker_async() : 위 블로킹 호출을 asyncio 스레드로 감싸 ToolResult로 변환
  - _truncate()         : 출력이 너무 길면 뒷부분만 남겨 컨텍스트 폭주를 방지

[의존/상속]
두 도구 모두 core.tools.base.BaseTool ABC를 상속하고, 권한 결과는
PermissionResult / PermissionBehavior, 실행 결과는 ToolResult로 표준화한다.
BaseTool 계약에 따라 name/description/input_schema/behavior flag/validate_input/
check_permissions/call 등을 구현한다.

[안전·정책]
두 도구 모두 requires_confirmation=True 라서 실행 전 사용자 확인(ASK)이 필요하다.
에어갭(폐쇄망) 환경이라 Docker Hub 같은 외부 레지스트리 접근은 불가하며,
로컬에 미리 준비(캐시)된 베이스 이미지·실행 이미지만 사용해야 한다.

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

# 이 모듈 전용 로거. 계층형 이름("nexus.tools.docker")으로 두면 로깅 설정에서
# Docker 도구 로그만 따로 켜고 끄거나 레벨을 조정하기 쉽다.
logger = logging.getLogger("nexus.tools.docker")

# 이미지 빌드 타임아웃(초). 빌드는 레이어가 많으면 오래 걸리므로 기본 5분으로 넉넉히 둔다.
_BUILD_TIMEOUT = 300

# 컨테이너 실행 타임아웃(초). 실행은 보통 빌드보다 짧으므로 기본 2분으로 둔다.
_RUN_TIMEOUT = 120

# docker 출력(stdout/stderr) 최대 보존 길이(문자 수). 이 값을 넘으면 _truncate()가
# 뒷부분만 남긴다. 컨텍스트 창이 로그로 가득 차는 것을 막기 위한 안전장치.
_MAX_OUTPUT_SIZE = 50_000


def _run_docker(args: list[str], cwd: str, timeout: int) -> subprocess.CompletedProcess:
    """
    docker CLI를 실제로 호출하는 공통 함수 (DockerBuild/DockerRun 양쪽에서 재사용).

    subprocess.run은 프로세스가 끝날 때까지 스레드를 붙잡는 '블로킹' 호출이다.
    따라서 asyncio 이벤트 루프를 막지 않으려면 이 함수를 직접 await하지 말고,
    반드시 asyncio.to_thread(_run_docker, ...) 형태로 별도 스레드에서 실행해야 한다.
    (실제로 _run_docker_async()가 그렇게 감싸서 호출한다.)

    Args:
        args: docker 뒤에 붙일 서브 명령어와 인자 목록.
              예: ["build", "-t", "myimg", "."] → 최종적으로 `docker build -t myimg .`
        cwd: 명령을 실행할 작업 디렉토리(빌드 컨텍스트의 기준 경로).
        timeout: 초 단위 타임아웃. 초과 시 subprocess.TimeoutExpired가 발생한다.

    Returns:
        subprocess.CompletedProcess: returncode/stdout/stderr를 담은 실행 결과 객체.
    """
    # 현재 프로세스의 환경변수를 복사해 자식(docker)에게 그대로 넘긴다.
    # 원본 os.environ을 직접 넘기지 않고 사본을 쓰는 것이 안전하다.
    env = os.environ.copy()
    # capture_output=True + text=True → stdout/stderr를 문자열로 모아서 받는다.
    # noqa 주석은 보안 린터(Bandit) 경고를 '의도된 동작'으로 명시적으로 무시하는 것.
    return subprocess.run(  # noqa: S603 — Docker 도구는 docker 명령 실행이 의도된 동작
        ["docker", *args],  # noqa: S607 — docker는 PATH에서 찾는 것이 정상
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
    )


def _truncate(text: str) -> str:
    """
    docker 출력이 너무 길면 '뒷부분'만 남기고 앞부분을 잘라낸다.

    빌드/실행 로그는 마지막 줄(에러 메시지, 최종 결과)이 가장 중요하기 때문에
    앞이 아니라 뒤를 보존한다. 이렇게 해야 컨텍스트 창을 아끼면서도
    실패 원인을 담은 마지막 로그를 놓치지 않는다.
    """
    # 한도 이내면 그대로 반환(가장 흔한 경로).
    if len(text) <= _MAX_OUTPUT_SIZE:
        return text
    # 얼마나 잘려나갔는지 계산해 사용자에게 안내 문구로 알려준다.
    removed = len(text) - _MAX_OUTPUT_SIZE
    # 슬라이스 [-_MAX_OUTPUT_SIZE:] 로 마지막 부분만 남긴다.
    return f"... (앞부분 생략, {removed}자 제거)\n{text[-_MAX_OUTPUT_SIZE:]}"


async def _run_docker_async(
    args: list[str],
    cwd: str,
    timeout: int,  # noqa: ASYNC109
) -> ToolResult:
    """
    블로킹 _run_docker()를 비동기로 감싸 실행하고, 그 결과를 표준 ToolResult로 변환한다.

    DockerBuildTool.call / DockerRunTool.call 이 공통으로 호출하는 헬퍼다.
    핵심 흐름:
      1) asyncio.to_thread로 docker CLI를 별도 스레드에서 실행(이벤트 루프 비차단).
      2) 실행 중 발생 가능한 예외를 종류별로 잡아 사람이 읽을 수 있는 에러로 변환.
      3) stdout/stderr를 _truncate로 자르고, 종료 코드까지 하나의 문자열로 합친다.
      4) 종료 코드가 0이 아니면 ToolResult.error, 0이면 ToolResult.success 로 반환.

    Args / Returns:
        args/cwd/timeout 은 _run_docker와 동일. 반환은 항상 ToolResult(성공/실패 래핑).

    # noqa: ASYNC109 — 타임아웃을 async 함수 인자로 직접 받는 패턴에 대한 린터 경고를
    # 의도적으로 무시한다(하위 to_thread가 실제 timeout을 처리하므로 문제 없음).
    """
    # 1) 블로킹 docker 실행을 스레드로 넘긴다. 여기서 나올 수 있는 예외를 세분화해 처리.
    try:
        result = await asyncio.to_thread(_run_docker, args, cwd, timeout)
    except subprocess.TimeoutExpired:
        # 제한 시간을 넘겼을 때(빌드가 너무 오래 걸리는 등).
        return ToolResult.error(f"docker 명령어가 {timeout}초 타임아웃을 초과했습니다.")
    except FileNotFoundError:
        # docker 실행 파일 자체가 PATH에 없을 때(미설치 등).
        return ToolResult.error("docker를 찾을 수 없습니다. Docker가 설치되어 있는지 확인하세요.")
    except OSError as e:
        # 그 외 OS 수준의 실행 실패(권한, 리소스 부족 등).
        return ToolResult.error(f"docker 명령어 실행 실패: {e}")

    # 2) 출력이 None일 수 있으므로 빈 문자열로 보정한 뒤 길이 제한을 적용한다.
    stdout = _truncate(result.stdout or "")
    stderr = _truncate(result.stderr or "")

    # 3) 사람이 읽기 좋은 순서로 조립: stdout → STDERR → 종료 코드.
    parts: list[str] = []
    if stdout:
        parts.append(stdout)
    if stderr:
        parts.append(f"STDERR:\n{stderr}")
    parts.append(f"Exit code: {result.returncode}")

    output = "\n".join(parts)

    # 4) 종료 코드로 성공/실패를 판정한다. 0이 아니면 실패로 래핑.
    if result.returncode != 0:
        return ToolResult.error(output, exit_code=result.returncode)

    return ToolResult.success(output, exit_code=result.returncode)


# ─────────────────────────────────────────────
# DockerBuildTool — Docker 이미지 빌드
# ─────────────────────────────────────────────
class DockerBuildTool(BaseTool):
    """
    Dockerfile을 이용해 Docker 이미지를 빌드하는 도구 (docker build 래퍼).

    [역할] 모델이 "이 디렉토리로 이미지를 빌드해줘"라고 도구를 호출하면,
    입력값(tag/context_path/dockerfile/build_args/no_cache)을 docker build 인자로
    바꿔 실행하고 결과를 ToolResult로 돌려준다.

    [안전] 빌드는 CPU/디스크 등 시스템 리소스를 많이 쓰므로
    requires_confirmation=True 로 두어 실행 전 사용자 확인을 받는다.

    [에어갭] FROM으로 지정하는 베이스 이미지는 외부에서 pull 할 수 없으므로,
    반드시 사전 다운로드/로컬 캐시된 이미지만 사용해야 빌드가 성공한다.

    BaseTool 계약에 따라 Identity/Schema/Behavior/Lifecycle/UI Hints 섹션을 구현한다.
    """

    # ═══ 1. Identity — 도구를 식별하는 이름/설명/그룹 ═══

    @property
    def name(self) -> str:
        # 모델이 도구를 호출할 때 사용하는 고유 이름. 레지스트리 조회 키이기도 하다.
        return "DockerBuild"

    @property
    def description(self) -> str:
        # 모델에게 노출되는 설명문. 언제/어떻게 쓰는 도구인지 모델이 판단하는 근거가 된다.
        return (
            "Docker 이미지를 빌드합니다. "
            "Dockerfile이 있는 디렉토리를 지정하고 태그를 붙여 빌드합니다. "
            "에어갭 환경이므로 베이스 이미지는 로컬에 사전 준비되어야 합니다."
        )

    @property
    def group(self) -> str:
        # 도구를 묶는 그룹명. UI 분류/권한 정책 등에서 "docker" 계열로 함께 다루기 위함.
        return "docker"

    # ═══ 2. Schema — 입력 파라미터를 정의하는 JSON Schema ═══

    @property
    def input_schema(self) -> dict[str, Any]:
        # 모델이 이 도구를 호출할 때 넘겨야 하는 인자 구조. tag만 필수, 나머지는 선택.
        return {
            "type": "object",
            "properties": {
                "tag": {
                    "type": "string",
                    "description": "이미지 태그 (예: myapp:latest)",
                },
                "context_path": {
                    "type": "string",
                    "description": "빌드 컨텍스트 경로 (기본: 현재 디렉토리)",
                    "default": ".",
                },
                "dockerfile": {
                    "type": "string",
                    "description": "Dockerfile 경로 (기본: context_path/Dockerfile)",
                },
                "build_args": {
                    "type": "object",
                    "description": "빌드 인자 (--build-arg KEY=VALUE)",
                    "additionalProperties": {"type": "string"},
                },
                "no_cache": {
                    "type": "boolean",
                    "description": "캐시 없이 빌드 (--no-cache)",
                    "default": False,
                },
            },
            "required": ["tag"],
        }

    # ═══ 3. Behavior Flags — 도구의 동작 특성(권한/타임아웃 등) ═══

    @property
    def requires_confirmation(self) -> bool:
        # True면 실행 전에 사용자 확인(ASK)이 필요하다. 빌드는 무거운 작업이라 확인 필수.
        return True

    @property
    def timeout_seconds(self) -> float:
        """빌드는 시간이 오래 걸릴 수 있으므로 기본값(_BUILD_TIMEOUT=5분)을 그대로 쓴다."""
        return float(_BUILD_TIMEOUT)

    # ═══ 5. Lifecycle — 입력 검증 → 권한 확인 → 실행 순서로 호출된다 ═══

    def validate_input(self, input_data: dict[str, Any]) -> str | None:
        """
        실행 전 입력값을 미리 검증한다. 문제가 있으면 에러 메시지(str)를,
        정상이면 None을 반환한다. 여기서는 필수값 tag가 비었는지만 확인한다.
        """
        tag = input_data.get("tag", "")
        # 공백만 있는 경우까지 걸러내기 위해 strip() 결과로 판단한다.
        if not tag or not tag.strip():
            return "tag는 비어 있을 수 없습니다."
        return None

    async def check_permissions(
        self, input_data: dict[str, Any], context: ToolUseContext
    ) -> PermissionResult:
        """
        권한 파이프라인에서 호출되는 훅. 여기서 ASK를 반환하면
        사용자에게 "이 빌드를 실행할까요?" 확인을 띄운다(빌드 태그/컨텍스트 표시).
        """
        tag = input_data.get("tag", "")
        context_path = input_data.get("context_path", ".")
        return PermissionResult(
            behavior=PermissionBehavior.ASK,
            message=f"Docker build: {tag} (context: {context_path})",
        )

    async def call(self, input_data: dict[str, Any], context: ToolUseContext) -> ToolResult:
        """
        실제로 docker build 명령을 조립해 실행하는 진입점.

        처리 순서:
          1. 입력값을 꺼내 docker build 인자 리스트(args)를 순서대로 구성한다.
             (docker build 문법상 컨텍스트 경로는 반드시 '마지막' 인자여야 한다.)
          2. _run_docker_async로 넘겨 별도 스레드에서 빌드를 실행한다.
          3. 실행 결과를 ToolResult로 반환한다.

        Args:
            input_data: input_schema에 정의된 사용자 입력.
            context: 실행 컨텍스트. 여기서는 작업 디렉토리(context.cwd)를 사용한다.
        Returns:
            ToolResult: 빌드 성공/실패 및 로그를 담은 결과.
        """
        # 필수값 tag는 [] 인덱싱으로 꺼낸다(validate_input에서 존재를 이미 보장).
        tag = input_data["tag"]
        context_path = input_data.get("context_path", ".")
        dockerfile = input_data.get("dockerfile")
        build_args = input_data.get("build_args", {})
        no_cache = input_data.get("no_cache", False)

        # 기본 골격: `docker build -t <tag>`
        args = ["build", "-t", tag]

        # Dockerfile 경로를 명시한 경우 -f 옵션으로 지정.
        if dockerfile:
            args.extend(["-f", dockerfile])

        # 캐시를 쓰지 않고 처음부터 다시 빌드하려면 --no-cache.
        if no_cache:
            args.append("--no-cache")

        # 빌드 인자들을 --build-arg KEY=VALUE 형태로 하나씩 추가.
        for key, value in build_args.items():
            args.extend(["--build-arg", f"{key}={value}"])

        # 빌드 컨텍스트 경로는 docker build 문법상 반드시 맨 끝에 온다.
        args.append(context_path)

        # 실행 전 어떤 명령이 나가는지 로그로 남겨 추적/디버깅에 활용.
        logger.info("DockerBuild: docker %s (cwd=%s)", " ".join(args), context.cwd)
        return await _run_docker_async(args, context.cwd, _BUILD_TIMEOUT)

    # ═══ 7. UI Hints — 진행 상황/요약을 UI에 보여주기 위한 문자열 ═══

    def get_progress_label(self, input_data: dict[str, Any]) -> str:
        # 실행 중 UI에 표시할 진행 라벨(예: "Building myapp:latest...").
        tag = input_data.get("tag", "")
        return f"Building {tag}..."

    def get_input_summary(self, input_data: dict[str, Any]) -> str:
        # 도구 호출 요약으로 태그만 간단히 보여준다.
        return input_data.get("tag", "")


# ─────────────────────────────────────────────
# DockerRunTool — Docker 컨테이너 실행
# ─────────────────────────────────────────────
class DockerRunTool(BaseTool):
    """
    지정한 이미지로 Docker 컨테이너를 실행하는 도구 (docker run 래퍼).

    [역할] 이미지 이름과 (선택적으로) 실행 명령어를 받아 컨테이너를 띄운다.
    볼륨 마운트(-v), 환경변수(-e), 포트 매핑(-p), 자동 삭제(--rm),
    컨테이너 이름(--name) 옵션을 지원한다.

    [안전] 컨테이너 실행은 호스트 시스템에 영향을 줄 수 있어
    requires_confirmation=True + is_destructive=True 로 표시하고 실행 전 확인을 받는다.

    [에어갭] 외부 레지스트리에서 pull 할 수 없으므로 로컬 이미지만 실행 가능하다.
    """

    # ═══ 1. Identity — 도구를 식별하는 이름/설명/그룹 ═══

    @property
    def name(self) -> str:
        # 모델이 도구를 호출할 때 사용하는 고유 이름(레지스트리 조회 키).
        return "DockerRun"

    @property
    def description(self) -> str:
        # 모델에게 노출되는 설명문. 지원 옵션(볼륨/환경변수/포트)을 함께 안내한다.
        return (
            "Docker 컨테이너를 실행합니다. "
            "이미지와 명령어를 지정하여 실행하며, "
            "볼륨 마운트, 환경 변수, 포트 매핑을 지원합니다."
        )

    @property
    def group(self) -> str:
        # DockerBuild와 동일한 "docker" 그룹으로 묶어 함께 다룬다.
        return "docker"

    # ═══ 2. Schema — 입력 파라미터를 정의하는 JSON Schema ═══

    @property
    def input_schema(self) -> dict[str, Any]:
        # image만 필수. 나머지(command/volumes/env_vars/ports/remove/name)는 선택.
        return {
            "type": "object",
            "properties": {
                "image": {
                    "type": "string",
                    "description": "실행할 Docker 이미지 (예: myapp:latest)",
                },
                "command": {
                    "type": "string",
                    "description": "컨테이너 내에서 실행할 명령어",
                },
                "volumes": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "볼륨 마운트 (-v host:container)",
                },
                "env_vars": {
                    "type": "object",
                    "description": "환경 변수 (-e KEY=VALUE)",
                    "additionalProperties": {"type": "string"},
                },
                "ports": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "포트 매핑 (-p host:container)",
                },
                "remove": {
                    "type": "boolean",
                    "description": "실행 완료 후 컨테이너 자동 삭제 (--rm)",
                    "default": True,
                },
                "name": {
                    "type": "string",
                    "description": "컨테이너 이름 (--name)",
                },
            },
            "required": ["image"],
        }

    # ═══ 3. Behavior Flags — 도구의 동작 특성(권한/파괴성 등) ═══

    @property
    def requires_confirmation(self) -> bool:
        # 컨테이너 실행은 사이드이펙트가 있으므로 실행 전 사용자 확인이 필요하다.
        return True

    @property
    def is_destructive(self) -> bool:
        """
        컨테이너 실행은 시스템 리소스를 소비하고 상태를 바꿀 수 있어 '파괴적'으로 분류한다.
        이 플래그는 권한/UI 계층에서 더 신중히 다루도록(경고 강조 등) 하는 신호다.
        """
        return True

    # ═══ 5. Lifecycle — 입력 검증 → 권한 확인 → 실행 순서로 호출된다 ═══

    def validate_input(self, input_data: dict[str, Any]) -> str | None:
        """
        실행 전 필수값 image가 비었는지 검증한다.
        문제가 있으면 에러 메시지를, 정상이면 None을 반환한다.
        """
        image = input_data.get("image", "")
        # 공백만 있는 경우도 무효로 간주한다.
        if not image or not image.strip():
            return "image는 비어 있을 수 없습니다."
        return None

    async def check_permissions(
        self, input_data: dict[str, Any], context: ToolUseContext
    ) -> PermissionResult:
        """
        권한 훅. ASK를 반환해 사용자 확인을 띄운다.
        확인 창에는 실행 이미지와(있다면) 실행 명령어를 함께 표시해준다.
        """
        image = input_data.get("image", "")
        command = input_data.get("command", "")
        msg = f"Docker run: {image}"
        # 실행 명령어가 있으면 메시지에 덧붙여 사용자가 무엇이 돌아갈지 알 수 있게 한다.
        if command:
            msg += f" — {command}"
        return PermissionResult(
            behavior=PermissionBehavior.ASK,
            message=msg,
        )

    async def call(self, input_data: dict[str, Any], context: ToolUseContext) -> ToolResult:
        """
        실제로 docker run 명령을 조립해 실행하는 진입점.

        처리 순서:
          1. 실행 옵션(--rm, --name, -v, -e, -p)을 순서대로 args에 쌓는다.
          2. docker run 문법상 이미지 이름을 옵션들 뒤, 컨테이너 명령어 앞에 놓는다.
          3. 컨테이너 안에서 돌릴 명령어가 있으면 공백으로 쪼개 뒤에 붙인다.
          4. _run_docker_async로 실행하고 ToolResult로 반환한다.

        주의: command.split()은 단순 공백 분리라 따옴표로 묶인 인자는 그대로 처리되지
        않는다(현 구현 그대로). 복잡한 셸 명령이 필요하면 이미지 엔트리포인트 쪽에서
        다루는 것이 안전하다.
        """
        # 필수값 image는 [] 인덱싱으로 꺼낸다(validate_input에서 존재 보장).
        image = input_data["image"]
        command = input_data.get("command")
        volumes = input_data.get("volumes", [])
        env_vars = input_data.get("env_vars", {})
        ports = input_data.get("ports", [])
        remove = input_data.get("remove", True)
        container_name = input_data.get("name")

        # 기본 골격: `docker run`
        args = ["run"]

        # 종료 후 컨테이너를 자동 정리하려면 --rm (기본 True).
        if remove:
            args.append("--rm")

        # 컨테이너에 이름을 붙이려면 --name.
        if container_name:
            args.extend(["--name", container_name])

        # 볼륨 마운트들을 -v host:container 형태로 하나씩 추가.
        for vol in volumes:
            args.extend(["-v", vol])

        # 환경변수들을 -e KEY=VALUE 형태로 추가.
        for key, value in env_vars.items():
            args.extend(["-e", f"{key}={value}"])

        # 포트 매핑들을 -p host:container 형태로 추가.
        for port in ports:
            args.extend(["-p", port])

        # 이미지 이름은 옵션 뒤 / 컨테이너 명령어 앞 위치에 온다.
        args.append(image)

        # 컨테이너 안에서 실행할 명령어가 있으면 토큰으로 나눠 이어 붙인다.
        if command:
            args.extend(command.split())

        # 실행 전 최종 명령을 로그로 남긴다(추적/디버깅용).
        logger.info("DockerRun: docker %s (cwd=%s)", " ".join(args), context.cwd)
        return await _run_docker_async(args, context.cwd, _RUN_TIMEOUT)

    # ═══ 7. UI Hints — 진행 상황/요약을 UI에 보여주기 위한 문자열 ═══

    def get_progress_label(self, input_data: dict[str, Any]) -> str:
        # 실행 중 UI에 표시할 진행 라벨(예: "Running myapp:latest...").
        image = input_data.get("image", "")
        return f"Running {image}..."

    def get_input_summary(self, input_data: dict[str, Any]) -> str:
        # 도구 호출 요약으로 이미지 이름만 간단히 보여준다.
        return input_data.get("image", "")
