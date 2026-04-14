"""
Docker 도구 모음 — Docker 컨테이너 빌드 및 실행.

2개의 Docker 도구를 제공한다:
  - DockerBuild: Docker 이미지 빌드 (docker build)
  - DockerRun: Docker 컨테이너 실행 (docker run)

모두 requires_confirmation=True로 사용자 확인이 필요하다.
에어갭 환경이므로 Docker Hub 등 외부 레지스트리 접근은 불가하며,
로컬 이미지 또는 사전 준비된 이미지만 사용해야 한다.
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

logger = logging.getLogger("nexus.tools.docker")

# 빌드 타임아웃 (기본 5분)
_BUILD_TIMEOUT = 300

# 실행 타임아웃 (기본 2분)
_RUN_TIMEOUT = 120

# 출력 최대 크기
_MAX_OUTPUT_SIZE = 50_000


def _run_docker(args: list[str], cwd: str, timeout: int) -> subprocess.CompletedProcess:
    """
    docker 명령어를 실행하는 공통 함수.
    블로킹 함수이므로 asyncio.to_thread에서 호출해야 한다.

    Args:
        args: docker 서브 명령어와 인자 목록 (예: ["build", "-t", "myimg", "."])
        cwd: 작업 디렉토리
        timeout: 실행 타임아웃 (초)

    Returns:
        subprocess.CompletedProcess: 실행 결과
    """
    env = os.environ.copy()
    return subprocess.run(  # noqa: S603 — Docker 도구는 docker 명령 실행이 의도된 동작
        ["docker", *args],  # noqa: S607 — docker는 PATH에서 찾는 것이 정상
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
    )


def _truncate(text: str) -> str:
    """출력이 너무 길면 뒷부분만 보존한다."""
    if len(text) <= _MAX_OUTPUT_SIZE:
        return text
    removed = len(text) - _MAX_OUTPUT_SIZE
    return f"... (앞부분 생략, {removed}자 제거)\n{text[-_MAX_OUTPUT_SIZE:]}"


async def _run_docker_async(
    args: list[str],
    cwd: str,
    timeout: int,  # noqa: ASYNC109
) -> ToolResult:
    """
    docker 명령어를 비동기로 실행하고 ToolResult를 반환하는 공통 헬퍼.
    """
    try:
        result = await asyncio.to_thread(_run_docker, args, cwd, timeout)
    except subprocess.TimeoutExpired:
        return ToolResult.error(f"docker 명령어가 {timeout}초 타임아웃을 초과했습니다.")
    except FileNotFoundError:
        return ToolResult.error("docker를 찾을 수 없습니다. Docker가 설치되어 있는지 확인하세요.")
    except OSError as e:
        return ToolResult.error(f"docker 명령어 실행 실패: {e}")

    stdout = _truncate(result.stdout or "")
    stderr = _truncate(result.stderr or "")

    parts: list[str] = []
    if stdout:
        parts.append(stdout)
    if stderr:
        parts.append(f"STDERR:\n{stderr}")
    parts.append(f"Exit code: {result.returncode}")

    output = "\n".join(parts)

    if result.returncode != 0:
        return ToolResult.error(output, exit_code=result.returncode)

    return ToolResult.success(output, exit_code=result.returncode)


# ─────────────────────────────────────────────
# DockerBuildTool — Docker 이미지 빌드
# ─────────────────────────────────────────────
class DockerBuildTool(BaseTool):
    """
    Docker 이미지를 빌드하는 도구.
    Dockerfile을 사용하여 이미지를 생성한다.
    빌드는 시스템 리소스를 많이 사용하므로 사용자 확인이 필요하다.
    에어갭 환경: 베이스 이미지는 사전 다운로드/로컬 캐시된 것만 사용 가능.
    """

    # ═══ 1. Identity ═══

    @property
    def name(self) -> str:
        return "DockerBuild"

    @property
    def description(self) -> str:
        return (
            "Docker 이미지를 빌드합니다. "
            "Dockerfile이 있는 디렉토리를 지정하고 태그를 붙여 빌드합니다. "
            "에어갭 환경이므로 베이스 이미지는 로컬에 사전 준비되어야 합니다."
        )

    @property
    def group(self) -> str:
        return "docker"

    # ═══ 2. Schema ═══

    @property
    def input_schema(self) -> dict[str, Any]:
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

    # ═══ 3. Behavior Flags ═══

    @property
    def requires_confirmation(self) -> bool:
        return True

    @property
    def timeout_seconds(self) -> float:
        """빌드는 시간이 오래 걸릴 수 있다."""
        return float(_BUILD_TIMEOUT)

    # ═══ 5. Lifecycle ═══

    def validate_input(self, input_data: dict[str, Any]) -> str | None:
        """tag가 비어 있는지 검증한다."""
        tag = input_data.get("tag", "")
        if not tag or not tag.strip():
            return "tag는 비어 있을 수 없습니다."
        return None

    async def check_permissions(
        self, input_data: dict[str, Any], context: ToolUseContext
    ) -> PermissionResult:
        """Docker 빌드는 사용자에게 확인을 요청한다."""
        tag = input_data.get("tag", "")
        context_path = input_data.get("context_path", ".")
        return PermissionResult(
            behavior=PermissionBehavior.ASK,
            message=f"Docker build: {tag} (context: {context_path})",
        )

    async def call(self, input_data: dict[str, Any], context: ToolUseContext) -> ToolResult:
        """
        docker build 명령어를 실행한다.

        처리 순서:
          1. 빌드 인자 구성 (태그, Dockerfile, build-args 등)
          2. subprocess로 docker build 실행
          3. 결과 반환
        """
        tag = input_data["tag"]
        context_path = input_data.get("context_path", ".")
        dockerfile = input_data.get("dockerfile")
        build_args = input_data.get("build_args", {})
        no_cache = input_data.get("no_cache", False)

        args = ["build", "-t", tag]

        # Dockerfile 지정
        if dockerfile:
            args.extend(["-f", dockerfile])

        # 캐시 비활성화
        if no_cache:
            args.append("--no-cache")

        # 빌드 인자 추가
        for key, value in build_args.items():
            args.extend(["--build-arg", f"{key}={value}"])

        # 빌드 컨텍스트 경로 (마지막 인자)
        args.append(context_path)

        logger.info("DockerBuild: docker %s (cwd=%s)", " ".join(args), context.cwd)
        return await _run_docker_async(args, context.cwd, _BUILD_TIMEOUT)

    # ═══ 7. UI Hints ═══

    def get_progress_label(self, input_data: dict[str, Any]) -> str:
        tag = input_data.get("tag", "")
        return f"Building {tag}..."

    def get_input_summary(self, input_data: dict[str, Any]) -> str:
        return input_data.get("tag", "")


# ─────────────────────────────────────────────
# DockerRunTool — Docker 컨테이너 실행
# ─────────────────────────────────────────────
class DockerRunTool(BaseTool):
    """
    Docker 컨테이너를 실행하는 도구.
    지정한 이미지로 컨테이너를 생성하고 명령어를 실행한다.
    시스템에 영향을 줄 수 있으므로 사용자 확인이 필요하다.
    에어갭 환경: 로컬 이미지만 사용 가능.
    """

    # ═══ 1. Identity ═══

    @property
    def name(self) -> str:
        return "DockerRun"

    @property
    def description(self) -> str:
        return (
            "Docker 컨테이너를 실행합니다. "
            "이미지와 명령어를 지정하여 실행하며, "
            "볼륨 마운트, 환경 변수, 포트 매핑을 지원합니다."
        )

    @property
    def group(self) -> str:
        return "docker"

    # ═══ 2. Schema ═══

    @property
    def input_schema(self) -> dict[str, Any]:
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

    # ═══ 3. Behavior Flags ═══

    @property
    def requires_confirmation(self) -> bool:
        return True

    @property
    def is_destructive(self) -> bool:
        """컨테이너 실행은 시스템 리소스를 사용한다."""
        return True

    # ═══ 5. Lifecycle ═══

    def validate_input(self, input_data: dict[str, Any]) -> str | None:
        """image가 비어 있는지 검증한다."""
        image = input_data.get("image", "")
        if not image or not image.strip():
            return "image는 비어 있을 수 없습니다."
        return None

    async def check_permissions(
        self, input_data: dict[str, Any], context: ToolUseContext
    ) -> PermissionResult:
        """Docker 실행은 사용자에게 확인을 요청한다."""
        image = input_data.get("image", "")
        command = input_data.get("command", "")
        msg = f"Docker run: {image}"
        if command:
            msg += f" — {command}"
        return PermissionResult(
            behavior=PermissionBehavior.ASK,
            message=msg,
        )

    async def call(self, input_data: dict[str, Any], context: ToolUseContext) -> ToolResult:
        """
        docker run 명령어를 실행한다.

        처리 순서:
          1. 실행 옵션 구성 (볼륨, 환경변수, 포트 등)
          2. 이미지와 명령어 지정
          3. subprocess로 docker run 실행
          4. 결과 반환
        """
        image = input_data["image"]
        command = input_data.get("command")
        volumes = input_data.get("volumes", [])
        env_vars = input_data.get("env_vars", {})
        ports = input_data.get("ports", [])
        remove = input_data.get("remove", True)
        container_name = input_data.get("name")

        args = ["run"]

        # 자동 삭제 옵션
        if remove:
            args.append("--rm")

        # 컨테이너 이름
        if container_name:
            args.extend(["--name", container_name])

        # 볼륨 마운트
        for vol in volumes:
            args.extend(["-v", vol])

        # 환경 변수
        for key, value in env_vars.items():
            args.extend(["-e", f"{key}={value}"])

        # 포트 매핑
        for port in ports:
            args.extend(["-p", port])

        # 이미지 지정
        args.append(image)

        # 컨테이너 내 명령어
        if command:
            args.extend(command.split())

        logger.info("DockerRun: docker %s (cwd=%s)", " ".join(args), context.cwd)
        return await _run_docker_async(args, context.cwd, _RUN_TIMEOUT)

    # ═══ 7. UI Hints ═══

    def get_progress_label(self, input_data: dict[str, Any]) -> str:
        image = input_data.get("image", "")
        return f"Running {image}..."

    def get_input_summary(self, input_data: dict[str, Any]) -> str:
        return input_data.get("image", "")
