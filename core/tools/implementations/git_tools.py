"""
Git 도구 모음 — Git 버전 관리 명령어 실행.

6개의 Git 도구를 제공한다:
  - GitLog: 커밋 로그 조회 (읽기 전용)
  - GitDiff: 변경 사항 비교 (읽기 전용)
  - GitStatus: 작업 트리 상태 조회 (읽기 전용)
  - GitCommit: 커밋 생성 (사용자 확인 필요)
  - GitBranch: 브랜치 목록 조회 (읽기 전용)
  - GitCheckout: 브랜치/파일 체크아웃 (사용자 확인 필요)

모든 도구는 subprocess.run으로 git 명령을 실행하며,
context.cwd를 작업 디렉토리로 사용한다.
에어갭 환경이므로 원격 저장소 접근(push/pull/fetch)은 지원하지 않는다.
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

logger = logging.getLogger("nexus.tools.git")

# git 명령 실행 타임아웃 (초)
_DEFAULT_TIMEOUT = 30

# stdout 최대 캡처 크기
_MAX_OUTPUT_SIZE = 50_000


# ─────────────────────────────────────────────
# 공통 유틸리티
# ─────────────────────────────────────────────
def _run_git(
    args: list[str],
    cwd: str,
    timeout: int = _DEFAULT_TIMEOUT,
) -> subprocess.CompletedProcess:
    """
    git 명령어를 실행하는 공통 함수.
    블로킹 함수이므로 asyncio.to_thread에서 호출해야 한다.

    Args:
        args: git 서브 명령어와 인자 목록 (예: ["log", "--oneline", "-10"])
        cwd: 작업 디렉토리 (git 저장소 경로)
        timeout: 실행 타임아웃 (초)

    Returns:
        subprocess.CompletedProcess: 실행 결과 (stdout, stderr, returncode)
    """
    env = os.environ.copy()
    # git 페이저 비활성화 (파이프로 출력할 때 멈추지 않도록)
    env["GIT_PAGER"] = "cat"
    # 색상 코드 비활성화 (순수 텍스트 출력)
    env["GIT_TERMINAL_PROMPT"] = "0"

    return subprocess.run(  # noqa: S603 — Git 도구는 git 명령 실행이 의도된 동작
        ["git", *args],  # noqa: S607 — git은 PATH에서 찾는 것이 정상
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


async def _run_git_async(
    args: list[str],
    cwd: str,
    timeout: int = _DEFAULT_TIMEOUT,  # noqa: ASYNC109
) -> ToolResult:
    """
    git 명령어를 비동기로 실행하고 ToolResult를 반환하는 공통 헬퍼.
    성공/실패에 관계없이 stdout + stderr + exit_code를 포함한다.
    """
    try:
        result = await asyncio.to_thread(_run_git, args, cwd, timeout)
    except subprocess.TimeoutExpired:
        return ToolResult.error(f"git 명령어가 {timeout}초 타임아웃을 초과했습니다.")
    except FileNotFoundError:
        return ToolResult.error("git을 찾을 수 없습니다. git이 설치되어 있는지 확인하세요.")
    except OSError as e:
        return ToolResult.error(f"git 명령어 실행 실패: {e}")

    stdout = _truncate(result.stdout or "")
    stderr = _truncate(result.stderr or "")

    # 결과 조합
    parts: list[str] = []
    if stdout:
        parts.append(stdout)
    if stderr:
        parts.append(f"STDERR:\n{stderr}")

    output = "\n".join(parts) if parts else "(출력 없음)"

    if result.returncode != 0:
        return ToolResult.error(
            f"{output}\nExit code: {result.returncode}",
            exit_code=result.returncode,
        )

    return ToolResult.success(output, exit_code=result.returncode)


# ─────────────────────────────────────────────
# GitLogTool — 커밋 로그 조회
# ─────────────────────────────────────────────
class GitLogTool(BaseTool):
    """
    Git 커밋 로그를 조회하는 도구.
    기본적으로 최근 20개 커밋을 한 줄 형식으로 보여준다.
    읽기 전용이므로 병렬 실행이 안전하다.
    """

    # ═══ 1. Identity ═══

    @property
    def name(self) -> str:
        return "GitLog"

    @property
    def description(self) -> str:
        return (
            "Git 커밋 로그를 조회합니다. "
            "최근 커밋 이력을 확인할 수 있으며, 다양한 포맷 옵션을 지원합니다."
        )

    @property
    def group(self) -> str:
        return "git"

    # ═══ 2. Schema ═══

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "max_count": {
                    "type": "integer",
                    "description": "표시할 최대 커밋 수 (기본: 20)",
                    "default": 20,
                    "minimum": 1,
                    "maximum": 500,
                },
                "oneline": {
                    "type": "boolean",
                    "description": "한 줄 형식으로 출력 (기본: true)",
                    "default": True,
                },
                "path": {
                    "type": "string",
                    "description": "특정 파일/디렉토리의 로그만 조회",
                },
                "revision": {
                    "type": "string",
                    "description": "특정 리비전 범위 (예: main..HEAD, v1.0..v2.0)",
                },
            },
            "required": [],
        }

    # ═══ 3. Behavior Flags ═══
    # 읽기 전용 — 안전하게 완화

    @property
    def is_read_only(self) -> bool:
        return True

    @property
    def is_concurrency_safe(self) -> bool:
        return True

    # ═══ 5. Lifecycle ═══

    async def check_permissions(
        self, input_data: dict[str, Any], context: ToolUseContext
    ) -> PermissionResult:
        """읽기 전용이므로 항상 허용한다."""
        return PermissionResult(behavior=PermissionBehavior.ALLOW)

    async def call(self, input_data: dict[str, Any], context: ToolUseContext) -> ToolResult:
        """
        git log 명령어를 실행한다.

        처리 순서:
          1. 옵션에 따라 git log 인자 구성
          2. subprocess로 실행
          3. 결과 반환
        """
        args = ["log"]

        # 최대 커밋 수 지정
        max_count = input_data.get("max_count", 20)
        args.append(f"--max-count={max_count}")

        # 한 줄 형식 여부
        if input_data.get("oneline", True):
            args.append("--oneline")

        # 리비전 범위가 있으면 추가
        revision = input_data.get("revision")
        if revision:
            args.append(revision)

        # 특정 경로 필터가 있으면 -- 뒤에 추가
        path = input_data.get("path")
        if path:
            args.extend(["--", path])

        logger.debug("GitLog: git %s (cwd=%s)", " ".join(args), context.cwd)
        return await _run_git_async(args, context.cwd)

    # ═══ 7. UI Hints ═══

    def get_progress_label(self, input_data: dict[str, Any]) -> str:
        return "Fetching git log..."


# ─────────────────────────────────────────────
# GitDiffTool — 변경 사항 비교
# ─────────────────────────────────────────────
class GitDiffTool(BaseTool):
    """
    Git 변경 사항을 비교하는 도구.
    작업 트리, 스테이징 영역, 커밋 간의 차이를 보여준다.
    읽기 전용이므로 병렬 실행이 안전하다.
    """

    # ═══ 1. Identity ═══

    @property
    def name(self) -> str:
        return "GitDiff"

    @property
    def description(self) -> str:
        return "Git 변경 사항을 비교합니다. 스테이징/비스테이징 변경, 커밋 간 비교를 지원합니다."

    @property
    def group(self) -> str:
        return "git"

    # ═══ 2. Schema ═══

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "staged": {
                    "type": "boolean",
                    "description": "스테이징된 변경만 보기 (--cached)",
                    "default": False,
                },
                "revision": {
                    "type": "string",
                    "description": "비교 대상 리비전 (예: HEAD~3, main..feature)",
                },
                "path": {
                    "type": "string",
                    "description": "특정 파일/디렉토리만 비교",
                },
                "stat": {
                    "type": "boolean",
                    "description": "변경 통계만 표시 (--stat)",
                    "default": False,
                },
            },
            "required": [],
        }

    # ═══ 3. Behavior Flags ═══

    @property
    def is_read_only(self) -> bool:
        return True

    @property
    def is_concurrency_safe(self) -> bool:
        return True

    # ═══ 5. Lifecycle ═══

    async def check_permissions(
        self, input_data: dict[str, Any], context: ToolUseContext
    ) -> PermissionResult:
        """읽기 전용이므로 항상 허용한다."""
        return PermissionResult(behavior=PermissionBehavior.ALLOW)

    async def call(self, input_data: dict[str, Any], context: ToolUseContext) -> ToolResult:
        """
        git diff 명령어를 실행한다.

        처리 순서:
          1. --cached, --stat 등 옵션 구성
          2. 리비전/경로 인자 추가
          3. subprocess로 실행
        """
        args = ["diff"]

        # 스테이징된 변경만 보기
        if input_data.get("staged", False):
            args.append("--cached")

        # 변경 통계만 표시
        if input_data.get("stat", False):
            args.append("--stat")

        # 리비전 범위 지정
        revision = input_data.get("revision")
        if revision:
            args.append(revision)

        # 특정 경로 필터
        path = input_data.get("path")
        if path:
            args.extend(["--", path])

        logger.debug("GitDiff: git %s (cwd=%s)", " ".join(args), context.cwd)
        return await _run_git_async(args, context.cwd)

    # ═══ 7. UI Hints ═══

    def get_progress_label(self, input_data: dict[str, Any]) -> str:
        return "Computing diff..."


# ─────────────────────────────────────────────
# GitStatusTool — 작업 트리 상태 조회
# ─────────────────────────────────────────────
class GitStatusTool(BaseTool):
    """
    Git 작업 트리 상태를 조회하는 도구.
    수정/추가/삭제된 파일 목록을 보여준다.
    읽기 전용이므로 병렬 실행이 안전하다.
    """

    # ═══ 1. Identity ═══

    @property
    def name(self) -> str:
        return "GitStatus"

    @property
    def description(self) -> str:
        return "Git 작업 트리의 현재 상태를 조회합니다. 변경/추가/삭제된 파일을 보여줍니다."

    @property
    def group(self) -> str:
        return "git"

    # ═══ 2. Schema ═══

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "short": {
                    "type": "boolean",
                    "description": "간단한 형식으로 출력 (-s)",
                    "default": False,
                },
            },
            "required": [],
        }

    # ═══ 3. Behavior Flags ═══

    @property
    def is_read_only(self) -> bool:
        return True

    @property
    def is_concurrency_safe(self) -> bool:
        return True

    # ═══ 5. Lifecycle ═══

    async def check_permissions(
        self, input_data: dict[str, Any], context: ToolUseContext
    ) -> PermissionResult:
        """읽기 전용이므로 항상 허용한다."""
        return PermissionResult(behavior=PermissionBehavior.ALLOW)

    async def call(self, input_data: dict[str, Any], context: ToolUseContext) -> ToolResult:
        """
        git status 명령어를 실행한다.
        -u 옵션을 기본 포함하여 추적되지 않는 파일도 보여준다.
        단, -uall은 사용하지 않는다 (대규모 저장소에서 메모리 이슈 방지).
        """
        args = ["status", "-u"]

        # 간단한 형식 옵션
        if input_data.get("short", False):
            args.append("-s")

        logger.debug("GitStatus: git %s (cwd=%s)", " ".join(args), context.cwd)
        return await _run_git_async(args, context.cwd)

    # ═══ 7. UI Hints ═══

    def get_progress_label(self, input_data: dict[str, Any]) -> str:
        return "Checking git status..."


# ─────────────────────────────────────────────
# GitCommitTool — 커밋 생성
# ─────────────────────────────────────────────
class GitCommitTool(BaseTool):
    """
    Git 커밋을 생성하는 도구.
    스테이징된 변경 사항을 커밋 메시지와 함께 저장한다.
    시스템 상태를 변경하므로 사용자 확인이 필요하다 (requires_confirmation=True).
    """

    # ═══ 1. Identity ═══

    @property
    def name(self) -> str:
        return "GitCommit"

    @property
    def description(self) -> str:
        return (
            "Git 커밋을 생성합니다. "
            "스테이징된 변경 사항을 지정한 메시지로 커밋합니다. "
            "커밋 전에 git add로 파일을 스테이징해야 합니다."
        )

    @property
    def group(self) -> str:
        return "git"

    # ═══ 2. Schema ═══

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "커밋 메시지",
                },
            },
            "required": ["message"],
        }

    # ═══ 3. Behavior Flags ═══
    # 쓰기 도구 — 사용자 확인 필요

    @property
    def requires_confirmation(self) -> bool:
        return True

    # ═══ 5. Lifecycle ═══

    def validate_input(self, input_data: dict[str, Any]) -> str | None:
        """커밋 메시지가 비어 있는지 검증한다."""
        message = input_data.get("message", "")
        if not message or not message.strip():
            return "커밋 메시지는 비어 있을 수 없습니다."
        return None

    async def check_permissions(
        self, input_data: dict[str, Any], context: ToolUseContext
    ) -> PermissionResult:
        """커밋은 사용자에게 확인을 요청한다."""
        message = input_data.get("message", "")
        return PermissionResult(
            behavior=PermissionBehavior.ASK,
            message=f"Git commit: {message}",
        )

    async def call(self, input_data: dict[str, Any], context: ToolUseContext) -> ToolResult:
        """
        git commit -m 명령어를 실행한다.

        처리 순서:
          1. 커밋 메시지 추출
          2. git commit -m "메시지" 실행
          3. 결과 반환 (새 커밋 해시 포함)
        """
        message = input_data["message"]
        args = ["commit", "-m", message]

        logger.info("GitCommit: message=%s (cwd=%s)", message[:50], context.cwd)
        return await _run_git_async(args, context.cwd)

    # ═══ 7. UI Hints ═══

    def get_progress_label(self, input_data: dict[str, Any]) -> str:
        return "Creating commit..."

    def get_input_summary(self, input_data: dict[str, Any]) -> str:
        return input_data.get("message", "")[:80]


# ─────────────────────────────────────────────
# GitBranchTool — 브랜치 목록 조회
# ─────────────────────────────────────────────
class GitBranchTool(BaseTool):
    """
    Git 브랜치 목록을 조회하는 도구.
    로컬 브랜치 목록과 현재 브랜치를 보여준다.
    읽기 전용이므로 병렬 실행이 안전하다.
    """

    # ═══ 1. Identity ═══

    @property
    def name(self) -> str:
        return "GitBranch"

    @property
    def description(self) -> str:
        return (
            "Git 브랜치 목록을 조회합니다. "
            "현재 브랜치가 *로 표시되며, -a 옵션으로 원격 브랜치도 볼 수 있습니다."
        )

    @property
    def group(self) -> str:
        return "git"

    # ═══ 2. Schema ═══

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "all": {
                    "type": "boolean",
                    "description": "원격 브랜치도 포함 (-a)",
                    "default": False,
                },
                "verbose": {
                    "type": "boolean",
                    "description": "각 브랜치의 최신 커밋 표시 (-v)",
                    "default": False,
                },
            },
            "required": [],
        }

    # ═══ 3. Behavior Flags ═══

    @property
    def is_read_only(self) -> bool:
        return True

    @property
    def is_concurrency_safe(self) -> bool:
        return True

    # ═══ 5. Lifecycle ═══

    async def check_permissions(
        self, input_data: dict[str, Any], context: ToolUseContext
    ) -> PermissionResult:
        """읽기 전용이므로 항상 허용한다."""
        return PermissionResult(behavior=PermissionBehavior.ALLOW)

    async def call(self, input_data: dict[str, Any], context: ToolUseContext) -> ToolResult:
        """
        git branch 명령어를 실행한다.
        옵션에 따라 -a(전체), -v(상세) 플래그를 추가한다.
        """
        args = ["branch"]

        if input_data.get("all", False):
            args.append("-a")

        if input_data.get("verbose", False):
            args.append("-v")

        logger.debug("GitBranch: git %s (cwd=%s)", " ".join(args), context.cwd)
        return await _run_git_async(args, context.cwd)

    # ═══ 7. UI Hints ═══

    def get_progress_label(self, input_data: dict[str, Any]) -> str:
        return "Listing branches..."


# ─────────────────────────────────────────────
# GitCheckoutTool — 브랜치/파일 체크아웃
# ─────────────────────────────────────────────
class GitCheckoutTool(BaseTool):
    """
    Git 체크아웃 도구.
    브랜치 전환 또는 특정 파일을 복원할 수 있다.
    작업 트리를 변경하므로 사용자 확인이 필요하다 (check_permissions=ASK).
    """

    # ═══ 1. Identity ═══

    @property
    def name(self) -> str:
        return "GitCheckout"

    @property
    def description(self) -> str:
        return (
            "Git 체크아웃을 수행합니다. "
            "브랜치를 전환하거나, 특정 파일을 이전 상태로 복원합니다. "
            "-b 옵션으로 새 브랜치를 생성할 수도 있습니다."
        )

    @property
    def group(self) -> str:
        return "git"

    # ═══ 2. Schema ═══

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "target": {
                    "type": "string",
                    "description": "체크아웃 대상 (브랜치 이름, 커밋 해시, 태그 등)",
                },
                "new_branch": {
                    "type": "boolean",
                    "description": "새 브랜치를 생성하며 전환 (-b)",
                    "default": False,
                },
                "path": {
                    "type": "string",
                    "description": "특정 파일만 복원 (-- path)",
                },
            },
            "required": ["target"],
        }

    # ═══ 3. Behavior Flags ═══
    # 작업 트리를 변경하므로 기본 fail-closed 유지

    @property
    def is_destructive(self) -> bool:
        """파일 복원 시 변경사항이 손실될 수 있다."""
        return True

    # ═══ 5. Lifecycle ═══

    async def check_permissions(
        self, input_data: dict[str, Any], context: ToolUseContext
    ) -> PermissionResult:
        """작업 트리 변경이므로 사용자에게 확인을 요청한다."""
        target = input_data.get("target", "")
        new_branch = input_data.get("new_branch", False)
        path = input_data.get("path")

        if path:
            msg = f"Git checkout: '{path}' from '{target}' (파일 복원)"
        elif new_branch:
            msg = f"Git checkout: 새 브랜치 '{target}' 생성"
        else:
            msg = f"Git checkout: '{target}'(으)로 전환"

        return PermissionResult(
            behavior=PermissionBehavior.ASK,
            message=msg,
        )

    async def call(self, input_data: dict[str, Any], context: ToolUseContext) -> ToolResult:
        """
        git checkout 명령어를 실행한다.

        처리 순서:
          1. 대상(target) 추출
          2. -b(새 브랜치) 또는 --(파일 복원) 옵션 구성
          3. subprocess로 실행
        """
        target = input_data["target"]
        new_branch = input_data.get("new_branch", False)
        path = input_data.get("path")

        args = ["checkout"]

        if new_branch:
            # 새 브랜치를 생성하며 전환
            args.extend(["-b", target])
        else:
            args.append(target)

        # 특정 파일만 복원하는 경우
        if path:
            args.extend(["--", path])

        logger.info("GitCheckout: git %s (cwd=%s)", " ".join(args), context.cwd)
        return await _run_git_async(args, context.cwd)

    # ═══ 7. UI Hints ═══

    def get_progress_label(self, input_data: dict[str, Any]) -> str:
        target = input_data.get("target", "")
        return f"Checking out {target}..."

    def get_input_summary(self, input_data: dict[str, Any]) -> str:
        return input_data.get("target", "")
