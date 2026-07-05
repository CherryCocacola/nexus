"""
Git 도구 모음 — 로컬 Git 버전 관리 명령어를 실행하는 BaseTool 구현들.

이 파일은 Nexus 오케스트레이터가 도구로 노출하는 6개의 Git 기능을 정의한다.
모델(에이전트)이 tool_call로 이 도구들을 호출하면, 각 도구는 내부적으로
`git` CLI를 subprocess로 실행하고 그 결과를 ToolResult로 감싸 돌려준다.

제공하는 6개 도구:
  - GitLogTool     : 커밋 로그 조회        (읽기 전용, 병렬 안전)
  - GitDiffTool    : 변경 사항 비교        (읽기 전용, 병렬 안전)
  - GitStatusTool  : 작업 트리 상태 조회   (읽기 전용, 병렬 안전)
  - GitCommitTool  : 커밋 생성            (쓰기 — 사용자 확인 ASK 필요)
  - GitBranchTool  : 브랜치 목록 조회      (읽기 전용, 병렬 안전)
  - GitCheckoutTool: 브랜치/파일 체크아웃  (파괴적 — 사용자 확인 ASK 필요)

공통 동작:
  - 모든 도구는 `_run_git_async()` 헬퍼를 통해 git 명령을 실행한다.
    이 헬퍼는 블로킹 subprocess를 asyncio.to_thread로 감싸 이벤트 루프를
    막지 않게 하고, stdout/stderr/exit_code를 하나의 ToolResult로 조합한다.
  - 작업 디렉토리는 항상 호출 시점의 `context.cwd`(현재 세션의 작업 경로)를 쓴다.
  - 읽기 전용 도구는 check_permissions에서 ALLOW를 반환해 곧바로 실행되고,
    쓰기/파괴적 도구는 ASK를 반환해 권한 파이프라인에서 사용자 확인을 거친다.

설계상 제약(에어갭 준수):
  - 폐쇄망(에어갭) 환경을 전제로 하므로 원격 저장소 접근(push/pull/fetch/clone)은
    의도적으로 지원하지 않는다. 오직 로컬 저장소에 대한 조회/커밋/체크아웃만 다룬다.

의존 모듈:
  - core.tools.base 의 BaseTool ABC 및 결과/권한 타입(ToolResult, PermissionResult 등)

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

# 이 모듈 전용 로거. 규칙상 "nexus.{module}" 네임스페이스를 사용한다.
# 로그 필터링/레벨 조정이 git 도구 단위로 가능해진다.
logger = logging.getLogger("nexus.tools.git")

# git 명령 1회 실행에 허용하는 최대 대기 시간(초).
# 이 시간을 넘기면 TimeoutExpired가 발생하고 도구는 오류 ToolResult를 돌려준다.
_DEFAULT_TIMEOUT = 30

# stdout/stderr 캡처 상한(문자 수). 초대형 diff/log 출력이 컨텍스트를
# 폭주시키는 것을 막기 위한 안전장치. 초과분은 _truncate()에서 잘라낸다.
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
    실제로 `git` CLI를 실행하는 저수준 공통 함수.

    subprocess.run은 완료될 때까지 블로킹되므로, 이 함수를 이벤트 루프에서
    직접 호출하면 안 된다. 반드시 상위의 _run_git_async()가 asyncio.to_thread로
    별도 스레드에서 실행해 준다(그래야 다른 비동기 작업이 멈추지 않는다).

    실행 전에 환경 변수를 손봐서 대화형/장식용 출력을 제거한다:
      - 페이저(less 등)를 끄고, 색상 이스케이프 코드를 없애 순수 텍스트만 받는다.
      - 원격 접근 시 자격증명 프롬프트가 떠서 멈추는 상황을 차단한다.

    Args:
        args: git 서브 명령과 인자 목록 (예: ["log", "--oneline", "-10"]).
              앞에 "git"을 붙이지 않은 순수 인자만 넘긴다.
        cwd: 명령을 실행할 작업 디렉토리(대상 git 저장소 경로).
        timeout: 실행 제한 시간(초). 초과 시 TimeoutExpired 예외 발생.

    Returns:
        subprocess.CompletedProcess: stdout, stderr, returncode를 담은 실행 결과 객체.
    """
    # 현재 프로세스 환경을 복사한 뒤 git 전용 설정만 덮어쓴다.
    # 원본 os.environ을 직접 수정하지 않기 위해 copy()를 쓴다.
    env = os.environ.copy()
    # git 페이저 비활성화 — 출력이 길어도 대화형 페이저에서 멈추지 않고
    # 그대로 stdout으로 흘려보내도록 "cat"으로 지정한다.
    env["GIT_PAGER"] = "cat"
    # 원격 접근이 자격증명 입력을 요구할 때 무한 대기하지 않고 즉시 실패하도록
    # 터미널 프롬프트를 0(비활성)으로 설정한다. (에어갭 환경 안전장치)
    env["GIT_TERMINAL_PROMPT"] = "0"

    # ["git", *args] 형태로 실행. shell=False(기본)이라 셸 인젝션 위험이 없다.
    # noqa 주석들은 보안 린터(bandit) 경고를 의도적으로 무시하는 표시이므로
    # 절대 지우면 안 된다 — git 실행 자체가 이 도구의 정당한 목적이다.
    return subprocess.run(  # noqa: S603 — Git 도구는 git 명령 실행이 의도된 동작
        ["git", *args],  # noqa: S607 — git은 PATH에서 찾는 것이 정상
        cwd=cwd,  # 어느 저장소에서 실행할지 지정
        capture_output=True,  # stdout/stderr를 파이프로 캡처
        text=True,  # bytes 대신 str로 디코딩해서 받음
        timeout=timeout,  # 지정 시간 초과 시 예외 발생
        env=env,  # 위에서 손본 환경 변수 사용
    )


def _truncate(text: str) -> str:
    """
    출력이 상한(_MAX_OUTPUT_SIZE)을 넘으면 뒷부분만 남기고 잘라낸다.

    git log/diff는 최신 항목이 앞쪽, 오래된/덜 중요한 내용이 뒤에 오는 경우가
    많지만, 여기서는 "뒤쪽(최근에 출력된 부분)"을 보존한다. diff처럼 마지막
    요약이나 최종 상태가 뒤에 오는 출력에서 핵심을 놓치지 않기 위함이다.
    잘린 경우 앞머리에 몇 글자를 생략했는지 안내 문구를 붙인다.
    """
    # 상한 이하면 그대로 반환 — 잘라낼 필요가 없다.
    if len(text) <= _MAX_OUTPUT_SIZE:
        return text
    # 생략된 글자 수를 계산해 사용자에게 명시한다.
    removed = len(text) - _MAX_OUTPUT_SIZE
    # 뒤쪽 _MAX_OUTPUT_SIZE 글자만 보존하고 앞쪽 생략 사실을 알린다.
    return f"... (앞부분 생략, {removed}자 제거)\n{text[-_MAX_OUTPUT_SIZE:]}"


async def _run_git_async(
    args: list[str],
    cwd: str,
    timeout: int = _DEFAULT_TIMEOUT,  # noqa: ASYNC109
) -> ToolResult:
    """
    git 명령을 비동기로 실행하고 그 결과를 ToolResult로 감싸 반환하는 공통 헬퍼.

    6개 Git 도구의 call()이 마지막에 공통으로 호출하는 지점이다. 블로킹인
    _run_git()을 asyncio.to_thread로 스레드에 넘겨 이벤트 루프를 막지 않게 한다.

    성공/실패와 무관하게 stdout, stderr, exit_code를 사람이 읽기 좋은 하나의
    문자열로 조합한다. exit_code가 0이 아니면 오류 ToolResult로, 0이면 성공
    ToolResult로 감싸되, 어느 쪽이든 출력 본문은 그대로 담아 디버깅을 돕는다.

    Args:
        args: git 서브 명령과 인자 목록.
        cwd: 실행 대상 저장소 경로(보통 context.cwd).
        timeout: 실행 제한 시간(초).

    Returns:
        ToolResult: 성공 시 success(...), 실패/오류 시 error(...). 둘 다 exit_code 포함.
    """
    # 블로킹 subprocess 호출을 스레드로 넘겨 실행한다.
    # 발생 가능한 예외를 유형별로 잡아 각각 친절한 오류 메시지로 변환한다.
    try:
        result = await asyncio.to_thread(_run_git, args, cwd, timeout)
    except subprocess.TimeoutExpired:
        # 제한 시간 초과 — 명령이 너무 오래 걸렸거나 멈춘 경우.
        return ToolResult.error(f"git 명령어가 {timeout}초 타임아웃을 초과했습니다.")
    except FileNotFoundError:
        # git 실행 파일 자체를 못 찾음 — 설치 여부/PATH 문제.
        return ToolResult.error("git을 찾을 수 없습니다. git이 설치되어 있는지 확인하세요.")
    except OSError as e:
        # 그 외 OS 수준 실행 실패(권한, 잘못된 cwd 등).
        return ToolResult.error(f"git 명령어 실행 실패: {e}")

    # stdout/stderr가 None일 수 있으므로 빈 문자열로 보정한 뒤 상한 초과분을 자른다.
    stdout = _truncate(result.stdout or "")
    stderr = _truncate(result.stderr or "")

    # 표준 출력과 표준 에러를 한 덩어리 텍스트로 조합한다.
    # stderr는 "STDERR:" 머리말을 붙여 사용자가 구분할 수 있게 한다.
    parts: list[str] = []
    if stdout:
        parts.append(stdout)
    if stderr:
        parts.append(f"STDERR:\n{stderr}")

    # 아무 출력도 없으면 빈 문자열 대신 명시적 안내를 넣는다.
    output = "\n".join(parts) if parts else "(출력 없음)"

    # 종료 코드가 0이 아니면 git이 실패한 것 — 오류 결과로 감싼다.
    if result.returncode != 0:
        return ToolResult.error(
            f"{output}\nExit code: {result.returncode}",
            exit_code=result.returncode,
        )

    # 정상 종료 — 성공 결과로 감싸 반환한다.
    return ToolResult.success(output, exit_code=result.returncode)


# ─────────────────────────────────────────────
# GitLogTool — 커밋 로그 조회
# ─────────────────────────────────────────────
class GitLogTool(BaseTool):
    """
    Git 커밋 로그(`git log`)를 조회하는 읽기 전용 도구.

    기본값은 최근 20개 커밋을 한 줄 요약(--oneline) 형식으로 보여준다.
    max_count로 개수를, revision으로 범위(예: main..HEAD)를, path로 특정
    파일의 변경 이력만 필터링할 수 있다.

    읽기 전용이라 상태를 바꾸지 않으므로 병렬 실행이 안전하고(check_permissions는
    항상 ALLOW), 다른 읽기 도구와 동시에 돌아도 문제가 없다.
    """

    # ═══ 1. Identity ═══

    @property
    def name(self) -> str:
        return "GitLog"

    @property
    def description(self) -> str:
        return "Show git commit history."

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
                    "description": "Number of commits",
                    "default": 20,
                    "minimum": 1,
                    "maximum": 500,
                },
                "oneline": {
                    "type": "boolean",
                    "description": "One-line format",
                    "default": True,
                },
                "path": {
                    "type": "string",
                    "description": "File path filter",
                },
                "revision": {
                    "type": "string",
                    "description": "Revision range (e.g. main..HEAD)",
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
        입력 옵션을 git log 인자로 변환해 실행한다.

        처리 순서:
          1. 기본 명령 ["log"]에서 시작
          2. max_count/oneline/revision/path 옵션을 순서대로 인자에 반영
          3. _run_git_async로 실행해 ToolResult 반환

        주의: 경로 필터(path)는 반드시 "--" 뒤에 두어, 브랜치 이름과 파일
        경로가 겹칠 때 git이 헷갈리지 않도록 명확히 구분한다.
        """
        # git 서브 명령 목록을 누적해 나가는 리스트. "log"부터 시작한다.
        args = ["log"]

        # 최대 커밋 수 — 미지정 시 기본 20개. --max-count=N 형태로 붙인다.
        max_count = input_data.get("max_count", 20)
        args.append(f"--max-count={max_count}")

        # 한 줄 요약 형식 여부. 기본 True라 별도 지정이 없으면 --oneline이 붙는다.
        if input_data.get("oneline", True):
            args.append("--oneline")

        # 리비전 범위(예: "main..HEAD")가 주어지면 그대로 인자에 추가.
        revision = input_data.get("revision")
        if revision:
            args.append(revision)

        # 특정 파일/경로의 이력만 보고 싶을 때 "--" 구분자 뒤에 경로를 붙인다.
        path = input_data.get("path")
        if path:
            args.extend(["--", path])

        # 실제 실행 직전 디버그 로그로 명령과 작업 경로를 남긴다.
        logger.debug("GitLog: git %s (cwd=%s)", " ".join(args), context.cwd)
        return await _run_git_async(args, context.cwd)

    # ═══ 7. UI Hints ═══

    def get_progress_label(self, input_data: dict[str, Any]) -> str:
        # 실행 중 UI에 표시할 진행 라벨.
        return "Fetching git log..."


# ─────────────────────────────────────────────
# GitDiffTool — 변경 사항 비교
# ─────────────────────────────────────────────
class GitDiffTool(BaseTool):
    """
    Git 변경 사항(`git diff`)을 비교하는 읽기 전용 도구.

    무엇과 무엇을 비교할지는 옵션으로 결정된다:
      - 기본(옵션 없음): 작업 트리 vs 스테이징 영역(아직 add 안 한 변경)
      - staged=True    : 스테이징 영역 vs 마지막 커밋(--cached)
      - revision 지정  : 해당 ref(브랜치/커밋)와의 차이
    stat=True면 파일별 증감 줄 수만 요약(diffstat)하고, path로 특정 파일에
    한정할 수 있다.

    읽기 전용이라 병렬 실행이 안전하다(check_permissions는 항상 ALLOW).
    """

    # ═══ 1. Identity ═══

    @property
    def name(self) -> str:
        return "GitDiff"

    @property
    def description(self) -> str:
        return "Show git diff."

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
                    "description": "Show staged changes only",
                    "default": False,
                },
                "revision": {
                    "type": "string",
                    "description": "Git ref (branch, commit)",
                },
                "path": {
                    "type": "string",
                    "description": "File path filter",
                },
                "stat": {
                    "type": "boolean",
                    "description": "Show diffstat only",
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
        입력 옵션을 git diff 인자로 변환해 실행한다.

        처리 순서:
          1. 기본 명령 ["diff"]에서 시작
          2. staged→--cached, stat→--stat 플래그 반영
          3. revision/path 인자 추가 후 _run_git_async로 실행
        """
        # git 서브 명령 목록. "diff"부터 시작해 옵션을 덧붙인다.
        args = ["diff"]

        # 스테이징된(add된) 변경만 마지막 커밋과 비교하려면 --cached를 붙인다.
        if input_data.get("staged", False):
            args.append("--cached")

        # 실제 diff 대신 파일별 증감 통계만 보고 싶으면 --stat.
        if input_data.get("stat", False):
            args.append("--stat")

        # 특정 ref(브랜치/커밋)와 비교할 경우 그 이름을 인자로 추가.
        revision = input_data.get("revision")
        if revision:
            args.append(revision)

        # 특정 파일/경로만 비교하려면 "--" 뒤에 경로를 넣어 모호성을 없앤다.
        path = input_data.get("path")
        if path:
            args.extend(["--", path])

        # 실행 직전 디버그 로그.
        logger.debug("GitDiff: git %s (cwd=%s)", " ".join(args), context.cwd)
        return await _run_git_async(args, context.cwd)

    # ═══ 7. UI Hints ═══

    def get_progress_label(self, input_data: dict[str, Any]) -> str:
        # 실행 중 UI에 표시할 진행 라벨.
        return "Computing diff..."


# ─────────────────────────────────────────────
# GitStatusTool — 작업 트리 상태 조회
# ─────────────────────────────────────────────
class GitStatusTool(BaseTool):
    """
    Git 작업 트리 상태(`git status`)를 조회하는 읽기 전용 도구.

    현재 브랜치, 스테이징된/안 된 변경, 추적되지 않는(untracked) 파일 목록을
    보여준다. short=True면 한 줄 요약(-s) 형식으로 간결하게 출력한다.

    읽기 전용이라 병렬 실행이 안전하다(check_permissions는 항상 ALLOW).
    """

    # ═══ 1. Identity ═══

    @property
    def name(self) -> str:
        return "GitStatus"

    @property
    def description(self) -> str:
        return "Show git working tree status."

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
                    "description": "Short format (-s)",
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
        git status 명령을 실행한다.

        -u(=-unormal) 옵션을 기본 포함해 추적되지 않는 파일도 함께 보여준다.
        다만 -uall(디렉토리 내부 파일까지 전부 나열)은 쓰지 않는다 — 파일이
        매우 많은 대규모 저장소에서 출력/메모리가 폭증하는 것을 막기 위함이다.
        """
        # "status" + "-u"로 시작. -u는 untracked 파일 표시 수준을 normal로 둔다.
        args = ["status", "-u"]

        # short=True면 -s를 붙여 한 줄 요약(간결) 형식으로 출력.
        if input_data.get("short", False):
            args.append("-s")

        # 실행 직전 디버그 로그.
        logger.debug("GitStatus: git %s (cwd=%s)", " ".join(args), context.cwd)
        return await _run_git_async(args, context.cwd)

    # ═══ 7. UI Hints ═══

    def get_progress_label(self, input_data: dict[str, Any]) -> str:
        # 실행 중 UI에 표시할 진행 라벨.
        return "Checking git status..."


# ─────────────────────────────────────────────
# GitCommitTool — 커밋 생성
# ─────────────────────────────────────────────
class GitCommitTool(BaseTool):
    """
    Git 커밋(`git commit -m`)을 생성하는 쓰기 도구.

    이미 스테이징된(git add된) 변경 사항을 주어진 메시지로 커밋한다. 이 도구
    자체는 add를 하지 않으므로, 호출 전에 변경이 스테이징되어 있어야 한다.

    저장소 이력을 실제로 바꾸는 쓰기 작업이므로 fail-closed 원칙에 따라
    requires_confirmation=True로 두고, check_permissions에서 ASK를 반환해
    권한 파이프라인이 사용자 확인을 거치게 한다. 병렬 실행 안전 플래그도
    기본값(False)을 유지한다.
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
        """
        커밋 메시지가 실제로 존재하는지(공백만은 아닌지) 사전 검증한다.

        실행 파이프라인에서 call() 이전에 호출된다. 문제가 있으면 오류 문자열을
        반환하고(그 순간 실행이 중단됨), 정상이면 None을 반환한다.
        """
        message = input_data.get("message", "")
        # 빈 문자열이거나 공백/개행만 있으면 커밋할 수 없으므로 거부한다.
        if not message or not message.strip():
            return "커밋 메시지는 비어 있을 수 없습니다."
        return None

    async def check_permissions(
        self, input_data: dict[str, Any], context: ToolUseContext
    ) -> PermissionResult:
        """
        커밋은 이력을 바꾸는 쓰기 작업이므로 항상 사용자 확인(ASK)을 요청한다.

        확인 창에 보여줄 메시지에 커밋 메시지를 포함해, 사용자가 무엇을 커밋하는지
        바로 알 수 있게 한다.
        """
        message = input_data.get("message", "")
        return PermissionResult(
            behavior=PermissionBehavior.ASK,
            message=f"Git commit: {message}",
        )

    async def call(self, input_data: dict[str, Any], context: ToolUseContext) -> ToolResult:
        """
        git commit -m 명령을 실행해 스테이징된 변경을 커밋한다.

        처리 순서:
          1. 검증을 통과한 커밋 메시지를 꺼낸다(validate_input에서 이미 확인됨).
          2. ["commit", "-m", 메시지] 인자를 구성한다.
          3. _run_git_async로 실행하고, 새 커밋 해시가 담긴 출력을 반환한다.
        """
        # validate_input을 통과했으므로 "message" 키가 반드시 존재한다.
        message = input_data["message"]
        args = ["commit", "-m", message]

        # 감사/추적을 위해 info 레벨 로그를 남긴다. 메시지는 앞 50자만 기록.
        logger.info("GitCommit: message=%s (cwd=%s)", message[:50], context.cwd)
        return await _run_git_async(args, context.cwd)

    # ═══ 7. UI Hints ═══

    def get_progress_label(self, input_data: dict[str, Any]) -> str:
        # 도구 실행 중 UI에 표시할 진행 라벨.
        return "Creating commit..."

    def get_input_summary(self, input_data: dict[str, Any]) -> str:
        # 승인 창/이력 등에 한 줄로 요약해 보여줄 문자열. 커밋 메시지 앞 80자.
        return input_data.get("message", "")[:80]


# ─────────────────────────────────────────────
# GitBranchTool — 브랜치 목록 조회
# ─────────────────────────────────────────────
class GitBranchTool(BaseTool):
    """
    Git 브랜치 목록(`git branch`)을 조회하는 읽기 전용 도구.

    로컬 브랜치 목록을 보여주며, 현재 체크아웃된 브랜치는 앞에 *로 표시된다.
    all=True면 원격 추적 브랜치까지(-a), verbose=True면 각 브랜치의 최신
    커밋 요약(-v)을 함께 보여준다.

    조회만 하고 브랜치를 만들거나 지우지 않으므로 읽기 전용이며 병렬 실행이
    안전하다(check_permissions는 항상 ALLOW).
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
        git branch 명령을 실행한다.

        옵션에 따라 -a(원격 포함 전체), -v(각 브랜치 상세) 플래그를 덧붙인다.
        인자 없이 실행하면 로컬 브랜치 목록만 나온다.
        """
        # "branch"부터 시작. 옵션이 없으면 로컬 목록만 출력된다.
        args = ["branch"]

        # all=True면 원격 추적 브랜치까지 포함(-a).
        if input_data.get("all", False):
            args.append("-a")

        # verbose=True면 각 브랜치의 최신 커밋 해시/제목을 함께 표시(-v).
        if input_data.get("verbose", False):
            args.append("-v")

        # 실행 직전 디버그 로그.
        logger.debug("GitBranch: git %s (cwd=%s)", " ".join(args), context.cwd)
        return await _run_git_async(args, context.cwd)

    # ═══ 7. UI Hints ═══

    def get_progress_label(self, input_data: dict[str, Any]) -> str:
        # 실행 중 UI에 표시할 진행 라벨.
        return "Listing branches..."


# ─────────────────────────────────────────────
# GitCheckoutTool — 브랜치/파일 체크아웃
# ─────────────────────────────────────────────
class GitCheckoutTool(BaseTool):
    """
    Git 체크아웃(`git checkout`)을 수행하는 파괴적 도구.

    세 가지 용도를 하나로 다룬다:
      - 브랜치/커밋 전환    : target만 지정
      - 새 브랜치 생성 후 전환: new_branch=True (-b)
      - 특정 파일 복원      : path 지정 (target 상태로 파일을 되돌림)

    특히 파일 복원은 작업 트리의 아직 커밋되지 않은 변경을 덮어써 되돌릴 수
    없게 만들 수 있으므로 is_destructive=True로 표시한다. 작업 트리를 바꾸는
    작업이라 check_permissions에서 ASK를 반환해 반드시 사용자 확인을 거친다.
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
        """
        작업 트리를 바꾸는 작업이므로 항상 사용자 확인(ASK)을 요청한다.

        입력 조합(path/new_branch/그 외)에 따라 확인 창 문구를 구분해, 사용자가
        지금 어떤 종류의 체크아웃을 승인하는지(파일 복원인지, 새 브랜치 생성인지,
        단순 전환인지) 한눈에 알 수 있게 한다.
        """
        target = input_data.get("target", "")
        new_branch = input_data.get("new_branch", False)
        path = input_data.get("path")

        # path가 있으면 "파일 복원" 의도 — 가장 위험(변경 손실 가능)하므로 명시.
        if path:
            msg = f"Git checkout: '{path}' from '{target}' (파일 복원)"
        # new_branch면 새 브랜치 생성 후 전환.
        elif new_branch:
            msg = f"Git checkout: 새 브랜치 '{target}' 생성"
        # 그 외에는 단순 브랜치/커밋 전환.
        else:
            msg = f"Git checkout: '{target}'(으)로 전환"

        return PermissionResult(
            behavior=PermissionBehavior.ASK,
            message=msg,
        )

    async def call(self, input_data: dict[str, Any], context: ToolUseContext) -> ToolResult:
        """
        입력 조합을 git checkout 인자로 변환해 실행한다.

        처리 순서:
          1. 필수 인자 target을 꺼낸다.
          2. new_branch면 "-b target"(새 브랜치 생성)로, 아니면 "target"만 넣는다.
          3. path가 있으면 "-- path"를 덧붙여 해당 파일만 복원한다.
          4. _run_git_async로 실행한다.
        """
        # target은 스키마상 required라 항상 존재한다.
        target = input_data["target"]
        new_branch = input_data.get("new_branch", False)
        path = input_data.get("path")

        args = ["checkout"]

        if new_branch:
            # -b는 새 브랜치를 만들면서 곧바로 그 브랜치로 전환한다.
            args.extend(["-b", target])
        else:
            # 기존 브랜치/커밋/태그로 전환하거나, 파일 복원의 소스로 사용한다.
            args.append(target)

        # path가 있으면 "--" 뒤에 붙여 "target 상태의 이 파일만 복원"을 의미한다.
        if path:
            args.extend(["--", path])

        # 파괴적 작업이므로 info 레벨로 실행 내역을 남겨 추적 가능하게 한다.
        logger.info("GitCheckout: git %s (cwd=%s)", " ".join(args), context.cwd)
        return await _run_git_async(args, context.cwd)

    # ═══ 7. UI Hints ═══

    def get_progress_label(self, input_data: dict[str, Any]) -> str:
        # 실행 중 UI 진행 라벨. 어떤 대상으로 체크아웃 중인지 함께 보여준다.
        target = input_data.get("target", "")
        return f"Checking out {target}..."

    def get_input_summary(self, input_data: dict[str, Any]) -> str:
        # 승인 창/이력에 한 줄로 요약할 문자열. 체크아웃 대상만 표시.
        return input_data.get("target", "")
