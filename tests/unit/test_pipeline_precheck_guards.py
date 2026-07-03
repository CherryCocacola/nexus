"""
PermissionPipeline Layer 2 사전검사(_precheck_security_guards) 단위 테스트.

무엇을 고정하는가 (구현 사실 core/permission/pipeline.py):
    파이프라인 생성 시 path_guard/command_filter를 "주입한 경우에만" Layer 2
    사전검사가 작동한다.
      - PathGuard 주입: 파일 도구의 file_path/path를 cwd-절대정규화 후
        보호경로(.env/.ssh)·경로순회(../../etc/passwd)면 deny, cwd 내 정상경로는
        deny 아님.
      - CommandFilter 주입: Bash(command 키)의 critical/high/medium은 deny,
        unknown은 차단하지 않는다(ASK 영역 → Layer 3).
    ★무회귀 핵심★: 사전검사기를 주입하지 않으면(둘 다 None) 동일 입력이 deny가
    아니어야 한다(기존 파이프라인 사용처 판정 불변).

왜 중요한가:
    이 사전검사는 배포(enforce)에서 실제 차단을 만들어내는 관문이다. 동시에
    미주입 경로가 조금이라도 바뀌면 기존 모든 파이프라인 테스트가 회귀한다.
    그 경계를 양방향으로 못박는다.
"""

from __future__ import annotations

from typing import Any

from core.permission.pipeline import PermissionPipeline
from core.permission.types import PermissionContext, PermissionMode
from core.security.command_filter import CommandFilter
from core.security.path_guard import PathGuard
from core.tools.base import (
    BaseTool,
    PermissionBehavior,
    PermissionResult,
    ToolResult,
    ToolUseContext,
)


class _MinimalTool(BaseTool):
    """
    사전검사 경로만 태우기 위한 최소 도구.

    - name/is_read_only는 파이프라인 카테고리 분류(FILE_WRITE/BASH)를 결정한다.
    - check_permissions는 항상 ALLOW를 반환한다 → Layer 2 도구 자체 검사는 통과.
      따라서 deny가 나온다면 그것은 오직 사전검사(PathGuard/CommandFilter) 때문이다.
    """

    def __init__(self, name: str, *, read_only: bool = False) -> None:
        self._name = name
        self._read_only = read_only

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return "test tool"

    @property
    def input_schema(self) -> dict[str, Any]:
        return {"type": "object"}

    @property
    def is_read_only(self) -> bool:
        return self._read_only

    async def check_permissions(
        self, input_data: dict[str, Any], context: ToolUseContext
    ) -> PermissionResult:
        # 항상 허용 — deny의 유일한 출처를 사전검사로 고정하기 위함.
        return PermissionResult(behavior=PermissionBehavior.ALLOW)

    async def call(self, input_data: dict[str, Any], context: ToolUseContext) -> ToolResult:
        return ToolResult.success("ok")


def _make_pipeline(
    *,
    path_guard: PathGuard | None = None,
    command_filter: CommandFilter | None = None,
) -> PermissionPipeline:
    """DEFAULT 모드 파이프라인을 만든다(사전검사기는 선택 주입)."""
    ctx = PermissionContext(mode=PermissionMode.DEFAULT, working_directory=".")
    return PermissionPipeline(
        ctx, path_guard=path_guard, command_filter=command_filter
    )


def _tool_ctx(cwd: str) -> ToolUseContext:
    """사전검사의 기준점이 되는 cwd를 담은 도구 실행 컨텍스트."""
    return ToolUseContext(cwd=cwd)


# ─────────────────────────────────────────────
# PathGuard 사전검사
# ─────────────────────────────────────────────
class TestPathGuardPrecheck:
    """PathGuard 주입 시 파일 경로 정책 위반이 deny로 판정돼야 한다."""

    async def test_write_env_file_denied(self, tmp_path) -> None:
        """cwd 내 .env 쓰기는 보호경로 위반으로 deny여야 한다."""
        pipe = _make_pipeline(path_guard=PathGuard())
        tool = _MinimalTool("Write")  # FILE_WRITE 카테고리
        decision = await pipe.check(tool, {"file_path": ".env"}, _tool_ctx(str(tmp_path)))
        assert decision.type == "deny"

    async def test_write_traversal_denied(self, tmp_path) -> None:
        """작업 디렉토리 밖으로 나가는 경로순회는 deny여야 한다."""
        pipe = _make_pipeline(path_guard=PathGuard())
        tool = _MinimalTool("Write")
        decision = await pipe.check(
            tool, {"file_path": "../../etc/passwd"}, _tool_ctx(str(tmp_path))
        )
        assert decision.type == "deny"

    async def test_write_ssh_key_denied(self, tmp_path) -> None:
        """cwd 내 .ssh/id_rsa 접근은 보호경로 위반으로 deny여야 한다."""
        pipe = _make_pipeline(path_guard=PathGuard())
        tool = _MinimalTool("Write")
        decision = await pipe.check(
            tool, {"file_path": ".ssh/id_rsa"}, _tool_ctx(str(tmp_path))
        )
        assert decision.type == "deny"

    async def test_write_normal_cwd_path_not_denied(self, tmp_path) -> None:
        """cwd 내 정상 파일 쓰기는 사전검사에서 deny가 아니어야 한다.

        (Layer 3에서 FILE_WRITE는 DEFAULT 모드상 ASK로 귀결되지만, 어쨌든 deny는
        아니다 — 사전검사 통과 확인이 목적이다.)
        """
        pipe = _make_pipeline(path_guard=PathGuard())
        tool = _MinimalTool("Write")
        decision = await pipe.check(
            tool, {"file_path": "notes.txt"}, _tool_ctx(str(tmp_path))
        )
        assert decision.type != "deny"

    async def test_env_not_denied_without_guard_injection(self, tmp_path) -> None:
        """★무회귀★ path_guard 미주입 시 .env 입력도 deny가 아니어야 한다(사전검사 skip)."""
        pipe = _make_pipeline()  # 사전검사기 없음
        tool = _MinimalTool("Write")
        decision = await pipe.check(tool, {"file_path": ".env"}, _tool_ctx(str(tmp_path)))
        assert decision.type != "deny"


# ─────────────────────────────────────────────
# CommandFilter 사전검사
# ─────────────────────────────────────────────
class TestCommandFilterPrecheck:
    """CommandFilter 주입 시 위험 Bash 명령이 deny로 판정돼야 한다."""

    async def test_install_denied_when_block_on(self, tmp_path) -> None:
        """block_package_install=True면 pip install이 deny여야 한다."""
        pipe = _make_pipeline(command_filter=CommandFilter(block_package_install=True))
        tool = _MinimalTool("Bash")
        decision = await pipe.check(
            tool, {"command": "pip install requests"}, _tool_ctx(str(tmp_path))
        )
        assert decision.type == "deny"

    async def test_install_not_denied_when_block_off(self, tmp_path) -> None:
        """block_package_install=False면 pip install은 deny가 아니어야 한다(개발)."""
        pipe = _make_pipeline(command_filter=CommandFilter(block_package_install=False))
        tool = _MinimalTool("Bash")
        decision = await pipe.check(
            tool, {"command": "pip install requests"}, _tool_ctx(str(tmp_path))
        )
        assert decision.type != "deny"

    async def test_rm_rf_root_always_denied(self, tmp_path) -> None:
        """rm -rf /는 critical이라 설치 게이팅과 무관하게 항상 deny여야 한다."""
        pipe = _make_pipeline(command_filter=CommandFilter(block_package_install=False))
        tool = _MinimalTool("Bash")
        decision = await pipe.check(
            tool, {"command": "rm -rf /"}, _tool_ctx(str(tmp_path))
        )
        assert decision.type == "deny"

    async def test_unknown_command_not_denied_by_precheck(self, tmp_path) -> None:
        """안전 목록에 없는 unknown 명령은 사전검사에서 차단하지 않는다(ASK 영역)."""
        pipe = _make_pipeline(command_filter=CommandFilter())
        tool = _MinimalTool("Bash")
        # 'frobnicate'는 안전 목록에도 위험 패턴에도 없음 → unknown → 사전검사 통과.
        decision = await pipe.check(
            tool, {"command": "frobnicate --now"}, _tool_ctx(str(tmp_path))
        )
        assert decision.type != "deny"

    async def test_install_not_denied_without_filter_injection(self, tmp_path) -> None:
        """★무회귀★ command_filter 미주입 시 rm -rf /조차 사전검사에서 deny가 아니어야 한다.

        (사전검사 skip을 증명하려 일부러 가장 위험한 입력을 쓴다 — 미주입이면
        사전검사가 통째로 건너뛰어지므로 여기서 deny가 나오면 안 된다.)
        """
        pipe = _make_pipeline()  # 사전검사기 없음
        tool = _MinimalTool("Bash")
        decision = await pipe.check(
            tool, {"command": "rm -rf /"}, _tool_ctx(str(tmp_path))
        )
        assert decision.type != "deny"
