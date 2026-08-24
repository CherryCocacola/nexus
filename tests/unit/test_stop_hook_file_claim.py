# STOP 훅 — "파일 만들었다"고만 하고 안 만든 턴의 종료를 막는다.
"""
2026-08-23 실측: 사후 검증기(file_claim)를 넣은 뒤 실서버 재검증에서도 같은 실패가
재발했다. 모델이 문서를 채팅에 출력하고 "작성했습니다"로 끝냈고, 경고는 붙었지만
**파일은 여전히 없었다.** 경고는 탐지지 예방이 아니다.

이 훅은 그 턴의 **종료 자체를 막아** 모델에게 한 번 더 기회를 준다.
차단 사유를 대화에 주입해야 모델이 무엇을 해야 하는지 안다 — 사유 없이 턴만
이어지면 같은 답을 반복해 차단이 루프로 바뀐다.

세션당 차단 상한(MAX_FILE_CLAIM_BLOCKS)이 반드시 있어야 한다. 모델이 끝내 파일을
안 쓰면 무한 루프가 되기 때문이다. 상한 도달 후에는 사후 검증 경고가 사실을 남긴다.
"""

from __future__ import annotations

import pytest

from core.hooks.builtin_hooks import MAX_FILE_CLAIM_BLOCKS, file_claim_stop_hook
from core.hooks.hook_manager import HookDecision, HookEvent, HookInput

CLAIM = "docs/TABLE_DESIGN.md 파일에 내용을 작성했습니다. 파일 존재를 확인했습니다."


def _stop(**meta) -> HookInput:
    base = {"assistant_text": "", "write_tools": [], "block_count": 0}
    base.update(meta)
    return HookInput(event=HookEvent.STOP, metadata=base)


class TestBlocksFalseClaim:
    @pytest.mark.asyncio
    async def test_claim_without_write_is_blocked(self) -> None:
        """★실제 사고 재현 — 주장은 있고 성공한 쓰기는 0건."""
        result = await file_claim_stop_hook(_stop(assistant_text=CLAIM))
        assert result.decision == HookDecision.BLOCK

    @pytest.mark.asyncio
    async def test_block_reason_quotes_the_claim(self) -> None:
        """사유에 문제 문장이 있어야 모델이 무엇을 고칠지 안다."""
        result = await file_claim_stop_hook(_stop(assistant_text=CLAIM))
        assert "TABLE_DESIGN" in result.block_reason


class TestDoesNotBlockLegitimateTurns:
    @pytest.mark.asyncio
    async def test_successful_write_passes(self) -> None:
        """실제로 썼으면 통과다."""
        result = await file_claim_stop_hook(
            _stop(assistant_text=CLAIM, write_tools=["Write"])
        )
        assert result.decision == HookDecision.CONTINUE

    @pytest.mark.asyncio
    async def test_bash_created_file_passes(self) -> None:
        """Bash 로 파일을 만드는 것도 정상이다."""
        result = await file_claim_stop_hook(
            _stop(assistant_text=CLAIM, write_tools=["Bash"])
        )
        assert result.decision == HookDecision.CONTINUE

    @pytest.mark.asyncio
    async def test_chat_only_answer_passes(self) -> None:
        """파일을 지목하지 않은 완료형은 채팅 전용 산출이라 정상이다."""
        result = await file_claim_stop_hook(
            _stop(assistant_text="요청하신 계획서를 작성했습니다. 위 내용을 참고하세요.")
        )
        assert result.decision == HookDecision.CONTINUE

    @pytest.mark.asyncio
    async def test_no_claim_passes(self) -> None:
        result = await file_claim_stop_hook(_stop(assistant_text="설명을 마쳤습니다."))
        assert result.decision == HookDecision.CONTINUE

    @pytest.mark.asyncio
    async def test_non_stop_event_ignored(self) -> None:
        """PRE_TOOL_USE 등 다른 시점에는 관여하지 않는다."""
        result = await file_claim_stop_hook(
            HookInput(event=HookEvent.PRE_TOOL_USE, metadata={"assistant_text": CLAIM})
        )
        assert result.decision == HookDecision.CONTINUE


class TestLoopGuard:
    """★상한이 없으면 차단이 무한 루프가 된다."""

    @pytest.mark.asyncio
    async def test_stops_blocking_at_limit(self) -> None:
        result = await file_claim_stop_hook(
            _stop(assistant_text=CLAIM, block_count=MAX_FILE_CLAIM_BLOCKS)
        )
        assert result.decision == HookDecision.CONTINUE

    @pytest.mark.asyncio
    async def test_still_blocks_below_limit(self) -> None:
        result = await file_claim_stop_hook(
            _stop(assistant_text=CLAIM, block_count=MAX_FILE_CLAIM_BLOCKS - 1)
        )
        assert result.decision == HookDecision.BLOCK

    @pytest.mark.asyncio
    async def test_limit_is_small(self) -> None:
        """기회를 너무 많이 주면 사용자가 먼저 떠난다."""
        assert 1 <= MAX_FILE_CLAIM_BLOCKS <= 3


class TestWiring:
    """배선 — 훅이 실제로 매니저에 등록되고 query_loop 이 근거를 넘기는가."""

    @pytest.mark.asyncio
    async def test_manager_runs_registered_hook(self) -> None:
        from core.hooks.hook_manager import HookManager

        manager = HookManager()
        manager.register(HookEvent.STOP, file_claim_stop_hook)
        result = await manager.run(HookEvent.STOP, _stop(assistant_text=CLAIM))
        assert result.decision == HookDecision.BLOCK

    def test_write_tool_collector_scopes_to_last_user_message(self) -> None:
        """★한 턴이 아니라 '마지막 사용자 메시지 이후'를 본다.

        모델이 세 턴 전에 파일을 쓰고 지금은 요약 중일 수 있다. 턴 하나만 보면
        그런 정상 케이스를 오탐한다.
        """
        from core.message import Message, Role, ToolUseBlock
        from core.orchestrator.query_loop import _successful_write_tools_since_user

        messages = [
            Message.user("이전 요청"),
            Message(role=Role.ASSISTANT, content=[ToolUseBlock(id="old", name="Write", input={})]),
            Message.tool_result(tool_use_id="old", content="ok"),
            Message.user("새 요청"),  # ← 여기부터가 이번 요청
            Message(role=Role.ASSISTANT, content=[ToolUseBlock(id="new", name="Read", input={})]),
            Message.tool_result(tool_use_id="new", content="ok"),
        ]
        # 이전 요청의 Write 는 세면 안 된다.
        assert _successful_write_tools_since_user(messages) == []

    def test_write_tool_collector_finds_current_request_write(self) -> None:
        from core.message import Message, Role, ToolUseBlock
        from core.orchestrator.query_loop import _successful_write_tools_since_user

        messages = [
            Message.user("요청"),
            Message(role=Role.ASSISTANT, content=[ToolUseBlock(id="w", name="Write", input={})]),
            Message.tool_result(tool_use_id="w", content="ok"),
        ]
        assert _successful_write_tools_since_user(messages) == ["Write"]
