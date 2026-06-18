"""
Phase 8.0 통합 테스트 — 핵심 모듈 통합.

사양서 Ch.22.6 Test 1~5에 해당한다:
  Test 1: TestQueryLoopIntegration — query_loop → model → tool → response 풀 플로우
  Test 2: TestToolChainIntegration — 다중 도구 체인 실행
  Test 3: TestSecurityIntegration — 보안 시스템 통합
  Test 4: TestThinkingIntegration — Thinking Engine 통합
  Test 5: TestMemoryIntegration — Memory 시스템 통합

GPU 서버 없이 EnhancedMockModelProvider로 모든 테스트를 수행한다.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.message import (
    Message,
    StreamEvent,
    StreamEventType,
)
from core.orchestrator.query_loop import query_loop
from core.security.command_filter import CommandFilter
from core.security.path_guard import PathGuard
from core.tools.base import BaseTool, ToolUseContext

# conftest에서 import (EnhancedMockModelProvider, MockResponse)
from tests.conftest import EnhancedMockModelProvider, MockResponse


# ─────────────────────────────────────────────
# Test 1: Query Loop 통합 테스트
# ─────────────────────────────────────────────
@pytest.mark.asyncio
class TestQueryLoopIntegration:
    """query_loop → model → tool → response 전체 플로우를 검증한다."""

    async def test_single_turn_text_only_response(
        self, workspace: Path, tool_use_context: ToolUseContext, basic_tools: list[BaseTool]
    ) -> None:
        """텍스트만 응답하는 단일 턴 — 도구 호출이 없으면 1턴으로 깔끔히 끝나는지 검증.

        가장 단순한 happy path다. 모델이 도구를 안 부르면 query_loop이
        추가 턴을 돌지 않고(_call_count==1), TEXT_DELTA·MESSAGE_STOP 이벤트와
        assistant 메시지를 정상 산출하는지 본다. 여기가 깨지면 그 위 모든
        다중 턴 시나리오도 의미가 없으므로 기초 회귀 가드 역할을 한다.
        """
        # 모델이 텍스트만 응답 (도구 호출 없음 → 1턴으로 종료)
        provider = EnhancedMockModelProvider(
            responses=[
                MockResponse(text="안녕하세요! 도움이 필요하시면 말씀해주세요."),
            ]
        )

        messages = [Message.user("안녕")]
        events: list[StreamEvent | Message] = []

        async for event in query_loop(
            messages=messages,
            system_prompt="당신은 도움을 주는 AI입니다.",
            model_provider=provider,
            tools=basic_tools,
            context=tool_use_context,
        ):
            events.append(event)

        # 모델이 1회만 호출되었는지 검증
        assert provider._call_count == 1

        # TEXT_DELTA 이벤트가 존재하는지 검증
        text_deltas = [
            e for e in events if isinstance(e, StreamEvent) and e.type == StreamEventType.TEXT_DELTA
        ]
        assert len(text_deltas) >= 1
        assert "안녕하세요" in text_deltas[0].text

        # MESSAGE_STOP 이벤트가 존재하는지 검증
        message_stops = [
            e
            for e in events
            if isinstance(e, StreamEvent) and e.type == StreamEventType.MESSAGE_STOP
        ]
        assert len(message_stops) >= 1

        # assistant 메시지가 대화에 추가되었는지 검증
        assistant_messages = [e for e in events if isinstance(e, Message)]
        assert len(assistant_messages) >= 1

    async def test_sampling_params_forwarded_from_query_loop_to_stream(
        self, workspace: Path, tool_use_context: ToolUseContext, basic_tools: list[BaseTool]
    ) -> None:
        """query_loop이 4개 샘플링 파라미터를 model_provider.stream()으로 전달한다.

        degeneration 버그 수정(2026-06-18): query_loop → stream() 구간이 누락되면
        repetition_penalty가 vLLM까지 도달하지 못해 무한 반복이 재발한다.
        EnhancedMockModelProvider가 마지막 호출값(_last_*)을 기록하므로 이를 검증한다.
        """
        provider = EnhancedMockModelProvider(
            responses=[MockResponse(text="확인했습니다.")]
        )

        messages = [Message.user("니체 철학 요약해줘")]

        async for _event in query_loop(
            messages=messages,
            system_prompt="당신은 도움을 주는 AI입니다.",
            model_provider=provider,
            tools=basic_tools,
            context=tool_use_context,
            # KNOWLEDGE 프로필 값을 그대로 흘려보낸다
            top_p=0.95,
            repetition_penalty=1.15,
            frequency_penalty=0.3,
            presence_penalty=0.0,
        ):
            pass

        # 마지막 stream() 호출에 KNOWLEDGE 샘플링 값이 그대로 전달돼야 한다
        assert provider._last_top_p == pytest.approx(0.95)
        assert provider._last_repetition_penalty == pytest.approx(1.15)
        assert provider._last_frequency_penalty == pytest.approx(0.3)
        assert provider._last_presence_penalty == pytest.approx(0.0)

    async def test_read_tool_then_respond(
        self, workspace: Path, tool_use_context: ToolUseContext, basic_tools: list[BaseTool]
    ) -> None:
        """Read 도구 호출 → 파일 읽기 → 최종 텍스트 응답 (2턴) 흐름을 검증한다.

        도구가 끼면 query_loop은 '턴1: 도구 호출 → 도구 실행 → 결과를 대화에 주입
        → 턴2: 모델이 결과 보고 최종 답변'으로 한 번 더 돌아야 한다. 모델 호출이
        정확히 2회였고 tool_result 메시지가 대화에 추가됐는지로 이 왕복을 확인한다.
        """
        # workspace에 테스트 파일 생성 — Read 도구가 실제로 읽을 대상
        test_file = workspace / "test.txt"
        test_file.write_text("hello world from test file", encoding="utf-8")

        # 턴 1: Read 도구 호출, 턴 2: 최종 텍스트 응답
        provider = EnhancedMockModelProvider(
            responses=[
                MockResponse(
                    tool_calls=[
                        {
                            "name": "Read",
                            "input": {"file_path": str(test_file)},
                        }
                    ],
                ),
                MockResponse(text="파일 내용을 확인했습니다: hello world from test file"),
            ]
        )

        messages = [Message.user("test.txt 파일을 읽어줘")]
        events: list[StreamEvent | Message] = []

        async for event in query_loop(
            messages=messages,
            system_prompt="You are a helpful AI.",
            model_provider=provider,
            tools=basic_tools,
            context=tool_use_context,
        ):
            events.append(event)

        # 모델이 2회 호출되었는지 검증 (턴 1: 도구 호출, 턴 2: 텍스트 응답)
        assert provider._call_count == 2

        # tool_result 메시지가 대화에 추가되었는지 검증
        # Message.role은 str 또는 Enum — 안전하게 문자열 비교
        tool_result_messages = [
            e for e in events if isinstance(e, Message) and str(e.role) == "tool_result"
        ]
        assert len(tool_result_messages) >= 1

    async def test_edit_tool_modifies_file(
        self, workspace: Path, tool_use_context: ToolUseContext, basic_tools: list[BaseTool]
    ) -> None:
        """Edit 도구 호출로 디스크의 파일이 실제로 바뀌는지 끝까지(부수효과까지) 검증한다.

        이벤트만 보는 게 아니라 실제 Edit 도구를 태워 파일 내용이 old→new로
        교체됐는지 read_text로 직접 확인한다. mock 도구였다면 잡지 못할,
        '진짜 쓰기가 일어났는가'를 보장하는 통합 성격의 테스트다.
        """
        test_file = workspace / "edit_target.py"
        test_file.write_text("x = old_value\ny = 2\n", encoding="utf-8")

        # 턴 1: Edit 도구 호출, 턴 2: 최종 텍스트 응답
        provider = EnhancedMockModelProvider(
            responses=[
                MockResponse(
                    tool_calls=[
                        {
                            "name": "Edit",
                            "input": {
                                "file_path": str(test_file),
                                "old_string": "old_value",
                                "new_string": "new_value",
                            },
                        }
                    ],
                ),
                MockResponse(text="파일을 수정했습니다."),
            ]
        )

        messages = [Message.user("old_value를 new_value로 바꿔줘")]

        async for _event in query_loop(
            messages=messages,
            system_prompt="You are a helpful AI.",
            model_provider=provider,
            tools=basic_tools,
            context=tool_use_context,
        ):
            pass

        # 파일이 실제로 수정되었는지 검증
        content = test_file.read_text(encoding="utf-8")
        assert "new_value" in content
        assert "old_value" not in content

    async def test_max_turns_limit(
        self, workspace: Path, tool_use_context: ToolUseContext, basic_tools: list[BaseTool]
    ) -> None:
        """max_turns 안전장치가 무한 루프를 끊는지 검증한다(중요한 안전 가드).

        모델이 매 턴 똑같이 도구 호출만 반환하도록 만들어 일부러 무한 루프를
        유발한다. query_loop은 max_turns=3에서 멈춰야 하고(_call_count==3),
        사용자에게 '최대 턴 도달'을 알리는 SYSTEM_WARNING을 내보내야 한다.
        이 가드가 없으면 폭주한 에이전트가 GPU/예산을 통째로 태울 수 있다.
        """
        # 테스트 파일 — Read 도구가 사용할 파일
        test_file = workspace / "loop_test.txt"
        test_file.write_text("loop content", encoding="utf-8")

        # 매번 도구 호출을 반환하여 무한 루프를 시뮬레이션
        provider = EnhancedMockModelProvider(
            responses=[
                MockResponse(
                    tool_calls=[
                        {
                            "name": "Read",
                            "input": {"file_path": str(test_file)},
                        }
                    ],
                ),
            ]
        )

        messages = [Message.user("계속 파일을 읽어줘")]
        events: list[StreamEvent | Message] = []

        async for event in query_loop(
            messages=messages,
            system_prompt="You are a helpful AI.",
            model_provider=provider,
            tools=basic_tools,
            context=tool_use_context,
            max_turns=3,  # 3턴으로 제한
        ):
            events.append(event)

        # 최대 3턴만 실행되었는지 검증
        assert provider._call_count == 3

        # SYSTEM_WARNING 이벤트가 최대 턴 수 도달을 알리는지 검증
        warnings = [
            e
            for e in events
            if isinstance(e, StreamEvent)
            and e.type == StreamEventType.SYSTEM_WARNING
            and e.message
            and "최대 턴" in e.message
        ]
        assert len(warnings) >= 1

    async def test_multi_turn_context_maintained(
        self, workspace: Path, tool_use_context: ToolUseContext, basic_tools: list[BaseTool]
    ) -> None:
        """여러 턴을 거치며 대화 이력이 누적·유지되는지 검증한다.

        query_loop은 넘겨준 messages 리스트를 직접 mutate(추가)한다. 도구를 한 번
        쓰는 시나리오를 돌린 뒤 messages에 user·assistant(도구호출)·tool_result·
        assistant(최종답변)이 모두 쌓였는지(>=4건, role 종류 확인) 본다. 컨텍스트가
        끊기면 모델이 직전 도구 결과를 못 보고 같은 작업을 반복하게 된다.
        """
        test_file = workspace / "context_test.txt"
        test_file.write_text("important data", encoding="utf-8")

        provider = EnhancedMockModelProvider(
            responses=[
                MockResponse(
                    tool_calls=[
                        {
                            "name": "Read",
                            "input": {"file_path": str(test_file)},
                        }
                    ],
                ),
                MockResponse(text="파일에서 important data를 읽었습니다."),
            ]
        )

        messages = [Message.user("파일을 읽어줘")]

        async for _ in query_loop(
            messages=messages,
            system_prompt="You are a helpful AI.",
            model_provider=provider,
            tools=basic_tools,
            context=tool_use_context,
        ):
            pass

        # messages가 mutate되어 tool_result와 assistant 메시지가 추가되었는지 검증
        # 원본 user 메시지 + assistant(tool_call) + tool_result + assistant(text)
        assert len(messages) >= 4
        roles = [str(m.role) for m in messages]
        assert "user" in roles
        assert "assistant" in roles
        assert "tool_result" in roles


# ─────────────────────────────────────────────
# Test 2: Tool Chain 통합 테스트
# ─────────────────────────────────────────────
@pytest.mark.asyncio
class TestToolChainIntegration:
    """다중 도구 체인 실행을 검증한다."""

    async def test_multi_tool_single_turn(
        self, workspace: Path, tool_use_context: ToolUseContext, basic_tools: list[BaseTool]
    ) -> None:
        """한 턴에서 읽기 도구 여러 개를 동시에 호출했을 때 모두 처리되는지 검증한다.

        Read는 읽기 전용(동시 실행 안전)이라 한 턴에 2개를 같이 부를 수 있다.
        두 파일 각각에 대한 tool_result가 messages에 빠짐없이 추가됐는지(>=2건)로
        '한 턴 다중 도구' 처리가 정상인지 확인한다.
        """
        # 여러 파일 생성
        (workspace / "a.py").write_text("print('hello')", encoding="utf-8")
        (workspace / "b.py").write_text("print('world')", encoding="utf-8")

        # 1턴에서 2개 Read 도구 동시 호출
        provider = EnhancedMockModelProvider(
            responses=[
                MockResponse(
                    tool_calls=[
                        {"name": "Read", "input": {"file_path": str(workspace / "a.py")}},
                        {"name": "Read", "input": {"file_path": str(workspace / "b.py")}},
                    ],
                ),
                MockResponse(text="두 파일을 모두 읽었습니다."),
            ]
        )

        messages = [Message.user("a.py와 b.py를 읽어줘")]
        events: list[StreamEvent | Message] = []

        async for event in query_loop(
            messages=messages,
            system_prompt="You are a helpful AI.",
            model_provider=provider,
            tools=basic_tools,
            context=tool_use_context,
        ):
            events.append(event)

        # 2개의 tool_result가 messages에 추가되었는지 검증
        tool_results = [m for m in messages if hasattr(m, "role") and str(m.role) == "tool_result"]
        assert len(tool_results) >= 2

    async def test_read_then_edit_chain(
        self, workspace: Path, tool_use_context: ToolUseContext, basic_tools: list[BaseTool]
    ) -> None:
        """Read → Edit → 최종 응답으로 이어지는 3턴 도구 체인을 검증한다.

        앞 도구 결과가 다음 도구 입력으로 자연스럽게 이어지는, 실제 작업에 가까운
        시나리오다. 모델 호출이 정확히 3회였고 파일이 최종적으로 value=100으로
        바뀌었는지로 '단계가 끊기지 않고 순서대로 흘렀는지'를 확인한다.
        """
        target_file = workspace / "chain_target.py"
        target_file.write_text("value = 42\n", encoding="utf-8")

        provider = EnhancedMockModelProvider(
            responses=[
                # 턴 1: Read
                MockResponse(
                    tool_calls=[
                        {
                            "name": "Read",
                            "input": {"file_path": str(target_file)},
                        }
                    ],
                ),
                # 턴 2: Edit
                MockResponse(
                    tool_calls=[
                        {
                            "name": "Edit",
                            "input": {
                                "file_path": str(target_file),
                                "old_string": "value = 42",
                                "new_string": "value = 100",
                            },
                        }
                    ],
                ),
                # 턴 3: 최종 응답
                MockResponse(text="값을 42에서 100으로 변경했습니다."),
            ]
        )

        messages = [Message.user("chain_target.py의 값을 100으로 바꿔줘")]

        async for _ in query_loop(
            messages=messages,
            system_prompt="You are a helpful AI.",
            model_provider=provider,
            tools=basic_tools,
            context=tool_use_context,
        ):
            pass

        # 파일이 수정되었는지 검증
        content = target_file.read_text(encoding="utf-8")
        assert "value = 100" in content
        # 3턴 실행
        assert provider._call_count == 3


# ─────────────────────────────────────────────
# Test 3: Security 통합 테스트
# ─────────────────────────────────────────────
class TestSecurityIntegration:
    """보안 시스템(PathGuard + CommandFilter + PermissionPipeline) 통합 검증."""

    def test_path_traversal_blocked(self, workspace: Path) -> None:
        """대표적 공격인 경로 순회(../../etc/passwd)가 차단되는지 검증한다.

        is_path_safe는 (안전여부, 사유) 튜플을 돌려준다. 차단(False)되면서
        사유에 '순회' 또는 '보호'가 담겨, 왜 막혔는지 운영자가 알 수 있어야 한다.
        """
        pg = PathGuard()
        safe, reason = pg.is_path_safe("../../etc/passwd", str(workspace))

        assert safe is False
        assert "순회" in reason or "보호" in reason

    def test_null_byte_injection_blocked(self, workspace: Path) -> None:
        """null 바이트 인젝션이 차단된다."""
        pg = PathGuard()
        safe, reason = pg.is_path_safe("file.txt\x00.evil", str(workspace))

        assert safe is False
        assert "null" in reason

    def test_unc_path_blocked(self, workspace: Path) -> None:
        """UNC 경로(네트워크 경로)가 차단된다."""
        pg = PathGuard()
        safe, reason = pg.is_path_safe("\\\\server\\share\\file.txt", str(workspace))

        assert safe is False
        assert "UNC" in reason

    def test_protected_path_env_blocked(self, workspace: Path) -> None:
        """보호 경로(.env 파일)가 차단된다."""
        pg = PathGuard()
        safe, reason = pg.is_path_safe(str(workspace / ".env"), str(workspace))

        assert safe is False
        assert "보호" in reason

    def test_safe_path_allowed(self, workspace: Path) -> None:
        """정상 경로는 허용된다(거짓 양성 방지).

        차단만 잘 되고 멀쩡한 경로까지 막으면 도구가 무용지물이 된다.
        workspace 내부의 평범한 파일은 통과해야 함을 보장하는 반대 방향 가드다.
        """
        pg = PathGuard()
        # workspace 내부의 일반 파일은 허용
        test_file = workspace / "safe_file.py"
        test_file.write_text("# safe", encoding="utf-8")

        safe, reason = pg.is_path_safe(str(test_file.resolve()), str(workspace.resolve()))
        assert safe is True

    def test_dangerous_command_rm_rf_blocked(self) -> None:
        """rm -rf / 명령이 차단된다."""
        cf = CommandFilter()
        safe, severity, reason = cf.check_command("rm -rf /")

        assert safe is False
        assert severity == "critical"

    def test_dangerous_command_curl_blocked(self) -> None:
        """curl 같은 네트워크 명령이 에어갭 위반으로 차단되는지 검증한다.

        check_command은 (안전여부, 심각도, 사유)를 돌려준다. 외부 통신 시도는
        severity='high'로 막히고 사유에 '에어갭'이 명시돼야 한다. 폐쇄망 원칙(P10)을
        명령어 레벨에서 강제하는 핵심 가드다.
        """
        cf = CommandFilter()
        safe, severity, reason = cf.check_command("curl http://example.com")

        assert safe is False
        assert severity == "high"
        assert "에어갭" in reason

    def test_safe_command_ls_allowed(self) -> None:
        """ls 명령이 허용된다."""
        cf = CommandFilter()
        safe, severity, reason = cf.check_command("ls -la")

        assert safe is True

    def test_safe_command_git_allowed(self) -> None:
        """git 명령이 허용된다."""
        cf = CommandFilter()
        safe, severity, reason = cf.check_command("git status")

        assert safe is True

    @pytest.mark.asyncio
    async def test_permission_pipeline_plan_mode_blocks_writes(self, workspace: Path) -> None:
        """PLAN 모드에서는 쓰기 도구가 막히는지(5계층 통합) 검증한다.

        PLAN 모드는 '실행 없이 계획만' 모드라 파일을 바꾸면 안 된다. 5계층
        파이프라인을 끝까지 통과시킨 결정(decision)이 DENY 또는 ASK여야 한다
        (Layer 5에서 PLAN의 쓰기를 ASK→DENY로 보정하기도 하므로 둘 다 허용한다).
        """
        from core.permission.pipeline import PermissionPipeline
        from core.permission.types import (
            PermissionBehavior,
            PermissionContext,
            PermissionMode,
        )
        from core.tools.implementations.write_tool import WriteTool

        # PLAN 모드 컨텍스트
        ctx = PermissionContext(
            mode=PermissionMode.PLAN,
            working_directory=str(workspace),
        )
        pipeline = PermissionPipeline(context=ctx)

        tool = WriteTool()
        tool_input = {"file_path": str(workspace / "test.py"), "content": "x = 1"}
        tool_ctx = ToolUseContext(cwd=str(workspace), permission_mode="plan")

        decision = await pipeline.check(tool, tool_input, tool_ctx)

        # PLAN 모드에서 쓰기는 DENY 또는 ASK (Layer 5에서 최종 보정)
        assert decision.behavior in (PermissionBehavior.DENY, PermissionBehavior.ASK)

    @pytest.mark.asyncio
    async def test_permission_pipeline_bypass_mode_allows(self, workspace: Path) -> None:
        """BYPASS 모드에서 읽기 도구가 곧바로 허용되는지 검증한다(반대 방향 가드).

        위 PLAN 테스트가 '막아야 하는 경우'라면 이건 '허용해야 하는 경우'다.
        BYPASS 모드 + 읽기 전용 도구는 5계층을 통과해 ALLOW로 떨어져야 한다.
        둘을 함께 둬서 파이프라인이 과하게 막지도, 과하게 풀지도 않음을 보장한다.
        """
        from core.permission.pipeline import PermissionPipeline
        from core.permission.types import (
            PermissionBehavior,
            PermissionContext,
            PermissionMode,
        )
        from core.tools.implementations.read_tool import ReadTool

        # BYPASS 모드 컨텍스트
        ctx = PermissionContext(
            mode=PermissionMode.BYPASS_PERMISSIONS,
            working_directory=str(workspace),
        )
        pipeline = PermissionPipeline(context=ctx)

        tool = ReadTool()
        test_file = workspace / "readable.txt"
        test_file.write_text("content", encoding="utf-8")
        tool_input = {"file_path": str(test_file)}
        tool_ctx = ToolUseContext(cwd=str(workspace), permission_mode="bypass_permissions")

        decision = await pipeline.check(tool, tool_input, tool_ctx)

        # BYPASS 모드에서 읽기는 ALLOW
        assert decision.behavior == PermissionBehavior.ALLOW


# ─────────────────────────────────────────────
# Test 4: Thinking Engine 통합 테스트
# ─────────────────────────────────────────────
@pytest.mark.asyncio
class TestThinkingIntegration:
    """Thinking Engine의 복잡도 평가 → 전략 선택 → 엔진 실행을 통합 검증한다."""

    async def test_simple_query_direct_strategy(self) -> None:
        """단순 질문은 DIRECT 전략(1-pass)으로 가서 비용을 아끼는지 검증한다.

        '쉬운 질문에 비싼 다단계 추론을 쓰지 않는다'가 핵심이다. 복잡도 점수가
        낮고(<0.4) 패스 수가 1이며 응답이 비어 있지 않은지로 가벼운 경로 선택을 본다.
        """
        from core.thinking.orchestrator import ThinkingOrchestrator
        from core.thinking.strategy import ThinkingStrategy

        provider = EnhancedMockModelProvider(
            responses=[
                MockResponse(text="Python은 프로그래밍 언어입니다."),
            ]
        )

        orch = ThinkingOrchestrator(model_provider=provider)
        result = await orch.think("Python이 뭐야?")

        assert result.strategy == ThinkingStrategy.DIRECT
        assert result.passes == 1
        assert len(result.response) > 0
        assert result.score < 0.4  # 단순 질문은 낮은 복잡도

    async def test_moderate_query_hidden_cot(self) -> None:
        """복잡도가 올라가면 DIRECT가 아닌 다단계 전략으로 승급하는지 검증한다.

        'implement/error handling/optimize' 같은 복잡 키워드를 일부러 섞어 점수를
        높인 뒤, 선택 전략이 DIRECT가 아니라 HIDDEN_COT/SELF_REFLECT/MULTI_AGENT 중
        하나인지 확인한다. 정확히 어느 것인지는 임계값 튜닝에 따라 흔들릴 수 있어
        '집합 멤버십'으로 느슨하게 검증해 깨지기 쉬움을 피한다.
        """
        from core.thinking.orchestrator import ThinkingOrchestrator
        from core.thinking.strategy import ThinkingStrategy

        # 2-pass: 분석 + 최종 응답
        provider = EnhancedMockModelProvider(
            responses=[
                MockResponse(text="<think>모듈 구조를 분석합니다...</think>"),
                MockResponse(text="모듈을 다음과 같이 구현하면 됩니다..."),
            ]
        )

        orch = ThinkingOrchestrator(model_provider=provider)
        # 복잡도를 높이는 키워드 조합
        result = await orch.think(
            "implement a new module with error handling and optimize performance"
        )

        # 중간 이상 복잡도 → DIRECT가 아닌 전략이 선택되어야 한다
        assert result.strategy in (
            ThinkingStrategy.HIDDEN_COT,
            ThinkingStrategy.SELF_REFLECT,
            ThinkingStrategy.MULTI_AGENT,
        )
        assert result.passes >= 1

    async def test_thinking_cache_hit(self) -> None:
        """같은 질문을 두 번 하면 캐시가 모델 재호출을 막는지 검증한다.

        모델 호출이 비싸므로 동일 입력은 캐시로 답해야 한다. 1회차 후 호출 수를
        기록해 두고 2회차에서 호출 수가 그대로(증가 없음)인지, 결과(response/strategy)가
        동일한지로 캐시 히트를 확인한다. 모델 호출 카운트가 곧 검증 지표다.
        """
        from core.thinking.orchestrator import ThinkingOrchestrator

        provider = EnhancedMockModelProvider(
            responses=[
                MockResponse(text="캐시될 응답입니다."),
            ]
        )

        orch = ThinkingOrchestrator(model_provider=provider)

        # 첫 번째 호출 — 캐시 미스
        result1 = await orch.think("hello")
        assert provider._call_count >= 1

        first_call_count = provider._call_count

        # 두 번째 호출 — 캐시 히트 (모델 호출 없음)
        result2 = await orch.think("hello")
        assert provider._call_count == first_call_count  # 추가 호출 없음

        # 결과가 동일한지 검증
        assert result1.response == result2.response
        assert result1.strategy == result2.strategy

    async def test_complexity_score_range(self) -> None:
        """ComplexityAssessor 점수가 0~1 범위를 지키고, 복잡할수록 커지는지 검증한다.

        절대 점수는 튜닝에 따라 변하므로 두 가지 불변식만 확인한다:
        (1) 항상 [0.0, 1.0] 안에 있을 것, (2) 복잡한 질문 점수 > 단순 질문 점수.
        이 상대 비교가 전략 승급 로직(위 테스트)의 토대가 된다.
        """
        from core.thinking.assessor import ComplexityAssessor

        assessor = ComplexityAssessor()

        # 단순 질문
        simple_score = assessor.assess("hi", None)
        assert 0.0 <= simple_score <= 1.0
        assert simple_score < 0.5

        # 복잡한 질문 (여러 키워드 조합)
        complex_score = assessor.assess(
            "debug the race condition in async pipeline and optimize memory with profiling",
            None,
        )
        assert 0.0 <= complex_score <= 1.0
        assert complex_score > simple_score


# ─────────────────────────────────────────────
# Test 5: Memory 시스템 통합 테스트
# ─────────────────────────────────────────────
@pytest.mark.asyncio
class TestMemoryIntegration:
    """Memory 시스템(ShortTerm + LongTerm + Manager)의 인메모리 폴백 통합 검증."""

    async def test_short_term_memory_in_memory_fallback(self) -> None:
        """Redis가 없어도 ShortTermMemory가 인메모리 딕셔너리로 정상 동작하는지 검증한다.

        redis_client=None을 주면 외부 의존성 없이 set→get→delete가 일관되게
        동작해야 한다(저장한 값이 그대로 나오고, 지우면 None). 단기 메모리가
        Redis 장애에도 죽지 않는 graceful degradation을 보장하는 가드다.
        """
        from core.memory.short_term import ShortTermMemory

        # redis_client=None → 인메모리 딕셔너리 폴백
        stm = ShortTermMemory(redis_client=None)

        # ShortTermMemory.set(key, value, ttl) — key는 단일 문자열
        await stm.set("session-1:greeting", "hello")
        # 데이터 조회
        result = await stm.get("session-1:greeting")
        assert result == "hello"

        # 삭제
        await stm.delete("session-1:greeting")
        result = await stm.get("session-1:greeting")
        assert result is None

    async def test_long_term_memory_in_memory_fallback(self) -> None:
        """PostgreSQL이 없어도 LongTermMemory가 인메모리 리스트로 저장/조회되는지 검증한다.

        pg_pool=None이면 DB 없이도 add()가 id를 돌려주고 get(id)로 같은 내용을
        다시 꺼낼 수 있어야 한다. 장기 메모리가 DB 부재에도 폴백으로 살아남는지를 본다.
        """
        from core.memory.long_term import LongTermMemory
        from core.memory.types import MemoryEntry, MemoryType

        # pg_pool=None → 인메모리 리스트 폴백
        ltm = LongTermMemory(pg_pool=None)

        entry = MemoryEntry(
            memory_type=MemoryType.SEMANTIC,
            content="Python은 프로그래밍 언어이다",
            importance=0.8,
        )

        # 저장
        memory_id = await ltm.add(entry)
        assert memory_id is not None

        # 조회
        retrieved = await ltm.get(memory_id)
        assert retrieved is not None
        assert "Python" in retrieved.content

    async def test_memory_manager_turn_lifecycle(self) -> None:
        """MemoryManager의 턴 시작(on_turn_start)/종료(on_turn_end) 흐름이 깨지지 않는지 검증한다.

        실제 운영 호출 순서를 그대로 흉내낸다: 첫 턴 시작은 빈 결과 → 턴 종료에서
        대화를 저장 → 다음 턴 시작에서 이전 데이터를 검색. 인메모리 폴백에서는 검색
        결과가 비어 있을 수도 있어 내용 단언은 하지 않고, 항상 list를 돌려주는지(타입
        계약)와 전체 라이프사이클이 예외 없이 도는지를 중심으로 본다.
        """
        from core.memory.long_term import LongTermMemory
        from core.memory.manager import MemoryManager
        from core.memory.short_term import ShortTermMemory

        stm = ShortTermMemory(redis_client=None)
        ltm = LongTermMemory(pg_pool=None)
        mm = MemoryManager(short_term=stm, long_term=ltm)

        # on_turn_start — 새 세션이므로 빈 결과
        entries = await mm.on_turn_start("session-1", "안녕하세요")
        assert isinstance(entries, list)

        # on_turn_end — 대화 데이터 저장
        await mm.on_turn_end(
            session_id="session-1",
            messages=[
                Message.user("안녕하세요"),
                Message.assistant(text="반갑습니다!"),
            ],
        )

        # on_turn_start — 이전 턴의 데이터가 메모리에 있으면 검색됨
        entries2 = await mm.on_turn_start("session-1", "이전에 뭐라고 했지?")
        # 인메모리 폴백에서도 메모리 엔트리가 반환될 수 있다
        assert isinstance(entries2, list)

    async def test_memory_importance_assessment(self) -> None:
        """ImportanceAssessor가 '평범한 대화 < 중요한 결정'으로 중요도를 매기는지 검증한다.

        장기 메모리는 중요한 것만 오래 남겨야 하므로 중요도 점수가 선별 기준이 된다.
        복잡도 평가와 마찬가지로 절대값 대신 두 불변식만 확인한다: [0,1] 범위 유지,
        그리고 '중요한 결정' 문장 점수 >= 일반 인사 점수(상대 비교).
        """
        from core.memory.importance import ImportanceAssessor
        from core.memory.types import MemoryType

        assessor = ImportanceAssessor()

        # 일반 대화 — 낮은 중요도
        simple = assessor.assess("안녕하세요", MemoryType.EPISODIC)
        assert 0.0 <= simple <= 1.0

        # 중요한 정보 — 높은 중요도
        important = assessor.assess(
            "중요한 결정: 아키텍처를 마이크로서비스로 변경합니다. "
            "이 결정은 프로젝트의 핵심 방향을 결정합니다.",
            MemoryType.SEMANTIC,
        )
        assert 0.0 <= important <= 1.0
        # 중요한 키워드가 많으므로 더 높은 중요도
        assert important >= simple
