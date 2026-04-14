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
        """텍스트만 응답하는 단일 턴 — 도구 호출 없이 종료한다."""
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

    async def test_read_tool_then_respond(
        self, workspace: Path, tool_use_context: ToolUseContext, basic_tools: list[BaseTool]
    ) -> None:
        """Read 도구 호출 → 파일 읽기 → 최종 텍스트 응답 (2턴)."""
        # workspace에 테스트 파일 생성
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
        """Edit 도구 호출로 파일이 실제로 변경되는지 검증한다."""
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
        """max_turns 제한으로 무한 루프가 방지되는지 검증한다."""
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
        """이전 턴의 대화 컨텍스트가 유지되는지 검증한다."""
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
        """한 턴에서 여러 Read 도구를 병렬로 호출한다."""
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
        """Read → Edit → 텍스트 응답 (3턴 체인)."""
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
        """경로 순회 공격(../../etc/passwd)이 차단된다."""
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
        """정상 경로는 허용된다."""
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
        """curl 명령이 차단된다 (에어갭 위반)."""
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
        """PLAN 모드에서 쓰기 도구가 거부되는지 검증한다."""
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
        """BYPASS 모드에서 읽기 도구가 허용되는지 검증한다."""
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
        """단순 질문 → DIRECT 전략 (1-pass) 선택을 검증한다."""
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
        """중간 복잡도 질문 → HIDDEN_COT 전략 (2-pass) 선택을 검증한다."""
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
        """동일 메시지 2회 호출 시 캐시 히트를 검증한다."""
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
        """ComplexityAssessor가 0.0~1.0 범위의 스코어를 반환하는지 검증한다."""
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
        """ShortTermMemory가 Redis 없이 인메모리로 동작하는지 검증한다."""
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
        """LongTermMemory가 PostgreSQL 없이 인메모리로 동작하는지 검증한다."""
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
        """MemoryManager의 턴 시작/종료 라이프사이클을 검증한다."""
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
        """ImportanceAssessor가 키워드 기반 중요도를 올바르게 평가하는지 검증한다."""
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
