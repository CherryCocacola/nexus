"""
Agent 도구 — 서브 에이전트 실행.

독립적인 서브 에이전트를 생성하여 복잡한 작업을 위임한다.
서브 에이전트는 자신만의 QueryEngine(대화 컨텍스트)을 가지며,
부모 에이전트의 메시지와 격리된다.

핵심 안전 규칙:
  - 서브 에이전트는 Agent 도구를 사용할 수 없다 (무한 재귀 방지)
  - DISALLOWED_TOOLS에 정의된 위험한 도구는 서브 에이전트에서 제외된다
  - 실제 QueryEngine은 아직 구현되지 않았으므로 stub으로 동작한다

에어갭 환경이므로 외부 API 호출 없이 로컬 vLLM 서버만 사용한다.
"""

from __future__ import annotations

import logging
from typing import Any

from core.tools.base import (
    BaseTool,
    PermissionBehavior,
    PermissionResult,
    ToolResult,
    ToolUseContext,
)

logger = logging.getLogger("nexus.tools.agent")

# 서브 에이전트가 사용할 수 없는 도구 목록 (무한 재귀 및 위험 방지)
DISALLOWED_TOOLS_FOR_AGENTS = frozenset(
    {
        "Agent",  # 재귀 에이전트 호출 방지
        "TaskCreate",  # 태스크 생성은 부모 에이전트만
        "TaskStop",  # 태스크 중지는 부모 에이전트만
        "TrainingTool",  # 학습은 부모 에이전트만
        "CheckpointTool",  # 체크포인트는 부모 에이전트만
    }
)


class AgentTool(BaseTool):
    """
    서브 에이전트를 생성하여 작업을 위임하는 도구.

    사용 시나리오:
      - 복잡한 작업을 독립된 컨텍스트에서 수행해야 할 때
      - 병렬로 여러 작업을 처리해야 할 때
      - 실험적 변경을 격리된 환경에서 시도할 때

    서브 에이전트는 부모와 messages[]가 격리되어 있어
    서로의 대화 컨텍스트를 오염시키지 않는다.
    """

    # ═══ 1. Identity ═══

    @property
    def name(self) -> str:
        return "Agent"

    @property
    def description(self) -> str:
        return (
            "서브 에이전트를 생성하여 작업을 위임합니다. "
            "각 에이전트는 독립된 대화 컨텍스트를 가지며, "
            "복잡한 작업을 격리된 환경에서 수행합니다."
        )

    @property
    def group(self) -> str:
        return "agent"

    @property
    def aliases(self) -> list[str]:
        return ["SubAgent", "Delegate"]

    # ═══ 2. Schema ═══

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "서브 에이전트에게 전달할 프롬프트 (수행할 작업 설명)",
                },
                "description": {
                    "type": "string",
                    "description": "에이전트의 역할/목적 설명 (시스템 프롬프트에 포함)",
                },
            },
            "required": ["prompt", "description"],
        }

    # ═══ 3. Behavior Flags ═══
    # 에이전트는 장시간 실행될 수 있으므로 타임아웃을 늘린다

    @property
    def timeout_seconds(self) -> float:
        """서브 에이전트는 최대 10분까지 실행 허용."""
        return 600.0

    # ═══ 5. Lifecycle ═══

    def validate_input(self, input_data: dict[str, Any]) -> str | None:
        """prompt와 description이 비어 있는지 검증한다."""
        prompt = input_data.get("prompt", "")
        if not prompt or not prompt.strip():
            return "prompt는 비어 있을 수 없습니다."

        description = input_data.get("description", "")
        if not description or not description.strip():
            return "description은 비어 있을 수 없습니다."

        return None

    async def check_permissions(
        self, input_data: dict[str, Any], context: ToolUseContext
    ) -> PermissionResult:
        """
        Agent 도구는 기본적으로 허용한다.
        서브 에이전트 내부에서 실행되는 개별 도구는
        각자의 check_permissions에서 권한을 확인한다.
        """
        return PermissionResult(behavior=PermissionBehavior.ALLOW)

    async def call(self, input_data: dict[str, Any], context: ToolUseContext) -> ToolResult:
        """
        서브 에이전트를 생성하고 작업을 실행한다.

        처리 순서:
          1. 에이전트 설정 구성 (허용 도구 필터링)
          2. 독립된 QueryEngine 생성
          3. 프롬프트 전달 및 실행
          4. 에이전트 응답 수집 및 반환

        현재 상태: QueryEngine이 아직 구현되지 않았으므로 stub 동작.
        Phase 3.0 (Orchestrator) 완성 후 실제 연동 필요.
        """
        prompt = input_data["prompt"]
        description = input_data["description"]

        logger.info(
            "Agent: starting sub-agent (description=%s, prompt=%s...)",
            description[:50],
            prompt[:50],
        )

        # ① 사용 가능한 도구를 필터링 — 재귀/위험 도구 제외
        # context.options에서 도구 리스트를 가져오거나 빈 리스트 사용
        all_tools: list[BaseTool] = context.options.get("available_tools", [])
        agent_tools = [
            t for t in all_tools if t.name not in DISALLOWED_TOOLS_FOR_AGENTS
        ]

        # ② 서브 에이전트용 독립 QueryEngine 생성
        try:
            from core.orchestrator.query_engine import QueryEngine

            # 모델 프로바이더를 context.options에서 가져온다
            model_provider = context.options.get("model_provider")
            if model_provider is None:
                # 모델 프로바이더가 없으면 stub 모드로 동작
                return ToolResult.success(
                    f"[Agent] 모델 프로바이더가 설정되지 않았습니다.\n"
                    f"설명: {description}\n프롬프트: {prompt}",
                    stub=True,
                )

            # 서브 에이전트용 컨텍스트 (부모 도구 ID를 상위 참조로 설정)
            sub_context = ToolUseContext(
                cwd=context.cwd,
                session_id=f"sub-{context.session_id}",
                agent_id=f"agent-{context.tool_use_id}",
                parent_tool_use_id=context.tool_use_id,
                permission_mode=context.permission_mode,
                options=context.options,
            )

            # 독립된 QueryEngine 생성 — messages[]가 부모와 격리됨
            engine = QueryEngine(
                model_provider=model_provider,
                tools=agent_tools,
                context=sub_context,
                system_prompt=description,
                max_turns=10,  # 서브 에이전트는 10턴 제한
            )

            # ③ 프롬프트를 전달하고 텍스트 응답을 수집한다
            from core.message import StreamEvent, StreamEventType

            result_parts: list[str] = []
            async for event in engine.submit_message(prompt):
                if (
                    isinstance(event, StreamEvent)
                    and event.type == StreamEventType.TEXT_DELTA
                    and event.text
                ):
                    result_parts.append(event.text)

            result_text = "".join(result_parts)
            if not result_text:
                result_text = "(서브 에이전트가 텍스트 응답을 생성하지 않았습니다)"

            logger.info(
                "Agent: sub-agent completed (%d turns, %d chars)",
                engine.total_turns,
                len(result_text),
            )
            return ToolResult.success(result_text, turns=engine.total_turns)

        except Exception as e:
            logger.error("Agent: sub-agent execution failed: %s", e)
            return ToolResult.error(f"서브 에이전트 실행 실패: {e}")

    # ═══ 7. UI Hints ═══

    def get_progress_label(self, input_data: dict[str, Any]) -> str:
        desc = input_data.get("description", "")
        if desc:
            return f"Agent: {desc[:40]}..."
        return "Running sub-agent..."

    def get_input_summary(self, input_data: dict[str, Any]) -> str:
        return input_data.get("description", "")[:100]
