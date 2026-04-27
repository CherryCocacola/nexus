"""
Agent 도구 — 서브 에이전트 실행.

독립적인 서브 에이전트를 생성하여 복잡한 작업을 위임한다.
서브 에이전트는 자신만의 QueryEngine(대화 컨텍스트)을 가지며,
부모 에이전트의 메시지와 격리된다.

v7.0 Phase 9 재설계: AgentDefinition 기반 호출을 지원한다.
  - subagent_type="scout" 호출 → AgentRegistry에서 SCOUT_AGENT를 조회
  - allowed_tools 기반으로 서브 에이전트 도구를 좁힘
  - model_override="scout" → ScoutModelProvider 사용 (CPU 4B 모델)
  - max_turns는 AgentDefinition에 선언된 값 사용

하위 호환:
  subagent_type을 주지 않으면 기존 방식대로 prompt + description으로 동작한다
  (DISALLOWED_TOOLS_FOR_AGENTS만 필터링, 부모 모델 재사용).

핵심 안전 규칙:
  - 서브 에이전트는 Agent 도구를 사용할 수 없다 (무한 재귀 방지)
  - DISALLOWED_TOOLS에 정의된 위험한 도구는 서브 에이전트에서 제외된다
  - 에어갭 환경이므로 외부 API 호출 없이 로컬 vLLM 서버만 사용한다
"""

from __future__ import annotations

import hashlib
import logging
import time
from collections import OrderedDict
from typing import Any, ClassVar

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

    사용 시나리오 (Worker가 판단해서 호출):
      - 복잡한 파일 탐색 (Scout 서브에이전트)
      - 전문 영역별 작업 (향후 code-reviewer, sql-explorer 등)
      - 병렬 독립 작업
      - 실험적 변경을 격리된 환경에서 시도

    두 가지 호출 방식:
      1. subagent_type 지정 (권장) — AgentRegistry의 AgentDefinition 사용
         Agent(prompt="...", subagent_type="scout")
      2. description 지정 (하위 호환) — ad-hoc 서브에이전트
         Agent(prompt="...", description="임시 역할 설명")
    """

    # 클래스 레벨 호출 통계 — Ch 17 /metrics가 참조한다.
    # agent_name(또는 "default") → {"calls": int, "total_latency_ms": float}
    _stats: ClassVar[dict[str, dict[str, Any]]] = {}

    # ─────────────────────────────────────────────
    # 결과 캐시 (2026-04-22, v0.14.2)
    # ─────────────────────────────────────────────
    # Scout 호출은 CPU 4B 모델 + llama.cpp라 한 번에 ~30초가 걸린다.
    # 같은 질문이 짧은 시간 안에 반복되면 (예: 사용자가 동일 문서를 여러 턴에 걸쳐
    # 질문하거나 Worker가 재시도하는 경우) 매번 Scout를 다시 돌리는 건 낭비다.
    # 여기서는 (subagent_type, prompt, system_prompt, 도구 조합, max_turns)를
    # 해시 키로 하여 결과 텍스트를 짧은 TTL 동안만 캐시한다.
    #
    # 주의:
    #   - subagent_type이 있는 호출만 캐시한다. 하위 호환 ad-hoc 경로는
    #     description이 매번 다를 수 있어 캐시 효과가 낮고 일관성 위험만 크다.
    #   - 에러 결과는 캐시하지 않는다 (일시적 장애가 장기간 캐시되면 안 됨).
    #   - TTL은 짧게(기본 5분). 도구가 파일 시스템에 의존하므로 길면 stale 위험.
    _cache: ClassVar[OrderedDict[str, tuple[str, int, float]]] = OrderedDict()
    _cache_stats: ClassVar[dict[str, int]] = {
        "hits": 0,
        "misses": 0,
        "stored": 0,
        "evicted": 0,
    }
    _cache_enabled: ClassVar[bool] = True
    _cache_ttl_seconds: ClassVar[float] = 300.0  # 5분
    _cache_max_entries: ClassVar[int] = 64

    @classmethod
    def _compute_cache_key(
        cls,
        subagent_type: str,
        prompt: str,
        system_prompt: str,
        tool_names: list[str],
        max_turns: int,
    ) -> str:
        """호출 입력으로부터 결정적 캐시 키를 계산한다.

        `\\0` 구분자로 필드를 이어 붙여 입력 경계를 명확히 한다.
        도구 순서는 정렬하여 동일 조합이 같은 키로 귀결되게 한다.
        """
        payload = "\0".join(
            [
                subagent_type,
                prompt,
                system_prompt,
                ",".join(sorted(tool_names)),
                str(max_turns),
            ]
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @classmethod
    def _cache_lookup(cls, key: str) -> tuple[str, int] | None:
        """TTL 안의 유효한 엔트리를 반환. 없거나 만료면 None."""
        entry = cls._cache.get(key)
        if entry is None:
            cls._cache_stats["misses"] += 1
            return None
        text, turns, stored_at = entry
        if time.monotonic() - stored_at > cls._cache_ttl_seconds:
            # 만료 — 제거 후 miss로 취급
            del cls._cache[key]
            cls._cache_stats["misses"] += 1
            return None
        # LRU 갱신 — 최근 사용으로 이동
        cls._cache.move_to_end(key)
        cls._cache_stats["hits"] += 1
        return text, turns

    @classmethod
    def _cache_store(cls, key: str, text: str, turns: int) -> None:
        """결과를 캐시에 저장. 크기 초과 시 가장 오래된 항목 제거."""
        cls._cache[key] = (text, turns, time.monotonic())
        cls._cache.move_to_end(key)
        cls._cache_stats["stored"] += 1
        while len(cls._cache) > cls._cache_max_entries:
            cls._cache.popitem(last=False)
            cls._cache_stats["evicted"] += 1

    @classmethod
    def get_cache_stats(cls) -> dict[str, int]:
        """/metrics 등이 조회할 캐시 통계."""
        return {**cls._cache_stats, "size": len(cls._cache)}

    @classmethod
    def reset_cache(cls) -> None:
        """테스트 격리용 — 캐시와 통계를 모두 비운다."""
        cls._cache.clear()
        for k in cls._cache_stats:
            cls._cache_stats[k] = 0

    # ═══ 1. Identity ═══

    @property
    def name(self) -> str:
        return "Agent"

    @property
    def description(self) -> str:
        # 간결한 영문 설명 — 토큰 절약을 위해 Worker에게는 시스템 프롬프트에서
        # 별도로 사용 가능한 subagent 목록을 동적 주입한다.
        return (
            "Delegate a task to a sub-agent. "
            "Use subagent_type to select a specialized agent (e.g. 'scout' for "
            "read-only exploration). Prefer subagent_type over ad-hoc description."
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
        # v7.0: subagent_type이 주요 파라미터, description은 하위 호환용
        return {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "Task for the sub-agent",
                },
                "subagent_type": {
                    "type": "string",
                    "description": (
                        "Sub-agent name from the registry (e.g. 'scout'). "
                        "When set, description is ignored."
                    ),
                },
                "description": {
                    "type": "string",
                    "description": (
                        "Fallback role description for ad-hoc sub-agent. "
                        "Required only when subagent_type is not set."
                    ),
                },
            },
            "required": ["prompt"],
        }

    # ═══ 3. Behavior Flags ═══

    @property
    def timeout_seconds(self) -> float:
        """서브 에이전트는 최대 10분까지 실행 허용."""
        return 600.0

    # ═══ 5. Lifecycle ═══

    def validate_input(self, input_data: dict[str, Any]) -> str | None:
        """prompt 필수. subagent_type이 없으면 description도 필수."""
        prompt = input_data.get("prompt", "")
        if not prompt or not prompt.strip():
            return "prompt는 비어 있을 수 없습니다."

        subagent_type = input_data.get("subagent_type", "")
        description = input_data.get("description", "")
        if not subagent_type and (not description or not description.strip()):
            return "subagent_type 또는 description 중 하나는 필요합니다."

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
          1. subagent_type이 있으면 AgentRegistry에서 AgentDefinition 조회
          2. 없으면 description 기반 ad-hoc 모드
          3. 도구 필터링 + 모델 선택 + 시스템 프롬프트 결정
          4. 독립된 QueryEngine 생성 후 submit_message 실행
          5. 텍스트 응답 수집 및 통계 누적
        """
        prompt = input_data["prompt"]
        subagent_type = input_data.get("subagent_type") or None

        # ① 서브 에이전트 설정 결정
        try:
            config = self._resolve_agent_config(subagent_type, input_data, context)
        except _AgentConfigError as e:
            logger.warning("Agent: 설정 결정 실패: %s", e)
            return ToolResult.error(str(e))

        logger.info(
            "Agent: 서브에이전트 실행 시작 (type=%s, tools=%d개, model_override=%s)",
            subagent_type or "ad-hoc",
            len(config.tools),
            config.model_override_label,
        )

        # ② 캐시 조회 — subagent_type이 있을 때만 시도한다.
        # ad-hoc(description) 호출은 description이 매번 미묘하게 달라질 수 있어
        # 캐시 일관성이 떨어진다. Scout 같은 정식 서브에이전트만 캐시 대상.
        cache_key: str | None = None
        stats_key = subagent_type or "ad-hoc"
        if subagent_type and self._cache_enabled:
            cache_key = self._compute_cache_key(
                subagent_type=subagent_type,
                prompt=prompt,
                system_prompt=config.system_prompt,
                tool_names=[t.name for t in config.tools],
                max_turns=config.max_turns,
            )
            cached = self._cache_lookup(cache_key)
            if cached is not None:
                text, turns = cached
                logger.info(
                    "Agent: 캐시 히트 (type=%s, %d자, %d턴) — Scout 재실행 생략",
                    subagent_type, len(text), turns,
                )
                # 캐시 히트도 관측성을 위해 latency=0으로 stats에 기록한다.
                self._record_stats(stats_key, 0.0)
                return ToolResult.success(text, turns=turns, cache_hit=True)

        # ③ 실행 + 통계 누적
        # 장기 실행 임계(60초)를 넘기면 완료 로그를 WARNING으로 승격하여
        # 운영자가 로그에서 hang 징후를 빠르게 발견할 수 있게 한다.
        # Scout(CPU 4B)는 평균 30~50초이므로 60초는 이상 징후 경계선으로 적절.
        long_running_threshold_s = 60.0
        start_time = time.monotonic()
        logger.info(
            "Agent: 실행 중 (type=%s, prompt_len=%d)",
            stats_key, len(prompt),
        )
        try:
            result_text, turns = await self._run_subagent(prompt, config, context)
            elapsed_ms = (time.monotonic() - start_time) * 1000.0
            self._record_stats(stats_key, elapsed_ms)
            elapsed_s = elapsed_ms / 1000.0

            if elapsed_s >= long_running_threshold_s:
                logger.warning(
                    "Agent: 장기 실행 완료 (type=%s, %.1fs, %d턴, %d자) — "
                    "임계 %ds 초과",
                    stats_key, elapsed_s, turns, len(result_text),
                    int(long_running_threshold_s),
                )
            else:
                logger.info(
                    "Agent: 완료 (type=%s, %.1fs, %d턴, %d자)",
                    stats_key, elapsed_s, turns, len(result_text),
                )
            # 성공한 결과만 캐시한다 (에러 결과는 일시 장애일 수 있어 캐시 금지).
            if cache_key is not None:
                self._cache_store(cache_key, result_text, turns)
            return ToolResult.success(result_text, turns=turns)
        except Exception as e:
            elapsed_ms = (time.monotonic() - start_time) * 1000.0
            self._record_stats(stats_key, elapsed_ms)
            logger.error(
                "Agent: 실행 실패 (type=%s, %.1fs): %s",
                stats_key, elapsed_ms / 1000.0, e,
            )
            return ToolResult.error(f"서브 에이전트 실행 실패: {e}")

    # ─── 내부 헬퍼: 설정 결정 ───────────────────────
    def _resolve_agent_config(
        self,
        subagent_type: str | None,
        input_data: dict[str, Any],
        context: ToolUseContext,
    ) -> _AgentConfig:
        """
        subagent_type 또는 description을 기반으로 서브에이전트 실행 설정을 만든다.

        두 경로:
          A) subagent_type 있음 → AgentRegistry에서 AgentDefinition 조회 후 사용
          B) subagent_type 없음 → description + 전체 도구(DISALLOWED 필터)
        """
        all_tools: list[BaseTool] = context.options.get("available_tools", [])
        parent_model_provider = context.options.get("model_provider")

        if subagent_type:
            registry = context.options.get("agent_registry")
            if registry is None:
                raise _AgentConfigError(
                    "agent_registry가 context.options에 없습니다."
                )
            agent_def = registry.get(subagent_type)
            if agent_def is None:
                # 안내 메시지에 사용 가능 목록을 포함해 Worker가 재시도할 수 있게 한다
                available = ", ".join(registry.list_names()) or "(none)"
                raise _AgentConfigError(
                    f"Unknown subagent_type: '{subagent_type}'. "
                    f"Available: {available}"
                )

            # 모델 선택 — model_override가 "scout"이면 ScoutModelProvider 사용
            model_provider = self._resolve_model_provider(
                agent_def.model_override, context, parent_model_provider
            )
            # 도구 필터링 — allowed_tools에 명시된 것만 + DISALLOWED 제외
            filtered_tools = [
                t for t in all_tools
                if t.name in agent_def.allowed_tools
                and t.name not in DISALLOWED_TOOLS_FOR_AGENTS
            ]

            return _AgentConfig(
                system_prompt=agent_def.system_prompt,
                tools=filtered_tools,
                max_turns=agent_def.max_turns,
                model_provider=model_provider,
                model_override_label=agent_def.model_override or "parent",
            )

        # B) ad-hoc 경로 (하위 호환)
        description = input_data.get("description", "").strip()
        if not description:
            raise _AgentConfigError(
                "subagent_type 또는 description 중 하나는 필요합니다."
            )
        filtered_tools = [
            t for t in all_tools if t.name not in DISALLOWED_TOOLS_FOR_AGENTS
        ]
        return _AgentConfig(
            system_prompt=description,
            tools=filtered_tools,
            max_turns=10,
            model_provider=parent_model_provider,
            model_override_label="parent",
        )

    def _resolve_model_provider(
        self,
        override_name: str | None,
        context: ToolUseContext,
        parent_model_provider: Any,
    ) -> Any:
        """
        model_override 이름을 실제 ModelProvider로 해석한다.

        현재 지원: "scout" → context.options["scout_provider"]
        None/알 수 없음 → 부모 Worker 프로바이더 재사용
        """
        if override_name is None:
            return parent_model_provider
        if override_name == "scout":
            scout_provider = context.options.get("scout_provider")
            if scout_provider is None:
                raise _AgentConfigError(
                    "scout_provider가 context.options에 없습니다 "
                    "(Scout 서버 미연결 또는 TIER_M/L 환경)"
                )
            return scout_provider
        # 알 수 없는 override → 보수적으로 부모 재사용
        logger.warning(
            "Agent: 알 수 없는 model_override '%s', 부모 Worker 재사용",
            override_name,
        )
        return parent_model_provider

    # ─── 내부 헬퍼: 서브에이전트 실행 ──────────────────
    async def _run_subagent(
        self,
        prompt: str,
        config: _AgentConfig,
        parent_context: ToolUseContext,
    ) -> tuple[str, int]:
        """
        서브에이전트 QueryEngine을 만들어 prompt를 실행하고 텍스트를 수집한다.

        model_provider가 None이면 stub 응답을 반환한다 (테스트 환경 호환).
        """
        if config.model_provider is None:
            return (
                "[Agent] 모델 프로바이더가 설정되지 않았습니다 (stub 모드).",
                0,
            )

        from core.config import RoutingConfig
        from core.message import StreamEvent, StreamEventType
        from core.orchestrator.query_engine import QueryEngine

        sub_context = ToolUseContext(
            cwd=parent_context.cwd,
            session_id=f"sub-{parent_context.session_id}",
            agent_id=f"agent-{parent_context.tool_use_id}",
            parent_tool_use_id=parent_context.tool_use_id,
            permission_mode=parent_context.permission_mode,
            options=parent_context.options,
        )

        # 서브에이전트는 라우팅 비활성 — Scout는 자기 전용 프로바이더(ScoutModelProvider)를
        # 쓰는데 부모의 라우팅 프로필이 개입하면 엉뚱한 model_override("nexus-phase3" 등)가
        # Scout 서버(qwen3.5-4b만 서빙)로 주입되어 오작동한다 (2026-04-21 재진단).
        # enabled=False면 프로필 치환 없이 프로바이더 기본 설정으로 동작.
        sub_routing = RoutingConfig(enabled=False)

        engine = QueryEngine(
            model_provider=config.model_provider,
            tools=config.tools,
            context=sub_context,
            system_prompt=config.system_prompt,
            max_turns=config.max_turns,
            routing_config=sub_routing,
        )

        text_parts: list[str] = []
        async for event in engine.submit_message(prompt):
            if (
                isinstance(event, StreamEvent)
                and event.type == StreamEventType.TEXT_DELTA
                and event.text
            ):
                text_parts.append(event.text)

        result_text = "".join(text_parts) or "(서브 에이전트가 텍스트 응답을 생성하지 않았습니다)"
        return result_text, engine.total_turns

    # ─── 통계 (Ch 17) ──────────────────────────────
    def _record_stats(self, key: str, elapsed_ms: float) -> None:
        """
        서브에이전트 호출 통계를 누적한다.

        클래스 레벨 딕셔너리에 저장하여 /metrics 엔드포인트가
        AgentTool.get_stats()로 조회할 수 있게 한다.
        """
        entry = AgentTool._stats.setdefault(
            key, {"calls": 0, "total_latency_ms": 0.0}
        )
        entry["calls"] += 1
        entry["total_latency_ms"] += elapsed_ms

    @classmethod
    def get_stats(cls) -> dict[str, dict[str, Any]]:
        """
        모든 서브에이전트의 호출 통계를 반환한다.

        반환 형식:
          {
              "scout": {"calls": 3, "total_latency_ms": 99000, "avg_latency_ms": 33000},
              "ad-hoc": {...},
          }
        """
        result: dict[str, dict[str, Any]] = {}
        for key, data in cls._stats.items():
            calls = data["calls"]
            total = data["total_latency_ms"]
            avg = total / calls if calls > 0 else 0.0
            result[key] = {
                "calls": calls,
                "total_latency_ms": round(total, 2),
                "avg_latency_ms": round(avg, 2),
            }
        return result

    @classmethod
    def reset_stats(cls) -> None:
        """테스트 격리를 위한 통계 리셋."""
        cls._stats.clear()

    # ═══ 7. UI Hints ═══

    def get_progress_label(self, input_data: dict[str, Any]) -> str:
        subagent_type = input_data.get("subagent_type", "")
        if subagent_type:
            return f"Agent[{subagent_type}]: {input_data.get('prompt', '')[:40]}..."
        desc = input_data.get("description", "")
        if desc:
            return f"Agent: {desc[:40]}..."
        return "Running sub-agent..."

    def get_input_summary(self, input_data: dict[str, Any]) -> str:
        subagent_type = input_data.get("subagent_type", "")
        if subagent_type:
            return f"[{subagent_type}] {input_data.get('prompt', '')[:80]}"
        return input_data.get("description", "")[:100]


# ─────────────────────────────────────────────
# 내부 데이터 클래스 — 모듈 외부 노출 안 함
# ─────────────────────────────────────────────
class _AgentConfigError(Exception):
    """서브에이전트 설정 해석 실패."""


class _AgentConfig:
    """서브에이전트 실행에 필요한 해석된 설정 묶음."""

    __slots__ = (
        "system_prompt",
        "tools",
        "max_turns",
        "model_provider",
        "model_override_label",
    )

    def __init__(
        self,
        system_prompt: str,
        tools: list[BaseTool],
        max_turns: int,
        model_provider: Any,
        model_override_label: str,
    ) -> None:
        self.system_prompt = system_prompt
        self.tools = tools
        self.max_turns = max_turns
        self.model_provider = model_provider
        self.model_override_label = model_override_label
