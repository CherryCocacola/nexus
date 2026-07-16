"""
Agent 도구 — 서브 에이전트(sub-agent) 실행 구현.

[이 파일이 하는 일]
Worker(부모 에이전트)가 복잡한 작업을 통째로 처리하기 어려울 때, 그 일부를
독립적인 "서브 에이전트"에게 위임할 수 있게 해 주는 도구다. 서브 에이전트는
자신만의 QueryEngine(= 완전히 별개의 대화 컨텍스트/메시지 히스토리)을 가지므로,
부모의 대화 내용과 섞이지 않고 격리된 상태에서 자기 일만 처리한 뒤 결과 텍스트만
부모에게 돌려준다. 대표 예: 파일 탐색 전담 "scout" 서브 에이전트.

[주요 구성 요소]
  - DISALLOWED_TOOLS_FOR_AGENTS : 서브 에이전트에게 절대 넘기지 않는 위험 도구 집합
  - AgentTool                   : BaseTool 구현체 — 실제 도구 본체(호출 진입점은 call())
  - _AgentConfig / _AgentConfigError : 서브 에이전트 실행 설정을 담는 내부 전용 클래스
  - AgentTool._stats / ._cache  : 관측성(통계)·성능(결과 캐시)용 클래스 레벨 상태

[호출/노출 관계]
  - Worker가 도구 이름 "Agent"로 호출한다(별칭: SubAgent, Delegate).
  - 서브 에이전트 실행 시 core.orchestrator.query_engine.QueryEngine을 새로 생성한다.
  - 통계(get_stats)·캐시 통계(get_cache_stats)는 Ch 17 /metrics 엔드포인트가 조회한다.

[v7.0 Phase 9 재설계] AgentDefinition 기반 호출을 지원한다.
  - subagent_type="scout" 호출 → AgentRegistry에서 SCOUT_AGENT 정의를 조회
  - allowed_tools 기반으로 서브 에이전트가 쓸 수 있는 도구를 좁힘
  - model_override="scout" → ScoutModelProvider 사용 (CPU 4B 모델)
  - max_turns는 AgentDefinition에 선언된 값을 사용

[하위 호환]
  subagent_type을 주지 않으면 기존 방식대로 prompt + description으로 동작한다
  (DISALLOWED_TOOLS_FOR_AGENTS만 필터링, 부모 모델을 그대로 재사용).

[핵심 안전 규칙]
  - 서브 에이전트는 Agent 도구 자신을 다시 사용할 수 없다 (무한 재귀 방지)
  - DISALLOWED_TOOLS에 정의된 위험한 도구는 서브 에이전트에서 제외된다
  - 에어갭 환경이므로 외부 API 호출 없이 로컬 vLLM 서버만 사용한다

작성자: 이현수 / 작성일: 2026-07-05
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

# 서브 에이전트가 사용할 수 없는 도구 목록 (무한 재귀 및 위험 방지).
# frozenset(불변 집합)으로 둔 이유: 이 목록은 실행 중 절대 바뀌면 안 되고,
# 도구 필터링 시 "t.name not in DISALLOWED..." 형태의 빠른 멤버십 검사에 쓰이기 때문.
DISALLOWED_TOOLS_FOR_AGENTS = frozenset(
    {
        "Agent",  # 재귀 에이전트 호출 방지
        "TaskCreate",  # 태스크 생성은 부모 에이전트만
        "TaskStop",  # 태스크 중지는 부모 에이전트만
        "TrainingTool",  # 학습은 부모 에이전트만
        "CheckpointTool",  # 체크포인트는 부모 에이전트만
    }
)

# ad-hoc 서브에이전트(경로 B)의 시스템 프롬프트에 자동으로 덧붙이는 "출력 규약".
# 왜 필요한가: ad-hoc 경로는 사용자가 준 description을 그대로 프롬프트로 쓰고 위험 도구만
# 뺀 전체 도구 풀(DocumentExport 등 산출물 도구 포함)을 받는다. 그런데 description에는
# 출력 형식 규칙이 없어, 모델이 "DocumentExport의 content 인자를 채워야 합니다" 같은 도구
# 내부 서술을 노출하거나 생성물 본문을 답변에 통째로 도배할 수 있다. Claude 앱/웹처럼
# 도구는 조용히 부르고 결과만 간결히 보여주도록, 모든 ad-hoc 서브에이전트에 공통 규약을 강제한다.
_SUBAGENT_OUTPUT_CONTRACT = (
    "\n\n---\n"
    "## 출력 규약 (반드시 준수)\n"
    "- 도구 사용 자체를 설명하지 마라. 도구 이름·인자·'인자를 채운다' 같은 내부 동작을 "
    "답변 텍스트에 쓰지 말고, 도구는 조용히 호출하라.\n"
    "- 파일/문서를 생성할 때는 생성물 본문 전체를 답변에 다시 쓰지 말고 해당 도구의 "
    "content(본문) 인자에 담아라. 완료 후에는 1~2문장의 짧은 확인만 남겨라.\n"
    "- 도구나 다른 에이전트가 돌려준 결과 원문을 그대로 복사해 붙여넣지 마라. "
    "핵심만 간결히 종합하라.\n"
    "- 사고 과정을 출력하지 마라."
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
    # ClassVar이므로 인스턴스가 아니라 클래스 전체에서 하나만 공유된다
    # (AgentTool을 몇 번 생성하든 통계는 한 곳에 누적된다).
    # 구조: agent_name(또는 "ad-hoc") → {"calls": int, "total_latency_ms": float}
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
    # OrderedDict를 쓰는 이유: LRU(가장 오래 안 쓴 항목 제거) 정책을 쉽게 구현하기 위함.
    # 값은 (결과 텍스트, 턴 수, 저장 시각[monotonic 초]) 튜플이다.
    _cache: ClassVar[OrderedDict[str, tuple[str, int, float]]] = OrderedDict()
    # 캐시 동작 관측용 카운터: 적중/불발/저장/축출 횟수.
    _cache_stats: ClassVar[dict[str, int]] = {
        "hits": 0,
        "misses": 0,
        "stored": 0,
        "evicted": 0,
    }
    _cache_enabled: ClassVar[bool] = True  # 전역 on/off 스위치(테스트 등에서 끌 수 있음)
    _cache_ttl_seconds: ClassVar[float] = 300.0  # 항목 유효 기간 5분 (지나면 stale 처리)
    _cache_max_entries: ClassVar[int] = 64  # 최대 보관 개수(초과 시 가장 오래된 것 축출)

    @classmethod
    def _compute_cache_key(
        cls,
        subagent_type: str,
        prompt: str,
        system_prompt: str,
        tool_names: list[str],
        max_turns: int,
    ) -> str:
        """호출 입력으로부터 결정적(deterministic) 캐시 키를 계산한다.

        같은 입력이면 언제 호출해도 반드시 같은 키가 나와야 캐시가 제대로 맞는다.
        그래서 입력 필드들을 이어 붙인 뒤 SHA-256 해시를 키로 쓴다.

        구현 포인트:
          - `\\0`(널 문자) 구분자로 필드를 이어 붙인다. 일반 문자로는 만들 수 없는
            경계라, 예컨대 prompt 끝과 system_prompt 시작이 뒤섞여 오인되는 것을 막는다.
          - 도구 목록은 sorted()로 정렬한다. 순서만 다르고 구성은 같은 도구 조합이
            서로 다른 키로 갈라지지 않게(= 같은 키로 귀결되게) 하기 위함이다.

        매개변수:
          subagent_type, prompt, system_prompt, tool_names, max_turns — 결과에 영향을
          주는 모든 입력. 하나라도 다르면 다른 키가 되어 캐시가 섞이지 않는다.
        반환: 16진수 SHA-256 다이제스트 문자열.
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
        """TTL(유효 기간) 안의 살아 있는 엔트리를 반환. 없거나 만료면 None.

        반환값이 있으면 (결과 텍스트, 턴 수) 튜플, 없으면 None이다.
        조회 성공/실패는 _cache_stats의 hits/misses에 그대로 집계된다.
        """
        entry = cls._cache.get(key)
        if entry is None:
            # 키 자체가 없음 → 불발(miss)
            cls._cache_stats["misses"] += 1
            return None
        text, turns, stored_at = entry
        # 저장 후 경과 시간이 TTL을 넘겼는지 검사.
        # time.monotonic()은 시스템 시계 변경에 영향받지 않는 단조 증가 시계라
        # 경과 시간 측정에 안전하다.
        if time.monotonic() - stored_at > cls._cache_ttl_seconds:
            # 만료 — 제거 후 miss로 취급
            del cls._cache[key]
            cls._cache_stats["misses"] += 1
            return None
        # 살아 있는 항목 → LRU 갱신(가장 최근 사용 위치로 이동)해 축출 대상에서 미룬다
        cls._cache.move_to_end(key)
        cls._cache_stats["hits"] += 1
        return text, turns

    @classmethod
    def _cache_store(cls, key: str, text: str, turns: int) -> None:
        """결과를 캐시에 저장. 최대 크기를 넘으면 가장 오래된 항목부터 축출한다.

        저장 시각(time.monotonic())을 함께 넣어 두어야 나중에 TTL 만료를 판정할 수 있다.
        """
        cls._cache[key] = (text, turns, time.monotonic())
        # 방금 저장한 항목을 맨 뒤(가장 최근)로 보내 LRU 순서를 맞춘다.
        cls._cache.move_to_end(key)
        cls._cache_stats["stored"] += 1
        # 최대 개수를 초과하면 맨 앞(last=False = 가장 오래된 항목)부터 제거한다.
        while len(cls._cache) > cls._cache_max_entries:
            cls._cache.popitem(last=False)
            cls._cache_stats["evicted"] += 1

    @classmethod
    def get_cache_stats(cls) -> dict[str, int]:
        """/metrics 등이 조회할 캐시 통계 스냅샷을 반환한다.

        기존 카운터(hits/misses/stored/evicted)에 현재 보관 개수 size를 덧붙여
        새 dict로 만들어 반환한다(원본 _cache_stats를 노출하지 않기 위한 복사).
        """
        return {**cls._cache_stats, "size": len(cls._cache)}

    @classmethod
    def reset_cache(cls) -> None:
        """테스트 격리용 — 캐시 내용과 통계 카운터를 모두 0으로 되돌린다.

        테스트 간에 이전 실행의 캐시가 남아 결과가 오염되는 것을 막기 위해 쓴다.
        """
        cls._cache.clear()
        for k in cls._cache_stats:
            cls._cache_stats[k] = 0

    # ═══ 1. Identity(도구 식별 정보) ═══
    # BaseTool이 요구하는 이름/설명/그룹/별칭. 레지스트리 등록과 모델 노출에 쓰인다.

    @property
    def name(self) -> str:
        # 모델(Worker)이 tool_calls에서 이 이름으로 호출한다. 절대 바뀌면 안 되는 식별자.
        return "Agent"

    @property
    def description(self) -> str:
        # 모델에게 보여줄 도구 설명. 일부러 짧은 영문으로 유지한다 — 토큰을 아끼기 위해
        # "사용 가능한 subagent 목록" 같은 상세 정보는 여기 넣지 않고, 시스템 프롬프트에서
        # 별도로 동적 주입한다.
        return (
            "Delegate a task to a sub-agent. "
            "Use subagent_type to select a specialized agent (e.g. 'scout' for "
            "read-only exploration). Prefer subagent_type over ad-hoc description."
        )

    @property
    def group(self) -> str:
        # 도구 분류용 그룹명. UI/권한에서 같은 성격의 도구를 묶는 데 쓴다.
        return "agent"

    @property
    def aliases(self) -> list[str]:
        # 별칭 — 레지스트리가 이 이름들로도 같은 도구를 조회할 수 있게 한다.
        return ["SubAgent", "Delegate"]

    # ═══ 2. Schema(입력 스키마) ═══
    # 모델이 이 도구를 호출할 때 넘겨야 하는 인자의 JSON Schema 정의.

    @property
    def input_schema(self) -> dict[str, Any]:
        # v7.0 기준: subagent_type이 주요 파라미터, description은 하위 호환용 폴백.
        # required는 prompt 하나뿐 — subagent_type/description 둘 중 최소 하나가 필요한
        # 규칙은 스키마만으로 표현하기 어려워 validate_input()에서 별도로 검사한다.
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

    # ═══ 3. Behavior Flags(동작 플래그) ═══

    @property
    def timeout_seconds(self) -> float:
        """이 도구(= 서브 에이전트 한 번 실행)의 최대 허용 시간.

        서브 에이전트는 여러 턴에 걸쳐 도구를 돌릴 수 있어 오래 걸린다. 특히 Scout는
        CPU 4B 모델이라 느리므로 넉넉히 10분(600초)까지 허용한다. 이 시간을 넘기면
        상위 실행 파이프라인이 타임아웃으로 강제 중단한다.
        """
        return 600.0

    # ═══ 5. Lifecycle(도구 실행 수명주기) ═══
    # 실행 순서: validate_input() → check_permissions() → call()

    def validate_input(self, input_data: dict[str, Any]) -> str | None:
        """호출 인자를 사전 검증한다. 문제가 없으면 None, 있으면 오류 메시지를 반환.

        규칙:
          - prompt는 반드시 있어야 하고 공백만 있으면 안 된다.
          - subagent_type이 없으면(ad-hoc 경로) description도 반드시 있어야 한다.
        반환값이 문자열이면 실행을 시작하지 않고 그 메시지를 오류로 돌려준다.
        """
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
        """Agent 도구 자체의 권한 판정 — 기본적으로 ALLOW(허용)한다.

        왜 무조건 허용해도 되나: 위임은 그 자체로 위험한 동작이 아니고, 실제 위험은
        서브 에이전트가 내부에서 돌리는 개별 도구(Read/Write/Bash 등)에서 발생한다.
        그 개별 도구들은 서브 에이전트의 QueryEngine 안에서 각자의 check_permissions를
        다시 통과하므로, 여기서 한 번 더 막을 필요가 없다(권한은 실행 지점에서 검사).
        """
        return PermissionResult(behavior=PermissionBehavior.ALLOW)

    async def call(self, input_data: dict[str, Any], context: ToolUseContext) -> ToolResult:
        """서브 에이전트를 생성하고 작업을 실행하는 메인 진입점.

        이 도구의 실질적인 본체다. 설정 결정 → (캐시 조회) → 실행 → 통계/캐시 저장을
        순서대로 수행하고, 최종적으로 ToolResult(성공/에러)를 돌려준다.

        처리 순서:
          1. subagent_type이 있으면 AgentRegistry에서 AgentDefinition을 조회
          2. 없으면 description 기반 ad-hoc(임시) 모드로 설정
          3. 도구 필터링 + 모델 선택 + 시스템 프롬프트를 _AgentConfig로 확정
          4. (정식 서브에이전트에 한해) 결과 캐시를 먼저 조회 — 있으면 재실행 생략
          5. 독립된 QueryEngine을 만들어 submit_message로 실행
          6. 텍스트 응답을 수집하고, 지연시간 통계와 결과 캐시를 갱신

        매개변수:
          input_data — prompt(필수), subagent_type/description(선택)
          context    — 부모의 실행 컨텍스트. 사용 가능 도구·모델 프로바이더·레지스트리가
                       context.options 안에 들어 있다.
        반환: ToolResult.success(결과 텍스트, turns=..) 또는 ToolResult.error(사유)
        """
        prompt = input_data["prompt"]
        # subagent_type이 빈 문자열("")이면 None으로 정규화 → 이후 분기를 단순하게 유지.
        subagent_type = input_data.get("subagent_type") or None

        # ① 서브 에이전트 설정 결정
        # 설정 해석 중 문제가 있으면 _AgentConfigError를 던지므로, 여기서 잡아
        # 에러 ToolResult로 변환한다(실행 자체는 시작하지 않는다).
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
        # cache_key는 나중에 성공 결과를 저장할 때 재사용하므로 여기서 잡아 둔다.
        # stats_key는 통계 집계 키(정식 이름 또는 "ad-hoc")로 캐시와 무관하게 항상 쓴다.
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
                # 캐시 적중 — 값비싼 서브 에이전트 재실행을 통째로 건너뛴다.
                text, turns = cached
                logger.info(
                    "Agent: 캐시 히트 (type=%s, %d자, %d턴) — Scout 재실행 생략",
                    subagent_type, len(text), turns,
                )
                # 캐시 히트도 "호출은 있었다"는 사실을 남기려고 latency=0으로 stats에 기록한다.
                self._record_stats(stats_key, 0.0)
                # cache_hit=True 플래그로 상위(예: /metrics, UI)가 캐시 응답임을 구분 가능.
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
            # 실제 서브 에이전트 실행 — 결과 텍스트와 소비한 턴 수를 받는다.
            result_text, turns = await self._run_subagent(prompt, config, context)
            elapsed_ms = (time.monotonic() - start_time) * 1000.0
            self._record_stats(stats_key, elapsed_ms)
            elapsed_s = elapsed_ms / 1000.0

            # 임계(60초) 초과면 WARNING, 아니면 INFO로 완료 로그를 남긴다.
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
            # 실패해도 소비한 시간은 통계에 남긴다(평균 지연 계산이 왜곡되지 않도록).
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
        """subagent_type 또는 description을 기반으로 서브에이전트 실행 설정을 만든다.

        결과로 _AgentConfig(시스템 프롬프트·도구 목록·최대 턴·모델 프로바이더·라벨)를
        돌려준다. 해석에 실패하면 _AgentConfigError를 던진다(call()이 잡아 처리).

        두 경로:
          A) subagent_type 있음 → AgentRegistry에서 AgentDefinition을 조회해 그대로 사용
          B) subagent_type 없음 → description을 프롬프트로 삼고, 위험 도구만 뺀 전체 도구 사용

        매개변수:
          subagent_type — 정식 서브에이전트 이름(없으면 ad-hoc 경로로 감)
          input_data    — description 등 원본 입력
          context       — available_tools / model_provider / agent_registry가 담긴 컨텍스트
        """
        # 부모가 넘겨준 "쓸 수 있는 전체 도구"와 "부모 모델 프로바이더"를 꺼낸다.
        all_tools: list[BaseTool] = context.options.get("available_tools", [])
        parent_model_provider = context.options.get("model_provider")

        # ── 경로 A: 정식 서브에이전트(subagent_type 지정) ──
        if subagent_type:
            registry = context.options.get("agent_registry")
            if registry is None:
                # 레지스트리가 주입되지 않았다면 정식 경로 자체를 진행할 수 없다.
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
            # 도구 필터링 — 정의의 allowed_tools에 명시된 것만 남기고, 그중에서도
            # DISALLOWED(위험/재귀 유발) 도구는 한 번 더 제외한다(이중 안전장치).
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

        # ── 경로 B: ad-hoc(임시) 경로 (하위 호환) ──
        # description을 곧 서브에이전트의 시스템 프롬프트로 사용한다.
        description = input_data.get("description", "").strip()
        if not description:
            raise _AgentConfigError(
                "subagent_type 또는 description 중 하나는 필요합니다."
            )
        # 정식 정의가 없으므로 allowed_tools 개념이 없다 → 위험 도구만 빼고 전부 허용.
        filtered_tools = [
            t for t in all_tools if t.name not in DISALLOWED_TOOLS_FOR_AGENTS
        ]
        # ad-hoc은 정의된 max_turns가 없으므로 보수적으로 10턴으로 고정하고,
        # 모델은 부모 것을 그대로 재사용한다(전용 프로바이더 없음).
        # description(사용자 지정 프롬프트) 뒤에 공통 출력 규약을 덧붙여, 도구 내부 서술·
        # 생성물 본문 도배를 막고 Claude식 간결 표시를 강제한다.
        return _AgentConfig(
            system_prompt=description + _SUBAGENT_OUTPUT_CONTRACT,
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
        """model_override 이름(문자열)을 실제 ModelProvider 객체로 해석한다.

        서브에이전트 정의에 어떤 모델을 쓸지 이름표만 들어 있으므로, 그 이름표를
        실제로 추론을 수행할 프로바이더 인스턴스로 바꿔 주는 역할이다.

        동작:
          - None            → 부모 Worker의 프로바이더를 그대로 재사용
          - "scout"         → context.options["scout_provider"](CPU 4B 전용) 사용.
                              단 미가용(TIER_M/L 등)이면 부모 Worker 프로바이더로 폴백한다.
          - 그 외 알 수 없는 이름 → 경고 로그 후 보수적으로 부모 프로바이더 재사용
        """
        if override_name is None:
            return parent_model_provider
        if override_name == "scout":
            scout_provider = context.options.get("scout_provider")
            if scout_provider is None:
                # Scout 전용 서버가 붙어 있지 않은 환경(TIER_M/L 등)에서는 죽이지 않고
                # 부모 Worker 모델로 폴백한다. TIER_M/L은 원래 Worker 단독 수행이 정상
                # 경로이므로(scout 위임 불필요), 폴백해도 결과 품질은 오히려 낫다.
                logger.warning(
                    "Agent: scout_provider 미가용(TIER_M/L 등) → 부모 Worker 모델로 폴백"
                )
                return parent_model_provider
            return scout_provider
        # 알 수 없는 override → 예외로 죽이지 않고 보수적으로 부모 모델로 폴백한다.
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
        """서브에이전트 전용 QueryEngine을 만들어 prompt를 실행하고 텍스트를 수집한다.

        핵심은 "부모와 완전히 분리된 새 대화 세션"을 여는 것이다. 새 세션은 자기만의
        메시지 히스토리를 갖고, submit_message로 흘러나오는 스트림 이벤트 중 텍스트
        조각(TEXT_DELTA)만 이어 붙여 최종 결과 문자열을 만든다.

        반환: (결과 텍스트, 사용한 총 턴 수) 튜플.
        model_provider가 None이면 실제 추론 없이 stub 응답을 돌려준다(테스트 환경 호환).
        """
        # 모델이 없으면(테스트/미구성 환경) 실행 대신 안내 문구를 반환한다.
        if config.model_provider is None:
            return (
                "[Agent] 모델 프로바이더가 설정되지 않았습니다 (stub 모드).",
                0,
            )

        # 지연 import(lazy import): 이 함수가 실제로 불릴 때만 QueryEngine 등을 들여온다.
        # 모듈 최상단에서 import하면 core.orchestrator ↔ core.tools 간 순환 참조가 생길 수
        # 있어 여기서 함수 지역으로 미룬다.
        from core.config import RoutingConfig
        from core.message import StreamEvent, StreamEventType
        from core.orchestrator.query_engine import QueryEngine

        # 서브에이전트 전용 컨텍스트 — session_id/agent_id에 접두어를 붙여 부모와 구분한다.
        # options는 부모 것을 그대로 물려줘야 도구/프로바이더 접근이 유지된다.
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

        # 스트림 이벤트를 하나씩 소비하며 텍스트 조각만 모은다.
        # (도구 사용/thinking 등 다른 이벤트는 여기서 무시 — 부모에는 최종 텍스트만 준다.)
        text_parts: list[str] = []
        async for event in engine.submit_message(prompt):
            if (
                isinstance(event, StreamEvent)
                and event.type == StreamEventType.TEXT_DELTA
                and event.text
            ):
                text_parts.append(event.text)

        # 조각을 이어 붙인다. 한 글자도 없으면(예: 도구만 쓰고 말로 답 안 함) 안내 문구로 대체.
        result_text = "".join(text_parts) or "(서브 에이전트가 텍스트 응답을 생성하지 않았습니다)"
        return result_text, engine.total_turns

    # ─── 통계 (Ch 17) ──────────────────────────────
    def _record_stats(self, key: str, elapsed_ms: float) -> None:
        """서브에이전트 호출 1건의 통계를 누적한다.

        key(서브에이전트 이름 또는 "ad-hoc")별로 누적 호출 수와 누적 지연시간을 더한다.
        인스턴스 메서드지만 저장은 클래스 레벨 _stats에 하므로 모든 호출이 한 곳에 쌓이고,
        /metrics 엔드포인트가 AgentTool.get_stats()로 그 값을 조회한다.

        매개변수:
          key        — 통계 집계 키(예: "scout", "ad-hoc")
          elapsed_ms — 이번 호출에 걸린 시간(밀리초). 캐시 히트일 땐 0.0으로 넣는다.
        """
        # setdefault: 해당 key가 처음이면 0으로 초기화한 항목을 만들어 준다.
        entry = AgentTool._stats.setdefault(
            key, {"calls": 0, "total_latency_ms": 0.0}
        )
        entry["calls"] += 1
        entry["total_latency_ms"] += elapsed_ms

    @classmethod
    def get_stats(cls) -> dict[str, dict[str, Any]]:
        """모든 서브에이전트의 호출 통계를 조회용으로 가공해 반환한다.

        내부 _stats에는 calls와 total_latency_ms만 쌓여 있으므로, 여기서 평균 지연시간
        (avg_latency_ms)을 계산해 덧붙여 준다. /metrics 응답에 그대로 실린다.

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
            # 0으로 나누기 방지 — 호출이 0이면 평균은 0.0으로 둔다.
            avg = total / calls if calls > 0 else 0.0
            result[key] = {
                "calls": calls,
                "total_latency_ms": round(total, 2),
                "avg_latency_ms": round(avg, 2),
            }
        return result

    @classmethod
    def reset_stats(cls) -> None:
        """테스트 격리를 위한 통계 리셋 — _stats를 통째로 비운다."""
        cls._stats.clear()

    # ═══ 7. UI Hints(터미널 표시용 라벨) ═══
    # 도구 실행 진행 상황이나 입력 요약을 사람이 보기 좋게 짧은 문자열로 만들어 준다.

    def get_progress_label(self, input_data: dict[str, Any]) -> str:
        # 실행 중 표시할 진행 라벨. subagent_type이 있으면 그 이름과 prompt 앞부분을,
        # 없으면 description을, 둘 다 없으면 기본 문구를 보여 준다.
        subagent_type = input_data.get("subagent_type", "")
        if subagent_type:
            return f"Agent[{subagent_type}]: {input_data.get('prompt', '')[:40]}..."
        desc = input_data.get("description", "")
        if desc:
            return f"Agent: {desc[:40]}..."
        return "Running sub-agent..."

    def get_input_summary(self, input_data: dict[str, Any]) -> str:
        # 로그/UI에 남길 입력 한 줄 요약. 긴 prompt/description은 앞부분만 잘라 보여 준다.
        subagent_type = input_data.get("subagent_type", "")
        if subagent_type:
            return f"[{subagent_type}] {input_data.get('prompt', '')[:80]}"
        return input_data.get("description", "")[:100]


# ─────────────────────────────────────────────
# 내부 데이터 클래스 — 이름 앞의 밑줄(_)은 "모듈 외부로 내보내지 않는 내부용"이라는 관례.
# ─────────────────────────────────────────────
class _AgentConfigError(Exception):
    """서브에이전트 설정 해석 실패를 나타내는 내부 전용 예외.

    _resolve_agent_config / _resolve_model_provider에서 설정을 확정하지 못할 때 던지며,
    call()이 이를 잡아 ToolResult.error로 변환한다(사용자에게는 사유 문자열만 전달).
    """


class _AgentConfig:
    """서브에이전트 실행에 필요한 값들을 한데 묶어 전달하는 설정 컨테이너.

    _resolve_agent_config가 만들어 _run_subagent로 넘긴다. 담는 값:
      system_prompt        — 서브에이전트의 시스템 프롬프트
      tools                — 서브에이전트가 쓸 수 있도록 필터링된 도구 목록
      max_turns            — 서브에이전트가 돌 수 있는 최대 턴 수
      model_provider       — 실제 추론을 수행할 프로바이더(부모 재사용 또는 Scout 전용)
      model_override_label — 어떤 모델을 골랐는지 로그/관측용 라벨("parent"/"scout" 등)

    __slots__를 지정해 인스턴스 dict를 없앤다 — 메모리를 아끼고, 정의되지 않은
    속성을 실수로 추가하는 것을 막는(오타 방지) 효과가 있다.
    """

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
        # 전달받은 값을 그대로 필드에 보관하는 단순 컨테이너 초기화.
        self.system_prompt = system_prompt
        self.tools = tools
        self.max_turns = max_turns
        self.model_provider = model_provider
        self.model_override_label = model_override_label
