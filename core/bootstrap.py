"""부트스트랩 — Nexus 애플리케이션의 2단계(2-Phase) 초기화 시스템.

이 파일은 CLI/웹/SDK 등 "모든 진입점"이 앱을 켤 때 가장 먼저 호출하는
공통 시동 코드다. Claude Code의 entrypoints/init.ts + setup.ts를
Python으로 재구현했다. 초기화를 딱 두 단계로 쪼개는 것이 핵심 설계다.

Phase 1 (init 함수): "환경 비의존" 초기화 — 외부 서비스가 없어도 되는 준비
  - GlobalState 생성 (세션 전역 상태 싱글톤)
  - 설정 로딩 + 검증 (YAML → NexusConfig)
  - 로깅 구성 (콘솔 + 파일)
  - 정상 종료(graceful shutdown) 핸들러 등록
  - GPU 서버 사전 연결 (fire-and-forget: 실패해도 진행)
  - 플랫폼(OS/셸) 감지

Phase 2 (init_phase2 함수): "환경 의존" 초기화 — 외부 서비스에 실제로 붙는 단계
  - ToolRegistry (도구 등록 — 티어에 따라 7개/23개)
  - MemoryManager (Redis 단기 + PostgreSQL 장기, 실패 시 인메모리 폴백)
  - RAG/지식/심볼 인덱서, MCP 연결, 권한 파이프라인
  - QueryEngine (Tier 1 세션 오케스트레이터) 최종 조립

왜 2단계로 나누는가:
  Phase 1은 core/ 내부의 다른 모듈을 "전혀 import하지 않는다". 덕분에
  이 파일이 의존성 그래프(DAG)의 leaf(말단)로 격리되어 순환 import가
  원천 차단된다. 무거운 모듈은 전부 Phase 2에서 함수 진입 시점에 lazy
  import 하므로, 시동 초반에는 가볍고 안전하게 뜬다.

주요 공개 함수:
  - init(): Phase 1 진입점
  - init_phase2(): Phase 2 진입점, 조립된 컴포넌트 dict 반환
  - register_cleanup(): 종료 시 실행할 정리 콜백 등록

작성자: 이현수 / 작성일: 2026-07-05
"""

from __future__ import annotations

import asyncio
import logging
import os
import platform
import signal
import sys
from pathlib import Path
from typing import Any

# ── Phase 1에서 허용되는 유일한 core import ──
# config/state 두 모듈만 끌어온다. 이 둘은 다른 core 모듈에 의존하지 않아
# leaf 격리(순환 import 방지) 원칙을 깨지 않는다. 나머지 무거운 모듈은
# 전부 Phase 2 함수 내부에서 lazy import 한다.
from core.config import load_and_validate_config
from core.state import GlobalState, get_initial_state

# 이 모듈 전용 로거. 규칙상 "nexus.{모듈}" 네임스페이스를 사용한다.
logger = logging.getLogger("nexus.bootstrap")


# ─────────────────────────────────────────────
# Phase 1: 환경 비의존 초기화
# ─────────────────────────────────────────────
async def init(
    config_path: str | None = None,
    cwd: str | None = None,
) -> GlobalState:
    """Phase 1(환경 비의존) 초기화를 수행한다.

    CLI, SDK, 웹 서버 등 어떤 진입점이든 앱 시동 시 이 함수를 "가장 먼저"
    호출해야 한다. 여기서는 GlobalState 생성, 설정 로드, 로깅, 종료 핸들러
    등록까지만 하고, DB/모델 등 외부 서비스 연결은 Phase 2로 미룬다.
    core/ 내부의 무거운 모듈은 import하지 않아 순환 의존을 원천 차단한다.

    Args:
        config_path: 설정 YAML 파일 경로. None이면 config가 자동 탐색한다.
        cwd: 작업 디렉토리. None이면 현재 프로세스의 디렉토리를 사용한다.

    Returns:
        초기화된 GlobalState 싱글톤. 이후 init_phase2에 그대로 넘겨준다.
    """
    # ① GlobalState 초기화 — 세션 ID/작업 디렉토리 등 전역 상태의 시작점.
    state = get_initial_state(cwd=cwd)
    logger.info(f"[Phase 1] GlobalState 초기화 완료: session={state.session_id}")

    # ② 설정 로딩 + 검증 — YAML을 읽어 NexusConfig(Pydantic)로 검증한 뒤
    #    state에 붙여둔다. 이후 모든 단계가 이 config를 단일 소스로 참조한다.
    # NEXUS_CONFIG 환경변수로 설정 파일 경로를 지정할 수 있다(config 헤더에 문서화된 방식).
    # 우선순위: 인자로 넘긴 config_path > NEXUS_CONFIG env > 자동 탐색(None).
    config_path = config_path or os.environ.get("NEXUS_CONFIG")
    config = load_and_validate_config(config_path)
    state.config = config
    logger.info(f"[Phase 1] 설정 로드 완료: gpu_server={config.gpu_server_url}")

    # ③ 로깅 구성 — 설정에서 읽은 레벨/파일로 콘솔+파일 핸들러를 세팅한다.
    _configure_logging(config.log_level, config.log_file)

    # ④ 정상 종료 핸들러 등록 — Ctrl+C(SIGINT)/SIGTERM 수신 시 세션 요약을
    #    남기고 등록된 정리 콜백을 실행하도록 시그널 핸들러를 건다.
    _setup_graceful_shutdown(state)
    logger.info("[Phase 1] 종료 핸들러 등록 완료")

    # ⑤ GPU 서버 사전 연결 (fire-and-forget)
    # 왜 fire-and-forget인가: GPU 서버가 아직 안 떠 있어도 부트스트랩은 진행해야 한다.
    # 첫 추론 요청 전까지 연결이 되면 100-200ms를 절약할 수 있다.
    asyncio.create_task(_preconnect_gpu_server(config.gpu_server_url))

    # ⑥ 플랫폼 감지 — OS/파이썬버전/아키텍처와 사용할 셸 경로를 파악해
    #    state에 저장한다. Bash 도구가 이 셸 정보를 참조한다.
    state.platform = _detect_platform()
    logger.info(f"[Phase 1] 플랫폼: {state.platform.get('os', 'unknown')}")

    return state


# ─────────────────────────────────────────────
# Phase 2: 환경 의존 초기화
# ─────────────────────────────────────────────
async def init_phase2(state: GlobalState) -> dict:
    """Phase 2(환경 의존) 초기화를 수행하고 조립된 컴포넌트 dict를 돌려준다.

    반드시 Phase 1(init)이 끝난 뒤 그 결과 state를 받아 호출한다. 여기서
    비로소 GPU(모델)/Redis/PostgreSQL/MCP 등 실제 외부 서비스에 연결하고,
    도구·메모리·RAG·권한 파이프라인을 거쳐 최종적으로 Tier 1 오케스트레이터인
    QueryEngine까지 조립한다.

    설계 원칙:
      - 외부 서비스 연결은 전부 "실패 격리"한다. Redis/PG/RAG/MCP 등이 죽어도
        각 블록이 예외를 삼키고 인메모리 폴백 또는 비활성으로 진행해, 채팅이라는
        본류 기능은 절대 멈추지 않는다(fail-closed로 부가기능만 끈다).
      - 무거운 core 모듈은 함수 진입 시점 lazy import로 끌어와 순환 의존을 막는다.
      - 하드웨어 티어(S/M/L)에 따라 도구 세트·컨텍스트 전략이 자동으로 갈린다.

    Args:
        state: Phase 1(init)에서 생성·구성된 GlobalState.

    Returns:
        조립된 컴포넌트 딕셔너리(components). 진입점이 이 dict에서 필요한
        객체(query_engine 등)를 꺼내 세션 루프를 구동한다. 주요 키:
          - model_provider: LocalModelProvider (LLM + 임베딩)
          - tool_registry: ToolRegistry (메트릭용 풀세트)
          - memory_manager: MemoryManager (Redis/PG, 없으면 인메모리 폴백)
          - query_engine: QueryEngine (Tier 1 세션 오케스트레이터)
          - 그 외 rag/knowledge/symbol/mcp/permission 관련 다수
    """
    # lazy import — Phase 2에서 필요한 무거운 모듈들을 여기서만 끌어온다.
    # Phase 1(파일 상단)에서는 절대 import하지 않아 leaf 격리를 유지한다.
    from core.memory.long_term import LongTermMemory
    from core.memory.manager import MemoryManager
    from core.memory.short_term import ShortTermMemory
    from core.memory.transcript import SessionTranscript
    from core.model.inference import LocalModelProvider
    from core.orchestrator.agent_definition import build_default_agent_registry
    from core.orchestrator.query_engine import QueryEngine
    from core.task import TaskManager
    from core.tools.base import ToolUseContext

    # 조립된 컴포넌트를 담아 반환할 그릇. 각 단계가 여기에 키를 채워 넣는다.
    components: dict = {}

    # ① ModelProvider 생성 — Machine B(GPU 서버)의 OpenAI 호환 API 클라이언트.
    #    LLM(추론)과 임베딩 서버가 서로 다른 포트/인스턴스라, 각각의 URL을
    #    따로 전달한다. 이후 모든 추론/임베딩은 이 provider를 통해서만 나간다
    #    (규칙: Machine A는 GPU를 직접 호출하지 않는다).
    config = state.config
    provider = LocalModelProvider(
        base_url=config.gpu_server_url,
        model_id=config.model.primary_model,
        max_context_tokens=config.model.max_context_tokens,
        max_output_tokens=config.model.default_max_tokens,
        embedding_base_url=config.gpu_server.embedding_url,
        # 구조화 출력 payload 주입 형태를 config에서 주입(하드코딩 회피, 안티패턴 #4).
        # 기본 "response_format"은 Phase 0 B200 실측으로 확정된 값이다.
        structured_output_injection_mode=config.structured_output.injection_mode,
    )
    components["model_provider"] = provider
    logger.info("[Phase 2] ModelProvider 초기화: %s", config.gpu_server_url)

    # ② ToolRegistry — 23개 도구 등록 (실측: _create_tool_registry 등록 개수)
    registry = _create_tool_registry()
    components["tool_registry"] = registry
    logger.info("[Phase 2] ToolRegistry 초기화: %d개 도구", registry.tool_count)

    # ③ MemoryManager — 단기(Redis)+장기(PostgreSQL) 메모리를 실제로 연결한다.
    #    두 연결 헬퍼는 실패 시 None을 돌려주고, 그때는 인메모리 폴백으로 동작해
    #    테스트/CI나 DB 장애 상황에서도 채팅이 끊기지 않는다.
    redis_client = await _create_redis_client(config)
    pg_pool = await _create_pg_pool(config)
    stm = ShortTermMemory(redis_client=redis_client)  # 세션 단기 메모리
    ltm = LongTermMemory(pg_pool=pg_pool)  # pgvector 기반 장기 메모리
    # tb_memories 스키마·인덱스 멱등 보장 — 새 PG 인스턴스(컨테이너 교체/재해
    # 복구)에서도 자동으로 운영 정의(varchar(12) PK + hnsw 인덱스)를 재현한다.
    # pg_pool=None이면 no-op이므로 인메모리 테스트는 영향 없음.
    if pg_pool is not None:
        try:
            await ltm.ensure_schema()
        except Exception as e:
            # 실패해도 인메모리 폴백으로 본류 응답 가능하므로 WARNING만.
            logger.warning("[Phase 2] tb_memories ensure_schema 실패 (무시): %s", e)
    memory_manager = MemoryManager(
        short_term=stm,
        long_term=ltm,
        model_provider=provider,
    )
    components["memory_manager"] = memory_manager
    components["redis_client"] = redis_client
    components["pg_pool"] = pg_pool
    mode = (
        "Redis+PG"
        if (redis_client and pg_pool)
        else ("Redis만" if redis_client else ("PG만" if pg_pool else "인메모리 폴백"))
    )
    logger.info("[Phase 2] MemoryManager 초기화: %s", mode)

    # ③-b TaskManager — 백그라운드 태스크(에이전트/모니터/워크플로우 등)의
    #    생성·조회·중단 라이프사이클을 관리한다. Task 계열 도구가 이걸 참조한다.
    task_manager = TaskManager()
    components["task_manager"] = task_manager
    logger.info("[Phase 2] TaskManager 초기화")

    # ③-c TodoStore — 계획 체크리스트(TodoWrite) 상태 저장소.
    #    TurnStateStore와 동일한 세션·에이전트 키 인메모리 저장소로, TodoWrite/
    #    TodoRead 도구가 context.options["todo_store"]로 참조한다(아래 ④-b).
    #    prompt_assembler가 매 턴 이 체크리스트를 시스템 프롬프트로 재주입한다.
    from core.todo_store import TodoStore

    todo_store = TodoStore()
    components["todo_store"] = todo_store
    logger.info("[Phase 2] TodoStore 초기화 (계획 체크리스트)")

    # ④ AgentRegistry — v7.0 Phase 9 서브에이전트 시스템의 정의 저장소.
    #    SCOUT_AGENT 등 기본 서브에이전트가 여기 등록되며, AgentTool이
    #    subagent_type 이름으로 정의를 조회하고 시스템 프롬프트에도 반영된다.
    agent_registry = build_default_agent_registry()
    components["agent_registry"] = agent_registry

    # ④-a 권한 강제 파이프라인 배선 (감사 Critical #1~3, 2026-07-03) — P0 하네스 + P1 shadow
    # ───────────────────────────────────────────────────────────────────────
    # 세션 1회만 만든다. executor까지 ToolUseContext.options로 전달한다(아래 ④-b).
    # ★무회귀★:
    #   - config.permission_enforcement.enabled 가 False(기본)면 파이프라인을
    #     아예 만들지 않는다(None). executor는 None을 보고 현행 로직(도구
    #     check_permissions의 DENY만 차단)을 그대로 탄다 → 동작 100% 동일.
    #   - True 여도 mode="shadow"면 executor가 판정을 감사 로그에 기록만 하고
    #     실행 경로는 바꾸지 않는다(차단 안 함). 실제 차단은 후속 단계(enforce).
    # lazy import — Phase 2 모듈 패턴(필요 시점 import, 순환 의존 방지)을 따른다.
    from core.permission.mode_mapping import map_mode_value_to_permission_mode
    from core.permission.pipeline import PermissionPipeline
    from core.permission.types import PermissionContext
    from core.security.audit import AuditLogger
    from core.security.command_filter import CommandFilter
    from core.security.path_guard import PathGuard

    perm_cfg = config.permission_enforcement
    audit_cfg = config.audit

    permission_pipeline = None
    audit_logger = None
    # AuditLogger는 audit.enabled만 켜져 있으면 만든다(관측은 파이프라인과 독립).
    # 파이프라인이 없어도(무회귀 상태) 감사 로거 자체는 존재할 수 있으나, executor는
    # 파이프라인이 있을 때만 기록하므로 enforcement가 꺼져 있으면 실제 쓰기는 없다.
    if audit_cfg.enabled:
        audit_logger = AuditLogger(log_path=audit_cfg.path)
    # 파이프라인은 enforcement.enabled가 True일 때만 생성한다(무회귀 핵심).
    if perm_cfg.enabled:
        # 세션 권한 모드(PermissionModeValue) → 파이프라인 모드(PermissionMode) 변환.
        perm_mode = map_mode_value_to_permission_mode(state.permission_mode)
        perm_context = PermissionContext(
            mode=perm_mode,
            working_directory=state.cwd or os.getcwd(),
            session_id=state.session_id,
        )
        # Layer 2 사전검사기 — PathGuard(경로 순회/보호경로/UNC) + CommandFilter
        # (위험 명령어 + 에어갭 설치차단 게이팅). 파이프라인에 주입하면 Layer 2가
        # 도구 자체 검사 이전에 이 둘로 먼저 걸러낸다(fail-closed).
        #   - PathGuard: 기본 보호경로 목록(.env/.ssh/*.pem/*.key/etc 등)만으로 충분해
        #     별도 override 없이 생성한다(경로 override는 후속 필요 시 확장).
        #   - CommandFilter: block_package_install을 config에서 받아 개발(false)/
        #     배포(true) 정책을 코드 하드코딩 없이 분기한다(anti #4).
        cmd_cfg = config.command_filter
        path_guard = PathGuard()
        command_filter = CommandFilter(
            block_package_install=cmd_cfg.block_package_install,
        )
        # hook_manager가 components에 있으면 주입(현재 부트스트랩엔 미배선 → None).
        # 주의(shadow 안전성): Layer 4 Hook는 PRE_TOOL_USE로 외부 명령을 실행할 수
        # 있어 "관측 전용"이 깨질 수 있다. 훗날 hook_manager가 실제로 배선되면
        # shadow 단계에서의 Hook 실행 정책을 반드시 재검토해야 한다.
        permission_pipeline = PermissionPipeline(
            context=perm_context,
            # TODO(nexus): config 기반 deny rule 로딩은 후속 단계. 현재 NexusConfig에
            #   PermissionRule 목록 필드가 없어 rules=None으로 둔다(파이프라인은 도구
            #   자체 검사 Layer 2 + 모드 기반 Layer 3만으로 판정).
            rules=None,
            hook_manager=components.get("hook_manager"),
            path_guard=path_guard,
            command_filter=command_filter,
        )
    components["permission_pipeline"] = permission_pipeline
    components["audit_logger"] = audit_logger
    logger.info(
        "[Phase 2] 권한 파이프라인 배선: enforcement=%s(mode=%s), pipeline=%s, audit=%s",
        perm_cfg.enabled,
        perm_cfg.mode,
        "생성" if permission_pipeline else "미생성(무회귀)",
        "생성" if audit_logger else "비활성",
    )

    # ④-b ToolUseContext 생성
    # options는 AgentTool이 서브에이전트를 해석할 때 필요한 모든 의존성을 제공한다:
    #   - agent_registry: subagent_type → AgentDefinition 조회
    #   - model_provider: 부모 Worker (model_override가 없을 때 재사용)
    #   - scout_provider: model_override="scout"일 때 사용 (없으면 AgentTool이 에러 반환)
    #   - available_tools: allowed_tools 필터링의 후보 집합 (Phase 2-b에서 채움)
    context = ToolUseContext(
        cwd=state.cwd or os.getcwd(),
        session_id=state.session_id,
        permission_mode=state.permission_mode.value,
        options={
            "memory_manager": memory_manager,
            "task_manager": task_manager,
            # 계획 체크리스트 저장소 — TodoWrite/TodoRead 도구가 여기서 꺼내 쓴다.
            # 미주입 시 도구가 모듈 전역 폴백 TodoStore로 동작한다(무회귀).
            "todo_store": todo_store,
            "agent_registry": agent_registry,
            "model_provider": provider,
            # 문서 청크 크기 — 하드코딩 외부화(2026-07-03). DocumentProcess 도구가
            # 이 값을 읽어 청크를 나눈다. 미주입 시 도구가 CHUNK_SIZE(2500)로 폴백.
            "document_chunk_size": config.context_budgets.document_chunk_size,
            # 권한 강제 파이프라인 배선(감사 Critical #1~3, 2026-07-03) — executor가
            # options에서 꺼내 쓴다. 왜 options인가: ToolUseContext는 이미 executor까지
            # 흐르고, memory_manager/agent_registry 등 세션 의존성도 전부 options로
            # 주입되는 기존 패턴을 따른다. query_loop/stream_handler 시그니처를 전혀
            # 바꾸지 않아 4-Tier 체인·무회귀를 지킨다.
            #   - permission_pipeline: enforcement 꺼짐 시 None → executor 현행 로직
            #   - audit_logger: shadow 판정을 JSONL로 기록(관측 전용)
            #   - permission_enforcement: {enabled, mode} — executor의 게이트 판단용
            "permission_pipeline": permission_pipeline,
            "audit_logger": audit_logger,
            "permission_enforcement": {
                "enabled": perm_cfg.enabled,
                "mode": perm_cfg.mode,
            },
        },
    )
    components["tool_use_context"] = context

    # ⑤ 하드웨어 티어 감지 — v7.0 적응형 오케스트레이션
    from core.model.hardware_tier import HardwareTier, detect_hardware_tier, get_tier_config
    from core.orchestrator.turn_state import TurnStateStore

    tier = detect_hardware_tier(config)
    tier_cfg = get_tier_config(tier)
    state.hardware_tier = tier.value
    state.orchestration_mode = tier_cfg["orchestration_mode"]
    components["hardware_tier"] = tier
    logger.info("[Phase 2] 하드웨어 티어: %s (%s)", tier.value, tier_cfg["description"])

    # ⑥ TurnStateStore — v7.0 턴 상태 외부화
    # TIER_S: 필수 (raw messages 누적 대신 요약 사용)
    # TIER_M/L: 메타데이터 용도로만 생성 (기존 동작 유지)
    turn_state_store = TurnStateStore() if tier_cfg["turn_state_enabled"] else None
    components["turn_state_store"] = turn_state_store
    if turn_state_store:
        logger.info("[Phase 2] TurnStateStore 초기화 (상태 외부화 활성)")

    # ⑦ Scout 초기화 — v7.0 Phase 9.5
    # TIER_S이고 config.scout.enabled이면 Scout 서버 연결을 시도한다.
    # 연결 실패 시 None 반환 (Worker 단독 모드로 fallback).
    scout_provider = None
    if tier == HardwareTier.TIER_S and config.scout.enabled:
        from core.model.scout_provider import create_scout_provider_if_available

        scout_provider = await create_scout_provider_if_available(
            base_url=config.scout.base_url,
            api_key=config.scout.api_key,
        )
        if scout_provider:
            state.scout_enabled = True
            logger.info("[Phase 2] Scout 초기화: %s", config.scout.base_url)
        else:
            logger.warning("[Phase 2] Scout 연결 실패, Worker 단독 모드")
    components["scout_provider"] = scout_provider
    # AgentTool이 model_override="scout"을 해석할 때 꺼내간다
    context.options["scout_provider"] = scout_provider

    # ⑧ 티어별 도구 레지스트리 자동 선택 (실측 개수)
    # TIER_S: 7개 도구 (_create_cli_tool_registry) — 컨텍스트 절약
    # TIER_M/L: 23개 도구 전체 (_create_tool_registry) — 컨텍스트 충분
    if tier == HardwareTier.TIER_S:
        cli_registry = _create_cli_tool_registry()
    else:
        cli_registry = _create_tool_registry()  # 23개 전체
    cli_tools = cli_registry.get_all_tools()

    # ⑧-b Scout 전용 읽기 전용 도구 세트 (TIER_S만 사용)
    # v7.0 사양서 Part 2.3: Scout는 Read/Glob/Grep/LS 4개로 탐색만 수행
    scout_registry = _create_scout_tool_registry()
    scout_tools = scout_registry.get_all_tools()
    components["scout_tools"] = scout_tools

    # AgentTool이 allowed_tools로 필터링할 후보 도구 풀을 context.options에 주입한다.
    # Worker가 쓰는 CLI 도구 + Scout가 쓰는 읽기 전용 도구를 합쳐 중복 없이 등록한다
    # (ToolRegistry.register_many가 내부적으로 사용하는 name 기반 중복 검사는
    # 여기서는 불필요하므로 단순 리스트로 관리한다).
    tool_pool_names: set[str] = set()
    combined_tools: list = []
    for tool in [*cli_tools, *scout_tools]:
        if tool.name not in tool_pool_names:
            combined_tools.append(tool)
            tool_pool_names.add(tool.name)
    context.options["available_tools"] = combined_tools

    # Phase 10.0 — SymbolSearchTool이 context.options["symbol_store"]로 조회한다
    # (실제 할당은 ⑨-c 이후에 수행되지만, 키를 미리 만들어 None 폴백 허용)
    context.options.setdefault("symbol_store", None)

    # ⑨ RAG 파이프라인 초기화 — 프로젝트 파일 인덱싱 + 검색
    rag_retriever = None
    try:
        from core.rag.indexer import ProjectIndexer
        from core.rag.retriever import RAGRetriever

        indexer = ProjectIndexer(
            model_provider=provider,
            memory_store=ltm,  # LongTermMemory (인메모리 폴백)
        )
        rag_retriever = RAGRetriever(
            model_provider=provider,
            memory_store=ltm,
        )
        components["rag_indexer"] = indexer
        components["rag_retriever"] = rag_retriever

        # 백그라운드 인덱싱 (fire-and-forget)
        # 인덱싱이 완료되기 전에도 채팅 가능 (검색 결과가 없을 뿐)
        cwd = state.cwd or os.getcwd()
        asyncio.create_task(_background_index(indexer, cwd))
        logger.info("[Phase 2] RAG 파이프라인 초기화 (백그라운드 인덱싱 시작)")
    except Exception as e:
        logger.warning("[Phase 2] RAG 초기화 실패 (무시): %s", e)

    # ⑨-b 지식 베이스 RAG — Part 2.5.8 (2026-04-21)
    # tb_knowledge에서 KNOWLEDGE_MODE 질의 시 자동 검색·주입.
    # pg_pool이 없으면 인메모리 폴백(테스트/CI용), 있으면 실 스키마 준비.
    knowledge_retriever = None
    try:
        from core.rag.knowledge_retriever import KnowledgeRetriever
        from core.rag.knowledge_store import KnowledgeStore

        knowledge_store = KnowledgeStore(pg_pool=pg_pool)
        # pg_pool이 있으면 스키마 멱등 생성 — 실제 PostgreSQL에만 DDL 실행됨
        if pg_pool is not None:
            await knowledge_store.ensure_schema()
        # 유사도 게이팅 임계값을 config(yaml 단일 소스)에서 주입한다.
        # KnowledgeRetriever 생성자 기본값은 "게이팅 무효"라, 여기서 실제 값을
        # 넘겨야 무관 청크 차단(abs_threshold)+노이즈 절단(relevance_margin)이
        # 작동한다. 코드에 임계를 박지 않아 운영 중 yaml로 튜닝할 수 있다.
        krag = config.knowledge_rag
        knowledge_retriever = KnowledgeRetriever(
            store=knowledge_store,
            embedding_provider=provider,  # e5-large 임베딩 서버 경유
            top_k=krag.top_k,
            min_similarity=krag.min_similarity,
            abs_threshold=krag.abs_threshold,
            relevance_margin=krag.relevance_margin,
            # MMR 리랭킹 (게이팅 이후 다양성 선별). yaml knowledge_rag.mmr에서 주입.
            # enabled 기본 False라 켜기 전까지 동작은 종전과 100% 동일하다.
            mmr_enabled=krag.mmr.enabled,
            mmr_fetch_k=krag.mmr.fetch_k,
            mmr_lambda=krag.mmr.lambda_,
            # 크로스인코더 리랭커 (게이팅 대체 + 재정렬). yaml knowledge_rag.rerank에서 주입.
            # enabled 기본 False라 켜기 전까지 동작은 종전과 100% 동일하다.
            rerank_enabled=krag.rerank.enabled,
            rerank_fetch_k=krag.rerank.fetch_k,
            rerank_top_k=krag.rerank.top_k,
            rerank_min_score=krag.rerank.min_score,
            rerank_min_similarity=krag.rerank.min_similarity,
            # ivfflat 재현율(probes) — search_by_vector가 SET LOCAL로 적용. lists=1000
            # 인덱스에서 기본 probes=1은 정답 문서를 놓쳐 그라운딩 실패를 유발한다.
            ivfflat_probes=krag.ivfflat_probes,
            # 출처 인용(Point 4-2). yaml knowledge_rag.citation에서 주입. enabled 기본
            # False라 켜기 전까지 헤더·주입·반환이 종전과 100% 동일하다(무회귀).
            citation_enabled=krag.citation.enabled,
            citation_label=krag.citation.label,
        )
        components["knowledge_store"] = knowledge_store
        components["knowledge_retriever"] = knowledge_retriever
        count = await knowledge_store.count()
        logger.info(
            "[Phase 2] KnowledgeStore 초기화: 레코드=%d (pg=%s)",
            count,
            "connected" if pg_pool else "in-memory",
        )

        # v0.14.8 — 임베딩 서버 워밍업(A) + 주기적 keep-warm(B)
        # 배경: e5-large 서버(:8002)가 idle 상태로 빠지면 첫 호출에서 ~60초의
        # cold start가 발생해 KNOWLEDGE_MODE 첫 호출이 매우 느려진다. 부트스트랩
        # 시점에 한 번 ping해 모델·CUDA 컨텍스트를 따뜻하게 만들고, 이후 주기적
        # ping으로 cold 상태로 떨어지지 않게 유지한다.
        # 둘 다 fire-and-forget — 워밍업·keepalive 실패는 본류 응답에 무영향.
        asyncio.create_task(_warmup_embedding(provider))
        keepalive_task = asyncio.create_task(_embedding_keepalive(provider))
        components["embedding_keepalive_task"] = keepalive_task
        logger.info(
            "[Phase 2] 임베딩 서버 워밍업 + keep-warm 시작 (interval=%ds)",
            _EMBEDDING_KEEPALIVE_INTERVAL_SEC,
        )
    except Exception as e:
        logger.warning("[Phase 2] 지식 베이스 초기화 실패 (무시): %s", e)

    # ⑨-c Phase 10.0 — 심볼 인덱스 (tb_symbols)
    # 프로젝트 Python 파일의 함수/클래스/메서드를 ast로 추출하여 pgvector에
    # 인덱싱한다. 기존 파일 청크 RAG(tb_memories)와 별도로 운영하여 심볼
    # 단위 정확도를 확보한다. 부트스트랩 후 백그라운드로 실행.
    try:
        from core.rag.symbol_indexer import SymbolProjectIndexer, background_index
        from core.rag.symbol_store import SymbolStore

        symbol_store = SymbolStore(pg_pool=pg_pool)
        if pg_pool is not None:
            await symbol_store.ensure_schema()
        components["symbol_store"] = symbol_store
        # SymbolSearchTool이 조회하는 경로
        context.options["symbol_store"] = symbol_store

        # 임베딩 배치 콜러블 (provider.embed 그대로 래핑)
        async def _embed_batch(texts: list[str]) -> list[list[float]]:
            return await provider.embed(texts)

        symbol_indexer = SymbolProjectIndexer(
            store=symbol_store,
            embedder=_embed_batch,
            project_source="nexus",
        )
        components["symbol_indexer"] = symbol_indexer
        cwd_for_symbols = state.cwd or os.getcwd()
        asyncio.create_task(background_index(symbol_indexer, cwd_for_symbols))
        count = await symbol_store.count()
        logger.info(
            "[Phase 2] SymbolStore 초기화: 레코드=%d (pg=%s, 백그라운드 인덱싱 시작)",
            count,
            "connected" if pg_pool else "in-memory",
        )
    except Exception as e:
        logger.warning("[Phase 2] 심볼 인덱스 초기화 실패 (무시): %s", e)
        components["symbol_store"] = None

    # ⑨-d v7.2 MCP 연결 — LAN 내부 MCP 서버 도구를 cli_registry에 흡수
    # KnowledgeStore/SymbolStore 초기화 직후, 동일한 fire-and-forget + 실패 격리
    # 패턴. config.mcp.enabled가 켜져 있을 때만 시도하며, 전체 실패도 본류 무영향.
    #
    # 왜 cli_registry인가: 아래 ⑩ ModelDispatcher(worker_tools)와 ⑫ QueryEngine
    # (tools)에 실제로 전달되는 도구 풀은 ⑧에서 만든 cli_tools다(line 240).
    # 따라서 MCP 도구가 모델에 노출되려면 cli_registry에 등록한 뒤 cli_tools를
    # 다시 취득해야 한다(:132의 registry에 넣으면 QueryEngine까지 전달되지 않음).
    #
    # mcp_tools 기본값: MCP 비활성/실패 시에도 웹 도구 구성(_build_web_query_engine)이
    # 안전하게 .get("mcp_tools", [])로 참조하도록 빈 리스트를 미리 둔다(fail-closed).
    components["mcp_tools"] = []
    if config.mcp.enabled:
        try:
            # lazy import — Phase 2 모듈 패턴(필요 시점 import)을 따른다
            from core.tools.mcp import McpConnectionManager

            mcp_manager = McpConnectionManager(config.mcp)
            registered = await mcp_manager.connect_and_register(cli_registry)
            components["mcp_manager"] = mcp_manager
            # 등록된 MCP 도구가 worker/QueryEngine 도구 풀에 포함되도록 재취득
            # (get_all_tools는 이름순 정렬을 보장 → prompt cache 안정성 P5 유지)
            cli_tools = cli_registry.get_all_tools()
            # 웹 진입점(web/app.py)은 cli_registry가 아니라 _create_web_tool_registry로
            # 자체 도구 풀을 만든다. 그 경로도 MCP 도구를 흡수할 수 있도록 등록된
            # MCP 어댑터(이름 prefix "mcp__")만 따로 추려 components로 노출한다.
            # (cli_registry 전체가 아니라 MCP 어댑터만 넘겨 웹 도구 구성의 독립성 유지)
            components["mcp_tools"] = [t for t in cli_tools if t.name.startswith("mcp__")]
            # GlobalState 에 MCP 가시성 정보를 채운다(메트릭/진단 노출용).
            #   mcp_servers: 서버명 → {등록 도구 목록, 개수} 상세.
            #   mcp_connected: 도구가 1개 이상 등록되어 "살아 있는" 서버명 집합.
            # connect_and_register 가 실패 서버를 빈 리스트로 돌려주므로,
            # 빈 리스트는 connected 에서 자연히 제외된다(fail-closed 요약).
            # 정책적으로 Worker 풀에서 제외된 서버(expose_to_worker=false)는
            # 도구 0개이지만 "연결 실패"가 아니라 "의도된 제외"임을 가시화한다.
            # excluded_from_worker = {서버명: 사유}. 빈 도구라도 사유를 덧붙여 노출.
            excluded = getattr(mcp_manager, "excluded_from_worker", {})
            state.mcp_servers = {
                name: {
                    "tools": list(tools),
                    "tool_count": len(tools),
                    # 제외된 서버에만 사유 표기를 덧붙인다(과설계 없이 간단히).
                    **({"excluded": excluded[name]} if name in excluded else {}),
                }
                for name, tools in registered.items()
            }
            state.mcp_connected = {name for name, tools in registered.items() if len(tools) >= 1}
            logger.info(
                "[Phase 2] MCP 연결: %s (연결 서버=%d개, cli_tools=%d개)",
                {s: len(t) for s, t in registered.items()},
                len(state.mcp_connected),
                len(cli_tools),
            )
        except Exception as e:
            # 최후 방어선(intentional broad except) — 여기서는 broad가 옳다.
            # 이유: MCP는 선택적 보조 기능이므로, 어떤 예외가 나더라도 부트스트랩
            # 본류(채팅·도구 풀)는 멈추면 안 된다(fail-closed로 MCP만 끈다).
            #
            # 설계 분담:
            #   - connect_and_register는 "예상 가능한 운영 실패"(연결/타임아웃/
            #     비-LAN/소켓)만 좁혀 서버별로 격리한다. 예상치 못한 버그
            #     (AttributeError 등)는 거기서 전파시켜 결함이 묻히지 않게 한다.
            #   - 그렇게 전파된 예외를 이 broad except가 최종 흡수해 본류를 보호한다.
            # 즉 "결함 가시성(좁은 except)"과 "본류 보호(넓은 최후 except)"를 양립시킨다.
            logger.warning("[Phase 2] MCP 초기화 실패 (무시): %s", e)
            components["mcp_manager"] = None
    else:
        components["mcp_manager"] = None

    # ⑩ ModelDispatcher — v7.0 Phase 9 멀티모델 라우터
    # TIER_S: Scout(CPU 4B) → Worker(GPU 27B) 2단계 실행
    # TIER_M/L: Worker 단독 passthrough (v6.1 경로 동일)
    # Scout 실패 시 Worker 단독 모드로 자동 fallback한다.
    from core.orchestrator.model_dispatcher import ModelDispatcher

    dispatcher = ModelDispatcher(
        tier=tier,
        worker_provider=provider,
        worker_tools=cli_tools,
        context=context,
        scout_provider=scout_provider,
        scout_tools=scout_tools,
        max_turns=200,
    )
    components["model_dispatcher"] = dispatcher
    logger.info(
        "[Phase 2] ModelDispatcher 초기화: tier=%s, scout=%s",
        tier.value,
        "활성" if dispatcher.scout_enabled else "비활성",
    )

    # ⑪ ContextManager — Ch 6 티어별 전략 공식화 (2026-04-21)
    # TIER_S: pass-through (TurnStateStore가 컨텍스트 관리)
    # TIER_M/L: 4단계 압축 파이프라인 활성
    # 기존 동작과 호환: QueryEngine은 None도 허용하므로 주입 없이도 동작함
    from core.orchestrator.context_manager import ContextManager

    # 컨텍스트 예산을 config에서 주입(하드코딩 외부화, 2026-07-03).
    # 기본값=현행 하드코딩 값(2048/3/2)이라 미지정 시 동작 불변(무회귀).
    _budgets = config.context_budgets
    context_manager = ContextManager(
        model_provider=provider,
        max_context_tokens=config.model.max_context_tokens,
        tool_result_budget=_budgets.tool_result_budget,
        preserve_recent_turns=_budgets.preserve_recent_turns,
        preserve_recent_tool_results=_budgets.preserve_recent_tool_results,
        tier=tier,  # HardwareTier 전달 → TIER_S에서는 자동 pass-through
    )
    components["context_manager"] = context_manager
    logger.info(
        "[Phase 2] ContextManager 초기화: tier=%s, passthrough=%s, "
        "budgets(tool_result=%d, recent_turns=%d, recent_tool_results=%d)",
        tier.value,
        context_manager.passthrough,
        _budgets.tool_result_budget,
        _budgets.preserve_recent_turns,
        _budgets.preserve_recent_tool_results,
    )

    # ⑫ QueryEngine — Tier 1 세션 오케스트레이터
    # model_dispatcher가 주입되면 submit_message는 dispatcher.route() 경로를 탄다.
    # 시스템 프롬프트에는 agent_registry를 반영하여 서브에이전트 사용 가이드를 넣는다.
    # Ch 16: CLI 세션용 JSONL 트랜스크립트 (기본 활성)
    cli_transcript = SessionTranscript(
        sessions_dir=config.session.sessions_dir,
        session_id=state.session_id,
        enabled=config.session.transcript_enabled,
    )
    components["transcript"] = cli_transcript

    engine = QueryEngine(
        model_provider=provider,
        tools=cli_tools,
        context=context,
        system_prompt=_build_default_system_prompt(agent_registry),
        context_manager=context_manager,  # Ch 6: 티어별 전략
        max_turns=200,
        turn_state_store=turn_state_store,
        todo_store=todo_store,  # 계획 체크리스트 — 매 턴 시스템 프롬프트에 재주입
        rag_retriever=rag_retriever,
        model_dispatcher=dispatcher,
        routing_config=config.routing,  # v7.0 Part 2.5 — 지식/도구 분기
        memory_manager=memory_manager,  # Ch 16: Redis + tb_memories 자동 저장
        transcript=cli_transcript,  # Ch 16: JSONL 영구 기록
        knowledge_retriever=knowledge_retriever,  # Part 2.5.8: tb_knowledge RAG
        context_budgets=_budgets,  # 하드코딩 외부화(2026-07-03): RAG 예산 + 출력 에스컬레이션
    )
    components["query_engine"] = engine
    logger.info(
        "[Phase 2] QueryEngine 초기화: session=%s, 도구=%d개, tier=%s, rag=%s",
        engine.session_id,
        len(cli_tools),
        tier.value,
        "활성" if rag_retriever else "비활성",
    )

    return components


async def _background_index(indexer: Any, cwd: str) -> None:
    """
    백그라운드에서 프로젝트 디렉토리를 인덱싱한다.

    fire-and-forget으로 실행되므로 실패해도 메인 흐름에 영향 없다.
    인덱싱이 완료되면 RAG 검색이 활성화된다.
    """
    try:
        stats = await indexer.index_directory(cwd)
        logger.info(
            "[RAG] 백그라운드 인덱싱 완료: %d파일, %d청크",
            stats["indexed_files"],
            stats["indexed_chunks"],
        )
    except Exception as e:
        logger.warning("[RAG] 백그라운드 인덱싱 실패: %s", e)


# ─────────────────────────────────────────────
# v0.14.8 — 임베딩 서버 워밍업/keep-warm
# ─────────────────────────────────────────────
# e5-large 임베딩 서버(:8002)는 idle 상태로 빠지면 다음 호출에서 모델·CUDA
# 컨텍스트 워밍업으로 ~60초가 추가로 걸린다. 사용자 첫 KNOWLEDGE 질의가
# 그 비용을 그대로 떠안는 일을 막기 위해 부트스트랩 직후 한 번 워밍업하고,
# 이후 주기적으로 ping을 보내 cold 상태로 떨어지지 않게 유지한다.

# keep-warm ping 간격(초). 5분 — kowiki 적재 후 임베딩이 자주 안 쓰이는
# 운영 패턴에서도 cold로 떨어지지 않을 정도로 짧고, 부담은 거의 없다.
_EMBEDDING_KEEPALIVE_INTERVAL_SEC: int = 300

# 워밍업 입력 — 의미와 무관한 짧은 토큰. 짧을수록 비용이 작다.
_EMBEDDING_WARMUP_TEXT: str = "warmup"


async def _warmup_embedding(provider: Any, retries: int = 2) -> bool:
    """첫 호출 cold start 비용을 부트스트랩 시점에 흡수한다.

    임베딩 서버가 아직 안 떴을 가능성을 고려해 짧은 backoff로 재시도한다.
    실패해도 본류 응답에 영향 없으므로 조용히 False 반환.
    """
    for attempt in range(retries + 1):
        try:
            vecs = await provider.embed([_EMBEDDING_WARMUP_TEXT])
            if vecs:
                logger.info("[Phase 2] 임베딩 서버 워밍업 성공 (시도 %d)", attempt + 1)
                return True
        except Exception as e:
            if attempt < retries:
                await asyncio.sleep(2)
            else:
                logger.warning("[Phase 2] 임베딩 서버 워밍업 실패: %s", e)
                return False
    return False


async def _embedding_keepalive(
    provider: Any,
    interval_sec: int = _EMBEDDING_KEEPALIVE_INTERVAL_SEC,
) -> None:
    """주기적으로 임베딩 서버에 짧은 ping을 보내 cold 상태를 회피한다.

    asyncio.CancelledError로 깨끗하게 종료되며, 그 외 예외는 디버그 로그만.
    """
    while True:
        try:
            await asyncio.sleep(interval_sec)
            await provider.embed([_EMBEDDING_WARMUP_TEXT])
            logger.debug("[Phase 2] 임베딩 keepalive ping 성공")
        except asyncio.CancelledError:
            logger.info("[Phase 2] 임베딩 keepalive 종료")
            return
        except Exception as e:
            # 일시 장애는 다음 주기에 재시도 — 본류 응답에 영향 없으므로 debug만.
            logger.debug("[Phase 2] 임베딩 keepalive 실패 (다음 주기 재시도): %s", e)


async def _create_redis_client(config: Any) -> Any:
    """Redis 비동기 클라이언트를 만들고 ping으로 연결을 확인한다.

    단기 메모리(ShortTermMemory)의 백엔드다. redis 패키지가 없거나 서버에
    붙지 못하면 예외를 삼키고 None을 반환한다 → 호출부는 인메모리 폴백으로
    진행한다(연결 실패가 앱 시동을 막지 않는다는 원칙).

    Returns:
        연결된 aioredis.Redis 인스턴스, 또는 실패 시 None.
    """
    try:
        import redis.asyncio as aioredis

        client = aioredis.Redis(
            host=config.redis.host,
            port=config.redis.port,
            db=config.redis.db,
            password=config.redis.password or None,
            socket_timeout=config.redis.socket_timeout,
            decode_responses=True,
        )
        await client.ping()
        logger.info(
            "[Phase 2] Redis 연결 성공: %s:%d (db=%d)",
            config.redis.host,
            config.redis.port,
            config.redis.db,
        )
        return client
    except ImportError:
        logger.warning("[Phase 2] redis 패키지 미설치 — 인메모리 폴백")
        return None
    except Exception as e:
        logger.warning("[Phase 2] Redis 연결 실패 — 인메모리 폴백: %s", e)
        return None


async def _create_pg_pool(config: Any) -> Any:
    """PostgreSQL 커넥션 풀(asyncpg)을 생성한다.

    장기 메모리(pgvector)·지식/심볼 스토어가 공유하는 DB 풀이다. asyncpg가
    없거나 서버에 붙지 못하면 예외를 삼키고 None을 반환한다 → 호출부는 각
    스토어를 인메모리 폴백으로 돌린다(DDL·검색이 no-op이 된다).

    Returns:
        생성된 asyncpg 풀, 또는 실패 시 None.
    """
    try:
        import asyncpg

        pool = await asyncpg.create_pool(
            host=config.postgresql.host,
            port=config.postgresql.port,
            database=config.postgresql.database,
            user=config.postgresql.user,
            password=config.postgresql.password,
            min_size=config.postgresql.min_connections,
            max_size=config.postgresql.max_connections,
            timeout=10.0,
        )
        logger.info(
            "[Phase 2] PostgreSQL 연결 성공: %s:%d/%s",
            config.postgresql.host,
            config.postgresql.port,
            config.postgresql.database,
        )
        return pool
    except ImportError:
        logger.warning("[Phase 2] asyncpg 패키지 미설치 — 인메모리 폴백")
        return None
    except Exception as e:
        logger.warning("[Phase 2] PostgreSQL 연결 실패 — 인메모리 폴백: %s", e)
        return None


def _create_tool_registry():  # noqa: ANN202 — ToolRegistry는 함수 내부에서 import
    """23개 도구를 모두 등록한 "풀세트" ToolRegistry를 생성한다(실측 등록 개수).

    TIER_M/L(컨텍스트 여유) 및 Phase 2 ②의 메트릭용 레지스트리로 쓰인다.
    TIER_S에서는 컨텍스트 절약을 위해 대신 _create_cli_tool_registry(7개)를 사용한다.
    lazy import 이유: 도구 구현 모듈이 core 하위를 import하므로 함수 진입 시점에만 끌어와
    Phase 1과의 의존성 분리(순환 import 방지)를 지킨다.
    """
    from core.tools.implementations.bash_tool import BashTool
    from core.tools.implementations.docker_tools import DockerBuildTool, DockerRunTool
    from core.tools.implementations.edit_tool import EditTool
    from core.tools.implementations.git_tools import (
        GitBranchTool,
        GitCheckoutTool,
        GitCommitTool,
        GitDiffTool,
        GitLogTool,
        GitStatusTool,
    )
    from core.tools.implementations.glob_tool import GlobTool
    from core.tools.implementations.grep_tool import GrepTool
    from core.tools.implementations.ls_tool import LSTool
    from core.tools.implementations.memory_tools import MemoryReadTool, MemoryWriteTool
    from core.tools.implementations.multi_edit_tool import MultiEditTool
    from core.tools.implementations.notebook_tools import NotebookEditTool, NotebookReadTool
    from core.tools.implementations.read_tool import ReadTool
    from core.tools.implementations.task_tools import TaskTool
    from core.tools.implementations.todo_tools import TodoReadTool, TodoWriteTool
    from core.tools.implementations.write_tool import WriteTool
    from core.tools.registry import ToolRegistry

    registry = ToolRegistry()
    registry.register_many(
        [
            # 파일 시스템 (4개)
            ReadTool(),
            WriteTool(),
            EditTool(),
            MultiEditTool(),
            # 실행 (1개)
            BashTool(),
            # 검색 (3개)
            GlobTool(),
            GrepTool(),
            LSTool(),
            # Git (6개)
            GitLogTool(),
            GitDiffTool(),
            GitStatusTool(),
            GitCommitTool(),
            GitBranchTool(),
            GitCheckoutTool(),
            # 노트북 (2개)
            NotebookReadTool(),
            NotebookEditTool(),
            # 태스크 (3개)
            TodoReadTool(),
            TodoWriteTool(),
            TaskTool(),
            # 메모리 (2개)
            MemoryReadTool(),
            MemoryWriteTool(),
            # Docker (2개)
            DockerBuildTool(),
            DockerRunTool(),
        ]
    )

    return registry


def _create_scout_tool_registry():  # noqa: ANN202
    """
    Scout 에이전트 전용 읽기 전용 도구 레지스트리.

    v7.0 Part 2.3 개정 (2026-04-17): DocumentProcess 포함 5개.

    왜 5개인가:
      - Scout는 CPU 4B 모델이지만 Worker의 컨텍스트를 절약하는 것이 본 목적.
        큰 문서를 Scout가 흡수해 요약만 Worker에 넘기면 Worker는 8K ctx 안에서
        여유롭게 동작할 수 있다.
      - Read/Glob/Grep/LS: 코드/파일 탐색
      - DocumentProcess: 업로드된 PDF/DOCX/XLSX를 청크 단위로 파싱
      - 전부 is_read_only=True라 권한 프롬프트가 뜨지 않음
      - 수정 도구(Edit/Write/Bash)는 포함하지 않음 (fail-closed)
    """
    from core.tools.implementations.document_tool import DocumentProcessTool
    from core.tools.implementations.glob_tool import GlobTool
    from core.tools.implementations.grep_tool import GrepTool
    from core.tools.implementations.ls_tool import LSTool
    from core.tools.implementations.read_tool import ReadTool
    from core.tools.implementations.symbol_search_tool import SymbolSearchTool
    from core.tools.registry import ToolRegistry

    registry = ToolRegistry()
    registry.register_many(
        [
            ReadTool(),
            GlobTool(),
            GrepTool(),
            LSTool(),
            DocumentProcessTool(),
            # Phase 10.0 — 심볼 단위 인덱스 기반 함수/클래스 위치 검색
            SymbolSearchTool(),
        ]
    )
    return registry


def _create_cli_tool_registry():  # noqa: ANN202
    """
    CLI Worker용 실행 전용 도구 레지스트리 (사양서 Part 2.4 원본 복원).

    원칙: Worker는 "실행에만 집중". 탐색/조회는 Scout에 위임한다.
    사양서 Part 2.4 WORKER_TOOLS_TIER_S 정의:
        ["Edit", "Write", "Bash", "GitCommit", "GitDiff"]
    본 구현은 여기에 Agent(서브에이전트 호출) + SymbolSearch를 추가해 실측 7개다.

    Read/Glob/Grep/LS/GitLog/GitStatus는 **Scout 전용**으로 이관됨.
    Worker가 파일 탐색이 필요하면 Agent(subagent_type="scout")를 호출한다.
    """
    from core.tools.implementations.agent_tool import AgentTool
    from core.tools.implementations.bash_tool import BashTool
    from core.tools.implementations.edit_tool import EditTool
    from core.tools.implementations.git_tools import GitCommitTool, GitDiffTool
    from core.tools.implementations.symbol_search_tool import SymbolSearchTool
    from core.tools.implementations.write_tool import WriteTool
    from core.tools.registry import ToolRegistry

    registry = ToolRegistry()
    registry.register_many(
        [
            # 실행 전용 도구 (Part 2.4 원본)
            EditTool(),
            WriteTool(),
            BashTool(),
            GitCommitTool(),
            GitDiffTool(),
            # 서브에이전트 호출 — Worker가 Scout 등 탐색자/전문가에 위임
            AgentTool(),
            # Phase 10.0 — Worker도 심볼 위치 빠르게 찾을 수 있도록 추가
            SymbolSearchTool(),
        ]
    )

    return registry


def _create_web_tool_registry(tier: Any = None):  # noqa: ANN202
    """
    웹 Worker용 도구 레지스트리 (하드웨어 티어 연동, B200 Phase 2).

    ── TIER_S (RTX 5090, 8K ctx) — 현행 그대로(무회귀) ──
    실행 전용 5개: Edit / Write / Bash / Agent / SymbolSearch.
    CLI보다 보수적으로 구성한다 — 파일 탐색(Read/Glob/Grep/LS)은 Scout 전용이며
    Agent(subagent_type="scout")로 위임한다. 8K 컨텍스트에 큰 데이터가 직접
    적재되는 상황을 구조적으로 차단하기 위함이다.

    ── TIER_M/L (80GB+ , 32K/128K ctx) — 문서 도구만 추가 ──
    컨텍스트가 넉넉하므로 문서 처리·생성 도구 3개를 추가한다:
    DocumentProcess / DocumentExport / GitDiff.
    ★ 파일탐색(Read/Glob/Grep/LS)은 웹 표면에서 **의도적으로 제외**한다(2026-07-08 결정).
      사양서 v7.0 Part 2.5는 "상위 티어 Worker는 직접 탐색"을 규정하지만, 그 규정은
      로컬 파일시스템을 가진 CLI/에이전트 Worker를 전제한 것이다. 웹 채팅 사용자는
      뒤질 파일시스템이 없어(세션 cwd 격리) Glob/Grep/LS/Read가 항상 빈손을 반환해
      혼란만 준다. 웹 사용자의 "파일"은 업로드 문서(DocumentProcess) + 지식 RAG(자동
      주입)이므로 그쪽으로만 노출한다. 코드 심볼 위치는 SymbolSearch(TIER_S 공통)가 담당.
    GitCommit은 웹 정책상 **계속 제외**한다(일반 사용자 UI에서 커밋 금지).

    fail-closed: tier가 None이거나 알 수 없는 값이면 TIER_S(5개)로 유지한다.

    P5(anti-pattern #5) 준수: 최종 도구 순서는 registry.get_all_tools()가
    name 기준으로 정렬하므로, 여기서의 등록 순서는 prompt cache 안정성에 무관하다.

    Args:
        tier: HardwareTier 열거형(또는 그 .value 문자열). 미지정 시 TIER_S.
    """
    from core.model.hardware_tier import HardwareTier
    from core.tools.implementations.agent_tool import AgentTool
    from core.tools.implementations.bash_tool import BashTool
    from core.tools.implementations.edit_tool import EditTool
    from core.tools.implementations.symbol_search_tool import SymbolSearchTool
    from core.tools.implementations.todo_tools import TodoReadTool, TodoWriteTool
    from core.tools.implementations.write_tool import WriteTool
    from core.tools.registry import ToolRegistry

    registry = ToolRegistry()
    # TIER_S(기본) — 실행 + 계획 체크리스트. 모든 티어의 공통 하위집합.
    # TodoWrite/TodoRead: 계획 메타데이터만 갱신(외부 부작용 없음). 웹 사용자에게
    # "지금 몇 번째 단계"를 가시화하고, compact 후에도 계획을 유지하기 위해 웹
    # Worker 풀에도 포함한다(파일 탐색 도구와 달리 세션 격리 cwd와 무관하게 유효).
    registry.register_many(
        [
            EditTool(),  # 편집 (~325 토큰)
            WriteTool(),  # 쓰기 (~225 토큰)
            BashTool(),  # 실행 (~275 토큰)
            AgentTool(),  # 서브에이전트 호출 (~300 토큰)
            SymbolSearchTool(),  # Phase 10.0 심볼 검색 (~200 토큰)
            TodoWriteTool(),  # 계획 체크리스트 갱신 (전체 목록 원자 교체)
            TodoReadTool(),  # 계획 체크리스트 조회 (읽기 전용)
        ]
    )

    # TIER_M/L 확장 — enum이면 .value, 문자열이면 그대로 비교(둘 다 허용).
    # 매칭 실패(None/미지 값) 시 아래 블록을 건너뛰어 TIER_S로 폴백 = fail-closed.
    tier_val = getattr(tier, "value", tier)
    if tier_val in (HardwareTier.TIER_M.value, HardwareTier.TIER_L.value):
        from core.tools.implementations.document_export_tool import DocumentExportTool
        from core.tools.implementations.document_tool import DocumentProcessTool
        from core.tools.implementations.git_tools import GitDiffTool

        registry.register_many(
            [
                DocumentProcessTool(),  # 업로드 문서(.pdf/.docx/.xlsx/.hwp/.pptx) 파싱
                DocumentExportTool(),  # 문서 생성(.docx/.pptx/.hwpx/.md/.txt) + 다운로드
                GitDiffTool(),  # git 변경 조회(읽기 전용). GitCommit은 제외.
                # ※ Read/Glob/Grep/LS는 웹 표면에서 제외 — 위 docstring 근거 참조.
            ]
        )

    return registry


def _build_default_system_prompt(agent_registry: Any | None = None) -> str:
    """
    CLI Worker용 기본 시스템 프롬프트 (사양서 Part 2.4 원본 복원).

    원칙 (사양서 Part 2.4):
      - Worker는 실행 전용 (Edit/Write/Bash/GitCommit/GitDiff/Agent)
      - 파일 탐색·읽기·검색은 모두 Scout에 위임
      - Worker는 Scout JSON 결과를 해석해 최종 답변 생성
    """
    base = (
        "You are Nexus, the Worker agent in an air-gapped environment.\n"
        "You are a 27B model — the brain. Scout (a 4B helper) does all file "
        "exploration for you.\n\n"
        "## Your tools (execution only)\n"
        "- Edit: edit an existing file\n"
        "- Write: create a new file (ONLY when the user explicitly asks)\n"
        "- Bash: run a shell command\n"
        "- GitCommit / GitDiff: git operations\n"
        "- Agent: delegate exploration to Scout (subagent_type='scout')\n\n"
        "You do NOT have Read/Glob/Grep/LS/DocumentProcess. Scout does. When you "
        "need to read a file, search code, list a directory, or analyze a document "
        "(.pdf/.docx/.xlsx/.hwp/.pptx), call:\n"
        "  Agent(prompt='<what you need>', subagent_type='scout')\n\n"
        "## Scout's response (markdown report)\n"
        "Scout returns 4 markdown sections:\n"
        "  ## relevant_files, ## file_summaries, ## plan, ## requires_tools\n"
        "Read `## plan` carefully — those bullets are the facts you need. Then "
        "produce a detailed natural answer in the user's language based on those "
        "facts.\n\n"
        "## CRITICAL — Scout invocation limit\n"
        "Call Agent(subagent_type='scout') AT MOST ONCE per user turn. After Scout "
        "returns, answer the user with whatever information you received. NEVER "
        "call Scout a second time in the same turn — it loops. If the plan seems "
        "sparse, mention that to the user and work with what you have.\n\n"
        "## Conversational style (greetings & small talk)\n"
        "For a short greeting or small talk (안녕, 좋은 아침, hi, thanks, 잘 자 등):\n"
        "- Reply briefly and warmly in the user's language — one or two short "
        "sentences — and stop.\n"
        "- Do NOT volunteer encyclopedic facts, song/movie/book references, or "
        "trivia even if the words look like a title.\n"
        "- Do NOT pivot to a topic the user did not ask about.\n\n"
        "## When a `--- Knowledge base ---` block is present\n"
        "Treat the snippets as a candidate reference, NOT as the answer:\n"
        "- Use them ONLY when clearly on-topic.\n"
        "- If the snippets are off-topic, irrelevant, or contradict common-sense, "
        "IGNORE them and answer from your own general knowledge.\n"
        "- Never quote/list off-topic snippets just because they were retrieved.\n\n"
        "## Hard rules\n"
        "- NEVER create a file the user didn't ask for.\n"
        "- NEVER attempt Read/Glob/Grep/LS — those tools aren't available to you.\n"
        "- Simple conversational questions → answer directly, no tools.\n"
    )

    # 서브에이전트 목록 동적 주입
    if agent_registry is None or len(agent_registry) == 0:
        return base

    agent_lines = []
    for name, desc in agent_registry.list_descriptions().items():
        agent_lines.append(f"  - {name}: {desc}")

    subagent_guide = "\n## Registered sub-agents\n" + "\n".join(agent_lines) + "\n"
    return base + subagent_guide


# ─────────────────────────────────────────────
# 로깅 구성
# ─────────────────────────────────────────────
def _configure_logging(level: str, log_file: str | None) -> None:
    """루트 로거를 구성한다 — 콘솔(stderr)과 선택적 파일 핸들러를 함께 건다.

    Args:
        level: "INFO"/"DEBUG" 등 문자열 레벨. 알 수 없으면 INFO로 폴백한다.
        log_file: 로그 파일 경로. None이면 콘솔 출력만 사용한다. 경로가 주어지면
            상위 디렉토리를 미리 만들고 UTF-8로 append 한다.
    """
    # 문자열 레벨명을 logging 상수로 변환. 오타/미지 값은 INFO로 안전 폴백.
    log_level = getattr(logging, level.upper(), logging.INFO)
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]

    if log_file:
        # 로그 디렉토리가 없으면 생성
        log_dir = Path(log_file).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,
    )

    # httpx의 요청 로그가 CLI에서 사용자 프롬프트를 덮어쓰므로
    # WARNING 이상만 표시한다 (RAG 인덱싱 시 대량 POST 로그 방지)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


# ─────────────────────────────────────────────
# 정상 종료 핸들러
# ─────────────────────────────────────────────
# 종료 시 실행할 클린업 함수 목록
_cleanup_handlers: list = []


def register_cleanup(handler) -> None:
    """종료 시 실행할 클린업 함수를 등록한다.

    진입점(CLI/웹)이 Redis·PG 풀 close 등 자원 정리 콜백을 미리 걸어두는 용도다.
    여기 등록된 핸들러는 SIGINT/SIGTERM 수신 시 _shutdown_handler에서 순서대로 실행된다.
    """
    _cleanup_handlers.append(handler)


def _setup_graceful_shutdown(state: GlobalState) -> None:
    """
    정상 종료 핸들러를 등록한다.
    Claude Code의 setupGracefulShutdown()에 대응한다.

    SIGINT/SIGTERM 수신 시:
      1. 진행 중인 도구 실행을 취소한다
      2. 세션 요약을 로깅한다
      3. 등록된 클린업 함수를 실행한다
    """

    def _shutdown_handler(signum, frame):
        """시그널 핸들러: 정상 종료를 수행한다."""
        sig_name = signal.Signals(signum).name
        logger.info(f"종료 시그널 수신: {sig_name}")

        # 세션 요약 로깅
        summary = state.get_session_summary()
        logger.info(f"세션 요약: {summary}")

        # 등록된 클린업 함수 실행
        for handler in _cleanup_handlers:
            try:
                handler()
            except Exception as e:
                logger.error(f"클린업 핸들러 실행 실패: {e}")

        sys.exit(0)

    # SIGINT (Ctrl+C), SIGTERM 핸들러 등록
    signal.signal(signal.SIGINT, _shutdown_handler)
    # Windows에서는 SIGTERM이 지원되지 않을 수 있다
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _shutdown_handler)


# ─────────────────────────────────────────────
# GPU 서버 사전 연결
# ─────────────────────────────────────────────
async def _preconnect_gpu_server(gpu_server_url: str) -> None:
    """
    GPU 서버에 사전 연결한다.
    Claude Code의 preconnectAnthropicApi()에 대응한다.
    HTTP 커넥션 풀을 워밍업하여 첫 추론 요청에서 100-200ms를 절약한다.
    """
    try:
        import httpx

        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{gpu_server_url}/health")
            if resp.status_code == 200:
                # vLLM /health는 빈 body(200 OK)를 반환한다
                # JSON body가 있으면 파싱, 없으면 연결 성공만 기록
                if resp.text.strip():
                    data = resp.json()
                    logger.info(
                        "[Phase 1] GPU 서버 사전 연결 성공: gpu=%s, tier=%s",
                        data.get("gpu", "unknown"),
                        data.get("gpu_tier", "unknown"),
                    )
                else:
                    logger.info("[Phase 1] GPU 서버 사전 연결 성공 (vLLM healthy)")
            else:
                logger.warning(f"[Phase 1] GPU 서버 상태 이상: status={resp.status_code}")
    except ImportError:
        logger.warning("[Phase 1] httpx가 설치되지 않아 사전 연결을 건너뜁니다")
    except Exception as e:
        # fire-and-forget이므로 실패해도 계속 진행한다
        logger.warning(f"[Phase 1] GPU 서버 사전 연결 실패: {e}")


# ─────────────────────────────────────────────
# 플랫폼 감지
# ─────────────────────────────────────────────
def _detect_platform() -> dict:
    """실행 중인 플랫폼 정보를 수집해 dict로 반환한다.

    OS/버전/파이썬버전/아키텍처와 함께, Bash 도구가 사용할 셸 경로를 고른다.
    Windows면 git-bash를 우선 찾고 없으면 COMSPEC(cmd), 그 외 OS는 SHELL을
    사용한다. Claude Code의 setShellIfWindows()에 대응한다.
    """
    info = {
        "os": platform.system(),
        "os_version": platform.version(),
        "python_version": platform.python_version(),
        "arch": platform.machine(),
    }

    # Windows에서 git-bash 감지
    if info["os"] == "Windows":
        git_bash = Path("C:/Program Files/Git/bin/bash.exe")
        if git_bash.exists():
            info["shell"] = str(git_bash)
        else:
            info["shell"] = os.environ.get("COMSPEC", "cmd.exe")
    else:
        info["shell"] = os.environ.get("SHELL", "/bin/bash")

    return info
