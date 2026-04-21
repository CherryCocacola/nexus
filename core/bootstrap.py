"""
부트스트랩 — 2-Phase 초기화 시스템.

Claude Code의 entrypoints/init.ts + setup.ts를 Python으로 재구현한다.

Phase 1 (이 파일): 환경 비의존 초기화
  - GlobalState 생성
  - 설정 로딩 + 검증
  - 로깅 구성
  - 정상 종료 핸들러 등록
  - GPU 서버 사전 연결 (fire-and-forget)
  - 플랫폼 감지

Phase 2 (init_phase2): 환경 의존 초기화
  - ToolRegistry (24개 도구 등록)
  - MemoryManager (단기+장기 메모리)
  - QueryEngine (Tier 1 세션 오케스트레이터)

왜 2-Phase인가: Claude Code가 이 패턴을 사용하는 이유는
Phase 1이 src/ 내부 모듈을 전혀 import하지 않아서
순환 의존이 원천 차단되기 때문이다 (DAG leaf 격리).
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

from core.config import load_and_validate_config
from core.state import GlobalState, get_initial_state

logger = logging.getLogger("nexus.bootstrap")


# ─────────────────────────────────────────────
# Phase 1: 환경 비의존 초기화
# ─────────────────────────────────────────────
async def init(
    config_path: str | None = None,
    cwd: str | None = None,
) -> GlobalState:
    """
    Phase 1 초기화. CLI, SDK, 웹 서버 등 어떤 진입점이든 이 함수를 먼저 호출한다.
    bootstrap 모듈만 사용하며, core/ 내부의 다른 모듈은 import하지 않는다.

    Args:
        config_path: 설정 파일 경로. None이면 자동 탐색한다.
        cwd: 작업 디렉토리. None이면 현재 디렉토리를 사용한다.

    Returns:
        초기화된 GlobalState 싱글톤
    """
    # ① GlobalState 초기화
    state = get_initial_state(cwd=cwd)
    logger.info(f"[Phase 1] GlobalState 초기화 완료: session={state.session_id}")

    # ② 설정 로딩 + 검증
    config = load_and_validate_config(config_path)
    state.config = config
    logger.info(f"[Phase 1] 설정 로드 완료: gpu_server={config.gpu_server_url}")

    # ③ 로깅 구성
    _configure_logging(config.log_level, config.log_file)

    # ④ 정상 종료 핸들러 등록
    _setup_graceful_shutdown(state)
    logger.info("[Phase 1] 종료 핸들러 등록 완료")

    # ⑤ GPU 서버 사전 연결 (fire-and-forget)
    # 왜 fire-and-forget인가: GPU 서버가 아직 안 떠 있어도 부트스트랩은 진행해야 한다.
    # 첫 추론 요청 전까지 연결이 되면 100-200ms를 절약할 수 있다.
    asyncio.create_task(_preconnect_gpu_server(config.gpu_server_url))

    # ⑥ 플랫폼 감지
    state.platform = _detect_platform()
    logger.info(f"[Phase 1] 플랫폼: {state.platform.get('os', 'unknown')}")

    return state


# ─────────────────────────────────────────────
# Phase 2: 환경 의존 초기화
# ─────────────────────────────────────────────
async def init_phase2(state: GlobalState) -> dict:
    """
    Phase 2 초기화. Phase 1(init) 이후 호출한다.
    core/ 내부 모듈(ToolRegistry, MemoryManager, QueryEngine)을 초기화한다.

    Args:
        state: Phase 1에서 생성된 GlobalState

    Returns:
        초기화된 컴포넌트 딕셔너리:
          - tool_registry: ToolRegistry (24개 도구 등록됨)
          - memory_manager: MemoryManager (인메모리 폴백)
          - model_provider: LocalModelProvider
          - query_engine: QueryEngine (Tier 1)
    """
    # lazy import — Phase 2 모듈은 Phase 1에서 import하지 않는다
    from core.memory.long_term import LongTermMemory
    from core.memory.manager import MemoryManager
    from core.memory.short_term import ShortTermMemory
    from core.memory.transcript import SessionTranscript
    from core.model.inference import LocalModelProvider
    from core.orchestrator.agent_definition import build_default_agent_registry
    from core.orchestrator.query_engine import QueryEngine
    from core.task import TaskManager
    from core.tools.base import ToolUseContext

    components: dict = {}

    # ① ModelProvider 생성
    # LLM과 임베딩 서버가 별도 포트/인스턴스이므로 각각 URL을 전달한다
    config = state.config
    provider = LocalModelProvider(
        base_url=config.gpu_server_url,
        model_id=config.model.primary_model,
        max_context_tokens=config.model.max_context_tokens,
        max_output_tokens=config.model.default_max_tokens,
        embedding_base_url=config.gpu_server.embedding_url,
    )
    components["model_provider"] = provider
    logger.info("[Phase 2] ModelProvider 초기화: %s", config.gpu_server_url)

    # ② ToolRegistry — 24개 도구 등록
    registry = _create_tool_registry()
    components["tool_registry"] = registry
    logger.info("[Phase 2] ToolRegistry 초기화: %d개 도구", registry.tool_count)

    # ③ MemoryManager — Redis/PostgreSQL 실 연결 (실패 시 인메모리 폴백)
    redis_client = await _create_redis_client(config)
    pg_pool = await _create_pg_pool(config)
    stm = ShortTermMemory(redis_client=redis_client)
    ltm = LongTermMemory(pg_pool=pg_pool)
    memory_manager = MemoryManager(
        short_term=stm,
        long_term=ltm,
        model_provider=provider,
    )
    components["memory_manager"] = memory_manager
    components["redis_client"] = redis_client
    components["pg_pool"] = pg_pool
    mode = "Redis+PG" if (redis_client and pg_pool) else (
        "Redis만" if redis_client else ("PG만" if pg_pool else "인메모리 폴백")
    )
    logger.info("[Phase 2] MemoryManager 초기화: %s", mode)

    # ③-b TaskManager — 비동기 태스크 라이프사이클 관리
    task_manager = TaskManager()
    components["task_manager"] = task_manager
    logger.info("[Phase 2] TaskManager 초기화")

    # ④ AgentRegistry 초기화 — v7.0 Phase 9 서브에이전트 시스템
    # SCOUT_AGENT 등 기본 에이전트가 등록된다.
    agent_registry = build_default_agent_registry()
    components["agent_registry"] = agent_registry

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
            "agent_registry": agent_registry,
            "model_provider": provider,
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

    # ⑧ 티어별 도구 레지스트리 자동 선택
    # TIER_S: 11개 도구 (~1,472토큰) — 컨텍스트 절약
    # TIER_M/L: 24개 도구 전체 — 컨텍스트 충분
    if tier == HardwareTier.TIER_S:
        cli_registry = _create_cli_tool_registry()
    else:
        cli_registry = _create_tool_registry()  # 24개 전체
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
        knowledge_retriever = KnowledgeRetriever(
            store=knowledge_store,
            embedding_provider=provider,  # e5-large 임베딩 서버 경유
        )
        components["knowledge_store"] = knowledge_store
        components["knowledge_retriever"] = knowledge_retriever
        count = await knowledge_store.count()
        logger.info(
            "[Phase 2] KnowledgeStore 초기화: 레코드=%d (pg=%s)",
            count, "connected" if pg_pool else "in-memory",
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
            count, "connected" if pg_pool else "in-memory",
        )
    except Exception as e:
        logger.warning("[Phase 2] 심볼 인덱스 초기화 실패 (무시): %s", e)
        components["symbol_store"] = None

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

    context_manager = ContextManager(
        model_provider=provider,
        max_context_tokens=config.model.max_context_tokens,
        tier=tier,  # HardwareTier 전달 → TIER_S에서는 자동 pass-through
    )
    components["context_manager"] = context_manager
    logger.info(
        "[Phase 2] ContextManager 초기화: tier=%s, passthrough=%s",
        tier.value, context_manager.passthrough,
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
        rag_retriever=rag_retriever,
        model_dispatcher=dispatcher,
        routing_config=config.routing,  # v7.0 Part 2.5 — 지식/도구 분기
        memory_manager=memory_manager,  # Ch 16: Redis + tb_memories 자동 저장
        transcript=cli_transcript,  # Ch 16: JSONL 영구 기록
        knowledge_retriever=knowledge_retriever,  # Part 2.5.8: tb_knowledge RAG
    )
    components["query_engine"] = engine
    logger.info(
        "[Phase 2] QueryEngine 초기화: session=%s, 도구=%d개, tier=%s, rag=%s",
        engine.session_id, len(cli_tools), tier.value,
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


async def _create_redis_client(config: Any) -> Any:
    """
    Redis 비동기 클라이언트를 생성한다.
    연결 실패 시 None 반환 (인메모리 폴백).
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
    """
    PostgreSQL asyncpg 풀을 생성한다.
    연결 실패 시 None 반환 (인메모리 폴백).
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
    """24개 도구를 등록한 ToolRegistry를 생성한다."""
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
    from core.tools.implementations.task_tools import TaskTool, TodoReadTool, TodoWriteTool
    from core.tools.implementations.write_tool import WriteTool
    from core.tools.registry import ToolRegistry

    registry = ToolRegistry()
    registry.register_many([
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
    ])

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
    registry.register_many([
        ReadTool(),
        GlobTool(),
        GrepTool(),
        LSTool(),
        DocumentProcessTool(),
        # Phase 10.0 — 심볼 단위 인덱스 기반 함수/클래스 위치 검색
        SymbolSearchTool(),
    ])
    return registry


def _create_cli_tool_registry():  # noqa: ANN202
    """
    CLI Worker용 실행 전용 도구 레지스트리 (사양서 Part 2.4 원본 복원).

    원칙: Worker는 "실행에만 집중". 탐색/조회는 Scout에 위임한다.
    사양서 Part 2.4 WORKER_TOOLS_TIER_S 정의:
        ["Edit", "Write", "Bash", "GitCommit", "GitDiff"]
    본 구현은 여기에 Agent(서브에이전트 호출)을 추가해 6개다.

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
    registry.register_many([
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
    ])

    return registry


def _create_web_tool_registry():  # noqa: ANN202
    """
    웹 Worker용 실행 전용 도구 레지스트리 (사양서 Part 2.4 원본 복원).

    CLI보다 더 보수적으로 구성한다 — GitCommit/GitDiff는 제외.
    웹 UI는 일반 사용자 인터페이스이므로 "대화 + 파일 편집 + 명령 실행 +
    서브에이전트 호출"만 제공한다.

    사양서 Part 2.4의 WORKER_TOOLS_TIER_S는 {Edit, Write, Bash, GitCommit,
    GitDiff}이지만, 웹에서는 Git은 일반 사용자 용도가 아니므로 CLI로만.

    Read/Glob/Grep/LS는 Scout 전용. 파일 탐색은 Agent(subagent_type="scout")
    로 위임한다. 이렇게 하면 Worker 컨텍스트(8K)에 큰 데이터가 직접 적재되는
    상황 자체를 구조적으로 차단한다.
    """
    from core.tools.implementations.agent_tool import AgentTool
    from core.tools.implementations.bash_tool import BashTool
    from core.tools.implementations.edit_tool import EditTool
    from core.tools.implementations.symbol_search_tool import SymbolSearchTool
    from core.tools.implementations.write_tool import WriteTool
    from core.tools.registry import ToolRegistry

    registry = ToolRegistry()
    registry.register_many([
        EditTool(),           # 편집 (~325 토큰)
        WriteTool(),          # 쓰기 (~225 토큰)
        BashTool(),           # 실행 (~275 토큰)
        AgentTool(),          # 서브에이전트 호출 (~300 토큰)
        SymbolSearchTool(),   # Phase 10.0 심볼 검색 (~200 토큰)
    ])

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

    subagent_guide = (
        "\n## Registered sub-agents\n"
        + "\n".join(agent_lines)
        + "\n"
    )
    return base + subagent_guide


# ─────────────────────────────────────────────
# 로깅 구성
# ─────────────────────────────────────────────
def _configure_logging(level: str, log_file: str | None) -> None:
    """
    로깅을 구성한다.
    콘솔(stderr) + 파일(선택) 핸들러를 설정한다.
    """
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
    """종료 시 실행할 클린업 함수를 등록한다."""
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
                        "[Phase 1] GPU 서버 사전 연결 성공: "
                        "gpu=%s, tier=%s",
                        data.get("gpu", "unknown"),
                        data.get("gpu_tier", "unknown"),
                    )
                else:
                    logger.info("[Phase 1] GPU 서버 사전 연결 성공 (vLLM healthy)")
            else:
                logger.warning(
                    f"[Phase 1] GPU 서버 상태 이상: status={resp.status_code}"
                )
    except ImportError:
        logger.warning("[Phase 1] httpx가 설치되지 않아 사전 연결을 건너뜁니다")
    except Exception as e:
        # fire-and-forget이므로 실패해도 계속 진행한다
        logger.warning(f"[Phase 1] GPU 서버 사전 연결 실패: {e}")


# ─────────────────────────────────────────────
# 플랫폼 감지
# ─────────────────────────────────────────────
def _detect_platform() -> dict:
    """
    플랫폼 정보를 감지한다.
    Claude Code의 setShellIfWindows()에 대응한다.
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
