"""
설정 시스템 — Pydantic v2 기반 설정 로딩 + 검증.

Claude Code의 enableConfigs() + applySafeConfigEnvironmentVariables()에 대응한다.
3단계 우선순위로 설정을 로드한다:
  1. 기본값 (이 파일에 정의)
  2. YAML 설정 파일 (config/nexus_config.yaml)
  3. 환경변수 (NEXUS_ 접두사)

에어갭 검증: GPU 서버 URL이 로컬/LAN 주소인지 자동으로 확인한다.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings

# LAN/에어갭 판정 헬퍼 — 보안 관심사이므로 core/security로 이동했다(P2: config → security 허용).
from core.security.network_guard import is_lan_hostname

logger = logging.getLogger("nexus.config")

# 하위호환 별칭 — 기존 import 경로(from core.config import _is_lan_hostname) 유지용.
# 신규 코드는 public 이름 is_lan_hostname을 직접 쓴다.
_is_lan_hostname = is_lan_hostname


# ─────────────────────────────────────────────
# GPU 서버 설정
# ─────────────────────────────────────────────
class GPUServerConfig(BaseModel):
    """
    GPU 서버 (Machine B) 연결 설정.

    2-Machine 토폴로지에서 추론을 담당하는 vLLM 서버(OpenAI 호환 API)의 접속
    정보를 담는다. Machine A(오케스트레이터)의 ModelClient/추론 클라이언트가
    이 값을 읽어 LAN 너머의 GPU 서버로 요청을 보낸다.
    """

    # vLLM 메인 추론 서버 URL. 개발 기본값은 localhost지만 실제 운영에서는
    # yaml/환경변수로 GPU 서버 LAN 주소(예: http://192.168.21.112:8000)를 넣는다.
    url: str = "http://localhost:8000"
    # 임베딩 전용 vLLM 인스턴스 — 메인 추론과 부하/모델을 분리하려고 포트를
    # 따로 둔다(8002). e5-large 임베딩 호출이 여기로 간다.
    embedding_url: str = "http://localhost:8002"
    # HTTP 요청 1건의 최대 대기 시간(초). 27B 모델의 긴 생성도 끊기지 않도록
    # 넉넉히 120초로 둔다(짧게 잡으면 정상 추론이 타임아웃으로 끊긴다).
    timeout_seconds: float = 120.0
    # 일시적 네트워크/서버 오류 시 재시도 횟수. GPU 서버 재기동·잠깐의 과부하를
    # 견디도록 10회로 둔다(Tier 4 with_retry에서 사용).
    max_retries: int = 10
    # 재시도 사이 대기의 기준 시간(초). 보통 지수 백오프의 base로 쓰여
    # 0.5 → 1.0 → 2.0초 식으로 점점 늘어난다.
    retry_base_delay: float = 0.5
    # GPU 서버 헬스체크 주기(초). 30초마다 살아 있는지 확인한다.
    health_check_interval: float = 30.0

    @field_validator("url")
    @classmethod
    def validate_url_is_local(cls, v: str) -> str:
        """
        에어갭 검증: GPU 서버 URL이 로컬/LAN 주소인지 확인한다.

        동작 단계:
          1. URL을 파싱해 hostname만 뽑는다.
          2. 허용 prefix(localhost/127.0.0.1/사설망 대역)로 시작하는지 본다.
          3. 어디에도 안 맞으면(=외부 주소로 의심) 경고만 띄운다.
        주의: 여기서는 예외를 던지지 않고 warning만 낸다 — 잘못된 설정으로
        로드 자체가 실패해 본류가 멈추는 것을 피하려는 의도다(경고로 알리되
        진행은 시킨다). 실제 차단은 에어갭 모드 검증/네트워크 계층에서 한다.
        """
        parsed = urlparse(v)
        hostname = parsed.hostname or ""
        # 사설망(RFC1918) + 로컬호스트 대역만 허용. 10./172./192.168.은 LAN 대역.
        allowed_prefixes = ("localhost", "127.0.0.1", "10.", "172.", "192.168.")
        if not any(hostname.startswith(p) for p in allowed_prefixes):
            import warnings

            warnings.warn(
                f"GPU 서버 URL '{v}'이(가) 외부 주소로 보입니다. "
                f"에어갭 환경에서는 작동하지 않을 수 있습니다.",
                UserWarning,
                stacklevel=2,
            )
        return v


# ─────────────────────────────────────────────
# Redis 설정 (단기 메모리 / 세션 캐시)
# ─────────────────────────────────────────────
class RedisConfig(BaseModel):
    """
    Redis 연결 설정 (단기 메모리 / 세션 캐시).

    core/memory/short_term이 세션·대화의 휘발성 상태를 여기에 저장한다.
    장기 기억(PostgreSQL+pgvector)과 달리 빠른 읽기/쓰기와 TTL 만료가 목적이다.
    """

    host: str = "localhost"  # Redis 서버 주소. 운영 시 LAN 주소로 오버라이드.
    port: int = 6379  # Redis 표준 포트.
    # 논리 DB 번호(0~15). 같은 Redis 인스턴스를 변형(예: G2)과 나눠 쓸 때
    # 번호로 격리한다(기본 0번 사용).
    db: int = 0
    password: str | None = None  # 인증이 없으면 None. LAN 내부라 기본은 무인증.
    # 소켓 연산 1건의 타임아웃(초). 단기 캐시는 빠르게 응답해야 하므로 5초로 짧게.
    socket_timeout: float = 5.0


# ─────────────────────────────────────────────
# PostgreSQL 설정 (장기 메모리 + pgvector)
# ─────────────────────────────────────────────
class PostgreSQLConfig(BaseModel):
    """
    PostgreSQL 연결 설정 (장기 메모리 + pgvector).

    core/memory/long_term이 쓰는 영속 저장소. pgvector 확장으로 임베딩 벡터를
    저장/검색하며, 지식 베이스(tb_knowledge)·기억(tb_memories) 등이 여기 있다.
    """

    host: str = "localhost"  # PG 서버 주소. 운영 시 DB 서버 LAN 주소로 오버라이드.
    port: int = 5432  # PostgreSQL 표준 포트.
    database: str = "nexus"  # 접속할 데이터베이스 이름.
    user: str = "nexus"  # 접속 계정.
    password: str = ""  # 비밀번호. 빈 문자열이면 yaml/환경변수로 주입 전제.
    # 커넥션 풀의 최소 유지 개수. 항상 2개는 열어두어 첫 요청 지연을 줄인다.
    min_connections: int = 2
    # 커넥션 풀의 최대 개수. 동시 요청이 몰려도 이 수를 넘기지 않아 DB를 보호한다.
    max_connections: int = 10


# ─────────────────────────────────────────────
# 모델 설정
# ─────────────────────────────────────────────
class ModelConfig(BaseModel):
    """
    LLM 모델 설정.

    추론에 쓰는 모델 이름과 기본 생성 파라미터를 담는다. 여기 이름들은 vLLM의
    served-model-name과 일치해야 GPU 서버가 올바른 모델로 라우팅한다.
    (질의 타입별 세부 분기는 RoutingConfig가 따로 담당한다.)
    """

    # 주 추론 모델 — 27B Qwen. 일반 대화/도구 호출의 기본.
    primary_model: str = "qwen3.5-27b"
    # 보조 모델 — 한국어 특화 ExaOne. 한국어 품질이 중요한 경로에서 보조로 쓴다.
    auxiliary_model: str = "exaone-7.8b"
    # 임베딩 모델 — 다국어 e5-large. RAG 벡터 검색용 임베딩 생성에 사용.
    embedding_model: str = "multilingual-e5-large"
    # 컨텍스트 윈도우 상한(토큰). RTX 5090(32GB) 제약상 보수적으로 4096.
    # (운영 라우팅에서는 프로필별로 별도 max_tokens를 두기도 한다.)
    max_context_tokens: int = 4096
    # 기본 샘플링 온도. 라우팅 프로필이 없을 때의 폴백 값(0.7=다소 창의적).
    default_temperature: float = 0.7
    # 응답 생성 토큰 상한의 기본값. 프로필이 지정되면 그 값이 우선한다.
    default_max_tokens: int = 4096


# ─────────────────────────────────────────────
# 세션 설정
# ─────────────────────────────────────────────
class SessionConfig(BaseModel):
    """
    세션 관리 설정.

    QueryEngine(Tier 1)이 한 세션의 수명·예산·기록을 통제하는 데 쓰는 값들이다.
    무한 루프나 폭주를 막는 안전장치(턴 수·시간 예산)가 핵심이다.
    """

    # 세션/대화 기록을 저장할 디렉토리(상대경로 — 작업 디렉토리 기준).
    sessions_dir: str = ".nexus/sessions"
    # 한 세션에서 허용하는 최대 에이전트 턴 수. query_loop의 while(True)가
    # 도구 호출로 끝없이 돌지 않도록 50턴에서 강제 종료한다(폭주 방지 안전장치).
    max_turns: int = 50
    # 한 세션의 누적 시간 예산(초). 5분을 넘기면 중단한다 — 응답이 한없이
    # 길어지는 것을 막는 시간 기준 안전장치.
    max_budget_seconds: float = 300.0
    # 세션 데이터의 유효 기간(시간). 24시간 지난 세션은 만료 대상으로 본다.
    session_ttl_hours: int = 24
    # 대화 전체를 JSONL transcript로 남길지 여부. 기본 True(감사/재현용 기록).
    transcript_enabled: bool = True


# ─────────────────────────────────────────────
# 보안 설정
# ─────────────────────────────────────────────
class ScoutConfig(BaseModel):
    """
    v7.0 Scout(CPU 4B 모델) 설정.

    2026-04-17 업데이트: Gemma 4 E4B → Qwen3.5-4B 전환.
      Worker(Qwen3.5-27B)와 동일 패밀리로 맞춰 토크나이저/chat template/
      tool_call 문법 일관성을 확보한다.
    """

    # Pydantic가 model_ 로 시작하는 필드를 예약어로 보고 경고하는데, 여기엔
    # model_id 등이 있으므로 그 경고를 끈다(실제 동작에는 영향 없음).
    model_config = {"protected_namespaces": ()}  # model_ 접두사 경고 방지

    # Scout 활성 여부. 단, 실제로는 하드웨어 티어가 TIER_S일 때만 자동 켜진다.
    enabled: bool = True  # TIER_S에서만 자동 활성화
    # Scout 전용 vLLM 인스턴스 주소(메인 추론 8000과 다른 8003 포트).
    base_url: str = "http://192.168.21.112:8003"
    api_key: str = "local-key"  # LAN 내부 인증 키(기본 placeholder).
    # served-model-name — Worker(27B)와 같은 Qwen 패밀리의 4B 경량 모델.
    model_id: str = "qwen3.5-4b"
    # Scout가 다룰 입력 컨텍스트 상한(토큰). 경량 보조 역할이라 4096이면 충분.
    max_context_tokens: int = 4096
    # Scout 응답의 출력 토큰 상한. 빠른 사전판단 용도라 짧게 512로 제한한다.
    max_output_tokens: int = 512


# ─────────────────────────────────────────────
# v7.2 MCP(Model Context Protocol) 통합 설정
# ─────────────────────────────────────────────
# 배경: 사내 시스템(PostgreSQL/임베딩 서버/DocUtil 등)을 LAN 내부 MCP 서버로
# 노출하고, 그 도구들을 Nexus 도구 풀에 흡수한다. 외부 SaaS MCP는 여전히 금지.
# 에어갭 원칙은 불변 — base_url은 LAN 대역(192.168/10/172.16~31/localhost)만 허용.
# fail-closed: 모든 enabled 기본값은 False. 운영자가 yaml에서 명시 활성해야만 동작.
# ─────────────────────────────────────────────
class McpServerConfig(BaseModel):
    """
    LAN MCP 서버 1개의 연결 설정.

    name은 도구 이름 규칙 mcp__{name}__{tool}의 {name}으로 쓰인다.
    trust는 신뢰 메타데이터로, {"read_only": true}처럼 어댑터의 동작 플래그를
    명시적으로 완화하는 데 사용한다(fail-closed: 명시하지 않으면 쓰기 도구로 간주).
    """

    name: str  # mcp__{name}__{tool}의 {name} (예: "db", "diag")
    transport: str = "http_sse"  # LAN HTTP/SSE만 채택 (stdio 비채택)
    base_url: str  # 반드시 LAN 대역 — 외부 도메인이면 강등됨
    api_key: str = "local-key"  # LAN 내부 인증 키 (기본 placeholder)
    enabled: bool = False  # fail-closed: 명시 활성만
    trust: dict = Field(default_factory=dict)  # {"read_only": true} 등 신뢰 메타
    # 쓰기 가능 MCP 서버를 의도적으로 등록 허용하는 운영자 명시 플래그.
    # 초기 제품 정책: read-only(trust.read_only=True) 서버만 자동 등록한다.
    # trust.read_only가 False(쓰기 가능)인 서버는 기본적으로 등록을 건너뛰며,
    # 운영자가 이 값을 True로 명시했을 때에만 예외적으로 등록한다(fail-closed 기본 False).
    # 등록된 쓰기 도구의 최종 권한 판단은 표준 5계층 파이프라인에 일임한다.
    allow_write: bool = False
    # 이 MCP 서버의 도구를 Nexus Worker(에이전트) 도구 풀에 노출할지 여부.
    # 기본 True(노출). False로 두면 서버 자체는 (외부 사내 앱 재사용 등을 위해)
    # 설정에는 남지만, 그 도구들은 Worker(ModelDispatcher/QueryEngine) 도구 풀에는
    # 등록하지 않는다.
    #
    # 왜 필요한가: 일부 MCP 서버는 Nexus 내부 경로와 기능이 중복된다.
    # 예) kowiki 검색 MCP는 KNOWLEDGE 모드의 "자동 RAG 주입"이 이미 담당한다.
    # 이 둘이 동시에 컨텍스트에 들어가면(자동 RAG 결과 + 동일 검색 도구 스키마)
    # RTX 5090의 8K 컨텍스트를 초과(overflow)한다. 그래서 자동 RAG가 책임지는
    # 서버는 Worker 도구로 중복 노출하지 않도록 expose_to_worker=False로 제외한다.
    expose_to_worker: bool = True


class McpConfig(BaseModel):
    """
    v7.2 MCP 통합 설정.

    enabled가 전역 마스터 스위치(기본 OFF)다. servers의 각 항목도 개별
    enabled를 가지며, 전역+개별이 모두 True여야 실제 연결을 시도한다.

    LAN URL 이중 검증의 1단계(설정 로드 단계)를 여기서 수행한다:
    base_url의 hostname이 LAN 대역이 아니면 그 서버를 강제로 enabled=False로
    강등하고 경고를 남긴다. 이렇게 하면 외부 도메인이 활성 상태로 남는 일이
    구조적으로 불가능하다(fail-closed). 2단계 검증은 McpClient 생성 시 재수행.
    """

    enabled: bool = False  # 전역 마스터 스위치 (기본 OFF — 에어갭 fail-closed)
    servers: list[McpServerConfig] = Field(default_factory=list)
    connect_timeout_sec: float = 5.0  # 연결 타임아웃 (실패 시 fail-closed)

    @model_validator(mode="after")
    def validate_lan_urls(self) -> McpConfig:
        """
        각 서버의 base_url이 LAN 대역인지 검증한다.

        왜 강등(enabled=False)인가: 외부 도메인을 단순히 거부(예외)하면
        설정 파일 전체 로드가 실패해 본류까지 멈춘다. 대신 해당 서버만
        비활성으로 강등하면 다른 LAN 서버는 정상 동작하면서 외부 연결만
        구조적으로 차단된다(fail-closed + 본류 무영향).
        """
        import warnings

        for server in self.servers:
            if not is_lan_hostname(urlparse(server.base_url).hostname or ""):
                if server.enabled:
                    # 외부 도메인인데 활성으로 설정돼 있으면 강제 강등
                    server.enabled = False
                    msg = (
                        f"MCP 서버 '{server.name}'의 base_url "
                        f"'{server.base_url}'이(가) LAN 대역이 아니므로 "
                        f"비활성으로 강등합니다 (에어갭 fail-closed)."
                    )
                    warnings.warn(msg, UserWarning, stacklevel=2)
                    logger.warning(msg)
        return self


# ─────────────────────────────────────────────
# 쿼리 라우팅 설정 (v7.0 Part 2.5, 2026-04-21)
# ─────────────────────────────────────────────
# 배경: Phase 3 LoRA가 도구 호출을 강화하는 대신 베이스 Qwen의 일반 지식
# 표현을 좁히는 부작용 발생. 2026-04-21 A/B/C/D 실측에서 베이스 모델(LoRA OFF)이
# 니체·카프카 같은 일반 교양 지식을 더 정확히 답변함을 확인.
# 조치: 질의 타입에 따라 런타임에 모델(LoRA ON/OFF)과 temperature를 분기한다.
# 이 분기는 TIER_S 한정 최적화이며, TIER_M 이상에서는 enabled=false로 끈다
# (베이스 모델 + 24개 도구 + 긴 컨텍스트가 이미 기본값이 되기 때문).
# ─────────────────────────────────────────────
# 기본 tool_keywords — yaml 누락 시 폴백 (2026-04-22 리팩토링 4)
# ─────────────────────────────────────────────
# yaml(config/nexus_config.yaml#routing.tool_keywords)이 단일 소스이나,
# 테스트/경량 실행 환경에서 yaml 로드 없이 RoutingConfig()를 만드는 경우를 위해
# 동일 리스트를 모듈 상수로 유지한다. 추가/변경은 양쪽 모두에 반영해야 한다.
_DEFAULT_TOOL_KEYWORDS: list[str] = [
    # 한국어 — 파일/프로젝트/도구 명시 힌트
    "파일",
    "첨부",
    "업로드",
    "이 프로젝트",
    "코드베이스",
    "디렉토리",
    "폴더",
    "리포지토리",
    "리포지터리",
    "읽어줘",
    "읽어 줘",
    "편집해",
    "수정해",
    "모듈 구조",
    "디렉토리 구조",
    "프로젝트 구조",
    # 영어
    "file",
    "attached",
    "upload",
    "this project",
    "codebase",
    "repository",
    "directory",
    "folder",
    # 도구 이름 — 괄호 포함(함수 호출 스타일)
    "Read(",
    "Write(",
    "Edit(",
    "Bash(",
    "Glob(",
    "Grep(",
    "Agent(",
    "DocumentProcess",
    # 단독 대문자 도구명 + 공백 — "Read 도구", "Edit the file" 등
    "Read ",
    "Write ",
    "Edit ",
    "Bash ",
    "Glob ",
    "Grep ",
    "Agent ",
    " LS ",
    # 확장자 힌트 (공백 뒤 경로 패턴)
    ".py ",
    ".md ",
    ".yaml ",
    ".json ",
    # 프로젝트 내부 디렉토리 prefix — core/orchestrator, web/app.py 등
    "core/",
    "web/",
    "tests/",
    "training/",
    "deployment/",
    "cli/",
    "config/",
    "scripts/",
    "tools/",
]


# ─────────────────────────────────────────────
# 기본 chat_keywords — 인사·잡담 식별용 (Part 2.5.9, v0.14.6)
# ─────────────────────────────────────────────
# kowiki 100만 청크 적재 후 "안녕"·"좋은 아침" 같은 인사도 KNOWLEDGE로 분류되어
# 가수 "안녕"·"굿모닝 예루살렘" 청크가 RAG로 주입되며 부자연스러운 답변이 발생.
# CHAT_MODE는 분류기에서만 의미를 가지며 (RoutingProfile 자체는 KNOWLEDGE와 동일
# 모델을 쓰되) PromptAssembler가 KB RAG 단계를 스킵하도록 신호하는 용도.
# 길이 임계와 AND 조건으로만 트리거 — 긴 입력이면 인사어가 들어 있어도 CHAT 아님.
_DEFAULT_CHAT_KEYWORDS: list[str] = [
    # 한국어 인사·잡담
    "안녕",
    "안뇽",
    "좋은 아침",
    "좋은 저녁",
    "좋은 밤",
    "굿모닝",
    "굿나잇",
    "잘 자",
    "잘자",
    "반가워",
    "반갑습니다",
    "반갑네",
    "고마워",
    "고맙습니다",
    "감사",
    "땡큐",
    "잘 가",
    "잘가",
    "다음에 봐",
    "다음에 보자",
    "별거 없",
    "ㅋㅋ",
    "ㅎㅎ",
    # 영어 인사·잡담
    "hi",
    "hello",
    "hey",
    "yo",
    "sup",
    "good morning",
    "good evening",
    "good night",
    "thanks",
    "thank you",
    "thx",
    "ty",
    "bye",
    "goodbye",
    "see you",
    "cya",
]


class RoutingProfile(BaseModel):
    """
    개별 라우팅 프로필 — 질의 타입별 모델/파라미터 조합.

    질의 분류(KNOWLEDGE/TOOL/CHAT) 결과에 따라 이 프로필 하나가 선택되어
    vLLM 호출 페이로드(모델 이름 + 샘플링 파라미터)로 그대로 전달된다.
    즉 "어떤 모델을, 어떤 온도/샘플링으로 부를지"를 한 묶음으로 정의한다.
    """

    # Pydantic의 model_ 예약어 경고를 끈다(아래 model 필드 때문). 동작엔 무영향.
    model_config = {"protected_namespaces": ()}  # model_ 접두사 경고 방지

    # vLLM served-model-name. LoRA 어댑터 이름(예: nexus-phase3)이거나
    # 베이스 모델 이름(qwen3.5-27b)이다. 이 값으로 GPU 서버가 모델을 고른다.
    model: str
    # 샘플링 온도. 낮을수록 결정적(사실 위주), 높을수록 다양/창의적.
    temperature: float = 0.3
    # 이 프로필로 생성할 최대 출력 토큰 수.
    max_tokens: int = 4096
    # Qwen3.5의 사고(thinking) 모드 on/off. chat_template_kwargs로 전달되며
    # 기본 False(추가 사고 토큰을 안 써 속도/컨텍스트를 아낀다).
    enable_thinking: bool = False  # Qwen3.5 chat_template_kwargs 인자
    # 사람용 설명 — 동작에는 영향 없고 운영자/테스트가 프로필을 식별하는 메모.
    description: str = ""  # 운영자/테스트용 설명
    # ── 샘플링 파라미터 (degeneration/무한 반복 방지) ──────────────────
    # 왜 추가하나: 이 4개가 누락되어 동일 문장이 무한 반복되는 결함이 있었다.
    # 모든 기본값은 "비활성"(vLLM이 무시하는 값)으로 두어 하위 호환을 보장한다.
    #   top_p=1.0            → nucleus 샘플링 비활성(전체 분포 사용)
    #   repetition_penalty=1.0 → 반복 페널티 없음(1.0이 중립값)
    #   frequency_penalty=0.0  → 빈도 페널티 없음
    #   presence_penalty=0.0   → 등장 페널티 없음
    # 실제 운영값은 config/nexus_config.yaml의 routing 프로필에서 주입한다.
    top_p: float = 1.0
    repetition_penalty: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0


class RoutingConfig(BaseModel):
    """
    질의 타입별 라우팅 설정.

    분류 규칙 (v0.14.6, Part 2.5.9):
      1. user_input 길이가 long_input_threshold 이상 → TOOL_MODE
         (첨부 문서/로그 분석 시나리오로 간주)
      2. tool_keywords 중 하나라도 포함 → TOOL_MODE
      3. user_input 길이가 chat_max_length 이하 AND chat_keywords 중 하나라도
         포함 → CHAT_MODE (인사·잡담, KB RAG 미주입)
      4. 그 외 → KNOWLEDGE_MODE (일반 QA, KB RAG 주입)

    enabled=False이면 분류기를 돌리지 않고 항상 tool_mode 프로필을 사용한다
    (하드웨어 업그레이드 또는 문제 발생 시 비상 스위치).
    """

    # 라우팅 분류기 마스터 스위치. True면 질의 타입을 분류해 프로필을 고르고,
    # False면 분류 없이 항상 tool_mode 프로필로 동작한다(위 docstring의 비상 스위치).
    enabled: bool = True
    long_input_threshold: int = 500  # 이 글자수 이상이면 TOOL_MODE
    # 분류기 종류 — 기본은 "heuristic" (HeuristicClassifier).
    # 향후 "llm_classifier", "embedding_classifier" 추가 시 여기서 선택한다.
    # 새 분류기를 추가하려면 routing.QueryClassifier를 상속한 뒤
    # routing._CLASSIFIER_REGISTRY에 매핑을 등록한다.
    classifier_type: str = "heuristic"
    tool_keywords: list[str] = Field(
        # yaml(config/nexus_config.yaml#routing.tool_keywords)이 단일 소스.
        # 이 default_factory는 yaml 없이 RoutingConfig()를 만드는 테스트·경량 환경용 폴백.
        default_factory=lambda: list(_DEFAULT_TOOL_KEYWORDS)
    )
    # 단어 경계 매칭 (2026-04-22 리팩토링 6) — "file"이 "filename"에 오탐되는 문제 해소.
    # 예: "file"을 등록하면 " file ", " file." 등 단어 경계에서만 매칭된다.
    # 한국어는 단어 경계 개념이 달라 주로 영어/코드 용어에 쓰인다.
    tool_word_patterns: list[str] = Field(default_factory=list)
    # 정규식 매칭 — 복잡한 패턴이 필요한 경우 (예: 함수 호출 `Read\s*\(`).
    # 운영자 책임으로 유효한 regex를 제공 — 잘못된 regex는 로드 시 경고 후 무시.
    tool_regex_patterns: list[str] = Field(default_factory=list)
    # CHAT_MODE 식별 임계 (Part 2.5.9, v0.14.6).
    # 입력이 짧고(chat_max_length 이하) 인사 어휘가 포함되면 CHAT으로 본다.
    # 너무 길게 잡으면 일반 지식 질의가 잡담으로 오분류될 수 있어 보수적으로 30자.
    chat_max_length: int = 30
    chat_keywords: list[str] = Field(default_factory=lambda: list(_DEFAULT_CHAT_KEYWORDS))
    knowledge_mode: RoutingProfile = Field(
        default_factory=lambda: RoutingProfile(
            model="qwen3.5-27b",
            temperature=0.2,
            max_tokens=2048,
            enable_thinking=False,
            description="일반 지식 QA — 베이스 Qwen + 낮은 temperature",
            # 지식 QA는 사실 위주라 반복 페널티를 가장 강하게 건다.
            # top_p=0.95로 꼬리 토큰을 약간 잘라 안정성 확보,
            # repetition_penalty=1.15로 동일 문장 반복(degeneration) 억제.
            top_p=0.95,
            repetition_penalty=1.15,
            frequency_penalty=0.3,
            presence_penalty=0.0,
        )
    )
    tool_mode: RoutingProfile = Field(
        default_factory=lambda: RoutingProfile(
            model="nexus-phase3",
            temperature=0.3,
            max_tokens=4096,
            enable_thinking=False,
            description="도구 호출 — Phase 3 LoRA + 중간 temperature",
            # 도구 호출 모드는 repetition_penalty=1.0(비활성)이 핵심이다.
            # 왜: tool_call JSON/XML은 같은 키("name","arguments" 등)와 괄호를
            # 반복할 수밖에 없는데, 반복 페널티를 걸면 이 필수 토큰이 왜곡되어
            # 파싱 실패를 유발한다. top_p만 0.95로 살짝 좁혀 안정성만 확보.
            top_p=0.95,
            repetition_penalty=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
        )
    )
    # CHAT_MODE 프로필 — 모델·온도는 KNOWLEDGE와 같지만 PromptAssembler가
    # query_class를 보고 KB RAG를 주입하지 않는다. max_tokens는 인사·잡담이
    # 길게 늘어지지 않도록 더 작게 설정.
    chat_mode: RoutingProfile = Field(
        default_factory=lambda: RoutingProfile(
            model="qwen3.5-27b",
            temperature=0.5,
            max_tokens=512,
            enable_thinking=False,
            description="인사·잡담 — 베이스 Qwen + RAG 미주입 + 짧은 응답",
            # 잡담은 다양성이 중요하므로 top_p=0.9로 살짝 더 좁히되,
            # 반복 페널티는 중간 강도(1.1)로 같은 인사말 반복을 막는다.
            top_p=0.9,
            repetition_penalty=1.1,
            frequency_penalty=0.2,
            presence_penalty=0.0,
        )
    )


# ─────────────────────────────────────────────
# 멀티테넌시 설정 (v7.0 Part 5 Ch 15, 2026-04-21)
# ─────────────────────────────────────────────
# 배경: 학교·기업별 서비스를 위해 LoRA 어댑터와 RAG 지식 베이스를 분리한다.
# 같은 GPU/DB 인프라에서 tenant별 격리를 제공한다.
#
# 작동 구조:
#   - 웹/API 요청의 X-Tenant-ID 헤더로 tenant 식별
#   - TenantConfig.model_override → QueryEngine 라우팅에서 사용하는 LoRA 지정
#   - TenantConfig.allowed_knowledge_sources → KnowledgeRetriever가 source 필터
#   - default 테넌트는 공통 지식(kowiki 등)만 허용
class TenantConfig(BaseModel):
    """단일 테넌트(학교·기업·기본) 설정."""

    # model_override 등 model_ 접두사 필드 때문에 Pydantic 예약어 경고를 끈다.
    model_config = {"protected_namespaces": ()}  # model_ 접두사 경고 방지

    id: str  # 테넌트 고유 식별자. X-Tenant-ID 헤더/조회 키로 쓰인다(필수).
    name: str = ""  # 사람용 표시 이름(예: "한양대학교"). 없으면 빈 문자열.
    description: str = ""  # 테넌트 설명 메모(운영용).

    # LoRA 라우팅 — tenant 전용 어댑터. None이면 라우팅/기본값 사용
    model_override: str | None = None

    # 이 tenant가 볼 수 있는 지식 소스 목록 (tb_knowledge.source)
    # 비어 있으면 전체 허용 (default 테넌트에만 권장)
    allowed_knowledge_sources: list[str] = Field(default_factory=list)

    # API 키 → tenant 매핑 (선택, 간이 인증)
    api_keys: list[str] = Field(default_factory=list)

    # 추가 메타데이터 (부서·계약정보 등)
    metadata: dict[str, Any] = Field(default_factory=dict)

    # M7: LoRA 학습 어댑터 이름 접두사 (선택)
    # 값이 있으면 adapter_name(phase)이 이 접두사를 우선 사용한다.
    # 예: "hy-custom" → hy-custom-phase3 (브랜딩·계약상 이름 규약용)
    adapter_name_prefix: str | None = None

    def adapter_name(self, phase: int) -> str:
        """이 테넌트의 phaseN LoRA 어댑터 이름을 반환한다 (M7).

        컴포지션은 `training.adapter_naming.compose_adapter_name`에 위임한다.
        default 테넌트는 `nexus-phaseN` (기존 호환), 그 외는 `nexus-{id}-phaseN`.
        `adapter_name_prefix`가 설정되면 해당 값이 우선.
        """
        # 순환 import 방지 — 함수 호출 시점에 지연 임포트
        from training.adapter_naming import compose_adapter_name

        return compose_adapter_name(
            self.id,
            phase,
            custom_prefix=self.adapter_name_prefix,
        )


class TenantRegistry(BaseModel):
    """
    테넌트 레지스트리 — 등록된 모든 테넌트의 목록과 조회 헬퍼.

    보통 config/tenants.yaml에서 통째로 로드되어 NexusConfig.tenants에 들어간다.
    요청이 들어오면 X-Tenant-ID(또는 API 키)로 이 레지스트리에서 해당
    TenantConfig를 찾아 LoRA/지식 소스 격리를 적용한다.
    """

    # 식별 실패 시 폴백할 기본 테넌트 id. resolve()가 이 값을 최후 보루로 쓴다.
    default_tenant: str = "default"
    # 등록된 테넌트 목록. 기본으로 공개 지식만 보는 "default" 한 개를 둔다.
    tenants: list[TenantConfig] = Field(
        default_factory=lambda: [
            TenantConfig(
                id="default",
                name="Default",
                description="공통 테넌트 — 공개 지식(kowiki 등)만 접근",
                allowed_knowledge_sources=["kowiki", "sample"],
            )
        ]
    )

    def get(self, tenant_id: str) -> TenantConfig | None:
        """id로 테넌트 조회. 없으면 None."""
        for t in self.tenants:
            if t.id == tenant_id:
                return t
        return None

    def resolve(self, tenant_id: str | None) -> TenantConfig:
        """tenant_id가 None·미등록이면 default로 폴백. 반드시 유효한 TenantConfig 반환."""
        if tenant_id:
            found = self.get(tenant_id)
            if found is not None:
                return found
        default = self.get(self.default_tenant)
        if default is not None:
            return default
        # 레지스트리가 비어 있을 때의 최후 폴백 — 공백 tenant
        return TenantConfig(id="default", name="Default")

    def resolve_by_api_key(self, api_key: str) -> TenantConfig | None:
        """API 키로 tenant 조회 (간이 인증 경로)."""
        if not api_key:
            return None
        for t in self.tenants:
            if api_key in t.api_keys:
                return t
        return None


# ─────────────────────────────────────────────
# 웹 API 인증 설정 (Security Critical #4, 2026-07-02)
# ─────────────────────────────────────────────
class WebAuthConfig(BaseModel):
    """
    웹(FastAPI) API 키 인증 설정.

    web/middleware.py의 ApiKeyAuthMiddleware가 이 설정을 읽어 요청을 통과/차단한다.
    인증이 켜지면 `Authorization: Bearer <key>` 헤더의 API 키가
    TenantRegistry.resolve_by_api_key()로 유효 테넌트를 찾을 때만 요청을 허용한다.

    왜 enabled 기본값이 False인가 (무회귀 원칙):
      기존 5090/로컬 개발 환경은 전 엔드포인트 무인증으로 동작해 왔다. 기본값을
      True로 바꾸면 기존 개발/테스트 흐름이 즉시 401로 깨진다(회귀). 따라서 기본은
      현행 동작(무인증)을 유지하고, 배포(에어갭 운영) 환경에서만 명시적으로 켠다.

    배포 시 반드시 True로 설정할 것:
      운영 배포에서는 NEXUS_WEB_AUTH__ENABLED=true (또는 config.yaml)로 인증을
      활성화해야 한다. 비활성 상태로 신뢰되지 않은 네트워크에 노출하면 안 된다.
    """

    # 인증 게이트 on/off. 기본 False = 현행 무인증(무회귀). 배포 시 True 필수.
    enabled: bool = False

    # 인증 없이 접근을 허용할 경로 목록.
    # 매칭 규칙: "/"는 정확히 루트("/")만 허용(prefix로 쓰면 모든 경로가 열림),
    #           그 외 항목은 prefix 매칭(예: "/static" → "/static/app.js"도 허용).
    exempt_paths: list[str] = Field(
        default_factory=lambda: [
            "/health",  # 헬스체크 — 모니터링 도구가 무인증으로 폴링
            "/",  # 루트 랜딩 페이지(정확 매칭)
            "/docs",  # Swagger UI
            "/openapi.json",  # OpenAPI 스키마
            "/static",  # 정적 자산(HTML/CSS/JS)
        ]
    )


# ─────────────────────────────────────────────
# OCR(스캔 PDF/이미지) 설정 — v7.3 로드맵 단계 8
# ─────────────────────────────────────────────
class OcrConfig(BaseModel):
    """
    Tesseract OCR 파서(core/ingest/parsers/ocr_tesseract.py) 설정.

    왜 설정으로 빼는가 (anti-pattern #4 — 하드코딩 금지):
      개발 환경과 배포(에어갭) 환경에서 tesseract 실행 파일 경로와 한국어
      학습 데이터(tessdata) 위치가 다르다. 코드에 박지 않고 여기서 받아
      yaml/환경변수(NEXUS_OCR__*)로 배포 시 덮어쓸 수 있게 한다.

    필드 설명:
      - tesseract_cmd: tesseract 실행 파일 경로. 개발 기본값은 Windows 설치
        경로. 배포 시 리눅스(예: "/usr/bin/tesseract") 등으로 오버라이드한다.
      - tessdata_dir: 언어 데이터(*.traineddata)가 든 폴더. 개발 환경에서는
        Program Files 쓰기 권한 문제로 사용자 LOCALAPPDATA 하위에 둔다.
        빈 문자열이면 tesseract 기본 위치(TESSDATA_PREFIX 등)를 따른다.
      - lang: OCR 언어 코드. "kor+eng" 는 한국어+영어 혼용 문서를 함께 인식한다
        (tessdata_dir 에 kor/eng traineddata 가 있어야 한다).
      - dpi: 스캔 PDF 페이지를 이미지로 렌더할 해상도. 200~300 이 OCR 품질과
        속도의 절충점이다(너무 낮으면 인식률↓, 너무 높으면 느리고 메모리↑).

    에어갭: 실행 파일/언어 데이터는 사전 배치 전제(런타임 설치 코드 없음).
    """

    # 개발 기본값 — 배포 시 yaml/환경변수(NEXUS_OCR__TESSERACT_CMD 등)로 오버라이드.
    tesseract_cmd: str = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    # 개발 기본값 — 사용자 LOCALAPPDATA 하위의 nexus_tessdata 폴더.
    # (Program Files 에 쓰기 권한이 없어 학습 데이터를 사용자 폴더로 분리했다.)
    tessdata_dir: str = os.path.join(os.environ.get("LOCALAPPDATA", ""), "nexus_tessdata")
    lang: str = "kor+eng"  # 한국어+영어 혼용 인식
    dpi: int = 250  # 스캔 PDF 렌더 해상도(품질/속도 절충)

    # ── PaddleOCR 고품질 OCR 파서 설정 (v7.3 단계 8 — 한국어 표/레이아웃 OCR, GPU) ──
    #
    # 왜 같은 OcrConfig 안에 두는가 (anti-pattern #4 — 하드코딩 금지):
    #   Tesseract(경량 CPU)와 PaddleOCR(고품질 GPU)는 "스캔 PDF/이미지 OCR" 라는
    #   같은 역할(v7.3 단계 8)을 품질 티어만 달리해 수행한다. 설정 묶음을 하나로
    #   두면 배포 시 yaml/환경변수(NEXUS_OCR__PADDLE_*)로 함께 관리하기 쉽다.
    #   tesseract_*/paddle_* 는 접두사로 구분되어 서로 충돌하지 않고 공존한다.
    #
    # 필드 설명:
    #   - paddle_lang: PaddleOCR 인식 언어 코드. PaddleOCR 의 언어 코드 체계는
    #     tesseract 와 달라 한국어는 "korean" 이다(tesseract 의 "kor" 이 아님).
    #     그래서 lang(tesseract용)과 별도 필드로 둔다.
    #   - paddle_use_gpu: GPU(CUDA) 사용 여부. 기본 True(고품질=GPU 권장).
    #     GPU 가 없는 호스트에서는 레지스트리 단계에서 이 파서를 아예 등록하지
    #     않으므로(아래 docingest_server 등록 정책 참조) 보통 이 값이 쓰이지
    #     않지만, 명시 호출/CPU 강제 실행 시 False 로 내려 CPU 로 동작시킨다.
    #   - paddle_enable_mkldnn: oneDNN(MKL-DNN) CPU 가속 사용 여부. 기본 False.
    #     Windows 개발 환경의 paddle 3.3.1 CPU 빌드에서 oneDNN 경로가
    #     "ConvertPirAttribute2RuntimeAttribute not support" 런타임 오류를 내는
    #     것을 실측으로 확인했다(이 옵션을 False 로 두면 정상 동작). Linux 배포
    #     GPU 환경에서는 영향이 없으므로 안전한 기본값으로 False 를 쓴다.
    paddle_lang: str = "korean"  # PaddleOCR 한국어 코드(tesseract "kor" 과 다름)
    paddle_use_gpu: bool = True  # 고품질=GPU 권장(미가용 시 레지스트리에서 미등록)
    paddle_enable_mkldnn: bool = False  # Windows CPU oneDNN PIR 버그 회피(실측)


# ─────────────────────────────────────────────
# .hwp(구포맷) 파서 설정 — v7.3 로드맵 단계 9
# ─────────────────────────────────────────────
class HwpConfig(BaseModel):
    """
    구포맷 .hwp 파서(core/ingest/parsers/hwp_libreoffice.py) 설정.

    왜 LibreOffice 경유인가:
      구포맷 .hwp(한글 v5, OLE 복합문서)는 개방형 OWPML(HWPX)과 달리 폐쇄
      바이너리 포맷이다. 이를 직접 파싱하는 청정 라이선스 파이썬 라이브러리가
      마땅치 않고(pyhwp 는 AGPL 이라 라이선스 정책상 배제), 사용자 문서에서
      구포맷 .hwp 비중이 높다. 따라서 LibreOffice 의 한글 import 필터로 .hwp 를
      .docx 로 변환한 뒤, 기존 python-docx 경로로 구조를 보존해 파싱한다.

    왜 설정으로 빼는가 (anti-pattern #4 — 하드코딩 금지):
      soffice 실행 파일 경로가 개발(Windows)과 배포(에어갭 Linux)에서 다르다.
      코드에 박지 않고 여기서 받아 yaml/환경변수(NEXUS_HWP__*)로 오버라이드한다.

    필드 설명:
      - soffice_cmd: LibreOffice headless 실행 파일 경로. 개발 기본값은 Windows
        설치 경로. 배포(Linux)에서는 "/usr/bin/soffice"(또는 libreoffice 런처)
        등으로 yaml/환경변수로 오버라이드한다. 빈 문자열이면 파서가 환경변수
        (NEXUS_SOFFICE_CMD) → 개발 기본값 순으로 폴백한다.
      - convert_timeout_sec: soffice 변환 1건의 최대 대기(초). 한글 대용량 문서가
        변환에 오래 걸릴 수 있어 넉넉히 둔다. 초과하면 TimeoutExpired → fail-soft.

    에어갭: soffice 실행 파일은 사전 설치 전제(런타임 설치 코드 없음 — anti #10).
    """

    # 개발 기본값 — 배포 시 yaml/환경변수(NEXUS_HWP__SOFFICE_CMD)로 오버라이드.
    soffice_cmd: str = r"C:\Program Files\LibreOffice\program\soffice.exe"
    # 변환 타임아웃(초) — 대용량 .hwp 도 수용하되 무한 대기는 막는다.
    convert_timeout_sec: float = 120.0


class SecurityConfig(BaseModel):
    """
    보안 및 샌드박스 설정.

    core/security(샌드박스/명령어 필터)와 권한 파이프라인 Layer 2가 참조한다.
    Bash 명령어 화이트/블랙리스트, 파일 크기·확장자 제한 등 "무엇을 허용/차단할지"
    의 경계값을 모아둔다. 기본값은 fail-closed 철학에 맞춰 보수적으로 잡혀 있다.
    """

    # 샌드박스 활성 여부. 기본 True — Bash 도구 실행을 격리 환경에서 돌린다.
    sandbox_enabled: bool = True
    # 샌드박스 안에서 명령 1건의 최대 실행 시간(초). 무한 루프/멈춤을 30초에서 끊는다.
    sandbox_timeout_seconds: float = 30.0
    # 명시적으로 허용할 Bash 패턴 화이트리스트. 비어 있으면 화이트리스트를
    # 적용하지 않는다(차단은 아래 deny 패턴이 담당).
    bash_allow_patterns: list[str] = Field(default_factory=list)
    # 무조건 차단할 위험 Bash 패턴(정규식). 시스템 파괴/포크 폭탄/디스크 덮어쓰기
    # 같은 치명적 명령을 막는 최소 안전망이다.
    bash_deny_patterns: list[str] = Field(
        default_factory=lambda: [
            r"rm\s+-rf\s+/",  # 루트부터 강제 삭제
            r":(){ :\|:& };:",  # 포크 폭탄(fork bomb) — 프로세스 무한 증식
            r"dd\s+if=/dev/zero",  # 디스크/파일을 0으로 덮어쓰기
            r"mkfs\.",  # 파일시스템 포맷
            r">\s*/dev/sd",  # 블록 디바이스에 직접 리다이렉트(디스크 손상)
        ]
    )
    # 도구가 읽거나 쓸 수 있는 파일 1건의 최대 크기(바이트). 10MB를 넘는
    # 거대 파일이 메모리를 잡아먹거나 컨텍스트를 폭주시키는 것을 막는다.
    max_file_size_bytes: int = 10 * 1024 * 1024  # 10MB
    # 파일 도구가 다룰 수 있는 확장자 화이트리스트. 텍스트/코드/설정 계열만
    # 허용해 바이너리나 위험 포맷 접근을 기본적으로 막는다.
    allowed_file_extensions: list[str] = Field(
        default_factory=lambda: [
            ".py",
            ".js",
            ".ts",
            ".json",
            ".yaml",
            ".yml",
            ".toml",
            ".md",
            ".txt",
            ".csv",
            ".html",
            ".css",
            ".sql",
            ".sh",
            ".bash",
            ".dockerfile",
            ".env.example",
        ]
    )


# ─────────────────────────────────────────────
# 지식 베이스 RAG 게이팅 설정 (2026-06-18)
# ─────────────────────────────────────────────
# 배경(실측으로 확정): e5-large 코사인 유사도는 "무관한 문서끼리"도 0.78~0.83
# 구간에 몰린다. 즉 절대 유사도만으로는 관련/무관을 가르기 어렵다.
#   - 메타질문 "rag에 이런 정보가 있었어?" → top1 sim=0.824 (무관: 비트토렌트류)
#   - "요한 제바스티안 바흐"                → top1 sim=0.857 (관련)
# 기존 KnowledgeRetriever.min_similarity=0.5(코드 기본값, config 미노출)는 너무
# 낮아 모든 검색 결과가 무조건 주입됐고, 모델이 무관 청크를 근거로 그럴듯한
# 오답을 만들어내는 할루시네이션의 원인이 됐다.
#
# 해법(2단 게이팅):
#   1) abs_threshold(절대 임계): 최상위 결과조차 이 값보다 낮으면 "관련 자료
#      없음"으로 보고 전부 드롭한다. 무관 분포 상한(~0.83)보다 살짝 위인 0.84로
#      잡아, 관련 질의(바흐 0.857)는 통과시키고 무관 질의(0.824)는 차단한다.
#   2) relevance_margin(상대 마진): 최상위 유사도(top_sim)에서 이 값만큼만
#      떨어진 결과까지만 남긴다. top과 크게 벌어진 "끼어든 노이즈 청크"를 잘라
#      가장 관련 높은 소수의 청크만 모델에 보여준다.
class KnowledgeRagConfig(BaseModel):
    """
    지식 베이스(tb_knowledge) RAG 검색·게이팅 설정.

    모든 임계값의 단일 소스는 config/nexus_config.yaml#knowledge_rag 이며,
    이 클래스의 기본값은 yaml 누락 시(테스트/경량 실행) 폴백으로만 쓰인다.
    """

    # 검색해 올 상위 청크 개수. 너무 많으면 무관 청크가 섞이고 컨텍스트를
    # 잡아먹으므로 5개로 보수적으로 둔다(RTX 5090 8K 컨텍스트 절약).
    top_k: int = 5
    # DB 벡터 검색 단계의 1차 컷오프(search_by_vector에 그대로 전달).
    # 명백히 무관한 하위 청크를 DB 단계에서 미리 떨군다. 기존 코드 기본값
    # 0.5는 e5 분포상 사실상 무필터였으므로 0.75로 현실화한다(아래 게이팅과
    # 별개의 1차 필터 — 게이팅은 그 위에서 다시 한 번 정밀하게 거른다).
    min_similarity: float = 0.75
    # 절대 임계 게이팅: 최상위 결과의 유사도가 이 값보다 낮으면 전부 드롭한다.
    # 실측 근거 — 무관 분포가 0.78~0.83에 몰리므로 그 상한 바로 위인 0.84로
    # 둔다. 무관 메타질문(top1=0.824)은 차단, 관련 질의(바흐 0.857)는 통과.
    abs_threshold: float = 0.84
    # 상대 마진 게이팅: 최상위 유사도(top_sim)에서 이 값 이내로 떨어진 결과만
    # 남긴다. e5는 관련 청크들끼리도 유사도가 촘촘해서 0.03(3%p)이면 진짜 핵심
    # 청크 1~몇 개만 통과하고, top과 동떨어진 노이즈 청크는 잘린다.
    relevance_margin: float = 0.03


# ─────────────────────────────────────────────
# 컨텍스트 예산 설정 — 하드코딩 외부화 (2026-07-03, B200 티어 준비 Phase 1)
# ─────────────────────────────────────────────
class ContextBudgetConfig(BaseModel):
    """
    컨텍스트 예산(context budget) 설정 — 여기저기 하드코딩돼 있던 토큰/청크
    상한값을 한곳에 모은 것.

    왜 모으는가:
      그동안 RAG 주입 토큰 수, 도구 결과 예산, 문서 청크 크기, 출력 토큰
      에스컬레이션 단계 등이 각 소비 파일(prompt_assembler / context_manager /
      document_tool / query_loop)에 상수로 박혀 있었다. RTX 5090(TIER_S,
      8K 컨텍스트) 기준으로 잡힌 값이라, 컨텍스트가 훨씬 큰 B200/H200 티어에서는
      그대로 두면 큰 컨텍스트를 못 살린다. 이 모델로 모아두면 티어별 config
      (예: B200용 nexus_config)에서 한 번에 상향 오버라이드할 수 있다.

    ★무회귀 원칙★:
      아래 모든 기본값은 "현재(2026-07-03) 각 파일에 하드코딩돼 있던 실측 값"과
      정확히 동일하다. 즉 이 섹션을 yaml에 쓰지 않아도(=기본값으로만 동작해도)
      5090(TIER_S)·기존 테스트 동작이 1비트도 바뀌지 않는다. B200 상향은 별도
      config에서 이 값들을 덮어쓰는 후속 작업으로 처리한다.
    """

    # ── RAG/프롬프트 조립 예산 (출처: core/orchestrator/prompt_assembler.py) ──
    # ① 이전 턴 요약(TurnState) 주입 상한. _attach_turn_state()에서
    #    turn_state_store.get_context(max_tokens=...)로 쓰이던 값(현행 1000).
    turn_state_tokens: int = 1000
    # ② 프로젝트 RAG(관련 파일 청크) 주입 상한. _attach_project_rag()에서
    #    rag_retriever.get_context(max_tokens=...)로 쓰이던 값(현행 1500).
    project_rag_tokens: int = 1500
    # ③ 지식베이스(KB) RAG 주입 상한. _attach_knowledge_base()에서
    #    knowledge_retriever.get_context(max_tokens=...)로 쓰이던 값(현행 1000).
    knowledge_rag_tokens: int = 1000

    # ── 컨텍스트 압축 예산 (출처: core/orchestrator/context_manager.py 생성자) ──
    # ④ 개별 도구 결과(tool_result)의 최대 토큰 수. 초과분은 head+tail로 잘린다.
    #    ContextManager(tool_result_budget=2048) 생성자 기본값과 동일(현행 2048).
    tool_result_budget: int = 2048
    # ⑤ 압축 시에도 항상 원본 보존할 최근 턴 수(현행 3).
    preserve_recent_turns: int = 3
    # ⑥ 압축 시에도 항상 원본 보존할 최근 도구 결과 수(현행 2).
    preserve_recent_tool_results: int = 2

    # ── 문서 도구 예산 (출처: core/tools/implementations/document_tool.py) ──
    # ⑦ DocumentProcess 도구가 문서를 나눌 청크 크기(글자 수). 8192 ctx 기준으로
    #    tool_result가 컨텍스트를 넘지 않도록 잡은 값(현행 CHUNK_SIZE=2500).
    document_chunk_size: int = 2500

    # ── 출력 토큰 에스컬레이션 (출처: core/orchestrator/query_loop.py) ──
    # ⑧ 응답이 max_tokens로 잘렸을 때 출력 한도를 점진 상향하는 단계.
    #    query_loop의 OUTPUT_TOKEN_ESCALATION 상수와 동일(현행 [4096,8192,16384]).
    #    list 기본값이므로 mutable 공유를 피하려 default_factory를 쓴다.
    output_token_escalation: list[int] = Field(
        default_factory=lambda: [4096, 8192, 16384]
    )


# ─────────────────────────────────────────────
# 권한 강제(permission enforcement) 설정 — 감사 Critical #1~3 무회귀 토대 (2026-07-03)
# ─────────────────────────────────────────────
class PermissionEnforcementConfig(BaseModel):
    """
    5계층 권한 파이프라인(PermissionPipeline)을 도구 실행 경로에 배선할지 결정하는 설정.

    배경(왜 필요한가):
      현재 executor(core/tools/executor.py)의 유일한 권한 게이트는 도구 자체
      check_permissions()의 DENY만 차단하고, 파이프라인의 Layer 1/3/4/5는 전혀
      실행하지 않는다(코드 주석에 "간소화 — Phase 4에서 전체 구현"이라 명시됨).
      이 설정은 그 파이프라인을 단계적으로 배선하기 위한 마스터 스위치다.

    ★무회귀 원칙(이 단계의 최우선 규칙)★:
      - enabled 기본값은 False = 파이프라인을 호출조차 하지 않는다. 즉 executor
        동작이 현행과 100% 동일하다(도구 check_permissions의 DENY만 차단).
      - enabled=True 여도 mode="shadow"이면 파이프라인 결정을 "기록만" 하고
        실행 경로는 전혀 바꾸지 않는다(차단 안 함 — 관측 전용). 새 차단이 절대
        생기지 않는다. 실제 차단(enforce)은 다음 단계에서 mode="enforce"로 전환한다.

    필드 설명:
      - enabled: 파이프라인 배선 on/off. 기본 False(무회귀). 배포/검증 시 명시 활성.
      - mode: "shadow"(판정을 감사 로그에 기록만, 차단 안 함) 또는
              "enforce"(파이프라인 결정으로 실제 차단 — 다음 단계). 기본 "shadow".
    """

    # 마스터 스위치. 기본 False = 현행 동작 100% 유지(파이프라인 미호출, 무회귀).
    enabled: bool = False
    # 강제 방식. "shadow"=기록만(차단 안 함), "enforce"=실제 차단(후속 단계).
    # 기본 "shadow" — enabled를 켜더라도 우선은 관측만 한다(fail-safe).
    mode: str = "shadow"


class AuditConfig(BaseModel):
    """
    감사 로그(AuditLogger, core/security/audit.py) 설정.

    권한 파이프라인이 내린 모든 결정을 JSONL로 남겨 무회귀·보안 검증의 근거로
    삼는다. AuditLogger는 이미 구현돼 있으나 런타임에 배선되지 않은 상태였고,
    이 설정으로 경로/활성 여부를 코드에 하드코딩하지 않고 외부화한다(anti #4).

    필드 설명:
      - enabled: 감사 로그 기록 on/off. 기본 True(관측은 항상 켜두는 게 안전).
      - path: 로그 파일 경로. 상대경로면 작업 디렉토리 기준. 기본 "logs/audit.log".
    """

    # 감사 로그 활성 여부. 기본 True — shadow 관측 결과를 남기려면 켜져 있어야 한다.
    enabled: bool = True
    # 로그 파일 경로(하드코딩 금지 — yaml/환경변수로 오버라이드 가능).
    path: str = "logs/audit.log"


# ─────────────────────────────────────────────
# 메인 설정 클래스 (Pydantic BaseSettings)
# ─────────────────────────────────────────────
class NexusConfig(BaseSettings):
    """
    Project Nexus 전체 설정.

    우선순위: CLI 인자 > 환경변수 > config.yaml > 기본값.
    환경변수는 NEXUS_ 접두사를 사용한다.
    중첩 구분자는 __ (더블 언더스코어)이다.
    예: NEXUS_REDIS__HOST=192.168.10.39
    """

    model_config = {"env_prefix": "NEXUS_", "env_nested_delimiter": "__"}

    # GPU 서버
    gpu_server: GPUServerConfig = Field(default_factory=GPUServerConfig)

    @property
    def gpu_server_url(self) -> str:
        """GPU 서버 URL — gpu_server.url에서 가져온다."""
        return self.gpu_server.url

    # 데이터 저장소
    redis: RedisConfig = Field(default_factory=RedisConfig)
    postgresql: PostgreSQLConfig = Field(default_factory=PostgreSQLConfig)

    # 편의 접근자 (flat config 호환)
    # 아래 property들은 config.redis.host처럼 중첩 접근하지 않고 config.redis_host로
    # 평평하게 읽을 수 있게 해주는 단순 위임자다. 과거 flat 설정을 쓰던 코드와의
    # 호환을 위해 남겨두며, 값을 가공하지 않고 그대로 돌려준다.
    @property
    def redis_host(self) -> str:
        return self.redis.host

    @property
    def redis_port(self) -> int:
        return self.redis.port

    @property
    def redis_db(self) -> int:
        return self.redis.db

    @property
    def pg_host(self) -> str:
        return self.postgresql.host

    @property
    def pg_port(self) -> int:
        return self.postgresql.port

    @property
    def pg_database(self) -> str:
        return self.postgresql.database

    @property
    def pg_user(self) -> str:
        return self.postgresql.user

    @property
    def pg_password(self) -> str:
        return self.postgresql.password

    # 모델
    model: ModelConfig = Field(default_factory=ModelConfig)

    # 세션
    session: SessionConfig = Field(default_factory=SessionConfig)

    @property
    def sessions_dir(self) -> str:
        return self.session.sessions_dir

    # 보안
    security: SecurityConfig = Field(default_factory=SecurityConfig)

    # v7.0 Scout (CPU 4B 모델)
    scout: ScoutConfig = Field(default_factory=ScoutConfig)

    # v7.3 OCR (스캔 PDF/이미지 — Tesseract, CPU 경량)
    ocr: OcrConfig = Field(default_factory=OcrConfig)

    # v7.3 단계 9 — 구포맷 .hwp (LibreOffice headless 변환 경유)
    hwp: HwpConfig = Field(default_factory=HwpConfig)

    # v7.0 Part 2.5 쿼리 라우팅 — 지식/도구 질의 분기 (2026-04-21 추가)
    routing: RoutingConfig = Field(default_factory=RoutingConfig)

    # 지식 베이스 RAG 게이팅 — 무관 청크 주입 차단 (2026-06-18 추가)
    # 임계값을 코드가 아닌 yaml에서 받아 운영 중 튜닝 가능하게 한다.
    knowledge_rag: KnowledgeRagConfig = Field(default_factory=KnowledgeRagConfig)

    # 멀티테넌시 (Part 5 Ch 15, 2026-04-21)
    tenants: TenantRegistry = Field(default_factory=TenantRegistry)

    # 웹 API 키 인증 (Security Critical #4, 2026-07-02)
    # 기본 비활성(무회귀). 배포 시 NEXUS_WEB_AUTH__ENABLED=true로 활성화한다.
    web_auth: WebAuthConfig = Field(default_factory=WebAuthConfig)

    # v7.2 MCP 통합 — LAN 내부 MCP 서버 연결 (기본 비활성, 에어갭 fail-closed)
    mcp: McpConfig = Field(default_factory=McpConfig)

    # 컨텍스트 예산 — 하드코딩 외부화(2026-07-03, B200 Phase 1)
    # 기본값=현행 5090(TIER_S) 하드코딩 값과 동일 → 미지정 시 무회귀.
    # B200/H200 티어는 별도 config에서 이 섹션을 상향 오버라이드한다.
    context_budgets: ContextBudgetConfig = Field(default_factory=ContextBudgetConfig)

    # 권한 강제 파이프라인 배선 (감사 Critical #1~3, 2026-07-03)
    # 기본 enabled=False = 현행 executor 동작 100% 유지(무회귀). enabled=True 여도
    # 기본 mode="shadow"라 판정을 기록만 하고 차단하지 않는다(관측 전용).
    permission_enforcement: PermissionEnforcementConfig = Field(
        default_factory=PermissionEnforcementConfig
    )

    # 감사 로그(AuditLogger) 배선 — 권한 결정을 JSONL로 기록.
    audit: AuditConfig = Field(default_factory=AuditConfig)

    # 하드웨어 티어 — Scout 활성화/컨텍스트 길이 등 동작을 좌우한다.
    # "auto"면 GPU VRAM을 감지해 TIER_S/M/L 등을 자동 결정한다. 특정 티어를
    # 강제하고 싶으면 그 값을 직접 넣는다(예: 테스트에서 티어 고정).
    hardware_tier: str = "auto"

    # 운영
    log_level: str = "INFO"  # 로그 레벨(DEBUG/INFO/WARNING/...). 기본 INFO.
    log_file: str | None = None  # 로그를 파일로도 남길 경로. None이면 콘솔만.
    # 설정/소스 파일 변경 감시 후 자동 리로드 여부. 기본 False(운영 안정성 우선).
    watch_files: bool = False
    debug: bool = False  # 디버그 모드 스위치. 켜면 추가 진단 동작/로그가 활성화된다.

    # 에어갭 모드 — 기본 True. 켜져 있으면 아래 validator가 GPU URL이 LAN인지
    # 한 번 더 확인한다(외부 주소면 경고). 폐쇄망 운영의 기본 전제.
    air_gap_mode: bool = True

    @model_validator(mode="after")
    def validate_air_gap(self) -> NexusConfig:
        """
        에어갭 모드가 켜져 있으면 GPU 서버 URL이 로컬인지 확인한다.

        GPUServerConfig의 url validator와 비슷하지만, 이건 전체 설정이 조립된
        뒤(after) 한 번 더 도는 최종 점검이다. air_gap_mode가 True인데 GPU URL이
        외부 주소면 경고를 띄운다. 여기서도 예외가 아니라 warning만 내는 이유는
        설정 로드 실패로 전체가 멈추는 것을 피하기 위함이다(알리되 진행).
        """
        if self.air_gap_mode:
            parsed = urlparse(self.gpu_server_url)
            hostname = parsed.hostname or ""
            # 로컬호스트 + 사설망(RFC1918) 대역만 "내부"로 인정한다.
            local_prefixes = ("localhost", "127.0.0.1", "10.", "172.", "192.168.")
            if not any(hostname.startswith(p) for p in local_prefixes):
                import warnings

                warnings.warn(
                    f"에어갭 모드가 활성화되어 있지만 GPU 서버 URL '{self.gpu_server_url}'이(가) "
                    f"외부 주소로 보입니다.",
                    UserWarning,
                    stacklevel=2,
                )
        return self


# ─────────────────────────────────────────────
# 설정 로딩 함수
# ─────────────────────────────────────────────
def load_and_validate_config(
    config_path: str | None = None,
) -> NexusConfig:
    """
    설정을 로드하고 검증한다.

    로딩 순서:
      1. 기본값
      2. 설정 파일 (.nexus/config.yaml 또는 지정 경로)
      3. 환경변수 (NEXUS_ 접두사)

    왜 이 순서인가: Claude Code도 settings.json → env vars 순서로 로드하며,
    환경변수가 파일 설정을 덮어쓸 수 있어야 배포 환경에서 유연하다.
    """
    # 설정 파일 경로 탐색
    # config_path를 명시하지 않으면 정해진 후보들을 위에서부터 훑어 처음 존재하는
    # 파일을 채택한다(프로젝트 로컬 → 사용자 홈 순). 운영에선 대개 첫 후보가 잡힌다.
    if config_path is None:
        candidates = [
            Path("config/nexus_config.yaml"),
            Path("config/nexus_config.yml"),
            Path(".nexus/config.yaml"),
            Path(os.path.expanduser("~/.nexus/config.yaml")),
        ]
        for candidate in candidates:
            if candidate.exists():
                config_path = str(candidate)
                break

    if config_path and Path(config_path).exists():
        # 찾은(또는 지정된) 설정 파일을 확장자에 따라 YAML/JSON으로 읽는다.
        path = Path(config_path)
        if path.suffix in (".yaml", ".yml"):
            try:
                import yaml

                # safe_load는 신뢰할 수 없는 태그 실행을 막아준다. 빈 파일이면
                # None이 나오므로 `or {}`로 빈 dict로 정규화한다.
                with open(path, encoding="utf-8") as f:
                    file_data = yaml.safe_load(f) or {}
            except ImportError:
                # PyYAML 부재는 치명적이지 않게 처리 — 기본값으로라도 뜨게 한다.
                logger.warning("PyYAML이 설치되지 않아 설정 파일을 로드할 수 없습니다.")
                file_data = {}
        else:
            # .yaml/.yml이 아니면 JSON으로 간주해 읽는다.
            import json

            with open(path, encoding="utf-8") as f:
                file_data = json.load(f)

        # 비밀번호 주입(보안): DB/Redis 비밀번호는 설정 파일(yaml)에 평문으로 두지
        # 않고, 실행 환경의 환경변수에서만 주입한다. 이렇게 하면 설정 파일을 git에
        # 올려도 자격증명이 노출되지 않는다.
        #   - NEXUS_PG_PASSWORD    → postgresql.password
        #   - NEXUS_REDIS_PASSWORD → redis.password
        # 왜 여기서 직접 채우나: 아래 NexusConfig(**file_data)는 yaml 값을 init 인자로
        # 넘기는데, pydantic-settings 우선순위상 "init 인자 > 환경변수"라 yaml에 빈
        # 값이 있으면 오히려 환경변수 주입을 덮어버린다. 그래서 yaml에서 비번을 비우고
        # (키 없음) 여기서 file_data에 직접 넣어 환경변수 값이 확실히 반영되게 한다.
        _pg_pw = os.environ.get("NEXUS_PG_PASSWORD")
        if _pg_pw:
            file_data.setdefault("postgresql", {})["password"] = _pg_pw
        _redis_pw = os.environ.get("NEXUS_REDIS_PASSWORD")
        if _redis_pw:
            file_data.setdefault("redis", {})["password"] = _redis_pw

        # 파일에서 읽은 dict를 펼쳐 Pydantic 모델에 주입한다. 이 시점에 환경변수
        # (NEXUS_*)와 각종 validator가 함께 적용되어 최종 설정이 검증·확정된다.
        config = NexusConfig(**file_data)
        logger.info(f"설정 파일 로드 완료: {config_path}")
        # fail-closed 경고: 원격 PostgreSQL인데 비밀번호가 비어 있으면(환경변수 미설정)
        # 접속이 인증 실패로 조용히 깨질 수 있으므로 기동 시 명확히 알린다.
        if not config.postgresql.password and config.postgresql.host not in (
            "localhost",
            "127.0.0.1",
        ):
            logger.warning(
                "PostgreSQL 비밀번호가 비어 있습니다(host=%s). "
                "NEXUS_PG_PASSWORD 환경변수로 자격증명을 주입하세요.",
                config.postgresql.host,
            )
    else:
        # 후보 파일이 하나도 없으면 전부 기본값으로 초기화한다(테스트/최초 실행).
        config = NexusConfig()
        logger.info("기본 설정으로 초기화 (설정 파일 없음)")

    # 멀티테넌시 설정은 별도 파일(config/tenants.yaml)에서 로드하여 병합
    # 이유: 고객별 설정은 자주 바뀌므로 메인 설정과 분리하는 게 운영에 유리
    tenants_path = Path("config/tenants.yaml")
    if tenants_path.exists():
        try:
            import yaml

            with open(tenants_path, encoding="utf-8") as f:
                t_data = yaml.safe_load(f) or {}
            # tenants.yaml이 실제로 테넌트 키를 담고 있을 때만 덮어쓴다(빈/잘못된
            # 파일이 멀쩡한 기본 레지스트리를 날리는 것을 막는 가드). model_copy로
            # tenants 필드만 교체해 나머지 설정은 그대로 보존한다.
            if isinstance(t_data, dict) and ("tenants" in t_data or "default_tenant" in t_data):
                config = config.model_copy(update={"tenants": TenantRegistry(**t_data)})
                logger.info(
                    "테넌트 설정 로드 완료: %s (%d 테넌트)",
                    tenants_path,
                    len(config.tenants.tenants),
                )
        except Exception as e:
            logger.warning("테넌트 설정 로드 실패 (기본값 사용): %s", e)

    return config
