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
    """GPU 서버 (Machine B) 연결 설정."""

    url: str = "http://localhost:8000"
    embedding_url: str = "http://localhost:8002"  # 임베딩 서버 (별도 vLLM 인스턴스)
    timeout_seconds: float = 120.0
    max_retries: int = 10
    retry_base_delay: float = 0.5
    health_check_interval: float = 30.0

    @field_validator("url")
    @classmethod
    def validate_url_is_local(cls, v: str) -> str:
        """에어갭 검증: GPU 서버 URL이 로컬/LAN 주소인지 확인한다."""
        parsed = urlparse(v)
        hostname = parsed.hostname or ""
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
    """Redis 연결 설정."""

    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: str | None = None
    socket_timeout: float = 5.0


# ─────────────────────────────────────────────
# PostgreSQL 설정 (장기 메모리 + pgvector)
# ─────────────────────────────────────────────
class PostgreSQLConfig(BaseModel):
    """PostgreSQL 연결 설정."""

    host: str = "localhost"
    port: int = 5432
    database: str = "nexus"
    user: str = "nexus"
    password: str = ""
    min_connections: int = 2
    max_connections: int = 10


# ─────────────────────────────────────────────
# 모델 설정
# ─────────────────────────────────────────────
class ModelConfig(BaseModel):
    """LLM 모델 설정."""

    primary_model: str = "qwen3.5-27b"
    auxiliary_model: str = "exaone-7.8b"
    embedding_model: str = "multilingual-e5-large"
    max_context_tokens: int = 4096
    default_temperature: float = 0.7
    default_max_tokens: int = 4096


# ─────────────────────────────────────────────
# 세션 설정
# ─────────────────────────────────────────────
class SessionConfig(BaseModel):
    """세션 관리 설정."""

    sessions_dir: str = ".nexus/sessions"
    max_turns: int = 50
    max_budget_seconds: float = 300.0
    session_ttl_hours: int = 24
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

    model_config = {"protected_namespaces": ()}  # model_ 접두사 경고 방지

    enabled: bool = True  # TIER_S에서만 자동 활성화
    base_url: str = "http://192.168.22.28:8003"
    api_key: str = "local-key"
    model_id: str = "qwen3.5-4b"
    max_context_tokens: int = 4096
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
    """개별 라우팅 프로필 — 질의 타입별 모델/파라미터 조합."""

    model_config = {"protected_namespaces": ()}  # model_ 접두사 경고 방지

    model: str  # vLLM served-model-name (LoRA 어댑터 이름 또는 베이스)
    temperature: float = 0.3
    max_tokens: int = 4096
    enable_thinking: bool = False  # Qwen3.5 chat_template_kwargs 인자
    description: str = ""  # 운영자/테스트용 설명


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
        )
    )
    tool_mode: RoutingProfile = Field(
        default_factory=lambda: RoutingProfile(
            model="nexus-phase3",
            temperature=0.3,
            max_tokens=4096,
            enable_thinking=False,
            description="도구 호출 — Phase 3 LoRA + 중간 temperature",
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

    model_config = {"protected_namespaces": ()}  # model_ 접두사 경고 방지

    id: str
    name: str = ""
    description: str = ""

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
    """테넌트 레지스트리."""

    default_tenant: str = "default"
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


class SecurityConfig(BaseModel):
    """보안 및 샌드박스 설정."""

    sandbox_enabled: bool = True
    sandbox_timeout_seconds: float = 30.0
    bash_allow_patterns: list[str] = Field(default_factory=list)
    bash_deny_patterns: list[str] = Field(
        default_factory=lambda: [
            r"rm\s+-rf\s+/",
            r":(){ :\|:& };:",
            r"dd\s+if=/dev/zero",
            r"mkfs\.",
            r">\s*/dev/sd",
        ]
    )
    max_file_size_bytes: int = 10 * 1024 * 1024  # 10MB
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

    # v7.0 Part 2.5 쿼리 라우팅 — 지식/도구 질의 분기 (2026-04-21 추가)
    routing: RoutingConfig = Field(default_factory=RoutingConfig)

    # 멀티테넌시 (Part 5 Ch 15, 2026-04-21)
    tenants: TenantRegistry = Field(default_factory=TenantRegistry)

    # v7.2 MCP 통합 — LAN 내부 MCP 서버 연결 (기본 비활성, 에어갭 fail-closed)
    mcp: McpConfig = Field(default_factory=McpConfig)

    # 하드웨어 티어 (auto: GPU VRAM 기반 자동 감지)
    hardware_tier: str = "auto"

    # 운영
    log_level: str = "INFO"
    log_file: str | None = None
    watch_files: bool = False
    debug: bool = False

    # 에어갭 모드
    air_gap_mode: bool = True

    @model_validator(mode="after")
    def validate_air_gap(self) -> NexusConfig:
        """에어갭 모드가 켜져 있으면 GPU 서버 URL이 로컬인지 확인한다."""
        if self.air_gap_mode:
            parsed = urlparse(self.gpu_server_url)
            hostname = parsed.hostname or ""
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
        # YAML 파일에서 로드
        path = Path(config_path)
        if path.suffix in (".yaml", ".yml"):
            try:
                import yaml

                with open(path, encoding="utf-8") as f:
                    file_data = yaml.safe_load(f) or {}
            except ImportError:
                logger.warning("PyYAML이 설치되지 않아 설정 파일을 로드할 수 없습니다.")
                file_data = {}
        else:
            import json

            with open(path, encoding="utf-8") as f:
                file_data = json.load(f)

        config = NexusConfig(**file_data)
        logger.info(f"설정 파일 로드 완료: {config_path}")
    else:
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
