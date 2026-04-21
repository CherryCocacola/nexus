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
from urllib.parse import urlparse

from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings

logger = logging.getLogger("nexus.config")


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
# 쿼리 라우팅 설정 (v7.0 Part 2.5, 2026-04-21)
# ─────────────────────────────────────────────
# 배경: Phase 3 LoRA가 도구 호출을 강화하는 대신 베이스 Qwen의 일반 지식
# 표현을 좁히는 부작용 발생. 2026-04-21 A/B/C/D 실측에서 베이스 모델(LoRA OFF)이
# 니체·카프카 같은 일반 교양 지식을 더 정확히 답변함을 확인.
# 조치: 질의 타입에 따라 런타임에 모델(LoRA ON/OFF)과 temperature를 분기한다.
# 이 분기는 TIER_S 한정 최적화이며, TIER_M 이상에서는 enabled=false로 끈다
# (베이스 모델 + 24개 도구 + 긴 컨텍스트가 이미 기본값이 되기 때문).
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

    분류 규칙:
      1. user_input 길이가 long_input_threshold 이상 → TOOL_MODE
         (첨부 문서/로그 분석 시나리오로 간주)
      2. tool_keywords 중 하나라도 포함 → TOOL_MODE
      3. 그 외 → KNOWLEDGE_MODE (일반 QA)

    enabled=False이면 분류기를 돌리지 않고 항상 tool_mode 프로필을 사용한다
    (하드웨어 업그레이드 또는 문제 발생 시 비상 스위치).
    """

    enabled: bool = True
    long_input_threshold: int = 500  # 이 글자수 이상이면 TOOL_MODE
    tool_keywords: list[str] = Field(
        default_factory=lambda: [
            # 한국어 — 파일/프로젝트/도구 명시 힌트
            "파일", "첨부", "업로드", "이 프로젝트", "코드베이스",
            "디렉토리", "폴더", "리포지토리", "리포지터리",
            "읽어줘", "읽어 줘", "편집해", "수정해",
            "모듈 구조", "디렉토리 구조", "프로젝트 구조",
            # 영어
            "file", "attached", "upload", "this project", "codebase",
            "repository", "directory", "folder",
            # 도구 이름 — 괄호 포함/미포함 양쪽 패턴
            # 괄호 포함(함수 호출 스타일)
            "Read(", "Write(", "Edit(", "Bash(", "Glob(", "Grep(",
            "Agent(", "DocumentProcess",
            # 단독 대문자 도구명 + 공백 — "Read 도구", "Edit the file" 등
            "Read ", "Write ", "Edit ", "Bash ", "Glob ", "Grep ",
            "Agent ", " LS ",
            # 확장자 힌트 (공백 뒤 경로 패턴)
            ".py ", ".md ", ".yaml ", ".json ",
            # 프로젝트 내부 디렉토리 prefix — core/orchestrator, web/app.py 등
            # 사용자가 특정 경로를 언급하면 코드 작업으로 간주
            "core/", "web/", "tests/", "training/", "deployment/",
            "cli/", "config/", "scripts/", "tools/",
        ]
    )
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
            ".py", ".js", ".ts", ".json", ".yaml", ".yml", ".toml",
            ".md", ".txt", ".csv", ".html", ".css", ".sql",
            ".sh", ".bash", ".dockerfile", ".env.example",
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

    return config
