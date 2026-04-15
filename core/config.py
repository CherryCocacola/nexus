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

    primary_model: str = "gemma-4-31b-it"
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
