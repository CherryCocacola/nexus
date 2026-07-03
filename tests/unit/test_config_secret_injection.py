"""설정 로더의 비밀번호 환경변수 주입(보안) 검증.

무엇을 검증하나:
    core/config.py 의 load_and_validate_config()가 DB/Redis 비밀번호를
    설정 파일(yaml)의 평문이 아니라 환경변수에서만 주입하는지 확인한다.

왜 중요한가 (Security Critical #4 후속):
    설정 파일은 git에 커밋되어 GitHub로 푸시된다. 비밀번호가 yaml에 평문으로
    남아 있으면 그대로 노출된다. 그래서 yaml에서는 비번을 비우고, 실행 환경의
    환경변수(NEXUS_PG_PASSWORD / NEXUS_REDIS_PASSWORD)로만 주입한다.

주의 (env 주입이 왜 로더에서 일어나는가):
    로더는 NexusConfig(**file_data)로 yaml을 init 인자로 넘긴다.
    pydantic-settings 우선순위상 "init 인자 > 환경변수"라, yaml에 빈 값이 있으면
    오히려 환경변수 주입을 덮는다. 그래서 로더가 file_data에 직접 비번을 채운다.
    이 테스트는 그 경로가 실제로 동작하는지(그리고 평문이 사라졌는지) 지킨다.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.config import load_and_validate_config

# 실제 커밋된 운영 설정 파일 — 이 파일에 평문 비번이 없어야 한다.
CONFIG_PATH = "config/nexus_config.yaml"


def test_pg_password_injected_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """NEXUS_PG_PASSWORD 환경변수가 있으면 postgresql.password에 주입된다."""
    monkeypatch.setenv("NEXUS_PG_PASSWORD", "PG_FROM_ENV")
    monkeypatch.delenv("NEXUS_REDIS_PASSWORD", raising=False)

    config = load_and_validate_config(CONFIG_PATH)

    assert config.postgresql.password == "PG_FROM_ENV"  # noqa: S105 (테스트용 더미)


def test_redis_password_injected_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """NEXUS_REDIS_PASSWORD 환경변수가 있으면 redis.password에 주입된다."""
    monkeypatch.setenv("NEXUS_REDIS_PASSWORD", "REDIS_FROM_ENV")
    monkeypatch.delenv("NEXUS_PG_PASSWORD", raising=False)

    config = load_and_validate_config(CONFIG_PATH)

    assert config.redis.password == "REDIS_FROM_ENV"  # noqa: S105 (테스트용 더미)


def test_pg_password_empty_when_env_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    """환경변수가 없으면 postgresql.password는 빈 문자열(fail-closed)이어야 한다.

    yaml에 평문 비번이 없으므로, 환경변수 미설정 시 비번은 기본값(빈 문자열)이며
    원격 DB 접속은 인증 실패로 즉시 드러난다(조용한 오작동 방지).
    """
    monkeypatch.delenv("NEXUS_PG_PASSWORD", raising=False)

    config = load_and_validate_config(CONFIG_PATH)

    assert config.postgresql.password == ""


def test_redis_password_none_when_env_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    """환경변수가 없으면 redis.password는 None(기본 무인증)이어야 한다."""
    monkeypatch.delenv("NEXUS_REDIS_PASSWORD", raising=False)

    config = load_and_validate_config(CONFIG_PATH)

    assert config.redis.password is None


def test_yaml_has_no_plaintext_db_passwords() -> None:
    """운영 설정 파일에 과거 평문 비밀번호가 남아 있지 않아야 한다(회귀 방지).

    구체적으로 과거 커밋됐던 값(PG 'idino@12', Redis requirepass 해시)이
    파일에서 완전히 제거됐는지 문자열 수준으로 확인한다.
    """
    text = Path(CONFIG_PATH).read_text(encoding="utf-8")

    # 과거 노출됐던 두 평문 자격증명이 파일에 존재하면 실패시킨다.
    assert "idino@12" not in text
    assert "2a4a59309d85062c" not in text  # Redis requirepass 해시 접두부
