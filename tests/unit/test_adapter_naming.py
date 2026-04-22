"""
M7 어댑터 네이밍 규약 테스트 (2026-04-22).

- training.adapter_naming 의 순수 함수들
- core.config.TenantConfig.adapter_name()
- training.trainer.TrainingConfig.resolved_output_dir / resolved_adapter_name

핵심 불변식:
  - default 테넌트: `nexus-phaseN` (기존 호환, 접두 없음)
  - 커스텀 테넌트: `nexus-{id}-phaseN`
  - custom_prefix: 접두 우회
  - phase 범위 벗어나면 ValueError
  - 허용되지 않는 문자(한글·공백·특수문자)는 ValueError
"""
from __future__ import annotations

import pytest

from core.config import TenantConfig
from training.adapter_naming import (
    DEFAULT_TENANT_ID,
    MAX_PHASE,
    MIN_PHASE,
    compose_adapter_name,
    compose_data_path,
    compose_output_dir,
    normalize_tenant_id,
)
from training.trainer import TrainingConfig


# ─────────────────────────────────────────────
# normalize_tenant_id
# ─────────────────────────────────────────────
@pytest.mark.parametrize(
    "given, expected",
    [
        (None, "default"),
        ("", "default"),
        ("  ", "default"),
        ("default", "default"),
        ("DEFAULT", "default"),      # 대문자 내림
        ("dongguk", "dongguk"),
        ("hy-univ", "hy-univ"),
        ("team_a_2", "team_a_2"),
        ("  dongguk ", "dongguk"),   # 주변 공백 제거
    ],
)
def test_normalize_tenant_id_accepts_valid(given, expected) -> None:
    """허용 문자셋과 정규화 규칙이 일치해야 한다."""
    assert normalize_tenant_id(given) == expected


@pytest.mark.parametrize(
    "invalid",
    [
        "동국",                 # 한글
        "my tenant",           # 공백
        "tenant@home",         # 특수문자
        "-leading-dash",       # 앞자리 dash 금지 (DNS label 규약)
        "_underscore-lead",    # 앞자리 underscore 금지
        "a" * 64,              # 63자 초과
    ],
)
def test_normalize_tenant_id_rejects_invalid(invalid) -> None:
    """허용 문자셋을 벗어나면 ValueError를 던져야 한다."""
    with pytest.raises(ValueError):
        normalize_tenant_id(invalid)


# ─────────────────────────────────────────────
# compose_adapter_name
# ─────────────────────────────────────────────
def test_compose_adapter_name_default_tenant_has_no_prefix() -> None:
    """default 테넌트는 기존 `nexus-phase3`와 호환되어야 한다 (접두 없음)."""
    assert compose_adapter_name(None, 3) == "nexus-phase3"
    assert compose_adapter_name("default", 3) == "nexus-phase3"


def test_compose_adapter_name_custom_tenant() -> None:
    """특정 테넌트는 `nexus-{id}-phaseN` 형식을 따라야 한다."""
    assert compose_adapter_name("dongguk", 3) == "nexus-dongguk-phase3"
    assert compose_adapter_name("hy-univ", 2) == "nexus-hy-univ-phase2"


def test_compose_adapter_name_custom_prefix_overrides_tenant() -> None:
    """custom_prefix가 지정되면 tenant_id 접두를 붙이지 않고 prefix를 그대로 사용."""
    assert (
        compose_adapter_name("hanyang", 2, custom_prefix="hy-custom")
        == "hy-custom-phase2"
    )


@pytest.mark.parametrize("phase", [-1, MAX_PHASE + 1, 10])
def test_compose_adapter_name_phase_out_of_range_raises(phase) -> None:
    """Phase 범위를 벗어나면 ValueError."""
    with pytest.raises(ValueError):
        compose_adapter_name("dongguk", phase)


@pytest.mark.parametrize("phase", list(range(MIN_PHASE, MAX_PHASE + 1)))
def test_compose_adapter_name_phase_in_range_ok(phase) -> None:
    """Phase 0~4 모두 유효해야 한다."""
    name = compose_adapter_name("dongguk", phase)
    assert name == f"nexus-dongguk-phase{phase}"


# ─────────────────────────────────────────────
# compose_output_dir / compose_data_path
# ─────────────────────────────────────────────
def test_compose_output_dir_default_tenant() -> None:
    """default 테넌트는 기존 경로(qwen35-phaseN) 유지."""
    assert (
        compose_output_dir(None, 3)
        == "/opt/nexus-gpu/checkpoints/qwen35-phase3"
    )


def test_compose_output_dir_custom_tenant() -> None:
    """테넌트별 경로는 tenant_id가 중간에 끼어야 한다."""
    assert (
        compose_output_dir("dongguk", 3)
        == "/opt/nexus-gpu/checkpoints/qwen35-dongguk-phase3"
    )


def test_compose_data_path_default_tenant_matches_legacy() -> None:
    """기존 default 학습 데이터 경로와 일치해야 한다 (회귀 방지)."""
    assert (
        compose_data_path(None)
        == "/opt/nexus-gpu/training/bootstrap_data.jsonl"
    )


def test_compose_data_path_custom_tenant_isolated_subdir() -> None:
    """테넌트별 데이터는 서브디렉토리로 격리된다."""
    assert (
        compose_data_path("dongguk")
        == "/opt/nexus-gpu/training/dongguk/bootstrap_data.jsonl"
    )


# ─────────────────────────────────────────────
# TenantConfig.adapter_name
# ─────────────────────────────────────────────
def test_tenant_config_adapter_name_default() -> None:
    """default 테넌트는 `nexus-phaseN` 반환."""
    t = TenantConfig(id=DEFAULT_TENANT_ID, name="Default")
    assert t.adapter_name(3) == "nexus-phase3"


def test_tenant_config_adapter_name_custom() -> None:
    """이름 있는 테넌트는 접두 포함 반환."""
    t = TenantConfig(id="dongguk", name="동국대학교")
    assert t.adapter_name(2) == "nexus-dongguk-phase2"


def test_tenant_config_adapter_name_with_prefix_override() -> None:
    """adapter_name_prefix가 있으면 해당 접두가 우선한다."""
    t = TenantConfig(id="hanyang", adapter_name_prefix="hy-custom")
    assert t.adapter_name(4) == "hy-custom-phase4"


# ─────────────────────────────────────────────
# TrainingConfig.resolved_*
# ─────────────────────────────────────────────
def test_training_config_resolved_without_tenant_returns_raw_output() -> None:
    """tenant/phase 미지정이면 직접 지정한 output_dir이 그대로 반환되어야 한다 (하위 호환)."""
    cfg = TrainingConfig(output_dir="./my-checkpoint/")
    assert cfg.resolved_output_dir() == "./my-checkpoint/"
    assert cfg.resolved_adapter_name() is None


def test_training_config_resolved_with_tenant_and_phase() -> None:
    """tenant_id + phase가 모두 있으면 자동 해석 경로·이름을 써야 한다."""
    cfg = TrainingConfig(tenant_id="dongguk", phase=3)
    assert (
        cfg.resolved_output_dir()
        == "/opt/nexus-gpu/checkpoints/qwen35-dongguk-phase3"
    )
    assert cfg.resolved_adapter_name() == "nexus-dongguk-phase3"


def test_training_config_to_dict_exposes_m7_fields() -> None:
    """API 페이로드(to_dict)에 tenant_id·phase·adapter_name이 포함돼야 한다."""
    cfg = TrainingConfig(tenant_id="dongguk", phase=3)
    payload = cfg.to_dict()
    assert payload["tenant_id"] == "dongguk"
    assert payload["phase"] == 3
    assert payload["adapter_name"] == "nexus-dongguk-phase3"
