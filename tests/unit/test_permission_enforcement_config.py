"""
권한 강제 배선 설정 단위 테스트 — PermissionEnforcementConfig / AuditConfig /
CommandFilterConfig 기본값 + B200 배포 템플릿 로드.

무엇을 고정하는가 (구현 사실):
    1) 코드 기본값(무회귀): permission_enforcement.enabled=False, mode="shadow",
       audit.enabled=True, command_filter.block_package_install=False(개발 허용).
    2) 배포 템플릿(config/examples/nexus_config.b200.yaml) 로드 시:
       enabled=True, mode="enforce", block_package_install=True.

왜 중요한가:
    기본값이 조용히 바뀌면(예: enabled 기본이 True가 되면) 개발 환경에서 갑자기
    도구가 차단되는 회귀가 생긴다. 반대로 b200 템플릿이 shadow로 후퇴하면 배포
    환경의 실제 차단이 사라진다. 이 테스트가 양쪽을 못박는다.
"""

from __future__ import annotations

from pathlib import Path

from core.config import (
    AuditConfig,
    CommandFilterConfig,
    NexusConfig,
    PermissionEnforcementConfig,
    load_and_validate_config,
)

# 이 파일(tests/unit/…) 기준으로 리포지토리 루트를 거슬러 올라가 b200 템플릿을 찾는다.
# 테스트 실행 cwd에 의존하지 않도록 __file__ 기준 절대경로를 쓴다.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_B200_CONFIG = _REPO_ROOT / "config" / "examples" / "nexus_config.b200.yaml"


class TestPermissionEnforcementDefaults:
    """PermissionEnforcementConfig의 무회귀 기본값(비활성 + shadow)을 고정한다."""

    def test_enabled_default_is_false(self) -> None:
        """기본은 파이프라인 미배선 — enabled=False(현행 executor 동작 100% 유지)."""
        assert PermissionEnforcementConfig().enabled is False

    def test_mode_default_is_shadow(self) -> None:
        """enabled를 켜더라도 우선은 관측만 — mode 기본은 'shadow'(fail-safe)."""
        assert PermissionEnforcementConfig().mode == "shadow"

    def test_nexus_config_wires_default_enforcement(self) -> None:
        """NexusConfig가 비활성 기본 PermissionEnforcementConfig를 자동으로 가진다."""
        cfg = NexusConfig()
        assert isinstance(cfg.permission_enforcement, PermissionEnforcementConfig)
        assert cfg.permission_enforcement.enabled is False
        assert cfg.permission_enforcement.mode == "shadow"


class TestAuditAndCommandFilterDefaults:
    """감사 로그(관측은 항상 켜둠)와 설치차단 게이팅(개발 허용) 기본값을 고정한다."""

    def test_audit_enabled_default_is_true(self) -> None:
        """shadow 관측 결과를 남기려면 감사 로그가 기본 활성이어야 한다."""
        assert AuditConfig().enabled is True

    def test_command_filter_block_install_default_is_false(self) -> None:
        """개발 방침: 개발 중엔 pip/npm 설치 허용 → 기본 block_package_install=False."""
        assert CommandFilterConfig().block_package_install is False

    def test_nexus_config_defaults_are_dev_safe(self) -> None:
        """NexusConfig 기본 조합이 '개발 안전'(설치 허용 + 감사 on)인지 확인한다."""
        cfg = NexusConfig()
        assert cfg.audit.enabled is True
        assert cfg.command_filter.block_package_install is False


class TestB200DeploymentTemplate:
    """B200 배포 템플릿은 실제 차단(enforce) + 설치차단(에어갭)을 켜야 한다."""

    def test_b200_template_file_exists(self) -> None:
        """전제 조건 — 배포 템플릿 파일이 실제로 존재해야 한다."""
        assert _B200_CONFIG.exists(), f"b200 템플릿 없음: {_B200_CONFIG}"

    def test_b200_enables_enforce_mode(self) -> None:
        """b200 로드 시 permission_enforcement가 활성 + enforce여야 한다(실제 차단)."""
        cfg = load_and_validate_config(str(_B200_CONFIG))
        assert cfg.permission_enforcement.enabled is True
        assert cfg.permission_enforcement.mode == "enforce"

    def test_b200_enables_install_block_and_audit(self) -> None:
        """b200 로드 시 설치차단(에어갭)과 감사 로그가 모두 켜져 있어야 한다."""
        cfg = load_and_validate_config(str(_B200_CONFIG))
        assert cfg.command_filter.block_package_install is True
        assert cfg.audit.enabled is True
