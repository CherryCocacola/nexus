"""
core/config.py 의 MCP 설정 단위 테스트.

검증 대상 (v7.2 MCP 통합 — 에어갭 fail-closed):
  1. is_lan_hostname(): LAN(사설망) 대역 판정 — 특히 172.16~31 경계.
     - 정식 위치 core.security.network_guard.is_lan_hostname
     - 별칭 core.config._is_lan_hostname (하위호환) 동일성 검증
  2. McpConfig 기본값: 전역 enabled=False (에어갭 fail-closed).
  3. McpConfig.validate_lan_urls(): 외부 도메인 + enabled=True 서버를
     enabled=False로 강제 강등하고 경고를 남긴다. LAN+enabled는 유지.

외부 네트워크/실제 서버 접근 없음 — 순수 설정 객체 검증만 수행한다.
"""

from __future__ import annotations

import pytest

from core.config import McpConfig, McpServerConfig, _is_lan_hostname
from core.security import is_lan_hostname as is_lan_hostname_from_pkg
from core.security.network_guard import is_lan_hostname


# ─────────────────────────────────────────────
# _is_lan_hostname — LAN 대역 판정
# ─────────────────────────────────────────────
class TestIsLanHostname:
    """_is_lan_hostname()의 사설/공인 IP 판정을 검증한다."""

    def test_is_lan_hostname_private_class_c_returns_true(self):
        """192.168.x (사설 C)는 LAN으로 판정되어야 한다."""
        assert _is_lan_hostname("192.168.1.1") is True

    def test_is_lan_hostname_private_class_a_returns_true(self):
        """10.x (사설 A)는 LAN으로 판정되어야 한다."""
        assert _is_lan_hostname("10.0.0.1") is True

    def test_is_lan_hostname_private_class_b_lower_bound_returns_true(self):
        """172.16.x (사설 B 하한)는 LAN으로 판정되어야 한다."""
        assert _is_lan_hostname("172.16.0.1") is True

    def test_is_lan_hostname_private_class_b_upper_bound_returns_true(self):
        """172.31.x (사설 B 상한)는 LAN으로 판정되어야 한다."""
        assert _is_lan_hostname("172.31.255.254") is True

    def test_is_lan_hostname_public_172_217_returns_false(self):
        """172.217.x (구글 등 공인 IP)는 LAN이 아니어야 한다 — 단순 prefix 오판 방지."""
        assert _is_lan_hostname("172.217.0.1") is False

    def test_is_lan_hostname_172_15_below_range_returns_false(self):
        """172.15.x (사설 B 범위 미만)는 공인이므로 LAN이 아니어야 한다."""
        assert _is_lan_hostname("172.15.0.1") is False

    def test_is_lan_hostname_172_32_above_range_returns_false(self):
        """172.32.x (사설 B 범위 초과)는 공인이므로 LAN이 아니어야 한다."""
        assert _is_lan_hostname("172.32.0.1") is False

    def test_is_lan_hostname_public_dns_ip_returns_false(self):
        """8.8.8.8 (공인 DNS)는 LAN이 아니어야 한다."""
        assert _is_lan_hostname("8.8.8.8") is False

    def test_is_lan_hostname_localhost_returns_true(self):
        """localhost는 루프백이므로 LAN으로 판정되어야 한다."""
        assert _is_lan_hostname("localhost") is True

    def test_is_lan_hostname_loopback_ip_returns_true(self):
        """127.0.0.1 (루프백 IP)은 LAN으로 판정되어야 한다."""
        assert _is_lan_hostname("127.0.0.1") is True

    def test_is_lan_hostname_empty_returns_false(self):
        """빈 hostname은 fail-closed로 LAN이 아니어야 한다."""
        assert _is_lan_hostname("") is False

    @pytest.mark.parametrize(
        "second_octet,expected",
        [(16, True), (17, True), (24, True), (31, True), (15, False), (32, False), (0, False)],
    )
    def test_is_lan_hostname_172_boundary_full_range(self, second_octet, expected):
        """172의 두 번째 옥텟 경계(16~31)를 전수 검증한다."""
        assert _is_lan_hostname(f"172.{second_octet}.0.1") is expected


# ─────────────────────────────────────────────
# is_lan_hostname 위치 이동 — 정식 경로 + 별칭 동일성 (v7.2)
# ─────────────────────────────────────────────
class TestIsLanHostnameCanonicalLocation:
    """is_lan_hostname의 정식 위치(core.security.network_guard)와 별칭 호환을 검증한다.

    이 함수는 core/config.py의 private 함수에서 core/security/network_guard.py의
    public 함수로 승격되었다(의존성 방향 명확화). 기존 import 경로
    (from core.config import _is_lan_hostname)는 별칭으로 계속 동작해야 한다.
    """

    def test_canonical_path_same_object_as_package_export(self):
        """network_guard.is_lan_hostname == core.security.is_lan_hostname (동일 객체)."""
        assert is_lan_hostname is is_lan_hostname_from_pkg

    def test_config_alias_is_same_object_as_canonical(self):
        """별칭 core.config._is_lan_hostname은 정식 함수와 동일 객체여야 한다."""
        assert _is_lan_hostname is is_lan_hostname

    @pytest.mark.parametrize(
        "hostname,expected",
        [
            ("192.168.1.1", True),
            ("10.0.0.1", True),
            ("172.16.0.1", True),
            ("172.31.255.254", True),
            ("172.15.0.1", False),
            ("172.32.0.1", False),
            ("172.217.0.1", False),
            ("8.8.8.8", False),
            ("localhost", True),
            ("127.0.0.1", True),
            ("", False),
        ],
    )
    def test_canonical_is_lan_hostname_judgement(self, hostname, expected):
        """정식 경로(network_guard.is_lan_hostname)로도 LAN 판정이 동일해야 한다."""
        assert is_lan_hostname(hostname) is expected


# ─────────────────────────────────────────────
# McpConfig 기본값 — 전역 fail-closed
# ─────────────────────────────────────────────
class TestMcpConfigDefaults:
    """McpConfig의 fail-closed 기본값을 검증한다."""

    def test_mcp_config_default_enabled_is_false(self):
        """전역 마스터 스위치는 기본 OFF여야 한다 (에어갭 fail-closed)."""
        config = McpConfig()
        assert config.enabled is False

    def test_mcp_config_default_servers_is_empty(self):
        """기본 서버 목록은 비어 있어야 한다."""
        config = McpConfig()
        assert config.servers == []

    def test_mcp_server_config_default_enabled_is_false(self):
        """개별 서버도 기본 enabled=False여야 한다 (명시 활성만)."""
        server = McpServerConfig(name="db", base_url="http://192.168.1.10:9000")
        assert server.enabled is False

    def test_mcp_server_config_default_allow_write_is_false(self):
        """allow_write는 기본 False여야 한다 (fail-closed — 쓰기 서버는 명시 허용만).

        초기 제품 정책: read-only 서버만 자동 등록한다. trust.read_only=False인
        쓰기 가능 서버는 운영자가 allow_write=True로 명시했을 때에만 등록된다.
        """
        server = McpServerConfig(name="db", base_url="http://192.168.1.10:9000")
        assert server.allow_write is False

    def test_mcp_server_config_allow_write_can_be_enabled(self):
        """운영자가 allow_write=True로 명시하면 그대로 반영되어야 한다."""
        server = McpServerConfig(name="db", base_url="http://192.168.1.10:9000", allow_write=True)
        assert server.allow_write is True

    def test_mcp_server_config_default_expose_to_worker_is_true(self):
        """expose_to_worker는 기본 True여야 한다 (기본 노출 — 명시적 제외만 차단).

        v7.3 회귀: expose_to_worker는 fail-closed가 아니라 "기본 노출(True)"이다.
        대부분의 MCP 서버는 Worker 도구 풀에 노출되어야 정상이고, kowiki처럼
        자동 RAG와 기능이 중복되는 특수 서버만 명시적으로 False로 제외한다.
        따라서 미지정 시에는 노출(True)이 안전한 기본값이다.
        """
        server = McpServerConfig(name="db", base_url="http://192.168.1.10:9000")
        assert server.expose_to_worker is True

    def test_mcp_server_config_expose_to_worker_can_be_disabled(self):
        """운영자가 expose_to_worker=False로 명시하면 그대로 반영되어야 한다.

        예) kowiki MCP는 KNOWLEDGE 모드 자동 RAG와 중복되어 8K overflow를
        유발하므로 Worker 풀에서 제외(False)한다.
        """
        server = McpServerConfig(
            name="kowiki",
            base_url="http://192.168.10.39:9001",
            expose_to_worker=False,
        )
        assert server.expose_to_worker is False


# ─────────────────────────────────────────────
# validate_lan_urls — 외부 도메인 강등 (model_validator)
# ─────────────────────────────────────────────
class TestValidateLanUrls:
    """McpConfig.validate_lan_urls()의 외부 도메인 강등 동작을 검증한다."""

    def test_validate_lan_urls_external_enabled_server_is_demoted(self):
        """외부 도메인 + enabled=True 서버는 enabled=False로 강등되고 경고가 발생해야 한다."""
        # pytest.warns는 전역 filterwarnings(ignore::UserWarning)를 로컬에서 무시하고
        # 경고 포착을 강제한다.
        with pytest.warns(UserWarning):
            config = McpConfig(
                enabled=True,
                servers=[
                    McpServerConfig(
                        name="evil",
                        base_url="https://api.openai.com",
                        enabled=True,
                    )
                ],
            )
        # 외부 서버는 강등되어 enabled=False가 되어야 한다.
        assert config.servers[0].enabled is False

    def test_validate_lan_urls_lan_enabled_server_is_kept(self):
        """LAN 대역 + enabled=True 서버는 그대로 유지되어야 한다."""
        config = McpConfig(
            enabled=True,
            servers=[
                McpServerConfig(
                    name="db",
                    base_url="http://192.168.10.39:9000",
                    enabled=True,
                )
            ],
        )
        # LAN 서버는 강등되지 않는다.
        assert config.servers[0].enabled is True

    def test_validate_lan_urls_external_disabled_server_stays_disabled(self):
        """외부 도메인이지만 enabled=False인 서버는 이미 비활성이므로 그대로 유지된다."""
        # enabled=False이면 강등 분기(if server.enabled)에 진입하지 않으므로 경고도 없다.
        config = McpConfig(
            enabled=True,
            servers=[
                McpServerConfig(
                    name="ext",
                    base_url="https://example.com",
                    enabled=False,
                )
            ],
        )
        assert config.servers[0].enabled is False

    def test_validate_lan_urls_mixed_servers_only_external_demoted(self):
        """LAN과 외부가 섞여 있으면 외부만 강등되고 LAN은 유지되어야 한다."""
        with pytest.warns(UserWarning):
            config = McpConfig(
                enabled=True,
                servers=[
                    McpServerConfig(name="db", base_url="http://10.0.0.5:9000", enabled=True),
                    McpServerConfig(name="ext", base_url="http://8.8.8.8:9000", enabled=True),
                ],
            )
        by_name = {s.name: s for s in config.servers}
        assert by_name["db"].enabled is True
        assert by_name["ext"].enabled is False
