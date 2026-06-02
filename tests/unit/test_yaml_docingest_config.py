"""
config/nexus_config.yaml 의 docingest MCP 서버 항목 정합 단위 테스트.

검증 대상 (후속 구현 B):
  실제 운영 설정 파일을 load_and_validate_config() 로 로드하여,
  v7.3 docingest MCP 서버 항목이 사양대로 들어 있고 docutil(과거명)은 더 이상
  존재하지 않는지 확인한다.
    - docutil 부재(이름 변경 정합).
    - docingest 존재 + base_url 포트 8814.
    - enabled=false (에어갭 fail-closed).
    - trust.read_only=true (조회 전용 신뢰 메타).
    - allow_write 기본 false (쓰기 도구 ingest 자동 등록 제외).
    - expose_to_worker 기본 true (Worker 도구 풀에 노출).

격리: 외부 네트워크/서버 접근 없음 — 디스크의 설정 파일을 파싱한 객체만 검증한다.
  load_and_validate_config()는 LAN 검증(validate_lan_urls)을 거치지만, 모든
  서버가 enabled=false 이거나 LAN 대역이라 강등 경고가 발생하지 않는다.
"""

from __future__ import annotations

import pytest

from core.config import load_and_validate_config


# ─────────────────────────────────────────────
# 설정 로드 픽스처 — 실제 config/nexus_config.yaml 1회 로드
# ─────────────────────────────────────────────
@pytest.fixture(scope="module")
def mcp_servers_by_name():
    """
    운영 설정 파일을 로드해 {서버이름: McpServerConfig} 매핑을 돌려준다.

    module scope: 파일 I/O 1회로 충분하며, 설정 객체는 읽기 전용으로만 다룬다.
    config_path 를 명시하지 않으면 load_and_validate_config 가 후보 경로
    (config/nexus_config.yaml)를 탐색해 로드한다.
    """
    config = load_and_validate_config()
    return {s.name: s for s in config.mcp.servers}


# ─────────────────────────────────────────────
# docutil → docingest 이름 변경 정합
# ─────────────────────────────────────────────
class TestDocutilRemoved:
    """과거 이름 'docutil' 이 설정에서 완전히 제거됐는지 검증한다."""

    def test_docutil_server_is_absent(self, mcp_servers_by_name):
        """'docutil' 이라는 이름의 MCP 서버는 더 이상 없어야 한다(이름 정합)."""
        assert "docutil" not in mcp_servers_by_name


# ─────────────────────────────────────────────
# docingest 서버 항목 정합
# ─────────────────────────────────────────────
class TestDocingestServerEntry:
    """docingest MCP 서버 항목이 사양대로 구성됐는지 검증한다."""

    def test_docingest_server_exists(self, mcp_servers_by_name):
        """docingest 서버가 설정에 존재해야 한다."""
        assert "docingest" in mcp_servers_by_name

    def test_docingest_base_url_port_8814(self, mcp_servers_by_name):
        """docingest base_url 의 포트는 8814 여야 한다(운영 placeholder 확정값)."""
        server = mcp_servers_by_name["docingest"]
        assert server.base_url.endswith(":8814"), server.base_url

    def test_docingest_enabled_is_false(self, mcp_servers_by_name):
        """docingest 는 enabled=false 여야 한다(에어갭 fail-closed)."""
        assert mcp_servers_by_name["docingest"].enabled is False

    def test_docingest_trust_read_only_is_true(self, mcp_servers_by_name):
        """trust.read_only=true (조회 전용으로 신뢰 — 자동 등록 대상)."""
        server = mcp_servers_by_name["docingest"]
        assert server.trust.get("read_only") is True

    def test_docingest_allow_write_defaults_false(self, mcp_servers_by_name):
        """allow_write 미지정 → 기본 false (쓰기 도구 ingest 는 자동 등록 제외)."""
        assert mcp_servers_by_name["docingest"].allow_write is False

    def test_docingest_expose_to_worker_defaults_true(self, mcp_servers_by_name):
        """expose_to_worker 미지정 → 기본 true (Worker 도구 풀에 parse/search 노출)."""
        assert mcp_servers_by_name["docingest"].expose_to_worker is True
