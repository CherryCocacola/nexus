"""
CommandFilter 설치차단 게이팅 단위 테스트 — block_package_install 플래그.

무엇을 고정하는가 (구현 사실 core/security/command_filter.py, 실측으로 검증):
    - block_package_install=True(배포/기본): "…패키지 설치" 사유가 붙은 install
      패턴(pip/npm/apt/yum/brew install)이 'high' 위험으로 차단된다(에어갭 준수).
    - block_package_install=False(개발): 그 install 패턴들만 위험 목록에서 제외된다.
      → 결과는 명령어별로 갈린다:
          · pip install …    : pip이 안전 목록(SAFE_COMMANDS)에 있어 safe=True로 통과.
          · npm/apt/yum …    : 이들은 안전 목록에 없어 'unknown'(ASK 영역)이 된다.
            핵심은 더 이상 'high'로 차단되지 않는다는 점이다(설치 차단 해제).
    - 완화는 오직 install 패턴에만 적용된다 → rm -rf /(critical)·curl/wget(high,
      네트워크)은 block 여부와 무관하게 항상 차단된다.

에어갭 우회 방지(2026-07-03 수정 반영):
    위험 패턴을 `pip[0-9.]*\\s+install`로 넓혀 'pip3 install …'·'pip3.11 install …'
    까지 잡는다. 따라서 block=True에서 pip3 경유 설치도 차단된다(아래 pip3 샘플로 고정).

왜 중요한가:
    개발 편의를 위한 설치 완화가 파괴적/네트워크 명령 차단까지 함께 풀어버리면
    치명적이다. 이 테스트가 "install 패턴만 완화, 나머지는 항상 차단" 경계를 못박는다.
"""

from __future__ import annotations

import pytest

from core.security.command_filter import CommandFilter

# "…패키지 설치" 사유로 차단되는 install 명령 — 네트워크 단어(curl/wget) 혼입이 없어
# 오직 install 패턴 때문에 high가 되는 것들만 골랐다(교란 변수 제거).
_INSTALL_COMMANDS_PURE = [
    "pip install requests",  # 사유: 런타임 패키지 설치
    "pip3 install requests",  # pip3 우회 방지(버전 접미사 포괄 패턴으로 차단)
    "npm install express",  # 사유: 런타임 패키지 설치
    "apt install vim",  # 사유: 시스템 패키지 설치
    "apt-get install build-essential",  # 사유: 시스템 패키지 설치
    "yum install httpd",  # 사유: 시스템 패키지 설치
    "brew install jq",  # 사유: 시스템 패키지 설치
]


class TestInstallBlockedWhenGatingOn:
    """block_package_install=True면 install 계열이 'high'(패키지 설치)로 차단된다."""

    @pytest.mark.parametrize("command", _INSTALL_COMMANDS_PURE)
    def test_install_command_denied_high_by_install_pattern(self, command: str) -> None:
        """설치 명령은 safe=False, severity='high', 사유에 '패키지 설치'가 담겨야 한다."""
        cf = CommandFilter(block_package_install=True)
        safe, severity, reason = cf.check_command(command)
        assert safe is False
        assert severity == "high"
        # 사유로 "install 패턴에 의한 차단"임을 못박는다(다른 high 패턴과 구분).
        assert "패키지 설치" in reason

    def test_default_gating_is_on(self) -> None:
        """인자 없는 기존 호출부(무회귀)는 설치를 차단해야 한다(기본 True)."""
        cf = CommandFilter()  # 기본값 = block_package_install=True
        safe, severity, _reason = cf.check_command("pip install requests")
        assert safe is False
        assert severity == "high"


class TestInstallGatingOffRemovesInstallVerdict:
    """block_package_install=False면 install 패턴의 'high' 차단이 사라진다(개발)."""

    @pytest.mark.parametrize("command", _INSTALL_COMMANDS_PURE)
    def test_install_no_longer_high(self, command: str) -> None:
        """개발 모드에서 install 명령은 더 이상 'high'로 차단되지 않아야 한다.

        (npm/apt/yum은 안전 목록에 없어 'unknown'(ASK)이 되지만, 그건 파괴적
        차단이 아니라 확인 영역이다 — 핵심은 install 'high' 차단의 해제다.)
        """
        cf = CommandFilter(block_package_install=False)
        _safe, severity, _reason = cf.check_command(command)
        assert severity != "high"

    def test_pip_install_passes_as_safe(self) -> None:
        """pip은 안전 목록에 있어, 게이팅을 끄면 pip install이 safe=True로 통과한다."""
        cf = CommandFilter(block_package_install=False)
        safe, severity, _reason = cf.check_command("pip install requests")
        assert safe is True
        assert severity == ""


class TestDestructiveAlwaysBlocked:
    """설치 게이팅과 무관하게 파괴적/네트워크 명령은 항상 차단돼야 한다."""

    @pytest.mark.parametrize("block", [True, False])
    def test_rm_rf_root_always_critical(self, block: bool) -> None:
        """rm -rf /는 block 여부와 무관하게 항상 'critical' 차단이다."""
        cf = CommandFilter(block_package_install=block)
        safe, severity, _reason = cf.check_command("rm -rf /")
        assert safe is False
        assert severity == "critical"

    @pytest.mark.parametrize("block", [True, False])
    def test_curl_always_high(self, block: bool) -> None:
        """curl(에어갭 네트워크)은 설치 완화와 무관하게 항상 'high'로 차단된다.

        완화는 오직 '패키지 설치' 사유 패턴만 대상이므로 다른 high 패턴은 남는다.
        """
        cf = CommandFilter(block_package_install=block)
        safe, severity, _reason = cf.check_command("curl http://192.168.10.39/x")
        assert safe is False
        assert severity == "high"
