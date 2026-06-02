"""보안 시스템 패키지 — 경로 보호, 명령어 필터, 감사 로그, 네트워크 가드."""

# LAN/에어갭 판정 헬퍼를 패키지 레벨에서 재export (편의 import)
from core.security.network_guard import is_lan_hostname

__all__ = ["is_lan_hostname"]
