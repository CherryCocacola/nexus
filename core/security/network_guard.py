"""
네트워크 보안 가드 — LAN(사설망) 주소 판정.

에어갭(폐쇄망) 원칙을 코드 레벨에서 강제하기 위한 공용 헬퍼다.
"이 호스트가 LAN 대역인가?"라는 판정은 GPU 서버 URL 검증, MCP 서버
연결 차단 등 여러 보안 관심사에서 반복적으로 필요하다.

왜 core/security/에 두는가:
  에어갭/LAN 판정은 본질적으로 보안 관심사다. 기존에는 core/config.py의
  private 함수(_is_lan_hostname)로 있었으나, core/tools/mcp/client.py가
  패키지 경계를 넘어 그 private 이름을 import하는 구조적 결함이 있었다.
  public 보안 헬퍼로 승격해 의존성 방향(P2)을 명확히 한다:
    - config → security      (설정 검증이 보안 헬퍼 사용)  ✅ 허용
    - tools/mcp → security    (도구가 보안 헬퍼 사용)        ✅ 허용
  두 경로 모두 "상위 → 하위(공용)" 방향이라 역방향 import가 아니다.
"""

from __future__ import annotations


# ─────────────────────────────────────────────
# LAN 주소 판정 헬퍼 (에어갭 / MCP 공용 — public)
# ─────────────────────────────────────────────
def is_lan_hostname(hostname: str) -> bool:
    """
    hostname이 LAN(사설망) 대역인지 판정한다.

    허용 대역:
      - localhost / 127.x (루프백)
      - 10.x (사설 A)
      - 172.16.x ~ 172.31.x (사설 B) — 172.0~15, 172.32~255는 공인이므로 제외
      - 192.168.x (사설 C)

    왜 172는 별도 처리하나: 172.x 전체가 사설이 아니라 172.16~172.31만
    사설이다. 단순 "172." prefix 검사는 172.217.x(구글 등 공인 IP)를
    LAN으로 오인하므로, 두 번째 옥텟을 반드시 16~31 범위로 검증한다.

    Args:
        hostname: urlparse(...).hostname 등에서 얻은 호스트 문자열.

    Returns:
        LAN 대역이면 True, 그 외(공인 IP/도메인/빈 문자열)면 False.
    """
    if not hostname:
        # 빈 호스트는 판정 불가 → fail-closed로 False
        return False
    host = hostname.lower()

    # 루프백 / localhost
    if host == "localhost" or host.startswith("127."):
        return True
    # 사설 A / 사설 C
    if host.startswith("10.") or host.startswith("192.168."):
        return True
    # 사설 B — 172.16~172.31만 허용 (두 번째 옥텟 범위 검증)
    if host.startswith("172."):
        parts = host.split(".")
        if len(parts) >= 2 and parts[1].isdigit():
            second_octet = int(parts[1])
            if 16 <= second_octet <= 31:
                return True
    return False
