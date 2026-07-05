"""
네트워크 보안 가드 — LAN(사설망) 주소 판정 공용 헬퍼 모듈.

■ 이 파일이 하는 일
  주어진 호스트 이름(또는 IP 문자열)이 "사설망(LAN) 대역인지"를 판정한다.
  즉 이 주소로 나가는 연결이 폐쇄망 내부로만 향하는지, 아니면 인터넷
  같은 외부(공인망)로 나가는지를 한 줄로 구분해 주는 역할을 한다.

■ 왜 필요한가 (에어갭 원칙)
  Nexus는 에어갭(폐쇄망) 환경에서 동작해야 하므로, 코드가 외부 네트워크로
  나가는 일을 절대 허용하면 안 된다. "이 호스트가 LAN 대역인가?"라는 판정은
  GPU 서버 URL 검증, MCP 서버 연결 차단 등 여러 보안 지점에서 반복적으로
  필요하다. 그 판정 로직을 이 한 곳에 모아 두어, 각 호출부가 같은 규칙으로
  일관되게 외부 연결을 걸러낼 수 있게 한다.

■ 공개 API
  - is_lan_hostname(hostname) -> bool : 호스트가 LAN 대역이면 True.

■ 왜 core/security/ 아래에 두는가 (의존성 방향)
  에어갭/LAN 판정은 본질적으로 보안 관심사다. 기존에는 core/config.py의
  private 함수(_is_lan_hostname)로 있었으나, core/tools/mcp/client.py가
  패키지 경계를 넘어 그 private 이름을 직접 import하는 구조적 결함이 있었다.
  이를 public 보안 헬퍼로 승격해 의존성 방향(architecture.md P2)을 명확히 했다:
    - config → security      (설정 검증이 보안 헬퍼 사용)  ✅ 허용
    - tools/mcp → security   (도구가 보안 헬퍼 사용)        ✅ 허용
  두 경로 모두 "상위 → 하위(공용)" 방향이라 금지된 역방향 import가 아니다.

작성자: 이현수 / 작성일: 2026-07-05
"""

from __future__ import annotations


# ─────────────────────────────────────────────
# LAN 주소 판정 헬퍼 (에어갭 / MCP 공용 — public)
# ─────────────────────────────────────────────
def is_lan_hostname(hostname: str) -> bool:
    """
    주어진 hostname이 LAN(사설망) 대역에 속하는지 판정한다.

    이 함수는 "이 주소로 나가는 연결이 폐쇄망 내부인가?"를 판단하는
    핵심 게이트다. True면 내부 LAN이라 연결을 허용해도 되고, False면
    외부(공인망)일 수 있으므로 호출부에서 연결을 차단하는 식으로 쓴다.

    허용(=LAN으로 인정)하는 대역:
      - localhost / 127.x        (루프백 — 자기 자신)
      - 10.x                     (사설 A 클래스)
      - 172.16.x ~ 172.31.x      (사설 B 클래스 — 아래 주의사항 참고)
      - 192.168.x                (사설 C 클래스)

    ※ 172 대역을 왜 특별 취급하나:
      172.x "전체"가 사설이 아니라 172.16 ~ 172.31 구간만 사설이다.
      단순히 "172." 접두사만 보면 172.217.x(구글 등 공인 IP)까지 LAN으로
      잘못 인정해 외부 연결이 뚫린다. 그래서 두 번째 옥텟(마디)이 반드시
      16~31 범위 안에 드는지까지 확인한다.

    Args:
        hostname: 판정할 호스트 문자열. 보통 urlparse(...).hostname 처럼
            URL에서 뽑아낸 호스트명이나 IP 문자열을 넘긴다.

    Returns:
        LAN 대역이면 True, 그 외(공인 IP·도메인·빈 문자열)면 False.
        판정이 애매하거나 입력이 비어 있으면 안전하게 False를 반환한다.
    """
    if not hostname:
        # 빈 호스트(None·빈 문자열)는 판정 자체가 불가능하다.
        # 보안 기본값(fail-closed)에 따라 "LAN 아님(False)"으로 처리해
        # 애매한 경우 연결을 막는 쪽으로 기운다.
        return False
    # 대소문자 구분 없이 비교하기 위해 소문자로 통일한다("Localhost" 등 대비).
    host = hostname.lower()

    # 루프백 / localhost — 자기 자신을 가리키는 주소이므로 항상 LAN으로 인정.
    if host == "localhost" or host.startswith("127."):
        return True
    # 사설 A(10.x) / 사설 C(192.168.x) — 두 번째 옥텟 검사가 필요 없어
    # 접두사만으로 곧바로 LAN으로 인정한다.
    if host.startswith("10.") or host.startswith("192.168."):
        return True
    # 사설 B — 172.16 ~ 172.31 구간만 허용(두 번째 옥텟 범위를 반드시 검증).
    if host.startswith("172."):
        # "172.20.10.5" → ["172", "20", "10", "5"] 처럼 마디 단위로 분리한다.
        parts = host.split(".")
        # 두 번째 마디가 존재하고 숫자일 때만 범위 비교를 시도한다
        # (예: "172.foo" 같은 비정상 입력에서 int 변환 오류를 피하기 위함).
        if len(parts) >= 2 and parts[1].isdigit():
            second_octet = int(parts[1])
            # 16~31 안에 들어야만 진짜 사설 B 대역 → LAN으로 인정.
            if 16 <= second_octet <= 31:
                return True
    # 위 어떤 사설 대역에도 해당하지 않으면 외부(공인) 주소로 간주 → False.
    return False
