"""보안 시스템 패키지 (core.security).

Nexus 오케스트레이터(Machine A)가 안전하게 동작하도록 지켜 주는 보안 유틸을
한곳에 모아 둔 패키지다. 크게 네 가지 축을 담당한다.

  - 경로 보호(path guard): 파일 도구가 프로젝트 밖이나 위험 경로에 접근하지
    못하도록 막는다.
  - 명령어 필터(command filter): Bash 도구가 위험한 명령을 실행하지 못하게
    사전에 걸러 낸다.
  - 감사 로그(audit): 보안 관련 사건을 JSONL로 기록해 추적할 수 있게 한다.
  - 네트워크 가드(network guard): 에어갭(폐쇄망) 원칙에 따라 외부(공인망)로
    나가는 접속을 차단하고, LAN 대역만 허용한다.

이 __init__.py 자체는 무거운 로직을 두지 않고, 다른 모듈에서 자주 쓰는 헬퍼를
패키지 최상단으로 끌어올려(재export) 짧은 import 경로를 제공하는 역할만 한다.
즉 `from core.security import is_lan_hostname` 처럼 바로 가져다 쓸 수 있다.

노출(공개) API:
  - is_lan_hostname: 주어진 호스트명이 LAN(사설망) 대역인지 판정하는 함수.
    실제 구현은 core.security.network_guard 에 있다.

작성자: 이현수 / 작성일: 2026-07-05
"""

# network_guard 모듈에 정의된 LAN/에어갭 판정 헬퍼를 패키지 레벨에서 다시 노출한다.
# 이렇게 재export 해 두면 호출부가 내부 파일 구조(network_guard)를 몰라도 되고,
# `core.security.is_lan_hostname` 라는 안정적인 짧은 경로로 접근할 수 있다.
from core.security.network_guard import is_lan_hostname

# `from core.security import *` 시 공개할 이름을 명시한다.
# 외부에 정식으로 노출하는 것은 is_lan_hostname 하나뿐임을 분명히 밝혀 둔다.
__all__ = ["is_lan_hostname"]
