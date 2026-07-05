"""
core.tools.mcp — v7.2 MCP(Model Context Protocol) 클라이언트 어댑터 패키지.

[이 패키지가 하는 일]
Nexus 외부(LAN 내부)에 떠 있는 MCP 서버들이 제공하는 "원격 도구"를
Nexus 자체 도구 풀에 자연스럽게 흡수시키는 역할을 한다. 즉, 다른 프로세스가
노출한 기능을 마치 Nexus에 내장된 도구인 것처럼 모델이 호출할 수 있게 해준다.

[핵심 설계 원칙 — 왜 이렇게 나눴나]
MCP 통신은 JSON-RPC(요청/응답 규약)로 이뤄지는데, 이 저수준 프로토콜은
반드시 이 패키지 내부(McpClient/McpToolAdapter)에만 갇혀 있어야 한다.
바깥의 query_loop(에이전트 턴 루프)와 executor(도구 실행기)는 원격 도구를
'그냥 하나의 BaseTool'로만 바라본다. 이렇게 해야 아키텍처 규칙 P3
(표준 내부 계약: 모든 도구는 OpenAI tool_calls 형식의 BaseTool로 통일)이
깨지지 않고, MCP 특유의 프로토콜 세부사항이 코어로 새어나가지 않는다.

[공개 심볼 — 이 패키지가 바깥에 노출하는 3가지]
  - McpClient            : LAN HTTP/SSE 위에서 JSON-RPC를 주고받는 저수준
                           클라이언트. 실제 원격 통신을 담당한다.
  - McpToolAdapter       : 원격 도구 1개를 감싸 Nexus의 BaseTool로 변환하는
                           래퍼. 모델은 이 어댑터를 일반 도구처럼 호출한다.
  - McpConnectionManager : MCP 서버 발견(discovery)·도구 등록·연결 정리를
                           총괄하는 관리자.

이 파일 자체는 패키지 진입점(__init__)으로, 위 3개 심볼을 하위 모듈에서
끌어와 한곳에 모아 재노출하는 얇은 파사드(facade) 역할만 한다.

작성자: 이현수 / 작성일: 2026-07-05
"""

# 파이썬의 지연 평가 어노테이션 활성화. 타입 힌트를 문자열로 취급해
# 순환 import를 피하고, 하위 모듈 간 타입 참조를 안전하게 해준다.
# (기능성 선언 — 반드시 파일 상단, docstring 바로 아래 위치해야 함)
from __future__ import annotations

# 하위 모듈에서 3대 공개 클래스를 끌어와, 사용하는 쪽이
# `from core.tools.mcp import McpClient`처럼 짧게 import할 수 있게 모아둔다.
from core.tools.mcp.adapter import McpToolAdapter  # 원격 도구 → BaseTool 래퍼
from core.tools.mcp.client import McpClient  # JSON-RPC 저수준 클라이언트
from core.tools.mcp.connection_manager import McpConnectionManager  # 발견·등록·정리

# 패키지의 공식 공개 API 목록. `from core.tools.mcp import *` 시 이 3개만
# 노출되며, 문서화 도구에도 "이것이 이 패키지의 안정적인 계약"임을 알린다.
__all__ = [
    "McpClient",
    "McpToolAdapter",
    "McpConnectionManager",
]
