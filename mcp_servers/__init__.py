"""
mcp_servers — 사내 시스템을 LAN MCP 서버로 노출하는 독립 서비스 패키지.

[이 패키지가 하는 일 — 한눈에 보기]
이 패키지는 Nexus 오케스트레이터(에이전트 두뇌) 쪽이 아니라, "사내 시스템을
MCP 프로토콜로 감싸서 서버처럼 제공하는 서버 측(server-side)" 코드다.
즉, DB/진단/지식검색/문서인제스트 같은 사내 기능을 외부(같은 LAN 안의 Nexus)
에서 도구처럼 호출할 수 있도록 HTTP + JSON-RPC 2.0 엔드포인트로 열어 준다.

Nexus 안에 이미 구현된 MCP "클라이언트"(core/tools/mcp/client.py)가 이 서버들에
JSON-RPC 2.0 요청을 보내고, 여기 서버들이 실제 사내 시스템(PostgreSQL 등)을
대신 조회해서 응답을 돌려주는 구조다. 클라이언트-서버가 짝을 이룬다고 보면 된다.

[의존성 방향 (아키텍처 규칙 P2) — 꼭 지켜야 함]
  mcp_servers/ → core/ (코어 코드를 가져다 쓰는 것)만 허용된다.
  반대로 core/ 가 mcp_servers/ 를 import 하는 것은 절대 금지다.
  (단방향 의존만 허용 — 순환 import를 막고 계층 구조를 지키기 위함.)

[패키지 구성 — 각 모듈의 역할]
  framework.py        — 모든 MCP 서버의 공통 뼈대.
                        McpServerTool(ABC: 각 도구가 상속하는 추상 베이스) +
                        create_mcp_app(JSON-RPC 디스패치 + 인증(API 키)을 붙인
                        FastAPI 앱을 만들어 주는 팩토리).
  run.py              — 실행 진입점. `python -m mcp_servers.run <name>` 형태로
                        특정 서버를 골라 uvicorn(ASGI 서버)으로 띄운다.
  db_server.py        — PostgreSQL 읽기 전용 조회를 제공(query). 쓰기는 없다.
  diag_server.py      — 인프라 진단용(예: reachability=도달성 확인,
                        rag_latency=RAG 검색 지연시간 측정).
  kowiki_server.py    — tb_knowledge 테이블 기반 지식 RAG 검색(search).
  docingest_server.py — 문서 인제스트 파이프라인(parse=파싱 / ingest=적재 /
                        search=검색).

[각 서버 모듈의 공통 계약]
  모든 서버 모듈은 `async def build_app(api_key) -> FastAPI` 함수를 노출한다.
  run.py 가 이 함수를 호출해 인증이 적용된 FastAPI 앱을 얻어 서비스한다.

[이 __init__.py 파일 자체의 역할]
  패키지의 "대문" 역할만 한다. 가장 많이 쓰는 공개 심볼
  (McpServerTool, create_mcp_app)을 framework 에서 끌어와, 바깥에서
  `from mcp_servers import McpServerTool` 처럼 짧게 쓸 수 있게 재노출한다.

작성자: 이현수 / 작성일: 2026-07-05
"""

# from __future__ import annotations:
#   타입 힌트를 문자열로 지연 평가(lazy)하게 만든다. 정의되기 전의 타입을
#   힌트로 써도 되고, 런타임 순환 참조 부담이 줄어든다. (기능성 import이므로
#   반드시 파일 최상단, docstring 바로 아래에 위치해야 한다.)
from __future__ import annotations

# 패키지의 핵심 공개 API를 framework 모듈에서 가져온다.
#   - McpServerTool  : 각 MCP 서버 도구가 상속하는 추상 베이스 클래스(ABC)
#   - create_mcp_app : JSON-RPC 디스패치 + 인증이 붙은 FastAPI 앱 팩토리
# 여기서 한 번 끌어와 아래 __all__ 로 재노출함으로써, 사용자는 내부 모듈
# 경로(mcp_servers.framework)를 몰라도 패키지 최상단에서 바로 임포트할 수 있다.
from mcp_servers.framework import McpServerTool, create_mcp_app

# __all__ : `from mcp_servers import *` 로 무엇이 공개될지 명시한다.
# 공개 표면(public surface)을 이 두 심볼로 좁혀, 내부 구현이 실수로
# 밖으로 새어 나가지 않도록 한다(캡슐화).
__all__ = ["McpServerTool", "create_mcp_app"]
