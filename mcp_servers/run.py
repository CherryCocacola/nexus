"""
mcp_servers 실행 진입점(entrypoint) — LAN 전용 MCP 서버 1개를 uvicorn 으로 기동한다.

이 파일이 하는 일(개요):
  Nexus 는 여러 개의 MCP(도구 제공) 서버를 각각 독립 프로세스로 띄운다. 이 스크립트
  는 그중 "한 개" 를 골라 실행하는 얇은 런처(launcher)다. 커맨드라인에서 서버 이름을
  받아 → 해당 서버 모듈의 FastAPI 앱을 만들고 → uvicorn HTTP 서버로 서빙한다.
  서버별 로직 자체는 각 *_server.py 모듈에 들어 있고, 여기서는 조립·기동만 담당한다.

사용법:
  python -m mcp_servers.run <name> [--host 0.0.0.0] [--port <p>] [--api-key ...]

  <name> 은 다음 중 하나(괄호는 지정하지 않았을 때 쓰는 기본 포트):
    db        — PostgreSQL read-only 조회 서버  (기본 포트 8810)
    diag      — 인프라 진단 서버                (기본 포트 8811)
    kowiki    — tb_knowledge RAG 검색 서버      (기본 포트 8813)
    docingest — 문서 인제스트(적재) 서버        (기본 포트 8814)

주요 구성 요소:
  - _SERVERS : 서버 이름 → 기본 포트 매핑 테이블(선택 가능한 이름의 원천).
  - _build() : 이름에 맞는 서버 모듈을 지연 import 하고 build_app() 을 호출하는 코루틴.
  - main()   : CLI 인자 파싱 → 앱 생성 → uvicorn 기동까지의 전체 흐름.

설계 메모(왜 이런 구조인가):
  각 서버 모듈은 async def build_app(api_key) -> FastAPI 를 노출한다. 이 진입점은
  선택된 서버의 build_app() 을 asyncio.run 으로 실행해 FastAPI 인스턴스를 얻은 뒤
  uvicorn 으로 띄운다. build_app 이 "동기 팩토리" 가 아니라 "코루틴" 인 이유는,
  DB 커넥션 풀처럼 만들 때 await 가 필요한 비동기 자원을 초기화해야 하기 때문이다.
  또한 서버 import 를 _build() 안에서 지연(lazy)으로 하는 이유는, 실제로 띄우는
  서버의 의존성만 로드하기 위함이다(예: db 서버만 띄울 때 docingest 의 문서 파서
  라이브러리까지 import 되지 않도록).

에어갭(폐쇄망) 관련:
  기본 host 는 0.0.0.0 으로 LAN 인터페이스에 바인드한다. 각 서버의 도구들이 외부가
  아닌 LAN 주소만 사용하도록 구현되어 있어, 외부로 나가는 발신 호출은 구조적으로
  차단된다. 기본 api_key 는 placeholder(예: "local-key") 이므로, 운영 환경에서는
  반드시 --api-key 인자 또는 별도 주입 방식으로 실제 키를 넣어 사용한다.

작성자: 이현수 / 작성일: 2026-07-05
"""

from __future__ import annotations

import argparse
import asyncio
import logging

import uvicorn
from fastapi import FastAPI

# 이 런처 전용 로거. 이름 규칙은 "nexus.{모듈경로}" 를 따른다(프로젝트 규칙 P7).
logger = logging.getLogger("nexus.mcp_servers.run")

# 서버 이름 → 기본 포트 매핑 테이블.
# 이 딕셔너리의 키(key) 집합이 곧 "선택 가능한 서버 이름" 의 원천이며, main() 의
# argparse choices 로도 그대로 재사용된다. 새 서버를 추가할 때는 여기에 이름/포트를
# 등록하고, _build() 의 분기에 해당 모듈 import 를 한 줄 추가하면 된다.
_SERVERS: dict[str, int] = {
    "db": 8810,
    "diag": 8811,
    "kowiki": 8813,
    "docingest": 8814,
}


async def _build(name: str, api_key: str) -> FastAPI:
    """선택된 서버 모듈을 지연 import 하고 build_app(api_key) 을 호출해 FastAPI 앱을 만든다.

    매개변수:
      name    — 기동할 서버 이름(_SERVERS 의 키 중 하나).
      api_key — 각 서버가 Bearer 인증에 사용할 키. build_app 으로 그대로 전달한다.

    반환:
      해당 서버의 build_app() 이 초기화까지 끝낸 FastAPI 인스턴스.

    왜 함수 안에서 import 하나:
      모듈 최상단이 아니라 이 분기 안에서 import 하면, 실제 선택된 서버의 의존성만
      로드된다(불필요한 라이브러리 로딩·초기화 비용 회피). 또 build_app 이 코루틴
      이라 await 로 호출해야 DB 풀 같은 비동기 자원 준비가 끝난 앱을 받을 수 있다.
    """
    if name == "db":
        from mcp_servers.db_server import build_app
    elif name == "diag":
        from mcp_servers.diag_server import build_app
    elif name == "kowiki":
        from mcp_servers.kowiki_server import build_app
    elif name == "docingest":
        from mcp_servers.docingest_server import build_app
    else:
        # 정상 경로에서는 argparse 의 choices 가 이미 잘못된 이름을 걸러낸다.
        # 그래도 _build() 를 다른 곳에서 직접 호출하는 경우를 대비한 방어 코드다.
        raise ValueError(f"알 수 없는 서버 이름: {name!r}")
    return await build_app(api_key=api_key)


def main() -> None:
    """CLI 인자를 파싱하고, 선택된 MCP 서버를 uvicorn 으로 기동하는 진입 함수.

    전체 흐름:
      1) argparse 로 서버 이름·호스트·포트·API 키를 읽는다.
      2) 로깅을 INFO 레벨로 초기화한다.
      3) 포트가 명시되지 않았으면 _SERVERS 의 서버별 기본 포트를 쓴다.
      4) _build() 를 asyncio.run 으로 실행해 FastAPI 앱을 만든다.
      5) uvicorn.run 으로 HTTP 서버를 띄운다(이 호출은 프로세스가 살아있는 동안 블로킹).
    """
    parser = argparse.ArgumentParser(
        prog="mcp_servers.run",
        description="LAN MCP 서버 기동(db/diag/kowiki/docingest)",
    )
    # 위치 인자: 어떤 서버를 띄울지. choices 를 _SERVERS 키로 고정해 오타를 사전 차단.
    parser.add_argument(
        "name",
        choices=sorted(_SERVERS.keys()),
        help="기동할 MCP 서버 이름",
    )
    # 바인드 호스트. 기본 0.0.0.0 은 LAN(사내 폐쇄망) 인터페이스에 바인드한다는 뜻.
    # noqa: S104 는 "0.0.0.0 바인드" 를 지적하는 보안 린트 경고를 의도적으로 끄는 것.
    parser.add_argument(
        "--host",
        default="0.0.0.0",  # noqa: S104 — LAN 바인드(에어갭 내부망). 외부 노출은 방화벽으로 통제.
        help="바인드 호스트(기본 0.0.0.0 — LAN)",
    )
    # 바인드 포트. None 으로 두면 아래에서 서버별 기본 포트로 대체한다.
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="바인드 포트(기본: 서버별 기본 포트)",
    )
    # Bearer 인증 키. 기본값은 개발용 placeholder 이며 운영 시 반드시 실제 키로 교체.
    parser.add_argument(
        "--api-key",
        default="local-key",
        help="Bearer 인증 키(기본 placeholder, 운영 시 명시 주입)",
    )
    args = parser.parse_args()

    # 표준 로깅 설정. 이후 logger.info 로 남기는 기동 안내가 콘솔에 보이게 한다.
    logging.basicConfig(level=logging.INFO)

    # 포트 결정: 사용자가 --port 를 줬으면 그 값을, 아니면 _SERVERS 의 기본 포트를 쓴다.
    port = args.port if args.port is not None else _SERVERS[args.name]

    # build_app 은 코루틴이므로 asyncio.run 으로 실행한다. DB 풀 등 비동기 자원을
    # 미리 초기화한 "완성된" FastAPI 앱을 여기서 받아 uvicorn 에 넘길 수 있다.
    app = asyncio.run(_build(args.name, args.api_key))

    # 어떤 서버가 어느 주소로 떴는지, 어떤 엔드포인트를 쓰는지 운영자에게 안내한다.
    # ('/' 는 JSON-RPC 요청 수신, '/health' 는 헬스체크용 GET 엔드포인트)
    logger.info(
        "MCP 서버 '%s' 기동: http://%s:%d (POST '/' JSON-RPC, GET '/health')",
        args.name,
        args.host,
        port,
    )
    # uvicorn 서버 시작. 이 호출은 서버가 종료될 때까지 반환하지 않는(블로킹) 호출이다.
    uvicorn.run(app, host=args.host, port=port)


if __name__ == "__main__":
    # 모듈을 `python -m mcp_servers.run ...` 형태로 직접 실행했을 때만 main() 을 호출.
    main()
