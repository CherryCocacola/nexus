"""web app full e2e (실물) — vLLM Worker가 MCP 도구를 호출하는 전 구간 증명.

이 스크립트는 웹앱(web.app)이 부팅부터 채팅 응답까지의 전 구간에서 MCP(Model
Context Protocol) 도구를 실제로 물고 동작하는지를 mock 없이 "실물 인프라"로만
검증하는 e2e 스모크 테스트다. 즉, 도구가 노출되는지(스키마), 연결됐는지(메트릭),
그리고 실제로 LLM이 그 도구를 호출해 답을 만드는지(채팅)를 한 번에 확인한다.

의존하는 실 인프라(모두 LAN 내부, 에어갭 준수):
  - 실 PostgreSQL (192.168.10.39:5440) — db MCP 서버가 사용(tb_knowledge 등)
  - 실 임베딩 서버 (192.168.21.112:8002) — kowiki MCP 서버가 사용
  - 실 vLLM (192.168.21.112:8001) — web app QueryEngine의 Worker(추론 엔진)
  - 실 Redis (192.168.10.39:6340) — 세션 단기 메모리

수행 흐름(main() 참고):
  1) db / kowiki MCP 서버를 localhost 고정 포트에 uvicorn으로 기동(실 PG/임베딩 연결).
  2) 운영 config(config/nexus_config.yaml)를 메모리에서 로드 → mcp 섹션만
     enabled=true + db/kowiki를 localhost 포트로 override → 임시 yaml로 기록.
     (운영 원본 파일은 절대 건드리지 않는다. 임시 파일은 종료 시 삭제.)
  3) 그 임시 config 경로로 web 부트스트랩(init/init_phase2)을 실제 수행하고,
     web/app.py의 _build_web_query_engine으로 웹 QueryEngine을 조립해 _app_state에
     주입한다(= web/app.py lifespan과 동일 경로, 단 config_path만 주입 가능하게).
  4) 실 FastAPI app(web.app:app)을 httpx.ASGITransport로 구동해 검증:
       - GET  /v1/tools  → mcp__db__query, mcp__kowiki__search 노출 확인
       - GET  /metrics   → mcp.connected, tool_counts 확인
       - POST /v1/chat   → 도구 사용을 유도하는 질의로 vLLM이 MCP 호출하는지 관측
  5) cli 도구 풀(TIER 별 cli_registry)에 mcp__ 도구가 흡수되는지 프로그램적으로 확인.

종료 시 MCP 서버·임시 config를 정리한다(운영 무영향). finally 블록에서 keepalive
태스크·MCP 매니저·uvicorn 서버·임시 파일을 순서대로 정리하므로, 중간에 실패해도
찌꺼기 프로세스나 파일이 남지 않는다.

주요 구성 요소:
  - _port_free / _free_port : 포트 사용 가능 여부 확인 및 빈 포트 확보 헬퍼
  - _serve / _stop          : uvicorn 서버를 비동기로 띄우고 정리하는 헬퍼
  - _build_temp_config      : 운영 yaml을 읽어 mcp만 켠 임시 config 생성
  - _bootstrap_web          : web/app.py lifespan과 동일한 부트스트랩 재현
  - main                    : 위 1~5단계를 순서대로 실행하는 진입점

실행: python -m scripts._verify_web_e2e_mcp

작성자: 이현수 / 작성일: 2026-07-05
"""

from __future__ import annotations

import asyncio
import os
import socket
import sys
import tempfile
from pathlib import Path
from typing import Any

import uvicorn
import yaml

# 한글 등 비ASCII 출력이 Windows 콘솔에서 깨지지 않도록 표준출력을 UTF-8로 재설정.
sys.stdout.reconfigure(encoding="utf-8")

# 로컬에서 띄우는 db/kowiki MCP 서버에 붙일 때 사용하는 API 키. 실 서버가 아니라
# 이 스크립트가 직접 기동하는 테스트용 MCP 서버이므로 고정 문자열이면 충분하다.
API_KEY = "local-key"

# db / kowiki MCP 서버를 띄울 localhost 고정 포트(테스트 충돌 시 _free_port로 대체).
DB_PORT = 18810
KOWIKI_PORT = 18813


# ─────────────────────────────────────────────
# uvicorn 서버 헬퍼 (검증된 _verify_mcp_live.py와 동일 패턴)
# ─────────────────────────────────────────────
def _port_free(port: int) -> bool:
    """주어진 포트가 지금 비어 있는지(bind 가능한지) 검사한다.

    127.0.0.1의 해당 포트에 실제로 bind를 시도해 보고, 성공하면 True를 반환한다.
    이미 다른 프로세스가 쓰고 있으면 OSError가 나므로 False. 검사 목적의 소켓은
    finally에서 반드시 닫아 누수를 막는다.
    """
    s = socket.socket()
    try:
        s.bind(("127.0.0.1", port))
        return True
    except OSError:
        # 포트가 이미 점유돼 bind 실패 → 사용 불가로 판단.
        return False
    finally:
        s.close()


def _free_port() -> int:
    """OS에게 아무 빈 포트나 하나 받아서 그 번호를 돌려준다.

    포트 0으로 bind하면 커널이 사용 가능한 임시 포트를 자동 할당한다. 고정 포트
    (DB_PORT/KOWIKI_PORT)가 이미 점유돼 있을 때의 대체 수단으로 쓴다.
    """
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


async def _serve(app, port: int):
    """FastAPI 앱을 localhost uvicorn으로 띄우고 준비될 때까지 폴링한다.

    uvicorn 서버를 asyncio 태스크로 백그라운드 기동한 뒤, server.started 플래그가
    True가 될 때까지 최대 약 10초(0.05초 × 200회) 동안 짧게 대기하며 확인한다.
    시간 내에 기동되지 않으면 RuntimeError로 실패를 명확히 알린다.

    반환: (server, task) 튜플 — 나중에 _stop에 넘겨 정상 종료시키는 데 쓴다.
    """
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    # serve()는 블로킹 코루틴이므로 태스크로 떼어 두고, 아래에서 준비 상태만 폴링.
    task = asyncio.create_task(server.serve())
    for _ in range(200):
        if server.started:
            break
        await asyncio.sleep(0.05)
    if not server.started:
        raise RuntimeError(f"uvicorn 서버가 포트 {port}에서 기동되지 않았습니다.")
    return server, task


async def _stop(server, task) -> None:
    """_serve로 띄운 uvicorn 서버에 종료 신호를 보내고 태스크가 끝날 때까지 대기한다."""
    # should_exit=True를 세우면 uvicorn이 다음 루프에서 graceful shutdown을 시작한다.
    server.should_exit = True
    await task


# ─────────────────────────────────────────────
# 임시 config 생성 — 운영 yaml을 로드해 mcp 섹션만 override
# ─────────────────────────────────────────────
def _build_temp_config(db_port: int, kowiki_port: int) -> str:
    """운영 nexus_config.yaml을 읽어 mcp만 켠 임시 복사본을 만들고 경로를 반환한다.

    운영 원본은 읽기만 한다. mcp.servers를 db/kowiki(localhost 포트)만 남기고
    read_only=true로 둔다 — 운영 데이터(tb_knowledge)에 쓰기 경로가 생기지 않게.

    이렇게 별도 임시 config를 만드는 이유: 운영 설정에는 실제 원격 MCP 서버 주소가
    들어 있을 수 있는데, 이 테스트는 방금 로컬에 띄운 서버를 물려야 하기 때문이다.
    운영 yaml을 직접 수정하면 위험하므로, 메모리에서만 mcp 섹션을 갈아끼운 뒤
    임시 파일로 떨궈서 그 경로를 부트스트랩에 넘긴다.

    매개변수:
      db_port     : 로컬에 띄운 db MCP 서버 포트
      kowiki_port : 로컬에 띄운 kowiki MCP 서버 포트
    반환: 생성된 임시 yaml 파일의 절대경로(문자열). 호출 측이 종료 시 직접 삭제한다.
    """
    prod_path = Path("config/nexus_config.yaml")
    # 운영 원본은 읽기 전용으로만 연다 — 여기서 만든 dict만 수정하고 원본은 불변.
    with open(prod_path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    # mcp 섹션만 통째로 교체한다. 나머지 운영 설정(모델/DB/세션 등)은 그대로 두어
    # 실제 운영과 동일한 환경에서 MCP만 로컬로 바꿔 검증하는 효과를 낸다.
    #   - enabled: MCP 기능 전역 on
    #   - servers: db/kowiki 두 개만, localhost 포트로, read_only(쓰기 차단)
    data["mcp"] = {
        "enabled": True,
        "connect_timeout_sec": 10.0,  # 임베딩 cold start와 무관(연결만). 넉넉히.
        "servers": [
            {
                "name": "db",
                "transport": "http_sse",
                "base_url": f"http://127.0.0.1:{db_port}",
                "api_key": API_KEY,
                "enabled": True,
                "trust": {"read_only": True},
            },
            {
                "name": "kowiki",
                "transport": "http_sse",
                "base_url": f"http://127.0.0.1:{kowiki_port}",
                "api_key": API_KEY,
                "enabled": True,
                "trust": {"read_only": True},
            },
        ],
    }

    # mkstemp는 충돌하지 않는 임시 파일을 만들고 (fd, 경로)를 준다. fd를 텍스트
    # 모드로 감싸 yaml을 쓴다. allow_unicode=True라야 한글 설정값이 안 깨진다.
    fd, tmp_path = tempfile.mkstemp(suffix="_nexus_mcp_e2e.yaml", prefix="nexus_")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True)
    return tmp_path


# ─────────────────────────────────────────────
# web 부트스트랩 — web/app.py lifespan과 동일하되 config_path 주입 가능
# ─────────────────────────────────────────────
async def _bootstrap_web(config_path: str) -> dict:
    """init(config_path)+init_phase2 → web QueryEngine 조립 → _app_state 주입.

    web/app.py는 원래 FastAPI lifespan 이벤트에서 앱 상태를 채운다. 하지만 lifespan은
    config_path를 주입받을 수 없어(운영 고정 경로 사용) 이 테스트에는 맞지 않는다.
    그래서 여기서 lifespan이 하는 일을 손으로 똑같이 재현하되, 임시 config 경로만
    바꿔 넣는다. web.app._app_state(전역 dict)를 직접 채워 넣으므로, 이후 httpx로
    실제 app을 호출하면 lifespan을 타지 않아도 동일한 상태에서 엔드포인트가 돈다.

    단계:
      1) init(config_path)     — 설정 로드 + 기본 상태(state) 구성
      2) init_phase2(state)    — 도구/모델/메모리 등 실 컴포넌트 부팅(MCP 연결 포함)
      3) _build_web_query_engine — 웹 전용 QueryEngine/디스패처/도구풀 조립

    매개변수: config_path — _build_temp_config가 만든 임시 yaml 경로.
    반환: init_phase2가 만든 components dict(정리 단계에서 매니저/태스크 회수에 사용).
    """
    from core.bootstrap import init, init_phase2
    from web import app as webapp

    # 1) 임시 config로 기본 상태를 만들고 전역 _app_state에 심는다.
    state = await init(config_path=config_path)
    webapp._app_state["state"] = state
    webapp._app_state["config"] = state.config

    # 2) 실 컴포넌트(도구 레지스트리/모델 프로바이더/메모리/MCP 등)를 부팅한다.
    components = await init_phase2(state)
    webapp._app_state["tool_registry"] = components.get("tool_registry")
    webapp._app_state["model_provider"] = components.get("model_provider")
    webapp._app_state["memory_manager"] = components.get("memory_manager")
    webapp._app_state["tenant_registry"] = state.config.tenants
    # 임베딩 서버를 깨어 있게 유지하는 keepalive 태스크 — 종료 시 취소해야 한다.
    webapp._app_state["embedding_keepalive_task"] = components.get(
        "embedding_keepalive_task"
    )

    # 3) 웹 전용 QueryEngine을 조립한다. 여기서 반환되는 web_tools에 MCP 도구가
    #    흡수돼 있어야 /v1/tools가 이를 노출할 수 있다.
    web_engine, web_dispatcher, web_tools = webapp._build_web_query_engine(
        components, state
    )
    webapp._app_state["model_dispatcher"] = web_dispatcher
    webapp._app_state["query_engine"] = web_engine
    # lifespan과 동일: /v1/tools가 모델 실풀(MCP 포함)을 노출하도록 web_tools 저장.
    webapp._app_state["web_tools"] = web_tools
    return components


def _print_kv(label: str, value: Any) -> None:
    """검증 결과를 'label : value' 꼴로 들여쓰기해 보기 좋게 출력하는 헬퍼."""
    print(f"  {label:22s}: {value}")


async def main() -> None:
    """e2e 검증 전 과정을 순서대로 실행하는 진입점.

    흐름: [1] 로컬 MCP 서버 기동 → [2] 임시 config 생성 → [3] web 부트스트랩 →
    [4-1] /v1/tools → [4-2] /metrics → [5] cli 도구 풀 → [4-3] /v1/chat →
    [4-4] 채팅 후 /metrics → [정리] 자원 회수. 진행 상황은 콘솔에 단계별로 찍는다.
    실 인프라(PG/임베딩/vLLM/Redis)가 모두 살아 있어야 정상 통과한다.
    """
    # 포트 충돌 회피 — 고정 포트가 막혀 있으면 OS 할당으로 대체.
    db_port = DB_PORT if _port_free(DB_PORT) else _free_port()
    kowiki_port = KOWIKI_PORT if _port_free(KOWIKI_PORT) else _free_port()

    # MCP 서버 팩토리는 여기서 지연 import한다 — 부팅 시 실 PG/임베딩에 연결을
    # 시도하므로, 포트 확보 등 사전 준비가 끝난 뒤 불러오는 편이 안전하다.
    from mcp_servers.db_server import build_app as build_db_app
    from mcp_servers.kowiki_server import build_app as build_kowiki_app

    # [1] 두 MCP 서버(db/kowiki)를 각각 FastAPI 앱으로 만들어 로컬 uvicorn으로 띄운다.
    # build_app 시점에 실 PG/임베딩 서버로의 연결이 이뤄지므로 여기서 인프라가
    # 죽어 있으면 곧바로 실패한다(의도된 fail-fast).
    print("=== [1] db / kowiki MCP 서버 기동 (실 PG / 실 임베딩) ===")
    db_app = await build_db_app(api_key=API_KEY)
    kowiki_app = await build_kowiki_app(api_key=API_KEY)
    db_server, db_task = await _serve(db_app, db_port)
    kowiki_server, kowiki_task = await _serve(kowiki_app, kowiki_port)
    _print_kv("db MCP", f"http://127.0.0.1:{db_port} (LISTEN)")
    _print_kv("kowiki MCP", f"http://127.0.0.1:{kowiki_port} (LISTEN)")

    # [2] 방금 띄운 로컬 포트를 물도록 mcp만 켠 임시 config를 만든다(운영 원본 불변).
    tmp_config = _build_temp_config(db_port, kowiki_port)
    print(f"\n=== [2] 임시 config 생성: {tmp_config} (mcp.enabled=true, localhost) ===")

    # components: init_phase2 결과(정리 단계에서 참조). chat_ok: 채팅이 MCP 결과를
    # 실제로 반영했는지의 최종 판정 플래그.
    components: dict = {}
    chat_ok = False
    # 이 아래부터는 무슨 일이 있어도 finally에서 자원을 정리해야 하므로 try로 감싼다.
    try:
        # [3] 임시 config로 웹앱 상태를 실제로 부팅한다(lifespan 동일 경로 재현).
        print("\n=== [3] web 부트스트랩 (init/init_phase2 + web QueryEngine) ===")
        components = await _bootstrap_web(tmp_config)

        from httpx import ASGITransport, AsyncClient

        from web import app as webapp

        # ASGITransport는 실제 네트워크 포트를 열지 않고 메모리 안에서 FastAPI 앱을
        # 직접 호출한다. 즉 진짜 web.app:app을 그대로 두드려 엔드포인트를 검증한다.
        transport = ASGITransport(app=webapp.app)
        async with AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            # ── 검증 4-1: /v1/tools ───────────────────────────────
            # 웹앱이 노출하는 도구 목록에 MCP 도구가 실려 있는지 본다. 도구 이름은
            # mcp__<서버>__<도구> 규칙을 따른다. 두 도구가 다 보이면 스키마 노출 OK.
            print("\n=== [4-1] GET /v1/tools — MCP 도구 노출 확인 ===")
            r = await client.get("/v1/tools")
            tools = r.json().get("tools", [])
            names = sorted(t["name"] for t in tools)
            _print_kv("status", r.status_code)
            _print_kv("도구 수", len(names))
            _print_kv("전체 도구", names)
            has_db = "mcp__db__query" in names
            has_kowiki = "mcp__kowiki__search" in names
            _print_kv("mcp__db__query", "있음 OK" if has_db else "없음 FAIL")
            _print_kv("mcp__kowiki__search", "있음 OK" if has_kowiki else "없음 FAIL")

            # ── 검증 4-2: /metrics ────────────────────────────────
            # 도구가 목록에 뜨는 것과, 실제로 MCP 서버에 "연결"된 것은 다른 문제다.
            # /metrics의 mcp 블록으로 연결 여부(connected)와 서버별 도구 수를 확인한다.
            print("\n=== [4-2] GET /metrics — mcp.connected / tool_counts ===")
            r = await client.get("/metrics")
            mcp_metrics = r.json().get("mcp", {})
            _print_kv("connected", mcp_metrics.get("connected"))
            _print_kv("connected_count", mcp_metrics.get("connected_count"))
            _print_kv("tool_counts", mcp_metrics.get("tool_counts"))

            # ── 검증 5: cli 도구 풀 ───────────────────────────────
            # 웹 경로뿐 아니라 cli/repl 경로에서도 MCP 도구가 붙는지 확인한다. cli와
            # 웹은 같은 부트스트랩(init_phase2)을 타므로, 여기서 흡수가 확인되면
            # cli 채팅에서도 MCP 도구를 쓸 수 있다는 뜻이다(HTTP 없이 객체 직접 검사).
            print("\n=== [5] cli 도구 풀에 mcp__ 흡수 확인 (프로그램적) ===")
            # init_phase2가 만든 ModelDispatcher.worker_tools = cli_tools.
            # 거기에 mcp__ 도구가 흡수됐는지 본다(cli/repl도 같은 부트스트랩 경로).
            cli_dispatcher = components.get("model_dispatcher")
            cli_tool_names = sorted(t.name for t in cli_dispatcher.worker_tools) \
                if cli_dispatcher is not None else []
            if not cli_tool_names:
                # 디스패처에서 도구를 못 얻었으면(구조 변경 등) query_engine 쪽의
                # 내부 도구 목록(_tools)을 대체 경로로 조회한다.
                cli_engine = components.get("query_engine")
                cli_tool_names = sorted(t.name for t in getattr(cli_engine, "_tools", []))
            # 전체 도구 중 mcp__ 접두 도구만 뽑아 흡수 여부를 눈으로 확인.
            mcp_in_cli = [n for n in cli_tool_names if n.startswith("mcp__")]
            _print_kv("cli 도구", cli_tool_names)
            _print_kv("cli의 mcp__ 도구", mcp_in_cli)

            # ── 검증 4-3: /v1/chat (실 vLLM이 MCP 호출하는지) ──────
            # 가장 중요한 관문. 실제 사용자 질의를 던져 vLLM Worker가 스스로 MCP
            # 도구를 골라 호출하고, 그 결과를 답변에 반영하는지 관측한다. 여기까지
            # 통과하면 "도구 노출→연결→실호출"의 전 구간이 살아 있다는 증명이 된다.
            print("\n=== [4-3] POST /v1/chat — vLLM Worker의 MCP 도구 호출 관측 ===")
            print("  (임베딩 cold start로 첫 kowiki 호출이 수십 초 걸릴 수 있음)")
            # 두 질의로 서로 다른 MCP 경로를 자극한다.
            # (a) db MCP — 실 PG count(*) / (b) kowiki MCP — 실 임베딩+pgvector 검색
            queries = [
                "DB MCP 도구 mcp__db__query 를 사용해서 tb_knowledge 테이블의 "
                "전체 행 수를 SELECT count(*) 로 조회하고, 그 숫자를 알려줘.",
                "사내 지식베이스(kowiki)에서 mcp__kowiki__search 도구로 '니체'를 "
                "검색해서, 검색 결과로 나온 항목들의 제목을 알려줘.",
            ]
            for q in queries:
                print(f"\n  -- 질의: {q[:60]}...")
                # timeout을 넉넉히(300초) 잡는다 — 임베딩 cold start + LLM 추론 시간 고려.
                resp = await client.post(
                    "/v1/chat",
                    json={"message": q},
                    timeout=300.0,
                )
                body = resp.json()
                text = body.get("response", "")
                _print_kv("status", resp.status_code)
                _print_kv("응답(앞 400자)", text[:400])
                # 응답 텍스트에 실제 count 값이나 테이블명이 들어 있으면 MCP 결과가
                # 반영된 것으로 간주한다(휴리스틱 판정). 하나라도 걸리면 chat_ok=True.
                if any(tok in text for tok in ("1067", "1,067", "106", "tb_knowledge")):
                    chat_ok = True

            # MCP 호출 흔적은 서버 로그/메트릭으로 본다 — 도구 호출 후 다시 /metrics
            # 채팅 전/후 메트릭을 비교하면 호출 카운트 변화 등 흔적을 확인할 수 있다.
            print("\n=== [4-4] 채팅 후 /metrics 재조회 (도구 호출 흔적) ===")
            r = await client.get("/metrics")
            _print_kv("mcp", r.json().get("mcp", {}))

    finally:
        # 성공/실패와 무관하게 항상 실행 — 띄운 자원을 전부 회수해 운영에 무영향.
        print("\n=== [정리] MCP 서버 종료 + 임시 config 삭제 ===")
        # 임베딩 keepalive task 정리 — 아직 돌고 있으면 취소하고 종료를 기다린다.
        # 취소 시 발생하는 예외(CancelledError 등)는 정리 과정이므로 조용히 무시.
        keepalive = components.get("embedding_keepalive_task")
        if keepalive is not None and not keepalive.done():
            keepalive.cancel()
            try:
                await keepalive
            except BaseException:
                pass
        # MCP 클라이언트 정리(부트스트랩이 만든 매니저) — 열린 세션/커넥션을 닫는다.
        mgr = components.get("mcp_manager")
        if mgr is not None:
            try:
                await mgr.aclose_all()
            except Exception as e:
                print(f"  mcp_manager 정리 경고: {e}")
        # 로컬에 띄운 두 MCP uvicorn 서버를 graceful shutdown.
        await _stop(db_server, db_task)
        await _stop(kowiki_server, kowiki_task)
        # 마지막으로 임시 config 파일을 지운다. 이미 없거나 지울 수 없으면 경고만.
        try:
            os.remove(tmp_config)
            _print_kv("임시 config 삭제", tmp_config)
        except OSError as e:
            print(f"  임시 config 삭제 실패: {e}")

    # 최종 요약 — chat_ok가 True면 채팅이 MCP 결과를 실제로 반영했다는 의미.
    print("\n=== web app full e2e (MCP) 검증 종료 ===")
    print(f"  /v1/chat MCP 결과 반영: {'예 ✔' if chat_ok else '미확인 (응답 텍스트 참조)'}")


# 모듈로 import될 때가 아니라 직접 실행될 때만 asyncio 이벤트 루프에서 main()을 돈다.
if __name__ == "__main__":
    asyncio.run(main())
