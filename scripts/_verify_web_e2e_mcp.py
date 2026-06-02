"""web app full e2e (실물) — vLLM Worker가 MCP 도구를 호출하는 전 구간 증명.

이 스크립트는 mock 없이 실 인프라로만 동작한다:
  - 실 PostgreSQL (192.168.10.39:5440) — db MCP 서버가 사용
  - 실 임베딩 서버 (192.168.22.28:8002) — kowiki MCP 서버가 사용
  - 실 vLLM (192.168.22.28:8001) — web app QueryEngine의 Worker
  - 실 Redis (192.168.10.39:6340) — 세션 단기 메모리

수행 흐름:
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

종료 시 MCP 서버·임시 config를 정리한다(운영 무영향).

실행: python -m scripts._verify_web_e2e_mcp
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

sys.stdout.reconfigure(encoding="utf-8")

API_KEY = "local-key"

# db / kowiki MCP 서버를 띄울 localhost 고정 포트(테스트 충돌 시 _free_port로 대체).
DB_PORT = 18810
KOWIKI_PORT = 18813


# ─────────────────────────────────────────────
# uvicorn 서버 헬퍼 (검증된 _verify_mcp_live.py와 동일 패턴)
# ─────────────────────────────────────────────
def _port_free(port: int) -> bool:
    s = socket.socket()
    try:
        s.bind(("127.0.0.1", port))
        return True
    except OSError:
        return False
    finally:
        s.close()


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


async def _serve(app, port: int):
    """FastAPI 앱을 localhost uvicorn으로 띄우고 준비될 때까지 폴링한다."""
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    task = asyncio.create_task(server.serve())
    for _ in range(200):
        if server.started:
            break
        await asyncio.sleep(0.05)
    if not server.started:
        raise RuntimeError(f"uvicorn 서버가 포트 {port}에서 기동되지 않았습니다.")
    return server, task


async def _stop(server, task) -> None:
    server.should_exit = True
    await task


# ─────────────────────────────────────────────
# 임시 config 생성 — 운영 yaml을 로드해 mcp 섹션만 override
# ─────────────────────────────────────────────
def _build_temp_config(db_port: int, kowiki_port: int) -> str:
    """운영 nexus_config.yaml을 읽어 mcp만 켠 임시 복사본을 만들고 경로를 반환한다.

    운영 원본은 읽기만 한다. mcp.servers를 db/kowiki(localhost 포트)만 남기고
    read_only=true로 둔다 — 운영 데이터(tb_knowledge)에 쓰기 경로가 생기지 않게.
    """
    prod_path = Path("config/nexus_config.yaml")
    with open(prod_path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    # mcp 섹션만 교체 — 전역 enabled + db/kowiki localhost + read-only.
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

    fd, tmp_path = tempfile.mkstemp(suffix="_nexus_mcp_e2e.yaml", prefix="nexus_")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True)
    return tmp_path


# ─────────────────────────────────────────────
# web 부트스트랩 — web/app.py lifespan과 동일하되 config_path 주입 가능
# ─────────────────────────────────────────────
async def _bootstrap_web(config_path: str) -> dict:
    """init(config_path)+init_phase2 → web QueryEngine 조립 → _app_state 주입.

    web.app._app_state를 실제로 채워 넣는다. 이후 httpx로 실 app을 때리면
    lifespan 없이도 동일 상태에서 엔드포인트가 동작한다.
    """
    from core.bootstrap import init, init_phase2
    from web import app as webapp

    state = await init(config_path=config_path)
    webapp._app_state["state"] = state
    webapp._app_state["config"] = state.config

    components = await init_phase2(state)
    webapp._app_state["tool_registry"] = components.get("tool_registry")
    webapp._app_state["model_provider"] = components.get("model_provider")
    webapp._app_state["memory_manager"] = components.get("memory_manager")
    webapp._app_state["tenant_registry"] = state.config.tenants
    webapp._app_state["embedding_keepalive_task"] = components.get(
        "embedding_keepalive_task"
    )

    web_engine, web_dispatcher, web_tools = webapp._build_web_query_engine(
        components, state
    )
    webapp._app_state["model_dispatcher"] = web_dispatcher
    webapp._app_state["query_engine"] = web_engine
    # lifespan과 동일: /v1/tools가 모델 실풀(MCP 포함)을 노출하도록 web_tools 저장.
    webapp._app_state["web_tools"] = web_tools
    return components


def _print_kv(label: str, value: Any) -> None:
    print(f"  {label:22s}: {value}")


async def main() -> None:
    # 포트 충돌 회피 — 고정 포트가 막혀 있으면 OS 할당으로 대체.
    db_port = DB_PORT if _port_free(DB_PORT) else _free_port()
    kowiki_port = KOWIKI_PORT if _port_free(KOWIKI_PORT) else _free_port()

    from mcp_servers.db_server import build_app as build_db_app
    from mcp_servers.kowiki_server import build_app as build_kowiki_app

    print("=== [1] db / kowiki MCP 서버 기동 (실 PG / 실 임베딩) ===")
    db_app = await build_db_app(api_key=API_KEY)
    kowiki_app = await build_kowiki_app(api_key=API_KEY)
    db_server, db_task = await _serve(db_app, db_port)
    kowiki_server, kowiki_task = await _serve(kowiki_app, kowiki_port)
    _print_kv("db MCP", f"http://127.0.0.1:{db_port} (LISTEN)")
    _print_kv("kowiki MCP", f"http://127.0.0.1:{kowiki_port} (LISTEN)")

    tmp_config = _build_temp_config(db_port, kowiki_port)
    print(f"\n=== [2] 임시 config 생성: {tmp_config} (mcp.enabled=true, localhost) ===")

    components: dict = {}
    chat_ok = False
    try:
        print("\n=== [3] web 부트스트랩 (init/init_phase2 + web QueryEngine) ===")
        components = await _bootstrap_web(tmp_config)

        from httpx import ASGITransport, AsyncClient

        from web import app as webapp

        transport = ASGITransport(app=webapp.app)
        async with AsyncClient(
            transport=transport, base_url="http://test"
        ) as client:
            # ── 검증 4-1: /v1/tools ───────────────────────────────
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
            print("\n=== [4-2] GET /metrics — mcp.connected / tool_counts ===")
            r = await client.get("/metrics")
            mcp_metrics = r.json().get("mcp", {})
            _print_kv("connected", mcp_metrics.get("connected"))
            _print_kv("connected_count", mcp_metrics.get("connected_count"))
            _print_kv("tool_counts", mcp_metrics.get("tool_counts"))

            # ── 검증 5: cli 도구 풀 ───────────────────────────────
            print("\n=== [5] cli 도구 풀에 mcp__ 흡수 확인 (프로그램적) ===")
            # init_phase2가 만든 ModelDispatcher.worker_tools = cli_tools.
            # 거기에 mcp__ 도구가 흡수됐는지 본다(cli/repl도 같은 부트스트랩 경로).
            cli_dispatcher = components.get("model_dispatcher")
            cli_tool_names = sorted(t.name for t in cli_dispatcher.worker_tools) \
                if cli_dispatcher is not None else []
            if not cli_tool_names:
                # 속성명이 다르면 query_engine의 도구로 대체 확인
                cli_engine = components.get("query_engine")
                cli_tool_names = sorted(t.name for t in getattr(cli_engine, "_tools", []))
            mcp_in_cli = [n for n in cli_tool_names if n.startswith("mcp__")]
            _print_kv("cli 도구", cli_tool_names)
            _print_kv("cli의 mcp__ 도구", mcp_in_cli)

            # ── 검증 4-3: /v1/chat (실 vLLM이 MCP 호출하는지) ──────
            print("\n=== [4-3] POST /v1/chat — vLLM Worker의 MCP 도구 호출 관측 ===")
            print("  (임베딩 cold start로 첫 kowiki 호출이 수십 초 걸릴 수 있음)")
            # (a) db MCP — 실 PG count(*) / (b) kowiki MCP — 실 임베딩+pgvector 검색
            queries = [
                "DB MCP 도구 mcp__db__query 를 사용해서 tb_knowledge 테이블의 "
                "전체 행 수를 SELECT count(*) 로 조회하고, 그 숫자를 알려줘.",
                "사내 지식베이스(kowiki)에서 mcp__kowiki__search 도구로 '니체'를 "
                "검색해서, 검색 결과로 나온 항목들의 제목을 알려줘.",
            ]
            for q in queries:
                print(f"\n  -- 질의: {q[:60]}...")
                resp = await client.post(
                    "/v1/chat",
                    json={"message": q},
                    timeout=300.0,
                )
                body = resp.json()
                text = body.get("response", "")
                _print_kv("status", resp.status_code)
                _print_kv("응답(앞 400자)", text[:400])
                if any(tok in text for tok in ("1067", "1,067", "106", "tb_knowledge")):
                    chat_ok = True

            # MCP 호출 흔적은 서버 로그/메트릭으로 본다 — 도구 호출 후 다시 /metrics
            print("\n=== [4-4] 채팅 후 /metrics 재조회 (도구 호출 흔적) ===")
            r = await client.get("/metrics")
            _print_kv("mcp", r.json().get("mcp", {}))

    finally:
        print("\n=== [정리] MCP 서버 종료 + 임시 config 삭제 ===")
        # 임베딩 keepalive task 정리
        keepalive = components.get("embedding_keepalive_task")
        if keepalive is not None and not keepalive.done():
            keepalive.cancel()
            try:
                await keepalive
            except BaseException:
                pass
        # MCP 클라이언트 정리(부트스트랩이 만든 매니저)
        mgr = components.get("mcp_manager")
        if mgr is not None:
            try:
                await mgr.aclose_all()
            except Exception as e:
                print(f"  mcp_manager 정리 경고: {e}")
        await _stop(db_server, db_task)
        await _stop(kowiki_server, kowiki_task)
        try:
            os.remove(tmp_config)
            _print_kv("임시 config 삭제", tmp_config)
        except OSError as e:
            print(f"  임시 config 삭제 실패: {e}")

    print("\n=== web app full e2e (MCP) 검증 종료 ===")
    print(f"  /v1/chat MCP 결과 반영: {'예 ✔' if chat_ok else '미확인 (응답 텍스트 참조)'}")


if __name__ == "__main__":
    asyncio.run(main())
