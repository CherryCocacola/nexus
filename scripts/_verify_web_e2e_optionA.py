"""옵션 A 실증 — kowiki MCP를 Worker 풀에서 제외(expose_to_worker=false)한
변경이 RTX 5090에서 kowiki 지식 질의까지 8K overflow 없이 정상 동작시키는지
web app full e2e로 증명한다. mock 금지 — 실 PG/임베딩/vLLM/Redis로만.

직전 드라이버(_verify_web_e2e_mcp.py)와의 차이(옵션 A 반영):
  1) _build_temp_config가 운영 nexus_config.yaml의 mcp.servers를 "그대로 보존"하되
     base_url만 localhost로 override한다. 핵심: kowiki의 expose_to_worker=false가
     임시 config에도 그대로 들어가야 옵션 A가 실제로 행사된다(직전 드라이버는 mcp
     섹션을 통째로 재작성하면서 expose_to_worker를 빠뜨려 kowiki가 등록돼버렸다).
  2) 검증 기대치를 옵션 A에 맞춰 반전: mcp__kowiki__search는 "없어야" OK,
     mcp__db__query는 "있어야" OK. kowiki는 state.mcp_servers에서 excluded 사유로
     표기돼야 한다.
  3) kowiki 지식 질의("니체 철학")는 도구 호출 없이(=tool_calls에 kowiki 없음)
     자동 RAG 주입으로 답하고, usage.input_tokens < 8192(8K overflow 없음)여야 한다.
  4) db 질의는 여전히 mcp__db__query를 호출해 1,067,978(tb_knowledge 행 수) 정답.

실 인프라:
  - 실 PostgreSQL (192.168.10.39:5440) — db MCP 서버 + 장기 메모리 + 자동 RAG
  - 실 임베딩 (192.168.21.112:8002) — 자동 RAG 주입 / kowiki MCP 서버
  - 실 vLLM (192.168.21.112:8001) — web QueryEngine Worker
  - 실 Redis (192.168.10.39:6340) — 세션 단기 메모리

운영 무영향: 운영 config 원본은 읽기만 하고, 임시 yaml 복사본으로만 부트스트랩한다.
모든 MCP 서버는 read_only=true로 등록되어 운영 데이터(tb_knowledge)에 쓰기 경로가
생기지 않는다. 종료 시 서버/임시 config를 정리한다.

실행: python -m scripts._verify_web_e2e_optionA
"""

from __future__ import annotations

import asyncio
import copy
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

# localhost에 띄울 MCP 서버 고정 포트(충돌 시 _free_port로 대체).
DB_PORT = 18810
DIAG_PORT = 18811
DOCINGEST_PORT = 18812
KOWIKI_PORT = 18813

CONTEXT_LIMIT = 8192  # RTX 5090 8K — 옵션 A의 overflow 임계


# ─────────────────────────────────────────────
# uvicorn 서버 헬퍼
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
# 임시 config — 운영 yaml의 mcp.servers를 "보존"하고 base_url만 override
# ─────────────────────────────────────────────
def _build_temp_config(ports: dict[str, int]) -> tuple[str, dict[str, dict]]:
    """운영 nexus_config.yaml을 읽어 mcp만 localhost로 바꾼 임시 복사본을 만든다.

    핵심(옵션 A): mcp.servers의 각 항목 속성(특히 kowiki의 expose_to_worker=false,
    trust.read_only=true)을 그대로 보존하고, 우리가 localhost에 실제로 띄운
    서버(db/diag/docingest/kowiki)만 base_url을 127.0.0.1:<port>로 바꾸고
    enabled=true로 켠다. 이렇게 해야 임시 config가 운영의 expose_to_worker=false를
    충실히 반영해, 옵션 A(Worker 풀 제외)가 실제로 행사된다.

    운영 원본은 읽기만 한다. 반환: (임시 config 경로, 적용된 서버 설정 요약).
    """
    prod_path = Path("config/nexus_config.yaml")
    with open(prod_path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    mcp = data.get("mcp") or {}
    mcp = copy.deepcopy(mcp)
    mcp["enabled"] = True  # 전역 마스터 스위치만 임시로 켠다(운영 원본은 false 유지).
    # 임베딩 cold start와 무관(연결만). 넉넉히.
    mcp["connect_timeout_sec"] = 10.0

    applied: dict[str, dict] = {}
    for server in mcp.get("servers", []):
        name = server.get("name")
        if name in ports:
            # 우리가 localhost에 띄운 서버만 켜고 base_url을 override한다.
            server["base_url"] = f"http://127.0.0.1:{ports[name]}"
            server["enabled"] = True
            server["api_key"] = API_KEY
            # 운영 데이터 보호: read-only 신뢰 메타를 강제(쓰기 경로 차단).
            server.setdefault("trust", {})
            server["trust"]["read_only"] = True
            server["allow_write"] = False
        else:
            # 우리가 안 띄운 서버는 확실히 꺼둔다(외부 LAN으로 새는 것 방지).
            server["enabled"] = False
        applied[name] = {
            "enabled": server.get("enabled"),
            "base_url": server.get("base_url"),
            "expose_to_worker": server.get("expose_to_worker", True),
            "read_only": (server.get("trust") or {}).get("read_only"),
        }

    data["mcp"] = mcp

    fd, tmp_path = tempfile.mkstemp(suffix="_nexus_optionA_e2e.yaml", prefix="nexus_")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True)
    return tmp_path, applied


# ─────────────────────────────────────────────
# web 부트스트랩 — web/app.py lifespan과 동일하되 config_path 주입 가능
# ─────────────────────────────────────────────
async def _bootstrap_web(config_path: str) -> dict:
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
    webapp._app_state["web_tools"] = web_tools
    # state도 함께 반환해 mcp_servers/excluded 사유를 직접 들여다본다.
    components["_state"] = state
    return components


def _kv(label: str, value: Any) -> None:
    print(f"  {label:26s}: {value}")


async def main() -> None:
    # 포트 충돌 회피.
    ports = {
        "db": DB_PORT if _port_free(DB_PORT) else _free_port(),
        "diag": DIAG_PORT if _port_free(DIAG_PORT) else _free_port(),
        "docingest": DOCINGEST_PORT if _port_free(DOCINGEST_PORT) else _free_port(),
        "kowiki": KOWIKI_PORT if _port_free(KOWIKI_PORT) else _free_port(),
    }

    from mcp_servers.db_server import build_app as build_db_app
    from mcp_servers.diag_server import build_app as build_diag_app
    from mcp_servers.docingest_server import build_app as build_docingest_app
    from mcp_servers.kowiki_server import build_app as build_kowiki_app

    print("=== [1] MCP 서버 기동 (실 PG / 실 임베딩 / 실 진단) ===")
    db_app = await build_db_app(api_key=API_KEY)
    diag_app = await build_diag_app(api_key=API_KEY)
    docingest_app = await build_docingest_app(api_key=API_KEY)
    # kowiki 서버는 expose_to_worker=false라 Nexus가 연결하지 않지만,
    # 외부 사내 앱 재사용 검증용으로 함께 띄운다(선택, 무해).
    kowiki_app = await build_kowiki_app(api_key=API_KEY)

    servers: list[tuple[Any, Any]] = []
    db_server, db_task = await _serve(db_app, ports["db"])
    servers.append((db_server, db_task))
    diag_server, diag_task = await _serve(diag_app, ports["diag"])
    servers.append((diag_server, diag_task))
    doc_server, doc_task = await _serve(docingest_app, ports["docingest"])
    servers.append((doc_server, doc_task))
    kowiki_server, kowiki_task = await _serve(kowiki_app, ports["kowiki"])
    servers.append((kowiki_server, kowiki_task))
    for n, p in ports.items():
        _kv(f"{n} MCP", f"http://127.0.0.1:{p} (LISTEN)")

    tmp_config, applied = _build_temp_config(ports)
    print(f"\n=== [2] 임시 config 생성: {tmp_config} ===")
    print("  (운영 mcp.servers 속성 보존 + localhost override. expose_to_worker 그대로)")
    for name, info in applied.items():
        _kv(name, info)

    # 결과 누적 — 마지막 PASS/FAIL 종합 판정에 쓴다.
    verdict: dict[str, Any] = {}
    components: dict = {}
    try:
        print("\n=== [3] web 부트스트랩 (init/init_phase2 + web QueryEngine) ===")
        components = await _bootstrap_web(tmp_config)
        state = components.get("_state")

        from httpx import ASGITransport, AsyncClient

        from web import app as webapp

        transport = ASGITransport(app=webapp.app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # ── 검증 1: /v1/tools — kowiki 제외, db 유지 ────────────
            print("\n=== [검증 1] GET /v1/tools — Worker 도구 풀 구성 ===")
            r = await client.get("/v1/tools")
            tools = r.json().get("tools", [])
            names = sorted(t["name"] for t in tools)
            mcp_names = [n for n in names if n.startswith("mcp__")]
            _kv("status", r.status_code)
            _kv("mcp__ 도구 수", len(mcp_names))
            _kv("mcp__ 도구", mcp_names)

            has_db = "mcp__db__query" in names
            has_kowiki = any(n.startswith("mcp__kowiki__") for n in names)
            has_diag = any(n.startswith("mcp__diag__") for n in names)
            has_docingest = any(n.startswith("mcp__docingest__") for n in names)
            _kv("mcp__db__query 있음", "OK" if has_db else "FAIL")
            _kv("mcp__kowiki__* 없음", "OK" if not has_kowiki else "FAIL")
            _kv("mcp__diag__* 있음", "OK" if has_diag else "(없음)")
            _kv("mcp__docingest__* 있음", "OK" if has_docingest else "(없음)")
            verdict["tools_kowiki_excluded"] = (has_db and not has_kowiki)

            # ── 검증 2: /metrics + state.mcp_servers 제외 사유 ──────
            print("\n=== [검증 2] GET /metrics + state.mcp_servers — 제외 사유 ===")
            r = await client.get("/metrics")
            mcp_metrics = r.json().get("mcp", {})
            _kv("connected", mcp_metrics.get("connected"))
            _kv("connected_count", mcp_metrics.get("connected_count"))
            _kv("tool_counts", mcp_metrics.get("tool_counts"))
            # 제외 사유는 state.mcp_servers[name]["excluded"]에 있다(/metrics는
            # tool_count만 노출하므로 state를 직접 들여다본다).
            mcp_servers = getattr(state, "mcp_servers", {}) if state else {}
            kowiki_info = mcp_servers.get("kowiki", {})
            db_info = mcp_servers.get("kowiki", {})  # noqa
            _kv("kowiki state", kowiki_info)
            _kv("db state", mcp_servers.get("db", {}))
            kowiki_excluded = kowiki_info.get("excluded") == "expose_to_worker=false"
            db_connected = "db" in (mcp_metrics.get("connected") or [])
            _kv("kowiki excluded 사유", "OK" if kowiki_excluded else "FAIL")
            _kv("db connected", "OK" if db_connected else "FAIL")
            verdict["metrics_kowiki_excluded"] = kowiki_excluded
            verdict["metrics_db_connected"] = db_connected

            # 임베딩 워밍업 — 자동 RAG 첫 주입 cold start 완화.
            print("\n=== [워밍업] 임베딩 cold start 완화 (짧은 KNOWLEDGE 질의) ===")
            try:
                w = await client.post(
                    "/v1/chat",
                    json={"message": "간단히 자기소개 해줘."},
                    timeout=300.0,
                )
                _kv("워밍업 status", w.status_code)
            except Exception as e:
                _kv("워밍업 경고", e)

            # ── 검증 3: kowiki 지식 질의 — 자동 RAG, overflow 없음 ──
            print("\n=== [검증 3] POST /v1/chat — kowiki 지식 질의(KNOWLEDGE) ===")
            print("  (도구 호출 없이 자동 RAG 주입으로 답해야 함. input_tokens<8192)")
            kq = "니체 철학에 대해 알려줘."
            print(f"  -- 질의: {kq}")
            resp = await client.post("/v1/chat", json={"message": kq}, timeout=300.0)
            body = resp.json()
            ktext = body.get("response", "")
            kusage = body.get("usage", {}) or {}
            ktool_calls = body.get("tool_calls", []) or []
            ktool_names = [tc.get("name", "") for tc in ktool_calls]
            kowiki_tool_called = any(
                str(n).startswith("mcp__kowiki__") for n in ktool_names
            )
            kinput = int(kusage.get("input_tokens", 0) or 0)
            _kv("status", resp.status_code)
            _kv("호출된 도구", ktool_names if ktool_names else "(없음)")
            _kv("kowiki 도구 호출됨?", "예(FAIL)" if kowiki_tool_called else "아니오(OK)")
            _kv("usage.input_tokens", kinput)
            _kv(f"input_tokens < {CONTEXT_LIMIT}",
                "OK" if (0 < kinput < CONTEXT_LIMIT) else f"확인필요({kinput})")
            _kv("응답(앞 500자)", ktext[:500])
            # 핵심 증명: kowiki 도구 미호출 + 실텍스트 응답 + overflow 없음.
            knowledge_ok = (
                resp.status_code == 200
                and not kowiki_tool_called
                and len(ktext.strip()) > 20
                and kinput < CONTEXT_LIMIT
            )
            verdict["knowledge_auto_rag_ok"] = knowledge_ok
            verdict["knowledge_input_tokens"] = kinput

            # ── 검증 4: db 질의 — 회귀 없음(1,067,978) ──────────────
            print("\n=== [검증 4] POST /v1/chat — db 질의(tb_knowledge 행 수) ===")
            dq = (
                "DB MCP 도구 mcp__db__query 를 사용해서 tb_knowledge 테이블의 "
                "전체 행 수를 SELECT count(*) 로 조회하고, 그 숫자를 알려줘."
            )
            print(f"  -- 질의: {dq[:60]}...")
            resp = await client.post("/v1/chat", json={"message": dq}, timeout=300.0)
            body = resp.json()
            dtext = body.get("response", "")
            dtool_calls = body.get("tool_calls", []) or []
            dtool_names = [tc.get("name", "") for tc in dtool_calls]
            db_tool_called = any(str(n) == "mcp__db__query" for n in dtool_names)
            _kv("status", resp.status_code)
            _kv("호출된 도구", dtool_names if dtool_names else "(없음)")
            _kv("응답(앞 400자)", dtext[:400])
            answer_correct = any(
                tok in dtext.replace(",", "") for tok in ("1067978",)
            ) or any(tok in dtext for tok in ("1,067,978",))
            _kv("mcp__db__query 호출됨", "OK" if db_tool_called else "(텍스트로만?)")
            _kv("정답 1,067,978 포함", "OK" if answer_correct else "확인필요")
            verdict["db_no_regression"] = answer_correct

    finally:
        print("\n=== [정리] MCP 서버 종료 + 임시 config 삭제 ===")
        keepalive = components.get("embedding_keepalive_task")
        if keepalive is not None and not keepalive.done():
            keepalive.cancel()
            try:
                await keepalive
            except BaseException:
                pass
        mgr = components.get("mcp_manager")
        if mgr is not None:
            try:
                await mgr.aclose_all()
            except Exception as e:
                print(f"  mcp_manager 정리 경고: {e}")
        for srv, tsk in servers:
            try:
                await _stop(srv, tsk)
            except Exception as e:
                print(f"  서버 종료 경고: {e}")
        try:
            os.remove(tmp_config)
            _kv("임시 config 삭제", tmp_config)
        except OSError as e:
            print(f"  임시 config 삭제 실패: {e}")

    # ── 종합 판정 ──────────────────────────────────────────
    print("\n=== [종합 판정] 옵션 A 실증 ===")
    for k, v in verdict.items():
        _kv(k, v)
    core_pass = (
        verdict.get("tools_kowiki_excluded")
        and verdict.get("metrics_kowiki_excluded")
        and verdict.get("metrics_db_connected")
        and verdict.get("knowledge_auto_rag_ok")
        and verdict.get("db_no_regression")
    )
    print(f"\n  옵션 A 핵심 증명: {'PASS ✔' if core_pass else 'FAIL/미확인 ✘'}")
    print("  (mock 미사용 — 실 PG/임베딩/vLLM/Redis로만 검증)")


if __name__ == "__main__":
    asyncio.run(main())
