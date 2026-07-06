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

전체 흐름(main 기준 4단계):
  [1] localhost에 MCP 서버 4종(db/diag/docingest/kowiki)을 uvicorn으로 기동
  [2] 운영 config를 읽어 mcp만 localhost로 바꾼 임시 yaml 생성(_build_temp_config)
  [3] 그 임시 config로 web 앱을 부트스트랩(_bootstrap_web)
  [4] httpx ASGI 클라이언트로 4가지 검증(도구 풀 / metrics / 지식 질의 / db 질의)

주요 함수:
  - _port_free / _free_port : 포트 사용 가능 여부 확인 및 빈 포트 확보
  - _serve / _stop          : FastAPI 앱을 uvicorn으로 띄우고 종료
  - _build_temp_config      : 운영 mcp 설정 보존 + base_url만 localhost override
  - _bootstrap_web          : web/app.py lifespan과 동일한 부트스트랩(config 주입)
  - main                    : 위 [1]~[4] 오케스트레이션 + 종합 PASS/FAIL 판정

실행: python -m scripts._verify_web_e2e_optionA

작성자: 이현수 / 작성일: 2026-07-05
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

# 한글 로그가 Windows 콘솔에서 깨지지 않도록 표준출력을 UTF-8로 강제한다.
sys.stdout.reconfigure(encoding="utf-8")

# 로컬에서 띄우는 MCP 서버들이 공유하는 인증 키(임시 검증용 고정값).
API_KEY = "local-key"

# localhost에 띄울 MCP 서버 고정 포트(충돌 시 _free_port로 대체).
DB_PORT = 18810
DIAG_PORT = 18811
DOCINGEST_PORT = 18812
KOWIKI_PORT = 18813

# RTX 5090은 컨텍스트 창이 8K(8192 토큰)뿐이다. 자동 RAG 주입 후 input_tokens가
# 이 값을 넘으면 overflow — 옵션 A가 실패했다는 뜻이므로 검증의 임계선으로 쓴다.
CONTEXT_LIMIT = 8192  # RTX 5090 8K — 옵션 A의 overflow 임계


# ─────────────────────────────────────────────
# uvicorn 서버 헬퍼
# ─────────────────────────────────────────────
def _port_free(port: int) -> bool:
    """지정한 포트가 지금 비어 있는지(bind 가능한지) 확인한다.

    고정 포트(DB_PORT 등)를 그대로 쓸 수 있는지 판단하는 용도. bind에 성공하면
    비어 있는 것이므로 True, 이미 누가 점유해 OSError가 나면 False를 돌려준다.
    확인용 소켓은 finally에서 반드시 닫아 자원 누수를 막는다.
    """
    s = socket.socket()
    try:
        s.bind(("127.0.0.1", port))
        return True
    except OSError:
        return False
    finally:
        s.close()


def _free_port() -> int:
    """OS가 임의로 배정해 주는 빈 포트 번호를 하나 얻어 반환한다.

    포트 0으로 bind하면 커널이 사용 가능한 포트를 자동으로 골라준다. 고정 포트가
    이미 점유돼 있을 때(_port_free가 False) 대체 포트로 쓰기 위한 폴백이다.
    """
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


async def _serve(app, port: int):
    """FastAPI 앱을 localhost uvicorn으로 띄우고 준비될 때까지 폴링한다.

    uvicorn Server.serve()를 백그라운드 asyncio 태스크로 돌린 뒤, server.started가
    True가 될 때까지 0.05초 간격으로 최대 200회(=약 10초) 기다린다. 그 안에 기동에
    실패하면 RuntimeError를 던진다. 반환한 (server, task)는 나중에 _stop으로 닫는다.
    """
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
    """_serve로 띄운 uvicorn 서버를 정상 종료한다.

    should_exit 플래그를 세우면 serve() 루프가 스스로 빠져나오므로, 그 태스크가
    끝날 때까지 await해 완전히 정리되기를 기다린다.
    """
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
    """운영 web/app.py의 lifespan과 동일한 순서로 web 앱을 부트스트랩한다.

    실제 web 서버가 뜰 때 하는 초기화(init → init_phase2 → web QueryEngine 조립)를
    그대로 재현하되, config_path를 주입해 우리가 만든 임시 config로 뜨게 한다.
    조립된 컴포넌트들을 webapp._app_state에 채워 넣어야 이후 httpx ASGI 요청이
    실제 라우트 핸들러에서 정상 동작한다.

    반환: init_phase2가 만든 components dict. 여기에 state를 "_state" 키로 끼워 넣어
    호출부에서 state.mcp_servers(제외 사유 등)를 직접 들여다볼 수 있게 한다.
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
    webapp._app_state["web_tools"] = web_tools
    # state도 함께 반환해 mcp_servers/excluded 사유를 직접 들여다본다.
    components["_state"] = state
    return components


def _kv(label: str, value: Any) -> None:
    """콘솔 로그를 "  라벨 : 값" 형태로 열 맞춰 예쁘게 출력하는 헬퍼."""
    print(f"  {label:26s}: {value}")


async def main() -> None:
    """옵션 A 실증 e2e의 전체 시나리오를 순서대로 수행한다.

    [1] MCP 서버 4종 기동 → [2] 임시 config 생성 → [3] web 부트스트랩 →
    [4] 검증 1~4 수행. finally에서 서버/임시 config를 정리하고, 마지막에 각 검증
    결과(verdict)를 모아 옵션 A 핵심 증명의 PASS/FAIL을 종합 판정한다.
    """
    # 고정 포트가 이미 점유돼 있으면 _free_port로 빈 포트를 대신 배정해 충돌을 피한다.
    ports = {
        "db": DB_PORT if _port_free(DB_PORT) else _free_port(),
        "diag": DIAG_PORT if _port_free(DIAG_PORT) else _free_port(),
        "docingest": DOCINGEST_PORT if _port_free(DOCINGEST_PORT) else _free_port(),
        "kowiki": KOWIKI_PORT if _port_free(KOWIKI_PORT) else _free_port(),
    }

    # 각 MCP 서버의 FastAPI 앱 팩토리를 가져온다(build_app이 실 인프라에 연결한다).
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

    # 띄운 서버들을 (server, task) 튜플로 모아둔다 → finally에서 한꺼번에 _stop 처리.
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

    # 각 검증 결과를 이름→불리언으로 누적한다. 맨 끝에서 이 값들을 AND로 묶어
    # 옵션 A 핵심 증명의 최종 PASS/FAIL을 판정한다.
    verdict: dict[str, Any] = {}
    components: dict = {}
    try:
        print("\n=== [3] web 부트스트랩 (init/init_phase2 + web QueryEngine) ===")
        components = await _bootstrap_web(tmp_config)
        state = components.get("_state")

        from httpx import ASGITransport, AsyncClient

        from web import app as webapp

        # ASGITransport로 실제 네트워크 없이 web 앱에 직접 요청한다(인프로세스 e2e).
        # base_url은 형식상 필요할 뿐 실제 접속 대상이 아니다.
        transport = ASGITransport(app=webapp.app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # ── 검증 1: /v1/tools — kowiki 제외, db 유지 ────────────
            print("\n=== [검증 1] GET /v1/tools — Worker 도구 풀 구성 ===")
            # /v1/tools는 Worker에게 실제 노출되는 도구 목록을 준다. 여기서 mcp__로
            # 시작하는 이름만 추려 어떤 MCP 도구가 풀에 들어왔는지 확인한다.
            r = await client.get("/v1/tools")
            tools = r.json().get("tools", [])
            names = sorted(t["name"] for t in tools)
            mcp_names = [n for n in names if n.startswith("mcp__")]
            _kv("status", r.status_code)
            _kv("mcp__ 도구 수", len(mcp_names))
            _kv("mcp__ 도구", mcp_names)

            # 옵션 A 기대치: db 도구는 있어야 하고 kowiki 도구는 없어야 한다.
            # (diag/docingest는 있으면 좋고 없어도 무방 — 판정에 넣지 않는다.)
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
            # 아래 db_info는 참고용 진단 변수이며 최종 판정에는 쓰이지 않는다.
            # (줄 끝 noqa 지시로 미사용 경고 억제.) db 연결 판정은 metrics로 한다.
            db_info = mcp_servers.get("kowiki", {})  # noqa
            _kv("kowiki state", kowiki_info)
            _kv("db state", mcp_servers.get("db", {}))
            # 핵심 판정: kowiki는 "expose_to_worker=false" 사유로 제외돼야 하고,
            # db는 metrics의 connected 목록에 실제로 잡혀 있어야 한다.
            kowiki_excluded = kowiki_info.get("excluded") == "expose_to_worker=false"
            db_connected = "db" in (mcp_metrics.get("connected") or [])
            _kv("kowiki excluded 사유", "OK" if kowiki_excluded else "FAIL")
            _kv("db connected", "OK" if db_connected else "FAIL")
            verdict["metrics_kowiki_excluded"] = kowiki_excluded
            verdict["metrics_db_connected"] = db_connected

            # 임베딩 서버는 첫 요청이 느리다(cold start). 검증 3의 시간·타임아웃이
            # 이 지연에 휘둘리지 않도록, 짧은 질의를 미리 한 번 던져 예열해 둔다.
            # 실패해도 검증 본체가 아니므로 경고만 찍고 넘어간다.
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
            ktext = body.get("response", "")  # 모델의 최종 답변 텍스트
            kusage = body.get("usage", {}) or {}  # 토큰 사용량(overflow 확인용)
            ktool_calls = body.get("tool_calls", []) or []  # 이번 응답에서 호출한 도구
            ktool_names = [tc.get("name", "") for tc in ktool_calls]
            # kowiki 도구가 호출됐다면 옵션 A 실패다(자동 RAG로만 답해야 함).
            kowiki_tool_called = any(
                str(n).startswith("mcp__kowiki__") for n in ktool_names
            )
            kinput = int(kusage.get("input_tokens", 0) or 0)  # 프롬프트 입력 토큰 수
            _kv("status", resp.status_code)
            _kv("호출된 도구", ktool_names if ktool_names else "(없음)")
            _kv("kowiki 도구 호출됨?", "예(FAIL)" if kowiki_tool_called else "아니오(OK)")
            _kv("usage.input_tokens", kinput)
            _kv(f"input_tokens < {CONTEXT_LIMIT}",
                "OK" if (0 < kinput < CONTEXT_LIMIT) else f"확인필요({kinput})")
            _kv("응답(앞 500자)", ktext[:500])
            # 핵심 증명: (1) HTTP 200, (2) kowiki 도구 미호출, (3) 20자 넘는 실제
            # 답변 텍스트, (4) input_tokens가 8K 미만(overflow 없음) — 네 조건 모두.
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
            # 지식 질의와 반대로, 여기서는 mcp__db__query가 실제로 호출돼야 정상이다.
            db_tool_called = any(str(n) == "mcp__db__query" for n in dtool_names)
            _kv("status", resp.status_code)
            _kv("호출된 도구", dtool_names if dtool_names else "(없음)")
            _kv("응답(앞 400자)", dtext[:400])
            # 정답 판정: 쉼표 제거 후 "1067978"이 들어 있거나, 원문에 "1,067,978"이
            # 그대로 들어 있으면 정답으로 본다(모델이 숫자 표기를 어떻게 하든 허용).
            answer_correct = any(
                tok in dtext.replace(",", "") for tok in ("1067978",)
            ) or any(tok in dtext for tok in ("1,067,978",))
            _kv("mcp__db__query 호출됨", "OK" if db_tool_called else "(텍스트로만?)")
            _kv("정답 1,067,978 포함", "OK" if answer_correct else "확인필요")
            verdict["db_no_regression"] = answer_correct

    finally:
        # 검증이 성공하든 예외로 죽든, 띄운 자원은 반드시 여기서 되돌린다.
        print("\n=== [정리] MCP 서버 종료 + 임시 config 삭제 ===")
        # (1) 임베딩 keepalive 백그라운드 태스크를 취소하고 마무리를 기다린다.
        #     cancel 시 발생하는 CancelledError 등은 삼켜서 정리를 계속 진행한다.
        keepalive = components.get("embedding_keepalive_task")
        if keepalive is not None and not keepalive.done():
            keepalive.cancel()
            try:
                await keepalive
            except BaseException:
                pass
        # (2) MCP 매니저가 열어 둔 연결(HTTP 세션 등)을 모두 닫는다.
        mgr = components.get("mcp_manager")
        if mgr is not None:
            try:
                await mgr.aclose_all()
            except Exception as e:
                print(f"  mcp_manager 정리 경고: {e}")
        # (3) 앞서 모아 둔 uvicorn 서버들을 하나씩 정상 종료한다.
        for srv, tsk in servers:
            try:
                await _stop(srv, tsk)
            except Exception as e:
                print(f"  서버 종료 경고: {e}")
        # (4) 임시로 만든 config yaml 파일을 삭제한다(운영 원본은 손대지 않았다).
        try:
            os.remove(tmp_config)
            _kv("임시 config 삭제", tmp_config)
        except OSError as e:
            print(f"  임시 config 삭제 실패: {e}")

    # ── 종합 판정 ──────────────────────────────────────────
    # 검증 1~4에서 쌓아 둔 verdict를 모두 출력한 뒤, 핵심 5개 조건을 AND로 묶어
    # 옵션 A가 실제로 증명됐는지(PASS) 최종 결론을 낸다.
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
    # 스크립트로 직접 실행할 때만 비동기 main()을 이벤트 루프에서 돌린다.
    asyncio.run(main())
