"""
diag MCP 서버 — Nexus 인프라 진단을 도구로 노출한다.

도구:
  reachability — 웹/GPU/DB 서비스 도달성 점검(scripts/_diag_all_services 로직 포팅).
                 GPU 는 paramiko SSH, DB 는 socket TCP, 웹은 urllib HTTP(S)로 점검.
  rag_latency  — tb_knowledge 벡터 검색 지연 측정 + EXPLAIN ANALYZE
                 (scripts/_diag_rag_latency 참고). 임베딩 서버 latency 도 측정.

왜 scripts 로직을 "포팅" 하는가 (재사용이 아니라 복제):
  scripts/_diag_*.py 는 일회성 진단 스크립트이며 mcp_servers 가 import 하면
  의존성 경계가 모호해진다. 동일 점검 로직을 이 서버 안에 정리해 담고, 자격
  정보(GPU/DB 비밀번호)는 scripts 와 동일하게 상수로 둔다(LAN 전용).

에어갭:
  모든 점검 대상은 LAN 주소(192.168.x / localhost)다. 외부 도메인 호출 없음.
  paramiko/asyncpg 가 없으면 해당 점검만 건너뛰고 명확한 사유를 결과에 담는다.

동시성 주의 (P12 정신):
  paramiko/socket/urllib 는 블로킹 I/O 다. asyncio 이벤트 루프를 막지 않도록
  asyncio.to_thread 로 워커 스레드에서 실행한다.
"""

from __future__ import annotations

import asyncio
import json
import logging
import socket
import ssl
import time
import urllib.request
from typing import Any

from fastapi import FastAPI

from mcp_servers.framework import McpServerTool, create_mcp_app

logger = logging.getLogger("nexus.mcp_servers.diag")

# ─────────────────────────────────────────────
# 서버 상수 (scripts/_diag_*.py 와 동일 — LAN 전용 자격)
# ─────────────────────────────────────────────
GPU_HOST = "192.168.21.112"
GPU_USER = "idino"
GPU_PASS = "dkdlelsh@12"  # noqa: S105 — LAN 내부 진단용 고정 자격(에어갭)

DB_HOST = "192.168.10.39"
DB_PORTS = {"PostgreSQL": 5440, "Redis": 6340}

WEB_URL = "https://localhost:8443/metrics"
EMBED_URL = "http://192.168.21.112:8002"

# RAG 지연 측정에 쓰는 DB 컨테이너 자격(scripts/_diag_rag_latency 와 동일).
PG_CONTAINER = "docutil-postgres"
PG_USER = "nexus"
PG_DB = "nexus"
PG_PASS = "idino@12"  # noqa: S105 — LAN 내부 진단용 고정 자격(에어갭)


# ─────────────────────────────────────────────
# 블로킹 점검 함수들 (to_thread 로 호출)
# ─────────────────────────────────────────────
def _tcp_probe(host: str, port: int, timeout: float) -> tuple[bool, str]:
    """TCP 연결 가능 여부를 점검한다. (성공여부, 사유) 반환."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(timeout)
    try:
        s.connect((host, port))
        return True, ""
    except OSError as e:
        return False, f"{type(e).__name__}: {e}"
    finally:
        s.close()


def _check_web_blocking() -> dict[str, Any]:
    """로컬 웹 서버 /metrics 응답을 점검한다(자체서명 인증서 허용)."""
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    try:
        with urllib.request.urlopen(WEB_URL, timeout=5, context=ctx) as r:  # noqa: S310 — LAN localhost 고정 URL
            body = r.read().decode("utf-8", errors="replace")
            data = json.loads(body)
            scout = data.get("scout") or {}
            return {
                "reachable": True,
                "status": r.status,
                "agent_cache": data.get("agent_cache"),
                "scout_enabled": scout.get("scout_enabled"),
            }
    except (OSError, ValueError) as e:
        return {"reachable": False, "error": f"{type(e).__name__}: {e}"}


def _check_db_blocking() -> dict[str, Any]:
    """DB 서버의 PostgreSQL/Redis 포트 TCP 도달성을 점검한다."""
    ports: dict[str, Any] = {}
    for name, port in DB_PORTS.items():
        ok, msg = _tcp_probe(DB_HOST, port, 5.0)
        ports[name] = {"port": port, "reachable": ok, "detail": msg}
    return {"host": DB_HOST, "ports": ports}


def _check_gpu_blocking() -> dict[str, Any]:
    """GPU 서버에 SSH 접속해 vLLM/Scout 프로세스·포트·GPU 메모리를 점검한다."""
    try:
        import paramiko
    except ImportError:
        return {"reachable": False, "error": "paramiko 미설치 — GPU 점검 건너뜀"}

    def _run(ssh: Any, cmd: str, timeout: int = 15) -> str:
        _, stdout, stderr = ssh.exec_command(cmd, timeout=timeout)
        out = stdout.read().decode(errors="replace")
        err = stderr.read().decode(errors="replace")
        return (out + (("\n[stderr]\n" + err) if err.strip() else "")).strip()

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())  # noqa: S507 — LAN 내부 고정 호스트(에어갭)
    try:
        ssh.connect(GPU_HOST, username=GPU_USER, password=GPU_PASS, timeout=10)
    except Exception as e:  # noqa: BLE001 — paramiko 예외 계층이 넓어 결과 dict 로 환원
        return {"reachable": False, "error": f"SSH 실패: {type(e).__name__}: {e}"}

    try:
        vllm_proc = _run(ssh, "pgrep -af 'vllm.entrypoints.openai' || echo '(없음)'")
        listen = _run(ssh, "ss -tln 2>/dev/null | awk 'NR==1 || /:(8001|8003)\\>/' || true")
        gpu_mem = _run(
            ssh,
            "nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader",
        )
        models = _run(
            ssh,
            "curl -s --max-time 5 http://localhost:8001/v1/models "
            '| python3 -c "import json,sys; '
            "d=json.load(sys.stdin); print([m['id'] for m in d['data']])\" "
            "2>&1 || echo '(응답 없음)'",
            timeout=12,
        )
        return {
            "reachable": True,
            "host": GPU_HOST,
            "vllm_process": vllm_proc,
            "listen_8001_8003": listen,
            "gpu_memory": gpu_mem,
            "vllm_models": models,
        }
    finally:
        ssh.close()


def _embed_query_blocking(query_text: str) -> tuple[list[float], float]:
    """임베딩 서버를 호출해 (벡터, 지연ms)를 반환한다."""
    body = json.dumps({"texts": [query_text]}).encode()
    req = urllib.request.Request(  # noqa: S310 — LAN 192.168 고정 URL(에어갭)
        f"{EMBED_URL}/v1/embed",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=15) as resp:  # noqa: S310 — LAN 192.168 고정 URL
        payload = json.loads(resp.read())
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return payload["embeddings"][0], elapsed_ms


# ─────────────────────────────────────────────
# ReachabilityTool
# ─────────────────────────────────────────────
class ReachabilityTool(McpServerTool):
    """웹/GPU/DB 도달성을 한 번에 점검한다."""

    @property
    def name(self) -> str:
        return "reachability"

    @property
    def description(self) -> str:
        return (
            "Nexus 인프라(로컬 웹 서버, GPU 서버, DB 서버)의 도달성을 점검한다. "
            "웹은 HTTP(S), GPU 는 SSH, DB 는 TCP 포트로 확인하고 서비스별 상태를 반환한다."
        )

    @property
    def input_schema(self) -> dict[str, Any]:
        # 입력 인자 없음 — 고정된 LAN 대상만 점검한다.
        return {"type": "object", "properties": {}}

    async def call(self, arguments: dict[str, Any]) -> Any:
        """
        세 점검을 워커 스레드에서 병렬로 수행한다(블로킹 I/O 이므로 to_thread).

        Returns:
            {"web": {...}, "db": {...}, "gpu": {...}}
        """
        web, db, gpu = await asyncio.gather(
            asyncio.to_thread(_check_web_blocking),
            asyncio.to_thread(_check_db_blocking),
            asyncio.to_thread(_check_gpu_blocking),
        )
        return {"web": web, "db": db, "gpu": gpu}


# ─────────────────────────────────────────────
# RagLatencyTool
# ─────────────────────────────────────────────
class RagLatencyTool(McpServerTool):
    """
    tb_knowledge 벡터 검색 지연을 측정한다.

    pg_pool 이 있으면 직접 EXPLAIN ANALYZE + 검색 지연을 측정하고, 없으면
    임베딩 서버 latency 만 측정한다(부분 결과 — fail-soft).
    """

    def __init__(self, pg_pool: Any | None) -> None:
        self._pg = pg_pool

    @property
    def name(self) -> str:
        return "rag_latency"

    @property
    def description(self) -> str:
        return (
            "RAG 파이프라인 지연을 진단한다. 임베딩 서버 호출 지연을 측정하고, "
            "DB 연결이 가능하면 tb_knowledge 벡터 검색의 EXPLAIN ANALYZE 와 "
            "실측 지연을 함께 반환한다."
        )

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "임베딩/검색에 사용할 샘플 질의(기본값 있음).",
                },
                "top_k": {
                    "type": "integer",
                    "description": "검색 상위 K(기본 5).",
                },
            },
        }

    async def call(self, arguments: dict[str, Any]) -> Any:
        """
        임베딩 지연 + (가능하면) 벡터 검색 EXPLAIN ANALYZE 를 측정한다.

        Returns:
            {
              "query": str,
              "embed_latency_ms": float,
              "embed_dim": int,
              "search": {... EXPLAIN/지연 ...} | {"skipped": "사유"},
            }
        """
        query_text = arguments.get("query") or "이기이원론"
        # top_k 는 미지정 시에만 기본값 — `or` 관용구는 0 을 falsy 로 보아
        # 기본값으로 조용히 치환하므로, get(키, 기본값) 으로 명시적으로 처리한다.
        # (top_k=0/음수는 잘못된 입력이므로 기본값 대체가 아니라 거부해야 한다.)
        top_k = arguments.get("top_k", 5)
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
            raise ValueError("top_k 는 1 이상의 정수여야 합니다.")

        # 1) 임베딩 지연 측정(블로킹 → 워커 스레드).
        try:
            embedding, embed_ms = await asyncio.to_thread(
                _embed_query_blocking, f"query: {query_text}"
            )
        except (OSError, ValueError, KeyError) as e:
            raise RuntimeError(f"임베딩 서버 호출 실패: {type(e).__name__}: {e}") from e

        result: dict[str, Any] = {
            "query": query_text,
            "embed_latency_ms": round(embed_ms, 1),
            "embed_dim": len(embedding),
        }

        # 2) DB 가 있으면 벡터 검색 EXPLAIN ANALYZE + 실측.
        if self._pg is None:
            result["search"] = {"skipped": "pg_pool 없음 — 임베딩 지연만 측정"}
            return result

        vec_literal = "[" + ",".join(f"{v:.6f}" for v in embedding) + "]"
        explain_sql = (
            "EXPLAIN (ANALYZE, BUFFERS) "
            "SELECT id, title, 1 - (embedding <=> $1::vector) AS sim "
            "FROM tb_knowledge WHERE embedding IS NOT NULL "
            "ORDER BY embedding <=> $1::vector LIMIT $2"
        )
        search_sql = (
            "SELECT id, title, 1 - (embedding <=> $1::vector) AS sim "
            "FROM tb_knowledge WHERE embedding IS NOT NULL "
            "ORDER BY embedding <=> $1::vector LIMIT $2"
        )
        try:
            async with self._pg.acquire() as conn:
                plan_rows = await conn.fetch(explain_sql, vec_literal, top_k)
                t0 = time.perf_counter()
                _ = await conn.fetch(search_sql, vec_literal, top_k)
                search_ms = (time.perf_counter() - t0) * 1000
            result["search"] = {
                "search_latency_ms": round(search_ms, 1),
                "explain_analyze": [r[0] for r in plan_rows],
            }
        except Exception as e:  # noqa: BLE001 — asyncpg 예외 계층 폭넓음 → 결과에 환원
            result["search"] = {"error": f"{type(e).__name__}: {e}"}

        return result


async def build_app(api_key: str = "local-key") -> FastAPI:
    """
    diag MCP 서버 FastAPI 앱을 조립한다.

    DB 연결은 best-effort 다 — 실패해도 reachability 점검과 임베딩 지연 측정은
    동작해야 하므로, pg_pool 이 None 이어도 앱을 정상 반환한다(fail-soft).
    """
    from core.config import load_and_validate_config

    config = load_and_validate_config()

    pg_pool: Any | None = None
    try:
        import asyncpg

        pg_pool = await asyncpg.create_pool(
            host=config.postgresql.host,
            port=config.postgresql.port,
            database=config.postgresql.database,
            user=config.postgresql.user,
            password=config.postgresql.password,
            min_size=1,
            max_size=4,
            timeout=10.0,
        )
        logger.info("diag MCP 서버: PostgreSQL 연결 성공")
    except ImportError:
        logger.warning("diag MCP 서버: asyncpg 미설치 — rag_latency 는 임베딩만 측정")
    except Exception as e:  # noqa: BLE001 — 연결 실패는 치명적 아님(부분 진단 유지)
        logger.warning("diag MCP 서버: PostgreSQL 연결 실패(부분 진단): %s", e)

    # pg_pool 이 있을 때만 종료 훅을 등록한다(lifespan shutdown).
    hooks: list[Any] = []
    if pg_pool is not None:

        async def _close_pool() -> None:
            await pg_pool.close()
            logger.info("diag MCP 서버: PostgreSQL 풀 종료")

        hooks.append(_close_pool)

    return create_mcp_app(
        tools=[ReachabilityTool(), RagLatencyTool(pg_pool)],
        api_key=api_key,
        title="Nexus Diagnostics MCP Server",
        shutdown_hooks=hooks,
    )
