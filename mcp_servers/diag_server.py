"""
diag MCP 서버 — Nexus 인프라 상태를 "MCP 도구" 형태로 노출하는 진단 서버.

[이 파일이 하는 일 — 한눈에]
  Nexus 는 여러 대의 서버(웹/GPU/DB/임베딩)로 나뉘어 돌아가는 시스템이다.
  운영 중 "어디가 죽었는지" 빠르게 확인하려면 각 서버에 직접 접속해봐야 하는데,
  그 확인 로직을 하나의 MCP 서버로 묶어 두 개의 도구(tool)로 노출한 것이 이 파일이다.
  AI 에이전트(또는 사람)는 이 도구를 호출하기만 하면 인프라 상태 JSON 을 돌려받는다.

[노출하는 두 도구]
  reachability — 웹/GPU/DB 세 서버의 "도달 가능 여부"를 한 번에 점검한다.
                 웹은 urllib HTTP(S), GPU 는 paramiko SSH, DB 는 socket TCP 로 확인한다.
                 (원본: scripts/_diag_all_services.py 의 점검 로직을 옮겨 담음)
  rag_latency  — RAG(검색증강) 파이프라인의 지연을 측정한다. 임베딩 서버 응답 시간과,
                 DB 가 붙으면 tb_knowledge 벡터 검색의 EXPLAIN ANALYZE·실측 지연을 함께 준다.
                 (원본: scripts/_diag_rag_latency.py 참고)

[주요 구성 요소]
  - 블로킹 점검 함수들(_tcp_probe / _check_web_blocking / _check_db_blocking /
    _check_gpu_blocking / _embed_query_blocking): 실제 네트워크 I/O 를 수행한다.
  - ReachabilityTool / RagLatencyTool: 위 함수들을 MCP 도구 규격으로 감싼 클래스.
  - build_app(): FastAPI 앱을 조립하는 진입점. DB 풀 생성·정리 훅까지 담당한다.

[왜 scripts 로직을 import 하지 않고 여기로 "복제(포팅)" 했나]
  scripts/_diag_*.py 는 손으로 돌려보는 일회성 진단 스크립트다. mcp_servers 가
  그걸 import 하면 "서버 코드가 스크립트에 의존"하는 이상한 방향의 의존성이 생겨
  경계가 흐려진다. 그래서 같은 점검 로직을 이 서버 안에 정리해 다시 담았다.
  자격 정보(GPU SSH 비밀번호)는 소스에 절대 박지 않고 환경변수
  (NEXUS_DIAG_GPU_PASS)에서 읽는다. 값이 없으면 GPU SSH 점검만 fail-soft 로
  건너뛰고 그 사유를 결과 JSON 에 담는다(에어갭·보안 fail-closed 원칙).

[에어갭 준수]
  모든 점검 대상은 LAN 주소(192.168.x / localhost)뿐이다 — 외부 도메인 호출은 전혀 없다.
  paramiko(SSH)·asyncpg(DB) 라이브러리가 없으면 그 점검만 건너뛰고 명확한 사유를 남긴다.
  즉 일부가 실패해도 서버 전체가 죽지 않고, 가능한 만큼만 진단해 부분 결과를 돌려준다.

[동시성 주의 — 규칙 P12 정신]
  paramiko/socket/urllib 는 모두 "블로킹 I/O"다. 이걸 asyncio 코루틴 안에서 그냥 부르면
  이벤트 루프 전체가 멈춘다. 그래서 asyncio.to_thread 로 워커 스레드에 넘겨 실행한다.

작성자: 이현수 / 작성일: 2026-07-05
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import socket
import ssl
import time
import urllib.request
from typing import Any

from fastapi import FastAPI

from mcp_servers.framework import McpServerTool, create_mcp_app

# 이 모듈 전용 로거. 로그 계층 이름을 "nexus.mcp_servers.diag" 로 고정해
# 상위(nexus)에서 일괄 레벨 제어·필터가 가능하게 한다.
logger = logging.getLogger("nexus.mcp_servers.diag")

# ─────────────────────────────────────────────
# 서버 상수 (scripts/_diag_*.py 와 동일 — 전부 LAN 전용 주소·자격)
# 여기 값들은 "어느 서버의 무슨 포트를 볼지"를 고정한 진단 대상 목록이다.
# 하드코딩된 IP/포트지만 전부 사내 폐쇄망(192.168.x / localhost)이라 에어갭에 안전하다.
# ─────────────────────────────────────────────
GPU_HOST = "192.168.21.112"  # GPU 서버(vLLM·임베딩이 도는 머신)의 LAN IP
GPU_USER = "idino"           # GPU 서버 SSH 로그인 계정명
# GPU SSH 비밀번호는 소스에 박지 않고 환경변수에서 읽는다(보안 fail-closed).
# 이렇게 하는 이유: 비밀번호를 코드/깃에 남기지 않기 위해서다.
#   - 값이 있으면 그 비밀번호로 GPU SSH 도달성 점검을 수행한다.
#   - 값이 없으면(None/빈문자열) GPU 점검만 건너뛰고 사유를 결과 JSON 에 담는다.
GPU_PASS = os.environ.get("NEXUS_DIAG_GPU_PASS")

DB_HOST = "192.168.10.39"  # PostgreSQL·Redis 가 도는 DB 서버의 LAN IP
# 점검할 DB 포트 목록. {서비스이름: 포트번호} 형태로 두어 결과에 이름을 붙여준다.
DB_PORTS = {"PostgreSQL": 5440, "Redis": 6340}

WEB_URL = "https://localhost:8443/metrics"  # 로컬 웹 서버의 상태(metrics) 엔드포인트
EMBED_URL = "http://192.168.21.112:8002"    # 임베딩 서버 베이스 URL(끝에 경로를 붙여 호출)


# ─────────────────────────────────────────────
# 블로킹 점검 함수들 (반드시 asyncio.to_thread 로 감싸 호출할 것)
#
# 아래 함수들은 전부 "동기(블로킹) I/O"를 수행한다. 소켓 연결, HTTP 요청, SSH 접속이
# 끝날 때까지 스레드가 멈춰 기다린다. 따라서 async 코드에서 직접 부르면 안 되고,
# 도구의 call() 안에서 asyncio.to_thread(...) 로 워커 스레드에 넘겨 실행한다.
# ─────────────────────────────────────────────
def _tcp_probe(host: str, port: int, timeout: float) -> tuple[bool, str]:
    """지정한 host:port 로 TCP 연결이 되는지 점검한다.

    "포트가 열려 응답하는가"만 확인하는 가장 단순한 도달성 체크다. 실제 프로토콜
    (Postgres/Redis 등)까지 말을 걸지는 않고, TCP 3-way handshake 성공 여부만 본다.

    Args:
        host: 접속 대상 IP/호스트명.
        port: 접속 대상 포트 번호.
        timeout: 연결 대기 최대 시간(초). 이 시간을 넘기면 실패로 본다.

    Returns:
        (성공여부, 사유) 튜플. 성공이면 (True, "") — 사유는 빈 문자열.
        실패면 (False, "예외타입: 메시지") 형태로 왜 실패했는지 담아 준다.
    """
    # AF_INET(IPv4) + SOCK_STREAM(TCP) 소켓을 만든다.
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(timeout)  # 연결이 무한정 매달리지 않도록 타임아웃을 건다.
    try:
        s.connect((host, port))  # 연결 시도 — 실패하면 OSError 계열 예외가 난다.
        return True, ""
    except OSError as e:
        # 연결 거부·타임아웃·호스트 없음 등은 모두 OSError 하위로 잡아 사유로 환원한다.
        return False, f"{type(e).__name__}: {e}"
    finally:
        s.close()  # 성공/실패와 무관하게 소켓 자원은 반드시 닫는다.


def _check_web_blocking() -> dict[str, Any]:
    """로컬 웹 서버의 /metrics 엔드포인트를 HTTP(S)로 호출해 상태를 점검한다.

    단순 도달성뿐 아니라 응답 JSON 에서 몇 가지 핵심 필드(agent_cache, scout_enabled)를
    뽑아 준다. 이 값들로 "웹 서버가 올바른 버전·구성으로 떠 있는지"까지 가늠할 수 있다.

    자체서명(self-signed) 인증서를 쓰는 로컬 HTTPS 라서, 인증서 검증을 끈 SSL 컨텍스트로
    접속한다. 대상이 localhost 고정 URL 이므로 이 완화는 에어갭 환경에서 안전하다.

    Returns:
        성공: {"reachable": True, "status": HTTP상태코드, "agent_cache": ..., "scout_enabled": ...}
        실패: {"reachable": False, "error": "예외타입: 메시지"}
    """
    # 기본 SSL 컨텍스트를 만든 뒤, 호스트명 검증과 인증서 검증을 모두 끈다.
    # (로컬 자체서명 인증서라 검증하면 오히려 접속이 실패하기 때문.)
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    try:
        # timeout=5 로 5초 안에 응답이 없으면 실패 처리. context=ctx 로 위 SSL 설정 적용.
        with urllib.request.urlopen(WEB_URL, timeout=5, context=ctx) as r:  # noqa: S310 — LAN localhost 고정 URL
            # 응답 본문을 문자열로 디코딩(깨지는 바이트는 대체문자로 치환).
            body = r.read().decode("utf-8", errors="replace")
            data = json.loads(body)  # metrics 는 JSON — 파싱해 dict 로 만든다.
            # scout 하위 객체가 없을 수도 있으므로 `or {}` 로 빈 dict 를 보장한다.
            scout = data.get("scout") or {}
            return {
                "reachable": True,
                "status": r.status,                       # HTTP 상태 코드(정상이면 200)
                "agent_cache": data.get("agent_cache"),   # 에이전트 캐시 상태(버전 판정용)
                "scout_enabled": scout.get("scout_enabled"),  # Scout 기능 on/off
            }
    except (OSError, ValueError) as e:
        # OSError=네트워크 계열 실패, ValueError=JSON 파싱 실패 등. 사유를 담아 환원한다.
        return {"reachable": False, "error": f"{type(e).__name__}: {e}"}


def _check_db_blocking() -> dict[str, Any]:
    """DB 서버의 PostgreSQL·Redis 포트가 열려 있는지 TCP 로 각각 점검한다.

    DB_PORTS 에 정의된 서비스별 포트를 하나씩 _tcp_probe 로 확인해, 어느 서비스가
    살아 있고 어느 것이 죽었는지 개별적으로 알 수 있게 결과를 이름별로 묶어 준다.

    Returns:
        {"host": DB서버IP,
         "ports": {"PostgreSQL": {"port": 5440, "reachable": bool, "detail": 사유},
                   "Redis":      {"port": 6340, "reachable": bool, "detail": 사유}}}
    """
    ports: dict[str, Any] = {}
    # 서비스 이름과 포트를 하나씩 돌며 각각 TCP 도달성을 점검한다.
    for name, port in DB_PORTS.items():
        ok, msg = _tcp_probe(DB_HOST, port, 5.0)  # 서비스당 5초 타임아웃
        ports[name] = {"port": port, "reachable": ok, "detail": msg}
    return {"host": DB_HOST, "ports": ports}


def _check_gpu_blocking() -> dict[str, Any]:
    """GPU 서버에 SSH 로 붙어 vLLM 프로세스·리슨 포트·GPU 메모리·로드된 모델을 점검한다.

    단순 도달성을 넘어 "GPU 서버가 추론을 서비스할 준비가 됐는지"를 실제로 들여다본다.
    SSH 로 몇 개의 진단 명령을 원격 실행하고, 그 표준출력을 모아 결과 dict 로 돌려준다.

    두 단계의 fail-soft 방어가 있다(둘 다 서버를 죽이지 않고 사유만 담아 반환):
      1) 환경변수(NEXUS_DIAG_GPU_PASS) 미설정 → SSH 점검 자체를 건너뛴다.
      2) paramiko 라이브러리 미설치 → 역시 건너뛴다.

    Returns:
        성공: {"reachable": True, "host": ..., "vllm_process": ..., "listen_8001_8003": ...,
               "gpu_memory": ..., "vllm_models": ...}
        건너뜀/실패: {"reachable": False, "error": 사유}
    """
    # [방어 1] 비밀번호 환경변수가 없으면 SSH 점검을 fail-soft 로 건너뛴다.
    # 비밀번호를 소스에 두지 않는 정책이라, 환경변수가 없으면 GPU 점검은 하지 않는다.
    if not GPU_PASS:
        return {
            "reachable": False,
            "error": "NEXUS_DIAG_GPU_PASS 미설정으로 GPU SSH 점검 생략",
        }

    # [방어 2] paramiko(순수 파이썬 SSH 라이브러리)가 없으면 역시 건너뛴다.
    # 함수 안에서 import 하는 이유: 이 점검을 안 할 때는 굳이 로딩하지 않으려는 것.
    try:
        import paramiko
    except ImportError:
        return {"reachable": False, "error": "paramiko 미설치 — GPU 점검 건너뜀"}

    def _run(ssh: Any, cmd: str, timeout: int = 15) -> str:
        """열린 SSH 세션으로 원격 명령 하나를 실행하고 그 출력을 문자열로 모아 준다.

        표준출력(stdout)을 기본으로 하되, 표준에러(stderr)에 내용이 있으면
        "[stderr]" 구분선과 함께 덧붙여 준다 — 진단 시 에러 메시지도 놓치지 않기 위해서다.
        """
        # exec_command 는 (stdin, stdout, stderr) 세 스트림을 준다. stdin 은 안 쓴다.
        _, stdout, stderr = ssh.exec_command(cmd, timeout=timeout)
        out = stdout.read().decode(errors="replace")
        err = stderr.read().decode(errors="replace")
        # stderr 에 실제 내용이 있을 때만 뒤에 붙이고, 앞뒤 공백은 정리해 반환한다.
        return (out + (("\n[stderr]\n" + err) if err.strip() else "")).strip()

    ssh = paramiko.SSHClient()
    # 처음 접속하는 호스트의 키를 자동 수락한다. 일반적으론 위험하지만 대상이
    # LAN 내부 고정 호스트(에어갭)라 안전하다고 판단해 완화한다.
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())  # noqa: S507 — LAN 내부 고정 호스트(에어갭)
    try:
        # 비밀번호 인증으로 SSH 접속(10초 타임아웃).
        ssh.connect(GPU_HOST, username=GPU_USER, password=GPU_PASS, timeout=10)
    except Exception as e:  # noqa: BLE001 — paramiko 예외 계층이 넓어 결과 dict 로 환원
        # 인증 실패·네트워크 오류 등 paramiko 예외 종류가 매우 다양해, 넓게 잡아
        # 서버를 죽이지 않고 사유만 담아 돌려준다.
        return {"reachable": False, "error": f"SSH 실패: {type(e).__name__}: {e}"}

    try:
        # (1) vLLM(OpenAI 호환 서버) 프로세스가 떠 있는지 — pgrep 로 명령줄까지 확인.
        vllm_proc = _run(ssh, "pgrep -af 'vllm.entrypoints.openai' || echo '(없음)'")
        # (2) 추론/보조 포트(8001, 8003)가 LISTEN 중인지 — ss 출력에서 해당 포트만 필터.
        listen = _run(ssh, "ss -tln 2>/dev/null | awk 'NR==1 || /:(8001|8003)\\>/' || true")
        # (3) GPU 메모리 사용량/총량 — nvidia-smi 로 "used,total" 을 CSV 로 뽑는다.
        gpu_mem = _run(
            ssh,
            "nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader",
        )
        # (4) 현재 로드된 모델 목록 — vLLM /v1/models 를 curl 로 치고, 그 JSON 에서
        #     id 만 뽑아 리스트로 출력한다. 응답이 없으면 '(응답 없음)' 을 남긴다.
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
            "vllm_process": vllm_proc,        # vLLM 프로세스 존재/명령줄
            "listen_8001_8003": listen,       # 8001/8003 포트 LISTEN 여부
            "gpu_memory": gpu_mem,            # GPU 메모리 used/total
            "vllm_models": models,           # 로드된 모델 id 목록
        }
    finally:
        ssh.close()  # 성공/실패와 무관하게 SSH 세션은 반드시 닫는다.


def _embed_query_blocking(query_text: str) -> tuple[list[float], float]:
    """임베딩 서버에 텍스트 하나를 보내 임베딩 벡터와 왕복 지연을 함께 측정한다.

    rag_latency 도구가 "임베딩 단계가 얼마나 느린지"를 재는 데 쓴다. 벡터 자체는
    이어지는 DB 벡터 검색(EXPLAIN ANALYZE)에도 재사용된다.

    Args:
        query_text: 임베딩할 문자열(호출부에서 "query: ..." 접두어를 붙여 넘긴다).

    Returns:
        (embedding, elapsed_ms) 튜플.
          - embedding: float 리스트 형태의 임베딩 벡터.
          - elapsed_ms: 요청~응답까지 걸린 시간(밀리초).
    """
    # 요청 본문을 JSON 으로 만들고 바이트로 인코딩한다. 서버는 {"texts": [...]} 형식을 받는다.
    body = json.dumps({"texts": [query_text]}).encode()
    req = urllib.request.Request(  # noqa: S310 — LAN 192.168 고정 URL(에어갭)
        f"{EMBED_URL}/v1/embed",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    # perf_counter 로 요청 직전 시각을 찍는다(고해상도 단조 증가 시계 — 지연 측정에 적합).
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=15) as resp:  # noqa: S310 — LAN 192.168 고정 URL
        payload = json.loads(resp.read())  # 응답 JSON 파싱
    elapsed_ms = (time.perf_counter() - t0) * 1000  # 경과 시간을 초→밀리초로 변환
    # 텍스트 하나만 보냈으므로 embeddings 리스트의 첫 번째가 우리가 원한 벡터다.
    return payload["embeddings"][0], elapsed_ms


# ─────────────────────────────────────────────
# ReachabilityTool — "reachability" MCP 도구
# ─────────────────────────────────────────────
class ReachabilityTool(McpServerTool):
    """웹·GPU·DB 세 서버의 도달성을 한 번의 호출로 점검하는 MCP 도구.

    McpServerTool(프레임워크가 정의한 도구 인터페이스)을 상속해, MCP 서버가
    요구하는 name/description/input_schema/call 을 구현한다. 앞서 정의한 블로킹
    점검 함수 3종을 워커 스레드에서 동시에 돌려, 결과를 하나의 dict 로 합쳐 준다.
    """

    @property
    def name(self) -> str:
        """도구의 고유 식별자. 호출 측이 이 이름으로 도구를 지정한다."""
        return "reachability"

    @property
    def description(self) -> str:
        """모델/사용자에게 보여줄 도구 설명. 무엇을 어떻게 점검하는지 요약한다."""
        return (
            "Nexus 인프라(로컬 웹 서버, GPU 서버, DB 서버)의 도달성을 점검한다. "
            "웹은 HTTP(S), GPU 는 SSH, DB 는 TCP 포트로 확인하고 서비스별 상태를 반환한다."
        )

    @property
    def input_schema(self) -> dict[str, Any]:
        """도구 입력의 JSON Schema. 인자 없음 — 대상이 고정된 LAN 서버뿐이라서다."""
        return {"type": "object", "properties": {}}

    async def call(self, arguments: dict[str, Any]) -> Any:
        """웹/DB/GPU 세 점검을 워커 스레드에서 "동시에" 수행한다.

        세 함수 모두 블로킹 I/O 라 to_thread 로 스레드에 넘기고, asyncio.gather 로
        묶어 병렬 대기한다. 이렇게 하면 세 점검이 순차가 아니라 겹쳐 진행돼
        전체 소요가 "가장 느린 하나"에 수렴한다(이벤트 루프도 막지 않는다).

        Args:
            arguments: 입력 인자(이 도구는 사용하지 않는다).

        Returns:
            {"web": {...}, "db": {...}, "gpu": {...}} — 각 서버별 점검 결과.
        """
        web, db, gpu = await asyncio.gather(
            asyncio.to_thread(_check_web_blocking),
            asyncio.to_thread(_check_db_blocking),
            asyncio.to_thread(_check_gpu_blocking),
        )
        return {"web": web, "db": db, "gpu": gpu}


# ─────────────────────────────────────────────
# RagLatencyTool — "rag_latency" MCP 도구
# ─────────────────────────────────────────────
class RagLatencyTool(McpServerTool):
    """RAG(검색증강) 파이프라인의 지연을 측정하는 MCP 도구.

    RAG 는 크게 (1) 질의를 임베딩 벡터로 바꾸고 (2) 그 벡터로 DB 에서 유사 문서를
    찾는 두 단계로 나뉜다. 이 도구는 두 단계의 지연을 각각 재서 병목이 어디인지 짚어 준다.

    DB 풀(pg_pool)이 주입돼 있으면 tb_knowledge 벡터 검색의 EXPLAIN ANALYZE 와 실측
    지연까지 재고, 없으면 임베딩 서버 지연만 측정한다(부분 결과 — fail-soft).
    """

    def __init__(self, pg_pool: Any | None) -> None:
        """DB 커넥션 풀을 주입받아 보관한다.

        Args:
            pg_pool: asyncpg 커넥션 풀. DB 연결에 실패했으면 None 이 들어오며,
                     그 경우 이 도구는 임베딩 지연만 측정하는 축소 모드로 동작한다.
        """
        self._pg = pg_pool

    @property
    def name(self) -> str:
        """도구의 고유 식별자."""
        return "rag_latency"

    @property
    def description(self) -> str:
        """모델/사용자에게 보여줄 도구 설명."""
        return (
            "RAG 파이프라인 지연을 진단한다. 임베딩 서버 호출 지연을 측정하고, "
            "DB 연결이 가능하면 tb_knowledge 벡터 검색의 EXPLAIN ANALYZE 와 "
            "실측 지연을 함께 반환한다."
        )

    @property
    def input_schema(self) -> dict[str, Any]:
        """도구 입력의 JSON Schema. 두 인자 모두 선택(생략 시 기본값 사용)이다."""
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
        """임베딩 지연을 재고, DB 가 있으면 벡터 검색 EXPLAIN ANALYZE 까지 측정한다.

        흐름:
          0) 입력 검증 — query 기본값 채우고, top_k 가 1 이상 정수인지 확인.
          1) 임베딩 지연 측정(블로킹이라 워커 스레드로).
          2) DB 풀이 없으면 여기서 임베딩 결과만 돌려주고 종료.
          3) DB 풀이 있으면 tb_knowledge 벡터 검색의 실행계획과 실측 지연을 잰다.

        Args:
            arguments: {"query": 샘플질의(선택), "top_k": 상위 K(선택, 기본 5)}.

        Returns:
            {
              "query": str,
              "embed_latency_ms": float,
              "embed_dim": int,
              "search": {... EXPLAIN/지연 ...} | {"skipped": "사유"} | {"error": ...},
            }
        """
        # query 미지정이면 임의의 한국어 샘플로 대체한다(빈 문자열도 기본값으로 흡수).
        query_text = arguments.get("query") or "이기이원론"
        # top_k 는 미지정 시에만 기본값 — `or` 관용구는 0 을 falsy 로 보아
        # 기본값으로 조용히 치환하므로, get(키, 기본값) 으로 명시적으로 처리한다.
        # (top_k=0/음수는 잘못된 입력이므로 기본값 대체가 아니라 거부해야 한다.)
        top_k = arguments.get("top_k", 5)
        # bool 은 int 의 하위형이라 True/False 가 정수로 통과하는 함정이 있어 따로 배제한다.
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
            raise ValueError("top_k 는 1 이상의 정수여야 합니다.")

        # 1) 임베딩 지연 측정(블로킹 → 워커 스레드).
        #    "query: " 접두어는 e5 계열 임베딩 모델이 질의를 구분하도록 요구하는 관례다.
        try:
            embedding, embed_ms = await asyncio.to_thread(
                _embed_query_blocking, f"query: {query_text}"
            )
        except (OSError, ValueError, KeyError) as e:
            # 네트워크 오류·JSON 파싱 실패·응답 키 누락을 하나로 묶어 명확한 에러로 올린다.
            raise RuntimeError(f"임베딩 서버 호출 실패: {type(e).__name__}: {e}") from e

        # 임베딩 단계 결과를 먼저 채운다. embed_dim(벡터 차원)은 임베딩 모델 확인에 유용.
        result: dict[str, Any] = {
            "query": query_text,
            "embed_latency_ms": round(embed_ms, 1),
            "embed_dim": len(embedding),
        }

        # 2) DB 풀이 없으면(연결 실패 등) 여기서 임베딩 결과만 돌려준다(fail-soft).
        if self._pg is None:
            result["search"] = {"skipped": "pg_pool 없음 — 임베딩 지연만 측정"}
            return result

        # 3) 벡터 검색 측정 준비 — 임베딩 리스트를 pgvector 리터럴 "[v1,v2,...]" 로 만든다.
        #    소수점 6자리로 고정해 SQL 문자열 길이를 안정화한다.
        vec_literal = "[" + ",".join(f"{v:.6f}" for v in embedding) + "]"
        # (a) 실행계획 확인용 — 코사인 거리(<=>) 정렬 + 인덱스 사용 여부를 EXPLAIN 으로 본다.
        explain_sql = (
            "EXPLAIN (ANALYZE, BUFFERS) "
            "SELECT id, title, 1 - (embedding <=> $1::vector) AS sim "
            "FROM tb_knowledge WHERE embedding IS NOT NULL "
            "ORDER BY embedding <=> $1::vector LIMIT $2"
        )
        # (b) 실측 지연용 — (a)와 동일한 검색을 EXPLAIN 없이 실제로 돌려 시간을 잰다.
        search_sql = (
            "SELECT id, title, 1 - (embedding <=> $1::vector) AS sim "
            "FROM tb_knowledge WHERE embedding IS NOT NULL "
            "ORDER BY embedding <=> $1::vector LIMIT $2"
        )
        try:
            # 풀에서 커넥션 하나를 빌려(acquire) 두 쿼리를 같은 연결에서 실행한다.
            async with self._pg.acquire() as conn:
                # 먼저 실행계획을 뽑는다($1=벡터 리터럴, $2=top_k).
                plan_rows = await conn.fetch(explain_sql, vec_literal, top_k)
                # 실제 검색을 돌려 왕복 시간을 측정한다(결과 행 자체는 버린다).
                t0 = time.perf_counter()
                _ = await conn.fetch(search_sql, vec_literal, top_k)
                search_ms = (time.perf_counter() - t0) * 1000
            result["search"] = {
                "search_latency_ms": round(search_ms, 1),
                # EXPLAIN 은 여러 행의 텍스트로 오므로 각 행의 첫 컬럼만 모아 리스트로 만든다.
                "explain_analyze": [r[0] for r in plan_rows],
            }
        except Exception as e:  # noqa: BLE001 — asyncpg 예외 계층 폭넓음 → 결과에 환원
            # DB 쿼리 실패도 서버를 죽이지 않고 search 필드에 사유만 담아 부분 결과를 유지한다.
            result["search"] = {"error": f"{type(e).__name__}: {e}"}

        return result


async def build_app(api_key: str = "local-key") -> FastAPI:
    """diag MCP 서버의 FastAPI 앱을 조립해 반환하는 진입점.

    하는 일:
      1) 설정을 로드한다(DB 접속 정보 등).
      2) DB 커넥션 풀(asyncpg)을 best-effort 로 만든다 — 실패해도 계속 진행한다.
      3) 두 도구(Reachability/RagLatency)를 등록하고, DB 풀이 있으면 종료 훅을 단다.

    DB 연결이 best-effort 인 이유: DB 가 죽어 있어도 reachability 점검과 임베딩 지연
    측정은 여전히 가치가 있다. 그래서 pg_pool 이 None 이어도 앱을 정상 반환한다(fail-soft).

    Args:
        api_key: MCP 서버 인증 키. 기본값은 로컬 개발용 "local-key".

    Returns:
        요청을 받을 준비가 된 FastAPI 애플리케이션.
    """
    # 설정 로더는 core 에 의존한다. 함수 안에서 import 해 모듈 로드 시점의 의존성을 줄인다.
    from core.config import load_and_validate_config

    config = load_and_validate_config()

    # DB 풀은 "있으면 좋고 없어도 되는" 선택 자원이므로 None 으로 시작한다.
    pg_pool: Any | None = None
    try:
        import asyncpg

        # 설정값으로 커넥션 풀을 만든다. min~max 로 연결 수를 제한하고 10초 타임아웃을 건다.
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
        # asyncpg 자체가 없으면 DB 검색은 포기하고 임베딩 지연만 측정하는 모드로 간다.
        logger.warning("diag MCP 서버: asyncpg 미설치 — rag_latency 는 임베딩만 측정")
    except Exception as e:  # noqa: BLE001 — 연결 실패는 치명적 아님(부분 진단 유지)
        # 접속 실패(서버 다운·인증 오류 등)도 치명적이지 않다. 경고만 남기고 진행한다.
        logger.warning("diag MCP 서버: PostgreSQL 연결 실패(부분 진단): %s", e)

    # pg_pool 이 실제로 만들어졌을 때만 종료 훅을 등록한다(앱 lifespan 종료 시 호출).
    # 이렇게 해야 서버가 내려갈 때 커넥션 풀이 깔끔히 닫혀 자원 누수를 막는다.
    hooks: list[Any] = []
    if pg_pool is not None:

        async def _close_pool() -> None:
            """앱 종료 시 호출되는 정리 훅 — DB 풀을 닫는다."""
            await pg_pool.close()
            logger.info("diag MCP 서버: PostgreSQL 풀 종료")

        hooks.append(_close_pool)

    # 프레임워크 헬퍼로 최종 앱을 만든다. 두 도구를 등록하고, RagLatencyTool 에는
    # (있을 수도 없을 수도 있는) DB 풀을 주입한다.
    return create_mcp_app(
        tools=[ReachabilityTool(), RagLatencyTool(pg_pool)],
        api_key=api_key,
        title="Nexus Diagnostics MCP Server",
        shutdown_hooks=hooks,
    )
