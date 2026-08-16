"""
FastAPI 웹 인터페이스 — Nexus HTTP API 서버.

[이 파일이 하는 일]
CLI(터미널) 말고도 HTTP/SSE로 Nexus를 사용할 수 있게 해 주는 웹 진입점이다.
브라우저 채팅 UI, 외부 시스템(AgentHub·.NET·LangChain 등 OpenAI 클라이언트),
관리/모니터링(도구·모델·테넌트·메트릭 조회)이 모두 이 서버를 통해 들어온다.
핵심 책임은 세 가지다:
  1) HTTP 요청을 받아 core의 QueryEngine(4-Tier 오케스트레이터)에 전달하고,
     QueryEngine이 yield하는 StreamEvent를 응답(JSON/SSE)으로 변환한다.
  2) 요청/세션별로 '격리된' QueryEngine을 조립해 멀티테넌트 동시 요청이
     서로의 대화·테넌트를 오염시키지 않게 한다(감사 Critical #5 수정).
  3) 세션 히스토리를 Redis(단기)·JSONL 트랜스크립트(영구)와 오가며 복원/저장한다.

[의존성 방향] web/ → core/ (단방향). 이 파일은 core를 import하지만 core는 web을
절대 import하지 않는다(아키텍처 규칙 P2). GPU/vLLM 직접 호출도 없다(전부 core 경유).

[주요 헬퍼/구성요소]
  - _strip_thinking / _sanitize_history_inplace : Qwen3.5 <think> 찌꺼기 정제
  - _extract_download / _collect_downloads      : 문서 생성 도구의 다운로드 URL 추출
  - _resolve_tenant                             : 요청→TenantConfig 해석(멀티테넌시)
  - _build_web_engine_parts / _assemble_session_engine
        : 무거운 '공유 부품'을 한 번만 만들고, 세션 전용 상태만 가볍게 격리 조립
  - _acquire_session_engine / _get_session_lock : 세션별 엔진 획득 + 세션 락 직렬화
  - Pydantic 요청/응답 모델(ChatRequest, ChatResponse, OpenAI* 등)

[제공 엔드포인트]
  POST /v1/chat          — 비스트리밍 채팅(모든 StreamEvent를 모아 한 번에 응답)
  POST /v1/chat/stream   — SSE 스트리밍 채팅(이벤트를 실시간 전송)
  POST /v1/chat/completions — OpenAI 호환 채팅(비스트림 JSON / stream=true SSE)
  GET  /v1/sessions      — 세션 목록 조회 (Redis + 트랜스크립트 병합)
  GET  /v1/sessions/{session_id}/messages — 특정 세션 대화 복원 (Ch 16)
  DELETE /v1/sessions/{session_id} — 특정 세션 삭제 (Redis + 트랜스크립트)
  GET  /v1/tools         — 도구 목록 조회(웹 Worker가 실제로 보는 풀, MCP 포함)
  GET  /v1/models        — 모델 목록 조회
  GET  /v1/tenants       — 테넌트 목록 조회 (멀티테넌시, Part 5 Ch 15)
  GET  /health           — 헬스체크(오케스트레이터 + GPU 서버 상태)
  GET  /metrics          — 메트릭스 조회(HTTP/세션/MCP/에이전트/테넌트 통계)
  POST /v1/upload        — 문서 분석용 파일 업로드
  GET  /v1/download/{filename} — DocumentExport가 생성한 문서 다운로드
  GET  /                 — 채팅 UI(정적 index.html) 서빙

작성자: 이현수 / 작성일: 2026-07-05
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import time
import uuid
from collections import OrderedDict
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# ─────────────────────────────────────────────
# thinking 태그/찌꺼기 정제 헬퍼
# ─────────────────────────────────────────────
# Qwen3.5 chat template이 <think>...</think> 블록을 삽입할 수 있다.
# enable_thinking=false로 대부분 예방되지만, 과거 세션이나 예외 상황에서
# 찌꺼기가 들어오면 다음 턴에서 Worker가 그 스타일을 모방할 수 있다.
_THINK_BLOCK = re.compile(r"<think>.*?</think>", flags=re.DOTALL)
_DANGLING_THINK_TAIL = re.compile(r"^.*?</think>\s*", flags=re.DOTALL)


def _strip_thinking(text: str) -> str:
    """<think>...</think> 블록 + 비정상 잘린 </think> 접두를 제거한다."""
    if not text:
        return text
    cleaned = _THINK_BLOCK.sub("", text)
    # 여는 <think> 없이 닫는 </think>만 남은 경우(스트리밍 중단 등)
    if "</think>" in cleaned and "<think>" not in cleaned:
        cleaned = _DANGLING_THINK_TAIL.sub("", cleaned)
    return cleaned.strip()


def _sanitize_history_inplace(history: list) -> None:
    """히스토리에 저장된 Message 중 thinking 찌꺼기가 있으면 정제된 Message로 교체."""
    from core.message import Message

    for i, msg in enumerate(history):
        role = msg.role if isinstance(msg.role, str) else msg.role.value
        if role not in ("user", "assistant"):
            continue
        content = msg.text_content if hasattr(msg, "text_content") else str(msg.content)
        if content and ("<think>" in content or "</think>" in content):
            cleaned = _strip_thinking(content)
            if cleaned:
                history[i] = (
                    Message.assistant(cleaned) if role == "assistant" else Message.user(cleaned)
                )


from fastapi import FastAPI, Header, HTTPException, Request, Response, UploadFile
from fastapi.exceptions import RequestValidationError  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# 지식 RAG 출처 인용(Point 4-2) — ChatResponse.sources 필드가 참조하는 출처 모델.
# core/message는 pydantic/표준만 의존하는 경량 모듈이라 안전하다(web → core 순방향,
# 순환 없음). 이 파일의 다른 import처럼 코드 뒤에 오므로(E402는 파일 전반의 기존
# 사항) noqa로 표기해 신규 lint를 만들지 않는다.
from core.message import KnowledgeCitation  # noqa: E402
from core.system_prompt.compose import compose_system_prompt  # noqa: E402
from web.middleware import ApiKeyAuthMiddleware, CORSConfig, RequestLoggingMiddleware

logger = logging.getLogger("nexus.web.app")

# DocumentExport 등 파일 생성 도구의 결과에서 다운로드 URL을 뽑는 정규식.
# 왜: 모델이 URL을 자유 텍스트로 다시 적을 때 uuid 한 글자를 틀리는 등 오탈자가
# 날 수 있다(실측). 그래서 "도구가 만든 정확한 URL"을 결과 본문에서 서버가 뽑아
# UI에 구조화(download 이벤트/필드)로 전달한다 — 모델 텍스트에 의존하지 않는다.
_DOWNLOAD_URL_RE = re.compile(r"/v1/download/[^\s)\]\"']+")


def _extract_download(content: str | None) -> dict[str, str] | None:
    """도구 결과 본문에서 다운로드 URL을 찾아 {url, filename} 으로 돌려준다(없으면 None).

    TODO(nexus): 정식으로는 도구 결과 본문 정규식이 아니라 ToolResult.metadata의
      download_url을 이벤트로 받아 쓰는 게 맞다(정규식 파싱 제거). TOOL_RESULT
      StreamEvent 노출 리팩터에서 함께 정리한다.
    """
    if not content:
        return None
    m = _DOWNLOAD_URL_RE.search(content)
    if not m:
        return None
    url = m.group(0)
    return {"url": url, "filename": url.rsplit("/", 1)[-1]}


# DocumentExport 계열 도구 이름(별칭 포함) — tool_use 입력에서 미리보기 content를 찾을 때 사용.
_DOC_EXPORT_NAMES = {"DocumentExport", "GenerateDocument", "SaveAs", "ExportDocument"}


def _collect_downloads(messages: list) -> list[dict[str, str]]:
    """
    이번 턴 메시지에서 생성 문서의 다운로드 정보를 모은다.

    - url/filename: tool_result 메시지에서 정규식으로 정확히 추출(모델 텍스트 오탈자 무관).
    - content/format: 대응하는 DocumentExport tool_use 입력에서 가져와 UI 미리보기(캔버스)에 쓴다.
      (tool_use_id로 tool_result ↔ tool_use 를 짝짓는다.)
    """
    # 1) tool_use_id → DocumentExport 입력(content/format) 매핑
    doc_inputs: dict[str, dict] = {}
    for msg in messages:
        for tub in getattr(msg, "tool_use_blocks", []):
            if getattr(tub, "name", "") in _DOC_EXPORT_NAMES:
                doc_inputs[tub.id] = tub.input or {}

    # 2) tool_result 메시지에서 URL 추출 + 입력에서 미리보기 content 결합
    out: list[dict[str, str]] = []
    seen: set[str] = set()
    for msg in messages:
        role = msg.role if isinstance(msg.role, str) else msg.role.value
        if role != "tool_result" or not isinstance(msg.content, str):
            continue
        dl = _extract_download(msg.content)
        if not dl or dl["url"] in seen:
            continue
        seen.add(dl["url"])
        inp = doc_inputs.get(getattr(msg, "tool_use_id", None), {})
        dl["content"] = inp.get("content", "") or ""  # 미리보기용 마크다운 본문
        dl["format"] = inp.get("format", "") or ""
        out.append(dl)
    return out


def _answer_warnings_for(answer: str, messages: list) -> str:
    """답변에 덧붙일 사후 검증 경고를 만든다(숫자 인용 + 리터럴 표기 + 실행 주장).

    실제 판단은 core/verification/post_check.py 가 한다. 웹과 CLI 가 같은 검증을
    받아야 하는데 CLI 는 web/ 을 import 할 수 없으므로(의존성 방향), 로직을 core 로
    내리고 여기서는 호출만 한다. 새 검증기가 늘어도 이 자리는 그대로다.
    """
    from core.verification.post_check import build_answer_warnings

    return build_answer_warnings(answer, messages)


def _uploads_dir() -> Any:
    """설정된 업로드 디렉토리를 돌려준다(없으면 런타임 폴백).

    [왜 헬퍼로 뺐나 — 2026-08-08]
      `resolve_uploads_dir(configured)` 는 설정값을 받을 수 있게 만들어져 있었는데,
      웹의 세 호출부가 **전부 인자 없이** 부르고 있었다. 그래서 설정에 무엇을 넣든 항상
      `{tempdir}/nexus_uploads` 로 폴백했다 — 컨테이너에서는 `/tmp` 라 재시작하면
      업로드가 통째로 사라진다(실측).

      호출부마다 config 를 꺼내 쓰게 두면 또 한 곳을 빠뜨린다. 한 자리에서만
      결정하게 한다(사후 검증기를 post_check 하나로 묶은 것과 같은 이유).
    """
    from core.config import UploadConfig
    from core.tools.implementations.analyze_image_tool import resolve_uploads_dir

    upload_cfg = getattr(_app_state.get("config"), "upload", None) or UploadConfig()
    return resolve_uploads_dir(getattr(upload_cfg, "uploads_dir", "") or None)


def _vision_max_bytes() -> int:
    """비전 입력 크기 상한 — **업로드 상한과 같은 값**을 단일 출처에서 읽는다.

    [왜 하나로 묶었나 — 2026-08-12]
      업로드는 20MB 를 받는데 AnalyzeImage 는 10MB 로 폴백하고 있었다
      (`vision_max_image_mb` 를 아무도 주입하지 않았다). 그래서 15MB 이미지가
      업로드는 성공하고 **분석에서만** 거부됐다 — 사용자에게는 원인이 안 보인다.

      상수를 두 개 두면 반드시 갈라진다. 업로드 상한 하나만 두고 여기서 파생시킨다.
    """
    from core.config import UploadConfig

    upload_cfg = getattr(_app_state.get("config"), "upload", None) or UploadConfig()
    return int(getattr(upload_cfg, "max_size_bytes", 0) or UploadConfig().max_size_bytes)


# sha256 계산 시 파일을 한 번에 읽지 않고 스트리밍하는 청크 크기(64KB).
_SHA256_CHUNK = 64 * 1024
# sha256을 계산할 파일 크기 상한(8MB). 이보다 큰 파일은 성능을 위해 해시를 생략한다.
_SHA256_MAX_BYTES = 8 * 1024 * 1024


def _sha256_file(path: Path) -> str | None:
    """파일 내용의 sha256 16진수 다이제스트를 계산한다(실패 시 None).

    64KB씩 스트리밍으로 읽어 큰 파일도 메모리를 적게 쓴다. 읽기 오류는
    삼키고 None을 돌려 호출부(메타 기록)가 해시 없이도 진행하게 한다.
    """
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(_SHA256_CHUNK), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


async def _record_artifacts(
    downloads: list, tenant: Any, session_id: str | None, turn: int | None = None
) -> None:
    """생성물(다운로드 확정분)의 메타데이터를 tb_artifacts에 fail-soft로 기록한다.

    [왜 web 계층에서 기록하는가]
      생성물이 "다운로드 URL"로 확정되는 지점(_collect_downloads)이 web이다.
      도구(image_generate/document_export)는 DB를 import하지 않는다(의존성 단방향
      유지 — core/tools/** → DB 금지). 그래서 저장/조회는 web + core/storage에서만
      한다. 결과적으로 CLI 경로는 web을 안 거치므로 자동으로 미기록된다(정상).

    [무엇을 남기는가 — 바이트는 파일시스템, 메타만 PG]
      각 다운로드에 대해 파일시스템(exports_dir)에서 크기(stat)와 sha256(선택)을
      뽑고, 확장자→MIME 매핑(MEDIA_TYPES 재사용)으로 mime을 정한 뒤 record_artifact로
      멱등 기록한다. tenant는 요청에서 해석된 TenantConfig의 id를 소유자로 쓴다.

    [fail-soft]
      pg_pool이 없으면 조용히 스킵한다. 개별 파일 stat/해시 오류도 삼켜, 메타 기록
      실패가 채팅 응답을 절대 깨뜨리지 않게 한다(가용성 우선).
    """
    if not downloads:
        return
    pool = _app_state.get("pg_pool")
    if pool is None:
        return

    # lazy import — 순환 import 방지 + DB가 없을 때 비용 회피.
    from core.storage.artifacts import record_artifact
    from core.tools.implementations.document_export_renderers import MEDIA_TYPES
    from core.tools.implementations.document_export_tool import resolve_exports_dir

    config = _app_state.get("config")
    configured = getattr(getattr(config, "document_export", None), "exports_dir", "")
    try:
        exports_dir = resolve_exports_dir(configured).resolve()
    except Exception as e:  # noqa: BLE001 — 경로 해석 실패 시 기록만 생략(무회귀)
        logger.warning("생성물 기록 스킵 — exports_dir 해석 실패: %s", e)
        return

    tenant_id = getattr(tenant, "id", None)
    for dl in downloads:
        filename = dl.get("filename") if isinstance(dl, dict) else None
        if not filename:
            continue
        # 다운로드 라우트와 동일한 경로 순회 방어: basename만 취하고 exports_dir 밖은 배제.
        safe_name = Path(filename).name
        target = (exports_dir / safe_name).resolve()
        if target.parent != exports_dir:
            continue

        size_bytes: int | None = None
        sha256: str | None = None
        try:
            if target.is_file():
                size_bytes = target.stat().st_size
                # 큰 파일은 sha256을 생략한다(성능). 상한 이하만 계산.
                if size_bytes is not None and size_bytes <= _SHA256_MAX_BYTES:
                    sha256 = _sha256_file(target)
        except OSError:
            # 파일 stat 실패는 치명적이지 않다 — 크기/해시 없이 메타만 남긴다.
            pass

        ext = target.suffix.lstrip(".").lower()
        mime = MEDIA_TYPES.get(ext, "application/octet-stream")
        await record_artifact(
            pool,
            filename=safe_name,
            tenant_id=tenant_id,
            session_id=session_id,
            mime=mime,
            size_bytes=size_bytes,
            sha256=sha256,
            turn=turn,
        )


async def _attach_session_artifacts(
    messages: list[dict[str, Any]], session_id: str
) -> list[dict[str, Any]]:
    """히스토리 복원 시 세션 생성물(이미지·문서)을 각 assistant 메시지에 되붙인다.

    [왜 필요한가]
      스트리밍 중에는 download 프레임으로 이미지 썸네일·다운로드 버튼을 그리지만,
      그 프레임은 트랜스크립트에 남지 않는다(assistant 텍스트만 저장). 그래서
      세션을 다시 열거나 새로고침하면 생성물이 사라진다. 여기서 tb_artifacts에
      기록해 둔 생성물을 세션 기준으로 끌어와, turn 값으로 정확한 메시지에 매칭해
      `downloads` 필드로 실어 준다(프론트가 라이브 때와 동일하게 렌더).

    [매칭 규칙 — 2026-08-08 turn → 시각 기준으로 교체]
      각 생성물을 **생성 시각 직전의 assistant 메시지**에 붙인다. 즉 생성물의
      created_at보다 작거나 같은 ts 중 가장 큰 것을 고른다.

      왜 turn을 버렸나: turn은 **항상 1이다.** 웹은 요청마다 엔진을 새로 만들어
      턴 카운터가 매번 리셋되기 때문이다. 실측으로 확인했다 —

        transcript : 10:50:18 turn=1 user/assistant  (이미지 요청)
                     10:50:20 turn=1 user/assistant  (문서 요청)
        tb_artifacts: 사과.png turn=1 / document.docx turn=1

      그래서 turn 매칭은 **성립할 수가 없었고**, 전부 "매칭 실패" 경로로 떨어져
      마지막 assistant 메시지 한 줄에 뭉쳤다(사용자 관측 C2). 시각은 두 기록 모두
      단조 증가하므로 신뢰할 수 있다.

      - ts가 없는 경우: Redis 복원 경로는 ts를 버린다(직렬화가 role/content만 담음).
        그때는 durable 기록인 트랜스크립트에서 ts를 보충한다. 길이가 다르면
        **뒤에서부터 맞춘다** — Redis는 최신 구간만 들고 있을 수 있어서다.
      - 그래도 자리를 못 정한 생성물(fork 상속분 등 created_at 없음)은 종전대로
        마지막 assistant 메시지에 모아 붙인다(가용성 우선 — 하나도 잃지 않는다).

    fail-soft: pg_pool이 없거나 생성물이 없으면 messages를 그대로 돌려준다.
    """
    pool = _app_state.get("pg_pool")
    if not messages:
        return messages
    from core.memory.transcript import read_session_meta
    from core.storage.artifacts import list_session_artifacts

    # 이 세션이 직접 만든 생성물(tb_artifacts, session_id 기준).
    arts: list[dict[str, Any]] = []
    if pool is not None:
        arts = list(await list_session_artifacts(pool, session_id))
    # fork로 상속받은 생성물 — tb_artifacts는 원본 session_id 기준이라 분기 세션엔
    # 레코드가 없다(filename UNIQUE라 재삽입 불가). 그래서 fork 시 원본 생성물 목록을
    # meta.json(inherited_artifacts)에 저장해 두고 여기서 병합한다(T3-1c 한계 해소).
    cfg = _app_state.get("config")
    sessions_dir = cfg.session.sessions_dir if cfg else ".nexus/sessions"
    inherited = read_session_meta(sessions_dir, session_id, channel="web").get(
        "inherited_artifacts"
    ) or []
    for a in inherited:
        if isinstance(a, dict) and a.get("filename"):
            arts.append(
                {"filename": a["filename"], "mime": a.get("mime"), "turn": a.get("turn")}
            )
    if not arts:
        return messages

    def _to_download(a: dict[str, Any]) -> dict[str, str]:
        # 라이브 download 프레임과 동일한 형식({url, filename, format})으로 맞춘다.
        # format은 확장자에서 뽑는다(이미지 판별·배지용). 문서 미리보기 content는
        # tb_artifacts에 없으므로 생략 — 다운로드 버튼·이미지 썸네일은 정상 동작한다.
        fn = a["filename"]
        ext = fn.rsplit(".", 1)[-1].lower() if "." in fn else ""
        return {"url": f"/v1/download/{fn}", "filename": fn, "format": ext}

    assistant_msgs = [m for m in messages if m.get("role") == "assistant"]
    if not assistant_msgs:
        return messages

    # ── assistant 메시지의 시각을 확보한다 ──
    # Redis 복원 경로는 ts를 버리므로(직렬화가 role/content만 담는다) 그때는
    # durable 기록인 트랜스크립트에서 보충한다. 길이가 다르면 뒤에서부터 맞춘다.
    ts_list = [_parse_ts(m.get("ts")) for m in assistant_msgs]
    if all(t is None for t in ts_list):
        from core.memory.transcript import read_transcript_messages

        tr = read_transcript_messages(sessions_dir, session_id, channel="web")
        tr_ts = [_parse_ts(m.get("ts")) for m in tr if m.get("role") == "assistant"]
        if len(tr_ts) >= len(assistant_msgs):
            ts_list = tr_ts[len(tr_ts) - len(assistant_msgs) :]

    # ── 생성물을 "생성 시각 직전의 assistant" 자리에 담는다 ──
    buckets: dict[int, list[dict[str, str]]] = {}
    leftovers: list[dict[str, str]] = []
    for a in arts:
        created = a.get("created_at")
        idx = _index_for_created_at(ts_list, created)
        if idx is None:
            leftovers.append(_to_download(a))
        else:
            buckets.setdefault(idx, []).append(_to_download(a))

    for idx, dls in buckets.items():
        assistant_msgs[idx]["downloads"] = dls

    # 자리를 못 정한 것(fork 상속분 등 created_at 없음)은 마지막에 모아 붙인다.
    if leftovers:
        last = assistant_msgs[-1]
        last["downloads"] = list(last.get("downloads", [])) + leftovers

    return messages


def _parse_ts(value: Any) -> datetime | None:
    """ISO-8601 문자열이나 datetime을 tz-aware datetime으로 정규화한다.

    두 기록의 시각 표현이 다르다 — 트랜스크립트는 ISO 문자열, tb_artifacts는
    asyncpg가 준 datetime이다. 비교하려면 한 종류로 맞춰야 한다. tz가 없는 값은
    UTC로 간주한다(두 기록 모두 UTC로 남긴다).
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _index_for_created_at(
    ts_list: list[datetime | None], created: Any
) -> int | None:
    """생성 시각이 속하는 assistant 메시지의 인덱스를 고른다.

    규칙은 "created_at보다 작거나 같은 ts 중 가장 큰 것". 생성물은 assistant
    기록이 남은 **직후** 삽입되므로(실측 0.5~1ms 차) 이 규칙이 정확히 그 턴을 집는다.

    맞는 자리가 없으면(시각 정보가 없거나 첫 메시지보다 이른 생성물) None을 돌려
    호출부가 레거시 경로로 보내게 한다 — 링크를 잃는 것보다 낫다.
    """
    created_dt = _parse_ts(created)
    if created_dt is None:
        return None
    best: int | None = None
    for i, t in enumerate(ts_list):
        if t is not None and t <= created_dt:
            best = i
    return best


def _build_todo_update_frame(engine: Any, session_id: str) -> dict[str, Any] | None:
    """
    TodoStore에서 현재 (메인 에이전트) 계획 체크리스트를 읽어 todo_update 프레임을 만든다.

    서버가 진실(TodoStore)에서 직접 구조화 목록을 뽑으므로 모델 텍스트 파싱이
    필요 없다(DocumentExport URL 추출과 동일 원칙). 목록이 비었거나 저장소가
    없으면 None을 돌려 프레임을 내보내지 않는다.

    프레임 형식(웹 전용 — StreamEvent 아님):
      {type: "todo_update", session_id, revision, stats, todos: [...]}
    프론트는 revision으로 중복/역순 프레임을 무시하고 #todoPanel을 전체 교체한다.
    """
    ctx = getattr(engine, "_context", None)
    options = getattr(ctx, "options", None)
    store = options.get("todo_store") if isinstance(options, dict) else None
    if store is None:
        return None
    try:
        # 메인 에이전트(agent_id=None) 목록만 UI에 노출한다(서브에이전트 격리).
        state = store.get(session_id, None)
    except Exception as e:  # noqa: BLE001 — 보조 UI 정보라 본류를 막지 않는다
        logger.debug("todo_update 조회 실패 (무시): %s", e)
        return None
    items = list(getattr(state, "items", ()))
    if not items:
        return None
    todos = [i.model_dump(mode="json") for i in items]
    completed = sum(1 for t in todos if t.get("status") == "completed")
    in_progress = sum(1 for t in todos if t.get("status") == "in_progress")
    return {
        "type": "todo_update",
        "session_id": session_id,
        "revision": getattr(state, "revision", 0),
        "stats": {"total": len(todos), "completed": completed, "in_progress": in_progress},
        "todos": todos,
    }


# ─────────────────────────────────────────────
# 도구 활동 표시(접힌 활동라인 + 접힌 요약) 헬퍼 — 웹 UX 개선
# ─────────────────────────────────────────────
# 프론트가 "지금 무슨 작업 중"(활동라인)과 "결과 요약(펼치면 원문)"을 그리기 위한
# 값들을 백엔드가 계산해 SSE 프레임에 실어 보낸다. 아래 두 상한은 큰 결과가
# 프레임을 폭파시키는 것을 막는 방어값이다.
_TOOL_PREVIEW_MAX = 160  # 접힌 라인용 요약 최대 글자수(초과 시 "…")
_TOOL_CONTENT_MAX = 16000  # 펼침용 전체 텍스트 최대 글자수(초과 시 "…(truncated)")


def _compute_tool_desc(tool_use: Any) -> str | None:
    """
    사람이 읽는 도구 라벨(tool_desc)을 계산한다. 없으면 None(프론트는 tool_name으로 폴백).

    규칙:
      - Agent 도구(서브에이전트 호출)면 입력의 subagent_type/description으로 라벨을 만든다.
        · description이 있으면 "{subagent_type}: {description}" (예: "scout: 문서 분석")
        · description이 없으면 "{subagent_type}"만.
        · subagent_type도 없으면(ad-hoc description만 있는 경우) description을 그대로 쓴다.
      - 그 외 도구는 None을 반환한다 → 프론트가 tool_name으로 폴백 표시.

    왜 백엔드에서 계산하나: subagent_type/description은 도구 입력(dict)에 들어 있어
    프론트가 알기 어렵고, "무슨 서브에이전트가 무슨 일을 하는지"를 일관되게 보여주려면
    이벤트가 흐르는 이 지점에서 라벨을 확정해 실어 보내는 것이 가장 정확하다.

    Args:
        tool_use: ToolUseBlock (name, input을 가진 도구 호출 정보).

    Returns:
        표시용 라벨 문자열, 또는 라벨을 만들 수 없으면 None.
    """
    name = getattr(tool_use, "name", "") or ""
    if name != "Agent":
        return None
    inp = getattr(tool_use, "input", None) or {}
    subagent_type = (inp.get("subagent_type") or "").strip()
    description = (inp.get("description") or "").strip()
    if subagent_type and description:
        return f"{subagent_type}: {description}"
    if subagent_type:
        return subagent_type
    if description:
        # subagent_type 없이 ad-hoc description만 준 하위 호환 경로.
        return description
    return None


# ─────────────────────────────────────────────
# 진입점 채널 해석 헬퍼 (히스토리 격리, 2026-08-05)
# ─────────────────────────────────────────────
# 왜 필요한가:
#   /v1/chat·/v1/chat/stream 은 웹 UI 전용으로 만들어져 channel="web"이 하드코딩돼
#   있었다. 그런데 외부 소비자(VSCode 플러그인 등)가 API 키로 이 엔드포인트를 쓰면
#   그 대화가 웹 사용자 히스토리 목록에 그대로 섞여 보인다(2026-08-05 실측).
#   클라이언트가 자기 채널을 선언하면 그 채널로 격리해 이 혼입을 막는다.
#
# 정책(사용자 요구):
#   - web / app  → 같은 "web" 채널. 브라우저와 앱 사용자는 대화 이력을 공유한다.
#   - cli / api  → 각자 독립 채널. 웹 목록·검색·삭제에서 보이지 않는다.
#   - 헤더 없음/모르는 값 → "web" (종전 동작 그대로 = 무회귀).
#
# 보안(중요):
#   channel 값은 그대로 디렉토리 이름({sessions_dir}/{channel}/...)과 Redis 키
#   네임스페이스가 된다. 임의 문자열을 허용하면 "../" 같은 값으로 경로를 벗어날 수
#   있으므로, 반드시 아래 화이트리스트에 있는 값만 통과시킨다(fail-closed).
_CHANNEL_ALIASES: dict[str, str] = {
    "web": "web",
    "app": "web",  # 앱은 웹과 이력을 공유한다(요구사항)
    "cli": "cli",
    "api": "api",
}


def _resolve_channel(header_channel: str | None) -> str:
    """X-Client-Channel 헤더를 검증된 저장 채널로 변환한다.

    Args:
        header_channel: 클라이언트가 선언한 채널("web"/"app"/"cli"/"api").
            None이거나 화이트리스트에 없으면 기본값 "web"으로 떨어진다.
            문자열이 아닌 값(FastAPI 의존성 주입을 거치지 않고 핸들러를 직접
            호출하는 테스트에서는 Header 객체가 그대로 들어온다)도 "web"으로
            안전하게 처리한다.

    Returns:
        "web" | "cli" | "api" 중 하나. 이 값만 경로·키에 쓰이므로 순회가 불가능하다.
    """
    if not header_channel or not isinstance(header_channel, str):
        return "web"
    return _CHANNEL_ALIASES.get(header_channel.strip().lower(), "web")


def _sanitize_client_id(raw: Any) -> str | None:
    """X-Client-Id 헤더를 안전한 식별자로 정규화한다 (2026-08-05).

    왜 필요한가:
      api 채널에는 여러 외부 소비자(VSCode 플러그인·AgentHub 등)의 대화가 함께
      쌓여 어느 것이 누구 것인지 구분할 수 없었다. 클라이언트가 자기 이름을
      선언하면 세션 메타에 남겨 나중에 골라볼 수 있게 한다.

    안전 규칙:
      메타 파일에 그대로 기록되므로 영문자·숫자·`-`·`_`만 남기고 32자로 자른다.
      (경로로 쓰이지는 않지만, 제어문자·개행이 로그와 JSON을 오염시키지 않도록.)

    Returns:
        정규화된 식별자. 값이 없거나 남는 문자가 없으면 None.
    """
    if not raw or not isinstance(raw, str):
        return None
    cleaned = "".join(ch for ch in raw.strip() if ch.isalnum() or ch in "-_")
    return cleaned[:32] or None


def _resolve_query_class(body_value: Any, header_value: Any) -> str | None:
    """요청이 지정한 질의 클래스를 정규화한다 (2026-08-16).

    왜 두 경로인가:
      body 는 프록시를 지나도 안 유실되고 OpenAI SDK 의 extra_body 로 보낼 수 있다.
      헤더는 body 를 못 건드리는 클라이언트용 폴백이다. 둘 다 오면 body 가 이긴다
      (요청 본문이 그 요청의 의도를 더 직접적으로 담는다).

    왜 잘못된 값을 무시하지 않고 400 을 내는가:
      조용히 무시하면 호출자는 "왜 여전히 RAG 가 붙지"의 원인을 영영 못 찾는다.
      이 리포는 같은 이유로 쓸 수 없는 도구를 받았을 때도 400 으로 거부한다.

    Returns:
        대문자 정규화된 클래스, 또는 지정이 없으면 None.

    Raises:
        HTTPException(400): 값이 유효 목록에 없을 때.
    """
    from core.orchestrator.routing import QUERY_CLASSES

    # 문자열이 아니면 "미지정"으로 본다. 엔드포인트 함수를 직접 부르는 테스트에서는
    # 헤더 인자에 FastAPI 의 Header 표식 객체가 들어오기 때문이다(실제 요청에서는
    # 항상 str 또는 None). body 쪽은 Pydantic 이 이미 str|None 으로 강제한다.
    body = body_value.strip() if isinstance(body_value, str) and body_value.strip() else None
    header = (
        header_value.strip() if isinstance(header_value, str) and header_value.strip() else None
    )
    raw = body or header
    if raw is None:
        return None

    value = raw.upper()
    if value not in QUERY_CLASSES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"알 수 없는 query_class: {raw!r}. "
                f"허용값: {', '.join(QUERY_CLASSES)}"
            ),
        )
    return value


def _sanitize_request_id(raw: Any) -> str | None:
    """X-Request-ID 헤더를 로그에 안전한 형태로 정규화한다 (2026-08-13).

    왜 필요한가:
      VSCode 플러그인은 요청마다 UUID를 만들어 `X-Request-ID`로 보내고, 사용자에게도
      그 값을 보여 준다. 그런데 서버가 이 헤더를 **읽지도 남기지도 않아서**, 사용자가
      "요청 ID xxxx가 실패했다"고 알려 와도 로그에서 찾을 수 없었다. 실제로
      2026-08-13 문의에서 두 건 모두 grep 0건이 나와 발생 시각으로 더듬어야 했다.

    안전 규칙:
      로그에 그대로 찍히므로 영문자·숫자·`-`·`_`만 남긴다(개행·제어문자로 로그 한 줄을
      위조하는 것을 막는다). UUID가 36자라 상한은 64자로 둔다.

    Returns:
        정규화된 요청 ID. 값이 없거나 남는 문자가 없으면 None.
    """
    if not raw or not isinstance(raw, str):
        return None
    cleaned = "".join(ch for ch in raw.strip() if ch.isalnum() or ch in "-_")
    return cleaned[:64] or None


# 검증 실패 응답·로그에 실을 오류 개수와 입력값 길이의 상한.
# 왜 자르나: Pydantic 오류의 `input`에는 **요청 본문 값이 그대로** 들어간다.
# 20MB base64 이미지가 들어오면 그게 통째로 로그와 응답으로 되돌아간다(증폭).
_MAX_VALIDATION_ERRORS = 20
_MAX_VALIDATION_INPUT_CHARS = 200


def _summarize_validation_errors(errors: Any) -> list[dict[str, Any]]:
    """Pydantic 검증 오류를 로그·응답에 싣기 안전한 크기로 줄인다 (2026-08-13).

    각 오류에서 `loc`(어느 필드)·`msg`(왜)·`type`(무슨 규칙)만 남기고, `input`은
    길이를 잘라 붙인다. 문자열이 아닌 입력(dict/list 등)은 **repr조차 만들지 않고**
    타입 이름만 남긴다 — 거대한 본문을 repr하는 순간 그 크기만큼 메모리를 쓴다.
    """
    out: list[dict[str, Any]] = []
    for err in list(errors)[:_MAX_VALIDATION_ERRORS]:
        if not isinstance(err, dict):
            continue
        item: dict[str, Any] = {
            "loc": [str(p) for p in err.get("loc", ())],
            "msg": str(err.get("msg", ""))[:300],
            "type": str(err.get("type", "")),
        }
        if "input" in err:
            raw = err["input"]
            if isinstance(raw, str):
                item["input"] = raw[:_MAX_VALIDATION_INPUT_CHARS] + (
                    "…(잘림)" if len(raw) > _MAX_VALIDATION_INPUT_CHARS else ""
                )
            elif raw is None or isinstance(raw, (bool, int, float)):
                item["input"] = str(raw)
            else:
                # dict/list 등 — 크기를 알 수 없으므로 타입만 알린다.
                item["input"] = f"<{type(raw).__name__}>"
        out.append(item)
    return out


def _record_client_meta(
    session_id: str, channel: str, client_id: str, tenant: Any = None
) -> None:
    """세션 메타(meta.json)에 어느 클라이언트가 만든 세션인지 남긴다 (fail-soft).

    api 채널에는 VSCode 플러그인·AgentHub 등 여러 소비자의 대화가 함께 쌓인다.
    테넌트(API 키)로는 구별되지만 저장소에서는 구분이 없어, 나중에 "플러그인
    대화만" 골라내거나 정리할 수 없었다. 이 값이 그 구분자다.

    메타 기록 실패가 대화 자체를 막아서는 안 되므로 예외는 삼키고 로그만 남긴다.
    """
    try:
        from core.memory.transcript import write_session_meta

        cfg = _app_state.get("config")
        sessions_dir = cfg.session.sessions_dir if cfg else ".nexus/sessions"
        write_session_meta(
            sessions_dir,
            session_id,
            {"client": client_id, "tenant": getattr(tenant, "id", None)},
            channel=channel,
        )
    except Exception as e:  # noqa: BLE001 — 부가 기능이므로 절대 요청을 깨뜨리지 않는다
        logger.debug("클라이언트 메타 기록 실패(무시): %s", e)


def _map_finish_reason(stop_reason: Any) -> str:
    """내부 StopReason을 OpenAI finish_reason으로 변환한다 (2026-08-05).

    왜 필요한가:
      기존 OpenAI 호환 응답은 finish_reason을 항상 "stop"으로 하드코딩했다. 그래서
      모델이 출력 토큰 한도에 걸려 **응답이 중간에 잘려도** 클라이언트는 완결된
      응답으로 오해했다(실측: VSCode 플러그인이 잘린 JSON을 받고 파싱 실패).
      OpenAI 규격대로 한도 초과는 "length"로 정직하게 알려, 클라이언트가 재시도·
      분할 요청 같은 대응을 할 수 있게 한다.

    Args:
        stop_reason: core.message.StopReason 또는 그 문자열 값. None이면 "stop".

    Returns:
        "length"(토큰 한도로 잘림) 또는 "stop"(그 외 정상 종료).
    """
    if stop_reason is None:
        return "stop"
    value = getattr(stop_reason, "value", None) or str(stop_reason)
    return "length" if value == "max_tokens" else "stop"


# ─────────────────────────────────────────────
# 멀티테넌시 해석 헬퍼 (Part 5 Ch 15, 2026-04-21)
# ─────────────────────────────────────────────
# 우선순위: body.tenant_id > X-Tenant-ID 헤더 > Authorization Bearer(API 키)
#          > 레지스트리의 default_tenant
def _resolve_tenant(
    body_tenant_id: str | None,
    header_tenant_id: str | None,
    authorization: str | None,
) -> Any:
    """요청 컨텍스트에서 TenantConfig를 해석한다. 항상 유효한 객체 반환."""
    registry = _app_state.get("tenant_registry")
    if registry is None:
        return None

    found = None
    # 1) body
    if body_tenant_id:
        found = registry.get(body_tenant_id)
    # 2) X-Tenant-ID 헤더
    if found is None and header_tenant_id:
        found = registry.get(header_tenant_id)
    # 3) Authorization Bearer — API key 기반
    if found is None and authorization and authorization.lower().startswith("bearer "):
        api_key = authorization[7:].strip()
        if api_key:
            found = registry.resolve_by_api_key(api_key)
    # 4) 기본 테넌트 폴백
    if found is None:
        found = registry.resolve(None)

    # per-tenant 카운트 누적 (M6)
    stats = _app_state.setdefault("tenant_stats", {})
    entry = stats.setdefault(found.id, {"requests": 0})
    entry["requests"] += 1
    return found


def _build_transcript(session_id: str, channel: str | None = None) -> Any:
    """세션 ID별 SessionTranscript 인스턴스를 만든다. config 실패 시 None 반환.

    /v1/chat과 /v1/chat/stream 두 핸들러에서 중복되던 로직을 한 지점으로 모음
    (2026-04-21 리팩토링). channel(web/api)을 주면 트랜스크립트를 채널 하위
    폴더로 격리해 진입점별 히스토리가 서로 안 보이게 한다.
    """
    try:
        from core.memory.transcript import SessionTranscript as _Trans

        cfg = _app_state.get("config")
        sessions_dir = cfg.session.sessions_dir if cfg else ".nexus/sessions"
        transcript_enabled = cfg.session.transcript_enabled if cfg else True
        return _Trans(
            sessions_dir=sessions_dir,
            session_id=session_id,
            enabled=transcript_enabled,
            channel=channel,
        )
    except Exception as e:
        logger.warning("트랜스크립트 생성 실패 (%s): %s", session_id, e)
        return None


# ─────────────────────────────────────────────
# 웹 QueryEngine 조립 헬퍼 (2026-04-21 리팩토링 3)
# ─────────────────────────────────────────────
# lifespan() 한 함수에 뭉쳐 있던 Scout 풀 합치기·시스템 프롬프트 빌드·ModelDispatcher
# 구성을 독립 함수로 분리해 가독성과 테스트 용이성을 확보한다.
def _combine_scout_pool(web_tools: list, scout_tools: list) -> list:
    """웹 도구 + Scout 도구를 name 중복 제거 후 하나의 풀로 합친다."""
    combined: list = []
    seen: set[str] = set()
    for t in [*web_tools, *scout_tools]:
        if t.name not in seen:
            combined.append(t)
            seen.add(t.name)
    return combined


def _load_worker_system_prompt(agent_registry: Any | None, tier: Any = None) -> str:
    """
    Worker 시스템 프롬프트를 하드웨어 티어에 맞춰 로드하고,
    AgentRegistry로부터 서브에이전트 가이드를 동적으로 추가한다(B200 Phase 2).

    티어별 프롬프트 파일:
      - TIER_S       → `web/prompts/worker_system.md`      (현행: Scout 위임)
      - TIER_M/L     → `web/prompts/worker_system_full.md` (탐색 도구 직접 사용)

    tier가 None이거나 알 수 없는 값이면 TIER_S(worker_system.md)로 폴백한다
    = fail-closed(현행 동작 유지). TIER_M/L 파일이 없으면 worker_system.md로,
    그것도 없으면 하드코딩 문자열로 단계적 폴백해 항상 유효한 프롬프트를 보장한다.
    """
    from core.model.hardware_tier import HardwareTier

    # enum이면 .value, 문자열이면 그대로 비교(둘 다 허용). 미매칭 시 TIER_S.
    tier_val = getattr(tier, "value", tier)
    is_expanded = tier_val in (HardwareTier.TIER_M.value, HardwareTier.TIER_L.value)
    fname = "worker_system_full.md" if is_expanded else "worker_system.md"

    prompts_dir = Path(__file__).parent / "prompts"
    prompt_path = prompts_dir / fname
    base: str | None = None
    try:
        base = prompt_path.read_text(encoding="utf-8")
    except OSError as e:
        logger.warning("Worker 프롬프트 파일 읽기 실패 (%s): %s", prompt_path, e)
        # TIER_M/L 전용 파일이 없으면 현행 TIER_S 프롬프트로 폴백.
        if fname != "worker_system.md":
            fallback_path = prompts_dir / "worker_system.md"
            try:
                base = fallback_path.read_text(encoding="utf-8")
                logger.warning("worker_system.md로 폴백: %s", fallback_path)
            except OSError:
                base = None
    if base is None:
        base = (
            "You are IDINO NOVA, the Worker agent developed by IDINO.\n"
            "Respond in the user's language. Be helpful and detailed."
        )

    if agent_registry is not None and len(agent_registry) > 0:
        # TIER_M/L에는 scout 전용 서버(scout_provider)가 없어(bootstrap: TIER_S에서만 생성)
        # scout 위임이 무의미하다 — Worker가 큰 컨텍스트로 직접 처리하는 게 정상 경로다.
        # 그래서 확장 티어에서는 목록·권장 모두에서 scout를 제외한다(불필요한 위임·오류 방지).
        agent_lines = [
            f"  - {name}: {desc}"
            for name, desc in agent_registry.list_descriptions().items()
            if not (is_expanded and name == "scout")
        ]
        base += "\n\n## Sub-agents (Agent tool)\n"
        base += "Delegate specialized tasks to sub-agents via the Agent tool.\n"
        if agent_lines:
            base += "Available sub-agents:\n" + "\n".join(agent_lines) + "\n"
        base += (
            "\nWhen to use sub-agents:\n"
            "  - Simple questions or greetings → answer directly, NO tools\n"
            "  - Editing/creating a known file → use Edit/Write directly "
            "(this web surface has NO Read/Glob/Grep/LS filesystem browsing)\n"
        )
        if is_expanded:
            base += (
                "  - Analyzing an uploaded document or broad exploration → handle it "
                "directly with your tools (large context; no scout delegation needed)\n"
            )
        else:
            base += (
                "  - Analyzing an uploaded document or broad exploration → "
                'Agent(subagent_type="scout")\n'
                "NEVER invoke scout for trivial tasks — it is slow (~30s on CPU).\n"
            )
        base += (
            "When you use any tool or sub-agent, do NOT narrate the tool mechanics "
            "(never mention tool names, arguments, or that you are 'filling in' a parameter); "
            "call it silently and give the user a concise, natural answer."
        )
    return base


def _build_web_engine_parts(components: dict, state: Any) -> dict:
    """
    요청과 무관한(상태 없는) 무거운 부품을 '한 번만' 조립해 모아 반환한다.

    왜 분리하는가 (동시성 결함 수정 — 감사 Critical #5, 2026-07-03):
      기존에는 QueryEngine을 앱 전역 싱글톤 1개로 두고, 모든 HTTP 요청이 그 하나의
      `_messages`/`_session_id`/tenant 상태를 락 없이 덮어써서 두 사용자의 대화·
      테넌트가 뒤섞였다(멀티테넌트 프로덕션 최대 블로커). 이를 고치려면 요청/세션
      별로 **독립된 QueryEngine 상태**를 써야 한다. 다만 도구 레지스트리 생성·시스템
      프롬프트 조립·MCP 머지는 무겁고 '요청과 무관'하므로, 여기서 딱 한 번 만들고
      모든 세션이 공유 재사용한다. 세션별로 격리하는 건 오직 _messages/_session_id/
      tenant/context 뿐이다(→ _assemble_session_engine).

    공유해도 안전한 이유(내부 mutable 상태 없음):
      model_provider·tool 인스턴스·knowledge_retriever·memory_manager 등은 요청별
      가변 상태를 인스턴스 필드에 담지 않고, 호출 시점에 넘어오는 context/인자로
      동작한다. 따라서 여러 세션이 동시에 참조해도 서로를 오염시키지 않는다.
      (반대로 tenant/cwd는 요청마다 다르므로 반드시 세션 context로 격리한다.)
    """
    from core.bootstrap import _create_web_tool_registry

    # 하드웨어 티어를 넘겨 웹 도구 풀을 티어별로 구성한다(B200 Phase 2).
    # TIER_S: 현행 5개, TIER_M/L: +Read/Glob/Grep/LS/DocumentProcess/GitDiff.
    # components에 hardware_tier가 없으면 None → _create_web_tool_registry가
    # TIER_S(5개)로 폴백 = fail-closed.
    web_registry = _create_web_tool_registry(components.get("hardware_tier"))
    web_tools = web_registry.get_all_tools()
    # v7.2 MCP — 부트스트랩이 LAN MCP 서버에서 등록한 도구(mcp__db__query 등)를
    # 웹 Worker 풀에도 흡수한다. 웹은 cli_registry가 아니라 _create_web_tool_registry
    # 로 자체 풀을 만들기 때문에, 이 머지가 없으면 모델이 MCP 도구를 볼 수 없다
    # (실측: Worker가 mcp__db__query를 못 보고 npx 셸 명령을 환각 호출하던 버그).
    # _combine_scout_pool과 동일한 name 중복 제거 규칙으로 합친다(P5 이름순 정렬은
    # registry/get_all_tools가 이미 보장 → prompt cache 안정).
    mcp_tools = components.get("mcp_tools") or []
    if mcp_tools:
        web_tools = _combine_scout_pool(web_tools, mcp_tools)
    scout_tools = components.get("scout_tools") or []
    combined_pool = _combine_scout_pool(web_tools, scout_tools)

    # 컨텍스트 예산(하드코딩 외부화, 2026-07-03). 실 NexusConfig에는 항상 존재하나,
    # 부분 config/테스트 더블에는 없을 수 있어 getattr로 방어(없으면 None → 하위가
    # 현행 상수로 폴백 = 무회귀). fail-closed가 아니라 fail-safe: 예산 미제공이
    # 곧 "현행 기본값 사용"을 의미하므로 엔진 조립을 막지 않는다.
    _web_budgets = getattr(state.config, "context_budgets", None)

    # ── 세션 샌드박스 '입력값'만 보관 ──
    # 실제 cwd 결정은 세션 ID가 정해지는 요청 시점(_session_sandbox_cwd)에서 한다.
    # 예전엔 부트스트랩 시점의 state.session_id 하나로 cwd를 고정해, 모든 세션이
    # 같은 샌드박스를 공유하는 잠재 결함이 있었다(이번 격리로 함께 해소).
    _pe_cfg = getattr(state.config, "permission_enforcement", None)
    _pe_enabled = bool(getattr(_pe_cfg, "enabled", False))
    _sessions_dir = getattr(getattr(state.config, "session", None), "sessions_dir", None)

    # AgentTool·SymbolSearchTool이 해석할 의존성 일체 — tenant를 제외한 '공용' 옵션.
    # 세션별 assemble에서 {**base_options, "tenant": tenant}로 얕은 복제해 격리한다.
    base_options = {
        "memory_manager": components.get("memory_manager"),
        "task_manager": components.get("task_manager"),
        # 계획 체크리스트 저장소 — 웹 TodoWrite/TodoRead 도구와 todo_update 프레임이
        # 같은 (세션 격리) 저장소를 공유하도록 주입한다. 미주입 시 도구는 모듈 전역
        # 폴백을 쓰지만, 그러면 _build_todo_update_frame이 읽지 못해 UI 갱신이 끊긴다.
        "todo_store": components.get("todo_store"),
        "agent_registry": components.get("agent_registry"),
        "model_provider": components["model_provider"],
        "scout_provider": components.get("scout_provider"),
        "available_tools": combined_pool,
        "symbol_store": components.get("symbol_store"),  # Phase 10.0
        # 문서 청크 크기 — 하드코딩 외부화(2026-07-03). DocumentProcess가 읽음.
        # 예산 미제공(None)이면 도구가 CHUNK_SIZE(2500)로 폴백.
        "document_chunk_size": (_web_budgets.document_chunk_size if _web_budgets else None),
        # 문서 통짜 반환 상한 — 이 이하 문서는 청크 없이 1회 반환. None/0이면 비활성
        # → 기존 청크 동작. CLI(bootstrap)와 쌍으로 주입해 표면 간 동작을 일치시킨다.
        "document_singleshot_chars": (
            _web_budgets.document_singleshot_chars if _web_budgets else None
        ),
        # 장기 기억 회상 — QueryEngine._recall_memories 가 읽는다(2026-08-06).
        # CLI(bootstrap)와 쌍으로 주입해 표면 간 동작을 일치시킨다. 기본 비활성이라
        # 미주입/꺼짐이면 기존 동작 그대로다(무회귀).
        "memory_recall": {
            "enabled": getattr(getattr(state.config, "memory", None), "recall_enabled", False),
            "max_items": getattr(getattr(state.config, "memory", None), "recall_max_items", 5),
            "max_chars": getattr(getattr(state.config, "memory", None), "recall_max_chars", 400),
        },
        # 생성 문서 저장 위치 — DocumentExport 도구가 읽는다. 빈 값이면 도구가
        # {tempdir}/nexus_exports 로 폴백(다운로드 라우트와 동일 경로).
        "exports_dir": getattr(getattr(state.config, "document_export", None), "exports_dir", ""),
        # 이미지 생성 서버 주소 — ImageGenerate 도구가 읽는다(config.gpu_server.image_url).
        # 미주입이면 도구가 DEFAULT_IMAGE_URL 로 폴백한다.
        "image_url": getattr(getattr(state.config, "gpu_server", None), "image_url", ""),
        # 비전 서버 주소·모델명 — AnalyzeImage 도구가 읽는다(config.gpu_server.vision_*).
        # 미주입이면 도구가 DEFAULT_VISION_URL/DEFAULT_VISION_MODEL 로 폴백한다.
        "vision_url": getattr(getattr(state.config, "gpu_server", None), "vision_url", ""),
        "vision_model": getattr(getattr(state.config, "gpu_server", None), "vision_model", ""),
        # 업로드 첨부 저장 디렉토리 — AnalyzeImage 가 "이 디렉토리 하위" 이미지만 읽도록
        # 제한하는 기준. /v1/upload 라우트와 같은 _uploads_dir() 로 단일 소스를 공유한다.
        "uploads_dir": str(_uploads_dir()),
        # 비전 입력 크기 상한 — 업로드 상한과 **같은 값**을 쓴다(2026-08-12).
        # 주입하지 않으면 도구가 10MB 로 폴백해, 20MB 로 올라간 파일이 업로드는
        # 통과하고 분석에서만 거부되는 불일치가 난다(실측 F10).
        "vision_max_image_mb": _vision_max_bytes() // (1024 * 1024),
        # 파일이 없을 때 "만료됐다"고 알려 주기 위한 값(2026-08-12). 그냥 "없다"고만
        # 하면 모델이 경로를 고쳐 가며 재시도해 턴을 낭비한다.
        "upload_retention_hours": getattr(
            getattr(state.config, "upload", None), "retention_hours", 0
        ),
    }

    # 시스템 프롬프트는 파일 읽기 + 서브에이전트 가이드 조립이라 비교적 무겁다 →
    # 한 번만 만들어 문자열로 공유한다(요청마다 다시 읽지 않는다).
    system_prompt = _load_worker_system_prompt(
        components.get("agent_registry"), components.get("hardware_tier")
    )

    return {
        "tier": components["hardware_tier"],
        "worker_provider": components["model_provider"],
        "scout_provider": components.get("scout_provider"),
        "web_tools": web_tools,
        "scout_tools": scout_tools,
        "combined_pool": combined_pool,
        "context_manager": components.get("context_manager"),
        "memory_manager": components.get("memory_manager"),
        "knowledge_retriever": components.get("knowledge_retriever"),
        "system_prompt": system_prompt,
        "routing_config": state.config.routing,
        "budgets": _web_budgets,
        "base_options": base_options,
        "permission_mode": state.permission_mode.value,
        "base_cwd": state.cwd or ".",
        "pe_enabled": _pe_enabled,
        "sessions_dir": _sessions_dir,
    }


def _session_sandbox_cwd(parts: dict, session_id: str) -> str:
    """세션별 샌드박스 작업 디렉토리를 결정한다.

    ★무회귀★: permission_enforcement.enabled=False(기본)면 항상 base_cwd(프로젝트
    루트)를 그대로 쓴다 — 파일 도구의 상대경로 해석이 현행과 100% 동일. 강제가
    켜진 경우에만 세션별 격리 디렉토리({sessions_dir}/{session_id}/workspace)로
    전환한다. PathGuard의 cwd-scope 순회 검사가 이 디렉토리를 기준으로 동작하므로,
    세션 밖(상위/시스템 경로) 쓰기가 자연히 차단된다.
    """
    base_cwd = parts["base_cwd"]
    if parts["pe_enabled"] and parts["sessions_dir"] and session_id:
        candidate = os.path.join(parts["sessions_dir"], str(session_id), "workspace")
        try:
            os.makedirs(candidate, exist_ok=True)
            return candidate
        except OSError as e:
            # 디렉토리 생성 실패 시 기존 cwd로 폴백(본류를 막지 않는다).
            logger.warning("[web] 세션 샌드박스 생성 실패, 기본 cwd 사용: %s", e)
    return base_cwd


def _mixed_tool_pool(client_tools: list[Any] | None, web_tools: list[Any]) -> list[Any]:
    """클라이언트 도구가 오면 그것으로 교체하되, 화이트리스트 서버 도구만 남긴다.

    [왜 완전 교체가 아닌가 — 2026-08-12]
      전부 교체하면 `AnalyzeImage` 가 사라져 플러그인 요청에서 **이미지를 영영 볼 수
      없다.** 클라이언트가 대신 실행할 수도 없다 — 비전 서버는 서버에만 있다.

    [왜 화이트리스트인가]
      "필요하면 남긴다" 로 두면 목록이 조용히 늘어난다. 서버 도구를 다시 여는 것은
      보안 결정이므로 상수 하나로 못 박고 테스트로 고정한다.

    Returns:
        client_tools 가 없으면 web_tools 그대로(무회귀). 있으면
        client_tools + 화이트리스트에 해당하는 서버 도구.
    """
    if not client_tools:
        return web_tools

    from core.tools.implementations.client_tool import SERVER_TOOLS_KEPT_WITH_CLIENT_TOOLS

    taken = {t.name for t in client_tools}
    kept = [
        t
        for t in web_tools
        # 이름이 겹치면 **클라이언트 것이 이긴다** — 호출자가 자기 구현을 선언했다면
        # 그쪽을 존중한다(서버가 몰래 가로채면 디버깅이 불가능해진다).
        if t.name in SERVER_TOOLS_KEPT_WITH_CLIENT_TOOLS and t.name not in taken
    ]
    return [*client_tools, *kept]


def _assemble_session_engine(
    parts: dict,
    session_id: str,
    tenant: Any,
    client_tools: list[Any] | None = None,
) -> tuple[Any, Any]:
    """
    공유 부품(parts)으로 '세션 전용' QueryEngine을 가볍게 조립한다.

    client_tools 가 오면 이번 세션의 도구 목록을 그것으로 교체한다. 섞으면 서버
    파일시스템을 만지는 도구가 다시 열려 웹에서 Bash 를 제거한 조치가 무의미해지기
    때문이다. 교체된 도구는 서버가 실행하지 않고 tool_calls 로 돌려준다.

    **예외 하나 — 비전 분석(2026-08-12).**
      A.X-4.0 은 텍스트 전용이고 비전은 별도 서버에 있다. 그 서버를 부르는 것은
      `AnalyzeImage` 뿐인데, 전부 교체하면 플러그인 요청에 이 도구가 없어 이미지를
      영영 볼 수 없다. 클라이언트가 대신 실행할 수도 없다(개발자 PC 에 비전 서버가
      없다). 그래서 `SERVER_TOOLS_KEPT_WITH_CLIENT_TOOLS` 에 열거된 도구만 남긴다.

      이 예외가 안전한 이유는 그 도구가 read-only 이고 업로드 디렉토리 밖 경로를
      DENY 하기 때문이다. 목록을 늘리는 것은 보안 결정이므로 테스트로 고정돼 있다.

    반환: (engine, dispatcher). dispatcher를 engine.model_dispatcher로 되꺼내지 않고
    직접 돌려주는 이유는, engine을 mock으로 대체하는 단위 테스트에서도 실제 조립된
    dispatcher를 검증할 수 있게 하기 위함이다.

    세션마다 새로 만드는 것(격리 대상):
      - ToolUseContext: session_id / cwd(샌드박스) / options["tenant"]
      - ModelDispatcher: 위 격리 context를 바인딩해 도구 실행이 올바른 tenant를 봄
        (dispatcher.route → query_loop → 도구가 context.options["tenant"]를 읽는다.
         만약 dispatcher를 공유하면 tenant가 세션 간 새어나간다 → 반드시 세션별 생성)
      - QueryEngine: _messages / _session_id / _cumulative_usage 등 요청별 가변 상태

    무거운 부품(도구 인스턴스·시스템 프롬프트·프로바이더·retriever)은 parts에서
    공유 재사용한다. 이 조립은 객체 참조 저장 + 로그 몇 줄 수준이라 매우 가벼워
    (수 마이크로초), 모델 추론(초 단위) 대비 무시할 만한 비용이다 → 요청/세션별
    생성이 정당하다.
    """
    from core.orchestrator.model_dispatcher import ModelDispatcher
    from core.orchestrator.query_engine import QueryEngine
    from core.tools.base import ToolUseContext

    context = ToolUseContext(
        cwd=_session_sandbox_cwd(parts, session_id),
        session_id=session_id,
        permission_mode=parts["permission_mode"],
        # 공용 base_options를 '새 dict'로 얕은 복제한 뒤 이 세션의 tenant만 얹는다.
        # (원본 base_options를 mutate하지 않아야 다른 세션이 오염되지 않는다.)
        options={**parts["base_options"], "tenant": tenant},
    )

    # 클라이언트 도구가 오면 전체 교체(위 docstring 참고), 아니면 종전 웹 도구 풀.
    session_tools = _mixed_tool_pool(client_tools, parts["web_tools"])

    dispatcher = ModelDispatcher(
        tier=parts["tier"],
        worker_provider=parts["worker_provider"],
        worker_tools=session_tools,
        context=context,
        scout_provider=parts["scout_provider"],
        scout_tools=parts["scout_tools"],
        max_turns=200,
    )

    engine = QueryEngine(
        model_provider=parts["worker_provider"],
        tools=session_tools,
        context=context,
        model_dispatcher=dispatcher,
        context_manager=parts["context_manager"],
        memory_manager=parts["memory_manager"],
        knowledge_retriever=parts["knowledge_retriever"],
        system_prompt=parts["system_prompt"],
        max_turns=200,
        routing_config=parts["routing_config"],
        # 컨텍스트 예산(하드코딩 외부화, 2026-07-03) — RAG 주입 예산 + 출력
        # 토큰 에스컬레이션. 실 config는 항상 존재, 없으면 None → 현행 상수 폴백.
        context_budgets=parts["budgets"],
    )
    return engine, dispatcher


def _build_web_query_engine(components: dict, state: Any) -> Any:
    """
    (하위 호환) Phase 2 부트스트랩 결과로 웹 QueryEngine 1개를 조립해
    (query_engine, dispatcher, web_tools) 3-튜플로 반환한다.

    동시성 수정(2026-07-03) 이후 실제 요청 처리는 세션별 엔진
    (_acquire_session_engine → _assemble_session_engine)을 쓴다. 이 함수는 부품
    조립 + MCP 머지 결과를 검증하는 기존 단위 테스트(tests/unit/test_mcp_web_pool.py)
    와의 계약(3-튜플 반환, web_tools에 MCP 머지)을 유지하기 위해 남긴다.

    web_tools(= MCP 머지 후 실제 Worker 도구 풀)를 반환한다 — /v1/tools가 모델이
    실제로 보는 도구와 일치하도록.
    """
    parts = _build_web_engine_parts(components, state)
    engine, dispatcher = _assemble_session_engine(parts, state.session_id or "", tenant=None)
    return engine, dispatcher, parts["web_tools"]


# ─────────────────────────────────────────────
# 요청/응답 모델 (Pydantic v2)
# ─────────────────────────────────────────────
class ChatRequest(BaseModel):
    """채팅 요청 본문 스키마 (POST /v1/chat, /v1/chat/stream 공용).

    브라우저 채팅 UI가 보내는 최소 입력이다. session_id를 함께 주면 그 세션의
    이전 대화를 이어가고, 없으면 서버가 새 UUID 세션을 만든다. tenant_id는
    멀티테넌시 선택 필드로, 헤더/API 키와 함께 _resolve_tenant에서 해석된다.
    """

    message: str = Field(..., description="사용자 메시지")
    session_id: str | None = Field(default=None, description="세션 ID (없으면 새 세션 생성)")
    model: str = Field(
        default="primary",
        description="사용할 모델 (primary: Qwen 3.5, auxiliary: ExaOne)",
    )
    # 멀티테넌시 (Part 5 Ch 15) — body로도 지정 가능 (헤더/API 키와 병행)
    tenant_id: str | None = Field(
        default=None,
        description="테넌트 ID. 헤더 X-Tenant-ID / Authorization Bearer와 같은 우선순위 중 하나.",
    )


class ToolCallInfo(BaseModel):
    """한 번의 도구 호출을 응답에 실어 보내기 위한 요약 스키마.

    채팅 핸들러가 StreamEvent(TOOL_USE_STOP → 이름/입력, TOOL_RESULT → 결과/에러)
    를 소비하며 tool_use_id 기준으로 채운다. UI가 "무슨 도구를 어떤 입력으로
    호출해 어떤 결과가 나왔는지"를 사용자에게 보여줄 때 쓴다.
    input_data는 외부로는 alias 'input'으로 직렬화된다(OpenAI 관례와 정합).
    """

    name: str
    input_data: dict[str, Any] = Field(default_factory=dict, alias="input")
    result: str | None = None
    is_error: bool = False


class UsageInfo(BaseModel):
    """이번 응답의 토큰 사용량(입력/출력/합계).

    USAGE_UPDATE StreamEvent에서 채워지며, UI의 사용량 표시와 예산 관측에 쓰인다.
    (OpenAI 규격의 prompt/completion/total 명칭과는 다른 Nexus 내부 표기다.)
    """

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


class ChatResponse(BaseModel):
    """비스트리밍 채팅(/v1/chat)의 최종 응답 스키마.

    submit_message가 흘려보낸 모든 StreamEvent를 서버가 다 모은 뒤 한 번에 담아
    돌려준다: assistant 최종 텍스트(response), 실행된 도구 목록(tool_calls),
    토큰 사용량(usage), 그리고 생성 문서 다운로드 링크(downloads).
    """

    session_id: str = Field(description="세션 ID")
    response: str = Field(description="assistant 응답 텍스트")
    tool_calls: list[ToolCallInfo] = Field(
        default_factory=list, description="실행된 도구 호출 목록"
    )
    usage: UsageInfo = Field(default_factory=UsageInfo, description="토큰 사용량")
    # 파일 생성 도구(DocumentExport)의 다운로드 링크 — 서버가 도구 결과에서 뽑은
    # 정확한 URL. UI는 모델 텍스트가 아니라 이 값으로 다운로드 버튼을 만든다.
    downloads: list[dict[str, str]] = Field(
        default_factory=list, description="생성 문서 다운로드 목록({url, filename})"
    )
    # 지식 RAG 출처 인용(Point 4-2) — 이번 답변에 주입된 지식 청크의 출처 목록.
    # downloads와 동일 원칙: 모델 텍스트가 아니라 서버가 retriever 메타에서 확보한
    # 진실이다. citation 비활성이면 빈 리스트라 기존 클라이언트에 무영향(무회귀).
    sources: list[KnowledgeCitation] = Field(
        default_factory=list, description="답변 근거 출처 목록([출처N]이 가리키는 실체)"
    )


# ─────────────────────────────────────────────
# OpenAI 호환 채팅 모델 (POST /v1/chat/completions)
# ─────────────────────────────────────────────
# 외부 서비스(AgentHub·.NET·LangChain 등 모든 OpenAI 클라이언트)가 커스텀 코드 없이
# Nexus를 "드롭인 LLM 프로바이더"로 쓰게 하기 위한 요청/응답 스키마다.
# OpenAI Chat Completions API 규격(https 규격 문서)의 필드명을 그대로 따른다.
class OpenAIChatMessage(BaseModel):
    """OpenAI 형식의 대화 메시지 한 줄.

    role: "system" | "user" | "assistant" | "tool" 등.
    content: 메시지 본문(멀티모달 배열은 이 에어갭 텍스트 파이프라인에서 미지원 → 문자열만).
    여분 필드(name, tool_call_id 등)는 무시(extra=ignore)한다.
    """

    model_config = {"extra": "ignore"}

    role: str = Field(description="메시지 역할 (system/user/assistant/tool)")
    # 배열도 받는다(2026-08-12). OpenAI 비전 규격은 content 를 파트 배열로 보낸다.
    #   [{"type":"text",...}, {"type":"image_url","image_url":{"url":"data:..."}}]
    # 예전에는 문자열만 받아 **이미지가 아니라 배열이라서** 422 가 났다(텍스트만 든
    # 배열도 마찬가지였다). _normalize_openai_content 가 여기서 문자열로 되돌린다.
    # 원소 타입은 `Any` 로 둔다 — `list[dict]` 로 좁히면 파트 하나가 이상할 때
    # Pydantic 이 **요청 전체를 422** 로 거부한다. 이 엔드포인트의 관례는 그 반대다
    # (build_client_tools: "하나 이상하다고 요청 전체를 죽이지 않는다"). 이상한 파트는
    # 정규화기가 건너뛰고 경고로 알린다.
    content: str | list[Any] | None = Field(
        default=None, description="메시지 본문. 문자열 또는 OpenAI 파트 배열."
    )
    # ── 클라이언트 실행 도구 루프용(2026-08-08) ──────────────
    # 클라이언트(VSCode 플러그인 등)가 도구를 직접 실행하는 방식에서는 대화가
    # `user → assistant(tool_calls) → tool(결과) → assistant …` 로 흐른다.
    # 이 두 필드가 없으면 그 히스토리를 서버가 재현할 수 없어, 모델이 자기가 무엇을
    # 요청했고 무엇을 돌려받았는지 모른 채 같은 도구를 반복 호출한다.
    tool_calls: list[dict[str, Any]] | None = Field(
        default=None, description="assistant가 요청한 도구 호출 목록(OpenAI 규격)"
    )
    tool_call_id: str | None = Field(
        default=None, description="role='tool'일 때 어느 호출의 결과인지 가리키는 id"
    )


class OpenAIChatCompletionRequest(BaseModel):
    """OpenAI `POST /v1/chat/completions` 요청 본문.

    - messages: 필수. 전체 대화 히스토리를 매 요청 그대로 담아 보낸다(클라이언트가 소유).
    - model/stream/temperature/max_tokens/top_p: 받되, 이 파이프라인이 모르는 값은
      무시하거나 기본값으로 동작한다(엔진이 내부 config로 샘플링을 관리하기 때문).
    - 알 수 없는 여분 필드(n, stop, presence_penalty 등)는 허용하고 무시한다(extra=ignore).
    """

    # protected_namespaces=() : `model` 필드가 pydantic의 model_ 예약 네임스페이스
    # 경고를 내지 않도록 한다. extra="ignore" : 모르는 필드는 조용히 버린다.
    model_config = {"extra": "ignore", "protected_namespaces": ()}

    messages: list[OpenAIChatMessage] = Field(description="OpenAI 형식 대화 메시지 배열")
    model: str = Field(default="primary", description="모델 식별자(응답에만 반향)")
    stream: bool = Field(default=False, description="true면 SSE 스트리밍 응답")
    # 샘플링 파라미터 — 받되 엔진이 자체 config로 관리하면 무시될 수 있다.
    temperature: float | None = Field(default=None, description="샘플링 온도(엔진 관리 시 무시)")
    max_tokens: int | None = Field(default=None, description="최대 생성 토큰(엔진 예산 우선)")
    top_p: float | None = Field(default=None, description="nucleus 샘플링(엔진 관리 시 무시)")
    # 멀티테넌시 — body로도 테넌트 지정 가능(헤더/API 키와 동일 우선순위 체계).
    tenant_id: str | None = Field(default=None, description="테넌트 ID(선택)")
    # 질의 클래스 고정(2026-08-16). 지정하면 서버의 휴리스틱 분류를 건너뛴다.
    #   "TOOL"      — 지식 RAG 미주입 + 도구/코드 작업 프로필(코드 분석에 권장)
    #   "KNOWLEDGE" — 지식 RAG 주입(테넌트 소스 필터는 그대로 적용)
    #   "CHAT"      — 짧은 응답 프로필, RAG 미주입
    # 왜 필요한가: 짧고 키워드 없는 코드 질문이 KNOWLEDGE 로 분류돼 사내 문서가
    # 주입되는 일이 실측됐다. 호출자가 의도를 밝힐 수 있어야 한다.
    query_class: str | None = Field(
        default=None, description='질의 클래스 고정: "TOOL" | "KNOWLEDGE" | "CHAT"'
    )
    # OpenAI 표준 response_format — 구조화 출력(guided decoding) 요청.
    #   {"type": "json_schema", "json_schema": {"name": ..., "schema": {...}, "strict": ...}}
    #   또는 {"type": "json_object"}(스키마 없는 JSON 강제).
    # 지정 시 엔진이 vLLM guided decoding으로 응답 JSON 문법을 강제한다. 과거에는
    # extra="ignore"로 이 필드가 조용히 버려졌는데(드롭인 프로바이더 규격 위반),
    # 이제 명시 필드로 받아 핸들러에서 StructuredOutputSpec으로 변환한다.
    response_format: dict[str, Any] | None = Field(
        default=None, description="구조화 출력 스펙(OpenAI response_format)"
    )
    # ── 클라이언트 실행 도구(2026-08-08) ─────────────────────
    # 클라이언트가 자기 도구 스키마를 보내면 모델에게 그대로 보여 주고, 호출 결정을
    # tool_calls로 돌려준다. **서버는 이 도구를 실행하지 않는다.**
    #
    # 왜 실행하지 않나 — 두 가지 모두 실행하면 안 되는 이유다.
    #   ① 보안: 서버 도구는 서버 파일시스템을 만진다. 실측으로 테넌트 키 하나에
    #      /app/config/tenants.yaml(전 테넌트 API 키)이 읽혔다(그래서 웹에서 Bash를
    #      제거했다). 코딩용이라고 서버 도구를 되돌려 주면 같은 구멍이 다시 열린다.
    #   ② 쓸모: 개발자의 코드는 개발자 PC에 있다. 서버의 Read는 /app을 읽는다.
    #
    # 이 필드가 오면 이번 요청의 도구 목록은 **클라이언트 도구로 전부 교체**된다
    # (서버 도구는 하나도 노출되지 않는다 — 부분 혼합은 위 ①을 다시 연다).
    tools: list[dict[str, Any]] | None = Field(
        default=None, description="클라이언트가 실행할 도구 스키마(OpenAI tools)"
    )
    tool_choice: Any | None = Field(
        default=None, description='도구 사용 방식("none"이면 도구 없이 답한다)'
    )


class OpenAIResponseMessage(BaseModel):
    """OpenAI 비스트림 응답에서 assistant가 낸 메시지 한 개.

    role은 관례상 항상 "assistant", content는 최종 텍스트(+다운로드 마크다운)이다.
    tool_calls는 클라이언트 실행 도구를 쓸 때만 채워진다(그 외에는 null). OpenAI
    규격도 도구를 안 쓰면 이 필드가 비어 있으므로 기존 소비자에 영향이 없다.
    """

    role: str = "assistant"
    content: str = ""
    tool_calls: list[dict[str, Any]] | None = None


class OpenAIChoice(BaseModel):
    """OpenAI 비스트림 응답의 choice 한 개(index/message/finish_reason).

    Nexus는 후보를 하나만 내므로 index=0 단일 choice만 반환한다. finish_reason은
    정상 종료 시 "stop"으로 고정한다(길이 제한/도구 호출 등 다른 사유는 미노출).
    """

    index: int = 0
    message: OpenAIResponseMessage
    finish_reason: str = "stop"


class FinishDetail(BaseModel):
    """finish_reason 만으로는 구분되지 않는 종료 사유를 기계가 읽을 수 있게 싣는다 (2026-08-13).

    왜 필요한가:
      OpenAI 규격의 finish_reason 은 값이 다섯 개뿐이라(stop/length/tool_calls/
      content_filter/function_call) "구조화 출력 JSON 이 깨졌다"를 담을 칸이 없다.
      그래서 2026-08-05 에 가장 가까운 칸인 `content_filter` 를 빌려 썼는데, 그 이름
      때문에 **콘텐츠 정책에 차단당했다는 오해**가 생겼다(2026-08-13 문의).
      이 서버에는 콘텐츠 정책 필터가 존재하지 않는다.

    왜 finish_reason 값 자체를 바꾸지 않았나:
      OpenAI SDK 들은 이 필드를 Literal 로 검증한다. 규격에 없는 값을 넣으면 표준
      클라이언트가 응답 파싱 단계에서 깨진다. 값은 유지하고 **사유를 따로 싣는다**.
      표준 클라이언트는 모르는 최상위 필드를 무시하므로 안전하다.
    """

    code: str = Field(description="기계가 분기할 사유 코드")
    message: str = Field(description="사람이 읽을 설명")


class OpenAIUsage(BaseModel):
    """OpenAI 규격 토큰 사용량(prompt/completion/total)."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class OpenAIChatCompletionResponse(BaseModel):
    """OpenAI `chat.completion` 비스트림 응답 본문."""

    model_config = {"protected_namespaces": ()}

    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[OpenAIChoice]
    usage: OpenAIUsage = Field(default_factory=OpenAIUsage)
    # 비표준 확장 필드 — 문서 생성 도구의 다운로드 정보를 표준 클라이언트가 무시해도
    # 되도록 별도 배열로도 노출한다(표준 클라이언트는 content의 마크다운 링크를 본다).
    downloads: list[dict[str, str]] = Field(default_factory=list)
    # 비표준 확장 필드 — 요청을 처리하며 **버린 것**을 알린다(현재는 제외된 클라이언트
    # 도구). 표준 클라이언트는 모르는 필드를 무시하고, 우리 플러그인은 읽어서 개발자에게
    # 보여줄 수 있다. 조용히 버리면 "왜 내 도구를 안 쓰지?"의 원인을 찾을 수 없다.
    warnings: list[str] = Field(default_factory=list)
    # 비표준 확장 필드 — finish_reason 이 담지 못하는 종료 사유(FinishDetail 참고).
    # 정상 종료면 None 이므로 기존 응답과 달라지지 않는다(무회귀).
    finish_detail: FinishDetail | None = None


class ToolInfo(BaseModel):
    """도구 목록 조회(GET /v1/tools)용 요약 스키마.

    BaseTool 인스턴스에서 UI/관리자가 알아야 할 최소 정보만 뽑아 노출한다:
    이름, 설명, 그룹, 그리고 읽기 전용 여부(권한/안전성 판단 힌트).
    """

    name: str
    description: str
    group: str = ""
    is_read_only: bool = False


class ModelInfo(BaseModel):
    """모델 목록 조회(GET /v1/models)용 요약 스키마.

    role은 모델의 쓰임을 구분한다: primary(주 추론), auxiliary(한국어 보조),
    embedding(임베딩). id는 config에 정의된 실제 모델 식별자다.
    """

    id: str
    name: str
    role: str  # primary, auxiliary, embedding


class TenantInfo(BaseModel):
    """테넌트 정보 (목록 조회용).

    Part 5 Ch 15 멀티테넌시 등록부를 외부에 노출한다.
    보안상 `api_keys` 원본은 절대 반환하지 않고, 개수(`api_key_count`)만 노출한다.
    """

    # Pydantic의 model_ 접두사 경고를 막는다 (model_override 필드명 때문).
    model_config = {"protected_namespaces": ()}

    id: str
    name: str = ""
    description: str = ""
    model_override: str | None = None
    allowed_knowledge_sources: list[str] = Field(default_factory=list)
    api_key_count: int = 0  # api_keys 원본은 비노출
    adapter_name_prefix: str | None = None  # M7
    metadata: dict[str, Any] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    """헬스체크(GET /health) 응답 스키마.

    status는 오케스트레이터(이 서버) 자체의 상태, gpu_server는 GPU 서버(Machine B)
    핑 결과다: healthy/unhealthy/unreachable/unknown 중 하나.
    """

    status: str = "ok"
    version: str = "0.1.0"
    gpu_server: str = "unknown"


# ─────────────────────────────────────────────
# 앱 상태 (모듈 레벨)
# ─────────────────────────────────────────────
_app_state: dict[str, Any] = {
    "state": None,  # GlobalState
    "config": None,  # NexusConfig
    "logging_middleware": None,  # RequestLoggingMiddleware 인스턴스
    "query_engine": None,  # QueryEngine (Phase 2에서 초기화)
    "tool_registry": None,  # ToolRegistry (Phase 2에서 초기화)
}


# ─────────────────────────────────────────────
# Lifespan (앱 시작/종료 이벤트)
# ─────────────────────────────────────────────
async def _uploads_cleanup_loop(
    uploads_dir: Path, retention_hours: float, interval_minutes: float
) -> None:
    """업로드 샌드박스를 주기적으로 쓸어내는 백그라운드 루프.

    왜 인프로세스 주기 태스크인가: 호스트 크론에 의존하면 배포처(고객 에어갭
    환경 포함)마다 별도 프로비저닝이 필요하다. 웹 서버가 스스로 도는 편이
    어디에 올려도 동작한다. 정리 자체는 파일시스템 작업이라 asyncio.to_thread
    로 돌려 이벤트 루프를 막지 않는다.

    기동 직후 한 번 먼저 돌린다 — 재시작 시점에 밀린 찌꺼기를 바로 걷어내기
    위함이다. 이후 interval_minutes 간격으로 반복한다.

    실패는 삼킨다(fail-soft). 정리는 부가 기능이며 서비스를 멈춰선 안 된다.
    다만 CancelledError 는 종료 신호이므로 그대로 올려보낸다.
    """
    from core.storage.uploads import cleanup_expired_uploads

    while True:
        try:
            await asyncio.to_thread(cleanup_expired_uploads, uploads_dir, retention_hours)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001 — 정리 실패가 서비스를 막지 않는다
            logger.warning("업로드 정리 실패(무시): %s", e)
        await asyncio.sleep(interval_minutes * 60)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """FastAPI 수명주기 훅 — 서버 기동/종료 시 딱 한 번씩 실행된다.

    기동(yield 이전):
      1) Phase 1 init() — 환경 비의존 초기화(설정 로드 등).
      2) Phase 2 init_phase2() — ToolRegistry/MemoryManager/모델 프로바이더 등 조립.
      3) 웹 전용 '공유 부품'(web_engine_parts)을 _build_web_engine_parts로 한 번만
         만들어 _app_state에 저장한다. 이후 각 요청은 이 부품으로 세션 전용 엔진을
         가볍게 조립한다(_acquire_session_engine). 여기서 부트스트랩이 실패해도
         서버는 뜨되(placeholder 응답), 채팅은 "미초기화" 안내로 폴백한다.

    종료(yield 이후):
      - 임베딩 keepalive task를 깨끗이 취소하고, 세션 요약을 로그로 남긴다.

    ★주의★ init/init_phase2는 무겁고 요청과 무관하므로 반드시 여기서 한 번만 한다.
    요청 핸들러 안에서 이 초기화를 다시 하지 않는다(비용·경합 방지).
    """
    # 시작: Phase 1 + Phase 2 부트스트랩
    try:
        from core.bootstrap import init, init_phase2

        # Phase 1: 환경 비의존 초기화
        state = await init()
        _app_state["state"] = state
        _app_state["config"] = state.config

        # Phase 2: ToolRegistry, MemoryManager, QueryEngine 초기화
        components = await init_phase2(state)
        _app_state["tool_registry"] = components.get("tool_registry")
        _app_state["model_provider"] = components.get("model_provider")
        _app_state["memory_manager"] = components.get("memory_manager")  # Ch 16
        _app_state["todo_store"] = components.get("todo_store")  # 계획 체크리스트 저장소
        _app_state["tenant_registry"] = state.config.tenants  # M2 — 헤더 해석용
        # v0.14.8: 임베딩 keepalive task — 종료 시 cancel하기 위해 보관
        _app_state["embedding_keepalive_task"] = components.get("embedding_keepalive_task")

        # 생성물 메타데이터(tb_artifacts)용 asyncpg 풀 보관 + 스키마 멱등 보장.
        # 바이트는 파일시스템(exports_dir), 메타/소유자만 이 풀로 PG에 남긴다.
        # pg_pool이 None(DB 미가동/테스트)이면 ensure_artifacts_schema는 no-op이라
        # 무회귀다. 스키마 보장 실패도 fail-soft로 서버 기동을 막지 않는다.
        _app_state["pg_pool"] = components.get("pg_pool")
        try:
            from core.storage.artifacts import ensure_artifacts_schema

            await ensure_artifacts_schema(_app_state["pg_pool"])
        except Exception as e:  # noqa: BLE001 — 스키마 준비 실패가 기동을 막지 않게 삼킴
            logger.warning("tb_artifacts 스키마 보장 실패(무시): %s", e)

        # 웹 전용 QueryEngine — 도구 8개로 축소 (토큰 예산 관리)
        # RTX 5090 (8192 ctx)에서 도구 24개(~6,102토큰)는 컨텍스트 초과.
        # 핵심 도구 8개(~1,851토큰)만 사용하여 입력+출력 공간 확보.
        # (2026-04-21 리팩토링 3: 인라인 조립 로직을 _build_web_query_engine으로 분리)
        # (2026-07-03 동시성 수정 #5: 공유 부품(parts)을 한 번만 만들어 저장하고,
        #  실제 요청은 세션별 격리 엔진을 조립해 쓴다. web_engine은 하위 호환/메타데이터
        #  용 템플릿으로만 남긴다 — 채팅 핸들러는 이걸 공유 mutable 상태로 쓰지 않는다.)
        web_parts = _build_web_engine_parts(components, state)
        web_engine, web_dispatcher = _assemble_session_engine(
            web_parts, state.session_id or "", tenant=None
        )
        web_tools = web_parts["web_tools"]
        # 세션별 엔진 팩토리가 참조할 공유 부품. 이 키가 존재하면 채팅 핸들러는
        # 요청마다 격리된 QueryEngine을 새로 조립한다(_acquire_session_engine).
        _app_state["web_engine_parts"] = web_parts
        _app_state["model_dispatcher"] = web_dispatcher
        _app_state["query_engine"] = web_engine
        # /v1/tools가 모델이 실제로 보는 웹 Worker 도구 풀(MCP 포함)을 노출하도록 저장.
        # 기존 _app_state["tool_registry"]는 부트스트랩의 23개 base 레지스트리라
        # 웹 Worker 실풀과 다르다(MCP 누락) — 웹 엔드포인트는 web_tools를 우선한다.
        _app_state["web_tools"] = web_tools
        logger.info(
            "웹 서버 부트스트랩 완료 (Phase 1 + 2, 웹 도구 %d개)",
            len(web_tools),
        )
    except Exception as e:
        logger.warning(f"부트스트랩 실패, 기본 설정으로 시작: {e}")

    # 업로드 정리 잡 기동 — 업로드본이 무한히 쌓이는 것을 막는다.
    # 부트스트랩 성공/실패와 무관하게 띄운다(설정을 못 읽으면 기본값 사용).
    # 저장 경로는 업로드 라우트와 같은 _uploads_dir() 단일 소스를 쓴다.
    try:
        from core.config import UploadConfig

        upload_cfg = getattr(_app_state.get("config"), "upload", None) or UploadConfig()
        interval = upload_cfg.cleanup_interval_minutes
        if interval > 0 and upload_cfg.retention_hours > 0:
            _app_state["uploads_cleanup_task"] = asyncio.create_task(
                _uploads_cleanup_loop(
                    _uploads_dir(), upload_cfg.retention_hours, interval
                )
            )
            logger.info(
                "업로드 정리 잡 기동: 보존 %d시간, 주기 %d분",
                upload_cfg.retention_hours,
                interval,
            )
        else:
            logger.info(
                "업로드 정리 잡 비활성 (retention_hours=%s, cleanup_interval_minutes=%s)",
                upload_cfg.retention_hours,
                interval,
            )
    except Exception as e:  # noqa: BLE001 — 정리 잡 기동 실패가 서버 기동을 막지 않는다
        logger.warning("업로드 정리 잡 기동 실패(무시): %s", e)

    yield

    # 종료: 리소스 정리
    # 업로드 정리 잡을 먼저 취소한다(파일시스템 작업이라 오래 붙잡지 않는다).
    cleanup_task = _app_state.get("uploads_cleanup_task")
    if cleanup_task is not None and not cleanup_task.done():
        cleanup_task.cancel()
        try:
            await cleanup_task
        except asyncio.CancelledError:
            pass  # 우리가 보낸 취소 신호 — 정상 종료 경로다.
        except Exception as e:  # noqa: BLE001 — 종료 경로: 사유만 남기고 계속 진행
            logger.debug("업로드 정리 잡 종료 중 예외(무시): %s", e)

    # v0.14.8 — 임베딩 keepalive task를 깨끗이 취소
    keepalive = _app_state.get("embedding_keepalive_task")
    if keepalive is not None and not keepalive.done():
        keepalive.cancel()
        try:
            await keepalive
        except (asyncio.CancelledError, Exception):
            # CancelledError 또는 task 내부 예외 모두 swallow — 종료 경로
            pass
    if _app_state["state"]:
        summary = _app_state["state"].get_session_summary()
        logger.info(f"웹 서버 종료. 세션 요약: {summary}")


# ─────────────────────────────────────────────
# FastAPI 앱 생성
# ─────────────────────────────────────────────
app = FastAPI(
    title="IDINO NOVA",
    description="에어갭 로컬 LLM 오케스트레이션 플랫폼 API",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS 미들웨어 적용 — 로컬/LAN만 허용
app.add_middleware(CORSMiddleware, **CORSConfig.get_cors_kwargs())


# API 키 인증 미들웨어 (Security Critical #4) — CORS 뒤에 등록한다.
# 설정·테넌트 레지스트리는 lifespan 기동 후에야 _app_state에 채워지므로,
# 미들웨어가 dispatch 시점에 지연 조회할 수 있도록 무인자 함수로 넘긴다
# (생성 시점 조회 금지 — 순환 import 및 기동 순서 문제 방지).
def _get_web_auth_config() -> Any:
    """현재 로드된 WebAuthConfig를 반환한다(없으면 None → 인증 비활성 취급)."""
    cfg = _app_state.get("config")
    return getattr(cfg, "web_auth", None) if cfg is not None else None


def _get_tenant_registry() -> Any:
    """현재 TenantRegistry를 반환한다(없으면 None → fail-closed 차단)."""
    return _app_state.get("tenant_registry")


app.add_middleware(
    ApiKeyAuthMiddleware,
    get_auth_config=_get_web_auth_config,
    get_tenant_registry=_get_tenant_registry,
)

@app.exception_handler(RequestValidationError)
async def _handle_validation_error(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """요청 검증 실패(422)의 **사유를 로그에 남긴다** (2026-08-13).

    왜 필요한가:
      기본 동작은 422 본문에만 사유를 담고 서버 로그에는 접근 로그 한 줄
      (`"POST /v1/chat/completions" 422 Unprocessable Entity`)만 남긴다. 그래서
      2026-08-12 플러그인 422 를 조사할 때 **어느 필드가 왜 걸렸는지 로그로는 알 수
      없어** 요청을 직접 재현해야 했다. 검증 실패는 클라이언트 개발자가 고쳐야 하는
      문제이므로, 원인을 서버가 먼저 알고 있어야 한다.

    함께 하는 일:
      - 응답 본문의 `input` 을 잘라 되돌린다. 자르지 않으면 20MB base64 이미지를
        보냈다가 실패했을 때 그 20MB 가 로그와 응답으로 **그대로 증폭**된다.
      - X-Request-ID 를 로그와 응답 헤더 양쪽에 싣는다(요청 ID 추적).
    """
    detail = _summarize_validation_errors(exc.errors())
    rid = _sanitize_request_id(request.headers.get("X-Request-ID"))
    logger.warning(
        "[요청검증실패] 422 %s %s request_id=%s detail=%s",
        request.method,
        request.url.path,
        rid or "-",
        json.dumps(detail, ensure_ascii=False)[:2000],
    )
    return JSONResponse(status_code=422, content={"detail": detail})


@app.middleware("http")
async def _echo_request_id(request: Request, call_next: Any) -> Response:
    """클라이언트가 보낸 X-Request-ID 를 응답에 되돌려준다 (2026-08-13).

    왜 엔드포인트가 아니라 미들웨어인가:
      ① 상태 코드를 가리지 않는다 — 200 뿐 아니라 400/422/500 응답에도 실린다.
         정작 필요한 건 실패했을 때인데, 핸들러 안에서 세팅하면 예외 경로를 놓친다.
      ② StreamingResponse 를 직접 반환하는 경로도 자동으로 덮는다.
      ③ 핸들러 시그니처를 건드리지 않는다(엔드포인트 함수를 직접 부르는 테스트가 있다).
    """
    response: Response = await call_next(request)
    rid = _sanitize_request_id(request.headers.get("X-Request-ID"))
    if rid:
        response.headers["X-Request-ID"] = rid
    return response


# 요청 로깅 미들웨어 적용
_logging_middleware = RequestLoggingMiddleware(app)
_app_state["logging_middleware"] = _logging_middleware

# 정적 파일 서빙 — 채팅 UI (HTML/CSS/JS)
_static_dir = Path(__file__).parent / "static"
if _static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")


# ─────────────────────────────────────────────
# 채팅 엔드포인트
# ─────────────────────────────────────────────
def _restore_messages_from_saved(saved: list[dict] | None, session_id: str) -> list:
    """Redis에서 가져온 직렬화 메시지(dict 목록)를 Message 객체 리스트로 복원한다.

    비스트리밍/스트리밍 두 핸들러가 동일하게 쓰던 복원 로직을 한 곳으로 모은다.

    왜 항목 단위로 예외를 격리하는가:
      과거 버전이 assistant content를 ContentBlock 리스트 형식으로 저장한
      "오염 데이터"가 Redis에 남아 있을 수 있다. 예전 코드는 복원 루프 전체를
      하나의 try/except로 감싸서, 한 항목(리스트 content)이 깨지면 그 뒤 항목까지
      전부 복원이 중단됐다(이력 유실의 직접 원인).
      따라서 항목마다 예외를 격리하고, content가 리스트면 텍스트만 추출해
      평문으로 되돌려 견고하게 복원한다.
    """
    from core.message import Message as _Msg

    restored: list[_Msg] = []
    for item in saved or []:
        try:
            role = item.get("role")
            content = item.get("content", "")
            # 과거 오염 데이터 호환: content가 ContentBlock 리스트면 텍스트만 추출한다.
            if isinstance(content, list):
                content = "".join(
                    block.get("text", "")
                    for block in content
                    if isinstance(block, dict) and block.get("type") == "text"
                )
            if not content:
                continue
            if role == "user":
                restored.append(_Msg.user(content))
            elif role == "assistant":
                restored.append(_Msg.assistant(content))
        except Exception as e:
            # 한 항목이 깨져도 나머지는 복원되도록 건너뛴다(부분 복원 보장).
            logger.warning("세션 복원 항목 건너뜀 (%s): %s", session_id, e)
            continue
    return restored


# ─────────────────────────────────────────────
# 세션별 엔진 격리 + 세션 락 (동시성 결함 수정 — 감사 Critical #5, 2026-07-03)
# ─────────────────────────────────────────────
# 기존: QueryEngine 싱글톤 1개를 모든 요청이 공유 → 동시 요청이 _messages/tenant를
# 락 없이 뒤섞음(멀티테넌트 최대 블로커). 수정: (1) 요청/세션별로 격리된 QueryEngine을
# 조립하고(_acquire_session_engine), (2) '같은 세션'의 동시 요청만 세션별 asyncio.Lock
# 으로 직렬화한다. 서로 다른 세션은 병렬을 유지하므로 멀티테넌트 처리량이 죽지 않는다.

# 세션 락 상한 — 무한 증가를 막는 바운드 LRU. 도달 시 '사용 중이 아닌' 가장 오래된
# 락부터 축출한다(사용 중 락은 절대 축출하지 않음).
_SESSION_LOCK_MAX = 4096


def _get_session_lock(session_id: str) -> asyncio.Lock:
    """세션 ID별 asyncio.Lock을 얻는다(없으면 생성). 바운드 LRU로 개수를 제한한다.

    반드시 실행 중인 이벤트 루프 안(async 핸들러)에서 호출한다.

    루프 인지(loop-aware)로 저장하는 이유: asyncio.Lock은 최초 acquire 시 특정
    이벤트 루프에 바인딩된다. 그런데 _app_state는 모듈 싱글톤이라 프로세스 수명 동안
    유지되고, pytest는 테스트마다 새 이벤트 루프를 쓴다(asyncio_default_fixture_
    loop_scope=function). 만약 락을 루프와 무관하게 캐시하면, 이전 테스트 루프에
    바인딩된 락을 다음 테스트 루프에서 acquire하다 'attached to a different loop'
    오류가 난다. 그래서 현재 실행 루프가 바뀌면 락 저장소를 새로 만든다. 프로덕션은
    단일 장수명 루프라 저장소가 유지되어 LRU가 정상 동작한다.

    동기 함수인 이유: dict 접근/삽입 사이에 await가 없어 단일 이벤트 루프에서 원자적
    이다(별도 async 가드 락이 필요 없다).
    """
    loop = asyncio.get_running_loop()
    store = _app_state.get("session_locks_store")
    if store is None or store[0] is not loop:
        # 최초 호출이거나 이벤트 루프가 교체됨(주로 테스트) → 저장소를 새로 시작.
        store = (loop, OrderedDict())
        _app_state["session_locks_store"] = store
    locks: OrderedDict[str, asyncio.Lock] = store[1]
    lock = locks.get(session_id)
    if lock is None:
        lock = asyncio.Lock()
        locks[session_id] = lock
    else:
        locks.move_to_end(session_id)  # 최근 사용 표시(LRU)
    # 바운드 축출 — 가장 오래된 것부터. 단, 현재 잠겨 있는(사용 중) 락이나 방금
    # 만든 이 세션의 락은 건드리지 않는다(사용 중 락 축출 시 직렬화 무력화).
    while len(locks) > _SESSION_LOCK_MAX:
        old_sid, old_lock = next(iter(locks.items()))
        if old_lock.locked() or old_sid == session_id:
            break
        locks.popitem(last=False)
    return lock


def _acquire_session_engine(
    session_id: str, tenant: Any, client_tools: list[Any] | None = None
) -> Any:
    """요청/세션별로 격리된 QueryEngine을 반환한다.

    - 프로덕션(부트스트랩 성공 → web_engine_parts 존재): 세션 전용 엔진을 새로
      조립한다. 공유 mutable 상태(_messages/_session_id/tenant)를 원천 제거한다.
    - 부트스트랩 미완/테스트(parts 없음): 기존 _app_state['query_engine'] 싱글톤을
      그대로 반환한다(무회귀 — 기존 웹 테스트가 주입한 fake 엔진/placeholder 경로 유지).
    """
    parts = _app_state.get("web_engine_parts")
    if parts is not None:
        engine, _dispatcher = _assemble_session_engine(
            parts, session_id, tenant, client_tools=client_tools
        )
        return engine
    return _app_state.get("query_engine")


@app.post("/v1/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    x_tenant_id: str | None = Header(default=None, alias="X-Tenant-ID"),
    authorization: str | None = Header(default=None),
    x_client_channel: str | None = Header(default=None, alias="X-Client-Channel"),
    x_client_id: str | None = Header(default=None, alias="X-Client-Id"),
) -> ChatResponse:
    """비스트리밍 채팅 엔드포인트 (POST /v1/chat).

    사용자 메시지를 세션 전용 QueryEngine에 넘기고, submit_message가 yield하는
    모든 StreamEvent를 끝까지 소비·누적한 뒤 한 번에 ChatResponse로 돌려준다
    (스트리밍이 아니라 "다 모아서 반환"). 처리 흐름:
      1) 세션 ID 확정 + 테넌트 해석(_resolve_tenant).
      2) 세션별 격리 엔진 획득(_acquire_session_engine) — 없으면 미초기화 폴백.
      3) 같은 세션 동시요청은 세션 락으로 직렬화하고, Redis에서 이전 이력을 복원.
      4) StreamEvent 소비: TEXT_DELTA(응답 텍스트), USAGE_UPDATE(토큰),
         TOOL_USE_STOP/TOOL_RESULT(도구 호출 요약), 그리고 생성 문서 다운로드 추출.

    매개변수:
      request        — ChatRequest(message/session_id/model/tenant_id).
      x_tenant_id    — X-Tenant-ID 헤더(테넌트 지정 경로 중 하나).
      authorization  — Authorization 헤더(Bearer API 키로도 테넌트 해석 가능).
    반환: ChatResponse.
    """
    # 세션 ID 생성 또는 재사용
    session_id = request.session_id or str(uuid.uuid4())
    tenant = _resolve_tenant(request.tenant_id, x_tenant_id, authorization)
    # 저장 채널 — 헤더 미지정이면 "web"(종전과 동일). 외부 소비자가 자기 채널을
    # 선언하면 그 채널로 격리돼 웹 사용자 히스토리에 섞이지 않는다.
    channel = _resolve_channel(x_client_channel)
    _client_id = _sanitize_client_id(x_client_id)
    if _client_id and channel != "web":
        _record_client_meta(session_id, channel, _client_id, tenant)

    # 동시성 수정(#5): 요청/세션별 격리 엔진을 얻는다(프로덕션). parts가 없으면
    # 기존 싱글톤/placeholder 경로로 폴백(무회귀).
    engine = _acquire_session_engine(session_id, tenant)
    if engine is None:
        # QueryEngine이 초기화되지 않은 경우 placeholder 응답
        return ChatResponse(
            session_id=session_id,
            response="QueryEngine이 아직 초기화되지 않았습니다.",
            tool_calls=[],
            usage=UsageInfo(),
        )

    from core.message import StreamEvent, StreamEventType

    response_text_parts: list[str] = []
    usage = UsageInfo()
    # ─── 도구 호출 수집 (tool_calls 응답 필드 보강) ─────────────────
    # 4-Tier 체인을 우회하지 않고 submit_message가 yield하는 StreamEvent만 소비하여
    # 누적한다. TOOL_USE_STOP에서 input이 완성되므로 STOP 기준 등록, TOOL_RESULT로
    # 같은 tool_use_id를 매칭해 요약/에러를 채운다(한 호출당 ToolCallInfo 하나).
    tool_calls_by_id: dict[str, ToolCallInfo] = {}
    tool_calls_order: list[str] = []
    downloads: list[dict[str, str]] = []  # 파일 생성 도구 다운로드 링크(정확한 URL)
    # 지식 RAG 출처 인용(Point 4-2) — KNOWLEDGE_SOURCES 이벤트로 받은 출처 목록.
    knowledge_sources: list[KnowledgeCitation] = []
    # 결과 본문이 과도하게 길면 응답이 비대해지므로 요약 길이를 제한한다(과설계 금지).
    result_summary_max = 500
    response_session_id = session_id

    # 같은 세션의 동시 요청만 직렬화(다른 세션은 병렬 유지). 프로덕션에서는 engine
    # 자체가 세션 전용이라 _messages는 이미 격리되지만, 공유 히스토리(Redis)·트랜스
    # 크립트 기록 순서를 안정화하기 위해 세션 단위로 감싼다(전역 처리량은 안 죽음).
    session_lock = _get_session_lock(session_id)
    async with session_lock:
        # 프로젝트(폴더=프로젝트명)가 있으면 지식소스를 narrow한 tenant로 교체(P3-3).
        _project = _resolve_project_for_session(getattr(tenant, "id", "default"), session_id)
        tenant = _narrow_tenant_for_project(tenant, _project)
        # Ch 16 + 리팩토링 2: 세션/tenant/transcript를 공식 bind_request로 한 번에 주입
        transcript = _build_transcript(session_id, channel=channel)
        engine.bind_request(
            session_id=session_id,
            tenant=tenant,
            transcript=transcript,
            channel=channel,
        )
        if tenant is not None:
            logger.info(
                "tenant 해석: %s (sources=%s)",
                tenant.id,
                tenant.allowed_knowledge_sources,
            )

        # 커스텀 인스트럭션(테넌트) + 프로젝트 인스트럭션을 덧붙인다(P2-2/P3-3, 비스트리밍).
        _instruction = _read_custom_instruction(getattr(tenant, "id", "default"))
        _proj_instr = _project.get("instruction", "") if _project else ""
        _style = _resolve_style_prompt_for(getattr(tenant, "id", "default"))
        if (_instruction or "").strip() or (_proj_instr or "").strip() or _style:
            # 문자열 덧붙이기 대신 base에서 재조립한다(멱등). 순서·형식이 세 진입점에서
            # 같아지고, 스타일 같은 섹션이 늘어도 헬퍼 한 곳만 고치면 된다.
            engine.update_system_prompt(
                compose_system_prompt(
                    engine.system_prompt,
                    style=_style,
                    user_instruction=_instruction,
                    project_instruction=_proj_instr,
                )
            )

        # Ch 16: Redis에서 해당 세션의 이전 히스토리 복원
        memory_manager = _app_state.get("memory_manager")
        if memory_manager is not None:
            try:
                engine.clear_messages()
                saved = await memory_manager.short_term.get_conversation_context(
                    session_id, channel=channel
                )
                engine._messages.extend(_restore_messages_from_saved(saved, session_id))
            except Exception as e:
                logger.warning("비스트리밍 세션 복원 실패 (%s): %s", session_id, e)

        # 이 턴에서 새로 추가되는 메시지의 시작 인덱스 — 아래에서 tool_result
        # 메시지의 다운로드 URL을 정확히 뽑기 위한 기준점(이전 턴 결과 오검출 방지).
        dl_start_idx = len(engine._messages)

        async for event in engine.submit_message(request.message):
            if not isinstance(event, StreamEvent):
                continue

            if event.type == StreamEventType.TEXT_DELTA and event.text:
                response_text_parts.append(event.text)

            elif event.type == StreamEventType.USAGE_UPDATE and event.usage:
                usage = UsageInfo(
                    input_tokens=event.usage.input_tokens,
                    output_tokens=event.usage.output_tokens,
                    total_tokens=event.usage.total_tokens,
                )

            elif (
                event.type == StreamEventType.KNOWLEDGE_SOURCES
                and event.knowledge_sources
            ):
                # 지식 RAG 출처 인용(Point 4-2) — 서버 진실인 출처 메타를 그대로 수집.
                # 응답 노출(max_sources 상한/strip)은 루프 종료 후 일괄 처리한다.
                knowledge_sources = list(event.knowledge_sources)

            elif event.type == StreamEventType.TOOL_USE_STOP and event.tool_use:
                # 도구 호출 확정 — 이름/입력/tool_use_id를 등록한다.
                tu = event.tool_use
                tu_id = tu.id
                if tu_id not in tool_calls_by_id:
                    tool_calls_by_id[tu_id] = ToolCallInfo(
                        name=tu.name,
                        input=tu.input,
                    )
                    tool_calls_order.append(tu_id)

            elif event.type == StreamEventType.TOOL_RESULT and event.tool_result:
                # 도구 결과 — 같은 tool_use_id의 호출 정보에 요약/에러 여부를 채운다.
                tr = event.tool_result
                info = tool_calls_by_id.get(tr.tool_use_id)
                if info is None:
                    # STOP 이벤트를 못 본 경우(방어적): 결과만으로 항목을 만든다.
                    info = ToolCallInfo(name="", input={})
                    tool_calls_by_id[tr.tool_use_id] = info
                    tool_calls_order.append(tr.tool_use_id)
                summary = tr.content or ""
                if len(summary) > result_summary_max:
                    summary = summary[:result_summary_max] + "…(truncated)"
                info.result = summary
                info.is_error = tr.is_error

        # 이 턴에 생성된 문서의 정확한 다운로드 URL을 tool_result 메시지에서 뽑는다.
        # (TOOL_RESULT StreamEvent는 발신되지 않으므로 메시지에서 직접 추출한다.)
        # TODO(nexus): orchestrator가 TOOL_RESULT StreamEvent(tool_result 채움 +
        #   ToolResult.metadata 전달)를 발신하도록 고치면, 이 engine._messages 스캔
        #   우회를 없애고 이벤트에서 downloads/result를 채울 수 있다. progress.md
        #   "TOOL_RESULT 이벤트 노출 리팩터" 항목 참조.
        downloads.extend(_collect_downloads(engine._messages[dl_start_idx:]))

        # 응답에 실을 세션 ID는 엔진이 확정한 값을 쓴다(fake 엔진 테스트 호환).
        response_session_id = engine.session_id

        # 생성물 메타데이터를 tb_artifacts에 fail-soft 기록(pg 없으면 조용히 스킵).
        await _record_artifacts(
            downloads, tenant, response_session_id, turn=getattr(engine, "total_turns", None)
        )

    # 등장 순서대로 ToolCallInfo 목록을 만든다.
    tool_calls_info: list[ToolCallInfo] = [tool_calls_by_id[tid] for tid in tool_calls_order]

    # ─── 지식 RAG 출처 인용(Point 4-2) 응답 shaping ─────────────────
    # (1) 응답 텍스트에서 '주입 범위 밖' 번호 마커([출처9] 등)를 제거(strip),
    # (2) sources 필드에 최대 max_sources개까지 노출(expose_in_response=True일 때만).
    # 출처가 없으면(citation 비활성/주입 없음) 두 처리 모두 무영향 → 기존 동작과 동일.
    response_text = "".join(response_text_parts)
    sources_out: list[KnowledgeCitation] = []
    if knowledge_sources:
        expose, strip_invalid, max_sources, label = _citation_settings()
        if strip_invalid:
            valid_indices = {c.index for c in knowledge_sources}
            response_text = _strip_invalid_citation_labels(
                response_text, valid_indices, label
            )
        if expose:
            sources_out = knowledge_sources[:max_sources]

    # 숫자 인용 검증 — 문서에서 옮긴 금액·수량의 자릿수가 틀리면 경고를 덧붙인다.
    # (도구를 쓰지 않은 턴이면 근거 자료가 없어 아무 것도 붙지 않는다.)
    if engine is not None:
        response_text += _answer_warnings_for(
            response_text, engine._messages[dl_start_idx:]
        )

    return ChatResponse(
        session_id=response_session_id,
        response=response_text,
        tool_calls=tool_calls_info,
        usage=usage,
        downloads=downloads,
        sources=sources_out,
    )


@app.post("/v1/chat/stream")
async def chat_stream(
    request: ChatRequest,
    x_tenant_id: str | None = Header(default=None, alias="X-Tenant-ID"),
    authorization: str | None = Header(default=None),
    x_client_channel: str | None = Header(default=None, alias="X-Client-Channel"),
    x_client_id: str | None = Header(default=None, alias="X-Client-Id"),
) -> StreamingResponse:
    """
    SSE 스트리밍 채팅.

    QueryEngine의 AsyncGenerator에서 yield되는 StreamEvent를
    Server-Sent Events 형식으로 실시간 전송한다.
    """
    session_id = request.session_id or str(uuid.uuid4())
    tenant = _resolve_tenant(request.tenant_id, x_tenant_id, authorization)
    # 저장 채널 — 헤더 미지정이면 "web"(무회귀). 아래 중첩 제너레이터들은 이 값을
    # 읽기만 하므로(재대입 없음) 클로저 캡처가 안전하다.
    channel = _resolve_channel(x_client_channel)
    _client_id = _sanitize_client_id(x_client_id)
    if _client_id and channel != "web":
        _record_client_meta(session_id, channel, _client_id, tenant)

    async def _locked_generate() -> AsyncGenerator[str, None]:
        """세션별 격리 엔진 획득 + 세션 락으로 감싼 뒤 실제 스트림을 위임한다.

        동시성 수정(#5): (1) 요청/세션별 격리 QueryEngine을 얻고(_acquire_session_engine),
        (2) 같은 세션의 동시 요청만 세션 락으로 직렬화한다(다른 세션은 병렬 유지 →
        멀티테넌트 처리량 보존). 락은 스트림 전체 수명 동안 유지되며, 클라이언트 연결
        종료로 제너레이터가 닫혀도 async with가 반드시 해제한다.
        """
        engine = _acquire_session_engine(session_id, tenant)
        if engine is None:
            placeholder = {
                "type": "text_delta",
                "text": "QueryEngine이 아직 초기화되지 않았습니다.",
                "session_id": session_id,
            }
            yield f"data: {json.dumps(placeholder, ensure_ascii=False)}\n\n"
            return
        session_lock = _get_session_lock(session_id)
        async with session_lock:
            async for _frame in generate(engine):
                yield _frame

    async def generate(engine: Any) -> AsyncGenerator[str, None]:
        """
        SSE 이벤트를 생성하는 AsyncGenerator.

        QueryEngine의 StreamEvent를 수신하여
        'data: {json}\n\n' 형식으로 실시간 전송한다. engine은 요청/세션별로 격리된
        인스턴스다(호출부 _locked_generate가 세션 락을 잡은 채 소비한다).
        """
        from core.message import StreamEvent

        # 세션별 대화 히스토리 관리 — 요청마다 엔진 messages를 해당 세션 이력으로
        # 복원한다. histories 자체는 세션 키로 분리돼 있고, 같은 세션의 동시 접근은
        # 상위 _locked_generate의 세션 락으로 직렬화된다.
        histories = _app_state.setdefault("chat_histories", {})

        # Ch 16: 세션 영속화 — 메모리 매니저가 있으면 Redis에서 복원
        # 첫 요청(인메모리 비어 있음)일 때만 Redis에서 이전 히스토리를 가져온다.
        # 이후 턴은 인메모리 + Redis 양쪽을 유지한다(write-through).
        memory_manager = _app_state.get("memory_manager")
        if session_id not in histories:
            histories[session_id] = []
            if memory_manager is not None:
                try:
                    saved = await memory_manager.short_term.get_conversation_context(
                        session_id, channel=channel
                    )
                    restored = _restore_messages_from_saved(saved, session_id)
                    histories[session_id].extend(restored)
                    if restored:
                        logger.info(
                            "세션 %s Redis 복원: %d개 메시지",
                            session_id,
                            len(restored),
                        )
                except Exception as e:
                    logger.warning("세션 Redis 복원 실패 (%s): %s", session_id, e)

        # Qwen3.5 thinking 찌꺼기가 들어있는 과거 메시지를 1회성 정제
        # (세션이 enable_thinking=false 전의 오염된 상태일 수 있다)
        _sanitize_history_inplace(histories[session_id])

        # Ch 16: 세션별 JSONL 트랜스크립트 주입 (요청/세션별 격리 엔진에 동적 세팅)
        try:
            from core.memory.transcript import SessionTranscript as _Trans

            cfg = _app_state.get("config")
            sessions_dir = cfg.session.sessions_dir if cfg else ".nexus/sessions"
            transcript_enabled = cfg.session.transcript_enabled if cfg else True
            engine._transcript = _Trans(
                sessions_dir=sessions_dir,
                session_id=session_id,
                enabled=transcript_enabled,
                channel=channel,
            )
        except Exception as e:
            logger.warning("트랜스크립트 주입 실패 (%s): %s", session_id, e)
            engine._transcript = None

        # 프로젝트(폴더=프로젝트명)가 있으면 지식소스를 narrow한 tenant 사본을 쓴다(P3-3).
        # ★tenant 자체를 재대입하면 이 중첩 함수에서 tenant가 지역변수로 잡혀
        #   UnboundLocalError가 나므로, 별도 변수(_eff_tenant)에 담는다.
        _project = _resolve_project_for_session(getattr(tenant, "id", "default"), session_id)
        _eff_tenant = _narrow_tenant_for_project(tenant, _project)
        # 리팩토링 2: 세션/tenant/transcript를 공식 bind_request로 주입
        # (이전엔 engine._session_id 등 비공개 필드를 직접 치환 — race condition 위험)
        transcript = _build_transcript(session_id, channel=channel)
        engine.bind_request(
            session_id=session_id,
            tenant=_eff_tenant,
            transcript=transcript,
            channel=channel,
        )
        if _eff_tenant is not None:
            logger.info(
                "tenant 해석: %s (sources=%s)",
                _eff_tenant.id,
                _eff_tenant.allowed_knowledge_sources,
            )

        # 커스텀 인스트럭션(테넌트) + 프로젝트 인스트럭션을 시스템 프롬프트에 덧붙인다(P2-2/P3-3).
        # 엔진은 요청마다 새로 조립되므로 base 프롬프트에 1회만 덧붙어 누적되지 않는다.
        _instruction = _read_custom_instruction(getattr(_eff_tenant, "id", "default"))
        _proj_instr = _project.get("instruction", "") if _project else ""
        _style = _resolve_style_prompt_for(getattr(_eff_tenant, "id", "default"))
        if (_instruction or "").strip() or (_proj_instr or "").strip() or _style:
            engine.update_system_prompt(
                compose_system_prompt(
                    engine.system_prompt,
                    style=_style,
                    user_instruction=_instruction,
                    project_instruction=_proj_instr,
                )
            )

        # QueryEngine의 messages를 해당 세션의 히스토리로 교체
        # 도구 호출/결과 메시지는 토큰을 많이 차지하므로 제외하고,
        # user/assistant 텍스트 메시지만 예산 내에서 복원한다.
        engine.clear_messages()

        # user/assistant 텍스트 메시지만 필터링

        text_messages = []
        for msg in histories[session_id]:
            role = msg.role if isinstance(msg.role, str) else msg.role.value
            # tool_result, tool_use 메시지는 건너뛰고 user/assistant만
            if role in ("user", "assistant"):
                # 도구 호출이 포함된 assistant 메시지도 텍스트만 추출
                text = msg.text_content if hasattr(msg, "text_content") else str(msg.content)
                if text and len(text) > 5:  # 빈 메시지 제외
                    text_messages.append(msg)

        # 예산 내에서 최근 메시지만 복원 (2,000 토큰 = ~6,000자)
        budget = 6000
        used = len(request.message)
        restored = []
        for msg in reversed(text_messages):
            content = msg.text_content if hasattr(msg, "text_content") else str(msg.content)
            if used + len(content) > budget:
                break
            restored.append(msg)
            used += len(content)
        restored.reverse()
        for msg in restored:
            engine._messages.append(msg)

        # 이 턴에서 새로 추가되는 메시지의 시작 인덱스 — 아래(스트림 종료 후)에서
        # tool_result 메시지의 다운로드 URL을 정확히 뽑기 위한 기준점.
        dl_start_idx = len(engine._messages)
        # 스트림으로 흘려보낸 답변 텍스트를 함께 모아 둔다 — 스트림이 끝난 뒤
        # 숫자 인용 검증(문서 원문과 자릿수 대조)에 쓴다. 표시는 이미 나간 뒤라
        # 경고는 마지막에 별도 text 프레임으로 덧붙인다.
        answer_parts: list[str] = []

        # ─── 요청 단위 타이밍/관측 로그 ───────────────────
        # 첨부 파일 경로가 메시지에 포함되면 업로드 케이스로 표시
        has_attach = "서버 경로:" in request.message or "[첨부파일:" in request.message
        req_start_mono = time.monotonic()
        event_count = 0
        stream_abort_error: BaseException | None = None

        # tool_use_id → 표시용 메타(name, tool_desc). TOOL_USE_START/STOP에서 채우고,
        # 뒤이어 오는 TOOL_RESULT 프레임에서 조회해 "접힌 활동라인 + 접힌 요약"의
        # 라벨(누가/무슨 작업)을 결과 라인과 일치시킨다.
        tool_meta: dict[str, dict[str, str]] = {}

        logger.info(
            "SSE 시작: session=%s, message_len=%d, has_attach=%s",
            session_id,
            len(request.message),
            has_attach,
        )

        # ─── Heartbeat/Producer 분리 구조 ──────────────────
        # submit_message의 이벤트 yield 사이에 긴 공백(Scout 호출 등)이 있으면
        # 브라우저/프록시가 연결을 끊거나 사용자가 "무한 로딩"으로 느낀다.
        # 이벤트는 Queue로 수거하고, 메인 루프는 get에 타임아웃을 걸어 일정 주기
        # 마다 SSE 주석(`: ping`) 프레임을 전송한다. SSE 주석은 EventSource 클라이언트
        # 에서 무시되므로 기존 JS 파서에 영향을 주지 않는다.
        sse_sentinel: tuple[str, Any] = ("done", None)
        sse_heartbeat_seconds = 20.0  # 20s마다 keep-alive

        event_queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()

        async def _producer() -> None:
            """submit_message 스트림을 큐로 옮긴다 (에러까지 포함)."""
            try:
                async for ev in engine.submit_message(request.message):
                    await event_queue.put(("event", ev))
            except BaseException as e:  # noqa: BLE001 — 모든 예외를 에러 프레임으로
                await event_queue.put(("error", e))
            finally:
                await event_queue.put(sse_sentinel)

        producer_task = asyncio.create_task(_producer())
        try:
            while True:
                try:
                    kind, payload = await asyncio.wait_for(
                        event_queue.get(), timeout=sse_heartbeat_seconds
                    )
                except TimeoutError:
                    # 이벤트 공백 → heartbeat. SSE 주석은 data 프레임이 아니므로
                    # 클라이언트 JSON 파서가 건드리지 않는다.
                    # (Python 3.11+에서 asyncio.TimeoutError는 builtin TimeoutError의 별칭)
                    elapsed = time.monotonic() - req_start_mono
                    yield f": heartbeat {elapsed:.0f}s\n\n"
                    continue

                if kind == "done":
                    break
                if kind == "error":
                    stream_abort_error = payload
                    # 에러를 클라이언트가 이해할 수 있는 형태로 변환
                    err_frame = {
                        "type": "error",
                        "session_id": session_id,
                        "error_code": "stream_aborted",
                        "message": (f"{type(payload).__name__}: {payload}"),
                    }
                    yield ("data: " + json.dumps(err_frame, ensure_ascii=False) + "\n\n")
                    break

                event = payload
                if isinstance(event, StreamEvent):
                    # 이벤트 타입 문자열(enum/문자열 양쪽 정규화). 아래 tool_result
                    # 분기 판정과 프레임 type 필드에 함께 쓴다.
                    etype = event.type if isinstance(event.type, str) else event.type.value
                    sse_data: dict[str, Any] = {
                        "type": etype,
                        "session_id": engine.session_id,
                    }
                    if event.text:
                        sse_data["text"] = event.text
                        # 본문 텍스트만 모은다(도구 라벨 등 다른 이벤트의 text 제외).
                        if etype == "text_delta":
                            answer_parts.append(event.text)
                    if event.message:
                        sse_data["message"] = event.message
                    # 중간 활동 표시용: 도구 이벤트(TOOL_USE_START/STOP)가 담고 온
                    # 도구 이름 + 사람이 읽는 라벨(tool_desc)을 프레임에 실어, UI가
                    # "지금 무슨 작업(누가) 중"을 클로드처럼 접힌 활동라인으로 렌더할 수
                    # 있게 한다. (표시는 이미 흐르는 이벤트를 그리는 것 — 추가 토큰/GPU 비용 없음)
                    if event.tool_use is not None:
                        tu = event.tool_use
                        sse_data["tool_name"] = tu.name
                        # tool_use_id → 표시 메타를 등록/갱신해, 뒤이어 오는 TOOL_RESULT
                        # 프레임이 같은 이름/라벨로 결과 라인을 그릴 수 있게 한다.
                        _meta = tool_meta.setdefault(tu.id, {"name": tu.name})
                        _meta["name"] = tu.name
                        # Agent(서브에이전트) 호출이면 "{subagent_type}: {description}"
                        # 라벨을 만든다. 그 외 도구는 None → 프론트가 tool_name으로 폴백.
                        _desc = _compute_tool_desc(tu)
                        if _desc:
                            sse_data["tool_desc"] = _desc
                            _meta["tool_desc"] = _desc
                    # ─── 신규 tool_result 프레임 ───
                    # 도구/서브에이전트 실행 결과를 "접힌 요약(preview) + 펼침 원문(content)"
                    # 으로 내려보낸다. 프론트는 이 프레임으로 결과를 접힌 채 보여주고,
                    # 펼치면 원문을 노출한다(메인 답변에 원문 재출력 방지).
                    if etype == "tool_result" and event.tool_result is not None:
                        tr = event.tool_result
                        _rmeta = tool_meta.get(tr.tool_use_id, {})
                        _raw = tr.content or ""
                        # preview: 접힌 라인용 — 개행을 공백으로 치환한 첫 N글자(넘으면 "…").
                        _preview = (
                            _raw.replace("\r\n", " ").replace("\n", " ").replace("\r", " ")
                        )
                        if len(_preview) > _TOOL_PREVIEW_MAX:
                            _preview = _preview[:_TOOL_PREVIEW_MAX] + "…"
                        # content: 펼침용 전체 텍스트 — 상한 초과 시 말미에 절단 표식(큰 결과 방어).
                        _content = _raw
                        if len(_content) > _TOOL_CONTENT_MAX:
                            _content = _content[:_TOOL_CONTENT_MAX] + "…(truncated)"
                        sse_data["tool_use_id"] = tr.tool_use_id
                        sse_data["name"] = _rmeta.get("name", "")
                        if _rmeta.get("tool_desc"):
                            sse_data["tool_desc"] = _rmeta["tool_desc"]
                        sse_data["is_error"] = bool(tr.is_error)
                        sse_data["preview"] = _preview
                        sse_data["content"] = _content
                    if event.error_code:
                        sse_data["error_code"] = event.error_code
                    # 지식 RAG 출처 인용(Point 4-2) — KNOWLEDGE_SOURCES 이벤트의 출처
                    # 목록을 SSE로 그대로 통과시켜 UI가 실시간 표시하게 한다(서버 진실).
                    if event.knowledge_sources is not None:
                        sse_data["sources"] = [
                            c.model_dump() for c in event.knowledge_sources
                        ]
                    if event.usage:
                        sse_data["usage"] = {
                            "input_tokens": event.usage.input_tokens,
                            "output_tokens": event.usage.output_tokens,
                        }
                    if event.stop_reason:
                        stop_val = event.stop_reason
                        sse_data["stop_reason"] = (
                            stop_val if isinstance(stop_val, str) else stop_val.value
                        )
                    yield f"data: {json.dumps(sse_data, ensure_ascii=False)}\n\n"
                    event_count += 1

                    # ─── todo_update 프레임 (계획 체크리스트 가시화) ───
                    # TodoWrite/TodoRead 결과가 왔으면, 서버가 TodoStore에서 최신
                    # 목록을 직접 읽어 웹 전용 todo_update 프레임을 추가로 내보낸다.
                    # (모델 텍스트 파싱이 아니라 서버 진실 — DocumentExport URL 추출과
                    # 동일 원칙.) StreamEvent를 새로 만들지 않는다(4-Tier 체인 불변).
                    if etype == "tool_result" and sse_data.get("name") in (
                        "TodoWrite",
                        "TodoRead",
                    ):
                        todo_frame = _build_todo_update_frame(engine, session_id)
                        if todo_frame is not None:
                            yield f"data: {json.dumps(todo_frame, ensure_ascii=False)}\n\n"
                            event_count += 1
        finally:
            # producer가 아직 살아 있으면 취소 (클라이언트가 연결을 끊은 경우 등)
            if not producer_task.done():
                producer_task.cancel()
                try:
                    await producer_task
                except (asyncio.CancelledError, Exception):  # noqa: BLE001, S110
                    # producer 취소 시 예외는 이미 stream_abort_error 경로에서 처리됨
                    pass
            elapsed_total = time.monotonic() - req_start_mono
            if stream_abort_error is not None:
                logger.warning(
                    "SSE 중단: session=%s, elapsed=%.1fs, events=%d, error=%s",
                    session_id,
                    elapsed_total,
                    event_count,
                    type(stream_abort_error).__name__,
                )
            else:
                logger.info(
                    "SSE 완료: session=%s, elapsed=%.1fs, events=%d",
                    session_id,
                    elapsed_total,
                    event_count,
                )

        # 이 턴에 생성된 문서의 정확한 다운로드 정보(+미리보기 content)를 download
        # 프레임으로 보낸다 → UI가 모델 텍스트(오탈자 가능) 대신 이걸로 버튼/미리보기 생성.
        # TODO(nexus): TOOL_RESULT StreamEvent 발신 리팩터 후에는 consume 루프 안에서
        #   이벤트로 바로 download 프레임을 내보내고 이 사후 스캔을 제거한다.
        # 숫자 인용 검증 — 스트림으로 이미 나간 본문은 고치지 않고, 확인이 필요한
        # 숫자가 있으면 경고만 별도 text 프레임으로 덧붙인다(사용자가 판단하도록).
        _warning = _answer_warnings_for(
            "".join(answer_parts), engine._messages[dl_start_idx:]
        )
        if _warning:
            _warn_frame = {"type": "text_delta", "session_id": session_id, "text": _warning}
            yield f"data: {json.dumps(_warn_frame, ensure_ascii=False)}\n\n"

        _dls = _collect_downloads(engine._messages[dl_start_idx:])
        # 생성물 메타데이터를 tb_artifacts에 fail-soft 기록(pg 없으면 조용히 스킵).
        await _record_artifacts(_dls, tenant, session_id, turn=getattr(engine, "total_turns", None))
        for _dl in _dls:
            yield f"data: {json.dumps({'type': 'download', **_dl}, ensure_ascii=False)}\n\n"

        # 이번 턴의 user/assistant 텍스트 메시지만 히스토리에 저장
        # tool_result/tool_use 메시지는 토큰이 크므로 저장하지 않는다
        # Qwen3.5의 <think>…</think> 블록이 혹여 섞여 들어오면 다음 턴의
        # in-context 모방을 유발하므로 저장 전에 제거한다 (safeguard).
        for msg in engine._messages:
            role = msg.role if isinstance(msg.role, str) else msg.role.value
            if role in ("user", "assistant"):
                content = msg.text_content if hasattr(msg, "text_content") else str(msg.content)
                if content and len(content) > 5 and msg not in histories[session_id]:
                    content_clean = _strip_thinking(content)
                    if content_clean and content_clean != content:
                        # content가 정제됐다면 원본 Message는 그대로 두되 저장용
                        # 얕은 복사본을 만들어 히스토리에 넣는다. 원본 Message는
                        # Pydantic frozen이므로 text를 바꿀 수 없다 → 새 Message 생성.
                        from core.message import Message

                        new_msg = (
                            Message.assistant(content_clean)
                            if role == "assistant"
                            else Message.user(content_clean)
                        )
                        histories[session_id].append(new_msg)
                    elif content_clean:
                        histories[session_id].append(msg)

        # Ch 16: Redis에 write-through — 인메모리 히스토리를 JSON 직렬화하여 저장
        # QueryEngine.on_turn_end도 자체적으로 save_conversation_context를 호출하지만,
        # 웹의 경우 histories[session_id]가 "여러 턴 누적된 완전한 대화"이므로
        # 여기서도 한 번 더 저장하여 서버 재기동 시 UI에 보이는 대화 그대로 복원.
        if memory_manager is not None:
            try:
                serialized: list[dict[str, Any]] = []
                for m in histories[session_id]:
                    role = m.role if isinstance(m.role, str) else m.role.value
                    if role not in ("user", "assistant"):
                        continue
                    text = m.text_content if hasattr(m, "text_content") else str(m.content)
                    if text:
                        serialized.append({"role": role, "content": text})
                await memory_manager.short_term.save_conversation_context(
                    session_id, serialized, ttl=86400, channel=channel
                )
            except Exception as e:
                logger.warning("세션 Redis 저장 실패 (%s): %s", session_id, e)

    return StreamingResponse(
        _locked_generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Session-ID": session_id,
        },
    )


# ─────────────────────────────────────────────
# OpenAI 호환 엔드포인트 (POST /v1/chat/completions)
# ─────────────────────────────────────────────
# 왜 필요한가: AgentHub·.NET·LangChain 등 "OpenAI 클라이언트"는 이미 이 규격을 말한다.
# 이 엔드포인트를 붙이면 그들이 base_url만 Nexus로 바꿔 커스텀 코드 없이 붙을 수 있다.
# 기존 /v1/chat·/v1/chat/stream 은 그대로 두고(무손상), 4-Tier 체인(submit_message가
# yield하는 StreamEvent만 소비)도 절대 우회하지 않는다.
def _normalize_openai_content(
    messages: list[OpenAIChatMessage],
) -> tuple[list[OpenAIChatMessage], list[str]]:
    """`content` 파트 배열을 문자열로 되돌린다. 이미지는 파일로 내리고 핸들만 남긴다.

    [왜 여기서 하나 — 2026-08-12]
      Pydantic 검증기에 넣고 싶지만 이미지 저장은 I/O 이고 설정(uploads_dir)이 필요해
      모델 안에 둘 수 없다. 그래서 `_split_openai_messages` **앞**에서 한 번 돌린다.
      이 아래 파이프라인은 종전대로 문자열 content 만 보게 되므로 무회귀다.

    Returns:
        (정규화된 메시지, 경고 목록). 경고는 응답의 `warnings` 로 나간다 —
        도구를 조용히 버리지 않기로 한 것과 같은 이유로, 이미지도 조용히 버리지 않는다.
    """
    from core.storage.inline_images import (
        InlineImageError,
        build_image_handle,
        parse_data_url,
        save_inline_image,
    )

    warnings: list[str] = []
    # 이미지가 하나도 없으면 디렉토리 생성조차 하지 않는다(불필요한 부작용 회피).
    uploads_dir: Path | None = None
    max_bytes = _vision_max_bytes()
    img_index = 0
    out: list[OpenAIChatMessage] = []

    for msg in messages:
        if not isinstance(msg.content, list):
            out.append(msg)
            continue

        texts: list[str] = []
        handles: list[str] = []
        for part in msg.content:
            if not isinstance(part, dict):
                warnings.append("형식이 올바르지 않은 content 파트를 건너뛰었습니다.")
                continue
            ptype = str(part.get("type") or "")
            if ptype == "text":
                texts.append(str(part.get("text") or ""))
            elif ptype == "image_url":
                raw_url = part.get("image_url")
                url = raw_url.get("url") if isinstance(raw_url, dict) else raw_url
                try:
                    mime, raw = parse_data_url(str(url or ""), max_bytes)
                    if uploads_dir is None:
                        uploads_dir = _uploads_dir()
                    path = save_inline_image(raw, mime, uploads_dir)
                except InlineImageError as e:
                    # 이미지 하나가 이상하다고 대화 전체를 죽이지 않는다. 다만 조용히
                    # 넘기지도 않는다 — 왜 못 봤는지 모델과 사용자 둘 다 알아야 한다.
                    warnings.append(f"이미지를 받아들이지 못했습니다: {e}")
                    texts.append(f"[이미지 첨부 실패: {e}]")
                    continue
                img_index += 1
                handles.append(build_image_handle(path, img_index))
            else:
                warnings.append(f"지원하지 않는 content 파트를 건너뛰었습니다: {ptype}")

        merged = "\n\n".join([t for t in texts if t] + handles)
        out.append(msg.model_copy(update={"content": merged}))

    return out, warnings


def _split_openai_messages(
    messages: list[OpenAIChatMessage],
) -> tuple[str | None, list[Any], str, bool]:
    """OpenAI messages 배열을
    (system_content, prior_messages, last_user_text, is_tool_continuation)로 분해한다.

    - system 메시지: 여러 개면 순서대로 이어붙여 하나의 지시문으로 만든다(없으면 None).
    - system 을 제외한 user/assistant 는 순서대로 core.Message 로 변환해 히스토리로 쓴다.
    - 마지막(가장 최근) 메시지는 반드시 user 여야 한다 → submit_message 에 넘길 질문.
      마지막이 user 가 아니면 400 (OpenAI 관례상 마지막은 사용자 발화).

    반환된 prior_messages 에는 '마지막 user'는 포함하지 않는다(그건 last_user_text).

    [도구 결과로 이어 도는 호출 — 2026-08-08]
    클라이언트가 도구를 직접 실행하는 방식에서는 대화가 user 로 끝나지 않는다.

        user → assistant(tool_calls) → tool(결과)   ← 여기서 다시 호출한다

    이때는 '마지막은 user' 규칙을 적용하지 않는다(적용하면 표준 도구 루프가
    400 으로 막힌다). 대신 히스토리 전체를 재현하고 새 user 발화 없이 이어 돌린다.
    반환 튜플의 마지막 값은 그 판정 결과(is_tool_continuation)다.
    """
    from core.message import Message

    if not messages:
        raise HTTPException(status_code=400, detail="messages 배열이 비어 있습니다.")

    # 1) system 지시문 수집(여러 개면 결합).
    system_parts = [m.content for m in messages if m.role == "system" and m.content]
    system_content = "\n\n".join(system_parts) if system_parts else None

    # 2) system 을 제외한 대화 흐름.
    convo = [m for m in messages if m.role != "system"]
    if not convo:
        raise HTTPException(status_code=400, detail="messages 배열이 비어 있습니다.")

    # 도구 결과로 끝나면 '이어 도는 호출'이다(새 user 발화 없음).
    is_tool_continuation = convo[-1].role == "tool"

    if is_tool_continuation:
        # 히스토리를 하나도 빼지 않고 전부 재현한다. 라우팅 판정에 쓸 텍스트로는
        # 가장 최근 user 발화를 그대로 넘긴다(히스토리에 다시 추가하지는 않는다).
        history_src = convo
        last_user_text = next(
            (m.content or "" for m in reversed(convo) if m.role == "user"), ""
        )
    else:
        if convo[-1].role != "user":
            raise HTTPException(
                status_code=400,
                detail="마지막 메시지는 role='user' 또는 role='tool' 이어야 합니다.",
            )
        last_user_text = convo[-1].content or ""
        if not last_user_text.strip():
            raise HTTPException(
                status_code=400, detail="마지막 user 메시지 content가 비어 있습니다."
            )
        history_src = convo[:-1]

    # 3) 이전 대화를 core.Message 로 변환.
    prior_messages: list[Any] = []
    for m in history_src:
        if m.role == "user":
            if m.content:
                prior_messages.append(Message.user(m.content))
        elif m.role == "assistant":
            # 도구를 요청한 assistant 턴은 content 가 비어 있는 것이 정상이다
            # (모델이 말 대신 도구를 불렀다). content 만 보고 건너뛰면 모델이
            # 자기가 무엇을 요청했는지 잊고 같은 도구를 다시 부른다.
            tool_uses = _openai_tool_calls_to_uses(m.tool_calls)
            if tool_uses or m.content:
                prior_messages.append(
                    Message.assistant(m.content or "", tool_uses=tool_uses or None)
                )
        elif m.role == "tool" and m.tool_call_id:
            prior_messages.append(
                Message.tool_result(
                    tool_use_id=m.tool_call_id,
                    content=m.content or "",
                )
            )

    return system_content, prior_messages, last_user_text, is_tool_continuation


def _openai_tool_calls_to_uses(
    tool_calls: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """OpenAI `tool_calls` → `Message.assistant(tool_uses=...)` 가 받는 dict 목록.

    OpenAI 규격은 인자를 **JSON 문자열**로 싣지만(`function.arguments`) 내부 계약은
    파싱된 dict 다. 여기서 한 번만 변환해 둔다.

    파싱에 실패하면 그 호출을 버리지 않고 인자를 빈 dict 로 둔다 — 모델이 "이 도구를
    불렀다"는 사실 자체는 남아야 같은 호출을 반복하지 않기 때문이다.
    """
    uses: list[dict[str, Any]] = []
    for call in tool_calls or []:
        if not isinstance(call, dict):
            continue
        fn = call.get("function") if isinstance(call.get("function"), dict) else {}
        name = str(fn.get("name") or call.get("name") or "").strip()
        if not name:
            continue
        raw_args = fn.get("arguments", call.get("arguments"))
        if isinstance(raw_args, dict):
            parsed = raw_args
        else:
            try:
                parsed = json.loads(raw_args) if raw_args else {}
            except (TypeError, ValueError):
                parsed = {}
            if not isinstance(parsed, dict):
                parsed = {}
        uses.append(
            {
                "id": str(call.get("id") or f"call_{uuid.uuid4().hex[:12]}"),
                "name": name,
                "input": parsed,
            }
        )
    return uses


def _tool_uses_to_openai_tool_calls(
    tool_uses: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """내부 도구 호출 → OpenAI `tool_calls` 응답 형식.

    인자는 규격대로 JSON **문자열**로 직렬화한다. `ensure_ascii=False` 로 한글을
    이스케이프하지 않는다 — 클라이언트가 로그에 그대로 찍어 읽을 수 있어야 한다.
    """
    return [
        {
            "id": use.get("id") or f"call_{uuid.uuid4().hex[:12]}",
            "type": "function",
            "function": {
                "name": use.get("name") or "",
                "arguments": json.dumps(use.get("input") or {}, ensure_ascii=False),
            },
        }
        for use in tool_uses
    ]


def _client_tools_instruction(client_tools: list[Any]) -> str:
    """클라이언트 도구로 교체했을 때 프롬프트를 실제 도구 목록에 맞춘다.

    기본 웹 프롬프트는 Read/Write/Edit 같은 **서버 도구**를 안내한다. 도구 풀을
    클라이언트 도구로 통째로 바꾸면 그 안내가 거짓이 되고, 모델은 없는 도구를
    부르려다 실패한다. 이 리포에서 이미 같은 원인으로 `알 수 없는 도구: 'Agent'`
    버그가 났었다 — 그래서 프롬프트를 실제 목록에서 유도한다.

    session_instruction 으로 들어가 기본 프롬프트 뒤에 붙으므로 앞의 안내를 덮는다.

    [잔류 서버 도구도 함께 안내한다 — 2026-08-12]
      혼합 풀(_mixed_tool_pool)이 `AnalyzeImage` 를 남기는데 여기서 "정확히 이것들뿐"
      이라고 말하면 **프롬프트가 실제 풀보다 좁아진다.** 모델은 있는 도구를 안 쓰게
      되고, 이미지를 받아 놓고도 분석하지 않는다. 위 `알 수 없는 도구: 'Agent'` 와
      정확히 대칭인 불일치다(그때는 넓게 말해 없는 도구를 불렀다).
    """
    if not client_tools:
        return ""
    from core.tools.implementations.client_tool import SERVER_TOOLS_KEPT_WITH_CLIENT_TOOLS

    names = ", ".join(t.name for t in client_tools)
    kept = ", ".join(SERVER_TOOLS_KEPT_WITH_CLIENT_TOOLS)
    return (
        "## Available tools (overrides any tool list above)\n"
        f"You have exactly these tools: {names}.\n"
        "Any other tool mentioned earlier is NOT available in this session — "
        "do not attempt to call it.\n"
        "These tools run on the user's own machine, so they see the user's project "
        "files, not the server's. Call them to inspect and change real code rather "
        "than guessing or printing code blocks and asking the user to apply them.\n"
        f"\nOne server-side tool is also available: {kept}. "
        "Use it whenever the conversation contains an uploaded image path and the "
        "user asks anything about that image. Call it again for each new question — "
        "an earlier summary may not contain the answer."
    )


def _inject_openai_context(
    engine: Any,
    system_content: str | None,
    prior_messages: list[Any],
    session_id: str,
    tenant: Any,
) -> int:
    """OpenAI 요청의 system/히스토리를 (세션별 격리) 엔진에 주입하고 다운로드 스캔 기준
    인덱스(dl_start_idx)를 돌려준다.

    - 소비자 system 지시문은 Nexus 기본 프롬프트를 유지한 채 '뒤에 덧붙인다'. 엔진은
      요청마다 새로 조립되는 세션 전용 인스턴스라 이 변형은 다른 요청에 영향을 주지 않는다.
      (base 원본을 먼저 engine.system_prompt 로 보관한 뒤 결합한다.)
    - OpenAI 클라이언트는 매 요청 전체 히스토리를 보내므로 Redis 세션 복원은 하지 않는다
      (무상태). clear 후 요청의 이전 user/assistant 만 순서대로 얹는다.
    """
    # system 지시문 반영 — 기본 프롬프트 원본을 먼저 보관 후 뒤에 덧붙인다.
    _style = _resolve_style_prompt_for(getattr(tenant, "id", "default"))
    if system_content or _style:
        # OpenAI 소비자가 보낸 system 메시지는 "이번 요청 한정" 지시로 취급한다.
        engine.update_system_prompt(
            compose_system_prompt(
                engine.system_prompt,
                style=_style,
                session_instruction=system_content,
            )
        )

    # 세션/tenant/transcript 를 공식 bind_request 로 주입(기존 핸들러와 동일 계약).
    # OpenAI 호환 API 경로이므로 channel="api"로 격리(web 히스토리 목록에 안 섞인다).
    transcript = _build_transcript(session_id, channel="api")
    engine.bind_request(
        session_id=session_id, tenant=tenant, transcript=transcript, channel="api"
    )

    # 무상태: 요청이 보낸 히스토리만 그대로 얹는다(Redis 복원 안 함).
    engine.clear_messages()
    engine._messages.extend(prior_messages)

    # 이 턴에 새로 생기는 tool_result 메시지에서만 다운로드를 추출하기 위한 기준점.
    return len(engine._messages)


def _citation_settings() -> tuple[bool, bool, int, str]:
    """config.knowledge_rag.citation에서 응답 노출 설정을 읽는다(출처 인용 Point 4-2).

    반환: (expose_in_response, strip_invalid_labels, max_sources, label).
    config 미로드/테스트에서는 CitationConfig 기본값(True/True/5/"출처")으로 폴백한다.
    이 값들은 '웹 계층 관심사'(응답 shaping)라 retriever가 아니라 여기서 읽는다.
    """
    cfg = _app_state.get("config")
    citation = getattr(getattr(cfg, "knowledge_rag", None), "citation", None)
    if citation is None:
        return True, True, 5, "출처"
    return (
        bool(citation.expose_in_response),
        bool(citation.strip_invalid_labels),
        int(citation.max_sources),
        str(citation.label),
    )


def _strip_invalid_citation_labels(
    text: str, valid_indices: set[int], label: str
) -> str:
    """응답 본문에서 '주입 범위 밖' 출처 번호 마커([출처9] 등)를 제거한다.

    모델이 지어낸(주입되지 않은 번호를 참조하는) 라벨을 지워 "근거 있어 보이는
    할루시네이션"을 줄인다. valid_indices(실제 주입된 출처 번호)에 있는 번호는
    그대로 둔다. label은 config에서 오며 정규식 특수문자를 이스케이프한다.
    """
    if not text:
        return text
    pattern = re.compile(r"\[" + re.escape(label) + r"(\d+)\]")

    def _repl(m: re.Match) -> str:
        # 유효 번호면 원문 유지, 아니면 빈 문자열로 치환(마커 제거).
        return m.group(0) if int(m.group(1)) in valid_indices else ""

    return pattern.sub(_repl, text)


def _downloads_markdown(downloads: list[dict[str, str]]) -> str:
    """다운로드 목록을 표준 OpenAI 클라이언트도 볼 수 있는 마크다운 링크 블록으로 만든다.

    비표준 downloads 배열 필드를 못 읽는 클라이언트라도 content 안의 링크는 볼 수 있게 한다.
    """
    if not downloads:
        return ""
    links = "\n".join(
        f"- [{d.get('filename', 'file')}]({d.get('url', '')})" for d in downloads
    )
    return "\n\n---\n**첨부 문서:**\n" + links


def _build_structured_output_spec(response_format: dict[str, Any] | None) -> Any:
    """OpenAI response_format 필드를 내부 StructuredOutputSpec으로 변환한다(fail-closed).

    None이면 None을 돌려주어 일반 생성 경로를 유지한다(무회귀). 값이 있으면 아래
    규칙으로 검증하며, 어긋나면 조용히 무시하지 않고 400을 던진다(과거 extra=ignore가
    바로 이 결함이었다 — 드롭인 프로바이더 규격 위반).

      - structured_output.enabled=false            → 400 (기능 자체가 꺼짐)
      - type == "json_schema": json_schema.schema(객체) 필수. name/strict 반영.
      - type == "json_object": 스키마 없는 JSON 강제 → {"type": "object"}로 정규화.
      - 그 외 type                                 → 400 (미지원)
      - 직렬화 크기 > max_schema_bytes             → 400 (xgrammar 컴파일 지연 차단)

    반환된 스펙은 호출 단위 인자로 submit_message에 넘긴다(세션 오염 방지, R8).
    """
    if response_format is None:
        return None

    from core.model.inference import StructuredOutputSpec

    # config 접근 — 부트스트랩 전/테스트에서는 None일 수 있어 기본값으로 폴백한다.
    cfg = _app_state.get("config")
    so_cfg = getattr(cfg, "structured_output", None) if cfg else None

    # 마스터 스위치 — 꺼져 있으면 조용히 무시하지 않고 명시적으로 거부한다.
    if so_cfg is not None and not so_cfg.enabled:
        raise HTTPException(
            status_code=400,
            detail="구조화 출력이 비활성화되어 있습니다(structured_output.enabled=false).",
        )

    default_strict = so_cfg.strict if so_cfg is not None else True
    max_bytes = so_cfg.max_schema_bytes if so_cfg is not None else 65536

    rf_type = response_format.get("type")
    if rf_type == "json_schema":
        js = response_format.get("json_schema") or {}
        schema = js.get("schema")
        if not isinstance(schema, dict):
            raise HTTPException(
                status_code=400,
                detail="response_format.json_schema.schema(객체)가 필요합니다.",
            )
        name = js.get("name", "nexus_structured")
        strict = js.get("strict", default_strict)
    elif rf_type == "json_object":
        # 스키마 없는 JSON 강제 — 최소 object 스키마로 정규화한다.
        schema = {"type": "object"}
        name = "nexus_structured"
        strict = default_strict
    else:
        raise HTTPException(
            status_code=400,
            detail=f"지원하지 않는 response_format.type: {rf_type!r} "
            "('json_schema' 또는 'json_object'만 허용)",
        )

    # 스키마 크기 상한 검사 — 거대/재귀 스키마의 문법 컴파일 지연을 사전 차단한다.
    schema_bytes = len(json.dumps(schema, ensure_ascii=False).encode("utf-8"))
    if schema_bytes > max_bytes:
        raise HTTPException(
            status_code=400,
            detail=f"스키마 크기({schema_bytes}B)가 상한({max_bytes}B)을 초과했습니다.",
        )

    return StructuredOutputSpec(json_schema=schema, name=name, strict=strict)


@app.post("/v1/chat/completions")
async def chat_completions(
    request: OpenAIChatCompletionRequest,
    x_tenant_id: str | None = Header(default=None, alias="X-Tenant-ID"),
    authorization: str | None = Header(default=None),
    x_client_id: str | None = Header(default=None, alias="X-Client-Id"),
    x_request_id: str | None = Header(default=None, alias="X-Request-ID"),
    x_query_class: str | None = Header(default=None, alias="X-Nexus-Query-Class"),
) -> Any:
    """OpenAI 호환 채팅 완성 엔드포인트.

    외부 OpenAI 클라이언트가 base_url 만 Nexus 로 바꿔 붙을 수 있게 한다.
    stream=false 면 chat.completion(JSON), stream=true 면 chat.completion.chunk(SSE)를 낸다.
    4-Tier 체인은 submit_message 이벤트만 소비하여 우회하지 않는다.
    """
    tenant = _resolve_tenant(request.tenant_id, x_tenant_id, authorization)

    # ── 요청 ID 추적 (2026-08-13) ────────────────────────────────
    # 세션 ID 를 여기서 먼저 만든다. 아래 도구 검증에서 400 으로 빠지는 경로가 있는데,
    # 그 전에 로그 한 줄을 남겨야 "요청 ID xxxx 가 실패했다"는 문의를 추적할 수 있다.
    # (실측: 2026-08-13 문의의 요청 ID 두 건 모두 로그 grep 0건이었다.)
    session_id = str(uuid.uuid4())
    _request_id = _sanitize_request_id(x_request_id)
    # 잘못된 값이면 여기서 400 — StreamingResponse 가 시작된 뒤에는 못 낸다.
    _forced_class = _resolve_query_class(request.query_class, x_query_class)
    logger.info(
        "[요청추적] session=%s request_id=%s client=%s tenant=%s stream=%s class=%s",
        session_id,
        _request_id or "-",
        _sanitize_client_id(x_client_id) or "-",
        # TenantConfig 의 식별자 필드는 `id` 다(`tenant_id` 가 아니다 — 요청 본문의
        # 필드명과 달라서, 틀리면 getattr 기본값 때문에 조용히 "-" 로만 찍힌다).
        getattr(tenant, "id", None) or "-",
        request.stream,
        _forced_class or "auto",
    )

    # 요청 검증/분해는 StreamingResponse 생성 '이전'에 수행해야 400을 정상 반환한다
    # (제너레이터 안에서 raise 하면 이미 200 스트림이 시작된 뒤라 400이 안 나감).
    # content 파트 배열(OpenAI 비전 규격)을 문자열로 되돌린다. 이미지는 파일로 내리고
    # 대화에는 서버 경로 핸들만 남는다 — 아래 파이프라인은 종전대로 문자열만 본다.
    normalized_messages, image_warnings = _normalize_openai_content(request.messages)

    (
        system_content,
        prior_messages,
        last_user_text,
        is_tool_continuation,
    ) = _split_openai_messages(normalized_messages)

    # 클라이언트가 실행할 도구 스키마(있으면 이번 요청의 도구 목록을 전부 교체한다).
    # tool_choice="none" 은 "도구 쓰지 말고 답하라"이므로 아예 만들지 않는다.
    from core.tools.implementations.client_tool import build_client_tools

    # 버려진 도구가 있으면 경고를 함께 받는다. 조용히 버리면 플러그인 개발자가
    # "왜 내 도구를 안 쓰지?"의 원인을 찾을 수 없다(2026-08-08).
    # 이미지 경고도 같은 통로로 내보낸다 — 조용히 버리지 않는다는 원칙은 도구와 같다.
    tool_warnings: list[str] = list(image_warnings)
    if request.tools and request.tool_choice != "none":
        # ★재대입하지 않는다 — 위에서 담은 이미지 경고가 덮여 사라진다.
        client_tools, _tool_w = build_client_tools(request.tools)
        tool_warnings.extend(_tool_w)
        if not client_tools:
            # 도구를 보냈는데 하나도 쓸 수 없다 — 조용히 도구 없는 대화로 강등하면
            # 클라이언트는 루프가 성립하지 않는 이유를 영영 모른다. 명시적으로 거부한다
            # (구조화 출력 비활성 시 400 으로 거부하는 것과 같은 원칙).
            raise HTTPException(
                status_code=400,
                detail="tools 를 보냈으나 사용할 수 있는 도구가 없습니다. "
                + (" / ".join(tool_warnings) if tool_warnings else "형식을 확인하세요."),
            )
        for _w in tool_warnings:
            # 세션 ID 를 실어야 위의 [요청추적] 줄과 이어붙일 수 있다("openai" 고정값이었다).
            logger.warning("클라이언트 도구 경고: session=%s, %s", session_id, _w)
    else:
        client_tools = []
    # 도구를 교체했으면 프롬프트도 실제 목록에 맞춘다(불일치 방지 — 위 함수 설명 참고).
    _tools_note = _client_tools_instruction(client_tools)
    if _tools_note:
        system_content = f"{system_content}\n\n{_tools_note}" if system_content else _tools_note

    # 구조화 출력 스펙 변환 — StreamingResponse '이전'에 수행해야 400을 정상 반환한다
    # (제너레이터 안에서 raise하면 이미 200 스트림이 시작된 뒤라 400이 안 나감).
    structured_output = _build_structured_output_spec(request.response_format)

    # session_id 는 위(요청 ID 추적)에서 이미 만들었다 — OpenAI 에는 session_id
    # 개념이 없어 요청마다 임시 세션으로 무상태 처리한다.
    # 소비자 식별(2026-08-05) — api 채널에는 여러 외부 클라이언트의 대화가 함께
    # 쌓여 구분이 불가능했다. 헤더를 준 클라이언트만 메타에 남긴다(선택 사항).
    _client_id = _sanitize_client_id(x_client_id)
    if _client_id:
        _record_client_meta(session_id, "api", _client_id, tenant)
    # 응답에 반향할 모델명(요청값 우선, 없으면 기본값).
    model_name = request.model or "ax-4.0"

    # ── 스트리밍 응답 ──────────────────────────────
    if request.stream:
        return StreamingResponse(
            _openai_stream_generate(
                session_id=session_id,
                tenant=tenant,
                model_name=model_name,
                system_content=system_content,
                prior_messages=prior_messages,
                last_user_text=last_user_text,
                structured_output=structured_output,
                max_tokens=request.max_tokens,
                client_tools=client_tools,
                is_tool_continuation=is_tool_continuation,
                tool_warnings=tool_warnings,
                forced_query_class=_forced_class,
                request_id=_request_id,
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )

    # ── 비스트림 응답 ─────────────────────────────
    engine = _acquire_session_engine(session_id, tenant, client_tools=client_tools)
    if engine is None:
        # 엔진 미초기화(부트스트랩 실패/테스트) — OpenAI 규격의 최소 응답으로 폴백.
        return OpenAIChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex}",
            created=int(time.time()),
            model=model_name,
            choices=[
                OpenAIChoice(
                    message=OpenAIResponseMessage(
                        content="QueryEngine이 아직 초기화되지 않았습니다.",
                    ),
                )
            ],
        )

    from core.message import StreamEvent, StreamEventType

    dl_start_idx = _inject_openai_context(
        engine, system_content, prior_messages, session_id, tenant
    )

    response_text_parts: list[str] = []
    usage = OpenAIUsage()
    # 마지막 종료 이유 — 토큰 한도로 잘렸는지(finish_reason="length") 판정에 쓴다.
    last_stop_reason: Any = None
    # 클라이언트가 실행해야 할 도구 호출(서버는 실행하지 않고 그대로 돌려준다).
    client_tool_names = {t.name for t in client_tools}
    pending_tool_uses: list[dict[str, Any]] = []

    # 4-Tier 체인 우회 금지: submit_message 가 yield 하는 StreamEvent 만 소비한다.
    # 요청의 max_tokens를 엔진까지 전달한다(2026-08-05). 이전에는 무시되어
    # 클라이언트가 출력 크기를 제어할 수 없었다. 0 이하는 무시(None 폴백).
    _max_tokens_req = request.max_tokens if (request.max_tokens or 0) > 0 else None
    async for event in engine.submit_message(
        last_user_text,
        structured_output=structured_output,
        max_tokens_override=_max_tokens_req,
        # 도구 결과로 이어 도는 호출이면 새 user 발화를 만들지 않는다.
        append_user_message=not is_tool_continuation,
        # 요청이 클래스를 지정했으면 서버 분류를 건너뛴다(지식 RAG 주입 여부가 갈린다).
        forced_query_class=_forced_class,
        request_id=_request_id,
    ):
        if not isinstance(event, StreamEvent):
            continue
        if event.type == StreamEventType.TEXT_DELTA and event.text:
            response_text_parts.append(event.text)
        elif (
            event.type == StreamEventType.TOOL_USE_STOP
            and event.tool_use
            and event.tool_use.name in client_tool_names
        ):
            # 클라이언트 실행 도구만 수집한다. 서버 도구(DocumentExport 등)는 서버가
            # 이미 실행했으므로 클라이언트에 넘기면 두 번 실행된다.
            pending_tool_uses.append(
                {
                    "id": event.tool_use.id,
                    "name": event.tool_use.name,
                    "input": event.tool_use.input or {},
                }
            )
        elif event.type == StreamEventType.USAGE_UPDATE and event.usage:
            usage = OpenAIUsage(
                prompt_tokens=event.usage.input_tokens,
                completion_tokens=event.usage.output_tokens,
                total_tokens=event.usage.total_tokens,
            )
        elif event.type == StreamEventType.MESSAGE_STOP and event.stop_reason:
            # 턴마다 갱신 — 마지막 값이 이 응답의 최종 종료 이유다.
            last_stop_reason = event.stop_reason
        # tool_use/tool_result/thinking 등은 OpenAI 표준 content 에 없으므로 무시한다.

    # 문서 생성 다운로드를 이 턴 메시지에서 추출(기존 헬퍼 재사용).
    downloads = _collect_downloads(engine._messages[dl_start_idx:])
    # 생성물 메타데이터를 tb_artifacts에 fail-soft 기록(pg 없으면 조용히 스킵).
    await _record_artifacts(
        downloads, tenant, session_id, turn=getattr(engine, "total_turns", None)
    )

    content = "".join(response_text_parts)
    # 숫자 인용 검증 — 문서에서 옮긴 금액·수량의 자릿수가 원문과 다르면 경고를 덧붙인다.
    content += _answer_warnings_for(content, engine._messages[dl_start_idx:])
    # 표준 클라이언트도 링크를 볼 수 있게 content 끝에 마크다운으로 덧붙인다.
    content += _downloads_markdown(downloads)

    # 토큰 한도로 잘렸으면 "length"로 정직하게 알린다(하드코딩 "stop" 제거).
    finish_reason = _map_finish_reason(last_stop_reason)

    # 클라이언트가 실행할 도구가 있으면 규격대로 알린다 — 이 신호를 보고 클라이언트가
    # 도구를 실행한 뒤 결과를 붙여 다시 호출한다(루프의 주인은 클라이언트다).
    response_tool_calls = (
        _tool_uses_to_openai_tool_calls(pending_tool_uses) if pending_tool_uses else None
    )
    if response_tool_calls:
        finish_reason = "tool_calls"

    # 구조화 출력(JSON) 요청이면 서버가 먼저 파싱해 본다. 깨진 JSON을 그대로
    # 200으로 흘려보내면 클라이언트는 원인을 알 수 없다(실측: 플러그인 파싱 실패).
    # 실패 원인 대부분은 "잘림"이므로 finish_reason과 함께 경고를 남긴다.
    finish_detail: FinishDetail | None = None
    if structured_output is not None and content:
        try:
            json.loads(content)
        except (ValueError, TypeError):
            logger.warning(
                "[structured_output] 응답이 유효한 JSON이 아닙니다 "
                "(finish_reason=%s, len=%d). 잘림이면 max_tokens를 늘리거나 "
                "요청을 분할해야 합니다. session=%s",
                finish_reason,
                len(content),
                session_id,
            )
            # 잘림이 아닌데도 깨졌다면 클라이언트가 구분할 수 있도록 신호를 준다.
            # finish_reason 값은 OpenAI 규격 안에 머물러야 표준 SDK 가 깨지지 않으므로
            # `content_filter` 를 유지하되, **왜 그런지는 finish_detail 로 분리해 싣는다**
            # (2026-08-13 — "정책에 차단당했다"는 오해가 실제로 발생했다).
            if finish_reason == "length":
                finish_detail = FinishDetail(
                    code="RESPONSE_TRUNCATED",
                    message=(
                        "출력 토큰 한도에 걸려 응답이 잘렸고 그래서 JSON 이 완성되지 "
                        "않았습니다. max_tokens 를 늘리거나 요청을 분할하세요."
                    ),
                )
            elif finish_reason == "stop":
                finish_reason = "content_filter"
                finish_detail = FinishDetail(
                    code="INVALID_STRUCTURED_OUTPUT",
                    message=(
                        "구조화 출력(JSON) 요청인데 응답이 유효한 JSON 이 아닙니다. "
                        "콘텐츠 정책에 의한 차단이 아닙니다 — 이 서버에는 프롬프트·"
                        "도구 결과·모델 출력을 검사하는 콘텐츠 정책 필터가 없습니다. "
                        "생성이 도중에 끊겼을 때(반복 붕괴 조기 절단 포함) 주로 "
                        "발생하므로 다시 시도하거나 요청 범위를 줄이세요."
                    ),
                )

    return OpenAIChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex}",
        created=int(time.time()),
        model=model_name,
        choices=[
            OpenAIChoice(
                message=OpenAIResponseMessage(
                    content=content, tool_calls=response_tool_calls
                ),
                finish_reason=finish_reason,
            )
        ],
        usage=usage,
        downloads=downloads,
        warnings=tool_warnings,
        finish_detail=finish_detail,
    )


async def _openai_stream_generate(
    session_id: str,
    tenant: Any,
    model_name: str,
    system_content: str | None,
    prior_messages: list[Any],
    last_user_text: str,
    structured_output: Any = None,
    max_tokens: int | None = None,
    client_tools: list[Any] | None = None,
    is_tool_continuation: bool = False,
    tool_warnings: list[str] | None = None,
    forced_query_class: str | None = None,
    request_id: str | None = None,
) -> AsyncGenerator[str, None]:
    """OpenAI `chat.completion.chunk` SSE 프레임을 생성한다.

    기존 /v1/chat/stream 의 Producer/Queue/Heartbeat 구조를 그대로 참고해 안정적으로
    스트리밍한다. Nexus text_delta → OpenAI delta.content 로만 매핑하고, tool_use/thinking
    등 OpenAI 표준에 없는 이벤트는 스트림 content 에 넣지 않는다. 다운로드 링크는 마지막
    content 청크로 덧붙이고, 종료 시 finish_reason='stop' 프레임 뒤 `data: [DONE]` 를 보낸다.
    """
    created = int(time.time())
    chatcmpl_id = f"chatcmpl-{uuid.uuid4().hex}"

    def _chunk(
        delta: dict[str, Any],
        finish: str | None = None,
        warnings: list[str] | None = None,
    ) -> str:
        """OpenAI chunk 한 프레임을 SSE `data: {...}` 문자열로 만든다.

        warnings 가 있으면 비표준 최상위 필드로 싣는다(비스트림 응답의 `warnings` 와
        같은 의미). 표준 클라이언트는 모르는 필드를 무시한다.
        """
        payload: dict[str, Any] = {
            "id": chatcmpl_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_name,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }
        if warnings:
            payload["warnings"] = warnings
        return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

    engine = _acquire_session_engine(session_id, tenant, client_tools=client_tools)
    if engine is None:
        # 엔진 미초기화 — 관례상 role 프레임 → 안내 content → 종료 순으로 최소 스트림.
        yield _chunk({"role": "assistant"})
        yield _chunk({"content": "QueryEngine이 아직 초기화되지 않았습니다."})
        yield _chunk({}, finish="stop")
        yield "data: [DONE]\n\n"
        return

    from core.message import StreamEvent, StreamEventType

    dl_start_idx = _inject_openai_context(
        engine, system_content, prior_messages, session_id, tenant
    )
    # 스트림으로 내보낸 본문을 모아 둔다 — 끝난 뒤 숫자 인용 검증에 쓴다.
    answer_parts: list[str] = []
    # 클라이언트가 실행할 도구 호출(서버는 실행하지 않고 종료 직전에 실어 보낸다).
    client_tool_names = {t.name for t in (client_tools or [])}
    pending_tool_uses: list[dict[str, Any]] = []

    # OpenAI 관례: 첫 프레임에 delta.role='assistant' 를 실어 스트림 시작을 알린다.
    # 제외된 도구가 있으면 이 첫 프레임에 함께 싣는다 — 스트림은 되돌릴 수 없으므로
    # 끝에 붙이면 클라이언트가 이미 결과를 처리한 뒤가 된다.
    yield _chunk({"role": "assistant"}, warnings=tool_warnings)

    # ── Producer/Queue/Heartbeat (기존 /v1/chat/stream 구조 미러링) ──
    # 이벤트 공백(Scout 호출 등)에 프록시가 끊지 않도록 주기적으로 SSE 주석(`: ping`)을
    # 보낸다. SSE 주석은 규격상 파서가 무시하므로 OpenAI 클라이언트 파싱에 영향 없음.
    sse_sentinel: tuple[str, Any] = ("done", None)
    sse_heartbeat_seconds = 20.0
    event_queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()

    async def _producer() -> None:
        """submit_message 스트림을 큐로 옮긴다(에러 포함)."""
        try:
            async for ev in engine.submit_message(
                last_user_text,
                structured_output=structured_output,
                max_tokens_override=(max_tokens if (max_tokens or 0) > 0 else None),
                # 도구 결과로 이어 도는 호출이면 새 user 발화를 만들지 않는다.
                append_user_message=not is_tool_continuation,
                forced_query_class=forced_query_class,
                request_id=request_id,
            ):
                await event_queue.put(("event", ev))
        except BaseException as e:  # noqa: BLE001 — 모든 예외를 에러 프레임/종료로 수렴
            await event_queue.put(("error", e))
        finally:
            await event_queue.put(sse_sentinel)

    producer_task = asyncio.create_task(_producer())
    stream_abort_error: BaseException | None = None
    # 마지막 종료 이유 — 토큰 한도로 잘리면 종료 프레임에 finish_reason="length"를 싣는다.
    stream_stop_reason: Any = None
    try:
        while True:
            try:
                kind, payload = await asyncio.wait_for(
                    event_queue.get(), timeout=sse_heartbeat_seconds
                )
            except TimeoutError:
                # 이벤트 공백 → keep-alive 주석 프레임(클라이언트 JSON 파서는 무시).
                yield ": heartbeat\n\n"
                continue

            if kind == "done":
                break
            if kind == "error":
                # 스트림 도중 오류 — 표준 content 로 노출하지 않고 종료한다(로그만 남김).
                stream_abort_error = payload
                break

            event = payload
            if isinstance(event, StreamEvent):
                # 텍스트 조각만 delta.content 로 흘린다. 나머지 이벤트는 무시.
                if event.type == StreamEventType.TEXT_DELTA and event.text:
                    # 스트림이 끝난 뒤 숫자 인용 검증에 쓰려고 본문을 함께 모아 둔다.
                    answer_parts.append(event.text)
                    yield _chunk({"content": event.text})
                elif (
                    event.type == StreamEventType.TOOL_USE_STOP
                    and event.tool_use
                    and event.tool_use.name in client_tool_names
                ):
                    # 클라이언트 실행 도구만 모은다(서버 도구는 서버가 이미 실행했다).
                    pending_tool_uses.append(
                        {
                            "id": event.tool_use.id,
                            "name": event.tool_use.name,
                            "input": event.tool_use.input or {},
                        }
                    )
                elif event.type == StreamEventType.MESSAGE_STOP and event.stop_reason:
                    # 턴마다 갱신 — 마지막 값이 이 응답의 최종 종료 이유다.
                    stream_stop_reason = event.stop_reason
    finally:
        # 클라이언트가 연결을 끊는 등으로 제너레이터가 닫히면 producer 를 취소한다.
        if not producer_task.done():
            producer_task.cancel()
            try:
                await producer_task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001, S110
                # 취소 시 예외는 위 error 경로에서 이미 다뤄졌거나 종료 경로다.
                pass
        if stream_abort_error is not None:
            logger.warning(
                "OpenAI 스트림 중단: session=%s, error=%s",
                session_id,
                type(stream_abort_error).__name__,
            )

    # 숫자 인용 검증 — 이미 흘려보낸 본문은 고치지 않고 경고만 마지막 청크로 덧붙인다.
    _num_warning = _answer_warnings_for(
        "".join(answer_parts), engine._messages[dl_start_idx:]
    )
    if _num_warning:
        yield _chunk({"content": _num_warning})

    # 이 턴에 생성된 문서 다운로드 링크를 마지막 content 청크로 덧붙인다.
    downloads = _collect_downloads(engine._messages[dl_start_idx:])
    # 생성물 메타데이터를 tb_artifacts에 fail-soft 기록(pg 없으면 조용히 스킵).
    await _record_artifacts(
        downloads, tenant, session_id, turn=getattr(engine, "total_turns", None)
    )
    dl_md = _downloads_markdown(downloads)
    if dl_md:
        yield _chunk({"content": dl_md})

    # 클라이언트가 실행할 도구가 있으면 종료 직전에 실어 보낸다.
    # OpenAI 규격은 tool_call 을 여러 delta 로 쪼개 보내는 것도, 한 delta 에 통째로
    # 싣는 것도 허용한다. 여기서는 통째로 싣는다 — 서버가 이미 완성된 인자를 갖고
    # 있어 굳이 쪼갤 이유가 없고, 쪼개면 클라이언트가 조각을 잇는 코드를 더 써야 한다.
    # (스트림 delta 에서는 각 호출에 index 가 필요하다 — 클라이언트가 조각을 잇는
    #  자리를 알아야 하기 때문이다. 비스트림 응답에는 index 가 없다.)
    if pending_tool_uses:
        _calls = _tool_uses_to_openai_tool_calls(pending_tool_uses)
        for i, call in enumerate(_calls):
            call["index"] = i
        yield _chunk({"tool_calls": _calls})

    # 종료 프레임 → OpenAI 관례상 빈 delta + finish_reason, 이어서 [DONE].
    # 토큰 한도로 잘렸으면 "length"를 실어 클라이언트가 미완결을 알 수 있게 한다.
    _finish = "tool_calls" if pending_tool_uses else _map_finish_reason(stream_stop_reason)
    yield _chunk({}, finish=_finish)
    yield "data: [DONE]\n\n"


# ─────────────────────────────────────────────
# 세션 엔드포인트
# ─────────────────────────────────────────────
@app.get("/v1/sessions")
async def list_sessions() -> dict[str, Any]:
    """
    저장된 세션 목록을 반환한다 (Ch 16).

    두 소스를 병합:
      1. Redis (단기, TTL 24h) — 최근 활성 세션 (session_id만)
      2. JSONL 트랜스크립트 (영구 기록) — 파일 시스템에 남아있는 모든 세션
    트랜스크립트가 상세 메타데이터(라인 수, 최종 수정 시각)를 갖고 있으므로
    이를 기본으로 삼고, Redis-only 세션은 이후에 머지한다.
    """
    from core.memory.transcript import list_transcript_sessions

    cfg = _app_state.get("config")
    sessions_dir = cfg.session.sessions_dir if cfg else ".nexus/sessions"

    # 1) 파일 트랜스크립트 기반 세션 (상세 정보 포함) — web 채널만(진입점 격리)
    disk_sessions = list_transcript_sessions(sessions_dir, limit=100, channel="web")

    # 2) Redis 단기 캐시 기반 세션 (session_id만) — 트랜스크립트에 없는 것만 추가
    memory_manager = _app_state.get("memory_manager")
    known_ids = {s["session_id"] for s in disk_sessions}
    redis_only: list[dict[str, Any]] = []
    if memory_manager is not None:
        try:
            for sid in await memory_manager.short_term.list_sessions(limit=100, channel="web"):
                if sid not in known_ids:
                    redis_only.append(
                        {
                            "session_id": sid,
                            "source": "redis_only",
                            "last_modified": None,
                            "entries": None,
                        }
                    )
        except Exception as e:
            # Redis 조회 실패는 치명적이지 않음 — disk 결과만으로 응답
            logger.debug("list_sessions Redis 조회 실패 (무시): %s", e)

    return {
        "sessions": disk_sessions + redis_only,
        "total": len(disk_sessions) + len(redis_only),
        "sessions_dir": sessions_dir,
    }


@app.get("/v1/sessions/search")
async def search_sessions(q: str = "") -> dict[str, Any]:
    """세션 대화를 질의어로 검색한다(사이드바 검색창).

    web 채널 트랜스크립트만 스캔(진입점 격리). 질의어가 2자 미만이면 빈 결과.
    반환: {"query", "results":[{session_id, last_modified, snippets:[...]}], "total"}.
    """
    from core.memory.transcript import search_transcript_sessions

    cfg = _app_state.get("config")
    sessions_dir = cfg.session.sessions_dir if cfg else ".nexus/sessions"
    results = search_transcript_sessions(sessions_dir, q, channel="web")
    return {"query": q, "results": results, "total": len(results)}


async def _load_session_messages(
    session_id: str,
) -> tuple[list[dict[str, Any]], str] | None:
    """세션 메시지를 Redis→트랜스크립트 순으로 로드한다. (messages, source) 또는 None.

    messages는 [{role, content, turn, ts}] (user/assistant만). 생성물 링크는 붙이지
    않은 원본이며, 호출부가 필요에 따라 _attach_session_artifacts로 보강한다. 조회/
    내보내기(messages 엔드포인트·export)가 이 헬퍼를 공유해 로직 중복을 없앤다.
    channel="web" 고정(진입점 격리). 어느 저장소에도 없으면 None.
    """
    memory_manager = _app_state.get("memory_manager")
    if memory_manager is not None:
        try:
            redis_msgs = await memory_manager.short_term.get_conversation_context(
                session_id, channel="web"
            )
            if redis_msgs:
                normalized = [
                    {
                        "role": m.get("role"),
                        "content": m.get("content"),
                        "turn": m.get("turn"),
                        "ts": m.get("ts"),
                    }
                    for m in redis_msgs
                    if m.get("role") in ("user", "assistant") and m.get("content")
                ]
                if normalized:
                    return normalized, "redis"
        except Exception as e:
            logger.debug("_load_session_messages Redis 조회 실패 (%s): %s", session_id, e)

    from core.memory.transcript import read_transcript_messages

    cfg = _app_state.get("config")
    sessions_dir = cfg.session.sessions_dir if cfg else ".nexus/sessions"
    disk_msgs = read_transcript_messages(sessions_dir, session_id, channel="web")
    if disk_msgs:
        out = [
            {
                "role": m["role"],
                "content": m["content"],
                "turn": m.get("turn"),
                "ts": m.get("ts"),
            }
            for m in disk_msgs
        ]
        return out, "transcript"
    return None


def _render_session_export(
    title: str, messages: list[dict[str, Any]], fmt: str
) -> str:
    """세션 대화를 내보내기용 텍스트로 렌더한다(md 또는 txt).

    md: 제목 헤더 + `## 사용자`/`## IDINO NOVA` 교대 섹션(본문은 마크다운 그대로).
    txt: `[사용자]`/`[IDINO NOVA]` 라벨 + 평문. 어느 쪽도 외부 의존 없이 순수 조립.
    """
    from datetime import UTC, datetime

    stamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    parts: list[str] = []
    if fmt == "md":
        parts.append(f"# {title}\n\n_{stamp}_\n")
        for m in messages:
            who = "사용자" if m.get("role") == "user" else "IDINO NOVA"
            parts.append(f"## {who}\n\n{m.get('content', '')}\n")
    else:  # txt
        parts.append(f"{title}\n{stamp}\n{'=' * 40}\n")
        for m in messages:
            who = "사용자" if m.get("role") == "user" else "IDINO NOVA"
            parts.append(f"[{who}]\n{m.get('content', '')}\n")
    return "\n".join(parts)


@app.get("/v1/sessions/{session_id}/messages")
async def get_session_messages(session_id: str) -> dict[str, Any]:
    """
    특정 세션의 대화 히스토리를 반환한다 (Ch 16 프론트 복원용).

    조회 우선순위:
      1) Redis 단기 캐시 (TTL 24h) — 가장 최신, JSON 직렬화된 user/assistant 페어
      2) JSONL 트랜스크립트 (영구 기록) — Redis 만료/없음 시 폴백

    어느 쪽에도 기록이 없으면 404 — UI가 "세션 없음" 분기로 전환할 수 있도록.

    응답:
      {
        "session_id": "...",
        "source": "redis" | "transcript",
        "messages": [{"role": "user"|"assistant", "content": "...",
                      "turn": N|None, "ts": ISO-8601|None}, ...],
        "total": N,
      }

    경로 파라미터 검증: session_id에 경로 분리자(슬래시/백슬래시/..)가 들어오면
    거부 — 트랜스크립트 파일 시스템 접근 시 디렉토리 탈출을 막는다.
    """
    from fastapi import HTTPException

    # 입력 검증 — 경로 탈출 차단 (파일 시스템 폴백 경로에서만 의미가 있지만
    # Redis 키 오염 방지 차원에서도 동일하게 적용)
    if not session_id or any(ch in session_id for ch in ("/", "\\", "..", "\x00")):
        raise HTTPException(status_code=400, detail="invalid session_id")

    # Redis→트랜스크립트 순으로 로드(공용 헬퍼). 어느 쪽에도 없으면 404.
    loaded = await _load_session_messages(session_id)
    if loaded is None:
        raise HTTPException(status_code=404, detail=f"session not found: {session_id}")
    messages, source = loaded
    # 생성물(이미지·문서) 링크를 각 assistant 메시지에 되붙인다.
    messages = await _attach_session_artifacts(messages, session_id)
    return {
        "session_id": session_id,
        "source": source,
        "messages": messages,
        "total": len(messages),
    }


@app.get("/v1/sessions/{session_id}/export")
async def export_session(session_id: str, fmt: str = "md"):
    """세션 대화를 md 또는 txt 파일로 내보낸다(첨부 다운로드).

    사이드바의 "내보내기" 버튼이 호출. 메시지는 get_session_messages와 같은
    로더(_load_session_messages, Redis→트랜스크립트)를 공유하고, 제목은 meta.json
    사용자 지정 제목을 우선한다. channel="web" 고정(진입점 격리).

    쿼리: fmt=md|txt (기본 md). 응답은 Content-Disposition 첨부 + UTF-8 파일명.
    """
    from urllib.parse import quote

    from fastapi import HTTPException
    from fastapi.responses import PlainTextResponse

    if not session_id or any(ch in session_id for ch in ("/", "\\", "..", "\x00")):
        raise HTTPException(status_code=400, detail="invalid session_id")
    fmt = (fmt or "md").lower()
    if fmt not in ("md", "txt"):
        raise HTTPException(status_code=400, detail="fmt must be 'md' or 'txt'")

    loaded = await _load_session_messages(session_id)
    if loaded is None:
        raise HTTPException(status_code=404, detail=f"session not found: {session_id}")
    messages, _source = loaded

    from core.memory.transcript import read_session_meta

    cfg = _app_state.get("config")
    sessions_dir = cfg.session.sessions_dir if cfg else ".nexus/sessions"
    meta = read_session_meta(sessions_dir, session_id, channel="web")
    title = meta.get("title") or f"대화 {session_id[:8]}"

    body = _render_session_export(title, messages, fmt)

    # 파일명 — 제목에서 경로/제어문자를 제거해 안전화한 뒤 확장자. 한글은 UTF-8 헤더로.
    safe = re.sub(r'[\\/:*?"<>|\x00-\x1f]', "_", title).strip()[:60] or "conversation"
    filename = f"{safe}.{fmt}"
    media = "text/markdown" if fmt == "md" else "text/plain"
    return PlainTextResponse(
        body,
        media_type=f"{media}; charset=utf-8",
        headers={"Content-Disposition": f"attachment; filename*=UTF-8''{quote(filename)}"},
    )


@app.post("/v1/sessions/{session_id}/truncate")
async def truncate_last_exchange(session_id: str) -> dict[str, Any]:
    """마지막 (user, assistant) 교환을 히스토리에서 제거한다(재생성용).

    프론트 "재생성" 버튼이 호출한다. 마지막 assistant 응답과 그 직전 user 메시지를
    chat_histories(인메모리 dict) + Redis 양쪽에서 제거하고, 제거된 user 메시지
    텍스트를 돌려준다. 프론트는 그 텍스트를 다시 전송해 새 응답을 받는다.

    [★왜 인메모리 dict까지 지우나 — Fable5 경고 지점]
      스트리밍 채팅은 요청마다 engine을 새로 조립하고 chat_histories[session_id]에서
      대화를 복원한다. Redis만 지우고 이 dict를 안 지우면 다음 요청이 인메모리 낡은
      이력을 그대로 써 "지운 메시지가 되살아나는" 버그가 난다. 그래서 둘 다 절단한다.

    channel="web" 고정. 세션 락으로 진행 중 생성과 직렬화한다(레이스 방지).
    """
    from fastapi import HTTPException

    if not session_id or any(ch in session_id for ch in ("/", "\\", "..", "\x00")):
        raise HTTPException(status_code=400, detail="invalid session_id")

    def _role(m: Any) -> str:
        return m.role if isinstance(m.role, str) else m.role.value

    def _text(m: Any) -> str:
        return m.text_content if hasattr(m, "text_content") else str(m.content)

    async with _get_session_lock(session_id):
        histories = _app_state.setdefault("chat_histories", {})
        hist = histories.get(session_id)
        memory_manager = _app_state.get("memory_manager")
        # 인메모리에 없으면 Redis에서 복원(서버 재기동/최초 접근 대비).
        if hist is None:
            saved = None
            if memory_manager is not None:
                try:
                    saved = await memory_manager.short_term.get_conversation_context(
                        session_id, channel="web"
                    )
                except Exception as e:
                    logger.debug("truncate Redis 조회 실패 (%s): %s", session_id, e)
            hist = _restore_messages_from_saved(saved, session_id)
            histories[session_id] = hist

        # 히스토리가 아예 없으면 없는 세션으로 보고 404(fork/export와 API 일관성).
        if not hist:
            raise HTTPException(status_code=404, detail=f"session not found: {session_id}")

        # 마지막 assistant와 그 직전 user를 찾아 그 지점부터 잘라낸다.
        last_asst = next(
            (i for i in range(len(hist) - 1, -1, -1) if _role(hist[i]) == "assistant"),
            None,
        )
        if last_asst is None:
            return {"session_id": session_id, "removed": 0, "last_user": None}
        last_user = next(
            (i for i in range(last_asst - 1, -1, -1) if _role(hist[i]) == "user"),
            None,
        )
        removed_user_text = _text(hist[last_user]) if last_user is not None else None
        cut = last_user if last_user is not None else last_asst
        removed = len(hist) - cut
        del hist[cut:]

        # Redis write-through(절단 반영) — 저장 형식은 스트리밍 경로와 동일.
        if memory_manager is not None:
            try:
                serialized = [
                    {"role": _role(m), "content": _text(m)}
                    for m in hist
                    if _role(m) in ("user", "assistant") and _text(m)
                ]
                await memory_manager.short_term.save_conversation_context(
                    session_id, serialized, ttl=86400, channel="web"
                )
            except Exception as e:
                logger.warning("truncate Redis 저장 실패 (%s): %s", session_id, e)

    return {"session_id": session_id, "removed": removed, "last_user": removed_user_text}


class ForkRequest(BaseModel):
    """대화 분기(fork) 요청 — at_index개의 앞 메시지를 새 세션에 복사한다."""

    at_index: int = 0


@app.post("/v1/sessions/{session_id}/fork")
async def fork_session(session_id: str, body: ForkRequest) -> dict[str, Any]:
    """대화를 특정 지점까지 복사한 새 세션(분기)을 만든다(메시지 편집·분기용).

    at_index개의 앞 메시지를 새 session_id로 복사한다(chat_histories + Redis +
    트랜스크립트). 원본은 그대로 두고 새 세션을 반환하므로 append-only 트랜스크립트와
    충돌하지 않고 원본이 보존된다(Fable5 설계). 프론트는 새 세션으로 전환한 뒤 편집한
    메시지를 전송해 분기를 이어간다. channel="web" 고정.
    """
    from fastapi import HTTPException

    from core.message import Message

    if not session_id or any(ch in session_id for ch in ("/", "\\", "..", "\x00")):
        raise HTTPException(status_code=400, detail="invalid session_id")

    loaded = await _load_session_messages(session_id)
    if loaded is None:
        raise HTTPException(status_code=404, detail=f"session not found: {session_id}")
    messages, _source = loaded
    at = max(0, min(body.at_index, len(messages)))
    prefix = messages[:at]

    new_id = str(uuid.uuid4())
    msg_objs: list[Any] = []
    for m in prefix:
        if m.get("role") == "user":
            msg_objs.append(Message.user(m.get("content", "")))
        elif m.get("role") == "assistant":
            msg_objs.append(Message.assistant(m.get("content", "")))

    async with _get_session_lock(new_id):
        histories = _app_state.setdefault("chat_histories", {})
        histories[new_id] = list(msg_objs)
        memory_manager = _app_state.get("memory_manager")
        if memory_manager is not None:
            try:
                serialized = [
                    {"role": m["role"], "content": m["content"]}
                    for m in prefix
                    if m.get("role") in ("user", "assistant") and m.get("content")
                ]
                await memory_manager.short_term.save_conversation_context(
                    new_id, serialized, ttl=86400, channel="web"
                )
            except Exception as e:
                logger.warning("fork Redis 저장 실패 (%s): %s", new_id, e)
        # 트랜스크립트에도 접두부를 남긴다(Redis TTL 이후에도 복원되도록).
        try:
            trans = _build_transcript(new_id, channel="web")
            if trans is not None:
                for m in prefix:
                    if m.get("role") in ("user", "assistant") and m.get("content"):
                        trans.append_entry(role=m["role"], content=m["content"], turn=0)
        except Exception as e:
            logger.warning("fork 트랜스크립트 기록 실패 (%s): %s", new_id, e)

    # 원본의 폴더·제목을 새 세션 메타에 상속(제목엔 '(분기)' 표시).
    from core.memory.transcript import read_session_meta, write_session_meta

    cfg = _app_state.get("config")
    sessions_dir = cfg.session.sessions_dir if cfg else ".nexus/sessions"
    src_meta = read_session_meta(sessions_dir, session_id, channel="web")
    new_meta: dict[str, Any] = {}
    if src_meta.get("folder"):
        new_meta["folder"] = src_meta["folder"]
    if src_meta.get("title"):
        new_meta["title"] = (str(src_meta["title"]) + " (분기)")[:200]

    # 원본의 생성물(이미지·문서)을 분기에 물려준다(T3-1c — tb_artifacts는 원본 session_id
    # 기준이라 그냥은 소실됨). 원본이 직접 만든 것 + 원본이 또 상속받은 것(연쇄 fork)을 합치되,
    # fork 지점(접두부 최대 turn) 이후에 생긴 생성물은 제외한다.
    from core.storage.artifacts import list_session_artifacts

    pool = _app_state.get("pg_pool")
    src_arts: list[dict[str, Any]] = []
    if pool is not None:
        src_arts = list(await list_session_artifacts(pool, session_id))
    src_arts += [a for a in (src_meta.get("inherited_artifacts") or []) if isinstance(a, dict)]
    prefix_turns = [m.get("turn") for m in prefix if isinstance(m.get("turn"), int)]
    max_turn = max(prefix_turns) if prefix_turns else None
    inherited: list[dict[str, Any]] = []
    seen_fn: set[str] = set()
    for a in src_arts:
        fn = a.get("filename")
        if not fn or fn in seen_fn:
            continue
        t = a.get("turn")
        if max_turn is not None and isinstance(t, int) and t > max_turn:
            continue  # fork 지점 이후 생성물은 분기에 넣지 않는다
        seen_fn.add(fn)
        inherited.append({"filename": fn, "mime": a.get("mime"), "turn": t})
    if inherited:
        new_meta["inherited_artifacts"] = inherited

    if new_meta:
        try:
            write_session_meta(sessions_dir, new_id, new_meta, channel="web")
        except ValueError:
            pass

    return {
        "new_session_id": new_id,
        "copied": len(prefix),
        "messages": prefix,
        "folder": new_meta.get("folder", ""),
        "title": new_meta.get("title"),
    }


# ─────────────────────────────────────────────
# 커스텀 인스트럭션 (테넌트 단위 시스템 프롬프트 추가) — P2-2
# ─────────────────────────────────────────────
# 최대 길이 — 시스템 프롬프트에 덧붙이는 사용자 지시문의 상한(과도한 프롬프트 방지).
_INSTRUCTION_MAX_CHARS = 4000


def _instructions_dir() -> Path:
    """커스텀 인스트럭션 파일이 저장되는 디렉토리(세션 디렉토리 하위 _instructions).

    세션 디렉토리는 배포에서 bind-mount로 영속화되므로 여기에 두면 재기동/재생성
    후에도 유지된다. 채널 세션 폴더(web/cli/api)와 이름이 겹치지 않고, transcript.jsonl이
    없어 세션 목록 스캔에도 잡히지 않는다.
    """
    cfg = _app_state.get("config")
    sessions_dir = cfg.session.sessions_dir if cfg else ".nexus/sessions"
    return Path(sessions_dir) / "_instructions"


def _read_custom_instruction(tenant_id: str | None) -> str:
    """테넌트의 커스텀 인스트럭션 텍스트를 읽는다(없거나 오류 시 빈 문자열)."""
    tid = re.sub(r"[^\w.-]", "_", tenant_id or "default")
    try:
        f = _instructions_dir() / f"{tid}.txt"
        if f.is_file():
            return f.read_text(encoding="utf-8")
    except OSError as e:
        logger.debug("커스텀 인스트럭션 읽기 실패 (%s): %s", tid, e)
    return ""


def _write_custom_instruction(tenant_id: str | None, text: str) -> bool:
    """테넌트의 커스텀 인스트럭션을 저장한다(성공 True, fail-soft)."""
    tid = re.sub(r"[^\w.-]", "_", tenant_id or "default")
    try:
        d = _instructions_dir()
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{tid}.txt").write_text(text, encoding="utf-8")
        return True
    except OSError as e:
        logger.warning("커스텀 인스트럭션 저장 실패 (%s): %s", tid, e)
        return False


class InstructionUpdate(BaseModel):
    """커스텀 인스트럭션 갱신 바디 — 시스템 프롬프트에 덧붙일 사용자 지시문."""

    text: str = ""


# ─────────────────────────────────────────────
# 응답 스타일 (W1, 2026-08-05)
# ─────────────────────────────────────────────
# 인스트럭션과 같은 저장 규약을 쓴다(_instructions/{tid}.style.json). 별도 저장소를
# 만들지 않는 이유: 테넌트 단위 개인화라는 성격이 같고, 백업·정리 대상도 같기 때문.
_STYLE_CUSTOM_MAX_CHARS = 2000


def _read_response_style(tenant_id: str | None) -> dict[str, str]:
    """테넌트의 응답 스타일 설정을 읽는다(없으면 기본값, fail-soft)."""
    from core.system_prompt.styles import default_style_id

    tid = re.sub(r"[^\w.-]", "_", tenant_id or "default")
    try:
        f = _instructions_dir() / f"{tid}.style.json"
        if f.is_file():
            data = json.loads(f.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return {
                    "style": str(data.get("style") or default_style_id()),
                    "custom": str(data.get("custom") or ""),
                }
    except (OSError, ValueError) as e:
        logger.debug("응답 스타일 읽기 실패 (%s): %s", tid, e)
    return {"style": default_style_id(), "custom": ""}


def _write_response_style(tenant_id: str | None, style: str, custom: str) -> bool:
    """테넌트의 응답 스타일을 저장한다(성공 True, fail-soft)."""
    tid = re.sub(r"[^\w.-]", "_", tenant_id or "default")
    try:
        d = _instructions_dir()
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{tid}.style.json").write_text(
            json.dumps({"style": style, "custom": custom}, ensure_ascii=False),
            encoding="utf-8",
        )
        return True
    except OSError as e:
        logger.warning("응답 스타일 저장 실패 (%s): %s", tid, e)
        return False


def _resolve_style_prompt_for(tenant_id: str | None) -> str:
    """테넌트 설정을 실제 [응답 스타일] 문구로 바꾼다(없으면 빈 문자열=무회귀)."""
    try:
        from core.system_prompt.styles import resolve_style_prompt

        cfg = _read_response_style(tenant_id)
        return resolve_style_prompt(cfg.get("style"), cfg.get("custom"))
    except Exception as e:  # noqa: BLE001 — 스타일은 부가 기능, 대화를 막지 않는다
        logger.debug("응답 스타일 해석 실패(무시): %s", e)
        return ""


class ResponseStyleUpdate(BaseModel):
    """응답 스타일 갱신 바디. custom이 있으면 프리셋 대신 그것을 쓴다."""

    style: str = ""
    custom: str = ""


@app.get("/v1/memories")
async def list_memories(
    limit: int = 50,
    x_tenant_id: str | None = Header(default=None, alias="X-Tenant-ID"),
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    """이 테넌트가 저장한 대화 기억 목록을 최신순으로 돌려준다(메모리 UI용).

    ★소유자는 요청에서 받지 않고 서버가 인증 정보로 정한다. 클라이언트가 owner를
      넘길 수 있게 하면 남의 기억을 조회하는 통로가 된다(IDOR).
    코드 RAG 청크는 소유자가 없어 자연히 제외된다 — 사용자가 볼 것은 자기 대화뿐이다.
    """
    manager = _app_state.get("memory_manager")
    if manager is None:
        return {"memories": [], "total": 0}

    tenant = _resolve_tenant(None, x_tenant_id, authorization)
    owner = str(getattr(tenant, "id", "") or "")
    if not owner:
        return {"memories": [], "total": 0}

    # 한 번에 너무 많이 긁어오지 않도록 상한을 둔다.
    limit = max(1, min(int(limit or 50), 200))
    try:
        entries = await manager.long_term.list_by_owner(owner, limit=limit)
    except Exception as e:  # noqa: BLE001 — 조회 실패가 화면을 깨뜨리지 않게
        logger.warning("메모리 목록 조회 실패: %s", e)
        return {"memories": [], "total": 0}

    return {
        "memories": [
            {
                "id": e.id,
                "content": e.content,
                "role": (e.metadata or {}).get("role", ""),
                "importance": e.importance,
                "created_at": e.created_at.isoformat() if e.created_at else "",
            }
            for e in entries
        ],
        "total": len(entries),
    }


@app.delete("/v1/memories/{memory_id}")
async def delete_memory(
    memory_id: str,
    x_tenant_id: str | None = Header(default=None, alias="X-Tenant-ID"),
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    """기억 한 건을 지운다 — 요청 테넌트가 소유한 것만.

    소유자 조건은 DELETE 문 자체에 들어가므로(delete_owned), 남의 id를 넣어도
    지워지지 않는다. 존재하지 않는 id와 남의 id를 같은 404로 응답해 '있는지 없는지'
    조차 알려주지 않는다(존재 은닉 — 다운로드 라우트와 같은 방침).
    """
    from fastapi import HTTPException

    manager = _app_state.get("memory_manager")
    tenant = _resolve_tenant(None, x_tenant_id, authorization)
    owner = str(getattr(tenant, "id", "") or "")
    if manager is None or not owner:
        raise HTTPException(status_code=404, detail="기억을 찾을 수 없습니다.")

    try:
        deleted = await manager.long_term.delete_owned(memory_id, owner)
    except Exception as e:  # noqa: BLE001 — 실패를 500으로 흘리지 않고 404로 수렴
        logger.warning("메모리 삭제 실패: %s", e)
        deleted = False

    if not deleted:
        raise HTTPException(status_code=404, detail="기억을 찾을 수 없습니다.")
    return {"status": "ok", "id": memory_id}


@app.get("/v1/response-style")
async def get_response_style(
    x_tenant_id: str | None = Header(default=None, alias="X-Tenant-ID"),
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    """현재 테넌트의 응답 스타일과 선택 가능한 프리셋 목록을 반환한다."""
    from core.system_prompt.styles import list_styles

    tenant = _resolve_tenant(None, x_tenant_id, authorization)
    tid = getattr(tenant, "id", "default")
    cfg = _read_response_style(tid)
    return {"tenant_id": tid, **cfg, "available": list_styles()}


@app.put("/v1/response-style")
async def put_response_style(
    body: ResponseStyleUpdate,
    x_tenant_id: str | None = Header(default=None, alias="X-Tenant-ID"),
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    """응답 스타일을 저장한다. 이후 채팅의 시스템 프롬프트에 [응답 스타일]로 붙는다."""
    from core.system_prompt.styles import default_style_id, list_styles

    tenant = _resolve_tenant(None, x_tenant_id, authorization)
    tid = getattr(tenant, "id", "default")
    valid = {s["id"] for s in list_styles()}
    # 모르는 프리셋 id는 기본값으로 떨어뜨린다(임의 문자열이 저장되지 않게).
    style = body.style if body.style in valid else default_style_id()
    custom = (body.custom or "")[:_STYLE_CUSTOM_MAX_CHARS]
    ok = _write_response_style(tid, style, custom)
    return {"tenant_id": tid, "ok": ok, "style": style, "custom": custom}


@app.get("/v1/instructions")
async def get_instructions(
    x_tenant_id: str | None = Header(default=None, alias="X-Tenant-ID"),
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    """현재 테넌트의 커스텀 인스트럭션을 반환한다(설정 화면 로드용)."""
    tenant = _resolve_tenant(None, x_tenant_id, authorization)
    tid = getattr(tenant, "id", "default")
    return {"tenant_id": tid, "text": _read_custom_instruction(tid)}


@app.put("/v1/instructions")
async def put_instructions(
    body: InstructionUpdate,
    x_tenant_id: str | None = Header(default=None, alias="X-Tenant-ID"),
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    """현재 테넌트의 커스텀 인스트럭션을 저장한다. 저장분은 이후 채팅 시스템
    프롬프트에 '[사용자 지시]'로 덧붙는다. 길이는 상한으로 자른다."""
    tenant = _resolve_tenant(None, x_tenant_id, authorization)
    tid = getattr(tenant, "id", "default")
    text = (body.text or "")[:_INSTRUCTION_MAX_CHARS]
    ok = _write_custom_instruction(tid, text)
    return {"tenant_id": tid, "ok": ok, "text": text}


# ─────────────────────────────────────────────
# 프롬프트 템플릿 (테넌트 단위 JSON) — P2-6
# ─────────────────────────────────────────────
_PROMPT_MAX_COUNT = 50           # 테넌트당 저장 가능한 템플릿 최대 개수
_PROMPT_TITLE_MAX = 60           # 제목 최대 길이
_PROMPT_TEXT_MAX = 2000          # 본문 최대 길이


def _prompts_dir() -> Path:
    """프롬프트 템플릿 파일 디렉토리(세션 디렉토리 하위 _prompts, bind-mount 영속)."""
    cfg = _app_state.get("config")
    sessions_dir = cfg.session.sessions_dir if cfg else ".nexus/sessions"
    return Path(sessions_dir) / "_prompts"


def _read_prompts(tenant_id: str | None) -> list[dict[str, str]]:
    """테넌트의 프롬프트 템플릿 목록을 읽는다([{title, text}], 없거나 오류 시 빈 목록)."""
    tid = re.sub(r"[^\w.-]", "_", tenant_id or "default")
    try:
        f = _prompts_dir() / f"{tid}.json"
        if f.is_file():
            data = json.loads(f.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return [
                    {"title": str(p.get("title", "")), "text": str(p.get("text", ""))}
                    for p in data
                    if isinstance(p, dict)
                ]
    except (OSError, json.JSONDecodeError) as e:
        logger.debug("프롬프트 템플릿 읽기 실패 (%s): %s", tid, e)
    return []


def _write_prompts(tenant_id: str | None, prompts: list[dict[str, str]]) -> bool:
    """테넌트의 프롬프트 템플릿 목록을 저장한다(성공 True, fail-soft)."""
    tid = re.sub(r"[^\w.-]", "_", tenant_id or "default")
    try:
        d = _prompts_dir()
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{tid}.json").write_text(
            json.dumps(prompts, ensure_ascii=False), encoding="utf-8"
        )
        return True
    except OSError as e:
        logger.warning("프롬프트 템플릿 저장 실패 (%s): %s", tid, e)
        return False


class PromptItem(BaseModel):
    """프롬프트 템플릿 1건 — 제목 + 본문."""

    title: str = ""
    text: str = ""


class PromptsUpdate(BaseModel):
    """프롬프트 템플릿 전체 목록 갱신 바디(목록 통째로 교체)."""

    prompts: list[PromptItem] = []


@app.get("/v1/prompts")
async def get_prompts(
    x_tenant_id: str | None = Header(default=None, alias="X-Tenant-ID"),
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    """현재 테넌트의 프롬프트 템플릿 목록을 반환한다."""
    tenant = _resolve_tenant(None, x_tenant_id, authorization)
    tid = getattr(tenant, "id", "default")
    return {"tenant_id": tid, "prompts": _read_prompts(tid)}


@app.put("/v1/prompts")
async def put_prompts(
    body: PromptsUpdate,
    x_tenant_id: str | None = Header(default=None, alias="X-Tenant-ID"),
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    """현재 테넌트의 프롬프트 템플릿 목록을 통째로 교체 저장한다.
    본문이 빈 항목은 버리고, 개수·길이는 상한으로 자른다."""
    tenant = _resolve_tenant(None, x_tenant_id, authorization)
    tid = getattr(tenant, "id", "default")
    items = [
        {"title": p.title.strip()[:_PROMPT_TITLE_MAX], "text": p.text[:_PROMPT_TEXT_MAX]}
        for p in body.prompts[:_PROMPT_MAX_COUNT]
        if p.text.strip()
    ]
    ok = _write_prompts(tid, items)
    return {"tenant_id": tid, "ok": ok, "prompts": items}


# ─────────────────────────────────────────────
# 프로젝트 (폴더 + 인스트럭션 + 지식소스 서브셋) — P3-3
# ─────────────────────────────────────────────
# 프로젝트는 "폴더명 = 프로젝트명"으로 세션에 연결된다(P2-3 폴더 재사용). 세션의 folder가
# 정의된 프로젝트명과 같으면 그 프로젝트의 인스트럭션·지식소스가 채팅에 적용된다.
_PROJECT_MAX_COUNT = 50


def _projects_dir() -> Path:
    cfg = _app_state.get("config")
    sessions_dir = cfg.session.sessions_dir if cfg else ".nexus/sessions"
    return Path(sessions_dir) / "_projects"


def _read_projects(tenant_id: str | None) -> list[dict[str, Any]]:
    """테넌트의 프로젝트 목록을 읽는다([{name, instruction, sources}], 없으면 빈 목록)."""
    tid = re.sub(r"[^\w.-]", "_", tenant_id or "default")
    try:
        f = _projects_dir() / f"{tid}.json"
        if f.is_file():
            data = json.loads(f.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return [
                    {
                        "name": str(p.get("name", "")),
                        "instruction": str(p.get("instruction", "")),
                        "sources": [str(x) for x in p.get("sources", []) if x],
                    }
                    for p in data
                    if isinstance(p, dict) and p.get("name")
                ]
    except (OSError, json.JSONDecodeError) as e:
        logger.debug("프로젝트 읽기 실패 (%s): %s", tid, e)
    return []


def _write_projects(tenant_id: str | None, projects: list[dict[str, Any]]) -> bool:
    tid = re.sub(r"[^\w.-]", "_", tenant_id or "default")
    try:
        d = _projects_dir()
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{tid}.json").write_text(
            json.dumps(projects, ensure_ascii=False), encoding="utf-8"
        )
        return True
    except OSError as e:
        logger.warning("프로젝트 저장 실패 (%s): %s", tid, e)
        return False


def _resolve_project_for_session(
    tenant_id: str | None, session_id: str
) -> dict[str, Any] | None:
    """세션의 folder(meta)와 이름이 같은 프로젝트를 찾는다(없으면 None)."""
    cfg = _app_state.get("config")
    sessions_dir = cfg.session.sessions_dir if cfg else ".nexus/sessions"
    from core.memory.transcript import read_session_meta

    folder = (
        read_session_meta(sessions_dir, session_id, channel="web").get("folder") or ""
    ).strip()
    if not folder:
        return None
    for p in _read_projects(tenant_id):
        if p.get("name") == folder:
            return p
    return None


def _narrow_tenant_for_project(tenant: Any, project: dict[str, Any] | None) -> Any:
    """프로젝트가 지식소스를 지정했으면 tenant를 narrow한 사본으로 바꿔 돌려준다.
    프로젝트는 테넌트 허용범위를 넓힐 수 없다(보안 — 교집합만). 아니면 원본 반환."""
    if not project:
        return tenant
    srcs = project.get("sources") or []
    if srcs and tenant is not None:
        base = list(getattr(tenant, "allowed_knowledge_sources", []) or [])
        narrowed = [s for s in srcs if (not base or s in base)]
        try:
            return tenant.model_copy(update={"allowed_knowledge_sources": narrowed})
        except Exception as e:  # noqa: BLE001 — 실패 시 원본 유지(무회귀)
            logger.warning("프로젝트 소스 제한 실패(무시): %s", e)
    return tenant


class ProjectItem(BaseModel):
    """프로젝트 1건 — 이름 + 인스트럭션 + 지식소스 서브셋."""

    name: str = ""
    instruction: str = ""
    sources: list[str] = Field(default_factory=list)


class ProjectsUpdate(BaseModel):
    """프로젝트 전체 목록 갱신 바디."""

    projects: list[ProjectItem] = []


@app.get("/v1/projects")
async def get_projects(
    x_tenant_id: str | None = Header(default=None, alias="X-Tenant-ID"),
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    """현재 테넌트의 프로젝트 목록 + 선택 가능한 지식소스 풀을 반환한다."""
    tenant = _resolve_tenant(None, x_tenant_id, authorization)
    tid = getattr(tenant, "id", "default")
    return {
        "tenant_id": tid,
        "projects": _read_projects(tid),
        "available_sources": list(getattr(tenant, "allowed_knowledge_sources", []) or []),
    }


@app.put("/v1/projects")
async def put_projects(
    body: ProjectsUpdate,
    x_tenant_id: str | None = Header(default=None, alias="X-Tenant-ID"),
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    """현재 테넌트의 프로젝트 목록을 통째로 교체 저장한다. 이름 없는 항목은 버린다."""
    tenant = _resolve_tenant(None, x_tenant_id, authorization)
    tid = getattr(tenant, "id", "default")
    items = [
        {
            "name": p.name.strip()[:60],
            "instruction": p.instruction[:_INSTRUCTION_MAX_CHARS],
            "sources": [s.strip() for s in p.sources if s.strip()][:20],
        }
        for p in body.projects[:_PROJECT_MAX_COUNT]
        if p.name.strip()
    ]
    ok = _write_projects(tid, items)
    return {"tenant_id": tid, "ok": ok, "projects": items}


class SessionMetaUpdate(BaseModel):
    """세션 메타 갱신 요청 바디 — 제목(수동)·핀 여부·폴더. 미지정(None)은 변경 안 함.

    folder는 빈 문자열("")로 보내면 '폴더 없음(미분류)'으로 되돌린다.
    """

    title: str | None = None
    pinned: bool | None = None
    folder: str | None = None


@app.patch("/v1/sessions/{session_id}")
async def update_session_meta(session_id: str, body: SessionMetaUpdate) -> dict[str, Any]:
    """세션의 메타데이터(사용자 지정 제목·핀 여부)를 갱신한다(meta.json 사이드카).

    사이드바의 "이름 변경"·"핀 고정"이 호출하는 엔드포인트. 트랜스크립트 본문은
    건드리지 않고 meta.json만 병합 갱신한다(부분 갱신 — 준 필드만 반영).

    channel="web" 고정 — cli/api 세션 메타에 닿지 않게 격리(fail-closed).
    title은 strip 후 200자로 제한(XSS는 프론트 렌더에서 escapeHtml로 방어).
    """
    from fastapi import HTTPException

    if not session_id or any(ch in session_id for ch in ("/", "\\", "..", "\x00")):
        raise HTTPException(status_code=400, detail="invalid session_id")

    # 갱신할 필드만 추린다(둘 다 None이면 요청 무의미 → 400).
    meta_update: dict[str, Any] = {}
    if body.title is not None:
        meta_update["title"] = body.title.strip()[:200]
    if body.pinned is not None:
        meta_update["pinned"] = bool(body.pinned)
    if body.folder is not None:
        # 폴더명(빈 문자열이면 '미분류'로 되돌림). 60자 제한.
        meta_update["folder"] = body.folder.strip()[:60]
    if not meta_update:
        raise HTTPException(status_code=400, detail="no fields to update")

    from core.memory.transcript import read_session_meta, write_session_meta

    cfg = _app_state.get("config")
    sessions_dir = cfg.session.sessions_dir if cfg else ".nexus/sessions"
    try:
        ok = write_session_meta(sessions_dir, session_id, meta_update, channel="web")
    except ValueError:
        raise HTTPException(status_code=400, detail="invalid session_id") from None
    meta = read_session_meta(sessions_dir, session_id, channel="web")
    return {
        "session_id": session_id,
        "ok": ok,
        "title": meta.get("title"),
        "pinned": bool(meta.get("pinned")),
        "folder": meta.get("folder") or "",
    }


@app.delete("/v1/sessions/{session_id}")
async def delete_session(session_id: str) -> dict[str, Any]:
    """
    특정 세션을 Redis(단기) + 트랜스크립트(영구) 양쪽에서 삭제한다 (Ch 16).

    프론트 사이드바의 세션 삭제 버튼이 호출하는 엔드포인트. 각 저장소는
    독립적이므로 한쪽만 성공해도 응답한다 (best-effort).

    응답:
      {
        "session_id": "...",
        "deleted_redis": bool,     # Redis 키가 실제로 있었고 삭제됐는지
        "deleted_disk":  bool,     # 트랜스크립트 디렉토리가 있었고 삭제됐는지
      }

    둘 다 False여도 200 — 이미 없었을 뿐 에러는 아님. 다만 session_id 자체가
    부적합(슬래시/백슬래시/'..')하면 400.
    """
    from fastapi import HTTPException

    if not session_id or any(ch in session_id for ch in ("/", "\\", "..", "\x00")):
        raise HTTPException(status_code=400, detail="invalid session_id")

    deleted_redis = False
    deleted_disk = False

    # 1) Redis — clear_session은 존재 여부와 무관하게 DEL을 호출하므로,
    #    실제 삭제 여부는 사전 존재 조회로 판정한다.
    memory_manager = _app_state.get("memory_manager")
    if memory_manager is not None:
        try:
            existing = await memory_manager.short_term.get_conversation_context(
                session_id, channel="web"
            )
            if existing:
                await memory_manager.short_term.clear_session(session_id, channel="web")
                deleted_redis = True
        except Exception as e:
            # Redis 장애는 치명적 아님 — 디스크 삭제는 독립적으로 시도
            logger.warning("Redis 세션 삭제 실패 (%s): %s", session_id, e)

    # 1-b) 계획 체크리스트(TodoStore) — 세션 삭제 시 함께 정리(메모리 누수 방지).
    todo_store = _app_state.get("todo_store")
    if todo_store is not None:
        try:
            todo_store.clear_session(session_id)
        except Exception as e:  # noqa: BLE001 — 보조 정리라 삭제 응답을 막지 않는다
            logger.warning("체크리스트 세션 삭제 실패 (%s): %s", session_id, e)

    # 2) 디스크 — 트랜스크립트 디렉토리 통째로 제거
    from core.memory.transcript import delete_transcript_session

    cfg = _app_state.get("config")
    sessions_dir = cfg.session.sessions_dir if cfg else ".nexus/sessions"
    try:
        deleted_disk = delete_transcript_session(sessions_dir, session_id, channel="web")
    except ValueError:
        # delete_transcript_session의 경로 검증 실패 — 이미 위에서 400 처리했지만
        # 방어적으로 한 번 더
        raise HTTPException(status_code=400, detail="invalid session_id") from None
    except OSError as e:
        logger.warning("트랜스크립트 삭제 실패 (%s): %s", session_id, e)

    logger.info(
        "세션 삭제: session=%s, redis=%s, disk=%s",
        session_id,
        deleted_redis,
        deleted_disk,
    )
    return {
        "session_id": session_id,
        "deleted_redis": deleted_redis,
        "deleted_disk": deleted_disk,
    }


# ─────────────────────────────────────────────
# 도구 엔드포인트
# ─────────────────────────────────────────────
@app.get("/v1/tools")
async def list_tools() -> dict[str, Any]:
    """
    등록된 도구 목록을 반환한다.

    ToolRegistry에서 등록된 모든 도구의 이름, 설명, 그룹을 반환한다.
    """
    # 웹 Worker가 실제로 보는 도구 풀을 우선 노출한다(MCP 도구 포함).
    # _app_state["web_tools"]는 _build_web_query_engine이 MCP 머지 후 저장한 실풀.
    # 부재 시(부트스트랩 실패 등) base tool_registry로 폴백.
    tools = _app_state.get("web_tools")
    if not tools:
        registry = _app_state.get("tool_registry")
        if registry is None:
            return {"tools": [], "total": 0}
        tools = registry.get_all_tools()

    tool_list = [
        ToolInfo(
            name=t.name,
            description=t.description,
            group=t.group,
            is_read_only=t.is_read_only,
        ).model_dump()
        for t in tools
    ]
    return {"tools": tool_list, "total": len(tool_list)}


# ─────────────────────────────────────────────
# 모델 엔드포인트
# ─────────────────────────────────────────────
@app.get("/v1/models")
async def list_models() -> dict[str, Any]:
    """사용 가능한 모델 목록을 반환한다 (GET /v1/models).

    id 는 config 의 모델 식별자(= vLLM served-model-name)이고, name 은 **실제로
    서빙 중인 모델**에게 물어본 값이다(vLLM `/v1/models` 의 `root`, 예 "skt/A.X-4.0").

    ★왜 물어보나 (2026-08-16):
      종전에는 name 을 코드에 박아 뒀다. 그래서 모델을 바꾼 뒤에도 옛 모델의 이름이
      그대로 남아, 실제 72B 를 서빙하면서 응답에는 27B 시절 이름이 나가고 있었다.
      연동하는 쪽이 그 값을 믿으면 파라미터 규모를 잘못 가정한다. 서빙 주체에게
      물어보면 모델을 바꿔도 저절로 맞는다 — 프롬프트를 도구 레지스트리에서 유도한
      것과 같은 원칙이다.

      조회에 실패하거나(추론 서버 다운) 그 서버가 안 가진 모델(임베딩은 별도 서버)이면
      name 을 id 로 둔다. **모르면 지어내지 않고 id 를 그대로 보여 준다.**
    """
    config = _app_state.get("config")
    if not config:
        # 부트스트랩 전/실패 — 조회가 500으로 죽지 않게 빈 목록을 돌려준다.
        # 종전에는 하드코딩된 기본 2종을 돌려줬는데, 그것이 바로 옛 이름이 남는 경로였다.
        return {"models": [], "total": 0}

    # 상위(vLLM)가 실제로 서빙 중인 목록 — id → root 로 만들어 둔다(fail-soft).
    upstream: dict[str, str] = {}
    provider = _app_state.get("model_provider")
    lister = getattr(provider, "list_upstream_models", None)
    if lister is not None:
        try:
            for item in await lister() or []:
                if isinstance(item, dict) and item.get("id"):
                    upstream[str(item["id"])] = str(item.get("root") or item["id"])
        except Exception as e:  # noqa: BLE001 — 정보 조회 실패가 응답을 막지 않게 한다
            logger.debug("[models] 상위 목록 병합 실패(무시): %s", e)

    models = [
        ModelInfo(id=mid, name=upstream.get(mid, mid), role=role)
        for mid, role in (
            (config.model.primary_model, "primary"),
            (config.model.auxiliary_model, "auxiliary"),
            (config.model.embedding_model, "embedding"),
        )
    ]
    return {"models": [m.model_dump() for m in models], "total": len(models)}


# ─────────────────────────────────────────────
# 테넌트 목록 엔드포인트 (Part 5 Ch 15)
# ─────────────────────────────────────────────
@app.get("/v1/tenants")
async def list_tenants() -> dict[str, Any]:
    """등록된 테넌트 목록을 반환한다.

    보안:
      - `api_keys` 원본은 응답에 포함하지 않는다. 개수(`api_key_count`)만 노출.
      - 이 엔드포인트는 내부 LAN 관리 용도. 외부 노출 시에는 프록시 단에서
        인증을 걸어야 한다 (현재 /metrics와 동일한 정책).

    응답 형식:
        {
            "tenants": [TenantInfo, ...],
            "default_tenant": "default",
            "total": N,
        }
    """
    registry = _app_state.get("tenant_registry")
    if registry is None:
        # 레지스트리가 아직 초기화되지 않은 경우 — 빈 목록 반환 (500 대신)
        return {"tenants": [], "default_tenant": "default", "total": 0}

    tenants = [
        TenantInfo(
            id=t.id,
            name=t.name,
            description=t.description,
            model_override=t.model_override,
            allowed_knowledge_sources=list(t.allowed_knowledge_sources),
            api_key_count=len(t.api_keys),
            adapter_name_prefix=t.adapter_name_prefix,
            metadata=dict(t.metadata),
        )
        for t in registry.tenants
    ]
    return {
        "tenants": [t.model_dump() for t in tenants],
        "default_tenant": registry.default_tenant,
        "total": len(tenants),
    }


# ─────────────────────────────────────────────
# 헬스체크 엔드포인트
# ─────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """헬스체크 (GET /health) — 오케스트레이터 + GPU 서버 상태를 반환한다.

    이 서버(오케스트레이터)가 응답한다는 것 자체가 status="ok"의 근거다. GPU 서버는
    config에 적힌 gpu_server_url/health 로 5초 타임아웃 핑을 보내 판정한다:
    200이면 healthy, 그 외 응답이면 unhealthy, 연결 실패면 unreachable,
    config가 없으면 unknown. GPU 핑 실패는 예외로 죽지 않고 상태 문자열로만 표시한다.
    """
    config = _app_state.get("config")
    gpu_status = "unknown"

    if config:
        try:
            import httpx

            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{config.gpu_server_url}/health")
                gpu_status = "healthy" if resp.status_code == 200 else "unhealthy"
        except Exception:
            gpu_status = "unreachable"

    return HealthResponse(
        status="ok",
        version="0.1.0",
        gpu_server=gpu_status,
    )


# ─────────────────────────────────────────────
# 메트릭스 엔드포인트
# ─────────────────────────────────────────────
@app.get("/metrics")
async def metrics() -> dict[str, Any]:
    """서버 메트릭스를 반환한다 (GET /metrics).

    여러 소스의 관측 지표를 하나의 dict로 합쳐 대시보드/모니터링에 노출한다:
      - http    : 요청 로깅 미들웨어가 집계한 HTTP 통계.
      - session : GlobalState의 세션 요약.
      - mcp     : 연결된 MCP 서버 수 + 서버별 도구 개수.
      - agents  : 서브에이전트(scout 등) 호출 통계.
      - agent_cache : Scout 결과 캐시 히트/미스 통계.
      - scout   : (하위 호환 alias) 기존 대시보드용 평탄화 뷰.
      - tenants : 등록 테넌트 목록 + 테넌트별 요청 카운트.
    각 소스는 없을 수 있어(부트스트랩 전 등) 존재할 때만 채운다 — 부분 실패에 견고.
    """
    result: dict[str, Any] = {}

    # 요청 메트릭스
    middleware = _app_state.get("logging_middleware")
    if middleware and hasattr(middleware, "metrics"):
        result["http"] = middleware.metrics

    # 세션 메트릭스
    state = _app_state.get("state")
    if state:
        result["session"] = state.get_session_summary()

        # MCP 가시성 — Phase 2 부트스트랩이 GlobalState 에 채운 등록 결과를
        # 간단히 노출한다(연결 서버 수 + 서버별 도구 개수). 과설계 없이
        # "몇 개 서버가 살아 있고 각자 도구가 몇 개인가" 만 보여준다.
        mcp_servers = getattr(state, "mcp_servers", {}) or {}
        mcp_connected = getattr(state, "mcp_connected", set()) or set()
        result["mcp"] = {
            "connected_count": len(mcp_connected),
            "connected": sorted(mcp_connected),
            "tool_counts": {name: info.get("tool_count", 0) for name, info in mcp_servers.items()},
        }

    # 서브에이전트 메트릭스 — Ch 17 (v7.0 Phase 9 재설계)
    # AgentTool.get_stats()가 subagent_type별 호출 통계를 집계한다.
    # 예: {"scout": {"calls": 3, "total_latency_ms": 99000, "avg_latency_ms": 33000}}
    from core.tools.implementations.agent_tool import AgentTool

    result["agents"] = AgentTool.get_stats()
    # v0.14.2: Scout 결과 캐시 통계. Scout 반복 호출 회피 효과를 관측한다.
    result["agent_cache"] = AgentTool.get_cache_stats()

    # 하위 호환: 기존 대시보드가 result["scout"]을 참조할 수 있으므로 alias 유지.
    # Scout 자동 전처리가 제거됐으므로 Dispatcher.stats는 0만 반환하지만,
    # AgentTool의 "scout" 항목을 평탄화해서 함께 노출한다.
    dispatcher = _app_state.get("model_dispatcher")
    scout_agent_stats = result["agents"].get("scout", {})
    result["scout"] = {
        "tier": dispatcher.tier.value if dispatcher is not None else "unknown",
        "scout_enabled": (dispatcher.scout_enabled if dispatcher is not None else False),
        "scout_calls": scout_agent_stats.get("calls", 0),
        "scout_avg_latency_ms": scout_agent_stats.get("avg_latency_ms", 0.0),
        "scout_fallback_count": 0,  # fallback 개념은 AgentTool 이관 후 의미 없음
        "note": "scout_calls/avg_latency_ms are sourced from AgentTool.get_stats().",
    }

    # 멀티테넌시 — 등록된 테넌트 목록과 테넌트별 호출 통계 (Part 5 Ch 15)
    registry = _app_state.get("tenant_registry")
    tenant_stats = _app_state.get("tenant_stats") or {}
    if registry is not None:
        result["tenants"] = {
            "registered": [
                {
                    "id": t.id,
                    "name": t.name,
                    "has_model_override": bool(t.model_override),
                    "allowed_source_count": len(t.allowed_knowledge_sources),
                    "api_key_count": len(t.api_keys),
                }
                for t in registry.tenants
            ],
            "default_tenant": registry.default_tenant,
            "per_tenant_stats": tenant_stats,
        }

    return result


# ─────────────────────────────────────────────
# 파일 업로드 (문서 분석용)
# ─────────────────────────────────────────────
@app.post("/v1/upload")
async def upload_file(file: UploadFile) -> dict[str, Any]:
    """문서 분석용 파일 업로드 (POST /v1/upload).

    브라우저가 첨부한 파일을 서버 임시 디렉토리({tempdir}/nexus_uploads)에 저장하고
    그 서버 경로를 돌려준다. 이후 채팅에서 그 경로를 DocumentProcess 도구로 넘기면
    모델이 파일 내용을 읽어 분석할 수 있다(모델은 파일 자체가 아니라 경로를 받는다).
    반환: {status, file_path, file_name, size_bytes}.

    입력 검증(2026-08-06 W8) — 브라우저 UI의 10MB 제한은 라우트를 직접 호출하면
    우회되므로 실제 방어선은 여기다(fail-closed).
      - 확장자가 허용 목록 밖이면 415로 거절한다(파싱도 못 하는 파일을 디스크에
        쌓지 않기 위함).
      - 크기가 상한을 넘으면 413으로 거절한다. 이때 전체를 메모리에 올린 뒤
        재는 것이 아니라 조각 단위로 읽으며 넘는 즉시 중단한다(메모리 폭주 방지).
    """
    # 업로드 저장 위치 — AnalyzeImage 도구의 경로 검증과 같은 _uploads_dir() 로
    # 단일 소스를 공유한다. 저장 경로와 분석 허용 경로가 어긋나지 않도록 하기 위함이다.
    # (설정 upload.uploads_dir 이 비어 있으면 {tempdir}/nexus_uploads 로 폴백하는데,
    #  컨테이너에서 그 자리는 재시작에 사라진다 — 배포는 영속 경로를 지정할 것.)
    import uuid
    from pathlib import Path as _Path

    from fastapi import HTTPException

    upload_dir = _uploads_dir()

    # 업로드 한계값은 설정에서 읽는다(하드코딩 금지). 설정이 없는 경량 경로에서도
    # 동작해야 하므로 UploadConfig 기본값으로 폴백한다.
    from core.config import UploadConfig

    upload_cfg = getattr(_app_state.get("config"), "upload", None) or UploadConfig()

    # 저장 파일명은 ASCII-safe로 만든다. 이유: 이후 채팅에서 모델이 이 "서버 경로"를
    # DocumentProcess 도구 인자로 다시 타이핑해야 하는데, 한글·특수문자가 긴 경로는
    # 모델이 재현하다 오타를 내 "파일을 찾을 수 없습니다"로 실패한다(긴 문자열 재현 취약성).
    # 짧은 ASCII 경로(upload-<uuid>.<ext>)면 안정적으로 재현된다. 원본 이름은 표시용으로만 반환.
    orig_name = file.filename or "upload"
    suffix = _Path(orig_name).suffix.lower()
    if len(suffix) > 8 or not all(c.isalnum() or c == "." for c in suffix):
        suffix = ""  # 확장자가 비정상이면 붙이지 않는다(경로 안전)

    # ── 확장자 검사 ─────────────────────────────────────────────
    # allowed_extensions 가 비어 있으면 검사를 건너뛴다(운영 중 임시 완화 탈출구).
    allowed = [e.lower() for e in (upload_cfg.allowed_extensions or [])]
    if allowed and suffix not in allowed:
        raise HTTPException(
            status_code=415,
            detail=(
                f"지원하지 않는 파일 형식입니다: {suffix or '확장자 없음'}. "
                f"허용 형식: {', '.join(allowed)}"
            ),
        )

    stored_name = f"upload-{uuid.uuid4().hex[:12]}{suffix}"
    file_path = upload_dir / stored_name

    # ── 크기 제한 + 저장 ────────────────────────────────────────
    # 조각(1MB) 단위로 읽어 바로 디스크에 흘려보낸다. 상한을 넘는 순간 쓰기를
    # 멈추고 이미 쓴 부분 파일을 지운 뒤 413으로 거절한다(찌꺼기 방지).
    max_bytes = max(int(upload_cfg.max_size_bytes or 0), 0)
    written = 0
    try:
        with file_path.open("wb") as out:
            while chunk := await file.read(1024 * 1024):
                written += len(chunk)
                if max_bytes and written > max_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=(
                            f"파일이 너무 큽니다. 최대 "
                            f"{max_bytes // (1024 * 1024)}MB까지 업로드할 수 있습니다."
                        ),
                    )
                out.write(chunk)
    except Exception:
        # 실패(크기 초과·디스크 오류)하면 반쯤 쓰인 파일을 남기지 않는다.
        file_path.unlink(missing_ok=True)
        raise

    return {
        "status": "ok",
        "file_path": str(file_path),   # ASCII-safe 실제 저장 경로(모델이 재현할 경로)
        "file_name": orig_name,        # 원본 파일명(표시용)
        "size_bytes": written,
    }


@app.get("/v1/download/{filename}")
async def download_file(filename: str, request: Request):
    """
    DocumentExport 도구가 생성한 문서 파일을 다운로드로 내려준다.

    경로 순회 차단(이중 방어):
      1) Path(filename).name 으로 디렉토리 성분(../, 절대경로 등)을 제거.
      2) 확정 경로의 부모가 정확히 exports 디렉토리인지 재확인.
    둘 중 하나라도 어긋나거나 파일이 없으면 404.

    조건부 테넌트 소유권 검사(IDOR 점진 차단):
      경로검증·파일존재 확인 뒤, tb_artifacts에 소유 테넌트가 기록돼 있고 그 값이
      요청 테넌트와 '불일치'하면 404로 숨긴다(403이 아니라 404 — 존재 은닉,
      fail-closed). 반면 (a) 행이 없음(레거시/기록 전 파일), (b) 소유자 미상,
      (c) 인증 미사용으로 요청 테넌트를 알 수 없음, (d) DB 없음/조회 오류 는 모두
      '통과'시킨다(하위호환·가용성 우선, fail-soft).
      TODO(nexus): 이 단계는 '점진 강화' 준비다. 로그인·세션 소유권이 도입되면
        소유자 미상/요청 테넌트 미상도 차단하는 fail-closed로 전환한다.
    """
    from fastapi import HTTPException

    from core.storage.artifacts import ARTIFACT_NOT_FOUND, get_artifact_owner
    from core.tools.implementations.document_export_renderers import MEDIA_TYPES
    from core.tools.implementations.document_export_tool import resolve_exports_dir

    config = _app_state.get("config")
    configured = getattr(getattr(config, "document_export", None), "exports_dir", "")
    exports_dir = resolve_exports_dir(configured).resolve()

    safe_name = Path(filename).name  # 경로 순회 차단(디렉토리 성분 제거)
    target = (exports_dir / safe_name).resolve()

    if target.parent != exports_dir or not target.is_file():
        raise HTTPException(status_code=404, detail="파일을 찾을 수 없습니다.")

    # ── 조건부 테넌트 소유권 검사 ─────────────────────────────────────
    # 요청 테넌트는 인증 미들웨어가 request.state.tenant에 실어준다(인증 성공 시).
    # 인증이 꺼져 있으면 request.state.tenant가 없어 req_tenant_id가 None이 되고,
    # 그 경우 '판정 불가'로 통과시킨다(신뢰 LAN 가정 — fail-soft).
    pool = _app_state.get("pg_pool")
    if pool is not None:
        owner = await get_artifact_owner(pool, safe_name)
        req_tenant = getattr(request.state, "tenant", None)
        req_tenant_id = getattr(req_tenant, "id", None)
        # 오직 "요청 테넌트를 알고 + 소유자가 특정 테넌트 + 서로 불일치"일 때만 404.
        if (
            req_tenant_id is not None
            and owner is not ARTIFACT_NOT_FOUND
            and owner is not None
            and owner != req_tenant_id
        ):
            raise HTTPException(status_code=404, detail="파일을 찾을 수 없습니다.")

    ext = target.suffix.lstrip(".").lower()
    media = MEDIA_TYPES.get(ext, "application/octet-stream")
    # attachment + filename 으로 브라우저가 "다운로드"로 처리하게 한다.
    return FileResponse(str(target), media_type=media, filename=safe_name)


# ─────────────────────────────────────────────
# 채팅 UI (루트 경로)
# ─────────────────────────────────────────────
@app.get("/")
async def root():
    """루트 경로(GET /) — 브라우저 채팅 UI(static/index.html)를 서빙한다.

    정적 index.html이 있으면 그 파일을 그대로 내려주고, 없으면(정적 자산 미배포 등)
    API 안내용 JSON을 폴백으로 반환한다(/docs 로 Swagger UI 안내).
    """
    index_path = Path(__file__).parent / "static" / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {"message": "IDINO NOVA API", "docs": "/docs"}
