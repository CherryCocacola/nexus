# 모델에 실제로 나간 프롬프트를 요청 ID 기준으로 파일에 남긴다(기본 비활성).
"""
프롬프트 덤프 — "모델이 실제로 무엇을 봤는가"를 나중에 확인하기 위한 진단 장치.

■ 왜 필요한가 (2026-08-16)
    `/v1/chat/completions` 는 무상태라 메시지도 조립된 프롬프트도 저장되지 않는다.
    그래서 "왜 이런 답이 나왔지"를 물으면 **재현하는 것 말고는 방법이 없었다.**
    특히 이 서버는 프롬프트를 그냥 흘려보내지 않는다 — 이전 턴 요약, 계획
    체크리스트, 지식 RAG(질의가 KNOWLEDGE 로 분류될 때), 클라이언트 도구 안내문이
    시스템 프롬프트에 덧붙는다. 무엇이 붙었는지 보이지 않으면 원인을 못 찾는다.

■ 왜 기본으로 꺼 두는가
    덤프에는 **사내 문서 RAG 본문과 사용자 대화가 그대로** 들어간다. 진단에 유용한
    만큼 남기면 위험한 내용이다. 그래서 `NEXUS_PROMPT_DUMP_DIR` 를 명시적으로
    주입했을 때만 동작한다(미설정 = 완전 비활성). CORS 오리진을 환경변수로만 여는
    것과 같은 원칙이다 — 위험을 늘리는 결정은 조용히 켜지지 않는다.

■ 무엇을 남기나
    `model_provider.stream()` **직전**의 값이다. 프롬프트 조립·압축이 모두 끝난
    진짜 최종본이라, 여기서 남긴 것이 곧 모델이 본 것이다.

■ 키
    요청 ID(`X-Request-ID`) 기준으로 턴마다 한 파일. 요청 ID 가 없으면 세션 ID 로
    떨어진다. 도구 루프는 한 요청이 여러 턴을 도므로 턴 번호로 나눈다.

작성자: 이현수 / 작성일: 2026-08-16
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger("nexus.orchestrator.prompt_dump")

# 이 환경변수가 있을 때만 덤프한다. 값이 곧 저장 디렉토리다.
DUMP_DIR_ENV = "NEXUS_PROMPT_DUMP_DIR"
# 보존 시간(시간). 민감 내용이라 오래 두지 않는다.
RETENTION_ENV = "NEXUS_PROMPT_DUMP_RETENTION_HOURS"
DEFAULT_RETENTION_HOURS = 24.0

# 파일명에 쓸 수 있는 문자만 남긴다 — 요청 ID 는 클라이언트가 정한 값이라
# 경로 조작(`../`)이 섞일 수 있다. 웹 계층에서 이미 걸러지지만 여기서도 막는다.
_SAFE = re.compile(r"[^A-Za-z0-9._-]")


def dump_dir() -> Path | None:
    """덤프 디렉토리. 환경변수가 없으면 None(= 비활성)."""
    raw = os.environ.get(DUMP_DIR_ENV, "").strip()
    return Path(raw) if raw else None


def is_enabled() -> bool:
    """덤프가 켜져 있는지 — 호출부가 값을 만들기 전에 먼저 확인해 비용을 아낀다."""
    return dump_dir() is not None


def _retention_hours() -> float:
    try:
        return float(os.environ.get(RETENTION_ENV, "") or DEFAULT_RETENTION_HOURS)
    except ValueError:
        return DEFAULT_RETENTION_HOURS


def _cleanup(directory: Path) -> None:
    """보존 시간이 지난 덤프를 지운다(민감 내용을 오래 남기지 않는다)."""
    cutoff = time.time() - _retention_hours() * 3600
    for f in directory.glob("*.json"):
        try:
            if f.stat().st_mtime < cutoff:
                f.unlink()
        except OSError:
            continue  # 지우기 실패는 다음 회차에 다시 시도한다


def dump_prompt(
    *,
    request_id: str | None,
    session_id: str,
    turn: int,
    system_prompt: str,
    messages: list[Any],
    tool_names: list[str],
    routing: dict[str, Any],
    sampling: dict[str, Any],
) -> Path | None:
    """이번 턴에 모델로 나가는 프롬프트를 파일로 남긴다.

    Returns:
        저장한 경로. 비활성이거나 실패하면 None.

    실패는 조용히 넘긴다 — 진단 장치가 본 요청을 막으면 본말이 전도된다.
    """
    directory = dump_dir()
    if directory is None:
        return None

    key = _SAFE.sub("_", (request_id or session_id or "unknown"))[:64]
    path = directory / f"{key}_turn{turn}.json"
    payload = {
        "request_id": request_id,
        "session_id": session_id,
        "turn": turn,
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "routing": routing,
        "sampling": sampling,
        # 도구는 스키마 전문이 아니라 이름만 — 프롬프트 본문을 가리지 않게.
        "tools": tool_names,
        "system_prompt": system_prompt,
        "messages": [
            {
                "role": getattr(m.role, "value", None) or str(getattr(m, "role", "?")),
                "content": str(getattr(m, "content", "")),
            }
            for m in messages
        ],
    }
    try:
        directory.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        _cleanup(directory)
        return path
    except Exception as e:  # noqa: BLE001 — 진단 실패가 응답을 막지 않게 한다
        logger.debug("[prompt_dump] 저장 실패(무시): %s", e)
        return None
