"""
세션 트랜스크립트 — JSONL 기반 영구 기록.

사양서 Part 5 Ch 16 (Session Management): 세션 대화 이력을 파일에 남겨
서버 재기동 후에도 복기 가능하게 한다. Redis(단기, TTL 24h)가 휘발성이라면,
이 파일 트랜스크립트는 **감사/재현/복구**용 영구 기록이다.

경로 규약:
  {sessions_dir}/{session_id}/transcript.jsonl

한 줄 형식(JSON 객체):
  {"ts": ISO-8601, "role": "user"|"assistant", "content": "...",
   "turn": N, "usage": {"input_tokens": ..., "output_tokens": ...}}

설계 결정:
  - append-only — 턴 순서로만 누적, 수정/삭제 없음
  - JSON Lines — 스트리밍 파싱이 쉽고 tail로 최근 N줄 관찰 가능
  - transcript_enabled=False면 아무 I/O도 하지 않음 (CI/테스트에서 비활성 가능)
  - 디렉토리 자동 생성 (없으면 mkdir parents=True)
"""

from __future__ import annotations

import json
import logging
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger("nexus.memory.transcript")


class SessionTranscript:
    """세션별 JSONL 트랜스크립트 기록기.

    QueryEngine이 턴 종료 시 append_turn()을 호출하여 user/assistant 쌍을
    기록한다. 디스크 I/O는 동기식 append로 단순하게 처리한다(한 줄 기록은
    밀리초 단위로 끝나며, 턴 간격이 초 단위라 논블로킹화 가치가 낮다).
    """

    def __init__(
        self,
        sessions_dir: str | Path,
        session_id: str,
        enabled: bool = True,
    ) -> None:
        self._enabled = bool(enabled)
        self._session_id = session_id
        self._base = Path(sessions_dir) / session_id
        self._path = self._base / "transcript.jsonl"
        if self._enabled:
            try:
                self._base.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                # 디렉토리 생성 실패 → 트랜스크립트 비활성 (치명적 아님)
                logger.warning(
                    "트랜스크립트 디렉토리 생성 실패 (%s): %s — 비활성 전환",
                    self._base, e,
                )
                self._enabled = False

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def path(self) -> Path:
        return self._path

    def append_entry(
        self,
        role: str,
        content: str,
        turn: int,
        usage: dict[str, Any] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """한 줄을 JSONL에 추가한다. enabled=False면 no-op."""
        if not self._enabled or not content:
            return
        entry: dict[str, Any] = {
            "ts": datetime.now(UTC).isoformat(),
            "session_id": self._session_id,
            "turn": turn,
            "role": role,
            "content": content,
        }
        if usage:
            entry["usage"] = usage
        if extra:
            entry["extra"] = extra
        try:
            with self._path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError as e:
            logger.warning("트랜스크립트 append 실패 (%s): %s", self._path, e)


def list_transcript_sessions(
    sessions_dir: str | Path,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """
    트랜스크립트 디렉토리를 스캔하여 최근 세션 목록을 반환한다.

    각 세션의 최종 수정 시각(mtime) 기준 내림차순 정렬.
    파일이 없거나 디렉토리가 비어 있으면 빈 리스트를 반환한다.

    Returns:
      [
        {"session_id": "...", "last_modified": ISO-8601, "turns": N,
         "path": "절대경로"},
        ...
      ]
    """
    base = Path(sessions_dir)
    if not base.exists() or not base.is_dir():
        return []

    out: list[dict[str, Any]] = []
    for sub in base.iterdir():
        if not sub.is_dir():
            continue
        tfile = sub / "transcript.jsonl"
        if not tfile.exists():
            continue
        try:
            stat = tfile.stat()
            # 한 번의 패스로 라인 수 + 첫 user 메시지를 동시에 추출 (I/O 절약).
            # title_hint는 사이드바에 "무슨 대화였는지" 즉시 보여주기 위해
            # 첫 user 메시지 앞 60자 정도를 돌려준다.
            lines = 0
            first_user: str | None = None
            with tfile.open("r", encoding="utf-8") as f:
                for raw in f:
                    lines += 1
                    if first_user is not None:
                        continue  # 라인 수 세기만 계속
                    stripped = raw.strip()
                    if not stripped:
                        continue
                    try:
                        entry = json.loads(stripped)
                    except json.JSONDecodeError:
                        continue
                    if entry.get("role") == "user":
                        content = entry.get("content") or ""
                        if isinstance(content, str):
                            first_user = content.strip().replace("\n", " ")
        except OSError:
            continue
        title_hint = None
        if first_user:
            # 60자를 넘기면 절단하고 말줄임표
            title_hint = first_user[:60] + ("…" if len(first_user) > 60 else "")
        out.append(
            {
                "session_id": sub.name,
                "last_modified": datetime.fromtimestamp(
                    stat.st_mtime, tz=UTC
                ).isoformat(),
                "entries": lines,
                "path": str(tfile.resolve()),
                "title_hint": title_hint,
            }
        )

    out.sort(key=lambda x: x["last_modified"], reverse=True)
    return out[:limit]


def read_transcript_messages(
    sessions_dir: str | Path,
    session_id: str,
    *,
    roles: tuple[str, ...] = ("user", "assistant"),
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """
    특정 세션의 트랜스크립트(JSONL)를 읽어 메시지 리스트로 반환한다.

    Ch 16 세션 영속화 — 프론트엔드가 과거 대화를 화면에 복원할 때 사용한다.

    필터:
      - roles에 포함된 role만 반환 (기본 user/assistant). system 에러 엔트리는
        기본 필터에서 제외되며, 명시적으로 ("user", "assistant", "system")을
        넘기면 포함된다.
      - 손상된 JSON 라인은 조용히 스킵 (감사 기록은 best-effort 복원).
      - limit가 지정되면 가장 최근 limit개만 반환 (오래된 것부터 순서 유지).

    Returns:
      [
        {"role": "user", "content": "...", "turn": N, "ts": "ISO-8601",
         "usage": {...}|None, "extra": {...}|None},
        ...
      ]

    파일이 없으면 빈 리스트.
    """
    base = Path(sessions_dir) / session_id
    tfile = base / "transcript.jsonl"
    if not tfile.exists():
        return []

    role_filter = set(roles)
    out: list[dict[str, Any]] = []
    try:
        with tfile.open("r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    # 손상된 라인은 무시 (감사 기록의 best-effort 원칙)
                    continue
                role = entry.get("role")
                content = entry.get("content")
                if role not in role_filter or not content:
                    continue
                out.append(
                    {
                        "role": role,
                        "content": content,
                        "turn": entry.get("turn"),
                        "ts": entry.get("ts"),
                        "usage": entry.get("usage"),
                        "extra": entry.get("extra"),
                    }
                )
    except OSError as e:
        logger.warning("트랜스크립트 읽기 실패 (%s): %s", tfile, e)
        return []

    if limit is not None and limit > 0 and len(out) > limit:
        out = out[-limit:]
    return out


def delete_transcript_session(
    sessions_dir: str | Path,
    session_id: str,
) -> bool:
    """
    특정 세션의 트랜스크립트 디렉토리를 통째로 삭제한다.

    사용처: Ch 16 세션 삭제 API. 프론트가 사이드바에서 세션을 제거할 때 호출.

    안전장치:
      - 경로 탈출 방지 — session_id에 슬래시/백슬래시/'..' 포함 시 거부
      - 대상이 sessions_dir의 직계 자식 디렉토리인지 resolve 후 재검증
      - shutil.rmtree(missing_ok=True) — 없는 세션 삭제는 성공으로 처리

    Returns:
      실제로 디렉토리를 지웠으면 True, 애초에 없었으면 False.
      검증 실패 시 ValueError.
    """
    if not session_id or any(ch in session_id for ch in ("/", "\\", "..", "\x00")):
        raise ValueError(f"invalid session_id: {session_id!r}")

    base = Path(sessions_dir).resolve()
    target = (base / session_id).resolve()

    # 탈출 방지 이중 검증 — target이 실제로 base의 자식인지
    try:
        target.relative_to(base)
    except ValueError as e:
        raise ValueError(
            f"session_id가 sessions_dir 바깥을 가리킴: {session_id!r}"
        ) from e

    if not target.exists() or not target.is_dir():
        return False

    try:
        shutil.rmtree(target)
    except OSError as e:
        logger.warning("트랜스크립트 디렉토리 삭제 실패 (%s): %s", target, e)
        raise
    return True
