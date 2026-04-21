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
            # 라인 수 = 기록된 엔트리 수 (턴 + 메시지 기준, 빠른 근사치)
            with tfile.open("r", encoding="utf-8") as f:
                lines = sum(1 for _ in f)
        except OSError:
            continue
        out.append(
            {
                "session_id": sub.name,
                "last_modified": datetime.fromtimestamp(
                    stat.st_mtime, tz=UTC
                ).isoformat(),
                "entries": lines,
                "path": str(tfile.resolve()),
            }
        )

    out.sort(key=lambda x: x["last_modified"], reverse=True)
    return out[:limit]
