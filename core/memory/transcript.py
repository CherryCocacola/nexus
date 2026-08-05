"""
세션 트랜스크립트 — JSONL(JSON Lines) 기반 영구 대화 기록 모듈.

[이 파일이 하는 일]
사용자와 AI가 주고받은 대화를 세션마다 파일 한 개(transcript.jsonl)에
"한 줄 = 한 발화" 형태로 차곡차곡 쌓아 둔다. 서버가 꺼졌다 켜져도, 또는
Redis 단기 메모리가 만료(TTL 24h)돼 사라져도, 이 파일만 있으면 과거 대화를
그대로 복구/재현/감사할 수 있다. 즉 Redis가 "휘발성 작업 기억"이라면 이
파일은 "영구 보관용 원장(ledger)"에 해당한다. (사양서 Part 5 Ch 16
Session Management 참고)

[이 파일이 제공하는 것 — 공개 API]
  - SessionTranscript            : 기록 담당 클래스. 턴이 끝날 때마다
                                   append_entry()로 한 줄씩 추가한다.
  - list_transcript_sessions()   : 세션 목록 조회(사이드바 표시용).
  - read_transcript_messages()   : 특정 세션의 대화 복원(화면 재표시용).
  - delete_transcript_session()  : 특정 세션 통째로 삭제.

[누가 호출하나 — 호출 관계]
  - QueryEngine(오케스트레이터)이 턴 종료 시 append_entry()를 호출.
  - 웹/프론트엔드가 세션 목록·복원·삭제 API를 통해 나머지 함수를 사용.

[디스크 경로 규약]
  flat(레거시)  : {sessions_dir}/{session_id}/transcript.jsonl
  채널 격리      : {sessions_dir}/{channel}/{session_id}/transcript.jsonl
  (진입점 channel=web/cli/api을 주면 채널 하위 폴더로 세션을 분리해, 서로의
   대화를 조회·복원·삭제하지 못하게 한다. channel=None이면 예전 flat 경로를
   그대로 써 하위호환을 유지한다. 레거시 flat 세션은 채널 폴더가 직속
   transcript.jsonl을 갖지 않으므로 채널 필터 조회에서 자연히 제외된다.)

[한 줄(JSON 객체) 형식 예시]
  {"ts": ISO-8601, "role": "user"|"assistant", "content": "...",
   "turn": N, "usage": {"input_tokens": ..., "output_tokens": ...}}

[핵심 설계 결정과 그 이유]
  - append-only(추가 전용): 턴 순서대로만 쌓고 수정/삭제하지 않는다.
    → 감사 로그의 신뢰성을 위해 과거 기록을 사후 변경하지 않는다.
  - JSON Lines 포맷: 한 줄이 곧 하나의 완결된 JSON이라 스트리밍 파싱이
    쉽고, tail 명령으로 최근 N줄만 실시간 관찰하기 좋다.
  - enabled=False면 어떤 파일 I/O도 하지 않는다(no-op).
    → CI/테스트 환경에서 디스크를 건드리지 않고 끌 수 있다.
  - 디렉토리는 없으면 자동 생성(mkdir parents=True)한다.

작성자: 이현수 / 작성일: 2026-07-05
"""

from __future__ import annotations

import json
import logging
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger("nexus.memory.transcript")


def _session_dir(
    sessions_dir: str | Path, session_id: str, channel: str | None = None
) -> Path:
    """
    세션 폴더 경로를 만든다 — 채널별 히스토리 격리의 단일 지점.

    channel 이 지정되면 진입점(web/cli/api)별 하위 폴더로 세션을 분리하고,
    None 이면 예전 flat 경로({sessions_dir}/{session_id})를 그대로 쓴다.
    기록기·목록·읽기·삭제가 모두 이 함수로 경로를 계산해 규약을 한 곳에서 지킨다.

    Args:
        sessions_dir: 모든 세션 폴더가 모이는 최상위 디렉토리.
        session_id: 세션 식별자.
        channel: "web"/"cli"/"api" 등 진입점 채널. None이면 flat 경로.

    Returns:
        channel 있으면 {sessions_dir}/{channel}/{session_id}, 없으면
        {sessions_dir}/{session_id} (Path).
    """
    root = Path(sessions_dir)
    if channel:
        return root / channel / session_id
    return root / session_id


class SessionTranscript:
    """세션 하나에 대응하는 JSONL 트랜스크립트 "기록기(writer)".

    [역할]
    특정 session_id에 묶인 대화 파일 하나를 열고, 턴이 끝날 때마다
    append_entry()로 user/assistant 발화를 한 줄씩 append 한다. 인스턴스
    하나가 세션 하나를 담당하며, 생성 시점에 세션 폴더를 미리 만들어 둔다.

    [왜 동기식 I/O인가]
    디스크 쓰기를 asyncio로 비동기화하지 않고 그냥 동기 append로 처리한다.
    한 줄 쓰기는 밀리초 단위로 끝나는 반면 턴과 턴 사이 간격은 초 단위라,
    논블로킹으로 만들 실익이 거의 없어 코드를 단순하게 유지하는 쪽을 택했다.

    [실패에 대한 태도 — fail-soft]
    트랜스크립트는 어디까지나 "부가 기록"이므로, 디렉토리 생성이나 파일
    쓰기가 실패해도 예외를 위로 던지지 않는다. 대신 경고 로그만 남기고
    조용히 비활성(enabled=False)으로 넘어가 본 대화 흐름을 막지 않는다.
    """

    def __init__(
        self,
        sessions_dir: str | Path,
        session_id: str,
        enabled: bool = True,
        channel: str | None = None,
    ) -> None:
        """기록기를 초기화하고 세션 폴더를 준비한다.

        매개변수:
          sessions_dir : 모든 세션 폴더가 모여 있는 최상위 디렉토리.
          session_id   : 이 기록기가 담당할 세션 식별자(폴더/파일 경로에 사용).
          enabled      : False면 이 기록기는 어떤 파일 I/O도 하지 않는다.
          channel      : 진입점 채널(web/cli/api). 주면 {sessions_dir}/{channel}/
                         하위로 세션을 격리한다. None이면 flat 경로(하위호환).

        동작:
          경로(self._base, self._path)를 계산해 두고, enabled일 때에 한해
          세션 폴더를 미리 생성한다. exist_ok=True라 폴더가 이미 있어도 OK.
          폴더 생성이 실패하면 예외를 던지지 않고 스스로 비활성으로 전환한다.
        """
        # 외부에서 어떤 값이 와도 명확한 bool로 정규화(예: None, 0 등 방어).
        self._enabled = bool(enabled)
        self._session_id = session_id
        self._channel = channel
        # 세션 전용 폴더 경로와 그 안의 트랜스크립트 파일 경로를 미리 계산.
        # 경로 규약(채널 격리 포함)은 _session_dir 한 곳에서 관장한다.
        self._base = _session_dir(sessions_dir, session_id, channel)
        self._path = self._base / "transcript.jsonl"
        if self._enabled:
            try:
                # parents=True: 중간 경로까지 한 번에 생성. exist_ok=True:
                # 이미 있으면 조용히 통과(재시작/재접속 시 정상 상황).
                self._base.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                # 폴더를 못 만들면(권한/디스크 등) 기록을 포기하되 서비스는
                # 계속 살린다 — 트랜스크립트는 치명적 기능이 아니므로 fail-soft.
                logger.warning(
                    "트랜스크립트 디렉토리 생성 실패 (%s): %s — 비활성 전환",
                    self._base, e,
                )
                self._enabled = False

    @property
    def enabled(self) -> bool:
        """이 기록기가 실제로 파일에 기록 중인지 여부(읽기 전용)."""
        return self._enabled

    @property
    def path(self) -> Path:
        """이 세션의 트랜스크립트 파일 경로(읽기 전용)."""
        return self._path

    def append_entry(
        self,
        role: str,
        content: str,
        turn: int,
        usage: dict[str, Any] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """발화 하나를 JSONL 파일 끝에 한 줄로 추가한다.

        매개변수:
          role    : 발화 주체("user" | "assistant" | 필요 시 "system" 등).
          content : 발화 본문. 빈 문자열이면 기록하지 않는다.
          turn    : 이 발화가 속한 턴 번호(대화 순서 복원/디버깅용).
          usage   : 토큰 사용량 등 부가 통계(있을 때만 기록).
          extra   : 그 밖의 임의 메타데이터(있을 때만 기록).

        동작:
          enabled=False이거나 content가 비어 있으면 아무 일도 하지 않는다
          (no-op). 그 외에는 타임스탬프를 찍은 JSON 한 줄을 파일에 append
          한다. 쓰기 실패는 경고 로그만 남기고 삼킨다(fail-soft).
        """
        # 비활성 상태이거나 내용이 비면 기록할 이유가 없으므로 즉시 반환.
        if not self._enabled or not content:
            return
        # 항상 존재하는 기본 필드로 한 줄(엔트리)을 구성. ts는 UTC ISO-8601.
        entry: dict[str, Any] = {
            "ts": datetime.now(UTC).isoformat(),
            "session_id": self._session_id,
            "turn": turn,
            "role": role,
            "content": content,
        }
        # usage/extra는 값이 있을 때만 넣어 파일을 불필요하게 키우지 않는다.
        if usage:
            entry["usage"] = usage
        if extra:
            entry["extra"] = extra
        try:
            # "a"(append) 모드로 열어 기존 내용 뒤에 이어 쓴다. 한글이 깨지지
            # 않도록 ensure_ascii=False로 직렬화하고, 줄 끝에 개행을 붙여
            # "한 줄 = 한 엔트리" 규약(JSON Lines)을 지킨다.
            with self._path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError as e:
            # 디스크 가득참/권한 등으로 실패해도 대화 흐름은 막지 않는다.
            logger.warning("트랜스크립트 append 실패 (%s): %s", self._path, e)


def list_transcript_sessions(
    sessions_dir: str | Path,
    limit: int = 50,
    channel: str | None = None,
) -> list[dict[str, Any]]:
    """
    트랜스크립트 디렉토리를 훑어 최근 세션 목록을 만들어 반환한다.

    [용도]
    웹 사이드바에 "지난 대화 목록"을 뿌리기 위한 함수. 각 세션 폴더 안의
    transcript.jsonl을 찾아 메타데이터(수정 시각·줄 수·미리보기 제목)를 뽑는다.

    [채널 격리]
      - channel 지정 → {sessions_dir}/{channel}/ 아래만 훑어 그 채널 세션만 반환.
      - channel None → {sessions_dir}/ 최상위만 훑는다. 채널 하위 폴더(web/cli/api)는
        직속 transcript.jsonl이 없어 자동으로 건너뛰므로, 레거시 flat 세션만 잡힌다
        (상호 비침범 — 채널 세션이 flat 목록에 새지 않는다).

    [정렬/필터]
      - 각 세션의 파일 최종 수정 시각(mtime) 기준 내림차순(최신이 위).
      - 최대 limit개까지만 반환.
      - 대상 디렉토리가 없거나 디렉토리가 아니면 빈 리스트.

    반환 예시:
      [
        {"session_id": "...", "last_modified": ISO-8601, "entries": N,
         "path": "절대경로", "title_hint": "첫 사용자 발화 미리보기"},
        ...
      ]
    """
    # 채널이 지정되면 그 하위 폴더를, 아니면 최상위를 스캔 기준으로 삼는다.
    base = Path(sessions_dir) / channel if channel else Path(sessions_dir)
    # 대상 폴더 자체가 없으면 보여줄 세션도 없으므로 빈 목록 반환.
    if not base.exists() or not base.is_dir():
        return []

    out: list[dict[str, Any]] = []
    # 최상위 폴더의 바로 아래 항목들을 하나씩 검사(각각이 세션 후보).
    for sub in base.iterdir():
        # 세션은 폴더 단위이므로 폴더가 아닌 항목은 건너뛴다.
        if not sub.is_dir():
            continue
        tfile = sub / "transcript.jsonl"
        # 트랜스크립트 파일이 없는 폴더는 유효한 세션이 아니므로 스킵.
        if not tfile.exists():
            continue
        try:
            # 파일 메타데이터(수정 시각 등)를 먼저 확보.
            stat = tfile.stat()
            # 파일을 딱 한 번만 읽으면서 (1) 전체 줄 수와 (2) 첫 user 발화를
            # 동시에 뽑아 I/O를 아낀다. title_hint(사이드바 미리보기)는
            # 첫 user 발화 앞부분을 잘라 "무슨 대화였는지" 바로 보여주려는 것.
            lines = 0
            first_user: str | None = None
            with tfile.open("r", encoding="utf-8") as f:
                for raw in f:
                    lines += 1
                    # 첫 user 발화를 이미 찾았다면 이후로는 줄 수만 계속 센다.
                    if first_user is not None:
                        continue  # 라인 수 세기만 계속
                    stripped = raw.strip()
                    # 빈 줄은 파싱 대상이 아니므로 건너뜀.
                    if not stripped:
                        continue
                    try:
                        entry = json.loads(stripped)
                    except json.JSONDecodeError:
                        # 손상된 줄은 미리보기 후보에서 제외(줄 수엔 이미 포함).
                        continue
                    # 첫 번째 user 역할 발화를 미리보기 원본으로 채택.
                    if entry.get("role") == "user":
                        content = entry.get("content") or ""
                        if isinstance(content, str):
                            # 줄바꿈을 공백으로 펴서 한 줄 미리보기로 만든다.
                            first_user = content.strip().replace("\n", " ")
        except OSError:
            # 특정 파일을 못 읽어도 전체 목록 작성은 계속 진행.
            continue
        title_hint = None
        if first_user:
            # 미리보기는 60자까지만. 넘치면 잘라내고 말줄임표(…)를 붙인다.
            title_hint = first_user[:60] + ("…" if len(first_user) > 60 else "")
        # 세션 메타(meta.json) — 사용자 지정 제목·핀 여부. 없으면 빈 dict.
        meta = read_session_meta(sessions_dir, sub.name, channel)
        out.append(
            {
                "session_id": sub.name,
                "last_modified": datetime.fromtimestamp(
                    stat.st_mtime, tz=UTC
                ).isoformat(),
                "entries": lines,
                "path": str(tfile.resolve()),
                "title_hint": title_hint,
                # 사용자 지정 제목(있으면 프론트가 우선 사용) + 핀 여부 + 폴더.
                "title": meta.get("title"),
                "pinned": bool(meta.get("pinned")),
                "folder": meta.get("folder") or "",
            }
        )

    # 최신 수정 순으로 정렬한 뒤 상위 limit개만 잘라 반환.
    out.sort(key=lambda x: x["last_modified"], reverse=True)
    return out[:limit]


def search_transcript_sessions(
    sessions_dir: str | Path,
    query: str,
    *,
    channel: str | None = None,
    limit: int = 20,
    max_snippets_per_session: int = 3,
    max_lines_per_file: int = 5000,
) -> list[dict[str, Any]]:
    """세션 트랜스크립트를 훑어 질의어를 포함한 세션과 매치 스니펫을 반환한다.

    [용도]
    사이드바 대화 검색. 세션 JSONL을 라인 단위로 스캔해 user/assistant 발화 중
    질의어(대소문자 무시 부분일치)를 포함한 줄을 찾아, 매치 주변 ±40자 스니펫을
    세션별 최대 max_snippets_per_session개까지 모은다.

    [채널 격리] channel을 주면 그 하위만 스캔(web/cli/api 상호 비침범).

    [방어]
      - query는 최소 2자(그보다 짧으면 빈 결과 — 전체 스캔 폭주 방지).
      - 파일당 max_lines_per_file 줄까지만 스캔(거대 세션 방어).
      - 손상된 JSON 라인은 건너뛴다.

    반환: [{session_id, last_modified(ISO), snippets:[{role, text, turn, ts}]}, ...]
          파일 최종 수정 시각 내림차순, 최대 limit개.
    """
    q = (query or "").strip()
    if len(q) < 2:
        return []
    ql = q.lower()

    base = Path(sessions_dir) / channel if channel else Path(sessions_dir)
    if not base.exists() or not base.is_dir():
        return []

    results: list[dict[str, Any]] = []
    for sub in base.iterdir():
        if not sub.is_dir():
            continue
        tfile = sub / "transcript.jsonl"
        if not tfile.exists():
            continue
        snippets: list[dict[str, Any]] = []
        last_modified: str | None = None
        try:
            last_modified = datetime.fromtimestamp(
                tfile.stat().st_mtime, tz=UTC
            ).isoformat()
            with tfile.open("r", encoding="utf-8") as f:
                for i, raw in enumerate(f):
                    if i >= max_lines_per_file or len(snippets) >= max_snippets_per_session:
                        break
                    line = raw.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    role = entry.get("role")
                    content = entry.get("content")
                    if role not in ("user", "assistant") or not isinstance(content, str):
                        continue
                    idx = content.lower().find(ql)
                    if idx < 0:
                        continue
                    # 매치 주변 ±40자를 스니펫으로 잘라 한 줄로 편다.
                    start = max(0, idx - 40)
                    end = min(len(content), idx + len(q) + 40)
                    text = content[start:end].replace("\n", " ").strip()
                    snippets.append(
                        {
                            "role": role,
                            "text": text,
                            "turn": entry.get("turn"),
                            "ts": entry.get("ts"),
                        }
                    )
        except OSError:
            # 특정 파일을 못 읽어도 전체 검색은 계속 진행.
            continue
        if snippets:
            results.append(
                {
                    "session_id": sub.name,
                    "last_modified": last_modified,
                    "snippets": snippets,
                }
            )

    results.sort(key=lambda x: x["last_modified"] or "", reverse=True)
    return results[:limit]


def read_transcript_messages(
    sessions_dir: str | Path,
    session_id: str,
    *,
    roles: tuple[str, ...] = ("user", "assistant"),
    limit: int | None = None,
    channel: str | None = None,
) -> list[dict[str, Any]]:
    """
    특정 세션의 트랜스크립트(JSONL)를 읽어 메시지 리스트로 복원해 반환한다.

    [용도]
    Ch 16 세션 영속화의 읽기 경로. 사용자가 사이드바에서 지난 세션을
    다시 열면, 프론트엔드가 이 함수로 과거 대화를 받아 화면에 복원한다.

    매개변수:
      sessions_dir : 세션 폴더들이 모인 최상위 디렉토리.
      session_id   : 읽어올 세션 식별자.
      roles        : 반환에 포함할 역할 집합(키워드 전용 인자).
      limit        : 최근 N개만 원할 때 지정(None이면 전체).
      channel      : 진입점 채널(web/cli/api). 기록 때와 같은 값을 넘겨야 그 세션을 읽는다.

    필터 규칙:
      - roles에 포함된 역할만 반환(기본은 user/assistant). system 에러
        엔트리는 기본값에서 제외되며, ("user","assistant","system")처럼
        명시적으로 넘기면 함께 포함된다.
      - 손상된 JSON 라인은 조용히 건너뛴다(감사 기록은 best-effort 복원).
      - limit가 있으면 가장 최근 limit개만 반환하되, 원래의 시간 순서
        (오래된 것 → 최신)는 그대로 유지한다.

    반환 예시:
      [
        {"role": "user", "content": "...", "turn": N, "ts": "ISO-8601",
         "usage": {...}|None, "extra": {...}|None},
        ...
      ]

    파일이 없으면 빈 리스트.
    """
    base = _session_dir(sessions_dir, session_id, channel)
    tfile = base / "transcript.jsonl"
    # 해당 세션 파일이 없으면 복원할 대화도 없다.
    if not tfile.exists():
        return []

    # 매 줄마다 in 검사를 빠르게 하려고 튜플을 set으로 변환.
    role_filter = set(roles)
    out: list[dict[str, Any]] = []
    try:
        with tfile.open("r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                # 빈 줄은 건너뜀.
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    # 손상된 라인은 무시 (감사 기록의 best-effort 원칙)
                    continue
                role = entry.get("role")
                content = entry.get("content")
                # 원하는 역할이 아니거나 내용이 비면 결과에서 제외.
                if role not in role_filter or not content:
                    continue
                # 프론트가 바로 쓰기 좋은 평평한 딕셔너리로 정규화해 담는다.
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
        # 파일을 여는 도중 I/O 오류가 나면 경고만 남기고 빈 목록 반환.
        logger.warning("트랜스크립트 읽기 실패 (%s): %s", tfile, e)
        return []

    # limit가 유효하면 뒤에서 limit개만 취한다. 슬라이싱이라 시간 순서는
    # 그대로 보존되고 "가장 최근 N개"만 남는다.
    if limit is not None and limit > 0 and len(out) > limit:
        out = out[-limit:]
    return out


def read_session_meta(
    sessions_dir: str | Path, session_id: str, channel: str | None = None
) -> dict[str, Any]:
    """세션 메타데이터(meta.json)를 읽어 dict로 반환한다(없거나 손상 시 빈 dict).

    [용도]
    트랜스크립트 본문(대화)과 분리된 부가 정보 — 사용자 지정 제목(title),
    핀 여부(pinned), (후속) 폴더/프로젝트 소속 등 — 를 세션 디렉토리의
    meta.json 사이드카에 둔다. 목록 조회(list_transcript_sessions)와
    웹 PATCH 엔드포인트가 이 함수를 공유해 규약을 한 곳에서 지킨다.

    fail-soft: 파일이 없거나 JSON이 깨져도 예외를 던지지 않고 빈 dict를 준다
    (메타 부재/손상이 세션 열람을 막지 않게 — 가용성 우선).
    """
    base = _session_dir(sessions_dir, session_id, channel)
    mfile = base / "meta.json"
    if not mfile.exists():
        return {}
    try:
        with mfile.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("세션 메타 읽기 실패 (%s): %s", mfile, e)
        return {}


def write_session_meta(
    sessions_dir: str | Path,
    session_id: str,
    meta: dict[str, Any],
    channel: str | None = None,
) -> bool:
    """세션 메타데이터를 meta.json에 "병합" 저장한다(성공 시 True).

    [병합 규칙]
    기존 meta.json을 읽어 전달된 meta의 값 중 None이 아닌 것만 덮어쓴다
    (pinned=False처럼 명시적 False는 반영, 미지정 None은 무시). updated_at은
    항상 현재 UTC로 갱신한다.

    [안전장치 — delete_transcript_session과 동일한 경로 탈출 방어]
      1) session_id에 슬래시/백슬래시/'..'/널문자가 있으면 ValueError.
      2) resolve() 후 대상이 sessions_dir 루트의 하위인지 재검증.
    Redis-only(트랜스크립트 파일이 아직 없는) 세션도 메타를 가질 수 있도록
    디렉토리를 없으면 만든다(mkdir parents=True).

    fail-soft: 쓰기 I/O 오류는 경고만 남기고 False를 반환한다(요청을 깨지 않음).
    """
    if not session_id or any(ch in session_id for ch in ("/", "\\", "..", "\x00")):
        raise ValueError(f"invalid session_id: {session_id!r}")

    base_root = Path(sessions_dir).resolve()
    target = _session_dir(sessions_dir, session_id, channel).resolve()
    try:
        target.relative_to(base_root)
    except ValueError as e:
        raise ValueError(
            f"session_id가 sessions_dir 바깥을 가리킴: {session_id!r}"
        ) from e

    # 기존 메타에 병합(None 값은 무시 — 부분 갱신 지원).
    current = read_session_meta(sessions_dir, session_id, channel)
    current.update({k: v for k, v in meta.items() if v is not None})
    current["updated_at"] = datetime.now(UTC).isoformat()
    try:
        target.mkdir(parents=True, exist_ok=True)
        with (target / "meta.json").open("w", encoding="utf-8") as f:
            json.dump(current, f, ensure_ascii=False)
        return True
    except OSError as e:
        logger.warning("세션 메타 쓰기 실패 (%s): %s", target / "meta.json", e)
        return False


def delete_transcript_session(
    sessions_dir: str | Path,
    session_id: str,
    channel: str | None = None,
) -> bool:
    """
    특정 세션의 트랜스크립트 디렉토리를 통째로 삭제한다.

    [용도]
    Ch 16 세션 삭제 API. 사용자가 사이드바에서 대화를 제거할 때 호출된다.
    파일 하나가 아니라 세션 폴더 전체(하위 내용 포함)를 지운다.

    [안전장치 — 왜 필요한가]
    session_id는 외부(프론트/사용자)에서 들어오는 값이므로, 그대로 경로에
    붙이면 "../../.." 같은 입력으로 엉뚱한 폴더를 지울 위험(경로 탈출)이 있다.
    그래서 두 단계로 방어한다:
      1) 문자 검사 — session_id에 슬래시/백슬래시/".."/널문자가 있으면 거부.
      2) 경로 재검증 — resolve() 후 대상이 sessions_dir의 하위인지 확인.
    또한 존재하지 않는 세션 삭제 요청은 오류가 아니라 "이미 없음"으로 보고
    False를 돌려준다.

    반환:
      실제로 디렉토리를 지웠으면 True, 애초에 없었으면 False.
      입력이 위험/부적절하면 ValueError.
    """
    # [방어 1] 위험 문자가 섞인 session_id는 경로 조작 시도로 보고 즉시 거부.
    if not session_id or any(ch in session_id for ch in ("/", "\\", "..", "\x00")):
        raise ValueError(f"invalid session_id: {session_id!r}")

    # 기준 폴더(sessions_dir 루트)와 삭제 대상 경로를 절대경로로 정규화.
    # 대상은 채널 격리 규약(_session_dir)으로 만들되, 방어 검사는 항상 루트 기준.
    base = Path(sessions_dir).resolve()
    target = _session_dir(sessions_dir, session_id, channel).resolve()

    # [방어 2] 정규화한 target이 정말 base(루트)의 하위인지 확인. 채널이 붙어도
    # 결국 sessions_dir 안에 있어야 한다. 바깥을 가리키면 relative_to가
    # ValueError를 내며, 이를 명확한 메시지로 다시 던진다.
    try:
        target.relative_to(base)
    except ValueError as e:
        raise ValueError(
            f"session_id가 sessions_dir 바깥을 가리킴: {session_id!r}"
        ) from e

    # 대상이 없거나 폴더가 아니면 지울 것이 없으므로 False(멱등적 처리).
    if not target.exists() or not target.is_dir():
        return False

    try:
        # 폴더와 그 안의 모든 내용을 재귀적으로 삭제.
        shutil.rmtree(target)
    except OSError as e:
        # 삭제 실패는 호출자가 알아야 하므로 로그 후 예외를 그대로 전파.
        logger.warning("트랜스크립트 디렉토리 삭제 실패 (%s): %s", target, e)
        raise
    return True
