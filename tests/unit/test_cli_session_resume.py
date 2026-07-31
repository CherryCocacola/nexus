# CLI --resume 세션 복원 검증 — 트랜스크립트를 엔진 메시지로 되살린다.
"""
`NexusREPL._load_resume_messages`가 이전 세션의 transcript.jsonl을 읽어
Message 리스트로 복원하고, 세션 ID를 승계(같은 파일에 이어쓰기)하는지 검증한다.

배경: `--resume` 옵션은 배너 표시만 하고 실제 복원 배선이 빠져 있었다(미완성
스텁). read_transcript_messages / bind_request 인프라는 이미 있었고, 이 테스트가
그 배선의 회귀를 막는다. REPL 전체를 인스턴스화하면 PromptSession이 콘솔을
요구해 테스트 환경에서 실패하므로, 스텁 self로 순수 로직만 호출한다.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from cli.repl import NexusREPL


def _write_transcript(sessions_dir: Path, session_id: str, entries: list[dict]) -> None:
    # CLI 세션은 'cli' 채널로 격리 저장된다({sessions_dir}/cli/{id}/). _load_resume_messages도
    # channel="cli"로 읽으므로 fixture를 같은 채널 경로에 둔다(진입점별 히스토리 격리).
    base = sessions_dir / "cli" / session_id
    base.mkdir(parents=True, exist_ok=True)
    (base / "transcript.jsonl").write_text(
        "\n".join(json.dumps(e, ensure_ascii=False) for e in entries) + "\n",
        encoding="utf-8",
    )


def _stub(sessions_dir: Path, resume_id: str | None, session_id: str = "new-uuid"):
    """PromptSession을 우회한 최소 self — _load_resume_messages가 쓰는 필드만 갖춘다."""
    cfg = SimpleNamespace(sessions_dir=str(sessions_dir))
    state = SimpleNamespace(config=cfg, session_id=session_id)
    return SimpleNamespace(_resume_session_id=resume_id, _state=state)


def test_resume_restores_messages_in_order(tmp_path):
    _write_transcript(tmp_path, "sess-A", [
        {"role": "user", "content": "내 이름은 홍길동이야."},
        {"role": "assistant", "content": "네, 홍길동님으로 기억하겠습니다."},
        {"role": "user", "content": "리스트 뒤집는 법?"},
        {"role": "assistant", "content": "lst[::-1] 입니다."},
    ])
    stub = _stub(tmp_path, "sess-A")
    msgs = NexusREPL._load_resume_messages(stub)

    assert len(msgs) == 4
    roles = [m.role if isinstance(m.role, str) else m.role.value for m in msgs]
    assert roles == ["user", "assistant", "user", "assistant"]
    first = msgs[0].text_content if hasattr(msgs[0], "text_content") else str(msgs[0].content)
    assert "홍길동" in first


def test_resume_inherits_session_id(tmp_path):
    """복원 성공 시 새 발화가 같은 트랜스크립트에 이어지도록 세션 ID를 승계한다."""
    _write_transcript(tmp_path, "sess-B", [{"role": "user", "content": "hi"}])
    stub = _stub(tmp_path, "sess-B", session_id="fresh-uuid")
    NexusREPL._load_resume_messages(stub)
    assert stub._state.session_id == "sess-B"


def test_resume_missing_session_falls_back(tmp_path):
    """없는 세션을 resume하면 빈 리스트 + resume 취소 + 새 세션 ID 유지."""
    stub = _stub(tmp_path, "does-not-exist", session_id="keep-me")
    msgs = NexusREPL._load_resume_messages(stub)
    assert msgs == []
    assert stub._resume_session_id is None      # 이어받기 취소
    assert stub._state.session_id == "keep-me"  # 새 세션으로 시작


def test_no_resume_returns_empty(tmp_path):
    """resume 미지정이면 아무 것도 하지 않는다."""
    stub = _stub(tmp_path, None)
    assert NexusREPL._load_resume_messages(stub) == []


def test_resume_skips_system_and_empty_entries(tmp_path):
    """system·빈 content는 복원 대상이 아니다(user/assistant만)."""
    _write_transcript(tmp_path, "sess-C", [
        {"role": "system", "content": "부팅 로그"},
        {"role": "user", "content": ""},
        {"role": "user", "content": "실제 질문"},
        {"role": "assistant", "content": "실제 답변"},
    ])
    stub = _stub(tmp_path, "sess-C")
    msgs = NexusREPL._load_resume_messages(stub)
    assert len(msgs) == 2
    texts = [m.text_content if hasattr(m, "text_content") else str(m.content) for m in msgs]
    assert "실제 질문" in texts[0]
    assert "실제 답변" in texts[1]
