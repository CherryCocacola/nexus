# Bash 서버/백그라운드 시작 관측 교정(P1-a) 검증 — exit 0 오인 방지.
"""
`_background_start_advisory`를 검증한다.

배경(FABLE5 처방, 2026-07-22): 서버를 띄우는 명령은 종료 코드 0이어도 서비스
정상 기동을 보장하지 않는다. 실측에서 `uvicorn ... &`가 빈 출력·exit 0으로
돌아오자 모델이 "서버가 재시작되었습니다"라고 오판했다(실제는 즉사). 서버/
백그라운드 시작 명령에만 검증 유도 안내가 붙고, 일반 명령에는 안 붙는지 확인한다.
"""

from __future__ import annotations

from core.tools.implementations.bash_tool import _background_start_advisory


def _has_advisory(command: str) -> bool:
    return _background_start_advisory(command) is not None


# ── 대상: 서버/데몬 시작 명령 ────────────────────────────────────────────


def test_advisory_for_uvicorn():
    assert _has_advisory("python -m uvicorn app.main:app --reload --port 8000")


def test_advisory_for_uvicorn_backgrounded():
    assert _has_advisory("cd backend && python -m uvicorn app.main:app > log 2>&1 &")


def test_advisory_for_common_dev_servers():
    for cmd in (
        "gunicorn app:app",
        "npm run dev",
        "npm start",
        "pnpm dev",
        "yarn start",
        "python manage.py runserver",
        "flask run",
        "python -m http.server 8080",
        "vite",
        "next dev",
    ):
        assert _has_advisory(cmd), cmd


def test_advisory_for_trailing_ampersand():
    """서버 실행기가 아니어도 POSIX 백그라운드(&)면 성공 오인 위험이 있어 대상."""
    assert _has_advisory("./run-something.sh &")


# ── 비대상: 일반 명령 (오탐 방지) ────────────────────────────────────────


def test_no_advisory_for_listing():
    assert not _has_advisory("ls -la")
    assert not _has_advisory("find . -name '*.py'")


def test_no_advisory_for_git_and_build():
    for cmd in (
        "git status",
        "git commit -m 'x'",
        "cat pyproject.toml",
        "grep -r foo .",
        "python -m pytest tests/",
        "mkdir -p logs",
        "echo done && exit 0",  # && 는 백그라운드가 아니다
    ):
        assert not _has_advisory(cmd), cmd


def test_double_ampersand_is_not_background():
    """`&&`(연결)는 백그라운드가 아니므로 그 자체로는 대상이 아니다."""
    assert not _has_advisory("cd backend && ls")


# ── 안내 내용 ────────────────────────────────────────────────────────────


def test_advisory_text_urges_verification():
    text = _background_start_advisory("uvicorn app:app")
    assert text is not None
    assert "종료 코드 0" in text
    assert "검증" in text
    assert "단정하지" in text
