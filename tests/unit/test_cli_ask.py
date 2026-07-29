# cli/commands.py ask 명령의 종료 코드·에러 정직화 검증 — 무출력 성공(exit0) 오인 방지.
"""
`nexus ask` 비대화형 명령의 종료 코드 계약을 검증한다.

배경(B-7): 과거 ask는 TEXT_DELTA만 stdout에 흘리고 ERROR 이벤트를 무시했다.
그 결과 GPU 서버가 죽어 답변이 한 글자도 안 나와도 종료 코드가 0(성공)이었다.
CI·A/B 자동화가 이 무출력 성공을 "정상"으로 오인하면 측정 전체가 오염된다.
이제 ERROR 이벤트는 stderr로 내보내고, 에러가 있었거나 답변이 전혀 없으면
종료 코드 1로 나간다. 이 테스트가 그 계약의 회귀를 막는다.
"""

from __future__ import annotations

from unittest.mock import patch

from click.testing import CliRunner

from cli.commands import cli
from core.message import StreamEvent, StreamEventType


async def _agen(events):
    """미리 준비한 StreamEvent 목록을 async generator로 흘려보낸다."""
    for e in events:
        yield e


class _MockEngine:
    """submit_message가 지정한 StreamEvent 시퀀스를 그대로 방출하는 가짜 엔진."""

    def __init__(self, events):
        self._events = events

    def submit_message(self, query):  # noqa: ARG002 — query는 테스트에서 무시
        return _agen(self._events)


def _invoke_ask(events):
    """부트스트랩을 가짜 엔진으로 대체하고 `nexus ask`를 실행해 결과를 돌려준다."""
    runner = CliRunner()  # click 8.2+ 는 stdout/stderr 를 기본 분리 캡처

    async def fake_init():
        return object()

    async def fake_init_phase2(state):  # noqa: ARG001
        return {"query_engine": _MockEngine(events)}

    with (
        patch("core.bootstrap.init", fake_init),
        patch("core.bootstrap.init_phase2", fake_init_phase2),
    ):
        return runner.invoke(cli, ["ask", "테스트 질문"])


def _text(t: str) -> StreamEvent:
    return StreamEvent(type=StreamEventType.TEXT_DELTA, text=t)


def _error(msg: str) -> StreamEvent:
    return StreamEvent(type=StreamEventType.ERROR, message=msg)


def test_ask_normal_text_exits_zero():
    """정상 답변(TEXT_DELTA)만 오면 종료 코드 0, 답변이 stdout에 나온다."""
    result = _invoke_ask([_text("정답은 "), _text("42입니다.")])
    assert result.exit_code == 0
    assert "42입니다." in result.stdout


def test_ask_error_event_exits_one():
    """스트림 도중 ERROR 이벤트가 있으면 종료 코드 1, 메시지는 stderr로 간다."""
    result = _invoke_ask([_error("GPU 서버 연결 실패")])
    assert result.exit_code == 1
    assert "GPU 서버 연결 실패" in result.stderr
    # 에러 메시지가 stdout(답변 채널)에 섞이면 파이프 파싱이 깨진다.
    assert "GPU 서버 연결 실패" not in result.stdout


def test_ask_empty_response_exits_one():
    """답변 텍스트가 하나도 없이 끝나면(빈 응답) 종료 코드 1."""
    result = _invoke_ask([])
    assert result.exit_code == 1


def test_ask_text_then_error_exits_one():
    """일부 답변이 나온 뒤라도 ERROR가 있었으면 성공으로 치지 않는다(exit 1)."""
    result = _invoke_ask([_text("부분 답변"), _error("중간에 연결 끊김")])
    assert result.exit_code == 1
    assert "중간에 연결 끊김" in result.stderr
