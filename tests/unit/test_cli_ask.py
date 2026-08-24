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


def test_ask_model_option_removed():
    """장식용이던 ask --model 옵션은 제거됐다(B-5). 넘기면 usage 에러로 거부된다."""
    result = CliRunner().invoke(cli, ["ask", "--model", "primary", "질문"])
    assert result.exit_code == 2  # click usage 에러
    assert "no such option" in result.output.lower()


# ─────────────────────────────────────────────
# 스트림 정합성 신호 (2026-08-23)
# ─────────────────────────────────────────────
# 배경(실측 사고): 생성 붕괴가 감지되면 엔진이 앞의 출력을 버리고 재생성한다.
# 재생성으로도 복구하지 못하면 **잘린 출력을 그대로** 내보내는데, 그 사실을 알리는
# 신호가 없어 `nexus ask` 가 잘린 답변을 exit 0 으로 내보냈다. 파이프·CI 는 성공으로
# 오인한다. 아래 테스트가 그 계약을 고정한다.
#
# ★가장 중요한 것은 "재시도는 실패가 아니다"이다. STREAM_DISCARD_RETRY 를 실패로
#   오배선하면 정상 복구가 CI 실패로 둔갑한다.


def _discard_retry(msg: str = "[생성 붕괴 감지] 다시 생성합니다") -> StreamEvent:
    from core.message import STREAM_DISCARD_RETRY

    return StreamEvent(
        type=StreamEventType.SYSTEM_WARNING,
        error_code=STREAM_DISCARD_RETRY,
        message=msg,
    )


def _truncated(msg: str = "[생성 붕괴] 잘린 출력을 그대로 사용합니다") -> StreamEvent:
    from core.message import STREAM_TRUNCATED

    return StreamEvent(
        type=StreamEventType.SYSTEM_WARNING,
        error_code=STREAM_TRUNCATED,
        message=msg,
    )


def test_truncated_signal_exits_one():
    """복구 실패로 잘린 답변이 나가면 실패로 처리한다 — 이게 없어서 exit 0 이었다."""
    result = _invoke_ask([_text("표를 만들다 잘린 본문"), _truncated()])
    assert result.exit_code == 1


def test_truncated_signal_reports_to_stderr():
    """경고는 stderr 로 — stdout(답변)을 오염시키면 파이프 소비자가 깨진다."""
    result = _invoke_ask([_text("본문"), _truncated()])
    assert "잘린 출력" in result.stderr
    assert "잘린 출력" not in result.stdout


def test_discard_retry_alone_still_exits_zero():
    """★재시도는 정상 복구 경로다. 실패로 치면 정상 실행이 CI 실패로 둔갑한다."""
    result = _invoke_ask([_text("버려질 본문"), _discard_retry(), _text("최종 답변입니다.")])
    assert result.exit_code == 0


def test_discard_retry_drops_earlier_output_from_stdout():
    """폐기된 절단본이 stdout 에 남으면 재생성본과 이어붙어 한 답변처럼 보인다."""
    result = _invoke_ask(
        [_text("버려질 절단본입니다."), _discard_retry(), _text("최종 답변입니다.")]
    )
    assert "최종 답변입니다." in result.stdout
    assert "버려질 절단본" not in result.stdout


def test_timeout_retry_success_exits_zero():
    """★오배선 방지 핵심 — 타임아웃 재시도 후 성공한 실행은 성공이어야 한다."""
    result = _invoke_ask(
        [
            _text("여기까지 쓰다 타임아웃"),
            _discard_retry("[스트림 idle 타임아웃] 재시도 1/3"),
            _text("재시도 후 완성된 답변."),
        ]
    )
    assert result.exit_code == 0
    assert "재시도 후 완성된 답변." in result.stdout
    assert "여기까지 쓰다 타임아웃" not in result.stdout


def test_retry_then_truncated_exits_one():
    """재시도했으나 끝내 복구 못 한 경우 — step5 에서 실제로 일어난 시퀀스다."""
    result = _invoke_ask(
        [_text("1차 절단본"), _discard_retry(), _text("2차도 잘림"), _truncated()]
    )
    assert result.exit_code == 1
    assert "1차 절단본" not in result.stdout


def test_plain_system_warning_without_code_is_ignored():
    """코드가 없는 일반 경고는 종료 코드에 영향을 주지 않는다(무회귀)."""
    result = _invoke_ask(
        [
            StreamEvent(type=StreamEventType.SYSTEM_WARNING, message="참고 사항"),
            _text("정상 답변입니다."),
        ]
    )
    assert result.exit_code == 0
