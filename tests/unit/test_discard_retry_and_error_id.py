# 붕괴 재생성 시 버려진 텍스트가 응답에 남지 않는지, 500 이 요청 ID 를 싣는지 검증한다.
"""
2026-08-26 실측 두 건을 고정한다.

■ ① 재생성한 뒤에도 절단본이 앞에 붙어 나갔다

    query_loop 은 생성 붕괴를 감지하면 재생성하면서 `STREAM_DISCARD_RETRY` 신호를
    낸다. 그 자리 주석이 "파이프 모드는 이 신호를 받으면 여태 모은 버퍼를 버려야
    한다"고 명시하고, 심지어 "비스트리밍 소비자는 아직 아무것도 못 봤으므로 완전히
    깨끗하게 교체된다"고 적혀 있었다. **그 문장이 사실이 아니었다** — 웹은 신호를
    듣지 않고 TEXT_DELTA 만 모으고 있었다.

        06:51:14  조기 절단 6,625자  →  재생성 255자  →  최종 content 6,877자
        07:42:11  조기 절단 6,015자  →  재생성 254자  →  최종 content 6,269자

    구조화 출력이면 그 순간 JSON 계약이 깨진다(실제로 두 건 다 그렇게 실패했다).
    구조화가 아니어도 사용자에게 붕괴한 6천 자가 그대로 보인다.

    CLI 는 이미 이 신호를 듣고 있었다(`cli/commands.py`). 웹만 빠져 있었다.

■ ② 500 응답에 요청 ID 가 없어 추적이 불가능했다

    3,011회 중 1건의 500 을 플러그인 팀이 관측했는데 본문이
    `Internal Server Error` 문자열뿐이라 서버 로그와 대조할 수 없었다.
"""

from __future__ import annotations

import json

from core.message import STREAM_DISCARD_RETRY, StreamEvent, StreamEventType


class _Ev:
    """StreamEvent 최소 스텁 — 판별에 쓰는 속성만 갖춘다."""

    def __init__(self, type_, text=None, error_code=None, message=""):
        self.type = type_
        self.text = text
        self.error_code = error_code
        self.message = message


def _fake_request(method: str, path: str, request_id: str | None):
    """Starlette Request 최소 대역 — 핸들러가 읽는 것만 갖춘다.

    `SimpleNamespace` 로 만든다. `request.url.path` 를 흉내내려면 중첩 속성이
    필요한데, 클래스로 쓰면 소문자 이름(`url`)이 명명 규약에 걸린다.
    """
    from types import SimpleNamespace

    headers = {"X-Request-ID": request_id} if request_id else {}
    return SimpleNamespace(
        method=method, headers=headers, url=SimpleNamespace(path=path)
    )


def _discard_signal() -> StreamEvent:
    """query_loop 이 실제로 내보내는 것과 같은 형태의 신호."""
    return StreamEvent(
        type=StreamEventType.SYSTEM_WARNING,
        error_code=STREAM_DISCARD_RETRY,
        message="[생성 붕괴 감지] 앞의 출력은 버리고 다시 생성합니다 (1/1)",
    )


# ── ① 폐기 신호 판별 ─────────────────────────────────────────


def test_discard_signal_is_recognized():
    """error_code 로 판별한다 — 문구는 바뀔 수 있다."""
    from web.app import _is_discard_retry

    assert _is_discard_retry(_discard_signal()) is True


def test_other_system_warnings_are_not_discard():
    """다른 경고까지 버퍼를 비우면 정상 응답이 사라진다."""
    from web.app import _is_discard_retry

    other = StreamEvent(
        type=StreamEventType.SYSTEM_WARNING, error_code="SOMETHING_ELSE", message="x"
    )
    assert _is_discard_retry(other) is False
    assert _is_discard_retry(_Ev(StreamEventType.TEXT_DELTA, text="본문")) is False


# ── ① 실측 재현 — 누적 규칙 ──────────────────────────────────


def test_truncated_text_is_dropped_on_regeneration():
    """★실측 재현★ 절단본 6,625자 + 재생성 255자 → 255자만 남아야 한다.

    소비 루프의 누적 규칙만 떼어내 검증한다(핸들러 전체를 띄우지 않는다).
    """
    from web.app import _is_discard_retry

    events = [
        _Ev(StreamEventType.TEXT_DELTA, text="붕괴" * 3312),  # 6,624자
        _discard_signal(),
        _Ev(StreamEventType.TEXT_DELTA, text='{"type":"chat_response"}'),
    ]

    parts: list[str] = []
    for ev in events:
        if getattr(ev, "type", None) == StreamEventType.TEXT_DELTA and ev.text:
            parts.append(ev.text)
        elif _is_discard_retry(ev):
            parts.clear()

    content = "".join(parts)
    assert content == '{"type":"chat_response"}', "절단본이 남았다"
    json.loads(content)  # 구조화 출력 계약이 지켜진다


def test_without_the_signal_the_bug_reproduces():
    """신호를 무시하면 실제로 깨진다는 것을 명시적으로 기록한다.

    이 테스트는 '고치기 전 동작'이다. 왜 신호를 들어야 하는지가 여기 있다.
    """
    parts = ["붕괴" * 10, '{"type":"chat_response"}']
    content = "".join(parts)

    try:
        json.loads(content)
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("절단본이 붙었는데도 JSON 이 유효하다 — 재현 실패")


# ── ② 500 응답 ───────────────────────────────────────────────


def test_internal_error_handler_carries_request_id():
    """500 본문에 request_id 와 오류 코드가 실려야 추적이 된다."""
    import asyncio

    from web.app import _handle_unexpected_error

    resp = asyncio.run(
        _handle_unexpected_error(
            _fake_request("POST", "/v1/chat/completions", "plugin-run-42"),
            RuntimeError("boom"),
        )
    )
    body = json.loads(resp.body)

    assert resp.status_code == 500
    assert body["error"]["request_id"] == "plugin-run-42"
    assert body["error"]["code"] == "INTERNAL_ERROR"
    assert body["error"]["type"] == "RuntimeError"
    assert resp.headers.get("X-Request-ID") == "plugin-run-42"


def test_internal_error_does_not_leak_stack_trace():
    """스택 트레이스는 응답에 싣지 않는다 — 내부 경로가 노출된다."""
    import asyncio

    from web.app import _handle_unexpected_error

    resp = asyncio.run(
        _handle_unexpected_error(
            _fake_request("GET", "/v1/models", None), RuntimeError("secret-path")
        )
    )
    raw = resp.body.decode("utf-8")

    assert "secret-path" not in raw, "예외 메시지가 그대로 나갔다"
    assert "Traceback" not in raw
    assert json.loads(raw)["error"]["request_id"] is None
