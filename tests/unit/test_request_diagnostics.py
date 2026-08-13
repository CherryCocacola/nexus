# 요청 진단 배선 — 요청 ID 추적 · 422 사유 로깅 · 종료 사유 분리를 고정한다
"""
2026-08-13. VSCode 플러그인이 "요청 ID xxxx 가 실패했다"고 알려 왔는데 서버 로그에서
그 ID 를 찾을 수 없었다. 원인 셋을 각각 테스트로 고정한다.

  ① `X-Request-ID` 를 플러그인이 보내는데 서버가 읽지도 남기지도 않았다
  ② 422 는 접근 로그 한 줄만 남아 **어느 필드가 왜** 걸렸는지 알 수 없었다
  ③ 구조화 출력 JSON 이 깨진 것을 `content_filter` 로 신호해, 콘텐츠 정책에
     차단당했다는 오해를 만들었다(이 서버에는 콘텐츠 정책 필터가 없다)

외부 서비스는 건드리지 않는다 — TestClient 를 컨텍스트 매니저로 쓰지 않으므로
lifespan(부트스트랩)이 돌지 않고, 검증 실패는 라우트 진입 전에 처리된다.
"""

from __future__ import annotations

import logging

import pytest
from starlette.testclient import TestClient

from web.app import (
    _MAX_VALIDATION_INPUT_CHARS,
    FinishDetail,
    OpenAIChatCompletionResponse,
    OpenAIChoice,
    OpenAIResponseMessage,
    _sanitize_request_id,
    _summarize_validation_errors,
    app,
)
from web.middleware import CORSConfig


@pytest.fixture()
def client() -> TestClient:
    """부트스트랩 없이 앱에 요청을 넣는다(lifespan 미실행)."""
    return TestClient(app)


# ─────────────────────────────────────────────
# ① 요청 ID 정규화
# ─────────────────────────────────────────────
def test_request_id_keeps_uuid_form() -> None:
    """플러그인이 보내는 UUID 는 그대로 살아남아야 한다(추적의 전제)."""
    rid = "4e4c3a90-fbd0-4aaf-a788-fa615f132ceb"
    assert _sanitize_request_id(rid) == rid


def test_request_id_strips_log_forging_characters() -> None:
    """개행·제어문자·공백을 남기면 로그 한 줄을 위조당한다.

    한글 등 문자 자체는 isalnum() 이 True 라 살아남는다 — 그래도 개행이 없으면
    새 로그 줄을 만들 수 없으므로 위조는 성립하지 않는다.
    """
    out = _sanitize_request_id("abc\n[ERROR] 위조\r\n2026-01-01")

    assert "\n" not in out and "\r" not in out and " " not in out
    assert out.startswith("abc")


def test_request_id_is_length_capped() -> None:
    assert len(_sanitize_request_id("a" * 500)) == 64


@pytest.mark.parametrize("bad", [None, "", "   ", 12345, "!!!"])
def test_request_id_rejects_unusable_values(bad: object) -> None:
    """남는 문자가 없으면 None — 빈 문자열을 헤더로 되돌리지 않는다."""
    assert _sanitize_request_id(bad) is None


# ─────────────────────────────────────────────
# ② 검증 오류 요약 — 증폭 차단
# ─────────────────────────────────────────────
def test_validation_summary_truncates_long_input() -> None:
    """★핵심. 20MB base64 를 보냈다 실패하면 그게 통째로 로그·응답으로 돌아온다."""
    huge = "A" * 50_000
    out = _summarize_validation_errors(
        [{"loc": ("body", "messages", 0, "content"), "msg": "err", "type": "x", "input": huge}]
    )

    assert len(out[0]["input"]) <= _MAX_VALIDATION_INPUT_CHARS + 10
    assert out[0]["input"].endswith("(잘림)")
    # loc 는 JSON 직렬화가 항상 되도록 문자열로 정규화한다(배열 인덱스도 "0").
    assert out[0]["loc"] == ["body", "messages", "0", "content"]


def test_validation_summary_does_not_repr_containers() -> None:
    """dict/list 는 repr 하는 순간 그 크기만큼 메모리를 쓴다 — 타입 이름만 남긴다."""
    out = _summarize_validation_errors(
        [{"loc": ("body",), "msg": "err", "type": "x", "input": {"a": "B" * 50_000}}]
    )
    assert out[0]["input"] == "<dict>"


def test_validation_summary_keeps_scalars_readable() -> None:
    out = _summarize_validation_errors(
        [{"loc": ("body", "n"), "msg": "m", "type": "t", "input": 7}]
    )
    assert out[0]["input"] == "7"


def test_validation_summary_caps_error_count() -> None:
    errors = [{"loc": ("body", i), "msg": "m", "type": "t"} for i in range(100)]
    assert len(_summarize_validation_errors(errors)) == 20


def test_validation_summary_skips_malformed_entries() -> None:
    """오류 목록에 dict 가 아닌 것이 섞여도 요약 자체가 죽지 않는다."""
    assert _summarize_validation_errors(["문자열", None]) == []


# ─────────────────────────────────────────────
# ② 422 응답 — 사유가 로그에 남고 요청 ID 가 반향된다
# ─────────────────────────────────────────────
def test_invalid_body_logs_reason_and_echoes_request_id(
    client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    rid = "eb936c3a-2b52-45b3-8e77-9add996ccb53"
    with caplog.at_level(logging.WARNING, logger="nexus.web.app"):
        resp = client.post(
            "/v1/chat/completions",
            json={"messages": "배열이 아니다"},
            headers={"X-Request-ID": rid},
        )

    assert resp.status_code == 422
    # 요청 ID 로 로그를 grep 할 수 있어야 한다 — 이것이 이번 수정의 목적이다.
    assert rid in caplog.text
    assert "[요청검증실패]" in caplog.text
    # 클라이언트도 같은 ID 로 되돌려 받는다.
    assert resp.headers.get("X-Request-ID") == rid
    # 어느 필드가 걸렸는지 본문에 남는다.
    assert resp.json()["detail"][0]["loc"][:2] == ["body", "messages"]


def test_invalid_body_without_request_id_omits_header(client: TestClient) -> None:
    """헤더를 안 보낸 클라이언트에게 빈 X-Request-ID 를 돌려주지 않는다."""
    resp = client.post("/v1/chat/completions", json={"messages": "배열이 아니다"})

    assert resp.status_code == 422
    assert "X-Request-ID" not in resp.headers


def test_huge_input_is_not_echoed_back(client: TestClient) -> None:
    """422 응답이 요청 본문을 그대로 되돌리는 증폭을 막는다."""
    resp = client.post("/v1/chat/completions", json={"messages": "X" * 200_000})

    assert resp.status_code == 422
    assert len(resp.content) < 5_000


# ─────────────────────────────────────────────
# ③ 종료 사유 분리
# ─────────────────────────────────────────────
def _response(finish_reason: str, detail: FinishDetail | None) -> OpenAIChatCompletionResponse:
    return OpenAIChatCompletionResponse(
        id="chatcmpl-x",
        created=0,
        model="ax-4.0",
        choices=[
            OpenAIChoice(
                message=OpenAIResponseMessage(content="{"), finish_reason=finish_reason
            )
        ],
        finish_detail=detail,
    )


def test_normal_response_has_no_finish_detail() -> None:
    """무회귀 — 정상 종료면 필드가 None 이라 기존 응답과 달라지지 않는다."""
    assert _response("stop", None).finish_detail is None


def test_finish_detail_states_it_is_not_policy_blocking() -> None:
    """★오해의 원인을 응답 자체가 부인해야 한다."""
    detail = FinishDetail(
        code="INVALID_STRUCTURED_OUTPUT",
        message="콘텐츠 정책에 의한 차단이 아닙니다.",
    )
    dumped = _response("content_filter", detail).model_dump()

    assert dumped["choices"][0]["finish_reason"] == "content_filter"
    assert dumped["finish_detail"]["code"] == "INVALID_STRUCTURED_OUTPUT"
    assert "차단이 아닙니다" in dumped["finish_detail"]["message"]


def test_finish_reason_stays_within_openai_enum() -> None:
    """규격 밖 값을 넣으면 표준 SDK 가 Literal 검증에서 깨진다 — 값은 유지한다."""
    allowed = {"stop", "length", "tool_calls", "content_filter", "function_call"}
    import inspect

    import web.app as webapp

    source = inspect.getsource(webapp.chat_completions)
    for line in source.splitlines():
        stripped = line.strip()
        if stripped.startswith("finish_reason = ") and '"' in stripped:
            value = stripped.split('"')[1]
            assert value in allowed, f"규격 밖 finish_reason: {value}"


# ─────────────────────────────────────────────
# CORS — 되돌려준 헤더를 브라우저가 읽을 수 있어야 한다
# ─────────────────────────────────────────────
def test_cors_exposes_request_id() -> None:
    """expose 하지 않으면 서버가 실어 보내도 JS 에서 안 보인다."""
    assert "X-Request-ID" in CORSConfig.get_cors_kwargs()["expose_headers"]


def test_tenant_identifier_field_is_id_not_tenant_id() -> None:
    """[요청추적] 로그가 읽는 필드명을 고정한다.

    실측: 처음에 `tenant_id` 로 읽어 실서버 로그가 조용히 `tenant=-` 로만 찍혔다.
    getattr 기본값 때문에 **틀려도 예외가 안 나므로** 테스트로 못 박아야 한다.
    """
    from core.config import TenantConfig

    tenant = TenantConfig(id="coding")

    assert getattr(tenant, "id", None) == "coding"
    assert not hasattr(tenant, "tenant_id")
