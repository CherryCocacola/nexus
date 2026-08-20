# 배포된 웹·OpenAI API·VSCode(구조화 출력) 표면이 실제로 도는지 확인한다.
"""
표면 회귀 — HTTP.

이 스위트가 잡는 것 (전부 실제로 겪은 사고다):
    - 코딩 모델 전환 시 앵커 모델명을 실어 보내 **HTTP 404**(2026-08-20)
    - 그 404 를 "입력이 너무 길다"로 오진해 사용자가 입력만 줄이던 문제
    - 모델을 갈아끼우자 답변 산문이 통째로 영어가 되던 문제
    - `/v1/models` 가 서빙하지 않는 옛 모델명을 보고하던 문제(2026-08-16)

단위 테스트로는 못 잡는다 — 값이 아니라 **경로**가 문제이기 때문이다.
"""

from __future__ import annotations

import json

import pytest

from tests.e2e.conftest import korean_ratio, requires_server

pytestmark = [pytest.mark.e2e, requires_server]


# ─────────────────────────────────────────────
# 생존·메타
# ─────────────────────────────────────────────
def test_health_reports_gpu_connected(api):
    """health 는 GPU 백엔드 연결까지 봐야 의미가 있다(200 만으로 정상 판정 금지)."""
    r = api.get("/health", timeout=20)
    assert r.status_code == 200
    assert r.json().get("gpu_server") == "healthy"


def test_models_endpoint_reports_served_models(api, key_primary):
    """실제 서빙 중인 모델을 보고해야 한다 — 박아 둔 옛 이름이 남으면 연동이 깨진다."""
    r = api.get("/v1/models", key=key_primary, timeout=20)
    assert r.status_code == 200
    ids = [m.get("id") for m in r.json().get("models", [])]
    assert "ax-4.0" in ids, f"주모델이 목록에 없다: {ids}"


def test_web_tool_pool_is_exposed(api, key_primary):
    """웹 Worker 도구 풀이 비면 문서 생성·이미지 같은 기능이 통째로 죽는다."""
    r = api.get("/v1/tools", key=key_primary, timeout=20)
    assert r.status_code == 200
    assert len(r.json().get("tools", [])) >= 10


# ─────────────────────────────────────────────
# 인증
# ─────────────────────────────────────────────
@pytest.mark.parametrize("key", [None, "wrong-key"])
def test_requests_without_valid_key_are_rejected(api, key):
    """인증 미들웨어 생존 확인 — 배포 사고로 이게 꺼진 적이 있다."""
    r = api.post(
        "/v1/chat/completions",
        {"model": "ax-4.0", "messages": [{"role": "user", "content": "hi"}]},
        key=key,
        timeout=20,
    )
    assert r.status_code == 401


# ─────────────────────────────────────────────
# 생성 — 비스트림·스트리밍
# ─────────────────────────────────────────────
def test_completion_returns_content(api, key_primary):
    status, text = api.chat("한 문장으로 자기소개해줘", key_primary, max_tokens=100)
    assert status == 200, text
    assert len(text.strip()) > 5


def test_answer_language_follows_user(api, key_primary):
    """한국어로 물으면 한국어로 답해야 한다.

    프롬프트의 언어 지시가 '잡담' 섹션에만 있어서, 모델을 코딩 전용으로 갈아끼우자
    설명이 통째로 영어가 됐다(실측 한국어 0%). 표면에서 이걸 지킨다.
    """
    status, text = api.chat(
        "재고 관리 시스템이 무엇인지 두 문장으로 설명해줘", key_primary, max_tokens=150
    )
    assert status == 200, text
    assert korean_ratio(text) > 0.3, f"한국어 비율 {korean_ratio(text):.2f} / {text[:80]}"


def test_streaming_sends_chunks_and_done(api, key_primary):
    """스트리밍이 [DONE] 으로 끝나야 프론트가 완료를 인식한다."""
    import httpx

    chunks, done = 0, False
    with httpx.Client(timeout=150.0) as c, c.stream(
        "POST",
        api.base + "/v1/chat/completions",
        json={
            "model": "ax-4.0",
            "stream": True,
            "max_tokens": 60,
            "messages": [{"role": "user", "content": "안녕하세요"}],
        },
        headers={"Authorization": f"Bearer {key_primary}"},
    ) as resp:
        assert resp.status_code == 200
        for line in resp.iter_lines():
            s = line.strip()
            if s.startswith("data: "):
                chunks += 1
                if s.endswith("[DONE]"):
                    done = True
    assert chunks > 1 and done, f"청크 {chunks}개, [DONE]={done}"


def test_multi_turn_context_is_kept(api, key_primary):
    """앞 턴 내용을 이어받아야 한다 — 컨텍스트 조립이 깨지면 여기서 드러난다."""
    r = api.post(
        "/v1/chat/completions",
        {
            "model": "ax-4.0",
            "max_tokens": 80,
            "messages": [
                {"role": "user", "content": "내 이름은 서현석이야."},
                {"role": "assistant", "content": "네, 서현석님 반갑습니다."},
                {"role": "user", "content": "내 이름이 뭐라고 했지?"},
            ],
        },
        key=key_primary,
    )
    assert r.status_code == 200
    assert "서현석" in r.json()["choices"][0]["message"]["content"]


# ─────────────────────────────────────────────
# 연동 계약 — 도구 호출·구조화 출력·질의 클래스
# ─────────────────────────────────────────────
def test_function_calling_returns_tool_calls(api, key_primary):
    """AgentHub 등 외부 연동이 의존하는 계약이다."""
    r = api.post(
        "/v1/chat/completions",
        {
            "model": "ax-4.0",
            "max_tokens": 200,
            "temperature": 0.0,
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "도시의 날씨를 조회한다",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                            "required": ["city"],
                        },
                    },
                }
            ],
            "messages": [{"role": "user", "content": "서울 날씨 알려줘"}],
        },
        key=key_primary,
    )
    assert r.status_code == 200
    tool_calls = r.json()["choices"][0]["message"].get("tool_calls")
    assert tool_calls, "tool_calls 가 비었다 — 툴파서 배선을 확인할 것"
    assert tool_calls[0]["function"]["name"] == "get_weather"


def test_structured_output_returns_valid_json(api, key_primary):
    """VSCode 플러그인 경로 — 스키마를 어기면 플러그인이 편집을 적용하지 못한다."""
    schema = {
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
            "steps": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["summary", "steps"],
    }
    r = api.post(
        "/v1/chat/completions",
        {
            "model": "ax-4.0",
            "max_tokens": 400,
            "temperature": 0.2,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "plan", "strict": True, "schema": schema},
            },
            "messages": [
                {"role": "user", "content": "파이썬 파일에 함수를 추가하는 절차를 알려줘"}
            ],
        },
        key=key_primary,
    )
    assert r.status_code == 200
    parsed = json.loads(r.json()["choices"][0]["message"]["content"])
    assert parsed.get("summary") and isinstance(parsed.get("steps"), list)


@pytest.mark.parametrize("style", ["body", "header"])
def test_query_class_can_be_forced(api, key_second, style):
    """연동 측이 클래스를 고정할 수 있어야 한다 — 안 되면 코드 질문에 사내 문서가 주입된다."""
    payload = {
        "model": "ax-4.0",
        "max_tokens": 60,
        "messages": [{"role": "user", "content": "리스트 정렬 코드 알려줘"}],
    }
    headers = None
    if style == "body":
        payload["query_class"] = "TOOL"
    else:
        headers = {"X-Nexus-Query-Class": "TOOL"}
    r = api.post("/v1/chat/completions", payload, key=key_second, headers=headers)
    assert r.status_code == 200


# ─────────────────────────────────────────────
# 웹 UI 경로
# ─────────────────────────────────────────────
def test_web_chat_non_stream(api, key_primary):
    r = api.post("/v1/chat", {"message": "안녕하세요"}, key=key_primary)
    assert r.status_code == 200
    body = r.json()
    assert body.get("response") or body.get("message")


def test_web_chat_stream(api, key_primary):
    import httpx

    lines = 0
    with httpx.Client(timeout=150.0) as c, c.stream(
        "POST",
        api.base + "/v1/chat/stream",
        json={"message": "안녕"},
        headers={"Authorization": f"Bearer {key_primary}"},
    ) as resp:
        assert resp.status_code == 200
        for line in resp.iter_lines():
            if line.strip():
                lines += 1
    assert lines > 1


# ─────────────────────────────────────────────
# 오류 처리 — 사유를 뭉뚱그리지 않는가
# ─────────────────────────────────────────────
def test_model_error_is_not_reported_as_context_overflow(api, key_primary):
    """어떤 오류든 '입력이 너무 길다'로 보고하면 사용자가 엉뚱한 처방을 한다."""
    r = api.post(
        "/v1/chat/completions",
        {
            "model": "존재하지-않는-모델",
            "max_tokens": 30,
            "messages": [{"role": "user", "content": "hi"}],
        },
        key=key_primary,
    )
    assert "너무 길어" not in r.text


def test_malformed_request_is_rejected(api, key_primary):
    """messages 없는 요청은 4xx 여야 한다(500 이면 검증이 빠진 것)."""
    r = api.post("/v1/chat/completions", {"model": "ax-4.0"}, key=key_primary, timeout=30)
    assert 400 <= r.status_code < 500
