# 입력 토큰을 '추정'에서 '세기'로 바꾼 배선을 고정한다
"""
2026-08-14. 종전 추정식은 `총글자수 // 3` 하나였다. A.X-4.0 토크나이저로 실측하니
문자 종류에 따라 100배 넘게 벌어진다(아래 표는 vLLM /tokenize 실측값).

  이 파일이 지키는 두 가지
    ① 폴백 추정기는 **어떤 표본에서도 과소추정하지 않는다.**
       과소추정하면 컨텍스트를 넘겨 vLLM 이 요청을 거부한다(긴급 압축 → 지연 + 손실).
       과대추정은 일찍 자를 뿐이라 훨씬 낫다.
    ② 정확히 세는 경로가 실패해도 턴이 죽지 않는다(폴백으로 넘어간다).
"""

from __future__ import annotations

from typing import Any

import pytest

from core.message import Message
from core.model.token_estimate import estimate_prompt_tokens, estimate_tokens
from core.orchestrator.query_loop import EXACT_COUNT_MIN_CHARS, _measure_input_tokens

# vLLM /tokenize 로 잰 실제 토큰 수. (표본 문자열, 실제 토큰)
# 이 값들은 측정 결과이지 계산값이 아니다 — 바꾸려면 다시 재야 한다.
MEASURED: dict[str, tuple[str, int]] = {
    "한글(공백없음)": ("가나다라마바사아자차카타파하각난달람" * 1200, 16800),
    "한글(문서체)": (
        (
            "이 문단은 공공기관 정보화사업 제안요청서의 본문 일부이며 "
            "사업 범위와 산출물 기준을 규정한다. "
        )
        * 250,
        5251,
    ),
    "한글(대화체)": ("어제 회의에서 나온 얘기를 정리해서 보냈어요 확인 부탁드립니다 " * 400, 3201),
    "영문소문자": ("the quick brown fox jumps over the lazy dog again and again " * 400, 4801),
    "코드": (
        "def compute(values):\n    total = sum(values)\n    return total / len(values)\n" * 300,
        6900,
    ),
    "숫자": ("1234567890 " * 2000, 22000),
    "공백만": (" " * 24000, 188),
    "개행만": ("\n" * 24000, 1500),
    "한자": ("漢字文書處理規定適用範圍" * 1200, 19200),
    "기호": ("-=+*/#@$%^&()[]{}<>" * 1200, 15601),
}


# ─────────────────────────────────────────────
# ① 폴백 추정기 — 과소추정 금지
# ─────────────────────────────────────────────
@pytest.mark.parametrize("name", list(MEASURED))
def test_fallback_never_underestimates(name: str) -> None:
    """★가장 중요한 성질. 실측보다 적게 추정하면 컨텍스트를 넘긴다."""
    text, actual = MEASURED[name]
    text = text[:24000]

    estimated = estimate_tokens(text)

    assert estimated >= actual, (
        f"{name}: 추정 {estimated} < 실제 {actual} — 가중치를 올려야 한다."
    )


@pytest.mark.parametrize("name", list(MEASURED))
def test_fallback_is_not_absurdly_high(name: str) -> None:
    """반대 방향 — 지나치게 부풀면 폴백 상황에서 쓸데없이 많이 잘린다."""
    text, actual = MEASURED[name]
    text = text[:24000]

    ratio = estimate_tokens(text) / actual

    # 공백만 있는 입력은 절대량이 작아(188토큰) 배수가 커도 무해하므로 예외로 둔다.
    limit = 10.0 if name == "공백만" else 3.0
    assert ratio <= limit, f"{name}: {ratio:.2f}배 과대추정"


def test_old_formula_would_have_underestimated_korean_documents() -> None:
    """이 수정의 출발점을 고정한다 — 종전 `글자수 // 3` 은 문서체 한국어에서 모자랐다."""
    text, actual = MEASURED["한글(문서체)"]
    text = text[:24000]

    assert len(text) // 3 < actual, "옛 추정식이 과소추정하지 않으면 이 수정의 전제가 틀린 것"
    assert estimate_tokens(text) >= actual


def test_empty_text_is_zero() -> None:
    assert estimate_tokens("") == 0


def test_prompt_estimate_adds_per_message_overhead() -> None:
    """채팅 템플릿은 메시지마다 역할 표시·특수 토큰을 붙인다 — 메시지가 많으면 무시 못 한다."""
    one = estimate_prompt_tokens(["안녕하세요"])
    many = estimate_prompt_tokens(["안녕하세요"] * 10)

    assert many > one * 5


def test_prompt_estimate_counts_tools_and_system() -> None:
    base = estimate_prompt_tokens(["안녕"])
    with_extras = estimate_prompt_tokens(
        ["안녕"], system_prompt="너는 도우미다" * 50, tool_schema_texts=['{"name":"Read"}'] * 20
    )
    assert with_extras > base


# ─────────────────────────────────────────────
# ② 세기 경로 — 되도록 세고, 실패해도 안 죽는다
# ─────────────────────────────────────────────
class _Provider:
    """count_prompt_tokens 를 흉내내는 최소 프로바이더."""

    def __init__(self, result: Any = 1234, raises: bool = False) -> None:
        self.result = result
        self.raises = raises
        self.calls = 0

    async def count_prompt_tokens(
        self, messages: list, system_prompt: str, tools: Any = None
    ) -> Any:
        self.calls += 1
        if self.raises:
            raise RuntimeError("서버 없음")
        return self.result


def _long_messages() -> list[Message]:
    """EXACT_COUNT_MIN_CHARS 를 확실히 넘기는 메시지."""
    return [Message.user("가" * (EXACT_COUNT_MIN_CHARS + 100))]


async def _measure(provider: Any, messages: list[Message]) -> tuple[int, bool]:
    total_chars = sum(len(str(m.content)) for m in messages)
    return await _measure_input_tokens(
        model_provider=provider,
        api_messages=messages,
        system_prompt="",
        tool_schemas=[],
        tool_schema_texts=[],
        total_chars=total_chars,
    )


@pytest.mark.asyncio
async def test_exact_count_is_used_when_available() -> None:
    provider = _Provider(result=4321)

    tokens, is_exact = await _measure(provider, _long_messages())

    assert (tokens, is_exact) == (4321, True)
    assert provider.calls == 1


@pytest.mark.asyncio
async def test_falls_back_when_provider_has_no_counter() -> None:
    """다른 프로바이더·테스트 더미에는 이 메서드가 없다 — 그래도 돌아야 한다."""

    class _Bare:
        pass

    tokens, is_exact = await _measure(_Bare(), _long_messages())

    assert is_exact is False
    assert tokens > 0


@pytest.mark.asyncio
async def test_falls_back_when_counter_raises() -> None:
    """★세기 실패가 턴을 죽이면 진단 장치가 본 기능을 막는 꼴이 된다."""
    provider = _Provider(raises=True)

    tokens, is_exact = await _measure(provider, _long_messages())

    assert is_exact is False
    assert tokens > 0


@pytest.mark.asyncio
@pytest.mark.parametrize("bad", [None, 0, -5, "1234"])
async def test_falls_back_on_unusable_result(bad: Any) -> None:
    """0·음수·문자열을 그대로 믿으면 출력 예산 계산이 무너진다."""
    provider = _Provider(result=bad)

    tokens, is_exact = await _measure(provider, _long_messages())

    assert is_exact is False
    assert tokens > 0


# ─────────────────────────────────────────────
# ③ /tokenize 요청 형식 — 실서버에서 400 을 맞고 배운 것
# ─────────────────────────────────────────────
@pytest.mark.asyncio
async def test_tool_schemas_are_converted_to_openai_shape() -> None:
    """★내부 스키마를 그대로 보내면 vLLM 이 400 을 낸다(실측).

    stream() 과 같은 변환기를 써야 한다. 이 테스트가 없으면 세기가 조용히 실패하고
    폴백 추정으로 떨어지는데, 폴백은 과대추정이라 **입력이 필요 이상으로 잘린다**
    (실측: 20만 자 요청이 창의 절반만 쓰고 잘렸다).
    """
    from core.model.inference import LocalModelProvider

    provider = LocalModelProvider(base_url="http://127.0.0.1:18001", model_id="ax-4.0")
    captured: dict[str, Any] = {}

    class _Resp:
        status_code = 200

        @staticmethod
        def json() -> dict[str, int]:
            return {"count": 777}

    async def _fake_post(url: str, **kwargs: Any) -> Any:
        captured["url"] = url
        captured["payload"] = kwargs.get("json")
        return _Resp()

    provider._client.post = _fake_post  # type: ignore[method-assign]

    nexus_schema = {
        "name": "Read",
        "description": "파일을 읽는다",
        "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}},
    }
    count = await provider.count_prompt_tokens([Message.user("안녕")], "시스템", [nexus_schema])

    assert count == 777
    assert captured["url"].endswith("/tokenize")
    tool = captured["payload"]["tools"][0]
    assert tool["type"] == "function"
    assert tool["function"]["name"] == "Read"
    # 내부 이름(input_schema)이 아니라 OpenAI 이름(parameters)이어야 한다.
    assert "parameters" in tool["function"]
    assert "input_schema" not in tool["function"]


@pytest.mark.asyncio
async def test_non_200_returns_none_not_garbage() -> None:
    """400/500 을 받고도 숫자를 지어내면 예산 계산이 조용히 틀어진다."""
    from core.model.inference import LocalModelProvider

    provider = LocalModelProvider(base_url="http://127.0.0.1:18001", model_id="ax-4.0")

    class _Resp:
        status_code = 400

        @staticmethod
        def json() -> dict[str, int]:
            return {}

    async def _fake_post(url: str, **kwargs: Any) -> Any:
        return _Resp()

    provider._client.post = _fake_post  # type: ignore[method-assign]

    assert await provider.count_prompt_tokens([Message.user("안녕")], "", None) is None


@pytest.mark.asyncio
async def test_short_input_skips_the_round_trip() -> None:
    """짧은 대화는 컨텍스트를 위협하지 못하므로 왕복을 아낀다."""
    provider = _Provider(result=999)

    tokens, is_exact = await _measure(provider, [Message.user("안녕")])

    assert provider.calls == 0
    assert is_exact is False
    assert tokens > 0
