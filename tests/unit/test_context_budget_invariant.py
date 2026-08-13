# 컨텍스트 총예산이 vLLM 창을 넘지 않는지 설정 파일 수준에서 고정한다
"""
2026-08-13. `max_context_tokens` 를 49152 → 53248 로 올렸다. 근거는 실서버 실측이다.

    입력 토큰 추정식은 `글자수 // 3`(0.3333 토큰/글자)이다. 60000자 차분 측정 결과
      문서체 한국어 0.3841 (추정 대비 1.152배 **과소**)  ← 최악
      일반 산문     0.2956 (0.89배)
      코드 0.2544 / 영어 0.1277

★첫 측정은 24000자로 재서 1.02배가 나왔고, 그 값으로 57344 를 잡았다가 실서버에서
  추정 48742 → 실제 56293(1.155배)이 관측돼 여유가 1047 토큰까지 떨어졌다. 표본이
  작으면 고정 오버헤드에 묻혀 실제 비율이 드러나지 않는다. 그래서 53248 로 물렸다.

이 파일은 **다음 사람이 숫자를 더 올릴 때 안전선을 넘는지 자동으로 걸리게** 한다.
주석으로만 적어 두면 다음 상향에서 그대로 반복된다(이 리포가 이미 겪은 방식이다 —
종전 주석은 max_context 를 입력 전용으로 오해해 창의 25%를 놀리고 있었다).

★계산은 query_loop 의 실제 식을 그대로 따른다.
    input_limit  = int(max_context * 0.85)              # truncate 임계
    dynamic_max  = max_context - estimated_input - 200   # 남은 공간
    max_tokens   = max(512, min(base_max_tokens, dynamic_max))
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

_REPO = Path(__file__).resolve().parents[2]

# 실측된 최악 과소추정 배수(문서체 한국어 1.152)에 8% 여유를 얹은 값.
# 다른 문자 구성이 섞여도 이보다 나빠지지 않는다는 가정을 명시적으로 둔다.
# ★이 값을 낮춰서 테스트를 통과시키지 말 것 — 재측정한 근거가 있을 때만 바꾼다.
_WORST_UNDERESTIMATE = 1.25

# query_loop 의 상수들 — 값이 바뀌면 이 테스트도 함께 실패해야 한다.
_TRUNCATE_RATIO = 0.85
_SAFETY_BUFFER = 200


def _load(path: str) -> dict:
    with (_REPO / path).open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _largest_output_ceiling(cfg: dict) -> int:
    """라우팅 모드 중 가장 큰 출력 상한 — 창을 가장 많이 쓰는 최악 조건이다."""
    modes = cfg["routing"].values()
    return max(m["max_tokens"] for m in modes if isinstance(m, dict) and "max_tokens" in m)


def test_query_loop_constants_match_this_test() -> None:
    """계산의 전제가 실제 코드와 어긋나면 이 테스트는 의미가 없다."""
    source = (_REPO / "core/orchestrator/query_loop.py").read_text(encoding="utf-8")

    assert "int(max_context * 0.85)" in source
    assert "max_context - estimated_input - 200" in source


def test_b200_context_budget_fits_the_vllm_window() -> None:
    """★핵심. 최악(전부 한국어) 조건에서도 vLLM 창을 넘지 않아야 한다."""
    cfg = _load("config/nexus_config.112.yaml")
    launch = _load("config/vllm_launch.yaml")

    max_context = cfg["model"]["max_context_tokens"]
    window = launch["profiles"]["axmodel"]["max_model_len"]
    base_max = _largest_output_ceiling(cfg)

    estimated_input = int(max_context * _TRUNCATE_RATIO)
    actual_input = estimated_input * _WORST_UNDERESTIMATE
    dynamic_max = max_context - estimated_input - _SAFETY_BUFFER
    max_tokens = max(512, min(base_max, dynamic_max))

    total = actual_input + max_tokens
    assert total <= window, (
        f"창 초과: 실제 입력 {actual_input:.0f} + 출력 {max_tokens} = {total:.0f} > {window}. "
        f"max_context_tokens({max_context})를 낮추거나 vLLM max_model_len 을 올려야 한다."
    )


def test_budget_is_not_left_unused() -> None:
    """반대 방향 — 너무 보수적이면 창이 놀아 긴 문서가 불필요하게 잘린다.

    이 테스트가 이번 변경의 이유다. 종전 49152 은 창의 25%를 쓰지 않았다.
    """
    cfg = _load("config/nexus_config.112.yaml")
    launch = _load("config/vllm_launch.yaml")

    max_context = cfg["model"]["max_context_tokens"]
    window = launch["profiles"]["axmodel"]["max_model_len"]

    assert max_context / window >= 0.80, (
        f"총예산 {max_context}이 창 {window}의 80% 미만이다 — 놀고 있는 공간이 크다."
    )


# 입력이 상한까지 찼을 때 보장할 출력 하한.
# 종전(49152)에는 7173 이었다. 설정 상한 8192 를 **전부** 쓰려면 max_context 가
# 55947 이상이어야 하는데, 그 값은 추정식 오차(최악 1.152배)를 흡수하지 못한다.
# 즉 "입력 상한에서도 8192 출력"은 지금 추정식으로는 안전하게 달성할 수 없다 —
# 창을 다 쓰려면 추정식을 먼저 고쳐야 한다(후속 과제).
_MIN_OUTPUT_AT_MAX_INPUT = 7500


def test_output_at_max_input_improved() -> None:
    """입력이 상한까지 차도 출력이 지나치게 깎이지 않아야 한다.

    종전(49152)에는 입력이 크면 dynamic_max 가 7173 으로 깎여, 설정이 8192 를
    허용해도 답이 더 일찍 잘렸다. 이것이 '문서 분석 → 보고서' 계열 잘림의 한 축이다.
    """
    cfg = _load("config/nexus_config.112.yaml")

    max_context = cfg["model"]["max_context_tokens"]
    base_max = _largest_output_ceiling(cfg)

    estimated_input = int(max_context * _TRUNCATE_RATIO)
    dynamic_max = max_context - estimated_input - _SAFETY_BUFFER
    effective = min(base_max, dynamic_max)

    assert effective >= _MIN_OUTPUT_AT_MAX_INPUT, (
        f"입력이 상한일 때 출력이 {effective}로 깎인다(직전 기준 7173, 목표 "
        f"{_MIN_OUTPUT_AT_MAX_INPUT} 이상)."
    )


@pytest.mark.parametrize("path", ["config/nexus_config.112.yaml"])
def test_context_value_is_a_positive_int(path: str) -> None:
    value = _load(path)["model"]["max_context_tokens"]
    assert isinstance(value, int) and value > 0
