# 컨텍스트 총예산이 vLLM 창을 넘지 않는지 설정 파일 수준에서 고정한다
"""
2026-08-14. `max_context_tokens` 를 61440 으로 올렸다. 근거는 **추정을 세기로 바꾼 것**이다.

  이력
    49152  글자수//3 추정. max_context 를 입력 전용으로 오해해 창의 25%가 놀았다.
    57344  첫 측정(24000자)에서 오차 1.02배로 나와 올렸다가, 실서버에서 1.155배가
           관측돼 여유가 1047 토큰까지 떨어져 기각.
    53248  60000자 재측정(문서체 한국어 1.152배 과소)을 반영한 보수적 값.
    61440  query_loop 이 vLLM `/tokenize` 로 **실제 프롬프트를 센다**. 실서버 검증에서
           로그의 `44490 (실측)` 과 응답 `prompt_tokens 44490` 이 정확히 일치했다.
           추정 오차를 흡수할 안전 계수가 필요 없어졌다.

★안전성의 두 경로
  ① 세기 성공(기본) — 오차 0 이므로 `입력 상한 + 출력 상한 ≤ 창` 만 지키면 된다.
  ② 세기 실패(폴백) — 문자 종류별 추정으로 넘어간다. 폴백은 **과대추정**하도록
     맞춰져 있어(tests/unit/test_token_counting.py 가 고정) 실제보다 크게 잡는다.
     즉 더 일찍 자를 뿐 창을 넘지 않는다. 그래서 ① 의 부등식이 양쪽을 모두 덮는다.

이 파일은 다음 사람이 숫자를 더 올릴 때 자동으로 걸리게 한다. 주석으로만 적어 두면
다음 상향에서 그대로 반복된다 — 실제로 종전 주석이 그 방식으로 틀린 채 유지됐다.

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

# query_loop 의 상수들 — 값이 바뀌면 이 테스트도 함께 실패해야 한다.
_TRUNCATE_RATIO = 0.85
_SAFETY_BUFFER = 200

# 창 대비 최소 여유. 채팅 템플릿 변경 같은 드리프트를 흡수할 몫이다.
_MIN_WINDOW_HEADROOM = 2048


def _load(path: str) -> dict:
    with (_REPO / path).open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _largest_output_ceiling(cfg: dict) -> int:
    """라우팅 모드 중 가장 큰 출력 상한 — 창을 가장 많이 쓰는 최악 조건이다."""
    modes = cfg["routing"].values()
    return max(m["max_tokens"] for m in modes if isinstance(m, dict) and "max_tokens" in m)


def _worst_case_total(cfg: dict) -> tuple[int, int]:
    """(입력 상한에서의 실제 총 토큰, 그때의 출력 토큰)."""
    max_context = cfg["model"]["max_context_tokens"]
    base_max = _largest_output_ceiling(cfg)

    estimated_input = int(max_context * _TRUNCATE_RATIO)
    dynamic_max = max_context - estimated_input - _SAFETY_BUFFER
    max_tokens = max(512, min(base_max, dynamic_max))
    return estimated_input + max_tokens, max_tokens


def test_query_loop_constants_match_this_test() -> None:
    """계산의 전제가 실제 코드와 어긋나면 이 테스트는 의미가 없다."""
    source = (_REPO / "core/orchestrator/query_loop.py").read_text(encoding="utf-8")

    assert "int(max_context * 0.85)" in source
    assert "max_context - estimated_input - 200" in source


def test_b200_context_budget_fits_the_vllm_window() -> None:
    """★핵심. 입력이 상한까지 차고 출력을 최대로 써도 창을 넘지 않아야 한다."""
    cfg = _load("config/nexus_config.112.yaml")
    window = _load("config/vllm_launch.yaml")["profiles"]["axmodel"]["max_model_len"]

    total, _ = _worst_case_total(cfg)

    assert total <= window, (
        f"창 초과: 총 {total} > {window}. max_context_tokens 를 낮추거나 "
        f"vLLM max_model_len 을 올려야 한다."
    )


def test_window_headroom_is_kept() -> None:
    """세기가 실패해 폴백으로 떨어지는 경우와 템플릿 드리프트를 위한 몫."""
    cfg = _load("config/nexus_config.112.yaml")
    window = _load("config/vllm_launch.yaml")["profiles"]["axmodel"]["max_model_len"]

    total, _ = _worst_case_total(cfg)

    assert window - total >= _MIN_WINDOW_HEADROOM, (
        f"여유 {window - total} < {_MIN_WINDOW_HEADROOM}"
    )


def test_budget_is_not_left_unused() -> None:
    """반대 방향 — 너무 보수적이면 창이 놀아 긴 문서가 불필요하게 잘린다.

    49152 는 창의 25%를 쓰지 않았다. 그게 이 작업의 출발점이었다.
    """
    cfg = _load("config/nexus_config.112.yaml")
    window = _load("config/vllm_launch.yaml")["profiles"]["axmodel"]["max_model_len"]

    ratio = cfg["model"]["max_context_tokens"] / window

    assert ratio >= 0.90, f"총예산이 창의 {ratio:.0%} — 놀고 있는 공간이 크다."


def test_output_ceiling_is_reachable_at_max_input() -> None:
    """입력이 상한까지 차도 설정된 출력 상한을 그대로 낼 수 있어야 한다.

    49152 에서는 입력이 크면 출력이 7173 으로 깎여, 설정이 8192 를 허용해도 답이 더
    일찍 잘렸다. 이것이 '문서 분석 → 보고서' 계열 잘림의 한 축이었다.
    """
    cfg = _load("config/nexus_config.112.yaml")
    base_max = _largest_output_ceiling(cfg)

    _, max_tokens = _worst_case_total(cfg)

    assert max_tokens >= base_max, (
        f"입력이 상한일 때 출력이 {max_tokens}로 깎인다(설정 상한 {base_max})."
    )


@pytest.mark.parametrize("path", ["config/nexus_config.112.yaml"])
def test_context_value_is_a_positive_int(path: str) -> None:
    value = _load(path)["model"]["max_context_tokens"]
    assert isinstance(value, int) and value > 0
