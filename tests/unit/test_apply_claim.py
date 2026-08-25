# 적용 주장 대조 검증기 — 2026-08-25 실측 사고를 그대로 재현해 고정한다.
"""
사고 요약.
    VSCode 플러그인 회귀 측정에서 모델이 "변경이 성공적으로 적용되었습니다" 라고
    답했는데 파일은 한 줄도 안 바뀌었다. 그 턴에 모델이 받은 도구 결과는
    read_file · search_text 뿐이었고 둘 다 ok:true 였다.
    1,760 세션 중 43건(2.4%), 전부 코딩 전용 모델 라우팅 세션.

이 테스트가 고정하는 것.
    ① 읽기 도구만 성공했는데 적용을 주장하면 경고가 나온다  (사고 재현)
    ② 쓰기 도구가 성공했으면 경고가 없다                    (정상 흐름 무회귀)
    ③ 계획형 표현은 주장이 아니다                            (오탐 방지)
    ④ 파일 표지 없는 완료형은 잡지 않는다                    (오탐 방지)
    ⑤ 클라이언트 도구 결과는 user 메시지의 JSON 에서 읽는다  (이 표면의 구조)
"""

from __future__ import annotations

import json

from core.verification.apply_claim import (
    build_apply_claim_warning,
    collect_client_tool_results,
    find_apply_claims,
    has_write_evidence,
)


class _Msg:
    """엔진 메시지 최소 스텁 — 실제 Message 의 role/text_content 만 흉내낸다."""

    def __init__(self, role: str, content: str) -> None:
        self.role = role
        self.content = content
        self.text_content = content


def _tool_envelope(*pairs: tuple[str, bool]) -> str:
    """플러그인이 user 메시지에 실어 보내는 도구 결과 봉투를 만든다."""
    body = json.dumps(
        {
            "iteration": 1,
            "results": [
                {"id": f"call_{i}", "name": name, "ok": ok, "data": {}}
                for i, (name, ok) in enumerate(pairs, start=1)
            ],
        },
        ensure_ascii=False,
    )
    return "[AGENT_TOOL_RESULTS]\n아래 결과는 참고 데이터입니다.\n" + body


# ── ① 사고 재현 ──────────────────────────────────────────────


def test_read_only_tools_with_apply_claim_warns():
    """★실측 사고★ read_file 만 성공했는데 적용을 주장하면 경고해야 한다."""
    answer = (
        "변경이 성공적으로 적용되었습니다.\n"
        "- backend/services/koje_stats.py 파일의 aggregate_stats 함수 위에 "
        "한 줄 주석이 추가되었습니다."
    )
    messages = [_Msg("user", _tool_envelope(("read_file", True), ("search_text", True)))]

    claims = find_apply_claims(answer)
    results = collect_client_tool_results(messages)

    assert claims, "적용 주장을 못 찾았다"
    assert results == [("read_file", True), ("search_text", True)]
    assert has_write_evidence(results) is False
    warning = build_apply_claim_warning(claims, results)
    assert warning.startswith("APPLY_CLAIM_UNVERIFIED:")
    assert "read_file" in warning and "search_text" in warning


def test_no_tool_results_at_all_warns():
    """도구 결과가 아예 없는데 적용을 주장하는 경우도 잡는다(실측 42건 계열)."""
    answer = "요청하신 변경사항은 이미 적용되었습니다. app.py 파일을 수정했습니다."
    warning = build_apply_claim_warning(find_apply_claims(answer), [])
    assert warning.startswith("APPLY_CLAIM_UNVERIFIED:")
    assert "성공한 도구 실행 기록이 없습니다" in warning


# ── ② 정상 흐름 무회귀 ────────────────────────────────────────


def test_successful_write_tool_suppresses_warning():
    """쓰기 도구가 성공했으면 주장에 근거가 있으므로 경고하지 않는다."""
    answer = "변경이 적용되었습니다. src/main.py 파일을 수정했습니다."
    messages = [_Msg("user", _tool_envelope(("read_file", True), ("apply_edit", True)))]
    results = collect_client_tool_results(messages)

    assert has_write_evidence(results) is True
    assert build_apply_claim_warning(find_apply_claims(answer), results) == ""


def test_failed_write_tool_still_warns():
    """쓰기를 시도했지만 실패했다면 "했다"의 근거가 못 된다."""
    answer = "변경이 적용되었습니다. src/main.py 파일을 수정했습니다."
    messages = [_Msg("user", _tool_envelope(("apply_edit", False)))]
    results = collect_client_tool_results(messages)

    assert has_write_evidence(results) is False
    assert build_apply_claim_warning(find_apply_claims(answer), results) != ""


# ── ③④ 오탐 방지 ─────────────────────────────────────────────


def test_planned_action_is_not_a_claim():
    """계획형은 거짓 주장이 아니다 — 아직 안 했다고 말하고 있다."""
    answer = "app.py 파일에 주석을 추가하겠습니다. 적용하려면 승인해 주세요."
    assert find_apply_claims(answer) == []


def test_completion_without_file_marker_is_ignored():
    """파일을 지목하지 않은 완료형은 대화형 산출일 수 있어 잡지 않는다."""
    answer = "요청하신 설명을 추가했습니다. 궁금한 점이 더 있으면 말씀해 주세요."
    assert find_apply_claims(answer) == []


def test_claim_inside_code_block_is_ignored():
    """코드 블록 안 문자열은 모델의 주장이 아니다."""
    answer = '```python\nprint("main.py 파일이 수정되었습니다")\n```\n확인해 보세요.'
    assert find_apply_claims(answer) == []


# ── ⑤ 봉투 파싱 ──────────────────────────────────────────────


def test_tool_results_parsed_from_user_message_json():
    """이 표면은 tool_result 역할이 없다 — user 본문의 JSON 에서 읽어야 한다."""
    messages = [
        _Msg("assistant", "무시되어야 한다"),
        _Msg("user", _tool_envelope(("list_files", True), ("run_command", False))),
    ]
    assert collect_client_tool_results(messages) == [
        ("list_files", True),
        ("run_command", False),
    ]


def test_malformed_envelope_is_ignored():
    """형식이 깨져도 예외를 내지 않는다 — 진단이 응답을 막으면 안 된다."""
    assert collect_client_tool_results([_Msg("user", '{"results": [oops')]) == []
    assert collect_client_tool_results([_Msg("user", "그냥 평범한 질문입니다")]) == []


def test_read_hint_wins_over_write_hint():
    """`read_file_and_apply` 같은 이름에 속아 쓰기로 판정하면 안 된다."""
    assert has_write_evidence([("read_file_and_apply", True)]) is False
    assert has_write_evidence([("write_file", True)]) is True
