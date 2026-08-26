# 구조화 출력에서 경고가 본문을 깨뜨리지 않는지, 변경안 제출이 오탐되지 않는지 검증한다.
"""
2026-08-25 야간 9시간 회차(19,620 요청)에서 관측된 두 결함을 고정한다.

■ 결함 1 — 서버가 자기 JSON을 깨뜨렸다 (32건, finish_reason=stop)

    모델은 정상 JSON을 냈다. 그런데 web 계층이 유효성 검사 **직전에** 사후 검증
    경고를 마크다운으로 본문에 이어 붙였고, 그 결과 JSON이 깨졌다. 그리고 그
    깨진 결과를 바로 다음 줄에서 우리가 다시 검사해 "유효한 JSON이 아닙니다"로
    판정했다.

        모델 출력            {"type":"chat_response","content":[...]}   ← 파싱 성공
        + 실행주장 경고 160자  \n---\n⚠️ **실행 확인 필요** …
        = 검사 결과          Extra data: line 10 column 1               ← 파싱 실패

    실행 주장 검증기가 발화한 이유는 이 표면 특성이다 — 도구를 클라이언트가
    실행하므로 서버 메시지에 role="tool_result" 가 없고, 검증기는 "실행했다면서
    도구 기록이 없다"로 읽는다. 클라이언트는 손쓸 방법이 없었다.

■ 결함 2 — 변경안 제출을 완료 주장으로 오탐 (222건 중 153건, 69%)

    클라이언트는 변경을 final_proposal 로 받아 사용자 승인 뒤 로컬에서 적용한다.
    정상 흐름에서도 서버는 쓰기 도구 성공을 볼 수 없다. 제안은 완료 선언이
    아니므로 검사 대상에서 뺀다. 정밀도 31% → 99%.
"""

from __future__ import annotations

import json

from core.verification.apply_claim import is_change_proposal
from core.verification.post_check import build_answer_warnings, build_structured_warnings


class _Msg:
    def __init__(self, role: str, content: str) -> None:
        self.role = role
        self.content = content
        self.text_content = content


# ── 결함 1: 본문 파괴 재현 ───────────────────────────────────

_REAL_MODEL_OUTPUT = json.dumps(
    {
        "type": "chat_response",
        "content": [
            "변경이 성공적으로 적용되었습니다.",
            "",
            "- backend/services/koje_stats.py 파일에 json 모듈이 추가로 import되었습니다.",
            "- 검증 명령 python -c import backend.services.koje_stats 실행 결과 성공하였습니다.",
        ],
    },
    ensure_ascii=False,
    indent=2,
)


def test_model_output_itself_is_valid_json():
    """전제 확인 — 모델은 정상 JSON을 냈다. 문제는 그 뒤에 있었다."""
    json.loads(_REAL_MODEL_OUTPUT)


def test_appending_answer_warning_breaks_the_json():
    """★실측 재현★ 사후 경고를 본문에 붙이면 JSON이 깨진다.

    이 테스트는 '고치기 전 동작'을 기록한다. web 계층이 구조화 출력에서
    본문 부착을 하지 않는 이유가 여기 있다.
    """
    warning = build_answer_warnings(_REAL_MODEL_OUTPUT, [])
    assert warning.strip(), "실행 주장 경고가 발화해야 이 사고가 재현된다"

    try:
        json.loads(_REAL_MODEL_OUTPUT + warning)
    except ValueError as e:
        assert "Extra data" in str(e)
    else:  # pragma: no cover
        raise AssertionError("경고를 붙였는데도 JSON이 유효하다 — 재현 실패")


def test_warning_text_survives_as_one_line():
    """본문에 못 붙이는 대신 한 줄로 압축해 warnings 로 옮길 수 있어야 한다.

    신호를 버리는 것이 아니라 통로를 바꾸는 것이므로, 압축 후에도 핵심 문구가
    남아야 한다.
    """
    warning = build_answer_warnings(_REAL_MODEL_OUTPUT, [])
    one_line = " ".join(warning.split())[:400]

    assert "\n" not in one_line
    assert "실행" in one_line


# ── 결함 2: 변경안 제출 오탐 ─────────────────────────────────


def test_final_proposal_is_a_change_proposal():
    """final_proposal 은 완료 선언이 아니라 제안이다."""
    answer = json.dumps(
        {"type": "final_proposal", "operations": [{"path": "a.py"}]}, ensure_ascii=False
    )
    assert is_change_proposal(answer) is True


def test_chat_response_is_not_a_change_proposal():
    """chat_response 로 "적용했다"고 말하는 것은 여전히 검사 대상이다."""
    assert is_change_proposal(_REAL_MODEL_OUTPUT) is False


def test_plain_text_answer_is_not_treated_as_proposal():
    """평문 답변은 판별 불가 → 기존 검사를 그대로 받는다(무회귀)."""
    assert is_change_proposal("변경을 적용했습니다. app.py 파일을 수정했습니다.") is False


def test_final_proposal_suppresses_apply_claim_warning():
    """★실측 오탐 153건★ 제안에는 경고가 붙지 않아야 한다."""
    proposal = json.dumps(
        {
            "type": "final_proposal",
            "summary": "변경이 적용되었습니다. app.py 파일을 수정했습니다.",
        },
        ensure_ascii=False,
    )
    msgs = [_Msg("user", '[AGENT_TOOL_RESULTS]{"results":[{"name":"read_file","ok":true}]}')]

    assert build_structured_warnings(proposal, msgs) == []


def test_completion_claim_still_warns():
    """정탐은 그대로 잡아야 한다 — final_proposal 제외가 과하면 신호가 죽는다."""
    answer = json.dumps(
        {
            "type": "chat_response",
            "content": ["backend/services/koje_stats.py 파일의 변경이 성공적으로 적용되었습니다."],
        },
        ensure_ascii=False,
    )
    msgs = [_Msg("user", '[AGENT_TOOL_RESULTS]{"results":[{"name":"read_file","ok":true}]}')]
    warnings = build_structured_warnings(answer, msgs)

    assert warnings, "chat_response 의 완료 주장은 계속 잡혀야 한다"
    assert warnings[0].startswith("APPLY_CLAIM_UNVERIFIED:")


def test_claim_and_file_marker_must_share_a_sentence():
    """의도적으로 보수적이다 — 주장과 파일 표지가 다른 줄이면 잡지 않는다.

    _REAL_MODEL_OUTPUT 이 그 형태다("적용되었습니다"와 파일 경로가 다른 원소).
    잡지 못하는 것은 설계상 감수한 거짓 음성이다. 문장 경계를 넘어 결합하면
    "설명을 추가했습니다" 류의 대화형 완료가 전부 오탐이 된다.
    """
    msgs = [_Msg("user", '[AGENT_TOOL_RESULTS]{"results":[{"name":"read_file","ok":true}]}')]
    assert build_structured_warnings(_REAL_MODEL_OUTPUT, msgs) == []
