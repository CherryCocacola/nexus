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


def _tool_envelope_with_content(code: str) -> str:
    """read_file 이 코드 본문을 실어 오고, 같은 봉투에 쓰기 결과가 함께 있는 형태."""
    body = json.dumps(
        {
            "iteration": 1,
            "results": [
                {"id": "call_1", "name": "read_file", "ok": True, "data": {"content": code}},
                {"id": "call_2", "name": "apply_patch", "ok": True, "data": {}},
            ],
        },
        ensure_ascii=False,
    )
    return "[AGENT_TOOL_RESULTS]\n" + body


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


def test_braces_inside_string_literals_do_not_break_parsing():
    """★2026-08-26 리뷰 실측★ 문자열 안의 중괄호가 증거를 삼키면 안 된다.

    도구 결과는 본질적으로 코드다. `log.info("start {")` 같은 줄이 흔한데, 중괄호를
    문자 단위로 세던 시절에는 균형이 깨져 봉투 전체가 파싱되지 않았다.
    실측 27,710자에서 642ms 를 쓰고도 apply_patch 증거를 통째로 잃었다
    (= 정상 적용 턴에 오탐 + async 핸들러에서 이벤트 루프 정지).
    """
    code = 'log.info("start {")\n' * 400
    envelope = _tool_envelope_with_content(code)

    results = collect_client_tool_results([_Msg("user", envelope)])

    assert ("apply_patch", True) in results, "문자열 안 중괄호에 증거가 삼켜졌다"
    assert has_write_evidence(results) is True


def test_second_envelope_in_a_later_message_is_not_skipped():
    """앞 메시지에서 결과를 얻었다고 뒤 메시지를 건너뛰면 안 된다.

    이전 구현은 결과를 하나라도 얻으면 **메시지 순회 자체를** 끊어서, 뒤 메시지의
    쓰기 증거를 놓쳤다.
    """
    messages = [
        _Msg("user", _tool_envelope(("read_file", True))),
        _Msg("user", '{"note":"이전 단계 요약"}\n' + _tool_envelope(("apply_patch", True))),
    ]

    results = collect_client_tool_results(messages)

    assert ("apply_patch", True) in results
    assert has_write_evidence(results) is True


def test_pure_read_tools_are_never_write():
    """읽기 전용 도구는 쓰기로 잡히면 안 된다 — 이게 오탐 억제의 핵심이다."""
    for name in ("read_file", "search_text", "list_files", "get_symbols", "grep"):
        assert has_write_evidence([(name, True)]) is False, f"{name} 이 쓰기로 잡혔다"
    assert has_write_evidence([("write_file", True)]) is True


def test_write_verb_wins_in_compound_names():
    """복합어에서는 쓰기 동사가 조작을 결정한다("검색해서 치환한다"는 치환이다).

    예전에는 읽기가 이겼다. `read_file_and_apply` 같은 **가상의** 이름을 막으려던
    것인데, 그 대가로 실재하는 편집 도구 5개가 전부 읽기로 뒤집혔다. 가상보다
    실측을 택한다.
    """
    assert has_write_evidence([("search_replace", True)]) is True


def test_real_world_write_tool_names_are_not_flipped_to_read():
    """★2026-08-26 리뷰★ 실재하는 편집 도구 이름이 읽기로 뒤집히면 안 된다.

    부분 문자열 매칭이던 시절 아래가 전부 읽기로 분류됐다 — `search`·`find`·
    `symbol` 이 이름 안에 들어 있기 때문이다. 그러면 정상적으로 적용에 성공한
    턴에 "근거 없음" 오탐이 붙는다. 토큰 경계로 쪼개 판정한다.
    """
    for name in (
        "search_replace",        # Cursor 편집 도구
        "find_and_replace",
        "replace_symbol_body",   # 심볼 편집 계열
        "insert_after_symbol",
        "edit_symbol",
    ):
        assert has_write_evidence([(name, True)]) is True, f"{name} 이 읽기로 뒤집혔다"


def test_shell_execution_counts_as_write():
    """셸 실행 계열은 리다이렉션으로 파일을 만들 수 있어 쓰기로 본다.

    이름을 하나만 열거하면(`run_command`) 플러그인이 `run_in_terminal` 을 쓰는
    순간 조용히 빠져나간다. 어간으로 잡는다.
    """
    for name in ("run_command", "run_in_terminal", "execute_shell", "bash"):
        assert has_write_evidence([(name, True)]) is True, f"{name} 이 쓰기로 안 잡혔다"


def test_verify_tools_are_not_write():
    """검증 도구는 쓰기가 아니다 — 프로토콜 문서의 VERIFY 분류와 일치해야 한다."""
    for name in ("run_diagnostics", "run_tests"):
        assert has_write_evidence([(name, True)]) is False, f"{name} 이 쓰기로 잡혔다"


# ── 구조화 출력에서 인용문이 JSON 원문이 되면 안 된다 (2026-08-27) ──


def test_structured_answer_quotes_prose_not_json():
    """★L4 재현★ 한 줄 JSON 은 문장 분리가 안 돼 통째로 인용됐다.

        주장: "{"type":"chat_response","message":"backend/x.py 에 주석을 추가했…

    사람이 읽을 수 없고, 90자 상한에 걸려 정작 주장 문장이 잘려 나간다.
    """
    answer = json.dumps(
        {
            "type": "chat_response",
            "message": (
                "backend/services/koje_stats.py 파일의 aggregate_stats 함수에 "
                "주석을 추가했습니다. 확인해 주세요."
            ),
        },
        ensure_ascii=False,
    )

    claims = find_apply_claims(answer)

    assert claims, "구조화 출력에서 주장을 찾지 못했다"
    assert not claims[0].startswith("{"), f"JSON 원문이 인용됐다: {claims[0][:40]}"
    assert "aggregate_stats" in claims[0]

    warning = build_apply_claim_warning(claims, [("read_file", True)])
    assert '"type"' not in warning, "경고에 JSON 키가 새어 나왔다"
    assert "aggregate_stats" in warning


def test_nested_structured_fields_are_searched():
    """필드 이름을 열거하지 않는다 — 중첩 값도 본다.

    플러그인이 필드를 하나 바꾸면 조용히 빠져나가는 것을 막기 위함이다
    (도구 이름을 어간으로 잡은 것과 같은 이유).
    """
    answer = json.dumps(
        {
            "type": "chat_response",
            "payload": {"detail": ["src/main.py 파일을 수정했습니다."]},
        },
        ensure_ascii=False,
    )

    claims = find_apply_claims(answer)

    assert claims and "src/main.py" in claims[0]


def test_plain_text_answer_is_unchanged():
    """평문 답변(웹 UI 표면)은 종전과 똑같아야 한다(무회귀)."""
    assert find_apply_claims("src/a.py 파일에 주석을 추가했습니다.") == [
        "src/a.py 파일에 주석을 추가했습니다."
    ]


# ── 확장자 목록은 file_claim 과 공유한다 (2026-08-27) ──


def test_document_extensions_are_recognized():
    """★L5 재현★ apply_claim 에만 문서 확장자가 빠져 있었다.

    두 검증기가 같은 목적으로 각자 목록을 들고 갈라져, 한쪽에만 있는 확장자는
    그쪽에서만 잡히는 조용한 구멍이 됐다.
    """
    for name in ("report.xlsx", "deck.pptx", "spec.docx", "data.csv", "old.hwp"):
        assert find_apply_claims(f"{name} 를 수정했습니다."), f"{name} 이 안 잡혔다"


def test_code_extensions_are_recognized():
    """반대 방향도 본다 — file_claim 에 없던 확장자들."""
    from core.verification.file_claim import find_file_claims

    for name in ("App.jsx", "main.tsx", "Foo.java", "server.go", "lib.rs"):
        assert find_file_claims(f"{name} 파일을 작성했습니다."), f"{name} 이 안 잡혔다"


def test_both_verifiers_share_one_extension_list():
    """목록이 다시 갈라지지 않게 고정한다."""
    from core.verification import apply_claim, file_claim
    from core.verification._markers import PATH_OR_FILENAME

    assert PATH_OR_FILENAME in apply_claim._FILE_MARKER_RE.pattern
    assert PATH_OR_FILENAME in file_claim._FILE_MARKER_RE.pattern
