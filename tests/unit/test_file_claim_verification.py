# "파일에 작성했다"는 주장과 실제 쓰기 도구 실행 기록의 대조 검증.
"""
2026-08-23 실측 사고: NOVA 가 문서를 파일에 쓰지 않고 채팅에 출력한 뒤
"docs/TABLE_DESIGN.md 파일에 내용을 작성했습니다. 파일 존재를 확인했습니다."
라고 답했다. Write 0회, 검증 0회. 8회 시도 중 2회 발생(확률적 실패).

기존 execution_claim 이 못 잡은 이유가 둘이라 두 조건을 모두 고정한다.
  ① 파일 작성 주장 패턴 자체가 없었다 → 이 파일의 TestClaimDetection
  ② 도구 결과 **개수**만 봐서 Read 2건에 억제됐다 → TestSuccessfulWriteJoin
"""

from __future__ import annotations

from core.message import Message, Role, ToolUseBlock
from core.verification.file_claim import (
    build_file_claim_warning,
    collect_successful_write_tools,
    find_file_claims,
)


def _turn(tool_name: str, is_error: bool = False, tool_id: str = "t1") -> list[Message]:
    """도구 1회 호출 + 그 결과로 이뤄진 한 턴을 만든다."""
    return [
        Message(role=Role.ASSISTANT, content=[ToolUseBlock(id=tool_id, name=tool_name, input={})]),
        Message.tool_result(tool_use_id=tool_id, content="결과", is_error=is_error),
    ]


class TestClaimDetection:
    """주장 탐지 — 파일을 지목한 완료형만 잡는다."""

    def test_actual_step3_sentence_detected(self) -> None:
        """★실제 사고 문장. 이게 안 잡히면 이 검증기의 존재 이유가 없다."""
        claims = find_file_claims(
            "docs/TABLE_DESIGN.md 파일에 내용을 작성했습니다. 파일 존재를 확인했습니다."
        )
        assert claims

    def test_existence_check_claim_detected(self) -> None:
        """"파일 존재를 확인했습니다"는 별도 유형이다 — 하지도 않은 검증 주장."""
        assert find_file_claims("생성 후 파일 존재를 확인했습니다.")

    def test_extension_counts_as_file_marker(self) -> None:
        assert find_file_claims("report.docx 를 생성했습니다.")

    def test_chat_only_output_not_flagged(self) -> None:
        """★오탐 방지 핵심 — 채팅 본문에 문서를 쓰고 "작성했습니다"는 정상이다."""
        assert find_file_claims("요청하신 계획서를 작성했습니다. 위 내용을 참고하세요.") == []

    def test_future_tense_not_flagged(self) -> None:
        """아직 안 했다고 말하는 것은 거짓 주장이 아니다."""
        assert find_file_claims("docs/PLAN.md 파일에 작성하겠습니다.") == []

    def test_path_mention_without_claim_not_flagged(self) -> None:
        assert find_file_claims("core/bootstrap.py 를 살펴보면 라우팅이 있습니다.") == []

    def test_code_block_content_not_flagged(self) -> None:
        """모델이 **작성한 코드** 안의 문자열은 주장이 아니다(실서버 실측 오탐)."""
        answer = '```python\nprint("파일을 생성했습니다")\n```\n확인해 보세요.'
        assert find_file_claims(answer) == []

    def test_empty_answer(self) -> None:
        assert find_file_claims("") == []


class TestSuccessfulWriteJoin:
    """★시도가 아니라 성공을 센다 — 실패한 Write 를 성공으로 치면 거짓 음성이 된다."""

    def test_successful_write_counted(self) -> None:
        assert collect_successful_write_tools(_turn("Write")) == {"Write"}

    def test_failed_write_not_counted(self) -> None:
        """권한 거부·에러로 실패했는데 "작성했습니다"라면 정확히 잡아야 할 케이스다."""
        assert collect_successful_write_tools(_turn("Write", is_error=True)) == set()

    def test_read_only_tools_not_counted(self) -> None:
        """step3 이 정확히 이 상태였다 — Read·LS 만 있고 쓰기는 0건."""
        assert collect_successful_write_tools(_turn("Read")) == set()

    def test_bash_counts_as_write_capable(self) -> None:
        """Bash 로 파일을 만드는 것은 정상이다. 빼면 그 경우가 전부 오탐이 된다."""
        assert collect_successful_write_tools(_turn("Bash")) == {"Bash"}

    def test_document_export_counted(self) -> None:
        assert collect_successful_write_tools(_turn("DocumentExport")) == {"DocumentExport"}

    def test_empty_messages(self) -> None:
        assert collect_successful_write_tools([]) == set()

    def test_mixed_turn_counts_only_write(self) -> None:
        msgs = _turn("Read", tool_id="a") + _turn("Write", tool_id="b")
        assert collect_successful_write_tools(msgs) == {"Write"}


class TestWarningComposition:
    """경고 발화 조건 — 주장이 있고 성공한 쓰기가 없을 때만."""

    CLAIM = ["docs/X.md 파일에 작성했습니다."]

    def test_warns_when_claim_without_write(self) -> None:
        warning = build_file_claim_warning(self.CLAIM, set())
        assert warning
        assert "파일 확인 필요" in warning

    def test_silent_when_write_succeeded(self) -> None:
        assert build_file_claim_warning(self.CLAIM, {"Write"}) == ""

    def test_silent_when_no_claim(self) -> None:
        assert build_file_claim_warning([], set()) == ""

    def test_quotes_the_offending_sentence(self) -> None:
        """어느 문장이 문제인지 보여 줘야 사람이 판단할 수 있다."""
        assert "docs/X.md" in build_file_claim_warning(self.CLAIM, set())


class TestEndToEndThroughPostCheck:
    """실제 진입점(build_answer_warnings)까지 배선됐는지 확인한다."""

    def test_step3_scenario_produces_warning(self) -> None:
        """★사고 재현 — Read 만 하고 파일에 썼다고 주장한 턴."""
        from core.verification.post_check import build_answer_warnings

        warning = build_answer_warnings(
            "docs/TABLE_DESIGN.md 파일에 내용을 작성했습니다. 파일 존재를 확인했습니다.",
            _turn("Read"),
        )
        assert "파일 확인 필요" in warning

    def test_normal_write_turn_produces_no_file_warning(self) -> None:
        from core.verification.post_check import build_answer_warnings

        warning = build_answer_warnings(
            "docs/TABLE_DESIGN.md 파일에 내용을 작성했습니다.", _turn("Write")
        )
        assert "파일 확인 필요" not in warning
