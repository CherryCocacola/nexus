# tool_call arguments 3단 폴백 파싱 — 특히 마크다운 이스케이프 혼입 복구 검증.
r"""
2026-08-23 실측: 테스트계획서 생성 중 tool_call arguments 파싱이 12회 실패했다.
분류하면 `Invalid \escape` 7 / `Expecting ',' delimiter` 3 / `Unterminated string` 2.

앞의 7건은 모델이 **마크다운 이스케이프**(`\_` `\.` `\*`)를 JSON 문자열 값 안에
그대로 쓴 것이다. 실제 실패 tail 이 `no\_line\_products\.csv` 였다. 파일명·경로에
밑줄과 점이 많은 문서 생성 도구에서 특히 잦다.

★뒤의 5건(문자열 경계 파손)은 **일부러 복구하지 않는다.** 경계를 추측해 고치면
조용히 다른 내용을 파일에 쓰게 된다. 빈 dict → 스키마 거부 → guided 재시도라는
기존 fail-closed 경로가 정답이다. 이 파일의 테스트가 그 경계를 고정한다.
"""

from __future__ import annotations

from core.model.inference import _repair_json_escapes, parse_tool_arguments

# 파이썬 리터럴에서 백슬래시를 다루기 번거로워 상수로 둔다.
BS = chr(92)


class TestRepairsInvalidEscapes:
    """복구 대상 — JSON 문법에 없는 이스케이프."""

    def test_markdown_underscore_and_dot(self) -> None:
        """실제 실패 tail 그대로. 경로의 밑줄·점 이스케이프."""
        args, failed = parse_tool_arguments(
            r'{"file_path": "no\_line\_products\.csv"}', "Write"
        )
        assert failed is False
        # 복구 결과는 모델이 쓴 바이트 그대로여야 한다(마크다운에선 이게 올바른 표기).
        assert args["file_path"] == "no" + BS + "_line" + BS + "_products" + BS + ".csv"

    def test_markdown_asterisk(self) -> None:
        args, failed = parse_tool_arguments(r'{"content": "제목 \*강조\* 끝"}', "Write")
        assert failed is False
        assert BS + "*강조" + BS + "*" in args["content"]

    def test_backtick_escape(self) -> None:
        args, failed = parse_tool_arguments(r'{"content": "\`코드\`"}', "Write")
        assert failed is False

    def test_unicode_escape_without_hex(self) -> None:
        """역슬래시+u 는 유효 집합에 있지만 hex 4자리가 없으면 역시 문법 위반이다."""
        args, failed = parse_tool_arguments('{"a": "' + BS + 'umm"}', "Write")
        assert failed is False
        assert args["a"] == BS + "umm"


class TestPreservesValidEscapes:
    """★복구가 멀쩡한 이스케이프를 망가뜨리면 안 된다."""

    def test_newline_escape_untouched(self) -> None:
        args, failed = parse_tool_arguments(r'{"content": "a\nb"}', "Write")
        assert failed is False
        assert args["content"] == "a\nb"  # 실제 개행 한 글자

    def test_escaped_backslash_untouched(self) -> None:
        """윈도우 경로. `\\` 를 건드리면 경로가 깨진다."""
        args, failed = parse_tool_arguments(r'{"p": "C:\Users\x"}', "Write")
        assert failed is False
        assert args["p"] == "C:" + BS + "Users" + BS + "x"

    def test_quote_escape_untouched(self) -> None:
        args, failed = parse_tool_arguments(r'{"q": "그는 \"안녕\" 이라 했다"}', "Write")
        assert failed is False
        assert args["q"] == '그는 "안녕" 이라 했다'

    def test_clean_json_unchanged_by_repair(self) -> None:
        """유효한 이스케이프만 든 JSON 은 복구기가 손대지 않아야 한다(멱등)."""
        clean = '{"a": "x' + BS + 'ny", "b": "C:' + BS + BS + 'p", "c": ' + BS + '"q' + BS + '"}'
        assert _repair_json_escapes(clean) == clean


class TestLenientPathStillWorks:
    """2단계(strict=False) 무회귀 — 제어문자 혼입 복구."""

    def test_raw_control_character_in_value(self) -> None:
        """A.X-4.0 가 긴 content 안에 실제 개행을 그대로 넣는 기존 실패 클래스."""
        args, failed = parse_tool_arguments('{"content": "첫줄\n둘째줄"}', "DocumentExport")
        assert failed is False
        assert args["content"] == "첫줄\n둘째줄"

    def test_escape_and_control_character_together(self) -> None:
        """두 결함이 한 인자에 공존할 수 있어 복구 후에도 strict=False 로 파싱한다."""
        args, failed = parse_tool_arguments('{"c": "a' + BS + '_b\n c"}', "Write")
        assert failed is False


class TestFailClosedOnStructuralDamage:
    """★경계가 깨진 것은 고치지 않는다 — 추측 복구가 조용한 오작성을 만든다."""

    def test_unterminated_string_stays_failed(self) -> None:
        """실제 실패 tail 형태. 값 안의 미이스케이프 따옴표로 문자열이 조기 종료됐다."""
        args, failed = parse_tool_arguments('{"content": "할 수 있습니다.""}', "Write")
        assert failed is True
        assert args == {}

    def test_missing_delimiter_stays_failed(self) -> None:
        args, failed = parse_tool_arguments('{"a": 1 "b": 2}', "Write")
        assert failed is True
        assert args == {}

    def test_truncated_json_stays_failed(self) -> None:
        """출력 한도로 잘린 경우 — 복구 대상이 아니다."""
        args, failed = parse_tool_arguments('{"content": "긴 문서가 여기서 잘렸', "Write")
        assert failed is True
        assert args == {}


class TestNormalPath:
    def test_valid_json_parses(self) -> None:
        assert parse_tool_arguments('{"a": 1}', "Write") == ({"a": 1}, False)

    def test_empty_arguments(self) -> None:
        assert parse_tool_arguments("{}", "TodoRead") == ({}, False)
