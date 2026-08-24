# Edit 공백 정규화 폴백의 들여쓰기 재정렬 — 파이썬 블록이 조용히 바뀌는 것을 막는다.
"""
2026-08-23 실측 사고: NOVA 가 data_loader.py 를 4회 편집(2회 fuzzy)한 뒤
`return orders` 가 `except` 블록 **안으로** 들어가 정상 경로가 None 을 반환했다.
로더 3개가 전부 같은 형태로 망가졌고, 프로그램은 에러 없이 빈 결과를 냈다.

구조적 원인: fuzzy 폴백은 **모델이 공백을 이미 틀렸을 때만** 발동하는데,
교체는 같은 모델이 쓴 new_string 의 공백을 100% 신뢰해 그대로 써 넣었다.
못 믿을 공백인 모집단을 골라내서 그 공백을 믿는 구조였다. 파이썬에서 들여쓰기는
의미이므로 블록이 바뀌고, 그런데도 **문법은 유효**해서 구문 검사도 통과한다.

수정: 첫 줄 기준 delta 로 정렬하되, 모델이 **바꾸지 않은 줄**의 들여쓰기가
파일 원본과 다르면 교체를 거부한다. 새로 추가된 줄은 delta 만 적용해 통과시켜
줄 수 불일치가 문제에서 빠지게 한다.
"""

from __future__ import annotations

from core.tools.implementations.edit_tool import realign_fuzzy_replacement

# 파일에 실제로 들어 있는 구간 — print 는 8칸(except 안), return 은 4칸(함수 레벨).
FILE_SEGMENT = "        print(e)\n    return items"


class TestRejectsSilentBlockMove:
    """★핵심 — 내용이 그대로인 줄의 들여쓰기가 바뀌면 거부한다."""

    def test_return_moved_into_except_is_rejected(self) -> None:
        """step6 사고 재현. return 이 4→8 로 밀려 except 안으로 들어간다."""
        aligned, mismatch = realign_fuzzy_replacement(
            FILE_SEGMENT,
            "        print(e)\n    return items",
            "        print(e)\n        return items",  # 모델이 8칸으로 써 보냄
        )
        assert aligned is None
        # 에러에 파일 원문을 실어 줘야 다음 시도에서 old_string 을 바로잡을 수 있다.
        assert mismatch == "    return items"

    def test_dedent_of_unchanged_line_is_rejected(self) -> None:
        """반대 방향(내어쓰기)도 마찬가지로 블록을 바꾼다."""
        aligned, _ = realign_fuzzy_replacement(
            "    if x:\n        do_it()",
            "    if x:\n        do_it()",
            "    if x:\n    do_it()",
        )
        assert aligned is None


class TestPreservesFallbackPurpose:
    """폴백의 존재 이유(A.X-4.0 공백 재현 실패 흡수)를 죽이면 안 된다."""

    def test_whitespace_only_mismatch_still_applies(self) -> None:
        """old_string 의 들여쓰기가 통째로 어긋난 흔한 케이스 — 계속 통과해야 한다."""
        aligned, _ = realign_fuzzy_replacement(
            FILE_SEGMENT,
            "print(e)\nreturn items",  # 모델이 들여쓰기를 전부 뺐다
            "print(e)\nreturn other",
        )
        assert aligned is not None
        # 파일 기준으로 delta 가 적용돼 첫 줄이 8칸으로 복원된다.
        assert aligned.startswith("        print(e)")

    def test_added_line_passes(self) -> None:
        """줄이 늘어나도 거부하지 않는다 — 대응 없는 줄은 delta 만 적용한다."""
        aligned, _ = realign_fuzzy_replacement(
            FILE_SEGMENT,
            "        print(e)\n    return items",
            "        print(e)\n        logger.warning(e)\n    return items",
        )
        assert aligned is not None
        assert "logger.warning(e)" in aligned

    def test_removed_line_passes(self) -> None:
        """줄이 줄어드는 편집도 통과한다."""
        aligned, _ = realign_fuzzy_replacement(
            FILE_SEGMENT, "        print(e)\n    return items", "    return items"
        )
        assert aligned is not None

    def test_changed_line_content_is_not_indentation_checked(self) -> None:
        """모델이 내용을 바꾼 줄은 의도를 알 수 없으므로 들여쓰기를 검사하지 않는다.

        (그 줄이 실제로 문법을 깨면 뒤따르는 구문 검사가 잡는다 — 이중 안전망.)
        """
        aligned, _ = realign_fuzzy_replacement(
            FILE_SEGMENT,
            "        print(e)\n    return items",
            "        print(e)\n    return transformed",
        )
        assert aligned is not None


class TestEdgeCases:
    def test_empty_new_string_returns_unchanged(self) -> None:
        aligned, _ = realign_fuzzy_replacement(FILE_SEGMENT, "print(e)", "")
        assert aligned == ""

    def test_single_line_replacement(self) -> None:
        aligned, _ = realign_fuzzy_replacement("    x = 1", "x = 1", "x = 2")
        assert aligned == "    x = 2"

    def test_trailing_newline_preserved(self) -> None:
        """끝 개행 구조가 사라지면 교체 위치의 줄 구성이 깨진다."""
        aligned, _ = realign_fuzzy_replacement("    a\n    b", "a\nb", "a\nc\n")
        assert aligned is not None
        assert aligned.endswith("\n")

    def test_blank_line_not_indented(self) -> None:
        """빈 줄에는 들여쓰기 개념이 없다 — 공백을 채워 넣으면 안 된다."""
        aligned, _ = realign_fuzzy_replacement(
            "    a\n\n    b", "a\n\nb", "a\n\nc"
        )
        assert aligned is not None
        assert "\n\n" in aligned
