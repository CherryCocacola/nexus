# cli/formatters.py OutputFormatter thinking 필터 검증 — 정상 응답 삼킴 회귀 방지.
"""
OutputFormatter.format_text_delta의 thinking 필터를 검증한다.

배경(B-3): 과거 필터는 "먼저"·"분석"·"let me" 같은 접두어로 사고 시작을 추측했고,
이는 한국어 정상 응답("먼저 …")을 통째로 삼키는 심각한 버그였다. 게다가 상태가
스트림 간 리셋되지 않아 한 번 삼키면 다음 턴까지 연쇄로 삼켰다. 리터럴 <think>
태그 기반으로 교체하고 reset_stream_state를 도입했다. 이 테스트가 회귀를 막는다.
"""

from __future__ import annotations

from cli.formatters import OutputFormatter


def test_korean_meonjeo_passes_through():
    """'먼저 …'로 시작하는 정상 한국어 응답은 삼켜지지 않아야 한다(B-3 핵심)."""
    f = OutputFormatter()
    out = f.format_text_delta("먼저, config 파일을 확인하겠습니다.")
    assert out == "먼저, config 파일을 확인하겠습니다."


def test_english_analysis_prefix_passes_through():
    """'Let me …' 등 영어 접두어도 더 이상 사고로 오판하지 않는다."""
    f = OutputFormatter()
    assert f.format_text_delta("Let me check the files.") == "Let me check the files."
    g = OutputFormatter()
    assert g.format_text_delta("분석 결과는 다음과 같습니다.") == "분석 결과는 다음과 같습니다."


def test_literal_think_tag_is_filtered():
    """리터럴 <think>로 시작하면 사고로 간주해 화면에서 숨긴다."""
    f = OutputFormatter()
    assert f.format_text_delta("<think>사용자가 무엇을 원하는가") == ""
    # 사고 구간 지속 — 계속 숨김
    assert f.format_text_delta(" 더 생각한다") == ""
    # </think> 뒤의 본문은 통과
    out = f.format_text_delta("</think>정답은 42입니다.")
    assert "정답은 42입니다." in out


def test_reset_clears_swallowing_state():
    """사고 구간을 닫지 못한 채 끝나도, 리셋 후 다음 응답은 정상 통과해야 한다."""
    f = OutputFormatter()
    # 사고 진입 후 태그를 못 닫고 끝난 상황
    assert f.format_text_delta("<think>아직 생각 중") == ""
    assert f.format_text_delta(" 계속 생각") == ""
    # 리셋 없이 다음 응답을 주면 여전히 삼킴(문제 상황 재현)
    assert f.format_text_delta("이 응답은 삼켜진다") == ""
    # 리셋하면 다음 응답이 정상 통과
    f.reset_stream_state()
    assert f.format_text_delta("이제 정상 응답입니다.") == "이제 정상 응답입니다."


def test_reset_stream_state_idempotent():
    """새 포매터에 reset을 호출해도 안전하며, 이후 정상 통과한다."""
    f = OutputFormatter()
    f.reset_stream_state()
    assert f.format_text_delta("정상 텍스트") == "정상 텍스트"


def test_inline_think_block_single_delta():
    """한 델타 안에 <think>...</think>답변이 모두 있으면 답변만 통과한다."""
    f = OutputFormatter()
    out = f.format_text_delta("<think>고민</think>결론입니다")
    assert "결론입니다" in out
    assert "고민" not in out
