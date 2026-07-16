# LocalModelProvider._finalize_tool_calls의 tool_call arguments JSON 파싱 복구 단위 테스트
"""
tool_call arguments JSON 파싱 복구 단위 테스트 (2026-07-13).

배경 (DocumentExport "content는 비어 있을 수 없습니다" 버그):
  A.X-4.0(FP8) 모델이 DocumentExport 등 긴 content 문자열 값 안에 실제 개행/탭
  (제어문자 0x00–0x1F)을 이스케이프(\\n) 없이 그대로 넣어, 기본 json.loads
  (strict=True)가 "Invalid control character"로 거부 → 빈 dict 폴백 → content=""
  → 도구 스키마 검증 실패로 파일이 생성되지 않던 결함.

  실측: 해당 턴은 output_tokens=3532(<max 8192), stop_reason="tool_use"로 정상
  종료 → 절단이 아니라 제어문자 malformation임이 확인됨.

수정: json.loads 1차 실패 시 strict=False로 재파싱해 문자열 내부 제어문자를
  허용(복구). strict=False로도 실패하면(진짜 절단/구조 파손) 기존대로 {} 폴백.

이 테스트는 _finalize_tool_calls(staticmethod)에 누적 arguments를 직접 넣어
  세 경로(복구/폴백/정상)를 검증한다. 모델·GPU 서버는 호출하지 않는다.
"""
from __future__ import annotations

from core.message import StreamEventType
from core.model.inference import LocalModelProvider


def _stop_block(raw_args, name="DocumentExport"):
    """누적 arguments 문자열을 넣어 첫 TOOL_USE_STOP의 ToolUseBlock을 돌려준다."""
    accumulated = {0: {"id": "call_0", "function": {"name": name, "arguments": raw_args}}}
    events = LocalModelProvider._finalize_tool_calls(accumulated)
    stops = [e for e in events if e.type == StreamEventType.TOOL_USE_STOP]
    assert stops, "TOOL_USE_STOP 이벤트가 생성되어야 한다"
    return stops[0].tool_use


def _stop_input(raw_args, name="DocumentExport"):
    """첫 TOOL_USE_STOP 이벤트의 input(dict)만 돌려주는 편의 헬퍼."""
    return _stop_block(raw_args, name).input


def test_finalize_tool_calls_literal_control_chars_recovered_strict_false():
    """content 문자열 안에 이스케이프 안 된 실제 개행/탭이 있어도 strict=False로 복구된다."""
    # 모델이 실제로 내는 malformed 형태: 값 안에 리터럴 \n, \t
    raw = '{"content": "## 보고서\n\n- 항목1\t설명\n- 항목2", "format": "docx", "title": "AI 포털"}'
    block = _stop_block(raw)
    args = block.input

    # 빈 dict로 폴백되지 않고 실제 값이 살아 있어야 한다
    assert args.get("content"), "content가 비어 있으면 안 된다(복구 실패)"
    assert "항목1" in args["content"] and "항목2" in args["content"]
    assert args["format"] == "docx"
    assert args["title"] == "AI 포털"
    # 리터럴 개행이 그대로 문자열에 보존됐는지(이스케이프 \n과 동일 결과)
    assert "\n" in args["content"]
    # 복구 성공 → parse_error 신호는 서지 않는다
    assert block.parse_error is False


def test_finalize_tool_calls_truncated_json_falls_back_to_empty_dict():
    """절단으로 문자열이 미종결된 경우 strict=False로도 실패 → 빈 dict 폴백."""
    # 문자열이 닫히지 않고 객체도 미완성 = 절단 시뮬레이션
    raw = '{"content": "## 보고서\\n긴 본문이 중간에서 잘림'
    block = _stop_block(raw)

    # 절단은 복구 대상이 아니다 → {} 폴백 + parse_error=True(상위 guided 재시도 트리거)
    assert block.input == {}
    assert block.parse_error is True


def test_finalize_tool_calls_valid_json_unchanged():
    """정상 JSON은 1차(strict=True) 경로에서 그대로 파싱된다(기존 동작 무변경)."""
    raw = '{"content": "정상 본문", "format": "md"}'
    block = _stop_block(raw)

    assert block.input == {"content": "정상 본문", "format": "md"}
    assert block.parse_error is False
