# Write 도구의 무변화 감지 — 같은 내용 반복 쓰기를 모델에게 알리는 in-band 신호.
"""
2026-08-23 실측 사고: 모델이 **똑같은 6321바이트를 11회** 다시 쓰며 30턴을
공전했다. 매번 "파일을 작성했습니다"가 성공으로 돌아오니, 모델 입장에서는
진전이 없다는 단서가 어디에도 없었다.

수정 방향은 오케스트레이터에 상태를 더하는 대신 **도구 결과 문구로 사실을
알리는 것**이다. pytest 반복 실행처럼 같은 호출이 정상인 워크플로와 충돌하지
않고, 모델이 다음 턴에 다른 행동을 고를 근거가 생긴다.

쓰기 자체는 막지 않는다 — 내용이 같으면 결과도 같아 해가 없고, 파일이 실제로
존재하게 만드는 것이 이 도구의 계약이다.
"""

from __future__ import annotations

import pytest

from core.tools.implementations.write_tool import WriteTool


async def _write(tool: WriteTool, path, content: str):
    return await tool.call({"file_path": str(path), "content": content}, None)


@pytest.mark.asyncio
async def test_new_file_is_not_marked_unchanged(tmp_path):
    """새 파일은 비교 대상이 없다."""
    result = await _write(WriteTool(), tmp_path / "a.txt", "내용")
    assert result.is_error is False
    assert result.metadata.get("unchanged") is False
    assert "변경 없음" not in result.data


@pytest.mark.asyncio
async def test_rewriting_identical_content_is_flagged(tmp_path):
    """★같은 내용을 다시 쓰면 결과 문구가 그 사실을 알린다."""
    tool = WriteTool()
    target = tmp_path / "b.txt"
    await _write(tool, target, "동일한 내용입니다\n")
    result = await _write(tool, target, "동일한 내용입니다\n")

    assert result.is_error is False
    assert result.metadata.get("unchanged") is True
    assert "변경 없음" in result.data
    # 행동 지시가 있어야 모델이 반복을 멈춘다("동일하다"만으론 또 쓴다).
    assert "다시 쓰지 마세요" in result.data


@pytest.mark.asyncio
async def test_changed_content_is_not_flagged(tmp_path):
    """내용이 달라졌으면 정상 쓰기다."""
    tool = WriteTool()
    target = tmp_path / "c.txt"
    await _write(tool, target, "처음 내용\n")
    result = await _write(tool, target, "바뀐 내용\n")

    assert result.metadata.get("unchanged") is False
    assert "변경 없음" not in result.data


@pytest.mark.asyncio
async def test_file_is_still_written_when_unchanged(tmp_path):
    """감지가 쓰기를 막으면 안 된다 — 파일 존재 보장이 이 도구의 계약이다."""
    tool = WriteTool()
    target = tmp_path / "d.txt"
    await _write(tool, target, "내용\n")
    target.unlink()  # 파일을 지운 뒤
    result = await _write(tool, target, "내용\n")  # 같은 내용을 다시 쓴다

    assert target.exists()
    assert target.read_text(encoding="utf-8") == "내용\n"
    assert result.metadata.get("unchanged") is False  # 없던 파일이므로 무변화 아님


@pytest.mark.asyncio
async def test_whitespace_difference_counts_as_changed(tmp_path):
    """공백 한 칸도 다르면 변경이다(정규화하지 않는다)."""
    tool = WriteTool()
    target = tmp_path / "e.txt"
    await _write(tool, target, "내용\n")
    result = await _write(tool, target, "내용 \n")
    assert result.metadata.get("unchanged") is False


@pytest.mark.asyncio
async def test_unreadable_existing_file_does_not_break_write(tmp_path):
    """기존 파일이 UTF-8 이 아니어도 쓰기가 실패하면 안 된다(비교는 안내용일 뿐)."""
    target = tmp_path / "f.bin"
    target.write_bytes(b"\xff\xfe\x00\x01")
    result = await _write(WriteTool(), target, "정상 텍스트\n")

    assert result.is_error is False
    assert result.metadata.get("unchanged") is False
    assert target.read_text(encoding="utf-8") == "정상 텍스트\n"
