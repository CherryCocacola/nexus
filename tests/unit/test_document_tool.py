# DocumentProcess 도구의 통짜 반환·최소 분할·캐시 무효화 동작을 검증하는 단위 테스트.
"""DocumentProcess 통짜/청크 전략 단위 테스트 (대용량 문서 반복 붕괴 수정).

무엇을 지키나:
    - 문서 전체가 document_singleshot_chars 이하이면 1청크(통짜)로 반환하고,
      footer가 "다시 호출하지 마세요"로 재호출을 막는다.
    - 통짜 상한을 넘으면 document_chunk_size로 최소 분할하되, 중간 청크 footer는
      "진행 안내 문장 없이" 이어 읽으라고 지시한다(반복 내레이션 억제 이중 방어).
    - singleshot 옵션이 없거나 0이면 기존 청크 동작(CHUNK_SIZE 폴백)으로 무회귀.
    - 파일 변경(크기)·분할 파라미터 변경 시 캐시가 무효화돼 재분할된다.

테스트 전략(.claude/rules/testing.md 준수):
    - 파일시스템은 tmp_path fixture. 대부분은 .txt 평문으로 청킹 로직을 직접 검증한다
      (_extract_text가 미지원 확장자를 UTF-8로 읽으므로 바이너리 생성 없이 빠르다).
    - 형식 경로(docx)가 동일 분기를 타는지는 python-docx로 실제 파일 하나로 확인한다.
    - asyncio_mode="auto"라 async def test_* 는 데코레이터 없이 동작한다.
"""

from __future__ import annotations

from pathlib import Path

from core.tools.base import ToolUseContext
from core.tools.implementations.document_tool import DocumentProcessTool


def _text_of_len(n: int, line_len: int = 80) -> str:
    """줄바꿈이 포함된 대략 n글자짜리 한글 텍스트를 만든다.

    _split_chunks는 줄(\\n) 경계에서만 자르므로, 분할이 실제로 일어나려면 텍스트에
    개행이 있어야 한다(개행 없는 한 줄은 chunk_size를 넘어도 안 쪼개진다). 그래서
    line_len 길이의 줄을 여러 개 이어 붙인 뒤 정확히 n글자로 자른다.
    """
    unit = "가나다라마바사아자차카타파하"  # 14자
    base = (unit * (line_len // len(unit) + 1))[:line_len]
    lines: list[str] = []
    total = 0
    while total < n:
        lines.append(base)
        total += len(base) + 1  # +1은 join 시 붙는 개행 몫
    return "\n".join(lines)[:n]


def _write_txt(tmp_path: Path, text: str, name: str = "doc.txt") -> str:
    """tmp_path에 텍스트 파일을 쓰고 그 경로 문자열을 돌려준다."""
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return str(p)


def _ctx(singleshot: int | None = None, chunk: int | None = None) -> ToolUseContext:
    """singleshot/chunk 예산을 옵션으로 주입한 도구 컨텍스트를 만든다."""
    options: dict = {}
    if singleshot is not None:
        options["document_singleshot_chars"] = singleshot
    if chunk is not None:
        options["document_chunk_size"] = chunk
    return ToolUseContext(cwd=".", options=options)


async def _run(path: str, ctx: ToolUseContext, chunk_index: int | None = None):
    """DocumentProcess를 한 번 호출해 ToolResult를 돌려주는 헬퍼."""
    tool = DocumentProcessTool()
    inp: dict = {"file_path": path}
    if chunk_index is not None:
        inp["chunk_index"] = chunk_index
    return await tool.call(inp, ctx)


# ─────────────────────────────────────────────
# 통짜 반환 (singleshot)
# ─────────────────────────────────────────────


async def test_document_process_within_singleshot_returns_whole_doc(tmp_path: Path):
    """문서가 통짜 상한 이하이면 1청크로 전문을 반환하고 재호출 금지를 안내한다."""
    path = _write_txt(tmp_path, _text_of_len(39000))
    result = await _run(path, _ctx(singleshot=40000, chunk=26000))

    assert not result.is_error
    assert result.metadata["total_chunks"] == 1
    # footer가 "다시 호출하지 마세요"로 반복 호출을 차단해야 한다.
    assert "다시 호출하지 마세요" in result.data
    # "다음 청크" 재호출 유도 문구는 없어야 한다(통짜이므로).
    assert "다음 청크" not in result.data


async def test_document_process_singleshot_boundary_exact_returns_whole(tmp_path: Path):
    """전체 길이가 통짜 상한과 정확히 같으면(경계) 여전히 통짜 반환이다."""
    text = _text_of_len(40000)
    path = _write_txt(tmp_path, text)
    result = await _run(path, _ctx(singleshot=40000, chunk=26000))

    assert not result.is_error
    assert result.metadata["total_chunks"] == 1


# ─────────────────────────────────────────────
# 최소 분할 (통짜 상한 초과)
# ─────────────────────────────────────────────


async def test_document_process_over_singleshot_splits_by_chunk_size(tmp_path: Path):
    """통짜 상한을 넘으면 chunk_size로 분할하되 청크 수가 소수여야 한다(8000자 시절 대비)."""
    path = _write_txt(tmp_path, _text_of_len(76000))
    result = await _run(path, _ctx(singleshot=40000, chunk=26000))

    assert not result.is_error
    total = result.metadata["total_chunks"]
    # 분할은 일어나되(≥2), 26000자 기준이라 소수(≤4)여야 한다.
    # (같은 문서를 8000자로 쪼개면 10개 안팎 — 그 반복 붕괴를 없애는 것이 목표.)
    assert 2 <= total <= 4, f"total_chunks={total} (기대 2~4)"


async def test_document_process_chunk_footer_suppresses_narration(tmp_path: Path):
    """중간 청크 footer는 청크마다 진행 안내를 반복하지 말라고 지시해야 한다."""
    path = _write_txt(tmp_path, _text_of_len(76000))
    result = await _run(path, _ctx(singleshot=40000, chunk=26000), chunk_index=0)

    assert not result.is_error
    assert result.metadata["total_chunks"] >= 2
    # footer에 반복 내레이션 억제 지시가 내장돼 있어야 한다(프롬프트와 이중 방어).
    assert "진행 안내 문장 없이" in result.data


# ─────────────────────────────────────────────
# 무회귀 — singleshot 비활성(0/미주입)
# ─────────────────────────────────────────────


async def test_document_process_singleshot_zero_keeps_legacy_chunking(tmp_path: Path):
    """singleshot 미주입이면 기존 청크 동작(CHUNK_SIZE 폴백)으로 쪼개진다(무회귀)."""
    # 옵션을 전혀 주지 않으면 chunk_size=CHUNK_SIZE(2500), singleshot=0(비활성).
    path = _write_txt(tmp_path, _text_of_len(10000))
    result = await _run(path, _ctx())  # 옵션 없음

    assert not result.is_error
    # 10000자 / 2500 → 여러 청크로 쪼개져야 한다(통짜 아님).
    assert result.metadata["total_chunks"] > 1


async def test_document_process_singleshot_zero_over_chunk_still_splits(tmp_path: Path):
    """singleshot=0이면 통짜 상한이 없으므로 chunk_size 초과 문서는 분할된다."""
    path = _write_txt(tmp_path, _text_of_len(60000))
    result = await _run(path, _ctx(singleshot=0, chunk=26000))

    assert not result.is_error
    assert result.metadata["total_chunks"] > 1


# ─────────────────────────────────────────────
# 캐시 무효화
# ─────────────────────────────────────────────


async def test_document_process_cache_invalidated_on_file_change(tmp_path: Path):
    """같은 경로에 크기가 다른 파일을 다시 쓰면 낡은 분할을 재사용하지 않는다."""
    path = _write_txt(tmp_path, _text_of_len(10000), name="same.txt")
    r1 = await _run(path, _ctx(singleshot=40000, chunk=26000))
    assert r1.metadata["total_chunks"] == 1  # 10000 ≤ 40000 → 통짜

    # 같은 경로에 통짜 상한을 넘는 더 큰 내용을 덮어쓴다(크기 변경 → 캐시 키 변경).
    _write_txt(tmp_path, _text_of_len(76000), name="same.txt")
    r2 = await _run(path, _ctx(singleshot=40000, chunk=26000))
    assert r2.metadata["total_chunks"] >= 2, "파일 변경 후 재분할되지 않았다(캐시 오염)"


async def test_document_process_cache_invalidated_on_chunk_size_change(tmp_path: Path):
    """분할 파라미터(chunk_size)가 바뀌면 캐시 키가 달라져 재분할된다."""
    path = _write_txt(tmp_path, _text_of_len(60000), name="cfg.txt")
    # singleshot 비활성 + 큰 청크 → 소수 청크
    r_big = await _run(path, _ctx(singleshot=0, chunk=26000))
    # 같은 파일, 작은 청크 → 더 많은 청크(캐시가 무효화돼야 반영됨)
    r_small = await _run(path, _ctx(singleshot=0, chunk=6000))
    assert r_small.metadata["total_chunks"] > r_big.metadata["total_chunks"]


# ─────────────────────────────────────────────
# 형식 경로 (docx가 동일 분기를 타는지)
# ─────────────────────────────────────────────


async def test_document_process_docx_within_singleshot_returns_whole(tmp_path: Path):
    """docx 형식도 통짜 상한 이하이면 1청크로 반환된다(형식 무관 동일 분기)."""
    from docx import Document

    doc = Document()
    for i in range(30):
        doc.add_paragraph(f"[{i}] " + "강원대학교 AI 포털 도입 제안 상세 내용 문단입니다. " * 3)
    p = tmp_path / "sample.docx"
    doc.save(str(p))

    result = await _run(str(p), _ctx(singleshot=40000, chunk=26000))
    assert not result.is_error
    # 짧은 docx라 통짜 반환(1청크) + 재호출 금지 안내.
    assert result.metadata["total_chunks"] == 1
    assert "다시 호출하지 마세요" in result.data
