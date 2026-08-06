# HWP 5.0(.hwp) 네이티브 파서의 컨테이너 판별·레코드 해석·제어문자 처리 검증.
"""
core.ingest.parsers.hwp_native.HwpNativeParser 단위 테스트.

무엇을 지키나:
    - can_parse 는 확장자·OLE2 매직·FileHeader 서명을 모두 확인한다(fail-closed).
      OLE2 이지만 HWP 가 아닌 파일(.doc 등)은 거부해야 한다.
    - 압축(raw deflate)/비압축 본문 모두에서 문단 텍스트를 뽑는다.
    - **확장·인라인 제어문자는 8 코드유닛을 통째로 건너뛴다.** 이걸 놓치면 개체의
      이진 데이터가 한글 글자로 잘못 해독돼 문서가 통째로 깨진다 — 이 파서에서
      가장 중요한 불변식이라 함정 데이터로 직접 검증한다.
    - 암호/배포용 문서는 빈 트리 + 경고로 끝낸다(예외 아님, fail-soft).
    - 손상된 스트림에서 무한루프·예외 없이 멈춘다.

픽스처:
    tests/fixtures/sample_hwp5*.hwp 는 실제 OLE2 복합문서다(Windows 의
    StgCreateDocfile 로 생성해 리포에 커밋). 테스트 자체는 파일만 읽으므로
    플랫폼 의존성이 없다.
"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

import pytest

from core.ingest.parsers.hwp_native import HwpNativeParser

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
COMPRESSED = FIXTURES / "sample_hwp5.hwp"
UNCOMPRESSED = FIXTURES / "sample_hwp5_uncompressed.hwp"

# 픽스처가 담고 있는 문단들(생성 스크립트와 일치해야 한다).
EXPECTED = [
    "한글 문서 첫 번째 문단입니다.",
    "표 안의 내용도 같은 방식으로 담긴다.",
    "제어문자 뒤 본문이 이어진다.",
]
# 확장 제어문자 뒤에 심어 둔 더미 7 코드유닛(한글 '가'~'갆'). 본문에 새어 나오면 실패.
LEAK_CANARY = "".join(chr(c) for c in range(0xAC00, 0xAC07))


def _parser() -> HwpNativeParser:
    return HwpNativeParser()


# ─────────────────────────────────────────────
# can_parse — fail-closed 판별
# ─────────────────────────────────────────────


def test_can_parse_accepts_real_hwp5():
    """실제 HWP 5.0 파일은 처리 가능으로 판정한다."""
    assert _parser().can_parse(COMPRESSED) is True


def test_can_parse_rejects_wrong_extension(tmp_path: Path):
    """내용이 HWP 여도 확장자가 다르면 이 파서가 맡지 않는다."""
    other = tmp_path / "doc.hwpx"
    other.write_bytes(COMPRESSED.read_bytes())
    assert _parser().can_parse(other) is False


def test_can_parse_rejects_non_ole2(tmp_path: Path):
    """OLE2 매직이 아니면 즉시 거부한다."""
    fake = tmp_path / "fake.hwp"
    fake.write_bytes(b"not an ole2 file at all")
    assert _parser().can_parse(fake) is False


def test_can_parse_rejects_ole2_without_hwp_signature(tmp_path: Path):
    """OLE2 이지만 FileHeader 서명이 없으면 거부한다(.doc/.xls 오인 방지).

    HWP 5.0 과 MS Word 97 은 둘 다 OLE2 복합문서라 매직바이트만으로는 구분되지
    않는다. 서명까지 봐야 한다.
    """
    fake = tmp_path / "word.hwp"
    # OLE2 매직만 흉내 낸 파일 — olefile 이 열지 못하거나 FileHeader 가 없다.
    fake.write_bytes(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"\x00" * 2048)
    assert _parser().can_parse(fake) is False


def test_can_parse_rejects_missing_file(tmp_path: Path):
    """없는 파일은 예외가 아니라 False."""
    assert _parser().can_parse(tmp_path / "nope.hwp") is False


# ─────────────────────────────────────────────
# parse — 본문 추출
# ─────────────────────────────────────────────


async def test_parse_compressed_extracts_paragraphs():
    """압축(raw deflate) 본문에서 문단들을 순서대로 뽑는다."""
    tree = await _parser().parse(COMPRESSED)

    assert [n.text for n in tree.nodes] == EXPECTED
    assert tree.doc_format == "hwp"
    assert tree.warnings == ()


async def test_parse_uncompressed_extracts_paragraphs():
    """압축 플래그가 꺼진 문서도 동일하게 읽는다."""
    tree = await _parser().parse(UNCOMPRESSED)

    assert [n.text for n in tree.nodes] == EXPECTED


async def test_parse_skips_extended_control_payload():
    """확장 제어문자 뒤 7 코드유닛이 본문으로 새어 나오지 않는다(핵심 불변식)."""
    tree = await _parser().parse(COMPRESSED)
    joined = "\n".join(n.text for n in tree.nodes)

    assert LEAK_CANARY not in joined
    # 제어문자 앞뒤 본문은 끊김 없이 이어져야 한다.
    assert "제어문자 뒤 본문이 이어진다." in joined


async def test_parse_preserves_paragraph_order():
    """문단 순서(order)가 문서 순서와 일치한다."""
    tree = await _parser().parse(COMPRESSED)

    assert [n.order for n in tree.nodes] == [0, 1, 2]


# ─────────────────────────────────────────────
# 순수 로직 — 레코드/제어문자
# ─────────────────────────────────────────────


def _rec(tag_id: int, payload: bytes, level: int = 0) -> bytes:
    """테스트용 레코드 조립(확장 크기 포함)."""
    size = len(payload)
    if size >= 0xFFF:
        return struct.pack("<II", tag_id | (level << 10) | (0xFFF << 20), size) + payload
    return struct.pack("<I", tag_id | (level << 10) | (size << 20)) + payload


def test_iter_records_handles_extended_size():
    """payload 가 0xFFF 이상이면 확장 크기 필드를 읽어 정확히 잘라낸다."""
    big = b"A" * 5000
    data = _rec(67, b"small") + _rec(66, big)

    got = list(HwpNativeParser._iter_records(data))

    assert [(t, len(p)) for t, p in got] == [(67, 5), (66, 5000)]


def test_iter_records_stops_on_truncated_stream():
    """스트림이 잘려 있으면 예외나 무한루프 없이 멈춘다(손상 파일 방어)."""
    data = _rec(67, b"ok") + struct.pack("<I", 67 | (100 << 20))  # 본문 없는 헤더

    got = list(HwpNativeParser._iter_records(data))

    assert [t for t, _ in got] == [67]


def test_decode_para_text_skips_inline_control():
    """인라인 제어문자(코드 4)도 8 코드유닛을 건너뛴다."""
    payload = (
        "앞".encode("utf-16-le")
        + struct.pack("<H", 4)
        + struct.pack("<7H", *([0xAC00] * 7))
        + "뒤".encode("utf-16-le")
    )

    assert HwpNativeParser._decode_para_text(payload) == "앞뒤"


def test_decode_para_text_maps_line_breaks():
    """줄바꿈 제어문자(10/13)는 개행으로 바꾼다."""
    payload = "가".encode("utf-16-le") + struct.pack("<H", 13) + "나".encode("utf-16-le")

    assert HwpNativeParser._decode_para_text(payload) == "가\n나"


def test_decode_para_text_drops_other_single_controls():
    """표시할 내용이 없는 1코드유닛 제어문자(0, 24~31)는 버린다."""
    payload = (
        "가".encode("utf-16-le")
        + struct.pack("<H", 0)
        + struct.pack("<H", 24)
        + "나".encode("utf-16-le")
    )

    assert HwpNativeParser._decode_para_text(payload) == "가나"


def test_maybe_decompress_falls_back_on_bad_data():
    """압축 플래그가 켜져 있어도 실제로 압축이 아니면 원본을 그대로 돌려준다."""
    raw = b"plain bytes not deflate"

    assert HwpNativeParser._maybe_decompress(raw, True) == raw
    assert HwpNativeParser._maybe_decompress(raw, False) == raw


def test_maybe_decompress_handles_raw_deflate():
    """raw deflate(zlib 헤더 없음)를 정상 해제한다."""
    original = b"hello hwp" * 20
    packed = zlib.compress(original)[2:-4]

    assert HwpNativeParser._maybe_decompress(packed, True) == original


# ─────────────────────────────────────────────
# 암호/배포용 — fail-soft
# ─────────────────────────────────────────────


class _StubOle:
    """FileHeader 하나만 흉내 내는 최소 OLE 스텁."""

    def __init__(self, flags: int, *, has_header: bool = True, short: bool = False) -> None:
        head = bytearray(256)
        head[0:17] = b"HWP Document File"
        head[32:36] = struct.pack("<I", 0x05000300)
        head[36:40] = struct.pack("<I", flags)
        self._data = bytes(head[:8] if short else head)
        self._has_header = has_header

    def exists(self, name: str) -> bool:
        return self._has_header and name == "FileHeader"

    def openstream(self, name: str):
        class _S:
            def __init__(self, d: bytes) -> None:
                self._d = d

            def read(self, n: int = -1) -> bytes:
                return self._d if n < 0 else self._d[:n]

        return _S(self._data)


@pytest.mark.parametrize(
    ("flags", "keyword"),
    [(0x02, "암호"), (0x04, "배포용")],
)
def test_read_file_header_blocks_protected_documents(flags: int, keyword: str):
    """암호(bit1)·배포용(bit2) 문서는 차단으로 판정하고 사유를 경고에 남긴다."""
    warnings: list[str] = []

    _compressed, blocked = HwpNativeParser._read_file_header(_StubOle(flags), warnings)

    assert blocked is True
    assert any(keyword in w for w in warnings)


def test_read_file_header_reports_compression_flag():
    """압축 플래그(bit0)를 올바로 읽고, 보호 문서가 아니면 통과시킨다."""
    warnings: list[str] = []

    compressed, blocked = HwpNativeParser._read_file_header(_StubOle(0x01), warnings)

    assert (compressed, blocked) == (True, False)
    assert warnings == []


def test_read_file_header_missing_stream_is_blocked():
    """FileHeader 가 없으면 차단하고 사유를 남긴다."""
    warnings: list[str] = []

    _compressed, blocked = HwpNativeParser._read_file_header(
        _StubOle(0x01, has_header=False), warnings
    )

    assert blocked is True
    assert warnings


async def test_parse_corrupted_file_is_fail_soft(tmp_path: Path):
    """손상된 파일도 예외를 던지지 않고 빈 트리 + 경고로 끝난다."""
    broken = tmp_path / "broken.hwp"
    broken.write_bytes(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"\xff" * 1024)

    tree = await _parser().parse(broken)

    assert tree.nodes == ()
    assert tree.warnings  # 사유가 남아야 한다
