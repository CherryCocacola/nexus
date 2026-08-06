# 한글 구포맷(.hwp, HWP 5.0)을 외부 변환기 없이 직접 파싱하는 파서.
"""
HwpNativeParser — HWP 5.0 문서를 순수 파이썬으로 파싱한다.

■ 왜 만들었나 (2026-08-06)
  기존 .hwp 경로는 LibreOffice 변환(HwpViaLibreOfficeParser)이었는데, 실측해 보니
  LibreOffice 의 한글 필터(libhwplo.so)는 **HWP V2.00/V2.10/V3.00(한글 97 이하)**
  만 인식한다. 한글 2002 이후가 쓰는 **HWP 5.0** 은 지원하지 않는다(필터 바이너리에
  HWP5 스트림명 BodyText/DocInfo/FileHeader 참조가 전혀 없음). 즉 LibreOffice 를
  설치해도(+373MB) 실사용 .hwp 는 한 건도 열리지 않는다. 그래서 포맷을 직접 읽는다.

■ HWP 5.0 구조 요약
  파일 자체는 OLE2 복합문서(CFB)이고, 그 안에 스트림들이 들어 있다.
    FileHeader          256바이트. 앞 32바이트가 "HWP Document File" 시그니처,
                        32~35 버전(DWORD), 36~39 플래그(DWORD).
                        플래그 bit0=압축, bit1=암호, bit2=배포용(암호화).
    DocInfo             문서 속성 레코드.
    BodyText/SectionN   본문 레코드. 실제 글자는 여기 있다.
    PrvText             미리보기 평문(UTF-16LE, 항상 비압축). 본문 파싱이 실패했을 때의
                        마지막 수단으로 쓴다 — 앞부분만 담고 있어 전문은 아니다.
  압축 플래그가 켜져 있으면 DocInfo/SectionN 은 raw deflate 라서
  zlib.decompress(data, -15) 로 푼다(zlib 헤더가 없으므로 -15).

■ 레코드 형식
  4바이트 리틀엔디언 DWORD 하나가 헤더다.
    tag_id = v & 0x3FF          (10비트)
    level  = (v >> 10) & 0x3FF  (10비트)
    size   = (v >> 20) & 0xFFF  (12비트)
  size 가 0xFFF 이면 뒤따르는 4바이트가 실제 크기다(확장 크기).
  본문 글자는 tag_id == HWPTAG_PARA_TEXT(67) 레코드의 payload 에 UTF-16LE 로 들어 있다.

■ 본문 텍스트의 제어문자 처리 (이 파서의 핵심 난점)
  PARA_TEXT payload 는 단순 문자열이 아니다. 코드값 32 미만은 제어문자이며 세 부류다.
    - char control  (1 코드유닛)  : 줄바꿈·문단 구분 등
    - inline control(8 코드유닛)  : 글자처럼 취급되는 개체
    - extended control(8 코드유닛): 표·그림 같은 확장 개체
  뒤의 두 부류는 **8 코드유닛(16바이트)을 통째로 건너뛰어야** 한다. 이걸 1개만 건너뛰면
  뒤따르는 이진 데이터가 한글 글자로 잘못 해독돼 문서 전체가 깨진 문자로 나온다.

■ fail-soft (DocumentParser 계약)
  섹션 하나가 깨져도 예외로 중단하지 않고, 읽은 만큼 트리에 담고 사유는 warnings 에
  모은다. 암호/배포용 문서처럼 아예 읽을 수 없는 경우도 빈 트리 + 경고로 표현한다.

■ 에어갭
  외부 네트워크를 쓰지 않는다. 의존성은 olefile(순수 파이썬, BSD) 하나뿐이다.

작성자: 이현수 / 작성일: 2026-08-06
"""

from __future__ import annotations

import logging
import zlib
from pathlib import Path

import olefile

from core.ingest.parser_base import DocumentParser
from core.ingest.types import DocumentNode, DocumentTree, ElementType

logger = logging.getLogger("nexus.ingest.hwp_native")

# OLE2(복합문서) 시그니처 8바이트. HWP 5.0 파일은 항상 이걸로 시작한다.
_OLE_MAGIC = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"

# FileHeader 스트림 맨 앞 32바이트에 들어가는 서명(뒤는 NUL 패딩).
_HWP_SIGNATURE = b"HWP Document File"

# FileHeader 플래그 비트.
_FLAG_COMPRESSED = 0x01  # 본문/문서정보가 raw deflate 로 압축됨
_FLAG_PASSWORD = 0x02  # 암호 설정됨 — 복호화 불가
_FLAG_DISTRIBUTION = 0x04  # 배포용 문서 — 별도 암호화, 복호화 불가

# 레코드 태그. HWPTAG_BEGIN(0x10=16) 기준 오프셋으로 정의돼 있다.
_HWPTAG_BEGIN = 0x010
_HWPTAG_PARA_HEADER = _HWPTAG_BEGIN + 50  # 66 — 문단 시작
_HWPTAG_PARA_TEXT = _HWPTAG_BEGIN + 51  # 67 — 문단 본문(UTF-16LE)
_HWPTAG_CTRL_HEADER = _HWPTAG_BEGIN + 55  # 71 — 개체 시작(표/그림 등)
_HWPTAG_LIST_HEADER = _HWPTAG_BEGIN + 56  # 72 — 문단 목록 시작(표에서는 '셀' 하나)
_HWPTAG_TABLE = _HWPTAG_BEGIN + 61  # 77 — 표 속성(행·열 수)

# CTRL_HEADER 의 개체 종류 식별자. 4바이트가 뒤집혀 저장돼 있어 [::-1] 로 읽는다.
_CTRL_ID_TABLE = b"tbl "

# 표 셀 정보(LIST_HEADER payload)의 오프셋.
#   앞 8바이트는 공통 헤더(문단 수 int32 + 속성 uint32)이고 그 뒤에 셀 주소가 온다.
#   실측(입찰공고문.hwp)으로 확인: 열주소·행주소·열병합·행병합이 uint16 으로 이어진다.
_CELL_COL_OFFSET = 8
_CELL_ROW_OFFSET = 10
_CELL_MIN_SIZE = 12  # 행/열 주소를 읽으려면 최소 이만큼은 있어야 한다

# PARA_TEXT 안의 제어문자 분류.
#   확장/인라인 제어문자는 자기 자신 포함 8 코드유닛을 차지한다(16바이트).
#   이 표가 틀리면 이진 데이터가 글자로 새어 나와 문서가 통째로 깨진다.
_CTRL_EXTENDED = frozenset({1, 2, 3, 11, 12, 14, 15, 16, 17, 18, 21, 22, 23})
_CTRL_INLINE = frozenset({4, 5, 6, 7, 8, 9, 19, 20})
# 자리를 1 코드유닛만 차지하는 제어문자 중 줄바꿈으로 볼 것들.
_CTRL_LINE_BREAK = frozenset({10, 13})


class HwpNativeParser(DocumentParser):
    """HWP 5.0(.hwp) 문서를 OLE2 스트림에서 직접 읽어 구조 트리로 만드는 파서."""

    # ── 메타 정보 ──────────────────────────────────────────────

    @property
    def supported_extensions(self) -> tuple[str, ...]:
        """이 파서가 담당하는 확장자. 신포맷 .hwpx 는 HwpxParser 가 맡는다."""
        return (".hwp",)

    @property
    def requires_gpu(self) -> bool:
        """파일 구조를 그대로 읽을 뿐이라 GPU 가 필요 없다."""
        return False

    # ── 처리 가능 여부 (fail-closed) ───────────────────────────

    def can_parse(self, path: Path) -> bool:
        """확장자 + OLE2 매직 + FileHeader 의 HWP 서명까지 확인한다.

        확장자만 믿지 않는 이유: .hwp 로 이름만 바꾼 다른 파일이 들어오면 엉뚱한
        해석 결과를 내놓게 된다. 서명까지 맞아야 True 를 돌려준다(불확실하면 거부).
        """
        if path.suffix.lower() != ".hwp":
            return False
        try:
            with path.open("rb") as f:
                if f.read(8) != _OLE_MAGIC:
                    return False
        except OSError as e:
            logger.debug("can_parse 파일 열기 실패: %s (%s)", path, e)
            return False

        # OLE2 이더라도 HWP 가 아닐 수 있다(.doc/.xls 도 OLE2). FileHeader 서명 확인.
        try:
            with olefile.OleFileIO(str(path)) as ole:
                if not ole.exists("FileHeader"):
                    return False
                head = ole.openstream("FileHeader").read(32)
            return head.startswith(_HWP_SIGNATURE)
        except Exception as e:  # noqa: BLE001 — 손상 파일 등 어떤 사유든 '처리 불가'
            logger.debug("can_parse HWP 서명 확인 실패: %s (%s)", path, e)
            return False

    # ── 핵심: parse ────────────────────────────────────────────

    async def parse(self, path: Path) -> DocumentTree:
        """.hwp 를 읽어 문단 노드들로 이루어진 구조 트리를 돌려준다.

        치명적 상황(암호 문서·손상)도 예외를 던지지 않고 빈 트리 + 경고로 표현한다.
        """
        warnings: list[str] = []
        nodes: list[DocumentNode] = []

        try:
            with olefile.OleFileIO(str(path)) as ole:
                compressed, blocked = self._read_file_header(ole, warnings)
                if blocked:
                    # 암호/배포용 — 복호화 수단이 없다. 여기서 끝낸다(경고만 남김).
                    return self._tree(path, nodes, warnings)

                paragraphs = self._read_body_paragraphs(ole, compressed, warnings)
                if not paragraphs:
                    # 본문을 못 읽었으면 미리보기 평문이라도 건진다(부분 가용성).
                    paragraphs = self._read_preview_text(ole, warnings)

            for order, (element_type, text) in enumerate(paragraphs):
                nodes.append(DocumentNode(element_type=element_type, text=text, order=order))
        except Exception as e:  # noqa: BLE001 — fail-soft: 사유를 경고로 남기고 빈 트리
            logger.warning("HWP 파싱 실패: %s — %s", path, e)
            warnings.append(f"HWP 파싱 실패: {type(e).__name__}: {e}")

        return self._tree(path, nodes, warnings)

    # ── 내부: FileHeader ───────────────────────────────────────

    @staticmethod
    def _read_file_header(ole: olefile.OleFileIO, warnings: list[str]) -> tuple[bool, bool]:
        """FileHeader 에서 (압축 여부, 읽기 차단 여부)를 뽑는다.

        차단 여부가 True 면 암호 또는 배포용 문서라 본문을 해독할 수 없다는 뜻이다.
        복호화를 시도하지 않는 이유는 명확하다 — 사용자의 암호 없이 푸는 것은
        기능이 아니라 우회이며, 이 시스템이 할 일이 아니다.
        """
        if not ole.exists("FileHeader"):
            warnings.append("FileHeader 스트림이 없습니다 — HWP 5.0 파일이 아닐 수 있습니다.")
            return False, True

        header = ole.openstream("FileHeader").read(40)
        if len(header) < 40:
            warnings.append("FileHeader 가 너무 짧습니다(손상 가능).")
            return False, True

        flags = int.from_bytes(header[36:40], "little")
        compressed = bool(flags & _FLAG_COMPRESSED)

        if flags & _FLAG_PASSWORD:
            warnings.append("암호가 걸린 문서라 본문을 읽을 수 없습니다.")
            return compressed, True
        if flags & _FLAG_DISTRIBUTION:
            warnings.append("배포용(암호화) 문서라 본문을 읽을 수 없습니다.")
            return compressed, True
        return compressed, False

    # ── 내부: 본문 ─────────────────────────────────────────────

    def _read_body_paragraphs(
        self, ole: olefile.OleFileIO, compressed: bool, warnings: list[str]
    ) -> list[tuple[ElementType, str]]:
        """BodyText/SectionN 을 순서대로 읽어 (요소종류, 텍스트) 목록을 만든다."""
        # 섹션 스트림 목록을 모아 번호순으로 정렬한다(Section10 이 Section2 보다
        # 뒤에 오도록 문자열이 아니라 숫자로 정렬 — 문서 순서가 뒤집히면 안 된다).
        sections: list[tuple[int, list[str]]] = []
        for entry in ole.listdir():
            if len(entry) == 2 and entry[0] == "BodyText" and entry[1].startswith("Section"):
                suffix = entry[1][len("Section") :]
                sections.append((int(suffix) if suffix.isdigit() else 0, entry))

        if not sections:
            warnings.append("BodyText 섹션이 없습니다.")
            return []

        paragraphs: list[tuple[ElementType, str]] = []
        for _num, entry in sorted(sections, key=lambda t: t[0]):
            name = "/".join(entry)
            try:
                raw = ole.openstream(entry).read()
                data = self._maybe_decompress(raw, compressed)
                paragraphs.extend(self._extract_paragraphs(data))
            except Exception as e:  # noqa: BLE001 — 섹션 하나 실패가 전체를 막지 않게
                logger.warning("섹션 읽기 실패: %s — %s", name, e)
                warnings.append(f"{name} 읽기 실패: {type(e).__name__}: {e}")
                continue
        return paragraphs

    @staticmethod
    def _maybe_decompress(raw: bytes, compressed: bool) -> bytes:
        """압축 플래그가 켜져 있으면 raw deflate 로 푼다.

        zlib 헤더가 없는 raw deflate 라 wbits=-15 를 쓴다. 플래그와 실제가 어긋난
        파일이 있을 수 있어, 실패하면 원본을 그대로 돌려 다음 단계가 판단하게 한다.
        """
        if not compressed:
            return raw
        try:
            return zlib.decompress(raw, -15)
        except zlib.error:
            return raw

    @classmethod
    def _extract_paragraphs(cls, data: bytes) -> list[tuple[ElementType, str]]:
        """레코드 스트림을 훑어 본문 문단과 표를 문서 순서대로 모은다.

        일반 문단: PARA_HEADER 를 만날 때마다 새 문단을 시작한다. 한 문단이 여러
        PARA_TEXT 레코드로 쪼개져 있을 수 있으므로 경계 전까지는 이어 붙인다.

        표: CTRL_HEADER 의 개체 식별자가 'tbl ' 이면 표가 시작된다. 그 안의
        LIST_HEADER 하나가 셀 하나이고, 셀의 (행, 열) 주소가 payload 에 들어 있다.
        이 주소로 셀을 행별로 묶어 " | " 로 이어 붙인다 — 주소를 쓰기 때문에
        병합 셀(rowspan/colspan)이 있어도 행이 뒤섞이지 않는다.
        (예전에는 셀을 각각 별도 문단으로 평탄화해, 어느 셀이 같은 행인지 알 수
         없었다. 입찰공고문·제안요청서처럼 표가 본문인 문서에서 특히 문제였다.)

        표는 중첩될 수 있으므로 스택으로 다룬다. 표 안의 레코드는 CTRL_HEADER 보다
        깊은 level 을 갖는다는 성질을 이용해, level 이 그 이하로 돌아오면 표를 닫는다.
        """
        out: list[tuple[ElementType, str]] = []
        current: list[str] = []  # 지금 모으는 중인 문단 조각들
        # 열려 있는 표들. 각 항목: {"level": 시작 level, "cells": {(행,열): [텍스트...]}}
        tables: list[dict] = []

        def flush_para() -> None:
            """모아 둔 문단 조각을 확정한다. 표 안이면 현재 셀에, 밖이면 본문에 넣는다."""
            text = "".join(current).strip()
            current.clear()
            if not text:
                return
            if tables and tables[-1]["current_cell"] is not None:
                tables[-1]["cells"].setdefault(tables[-1]["current_cell"], []).append(text)
            else:
                out.append((ElementType.PARAGRAPH, text))

        def close_table() -> None:
            """가장 안쪽 표를 닫아 행 단위 텍스트로 만들어 내보낸다."""
            table = tables.pop()
            cells: dict[tuple[int, int], list[str]] = table["cells"]
            if not cells:
                return
            rows: dict[int, list[tuple[int, str]]] = {}
            for (row, col), parts in cells.items():
                # 한 셀 안의 여러 문단은 공백으로 이어 한 줄로 만든다(행이 깨지지 않게).
                rows.setdefault(row, []).append((col, " ".join(parts).strip()))
            lines = []
            for row in sorted(rows):
                ordered = [text for _col, text in sorted(rows[row])]
                line = " | ".join(t for t in ordered if t)
                if line:
                    lines.append(line)
            if not lines:
                return
            text = "\n".join(lines)
            # 중첩 표였다면 바깥 표의 현재 셀 안에 텍스트로 넣고, 아니면 본문에 내보낸다.
            if tables and tables[-1]["current_cell"] is not None:
                tables[-1]["cells"].setdefault(tables[-1]["current_cell"], []).append(text)
            else:
                out.append((ElementType.TABLE, text))

        for tag_id, level, payload in cls._iter_records(data):
            # 표 밖으로 빠져나왔으면(level 이 시작 level 이하) 열린 표를 닫는다.
            while tables and level <= tables[-1]["level"]:
                flush_para()
                close_table()

            if tag_id == _HWPTAG_CTRL_HEADER:
                flush_para()
                # 개체 식별자 4바이트는 뒤집혀 저장돼 있다('tbl ' → ' lbt').
                if payload[:4][::-1] == _CTRL_ID_TABLE:
                    tables.append({"level": level, "cells": {}, "current_cell": None})
            elif tag_id == _HWPTAG_LIST_HEADER and tables:
                # 표 안의 LIST_HEADER = 셀 하나의 시작. 앞 셀 내용을 확정하고 주소를 읽는다.
                flush_para()
                tables[-1]["current_cell"] = cls._read_cell_address(payload)
            elif tag_id == _HWPTAG_PARA_HEADER:
                flush_para()
            elif tag_id == _HWPTAG_PARA_TEXT:
                current.append(cls._decode_para_text(payload))

        flush_para()
        while tables:
            close_table()
        return out

    @staticmethod
    def _read_cell_address(payload: bytes) -> tuple[int, int]:
        """표 셀(LIST_HEADER)의 (행, 열) 주소를 읽는다.

        payload 앞 8바이트는 공통 헤더(문단 수 + 속성)이고 그 뒤에 열주소·행주소가
        uint16 으로 이어진다. 주소를 읽을 수 없을 만큼 짧으면 (0, 0) 으로 돌려
        최소한 내용은 살린다(형식이 다른 파일에서도 죽지 않게 — fail-soft).
        """
        if len(payload) < _CELL_MIN_SIZE:
            return (0, 0)
        col = int.from_bytes(payload[_CELL_COL_OFFSET : _CELL_COL_OFFSET + 2], "little")
        row = int.from_bytes(payload[_CELL_ROW_OFFSET : _CELL_ROW_OFFSET + 2], "little")
        return (row, col)

    @staticmethod
    def _iter_records(data: bytes):
        """레코드 스트림을 (tag_id, level, payload) 로 순회한다.

        헤더 DWORD 에서 tag/level/size 를 뽑고, size 가 0xFFF 이면 뒤 4바이트가
        실제 크기다. level 은 개체 중첩 깊이라 표의 시작·끝 판정에 쓴다.
        남은 바이트가 모자라면 조용히 멈춘다(손상 파일 방어).
        """
        pos = 0
        end = len(data)
        while pos + 4 <= end:
            header = int.from_bytes(data[pos : pos + 4], "little")
            pos += 4
            tag_id = header & 0x3FF
            level = (header >> 10) & 0x3FF
            size = (header >> 20) & 0xFFF
            if size == 0xFFF:  # 확장 크기 — 뒤 4바이트가 진짜 크기
                if pos + 4 > end:
                    return
                size = int.from_bytes(data[pos : pos + 4], "little")
                pos += 4
            if pos + size > end:  # 손상/잘린 스트림
                return
            yield tag_id, level, data[pos : pos + size]
            pos += size

    @staticmethod
    def _decode_para_text(payload: bytes) -> str:
        """PARA_TEXT payload(UTF-16LE + 제어문자)를 사람이 읽는 문자열로 바꾼다.

        확장·인라인 제어문자는 자기 자신 포함 8 코드유닛을 차지하므로 통째로
        건너뛴다. 이 건너뛰기를 빠뜨리면 개체의 이진 데이터가 한글 글자로 잘못
        해독돼 문서 전체가 깨진 문자로 나온다(이 파서에서 가장 중요한 부분).
        """
        # 코드유닛 단위로 다뤄야 해서 2바이트씩 끊어 정수 배열로 만든다.
        count = len(payload) // 2
        out: list[str] = []
        i = 0
        while i < count:
            code = int.from_bytes(payload[i * 2 : i * 2 + 2], "little")
            if code in _CTRL_EXTENDED or code in _CTRL_INLINE:
                i += 8  # 제어문자 1 + 데이터 6 + 종료 1 = 8 코드유닛
                continue
            if code in _CTRL_LINE_BREAK:
                out.append("\n")
            elif code >= 32:
                # 서로게이트 쌍은 나오지 않지만, 혹시 몰라 chr() 실패는 무시한다.
                try:
                    out.append(chr(code))
                except ValueError:
                    pass
            # 그 밖의 1코드유닛 제어문자(0, 24~31 등)는 표시할 내용이 없어 버린다.
            i += 1
        return "".join(out)

    # ── 내부: 미리보기 폴백 ────────────────────────────────────

    @staticmethod
    def _read_preview_text(
        ole: olefile.OleFileIO, warnings: list[str]
    ) -> list[tuple[ElementType, str]]:
        """본문을 못 읽었을 때 PrvText(미리보기 평문)라도 건진다.

        PrvText 는 항상 비압축 UTF-16LE 이고 문서 앞부분만 담는다. 전문이 아니므로
        경고를 함께 남겨, 이것이 '일부'라는 사실이 사용자에게 전달되게 한다.
        """
        if not ole.exists("PrvText"):
            return []
        try:
            text = ole.openstream("PrvText").read().decode("utf-16-le", errors="replace")
        except Exception as e:  # noqa: BLE001 — 폴백이므로 실패해도 조용히 포기
            warnings.append(f"미리보기 텍스트 읽기 실패: {type(e).__name__}: {e}")
            return []

        text = text.replace("\r\n", "\n").replace("\r", "\n").strip("\x00").strip()
        if not text:
            return []
        warnings.append("본문을 읽지 못해 미리보기 텍스트만 추출했습니다(문서 앞부분 일부).")
        return [(ElementType.PARAGRAPH, p.strip()) for p in text.split("\n") if p.strip()]

    # ── 내부: 결과 조립 ────────────────────────────────────────

    @staticmethod
    def _tree(path: Path, nodes: list[DocumentNode], warnings: list[str]) -> DocumentTree:
        """노드/경고를 DocumentTree 로 감싼다."""
        return DocumentTree(
            title=path.stem,
            source_path=str(path),
            nodes=tuple(nodes),
            doc_format="hwp",
            warnings=tuple(warnings),
        )
