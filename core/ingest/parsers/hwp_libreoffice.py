"""
구포맷 .hwp 파서 — LibreOffice headless 로 .docx 변환 후 python-docx 로 파싱한다.

왜 LibreOffice 경유인가 (v7.3 로드맵 단계 9 — 구포맷 .hwp):
  구포맷 .hwp(한글 v5)는 OLE 복합문서 기반의 폐쇄 바이너리 포맷이다. 개방형
  HWPX(OWPML, ZIP+XML)와 달리 구조를 직접 읽을 수 없고, 이를 파싱하는 청정
  라이선스 파이썬 라이브러리가 마땅치 않다(대표격 pyhwp 는 AGPL 이라 라이선스
  정책상 배제). 그런데 사용자 문서에서 구포맷 .hwp 비중이 높아(사용자 확정)
  반드시 지원해야 한다. 해법: LibreOffice 의 한글 import 필터로 .hwp →
  .docx 변환 후, 기존 python-docx 경로(document_tool._parse_docx)와 같은
  방식으로 — 단 "구조 보존" 버전으로 — 문단/표/제목을 노드 트리로 만든다.

처리 개요:
  1. soffice --headless --convert-to docx --outdir <tmp> <hwp>  (subprocess)
       → LibreOffice 가 .hwp import 필터로 같은 이름의 .docx 를 tmp 에 생성.
  2. 생성된 .docx 를 docx.Document 로 열고, 문서 본문(body)을 "문서 순서대로"
     훑어 노드를 만든다:
       · 문단 스타일이 "Heading 1" → SECTION, "Heading 2/3..." → SUBHEADING.
         (heading_path 누적 — 이후 본문 PARAGRAPH 에 맥락 경로로 전파.)
       · 일반 문단 → PARAGRAPH (현재까지의 heading_path 부여).
       · 표 → TABLE (행/열 보존, 셀을 " | " 로, 행은 줄바꿈으로).
       · 빈 문단(공백만)은 생략(검색 노이즈 방지).
  3. tmpdir(및 그 안의 변환 .docx) 정리.

왜 body 를 직접 순회하는가 (구조 보존):
  document_tool._parse_docx 는 doc.paragraphs 와 doc.tables 를 따로 돌려
  문단/표의 "문서 내 순서"를 잃는다. 구조 보존을 위해 body XML 자식을 순서대로
  훑어, 표가 문단들 사이에 끼어 있던 원래 흐름을 그대로 노드 순서로 복원한다.

soffice 경로 (anti-pattern #4 — 하드코딩 금지):
  HwpConfig(core/config.py)에서 soffice_cmd / convert_timeout_sec 를 읽는다.
  config 로딩이 안 되면 환경변수(NEXUS_SOFFICE_CMD) → 개발 기본값(Windows
  설치 경로) 순으로 폴백한다. 배포(에어갭 Linux)에서는 yaml/환경변수로
  "/usr/bin/soffice" 등 실제 경로를 덮어쓴다.

fail-soft (anti-pattern #8):
  soffice 미설치(FileNotFoundError) / 변환 실패(CalledProcessError) /
  변환 시간 초과(TimeoutExpired) / 변환물(.docx) 미생성 / python-docx 파싱
  실패는 모두 구체 예외로 포착해 빈 트리(또는 부분 트리) + warnings 로
  표현하고 예외를 전파하지 않는다. bare except 금지 — 구체 예외만.

보안:
  subprocess 는 shell=False + 인자 리스트로 호출한다(셸 인젝션 방지 — 파일명에
  특수문자가 있어도 셸 해석을 거치지 않는다). 임시 디렉토리는 tempfile.mkdtemp
  로 만들고 finally 에서 정리한다(동시 호출 간 파일 충돌 방지).

의존성 방향 (P2): core.ingest.types / parser_base + core.config 만 의존.
  core/rag·core/model 무관(역방향/순환 없음).
에어갭: soffice/python-docx 모두 로컬 파일만 다룬다(외부 네트워크 없음).
  import/호출만 하며 런타임 설치 코드는 넣지 않는다(anti-pattern #10).
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from docx import Document
from docx.opc.exceptions import OpcError, PackageNotFoundError
from docx.oxml.ns import qn

from core.ingest.parser_base import DocumentParser
from core.ingest.types import DocumentNode, DocumentTree, ElementType

logger = logging.getLogger("nexus.ingest.parsers.hwp_libreoffice")

# 구포맷 .hwp(한글 v5)의 매직바이트 — OLE2 복합문서(Compound File Binary)
# 시그니처다. .hwp v5 는 이 OLE 컨테이너 안에 한글 스트림을 담는다.
# (HWPX 의 ZIP "PK\x03\x04" 와 다르다 — 그쪽은 hwpx.py 가 가져간다.)
_HWP_OLE_MAGIC = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"

# 개발 기본값 — config/환경변수가 모두 비었을 때의 최후 폴백(Windows 설치 경로).
# 배포(Linux)에서는 HwpConfig.soffice_cmd 또는 NEXUS_SOFFICE_CMD 로 오버라이드.
_DEFAULT_SOFFICE_CMD = r"C:\Program Files\LibreOffice\program\soffice.exe"

# 변환 타임아웃 기본값(초) — config 로딩 실패 시의 폴백.
_DEFAULT_TIMEOUT_SEC = 120.0


class HwpViaLibreOfficeParser(DocumentParser):
    """
    구포맷 .hwp 파일을 LibreOffice 로 .docx 변환 후 python-docx 로 파싱하는 파서.

    LibreOffice 변환은 CPU 만으로 도는 외부 프로세스이므로 GPU 가 필요 없다
    (requires_gpu=False). HWPX(.hwpx)는 HwpxParser 가 담당하고, 이 파서는 구포맷
    바이너리 .hwp 만 처리한다.
    """

    def __init__(self) -> None:
        """
        파서 인스턴스를 만든다.

        왜 soffice 설정을 지연(lazy)으로 읽는가:
          soffice 경로/타임아웃은 HwpConfig 에서 읽는데, config 로딩이 가능한지·
          필요한지는 첫 parse() 시점에 판단하는 편이 안전하다(레지스트리에 등록만
          되고 한 번도 안 쓰는 호스트에서 config 접근/경고를 피한다). 따라서 첫
          parse() 호출 때 한 번만 읽어 캐시한다.
        """
        # (soffice_cmd, timeout_sec) 캐시 — 첫 parse 때 채운다.
        self._cfg: tuple[str, float] | None = None

    # ─── 정체성/플래그 ───

    @property
    def supported_extensions(self) -> tuple[str, ...]:
        # 구포맷 .hwp 만 처리. .hwpx(개방형)는 HwpxParser 담당.
        return (".hwp",)

    @property
    def requires_gpu(self) -> bool:
        # LibreOffice 변환 + python-docx 파싱 — GPU 불필요.
        return False

    # ─── 처리 가능 여부 (fail-closed) ───

    def can_parse(self, path: Path) -> bool:
        """
        확장자(.hwp) + 매직바이트(OLE2 복합문서 시그니처)로 처리 가능 여부 판단.

        fail-closed: 확장자가 다르거나, 파일을 열 수 없거나, 시그니처가 OLE2 가
        아니면 False(불확실하면 거부). HWPX(ZIP "PK")는 여기서 자연히 걸러진다.
        """
        if path.suffix.lower() != ".hwp":
            return False
        try:
            with path.open("rb") as f:
                head = f.read(8)  # OLE2 시그니처는 8바이트.
        except OSError as e:
            logger.debug("can_parse 파일 열기 실패: %s (%s)", path, e)
            return False
        return head == _HWP_OLE_MAGIC

    # ─── 핵심: parse ───

    async def parse(self, path: Path) -> DocumentTree:
        """
        구포맷 .hwp 를 LibreOffice 로 .docx 변환 후 구조 트리로 파싱한다.

        반환: DocumentTree(title, source_path, nodes=구조 노드들, doc_format="hwp",
        warnings=fail-soft 경고).

        - 치명적 상황(soffice 미설치/변환 실패/타임아웃/파싱 실패)도 예외를 던지지
          않고 빈 트리 + 경고로 표현한다(파이프라인 중단 방지 — fail-soft).
        """
        warnings: list[str] = []

        # 1) soffice 설정(경로/타임아웃) 읽기 — 실패해도 환경변수/기본값 폴백.
        soffice_cmd, timeout_sec = self._ensure_config()

        # 2) 임시 디렉토리에서 .hwp → .docx 변환. 변환물 경로를 받아온다.
        #    (동시 호출/파일 충돌 방지를 위해 호출마다 독립 tmpdir 사용.)
        tmpdir = tempfile.mkdtemp(prefix="nexus_hwp_")
        try:
            docx_path = self._convert_to_docx(path, soffice_cmd, timeout_sec, tmpdir, warnings)
            if docx_path is None:
                # 변환 실패 — 경고는 _convert_to_docx 가 이미 쌓았다. 빈 트리.
                return self._empty_tree(path, warnings)

            # 3) 변환된 .docx 를 구조 보존하며 노드로 변환.
            nodes = self._parse_docx_structured(docx_path, warnings)
        finally:
            # 4) 임시 디렉토리(및 변환 .docx) 정리 — 변환/파싱 성패와 무관.
            self._cleanup_tmpdir(tmpdir)

        return DocumentTree(
            title=path.stem,
            source_path=str(path),
            nodes=tuple(nodes),
            doc_format="hwp",
            warnings=tuple(warnings),
        )

    # ─────────────────────────────────────────
    # 내부: 설정
    # ─────────────────────────────────────────

    def _ensure_config(self) -> tuple[str, float]:
        """
        HwpConfig 에서 soffice 경로/타임아웃을 한 번만 읽어 캐시한다.

        우선순위(경로): HwpConfig.soffice_cmd → 환경변수(NEXUS_SOFFICE_CMD) →
        개발 기본값(_DEFAULT_SOFFICE_CMD). config 로딩이 어떤 이유로든 실패해도
        변환은 환경변수/기본값으로 시도할 수 있게 한다(부분 가용성 우선).
        """
        if self._cfg is not None:
            return self._cfg

        soffice_cmd = ""
        timeout_sec = _DEFAULT_TIMEOUT_SEC

        # HwpConfig 에서 읽기 — config 시스템 문제는 환경변수/기본값 폴백으로 흡수.
        try:
            from core.config import load_and_validate_config

            cfg = load_and_validate_config()
            hwp = cfg.hwp
            soffice_cmd = (hwp.soffice_cmd or "").strip()
            timeout_sec = float(hwp.convert_timeout_sec or _DEFAULT_TIMEOUT_SEC)
        except (OSError, ValueError, RuntimeError, ImportError) as e:
            logger.warning("HwpConfig 로딩 실패 — 환경변수/기본값 폴백: %s", e)

        # 환경변수 폴백(config 값이 비어 있을 때만 보강).
        if not soffice_cmd:
            soffice_cmd = os.environ.get("NEXUS_SOFFICE_CMD", "").strip()
        # 그래도 비면 개발 기본값.
        if not soffice_cmd:
            soffice_cmd = _DEFAULT_SOFFICE_CMD

        # 타임아웃 하한 보호(비정상적으로 0/음수면 기본값으로).
        if timeout_sec <= 0:
            timeout_sec = _DEFAULT_TIMEOUT_SEC

        self._cfg = (soffice_cmd, timeout_sec)
        return self._cfg

    # ─────────────────────────────────────────
    # 내부: LibreOffice 변환
    # ─────────────────────────────────────────

    def _convert_to_docx(
        self,
        path: Path,
        soffice_cmd: str,
        timeout_sec: float,
        outdir: str,
        warnings: list[str],
    ) -> Path | None:
        """
        soffice 를 headless 로 호출해 .hwp 를 outdir 에 .docx 로 변환한다.

        명령: soffice --headless --convert-to docx --outdir <outdir> <hwp>
          LibreOffice 가 한글 import 필터로 .hwp 를 읽어 같은 stem 의 .docx 를
          outdir 에 만든다(예: report.hwp → outdir/report.docx).

        보안: shell=False + 인자 리스트(셸 인젝션 방지). 파일명에 공백/특수문자가
        있어도 셸을 거치지 않아 안전하다.

        fail-soft: 미설치(FileNotFoundError) / 비정상 종료(CalledProcessError) /
        시간 초과(TimeoutExpired) / 변환물 미발견은 경고만 쌓고 None 을 돌린다.

        반환: 생성된 .docx 경로(성공), 실패 시 None.
        """
        cmd = [
            soffice_cmd,
            "--headless",  # GUI 없이 백그라운드 실행.
            "--convert-to",
            "docx",  # 출력 포맷(import 필터는 입력 .hwp 확장자로 자동 선택).
            "--outdir",
            outdir,
            str(path),
        ]

        try:
            # check=True: 비정상 종료(0 아님)면 CalledProcessError 를 던지게 한다.
            # capture_output: soffice 의 stdout/stderr 를 잡아 경고에 담는다.
            # shell=False(기본): 인자 리스트를 그대로 전달 — 셸 해석 없음(인젝션 차단).
            result = subprocess.run(  # noqa: S603 — shell=False + 신뢰 가능한 인자 리스트
                cmd,
                check=True,
                capture_output=True,
                timeout=timeout_sec,
            )
            logger.debug(
                "soffice 변환 완료: %s (rc=%d, stdout=%r)",
                path,
                result.returncode,
                (result.stdout or b"")[:200],
            )
        except FileNotFoundError as e:
            # soffice 실행 파일이 경로에 없음 — 미설치/경로 오타.
            msg = (
                f"LibreOffice(soffice) 실행 파일을 찾지 못함 — .hwp 변환 불가: "
                f"{soffice_cmd} ({type(e).__name__}: {e})"
            )
            logger.warning("%s (%s)", msg, path)
            warnings.append(msg)
            return None
        except subprocess.TimeoutExpired as e:
            msg = f".hwp→docx 변환 시간 초과({timeout_sec}s) — 건너뜀 ({type(e).__name__})"
            logger.warning("%s (%s)", msg, path)
            warnings.append(msg)
            return None
        except subprocess.CalledProcessError as e:
            # soffice 가 비정상 종료 — stderr 일부를 경고에 담아 진단을 돕는다.
            stderr = (e.stderr or b"")[:300]
            msg = f".hwp→docx 변환 실패(rc={e.returncode}): {stderr!r}"
            logger.warning("%s (%s)", msg, path)
            warnings.append(msg)
            return None
        except OSError as e:
            # 권한/리소스 등 기타 OS 수준 실패.
            msg = f".hwp→docx 변환 중 OS 오류: {type(e).__name__}: {e}"
            logger.warning("%s (%s)", msg, path)
            warnings.append(msg)
            return None

        # 변환물 경로: outdir/<원본 stem>.docx. LibreOffice 는 입력 stem 을 그대로
        # 쓰므로 이를 우선 확인하고, 못 찾으면 outdir 내 .docx 를 폭넓게 탐색한다.
        expected = Path(outdir) / f"{path.stem}.docx"
        if expected.is_file():
            return expected

        # 일부 환경에서 stem 이 정규화될 수 있어, outdir 내 첫 .docx 로 폴백.
        produced = sorted(Path(outdir).glob("*.docx"))
        if produced:
            return produced[0]

        msg = ".hwp→docx 변환물(.docx)을 outdir 에서 찾지 못함 — 변환 무산"
        logger.warning("%s (%s)", msg, path)
        warnings.append(msg)
        return None

    # ─────────────────────────────────────────
    # 내부: 변환된 .docx → 구조 노드
    # ─────────────────────────────────────────

    def _parse_docx_structured(self, docx_path: Path, warnings: list[str]) -> list[DocumentNode]:
        """
        변환된 .docx 를 python-docx 로 열어 "구조 보존" 노드 목록으로 만든다.

        문서 본문(body)의 XML 자식을 문서 순서대로 훑어 문단/표가 섞인 원래 흐름을
        유지한다(doc.paragraphs/doc.tables 를 따로 도는 방식은 순서를 잃으므로
        쓰지 않는다). 문단 스타일로 SECTION/SUBHEADING 을 판정하고 heading_path 를
        누적해 이후 본문 PARAGRAPH 에 맥락 경로로 부여한다.

        fail-soft: docx 열기/요소 접근 실패는 경고만 쌓고 부분/빈 목록을 돌린다.
        """
        # 1) .docx 열기 — 변환물이 손상됐거나 docx 가 아니면 부분 결과(빈)+경고.
        try:
            document = Document(str(docx_path))
        except (PackageNotFoundError, OpcError, OSError, ValueError, KeyError) as e:
            msg = f"변환 docx 열기 실패: {type(e).__name__}: {e}"
            logger.warning("%s (%s)", msg, docx_path)
            warnings.append(msg)
            return []

        # 지연 import — python-docx 의 문단/표 래퍼. 모듈 상단 import 시 docx 미설치
        # 호스트에서 import 자체가 깨지지 않도록 여기서 가져온다(파서는 .hwp 가
        # 실제로 들어와 parse() 가 호출될 때만 이 경로를 탄다).
        try:
            from docx.table import Table
            from docx.text.paragraph import Paragraph
        except ImportError as e:
            msg = f"python-docx 내부 모듈 import 실패: {type(e).__name__}: {e}"
            logger.warning("%s (%s)", msg, docx_path)
            warnings.append(msg)
            return []

        nodes: list[DocumentNode] = []
        order = 0
        # 현재까지의 제목 경로(heading_path). 본문 문단에 맥락으로 전파한다.
        heading_path: tuple[str, ...] = ()

        # body 의 직속 자식(<w:p> 문단, <w:tbl> 표)을 문서 순서대로 순회.
        try:
            body = document.element.body
            body_children = list(body.iterchildren())
        except (AttributeError, TypeError) as e:
            msg = f"docx 본문(body) 접근 실패: {type(e).__name__}: {e}"
            logger.warning("%s (%s)", msg, docx_path)
            warnings.append(msg)
            return []

        for child in body_children:
            tag = child.tag
            # 문단(<w:p>) — 텍스트/스타일을 읽어 PARAGRAPH/SECTION/SUBHEADING 분류.
            if tag == qn("w:p"):
                made, order, heading_path = self._handle_paragraph(
                    Paragraph(child, document), order, heading_path, warnings
                )
                if made is not None:
                    nodes.append(made)
            # 표(<w:tbl>) — 행/열 보존 TABLE 노드.
            elif tag == qn("w:tbl"):
                table_node = self._table_node(Table(child, document), order, heading_path, warnings)
                if table_node is not None:
                    nodes.append(table_node)
                    order += 1
            # 그 외(섹션 속성 <w:sectPr> 등)는 본문이 아니므로 건너뛴다.

        return nodes

    def _handle_paragraph(
        self,
        para,
        order: int,
        heading_path: tuple[str, ...],
        warnings: list[str],
    ) -> tuple[DocumentNode | None, int, tuple[str, ...]]:
        """
        문단 1개를 노드로 변환하고, 갱신된 (order, heading_path)를 함께 돌린다.

        - 빈 문단(공백만)은 노드 없이 건너뛴다(검색 노이즈 방지).
        - 스타일이 "Heading 1" → SECTION: heading_path 를 (제목,) 으로 재설정.
        - 스타일이 "Heading N"(N≥2) → SUBHEADING: heading_path 의 N-1 깊이로 보정.
        - 일반 문단 → PARAGRAPH: 현재 heading_path 를 맥락으로 부여.

        문단 텍스트/스타일 접근 실패는 경고만 남기고 건너뛴다(fail-soft).
        반환: (생성 노드 또는 None, 갱신 order, 갱신 heading_path).
        """
        try:
            text = (para.text or "").strip()
        except (AttributeError, ValueError) as e:
            warnings.append(f"문단 텍스트 접근 실패 — 건너뜀 ({type(e).__name__}: {e})")
            return None, order, heading_path

        if not text:
            # 빈 문단 — 노드/순서 변화 없음.
            return None, order, heading_path

        # 스타일 이름(예: "Heading 1", "Normal", "제목 1" 등). 접근 실패는 무명 처리.
        try:
            style_name = (para.style.name or "") if para.style is not None else ""
        except (AttributeError, ValueError):
            style_name = ""

        heading_level = self._heading_level(style_name)

        if heading_level == 1:
            # 대제목 — heading_path 를 이 제목만으로 재설정(새 섹션 시작).
            heading_path = (text,)
            node = DocumentNode(
                element_type=ElementType.SECTION,
                text=text,
                heading_path=heading_path,
                page=None,
                order=order,
            )
            return node, order + 1, heading_path

        if heading_level >= 2:
            # 소제목 — 경로를 (level-1) 깊이로 자른 뒤 이 제목을 덧붙인다.
            # 예: 기존 ("1장",) 에서 Heading 2 → ("1장", "1.1 절").
            depth = heading_level - 1
            base = heading_path[:depth]
            heading_path = (*base, text)
            node = DocumentNode(
                element_type=ElementType.SUBHEADING,
                text=text,
                heading_path=heading_path,
                page=None,
                order=order,
            )
            return node, order + 1, heading_path

        # 일반 본문 문단 — 현재 heading_path 를 맥락으로 부여.
        node = DocumentNode(
            element_type=ElementType.PARAGRAPH,
            text=text,
            heading_path=heading_path,
            page=None,
            order=order,
        )
        return node, order + 1, heading_path

    @staticmethod
    def _heading_level(style_name: str) -> int:
        """
        문단 스타일 이름에서 제목 레벨을 추출한다(0=본문, 1=대제목, N=소제목).

        LibreOffice 가 .hwp 의 제목을 .docx 표준 스타일 "Heading 1/2/3..." 로
        매핑하는 것이 일반적이다. 일부 로캘에서는 "제목 1" 같은 한글 이름이 올 수
        있어 둘 다 인식한다. 숫자가 붙지 않은 일반 스타일("Normal", "Title" 등)은
        0(본문)으로 본다 — 과탐 방지(애매하면 본문).
        """
        if not style_name:
            return 0
        name = style_name.strip().lower()
        # 영문 "heading N" / 한글 "제목 N" 패턴에서 끝의 숫자를 레벨로 본다.
        prefixes = ("heading", "제목")
        for prefix in prefixes:
            if name.startswith(prefix):
                tail = name[len(prefix) :].strip()
                if tail.isdigit():
                    level = int(tail)
                    # 비정상적으로 깊은 레벨은 9 로 클램프(heading_path 폭주 방지).
                    return min(level, 9)
        return 0

    @staticmethod
    def _table_node(
        table,
        order: int,
        heading_path: tuple[str, ...],
        warnings: list[str],
    ) -> DocumentNode | None:
        """
        docx 표 1개를 TABLE 노드로 변환한다(행/열 구조 보존).

        직렬화 형식은 다른 파서(PPTX/PDF/HWPX)와 동일하게 맞춘다: 각 행을 셀들을
        " | " 로 이어 한 줄로, 행은 줄바꿈으로 연결한다. 빈 표는 None(노드 생략).

        표/행/셀 접근 실패는 fail-soft: 셀 단위 실패는 빈칸으로 흡수하고, 표 전체
        접근이 깨지면 경고 후 None.
        """
        rows_text: list[str] = []
        try:
            rows = list(table.rows)
        except (AttributeError, ValueError, IndexError) as e:
            warnings.append(f"표 행 접근 실패 — 건너뜀 ({type(e).__name__}: {e})")
            return None

        for row in rows:
            cells: list[str] = []
            try:
                row_cells = list(row.cells)
            except (AttributeError, ValueError, IndexError):
                # 한 행 접근 실패는 그 행만 건너뛴다(표 전체를 버리지 않음).
                continue
            for cell in row_cells:
                # 셀 1개 접근 실패는 그 셀만 빈칸 처리.
                try:
                    cells.append((cell.text or "").strip())
                except (AttributeError, ValueError):
                    cells.append("")
            rows_text.append(" | ".join(cells))

        content = "\n".join(rt for rt in rows_text if rt.strip(" |"))
        if not content.strip():
            return None

        return DocumentNode(
            element_type=ElementType.TABLE,
            text=content,
            heading_path=heading_path,
            page=None,
            order=order,
        )

    # ─────────────────────────────────────────
    # 내부: 공통 헬퍼
    # ─────────────────────────────────────────

    @staticmethod
    def _cleanup_tmpdir(tmpdir: str) -> None:
        """임시 디렉토리(및 변환 .docx)를 통째로 정리한다(실패는 디버그 로그만)."""
        try:
            shutil.rmtree(tmpdir, ignore_errors=True)
        except OSError as e:
            # 정리 실패는 본류에 영향 없음 — 디버그 로그만 남긴다.
            logger.debug("임시 디렉토리 정리 실패(무시): %s (%s)", tmpdir, e)

    @staticmethod
    def _empty_tree(path: Path, warnings: list[str]) -> DocumentTree:
        """치명적 실패 시 돌려줄 빈 트리(파일명을 제목으로, 경고만 담는다)."""
        return DocumentTree(
            title=path.stem,
            source_path=str(path),
            nodes=(),
            doc_format="hwp",
            warnings=tuple(warnings),
        )
