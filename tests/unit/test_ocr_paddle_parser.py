"""
core/ingest/parsers/ocr_paddle.py — PaddleOcrParser 단위 테스트.

대상: v7.3 단계 8 — 스캔 PDF/이미지 OCR(PaddleOCR, 고품질·한국어 특화, GPU 권장).

검증 의도(일곱 갈래 — 작업 지시 매핑):
  1. can_parse — fail-closed: 확장자(.png/.jpg/.tiff/.pdf) + 매직바이트까지 봐야 True.
     깨진 시그니처 / 미지원 확장자 / 없는 파일은 False. requires_gpu=True 확인.
  2. parse 정상(mock) — _ensure_ocr 를 stub 로 대체해 predict 가 rec_texts/rec_scores
     를 돌려주게 만들고, 이미지 → PARAGRAPH 노드(text 줄바꿈 합치기, page=1,
     doc_format=ocr_image)로 매핑되는지 확인.
  3. 빈 결과 — rec_texts=[] → 노드 0 + "비어 있음" 경고.
  4. fail-soft — import 실패(ImportError) / 모델 초기화 실패(RuntimeError) /
     OCR 추론 예외(ValueError 등) 각각 → 빈/부분 트리 + warning, 예외 비전파.
  5. 스캔 PDF(mock) — pypdfium2 렌더를 mock 하고 predict stub → 페이지별 노드,
     page 매핑, 한 페이지만 OCR 실패 시 그 페이지만 skip(나머지 계속).
  6. config — OcrConfig 의 PaddleOCR 필드 기본값(paddle_lang 등), DPI 클램프.
  7. 등록 정책 — _gpu_available / _paddleocr_available 4조합 monkeypatch →
     PaddleOcrParser 등록/미등록 + priority(-5, Tesseract -10 보다 우선),
     get_for_path 로 이미지=Paddle 우선 / .pdf=디지털(pdfplumber) 우선.

mock 전략 (왜 _ensure_ocr 를 patch 하는가):
  실 paddleocr 실행은 금지(모델 다운로드 + Windows KMP/OpenMP 불안정)다. 파서는
  OCR 호출을 _ocr_pil() 한 곳으로 모으고, 그 안에서 `ocr.predict(arr)` 만 부른다
  (ocr 는 _ensure_ocr() 가 만든 PaddleOCR 인스턴스). 따라서:
    - 정상/빈결과/PDF: _ensure_ocr 를 patch 해 predict 를 가진 stub 를 돌려주면
      실 paddleocr 없이도 변환 로직(rec_texts→PARAGRAPH, page 매핑)을 결정론적으로
      검증할 수 있다. _ocr_pil 은 numpy/Pillow 만 쓰므로 실 패키지 그대로 통과한다.
    - fail-soft: _ensure_ocr 를 side_effect 로 ImportError/RuntimeError 를 내게 하면
      parse() 의 import/초기화 fail-soft 분기를, OCR 단계 예외는 stub.predict 가
      ValueError 를 내게 해 추론 fail-soft 분기를 각각 정확히 친다.
  patch 대상은 인스턴스 메서드(parser._ensure_ocr)라 다른 테스트로 누수가 없다.

config 격리 (왜 _cfg 를 직접 주입하는가):
  _ensure_config 는 첫 parse() 때 OcrConfig 를 읽어 self._cfg 에 캐시한다. 변환/
  fail-soft 테스트는 실 config 의존을 피하려고 parser._cfg 를 미리 채워
  _ensure_config 를 단락(short-circuit)시킨다(테스트 간 격리 — FIRST 의 Independent).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from PIL import Image, ImageDraw

import mcp_servers.docingest_server as docingest_mod
from core.config import OcrConfig
from core.ingest.parsers.ocr_paddle import PaddleOcrParser
from core.ingest.parsers.ocr_tesseract import TesseractParser
from core.ingest.parsers.pdf_plumber import PdfPlumberParser
from core.ingest.types import ElementType


# ─────────────────────────────────────────────
# fixture 생성 헬퍼 — 실 OCR 은 mock 이므로 픽셀 내용은 무의미하다.
# (단, PIL Image.open/load 와 pypdfium2 렌더 경로는 실제로 통과하므로
#  "진짜 디코딩 가능한" 이미지/PDF 바이트가 필요하다.)
# ─────────────────────────────────────────────
def _make_image(text: str = "x") -> Image.Image:
    """흰 배경에 검은 글자를 그린 작은 RGB 이미지를 만든다(디코딩 가능한 실 픽셀용)."""
    img = Image.new("RGB", (320, 120), "white")
    draw = ImageDraw.Draw(img)
    draw.text((10, 40), text, fill="black")
    return img


def _save_png(tmp_path: Path, name: str = "ocr.png") -> Path:
    """디코딩 가능한 PNG 를 저장하고 경로를 돌려준다."""
    path = tmp_path / name
    _make_image().save(path)
    return path


def _save_multipage_pdf(tmp_path: Path, page_count: int = 3) -> Path:
    """여러 페이지 이미지를 PIL save_all 로 묶어 진짜 '%PDF'(스캔 PDF)를 만든다.

    왜 PIL save_all 인가: reportlab/img2pdf 없이도 PIL 은 이미지 → 멀티페이지 PDF 를
    네이티브로 쓸 수 있다. 결과는 진짜 '%PDF' 헤더라 can_parse 와 pypdfium2 렌더
    경로를 실제로 통과한다(OCR 만 mock — 바이너리 미커밋)."""
    path = tmp_path / "scan.pdf"
    images = [_make_image(f"p{i}") for i in range(page_count)]
    images[0].save(path, save_all=True, append_images=images[1:])
    return path


# ─────────────────────────────────────────────
# predict stub — PaddleOCR 인스턴스를 흉내 내는 가짜
# ─────────────────────────────────────────────
class _StubOcr:
    """
    _ensure_ocr 가 돌려주는 PaddleOCR 인스턴스 자리의 가짜.

    _ocr_pil 은 numpy RGB 배열을 넘기며 `ocr.predict(arr)` 만 호출하고, 결과의
    각 항목에서 .get("rec_texts") / .get("rec_scores") 로 라인/점수를 꺼낸다.
    따라서 predict 는 dict 들의 리스트(또는 1회용 이터러블)를 돌려주면 충분하다.

    pages: predict 가 호출 순서대로 돌려줄 결과 리스트들. 각 원소는 dict 리스트.
    fail_on_call: 이 (1-base) 호출 차수에서 ValueError 를 던진다(페이지 OCR 실패 모사).
    """

    def __init__(
        self,
        pages: list[list[dict[str, Any]]],
        *,
        fail_on_call: int | None = None,
    ) -> None:
        self._pages = pages
        self._fail_on_call = fail_on_call
        self.calls = 0  # predict 호출 횟수(검증용)

    def predict(self, _arr: Any) -> list[dict[str, Any]]:
        """numpy 배열을 받아(무시) 미리 준비한 predict 결과를 순서대로 돌려준다."""
        idx = self.calls
        self.calls += 1
        if self._fail_on_call is not None and self.calls == self._fail_on_call:
            # PaddleOCR/paddle 추론 실패로 관찰되는 예외(_OCR_RUNTIME_ERRORS 중 하나).
            raise ValueError("테스트: 페이지 OCR 추론 실패 모사")
        # 호출 횟수가 준비한 결과보다 많으면 빈 결과로 폴백(방어적).
        if idx >= len(self._pages):
            return []
        return self._pages[idx]


def _primed_parser(dpi: int = 250) -> PaddleOcrParser:
    """_ensure_config 를 단락시키기 위해 _cfg 를 미리 채운 파서.

    _cfg = (paddle_lang, paddle_use_gpu, paddle_enable_mkldnn, dpi). 실 config
    로딩/부수효과 없이 dpi 만 통제한다(스캔 PDF 렌더 scale 계산에 쓰인다)."""
    parser = PaddleOcrParser()
    parser._cfg = ("korean", True, False, dpi)
    return parser


# 한 줄 결과(rec_texts/rec_scores)를 가진 predict 결과 1건을 만든다.
def _rec(texts: list[str], scores: list[float] | None = None) -> list[dict[str, Any]]:
    """predict 결과 1건 = [{"rec_texts": [...], "rec_scores": [...]}] 형태로 감싼다."""
    if scores is None:
        scores = [0.99] * len(texts)
    return [{"rec_texts": texts, "rec_scores": scores}]


# ─────────────────────────────────────────────
# 1) can_parse — fail-closed + 정체성 플래그
# ─────────────────────────────────────────────
def test_can_parse_valid_png_returns_true(tmp_path: Path) -> None:
    """정상 PNG(매직바이트 일치)는 True."""
    path = _save_png(tmp_path)
    assert PaddleOcrParser().can_parse(path) is True


def test_can_parse_valid_jpg_returns_true(tmp_path: Path) -> None:
    """정상 JPEG(\\xff\\xd8\\xff SOI 마커)는 True."""
    path = tmp_path / "p.jpg"
    _make_image().save(path, format="JPEG")
    assert PaddleOcrParser().can_parse(path) is True


def test_can_parse_valid_tiff_returns_true(tmp_path: Path) -> None:
    """정상 TIFF(II*\\0 또는 MM\\0* 시그니처)는 True."""
    path = tmp_path / "p.tiff"
    _make_image().save(path, format="TIFF")
    assert PaddleOcrParser().can_parse(path) is True


def test_can_parse_valid_pdf_returns_true(tmp_path: Path) -> None:
    """정상 PDF('%PDF' 시그니처)는 True."""
    path = _save_multipage_pdf(tmp_path, page_count=1)
    assert PaddleOcrParser().can_parse(path) is True


def test_can_parse_broken_png_magic_returns_false(tmp_path: Path) -> None:
    """확장자는 .png 지만 매직바이트가 PNG 시그니처가 아니면 False(fail-closed)."""
    fake = tmp_path / "fake.png"
    fake.write_bytes(b"NOT-A-PNG-FILE-AT-ALL")
    assert PaddleOcrParser().can_parse(fake) is False


def test_can_parse_broken_pdf_magic_returns_false(tmp_path: Path) -> None:
    """확장자는 .pdf 지만 '%PDF' 로 시작하지 않으면 False."""
    fake = tmp_path / "fake.pdf"
    fake.write_bytes(b"not a pdf header")
    assert PaddleOcrParser().can_parse(fake) is False


def test_can_parse_unsupported_extension_returns_false(tmp_path: Path) -> None:
    """지원하지 않는 확장자(.bmp)는 내용과 무관하게 False(확장자 우선 거부)."""
    png = _save_png(tmp_path)
    other = tmp_path / "img.bmp"
    other.write_bytes(png.read_bytes())  # PNG 본문이지만 확장자가 .bmp
    assert PaddleOcrParser().can_parse(other) is False


def test_can_parse_missing_file_returns_false(tmp_path: Path) -> None:
    """없는 파일은 OSError 를 삼키고 False(fail-closed)."""
    assert PaddleOcrParser().can_parse(tmp_path / "nope.png") is False


def test_supported_extensions_and_requires_gpu() -> None:
    """정체성 플래그 — 이미지 확장자 + .pdf 지원, requires_gpu=True(고품질 GPU OCR)."""
    parser = PaddleOcrParser()
    assert parser.supported_extensions == (".png", ".jpg", ".jpeg", ".tiff", ".tif", ".pdf")
    assert parser.requires_gpu is True


# ─────────────────────────────────────────────
# 2) parse 정상 (mock) — 이미지
# ─────────────────────────────────────────────
@pytest.mark.asyncio
async def test_parse_image_mock_maps_to_single_paragraph(tmp_path: Path) -> None:
    """_ensure_ocr 를 stub: predict→rec_texts 2줄을 줄바꿈으로 합쳐 PARAGRAPH 1개.

    page=1, order=0, doc_format=ocr_image, 경고 없음 을 확인한다(변환 경로 검증)."""
    path = _save_png(tmp_path)
    parser = _primed_parser()
    stub = _StubOcr(pages=[_rec(["안녕", "넥서스"], [0.96, 0.98])])

    with patch.object(parser, "_ensure_ocr", return_value=stub):
        tree = await parser.parse(path)

    assert tree.doc_format == "ocr_image"
    assert tree.title == path.stem
    assert tree.warnings == ()
    assert len(tree.nodes) == 1
    node = tree.nodes[0]
    assert node.element_type == ElementType.PARAGRAPH
    assert node.text == "안녕\n넥서스"  # rec_texts 를 줄바꿈으로 합침
    assert node.page == 1
    assert node.order == 0
    # _ocr_pil 은 predict 를 정확히 1번 호출한다(이미지 1장).
    assert stub.calls == 1


# ─────────────────────────────────────────────
# 3) 빈 결과 — 노드 0 + 경고
# ─────────────────────────────────────────────
@pytest.mark.asyncio
async def test_parse_image_mock_empty_rec_texts_produces_no_node(tmp_path: Path) -> None:
    """rec_texts=[] (인식 라인 없음) → 노드를 만들지 않고 '비어 있음' 경고만 남긴다."""
    path = _save_png(tmp_path)
    parser = _primed_parser()
    stub = _StubOcr(pages=[_rec([], [])])  # 빈 인식 결과

    with patch.object(parser, "_ensure_ocr", return_value=stub):
        tree = await parser.parse(path)

    assert tree.doc_format == "ocr_image"
    assert tree.nodes == ()
    assert any("비어 있음" in w for w in tree.warnings)


# ─────────────────────────────────────────────
# 4) fail-soft — import / 초기화 / 추론 예외
# ─────────────────────────────────────────────
@pytest.mark.asyncio
async def test_parse_import_error_fails_soft(tmp_path: Path) -> None:
    """paddleocr import 실패(ImportError) → 빈 트리 + 경고, 예외 비전파.

    _ensure_ocr 의 지연 `from paddleocr import PaddleOCR` 가 미설치로 실패하는 상황을
    side_effect=ImportError 로 모사한다. parse() 의 ImportError 절이 흡수해야 한다."""
    path = _save_png(tmp_path)
    parser = _primed_parser()

    err = ImportError("No module named 'paddleocr'")
    with patch.object(parser, "_ensure_ocr", side_effect=err):
        tree = await parser.parse(path)

    assert tree.nodes == ()
    # parse() 의 import 실패 분기는 doc_format="ocr" 빈 트리를 돌려준다.
    assert tree.doc_format == "ocr"
    assert any("import 실패" in w for w in tree.warnings)


@pytest.mark.asyncio
async def test_parse_init_runtime_error_fails_soft(tmp_path: Path) -> None:
    """모델 초기화 실패(RuntimeError, 에어갭 모델 미배치 등) → 빈 트리 + 경고, 비전파.

    _ensure_ocr 가 PaddleOCR 생성 중 RuntimeError 를 던지는 상황을 모사한다.
    parse() 의 _OCR_RUNTIME_ERRORS(RuntimeError 포함) 절이 흡수해야 한다."""
    path = _save_png(tmp_path)
    parser = _primed_parser()

    with patch.object(parser, "_ensure_ocr", side_effect=RuntimeError("모델 가중치 적재 실패")):
        tree = await parser.parse(path)

    assert tree.nodes == ()
    assert tree.doc_format == "ocr"
    assert any("초기화 실패" in w for w in tree.warnings)


@pytest.mark.asyncio
async def test_parse_image_ocr_inference_error_fails_soft(tmp_path: Path) -> None:
    """OCR 추론 예외(predict 가 ValueError) → 이미지 OCR 실패 경고 + 노드 0, 비전파.

    _ensure_ocr 는 정상 stub 를 돌려주되, stub.predict 가 첫 호출에서 ValueError 를
    던지게 해 _parse_image 의 OCR 단계 fail-soft 절을 친다."""
    path = _save_png(tmp_path)
    parser = _primed_parser()
    stub = _StubOcr(pages=[], fail_on_call=1)  # 첫 predict 호출에서 ValueError

    with patch.object(parser, "_ensure_ocr", return_value=stub):
        tree = await parser.parse(path)

    assert tree.nodes == ()
    assert tree.doc_format == "ocr_image"
    assert any("이미지 OCR 실패" in w for w in tree.warnings)


# ─────────────────────────────────────────────
# 5) 스캔 PDF (mock) — 페이지 매핑 + 페이지 단위 skip
# ─────────────────────────────────────────────
@pytest.mark.asyncio
async def test_parse_pdf_mock_maps_pages(tmp_path: Path) -> None:
    """스캔 PDF(2페이지) mock: pypdfium2 렌더는 실제로 통과, predict 만 stub.

    페이지별 결과를 순서대로 돌려줘 page=1/2 매핑과 order 단조 증가를 확인한다.
    (실 paddleocr 없이 변환 경로 — render→to_pil→_ocr_pil→노드 — 를 검증.)"""
    path = _save_multipage_pdf(tmp_path, page_count=2)
    parser = _primed_parser()
    stub = _StubOcr(pages=[_rec(["페이지 일"]), _rec(["페이지 이"])])

    with patch.object(parser, "_ensure_ocr", return_value=stub):
        tree = await parser.parse(path)

    assert tree.doc_format == "ocr_pdf"
    pages = sorted(n.page for n in tree.nodes)
    assert pages == [1, 2], f"페이지 매핑이 예상과 다름: {pages}"
    assert all(n.element_type == ElementType.PARAGRAPH for n in tree.nodes)
    texts = {n.text for n in tree.nodes}
    assert texts == {"페이지 일", "페이지 이"}
    # order 는 문서 전체 읽기 순서로 단조 증가.
    orders = [n.order for n in tree.nodes]
    assert orders == sorted(orders)


@pytest.mark.asyncio
async def test_parse_pdf_mock_skips_failed_page_only(tmp_path: Path) -> None:
    """스캔 PDF(3페이지) mock: 2번째 페이지 OCR 만 실패 → 그 페이지만 skip(나머지 계속).

    predict 2번째 호출에서 ValueError 를 던지게 해 _ocr_pdf_page 의 페이지 단위
    fail-soft 를 친다. 결과 page={1,3} + 해당 페이지 경고가 남고 예외는 전파되지 않는다."""
    path = _save_multipage_pdf(tmp_path, page_count=3)
    parser = _primed_parser()
    # 호출 순서: p1=정상, p2=ValueError(skip), p3=정상.
    stub = _StubOcr(
        pages=[_rec(["일 내용"]), [], _rec(["삼 내용"])],
        fail_on_call=2,
    )

    with patch.object(parser, "_ensure_ocr", return_value=stub):
        tree = await parser.parse(path)

    assert tree.doc_format == "ocr_pdf"
    pages = sorted(n.page for n in tree.nodes)
    assert pages == [1, 3], f"실패한 2페이지만 생략되어야 함: {pages}"
    texts = {n.text for n in tree.nodes}
    assert texts == {"일 내용", "삼 내용"}
    # 빈 페이지가 생략돼도 살아남은 노드의 order 는 0,1 로 연속 재부여.
    orders = sorted(n.order for n in tree.nodes)
    assert orders == [0, 1]
    # 2페이지 실패 경고가 남는다(나머지 페이지는 계속 처리됨 — fail-soft).
    assert any("페이지 2" in w and "OCR 실패" in w for w in tree.warnings)


@pytest.mark.asyncio
async def test_parse_pdf_mock_all_empty_warns(tmp_path: Path) -> None:
    """스캔 PDF 모든 페이지가 빈 인식 결과 → 노드 0 + '비어 있음' 경고."""
    path = _save_multipage_pdf(tmp_path, page_count=2)
    parser = _primed_parser()
    stub = _StubOcr(pages=[_rec([], []), _rec([], [])])

    with patch.object(parser, "_ensure_ocr", return_value=stub):
        tree = await parser.parse(path)

    assert tree.doc_format == "ocr_pdf"
    assert tree.nodes == ()
    assert any("비어 있음" in w for w in tree.warnings)


# ─────────────────────────────────────────────
# 6) config — PaddleOCR 필드 기본값 / DPI 클램프
# ─────────────────────────────────────────────
def test_ocr_config_paddle_defaults() -> None:
    """OcrConfig 의 PaddleOCR 필드 기본값 — paddle_lang=korean, use_gpu=True, mkldnn=False."""
    cfg = OcrConfig()
    assert cfg.paddle_lang == "korean"  # tesseract "kor" 과 다른 코드
    assert cfg.paddle_use_gpu is True
    assert cfg.paddle_enable_mkldnn is False
    # dpi 는 tesseract 파서와 공유하는 렌더 해상도(기본 250).
    assert cfg.dpi == 250


def test_ensure_config_reads_paddle_defaults_and_clamps_dpi() -> None:
    """_ensure_config 가 OcrConfig 의 paddle_* 기본값을 읽고 dpi 를 [72,600] 으로 클램프.

    기본 dpi=250 은 범위 안이므로 그대로 250 이 캐시돼야 한다."""
    parser = PaddleOcrParser()
    paddle_lang, paddle_use_gpu, paddle_enable_mkldnn, dpi = parser._ensure_config()
    assert paddle_lang == "korean"
    assert paddle_use_gpu is True
    assert paddle_enable_mkldnn is False
    assert dpi == 250


def test_ensure_config_dpi_clamp_low(monkeypatch: pytest.MonkeyPatch) -> None:
    """dpi 가 하한(72) 미만이면 72 로 클램프(NEXUS_OCR__DPI=10 → 72)."""
    monkeypatch.setenv("NEXUS_OCR__DPI", "10")
    parser = PaddleOcrParser()
    _lang, _gpu, _mkldnn, dpi = parser._ensure_config()
    assert dpi == 72


def test_ensure_config_dpi_clamp_high(monkeypatch: pytest.MonkeyPatch) -> None:
    """dpi 가 상한(600) 초과면 600 으로 클램프(NEXUS_OCR__DPI=9999 → 600)."""
    monkeypatch.setenv("NEXUS_OCR__DPI", "9999")
    parser = PaddleOcrParser()
    _lang, _gpu, _mkldnn, dpi = parser._ensure_config()
    assert dpi == 600


def test_ensure_config_paddle_lang_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """NEXUS_OCR__PADDLE_LANG 로 인식 언어를 오버라이드할 수 있다(en 단일)."""
    monkeypatch.setenv("NEXUS_OCR__PADDLE_LANG", "en")
    parser = PaddleOcrParser()
    lang, _gpu, _mkldnn, _dpi = parser._ensure_config()
    assert lang == "en"


def test_ensure_config_caches_result() -> None:
    """_ensure_config 는 첫 호출 결과를 캐시해 동일 객체를 재반환한다(지연 1회 적용)."""
    parser = PaddleOcrParser()
    first = parser._ensure_config()
    second = parser._ensure_config()
    assert first is second


# ─────────────────────────────────────────────
# 7) 등록 정책 — _gpu_available / _paddleocr_available 조합 + priority + 라우팅
# ─────────────────────────────────────────────
class _DummyModelProvider:
    """build_app 이 만드는 LocalModelProvider 자리 — 어떤 인자든 받고 아무것도 안 한다."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    async def close(self) -> None:
        """리소스 정리(가짜)."""


class _CapturingPipeline:
    """DocumentIngestPipeline 자리에 끼워 생성자 인자(parser_registry)를 포착하는 가짜.

    (test_docingest_registry_routing.py 와 동일 패턴 — 라우팅 결정만 검증한다.)"""

    last_registry = None

    def __init__(self, parser_registry: Any, model_provider: Any, knowledge_store: Any) -> None:
        type(self).last_registry = parser_registry
        self._registry = parser_registry


async def _build_app_capture_registry(*, gpu: bool, paddle: bool):
    """_gpu_available/_paddleocr_available 를 고정하고 build_app 을 돌려 레지스트리를 포착.

    네트워크/모델/실 PG 미접속(asyncpg.create_pool → ConnectionError 폴백,
    LocalModelProvider 더미)으로 등록 라우팅 결정만 격리해 검증한다.
    build_app 은 함수 본문에서 from ... import 로 심볼을 끌어오므로 import 대상
    모듈(core.model.inference / core.ingest.pipeline)을 직접 patch 한다."""
    _CapturingPipeline.last_registry = None
    with (
        patch.object(docingest_mod, "_gpu_available", return_value=gpu),
        patch.object(docingest_mod, "_paddleocr_available", return_value=paddle),
        patch("asyncpg.create_pool", side_effect=ConnectionError("테스트: PG 미접속")),
        patch("core.model.inference.LocalModelProvider", _DummyModelProvider),
        patch("core.ingest.pipeline.DocumentIngestPipeline", _CapturingPipeline),
    ):
        await docingest_mod.build_app(api_key="local-key")

    assert _CapturingPipeline.last_registry is not None
    return _CapturingPipeline.last_registry


@pytest.mark.parametrize(
    ("gpu", "paddle"),
    [(False, False), (False, True), (True, False)],
)
@pytest.mark.asyncio
async def test_paddle_not_registered_unless_both_true(
    tmp_path: Path, gpu: bool, paddle: bool
) -> None:
    """GPU/paddleocr 중 하나라도 없으면 PaddleOcrParser 미등록 → 이미지는 Tesseract.

    이미지(.png)는 Paddle 미등록 시 유일한 후보인 TesseractParser 로 라우팅된다."""
    registry = await _build_app_capture_registry(gpu=gpu, paddle=paddle)

    png = tmp_path / "img.png"
    _make_image().save(png)
    parser = registry.get_for_path(png)

    assert not isinstance(parser, PaddleOcrParser)
    assert isinstance(parser, TesseractParser)


@pytest.mark.asyncio
async def test_paddle_registered_when_both_true_and_wins_image(tmp_path: Path) -> None:
    """GPU+paddleocr 둘 다 True → PaddleOcrParser 등록, 이미지(.png)에서 Tesseract 보다 우선.

    priority=-5(Paddle) > -10(Tesseract) 이므로 이미지의 1차 파서가 PaddleOcrParser 다."""
    registry = await _build_app_capture_registry(gpu=True, paddle=True)

    png = tmp_path / "img.png"
    _make_image().save(png)
    parser = registry.get_for_path(png)

    assert isinstance(parser, PaddleOcrParser)


@pytest.mark.asyncio
async def test_paddle_does_not_win_pdf_over_digital(tmp_path: Path) -> None:
    """GPU+paddleocr 둘 다 True 라도 .pdf 는 디지털 텍스트 파서(pdfplumber)가 우선.

    OCR(Paddle=-5/Tesseract=-10)은 .pdf 의 자동 대상이 아니라 명시 호출/폴백용이다.
    (GPU=True 면 Docling 도 등록되지만 이 호스트에 docling 모델이 없을 수 있어,
     '디지털 우선 = OCR 이 아님' 만 단언한다. 최소한 pdfplumber 가 PaddleOcrParser 를
     이긴다는 것은 priority 규칙상 보장된다.)"""
    registry = await _build_app_capture_registry(gpu=True, paddle=True)

    pdf = _save_multipage_pdf(tmp_path, page_count=1)
    parser = registry.get_for_path(pdf)

    # OCR 파서(Paddle/Tesseract)가 .pdf 의 1차로 선택돼서는 안 된다.
    assert not isinstance(parser, PaddleOcrParser)
    assert not isinstance(parser, TesseractParser)


@pytest.mark.asyncio
async def test_pdf_falls_to_pdfplumber_without_gpu(tmp_path: Path) -> None:
    """GPU 없음: Docling/Paddle 미등록 → .pdf 는 경량 PdfPlumberParser 로 라우팅."""
    registry = await _build_app_capture_registry(gpu=False, paddle=False)

    pdf = _save_multipage_pdf(tmp_path, page_count=1)
    parser = registry.get_for_path(pdf)

    assert isinstance(parser, PdfPlumberParser)
