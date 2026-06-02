"""
core/ingest/parsers/ocr_tesseract.py — TesseractParser 단위 테스트.

대상: v7.3 단계 8 — 스캔 PDF/이미지 OCR(Tesseract, CPU 경량).

검증 의도(다섯 갈래):
  1. can_parse — fail-closed: 확장자(.png/.jpg/.tiff/.pdf) + 매직바이트까지 봐야 True.
     깨진 시그니처 / 미지원 확장자 / 없는 파일은 False.
  2. 실 OCR(skipif 가드) — PIL 로 한글+영문 텍스트 이미지를 만들어 실제 tesseract 로
     OCR. 이미지→PARAGRAPH(page=1, doc_format=ocr_image), 스캔 PDF→페이지별
     PARAGRAPH(page 매핑, doc_format=ocr_pdf). tesseract/kor 데이터/한글 폰트가
     없는 호스트로의 이식성을 위해 skipif 로 가드한다.
  3. 변환 로직(mock) — pytesseract.image_to_string 을 고정 텍스트로 mock 하여
     tesseract 없이도 이미지→노드 매핑·빈 페이지 노드 생략을 결정론적으로 검증.
  4. fail-soft — 잘못된 tesseract_cmd → 노드 0 + warning(예외 비전파).
     손상 이미지/PDF(UnidentifiedImageError/PdfiumError) → warning.
  5. config/환경변수 — OcrConfig 기본값, NEXUS_OCR__* 오버라이드, DPI 클램프,
     _ensure_config 가 TESSDATA_PREFIX 를 설정하는지.

mock 전략(왜 image_to_string 을 patch 하는가):
  파서는 OCR 호출을 _ocr_image() 한 곳으로 모아 pytesseract.image_to_string 만
  호출한다. ocr_tesseract 모듈이 `import pytesseract` 한 뒤 모듈 경유로 부르므로
  patch 대상은 모듈에서 본 이름인 "core.ingest.parsers.ocr_tesseract.pytesseract"
  의 image_to_string 이다(import 위치 기준 patch — 다른 모듈의 pytesseract 를
  건드리지 않아 테스트 간 누수가 없다).

config 격리(왜 _cfg 를 직접 주입하는가):
  _ensure_config 는 첫 parse() 때 OcrConfig 를 읽어 self._cfg 에 캐시하고,
  그 부수효과로 pytesseract.pytesseract.tesseract_cmd 와 os.environ 을 바꾼다.
  변환/ fail-soft 테스트는 이 전역 부수효과와 실 config 의존을 피하려고
  parser._cfg 를 미리 채워 _ensure_config 를 단락(short-circuit)시킨다
  (테스트 간 전역 오염 방지 — FIRST 의 Independent).
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest
from PIL import Image, ImageDraw, ImageFont

from core.config import OcrConfig
from core.ingest.parsers.ocr_tesseract import TesseractParser
from core.ingest.types import ElementType


# ─────────────────────────────────────────────
# 실 OCR 가드 — tesseract 바이너리 + kor 데이터 + 한글 폰트가 있어야만 실행
# ─────────────────────────────────────────────
def _tesseract_cmd() -> str:
    """OcrConfig 기본 tesseract 실행 파일 경로(개발 기본값)."""
    return OcrConfig().tesseract_cmd


def _tessdata_dir() -> str:
    """OcrConfig 기본 tessdata 폴더 경로."""
    return OcrConfig().tessdata_dir


def _korean_font_path() -> str | None:
    """한글 글리프가 있는 시스템 폰트를 찾는다(없으면 None → 한글 OCR 테스트 skip)."""
    candidates = (
        r"C:\Windows\Fonts\malgun.ttf",  # 맑은 고딕
        r"C:\Windows\Fonts\gulim.ttc",  # 굴림
        r"C:\Windows\Fonts\batang.ttc",  # 바탕
    )
    for fp in candidates:
        if os.path.exists(fp):
            return fp
    return None


def _real_ocr_available() -> bool:
    """실 tesseract OCR + kor 데이터 + 한글 폰트가 모두 갖춰졌는지(이식성 가드)."""
    cmd = _tesseract_cmd()
    if not cmd or not os.path.exists(cmd):
        return False
    kor = os.path.join(_tessdata_dir(), "kor.traineddata")
    if not os.path.exists(kor):
        return False
    return _korean_font_path() is not None


# tesseract/kor/한글폰트 미설치 호스트에서는 실 OCR 테스트를 통째로 skip 한다.
_skip_no_ocr = pytest.mark.skipif(
    not _real_ocr_available(),
    reason="실 tesseract 바이너리 / kor.traineddata / 한글 폰트가 없어 실 OCR 검증을 건너뜀",
)


# ─────────────────────────────────────────────
# fixture 생성 헬퍼
# ─────────────────────────────────────────────
def _make_text_image(text: str, font_path: str | None = None, size: int = 64) -> Image.Image:
    """흰 배경에 검은 글자를 그린 이미지를 만든다(OCR 입력용).

    글자가 충분히 크고 대비가 높아야 인식률이 안정적이라, 큰 폰트 + 흑백 대비를
    쓴다. font_path 가 없으면 영문 전용 truetype(arial)로 폴백한다."""
    img = Image.new("RGB", (1000, 240), "white")
    draw = ImageDraw.Draw(img)
    font: ImageFont.ImageFont | ImageFont.FreeTypeFont
    if font_path is not None:
        font = ImageFont.truetype(font_path, size)
    else:
        try:
            font = ImageFont.truetype("arial.ttf", size)
        except OSError:
            font = ImageFont.load_default()
    draw.text((20, 80), text, fill="black", font=font)
    return img


def _save_png(
    tmp_path: Path, text: str, font_path: str | None = None, name: str = "ocr.png"
) -> Path:
    """텍스트 이미지를 PNG 로 저장하고 경로를 돌려준다."""
    path = tmp_path / name
    _make_text_image(text, font_path).save(path)
    return path


def _save_multipage_pdf(tmp_path: Path, texts: list[str], font_path: str | None = None) -> Path:
    """여러 페이지 텍스트 이미지를 PIL 로 묶어 '스캔 PDF' 를 만든다.

    왜 PIL save_all 인가: reportlab/img2pdf 없이도 PIL 은 이미지 → 멀티페이지 PDF 를
    네이티브로 쓸 수 있다. 결과는 진짜 '%PDF' 라 pypdfium2 가 렌더할 수 있고,
    각 페이지가 '글자 이미지'이므로 OCR 대상(스캔 PDF)으로 적합하다(바이너리 미커밋)."""
    path = tmp_path / "scan.pdf"
    images = [_make_text_image(t, font_path) for t in texts]
    images[0].save(path, save_all=True, append_images=images[1:])
    return path


# ─────────────────────────────────────────────
# 1) can_parse — fail-closed
# ─────────────────────────────────────────────
def test_can_parse_valid_png_returns_true(tmp_path: Path) -> None:
    """정상 PNG(매직바이트 일치)는 True."""
    path = _save_png(tmp_path, "hello")
    assert TesseractParser().can_parse(path) is True


def test_can_parse_valid_jpg_returns_true(tmp_path: Path) -> None:
    """정상 JPEG(\\xff\\xd8\\xff SOI 마커)는 True."""
    path = tmp_path / "p.jpg"
    _make_text_image("hello").save(path, format="JPEG")
    assert TesseractParser().can_parse(path) is True


def test_can_parse_valid_tiff_returns_true(tmp_path: Path) -> None:
    """정상 TIFF(II*\\0 또는 MM\\0* 시그니처)는 True."""
    path = tmp_path / "p.tiff"
    _make_text_image("hello").save(path, format="TIFF")
    assert TesseractParser().can_parse(path) is True


def test_can_parse_valid_pdf_returns_true(tmp_path: Path) -> None:
    """정상 PDF('%PDF' 시그니처)는 True."""
    path = _save_multipage_pdf(tmp_path, ["page one"])
    assert TesseractParser().can_parse(path) is True


def test_can_parse_broken_magic_returns_false(tmp_path: Path) -> None:
    """확장자는 .png 지만 매직바이트가 PNG 시그니처가 아니면 False(fail-closed)."""
    fake = tmp_path / "fake.png"
    fake.write_bytes(b"NOT-A-PNG-FILE-AT-ALL")
    assert TesseractParser().can_parse(fake) is False


def test_can_parse_broken_pdf_magic_returns_false(tmp_path: Path) -> None:
    """확장자는 .pdf 지만 '%PDF' 로 시작하지 않으면 False."""
    fake = tmp_path / "fake.pdf"
    fake.write_bytes(b"not a pdf header")
    assert TesseractParser().can_parse(fake) is False


def test_can_parse_unsupported_extension_returns_false(tmp_path: Path) -> None:
    """지원하지 않는 확장자(.bmp)는 내용과 무관하게 False(확장자 우선 거부)."""
    png = _save_png(tmp_path, "hi")
    other = tmp_path / "img.bmp"
    other.write_bytes(png.read_bytes())  # PNG 본문이지만 확장자가 .bmp
    assert TesseractParser().can_parse(other) is False


def test_can_parse_missing_file_returns_false(tmp_path: Path) -> None:
    """없는 파일은 OSError 를 삼키고 False(fail-closed)."""
    assert TesseractParser().can_parse(tmp_path / "nope.png") is False


def test_supported_extensions_and_no_gpu() -> None:
    """정체성 플래그 — 이미지 확장자 + .pdf 지원, GPU 불필요(CPU 경량 OCR)."""
    parser = TesseractParser()
    assert parser.supported_extensions == (".png", ".jpg", ".jpeg", ".tiff", ".tif", ".pdf")
    assert parser.requires_gpu is False


# ─────────────────────────────────────────────
# 2) 실 OCR (skipif 가드) — 이미지
# ─────────────────────────────────────────────
@_skip_no_ocr
@pytest.mark.asyncio
async def test_parse_real_image_produces_paragraph_node(tmp_path: Path) -> None:
    """실 tesseract: 한글+영문 이미지 → PARAGRAPH 노드(page=1, doc_format=ocr_image).

    한글 OCR 은 노이즈가 있을 수 있으므로 '한글 코드포인트(U+AC00~U+D7A3) 또는
    영문 토큰 포함' 으로 느슨하게 검증한다(인식 품질 자체가 아니라 변환 경로 검증)."""
    font = _korean_font_path()
    path = _save_png(tmp_path, "보안 정책 Report", font_path=font)

    tree = await TesseractParser().parse(path)

    assert tree.doc_format == "ocr_image"
    assert tree.title == "ocr"  # 파일명 stem
    assert len(tree.nodes) == 1
    node = tree.nodes[0]
    assert node.element_type == ElementType.PARAGRAPH
    assert node.page == 1
    text = node.text
    has_hangul = any(0xAC00 <= ord(c) <= 0xD7A3 for c in text)
    has_english = any(c.isalpha() and c.isascii() for c in text)
    assert has_hangul or has_english, f"OCR 결과에 한글/영문이 없음: {text!r}"


# ─────────────────────────────────────────────
# 3) 실 OCR (skipif 가드) — 스캔 PDF
# ─────────────────────────────────────────────
@_skip_no_ocr
@pytest.mark.asyncio
async def test_parse_real_scan_pdf_maps_pages(tmp_path: Path) -> None:
    """실 tesseract: 2페이지 스캔 PDF → 페이지별 PARAGRAPH(page 매핑, doc_format=ocr_pdf).

    영문은 인식이 안정적이므로 페이지 본문에 영문 토큰을 넣어 page=1/2 분리를
    결정론적으로 확인한다(한글 노이즈 의존 회피)."""
    font = _korean_font_path()
    path = _save_multipage_pdf(tmp_path, ["Page One Alpha", "Page Two Bravo"], font_path=font)

    tree = await TesseractParser().parse(path)

    assert tree.doc_format == "ocr_pdf"
    pages = {n.page for n in tree.nodes}
    assert pages == {1, 2}, f"페이지 매핑이 예상과 다름: {pages}"
    assert all(n.element_type == ElementType.PARAGRAPH for n in tree.nodes)
    # order 는 페이지를 넘어가도 단조 증가(문서 전체 읽기 순서).
    orders = [n.order for n in tree.nodes]
    assert orders == sorted(orders)


# ─────────────────────────────────────────────
# 4) 변환 로직 (mock) — tesseract 없이 매핑/빈페이지 생략 검증
# ─────────────────────────────────────────────
def _primed_parser(lang: str = "kor+eng", dpi: int = 250) -> TesseractParser:
    """_ensure_config 를 단락시키기 위해 _cfg 를 미리 채운 파서.

    (tesseract_cmd, lang, tessdata_dir, dpi) — cmd/tessdata 는 빈 문자열로 두어
    pytesseract 전역/os.environ 부수효과를 일으키지 않는다(테스트 격리)."""
    parser = TesseractParser()
    parser._cfg = ("", lang, "", dpi)
    return parser


_OCR_PATH = "core.ingest.parsers.ocr_tesseract.pytesseract.image_to_string"


@pytest.mark.asyncio
async def test_parse_image_mock_maps_to_single_paragraph(tmp_path: Path) -> None:
    """image_to_string 을 mock: 이미지 1장 → PARAGRAPH 1개(page=1, doc_format=ocr_image)."""
    path = _save_png(tmp_path, "ignored-pixels")  # 실제 픽셀은 mock 이라 무의미
    parser = _primed_parser()

    with patch(_OCR_PATH, return_value="가짜 OCR 텍스트 fixed"):
        tree = await parser.parse(path)

    assert tree.doc_format == "ocr_image"
    assert len(tree.nodes) == 1
    node = tree.nodes[0]
    assert node.element_type == ElementType.PARAGRAPH
    assert node.text == "가짜 OCR 텍스트 fixed"
    assert node.page == 1
    assert node.order == 0
    assert tree.warnings == ()


@pytest.mark.asyncio
async def test_parse_image_mock_empty_text_produces_no_node(tmp_path: Path) -> None:
    """OCR 결과가 공백뿐이면 노드를 만들지 않고 '비어 있음' 경고만 남긴다(노이즈 방지)."""
    path = _save_png(tmp_path, "x")
    parser = _primed_parser()

    with patch(_OCR_PATH, return_value="   \n  "):  # strip 후 빈 문자열
        tree = await parser.parse(path)

    assert tree.nodes == ()
    assert any("비어 있음" in w for w in tree.warnings)


@pytest.mark.asyncio
async def test_parse_pdf_mock_maps_pages_and_skips_empty(tmp_path: Path) -> None:
    """스캔 PDF(3페이지) mock: 2페이지만 텍스트, 1페이지는 공백 → 빈 페이지 노드 생략.

    페이지별 결과를 side_effect 로 제어해, 텍스트 페이지만 노드가 되고 page 번호가
    원본 페이지에 매핑되는지(빈 2페이지는 건너뛰어 page={1,3}) 확인한다."""
    path = _save_multipage_pdf(tmp_path, ["p1", "p2-empty", "p3"])
    parser = _primed_parser()

    # 호출 순서대로: page1=텍스트, page2=공백(생략), page3=텍스트.
    with patch(_OCR_PATH, side_effect=["페이지 일 내용", "   ", "page three body"]):
        tree = await parser.parse(path)

    assert tree.doc_format == "ocr_pdf"
    pages = {n.page for n in tree.nodes}
    assert pages == {1, 3}, f"빈 2페이지는 생략되어야 함: {pages}"
    # 빈 페이지가 생략돼도 order 는 0,1 로 연속(빈 노드만큼 건너뛴 게 아니라 재부여).
    orders = sorted(n.order for n in tree.nodes)
    assert orders == [0, 1]
    texts = {n.text for n in tree.nodes}
    assert texts == {"페이지 일 내용", "page three body"}


# ─────────────────────────────────────────────
# 5) fail-soft
# ─────────────────────────────────────────────
@pytest.mark.asyncio
async def test_parse_image_bad_cmd_fails_soft(tmp_path: Path) -> None:
    """잘못된 tesseract_cmd → 노드 0 + warning, 예외 비전파(fail-soft).

    실제 호출이 TesseractNotFoundError 를 던지지만 _parse_image 가 포착해
    경고로 환원해야 한다(mock 없이 잘못된 cmd 로 실 동작 검증).

    주의(예외 절 그늘짐): TesseractNotFoundError 는 FileNotFoundError →
    OSError 의 하위 타입이라, _parse_image 의 첫 except (OSError,
    UnidentifiedImageError) 절에 먼저 걸린다. 따라서 미설치 케이스의 경고는
    뒤쪽 'OCR 실패' 절이 아니라 앞쪽 '이미지 열기 실패' 메시지로 남는다.
    (fail-soft 계약 자체는 만족 — 노드 0 + 경고 + 예외 비전파.)"""
    path = _save_png(tmp_path, "irrelevant")
    parser = TesseractParser()
    # 존재하지 않는 실행 파일 경로를 강제 주입 → image_to_string 이 미설치로 인식.
    bad_cmd = str(tmp_path / "no-such-tesseract.exe")
    parser._cfg = (bad_cmd, "eng", "", 250)
    # cmd 부수효과는 _ensure_config 가 단락되어 적용되지 않으므로, 명시 적용.
    import pytesseract

    with patch.object(pytesseract.pytesseract, "tesseract_cmd", bad_cmd):
        tree = await parser.parse(path)

    assert tree.nodes == ()
    assert tree.doc_format == "ocr_image"
    assert len(tree.warnings) >= 1
    # 미설치 예외가 OSError 그늘에 걸려 '이미지 열기 실패' 로 남는다(위 주의 참조).
    assert any(
        "이미지 열기 실패" in w or "OCR 실패" in w or "비어 있음" in w for w in tree.warnings
    )


@pytest.mark.asyncio
async def test_parse_corrupted_image_fails_soft(tmp_path: Path) -> None:
    """손상 이미지(매직바이트는 PNG 지만 본문이 깨짐) → UnidentifiedImageError 흡수 + warning.

    can_parse 는 매직바이트만 보므로 통과하지만, Image.open/load 단계에서 깨진다.
    파서가 (OSError/UnidentifiedImageError) 를 포착해 빈 트리 + 경고로 환원해야 한다."""
    broken = tmp_path / "broken.png"
    # 유효 PNG 시그니처 + 쓰레기 본문(디코딩 불가).
    broken.write_bytes(b"\x89PNG\r\n\x1a\n" + b"garbage-not-a-real-png-body" * 4)
    parser = _primed_parser()

    tree = await parser.parse(broken)

    assert tree.nodes == ()
    assert len(tree.warnings) >= 1
    assert any("이미지 열기 실패" in w for w in tree.warnings)


@pytest.mark.asyncio
async def test_parse_corrupted_pdf_fails_soft(tmp_path: Path) -> None:
    """손상 PDF('%PDF' 헤더만 있고 본문 깨짐) → PdfiumError 흡수 + warning(예외 비전파)."""
    broken = tmp_path / "broken.pdf"
    broken.write_bytes(b"%PDF-1.4\nthis is not a valid pdf body at all\n")
    parser = _primed_parser()

    # can_parse 는 매직바이트만 보므로 True.
    assert parser.can_parse(broken) is True

    tree = await parser.parse(broken)

    assert tree.doc_format == "ocr_pdf"
    assert tree.nodes == ()
    assert len(tree.warnings) >= 1
    # 문서 열기 실패 또는 (열려도) 빈 결과 경고 중 하나가 남는다.
    assert any("스캔 PDF 열기 실패" in w or "비어 있음" in w for w in tree.warnings)


# ─────────────────────────────────────────────
# 6) config / 환경변수 / DPI 클램프 / TESSDATA_PREFIX
# ─────────────────────────────────────────────
def test_ocr_config_defaults() -> None:
    """OcrConfig 기본값 — lang=kor+eng, dpi=250, tesseract_cmd 는 Windows 설치 경로."""
    cfg = OcrConfig()
    assert cfg.lang == "kor+eng"
    assert cfg.dpi == 250
    assert cfg.tesseract_cmd.endswith("tesseract.exe")
    # tessdata_dir 은 LOCALAPPDATA 하위 nexus_tessdata 로 끝난다(개발 기본값).
    assert cfg.tessdata_dir.endswith("nexus_tessdata")


def test_ensure_config_reads_defaults_and_clamps_dpi() -> None:
    """_ensure_config 가 OcrConfig 기본값을 읽고 dpi 를 [72,600] 으로 클램프한다.

    기본 dpi=250 은 범위 안이므로 그대로 250 이 캐시돼야 한다."""
    parser = TesseractParser()
    cmd, lang, _tessdata, dpi = parser._ensure_config()
    assert lang == "kor+eng"
    assert dpi == 250


def test_ensure_config_dpi_clamp_low(monkeypatch: pytest.MonkeyPatch) -> None:
    """dpi 가 하한(72) 미만이면 72 로 클램프된다(NEXUS_OCR__DPI=10 → 72)."""
    monkeypatch.setenv("NEXUS_OCR__DPI", "10")
    parser = TesseractParser()
    _cmd, _lang, _tessdata, dpi = parser._ensure_config()
    assert dpi == 72


def test_ensure_config_dpi_clamp_high(monkeypatch: pytest.MonkeyPatch) -> None:
    """dpi 가 상한(600) 초과면 600 으로 클램프된다(NEXUS_OCR__DPI=9999 → 600)."""
    monkeypatch.setenv("NEXUS_OCR__DPI", "9999")
    parser = TesseractParser()
    _cmd, _lang, _tessdata, dpi = parser._ensure_config()
    assert dpi == 600


def test_ensure_config_lang_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """NEXUS_OCR__LANG 로 언어를 오버라이드할 수 있다(eng 단일)."""
    monkeypatch.setenv("NEXUS_OCR__LANG", "eng")
    parser = TesseractParser()
    _cmd, lang, _tessdata, _dpi = parser._ensure_config()
    assert lang == "eng"


def test_ensure_config_sets_tessdata_prefix(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """_ensure_config 가 tessdata_dir 을 os.environ['TESSDATA_PREFIX'] 로 설정한다.

    pytesseract 는 config 문자열이 아닌 TESSDATA_PREFIX 환경변수로 언어 데이터
    폴더를 받으므로(OS/공백 quoting 안전), 이 부수효과가 정확히 일어나는지 확인한다."""
    custom = str(tmp_path / "my_tessdata")
    monkeypatch.setenv("NEXUS_OCR__TESSDATA_DIR", custom)
    monkeypatch.delenv("TESSDATA_PREFIX", raising=False)

    parser = TesseractParser()
    _cmd, _lang, tessdata, _dpi = parser._ensure_config()

    assert tessdata == custom
    assert os.environ.get("TESSDATA_PREFIX") == custom


def test_ensure_config_applies_tesseract_cmd(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """tesseract_cmd 설정값이 pytesseract 전역(tesseract_cmd)에 적용된다.

    config 기본 cmd 가 비어 있지 않으므로 환경변수로 명시 오버라이드해 검증한다."""
    import pytesseract

    custom_cmd = str(tmp_path / "custom-tesseract.exe")
    monkeypatch.setenv("NEXUS_OCR__TESSERACT_CMD", custom_cmd)

    parser = TesseractParser()
    cmd, _lang, _tessdata, _dpi = parser._ensure_config()

    assert cmd == custom_cmd
    assert pytesseract.pytesseract.tesseract_cmd == custom_cmd


def test_ensure_config_caches_result() -> None:
    """_ensure_config 는 첫 호출 결과를 캐시해 동일 객체를 재반환한다(지연 1회 적용)."""
    parser = TesseractParser()
    first = parser._ensure_config()
    second = parser._ensure_config()
    assert first is second
