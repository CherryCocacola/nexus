"""
DocumentExport 도구의 포맷별 렌더러 — 마크다운 유사 텍스트를 실제 문서 파일로 변환.

지원 포맷: md, txt, docx, pptx, hwpx (전부 순수 파이썬 라이브러리 = 에어갭 안전).

설계:
  1) parse_blocks() 로 입력 텍스트를 공통 "블록" 목록으로 한 번만 파싱한다.
     (제목/글머리표/문단 3종) → 각 포맷 렌더러는 이 블록을 소비만 하므로
     포맷마다 파싱을 중복하지 않는다.
  2) 무거운 라이브러리(python-docx/pptx/hwpx) import 는 각 렌더러 함수 "안"에서
     지연(lazy) import 한다. 도구 로드 시점에 라이브러리가 없어도 죽지 않고,
     실제로 그 포맷을 요청할 때만 필요하다.

입력 문법(모델이 자연스럽게 쓰는 마크다운 부분집합):
  - "# / ## / ###"  → 제목(레벨 1~3)
  - "- 항목" / "* 항목" → 글머리표(연속 줄은 한 묶음)
  - "**굵게**"       → 굵은 글씨(포맷이 지원하면 반영, 아니면 평문)
  - 그 외 비어있지 않은 줄 → 일반 문단
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

# 블록 표현:
#   ("h", level:int, text:str)  — 제목
#   ("bullet", items:list[str]) — 글머리표 묶음
#   ("p", text:str)             — 일반 문단
Block = tuple


def strip_md(text: str) -> str:
    """평문 포맷(txt)·굵기 미지원 렌더러용 — '**' 굵기 표시를 제거한다."""
    return re.sub(r"\*\*(.+?)\*\*", r"\1", text)


def parse_blocks(content: str) -> list[Block]:
    """
    마크다운 유사 텍스트를 공통 블록 목록으로 파싱한다.

    연속된 글머리표 줄은 하나의 ("bullet", [...]) 로 묶는다 — 각 포맷에서
    <ul>/불릿 문단 그룹으로 자연스럽게 렌더하기 위함이다.
    """
    blocks: list[Block] = []
    bullet_buf: list[str] = []

    def flush_bullets() -> None:
        # 모아둔 글머리표 항목이 있으면 하나의 블록으로 확정한다.
        nonlocal bullet_buf
        if bullet_buf:
            blocks.append(("bullet", bullet_buf))
            bullet_buf = []

    for raw in content.splitlines():
        line = raw.rstrip()
        stripped = line.strip()
        if not stripped:
            flush_bullets()
            continue

        # 제목: 앞의 '#' 개수로 레벨 결정(최대 3)
        m = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if m:
            flush_bullets()
            level = min(len(m.group(1)), 3)
            blocks.append(("h", level, m.group(2).strip()))
            continue

        # 글머리표: "- " 또는 "* " 로 시작
        m = re.match(r"^[-*]\s+(.*)$", stripped)
        if m:
            bullet_buf.append(m.group(1).strip())
            continue

        # 그 외 → 일반 문단
        flush_bullets()
        blocks.append(("p", stripped))

    flush_bullets()
    return blocks


# ─────────────────────────────────────────────
# 평문 계열: md / txt
# ─────────────────────────────────────────────
def render_md(content: str, path: Path, title: str = "") -> None:
    """md — 입력이 이미 마크다운이므로 (선택적 제목만 얹어) 그대로 저장한다."""
    body = f"# {title}\n\n{content}" if title else content
    path.write_text(body, encoding="utf-8")


def render_txt(content: str, path: Path, title: str = "") -> None:
    """txt — 굵기 표시(**)를 제거한 평문으로 저장한다."""
    body = f"{title}\n\n{content}" if title else content
    path.write_text(strip_md(body), encoding="utf-8")


# ─────────────────────────────────────────────
# DOCX (python-docx)
# ─────────────────────────────────────────────
def _docx_add_runs(paragraph: Any, text: str) -> None:
    """'**굵게**' 를 실제 굵은 런으로 바꿔 문단에 추가한다."""
    parts = re.split(r"\*\*(.+?)\*\*", text)
    for i, part in enumerate(parts):
        if part == "":
            continue
        run = paragraph.add_run(part)
        if i % 2 == 1:  # 캡처된 굵기 대상은 홀수 인덱스
            run.bold = True


def render_docx(content: str, path: Path, title: str = "") -> None:
    """DOCX 생성 — 제목/글머리표/굵기를 반영한다."""
    from docx import Document  # lazy import

    doc = Document()
    if title:
        doc.add_heading(title, level=0)

    for block in parse_blocks(content):
        if block[0] == "h":
            doc.add_heading(block[2], level=block[1])
        elif block[0] == "bullet":
            for item in block[1]:
                p = doc.add_paragraph(style="List Bullet")
                _docx_add_runs(p, item)
        else:  # "p"
            p = doc.add_paragraph()
            _docx_add_runs(p, block[1])

    doc.save(str(path))


# ─────────────────────────────────────────────
# PPTX (python-pptx)
# ─────────────────────────────────────────────
def render_pptx(content: str, path: Path, title: str = "") -> None:
    """
    PPTX 생성.

    슬라이드 구성 규칙:
      - 맨 앞에 제목 슬라이드 1장(제목 = title 또는 첫 제목 블록).
      - 이후 제목 블록(##/###)을 만날 때마다 새 본문 슬라이드를 시작하고,
        뒤따르는 글머리표/문단을 그 슬라이드 본문으로 채운다.
      - 제목 블록이 하나도 없으면 본문 슬라이드 1장에 전부 담는다.
    """
    from pptx import Presentation  # lazy import

    prs = Presentation()
    blocks = parse_blocks(content)

    # 제목 슬라이드
    first_heading = next((b[2] for b in blocks if b[0] == "h"), "")
    deck_title = title or first_heading or "문서"
    s0 = prs.slides.add_slide(prs.slide_layouts[0])
    s0.shapes.title.text = deck_title
    if len(s0.placeholders) > 1:
        s0.placeholders[1].text = "Nexus 생성 문서"

    # 본문 슬라이드들
    body_tf = None  # 현재 채우는 중인 본문 text_frame

    def new_body_slide(heading_text: str):
        # 제목+본문 레이아웃(1)으로 새 슬라이드를 만든다.
        slide = prs.slides.add_slide(prs.slide_layouts[1])
        slide.shapes.title.text = heading_text
        tf = slide.placeholders[1].text_frame
        tf.clear()
        return tf

    def add_body_line(tf, text: str, first_flag: list[bool]) -> None:
        # text_frame 의 첫 줄은 기본 문단을 재사용하고, 이후엔 add_paragraph.
        clean = strip_md(text)
        if first_flag[0]:
            tf.paragraphs[0].text = clean
            first_flag[0] = False
        else:
            tf.add_paragraph().text = clean

    first_line = [True]
    for block in blocks:
        if block[0] == "h":
            body_tf = new_body_slide(block[2])
            first_line = [True]
        else:
            if body_tf is None:  # 제목 블록 없이 본문이 먼저 온 경우
                body_tf = new_body_slide(deck_title)
                first_line = [True]
            if block[0] == "bullet":
                for item in block[1]:
                    add_body_line(body_tf, item, first_line)
            else:  # "p"
                add_body_line(body_tf, block[1], first_line)

    prs.save(str(path))


# ─────────────────────────────────────────────
# HWPX (python-hwpx)
# ─────────────────────────────────────────────
def render_hwpx(content: str, path: Path, title: str = "") -> None:
    """HWPX(한글) 생성 — hwpx.builder 고수준 API 사용."""
    from hwpx.builder import Bullet, Document, Heading, Paragraph, Section  # lazy

    children: list[Any] = []
    if title:
        children.append(Heading(level=1, text=title))

    for block in parse_blocks(content):
        if block[0] == "h":
            children.append(Heading(level=block[1], text=strip_md(block[2])))
        elif block[0] == "bullet":
            children.append(Bullet(items=[strip_md(x) for x in block[1]]))
        else:  # "p"
            children.append(Paragraph(text=strip_md(block[1])))

    doc = Document(sections=[Section(children=children)])
    doc.save_to_path(str(path))


# 포맷명 → (렌더 함수, MIME 타입) 매핑.
# 도구/다운로드 라우트가 공유하는 단일 진실원(single source of truth)이다.
RENDERERS: dict[str, Any] = {
    "md": render_md,
    "txt": render_txt,
    "docx": render_docx,
    "pptx": render_pptx,
    "hwpx": render_hwpx,
}

MEDIA_TYPES: dict[str, str] = {
    "md": "text/markdown; charset=utf-8",
    "txt": "text/plain; charset=utf-8",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "hwpx": "application/hwp+zip",
}
