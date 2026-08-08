# 첨부 UI 두 기능(클립보드 붙여넣기·이미지 썸네일)의 배선을 고정한다.
"""
2026-08-08 사용자 요청으로 넣은 두 기능.

  ① 다른 곳에서 복사한 이미지를 대화창에 Ctrl+V 로 붙여넣기
  ② 첨부한 이미지를 작게라도 미리보기로 표시

프런트엔드라 단위 테스트로 동작을 돌려볼 수는 없다. 대신 **배선이 사라지지 않도록**
못 박는다 — 이 리포에서 "코드는 있는데 안 걸려 있음"이 실제로 여러 번 났다
(검증기가 웹에만 배선된 사고, 프롬프트↔도구 불일치). 함수만 있고 `onpaste` 가 빠지면
기능은 조용히 죽는데 아무도 모른다.

실제 동작(붙여넣기가 되는지, 썸네일이 보이는지)은 사람이 눈으로 봐야 한다
— `user_mig/TEST_사용자수동_20260808.md` 의 해당 항목.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

INDEX = Path(__file__).resolve().parents[2] / "web" / "static" / "index.html"


@pytest.fixture(scope="module")
def html() -> str:
    return INDEX.read_text(encoding="utf-8")


# ─────────────────────────────────────────────
# ① 클립보드 붙여넣기
# ─────────────────────────────────────────────
def test_paste_handler_is_defined(html: str) -> None:
    assert "function handlePasteFiles(" in html


def test_paste_handler_is_wired_to_the_input(html: str) -> None:
    """★배선 — 입력창에 onpaste 가 걸려 있어야 기능이 산다."""
    textarea = re.search(r'<textarea id="userInput".*?>', html, re.S)
    assert textarea is not None, "userInput textarea 를 찾지 못했다"
    assert "onpaste=" in textarea.group(0), "onpaste 가 입력창에 걸려 있지 않다"
    assert "handlePasteFiles" in textarea.group(0)


def test_paste_keeps_text_paste_intact(html: str) -> None:
    """이미지가 없으면 그냥 반환해 브라우저 기본 텍스트 붙여넣기를 막지 않는다.

    여기서 preventDefault 를 무조건 부르면 **텍스트 붙여넣기가 통째로 죽는다.**
    조기 반환이 그 방어이므로 사라지지 않게 고정한다.
    """
    body = html[html.index("function handlePasteFiles("):]
    body = body[: body.index("\n}\n") + 3]
    early_return = body.index("if (images.length === 0) return;")
    prevent = body.index("event.preventDefault();")
    assert early_return < prevent, "이미지가 없을 때도 기본 동작을 막고 있다"


def test_paste_enforces_size_limit(html: str) -> None:
    """서버 상한(20MB)과 같은 사전 안내 — 헛된 업로드를 줄인다."""
    body = html[html.index("function handlePasteFiles("):]
    assert "20 * 1024 * 1024" in body[:1500]


def test_pasted_image_gets_unique_name(html: str) -> None:
    """붙여넣은 이미지는 이름이 없거나 image.png 로 겹친다 → 시각을 붙여 구분."""
    body = html[html.index("function handlePasteFiles("):]
    assert "pasted-" in body[:2000]


# ─────────────────────────────────────────────
# ② 이미지 썸네일
# ─────────────────────────────────────────────
def test_thumbnail_css_exists(html: str) -> None:
    assert ".attach-chip .thumb" in html


def test_preview_renders_thumbnail_for_images(html: str) -> None:
    body = html[html.index("function renderAttachPreview("):]
    body = body[: body.index("\n}\n") + 3]
    assert "_thumbUrl(" in body
    assert 'class="thumb"' in body


def test_filename_is_escaped_in_preview(html: str) -> None:
    """파일명이 HTML 로 조립되므로 이스케이프해야 한다.

    사용자가 올린 파일 이름이 그대로 마크업이 되면 남의 파일명 하나로 화면이
    깨지거나 스크립트가 돈다.
    """
    body = html[html.index("function renderAttachPreview("):]
    body = body[: body.index("\n}\n") + 3]
    assert "escapeHtml(f.name)" in body


def test_object_urls_are_released(html: str) -> None:
    """★미리보기 URL 은 반드시 해제한다 — 안 하면 이미지가 메모리에 계속 남는다.

    제거(removeFile)와 전송 후 비우기 두 경로 모두에서 풀어야 한다. 한쪽만 풀면
    긴 세션에서 조용히 샌다.
    """
    assert "URL.revokeObjectURL" in html
    remove_fn = html[html.index("function removeFile("):]
    remove_fn = remove_fn[: remove_fn.index("\n}\n") + 3]
    assert "_releaseThumb" in remove_fn, "제거 경로에서 URL 을 해제하지 않는다"
    assert "attachedFiles.forEach(_releaseThumb)" in html, "전송 후 해제가 빠졌다"


def test_image_detection_covers_pasted_files_without_name(html: str) -> None:
    """붙여넣기 이미지는 이름이 없을 수 있어 MIME 을 먼저 본다."""
    body = html[html.index("function _isImageFile("):]
    body = body[: body.index("\n}\n") + 3]
    assert "f.type" in body and "image/" in body
