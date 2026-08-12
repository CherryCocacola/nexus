# 인라인 이미지 수용 — 검증 순서와 **해시 멱등 저장**을 고정한다.
"""
2026-08-12. 코딩 API 가 OpenAI 비전 규격(`content` 파트 배열)을 못 받아 422 였다.
실측으로 막히는 이유는 이미지가 아니라 **배열 형식**이었다(텍스트만 든 배열도 422).

이 테스트가 지키는 것 둘.

★① 해시 멱등 저장. 코딩 API 는 무상태라 클라이언트가 히스토리를 매 턴 재전송한다.
   파일명을 새로 만들면 같은 20MB 가 턴마다 쌓여 30턴이면 600MB 다.

★② 디코딩 **전** 크기 차단. 디코딩 후에 재면 이미 메모리를 다 쓴 뒤다.
"""

from __future__ import annotations

import base64
import hashlib
from pathlib import Path

import pytest

from core.storage.inline_images import (
    InlineImageError,
    build_image_handle,
    parse_data_url,
    save_inline_image,
)

# 1x1 PNG (실제 시그니처를 가진 최소 이미지)
PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)
PNG_URL = "data:image/png;base64," + base64.b64encode(PNG_BYTES).decode()
MB = 1024 * 1024


# ─────────────────────────────────────────────
# 파싱 · 검증
# ─────────────────────────────────────────────
def test_parses_valid_png() -> None:
    mime, raw = parse_data_url(PNG_URL, 20 * MB)
    assert mime == "image/png"
    assert raw == PNG_BYTES


def test_remote_url_is_rejected() -> None:
    """★에어갭 위반이자 SSRF 통로 — 원격 URL 은 받지 않는다."""
    with pytest.raises(InlineImageError, match="data URL"):
        parse_data_url("http://192.168.21.112/x.png", 20 * MB)


def test_unsupported_mime_is_rejected() -> None:
    """AnalyzeImage 가 다루지 못하는 형식을 받아 두면 나중에 분석에서 죽는다."""
    url = "data:image/gif;base64," + base64.b64encode(b"GIF89a").decode()
    with pytest.raises(InlineImageError, match="지원하지 않는"):
        parse_data_url(url, 20 * MB)


def test_mime_lie_is_caught_by_magic_bytes() -> None:
    """MIME 은 클라이언트가 만든 값이다 — 내용과 다르면 잡아야 한다."""
    url = "data:image/png;base64," + base64.b64encode(b"NOT A PNG AT ALL").decode()
    with pytest.raises(InlineImageError, match="일치하지 않습니다"):
        parse_data_url(url, 20 * MB)


def test_broken_base64_is_rejected() -> None:
    with pytest.raises(InlineImageError, match="base64"):
        parse_data_url("data:image/png;base64,!!!not base64!!!", 20 * MB)


def test_empty_content_is_rejected() -> None:
    with pytest.raises(InlineImageError, match="비어"):
        parse_data_url("data:image/png;base64,", 20 * MB)


# ─────────────────────────────────────────────
# ★크기 — 디코딩 전에 잘라야 한다
# ─────────────────────────────────────────────
def test_oversize_is_rejected_before_decoding() -> None:
    """base64 문자열 길이로 먼저 자른다.

    디코딩까지 간 뒤 재면 그 시점에 이미 메모리를 다 썼다. 여기서는 '디코딩하면
    수십 MB 가 되는 문자열'을 넘겨도 즉시 거부되는지 본다.
    """
    huge_b64 = "A" * (3 * MB)  # 디코딩하면 약 2.25MB
    url = "data:image/png;base64," + huge_b64
    with pytest.raises(InlineImageError, match="너무 큽니다"):
        parse_data_url(url, 1 * MB)


def test_within_limit_passes() -> None:
    mime, raw = parse_data_url(PNG_URL, 1 * MB)
    assert mime == "image/png" and raw


# ─────────────────────────────────────────────
# ★해시 멱등 저장 — 이게 없으면 디스크가 터진다
# ─────────────────────────────────────────────
def test_same_image_saved_once(tmp_path: Path) -> None:
    """무상태 재전송 때문에 같은 이미지가 매 턴 도착한다 — 한 번만 써야 한다."""
    p1 = save_inline_image(PNG_BYTES, "image/png", tmp_path)
    p2 = save_inline_image(PNG_BYTES, "image/png", tmp_path)

    assert p1 == p2
    assert len(list(tmp_path.glob("img-*"))) == 1, "같은 이미지가 두 벌 쌓였다"


def test_filename_is_content_hash(tmp_path: Path) -> None:
    """경로가 내용으로 결정돼야 만료 후 재전송에서도 **같은 경로**로 되살아난다."""
    path = save_inline_image(PNG_BYTES, "image/png", tmp_path)
    assert hashlib.sha256(PNG_BYTES).hexdigest()[:32] in path.name
    assert path.name.startswith("img-"), "업로드(upload-)와 접두사가 구분돼야 한다"
    assert path.suffix == ".png"


def test_different_images_get_different_files(tmp_path: Path) -> None:
    other = PNG_BYTES + b"\x00"
    a = save_inline_image(PNG_BYTES, "image/png", tmp_path)
    b = save_inline_image(other, "image/png", tmp_path)
    assert a != b
    assert len(list(tmp_path.glob("img-*"))) == 2


def test_no_partial_file_left_behind(tmp_path: Path) -> None:
    """중간 파일(.part)이 남으면 다음 분석이 깨진 파일을 읽는다."""
    save_inline_image(PNG_BYTES, "image/png", tmp_path)
    assert not list(tmp_path.glob("*.part"))


def test_creates_directory_if_missing(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "uploads"
    path = save_inline_image(PNG_BYTES, "image/png", target)
    assert path.is_file()


# ─────────────────────────────────────────────
# 핸들 문구 — 웹과 같은 규약이어야 한다
# ─────────────────────────────────────────────
def test_handle_matches_web_convention() -> None:
    """문구가 갈리면 모델이 웹에서 하던 대로 도구를 부르지 못한다."""
    text = build_image_handle(Path("/app/data/uploads/img-abc.png"), 1)

    assert "서버 경로: /app/data/uploads/img-abc.png" in text
    assert "AnalyzeImage" in text
    assert "사용자가 이미지 파일을 업로드했습니다" in text


def test_handle_tells_model_to_recall_for_new_questions() -> None:
    """재질의 유도 — 이것이 없으면 모델이 앞선 요약으로 때운다."""
    text = build_image_handle(Path("/x/img-a.png"), 1)
    assert "다시 호출" in text


def test_handle_uses_posix_path() -> None:
    """서버는 리눅스다. 윈도우에서 만들어도 역슬래시가 새면 안 된다."""
    assert "\\" not in build_image_handle(Path("/app/data/uploads/img-a.png"), 2)
