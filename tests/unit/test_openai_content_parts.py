# OpenAI `content` 파트 배열 정규화 — 아래 파이프라인이 문자열만 보게 유지한다.
"""
2026-08-12. 코딩 API 는 `content` 가 문자열일 때만 받았다. 실측 3케이스.

    content: "안녕"                      → 200
    content: [{text}, {image_url}]        → 422
    content: [{text}]                     → 422   ← ★이미지와 무관

즉 막힌 원인은 이미지가 아니라 **배열 형식**이었다. 그래서 정규화 계층을 앞에 두고,
그 아래(`_split_openai_messages` 이후)는 종전대로 문자열만 보게 한다 — 무회귀의 핵심.
"""

from __future__ import annotations

import base64
from pathlib import Path

import pytest

from web.app import OpenAIChatMessage, _normalize_openai_content

PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)
PNG_URL = "data:image/png;base64," + base64.b64encode(PNG_BYTES).decode()


@pytest.fixture()
def uploads(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """업로드 디렉토리를 임시 폴더로 돌린다(실제 배포 경로를 건드리지 않도록)."""
    import web.app as webapp

    monkeypatch.setattr(webapp, "_uploads_dir", lambda: tmp_path)
    return tmp_path


def _msg(role: str, content: object) -> OpenAIChatMessage:
    return OpenAIChatMessage(role=role, content=content)


# ─────────────────────────────────────────────
# 무회귀 — 문자열 경로는 손대지 않는다
# ─────────────────────────────────────────────
def test_string_content_is_untouched(uploads: Path) -> None:
    out, warns = _normalize_openai_content([_msg("user", "안녕")])

    assert out[0].content == "안녕"
    assert warns == []
    assert not list(uploads.iterdir()), "이미지가 없으면 디스크를 건드리지 않는다"


def test_none_content_is_untouched(uploads: Path) -> None:
    out, _ = _normalize_openai_content([_msg("assistant", None)])
    assert out[0].content is None


# ─────────────────────────────────────────────
# ★배열 — 이미지가 없어도 받아야 한다
# ─────────────────────────────────────────────
def test_text_only_array_becomes_string(uploads: Path) -> None:
    """실측 3번째 케이스. 이미지와 무관하게 배열이라서 422 였다."""
    out, warns = _normalize_openai_content([_msg("user", [{"type": "text", "text": "안녕"}])])

    assert out[0].content == "안녕"
    assert warns == []


def test_multiple_text_parts_are_joined(uploads: Path) -> None:
    out, _ = _normalize_openai_content(
        [_msg("user", [{"type": "text", "text": "앞"}, {"type": "text", "text": "뒤"}])]
    )
    assert "앞" in out[0].content and "뒤" in out[0].content


def test_empty_array_becomes_empty_string(uploads: Path) -> None:
    out, _ = _normalize_openai_content([_msg("user", [])])
    assert out[0].content == ""


# ─────────────────────────────────────────────
# 이미지 → 파일 + 핸들
# ─────────────────────────────────────────────
def test_image_is_saved_and_replaced_by_handle(uploads: Path) -> None:
    out, warns = _normalize_openai_content(
        [
            _msg(
                "user",
                [
                    {"type": "text", "text": "이 그림 뭐야?"},
                    {"type": "image_url", "image_url": {"url": PNG_URL}},
                ],
            )
        ]
    )

    saved = list(uploads.glob("img-*.png"))
    assert len(saved) == 1, "이미지가 파일로 내려가야 AnalyzeImage 가 볼 수 있다"
    assert "이 그림 뭐야?" in out[0].content
    assert saved[0].as_posix() in out[0].content, "대화에 서버 경로가 남아야 한다"
    assert "AnalyzeImage" in out[0].content
    assert warns == []


def test_image_url_as_plain_string_is_accepted(uploads: Path) -> None:
    """일부 클라이언트는 image_url 을 dict 가 아니라 문자열로 보낸다."""
    out, warns = _normalize_openai_content(
        [_msg("user", [{"type": "image_url", "image_url": PNG_URL}])]
    )
    assert list(uploads.glob("img-*.png"))
    assert warns == []
    assert "서버 경로" in out[0].content


def test_multiple_images_are_numbered(uploads: Path) -> None:
    other = "data:image/png;base64," + base64.b64encode(PNG_BYTES + b"\x00").decode()
    out, _ = _normalize_openai_content(
        [
            _msg(
                "user",
                [
                    {"type": "image_url", "image_url": {"url": PNG_URL}},
                    {"type": "image_url", "image_url": {"url": other}},
                ],
            )
        ]
    )
    assert "이미지 #1" in out[0].content
    assert "이미지 #2" in out[0].content


# ─────────────────────────────────────────────
# ★멱등 — 무상태 재전송이 디스크를 불리면 안 된다
# ─────────────────────────────────────────────
def test_resending_same_history_does_not_grow_disk(uploads: Path) -> None:
    """코딩 API 는 무상태라 클라이언트가 히스토리를 매 턴 다시 보낸다.

    같은 이미지가 턴마다 새 파일이 되면 30턴 대화에서 같은 20MB 가 30벌 쌓인다.
    """
    msgs = [_msg("user", [{"type": "image_url", "image_url": {"url": PNG_URL}}])]

    for _ in range(5):
        _normalize_openai_content(msgs)

    assert len(list(uploads.glob("img-*"))) == 1


# ─────────────────────────────────────────────
# 실패는 조용하지 않게 — 대화 전체를 죽이지도 않게
# ─────────────────────────────────────────────
def test_bad_image_warns_but_keeps_conversation(uploads: Path) -> None:
    out, warns = _normalize_openai_content(
        [
            _msg(
                "user",
                [
                    {"type": "text", "text": "이거 봐줘"},
                    {"type": "image_url", "image_url": {"url": "http://x/y.png"}},
                ],
            )
        ]
    )

    assert warns and "받아들이지 못했" in warns[0]
    assert "이거 봐줘" in out[0].content
    # 모델도 알아야 한다 — 본문에 실패가 남지 않으면 "왜 이미지 얘기가 없지"가 된다.
    assert "이미지 첨부 실패" in out[0].content
    assert not list(uploads.glob("img-*"))


def test_unknown_part_type_is_warned(uploads: Path) -> None:
    out, warns = _normalize_openai_content(
        [_msg("user", [{"type": "input_audio"}, {"type": "text", "text": "안녕"}])]
    )
    assert any("지원하지 않는" in w for w in warns)
    assert out[0].content == "안녕"


def test_malformed_part_is_warned(uploads: Path) -> None:
    out, warns = _normalize_openai_content(
        [_msg("user", ["문자열", {"type": "text", "text": "x"}])]
    )
    assert any("형식이 올바르지 않은" in w for w in warns)
    assert out[0].content == "x"
