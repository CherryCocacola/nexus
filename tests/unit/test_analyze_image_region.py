# region 크롭 — 작은 글씨를 읽으려고 자르는 기능의 계약을 고정한다.
"""
2026-08-12 실측이 출발점이다. 640x400 이미지의 우하단 `OMEGA77` 을 전체 분석에서
**`OMEGA777` 로 오독**했다(7이 하나 더 붙었다). 재질의는 잘 됐지만 정확도는 별개였다.

원인은 VLM 이 입력을 자체 해상도로 줄여 보기 때문이다. 잘라 보내면 같은 픽셀 예산이
그 영역에만 쓰여 글자가 살아난다.

여기서 고정하는 것
  ★비율 좌표(0~1) — 모델은 원본 픽셀 크기를 모른다. 픽셀을 요구하면 찍게 된다.
  ★잘못된 좌표는 **원본으로 조용히 넘어가지 않는다** — 모델이 자기가 자른 줄 알고
    엉뚱한 답을 확신하게 되면 더 나쁘다.
"""

from __future__ import annotations

import io

import pytest
from PIL import Image

from core.tools.implementations.analyze_image_tool import AnalyzeImageTool


def _png(w: int = 400, h: int = 200, color: str = "white") -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), color).save(buf, format="PNG")
    return buf.getvalue()


def _size(data: bytes) -> tuple[int, int]:
    with Image.open(io.BytesIO(data)) as im:
        return im.size


TOOL = AnalyzeImageTool()


# ─────────────────────────────────────────────
# 스키마 — 모델이 쓸 수 있게 설명돼 있는가
# ─────────────────────────────────────────────
def test_region_is_declared_as_ratio() -> None:
    schema = TOOL.input_schema["properties"]["region"]

    assert schema["minItems"] == 4 and schema["maxItems"] == 4
    assert "0~1" in schema["description"] or "RATIO" in schema["description"].upper()
    assert "region" not in TOOL.input_schema["required"], "선택 인자여야 한다(무회귀)"


def test_region_description_tells_when_to_use_it() -> None:
    """언제 쓰는지 안 적으면 모델이 영영 안 쓴다."""
    desc = TOOL.input_schema["properties"]["region"]["description"]
    assert "small" in desc.lower()


# ─────────────────────────────────────────────
# ★크롭 동작
# ─────────────────────────────────────────────
def test_crops_to_requested_ratio() -> None:
    data, mime, err = TOOL._crop_region(_png(400, 200), "image/png", [0, 0, 0.5, 0.5])

    assert err is None
    assert _size(data) == (200, 100)
    assert mime == "image/png"


def test_bottom_right_quadrant() -> None:
    """실측 실패 사례의 좌표 — 우하단."""
    data, _mime, err = TOOL._crop_region(_png(640, 400), "image/png", [0.5, 0.5, 0.5, 0.5])

    assert err is None
    assert _size(data) == (320, 200)


def test_horizontal_band() -> None:
    data, _mime, err = TOOL._crop_region(_png(400, 200), "image/png", [0, 0.4, 1, 0.2])

    assert err is None
    assert _size(data) == (400, 40)


def test_output_is_png_even_from_jpeg() -> None:
    """JPEG 로 다시 압축하면 글자 경계에 블록 노이즈가 생겨 자른 의미가 준다."""
    buf = io.BytesIO()
    Image.new("RGB", (400, 200), "white").save(buf, format="JPEG")

    _data, mime, err = TOOL._crop_region(buf.getvalue(), "image/jpeg", [0, 0, 0.5, 1])

    assert err is None
    assert mime == "image/png"


def test_full_region_is_effectively_the_whole_image() -> None:
    data, _mime, err = TOOL._crop_region(_png(300, 150), "image/png", [0, 0, 1, 1])

    assert err is None
    assert _size(data) == (300, 150)


# ─────────────────────────────────────────────
# ★잘못된 좌표는 조용히 넘어가지 않는다
# ─────────────────────────────────────────────
@pytest.mark.parametrize(
    "bad",
    [
        [0, 0, 0.5],  # 개수 부족
        [0, 0, 0.5, 0.5, 0.5],  # 개수 초과
        "왼쪽 위",  # 배열이 아님
        [0, 0, "반", 0.5],  # 숫자가 아님
    ],
)
def test_malformed_region_is_reported(bad: object) -> None:
    _data, _mime, err = TOOL._crop_region(_png(), "image/png", bad)
    assert err is not None


def test_pixel_coordinates_are_rejected_with_a_hint() -> None:
    """★모델이 픽셀로 줄 가능성이 높다 — 조용히 자르지 말고 비율임을 알려준다."""
    _data, _mime, err = TOOL._crop_region(_png(640, 400), "image/png", [320, 200, 320, 200])

    assert err is not None
    assert "비율" in err
    assert "0.5" in err, "올바른 예를 함께 줘야 모델이 고칠 수 있다"


@pytest.mark.parametrize("bad", [[0, 0, 0, 0.5], [0, 0, 0.5, 0], [0, 0, -0.1, 0.5]])
def test_zero_or_negative_size_is_rejected(bad: list[float]) -> None:
    _data, _mime, err = TOOL._crop_region(_png(), "image/png", bad)
    assert err is not None


def test_corrupt_image_is_reported_not_silently_passed() -> None:
    _data, _mime, err = TOOL._crop_region(b"not an image at all", "image/png", [0, 0, 1, 1])
    assert err is not None
    assert "자르지 못했" in err


# ─────────────────────────────────────────────
# 경계 — 넘치는 좌표도 죽지 않는다
# ─────────────────────────────────────────────
def test_region_beyond_edge_is_clamped() -> None:
    """x+w 가 1을 넘어도(모델이 흔히 낸다) 경계까지만 자른다."""
    data, _mime, err = TOOL._crop_region(_png(400, 200), "image/png", [0.8, 0.8, 0.5, 0.5])

    assert err is None
    w, h = _size(data)
    assert 0 < w <= 80 and 0 < h <= 40


def test_tiny_region_still_yields_at_least_one_pixel() -> None:
    data, _mime, err = TOOL._crop_region(_png(400, 200), "image/png", [0.5, 0.5, 0.0001, 0.0001])

    assert err is None
    w, h = _size(data)
    assert w >= 1 and h >= 1


# ─────────────────────────────────────────────
# 의존성 선언 — 에어갭 번들에서 빠지면 안 된다
# ─────────────────────────────────────────────
def test_pillow_is_declared_not_transitive() -> None:
    """설치돼 있다는 것과 선언돼 있다는 것은 다르다.

    이전에는 전이 의존성이라, 상위 패키지가 바뀌면 조용히 사라지고 오프라인 wheel
    번들에서도 누락됐다.
    """
    from pathlib import Path

    req = (Path(__file__).resolve().parents[2] / "requirements.txt").read_text(encoding="utf-8")
    assert "pillow" in req.lower()
