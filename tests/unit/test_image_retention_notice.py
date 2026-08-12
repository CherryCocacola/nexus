# 이미지 재질의의 전제 — 원본 보존과 "만료됐다"는 안내를 고정한다.
"""
2026-08-12. 재질의(같은 이미지에 새 질문)는 **원본이 남아 있는 동안만** 유효하다.
실측으로 재질의 자체는 동작했다(2턴에서 AnalyzeImage 재호출 확인). 그래서 남은
위험은 하나다 — **파일이 사라졌을 때 무슨 일이 벌어지는가.**

그냥 "파일이 없습니다" 라고만 하면 모델이 경로를 고쳐 가며 재시도해 턴을 낭비하고,
사용자는 어제 되던 것이 왜 오늘 안 되는지 알 수 없다. 그래서 만료를 명시한다.

보존 기간은 24→72 로 올렸다. 해시 멱등 저장(S1) 덕에 같은 이미지가 겹쳐 쌓이지
않으므로 늘려도 디스크 부담이 크지 않다.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from core.tools.base import ToolUseContext
from core.tools.implementations.analyze_image_tool import AnalyzeImageTool

ROOT = Path(__file__).resolve().parents[2]


# ─────────────────────────────────────────────
# ★만료 안내 — 조용한 실패를 막는다
# ─────────────────────────────────────────────
@pytest.mark.asyncio
async def test_missing_file_explains_expiry(tmp_path: Path) -> None:
    tool = AnalyzeImageTool()
    ctx = ToolUseContext(
        cwd=".",
        options={"uploads_dir": str(tmp_path), "upload_retention_hours": 72},
    )

    result = await tool.check_permissions({"image_path": str(tmp_path / "img-gone.png")}, ctx)

    assert result.behavior.value == "deny"
    assert "72시간" in result.message, "언제 지워지는지 알려야 한다"
    assert "다시 올려야" in result.message, "무엇을 하면 되는지 알려야 한다"


@pytest.mark.asyncio
async def test_missing_file_does_not_invite_path_retries(tmp_path: Path) -> None:
    """★경로를 고쳐 가며 재시도하면 턴만 낭비된다 — 하지 말라고 명시한다."""
    tool = AnalyzeImageTool()
    ctx = ToolUseContext(cwd=".", options={"uploads_dir": str(tmp_path)})

    result = await tool.check_permissions({"image_path": str(tmp_path / "x.png")}, ctx)

    assert "재시도하지 마세요" in result.message


@pytest.mark.asyncio
async def test_retention_hours_absent_still_explains(tmp_path: Path) -> None:
    """설정 미주입 경량 경로에서도 문구가 깨지지 않는다."""
    tool = AnalyzeImageTool()
    ctx = ToolUseContext(cwd=".", options={"uploads_dir": str(tmp_path)})

    result = await tool.check_permissions({"image_path": str(tmp_path / "x.png")}, ctx)

    assert "일정 시간" in result.message


# ─────────────────────────────────────────────
# 배선 — 값이 실제로 도구까지 가는가
# ─────────────────────────────────────────────
def test_web_injects_retention_hours() -> None:
    """주입을 빠뜨리면 문구가 '일정 시간'으로 뭉개진다(F10 과 같은 종류의 실수)."""
    src = (ROOT / "web" / "app.py").read_text(encoding="utf-8")
    assert '"upload_retention_hours"' in src


# ─────────────────────────────────────────────
# 보존 기간 — 재질의의 전제
# ─────────────────────────────────────────────
@pytest.mark.parametrize("cfg", ["nexus_config.yaml", "nexus_config.112.yaml"])
def test_retention_is_at_least_72_hours(cfg: str) -> None:
    """하루짜리 보존은 '어제 그 그림' 을 못 받는다."""
    text = (ROOT / "config" / cfg).read_text(encoding="utf-8")
    m = re.search(r"^\s*retention_hours:\s*(\d+)", text, re.M)

    assert m, f"{cfg} 에 retention_hours 가 없다"
    assert int(m.group(1)) >= 72, f"{cfg}: {m.group(1)}시간 — 재질의 전제가 깨진다"


# ─────────────────────────────────────────────
# 재질의 유도 문구 — 웹·API 양쪽에 있어야 한다
# ─────────────────────────────────────────────
def test_web_handle_nudges_recall() -> None:
    """웹만 빠지면 같은 대화가 표면에 따라 다르게 동작한다."""
    html = (ROOT / "web" / "static" / "index.html").read_text(encoding="utf-8")
    assert "새로운 질문이면 다시 호출" in html


def test_api_handle_nudges_recall() -> None:
    from core.storage.inline_images import build_image_handle

    assert "다시 호출" in build_image_handle(Path("/x/img-a.png"), 1)
