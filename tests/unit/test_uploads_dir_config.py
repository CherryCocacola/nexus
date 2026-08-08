# 업로드 저장 위치가 설정을 따르는지 고정한다 — /tmp 폴백은 재시작에 사라진다.
"""
2026-08-08 실측 사고. 컨테이너를 재시작했더니 `/tmp/nexus_uploads` 가 비워져
업로드가 전부 사라졌고, 비전 도구가 "이미지 없음"으로 떨어졌다.

원인은 설정 부재가 아니라 **배선**이었다. `resolve_uploads_dir()` 는 처음부터
설정값을 받을 수 있게 만들어져 있었는데, 웹의 세 호출부가 **전부 인자 없이**
부르고 있었다. 그래서 설정에 무엇을 넣든 항상 임시폴더로 폴백했다.

그래서 이 테스트는 "설정을 읽는가"보다 **"호출부가 설정을 실제로 넘기는가"**를 본다.
함수만 고치고 호출부를 놓치면 같은 사고가 그대로 재현된다.
"""

from __future__ import annotations

import re
import tempfile
from pathlib import Path

from core.config import UploadConfig
from core.tools.implementations.analyze_image_tool import resolve_uploads_dir

WEB_APP = Path(__file__).resolve().parents[2] / "web" / "app.py"


# ─────────────────────────────────────────────
# 설정 필드
# ─────────────────────────────────────────────
def test_upload_config_has_uploads_dir_defaulting_to_empty() -> None:
    """기본값은 빈 문자열 — 개발 환경은 종전대로 임시폴더 폴백(무회귀)."""
    assert UploadConfig().uploads_dir == ""


def test_resolve_uses_configured_path(tmp_path: Path) -> None:
    target = tmp_path / "persistent_uploads"
    got = resolve_uploads_dir(str(target))
    assert got == target.resolve()
    assert got.is_dir(), "디렉토리를 만들어 주지 않으면 첫 업로드가 실패한다"


def test_resolve_falls_back_to_tempdir_when_unset() -> None:
    """설정이 비면 종전 동작 — 이 폴백 자체는 유지한다(개발 편의)."""
    got = resolve_uploads_dir(None)
    assert got == (Path(tempfile.gettempdir()) / "nexus_uploads").resolve()


def test_blank_string_is_treated_as_unset() -> None:
    """공백만 있는 값도 미설정으로 본다 — YAML 에서 흔한 실수다."""
    assert resolve_uploads_dir("   ") == resolve_uploads_dir(None)


# ─────────────────────────────────────────────
# ★배선 — 여기가 실제로 깨져 있던 곳
# ─────────────────────────────────────────────
def test_web_has_single_uploads_dir_helper() -> None:
    src = WEB_APP.read_text(encoding="utf-8")
    assert "def _uploads_dir(" in src


def test_helper_passes_config_value_to_resolver() -> None:
    src = WEB_APP.read_text(encoding="utf-8")
    body = src[src.index("def _uploads_dir("):]
    body = body[: body.index("\n\n\n")]
    assert 'getattr(upload_cfg, "uploads_dir", "")' in body
    assert "resolve_uploads_dir(" in body


def test_no_bare_resolve_calls_remain_in_web() -> None:
    """★핵심 — 인자 없는 `resolve_uploads_dir()` 호출이 하나라도 남으면 안 된다.

    남아 있으면 그 경로만 조용히 임시폴더를 쓰게 되고, 어디가 새는지 찾기 어렵다.
    (문서·주석의 언급은 괜찮다 — 실제 호출만 본다.)
    """
    src = WEB_APP.read_text(encoding="utf-8")
    bare = re.findall(r"(?<!def )resolve_uploads_dir\(\s*\)", src)
    assert not bare, f"인자 없는 호출이 남아 있다: {len(bare)}건"


def test_all_three_sites_use_the_helper() -> None:
    """부트스트랩 옵션·정리 잡·업로드 라우트 세 곳 모두 헬퍼를 쓴다."""
    src = WEB_APP.read_text(encoding="utf-8")
    assert src.count("_uploads_dir()") >= 3, "세 호출부 중 빠진 곳이 있다"


# ─────────────────────────────────────────────
# 배포 설정 — 실제로 영속 경로가 들어가 있는지
# ─────────────────────────────────────────────
def test_deployed_config_sets_persistent_uploads_dir() -> None:
    """112 배포 설정은 반드시 영속 경로를 지정해야 한다.

    이 값이 비면 코드를 아무리 고쳐도 /tmp 로 돌아간다 — 사고가 그대로 재현된다.
    """
    cfg = (Path(__file__).resolve().parents[2] / "config" / "nexus_config.112.yaml").read_text(
        encoding="utf-8"
    )
    m = re.search(r"^\s*uploads_dir:\s*\"?([^\"\n#]+)", cfg, re.M)
    assert m, "112 설정에 uploads_dir 이 없다"
    value = m.group(1).strip()
    assert value, "112 설정의 uploads_dir 이 비어 있다(= /tmp 폴백)"
    assert not value.startswith("/tmp"), f"/tmp 는 재시작에 사라진다: {value}"
