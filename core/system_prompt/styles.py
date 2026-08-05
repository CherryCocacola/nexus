# 응답 스타일 프리셋 로더 (W1) — 웹·CLI가 공유하는 단일 진입점.
"""
응답 스타일 프리셋을 YAML에서 읽어 프롬프트 문구로 바꿔 준다 (2026-08-05).

[설계 원칙]
  - **단일 소스**: 문구는 `config/response_styles.yaml` 한 곳에만 있다. 웹과 CLI가
    같은 파일을 읽으므로 표면마다 말투가 갈리지 않는다.
  - **fail-soft**: 파일이 없거나 깨져도 예외를 내지 않는다. 스타일은 부가 기능이라
    이것 때문에 대화가 막히면 안 된다 — 그런 경우 "스타일 없음"으로 동작한다.
  - **무회귀 기본값**: 기본 스타일(normal)의 문구는 빈 문자열이다. 빈 값이면
    compose_system_prompt가 섹션 자체를 넣지 않으므로 기존 프롬프트와 완전히 같다.

[캐시]
  파일을 매 요청 읽지 않도록 mtime 기준으로 캐시한다. 운영 중 YAML을 고치면
  다음 요청부터 반영된다(재기동 불필요).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger("nexus.system_prompt.styles")

# 기본 위치 — 이 파일(core/system_prompt/) 기준 리포 루트의 config/.
_DEFAULT_PATH = Path(__file__).resolve().parents[2] / "config" / "response_styles.yaml"

# (경로, mtime) → 파싱 결과 캐시. 파일이 바뀌면 자동으로 다시 읽는다.
_cache: dict[str, tuple[float, dict[str, Any]]] = {}


def _load(path: Path) -> dict[str, Any]:
    """YAML을 읽어 dict로 돌려준다(실패 시 빈 dict — fail-soft)."""
    try:
        stat = path.stat()
    except OSError:
        return {}
    key = str(path)
    cached = _cache.get(key)
    if cached and cached[0] == stat.st_mtime:
        return cached[1]
    try:
        import yaml

        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            data = {}
    except Exception as e:  # noqa: BLE001 — 설정 오류가 대화를 막지 않는다
        logger.warning("응답 스타일 로딩 실패(무시): %s", e)
        data = {}
    _cache[key] = (stat.st_mtime, data)
    return data


def list_styles(path: Path | None = None) -> list[dict[str, str]]:
    """선택 가능한 스타일 목록을 돌려준다(UI 드롭다운용).

    Returns:
        [{"id": "concise", "label": "간결", "description": "..."}] 형태. 파일이
        없으면 빈 리스트 — 호출부는 이 경우 스타일 선택을 노출하지 않으면 된다.
    """
    data = _load(path or _DEFAULT_PATH)
    styles = data.get("styles") or {}
    if not isinstance(styles, dict):
        return []
    out: list[dict[str, str]] = []
    for style_id, meta in styles.items():
        if not isinstance(meta, dict):
            continue
        out.append(
            {
                "id": str(style_id),
                "label": str(meta.get("label", style_id)),
                "description": str(meta.get("description", "")),
            }
        )
    return out


def default_style_id(path: Path | None = None) -> str:
    """설정된 기본 스타일 id(없으면 "normal")."""
    data = _load(path or _DEFAULT_PATH)
    return str(data.get("default") or "normal")


def resolve_style_prompt(
    style_id: str | None,
    custom_text: str | None = None,
    path: Path | None = None,
) -> str:
    """스타일 선택을 실제 프롬프트 문구로 바꾼다.

    Args:
        style_id: 프리셋 id("concise" 등). None·빈 값·모르는 값이면 스타일 없음으로
            처리한다(fail-soft — 잘못된 값 때문에 대화가 막히지 않는다).
        custom_text: 사용자가 직접 쓴 문구. 있으면 **프리셋 대신** 이것을 쓴다
            (사용자가 명시적으로 쓴 것이 프리셋보다 우선한다).
        path: 프리셋 YAML 경로(테스트 주입용).

    Returns:
        [응답 스타일] 섹션에 넣을 문구. 스타일이 없으면 빈 문자열이며, 그 경우
        compose_system_prompt가 섹션을 통째로 생략해 기존 프롬프트와 동일해진다.
    """
    if custom_text and custom_text.strip():
        return custom_text.strip()
    if not style_id or not str(style_id).strip():
        return ""
    data = _load(path or _DEFAULT_PATH)
    styles = data.get("styles") or {}
    meta = styles.get(str(style_id).strip()) if isinstance(styles, dict) else None
    if not isinstance(meta, dict):
        return ""
    return str(meta.get("prompt") or "").strip()
