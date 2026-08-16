# /v1/models 가 실제 서빙 중인 모델을 보고하는지 고정한다
"""
2026-08-16. `GET /v1/models` 가 표시 이름을 코드에 박아 두고 있었다.

    실제: skt/A.X-4.0 (72B)      →  응답: "Qwen 3.5 27B"

모델을 바꾼 뒤에도 옛 이름이 남은 것이다. 연동하는 쪽이 그 값을 믿으면 파라미터
규모·성능을 잘못 가정한다. **이름은 서빙 주체에게 물어봐야** 모델이 바뀌어도 맞는다.

이 파일이 지키는 것
  ① 코드에 박힌 모델명이 다시 생기지 않는다(정적 검사)
  ② 상위(vLLM) 조회가 되면 그 값을 쓰고, 안 되면 **지어내지 않고 id 를 보여 준다**
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import web.app as webapp

_APP_SRC = Path(webapp.__file__).read_text(encoding="utf-8")


class _Cfg:
    class model:  # noqa: N801 — config 객체 흉내
        primary_model = "ax-4.0"
        auxiliary_model = "exaone-7.8b"
        embedding_model = "multilingual-e5-large"


class _Provider:
    def __init__(self, data: Any = None, raises: bool = False) -> None:
        self.data = data
        self.raises = raises

    async def list_upstream_models(self) -> Any:
        if self.raises:
            raise RuntimeError("추론 서버 없음")
        return self.data


@pytest.fixture()
def app_state(monkeypatch):
    """_app_state 를 테스트용으로 갈아 끼운다(원본 훼손 방지)."""
    state: dict[str, Any] = {"config": _Cfg()}
    monkeypatch.setattr(webapp, "_app_state", state)
    return state


# ─────────────────────────────────────────────
# ① 하드코딩된 모델명이 다시 생기지 않게
# ─────────────────────────────────────────────
@pytest.mark.parametrize("stale", ["Qwen 3.5 27B", "qwen3.5-27b"])
def test_no_hardcoded_model_names(stale: str) -> None:
    """★이 값들이 코드에 남아 있었던 것이 이번 결함이다."""
    assert stale not in _APP_SRC, f"코드에 박힌 모델명이 남아 있다: {stale}"


def test_model_ids_come_from_config() -> None:
    """id 는 설정에서 와야 한다 — 여기가 어긋나면 라우팅과 응답이 따로 논다."""
    assert "config.model.primary_model" in _APP_SRC
    assert "config.model.embedding_model" in _APP_SRC


# ─────────────────────────────────────────────
# ② 이름은 서빙 주체에게 물어본다
# ─────────────────────────────────────────────
@pytest.mark.asyncio
async def test_name_comes_from_upstream(app_state) -> None:
    app_state["model_provider"] = _Provider(
        [{"id": "ax-4.0", "root": "skt/A.X-4.0", "max_model_len": 65536}]
    )

    r = await webapp.list_models()
    by_id = {m["id"]: m for m in r["models"]}

    assert by_id["ax-4.0"]["name"] == "skt/A.X-4.0"
    assert by_id["ax-4.0"]["role"] == "primary"


@pytest.mark.asyncio
async def test_unknown_model_falls_back_to_id(app_state) -> None:
    """임베딩은 다른 서버가 서빙한다 — 추론 서버는 모른다. 지어내면 안 된다."""
    app_state["model_provider"] = _Provider([{"id": "ax-4.0", "root": "skt/A.X-4.0"}])

    r = await webapp.list_models()
    by_id = {m["id"]: m for m in r["models"]}

    assert by_id["multilingual-e5-large"]["name"] == "multilingual-e5-large"


@pytest.mark.asyncio
async def test_upstream_failure_is_fail_soft(app_state) -> None:
    """추론 서버가 죽어도 목록 조회는 살아야 한다(정보성 엔드포인트)."""
    app_state["model_provider"] = _Provider(raises=True)

    r = await webapp.list_models()

    assert r["total"] == 3
    assert all(m["name"] == m["id"] for m in r["models"])


@pytest.mark.asyncio
async def test_provider_without_the_method_is_tolerated(app_state) -> None:
    """다른 프로바이더·테스트 더미에는 이 메서드가 없을 수 있다."""
    app_state["model_provider"] = object()

    r = await webapp.list_models()

    assert r["total"] == 3


@pytest.mark.asyncio
async def test_no_config_returns_empty_not_stale_defaults(app_state) -> None:
    """부트스트랩 전에는 빈 목록 — 종전의 하드코딩 폴백이 옛 이름이 남던 경로였다."""
    app_state["config"] = None

    r = await webapp.list_models()

    assert r == {"models": [], "total": 0}
