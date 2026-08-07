# 부팅 시 모델 서버 "신원"을 대조하는 자가검증을 검증하는 단위 테스트.
"""
`_verify_model_endpoints()` 의 계약을 고정한다 (2026-08-07).

[왜 이 검사가 생겼나 — 하루에 두 번 같은 모양으로 당했다]
    ① 재부팅 때 다른 PostgreSQL 이 포트를 선점해 NOVA 가 46MB 짜리 빈 DB 에 붙었는데
       로그는 "연결 성공"이었다.
    ② 비전·이미지 설정이 B200 이관 후에도 옛 주소를 가리켜 두 도구가 죽어 있었는데
       부팅 로그는 깨끗했다. vision_model 이름도 실제 서빙명과 달랐다.

    공통점은 **살아있음만 보고 정합성은 아무도 보지 않았다**는 것이다. 그래서 두 실패가
    전부 조용했다. 아래 테스트들이 지키는 것은 "조용하지 않을 것" 하나다.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from core.bootstrap import _verify_model_endpoints


class _FakeResponse:
    def __init__(self, payload: dict, status: int = 200) -> None:
        self._payload = payload
        self._status = status

    def raise_for_status(self) -> None:
        if self._status >= 400:
            raise RuntimeError(f"HTTP {self._status}")

    def json(self) -> dict:
        return self._payload


def _fake_client_factory(routes: dict[str, dict]):
    """주어진 URL 표에만 응답하는 httpx.AsyncClient 대역을 만든다."""

    class _FakeClient:
        def __init__(self, *_a, **_k) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_a):
            return False

        async def get(self, url: str):
            if url not in routes:
                raise ConnectionError(f"연결 실패: {url}")
            return _FakeResponse(routes[url])

    return _FakeClient


def _config(**over) -> SimpleNamespace:
    gs = SimpleNamespace(
        vision_url=over.get("vision_url", "http://v:8004"),
        vision_model=over.get("vision_model", "gemma-3-27b"),
        coder_url=over.get("coder_url", ""),
        coder_model=over.get("coder_model", "devstral-small"),
        image_url=over.get("image_url", "http://i:8003"),
        embedding_url=over.get("embedding_url", "http://e:8002"),
    )
    return SimpleNamespace(
        gpu_server_url=over.get("url", "http://m:8001"),
        gpu_server=gs,
        model=SimpleNamespace(primary_model=over.get("primary_model", "ax-4.0")),
    )


ALL_HEALTHY = {
    "http://m:8001/v1/models": {"data": [{"id": "ax-4.0"}]},
    "http://v:8004/v1/models": {"data": [{"id": "gemma-3-27b"}]},
    "http://i:8003/health": {"status": "ok", "model": "stabilityai/stable-diffusion-3.5-large"},
    "http://e:8002/health": {
        "status": "ok",
        "model": "/opt/nexus-gpu/models/e5-large",
        "dimension": 1024,
    },
}


@pytest.mark.asyncio
async def test_all_healthy_logs_summary_without_warning(caplog, monkeypatch):
    """정상이면 한 줄 요약만 남고 경고는 없다."""
    import httpx

    monkeypatch.setattr(httpx, "AsyncClient", _fake_client_factory(ALL_HEALTHY))

    with caplog.at_level(logging.INFO, logger="nexus.bootstrap"):
        await _verify_model_endpoints(_config())

    assert "모델 서버 신원" in caplog.text
    assert "추론=ax-4.0" in caplog.text
    assert "비전=gemma-3-27b" in caplog.text
    # 경로형 모델명은 마지막 조각만, 차원은 괄호로
    assert "임베딩=e5-large(1024)" in caplog.text
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


@pytest.mark.asyncio
async def test_model_name_mismatch_warns_with_both_names(caplog, monkeypatch):
    """★실제로 겪은 케이스 — 설정은 gemma-4-12b 인데 서버는 gemma-3-27b 를 서빙."""
    import httpx

    monkeypatch.setattr(httpx, "AsyncClient", _fake_client_factory(ALL_HEALTHY))

    with caplog.at_level(logging.INFO, logger="nexus.bootstrap"):
        await _verify_model_endpoints(_config(vision_model="gemma-4-12b"))

    warns = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert warns, "불일치인데 경고가 없다"
    assert "gemma-4-12b" in warns[0] and "gemma-3-27b" in warns[0]  # 양쪽을 다 보여준다
    assert "불일치" in caplog.text


@pytest.mark.asyncio
async def test_unreachable_endpoint_warns(caplog, monkeypatch):
    """★실제로 겪은 케이스 — 설정이 죽은 주소를 가리킬 때."""
    import httpx

    routes = dict(ALL_HEALTHY)
    del routes["http://i:8003/health"]  # 이미지 서버만 닿지 않게
    monkeypatch.setattr(httpx, "AsyncClient", _fake_client_factory(routes))

    with caplog.at_level(logging.INFO, logger="nexus.bootstrap"):
        await _verify_model_endpoints(_config())

    warns = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warns) == 1
    assert "이미지" in warns[0] and "연결할 수 없습니다" in warns[0]
    # 나머지는 정상 요약에 그대로 남는다(하나 죽었다고 전부 숨기지 않는다)
    assert "추론=ax-4.0" in caplog.text


@pytest.mark.asyncio
async def test_unconfigured_endpoint_is_skipped(caplog, monkeypatch):
    """주소가 비어 있으면 아예 검사하지 않는다 — 코딩 서브모델은 선택 구성이다."""
    import httpx

    monkeypatch.setattr(httpx, "AsyncClient", _fake_client_factory(ALL_HEALTHY))

    with caplog.at_level(logging.INFO, logger="nexus.bootstrap"):
        await _verify_model_endpoints(_config(coder_url=""))

    assert "코딩" not in caplog.text
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


@pytest.mark.asyncio
async def test_health_not_ok_warns(caplog, monkeypatch):
    """/health 가 200 이어도 status 가 ok 가 아니면 경고한다."""
    import httpx

    routes = dict(ALL_HEALTHY)
    routes["http://e:8002/health"] = {"status": "loading", "model": "e5-large"}
    monkeypatch.setattr(httpx, "AsyncClient", _fake_client_factory(routes))

    with caplog.at_level(logging.INFO, logger="nexus.bootstrap"):
        await _verify_model_endpoints(_config())

    warns = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert warns and "임베딩" in warns[0]


@pytest.mark.asyncio
async def test_empty_model_list_warns(caplog, monkeypatch):
    """모델 목록이 비어 있으면(기동 중) 경고한다 — 200 만으로 정상이라 하지 않는다."""
    import httpx

    routes = dict(ALL_HEALTHY)
    routes["http://m:8001/v1/models"] = {"data": []}
    monkeypatch.setattr(httpx, "AsyncClient", _fake_client_factory(routes))

    with caplog.at_level(logging.INFO, logger="nexus.bootstrap"):
        await _verify_model_endpoints(_config())

    warns = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert warns and "추론" in warns[0]


@pytest.mark.asyncio
async def test_never_raises(monkeypatch):
    """어떤 실패도 부트스트랩을 막지 않는다 — 진단이 서비스를 죽이면 안 된다."""
    import httpx

    class _Exploding:
        def __init__(self, *_a, **_k):
            raise RuntimeError("클라이언트 생성 자체가 실패")

    monkeypatch.setattr(httpx, "AsyncClient", _Exploding)
    await _verify_model_endpoints(_config())  # 예외가 새면 이 줄에서 실패한다
