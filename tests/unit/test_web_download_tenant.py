# web/app.py 다운로드 라우트 — 조건부 테넌트 소유권 검사(IDOR 점진 차단) 단위 테스트.
"""
web.app.download_file 의 조건부 테넌트 소유권 검사 단위 테스트.

정책(작업 지시):
  경로검증·파일존재 뒤 tb_artifacts 소유자를 조회해,
    - 소유자가 특정 테넌트 + 요청 테넌트와 불일치 → 404(존재 은닉, fail-closed)
    - 행 없음(레거시) / 소유자 미상 / 요청 테넌트 미상 / DB 없음 → 통과(fail-soft)

테스트 격리:
  실제 PG/HTTP 없이 download_file 을 직접 await 한다. _app_state 의 config/pg_pool 을
  가짜로 갈아끼우고 tmp 파일을 exports_dir 로 지정한다. request 는 state.tenant 만
  갖는 최소 객체로 흉내 낸다. asyncio_mode="auto" 라 데코레이터 없이 async 테스트가 돈다.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import HTTPException

from web.app import _app_state, download_file


# ─────────────────────────────────────────────
# 가짜 pg 풀 — get_artifact_owner가 쓰는 acquire()/fetchrow만 흉내
# ─────────────────────────────────────────────
class _Conn:
    def __init__(self, store: dict[str, dict[str, Any]]) -> None:
        self._store = store

    async def fetchrow(self, sql: str, *args: Any) -> dict[str, Any] | None:
        return self._store.get(args[0])


class _Ctx:
    def __init__(self, conn: _Conn) -> None:
        self._conn = conn

    async def __aenter__(self) -> _Conn:
        return self._conn

    async def __aexit__(self, *exc: Any) -> bool:
        return False


class _Pool:
    def __init__(self, store: dict[str, dict[str, Any]] | None = None) -> None:
        self._conn = _Conn(store or {})

    def acquire(self) -> _Ctx:
        return _Ctx(self._conn)


def _request(tenant_id: str | None) -> Any:
    """download_file이 읽는 request.state.tenant만 갖춘 최소 요청 객체."""
    tenant = SimpleNamespace(id=tenant_id) if tenant_id is not None else None
    return SimpleNamespace(state=SimpleNamespace(tenant=tenant))


@pytest.fixture
def _exports(tmp_path: Path):
    """exports_dir(tmp)에 테스트 파일 1개를 만들고 _app_state를 세팅/원복한다."""
    fname = "report.docx"
    (tmp_path / fname).write_text("hello", encoding="utf-8")

    saved_config = _app_state.get("config")
    saved_pool = _app_state.get("pg_pool")
    # config.document_export.exports_dir 가 tmp를 가리키게 한다.
    _app_state["config"] = SimpleNamespace(
        document_export=SimpleNamespace(exports_dir=str(tmp_path))
    )
    yield fname
    _app_state["config"] = saved_config
    _app_state["pg_pool"] = saved_pool


async def test_download_owner_match_passes(_exports):
    """소유자 == 요청 테넌트 → 정상 다운로드(404 아님)."""
    fname = _exports
    _app_state["pg_pool"] = _Pool({fname: {"tenant_id": "tenant-A"}})
    resp = await download_file(fname, _request("tenant-A"))
    assert getattr(resp, "status_code", 200) == 200


async def test_download_owner_mismatch_404(_exports):
    """소유자 != 요청 테넌트 → 404(존재 은닉)."""
    fname = _exports
    _app_state["pg_pool"] = _Pool({fname: {"tenant_id": "tenant-A"}})
    with pytest.raises(HTTPException) as ei:
        await download_file(fname, _request("tenant-B"))
    assert ei.value.status_code == 404


async def test_download_no_row_legacy_passes(_exports):
    """tb_artifacts에 행이 없으면(레거시) 통과(하위호환)."""
    fname = _exports
    _app_state["pg_pool"] = _Pool({})  # 행 없음
    resp = await download_file(fname, _request("tenant-B"))
    assert getattr(resp, "status_code", 200) == 200


async def test_download_owner_none_passes(_exports):
    """소유자 미상(tenant_id NULL)이면 통과(불일치 아님)."""
    fname = _exports
    _app_state["pg_pool"] = _Pool({fname: {"tenant_id": None}})
    resp = await download_file(fname, _request("tenant-B"))
    assert getattr(resp, "status_code", 200) == 200


async def test_download_request_tenant_none_passes(_exports):
    """요청 테넌트 미상(인증 off)이면 소유자가 있어도 통과(판정 불가 → fail-soft)."""
    fname = _exports
    _app_state["pg_pool"] = _Pool({fname: {"tenant_id": "tenant-A"}})
    resp = await download_file(fname, _request(None))
    assert getattr(resp, "status_code", 200) == 200


async def test_download_pool_none_passes(_exports):
    """pg_pool이 없으면 소유권 검사를 건너뛰고 통과(fail-soft)."""
    fname = _exports
    _app_state["pg_pool"] = None
    resp = await download_file(fname, _request("tenant-B"))
    assert getattr(resp, "status_code", 200) == 200


async def test_download_missing_file_404(_exports):
    """파일 자체가 없으면 소유권 검사 이전에 404(경로/존재 검증)."""
    _app_state["pg_pool"] = _Pool({})
    with pytest.raises(HTTPException) as ei:
        await download_file("does_not_exist.docx", _request("tenant-A"))
    assert ei.value.status_code == 404
