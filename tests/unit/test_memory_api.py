# 메모리 조회·삭제 API의 소유자 격리(IDOR 차단)를 검증하는 단위 테스트.
"""
web.app 의 /v1/memories 조회·삭제 라우트 검증.

무엇을 지키나 (가장 중요한 것부터):
    - **소유자는 서버가 정한다.** 클라이언트가 owner를 넘길 수 없고, 인증 정보에서만
      해석한다. 이게 뚫리면 남의 대화 기억(사번·소속 등 개인정보)이 그대로 노출된다.
    - 조회는 자기 테넌트 것만 돌려준다.
    - 삭제는 자기 것만 지운다. 남의 id를 넣으면 지워지지 않고 404다
      (존재 여부조차 알려주지 않는다 — 존재 은닉).
    - 매니저가 없거나 소유자를 알 수 없으면 빈 목록/404 (fail-closed).

테스트 격리:
    실제 DB·HTTP 없이 라우트 함수를 직접 await 한다. LongTermMemory 는 소유자 규칙을
    실제로 수행하는 스텁으로 대체하고, _app_state 는 테스트 뒤 원복한다.
"""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import HTTPException

from core.memory.types import MemoryEntry, MemoryType
from web.app import _app_state, delete_memory, list_memories


def _entry(mem_id: str, content: str, owner: str | None) -> MemoryEntry:
    meta: dict[str, Any] = {"role": "user"}
    if owner:
        meta["owner"] = owner
    return MemoryEntry(
        id=mem_id,
        memory_type=MemoryType.EPISODIC,
        content=content,
        key=f"turn:s:{mem_id}",
        importance=0.7,
        created_at=datetime.now(UTC),
        metadata=meta,
    )


class _StubLongTerm:
    """소유자 규칙을 실제로 수행하는 LongTermMemory 스텁."""

    def __init__(self, entries: list[MemoryEntry]) -> None:
        self.entries = entries

    async def list_by_owner(self, owner: str, limit: int = 50) -> list[MemoryEntry]:
        if not owner:
            return []
        return [e for e in self.entries if e.metadata.get("owner") == owner][:limit]

    async def delete_owned(self, memory_id: str, owner: str) -> bool:
        for i, e in enumerate(self.entries):
            if e.id == memory_id and e.metadata.get("owner") == owner:
                del self.entries[i]
                return True
        return False


@pytest.fixture
def memories(monkeypatch: pytest.MonkeyPatch):
    """A/B 두 테넌트의 기억을 심고, 테넌트 해석을 고정한다."""
    entries = [
        _entry("a1", "A사 사번은 NX-8842", "tenant-a"),
        _entry("a2", "A사 회의는 월요일", "tenant-a"),
        _entry("b1", "B사 사번은 QQ-1111", "tenant-b"),
        _entry("x1", "소유자 없는 레거시", None),
    ]
    stub = _StubLongTerm(entries)
    saved = _app_state.get("memory_manager")
    _app_state["memory_manager"] = SimpleNamespace(long_term=stub)

    def fake_resolve(_body, _header, authorization):
        # Authorization 값을 그대로 테넌트 id 로 삼는 단순 규칙(테스트 전용).
        return SimpleNamespace(id=authorization) if authorization else None

    monkeypatch.setattr("web.app._resolve_tenant", fake_resolve)
    yield stub
    if saved is None:
        _app_state.pop("memory_manager", None)
    else:
        _app_state["memory_manager"] = saved


# ─────────────────────────────────────────────
# 조회
# ─────────────────────────────────────────────


async def test_list_returns_only_own_memories(memories):
    """자기 테넌트의 기억만 보인다."""
    result = await list_memories(authorization="tenant-a")

    assert [m["id"] for m in result["memories"]] == ["a1", "a2"]
    assert result["total"] == 2


async def test_list_excludes_other_tenant(memories):
    """다른 테넌트의 기억은 목록에 없다."""
    result = await list_memories(authorization="tenant-b")

    contents = " ".join(m["content"] for m in result["memories"])
    assert "NX-8842" not in contents
    assert "QQ-1111" in contents


async def test_list_excludes_ownerless_rows(memories):
    """소유자 없는 행(코드 RAG·레거시)은 노출되지 않는다."""
    result = await list_memories(authorization="tenant-a")

    assert all("레거시" not in m["content"] for m in result["memories"])


async def test_list_without_tenant_is_empty(memories):
    """테넌트를 알 수 없으면 아무 것도 보여주지 않는다(fail-closed)."""
    assert await list_memories(authorization=None) == {"memories": [], "total": 0}


async def test_list_without_manager_is_empty(monkeypatch):
    """메모리 매니저가 없으면(경량 경로) 빈 목록."""
    saved = _app_state.get("memory_manager")
    _app_state["memory_manager"] = None
    try:
        assert await list_memories(authorization="tenant-a") == {"memories": [], "total": 0}
    finally:
        if saved is not None:
            _app_state["memory_manager"] = saved


async def test_list_caps_limit(memories):
    """limit 이 과도해도 상한(200)으로 잘린다 — 한 번에 통째로 긁어가지 못하게."""
    result = await list_memories(limit=99999, authorization="tenant-a")

    assert result["total"] <= 200


# ─────────────────────────────────────────────
# 삭제 — IDOR 차단
# ─────────────────────────────────────────────


async def test_delete_removes_own_memory(memories):
    """자기 기억은 지워진다."""
    result = await delete_memory("a1", authorization="tenant-a")

    assert result["status"] == "ok"
    assert all(e.id != "a1" for e in memories.entries)


async def test_delete_other_tenant_memory_is_404(memories):
    """★남의 기억은 지워지지 않고 404 — 존재 여부도 알려주지 않는다."""
    with pytest.raises(HTTPException) as exc:
        await delete_memory("b1", authorization="tenant-a")

    assert exc.value.status_code == 404
    # 실제로 남아 있어야 한다(삭제 시도가 통과하면 안 된다).
    assert any(e.id == "b1" for e in memories.entries)


async def test_delete_unknown_id_is_404(memories):
    """없는 id도 남의 id와 똑같이 404 — 응답으로 구분되지 않는다."""
    with pytest.raises(HTTPException) as exc:
        await delete_memory("does-not-exist", authorization="tenant-a")

    assert exc.value.status_code == 404


async def test_delete_without_tenant_is_404(memories):
    """테넌트를 모르면 삭제할 수 없다."""
    with pytest.raises(HTTPException) as exc:
        await delete_memory("a1", authorization=None)

    assert exc.value.status_code == 404
    assert any(e.id == "a1" for e in memories.entries)
