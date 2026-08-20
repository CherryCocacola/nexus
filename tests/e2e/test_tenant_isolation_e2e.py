# 배포 서버에서 테넌트 격리가 실제로 서는지, 그리고 정상 사용이 살아 있는지 함께 본다.
"""
표면 회귀 — 테넌트 격리.

★이 스위트의 설계 원칙: **차단과 정상동작을 쌍으로 본다.**
    2026-08-20 에 격리를 넣고 차단만 확인했더니, 정작 소유 테넌트가 자기 세션을
    못 읽는 회귀가 배포된 채로 있었다(소유자 기록이 애초에 없었기 때문).
    "막혔다"만 보는 테스트는 절반짜리다.

계약:
    - 소유자 = 세션 meta.json 의 tenant (세 진입점에서 기록)
    - 기록이 없는 과거 세션 = 기본 테넌트 소유
    - 남의 세션 접근은 403 이 아니라 404 (존재를 알려주지 않는다)
"""

from __future__ import annotations

import time

import pytest

from tests.e2e.conftest import requires_server

pytestmark = [pytest.mark.e2e, requires_server]


@pytest.fixture
def owned_session(api, key_primary):
    """기본 테넌트로 표식이 든 대화를 만들고 (session_id, 표식) 을 돌려준다."""
    mark = f"격리표식-{int(time.time() * 1000)}"
    r = api.post(
        "/v1/chat",
        {"message": f"'{mark}' 를 그대로 한 번만 따라 써줘"},
        key=key_primary,
        timeout=300,
    )
    assert r.status_code == 200, r.text
    sid = r.json().get("session_id")
    assert sid, "세션 id 가 없다"
    time.sleep(2)  # 트랜스크립트·메타 기록이 디스크에 닿을 여유
    return sid, mark


# ─────────────────────────────────────────────
# 차단 — 남의 것은 못 본다
# ─────────────────────────────────────────────
def test_other_tenant_cannot_read_messages(api, key_second, owned_session):
    sid, mark = owned_session
    r = api.get(f"/v1/sessions/{sid}/messages", key=key_second)
    assert r.status_code == 404, f"남의 대화가 열렸다: {r.status_code}"
    assert mark not in r.text


def test_other_tenant_cannot_export(api, key_second, owned_session):
    sid, mark = owned_session
    r = api.get(f"/v1/sessions/{sid}/export", key=key_second, params={"fmt": "md"})
    assert r.status_code == 404
    assert mark not in r.text


def test_other_tenant_cannot_see_it_in_list(api, key_second, owned_session):
    sid, _ = owned_session
    r = api.get("/v1/sessions", key=key_second)
    assert r.status_code == 200
    ids = {s.get("session_id") for s in r.json().get("sessions", [])}
    assert sid not in ids


def test_other_tenant_search_finds_nothing(api, key_second, owned_session):
    """검색 결과 건수로 본다 — 응답의 query 필드에는 검색어가 그대로 되돌아온다."""
    _, mark = owned_session
    r = api.get("/v1/sessions/search", key=key_second, params={"q": mark})
    assert r.status_code == 200
    assert r.json().get("total", 0) == 0


def test_other_tenant_cannot_delete(api, key_second, key_primary, owned_session):
    """파괴적 조작이 특히 위험하다 — 남의 대화를 지울 수 있으면 안 된다."""
    sid, _ = owned_session
    r = api.post(f"/v1/sessions/{sid}/truncate", None, key=key_second, timeout=60)
    assert r.status_code == 404

    # 차단 후에도 원래 소유자에게는 그대로 남아 있어야 한다.
    check = api.get(f"/v1/sessions/{sid}/messages", key=key_primary)
    assert check.status_code == 200


# ─────────────────────────────────────────────
# 정상동작 — 자기 것은 그대로 쓴다 (여기가 빠지면 절반짜리다)
# ─────────────────────────────────────────────
def test_owner_can_read_own_session(api, key_primary, owned_session):
    sid, mark = owned_session
    r = api.get(f"/v1/sessions/{sid}/messages", key=key_primary)
    assert r.status_code == 200
    assert mark in r.text


def test_owner_can_export_own_session(api, key_primary, owned_session):
    sid, mark = owned_session
    r = api.get(f"/v1/sessions/{sid}/export", key=key_primary, params={"fmt": "md"})
    assert r.status_code == 200
    assert mark in r.text


def test_owner_sees_own_session_in_list(api, key_primary, owned_session):
    sid, _ = owned_session
    r = api.get("/v1/sessions", key=key_primary)
    assert r.status_code == 200
    ids = {s.get("session_id") for s in r.json().get("sessions", [])}
    assert sid in ids


def test_legacy_sessions_are_still_visible(api, key_primary):
    """과거 세션(tenant 미기록)은 기본 테넌트 소유로 본다 — 사용자가 기록을 잃으면 안 된다."""
    r = api.get("/v1/sessions", key=key_primary)
    assert r.status_code == 200
    assert len(r.json().get("sessions", [])) > 1, "목록이 비었다 — 과거 기록이 사라졌을 수 있다"


def test_second_tenant_can_use_its_own_session(api, key_second):
    """두 번째 테넌트도 자기 세션은 정상적으로 읽어야 한다(1차 수정 때 여기서 회귀가 났다)."""
    mark = f"자기세션-{int(time.time() * 1000)}"
    r = api.post(
        "/v1/chat",
        {"message": f"'{mark}' 를 그대로 한 번만 따라 써줘"},
        key=key_second,
        timeout=300,
    )
    assert r.status_code == 200, r.text
    sid = r.json().get("session_id")
    time.sleep(2)

    own = api.get(f"/v1/sessions/{sid}/messages", key=key_second)
    assert own.status_code == 200, "자기 세션을 못 읽는다 — 소유자 기록을 확인할 것"
