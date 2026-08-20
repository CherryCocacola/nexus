# 세션 계열 엔드포인트가 다른 테넌트의 대화를 넘겨주지 않는지 검증한다.
"""
테넌트 세션 격리 테스트 (2026-08-20).

무엇을 막는가 (실증된 결함):
    `default` 테넌트 키로 만든 대화를 `coding` 테넌트 키로 그대로 읽을 수 있었다.
    목록·검색·조회·내보내기뿐 아니라 truncate·fork·delete 같은 **파괴적 조작**까지
    남의 세션에 가능했다. 메모리·지침·프로젝트는 이미 테넌트로 막혀 있었으므로
    설계 의도가 아니라 누락이었다.

계약:
    - 소유자는 세션 meta.json 의 `tenant`(= _record_client_meta 가 기록).
    - 기록이 없는 과거 세션은 **기본 테넌트 소유**로 본다(사용자가 자기 기록을
      잃지 않게 한 결정). 이 방침이 바뀌면 이 테스트가 먼저 깨진다.
    - 남의 세션 접근은 403 이 아니라 **404** — 존재 여부를 알려주지 않는다.
"""

from __future__ import annotations

import pytest

from web import app as web_app


class _Tenant:
    def __init__(self, tid: str) -> None:
        self.id = tid


class _Registry:
    """기본 테넌트만 아는 최소 레지스트리 스텁."""

    def resolve(self, _key):
        return _Tenant("default")


@pytest.fixture
def app_state(monkeypatch, tmp_path):
    """세션 디렉토리와 레지스트리를 갈아끼운 _app_state 를 만든다."""

    class _Cfg:
        class session:  # noqa: N801 — config 객체 흉내
            sessions_dir = str(tmp_path)

    state = {"config": _Cfg(), "tenant_registry": _Registry()}
    monkeypatch.setattr(web_app, "_app_state", state)
    return state


def _write_meta(tmp_path, session_id: str, meta: dict, channel: str = "web") -> None:
    from core.memory.transcript import write_session_meta

    write_session_meta(str(tmp_path), session_id, meta, channel=channel)


# ─────────────────────────────────────────────
# 소유자 판정
# ─────────────────────────────────────────────
def test_owner_comes_from_meta(app_state, tmp_path):
    """meta.json 의 tenant 가 소유자다."""
    _write_meta(tmp_path, "s-1", {"tenant": "coding"})
    assert web_app._session_owner("s-1") == "coding"


def test_legacy_session_belongs_to_default_tenant(app_state, tmp_path):
    """tenant 기록이 없는 과거 세션은 기본 테넌트 소유로 본다."""
    _write_meta(tmp_path, "s-legacy", {"pinned": True})
    assert web_app._session_owner("s-legacy") == "default"


def test_unknown_session_is_default_owned(app_state):
    """meta 자체가 없어도 판정이 깨지지 않는다(기본 테넌트)."""
    assert web_app._session_owner("s-missing") == "default"


def test_api_channel_session_owner_is_found(app_state, tmp_path):
    """api 채널(플러그인·AgentHub) 세션의 소유자도 찾는다."""
    _write_meta(tmp_path, "s-api", {"tenant": "agenthub"}, channel="api")
    assert web_app._session_owner("s-api") == "agenthub"


# ─────────────────────────────────────────────
# 접근 차단
# ─────────────────────────────────────────────
def test_other_tenant_gets_404(app_state, tmp_path):
    """남의 세션은 403 이 아니라 404 — 존재를 알려주지 않는다."""
    from fastapi import HTTPException

    _write_meta(tmp_path, "s-2", {"tenant": "default"})
    with pytest.raises(HTTPException) as e:
        web_app._require_session_owner("s-2", _Tenant("coding"))
    assert e.value.status_code == 404


def test_owner_passes(app_state, tmp_path):
    """소유자는 통과한다."""
    _write_meta(tmp_path, "s-3", {"tenant": "coding"})
    web_app._require_session_owner("s-3", _Tenant("coding"))  # 예외 없음


def test_no_tenant_registry_keeps_old_behavior(app_state, tmp_path):
    """테넌트를 해석할 수 없으면(단일 테넌트 개발) 검사하지 않는다 — 무회귀."""
    _write_meta(tmp_path, "s-4", {"tenant": "coding"})
    web_app._require_session_owner("s-4", None)  # 예외 없음


# ─────────────────────────────────────────────
# 목록 필터
# ─────────────────────────────────────────────
def test_listed_session_filter(app_state):
    """목록 항목 필터 — 미기록은 기본 테넌트 것으로 본다."""
    assert web_app._owns_listed_session({"tenant": "coding"}, "coding") is True
    assert web_app._owns_listed_session({"tenant": "coding"}, "default") is False
    assert web_app._owns_listed_session({"tenant": None}, "default") is True
    assert web_app._owns_listed_session({"tenant": None}, "coding") is False
    # 호출자를 모르면(레지스트리 없음) 거르지 않는다.
    assert web_app._owns_listed_session({"tenant": "coding"}, "") is True


# ─────────────────────────────────────────────
# 배선 — 엔드포인트가 실제로 검사를 부르는가
# ─────────────────────────────────────────────
@pytest.mark.parametrize(
    "func",
    [
        "get_session_messages",
        "export_session",
        "truncate_last_exchange",
        "fork_session",
        "delete_session",
    ],
)
def test_endpoint_calls_owner_guard(func: str):
    """값이 아니라 배선을 본다 — 한 곳만 빠져도 그 경로로 다 새어 나간다."""
    import inspect

    src = inspect.getsource(getattr(web_app, func))
    assert "_require_session_owner(" in src, f"{func} 에 소유자 검사가 없다"
    # 인증 정보를 받는지도 함께 확인한다(헤더가 없으면 항상 기본 테넌트로 해석된다).
    assert "authorization" in src, f"{func} 이 Authorization 을 받지 않는다"


@pytest.mark.parametrize("func", ["list_sessions", "search_sessions"])
def test_list_endpoints_filter_by_tenant(func: str):
    """목록·검색은 예외가 아니라 필터로 막는다."""
    import inspect

    src = inspect.getsource(getattr(web_app, func))
    assert "_resolve_tenant(" in src, f"{func} 이 테넌트를 해석하지 않는다"
    assert ("_owns_listed_session(" in src) or ("_session_owner(" in src), (
        f"{func} 에 소유자 필터가 없다"
    )

# ─────────────────────────────────────────────
# 소유자 기록 — 격리의 전제 데이터
# ─────────────────────────────────────────────
def test_owner_is_recorded_for_new_session(app_state, tmp_path):
    """소유자를 기록해야 격리가 성립한다.

    실측 배경: 기존 `_record_client_meta` 는 X-Client-Id 가 있고 채널이 web 이 아닐
    때만 돌아, **tenant 가 기록된 세션이 하나도 없었다**(meta.json 13개 중 0개).
    그 상태에서는 모든 세션이 기본 테넌트 소유로 보여 격리가 무의미하다.
    """
    web_app._record_session_owner("s-new", "web", _Tenant("coding"))
    assert web_app._session_owner("s-new") == "coding"


def test_owner_record_is_idempotent(app_state, tmp_path):
    """같은 값이면 다시 쓰지 않는다(매 턴 파일 I/O 방지)."""
    from core.memory.transcript import read_session_meta

    web_app._record_session_owner("s-idem", "web", _Tenant("coding"))
    before = read_session_meta(str(tmp_path), "s-idem", channel="web")
    web_app._record_session_owner("s-idem", "web", _Tenant("coding"))
    after = read_session_meta(str(tmp_path), "s-idem", channel="web")
    assert before == after


def test_owner_record_keeps_other_meta(app_state, tmp_path):
    """제목·핀 같은 기존 메타를 지우지 않는다(병합 저장)."""
    from core.memory.transcript import read_session_meta

    _write_meta(tmp_path, "s-merge", {"title": "내 대화", "pinned": True})
    web_app._record_session_owner("s-merge", "web", _Tenant("default"))
    meta = read_session_meta(str(tmp_path), "s-merge", channel="web")
    assert meta.get("title") == "내 대화" and meta.get("pinned") is True
    assert meta.get("tenant") == "default"


def test_owner_record_without_tenant_is_noop(app_state, tmp_path):
    """테넌트를 모르면 아무것도 쓰지 않는다(단일 테넌트 개발 환경)."""
    web_app._record_session_owner("s-none", "web", None)
    assert web_app._session_owner("s-none") == "default"


@pytest.mark.parametrize("handler", ["chat", "chat_stream", "chat_completions"])
def test_every_chat_handler_records_owner(handler: str):
    """세 진입점 모두 기록해야 한다 — 한 곳만 빠져도 그 경로 세션이 무주공산이 된다."""
    import inspect

    src = inspect.getsource(getattr(web_app, handler))
    assert "_record_session_owner(" in src, f"{handler} 이 소유자를 기록하지 않는다"
