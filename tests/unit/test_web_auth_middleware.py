"""
웹 API 키 인증 미들웨어 단위 테스트 (Security Critical #4, 2026-07-02).

검증 대상:
  - web/middleware.py::ApiKeyAuthMiddleware — Bearer API 키 인증 게이트(fail-closed)
  - core/config.py::WebAuthConfig — 인증 설정 기본값 + 환경변수 배선

Mock/격리 전략:
  - 실제 web/app.py(FastAPI 전체)를 띄우지 않는다. 미들웨어만 검증하기 위해
    최소 Starlette 앱(더미 라우트)에 미들웨어를 얹고 starlette TestClient로
    요청한다. 외부 서비스(vLLM/DB/Redis)는 전혀 건드리지 않는다.
  - 미들웨어는 설정/테넌트 레지스트리를 "무인자 콜러블"로 주입받으므로,
    테스트마다 원하는 WebAuthConfig / TenantRegistry 를 클로저로 넘겨 격리한다.
  - TenantRegistry / WebAuthConfig 는 실제 Pydantic 모델을 그대로 사용한다
    (enabled 가 실제 bool 로 강제되는 동작까지 함께 검증하기 위함).
"""

from __future__ import annotations

from typing import Any

from starlette.applications import Starlette
from starlette.responses import PlainTextResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from core.config import TenantConfig, TenantRegistry, WebAuthConfig
from web.middleware import ApiKeyAuthMiddleware

# 유효 API 키가 등록된 테넌트가 반환할 표준 401 본문(소스와 정확히 일치해야 함).
_UNAUTHORIZED_DETAIL = "인증 실패: 유효한 API 키가 필요합니다"
# 테스트에서 사용할 유효 API 키.
_VALID_KEY = "secret-key-123"


def _make_client(auth: Any, registry: Any) -> TestClient:
    """주어진 인증 설정/레지스트리로 미들웨어를 얹은 TestClient 를 만든다.

    더미 라우트:
      - "/"          : 루트(정확 매칭 면제 경로 검증용)
      - "/v1/chat"   : 보호 대상 API 경로
      - "/health"    : 면제 경로(prefix) 검증용
      - "/static/app.js" : "/static" prefix 면제 검증용
    OPTIONS 를 허용 메서드에 포함해 프리플라이트 통과 시 200 을 확인할 수 있게 한다.
    """

    async def _ok(request: Any) -> PlainTextResponse:
        return PlainTextResponse("ok")

    routes = [
        Route("/", _ok, methods=["GET", "POST", "OPTIONS"]),
        Route("/v1/chat", _ok, methods=["GET", "POST", "OPTIONS"]),
        Route("/health", _ok, methods=["GET"]),
        Route("/static/app.js", _ok, methods=["GET"]),
    ]
    app = Starlette(routes=routes)
    # 미들웨어는 dispatch 시점에 콜러블을 호출해 최신 설정을 지연 조회한다.
    app.add_middleware(
        ApiKeyAuthMiddleware,
        get_auth_config=lambda: auth,
        get_tenant_registry=lambda: registry,
    )
    # raise_server_exceptions=False 는 여기선 불필요하지만, 401 JSON 응답을
    # 그대로 받기 위해 기본값(True)으로 둔다.
    return TestClient(app)


def _registry_with_valid_key() -> TenantRegistry:
    """유효 API 키를 가진 테넌트가 등록된 레지스트리를 만든다."""
    return TenantRegistry(
        tenants=[
            TenantConfig(id="acme", name="Acme", api_keys=[_VALID_KEY]),
        ]
    )


# ─────────────────────────────────────────────
# ApiKeyAuthMiddleware — 인증 비활성(무회귀)
# ─────────────────────────────────────────────
def test_auth_disabled_allows_request_without_key() -> None:
    """enabled=False 면 키 없이도 보호 경로가 통과된다(기존 무인증 동작 유지)."""
    client = _make_client(auth=WebAuthConfig(enabled=False), registry=_registry_with_valid_key())

    resp = client.get("/v1/chat")

    assert resp.status_code == 200
    assert resp.text == "ok"


def test_auth_config_none_allows_request_without_key() -> None:
    """설정이 아직 없으면(기동 전, auth=None) 통과시킨다(무회귀)."""
    client = _make_client(auth=None, registry=None)

    resp = client.get("/v1/chat")

    assert resp.status_code == 200


# ─────────────────────────────────────────────
# ApiKeyAuthMiddleware — 인증 활성(fail-closed)
# ─────────────────────────────────────────────
def test_auth_enabled_missing_header_returns_401() -> None:
    """enabled=True 인데 Authorization 헤더가 없으면 401."""
    client = _make_client(auth=WebAuthConfig(enabled=True), registry=_registry_with_valid_key())

    resp = client.get("/v1/chat")

    assert resp.status_code == 401
    assert resp.json()["detail"] == _UNAUTHORIZED_DETAIL


def test_auth_enabled_malformed_header_returns_401() -> None:
    """Bearer 스킴이 아닌 헤더(예: 'Token abc')는 401."""
    client = _make_client(auth=WebAuthConfig(enabled=True), registry=_registry_with_valid_key())

    resp = client.get("/v1/chat", headers={"Authorization": "Token abc"})

    assert resp.status_code == 401
    assert resp.json()["detail"] == _UNAUTHORIZED_DETAIL


def test_auth_enabled_empty_key_returns_401() -> None:
    """'Bearer ' 뒤에 키가 비어 있으면 401(빈 키 거부)."""
    client = _make_client(auth=WebAuthConfig(enabled=True), registry=_registry_with_valid_key())

    resp = client.get("/v1/chat", headers={"Authorization": "Bearer    "})

    assert resp.status_code == 401
    assert resp.json()["detail"] == _UNAUTHORIZED_DETAIL


def test_auth_enabled_invalid_key_returns_401() -> None:
    """레지스트리에 없는 키는 401(미매칭 차단)."""
    client = _make_client(auth=WebAuthConfig(enabled=True), registry=_registry_with_valid_key())

    resp = client.get("/v1/chat", headers={"Authorization": "Bearer wrong-key"})

    assert resp.status_code == 401
    assert resp.json()["detail"] == _UNAUTHORIZED_DETAIL


def test_auth_enabled_valid_key_allows_request() -> None:
    """레지스트리에 등록된 유효 키는 통과(200)."""
    client = _make_client(auth=WebAuthConfig(enabled=True), registry=_registry_with_valid_key())

    resp = client.get("/v1/chat", headers={"Authorization": f"Bearer {_VALID_KEY}"})

    assert resp.status_code == 200
    assert resp.text == "ok"


def test_auth_enabled_bearer_scheme_case_insensitive_allows() -> None:
    """스킴 매칭은 대소문자 무시('bearer ...')여도 유효 키면 통과."""
    client = _make_client(auth=WebAuthConfig(enabled=True), registry=_registry_with_valid_key())

    resp = client.get("/v1/chat", headers={"Authorization": f"bearer {_VALID_KEY}"})

    assert resp.status_code == 200


def test_auth_enabled_exempt_path_allows_without_key() -> None:
    """면제 경로(/health)는 키 없이 통과(모니터링 폴링 허용)."""
    client = _make_client(auth=WebAuthConfig(enabled=True), registry=_registry_with_valid_key())

    resp = client.get("/health")

    assert resp.status_code == 200


def test_auth_enabled_exempt_prefix_allows_static_asset() -> None:
    """'/static' prefix 면제로 하위 자산(/static/app.js)도 키 없이 통과."""
    client = _make_client(auth=WebAuthConfig(enabled=True), registry=_registry_with_valid_key())

    resp = client.get("/static/app.js")

    assert resp.status_code == 200


def test_auth_enabled_root_path_exact_match_exempt() -> None:
    """'/' 면제는 정확 매칭이라 루트는 통과하지만 '/v1/...'는 401.

    prefix 오남용(모든 경로가 '/'로 시작 → 인증 무력화)을 막는 회귀 테스트.
    """
    client = _make_client(auth=WebAuthConfig(enabled=True), registry=_registry_with_valid_key())

    # 루트는 정확 매칭 면제 → 200
    root_resp = client.get("/")
    assert root_resp.status_code == 200

    # '/'로 시작하지만 루트가 아닌 보호 경로 → 401(인증 요구)
    api_resp = client.get("/v1/chat")
    assert api_resp.status_code == 401
    assert api_resp.json()["detail"] == _UNAUTHORIZED_DETAIL


def test_auth_enabled_options_preflight_bypasses() -> None:
    """OPTIONS(CORS 프리플라이트)는 인증 면제로 통과한다(401 아님)."""
    client = _make_client(auth=WebAuthConfig(enabled=True), registry=_registry_with_valid_key())

    resp = client.options("/v1/chat")

    # 프리플라이트는 자격증명을 싣지 않으므로 인증 게이트를 통과해야 한다.
    assert resp.status_code != 401
    assert resp.status_code == 200


def test_auth_registry_none_returns_401() -> None:
    """enabled=True 인데 테넌트 레지스트리가 미구성(None)이면 검증 불가 → 401."""
    client = _make_client(auth=WebAuthConfig(enabled=True), registry=None)

    resp = client.get("/v1/chat", headers={"Authorization": f"Bearer {_VALID_KEY}"})

    assert resp.status_code == 401
    assert resp.json()["detail"] == _UNAUTHORIZED_DETAIL


def test_auth_enabled_non_bool_truthy_does_not_enable() -> None:
    """enabled 가 bool True 가 아닌 truthy 값이면 인증을 켜지 않는다(fail-safe).

    미들웨어는 `getattr(auth, 'enabled', False) is not True` 로 판정하므로,
    잘못 배선된 truthy 값(예: 문자열 'true')으로 실수로 인증이 켜지지 않는다.
    """

    class _FakeAuth:
        # bool 이 아닌 truthy 문자열 — 실제 인증을 켜면 안 된다.
        enabled = "true"
        exempt_paths: list[str] = []

    client = _make_client(auth=_FakeAuth(), registry=_registry_with_valid_key())

    # 키가 없어도 통과해야 한다(enabled is not True → 무회귀 통과 경로).
    resp = client.get("/v1/chat")

    assert resp.status_code == 200


# ─────────────────────────────────────────────
# WebAuthConfig — 기본값 + 환경변수 배선
# ─────────────────────────────────────────────
def test_web_auth_config_default_disabled() -> None:
    """WebAuthConfig 기본 enabled 는 False(무회귀 원칙)."""
    cfg = WebAuthConfig()

    assert cfg.enabled is False


def test_web_auth_config_default_exempt_paths() -> None:
    """WebAuthConfig 기본 면제 경로 목록이 사양과 일치한다."""
    cfg = WebAuthConfig()

    assert cfg.exempt_paths == [
        "/health",
        "/",
        "/docs",
        "/openapi.json",
        "/static",
    ]


def test_nexus_config_web_auth_default_instance_exists() -> None:
    """NexusConfig.web_auth 기본 인스턴스가 존재하고 기본 비활성이다."""
    from core.config import NexusConfig

    # 기본 팩토리로 만든 WebAuthConfig 가 배선되어 있는지 확인.
    default_web_auth = NexusConfig.model_fields["web_auth"].default_factory()
    assert isinstance(default_web_auth, WebAuthConfig)
    assert default_web_auth.enabled is False


def test_web_auth_config_env_enables_auth(monkeypatch: Any) -> None:
    """NEXUS_WEB_AUTH__ENABLED=true 환경변수로 인증이 켜진다(중첩 델리미터 배선)."""
    from core.config import NexusConfig

    # env_prefix='NEXUS_' + env_nested_delimiter='__' → web_auth.enabled 로 배선.
    monkeypatch.setenv("NEXUS_WEB_AUTH__ENABLED", "true")

    cfg = NexusConfig()

    assert cfg.web_auth.enabled is True
