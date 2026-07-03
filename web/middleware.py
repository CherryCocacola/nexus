"""
미들웨어 — 요청 로깅 + CORS 설정.

FastAPI 앱에 적용되는 미들웨어를 정의한다.

의존성 방향: web/ → core/ (단방향)

주요 구성:
  - RequestLoggingMiddleware: 모든 요청/응답을 로깅하고 메트릭스를 수집한다
  - CORSConfig: 로컬 환경에 적합한 CORS 설정을 제공한다
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

logger = logging.getLogger("nexus.web.middleware")


# ─────────────────────────────────────────────
# API 키 인증 미들웨어 (Security Critical #4, 2026-07-02)
# ─────────────────────────────────────────────
class ApiKeyAuthMiddleware(BaseHTTPMiddleware):
    """
    웹 API 키 인증 게이트 (fail-closed).

    동작 요약:
      - 인증이 꺼져 있으면(enabled=False) 모든 요청을 통과시키되, 최초 1회
        경고 로그를 남긴다(신뢰된 네트워크 전용임을 운영자에게 알림).
      - 인증이 켜져 있으면(enabled=True) exempt_paths(면제 경로)가 아닌 모든
        요청에 대해 `Authorization: Bearer <key>` 헤더를 요구하고, 그 키가
        TenantRegistry.resolve_by_api_key()로 유효 테넌트를 찾을 때만 통과시킨다.
        헤더 없음/형식 오류/미매칭은 모두 401로 차단한다(fail-closed).

    왜 설정/레지스트리를 "콜러블"로 주입받는가 (순환 import 방지):
      web/middleware.py는 core/, web/app.py를 직접 import하지 않는다. 대신
      app.py가 `_app_state`에서 설정·테넌트 레지스트리를 읽어오는 무인자 함수를
      넘겨준다. 미들웨어는 요청 시점(dispatch)에 그 함수를 호출해 최신 값을 얻는다.
      (미들웨어는 앱 import 시점에 등록되지만, 설정은 lifespan 기동 후에야 채워지므로
       생성 시점이 아니라 dispatch 시점에 지연 조회해야 한다.)

    인증과 테넌트 해석의 분리:
      이 미들웨어는 "유효한 키인가"만 판정한다(인증). 실제 어떤 테넌트로 라우팅할지
      (body.tenant_id > X-Tenant-ID > Bearer 키 우선순위)는 기존 app._resolve_tenant가
      그대로 담당한다(인가/해석). 두 관심사를 섞지 않는다.
    """

    def __init__(
        self,
        app: Any,
        *,
        get_auth_config: Callable[[], Any],
        get_tenant_registry: Callable[[], Any],
    ):
        """
        미들웨어를 초기화한다.

        Args:
            app: FastAPI/Starlette ASGI 앱 인스턴스
            get_auth_config: WebAuthConfig(또는 None)를 반환하는 무인자 함수
            get_tenant_registry: TenantRegistry(또는 None)를 반환하는 무인자 함수
        """
        super().__init__(app)
        self._get_auth_config = get_auth_config
        self._get_tenant_registry = get_tenant_registry
        # 인증 비활성 경고를 최초 1회만 남기기 위한 플래그(dispatch마다 폭주 방지).
        self._warned_disabled: bool = False

    def _is_exempt(self, path: str, exempt_paths: list[str]) -> bool:
        """요청 경로가 면제 경로에 해당하는지 판정한다.

        매칭 규칙:
          - "/" 항목은 정확히 루트 경로("/")에만 매칭한다. (prefix로 처리하면
            모든 경로가 "/"로 시작하므로 인증이 통째로 무력화되는 사고를 막는다.)
          - 그 외 항목은 prefix 매칭한다(예: "/static" → "/static/app.js" 허용).
        """
        for exempt in exempt_paths:
            if exempt == "/":
                if path == "/":
                    return True
            elif path.startswith(exempt):
                return True
        return False

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """요청을 인증 검사한 뒤 다음 핸들러로 넘기거나 401로 차단한다."""
        auth = self._get_auth_config()

        # 설정이 아직 없거나(기동 전) 인증이 꺼져 있으면 통과시킨다(무회귀).
        # enabled는 반드시 "명시적 boolean True"일 때만 인증을 켠다(`is True`).
        #   - WebAuthConfig.enabled는 Pydantic이 실제 bool로 강제하므로 정상 동작.
        #   - 설정이 잘못 배선돼 bool이 아닌 truthy 값(예: 테스트의 MagicMock)이
        #     들어와도 "실수로 인증이 켜지는" 오작동을 막는다(fail-safe 방향).
        if auth is None or getattr(auth, "enabled", False) is not True:
            if not self._warned_disabled:
                self._warned_disabled = True
                logger.warning(
                    "웹 인증이 비활성화됨 — 신뢰된 네트워크(에어갭 LAN)에서만 사용하십시오. "
                    "배포 환경에서는 NEXUS_WEB_AUTH__ENABLED=true로 인증을 켜야 합니다."
                )
            return await call_next(request)

        path = request.url.path

        # CORS 프리플라이트(OPTIONS)는 자격증명을 싣지 않으므로 인증에서 면제한다.
        # (이 미들웨어가 CORS보다 바깥이라 프리플라이트를 먼저 만나므로, 여기서
        #  막으면 브라우저 CORS가 깨진다. 실제 요청은 여전히 인증 대상이다.)
        if request.method == "OPTIONS":
            return await call_next(request)

        # 면제 경로는 인증 없이 통과.
        if self._is_exempt(path, list(getattr(auth, "exempt_paths", []))):
            return await call_next(request)

        # Authorization: Bearer <key> 헤더를 요구한다(fail-closed).
        authorization = request.headers.get("Authorization", "")
        if not authorization.lower().startswith("bearer "):
            return self._unauthorized()

        api_key = authorization[7:].strip()
        if not api_key:
            return self._unauthorized()

        registry = self._get_tenant_registry()
        # 레지스트리가 없으면 검증 불가 → fail-closed로 차단한다.
        if registry is None:
            return self._unauthorized()

        tenant = registry.resolve_by_api_key(api_key)
        if tenant is None:
            return self._unauthorized()

        # 인증 통과 — 다음 핸들러로 진행(테넌트 해석은 기존 _resolve_tenant가 담당).
        return await call_next(request)

    @staticmethod
    def _unauthorized() -> JSONResponse:
        """표준 401 JSON 응답을 만든다."""
        return JSONResponse(
            status_code=401,
            content={"detail": "인증 실패: 유효한 API 키가 필요합니다"},
        )


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    요청/응답 로깅 + 메트릭스 수집 미들웨어.

    모든 HTTP 요청에 대해:
      1. 요청 시작 시간을 기록한다
      2. 응답 완료 후 지연 시간을 계산한다
      3. JSONL 형식으로 로그를 남긴다
      4. 메트릭스를 누적한다 (총 요청 수, 에러 수, 평균 지연)
    """

    def __init__(self, app: Any):
        """
        미들웨어를 초기화한다.

        Args:
            app: FastAPI/Starlette ASGI 앱 인스턴스
        """
        super().__init__(app)
        # 메트릭스 누적 변수
        self._total_requests: int = 0
        self._total_errors: int = 0
        self._total_latency_ms: float = 0.0

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """
        요청을 처리하고 로깅한다.

        모든 요청에 대해 시작/종료 시간을 기록하고,
        지연 시간과 상태 코드를 로깅한다.

        Args:
            request: 수신된 HTTP 요청
            call_next: 다음 미들웨어/핸들러 호출 함수

        Returns:
            처리된 HTTP 응답
        """
        # 요청 시작 시간 기록
        start_time = time.monotonic()
        self._total_requests += 1

        # 요청 정보 로깅
        method = request.method
        path = request.url.path
        client_ip = request.client.host if request.client else "unknown"

        logger.info(f"요청 시작: {method} {path} | client={client_ip}")

        try:
            # 다음 핸들러 실행
            response = await call_next(request)

            # 지연 시간 계산
            latency_ms = (time.monotonic() - start_time) * 1000
            self._total_latency_ms += latency_ms

            # 에러 응답 카운트
            if response.status_code >= 400:
                self._total_errors += 1

            # 응답 로깅
            logger.info(
                f"요청 완료: {method} {path} | "
                f"status={response.status_code} | "
                f"latency={latency_ms:.1f}ms | "
                f"client={client_ip}"
            )

            # 응답 헤더에 처리 시간을 추가한다
            response.headers["X-Process-Time-Ms"] = f"{latency_ms:.1f}"

            return response

        except Exception as e:
            # 처리되지 않은 예외 로깅
            latency_ms = (time.monotonic() - start_time) * 1000
            self._total_errors += 1
            self._total_latency_ms += latency_ms

            logger.error(
                f"요청 실패: {method} {path} | "
                f"error={type(e).__name__}: {e} | "
                f"latency={latency_ms:.1f}ms"
            )
            raise

    @property
    def metrics(self) -> dict[str, Any]:
        """
        누적 메트릭스를 반환한다.

        /metrics 엔드포인트에서 이 데이터를 노출한다.
        """
        avg_latency = (
            self._total_latency_ms / self._total_requests if self._total_requests > 0 else 0.0
        )
        return {
            "total_requests": self._total_requests,
            "total_errors": self._total_errors,
            "error_rate": (
                self._total_errors / self._total_requests if self._total_requests > 0 else 0.0
            ),
            "avg_latency_ms": round(avg_latency, 2),
        }


class CORSConfig:
    """
    로컬 환경 CORS 설정.

    에어갭 환경이므로 외부 도메인은 허용하지 않는다.
    로컬 개발 환경(localhost, LAN IP)만 허용한다.
    """

    # 허용할 오리진 목록 — 로컬/LAN 주소만 허용한다
    ALLOWED_ORIGINS: list[str] = [
        "http://localhost:3000",  # 프론트엔드 개발 서버
        "http://localhost:8080",  # 웹 UI
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8080",
    ]

    # 허용할 HTTP 메서드
    ALLOWED_METHODS: list[str] = ["GET", "POST", "PUT", "DELETE", "OPTIONS"]

    # 허용할 헤더
    ALLOWED_HEADERS: list[str] = [
        "Content-Type",
        "Authorization",
        "X-Session-ID",
        "X-Request-ID",
    ]

    @classmethod
    def get_cors_kwargs(cls) -> dict[str, Any]:
        """
        FastAPI CORSMiddleware에 전달할 설정을 반환한다.

        사용 예시:
            from fastapi.middleware.cors import CORSMiddleware
            app.add_middleware(CORSMiddleware, **CORSConfig.get_cors_kwargs())
        """
        return {
            "allow_origins": cls.ALLOWED_ORIGINS,
            "allow_credentials": True,
            "allow_methods": cls.ALLOWED_METHODS,
            "allow_headers": cls.ALLOWED_HEADERS,
        }

    @classmethod
    def add_lan_origin(cls, ip: str, port: int = 8080) -> None:
        """
        LAN IP 주소를 CORS 허용 목록에 추가한다.

        에어갭 환경에서 같은 네트워크 내 다른 머신에서 접근할 때 사용한다.
        외부 네트워크 주소는 거부한다.

        Args:
            ip: LAN IP 주소 (192.168.x.x, 10.x.x.x, 172.16~31.x.x)
            port: 포트 번호
        """
        # 에어갭 검증: LAN 주소만 허용한다
        allowed_prefixes = (
            "192.168.",
            "10.",
            "172.16.",
            "172.17.",
            "172.18.",
            "172.19.",
            "172.20.",
            "172.21.",
            "172.22.",
            "172.23.",
            "172.24.",
            "172.25.",
            "172.26.",
            "172.27.",
            "172.28.",
            "172.29.",
            "172.30.",
            "172.31.",
        )
        if not any(ip.startswith(prefix) for prefix in allowed_prefixes):
            logger.warning(f"LAN이 아닌 주소 거부: {ip}")
            return

        origin = f"http://{ip}:{port}"
        if origin not in cls.ALLOWED_ORIGINS:
            cls.ALLOWED_ORIGINS.append(origin)
            logger.info(f"CORS 오리진 추가: {origin}")
