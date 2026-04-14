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
from typing import Any

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger("nexus.web.middleware")


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
