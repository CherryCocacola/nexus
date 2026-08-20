# 배포 표면 E2E 테스트가 공유하는 접속 정보와 자동 스킵 규칙을 모아 둔다.
"""
표면 회귀 E2E 공통 설정.

왜 이 파일이 필요한가 (2026-08-20):
    단위 테스트 2,600여 개가 전부 통과하는 동안 실서버는 404 를 뱉고 있었다.
    단위 테스트는 값을 검증하지 **경로를 통과시키지 않는다**. 그래서 배포된
    서버를 실제로 두들기는 스위트를 두고, 배포 전후에 돌린다.

실행 방법:
    pytest tests/e2e -m e2e                      # 기본 대상(배포 서버)
    NEXUS_E2E_BASE=http://127.0.0.1:8600 pytest tests/e2e -m e2e
    NEXUS_E2E_KEY=... NEXUS_E2E_KEY2=... pytest tests/e2e -m e2e

    서버에 닿지 않으면 **자동 스킵**된다(기존 e2e 관례와 동일). 그러니 CI 나
    개발 PC 에서 전체 pytest 를 돌려도 실패하지 않는다.

키를 코드에 넣지 않는 이유:
    테넌트 키는 배포마다 다르고, 리포에 박으면 그 자체가 노출이다. 환경변수로
    받고, 없으면 배포 기본 테스트 키를 쓴다(운영 키가 아니다).
"""

from __future__ import annotations

import os
import socket
from urllib.parse import urlparse

import pytest

# 기본 대상 — 112 서비스 서버. 환경변수로 갈아끼울 수 있다.
BASE_URL = os.environ.get("NEXUS_E2E_BASE", "http://192.168.21.112:8600")

# 테넌트 두 개가 필요하다 — 격리는 "서로 못 본다"를 봐야 검증되기 때문이다.
KEY_PRIMARY = os.environ.get("NEXUS_E2E_KEY", "nexus-b200-test-key-001")
KEY_SECOND = os.environ.get("NEXUS_E2E_KEY2", "nexus-coding-key-001")


def _server_reachable(url: str, timeout: float = 3.0) -> bool:
    """대상 서버의 포트가 열려 있는지 확인한다(기존 e2e 스킵 관례와 동일)."""
    parsed = urlparse(url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        s = socket.socket()
        s.settimeout(timeout)
        ok = s.connect_ex((host, port)) == 0
        s.close()
        return ok
    except OSError:
        return False


requires_server = pytest.mark.skipif(
    not _server_reachable(BASE_URL),
    reason=f"배포 서버에 닿지 않는다: {BASE_URL}",
)


@pytest.fixture(scope="session")
def base_url() -> str:
    return BASE_URL


@pytest.fixture(scope="session")
def key_primary() -> str:
    return KEY_PRIMARY


@pytest.fixture(scope="session")
def key_second() -> str:
    return KEY_SECOND


@pytest.fixture(scope="session")
def api():
    """간단한 HTTP 헬퍼 — 표면 테스트가 공유한다."""
    import httpx

    class _Api:
        def __init__(self) -> None:
            self.base = BASE_URL

        def post(self, path: str, payload=None, key: str | None = None,
                 headers: dict | None = None, timeout: float = 180.0, files=None):
            h = dict(headers or {})
            if key:
                h["Authorization"] = f"Bearer {key}"
            with httpx.Client(timeout=timeout) as c:
                if files is not None:
                    return c.post(self.base + path, files=files, headers=h)
                return c.post(self.base + path, json=payload, headers=h)

        def get(self, path: str, key: str | None = None, params: dict | None = None,
                timeout: float = 60.0):
            h = {"Authorization": f"Bearer {key}"} if key else {}
            with httpx.Client(timeout=timeout) as c:
                return c.get(self.base + path, headers=h, params=params)

        def chat(self, message: str, key: str, max_tokens: int = 120,
                 timeout: float = 300.0, **extra):
            """OpenAI 호환 경로로 한 번 물어보고 본문 텍스트를 돌려준다."""
            payload = {
                "model": "ax-4.0",
                "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": message}],
            }
            payload.update(extra)
            r = self.post("/v1/chat/completions", payload, key=key, timeout=timeout)
            if r.status_code != 200:
                return r.status_code, r.text
            body = r.json()
            return 200, body["choices"][0]["message"]["content"]

    return _Api()


def korean_ratio(text: str) -> float:
    """한글 비율 — 모델을 갈아끼웠을 때 답변 언어가 바뀌는 것을 잡는다."""
    ko = sum(1 for ch in text if "가" <= ch <= "힣")
    en = sum(1 for ch in text if ch.isascii() and ch.isalpha())
    return ko / (ko + en) if (ko + en) else 0.0
