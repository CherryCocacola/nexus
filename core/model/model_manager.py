"""
모델 매니저 — Machine A 측 모델 관리 및 hot-swap 조율 모듈.

[이 파일이 하는 일 — 3줄 요약]
  이 모듈은 Machine A(오케스트레이터)에서 실행되며, 실제 GPU 연산이 도는
  Machine B(GPU 서버)의 모델 상태를 "원격으로" 관리한다. 즉 이 파일 안에는
  torch/CUDA가 전혀 없고, 오직 HTTP(httpx)로 GPU 서버에 명령을 보낼 뿐이다.

[2-Machine 토폴로지 배경 지식 — 초보자용]
  Nexus는 두 대의 머신으로 나뉜다.
    - Machine A(오케스트레이터): 파이썬 asyncio, CLI/쿼리루프/도구/권한/메모리 담당.
    - Machine B(GPU 서버): vLLM + FastAPI, 실제 모델 추론(Qwen/ExaOne)을 수행.
  두 머신은 LAN 전용 HTTP/SSE로만 통신한다. 그래서 Machine A에 있는 이 매니저는
  GPU를 직접 만지지 못하고, Machine B가 열어 둔 REST 엔드포인트를 호출해서
  "모델을 바꿔줘 / LoRA를 올려줘 / 지금 상태 어때?" 하고 부탁하는 역할만 한다.

[주요 책임 4가지]
  1. 현재 활성 모델 추적 (primary=Qwen / auxiliary=ExaOne 중 무엇이 떠 있는지)
  2. Hot-swap 요청 조율 (primary ↔ auxiliary 전환을 GPU 서버에 요청)
  3. LoRA 체크포인트 로드/언로드 요청 (테넌트/작업별 어댑터 부착·제거)
  4. 모델 헬스 체크 (GPU 서버가 살아 있는지, 활성 모델·LoRA 목록 동기화)

[왜 Machine A에 이런 매니저가 따로 필요한가 — 설계 의도]
  QueryEngine 같은 상위 코드가 모델을 전환하거나 상태를 확인할 때, GPU 서버 API를
  여기저기서 직접 호출하면 URL·인증 헤더·타임아웃 같은 세부사항이 코드 전반에
  흩어져 결합도가 높아진다. 그래서 그 통신 세부사항을 이 클래스 한 곳에 모아
  추상화하면, 상위 코드는 swap_model()/health_check() 같은 간단한 메서드만
  부르면 되고, 테스트할 때도 이 클래스만 mock 하면 되어 훨씬 편해진다.

[주요 구성 요소]
  - ActiveModel: 활성 모델을 나타내는 문자열 Enum (primary/auxiliary)
  - ModelManager: 위 4가지 책임을 담당하는 핵심 클래스

작성자: 이현수 / 작성일: 2026-07-05
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any

import httpx

logger = logging.getLogger("nexus.model.manager")


class ActiveModel(str, Enum):
    """
    현재 GPU 서버에 떠 있는(활성) 모델을 나타내는 식별자.

    str를 함께 상속하므로 이 Enum 멤버는 그 자체로 문자열처럼 쓸 수 있다.
    즉 ActiveModel.PRIMARY == "primary"가 성립해서, JSON 직렬화나 서버가
    돌려준 문자열("primary")과의 비교가 편리하다. 값이 곧 API가 쓰는
    문자열 프로토콜이므로 함부로 바꾸면 GPU 서버와 계약이 깨진다.
    """

    PRIMARY = "primary"  # Qwen 3.5 27B — 기본 추론 모델(주력)
    AUXILIARY = "auxiliary"  # ExaOne 7.8B — 한국어 보조 모델(경량)


class ModelManager:
    """
    Machine A 측 모델 매니저 — GPU 서버(Machine B)의 모델 상태를 원격 관리.

    이 클래스는 내부에 httpx.AsyncClient를 하나 들고 있으면서, GPU 서버가
    노출한 REST 엔드포인트(/health, /v1/models/swap, /v1/lora/load 등)를
    호출한다. 또한 "지금 무슨 모델이 떠 있는지(_active_model)", "서버가
    살아 있는지(_server_healthy)", "어떤 LoRA가 붙어 있는지(_active_loras)"
    같은 상태를 로컬에 캐시해 두어, 상위 코드가 매번 서버에 묻지 않아도
    되게 한다. (이 캐시는 health_check()가 호출될 때 서버 값으로 갱신된다.)

    사용 예:
        manager = ModelManager(gpu_server_url="http://192.168.21.112:8000")
        await manager.initialize()
        health = await manager.health_check()
        await manager.swap_model(ActiveModel.AUXILIARY)
    """

    def __init__(
        self,
        gpu_server_url: str = "http://localhost:8000",
        api_key: str = "local-key",
    ):
        """
        모델 매니저를 생성한다. (아직 서버에 연결하지는 않는다)

        매개변수:
          gpu_server_url: Machine B(GPU 서버)의 베이스 URL. 에어갭 원칙상
            반드시 LAN 주소(192.168.x.x 등)나 localhost여야 한다.
          api_key: GPU 서버 호출 시 Authorization: Bearer 헤더에 실을 키.

        참고: 실제 값은 하드코딩이 아니라 config/*.yaml에서 읽어 넘겨주는 것이
        원칙이며, 여기 기본값(localhost/local-key)은 테스트·로컬 편의용이다.
        """
        # URL 끝의 슬래시를 제거해 둔다. 이후 f"{url}/health"처럼 경로를 이어
        # 붙일 때 슬래시가 겹쳐 "//health"가 되는 실수를 막기 위함이다.
        self.gpu_server_url = gpu_server_url.rstrip("/")
        self.api_key = api_key
        # 로컬 캐시 상태들. health_check() 성공 시 서버 값으로 동기화된다.
        self._active_model = ActiveModel.PRIMARY  # 기본은 primary라고 가정
        self._server_healthy = False  # 아직 연결 확인 전이므로 안전하게 False
        self._active_loras: list[str] = []  # 현재 부착된 LoRA 이름 목록

        # 재사용 가능한 비동기 HTTP 클라이언트. 세부 타임아웃을 단계별로 지정:
        #   connect(연결 10s)/read(응답 대기 30s)/write(요청 전송 10s)/pool(10s).
        # 개별 호출에서 timeout=... 을 넘기면 그 호출만 이 값을 덮어쓴다.
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=10.0, read=30.0, write=10.0, pool=10.0),
        )

    @property
    def active_model(self) -> ActiveModel:
        """
        현재 활성 모델을 반환한다. (로컬 캐시 값 — 서버에 재조회하지 않음)

        최신값이 필요하면 먼저 health_check()를 호출해 캐시를 갱신한 뒤 읽는다.
        """
        return self._active_model

    @property
    def is_healthy(self) -> bool:
        """
        GPU 서버가 마지막 확인 시점에 정상이었는지 반환한다. (캐시 값)

        이 값 역시 health_check() 호출 결과로만 갱신되는 캐시이므로,
        실시간 판단이 필요하면 health_check()를 직접 await 하는 편이 정확하다.
        """
        return self._server_healthy

    # ─── 초기화 ───

    async def initialize(self) -> dict[str, Any]:
        """
        GPU 서버 상태를 확인하고 초기 정보를 로드하는 진입점.

        부트스트랩 Phase 2에서 한 번 호출된다. 내부적으로 health_check()를
        불러 서버 연결 여부를 확인하고, 결과를 로그로 남긴 뒤 요약 dict를 돌려준다.

        중요: 서버 연결에 실패해도 예외를 던지지 않고 healthy=False로 반환한다.
        이는 GPU 서버가 아직 안 떠 있어도 Machine A 부트스트랩은 계속 진행하고,
        나중에 서버가 준비되면 재시도할 수 있게 하기 위한 fail-soft 설계다.

        반환: {"healthy": bool, "active_model": str} 형태의 초기화 요약.
        """
        health = await self.health_check()
        if health:
            logger.info(
                f"모델 매니저 초기화 완료: "
                f"active_model={self._active_model.value}, "
                f"server={self.gpu_server_url}"
            )
        else:
            logger.warning(
                f"GPU 서버 연결 실패: {self.gpu_server_url}. "
                f"서버 시작 후 재시도가 필요합니다."
            )
        return {"healthy": health, "active_model": self._active_model.value}

    # ─── 헬스 체크 ───

    async def health_check(self) -> bool:
        """
        GPU 서버의 헬스 상태를 확인하고 로컬 캐시를 서버 값으로 동기화한다.

        흐름:
          1. GET /health 를 5초 타임아웃으로 호출한다.
          2. 200 OK면 응답 JSON에서 active_model / active_loras를 읽어
             로컬 캐시(_active_model, _active_loras)를 서버 기준으로 갱신한다.
          3. 200이 아니거나 예외(네트워크 오류·타임아웃 등)면 서버를 비정상으로
             간주하고 _server_healthy=False로 두고 False를 반환한다.

        반환: 서버가 정상(200)이면 True, 아니면 False.

        설계 메모: 예외를 밖으로 던지지 않고 조용히 False로 흡수한다. 헬스 체크는
        "서버가 살아 있나?"를 묻는 용도라, 죽어 있는 것도 정상적인 답(False)으로
        다루는 편이 상위 로직을 단순하게 만든다. (fail-closed: 확신 없으면 False)
        """
        try:
            resp = await self._client.get(
                f"{self.gpu_server_url}/health",
                timeout=5.0,  # 헬스 체크는 빨리 끝나야 하므로 짧게 잡는다
            )
            if resp.status_code == 200:
                data = resp.json()
                self._server_healthy = True

                # 서버가 보고한 활성 모델로 로컬 캐시를 맞춘다. 없으면 "primary"로
                # 가정한다. 알 수 없는 값이 오면 무시하고 기존 값을 유지한다
                # (아래 in 검사로 이상값 방어 — Enum 변환 시 예외를 막는다).
                active = data.get("active_model", "primary")
                if active in ("primary", "auxiliary"):
                    self._active_model = ActiveModel(active)

                # 서버가 알려준 부착 LoRA 목록으로 캐시를 통째로 교체한다.
                self._active_loras = data.get("active_loras", [])
                return True

            # 200이 아니면(예: 500/503) 비정상으로 처리한다.
            self._server_healthy = False
            return False
        except Exception as e:
            # 연결 실패·타임아웃·JSON 파싱 오류 등 모든 예외를 여기서 흡수한다.
            # 헬스 체크는 자주 호출될 수 있어 로그는 debug 레벨로만 남긴다.
            logger.debug(f"헬스 체크 실패: {e}")
            self._server_healthy = False
            return False

    # ─── 모델 전환 (hot-swap) ───

    async def swap_model(self, target: ActiveModel) -> dict[str, Any]:
        """
        Primary ↔ Auxiliary 모델을 hot-swap(무중단 교체)한다.

        GPU 서버의 POST /v1/models/swap 엔드포인트를 호출해, 원하는 모델
        (target)을 올려달라고 요청한다. 성공하면 로컬 캐시(_active_model)도
        target으로 갱신한다.

        매개변수:
          target: 전환하고 싶은 대상 모델(ActiveModel.PRIMARY/AUXILIARY).
        반환:
          - 이미 target이 떠 있으면 {"status": "already_active", ...} (조기 반환).
          - 성공 시 서버가 돌려준 JSON 그대로.
          - 실패 시 {"status": "error", "message": ...}. (예외를 던지지 않음)

        왜 hot-swap이 필요한가: RTX 5090(32GB)에서는 두 모델을 동시에 VRAM에
        올릴 수 없다. 그래서 하나를 내리고 다른 하나를 올리는 방식으로 전환한다.
        VRAM이 넉넉한 H200에서는 동시 로딩이 가능해 이 과정이 필요 없어진다.
        """
        # 이미 원하는 모델이 떠 있으면 서버를 호출할 필요 없이 즉시 반환한다.
        # (불필요한 무거운 스왑 작업을 피하는 최적화)
        if target == self._active_model:
            return {"status": "already_active", "model": target.value}

        try:
            resp = await self._client.post(
                f"{self.gpu_server_url}/v1/models/swap",
                json={"target": target.value},
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=120.0,  # 모델 로드/언로드는 오래 걸릴 수 있어 넉넉히 준다
            )
            # 4xx/5xx면 여기서 예외를 일으켜 아래 except로 넘긴다.
            resp.raise_for_status()
            result = resp.json()
            # 서버 전환이 확인된 뒤에만 로컬 캐시를 갱신한다(순서 중요).
            self._active_model = target
            logger.info(f"모델 전환 완료: {target.value}")
            return result
        except Exception as e:
            # 전환 실패 시 캐시는 건드리지 않고(기존 모델 유지) 에러 dict를 반환.
            logger.error(f"모델 전환 실패: {e}")
            return {"status": "error", "message": str(e)}

    # ─── LoRA 체크포인트 관리 ───

    async def load_lora(self, name: str, path: str) -> dict[str, Any]:
        """
        LoRA 체크포인트를 GPU 서버에 부착(로드)하도록 요청한다.

        LoRA는 베이스 모델 위에 얹는 경량 어댑터로, 테넌트별·작업별 미세조정
        결과를 베이스 모델 전체를 갈아끼우지 않고도 붙였다 뗄 수 있게 해준다.

        매개변수:
          name: LoRA 어댑터의 논리적 이름(이후 언로드·라우팅 시 식별자).
          path: GPU 서버 파일시스템 기준의 체크포인트 경로.
        반환: 성공 시 서버 JSON, 실패 시 {"status": "error", ...}.

        성공하면 로컬 캐시 _active_loras에 name을 추가한다(중복은 방지).
        """
        try:
            resp = await self._client.post(
                f"{self.gpu_server_url}/v1/lora/load",
                json={"name": name, "path": path},
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=60.0,  # 체크포인트 로드에 다소 시간이 걸릴 수 있다
            )
            resp.raise_for_status()
            result = resp.json()
            # 같은 이름이 두 번 쌓이지 않도록 없을 때만 목록에 추가한다.
            if name not in self._active_loras:
                self._active_loras.append(name)
            logger.info(f"LoRA 체크포인트 로드: {name}")
            return result
        except Exception as e:
            logger.error(f"LoRA 로드 실패: {e}")
            return {"status": "error", "message": str(e)}

    async def unload_lora(self, name: str) -> dict[str, Any]:
        """
        LoRA 체크포인트를 GPU 서버에서 분리(언로드)하도록 요청한다.

        load_lora의 반대 동작. VRAM을 회수하거나 더 이상 쓰지 않는 어댑터를
        정리할 때 호출한다. 성공하면 로컬 캐시 _active_loras에서 name을 제거한다.

        매개변수:
          name: 분리할 LoRA 어댑터의 이름(load_lora에서 쓴 것과 동일).
        반환: 성공 시 서버 JSON, 실패 시 {"status": "error", ...}.
        """
        try:
            resp = await self._client.post(
                f"{self.gpu_server_url}/v1/lora/unload",
                json={"name": name},
                headers={"Authorization": f"Bearer {self.api_key}"},
            )
            resp.raise_for_status()
            result = resp.json()
            # 캐시에 있을 때만 제거한다(없는 항목 remove 시 ValueError 방지).
            if name in self._active_loras:
                self._active_loras.remove(name)
            logger.info(f"LoRA 체크포인트 언로드: {name}")
            return result
        except Exception as e:
            logger.error(f"LoRA 언로드 실패: {e}")
            return {"status": "error", "message": str(e)}

    # ─── 상태 정보 ───

    def get_status(self) -> dict[str, Any]:
        """
        현재 모델 매니저의 로컬 캐시 상태를 dict로 스냅샷해 반환한다.

        네트워크 호출 없이 즉시 반환하는 동기 메서드다. 상태 표시(UI/로그)나
        디버깅에 쓰기 좋다. 최신 서버 상태가 필요하면 먼저 health_check()로
        캐시를 갱신한 뒤 이 메서드를 호출한다.

        반환 키: gpu_server_url, active_model, server_healthy, active_loras.
        """
        return {
            "gpu_server_url": self.gpu_server_url,
            "active_model": self._active_model.value,
            "server_healthy": self._server_healthy,
            "active_loras": self._active_loras,
        }

    # ─── 정리 ───

    async def close(self) -> None:
        """
        내부 httpx.AsyncClient를 닫아 열린 연결·리소스를 정리한다.

        매니저 사용을 마칠 때(앱 종료·셧다운 훅 등) 반드시 한 번 호출해야
        커넥션 누수를 막을 수 있다. 이 메서드 이후에는 이 매니저로 더 이상
        서버를 호출하면 안 된다(클라이언트가 닫혀 있어 실패한다).
        """
        await self._client.aclose()
