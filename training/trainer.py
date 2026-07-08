"""
LoRA/QLoRA 트레이너 — GPU 서버에 학습 요청을 보내고 상태를 추적한다.

[이 파일이 하는 일 — 한눈에 보기]
Nexus는 두 대의 머신으로 구성된다. Machine A(오케스트레이터, 지금 이 코드가
도는 곳)는 실제 GPU 연산을 절대 하지 않고, Machine B(GPU 서버)에게 "이 데이터로
이렇게 학습해줘" 하고 HTTP로 부탁만 한다. 이 파일은 그 "부탁"과 "진행 상황
확인", "취소"를 담당하는 클라이언트다. torch/CUDA를 직접 import 하지 않는 이유가
바로 이것이며, 이는 Nexus 아키텍처 원칙 P4(2-Machine 토폴로지)를 지키기 위함이다.

[주요 구성 요소]
  - TrainingJobStatus : 학습 작업의 5가지 상태를 정의한 문자열 Enum.
  - TrainingConfig    : LoRA/QLoRA 하이퍼파라미터 묶음(dataclass). GPU 서버로
                        보내기 좋게 to_dict()로 직렬화한다.
  - _validate_lan_url : 에어갭 위반(외부 URL) 차단용 내부 검증 함수.
  - LoRATrainer       : 실제 HTTP 호출 3종(start/status/cancel)을 감싼 클래스.

[지원하는 학습 방법(method)]
  - LoRA  : Low-Rank Adaptation (Phase 1 부트스트랩)
  - QLoRA : Quantized LoRA (Phase 2~4 본격 학습, 기본값)
  - full  : Full fine-tuning (실험용, 메모리 집약적)

[의존/연동]
  - httpx(비동기 HTTP)로 GPU 서버의 /v1/training/* 엔드포인트를 호출한다.
  - M7(멀티테넌시) 경로/이름 해석은 core.adapter_naming 모듈에 위임한다.

[에어갭 준수]
  GPU 서버 URL은 반드시 LAN 주소(localhost/127.0.0.1/10.x/172.x/192.168.x)여야
  하며, 그 외 주소는 생성 시점에 ValueError로 즉시 거부한다.

작성자: 이현수 / 작성일: 2026-07-05
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx

logger = logging.getLogger("nexus.training.trainer")


# ─────────────────────────────────────────────
# 학습 작업 상태
# ─────────────────────────────────────────────
class TrainingJobStatus(str, Enum):
    """
    학습 작업(job)의 생명주기 상태를 나타내는 Enum.

    str을 함께 상속하므로 값 자체가 문자열이다. 즉 JSON 직렬화나 딕셔너리 비교 시
    TrainingJobStatus.RUNNING == "running" 이 성립해 GPU 서버 API 응답(문자열)과
    바로 비교/저장하기 편하다. 상태 흐름은 보통
    PENDING → RUNNING → (COMPLETED | FAILED | CANCELLED) 순서로 흘러간다.
    """

    PENDING = "pending"  # 대기 중 — 아직 GPU 서버가 학습을 시작하지 않은 상태
    RUNNING = "running"  # 학습 진행 중 — 실제로 GPU가 파라미터를 업데이트하는 중
    COMPLETED = "completed"  # 완료 — 체크포인트 저장까지 정상 종료
    FAILED = "failed"  # 실패 — OOM/데이터 오류/서버 장애 등으로 중단
    CANCELLED = "cancelled"  # 취소됨 — 사용자가 cancel()로 명시적으로 중단


# ─────────────────────────────────────────────
# 학습 설정
# ─────────────────────────────────────────────
@dataclass
class TrainingConfig:
    """
    LoRA/QLoRA 학습에 필요한 하이퍼파라미터를 한 곳에 모아둔 설정 dataclass.

    이 객체 하나만 채워서 LoRATrainer에 넘기면, 나머지 값들이 to_dict()로
    직렬화되어 GPU 서버로 그대로 전달된다. 즉 "학습을 어떻게 돌릴지"에 대한
    단일 진실 공급원(single source of truth) 역할을 한다.

    아래 기본값은 RTX 5090(32GB VRAM) 한 장에서 Qwen 3.5 27B의 INT4 양자화
    모델을 QLoRA로 무리 없이 학습하도록 맞춘 보수적 값이다. VRAM이 더 큰
    H100/H200 등에서는 batch_size나 max_seq_length를 키울 수 있다.

    각 필드 옆 주석은 "이 값을 올리면/내리면 어떤 트레이드오프가 있는지"를
    빠르게 파악하도록 달아 두었다.
    """

    method: str = "qlora"  # 학습 방식: "qlora"(양자화 LoRA) / "lora" / "full"
    model_path: str = "./models/qwen3.5-27b"  # 학습의 출발점이 되는 베이스 모델 경로
    output_dir: str = "./checkpoints/"  # 학습 산출물(체크포인트/어댑터) 저장 경로
    lora_rank: int = 16  # LoRA rank — 클수록 표현력↑ 이지만 메모리·과적합 위험↑
    lora_alpha: int = 32  # LoRA alpha — 스케일 계수. 관례상 rank의 2배로 둔다
    lora_dropout: float = 0.05  # 학습 중 일부 연결을 끊어 과적합을 억제하는 비율
    # LoRA를 끼워 넣을 어텐션 서브모듈 목록. q/k/v/o 프로젝션에만 붙여
    # 파라미터 수를 최소화하면서 성능을 확보하는 표준 조합이다.
    target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"]
    )
    batch_size: int = 1  # 1스텝에 올리는 샘플 수. VRAM이 빠듯해 1을 권장
    # 그래디언트 누적: 여러 스텝의 기울기를 모아 한 번에 업데이트한다.
    # 실질(유효) 배치 크기 = batch_size * gradient_accumulation_steps = 1 * 8 = 8.
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4  # 학습률 — LoRA에서 흔히 쓰는 값. 크면 발산 위험
    num_epochs: int = 3  # 전체 데이터를 몇 번 반복 학습할지
    max_seq_length: int = 2048  # 한 샘플의 최대 토큰 길이. 길수록 VRAM 사용↑
    warmup_ratio: float = 0.03  # 초반에 학습률을 서서히 올리는 구간 비율(전체의 3%)
    weight_decay: float = 0.01  # 가중치 감쇠(L2 정규화) — 과적합 완화
    save_steps: int = 100  # 몇 스텝마다 체크포인트를 디스크에 저장할지
    logging_steps: int = 10  # 몇 스텝마다 학습 로그(loss 등)를 남길지

    # ── M7(멀티테넌시) 관련 필드 ──────────────────────────
    # 여러 고객(테넌트)이 같은 GPU 서버를 공유할 때, 누구를 위한 학습인지
    # 구분하기 위한 식별자. None이면 default 테넌트로 취급(기존 코드 호환).
    tenant_id: str | None = None
    # 학습 Phase(0~4). tenant_id와 함께 output_dir/어댑터 이름을 자동으로
    # 해석하는 데 쓰인다. None이면 자동 해석을 하지 않는다.
    phase: int | None = None

    def resolved_output_dir(self, base_dir: str = "/opt/nexus-gpu/checkpoints") -> str:
        """
        M7: tenant_id + phase 조합으로 실제 출력 디렉토리 경로를 계산해 돌려준다.

        멀티테넌시 환경에서는 테넌트/단계별로 체크포인트를 격리 저장해야 하므로,
        경로 규칙을 이 클래스에 하드코딩하지 않고 core.adapter_naming 모듈의
        compose_output_dir()에 위임한다(경로 규칙 변경 시 한 곳만 고치면 됨).

        단, tenant_id나 phase 중 하나라도 None이면 멀티테넌시를 쓰지 않는
        것으로 보고, 사용자가 직접 지정한 self.output_dir을 그대로 반환한다
        (하위 호환).

        Args:
            base_dir: 테넌트/단계 경로가 붙는 기준 루트 디렉토리.

        Returns:
            해석된 출력 디렉토리 경로 문자열.
        """
        if self.tenant_id is None or self.phase is None:
            return self.output_dir
        # 순환 import를 피하기 위해 함수 내부에서 지연 import 한다.
        from core.adapter_naming import compose_output_dir
        return compose_output_dir(self.tenant_id, self.phase, base_dir=base_dir)

    def resolved_adapter_name(self) -> str | None:
        """
        M7: tenant_id + phase 조합으로 LoRA 어댑터 이름을 계산해 돌려준다.

        GPU 서버가 학습된 어댑터를 이 이름으로 등록/핫로딩할 수 있게 한다.
        둘 중 하나라도 None이면(=멀티테넌시 비활성) 이름을 만들지 않고 None을
        반환한다. 실제 이름 규칙은 adapter_naming.compose_adapter_name()이 정한다.

        Returns:
            어댑터 이름 문자열, 또는 멀티테넌시 미사용 시 None.
        """
        if self.tenant_id is None or self.phase is None:
            return None
        # resolved_output_dir()과 마찬가지로 순환 import 회피용 지연 import.
        from core.adapter_naming import compose_adapter_name
        return compose_adapter_name(self.tenant_id, self.phase)

    def to_dict(self) -> dict[str, Any]:
        """
        설정 전체를 순수 딕셔너리로 직렬화한다(GPU 서버 API 전송용).

        httpx로 JSON 본문에 실어 보내기 위해 dataclass를 평범한 dict로 편다.
        멀티테넌시 필드(tenant_id/phase/adapter_name)도 함께 담아, GPU 서버가
        요청을 올바른 테넌트/어댑터로 라우팅할 수 있게 한다.
        """
        return {
            "method": self.method,
            "model_path": self.model_path,
            "output_dir": self.output_dir,
            "lora_rank": self.lora_rank,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "target_modules": self.target_modules,
            "batch_size": self.batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "learning_rate": self.learning_rate,
            "num_epochs": self.num_epochs,
            "max_seq_length": self.max_seq_length,
            "warmup_ratio": self.warmup_ratio,
            "weight_decay": self.weight_decay,
            "save_steps": self.save_steps,
            "logging_steps": self.logging_steps,
            # M7 멀티테넌시 필드는 GPU 서버 API가 라우팅에 쓸 수 있도록 함께 전송
            "tenant_id": self.tenant_id,
            "phase": self.phase,
            "adapter_name": self.resolved_adapter_name(),
        }


def _validate_lan_url(url: str) -> None:
    """
    주어진 URL의 호스트가 LAN(사설망) 범위인지 검증하고, 아니면 예외를 던진다.

    Nexus는 에어갭(폐쇄망) 환경을 전제로 하므로 외부 인터넷으로 나가는 통신이
    있어서는 안 된다. LoRATrainer 생성 시 이 함수를 통과해야만 GPU 서버 URL을
    받아들여, 실수로 외부 주소를 넣는 것을 시작 시점에 원천 차단한다(fail-closed).

    검증 방식은 화이트리스트다 — 허용된 접두사로 시작하는 호스트명만 통과시킨다.
    (참고: 접두사 매칭이라 예컨대 "172."는 실제 사설망 대역 172.16~31 외의
    172.x도 통과시킬 수 있으나, 폐쇄망 운영을 전제로 한 의도적 단순화다.)

    Args:
        url: 검증할 GPU 서버 URL.

    Raises:
        ValueError: 호스트가 허용된 LAN 접두사 중 어느 것으로도 시작하지 않을 때.
    """
    parsed = urlparse(url)
    # 스킴/포트를 제외한 순수 호스트명만 추출. 파싱 실패 시 빈 문자열로 처리.
    hostname = parsed.hostname or ""
    # 허용하는 로컬/사설망 접두사 목록.
    allowed_prefixes = ("localhost", "127.0.0.1", "10.", "172.", "192.168.")
    # 하나라도 매칭되면 통과. 전부 불일치면 에어갭 위반으로 간주해 예외.
    if not any(hostname.startswith(p) for p in allowed_prefixes):
        raise ValueError(
            f"에어갭 위반: GPU 서버 URL '{url}'이(가) LAN 주소가 아닙니다. "
            f"허용 범위: localhost, 127.0.0.1, 10.x, 172.x, 192.168.x"
        )


# ─────────────────────────────────────────────
# LoRA/QLoRA 트레이너
# ─────────────────────────────────────────────
class LoRATrainer:
    """
    LoRA/QLoRA 학습의 전 과정을 GPU 서버 API 호출로 감싸는 클라이언트 클래스.

    이 클래스는 "학습 오케스트레이터"다. Machine A에서 Machine B(GPU 서버)로
    HTTP 요청을 보내 학습을 시작(start_training)하고, 상태를 폴링(get_status)하며,
    필요하면 취소(cancel)한다. GPU/CUDA를 직접 만지는 코드는 전혀 없으며, 이는
    Nexus 아키텍처 원칙 P4(2-Machine 토폴로지)를 지키기 위한 의도적 설계다.

    또한 GPU 서버와 통신이 끊기더라도 클라이언트가 완전히 죽지 않도록,
    진행 중 작업 정보를 self._active_jobs에 캐시해 두고 연결 실패 시 이를
    폴백(fallback)으로 활용한다.
    """

    def __init__(self, gpu_server_url: str, config: TrainingConfig) -> None:
        """
        트레이너를 초기화한다. 이때 GPU 서버 URL의 LAN 여부를 즉시 검증한다.

        Args:
            gpu_server_url: GPU 서버(Machine B)의 URL. LAN 주소만 허용되며,
                외부 주소면 _validate_lan_url가 여기서 ValueError를 던진다.
            config: 학습에 사용할 하이퍼파라미터 설정(TrainingConfig).

        Raises:
            ValueError: gpu_server_url이 LAN 범위가 아닐 때(에어갭 위반).
        """
        # 생성 시점에 에어갭 규칙을 강제 — 외부 URL이면 여기서 바로 실패한다.
        _validate_lan_url(gpu_server_url)
        # 뒤에서 경로를 f-string으로 이어붙이므로, 끝의 슬래시를 제거해
        # "http://host//v1/..." 같은 이중 슬래시를 방지한다.
        self._gpu_server_url = gpu_server_url.rstrip("/")
        self._config = config
        # 진행 중인 학습 작업을 추적하는 캐시 (job_id → 상태 정보 딕셔너리).
        # 서버 연결이 끊겼을 때 마지막으로 알던 상태를 돌려주는 용도로도 쓰인다.
        self._active_jobs: dict[str, dict[str, Any]] = {}

    @property
    def config(self) -> TrainingConfig:
        """현재 이 트레이너가 사용 중인 학습 설정 객체를 반환한다."""
        return self._config

    @property
    def gpu_server_url(self) -> str:
        """정규화된(끝 슬래시 제거된) GPU 서버 URL을 반환한다."""
        return self._gpu_server_url

    async def start_training(self, data_path: str) -> str:
        """
        GPU 서버에 "학습을 시작하라"고 요청하고, 부여받은 job_id를 돌려준다.

        전체 흐름:
          1) 로컬에서 학습 데이터 파일이 실제로 존재하는지 먼저 확인한다.
          2) 데이터 경로 + 직렬화한 설정을 페이로드로 만들어 POST 한다.
          3) 성공하면 서버가 준 job_id를 받아 RUNNING 상태로 캐시에 등록한다.
          4) 만약 서버에 연결조차 안 되면(ConnectError), 예외로 죽지 않고
             로컬 job_id("local_...")를 발급해 PENDING으로 기록한 뒤 반환한다.
             → 나중에 서버가 살아나면 이 작업을 재시도/추적할 수 있게 하기 위함.

        Args:
            data_path: 학습 데이터(JSONL) 파일 경로.

        Returns:
            job_id — 이후 get_status/cancel에서 이 작업을 가리키는 식별자.

        Raises:
            FileNotFoundError: 학습 데이터 파일이 존재하지 않을 때.
            httpx.HTTPStatusError: 서버에 연결은 됐지만 4xx/5xx 응답이 온 경우
                (raise_for_status가 던짐 — ConnectError와 달리 여기서 잡지 않음).
        """
        # (1) 데이터가 없으면 굳이 서버를 호출할 필요 없이 즉시 실패시킨다.
        if not Path(data_path).exists():
            raise FileNotFoundError(f"학습 데이터 파일을 찾을 수 없습니다: {data_path}")

        # (2) 서버로 보낼 요청 본문 구성 — 데이터 경로와 전체 하이퍼파라미터.
        payload = {
            "data_path": data_path,
            "config": self._config.to_dict(),
        }

        try:
            # 학습 시작 요청은 서버측 준비가 있을 수 있어 타임아웃을 30초로 넉넉히.
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self._gpu_server_url}/v1/training/start",
                    json=payload,
                )
                # 4xx/5xx면 예외를 던져 호출자에게 실패를 알린다.
                response.raise_for_status()
                result = response.json()
                # 서버가 job_id를 안 줬다면 방어적으로 새 UUID를 생성해 쓴다.
                job_id = result.get("job_id", str(uuid.uuid4()))
        except httpx.ConnectError as e:
            # (4) 서버 자체에 닿지 못한 경우 — 로컬 job_id를 발급해 작업을
            #     유실하지 않고 대기(PENDING) 상태로 남겨 추후 재시도를 가능케 한다.
            job_id = f"local_{uuid.uuid4().hex[:12]}"
            logger.warning("GPU 서버 연결 실패, 로컬 job_id 발급: %s (에러: %s)", job_id, e)
            self._active_jobs[job_id] = {
                "job_id": job_id,
                "status": TrainingJobStatus.PENDING.value,
                "data_path": data_path,
                "config": self._config.to_dict(),
                "error": str(e),
            }
            return job_id

        # (3) 정상 시작 — 활성 작업 캐시에 RUNNING 상태로 등록한다.
        self._active_jobs[job_id] = {
            "job_id": job_id,
            "status": TrainingJobStatus.RUNNING.value,
            "data_path": data_path,
            "config": self._config.to_dict(),
        }

        logger.info(
            "학습 시작: job_id=%s, method=%s, data=%s",
            job_id,
            self._config.method,
            data_path,
        )
        return job_id

    async def get_status(self, job_id: str) -> dict[str, Any]:
        """
        특정 학습 작업의 현재 진행 상태를 GPU 서버에 폴링해 조회한다.

        정상 경로에서는 서버 응답(진행률/메트릭 등)을 받아 로컬 캐시를 갱신한 뒤
        그대로 반환한다. 서버 연결 실패나 에러 응답이 오면 예외를 밖으로 던지지
        않고, 마지막으로 알던 캐시 상태를 대신 돌려준다(호출자가 폴링 루프를
        계속 돌릴 수 있게 하기 위함). 캐시에도 없는 미지의 job_id라면 FAILED
        상태를 만들어 반환한다.

        Args:
            job_id: 상태를 조회할 학습 작업 식별자.

        Returns:
            상태 딕셔너리. 보통 job_id, status, progress, metrics 등을 포함한다.
        """
        try:
            # 상태 조회는 가벼운 GET이므로 타임아웃을 10초로 짧게 잡는다.
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    f"{self._gpu_server_url}/v1/training/status/{job_id}",
                )
                response.raise_for_status()
                status_data = response.json()
        except (httpx.ConnectError, httpx.HTTPStatusError) as e:
            # 연결 실패/에러 응답 — 폴링이 멈추지 않도록 캐시된 상태로 대체한다.
            logger.warning("GPU 서버 상태 조회 실패: %s", e)
            cached = self._active_jobs.get(job_id)
            if cached:
                # 캐시된 상태에 이번 에러 내용을 덧붙여 돌려준다.
                cached["error"] = str(e)
                return cached
            # 캐시에도 흔적이 없는 job_id면 실패로 간주해 알린다.
            return {
                "job_id": job_id,
                "status": TrainingJobStatus.FAILED.value,
                "error": f"상태 조회 실패: {e}",
            }

        # 정상 응답 — 우리가 알고 있는 작업이면 최신 서버 상태로 캐시를 갱신한다.
        if job_id in self._active_jobs:
            self._active_jobs[job_id].update(status_data)

        return status_data

    async def cancel(self, job_id: str) -> bool:
        """
        진행 중인 학습 작업을 취소하도록 GPU 서버에 요청한다.

        서버가 취소를 확인(cancelled=true)하면 로컬 캐시의 상태도 CANCELLED로
        바꿔 둔다. 연결 실패나 에러 응답이 오면 예외를 던지지 않고 취소 실패
        (False)로 처리한다 — 취소는 "되면 좋고 안 되면 마는" 성격이라 호출자
        입장에서 예외보다 불리언 결과가 다루기 쉽기 때문이다.

        Args:
            job_id: 취소할 학습 작업 식별자.

        Returns:
            취소 성공 여부(bool). 서버가 취소를 확인했을 때만 True.
        """
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    f"{self._gpu_server_url}/v1/training/cancel/{job_id}",
                )
                response.raise_for_status()
                # 서버 응답의 "cancelled" 플래그가 실제 취소 성공 여부다.
                success = response.json().get("cancelled", False)
        except (httpx.ConnectError, httpx.HTTPStatusError) as e:
            # 통신 문제는 로그만 남기고 "취소 실패"로 간주한다.
            logger.warning("학습 취소 요청 실패: %s", e)
            success = False

        # 로컬 캐시 반영 — 우리가 추적 중인 작업일 때만.
        if job_id in self._active_jobs:
            # 성공했을 때만 상태를 CANCELLED로 덮어쓴다(실패면 기존 상태 유지).
            if success:
                self._active_jobs[job_id]["status"] = TrainingJobStatus.CANCELLED.value
            logger.info("학습 취소 %s: job_id=%s", "성공" if success else "실패", job_id)

        return success

    @property
    def active_jobs(self) -> dict[str, dict[str, Any]]:
        """
        현재 추적 중인 활성 학습 작업들의 얕은 복사본을 반환한다.

        내부 캐시(self._active_jobs)를 그대로 노출하면 외부에서 실수로 수정할 수
        있으므로, dict()로 감싼 복사본을 돌려줘 내부 상태를 보호한다.
        """
        return dict(self._active_jobs)
