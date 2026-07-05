"""
하드웨어 티어 감지 모듈 — GPU VRAM 용량을 근거로 Nexus의 오케스트레이션 모드를 자동 결정한다.

이 파일이 하는 일(개요):
  - 현재 머신에 장착된 GPU의 VRAM(비디오 메모리) 크기를 여러 방법으로 조사한다.
  - 그 용량에 따라 3단계 하드웨어 티어(TIER_S / TIER_M / TIER_L) 중 하나로 분류한다.
  - 각 티어에 맞는 오케스트레이션 설정(Scout 사용 여부, 컨텍스트 창 크기 전략 등)을 제공한다.
  - 즉, "이 하드웨어에서 Nexus를 어떻게 굴릴지"를 코드가 스스로 판단하게 해 주는 진입점이다.

왜 티어를 나누는가:
  - GPU마다 쓸 수 있는 메모리가 다르고, 메모리가 곧 컨텍스트 창 크기와 모델 배치 전략을 좌우한다.
  - TIER_S (32GB): 컨텍스트 8K로 좁다 → 무거운 Worker(GPU)만으로는 벅차서
    가벼운 Scout(CPU)가 앞단에서 도구/계획을 정리하는 다중 모델 구조가 필요하다.
  - TIER_M (80GB): 컨텍스트 32K → Worker 단독(single_model)으로 충분히 감당된다.
  - TIER_L (128GB+): 컨텍스트 128K → v6.1 원래 설계 그대로 여유롭게 동작한다.

핵심 원칙 — 상위 티어는 하위 티어의 상위집합(superset)이다:
  TIER_L ⊃ TIER_M ⊃ TIER_S
  (상위 티어는 하위 티어가 하는 일을 모두 할 수 있고, 거기에 더 큰 여유가 있다는 뜻.)

주요 구성 요소:
  - HardwareTier      : 세 가지 티어를 나타내는 문자열 Enum.
  - TIER_CONFIG       : 티어별 오케스트레이션 설정을 담은 상수 딕셔너리.
  - detect_hardware_tier() : 설정 우선 → GPU 자동 감지 순으로 티어를 결정하는 공개 함수.
  - _detect_gpu_vram_gb()  : torch / pynvml / nvidia-smi 순으로 VRAM을 조사하는 내부 헬퍼.
  - get_tier_config()      : 특정 티어의 설정 사본을 돌려주는 조회 함수.

작성자: 이현수 / 작성일: 2026-07-05
"""

# 아래 import는 미래 문법(PEP 563) 활성화용 — 파일 최상단에 있어야 하므로 절대 위치 이동 금지.
from __future__ import annotations

import logging
from enum import Enum
from typing import Any

# 이 모듈 전용 로거. "nexus.{module}" 네이밍 규칙을 따라 로그 출처를 명확히 한다.
logger = logging.getLogger("nexus.model.hardware_tier")


class HardwareTier(str, Enum):
    """
    하드웨어 티어를 나타내는 문자열 기반 Enum.

    str을 함께 상속하므로 각 멤버는 문자열처럼 다룰 수 있다(예: 설정 파일의
    "small" 값과 직접 비교/변환 가능). 이 값 하나로 오케스트레이션 모드 전체가
    갈리므로, 시스템이 "지금 어떤 급의 하드웨어 위에서 도는지"를 대표하는 핵심 키다.
    """

    TIER_S = "small"    # RTX 5090 (32GB), 8K 컨텍스트
    TIER_M = "medium"   # H100 (80GB), 32K 컨텍스트
    TIER_L = "large"    # H200 (141GB) / GB10 (128GB), 128K 컨텍스트


# 티어별 오케스트레이션 설정 테이블.
#
# 각 티어가 어떤 모드로 돌아야 하는지, Scout를 켜야 하는지, 턴 상태를 외부화해야
# 하는지 등을 한곳에 모아 둔 상수 딕셔너리다. detect_hardware_tier()로 티어를 정한
# 뒤 get_tier_config()로 여기서 해당 설정을 꺼내 쓰는 흐름이다.
#
# 참고 — max_worker_tools는 "정보용(문서용) 필드"이며 실제 강제(enforcement)는
# 하지 않는다. Worker가 실제로 보는 도구 개수는 bootstrap._create_*_tool_registry
# 계열이 결정한다. 여기의 값은 각 티어에서 CLI Worker가 받는 레지스트리의 실측
# 등록 개수로 맞춰 둔 것이다:
#   - TIER_S : _create_cli_tool_registry() = 7개
#   - TIER_M/L: _create_tool_registry()    = 23개(풀세트)
# (레지스트리 구성이 바뀌면 이 값도 함께 갱신할 것 — 어긋나도 동작에는 영향 없음.)
TIER_CONFIG = {
    HardwareTier.TIER_S: {
        # multi_model: 가벼운 Scout(CPU)가 앞단, 무거운 Worker(GPU)가 뒷단인 2단 구조.
        "orchestration_mode": "multi_model",   # Scout(CPU) + Worker(GPU)
        "scout_enabled": True,                 # Scout 단계 사용(좁은 컨텍스트를 보완)
        "max_worker_tools": 7,                 # 정보용(강제 아님) — CLI Worker 실측
        # 컨텍스트가 좁아 raw 메시지를 계속 쌓을 수 없으므로 턴 상태를 외부화한다.
        "turn_state_enabled": True,            # 상태 외부화 활성
        "description": "RTX 5090 (32GB, 8K ctx)",
    },
    HardwareTier.TIER_M: {
        # single_model: 컨텍스트 여유가 있어 Worker 하나로 전 과정을 처리한다.
        "orchestration_mode": "single_model",  # Worker 단독
        "scout_enabled": False,                # Scout 불필요
        "max_worker_tools": 23,                # 정보용(강제 아님) — 풀세트 실측
        # 컨텍스트가 넉넉해 raw 메시지를 그대로 누적해도 되므로 외부화 비활성.
        "turn_state_enabled": False,           # raw messages 누적 가능
        "description": "H100 (80GB, 32K ctx)",
    },
    HardwareTier.TIER_L: {
        # 최상위 티어 — TIER_M과 동일한 단독 모드에 컨텍스트만 훨씬 넓다.
        "orchestration_mode": "single_model",  # Worker 단독
        "scout_enabled": False,
        "max_worker_tools": 23,                # 정보용(강제 아님) — 풀세트 실측
        "turn_state_enabled": False,
        "description": "H200/GB10 (128GB+, 128K ctx)",
    },
}


def detect_hardware_tier(config: Any = None) -> HardwareTier:
    """
    GPU VRAM을 기반으로 현재 머신의 하드웨어 티어를 결정한다.

    이 함수는 "설정에 사람이 못 박아 둔 값"을 GPU 실측보다 우선한다. 운영자가
    특정 티어로 강제하고 싶을 때(예: 테스트/디버깅) 자동 감지를 덮어쓸 수 있게 하려는
    의도다. 명시값이 없거나 "auto"면 실제 VRAM을 재서 자동 분류한다.

    감지 순서(위에서부터 우선):
      1. config에 명시적 tier가 있으면(그리고 "auto"가 아니면) 그대로 사용한다.
      2. GPU VRAM을 실제로 조사해 임계값 기준으로 티어를 자동 분류한다.
      3. GPU 정보를 전혀 가져올 수 없으면 TIER_S(가장 보수적)로 안전하게 후퇴한다.
         — 큰 티어로 잘못 잡으면 메모리 초과로 터질 위험이 있으니, 모르면 작게 잡는다.

    Args:
        config: NexusConfig 객체(선택). hardware_tier 또는 hardware.tier 필드를
                읽어 명시적 티어 지정 여부를 판단한다. None이면 곧장 자동 감지로 간다.

    Returns:
        결정된 HardwareTier 값.
    """
    # 1. config에서 명시적 tier 확인
    if config is not None:
        explicit_tier = None
        # config.hardware.tier 필드가 있으면 사용
        # 두 가지 필드 형태(평면형 hardware_tier / 중첩형 hardware.tier)를 모두 지원한다.
        if hasattr(config, "hardware_tier") and config.hardware_tier:
            explicit_tier = config.hardware_tier
        elif hasattr(config, "hardware") and hasattr(config.hardware, "tier"):
            explicit_tier = config.hardware.tier

        # "auto"는 "자동 감지에 맡긴다"는 뜻이므로 명시값으로 취급하지 않는다.
        if explicit_tier and explicit_tier != "auto":
            try:
                # 문자열을 Enum으로 변환 — 알 수 없는 값이면 ValueError가 난다.
                tier = HardwareTier(explicit_tier)
                logger.info("하드웨어 티어 (설정에서 지정): %s", tier.value)
                return tier
            except ValueError:
                # 오타나 지원하지 않는 값 → 무시하고 자동 감지로 넘어간다(중단하지 않음).
                logger.warning("알 수 없는 하드웨어 티어: %s, 자동 감지로 전환", explicit_tier)

    # 2. GPU VRAM 기반 자동 감지
    vram_gb = _detect_gpu_vram_gb()

    if vram_gb is None:
        # GPU 정보 없음 → 가장 보수적인 TIER_S
        # (VRAM을 모르는데 큰 티어로 잡으면 위험하므로, 가장 안전한 최소 티어로 후퇴.)
        logger.warning("GPU VRAM 감지 실패, TIER_S(보수적)로 fallback")
        return HardwareTier.TIER_S

    # VRAM 실측값을 임계값과 비교해 티어를 나눈다.
    #   120GB 이상 → TIER_L (H200/GB10 급)
    #    64GB 이상 → TIER_M (H100 급)
    #    그 미만   → TIER_S (RTX 5090 급)
    if vram_gb >= 120:
        tier = HardwareTier.TIER_L
    elif vram_gb >= 64:
        tier = HardwareTier.TIER_M
    else:
        tier = HardwareTier.TIER_S

    logger.info(
        "하드웨어 티어 자동 감지: %s (VRAM: %.1fGB)",
        tier.value,
        vram_gb,
    )
    return tier


def _detect_gpu_vram_gb() -> float | None:
    """
    GPU VRAM 용량을 GB 단위(float)로 조사해 반환한다. 실패하면 None.

    왜 여러 방법을 시도하나:
      에어갭(폐쇄망) 환경에서는 torch나 pynvml 같은 무거운 라이브러리가 설치돼 있지
      않을 수 있다. 그래서 "정확하지만 무거운 방법"부터 "가벼운 CLI 방법"까지 순차로
      시도하고, 하나라도 성공하면 즉시 그 값을 돌려준다. 모두 실패하면 None을 반환해
      호출부가 보수적으로 처리하도록 한다.

    시도 순서:
      1. torch.cuda — 가장 정확. 설치돼 있고 CUDA를 쓸 수 있을 때만 성공.
      2. pynvml     — NVIDIA 관리 라이브러리. torch가 없을 때의 대안.
      3. nvidia-smi — 드라이버에 딸려 오는 CLI. Windows/Linux 공통으로 가장 널리 존재.

    Returns:
        조사에 성공하면 VRAM 용량(GB), 어떤 방법으로도 알아낼 수 없으면 None.
    """
    # 방법 1: torch.cuda (가장 정확)
    try:
        import torch

        # CUDA 사용 가능할 때만 0번 GPU의 총 메모리(byte)를 읽어 GB로 환산한다.
        if torch.cuda.is_available():
            vram_bytes = torch.cuda.get_device_properties(0).total_mem
            return vram_bytes / (1024**3)
    except ImportError:
        # torch 자체가 설치돼 있지 않은 흔한 경우 — 조용히 다음 방법으로 넘어간다.
        pass
    except Exception as e:
        # 그 외 예외(드라이버 문제 등)는 디버그 로그만 남기고 계속 진행한다.
        logger.debug("torch.cuda VRAM 감지 실패: %s", e)

    # 방법 2: pynvml
    try:
        import pynvml  # type: ignore[import-untyped]

        # NVML 초기화 → 0번 GPU 핸들 획득 → 메모리 정보 조회 → 종료.
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        pynvml.nvmlShutdown()
        return info.total / (1024**3)
    except ImportError:
        # pynvml 미설치 → 다음 방법으로.
        pass
    except Exception as e:
        logger.debug("pynvml VRAM 감지 실패: %s", e)

    # 방법 3: nvidia-smi CLI (Windows/Linux 공통)
    try:
        import subprocess

        # 메모리 총량만 숫자(단위 없이)로 뽑도록 쿼리 옵션을 지정해 파싱을 단순화한다.
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        # 성공(returncode 0)이면 첫 줄(=첫 번째 GPU)의 MB 값을 읽어 GB로 환산.
        if result.returncode == 0:
            vram_mb = float(result.stdout.strip().split("\n")[0])
            return vram_mb / 1024
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        # nvidia-smi 부재 / 시간초과 / 숫자 파싱 실패 — 예상 가능한 실패라 조용히 넘긴다.
        pass
    except Exception as e:
        logger.debug("nvidia-smi VRAM 감지 실패: %s", e)

    # 세 방법 모두 실패 → 알 수 없음을 뜻하는 None 반환(호출부가 보수적으로 처리).
    return None


def get_tier_config(tier: HardwareTier) -> dict[str, Any]:
    """
    주어진 티어의 오케스트레이션 설정을 새 딕셔너리(사본)로 반환한다.

    dict(...)로 감싸 사본을 만드는 이유: 호출자가 반환값을 수정하더라도 원본
    TIER_CONFIG가 오염되지 않도록 하기 위함이다(공유 상수 보호). 혹시 알 수 없는
    티어가 들어오면 가장 보수적인 TIER_S 설정으로 안전하게 대체한다.
    """
    return dict(TIER_CONFIG.get(tier, TIER_CONFIG[HardwareTier.TIER_S]))
