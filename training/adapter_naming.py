"""
테넌트별 LoRA 어댑터 네이밍 규약 (M7, 2026-04-22).

Nexus의 어댑터 이름은 vLLM에 `--lora-modules`로 등록되는 식별자와 동일하다.
멀티테넌시 환경에서는 동일한 베이스 모델(qwen3.5-27b) 위에 학교·기업별 LoRA를
올려야 하므로, 어댑터 이름이 tenant를 포함해 충돌·오라우팅을 구조적으로 막는다.

규약:
  - default 테넌트: `nexus-phaseN`                  (기존 호환 유지)
  - 특정 테넌트:   `nexus-{tenant_id}-phaseN`       (예: nexus-dongguk-phase3)
  - 자유 이름:    TenantConfig.adapter_name_prefix가 있으면 그 값을 우선

왜 default는 접두 없이 두는가:
  - 이미 운영 중인 nexus-phase3 어댑터를 호환 유지하기 위함
  - default 테넌트 자체가 tenant_id="default"인 "공통" 역할이라 이름에 중복 표기
    (nexus-default-phase3)는 혼동을 유발

경로 규약 (학습 산출물 저장):
  - /opt/nexus-gpu/checkpoints/qwen35-phaseN               (default)
  - /opt/nexus-gpu/checkpoints/qwen35-{tenant_id}-phaseN   (테넌트별)

이 모듈은 순수 문자열 가공만 하고 GPU·파일시스템에 닿지 않아
학습/서빙/테스트 어디서나 재사용 가능하다.
"""

from __future__ import annotations

import re

# 유효한 tenant_id 문자 — DNS label 수준으로 제한 (vLLM 어댑터 이름이 경로·URL에
# 섞여 쓰일 수 있어 안전한 문자만 허용). 한국어·공백·특수문자 금지.
_VALID_TENANT_ID = re.compile(r"^[a-z0-9][a-z0-9\-_]{0,62}$")

# 허용 phase 정수 범위 — Nexus의 5-Phase 전략과 일치 (Phase 0~4)
MIN_PHASE = 0
MAX_PHASE = 4

# 기본 테넌트 id — 이 값과 같으면 접두를 생략해 기존 이름 규약과 호환된다.
DEFAULT_TENANT_ID = "default"


def normalize_tenant_id(tenant_id: str | None) -> str:
    """tenant_id를 검증·정규화한다.

    - None/빈 값/'default' → "default"
    - 영소문자·숫자·`-_`만 허용 (DNS label 규약과 동일한 안전 문자셋)
    - 대문자는 소문자로 내림 (운영 혼동 방지)

    Raises:
        ValueError: 허용 문자셋을 벗어난 경우.
    """
    if not tenant_id:
        return DEFAULT_TENANT_ID
    tid = tenant_id.strip().lower()
    if not tid or tid == DEFAULT_TENANT_ID:
        return DEFAULT_TENANT_ID
    if not _VALID_TENANT_ID.match(tid):
        raise ValueError(
            f"invalid tenant_id={tenant_id!r} — 소문자/숫자/'-'/'_'만 허용, "
            "1~63자, 첫 글자는 영숫자"
        )
    return tid


def compose_adapter_name(
    tenant_id: str | None,
    phase: int,
    *,
    prefix: str = "nexus",
    custom_prefix: str | None = None,
) -> str:
    """테넌트·Phase 조합에서 LoRA 어댑터 이름을 만든다.

    Args:
        tenant_id: 테넌트 식별자. None/빈값/'default'는 기본 테넌트로 간주해
            기존 호환 이름 (`nexus-phaseN`)을 돌려준다.
        phase: 학습 Phase 번호 (0~4). 범위를 벗어나면 ValueError.
        prefix: 기본 접두사 (기본 "nexus"). 프로젝트가 포크되는 경우를 위해 열어둠.
        custom_prefix: `TenantConfig.adapter_name_prefix`처럼 테넌트가 자기 이름
            규칙을 원하면 이 값으로 덮어쓴다. 설정되면 tenant_id 접두를 붙이지 않는다.

    Returns:
        `"nexus-phase3"` | `"nexus-dongguk-phase3"` | custom prefix 기반 이름.

    Examples:
        >>> compose_adapter_name(None, 3)
        'nexus-phase3'
        >>> compose_adapter_name('dongguk', 3)
        'nexus-dongguk-phase3'
        >>> compose_adapter_name('hanyang', 2, custom_prefix='hy-custom')
        'hy-custom-phase2'
    """
    if not (MIN_PHASE <= int(phase) <= MAX_PHASE):
        raise ValueError(
            f"phase {phase} 범위 위반 — {MIN_PHASE}~{MAX_PHASE} 허용"
        )

    # custom_prefix가 명시되면 tenant_id 접두사를 덧붙이지 않는다.
    # 이 경로는 테넌트가 완전 자기만의 이름 체계를 원할 때만 사용 (예: 계약상 브랜딩).
    if custom_prefix:
        return f"{custom_prefix}-phase{phase}"

    tid = normalize_tenant_id(tenant_id)
    if tid == DEFAULT_TENANT_ID:
        return f"{prefix}-phase{phase}"
    return f"{prefix}-{tid}-phase{phase}"


def compose_output_dir(
    tenant_id: str | None,
    phase: int,
    *,
    base_dir: str = "/opt/nexus-gpu/checkpoints",
    model_stem: str = "qwen35",
) -> str:
    """체크포인트 출력 디렉토리 경로를 만든다.

    경로 규약:
      - default 테넌트: `{base_dir}/{model_stem}-phaseN`
      - 테넌트별:        `{base_dir}/{model_stem}-{tenant_id}-phaseN`

    Args:
        tenant_id: 테넌트 식별자.
        phase: 학습 Phase.
        base_dir: 체크포인트 루트 (GPU 서버 로컬 경로).
        model_stem: 모델 식별 접두사 (기본 "qwen35").

    Examples:
        >>> compose_output_dir(None, 3)
        '/opt/nexus-gpu/checkpoints/qwen35-phase3'
        >>> compose_output_dir('dongguk', 3)
        '/opt/nexus-gpu/checkpoints/qwen35-dongguk-phase3'
    """
    if not (MIN_PHASE <= int(phase) <= MAX_PHASE):
        raise ValueError(
            f"phase {phase} 범위 위반 — {MIN_PHASE}~{MAX_PHASE} 허용"
        )
    tid = normalize_tenant_id(tenant_id)
    base = base_dir.rstrip("/")
    if tid == DEFAULT_TENANT_ID:
        return f"{base}/{model_stem}-phase{phase}"
    return f"{base}/{model_stem}-{tid}-phase{phase}"


def compose_data_path(
    tenant_id: str | None,
    *,
    base_dir: str = "/opt/nexus-gpu/training",
    filename: str = "bootstrap_data.jsonl",
) -> str:
    """학습 데이터 JSONL 경로를 만든다.

    경로 규약:
      - default: `{base_dir}/{filename}`                     (기존 호환)
      - 테넌트: `{base_dir}/{tenant_id}/{filename}`          (격리)
    """
    tid = normalize_tenant_id(tenant_id)
    base = base_dir.rstrip("/")
    if tid == DEFAULT_TENANT_ID:
        return f"{base}/{filename}"
    return f"{base}/{tid}/{filename}"
