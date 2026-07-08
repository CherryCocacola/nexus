"""
테넌트별 LoRA 어댑터 네이밍 규약 (M7, 2026-04-22).

■ 이 파일이 하는 일 (한 줄 요약)
  "어떤 테넌트의 Phase 몇 학습이냐"라는 입력을 받아, 그에 대응하는
  LoRA 어댑터 이름 / 체크포인트 디렉토리 / 학습데이터 경로 문자열을
  일관된 규칙으로 만들어 주는 "이름 짓기 전용" 헬퍼 모듈이다.

■ 왜 필요한가 (배경)
  Nexus의 어댑터 이름은 vLLM에 `--lora-modules`로 등록되는 식별자와 그대로 같다.
  멀티테넌시 환경에서는 하나의 베이스 모델(qwen3.5-27b) 위에 학교·기업별 LoRA를
  여러 개 동시에 올려야 한다. 이때 이름이 겹치거나 규칙이 제각각이면
  A테넌트 요청이 B테넌트 어댑터로 잘못 라우팅(오라우팅)될 수 있다.
  그래서 어댑터 이름에 tenant_id를 구조적으로 박아 넣어 충돌을 원천 차단한다.

■ 어댑터 이름 규약 (compose_adapter_name)
  - default 테넌트: `nexus-phaseN`                (기존 이름과의 호환 유지)
  - 특정 테넌트:    `nexus-{tenant_id}-phaseN`    (예: nexus-dongguk-phase3)
  - 자유 이름:      custom_prefix가 있으면 그 값을 최우선으로 사용
    (TenantConfig.adapter_name_prefix 같은 테넌트 커스텀 브랜딩 대응)

■ 왜 default만 접두 없이 두는가
  - 이미 운영 중인 nexus-phase3 어댑터를 그대로 재사용하기 위한 하위호환.
  - default 테넌트는 tenant_id="default"인 "공통" 역할이라, 이름에까지
    default를 또 넣으면(nexus-default-phase3) 오히려 헷갈린다.

■ 경로 규약 (학습 산출물 저장)
  - 체크포인트: /opt/nexus-gpu/checkpoints/qwen35-phaseN            (default)
  - 체크포인트: /opt/nexus-gpu/checkpoints/qwen35-{tenant_id}-phaseN (테넌트별)
  - 학습데이터: /opt/nexus-gpu/training/bootstrap_data.jsonl        (default)
  - 학습데이터: /opt/nexus-gpu/training/{tenant_id}/bootstrap_data.jsonl (격리)

■ 공개 함수 (이 모듈이 노출하는 API)
  - normalize_tenant_id : tenant_id를 검증·정규화 (안전한 문자셋으로 통일)
  - compose_adapter_name: vLLM 어댑터 이름 생성
  - compose_output_dir  : 체크포인트 출력 디렉토리 경로 생성
  - compose_data_path   : 학습 데이터 JSONL 경로 생성

■ 설계 의도
  이 모듈은 오직 문자열만 가공한다. GPU·파일시스템·네트워크에 전혀 닿지 않으므로
  학습 쪽이든 서빙 쪽이든 테스트 쪽이든 부작용 없이 어디서나 재사용할 수 있다.

작성자: 이현수 / 작성일: 2026-07-05
"""

from __future__ import annotations

import re

# 유효한 tenant_id 문자 규칙 — DNS label 수준으로 제한한다.
# 이유: tenant_id는 어댑터 이름·파일 경로·URL 등에 그대로 끼어 들어갈 수 있어,
#       위험 문자가 섞이면 경로 조작·오라우팅 사고로 이어질 수 있기 때문이다.
# 패턴 뜻: 첫 글자는 영소문자/숫자, 이후 최대 62자까지 영소문자/숫자/'-'/'_'.
#          (전체 1~63자) 한국어·공백·대문자·특수문자는 허용하지 않는다.
_VALID_TENANT_ID = re.compile(r"^[a-z0-9][a-z0-9\-_]{0,62}$")

# 허용하는 phase 정수 범위 — Nexus의 5-Phase 학습 전략과 일치한다 (Phase 0~4).
# 이 범위를 벗어난 phase는 아래 compose_* 함수들이 ValueError로 즉시 거부한다.
MIN_PHASE = 0
MAX_PHASE = 4

# 기본(공통) 테넌트 id — 입력이 이 값과 같으면 이름에서 tenant 접두를 생략한다.
# 그래야 예전부터 쓰던 `nexus-phaseN` 이름과 하위호환이 유지된다.
DEFAULT_TENANT_ID = "default"


def normalize_tenant_id(tenant_id: str | None) -> str:
    """tenant_id를 검증하고 표준 형태로 정규화한다.

    이 모듈의 모든 이름/경로 생성 함수는 tenant_id를 쓰기 전에 반드시 이 함수를
    거친다. 즉 "입력을 신뢰하기 전에 한 번 통과시키는 관문" 역할이다.
    덕분에 뒤쪽 함수들은 tenant_id가 항상 안전하다고 가정할 수 있다.

    처리 규칙:
      - None·빈 문자열·'default' → 모두 "default"로 통일 (공통 테넌트로 간주)
      - 앞뒤 공백 제거 후 소문자로 내림 (예: " Dongguk " → "dongguk")
        → 대소문자만 다른 이름이 서로 다른 테넌트로 오인되는 사고를 막는다.
      - 최종 값이 허용 문자셋(_VALID_TENANT_ID)을 벗어나면 예외를 던진다.

    Args:
        tenant_id: 원본 테넌트 식별자 (None 가능).

    Returns:
        정규화된 tenant_id 문자열. 기본 테넌트면 "default".

    Raises:
        ValueError: 정규화 후에도 허용 문자셋을 벗어난 경우.
    """
    # 1) None이나 빈 문자열이면 곧바로 기본 테넌트로 처리한다.
    if not tenant_id:
        return DEFAULT_TENANT_ID
    # 2) 앞뒤 공백을 제거하고 전부 소문자로 낮춰 표기를 통일한다.
    tid = tenant_id.strip().lower()
    # 3) 공백만 있었거나 명시적으로 'default'면 역시 기본 테넌트로 수렴시킨다.
    if not tid or tid == DEFAULT_TENANT_ID:
        return DEFAULT_TENANT_ID
    # 4) 안전 문자셋 검사 — 통과하지 못하면 뒤 단계로 넘기지 않고 즉시 거부한다.
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
    """테넌트·Phase 조합으로 vLLM에 등록할 LoRA 어댑터 이름을 만든다.

    이 함수가 반환하는 문자열이 곧 vLLM `--lora-modules`의 어댑터 식별자이자,
    런타임에 요청을 특정 LoRA로 라우팅할 때 쓰는 키가 된다. 그래서 이름이
    테넌트별로 확실히 갈라지도록 규칙을 한곳(여기)에 모아 관리한다.

    이름 결정 우선순위 (위에서부터 먼저 적용):
      1) custom_prefix가 있으면 → `{custom_prefix}-phaseN` (테넌트 접두 생략)
      2) 정규화된 tenant가 default면 → `{prefix}-phaseN`   (하위호환 이름)
      3) 그 외 테넌트면 → `{prefix}-{tenant_id}-phaseN`

    Args:
        tenant_id: 테넌트 식별자. None/빈값/'default'는 기본 테넌트로 간주해
            기존 호환 이름(`nexus-phaseN`)을 돌려준다.
        phase: 학습 Phase 번호 (0~4). 범위를 벗어나면 ValueError.
        prefix: 기본 접두사 (기본 "nexus"). 프로젝트가 포크되는 경우를 위해 열어둠.
        custom_prefix: `TenantConfig.adapter_name_prefix`처럼 테넌트가 자기 이름
            규칙을 원하면 이 값으로 덮어쓴다. 설정되면 tenant_id 접두를 붙이지 않는다.

    Returns:
        `"nexus-phase3"` | `"nexus-dongguk-phase3"` | custom prefix 기반 이름.

    Raises:
        ValueError: phase가 허용 범위(MIN_PHASE~MAX_PHASE)를 벗어난 경우.

    Examples:
        >>> compose_adapter_name(None, 3)
        'nexus-phase3'
        >>> compose_adapter_name('dongguk', 3)
        'nexus-dongguk-phase3'
        >>> compose_adapter_name('hanyang', 2, custom_prefix='hy-custom')
        'hy-custom-phase2'
    """
    # phase가 허용 범위 안인지 먼저 확인한다. int()로 감싸는 이유는 "3" 같은
    # 문자열/실수형이 넘어와도 정수로 비교하기 위함이다. 벗어나면 바로 거부한다.
    if not (MIN_PHASE <= int(phase) <= MAX_PHASE):
        raise ValueError(
            f"phase {phase} 범위 위반 — {MIN_PHASE}~{MAX_PHASE} 허용"
        )

    # [우선순위 1] custom_prefix가 명시되면 tenant_id 접두를 아예 붙이지 않는다.
    # 이 경로는 테넌트가 완전히 자기만의 이름 체계를 원할 때만 사용한다
    # (예: 계약상 브랜딩 요구). 이때 tenant_id 정규화 자체를 건너뛴다.
    if custom_prefix:
        return f"{custom_prefix}-phase{phase}"

    # [우선순위 2·3] tenant_id를 안전하게 정규화한 뒤 default 여부로 분기한다.
    tid = normalize_tenant_id(tenant_id)
    if tid == DEFAULT_TENANT_ID:
        # default는 접두 없이 기존 이름 규약을 그대로 유지한다.
        return f"{prefix}-phase{phase}"
    # 특정 테넌트는 이름 사이에 tenant_id를 끼워 충돌을 구조적으로 막는다.
    return f"{prefix}-{tid}-phase{phase}"


def compose_output_dir(
    tenant_id: str | None,
    phase: int,
    *,
    base_dir: str = "/opt/nexus-gpu/checkpoints",
    model_stem: str = "qwen35",
) -> str:
    """학습 산출물(체크포인트)을 저장할 출력 디렉토리 경로를 만든다.

    어댑터 이름(compose_adapter_name)과 같은 테넌트/phase 분기 규칙을 경로에도
    그대로 적용해, "이름 규약"과 "저장 위치 규약"이 어긋나지 않도록 맞춘다.

    경로 규약:
      - default 테넌트: `{base_dir}/{model_stem}-phaseN`
      - 테넌트별:        `{base_dir}/{model_stem}-{tenant_id}-phaseN`

    Args:
        tenant_id: 테넌트 식별자.
        phase: 학습 Phase.
        base_dir: 체크포인트 루트 (GPU 서버 로컬 경로).
        model_stem: 모델 식별 접두사 (기본 "qwen35").

    Returns:
        체크포인트를 저장할 절대 디렉토리 경로 문자열.

    Raises:
        ValueError: phase가 허용 범위를 벗어난 경우.

    Examples:
        >>> compose_output_dir(None, 3)
        '/opt/nexus-gpu/checkpoints/qwen35-phase3'
        >>> compose_output_dir('dongguk', 3)
        '/opt/nexus-gpu/checkpoints/qwen35-dongguk-phase3'
    """
    # 어댑터 이름과 동일하게 phase 범위를 먼저 검증한다 (일관성 유지).
    if not (MIN_PHASE <= int(phase) <= MAX_PHASE):
        raise ValueError(
            f"phase {phase} 범위 위반 — {MIN_PHASE}~{MAX_PHASE} 허용"
        )
    tid = normalize_tenant_id(tenant_id)
    # base_dir 끝에 실수로 붙은 슬래시를 제거해 `//` 중복 경로가 생기지 않게 한다.
    base = base_dir.rstrip("/")
    if tid == DEFAULT_TENANT_ID:
        # default는 tenant 세그먼트 없이 모델stem-phase만으로 경로를 구성한다.
        return f"{base}/{model_stem}-phase{phase}"
    # 테넌트별은 디렉토리 이름에 tenant_id를 넣어 산출물을 물리적으로 분리한다.
    return f"{base}/{model_stem}-{tid}-phase{phase}"


def compose_data_path(
    tenant_id: str | None,
    *,
    base_dir: str = "/opt/nexus-gpu/training",
    filename: str = "bootstrap_data.jsonl",
) -> str:
    """학습에 사용할 데이터(JSONL) 파일 경로를 만든다.

    앞의 두 함수와 달리 여기서는 tenant_id를 파일명 접두가 아니라 "하위 디렉토리"로
    분리한다. 즉 테넌트별 학습 데이터를 폴더 단위로 격리해, 서로 다른 테넌트의
    데이터가 같은 파일명으로 섞이는 사고를 구조적으로 막는다.

    경로 규약:
      - default: `{base_dir}/{filename}`                     (기존 호환)
      - 테넌트: `{base_dir}/{tenant_id}/{filename}`          (하위 폴더로 격리)

    Args:
        tenant_id: 테넌트 식별자.
        base_dir: 학습 데이터 루트 (GPU 서버 로컬 경로).
        filename: 데이터 파일명 (기본 "bootstrap_data.jsonl").

    Returns:
        학습 데이터 JSONL의 절대 경로 문자열.
    """
    tid = normalize_tenant_id(tenant_id)
    # 위 함수들과 동일하게 끝 슬래시를 정리해 경로 결합 시 `//`를 방지한다.
    base = base_dir.rstrip("/")
    if tid == DEFAULT_TENANT_ID:
        # default는 루트 바로 아래에 파일을 둔다 (기존 경로와 호환).
        return f"{base}/{filename}"
    # 테넌트별은 tenant_id 폴더를 한 단계 끼워 데이터를 격리한다.
    return f"{base}/{tid}/{filename}"
