#!/usr/bin/env python3
"""
테넌트별 부트스트랩(초기) 학습 데이터 생성 CLI (M7 마일스톤, 2026-04-22).

[이 파일이 하는 일 — 한눈에 보기]
Nexus는 테넌트(고객사/조직)마다 별도의 LoRA 어댑터를 학습한다. 학습을 시작하려면
"부트스트랩 데이터"라 부르는 Phase 1용 합성(synthetic) 학습 샘플이 먼저 필요하다.
실제 데이터를 만드는 무거운 로직은 `training.bootstrap_generator.BootstrapGenerator`
라이브러리 안에 들어 있고, 이 스크립트는 그 라이브러리를 커맨드라인에서 쉽게 돌릴 수
있도록 감싸주는 얇은 진입점(entry point)이다.

[동작 흐름 요약]
1) 커맨드라인 인자를 파싱한다 (테넌트 ID, 샘플 개수, 출력 경로, 시드, dry-run).
2) 테넌트 ID를 표준 형태로 정규화한다(normalize_tenant_id).
3) BootstrapGenerator를 만들고 비동기 generate()를 실행해 JSONL 파일을 생성한다.
4) 생성 결과 통계를 로그로 출력한다.

[출력 파일 규약]
결과 JSONL은 `scripts/train_tenant_lora.py`(실제 LoRA 학습 스크립트)가 기본으로 찾는
위치와 동일한 규약으로 저장된다. 그래서 여기서 데이터를 만든 뒤 학습 스크립트를 바로
이어서 돌릴 수 있다. 테넌트 ID가 주어지면 출력 루트 아래에 테넌트별 서브디렉토리가
자동으로 만들어진다.

[주요 구성 요소]
- parse_args(): argparse 기반 CLI 인자 정의/파싱
- main(): 실제 실행 오케스트레이션 (정규화 → 생성 → 통계 로깅)

[외부 의존]
- core.adapter_naming.normalize_tenant_id — 테넌트 ID 정규화 규칙의 단일 소스
- training.bootstrap_generator.BootstrapGenerator — 데이터 생성 본체(비동기)

사용 예시:
  # default 테넌트 (기존 호환 경로)
  python scripts/generate_bootstrap.py --count 5000

  # 특정 테넌트
  python scripts/generate_bootstrap.py --tenant-id dongguk --count 3000

  # 출력 루트 변경 (에어갭 서버 경로와 맞추기)
  python scripts/generate_bootstrap.py \\
      --tenant-id dongguk --count 3000 \\
      --output-root /opt/nexus-gpu/training

작성자: 이현수 / 작성일: 2026-07-05
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

# 이 스크립트는 리포지토리 어디서 실행하든 `training` 패키지를 import할 수 있어야 한다.
# 그래서 파일 위치(scripts/) 기준으로 상위 폴더(=리포 루트)를 계산해 sys.path 맨 앞에
# 끼워 넣는다. 이미 들어 있으면 중복 추가하지 않는다.
# 주의: 이 경로 조작이 아래 training.* import보다 반드시 먼저 실행돼야 하므로, 해당
# import 줄에는 "모듈 최상단이 아니어도 된다"는 의미의 # noqa: E402가 붙어 있다.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# training 패키지 의존성 — sys.path 설정 이후에 import해야 하므로 E402 예외 처리.
from core.adapter_naming import normalize_tenant_id  # noqa: E402
from training.bootstrap_generator import BootstrapGenerator  # noqa: E402

# 로깅 기본 설정: 시각 + 레벨 + 메시지 형식으로 INFO 이상을 표준 출력한다.
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
# 이 스크립트 전용 로거. 다른 모듈 로그와 구분되도록 고유 이름을 부여한다.
logger = logging.getLogger("generate_bootstrap")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """커맨드라인 인자를 정의하고 파싱한다.

    이 스크립트가 받는 모든 옵션(--tenant-id, --count, --output-root, --seed,
    --dry-run)을 argparse로 선언하고, 파싱 결과인 Namespace 객체를 돌려준다.

    매개변수:
        argv: 파싱할 인자 리스트. None이면 실제 프로세스 인자(sys.argv[1:])를 쓴다.
              테스트에서 인자를 직접 주입할 때 리스트를 넘길 수 있게 열어 둔 것.

    반환:
        argparse.Namespace — 각 옵션 값이 속성으로 담긴 객체.
    """
    parser = argparse.ArgumentParser(
        description="테넌트별 Phase 1 부트스트랩 데이터 JSONL 생성기 (M7)"
    )
    parser.add_argument(
        "--tenant-id",
        default=None,
        help="테넌트 식별자. 생략/'default'면 기존 공용 경로에 저장.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=1000,
        help="생성할 총 샘플 수 (기본 1000).",
    )
    parser.add_argument(
        "--output-root",
        default="data/bootstrap",
        help=(
            "출력 루트 디렉토리. tenant_id가 주어지면 이 경로 아래 "
            "{tenant_id}/ 서브디렉토리가 생성된다. (기본: data/bootstrap)"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="재현성을 위한 난수 시드. 미지정 시 비결정적.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="경로/파라미터만 출력하고 실제 생성은 하지 않는다.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """스크립트의 실제 진입점 — 인자 파싱부터 데이터 생성, 통계 로깅까지 담당한다.

    흐름:
        1) parse_args로 옵션을 읽는다.
        2) normalize_tenant_id로 테넌트 ID를 표준형으로 바꾼다(None/'default' 처리 포함).
        3) 실행 파라미터를 로그로 한 번 찍어 무엇으로 돌리는지 눈으로 확인시킨다.
        4) --dry-run이면 여기서 조기 종료(파일 생성 없음).
        5) BootstrapGenerator를 만들고 비동기 generate()를 asyncio.run으로 실행한다.
        6) 생성 통계(dict)를 보기 좋은 JSON으로 로그에 출력한다.

    매개변수:
        argv: 테스트/재사용을 위해 인자 리스트를 직접 주입할 수 있다. None이면 실제 CLI 인자.

    반환:
        int — 프로세스 종료 코드. 정상 종료는 항상 0.
    """
    # 1) CLI 인자 파싱 → 2) 테넌트 ID 정규화(빈 값/'default' 규칙은 라이브러리가 담당).
    args = parse_args(argv)
    tid = normalize_tenant_id(args.tenant_id)

    # 3) 실제 어떤 파라미터로 돌아가는지 시작 배너로 남긴다(재현/디버깅에 유용).
    logger.info("=" * 60)
    logger.info("M7 부트스트랩 데이터 생성기")
    logger.info("  tenant_id   = %s", tid)
    logger.info("  count       = %d", args.count)
    logger.info("  output_root = %s", args.output_root)
    logger.info("  seed        = %s", args.seed)
    logger.info("=" * 60)

    # 4) dry-run 모드: 경로/파라미터만 확인하고 싶을 때. 실제 파일은 만들지 않고 종료.
    if args.dry_run:
        logger.info("--dry-run — 실제 생성 없이 종료")
        return 0

    # 5) 생성기 인스턴스화. seed를 넘기면 같은 입력에 항상 같은 데이터가 나와 재현 가능.
    generator = BootstrapGenerator(seed=args.seed)
    # generate()는 비동기 코루틴이라 asyncio.run으로 이벤트 루프를 열어 끝까지 돌린다.
    # 반환되는 stats는 생성 개수 등 요약 정보가 담긴 dict.
    stats = asyncio.run(
        generator.generate(
            count=args.count,
            output_path=args.output_root,
            tenant_id=tid,
        )
    )
    # 6) 통계를 JSON으로 보기 좋게 출력. ensure_ascii=False로 한글이 깨지지 않게 한다.
    logger.info("생성 통계: %s", json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


# 모듈로 import될 때가 아니라 직접 실행될 때만 main()을 돌린다.
# main()이 돌려준 종료 코드를 그대로 프로세스 종료 코드로 전달한다.
if __name__ == "__main__":
    raise SystemExit(main())
