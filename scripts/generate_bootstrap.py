#!/usr/bin/env python3
"""
테넌트별 부트스트랩 데이터 생성 CLI (M7, 2026-04-22).

`BootstrapGenerator`는 Phase 1 학습용 합성 데이터를 만드는 라이브러리다.
이 스크립트는 그 라이브러리를 CLI로 노출하여, 테넌트별로 분리된 JSONL 파일을
쉽게 생성할 수 있게 한다. 결과물은 `scripts/train_tenant_lora.py`가 기본
경로로 잡는 위치와 같은 규약으로 저장된다.

사용 예시:
  # default 테넌트 (기존 호환 경로)
  python scripts/generate_bootstrap.py --count 5000

  # 특정 테넌트
  python scripts/generate_bootstrap.py --tenant-id dongguk --count 3000

  # 출력 루트 변경 (에어갭 서버 경로와 맞추기)
  python scripts/generate_bootstrap.py \\
      --tenant-id dongguk --count 3000 \\
      --output-root /opt/nexus-gpu/training
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from training.adapter_naming import normalize_tenant_id  # noqa: E402
from training.bootstrap_generator import BootstrapGenerator  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("generate_bootstrap")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
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
    args = parse_args(argv)
    tid = normalize_tenant_id(args.tenant_id)

    logger.info("=" * 60)
    logger.info("M7 부트스트랩 데이터 생성기")
    logger.info("  tenant_id   = %s", tid)
    logger.info("  count       = %d", args.count)
    logger.info("  output_root = %s", args.output_root)
    logger.info("  seed        = %s", args.seed)
    logger.info("=" * 60)

    if args.dry_run:
        logger.info("--dry-run — 실제 생성 없이 종료")
        return 0

    generator = BootstrapGenerator(seed=args.seed)
    stats = asyncio.run(
        generator.generate(
            count=args.count,
            output_path=args.output_root,
            tenant_id=tid,
        )
    )
    logger.info("생성 통계: %s", json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
