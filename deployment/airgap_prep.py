"""
에어갭 번들 준비 — 폐쇄망 배포를 위한 오프라인 패키지 생성.

외부 네트워크 없이 Nexus를 배포하기 위한 번들을 준비한다.
Python 패키지(wheel), 모델 weights, 설정 파일, 데이터를 하나의 디렉토리에 모아서
SHA256 매니페스트와 함께 패키징한다.

사용 시나리오:
  1. 온라인 환경에서 prepare_bundle()로 번들을 준비한다
  2. 준비된 번들을 USB/물리 매체로 에어갭 환경에 이동한다
  3. 에어갭 환경에서 verify_manifest()로 무결성을 확인한다
  4. 번들에서 설치를 진행한다

의존성: deployment/integrity.py (무결성 검증)
"""

from __future__ import annotations

import json
import logging
import shutil
from datetime import UTC, datetime
from pathlib import Path

from deployment.integrity import IntegrityVerifier

logger = logging.getLogger("nexus.deployment.airgap_prep")


class AirGapPrep:
    """
    에어갭 환경 배포를 위한 번들을 준비한다.

    번들 구조:
      bundle/
        packages/     — Python wheel 파일 (.whl)
        models/       — 모델 weights (.safetensors, .bin)
        config/       — 설정 파일 (.yaml)
        data/         — 초기 데이터 (벡터 DB seed 등)
        scripts/      — 설치/업데이트 스크립트
        manifest.json — SHA256 매니페스트
        README.txt    — 설치 가이드
    """

    def __init__(self):
        """번들 준비기를 초기화한다."""
        self._verifier = IntegrityVerifier()

    async def prepare_bundle(self, output_dir: str) -> dict:
        """
        전체 배포 번들을 준비한다.

        패키지, 모델, 설정, 데이터를 수집하고
        SHA256 매니페스트를 생성한다.

        Args:
            output_dir: 번들을 생성할 출력 디렉토리 경로

        Returns:
            번들 요약 정보 딕셔너리:
              - bundle_dir: 번들 경로
              - total_files: 파일 수
              - total_size_bytes: 전체 크기
              - manifest_path: 매니페스트 경로
              - created_at: 생성 시각 (ISO 8601)
        """
        bundle_dir = Path(output_dir)

        # 번들 디렉토리 구조를 생성한다
        subdirs = ["packages", "models", "config", "data", "scripts"]
        for subdir in subdirs:
            (bundle_dir / subdir).mkdir(parents=True, exist_ok=True)
        logger.info(f"번들 디렉토리 생성: {bundle_dir}")

        # ① Python 패키지 복사
        packages_count = await self._collect_packages(bundle_dir / "packages")

        # ② 모델 weights 복사
        models_count = await self._collect_models(bundle_dir / "models")

        # ③ 설정 파일 복사
        config_count = await self._collect_configs(bundle_dir / "config")

        # ④ 데이터 파일 복사
        data_count = await self._collect_data(bundle_dir / "data")

        # ⑤ 설치 스크립트 생성
        await self._generate_install_scripts(bundle_dir / "scripts")

        # ⑥ SHA256 매니페스트 생성
        manifest = self.generate_manifest(str(bundle_dir))
        manifest_path = bundle_dir / "manifest.json"
        manifest_json = json.dumps(manifest, indent=2, ensure_ascii=False)
        manifest_path.write_text(manifest_json, encoding="utf-8")
        logger.info(f"매니페스트 생성: {manifest_path} ({len(manifest)}개 파일)")

        # ⑦ README 생성
        await self._generate_readme(bundle_dir)

        # 번들 요약 반환
        total_size = sum(f.stat().st_size for f in bundle_dir.rglob("*") if f.is_file())
        summary = {
            "bundle_dir": str(bundle_dir),
            "total_files": len(manifest),
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "manifest_path": str(manifest_path),
            "created_at": datetime.now(UTC).isoformat(),
            "breakdown": {
                "packages": packages_count,
                "models": models_count,
                "config": config_count,
                "data": data_count,
            },
        }
        logger.info(f"번들 준비 완료: {summary}")
        return summary

    def generate_manifest(self, bundle_dir: str) -> dict[str, str]:
        """
        번들 디렉토리의 SHA256 매니페스트를 생성한다.

        IntegrityVerifier를 위임하여 모든 파일의 해시를 계산한다.

        Args:
            bundle_dir: 번들 디렉토리 경로

        Returns:
            {상대경로: SHA256 해시} 딕셔너리
        """
        return self._verifier.generate_manifest(bundle_dir)

    def verify_manifest(self, bundle_dir: str) -> tuple[bool, list[str]]:
        """
        번들의 매니페스트를 검증한다.

        bundle_dir/manifest.json 파일을 읽어서
        각 파일의 SHA256 해시를 검증한다.

        Args:
            bundle_dir: 검증할 번들 디렉토리 경로

        Returns:
            (전체 통과 여부, 실패한 파일 경로 목록)
        """
        manifest_path = Path(bundle_dir) / "manifest.json"
        if not manifest_path.exists():
            logger.error(f"매니페스트 파일을 찾을 수 없습니다: {manifest_path}")
            return False, [f"매니페스트 없음: {manifest_path}"]

        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)

        return self._verifier.verify_directory(bundle_dir, manifest)

    # ─── 내부 수집 메서드 ───

    async def _collect_packages(self, target_dir: Path) -> int:
        """
        Python wheel 패키지를 수집한다.

        프로젝트 루트의 offline_packages/ 또는 pip download 결과에서
        .whl 파일을 복사한다.

        Args:
            target_dir: 패키지를 복사할 대상 디렉토리

        Returns:
            복사된 패키지 수
        """
        # 오프라인 패키지 소스 디렉토리 탐색
        source_dirs = [
            Path("offline_packages"),
            Path("deployment/offline_packages"),
        ]

        count = 0
        for source in source_dirs:
            if source.exists():
                for whl in source.glob("*.whl"):
                    shutil.copy2(str(whl), str(target_dir / whl.name))
                    count += 1
                logger.info(f"패키지 {count}개 복사: {source} → {target_dir}")
                break

        if count == 0:
            logger.warning(
                "오프라인 패키지 소스를 찾을 수 없습니다. "
                "번들 생성 전에 'pip download' 명령으로 패키지를 준비하세요."
            )

        return count

    async def _collect_models(self, target_dir: Path) -> int:
        """
        모델 weights 파일을 수집한다.

        프로젝트 루트의 models/ 디렉토리에서 모델 파일을 복사한다.

        Args:
            target_dir: 모델을 복사할 대상 디렉토리

        Returns:
            복사된 모델 파일 수
        """
        model_dir = Path("models")
        if not model_dir.exists():
            logger.warning("models/ 디렉토리를 찾을 수 없습니다.")
            return 0

        # 모델 파일 확장자
        model_extensions = {".safetensors", ".bin", ".gguf", ".json", ".txt", ".model"}
        count = 0

        for model_path in model_dir.rglob("*"):
            if model_path.is_file() and model_path.suffix in model_extensions:
                # 모델 디렉토리 구조를 유지하며 복사한다
                relative = model_path.relative_to(model_dir)
                dest = target_dir / relative
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(str(model_path), str(dest))
                count += 1

        logger.info(f"모델 파일 {count}개 복사: {model_dir} → {target_dir}")
        return count

    async def _collect_configs(self, target_dir: Path) -> int:
        """
        설정 파일을 수집한다.

        config/ 디렉토리의 YAML 파일을 복사한다.

        Args:
            target_dir: 설정을 복사할 대상 디렉토리

        Returns:
            복사된 설정 파일 수
        """
        config_dir = Path("config")
        if not config_dir.exists():
            logger.warning("config/ 디렉토리를 찾을 수 없습니다.")
            return 0

        count = 0
        for config_file in config_dir.glob("*.yaml"):
            shutil.copy2(str(config_file), str(target_dir / config_file.name))
            count += 1

        # .yml 확장자도 복사한다
        for config_file in config_dir.glob("*.yml"):
            shutil.copy2(str(config_file), str(target_dir / config_file.name))
            count += 1

        logger.info(f"설정 파일 {count}개 복사: {config_dir} → {target_dir}")
        return count

    async def _collect_data(self, target_dir: Path) -> int:
        """
        초기 데이터 파일을 수집한다.

        data/ 디렉토리의 데이터 파일을 복사한다.

        Args:
            target_dir: 데이터를 복사할 대상 디렉토리

        Returns:
            복사된 데이터 파일 수
        """
        data_dir = Path("data")
        if not data_dir.exists():
            logger.warning("data/ 디렉토리를 찾을 수 없습니다.")
            return 0

        count = 0
        for data_file in data_dir.rglob("*"):
            if data_file.is_file():
                relative = data_file.relative_to(data_dir)
                dest = target_dir / relative
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(str(data_file), str(dest))
                count += 1

        logger.info(f"데이터 파일 {count}개 복사: {data_dir} → {target_dir}")
        return count

    async def _generate_install_scripts(self, scripts_dir: Path) -> None:
        """
        설치 스크립트를 생성한다.

        에어갭 환경에서 번들을 설치하기 위한 스크립트를 작성한다.
        """
        # Linux/macOS 설치 스크립트
        install_sh = scripts_dir / "install.sh"
        install_sh.write_text(
            "#!/bin/bash\n"
            "# Project Nexus 에어갭 설치 스크립트\n"
            "set -e\n\n"
            'SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"\n'
            'BUNDLE_DIR="$(dirname "$SCRIPT_DIR")"\n\n'
            "# 1. 무결성 검증\n"
            'echo "번들 무결성 검증 중..."\n'
            'python3 -c "\n'
            "from deployment.integrity import IntegrityVerifier\n"
            "import json\n"
            "v = IntegrityVerifier()\n"
            "with open('${BUNDLE_DIR}/manifest.json') as f:\n"
            "    manifest = json.load(f)\n"
            "ok, failures = v.verify_directory('${BUNDLE_DIR}', manifest)\n"
            "if not ok:\n"
            "    print(f'실패: {failures}')\n"
            "    exit(1)\n"
            "print('무결성 검증 통과')\n"
            '"\n\n'
            "# 2. 패키지 설치 (오프라인)\n"
            'echo "패키지 설치 중..."\n'
            "pip install --no-index --find-links=${BUNDLE_DIR}/packages "
            "-r requirements.txt\n\n"
            "# 3. 설정 복사\n"
            'echo "설정 복사 중..."\n'
            "cp -r ${BUNDLE_DIR}/config/* config/\n\n"
            'echo "설치 완료!"\n',
            encoding="utf-8",
        )

        logger.info(f"설치 스크립트 생성: {install_sh}")

    async def _generate_readme(self, bundle_dir: Path) -> None:
        """
        번들 README를 생성한다.

        설치 절차와 검증 방법을 안내한다.
        """
        readme = bundle_dir / "README.txt"
        readme.write_text(
            "=" * 60 + "\n"
            "Project Nexus 에어갭 배포 번들\n"
            "=" * 60 + "\n\n"
            "1. 무결성 검증:\n"
            '   python -c "from deployment.airgap_prep import AirGapPrep; '
            "ok, f = AirGapPrep().verify_manifest('.'); "
            "print('OK' if ok else f)\"\n\n"
            "2. 설치:\n"
            "   bash scripts/install.sh\n\n"
            "3. 번들 구조:\n"
            "   packages/  — Python wheel 패키지\n"
            "   models/    — 모델 weights\n"
            "   config/    — 설정 파일\n"
            "   data/      — 초기 데이터\n"
            "   scripts/   — 설치 스크립트\n\n"
            f"생성 시각: {datetime.now(UTC).isoformat()}\n",
            encoding="utf-8",
        )
        logger.info(f"README 생성: {readme}")
