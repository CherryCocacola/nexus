"""
에어갭(폐쇄망) 배포 번들 준비 모듈 — 오프라인 설치용 패키지 일괄 생성.

[이 파일이 하는 일]
인터넷이 완전히 차단된 고객사/기관 서버(에어갭 환경)에 Project Nexus를 설치하려면,
설치에 필요한 모든 것을 온라인 환경에서 미리 하나의 폴더(번들)로 묶어야 한다.
이 모듈은 그 번들을 만드는 역할을 담당한다. 구체적으로:
  - Python 의존 패키지(.whl 휠 파일)  → 에어갭에서 pip이 인터넷 없이 설치할 수 있게
  - 모델 weights(.safetensors 등)      → LLM/임베딩 모델 파일
  - 설정 파일(config/*.yaml)            → 서비스 구동에 필요한 설정
  - 초기 데이터(data/)                  → 벡터 DB seed 등
  - 설치 스크립트(install.sh)와 README  → 현장에서 그대로 실행/참고
그리고 위 모든 파일의 SHA256 해시를 담은 manifest.json(무결성 매니페스트)을 함께 만든다.
이 매니페스트는 USB로 옮기는 도중 파일이 손상/변조되지 않았는지 나중에 검증하는 데 쓰인다.

[주요 클래스/함수]
  - AirGapPrep                : 번들 준비 전체를 담당하는 핵심 클래스
  - AirGapPrep.prepare_bundle : (온라인) 번들 폴더 생성 + 모든 자산 수집 + 매니페스트 작성
  - AirGapPrep.generate_manifest : 번들 디렉토리의 SHA256 매니페스트 생성
                                   (내부는 IntegrityVerifier에 위임)
  - AirGapPrep.verify_manifest   : (에어갭) manifest.json을 읽어 파일 무결성 검증

[사용 시나리오 — 온라인에서 만들고 폐쇄망에서 검증·설치]
  1. 인터넷이 되는 온라인 환경에서 prepare_bundle()로 번들을 준비한다
  2. 준비된 번들 폴더를 USB/물리 매체로 에어갭 환경에 물리적으로 옮긴다
  3. 에어갭 환경에서 verify_manifest()로 SHA256 무결성을 확인한다(손상/변조 감지)
  4. scripts/install.sh 등으로 번들 안에서 오프라인 설치를 진행한다

[의존]
  - deployment/integrity.py 의 IntegrityVerifier — 해시 계산 및 디렉토리 무결성 검증을 위임받아 수행

작성자: 이현수 / 작성일: 2026-07-05
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
    에어갭 환경 배포를 위한 번들을 준비하는 핵심 클래스.

    이 클래스 하나가 "온라인에서 번들 만들기(prepare_bundle)"와
    "폐쇄망에서 번들 검증하기(verify_manifest)"를 모두 제공한다.
    실제 해시 계산/검증 같은 무거운 일은 직접 하지 않고,
    __init__에서 만들어 두는 IntegrityVerifier(deployment/integrity.py)에 위임한다.
    즉 이 클래스는 "무엇을 어디에 모을지"를 결정하는 오케스트레이터 역할이다.

    prepare_bundle()이 만들어내는 번들 디렉토리 구조는 다음과 같다:
      bundle/
        packages/     — Python wheel 파일 (.whl) — 오프라인 pip 설치용
        models/       — 모델 weights (.safetensors, .bin 등)
        config/       — 설정 파일 (.yaml)
        data/         — 초기 데이터 (벡터 DB seed 등)
        scripts/      — 설치/업데이트 스크립트 (install.sh 등)
        manifest.json — 위 모든 파일의 SHA256 매니페스트(무결성 검증용)
        README.txt    — 설치 가이드
    """

    def __init__(self):
        """
        번들 준비기를 초기화한다.

        무결성 검증(해시 계산/디렉토리 검증)을 담당할 IntegrityVerifier 인스턴스를
        미리 하나 만들어 보관한다. 이후 generate_manifest / verify_manifest 등에서 재사용한다.
        """
        # 해시 계산·무결성 검증 전담 객체. 이 클래스의 매니페스트 관련 작업은 모두 여기에 위임한다.
        self._verifier = IntegrityVerifier()

    async def prepare_bundle(self, output_dir: str) -> dict:
        """
        전체 배포 번들을 준비한다. (온라인 환경에서 호출하는 진입점)

        [흐름] 이 메서드 하나가 번들 생성의 전 과정을 순서대로 지휘한다.
          1) 번들 폴더 뼈대(packages/models/config/data/scripts)를 만든다
          2) 각 자산을 해당 하위 폴더로 수집(복사)한다 — ①~④
          3) 설치 스크립트(install.sh)를 생성한다 — ⑤
          4) 지금까지 복사된 모든 파일의 SHA256 매니페스트를 계산해 manifest.json으로 저장 — ⑥
             (주의: 매니페스트는 반드시 모든 자산 복사가 끝난 뒤에 만들어야 전체 파일이 포함된다)
          5) README.txt를 생성한다 — ⑦
          6) 번들 전체 용량/파일 수 등을 요약 딕셔너리로 만들어 반환한다

        각 _collect_* 메서드는 "복사한 파일 개수"를 돌려주며, 소스 폴더가 없으면
        경고 로그만 남기고 0을 반환한다(예외를 던지지 않음 → 일부 자산이 없어도 번들 생성은 진행됨).

        Args:
            output_dir: 번들을 생성할 출력 디렉토리 경로

        Returns:
            번들 요약 정보 딕셔너리:
              - bundle_dir: 번들 경로
              - total_files: 매니페스트에 포함된 파일 수
              - total_size_bytes: 번들 전체 크기(바이트)
              - total_size_mb: 번들 전체 크기(MB, 소수 2자리)
              - manifest_path: 매니페스트 경로
              - created_at: 생성 시각 (ISO 8601, UTC)
              - breakdown: 자산 종류별 복사 개수(packages/models/config/data)
        """
        bundle_dir = Path(output_dir)

        # 번들 디렉토리 구조를 생성한다.
        # parents=True: 중간 경로가 없어도 함께 생성 / exist_ok=True: 이미 있어도 에러 없이 통과
        subdirs = ["packages", "models", "config", "data", "scripts"]
        for subdir in subdirs:
            (bundle_dir / subdir).mkdir(parents=True, exist_ok=True)
        logger.info(f"번들 디렉토리 생성: {bundle_dir}")

        # ① Python 패키지(.whl) 복사 — 에어갭에서 오프라인 pip 설치에 사용
        packages_count = await self._collect_packages(bundle_dir / "packages")

        # ② 모델 weights 복사 — models/ 하위 구조를 유지하며 복사
        models_count = await self._collect_models(bundle_dir / "models")

        # ③ 설정 파일(.yaml/.yml) 복사
        config_count = await self._collect_configs(bundle_dir / "config")

        # ④ 데이터 파일 복사 — data/ 하위 구조를 유지하며 복사
        data_count = await self._collect_data(bundle_dir / "data")

        # ⑤ 설치 스크립트(install.sh) 생성
        await self._generate_install_scripts(bundle_dir / "scripts")

        # ⑥ SHA256 매니페스트 생성 — 반드시 모든 자산 복사가 끝난 뒤에 실행해야 누락이 없다.
        #    manifest는 {상대경로: SHA256} 딕셔너리이며, JSON으로 직렬화해 저장한다.
        #    ensure_ascii=False: 한글 경로/내용이 이스케이프되지 않고 그대로 저장되도록
        manifest = self.generate_manifest(str(bundle_dir))
        manifest_path = bundle_dir / "manifest.json"
        manifest_json = json.dumps(manifest, indent=2, ensure_ascii=False)
        manifest_path.write_text(manifest_json, encoding="utf-8")
        logger.info(f"매니페스트 생성: {manifest_path} ({len(manifest)}개 파일)")

        # ⑦ 설치 안내 README 생성
        await self._generate_readme(bundle_dir)

        # 번들 요약 반환 — 번들 폴더를 재귀 순회하며 모든 '파일'의 크기를 합산해 총용량을 구한다.
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

        실제 해시 계산은 하지 않고 IntegrityVerifier에 그대로 위임한다(단순 래퍼).
        이렇게 감싸 두면 호출부(prepare_bundle 등)가 IntegrityVerifier를 직접 알 필요 없이
        AirGapPrep의 메서드만 사용하면 되므로 결합도가 낮아진다.

        Args:
            bundle_dir: 매니페스트를 만들 대상(번들) 디렉토리 경로

        Returns:
            {번들 기준 상대경로: 해당 파일의 SHA256 해시} 형태의 딕셔너리
        """
        return self._verifier.generate_manifest(bundle_dir)

    def verify_manifest(self, bundle_dir: str) -> tuple[bool, list[str]]:
        """
        번들의 매니페스트를 검증한다. (에어갭 현장에서 호출하는 진입점)

        USB 등으로 옮겨온 번들이 손상/변조되지 않았는지 확인하는 용도다.
        절차: bundle_dir/manifest.json을 읽어 {상대경로: 기대 해시} 목록을 얻고,
        실제 파일들의 SHA256을 다시 계산해 하나씩 대조한다(실제 대조는 IntegrityVerifier가 수행).

        Args:
            bundle_dir: 검증할 번들 디렉토리 경로

        Returns:
            (전체 통과 여부, 실패한 파일 경로 목록)
              - manifest.json 자체가 없으면 (False, ["매니페스트 없음: ..."])을
                반환한다(fail-closed).
              - 모두 통과하면 (True, []) 형태.
        """
        manifest_path = Path(bundle_dir) / "manifest.json"
        # 매니페스트가 아예 없으면 검증 자체가 불가능 → 실패로 간주하고 조기 반환(fail-closed)
        if not manifest_path.exists():
            logger.error(f"매니페스트 파일을 찾을 수 없습니다: {manifest_path}")
            return False, [f"매니페스트 없음: {manifest_path}"]

        # 매니페스트(JSON)를 읽어 {상대경로: 기대 SHA256} 딕셔너리로 복원한다
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)

        # 실제 파일 해시와 기대 해시를 대조하는 작업은 IntegrityVerifier에 위임한다
        return self._verifier.verify_directory(bundle_dir, manifest)

    # ─── 내부 수집 메서드 (prepare_bundle이 순서대로 호출) ───
    # 아래 _collect_* 메서드들은 모두 "소스 폴더 → 번들의 target_dir로 복사"하고
    # 복사한 파일 개수를 반환한다는 공통 규약을 따른다. 소스 폴더가 없으면
    # 예외를 던지지 않고 경고 로그 후 0을 반환한다(일부 자산이 없어도 번들 생성은 계속 진행).

    async def _collect_packages(self, target_dir: Path) -> int:
        """
        Python wheel(.whl) 패키지를 수집한다.

        온라인에서 미리 'pip download'로 받아 둔 오프라인 패키지들을 번들로 복사한다.
        후보 소스 폴더를 순서대로 확인해 '먼저 발견되는 하나'만 사용한다.

        Args:
            target_dir: 패키지를 복사할 대상 디렉토리 (번들의 packages/)

        Returns:
            복사된 패키지 수 (소스를 못 찾으면 0)
        """
        # 오프라인 패키지 소스 후보. 앞쪽(offline_packages/)을 우선 탐색한다.
        source_dirs = [
            Path("offline_packages"),
            Path("deployment/offline_packages"),
        ]

        count = 0
        for source in source_dirs:
            if source.exists():
                # 해당 소스 폴더의 모든 .whl 파일을 번들로 복사.
                # copy2는 파일 내용뿐 아니라 수정시각 등 메타데이터도 함께 보존한다.
                for whl in source.glob("*.whl"):
                    shutil.copy2(str(whl), str(target_dir / whl.name))
                    count += 1
                logger.info(f"패키지 {count}개 복사: {source} → {target_dir}")
                # 첫 번째로 존재하는 소스만 쓰고 멈춘다(여러 소스를 중복 복사하지 않음)
                break

        # 하나도 못 찾았다면 배포 담당자가 pip download를 빠뜨렸을 가능성 → 경고로 알린다
        if count == 0:
            logger.warning(
                "오프라인 패키지 소스를 찾을 수 없습니다. "
                "번들 생성 전에 'pip download' 명령으로 패키지를 준비하세요."
            )

        return count

    async def _collect_models(self, target_dir: Path) -> int:
        """
        모델 weights 파일을 수집한다.

        프로젝트 루트의 models/ 디렉토리를 재귀적으로 훑어, 모델 관련 확장자만 골라
        원래의 하위 폴더 구조를 그대로 유지한 채 번들로 복사한다.
        (하위 구조 유지가 중요한 이유: HuggingFace 모델은 config.json/tokenizer 등
         여러 파일이 특정 폴더 배치를 전제로 로드되기 때문.)

        Args:
            target_dir: 모델을 복사할 대상 디렉토리 (번들의 models/)

        Returns:
            복사된 모델 파일 수 (models/ 폴더가 없으면 0)
        """
        model_dir = Path("models")
        if not model_dir.exists():
            logger.warning("models/ 디렉토리를 찾을 수 없습니다.")
            return 0

        # 복사 대상으로 인정할 모델 관련 파일 확장자 화이트리스트.
        # weights(.safetensors/.bin/.gguf/.model)뿐 아니라 설정/토크나이저(.json/.txt)도 포함한다.
        model_extensions = {".safetensors", ".bin", ".gguf", ".json", ".txt", ".model"}
        count = 0

        # rglob("*")로 models/ 아래 모든 항목을 재귀 순회한다
        for model_path in model_dir.rglob("*"):
            if model_path.is_file() and model_path.suffix in model_extensions:
                # models/ 기준 상대경로를 구해 번들 안에 동일한 폴더 구조로 재현한다
                relative = model_path.relative_to(model_dir)
                dest = target_dir / relative
                # 대상 쪽 하위 폴더가 아직 없을 수 있으므로 먼저 만들어 둔다
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(str(model_path), str(dest))
                count += 1

        logger.info(f"모델 파일 {count}개 복사: {model_dir} → {target_dir}")
        return count

    async def _collect_configs(self, target_dir: Path) -> int:
        """
        설정 파일을 수집한다.

        config/ 디렉토리 '바로 아래'(재귀 아님)의 YAML 파일을 번들로 복사한다.
        .yaml과 .yml 두 확장자 표기를 모두 처리한다.

        Args:
            target_dir: 설정을 복사할 대상 디렉토리 (번들의 config/)

        Returns:
            복사된 설정 파일 수 (config/ 폴더가 없으면 0)
        """
        config_dir = Path("config")
        if not config_dir.exists():
            logger.warning("config/ 디렉토리를 찾을 수 없습니다.")
            return 0

        count = 0
        # glob("*.yaml")은 config/ 최상위의 .yaml 파일만 매칭한다(하위 폴더는 훑지 않음)
        for config_file in config_dir.glob("*.yaml"):
            shutil.copy2(str(config_file), str(target_dir / config_file.name))
            count += 1

        # 같은 의미의 .yml 확장자도 빠짐없이 복사한다
        for config_file in config_dir.glob("*.yml"):
            shutil.copy2(str(config_file), str(target_dir / config_file.name))
            count += 1

        logger.info(f"설정 파일 {count}개 복사: {config_dir} → {target_dir}")
        return count

    async def _collect_data(self, target_dir: Path) -> int:
        """
        초기 데이터 파일을 수집한다.

        data/ 디렉토리 전체를 재귀적으로 복사한다. 모델 수집과 달리 확장자 필터가 없어
        data/ 아래의 '모든 파일'을 하위 폴더 구조를 유지한 채 그대로 옮긴다.
        (예: 벡터 DB seed, 초기 사전/샘플 데이터 등.)

        Args:
            target_dir: 데이터를 복사할 대상 디렉토리 (번들의 data/)

        Returns:
            복사된 데이터 파일 수 (data/ 폴더가 없으면 0)
        """
        data_dir = Path("data")
        if not data_dir.exists():
            logger.warning("data/ 디렉토리를 찾을 수 없습니다.")
            return 0

        count = 0
        # data/ 아래 모든 항목을 재귀 순회하며 '파일'만 복사한다(폴더 자체는 건너뜀)
        for data_file in data_dir.rglob("*"):
            if data_file.is_file():
                # data/ 기준 상대경로를 그대로 번들 안에 재현해 폴더 구조를 유지한다
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

        에어갭 환경 현장에서 그대로 실행할 수 있는 install.sh(Linux/macOS용)를 작성한다.
        스크립트가 하는 일은 3단계다:
          1) manifest.json으로 번들 무결성 검증(손상 시 exit 1로 중단)
          2) 번들의 packages/에서 pip을 '오프라인 모드(--no-index)'로 실행해 의존성 설치
          3) 번들의 config/를 프로젝트 config/로 복사
        이 메서드는 스크립트를 '실행'하는 게 아니라, 위 내용을 담은 텍스트 파일을 '생성'만 한다.

        Args:
            scripts_dir: install.sh를 생성할 대상 디렉토리 (번들의 scripts/)
        """
        # Linux/macOS 설치 스크립트. 아래 문자열이 그대로 install.sh 파일 내용이 된다.
        # (여러 문자열 리터럴이 인접해 하나로 이어붙여지는 Python 규칙을 사용 중)
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
        번들 README(README.txt)를 생성한다.

        현장 담당자가 번들만 보고도 따라 할 수 있도록 무결성 검증 명령, 설치 명령,
        번들 폴더 구조 설명, 생성 시각을 담은 안내문을 번들 최상위에 만든다.

        Args:
            bundle_dir: README.txt를 둘 번들 최상위 디렉토리
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
