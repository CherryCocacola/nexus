"""
무결성 검증 — SHA256 기반 파일 무결성 검증.

에어갭 환경에서 배포 번들의 무결성을 보장한다.
파일 단위 및 디렉토리 단위 검증을 지원한다.

사용 시나리오:
  1. 번들 생성 시: generate_manifest()로 SHA256 매니페스트를 생성한다
  2. 배포 시: verify_manifest()로 파일 변조 여부를 확인한다
  3. 런타임: verify_file()로 특정 파일의 무결성을 확인한다

의존성: 표준 라이브러리만 사용 (hashlib, pathlib)
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

logger = logging.getLogger("nexus.deployment.integrity")

# SHA256 해시 계산 시 파일을 읽는 블록 크기 (8KB)
# 대용량 모델 파일(수 GB)도 메모리 부담 없이 처리한다
_HASH_BLOCK_SIZE = 8192


class IntegrityVerifier:
    """
    SHA256 기반 파일 무결성 검증기.

    에어갭 배포 번들의 모든 파일이 원본과 동일한지 확인한다.
    모델 파일(.safetensors), 코드(.py), 설정(.yaml) 등
    모든 파일 타입을 검증할 수 있다.
    """

    def compute_hash(self, path: str) -> str:
        """
        파일의 SHA256 해시를 계산한다.

        대용량 파일(모델 weights 등)도 스트리밍 방식으로 읽어서
        메모리 사용량을 최소화한다.

        Args:
            path: 해시를 계산할 파일의 절대 경로

        Returns:
            SHA256 해시 문자열 (16진수, 64자)

        Raises:
            FileNotFoundError: 파일이 존재하지 않을 때
            PermissionError: 파일 읽기 권한이 없을 때
        """
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {path}")
        if not file_path.is_file():
            raise ValueError(f"디렉토리가 아닌 파일 경로를 지정하세요: {path}")

        # SHA256 해시를 스트리밍 방식으로 계산한다
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            while True:
                block = f.read(_HASH_BLOCK_SIZE)
                if not block:
                    break
                sha256.update(block)

        return sha256.hexdigest()

    def verify_file(self, path: str, expected_hash: str) -> bool:
        """
        파일의 SHA256 해시를 예상값과 비교한다.

        Args:
            path: 검증할 파일의 절대 경로
            expected_hash: 예상 SHA256 해시 (16진수)

        Returns:
            True이면 해시 일치 (무결성 통과), False이면 불일치
        """
        try:
            actual_hash = self.compute_hash(path)
            is_valid = actual_hash == expected_hash.lower()

            if not is_valid:
                logger.warning(
                    f"무결성 검증 실패: {path}\n  예상: {expected_hash}\n  실제: {actual_hash}"
                )
            else:
                logger.debug(f"무결성 검증 통과: {path}")

            return is_valid

        except (FileNotFoundError, PermissionError) as e:
            logger.error(f"무결성 검증 에러: {path} — {e}")
            return False

    def verify_directory(self, dir_path: str, manifest: dict[str, str]) -> tuple[bool, list[str]]:
        """
        디렉토리 내 모든 파일의 무결성을 매니페스트와 비교한다.

        매니페스트에 있는 모든 파일을 검증하고,
        실패한 파일 목록을 반환한다.

        Args:
            dir_path: 검증할 디렉토리의 절대 경로
            manifest: {상대경로: SHA256 해시} 딕셔너리

        Returns:
            (전체 통과 여부, 실패한 파일 경로 목록)
            실패 목록에는 해시 불일치 파일과 누락된 파일이 포함된다
        """
        base = Path(dir_path)
        if not base.exists():
            logger.error(f"디렉토리를 찾을 수 없습니다: {dir_path}")
            return False, [f"디렉토리 없음: {dir_path}"]

        failures: list[str] = []

        for relative_path, expected_hash in manifest.items():
            file_path = base / relative_path

            if not file_path.exists():
                # 매니페스트에 있지만 실제 파일이 없는 경우
                failures.append(f"누락: {relative_path}")
                logger.warning(f"파일 누락: {file_path}")
                continue

            if not self.verify_file(str(file_path), expected_hash):
                failures.append(f"불일치: {relative_path}")

        is_valid = len(failures) == 0

        if is_valid:
            logger.info(f"디렉토리 무결성 검증 통과: {dir_path} ({len(manifest)}개 파일)")
        else:
            logger.error(
                f"디렉토리 무결성 검증 실패: {dir_path} "
                f"({len(failures)}/{len(manifest)}개 파일 이상)"
            )

        return is_valid, failures

    def generate_manifest(self, dir_path: str) -> dict[str, str]:
        """
        디렉토리 내 모든 파일의 SHA256 매니페스트를 생성한다.

        재귀적으로 모든 파일을 탐색하여 {상대경로: SHA256 해시} 딕셔너리를 만든다.
        숨김 파일(.git, __pycache__ 등)은 제외한다.

        Args:
            dir_path: 매니페스트를 생성할 디렉토리의 절대 경로

        Returns:
            {상대경로: SHA256 해시} 딕셔너리
        """
        base = Path(dir_path)
        if not base.exists():
            raise FileNotFoundError(f"디렉토리를 찾을 수 없습니다: {dir_path}")

        # 제외할 디렉토리 패턴
        exclude_dirs = {".git", "__pycache__", ".mypy_cache", ".ruff_cache", "node_modules"}

        manifest: dict[str, str] = {}

        for file_path in sorted(base.rglob("*")):
            # 디렉토리는 건너뛴다
            if not file_path.is_file():
                continue

            # 제외 디렉토리 안의 파일은 건너뛴다
            parts = file_path.relative_to(base).parts
            if any(part in exclude_dirs for part in parts):
                continue

            # 상대 경로를 키로 사용한다 (POSIX 형식)
            relative = file_path.relative_to(base).as_posix()

            try:
                file_hash = self.compute_hash(str(file_path))
                manifest[relative] = file_hash
            except (PermissionError, OSError) as e:
                logger.warning(f"해시 계산 실패 (건너뜀): {file_path} — {e}")

        logger.info(f"매니페스트 생성 완료: {dir_path} ({len(manifest)}개 파일)")
        return manifest
