"""
무결성 검증 모듈 — SHA256 해시 기반으로 배포 파일이 변조되지 않았는지 확인한다.

[이 파일이 하는 일]
에어갭(폐쇄망)에 Nexus를 배포할 때는 USB나 오프라인 매체로 파일 번들을
옮긴다. 이 과정에서 파일이 깨지거나(전송 오류) 누군가 몰래 바꿔치기하는
(변조) 일이 없었는지 반드시 확인해야 한다. 이 모듈은 각 파일의 SHA256
해시값(파일 내용을 대표하는 64자리 지문)을 계산하고, 미리 기록해 둔 값과
비교해서 "원본과 100% 동일한가?"를 판정한다.

[핵심 클래스]
  - IntegrityVerifier : 무결성 검증기. 아래 4개 메서드로 구성된다.
      * compute_hash(path)                 파일 하나의 SHA256 해시를 계산
      * verify_file(path, expected)        파일 하나를 예상 해시와 비교
      * verify_directory(dir, manifest)    디렉토리 전체를 매니페스트와 비교
      * generate_manifest(dir)             디렉토리 전체의 매니페스트를 생성

[전형적인 사용 흐름(3단계)]
  1. 번들을 만드는 쪽(빌드 서버)에서 generate_manifest()를 호출해
     {상대경로: 해시} 형태의 매니페스트를 만들어 함께 배포한다.
  2. 번들을 받는 쪽(배포 대상)에서 verify_directory()로 매니페스트와
     실제 파일들을 대조해 변조/누락을 잡아낸다.
  3. 런타임에 특정 파일 하나만 확인하고 싶으면 verify_file()을 쓴다.

[의존성]
  외부 패키지 없이 파이썬 표준 라이브러리만 사용한다(hashlib, pathlib,
  logging). 에어갭 환경에서 추가 설치 없이 그대로 동작해야 하기 때문이다.

작성자: 이현수 / 작성일: 2026-07-05
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

# 이 모듈 전용 로거. 프로젝트 규칙상 "nexus.{모듈경로}" 네임스페이스를 쓴다.
# 검증 성공/실패, 파일 누락 등의 사건을 이 로거로 남겨 배포 후 추적이 가능하다.
logger = logging.getLogger("nexus.deployment.integrity")

# SHA256 해시를 계산할 때 파일을 한 번에 다 읽지 않고 이 크기(8KB)씩 잘라
# 읽는다. 모델 weight 파일은 수 GB에 달하므로, 통째로 메모리에 올리면
# OOM(메모리 부족)이 날 수 있다. 8KB씩 스트리밍하면 파일 크기와 무관하게
# 일정한(작은) 메모리만 사용한다.
_HASH_BLOCK_SIZE = 8192


class IntegrityVerifier:
    """
    SHA256 기반 파일 무결성 검증기.

    에어갭 배포 번들의 모든 파일이 원본과 바이트 단위로 동일한지 확인한다.
    파일 타입을 가리지 않으므로 모델 weight(.safetensors), 파이썬 코드(.py),
    YAML 설정(.yaml) 등 무엇이든 검증 대상이 될 수 있다.

    상태(state)를 가지지 않는 순수 유틸리티 클래스라, 인스턴스를 하나 만들어
    두고 여러 파일에 재사용해도 안전하다. (예: verifier = IntegrityVerifier())
    """

    def compute_hash(self, path: str) -> str:
        """
        파일 하나의 SHA256 해시(64자리 16진수 지문)를 계산해 돌려준다.

        [왜 스트리밍으로 읽는가]
        f.read()로 파일 전체를 한 번에 읽으면 수 GB짜리 모델 파일에서
        메모리가 터질 수 있다. 그래서 _HASH_BLOCK_SIZE(8KB)만큼씩 반복해서
        읽고, 읽은 조각을 그때그때 sha256에 누적(update)한다. 이렇게 하면
        파일이 아무리 커도 메모리 사용량은 8KB 수준으로 일정하다.

        Args:
            path: 해시를 계산할 파일의 절대 경로(문자열)

        Returns:
            SHA256 해시 문자열 (소문자 16진수, 정확히 64자)

        Raises:
            FileNotFoundError: 경로에 파일이 존재하지 않을 때
            ValueError: 경로가 파일이 아니라 디렉토리일 때
            PermissionError: 파일을 읽을 권한이 없을 때 (open 단계에서 발생)
        """
        file_path = Path(path)
        # 방어적 사전 검사: 존재하지 않거나 디렉토리인 경로를 먼저 걸러내
        # 호출한 쪽이 원인을 바로 알 수 있는 명확한 예외를 던진다.
        if not file_path.exists():
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {path}")
        if not file_path.is_file():
            raise ValueError(f"디렉토리가 아닌 파일 경로를 지정하세요: {path}")

        # SHA256 계산기를 만든 뒤, 파일을 바이너리("rb")로 열어 블록 단위로
        # 흘려 넣는다. 텍스트 모드가 아니라 바이너리 모드여야 OS/개행 차이
        # 없이 원본 바이트 그대로 해시된다.
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            while True:
                # 8KB씩 읽는다. 파일 끝에 도달하면 read()는 빈 bytes(b"")를
                # 반환하고, 그때 루프를 빠져나간다.
                block = f.read(_HASH_BLOCK_SIZE)
                if not block:
                    break
                sha256.update(block)

        # 누적된 해시를 사람이 비교하기 쉬운 16진수 문자열로 변환해 반환.
        return sha256.hexdigest()

    def verify_file(self, path: str, expected_hash: str) -> bool:
        """
        파일 하나의 실제 해시를 "예상 해시"와 비교해 무결성을 판정한다.

        compute_hash()와 달리 이 메서드는 예외를 밖으로 던지지 않는다.
        파일이 없거나 권한이 없어 해시를 못 구하는 상황도 "검증 실패(False)"
        로 취급한다. 배포 검증 루프가 파일 하나 때문에 중단되지 않고 끝까지
        돌아 전체 결과를 모을 수 있도록 하기 위해서다(fail-closed).

        Args:
            path: 검증할 파일의 절대 경로
            expected_hash: 기대하는 SHA256 해시(16진수). 대문자로 들어와도
                           되도록 아래에서 .lower()로 소문자로 맞춰 비교한다.

        Returns:
            해시가 일치하면 True(무결성 통과), 불일치하거나 파일 접근 실패면 False
        """
        try:
            # 실제 파일의 해시를 계산하고, 예상 해시와 대소문자 차이를 없앤
            # 상태로 정확히 일치하는지 비교한다.
            actual_hash = self.compute_hash(path)
            is_valid = actual_hash == expected_hash.lower()

            # 실패/성공을 로그로 남긴다. 실패는 WARNING으로 예상/실제 해시를
            # 함께 찍어 원인 추적을 돕고, 성공은 잡음을 줄이려 DEBUG로 남긴다.
            if not is_valid:
                logger.warning(
                    f"무결성 검증 실패: {path}\n  예상: {expected_hash}\n  실제: {actual_hash}"
                )
            else:
                logger.debug(f"무결성 검증 통과: {path}")

            return is_valid

        except (FileNotFoundError, PermissionError) as e:
            # 해시 계산 단계에서 나는 예상 가능한 예외만 좁게 잡아 실패로
            # 처리한다(anti-patterns의 bare except 금지 원칙). 그 외 예외는
            # 진짜 버그일 수 있으니 삼키지 않고 그대로 위로 전파시킨다.
            logger.error(f"무결성 검증 에러: {path} — {e}")
            return False

    def verify_directory(self, dir_path: str, manifest: dict[str, str]) -> tuple[bool, list[str]]:
        """
        매니페스트에 적힌 모든 파일을 실제 디렉토리와 대조해 통째로 검증한다.

        배포 번들 전체를 한 번에 확인할 때 쓰는 핵심 메서드다. 매니페스트의
        각 항목(상대경로 → 예상 해시)을 하나씩 돌면서 두 가지를 확인한다.
          (1) 그 경로에 파일이 실제로 존재하는가?  → 없으면 "누락"
          (2) 존재한다면 해시가 예상값과 같은가?    → 다르면 "불일치"
        하나라도 걸리면 실패 목록에 사유와 함께 담고, 도중에 멈추지 않고
        끝까지 검사해 전체 이상 목록을 한꺼번에 돌려준다.

        Args:
            dir_path: 검증할 디렉토리의 절대 경로(매니페스트 상대경로의 기준점)
            manifest: {상대경로: SHA256 해시} 딕셔너리. generate_manifest()가
                      만든 것을 그대로 넣으면 된다.

        Returns:
            (전체 통과 여부, 실패 사유 문자열 목록)의 튜플.
            모두 통과하면 (True, []), 문제가 있으면 (False, ["누락: ...",
            "불일치: ..."]) 형태. 실패 목록에는 누락 파일과 해시 불일치 파일이
            섞여 들어간다.
        """
        base = Path(dir_path)
        # 기준 디렉토리 자체가 없으면 개별 파일을 볼 필요도 없이 즉시 실패.
        if not base.exists():
            logger.error(f"디렉토리를 찾을 수 없습니다: {dir_path}")
            return False, [f"디렉토리 없음: {dir_path}"]

        # 검증에 실패한 항목들을 사유 문자열로 모아 둘 리스트.
        failures: list[str] = []

        # 매니페스트에 기록된 파일들을 하나씩 검증한다.
        for relative_path, expected_hash in manifest.items():
            # 기준 디렉토리에 상대경로를 붙여 실제 파일의 전체 경로를 만든다.
            file_path = base / relative_path

            if not file_path.exists():
                # 매니페스트에는 있는데 실제로는 없는 파일 = 전송 누락/삭제.
                failures.append(f"누락: {relative_path}")
                logger.warning(f"파일 누락: {file_path}")
                # 없는 파일은 해시 비교가 불가능하므로 다음 항목으로 넘어간다.
                continue

            # 파일은 있으니 해시를 비교한다. verify_file은 실패해도 예외 대신
            # False를 주므로, False면 "불일치"로 기록한다.
            if not self.verify_file(str(file_path), expected_hash):
                failures.append(f"불일치: {relative_path}")

        # 실패 항목이 하나도 없어야 전체 통과다.
        is_valid = len(failures) == 0

        # 최종 결과를 요약 로그로 남긴다. 통과는 INFO, 실패는 "이상 개수/전체
        # 개수"를 담아 ERROR로 남겨 운영자가 심각도를 바로 파악하게 한다.
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
        디렉토리 전체를 재귀 탐색해 {상대경로: SHA256 해시} 매니페스트를 만든다.

        verify_directory()가 나중에 대조할 "정답표"를 만드는 메서드다. 보통
        번들을 배포하기 직전에 빌드 서버에서 한 번 호출해, 그 결과를 번들과
        함께 내보낸다.

        [설계상 눈여겨볼 점]
          - sorted(...)로 파일을 정렬해 순회하므로, 같은 디렉토리에 대해 항상
            동일한 순서로 매니페스트가 만들어진다(결과의 재현성/안정성).
          - .git, __pycache__ 같은 빌드 부산물 디렉토리는 배포 대상이 아니고
            내용이 매번 바뀌므로 exclude_dirs로 통째로 제외한다.
          - 개별 파일 해시 실패(권한/OS 오류)는 전체를 중단시키지 않고 경고만
            남기고 건너뛴다. 한 파일 때문에 매니페스트 생성이 통째로 깨지는
            것을 막기 위해서다.

        Args:
            dir_path: 매니페스트를 생성할 디렉토리의 절대 경로

        Returns:
            {상대경로(POSIX 형식): SHA256 해시} 딕셔너리

        Raises:
            FileNotFoundError: 대상 디렉토리가 존재하지 않을 때
        """
        base = Path(dir_path)
        if not base.exists():
            raise FileNotFoundError(f"디렉토리를 찾을 수 없습니다: {dir_path}")

        # 매니페스트에서 통째로 제외할 디렉토리 이름들. 버전 관리 메타데이터나
        # 캐시, 의존성 폴더처럼 배포 대상이 아니고 내용이 유동적인 것들이다.
        exclude_dirs = {".git", "__pycache__", ".mypy_cache", ".ruff_cache", "node_modules"}

        manifest: dict[str, str] = {}

        # rglob("*")로 하위 모든 항목을 재귀 탐색한다. sorted로 감싸 순서를
        # 고정하면 매번 동일한 매니페스트가 나와 diff/재현이 쉬워진다.
        for file_path in sorted(base.rglob("*")):
            # 디렉토리 자체는 해시 대상이 아니므로 건너뛴다(파일만 처리).
            if not file_path.is_file():
                continue

            # 경로를 base 기준 상대 경로로 쪼갠 뒤, 그 경로 구성요소 중
            # 하나라도 제외 목록에 있으면(예: .../__pycache__/foo.pyc) 건너뛴다.
            parts = file_path.relative_to(base).parts
            if any(part in exclude_dirs for part in parts):
                continue

            # 딕셔너리 키로 쓸 상대경로. OS에 따라 구분자가 \ 또는 /로 갈리지
            # 않도록 as_posix()로 항상 슬래시(/) 형식으로 통일한다. 이래야
            # 윈도우에서 만든 매니페스트를 리눅스에서 검증해도 키가 일치한다.
            relative = file_path.relative_to(base).as_posix()

            try:
                # 파일 하나의 해시를 계산해 매니페스트에 등록한다.
                file_hash = self.compute_hash(str(file_path))
                manifest[relative] = file_hash
            except (PermissionError, OSError) as e:
                # 읽기 권한 부족이나 OS 레벨 오류가 난 파일은 경고만 남기고
                # 건너뛴다. 나머지 파일들의 매니페스트는 정상적으로 이어 만든다.
                logger.warning(f"해시 계산 실패 (건너뜀): {file_path} — {e}")

        logger.info(f"매니페스트 생성 완료: {dir_path} ({len(manifest)}개 파일)")
        return manifest
