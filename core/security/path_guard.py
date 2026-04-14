"""
경로 보호 — PathGuard.

Ch.9.2 사양서 기반. 파일 경로의 보안을 검증한다.
다음 위협을 탐지하고 차단한다:
  - 경로 순회 공격 (../../etc/passwd)
  - 보호 경로 접근 (.env, .ssh, credentials 등)
  - UNC 경로 (\\\\server\\share — Windows 네트워크 경로)
  - 심볼릭 링크를 통한 우회
  - null 바이트 인젝션

핵심 원칙: 의심스러운 경로는 모두 차단한다 (fail-closed).
"""

from __future__ import annotations

import fnmatch
import logging
import os
from pathlib import Path

logger = logging.getLogger("nexus.security")


class PathGuard:
    """
    파일 경로 보안 검증기.

    모든 파일 접근 도구(Read, Write, Edit, Glob 등)가
    실행 전에 이 검증기를 통해 경로를 확인해야 한다.
    """

    # 절대 접근할 수 없는 보호 경로 패턴
    # glob 패턴으로 매칭한다
    PROTECTED_PATHS: list[str] = [
        # 시스템 보안 파일
        "/etc/shadow",
        "/etc/passwd",
        "/etc/sudoers",
        # SSH 키 및 설정
        "**/.ssh/*",
        "**/.ssh",
        # GPG 키
        "**/.gnupg/*",
        "**/.gnupg",
        # 환경변수 / 시크릿 파일
        "**/.env",
        "**/.env.local",
        "**/.env.production",
        "**/.env.staging",
        "**/credentials.json",
        "**/secrets.yaml",
        "**/secrets.yml",
        "**/*.pem",
        "**/*.key",
        # Windows 보호 경로
        "**/SAM",
        "**/SYSTEM",
        "**/SECURITY",
    ]

    # 읽기 전용으로만 접근할 수 있는 경로
    READ_ONLY_PATHS: list[str] = [
        "/etc/*",
        "/usr/*",
        "/bin/*",
        "/sbin/*",
    ]

    def __init__(
        self,
        protected_paths: list[str] | None = None,
        read_only_paths: list[str] | None = None,
    ) -> None:
        """
        PathGuard를 초기화한다.

        Args:
            protected_paths: 추가 보호 경로 패턴 (기본 목록에 추가)
            read_only_paths: 추가 읽기 전용 경로 패턴
        """
        self._protected = list(self.PROTECTED_PATHS)
        if protected_paths:
            self._protected.extend(protected_paths)

        self._read_only = list(self.READ_ONLY_PATHS)
        if read_only_paths:
            self._read_only.extend(read_only_paths)

    def is_path_safe(self, path: str, cwd: str) -> tuple[bool, str]:
        """
        경로가 안전한지 종합 검증한다.

        모든 보안 검사를 순서대로 실행하고,
        첫 번째 실패에서 즉시 (False, 이유) 를 반환한다.
        모든 검사를 통과하면 (True, "") 를 반환한다.

        Args:
            path: 검증할 파일 경로
            cwd: 현재 작업 디렉토리 (상대 경로 해석 기준)

        Returns:
            (safe, reason): safe=True면 안전, False면 reason에 차단 이유
        """
        # 1. null 바이트 인젝션 검사
        if "\x00" in path:
            return False, "경로에 null 바이트가 포함되어 있습니다"

        # 2. UNC 경로 검사 (Windows 네트워크 경로)
        if self._check_unc_path(path):
            return False, "UNC 경로(네트워크 경로)는 허용되지 않습니다"

        # 3. 경로 순회 공격 검사
        if not self._check_traversal(path, cwd):
            return False, "경로 순회 공격이 감지되었습니다 (작업 디렉토리 밖으로 이동)"

        # 4. 보호 경로 검사
        if self._check_protected(path):
            return False, f"보호된 경로입니다: {path}"

        return True, ""

    def is_path_writable(self, path: str, cwd: str) -> tuple[bool, str]:
        """
        경로에 쓰기가 가능한지 확인한다.

        is_path_safe() 검사 + 읽기 전용 경로 검사를 수행한다.

        Args:
            path: 검증할 파일 경로
            cwd: 현재 작업 디렉토리

        Returns:
            (writable, reason): writable=True면 쓰기 가능
        """
        # 기본 안전성 검사
        safe, reason = self.is_path_safe(path, cwd)
        if not safe:
            return False, reason

        # 읽기 전용 경로 검사
        if self._check_read_only(path):
            return False, f"읽기 전용 경로입니다: {path}"

        return True, ""

    def _check_traversal(self, path: str, cwd: str) -> bool:
        """
        경로 순회 공격을 감지한다.

        ../../ 패턴으로 작업 디렉토리 밖으로 벗어나는 시도를 차단한다.
        resolve()로 정규화한 후 작업 디렉토리 하위인지 확인한다.

        Returns:
            True면 안전 (순회 없음), False면 순회 감지
        """
        try:
            cwd_path = Path(cwd).resolve()
            # 절대 경로면 그대로, 상대 경로면 cwd 기준으로 해석
            if os.path.isabs(path):
                resolved = Path(path).resolve()
            else:
                resolved = (cwd_path / path).resolve()

            # 작업 디렉토리 하위인지 확인
            # 작업 디렉토리 자체도 허용
            try:
                resolved.relative_to(cwd_path)
                return True
            except ValueError:
                # 작업 디렉토리 밖 — 순회 감지
                return False
        except (OSError, ValueError):
            # 경로 해석 실패 — 안전하지 않은 것으로 간주
            return False

    def _check_protected(self, path: str) -> bool:
        """
        보호 경로 접근을 감지한다.

        PROTECTED_PATHS의 glob 패턴과 매칭되면 True를 반환한다.
        대소문자 무시 매칭을 사용한다 (Windows 호환).

        Returns:
            True면 보호 경로 (접근 차단), False면 비보호
        """
        # 경로를 정규화한다 (역슬래시 → 슬래시)
        normalized = path.replace("\\", "/")

        # 홈 디렉토리 확장
        if normalized.startswith("~"):
            normalized = os.path.expanduser(normalized)
            normalized = normalized.replace("\\", "/")

        for pattern in self._protected:
            # 홈 디렉토리 패턴 확장
            expanded_pattern = pattern.replace("\\", "/")
            if expanded_pattern.startswith("~"):
                expanded_pattern = os.path.expanduser(expanded_pattern)
                expanded_pattern = expanded_pattern.replace("\\", "/")

            # fnmatch로 패턴 매칭
            if fnmatch.fnmatch(normalized, expanded_pattern):
                return True
            # 파일 이름만으로도 매칭 시도 (경로 어디서든 매칭)
            basename = os.path.basename(normalized)
            pattern_basename = os.path.basename(expanded_pattern)
            # pattern_basename이 "*"이면 너무 관대하므로 제외
            if (
                pattern_basename
                and pattern_basename != "*"
                and fnmatch.fnmatch(basename, pattern_basename)
                and "**" in pattern
            ):
                return True

        return False

    def _check_read_only(self, path: str) -> bool:
        """
        읽기 전용 경로인지 확인한다.

        Returns:
            True면 읽기 전용 (쓰기 차단), False면 쓰기 허용
        """
        normalized = path.replace("\\", "/")
        for pattern in self._read_only:
            if fnmatch.fnmatch(normalized, pattern):
                return True
        return False

    def _check_unc_path(self, path: str) -> bool:
        """
        UNC 경로 (Windows 네트워크 경로)를 감지한다.

        \\\\server\\share 형태의 경로를 차단한다.
        에어갭 환경에서 네트워크 파일 접근을 방지한다.

        Returns:
            True면 UNC 경로 (차단), False면 비UNC
        """
        # \\ 로 시작하는 Windows UNC 경로
        if path.startswith("\\\\") or path.startswith("//"):
            return True
        return False
