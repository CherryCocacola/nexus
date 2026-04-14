"""
명령어 필터 — CommandFilter.

Ch.9.1 사양서 기반. Bash 명령어의 보안을 검증한다.
에어갭 환경에 맞는 allowlist 기반 필터링을 수행한다.

검사 순서:
  1. 위험 패턴 매칭 (즉시 차단)
  2. 안전 명령어 목록 확인
  3. 알 수 없는 명령어 → ASK (사용자 확인 필요)

핵심 원칙: 허용 목록에 없는 명령어는 사용자에게 확인한다 (fail-closed).
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger("nexus.security")


class CommandFilter:
    """
    Bash 명령어 보안 필터.

    모든 Bash 도구 실행 전에 이 필터를 통해 명령어를 검증해야 한다.
    allowlist 기반으로 안전한 명령어만 자동 허용하고,
    위험 패턴은 즉시 차단한다.
    """

    # 자동 허용되는 안전한 명령어 목록
    # 이 목록의 명령어는 사용자 확인 없이 실행된다
    SAFE_COMMANDS: list[str] = [
        # 파일 탐색/읽기
        "ls",
        "cat",
        "head",
        "tail",
        "grep",
        "find",
        "wc",
        "echo",
        "pwd",
        "whoami",
        "date",
        "file",
        "stat",
        "tree",
        "less",
        "more",
        "sort",
        "uniq",
        "diff",
        "basename",
        "dirname",
        "realpath",
        "readlink",
        # 개발 도구
        "git",
        "python",
        "python3",
        "pytest",
        "ruff",
        "mypy",
        "pip",
        "pip3",
        "black",
        "isort",
        "flake8",
        # 텍스트 처리
        "awk",
        "sed",
        "tr",
        "cut",
        "paste",
        "tee",
        "xargs",
        "printf",
        # 시스템 정보 (읽기 전용)
        "uname",
        "hostname",
        "df",
        "du",
        "free",
        "uptime",
        "env",
        "printenv",
        "which",
        "type",
        "id",
        # 디렉토리 이동
        "cd",
        "pushd",
        "popd",
        # 프로세스 조회 (읽기 전용)
        "ps",
        "top",
        "htop",
    ]

    # 위험 패턴 목록 — (패턴, 심각도, 이유)
    # 이 패턴이 매칭되면 즉시 차단한다
    DANGEROUS_PATTERNS: list[tuple[str, str, str]] = [
        # ── critical: 시스템 파괴 가능 ──
        (r"rm\s+-rf\s+/", "critical", "루트 파일시스템 삭제 시도"),
        (r"rm\s+-rf\s+~", "critical", "홈 디렉토리 삭제 시도"),
        (r"rm\s+-rf\s+\*", "critical", "전체 삭제 시도"),
        (r":()\s*\{\s*:\|:&\s*\}\s*;?\s*:", "critical", "포크 폭탄"),
        (r"dd\s+if=/dev/zero", "critical", "디스크 와이프"),
        (r"dd\s+if=/dev/random", "critical", "디스크 와이프"),
        (r"mkfs\.", "critical", "파일시스템 포맷"),
        (r"fdisk", "critical", "파티션 조작"),
        (r">\s*/dev/sd[a-z]", "critical", "블록 디바이스 직접 쓰기"),
        (r"format\s+[a-zA-Z]:", "critical", "Windows 드라이브 포맷"),
        # ── high: 보안 위험 ──
        (r"\bcurl\b", "high", "에어갭: 외부 네트워크 요청 (curl)"),
        (r"\bwget\b", "high", "에어갭: 외부 네트워크 요청 (wget)"),
        (r"\bssh\b", "high", "에어갭: SSH 연결 시도"),
        (r"\bscp\b", "high", "에어갭: SCP 파일 전송 시도"),
        (r"\brsync\b", "high", "에어갭: rsync 전송 시도"),
        (r"\bnc\b", "high", "에어갭: netcat 연결 시도"),
        (r"\bncat\b", "high", "에어갭: ncat 연결 시도"),
        (r"\btelnet\b", "high", "에어갭: telnet 연결 시도"),
        (r"\bftp\b", "high", "에어갭: FTP 연결 시도"),
        (r"pip\s+install", "high", "에어갭: 런타임 패키지 설치"),
        (r"npm\s+install", "high", "에어갭: 런타임 패키지 설치"),
        (r"apt(-get)?\s+install", "high", "에어갭: 시스템 패키지 설치"),
        (r"yum\s+install", "high", "에어갭: 시스템 패키지 설치"),
        (r"brew\s+install", "high", "에어갭: 시스템 패키지 설치"),
        # ── medium: 시스템 변경 ──
        (r"chmod\s+777", "medium", "과도한 권한 부여"),
        (r"chmod\s+-R", "medium", "재귀적 권한 변경"),
        (r"chown\s+-R", "medium", "재귀적 소유자 변경"),
        (r"sudo\s+", "medium", "관리자 권한 실행"),
        (r"su\s+", "medium", "사용자 전환"),
        (r"systemctl", "medium", "시스템 서비스 조작"),
        (r"service\s+", "medium", "시스템 서비스 조작"),
        (r"crontab", "medium", "크론 작업 조작"),
        (r"iptables", "medium", "방화벽 규칙 조작"),
        (r"kill\s+-9", "medium", "프로세스 강제 종료"),
        (r"pkill", "medium", "프로세스 종료"),
        (r"reboot", "medium", "시스템 재부팅"),
        (r"shutdown", "medium", "시스템 종료"),
        (r"eval\s+", "medium", "동적 코드 실행"),
    ]

    def __init__(
        self,
        safe_commands: list[str] | None = None,
        extra_dangerous_patterns: list[tuple[str, str, str]] | None = None,
    ) -> None:
        """
        CommandFilter를 초기화한다.

        Args:
            safe_commands: 안전 명령어 목록을 덮어쓴다 (None이면 기본 목록 사용)
            extra_dangerous_patterns: 추가 위험 패턴 (기본 목록에 추가)
        """
        self._safe_commands: set[str] = set(
            safe_commands if safe_commands is not None else self.SAFE_COMMANDS
        )
        self._dangerous_patterns = list(self.DANGEROUS_PATTERNS)
        if extra_dangerous_patterns:
            self._dangerous_patterns.extend(extra_dangerous_patterns)

        # 위험 패턴을 사전에 컴파일한다 (성능 최적화)
        self._compiled_patterns: list[tuple[re.Pattern, str, str]] = [
            (re.compile(pattern), severity, reason)
            for pattern, severity, reason in self._dangerous_patterns
        ]

    def check_command(self, command: str) -> tuple[bool, str, str]:
        """
        명령어의 안전성을 검증한다.

        검사 순서:
        1. 빈 명령어 → 안전
        2. 위험 패턴 매칭 → (False, severity, reason)
        3. 안전 명령어 목록 → (True, "", "")
        4. 알 수 없는 명령어 → (False, "unknown", 설명)

        Args:
            command: 검증할 bash 명령어 문자열

        Returns:
            (safe, severity, reason):
                safe=True면 안전 (자동 허용)
                safe=False면 차단 또는 확인 필요
                severity: "critical", "high", "medium", "unknown", ""
                reason: 차단/확인 이유 설명
        """
        # 빈 명령어는 안전 (아무것도 실행하지 않음)
        if not command or not command.strip():
            return True, "", ""

        stripped = command.strip()

        # 1단계: 위험 패턴 매칭 (전체 명령어 문자열에서 검색)
        for compiled, severity, reason in self._compiled_patterns:
            if compiled.search(stripped):
                logger.warning(
                    "위험 명령어 감지: severity=%s, reason=%s, command=%s",
                    severity,
                    reason,
                    stripped[:100],
                )
                return False, severity, reason

        # 2단계: 첫 번째 명령어 추출 (파이프, &&, || 등 고려)
        base_commands = self._extract_base_commands(stripped)

        # 모든 기본 명령어가 안전 목록에 있는지 확인
        for base_cmd in base_commands:
            if base_cmd not in self._safe_commands:
                return (
                    False,
                    "unknown",
                    f"알 수 없는 명령어: '{base_cmd}' (안전 목록에 없음)",
                )

        # 모든 검사를 통과 — 안전
        return True, "", ""

    def _extract_base_commands(self, command: str) -> list[str]:
        """
        명령어 문자열에서 기본 명령어(첫 번째 토큰)를 추출한다.

        파이프(|), AND(&&), OR(||), 세미콜론(;) 으로 연결된
        여러 명령어를 모두 추출한다.

        Args:
            command: 전체 명령어 문자열

        Returns:
            기본 명령어 이름 목록 (예: ["ls", "grep", "wc"])
        """
        # 파이프, AND, OR, 세미콜론으로 분리
        # 간단한 분리 — 따옴표 안의 구분자는 고려하지 않는다
        parts = re.split(r"\s*(?:\|{1,2}|&&|;)\s*", command)

        base_commands: list[str] = []
        for part in parts:
            part = part.strip()
            if not part:
                continue

            # 환경변수 접두사 건너뛰기 (VAR=value command)
            tokens = part.split()
            cmd_idx = 0
            for i, token in enumerate(tokens):
                if "=" in token and not token.startswith("-"):
                    cmd_idx = i + 1
                else:
                    break

            if cmd_idx < len(tokens):
                # 명령어 이름만 추출 (경로 제거)
                cmd = tokens[cmd_idx]
                # /usr/bin/python → python
                cmd = cmd.rsplit("/", 1)[-1]
                # ./ 접두사 제거
                if cmd.startswith("./"):
                    cmd = cmd[2:]
                base_commands.append(cmd)

        return base_commands

    def is_safe_command(self, command_name: str) -> bool:
        """주어진 명령어 이름이 안전 목록에 있는지 확인한다."""
        return command_name in self._safe_commands

    def add_safe_command(self, command_name: str) -> None:
        """안전 명령어 목록에 명령어를 추가한다."""
        self._safe_commands.add(command_name)

    def get_safe_commands(self) -> list[str]:
        """현재 안전 명령어 목록을 정렬하여 반환한다."""
        return sorted(self._safe_commands)
