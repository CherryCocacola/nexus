"""
명령어 필터 — CommandFilter.

[이 파일이 하는 일]
Bash 도구가 실제로 셸 명령을 실행하기 "직전"에, 그 명령 문자열이 안전한지
검사하는 보안 게이트다. Ch.9.1 사양서를 기반으로 하며, 폐쇄망(에어갭)
환경에 맞춘 allowlist(허용 목록) 기반 필터링을 수행한다.

[왜 필요한가]
로컬 LLM(Qwen/ExaOne 등)이 자율적으로 셸 명령을 만들어 낼 수 있는데,
모델이 실수로든 악의적 프롬프트로든 `rm -rf /` 같은 파괴적 명령이나
`curl`/`pip install` 같은 외부 네트워크·설치 명령을 시도할 수 있다.
이 필터가 그런 명령을 실행 전에 걸러 내는 1차 방어선 역할을 한다.

[검사 순서 — check_command()의 흐름]
  1. 위험 패턴(DANGEROUS_PATTERNS) 매칭 → 걸리면 즉시 차단(safe=False)
  2. 안전 명령어(SAFE_COMMANDS) 목록에 있으면 → 자동 허용(safe=True)
  3. 위 둘 다 아닌 알 수 없는 명령어 → 차단(unknown) 후 상위 계층에서
     사용자에게 확인(ASK)하도록 넘긴다

[핵심 원칙 — fail-closed]
"명시적으로 허용된 것만 통과"가 기본 방침이다. 허용 목록에 없으면
자동 통과시키지 않고 사용자 확인이 필요한 상태로 되돌린다.
잘 모르는 명령을 일단 막는 쪽이 뚫는 쪽보다 안전하기 때문이다.

[주요 구성 요소]
  - CommandFilter: 이 파일의 유일한 클래스. 필터 규칙과 검사 로직을 담는다.
  - SAFE_COMMANDS / DANGEROUS_PATTERNS: 규칙 데이터(클래스 상수)
  - check_command(): 외부에서 호출하는 주 진입점

작성자: 이현수 / 작성일: 2026-07-05
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger("nexus.security")


class CommandFilter:
    """
    Bash 명령어 보안 필터.

    모든 Bash 도구 실행 전에 이 필터를 통해 명령어를 검증해야 한다.
    allowlist(허용 목록) 기반으로 안전한 명령어만 자동 허용하고,
    위험 패턴은 즉시 차단한다.

    [사용법 요약]
      filter = CommandFilter()                 # 기본(배포용) 설정으로 생성
      safe, severity, reason = filter.check_command("ls -al")
      # safe=True  → 그대로 실행
      # safe=False → severity/reason을 보고 차단하거나 사용자에게 확인

    [규칙의 두 축]
      - SAFE_COMMANDS: 자동 허용할 안전 명령어 이름 목록(화이트리스트)
      - DANGEROUS_PATTERNS: 즉시 차단할 위험 정규식 패턴 목록(블랙리스트)
    위험 패턴이 안전 목록보다 항상 우선한다(먼저 검사해 즉시 차단).
    """

    # 자동 허용되는 안전한 명령어 목록.
    # 여기에 "명령어 이름"이 들어 있으면 사용자 확인 없이 바로 실행된다.
    # 모두 읽기 전용이거나 부작용이 작은, 개발에 흔히 쓰는 명령들이다.
    # 주의: 이 목록은 "명령어 이름"만 본다(인자는 위험 패턴 쪽에서 검사).
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

    # 위험 패턴 목록 — 각 항목은 (정규식 패턴, 심각도, 사람이 읽을 이유) 3-튜플.
    # 명령어 전체 문자열을 이 정규식들로 훑어(search) 하나라도 걸리면 즉시 차단한다.
    # 심각도는 3단계로 나눈다:
    #   - critical: 시스템/디스크를 파괴할 수 있는 명령(복구 불가 위험)
    #   - high    : 에어갭 위반(외부 통신·패키지 설치 등) 보안 위험
    #   - medium  : 시스템 상태를 바꾸는 관리성 명령(권한/서비스/전원 등)
    # 순서 자체는 심각도별로 묶어 두었을 뿐, 검사 시에는 위에서부터 순차로 본다.
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
        # pip / pip3 / pip3.11 등 버전 접미사까지 포괄한다(에어갭 우회 방지).
        # (`python -m pip install`은 "pip install" 부분문자열이 이미 걸린다.)
        (r"pip[0-9.]*\s+install", "high", "에어갭: 런타임 패키지 설치"),
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
        block_package_install: bool = True,
    ) -> None:
        """
        CommandFilter를 초기화한다.

        Args:
            safe_commands: 안전 명령어 목록을 덮어쓴다 (None이면 기본 목록 사용)
            extra_dangerous_patterns: 추가 위험 패턴 (기본 목록에 추가)
            block_package_install: pip/npm/apt/yum/brew install 계열을 위험
                패턴으로 취급할지 여부(에어갭 설치차단 게이팅).
                - True(기본, 배포용): install 계열을 그대로 위험 패턴에 포함 →
                  런타임 패키지 설치를 차단한다(에어갭 준수).
                - False(개발용): install 계열 위험 패턴만 목록에서 제외한다 →
                  개발 중에는 라이브러리 설치를 허용한다(사용자 방침: 에어갭은
                  배포물에만 적용, 개발 중엔 설치해 진행).
                기본값을 True로 둔 이유: 이 인자를 주지 않는 기존 호출부(및 기존
                단위 테스트)의 동작을 100% 그대로 유지하기 위함이다(무회귀).
        """
        self._safe_commands: set[str] = set(
            safe_commands if safe_commands is not None else self.SAFE_COMMANDS
        )
        # 위험 패턴 기본 목록을 복사한다(원본 클래스 상수는 건드리지 않는다).
        self._dangerous_patterns = list(self.DANGEROUS_PATTERNS)
        # 개발 모드(block_package_install=False)에서는 "…패키지 설치" 사유가 붙은
        # install 계열 위험 패턴만 골라 제외한다. 사유 문자열로 식별하는 이유:
        # pip/npm/apt/yum/brew 5종이 모두 동일하게 "…패키지 설치" 사유를 쓰므로,
        # 패턴 정규식을 하드코딩하지 않고도 한 번에 정확히 걸러낼 수 있다.
        if not block_package_install:
            self._dangerous_patterns = [
                (pattern, severity, reason)
                for (pattern, severity, reason) in self._dangerous_patterns
                if "패키지 설치" not in reason
            ]
        if extra_dangerous_patterns:
            self._dangerous_patterns.extend(extra_dangerous_patterns)

        # 위험 패턴 정규식을 미리 컴파일해 둔다(성능 최적화).
        # check_command()는 명령마다 여러 번 호출되므로, 매번 re.compile 하지 않고
        # 생성 시점에 한 번만 컴파일해 재사용하면 반복 검사 비용을 크게 줄인다.
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
        # 빈 명령어(None·공백뿐)는 실행할 것이 없으므로 그냥 안전으로 통과시킨다.
        if not command or not command.strip():
            return True, "", ""

        # 앞뒤 공백을 제거한 실제 검사 대상 문자열.
        stripped = command.strip()

        # 1단계: 위험 패턴 매칭. 명령 전체 문자열에서 위험 정규식을 순서대로 검색해,
        # 하나라도 걸리면(search 성공) 더 볼 것 없이 즉시 차단(safe=False)한다.
        # 안전 목록 검사보다 먼저 하는 이유: "git"은 안전 명령이지만
        # "git ... | curl ..."처럼 위험 요소가 섞이면 반드시 먼저 잡아야 하기 때문.
        for compiled, severity, reason in self._compiled_patterns:
            if compiled.search(stripped):
                logger.warning(
                    "위험 명령어 감지: severity=%s, reason=%s, command=%s",
                    severity,
                    reason,
                    stripped[:100],
                )
                return False, severity, reason

        # 2단계: 실행될 각 기본 명령어(첫 토큰)를 추출한다.
        # 파이프(|)·AND(&&)·OR(||)·세미콜론(;)으로 이어진 복합 명령이면
        # 그 안의 명령들을 모두 뽑아 하나하나 검사할 수 있게 한다.
        base_commands = self._extract_base_commands(stripped)

        # 추출한 기본 명령어가 "전부" 안전 목록에 있어야 통과시킨다.
        # 하나라도 목록에 없으면 unknown으로 차단 → 상위 계층에서 사용자에게
        # 확인(ASK)을 받게 된다. 이것이 fail-closed 원칙의 실제 구현부다.
        for base_cmd in base_commands:
            if base_cmd not in self._safe_commands:
                return (
                    False,
                    "unknown",
                    f"알 수 없는 명령어: '{base_cmd}' (안전 목록에 없음)",
                )

        # 위험 패턴에도 안 걸리고 모든 기본 명령어가 안전 목록에 있음 → 최종 안전.
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
        # 파이프(|, ||)·AND(&&)·세미콜론(;)을 구분자로 삼아 명령을 조각낸다.
        # 주의: 이것은 "간단한" 분리라, 따옴표("...") 안에 든 구분자까지는
        # 구분하지 못한다. 정밀한 셸 파싱이 아니라 1차 스크리닝 용도임을 기억할 것.
        parts = re.split(r"\s*(?:\|{1,2}|&&|;)\s*", command)

        base_commands: list[str] = []
        for part in parts:
            part = part.strip()
            if not part:
                # 구분자 사이가 비어 있는 조각(예: "ls ;; wc")은 건너뛴다.
                continue

            # 환경변수 접두사(VAR=value cmd 형태)를 건너뛰고 진짜 명령어를 찾는다.
            # 예: "FOO=1 BAR=2 python x.py"에서 앞의 FOO=1, BAR=2는 명령이 아니라
            # 환경변수 지정이므로 넘기고, 실제 명령인 python부터 잡아야 한다.
            tokens = part.split()
            cmd_idx = 0
            for i, token in enumerate(tokens):
                # "="이 있고 옵션(-)으로 시작하지 않으면 환경변수 지정으로 보고 스킵.
                if "=" in token and not token.startswith("-"):
                    cmd_idx = i + 1
                else:
                    # 환경변수가 아닌 첫 토큰을 만나면 그게 명령어이므로 멈춘다.
                    break

            if cmd_idx < len(tokens):
                # 환경변수를 건너뛴 위치의 토큰이 실제 명령어다.
                cmd = tokens[cmd_idx]
                # 경로가 붙어 있으면 마지막 요소만 남긴다: /usr/bin/python → python.
                cmd = cmd.rsplit("/", 1)[-1]
                # 위 rsplit로도 안 떨어지는 "./name" 형태의 앞 "./"를 제거한다.
                if cmd.startswith("./"):
                    cmd = cmd[2:]
                base_commands.append(cmd)

        return base_commands

    def is_safe_command(self, command_name: str) -> bool:
        """
        주어진 "명령어 이름 하나"가 안전 목록에 있는지 단순 조회한다.

        check_command()와 달리 위험 패턴 검사나 명령어 분해를 하지 않고,
        순수하게 이름 하나가 화이트리스트에 속하는지만 O(1)로 확인한다.
        (예: is_safe_command("ls") → True)
        """
        return command_name in self._safe_commands

    def add_safe_command(self, command_name: str) -> None:
        """
        실행 중에 안전 명령어 목록에 이름을 하나 추가한다.

        이 인스턴스의 목록만 바꾸며(내부 set), 클래스 상수 SAFE_COMMANDS나
        다른 인스턴스에는 영향을 주지 않는다. set이라 중복 추가는 무해하다.
        """
        self._safe_commands.add(command_name)

    def get_safe_commands(self) -> list[str]:
        """
        현재 안전 명령어 목록을 정렬된 리스트로 반환한다.

        내부 저장은 순서가 없는 set이므로, 로그 출력·표시·테스트에서
        결과가 항상 일정하도록 이름순으로 정렬해 돌려준다.
        """
        return sorted(self._safe_commands)
