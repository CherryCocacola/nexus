"""
경로 보호 계층 — PathGuard.

이 파일은 Nexus의 모든 "파일 경로 관련 보안 검증"을 한곳에 모아 담당한다.
Read/Write/Edit/Glob 같은 파일 접근 도구가 실제로 디스크를 건드리기 전에,
사용자(또는 모델)가 넘긴 경로가 위험하지 않은지 이 클래스에게 먼저 물어본다.

막아야 하는 대표적인 위협:
  - 경로 순회 공격: ../../etc/passwd 처럼 작업 디렉토리 밖으로 빠져나가는 시도
  - 보호 경로 접근: .env, .ssh, credentials.json 등 비밀·자격증명 파일 접근
  - UNC 경로: \\\\server\\share 형태의 Windows 네트워크 경로 (에어갭 위반 소지)
  - 심볼릭 링크를 통한 우회 (resolve()로 실제 경로까지 펼쳐서 검사)
  - null 바이트 인젝션: 경로 문자열에 \\x00 을 끼워 넣어 검사를 속이는 기법

핵심 원칙은 fail-closed 이다. 즉 "확실히 안전하다고 판단되지 않으면 무조건 차단"한다.
검사 도중 예외가 나거나 판단이 애매하면 안전한 쪽(차단)으로 결정한다.

주요 구성:
  - PathGuard 클래스: 검증기 본체
  - is_path_safe():     읽기/쓰기 공통으로 통과해야 하는 기본 안전성 검사
  - is_path_writable(): 위 검사 + 읽기 전용 경로 여부까지 확인 (쓰기 도구용)
  - 내부 헬퍼 _check_* : 각 위협 유형별 세부 검사

사양서 근거: Ch.9.2 (경로 보안).
이 클래스는 보통 권한 파이프라인 Layer 2 (파일 경로 검증) 단계에서 호출된다.

작성자: 이현수 / 작성일: 2026-07-05
"""

from __future__ import annotations

import fnmatch
import logging
import os
from pathlib import Path

# 이 모듈 전용 로거. 보안 관련 로그는 "nexus.security" 네임스페이스로 모은다.
logger = logging.getLogger("nexus.security")


class PathGuard:
    """
    파일 경로 보안 검증기.

    모든 파일 접근 도구(Read, Write, Edit, Glob 등)는 실제 파일 작업을
    수행하기 전에 반드시 이 검증기를 거쳐 경로가 안전한지 확인해야 한다.
    즉 "검증 없이 파일에 접근하는 경로"가 존재하면 안 된다.

    보호 대상 패턴과 읽기 전용 패턴은 클래스 변수(PROTECTED_PATHS,
    READ_ONLY_PATHS)로 기본 제공되며, 생성자에서 추가 패턴을 덧붙일 수 있다.
    """

    # 절대 접근할 수 없는(읽기조차 막는) 보호 경로 패턴 목록.
    # 여기 적힌 항목들은 glob 패턴으로 매칭한다. (예: "**/.env" 는 어떤 깊이의
    # 디렉토리에 있든 이름이 .env 인 파일을 의미한다.)
    PROTECTED_PATHS: list[str] = [
        # 시스템 보안 파일 — 사용자·권한·sudo 설정이 담긴 민감 파일들
        "/etc/shadow",
        "/etc/passwd",
        "/etc/sudoers",
        # SSH 키 및 설정 — 원격 접속 자격증명이 들어있는 디렉토리 전체
        "**/.ssh/*",
        "**/.ssh",
        # GPG 키 — 암호화·서명용 개인키 디렉토리 전체
        "**/.gnupg/*",
        "**/.gnupg",
        # 환경변수 / 시크릿 파일 — API 키, 비밀번호 등이 흔히 저장되는 파일들
        "**/.env",
        "**/.env.local",
        "**/.env.production",
        "**/.env.staging",
        "**/credentials.json",
        "**/secrets.yaml",
        "**/secrets.yml",
        "**/*.pem",
        "**/*.key",
        # Windows 보호 경로 — 레지스트리 하이브(계정·시스템·보안 정보) 파일들
        "**/SAM",
        "**/SYSTEM",
        "**/SECURITY",
    ]

    # 읽기는 허용하되 쓰기는 막아야 하는 경로 패턴 목록.
    # 시스템 실행 파일·설정이 들어있는 디렉토리라 덮어쓰면 위험하다.
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

        기본 제공되는 PROTECTED_PATHS / READ_ONLY_PATHS 를 복사해서 인스턴스
        상태로 들고 있고, 인자로 추가 패턴이 넘어오면 그 뒤에 이어 붙인다.
        (클래스 변수를 직접 수정하지 않고 복사본을 쓰는 이유: 인스턴스마다
        다른 추가 규칙을 가질 수 있게 하고, 원본 기본 목록을 오염시키지 않기
        위해서다.)

        Args:
            protected_paths: 기본 보호 목록에 더할 추가 보호 경로 패턴
            read_only_paths: 기본 읽기 전용 목록에 더할 추가 패턴
        """
        # 기본 보호 목록을 복사(list())해서 인스턴스 전용 목록으로 만든다.
        self._protected = list(self.PROTECTED_PATHS)
        if protected_paths:
            self._protected.extend(protected_paths)

        # 읽기 전용 목록도 동일하게 복사 후 추가 패턴을 이어 붙인다.
        self._read_only = list(self.READ_ONLY_PATHS)
        if read_only_paths:
            self._read_only.extend(read_only_paths)

    def is_path_safe(self, path: str, cwd: str) -> tuple[bool, str]:
        """
        경로가 안전한지 종합 검증한다. (읽기·쓰기 공통 기본 검사)

        여러 보안 검사를 정해진 순서대로 실행하며, 하나라도 걸리면 즉시
        (False, 차단 이유) 를 돌려준다. 모든 검사를 통과하면 (True, "")
        를 반환한다. 이런 "먼저 걸리면 바로 실패" 방식은 fail-closed 원칙과
        맞물려, 가장 위험한 신호를 우선 잡아낸다.

        검사 순서(의도적으로 값싸고 확실한 것부터):
          1. null 바이트 인젝션
          2. UNC(네트워크) 경로
          3. 경로 순회(디렉토리 탈출)
          4. 보호 경로 접근

        Args:
            path: 검증할 파일 경로 (절대/상대 모두 가능)
            cwd: 현재 작업 디렉토리 — 상대 경로를 해석하는 기준점

        Returns:
            (safe, reason): safe=True면 안전, False면 reason에 차단 사유가 담김
        """
        # 1. null 바이트 인젝션 검사.
        #    경로 안에 \x00 이 있으면 하위 OS 호출에서 문자열이 잘려 엉뚱한
        #    파일을 가리킬 수 있으므로 즉시 차단한다.
        if "\x00" in path:
            return False, "경로에 null 바이트가 포함되어 있습니다"

        # 2. UNC 경로(Windows 네트워크 경로) 검사.
        #    에어갭 환경에서는 네트워크 공유 파일 접근 자체를 막는다.
        if self._check_unc_path(path):
            return False, "UNC 경로(네트워크 경로)는 허용되지 않습니다"

        # 3. 경로 순회 공격 검사.
        #    _check_traversal은 "안전하면 True"를 주므로, not 으로 뒤집어
        #    "순회가 감지되면(=False면)" 차단한다.
        if not self._check_traversal(path, cwd):
            return False, self._traversal_reason(path, cwd)

        # 4. 보호 경로 검사.
        #    .env, .ssh 등 민감 파일 패턴에 걸리면 차단한다.
        if self._check_protected(path):
            return False, f"보호된 경로입니다: {path}"

        # 모든 검사를 통과 — 안전한 경로로 판단한다.
        return True, ""

    def is_path_writable(self, path: str, cwd: str) -> tuple[bool, str]:
        """
        경로에 "쓰기"가 가능한지 확인한다. (Write/Edit 등 쓰기 도구용)

        먼저 is_path_safe()로 읽기·쓰기 공통 안전성을 확인하고, 그다음
        읽기 전용 경로(/usr, /bin 등)에 해당하는지 추가로 검사한다.
        따라서 이 메서드를 통과하려면 "안전하면서 동시에 읽기 전용이 아닌"
        경로여야 한다.

        Args:
            path: 검증할 파일 경로
            cwd: 현재 작업 디렉토리

        Returns:
            (writable, reason): writable=True면 쓰기 가능, False면 사유 포함
        """
        # 기본 안전성 검사부터 통과해야 한다. 여기서 걸리면 그 이유를 그대로
        # 전달한다.
        safe, reason = self.is_path_safe(path, cwd)
        if not safe:
            return False, reason

        # 안전하더라도 시스템 읽기 전용 경로라면 덮어쓰기를 막는다.
        if self._check_read_only(path):
            return False, f"읽기 전용 경로입니다: {path}"

        # 안전하고 읽기 전용도 아님 — 쓰기 허용.
        return True, ""

    def _traversal_reason(self, path: str, cwd: str) -> str:
        """차단 사유 문구를 만든다 — '진짜 순회'와 '단순히 밖'을 갈라 준다.

        왜 나누는가 (2026-08-19, 실측):
            둘을 같은 문구로 뭉뚱그리면 두 가지가 망가진다.

            ① 모델이 스스로 못 고친다. A.X-4.0 은 긴 절대경로를 재현하다 글자를
               틀리는데(실측: `nexus-b200` → `nexus-b2200`), 돌아오는 말이
               "경로 순회 공격이 감지되었습니다" 뿐이라 **경로가 틀렸다는 정보를
               받지 못한다.** 그래서 사과만 하고 같은 경로를 그대로 반복했다.
               작업 디렉토리를 알려 주면 스스로 교정할 재료가 생긴다.
            ② 감사 로그가 오염된다. 오타 수백 건과 진짜 침입 시도가 같은 문구로
               쌓이면, 나중에 진짜를 찾을 때 묻힌다.

        자동 교정(퍼지 매칭)은 일부러 넣지 않는다. `b2200` 을 `b200` 으로 알아서
        고쳐 읽으면 요청한 것과 **다른 파일**을 조용히 읽게 된다. 지금처럼 막고
        이유를 정확히 말해 주는 편이 낫다(fail-closed 유지).
        """
        # 경로 성분에 실제로 '..' 이 들어 있으면 상위로 빠져나가려는 시도다.
        # (문자열 검사만으로 충분하지 않은 우회는 _check_traversal 이 이미 걸렀다.)
        raw = str(path).replace("\\", "/")
        has_dotdot = ".." in raw.split("/")
        if has_dotdot:
            return "경로 순회 공격이 감지되었습니다 (작업 디렉토리 밖으로 이동)"

        # 순회 성분이 없는데 밖으로 나갔다 = 그냥 다른 위치를 가리킨 것.
        # 오타이거나, 애초에 접근 범위 밖이거나. 작업 디렉토리를 함께 알려 준다.
        try:
            shown_cwd = str(Path(cwd).resolve())
        except (OSError, ValueError):
            shown_cwd = cwd
        return (
            f"작업 디렉토리 밖의 경로입니다. 현재 작업 디렉토리는 '{shown_cwd}' 이며 "
            "그 하위 경로만 접근할 수 있습니다. 경로에 오타가 없는지 확인하거나, "
            "작업 디렉토리 기준 상대경로를 사용하십시오."
        )

    def _check_traversal(self, path: str, cwd: str) -> bool:
        """
        경로 순회(디렉토리 탈출) 공격을 감지한다.

        ../../ 같은 상위 이동으로 작업 디렉토리 바깥의 파일에 손대려는 시도를
        차단하기 위한 검사다. 문자열에서 ".." 를 찾는 방식은 심볼릭 링크나
        중복 슬래시로 쉽게 우회되므로, Path.resolve()로 경로를 "실제 최종
        위치"까지 정규화한 뒤, 그 결과가 작업 디렉토리 하위에 있는지를 본다.

        흐름:
          1. cwd를 resolve()로 정규화한다.
          2. path가 절대 경로면 그대로, 상대 경로면 cwd에 붙여서 resolve().
          3. 최종 경로가 cwd 하위인지 relative_to()로 확인한다.

        Returns:
            True면 안전(순회 없음), False면 순회 감지 또는 해석 실패
        """
        try:
            # 기준이 되는 작업 디렉토리를 실제 경로로 정규화한다.
            cwd_path = Path(cwd).resolve()
            # 절대 경로면 그대로 resolve, 상대 경로면 cwd 기준으로 이어 붙인다.
            if os.path.isabs(path):
                resolved = Path(path).resolve()
            else:
                resolved = (cwd_path / path).resolve()

            # 최종 경로가 작업 디렉토리 하위(또는 자기 자신)인지 확인한다.
            # relative_to()는 하위가 아니면 ValueError를 던진다.
            try:
                resolved.relative_to(cwd_path)
                # 작업 디렉토리 안쪽 — 안전.
                return True
            except ValueError:
                # 작업 디렉토리 밖으로 벗어남 — 순회로 판단해 차단.
                return False
        except (OSError, ValueError):
            # 경로 해석 자체가 실패(잘못된 경로 등)하면 fail-closed 원칙에 따라
            # 안전하지 않은 것으로 간주한다.
            return False

    def _check_protected(self, path: str) -> bool:
        """
        보호 경로(비밀·자격증명 파일 등) 접근 여부를 판단한다.

        self._protected에 담긴 glob 패턴들과 대조해, 하나라도 매칭되면
        True(=보호 대상, 접근 차단)를 반환한다. Windows/유닉스 경로 표기
        차이를 흡수하기 위해 역슬래시를 슬래시로 바꾸고, ~ 로 시작하는 홈
        디렉토리 표기는 실제 경로로 펼친 뒤 비교한다.

        매칭은 두 단계로 시도한다:
          (1) 전체 경로 대 패턴 매칭 (fnmatch)
          (2) 파일 이름만 떼어서 패턴의 파일 이름 부분과 매칭
              — 단, 패턴이 "**"를 포함(어느 깊이든 허용)하고 파일 이름
                 부분이 "*"처럼 지나치게 관대하지 않을 때만 적용한다.

        Returns:
            True면 보호 경로(접근 차단), False면 비보호(통과)
        """
        # 경로 구분자를 슬래시로 통일한다(역슬래시 → 슬래시).
        normalized = path.replace("\\", "/")

        # ~ 로 시작하면 홈 디렉토리 실제 경로로 확장한 뒤 다시 슬래시 통일.
        if normalized.startswith("~"):
            normalized = os.path.expanduser(normalized)
            normalized = normalized.replace("\\", "/")

        for pattern in self._protected:
            # 패턴 쪽도 동일하게 슬래시 통일 + 홈 디렉토리 확장을 해준다.
            expanded_pattern = pattern.replace("\\", "/")
            if expanded_pattern.startswith("~"):
                expanded_pattern = os.path.expanduser(expanded_pattern)
                expanded_pattern = expanded_pattern.replace("\\", "/")

            # (1) 전체 경로를 패턴과 직접 매칭한다.
            if fnmatch.fnmatch(normalized, expanded_pattern):
                return True
            # (2) 파일 이름만 떼어 매칭 시도 — 경로 어디에 있든 잡기 위해서다.
            basename = os.path.basename(normalized)
            pattern_basename = os.path.basename(expanded_pattern)
            # pattern_basename이 "*"이면 아무 파일이나 걸려 너무 관대해지므로
            # 제외하고, 원래 패턴이 "**"(임의 깊이)를 포함할 때만 적용한다.
            if (
                pattern_basename
                and pattern_basename != "*"
                and fnmatch.fnmatch(basename, pattern_basename)
                and "**" in pattern
            ):
                return True

        # 어떤 보호 패턴에도 걸리지 않음 — 비보호 경로.
        return False

    def _check_read_only(self, path: str) -> bool:
        """
        경로가 읽기 전용 대상인지 확인한다. (쓰기만 막고 읽기는 허용)

        self._read_only의 glob 패턴과 매칭되면 True를 반환한다. 시스템
        실행 파일·설정 디렉토리를 덮어쓰지 못하게 하는 용도다.

        Returns:
            True면 읽기 전용(쓰기 차단), False면 쓰기 허용
        """
        # 경로 구분자를 슬래시로 통일한 뒤 각 패턴과 비교한다.
        normalized = path.replace("\\", "/")
        for pattern in self._read_only:
            if fnmatch.fnmatch(normalized, pattern):
                return True
        return False

    def _check_unc_path(self, path: str) -> bool:
        """
        UNC 경로(Windows 네트워크 경로)를 감지한다.

        \\\\server\\share 형태처럼 백슬래시 두 개(또는 슬래시 두 개)로
        시작하는 네트워크 공유 경로를 차단하기 위한 검사다. 에어갭 환경에서는
        네트워크를 통한 파일 접근 자체가 정책 위반이므로 원천 차단한다.

        Returns:
            True면 UNC 경로(차단 대상), False면 일반 로컬 경로
        """
        # \\ (백슬래시 둘) 또는 // (슬래시 둘)로 시작하면 UNC로 간주한다.
        if path.startswith("\\\\") or path.startswith("//"):
            return True
        return False
