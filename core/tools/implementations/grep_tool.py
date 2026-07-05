"""
Grep 도구 — 파일 내용 정규표현식 검색.

이 파일은 Nexus의 24개 도구 중 하나인 "Grep" 도구를 구현한다. 모델(에이전트)이
파일 내용을 정규표현식(regex) 패턴으로 검색할 때 이 도구를 호출한다.
Claude Code의 Grep 도구와 동일한 역할을 로컬/에어갭 환경에서 재현한 것이다.

핵심 동작:
  - 검색 엔진으로 ripgrep(rg)을 우선 사용하고, 시스템에 rg가 없으면
    표준 grep으로 자동 폴백한다. (rg가 훨씬 빠르고 .gitignore를 존중한다.)
  - 3가지 출력 모드를 지원한다.
      * content            — 매칭된 라인을 라인 번호와 함께 출력
      * files_with_matches — 매칭이 있는 "파일 경로"만 출력 (기본값)
      * count              — 파일별 매칭 "개수"만 출력
  - 파일을 읽기만 하므로 완전한 읽기 전용 도구다. 따라서 behavior flag를
    is_read_only=True, is_concurrency_safe=True 로 완화해 병렬 실행을 허용한다.

주요 구성 요소:
  - GrepTool          : BaseTool을 상속한 도구 본체 (스키마·권한·실행 담당)
  - _build_rg_command : ripgrep용 명령어(list[str])를 조립하는 헬퍼 함수
  - _build_grep_command : grep 폴백용 명령어를 조립하는 헬퍼 함수

의존:
  - core.tools.base 의 BaseTool / ToolResult / PermissionResult 등 공용 계약을 사용한다.
  - 실제 검색은 subprocess로 외부 rg/grep 실행 파일을 호출해 수행한다.

작성자: 이현수 / 작성일: 2026-07-05
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import subprocess
from typing import Any

from core.tools.base import (
    BaseTool,
    PermissionBehavior,
    PermissionResult,
    ToolResult,
    ToolUseContext,
)

# 모듈 전용 로거. "nexus.tools.grep" 네임스페이스를 써서 로그를 분류한다.
logger = logging.getLogger("nexus.tools.grep")

# ── 결과 크기 관련 기본 상수 ──
# _DEFAULT_HEAD_LIMIT: head_limit 인자를 주지 않았을 때 반환할 최대 라인 수.
#   결과가 폭주해서 컨텍스트를 잡아먹는 것을 막는 안전장치다.
_DEFAULT_HEAD_LIMIT = 250
# _MAX_OUTPUT_SIZE: 최종 출력 문자열의 최대 길이(문자 수). 이 값을 넘으면 뒤를 자른다.
_MAX_OUTPUT_SIZE = 100_000  # 최대 출력 크기 (문자 수)


class GrepTool(BaseTool):
    """
    파일 내용 검색 도구 (BaseTool 구현체).

    모델이 정규표현식으로 코드/텍스트를 찾을 때 호출한다. 내부적으로는
    ripgrep(rg)을 우선 실행하고, 없으면 grep으로 폴백해 동일한 결과를 낸다.

    BaseTool이 요구하는 4단계 규약을 이 클래스에서 구현한다.
      1. Identity  — name / description / group (도구를 식별하는 메타데이터)
      2. Schema    — input_schema (모델에게 노출되는 입력 JSON Schema)
      3. Behavior  — is_read_only / is_concurrency_safe (실행 안전성 플래그)
      5. Lifecycle — validate_input → check_permissions → call (검증·권한·실행)
    (섹션 번호는 BaseTool의 표준 골격을 그대로 따른 것이라 4·6이 비어 있을 수 있다.)
    """

    # ═══ 1. Identity ═══
    # 도구를 식별하는 메타데이터. name은 모델이 호출할 때 쓰는 고유 이름이며
    # 레지스트리 등록·권한 규칙 매칭의 키가 되므로 절대 바꾸면 안 된다.

    @property
    def name(self) -> str:
        """도구의 고유 이름. 모델은 이 이름("Grep")으로 도구를 호출한다."""
        return "Grep"

    @property
    def description(self) -> str:
        """모델에게 보여줄 한 줄 설명. 프롬프트에 그대로 실린다."""
        return "Search file contents with regex."

    @property
    def group(self) -> str:
        """도구 분류(그룹). UI/문서에서 검색 계열 도구로 묶는 용도다."""
        return "search"

    # ═══ 2. Schema ═══
    # 모델에게 노출되는 입력 파라미터의 JSON Schema. 이 스키마가 곧
    # 모델이 채울 수 있는 인자 목록이다. required에 없는 값은 선택 인자.

    @property
    def input_schema(self) -> dict[str, Any]:
        """
        Grep 도구의 입력 스키마(JSON Schema) 를 반환한다.

        각 프로퍼티 의미:
          - pattern          : (필수) 검색할 정규표현식 패턴
          - path             : 검색 대상 디렉토리/파일 (없으면 현재 작업 디렉토리)
          - output_mode      : 출력 형식. content / files_with_matches / count
          - glob             : 파일 이름 필터 (예: *.py)
          - head_limit       : 최대 결과 라인 수 (기본 250)
          - case_insensitive : 대소문자 무시 여부 (기본 False)
        """
        return {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Regex pattern to search",
                },
                "path": {
                    "type": "string",
                    "description": "Directory or file to search",
                },
                "output_mode": {
                    "type": "string",
                    "enum": ["content", "files_with_matches", "count"],
                    "description": "Output mode (default: files_with_matches)",
                    "default": "files_with_matches",
                },
                "glob": {
                    "type": "string",
                    "description": "File glob filter (e.g. *.py)",
                },
                "head_limit": {
                    "type": "integer",
                    "description": "Max result lines (default: 250)",
                    "default": 250,
                    "minimum": 1,
                },
                "case_insensitive": {
                    "type": "boolean",
                    "description": "Case insensitive (default: false)",
                    "default": False,
                },
            },
            "required": ["pattern"],
        }

    # ═══ 3. Behavior Flags ═══
    # BaseTool의 기본값은 fail-closed(가장 제한적)이지만, Grep은 파일을 읽기만
    # 하고 시스템 상태를 바꾸지 않으므로 아래 두 플래그를 안전하게 완화한다.

    @property
    def is_read_only(self) -> bool:
        """읽기 전용 도구임을 표시한다. 쓰기/파괴 동작이 전혀 없다."""
        return True

    @property
    def is_concurrency_safe(self) -> bool:
        """동시 실행해도 안전하다. 여러 Grep을 병렬로 돌려도 부작용이 없다."""
        return True

    # ═══ 5. Lifecycle ═══
    # 도구 한 번의 실행은 validate_input(사전 검증) → check_permissions(권한)
    #   → call(실제 실행) 순서로 진행된다. 아래 세 메서드가 그 단계를 담당한다.

    def validate_input(self, input_data: dict[str, Any]) -> str | None:
        """
        실행 전에 입력을 가볍게 검증한다.

        pattern이 없거나 공백뿐이면 검색 자체가 무의미하므로 에러 메시지
        문자열을 반환한다. 문제가 없으면 None을 반환해 통과시킨다.
        (반환값이 문자열이면 호출자가 그것을 검증 실패 사유로 취급한다.)
        """
        pattern = input_data.get("pattern", "")
        if not pattern or not pattern.strip():
            return "pattern은 비어 있을 수 없습니다."
        return None

    async def check_permissions(
        self,
        input_data: dict[str, Any],
        context: ToolUseContext,
    ) -> PermissionResult:
        """
        이 도구의 권한 판정. 읽기 전용이라 위험이 없으므로 항상 ALLOW를 반환한다.

        (파일 경로 순회 차단 등 세부 검증은 상위 권한 파이프라인 Layer 2가
        별도로 담당한다. 여기서는 도구 수준에서 무조건 허용만 한다.)
        """
        return PermissionResult(behavior=PermissionBehavior.ALLOW)

    async def call(
        self,
        input_data: dict[str, Any],
        context: ToolUseContext,
    ) -> ToolResult:
        """
        정규표현식 패턴으로 파일 내용을 검색해 ToolResult로 돌려준다.

        이 메서드가 Grep 도구의 핵심이다. 실제 검색은 외부 실행 파일(rg/grep)을
        subprocess로 돌려서 수행하며, blocking 호출이므로 asyncio.to_thread로
        별도 스레드에 넘겨 이벤트 루프가 막히지 않게 한다.

        매개변수:
          - input_data : 스키마에 정의된 입력 값들이 담긴 딕셔너리
          - context    : 실행 컨텍스트. 여기서는 context.cwd(작업 디렉토리)를 사용

        반환:
          - ToolResult.success(...) : 검색 결과 텍스트 + 메타데이터
          - ToolResult.error(...)   : 타임아웃/도구 부재/실행 오류 등 실패

        처리 순서:
          1. ripgrep(rg) 사용 가능 여부 확인 (shutil.which)
          2. rg 또는 grep 용 검색 명령어(list[str]) 구성
          3. subprocess.run으로 실행 (60초 타임아웃)
          4. exit code 판정 후 head_limit·출력 크기에 맞게 결과 자르기
          5. ToolResult로 결과 반환
        """
        # ── 입력 값 추출 ── (없는 값은 스키마 기본값/컨텍스트로 보충)
        pattern = input_data["pattern"]
        search_path = input_data.get("path", context.cwd)
        output_mode = input_data.get("output_mode", "files_with_matches")
        glob_filter = input_data.get("glob")
        head_limit = input_data.get("head_limit", _DEFAULT_HEAD_LIMIT)
        case_insensitive = input_data.get("case_insensitive", False)

        # ── 검색 엔진 선택 ──
        # shutil.which로 PATH에서 rg 실행 파일을 찾는다. 있으면 rg를 쓰고,
        # 없으면(None) grep 폴백으로 전환한다. use_rg 값은 이후 결과 처리에서도
        # "rg를 썼는지"를 구분하는 플래그로 재사용된다.
        rg_path = shutil.which("rg")
        use_rg = rg_path is not None

        # 선택된 엔진에 맞는 명령어 배열을 조립한다. (문자열이 아니라 인자
        # 리스트로 만들어 shell=True 없이 안전하게 실행한다.)
        if use_rg:
            cmd = _build_rg_command(
                pattern,
                search_path,
                output_mode,
                glob_filter,
                head_limit,
                case_insensitive,
            )
        else:
            cmd = _build_grep_command(
                pattern,
                search_path,
                output_mode,
                glob_filter,
                case_insensitive,
            )

        # 디버깅용: 실제로 실행되는 명령어를 로그로 남긴다.
        logger.debug("Grep command: %s", " ".join(cmd))

        # ── 검색 실행 ──
        # subprocess.run은 동기(blocking) 함수이므로 asyncio.to_thread로 감싸
        # 스레드풀에서 돌린다. 그래야 검색이 도는 동안 이벤트 루프가 멈추지 않는다.
        # capture_output=True: stdout/stderr를 문자열로 수집, text=True: 바이트→문자열 디코드.
        try:
            result = await asyncio.to_thread(
                subprocess.run,
                cmd,
                capture_output=True,
                text=True,
                timeout=60,  # 무한정 매달리지 않도록 60초 상한을 둔다.
                cwd=context.cwd,  # 상대 경로 검색의 기준이 될 작업 디렉토리
            )
        except subprocess.TimeoutExpired:
            # 60초를 넘기면 결과를 포기하고 에러로 반환한다.
            return ToolResult.error("검색이 60초 타임아웃을 초과했습니다.")
        except FileNotFoundError:
            # rg/grep 실행 파일 자체가 없을 때. (which 통과 후에도 방어적으로 처리)
            return ToolResult.error(
                "검색 도구를 찾을 수 없습니다. rg 또는 grep이 설치되어 있는지 확인하세요."
            )
        except OSError as e:
            # 권한 문제 등 그 밖의 OS 수준 실행 실패.
            return ToolResult.error(f"검색 실행 실패: {e}")

        # ── 종료 코드 판정 ──
        # rg/grep 공통 규약: 0=매칭 있음, 1=매칭 없음(정상), 2 이상=실제 오류.
        # 따라서 2 이상일 때만 에러로 처리하고, 1은 "결과 없음"으로 아래에서 다룬다.
        if result.returncode >= 2:
            stderr = result.stderr.strip()
            return ToolResult.error(f"검색 오류: {stderr or '알 수 없는 오류'}")

        # 표준출력이 곧 검색 결과다. None 방지를 위해 빈 문자열로 보정한다.
        output = result.stdout or ""

        # 매칭이 하나도 없으면(공백만 있으면) 성공이되 total=0으로 안내 메시지를 준다.
        if not output.strip():
            return ToolResult.success(
                f"패턴 '{pattern}'에 일치하는 결과가 없습니다.",
                total=0,
            )

        # ── 라인 수 제한(head_limit) 적용 ──
        # rg는 명령어에 이미 --max-count로 제한을 걸었으므로 여기서 다시 자르지 않는다.
        # grep 폴백은 그런 옵션을 안 줬으므로, 결과가 head_limit을 넘으면
        # 여기서 앞부분만 남기고 "총 N줄 중 M줄만 표시" 안내를 덧붙인다.
        lines = output.splitlines()
        total_lines = len(lines)
        if not use_rg and total_lines > head_limit:
            lines = lines[:head_limit]
            output = "\n".join(lines)
            output += f"\n\n... (총 {total_lines}줄 중 {head_limit}줄만 표시)"

        # ── 전체 출력 크기 제한 ──
        # 라인 수와 별개로, 한 줄이 매우 길 수도 있으니 문자 수 상한도 둔다.
        # _MAX_OUTPUT_SIZE를 넘으면 뒤를 잘라 컨텍스트 폭주를 막는다.
        if len(output) > _MAX_OUTPUT_SIZE:
            output = output[:_MAX_OUTPUT_SIZE] + "\n... (출력 크기 초과로 잘림)"

        # 성공 결과 반환. total(원본 라인 수)과 tool_used(사용 엔진)를
        # 메타데이터로 함께 넘겨 호출자가 후처리/표시에 활용하게 한다.
        return ToolResult.success(
            output,
            total=total_lines,
            tool_used="rg" if use_rg else "grep",
        )

    # ═══ 7. UI Hints ═══
    # 실행 중/실행 전에 사용자 화면에 보여줄 짧은 문구를 제공하는 힌트 메서드들.
    # 검색 로직과 무관하며 순수하게 표시(UX) 용도다.

    def get_progress_label(self, input_data: dict[str, Any]) -> str:
        """검색 진행 중 표시할 라벨. 예: Searching for 'foo'."""
        return f"Searching for '{input_data.get('pattern', '...')}'"

    def get_input_summary(self, input_data: dict[str, Any]) -> str:
        """도구 호출 요약으로 보여줄 문자열. 여기서는 검색 패턴 그대로다."""
        return input_data.get("pattern", "")


# ─────────────────────────────────────────────
# ripgrep 명령어 구성
# ─────────────────────────────────────────────
def _build_rg_command(
    pattern: str,
    search_path: str,
    output_mode: str,
    glob_filter: str | None,
    head_limit: int,
    case_insensitive: bool,
) -> list[str]:
    """
    ripgrep(rg) 실행용 인자 리스트를 조립해 반환한다.

    입력 옵션들을 rg의 CLI 플래그로 하나씩 번역하는 순수 함수다.
    문자열이 아니라 list[str]로 만들어 shell 없이 subprocess에 그대로 넘긴다.

    매개변수:
      - pattern          : 검색할 정규표현식
      - search_path      : 검색 대상 경로
      - output_mode      : content / files_with_matches / count
      - glob_filter      : 파일 이름 필터 (없으면 None)
      - head_limit       : content 모드에서 파일당 최대 매칭 수 상한
      - case_insensitive : True면 대소문자 무시
    반환: rg에 넘길 인자 리스트 (예: ["rg", "--line-number", ...])
    """
    cmd = ["rg"]

    # 출력 모드별로 대응하는 rg 플래그를 붙인다.
    if output_mode == "files_with_matches":
        cmd.append("--files-with-matches")  # 매칭된 파일 경로만 출력
    elif output_mode == "count":
        cmd.append("--count")  # 파일별 매칭 개수만 출력
    else:
        # content 모드: 매칭 라인을 라인 번호와 함께 출력
        cmd.append("--line-number")

    # 대소문자 무시 옵션
    if case_insensitive:
        cmd.append("--ignore-case")

    # 파일 이름 glob 필터 (예: *.py)
    if glob_filter:
        cmd.extend(["--glob", glob_filter])

    # 결과 수 제한. rg에서 --max-count는 "파일당 최대 매칭 수"라서
    # count 모드(개수 집계)에는 의미가 없으므로 그때는 붙이지 않는다.
    if output_mode != "count":
        cmd.extend(["--max-count", str(head_limit)])

    # 마지막에 검색 패턴과 대상 경로를 순서대로 붙인다.
    cmd.append(pattern)
    cmd.append(search_path)

    return cmd


# ─────────────────────────────────────────────
# grep 폴백 명령어 구성
# ─────────────────────────────────────────────
def _build_grep_command(
    pattern: str,
    search_path: str,
    output_mode: str,
    glob_filter: str | None,
    case_insensitive: bool,
) -> list[str]:
    """
    표준 grep 실행용 인자 리스트를 조립한다. 시스템에 rg가 없을 때의 폴백.

    _build_rg_command과 짝을 이루는 함수로, 같은 옵션을 grep CLI 플래그로
    번역한다. rg와 달리 grep은 head_limit 상한을 명령어로 걸지 않으므로
    (call()에서 후처리로 잘라낸다) 인자로 head_limit을 받지 않는다.

    매개변수:
      - pattern          : 검색할 정규표현식
      - search_path      : 검색 대상 경로
      - output_mode      : content / files_with_matches / count
      - glob_filter      : 파일 이름 필터 (없으면 None)
      - case_insensitive : True면 대소문자 무시
    반환: grep에 넘길 인자 리스트
    """
    # --recursive: 디렉토리 하위를 재귀적으로 검색한다.
    cmd = ["grep", "--recursive"]

    # 출력 모드별 플래그 (rg 쪽과 동일한 의미로 매핑)
    if output_mode == "files_with_matches":
        cmd.append("--files-with-matches")  # 매칭 파일 경로만
    elif output_mode == "count":
        cmd.append("--count")  # 매칭 개수만
    else:
        cmd.append("--line-number")  # content 모드: 라인 번호 포함

    # 대소문자 무시 옵션
    if case_insensitive:
        cmd.append("--ignore-case")

    # 파일 이름 필터. grep에서는 rg의 --glob 대신 --include를 사용한다.
    if glob_filter:
        cmd.extend(["--include", glob_filter])

    # 바이너리 파일은 매칭 대상에서 제외한다(깨진 출력·오탐 방지).
    cmd.append("--binary-files=without-match")

    # 마지막에 패턴과 경로. --regexp로 다음 인자가 패턴임을 명시해,
    # 패턴이 '-'로 시작해도 옵션으로 오해되지 않게 한다.
    cmd.extend(["--regexp", pattern, search_path])

    return cmd
