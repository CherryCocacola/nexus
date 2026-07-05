"""
Glob 도구 — glob 패턴으로 파일 경로를 검색하는 읽기 전용 도구.

이 파일이 하는 일 (한눈에):
  - 사용자가 넘긴 glob 패턴(예: ``**/*.py``, ``src/*.md``)에 일치하는 파일을 찾는다.
  - 찾은 결과를 파일 수정 시간 기준 내림차순(가장 최근에 바뀐 파일이 위)으로 정렬한다.
  - 결과가 너무 많으면 상위 250개(_MAX_RESULTS)까지만 잘라서 반환한다.
  - 파일 시스템을 읽기만 하므로 절대 아무것도 수정하지 않는다.

핵심 클래스:
  - ``GlobTool`` — ``BaseTool`` ABC를 상속한 24개 도구 중 하나. 모델이 "Glob"이라는
    이름으로 호출하며, 4-Tier 체인의 도구 실행 단계(Executor)에서 ``call()``이 불린다.

이 도구가 읽기 전용/동시성 안전인 이유:
  - 파일 목록만 조회하고 쓰기·삭제가 없으므로, fail-closed 기본값(쓰기/순차)을 안전하게
    완화하여 ``is_read_only=True``, ``is_concurrency_safe=True``로 선언한다. 덕분에
    Executor가 이 도구를 다른 읽기 도구와 병렬로 실행할 수 있다.

작성자: 이현수 / 작성일: 2026-07-05
"""

from __future__ import annotations

import glob
import logging
import os
from pathlib import Path
from typing import Any

from core.tools.base import (
    BaseTool,
    PermissionBehavior,
    PermissionResult,
    ToolResult,
    ToolUseContext,
)

# 프로젝트 규칙(P7)에 따라 모듈별 네임스페이스 로거를 사용한다.
# "nexus.tools.glob" 계층 이름으로 로그를 남겨, 나중에 도구별로 필터링하기 쉽게 한다.
logger = logging.getLogger("nexus.tools.glob")

# 한 번의 검색에서 모델에게 돌려줄 파일 경로의 최대 개수.
# 결과가 폭주해 컨텍스트를 낭비하는 것을 막기 위한 상한선이다.
# 여기서 값을 조정하면 잘리는 기준이 바뀐다(하드코딩 대신 상수로 분리).
_MAX_RESULTS = 250


class GlobTool(BaseTool):
    """
    glob 패턴으로 파일 경로를 찾아 주는 검색 도구.

    동작 요약:
      - 입력받은 패턴을 검색 기준 디렉토리와 합쳐 ``glob.glob``으로 재귀 검색한다.
      - 디렉토리는 버리고 실제 파일만 남긴 뒤, 수정 시간 내림차순으로 정렬한다.
      - 상위 _MAX_RESULTS개까지만 반환하고, 잘렸다면 전체 개수를 함께 알려 준다.

    ``BaseTool``이 정의한 수명주기(validate_input → check_permissions → call)를 그대로
    따르며, 각 단계 구현은 아래 메서드에 있다.
    """

    # ═══ 1. Identity(정체성) ═══
    # 도구의 이름·설명·분류 등 "이 도구가 무엇인가"를 알려 주는 메타데이터.

    @property
    def name(self) -> str:
        # 모델과 Registry가 이 도구를 식별하는 고유 이름. 절대 바꾸면 안 된다
        # (프롬프트/캐시/도구 매핑이 모두 이 문자열에 의존한다).
        return "Glob"

    @property
    def description(self) -> str:
        # 도구 스키마에 실려 모델에게 전달되는 한 줄 설명.
        # 모델이 "이 도구를 언제 써야 하는가"를 판단하는 근거가 된다.
        return "Find files matching a glob pattern."

    @property
    def group(self) -> str:
        # 도구 분류(UI 그룹핑·통계 용도). 파일을 찾는 도구이므로 "search" 그룹.
        return "search"

    # ═══ 2. Schema(입력 스키마) ═══
    # 모델이 이 도구를 호출할 때 넘겨야 하는 인자의 형태를 JSON Schema로 정의한다.

    @property
    def input_schema(self) -> dict[str, Any]:
        # pattern은 필수, path는 선택. path를 생략하면 call()에서 현재 작업
        # 디렉토리(context.cwd)를 기본값으로 사용한다.
        return {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Glob pattern (e.g. **/*.py)",
                },
                "path": {
                    "type": "string",
                    "description": "Base directory path",
                },
            },
            "required": ["pattern"],
        }

    # ═══ 3. Behavior Flags(동작 플래그) ═══
    # BaseTool의 기본값은 fail-closed(가장 안전한 쪽: 쓰기·순차 실행)이다.
    # 이 도구는 파일을 읽기만 하므로, 아래 두 플래그를 명시적으로 True로 완화한다.

    @property
    def is_read_only(self) -> bool:
        # 파일 시스템을 조회만 하고 변경하지 않으므로 읽기 전용으로 표시.
        # 권한 파이프라인이 이 도구를 안전한 READONLY 범주로 취급하게 된다.
        return True

    @property
    def is_concurrency_safe(self) -> bool:
        # 상태를 바꾸지 않아 다른 읽기 도구와 동시에 돌려도 안전하다.
        # Executor가 이 도구를 병렬 실행 대상으로 묶을 수 있게 해 준다.
        return True

    # ═══ 5. Lifecycle(수명주기) ═══
    # BaseTool이 정한 실행 순서: validate_input → check_permissions → call.

    def validate_input(self, input_data: dict[str, Any]) -> str | None:
        """
        call() 실행 전에 입력값이 올바른지 미리 검사한다.

        여기서는 필수 인자인 pattern이 비어 있거나 공백뿐인지만 확인한다.
        문제가 있으면 사용자에게 보여 줄 오류 메시지(문자열)를 반환하고,
        정상이면 None을 반환한다(관례상 None = 통과).
        """
        pattern = input_data.get("pattern", "")
        # 빈 문자열이거나 공백만 있으면 검색 자체가 무의미하므로 거부한다.
        if not pattern or not pattern.strip():
            return "pattern은 비어 있을 수 없습니다."
        return None

    async def check_permissions(
        self,
        input_data: dict[str, Any],
        context: ToolUseContext,
    ) -> PermissionResult:
        """
        이 도구를 실행해도 되는지 권한을 판정한다.

        파일을 읽기만 하는 안전한 도구라, 별도의 사용자 확인 없이 항상 ALLOW를
        반환한다. (경로 순회 차단 등 더 엄격한 검증이 필요하면 상위 권한
        파이프라인의 Layer 2에서 처리된다.)
        """
        return PermissionResult(behavior=PermissionBehavior.ALLOW)

    async def call(
        self,
        input_data: dict[str, Any],
        context: ToolUseContext,
    ) -> ToolResult:
        """
        glob 패턴으로 파일을 검색해 그 목록을 ToolResult로 돌려주는 실제 본체.

        매개변수:
          - input_data: 모델이 넘긴 인자. "pattern"(필수)과 "path"(선택)를 담는다.
          - context: 실행 컨텍스트. path가 없을 때 기준 디렉토리로 쓸 cwd 등을 제공.

        반환:
          - 성공 시 ToolResult.success — 파일 경로를 줄바꿈으로 이어 붙인 문자열과
            함께 total/shown 메타데이터를 담는다.
          - 실패 시 ToolResult.error — 사람이 읽을 수 있는 오류 메시지를 담는다.

        처리 순서:
          1. 검색 디렉토리 결정 (path 또는 context.cwd)
          2. glob.glob으로 재귀 검색
          3. 파일만 필터링 (디렉토리 제외)
          4. 수정 시간순 정렬 (최신 먼저)
          5. 최대 250개로 제한하여 반환
        """
        pattern = input_data["pattern"]
        # path가 주어지면 그 디렉토리를, 없으면 현재 작업 디렉토리를 검색 기준으로 삼는다.
        search_dir = input_data.get("path", context.cwd)

        # 검색 디렉토리 유효성 확인 — 존재하지 않거나 파일이면 검색이 불가능하므로
        # 여기서 미리 걸러 명확한 오류를 돌려준다.
        search_path = Path(search_dir)
        if not search_path.exists():
            return ToolResult.error(f"디렉토리를 찾을 수 없습니다: {search_dir}")
        if not search_path.is_dir():
            return ToolResult.error(f"디렉토리가 아닙니다: {search_dir}")

        # glob 검색: 기준 디렉토리와 패턴을 합쳐 완전한 검색 경로를 만든다.
        # recursive=True라야 ``**``(하위 디렉토리 전체)가 실제로 동작한다.
        full_pattern = os.path.join(search_dir, pattern)
        try:
            matches = glob.glob(full_pattern, recursive=True)
        except Exception as e:
            # 잘못된 패턴 등으로 glob이 예외를 던져도 도구가 죽지 않도록
            # 오류를 ToolResult로 감싸 정상적으로 반환한다.
            return ToolResult.error(f"glob 검색 실패: {e}")

        # 파일만 필터링 (디렉토리는 결과에서 제외) — 사용자는 보통 실제 파일을 원한다.
        files = [m for m in matches if os.path.isfile(m)]

        # 수정 시간순 정렬을 위한 보조 함수.
        # 검색과 정렬 사이에 파일이 삭제되면 getmtime이 OSError를 낼 수 있으므로,
        # 그런 경우 0.0(가장 오래된 것으로 취급)을 돌려 정렬이 깨지지 않게 한다.
        def _safe_mtime(filepath: str) -> float:
            try:
                return os.path.getmtime(filepath)
            except OSError:
                return 0.0

        # 최신 파일이 위로 오도록 수정 시간 내림차순 정렬.
        files.sort(key=_safe_mtime, reverse=True)

        # 자르기 전의 전체 매칭 개수를 기억해 둔다(결과 요약에 사용).
        total_found = len(files)
        # 상한선(_MAX_RESULTS)까지만 남긴다 — 너무 많은 경로로 컨텍스트를 낭비하지 않기 위함.
        files = files[:_MAX_RESULTS]

        # 결과 포맷팅 — 매칭이 하나도 없으면 그 사실을 명확히 알려 준다.
        if not files:
            return ToolResult.success(
                f"패턴 '{pattern}'에 일치하는 파일이 없습니다.",
                total=0,
            )

        # 각 파일 경로를 줄바꿈으로 구분해 하나의 문자열로 합친다.
        result_lines = "\n".join(files)
        # 상한선 때문에 잘렸다면, 전체 몇 개 중 몇 개만 보여 주는지 꼬리말로 덧붙인다.
        if total_found > _MAX_RESULTS:
            result_lines += f"\n\n... (총 {total_found}개 중 {_MAX_RESULTS}개만 표시)"

        # 운영 중 검색 동작을 추적할 수 있도록 디버그 로그를 남긴다(패턴·위치·개수).
        logger.debug("Glob '%s' in '%s': %d files found", pattern, search_dir, total_found)

        # total(전체 매칭 수)과 shown(실제 표시 수)을 메타데이터로 함께 전달한다.
        return ToolResult.success(
            result_lines,
            total=total_found,
            shown=len(files),
        )

    # ═══ 7. UI Hints(사용자 인터페이스 힌트) ═══
    # 도구 실행 중/후에 CLI·웹 UI가 보여 줄 짧은 문구를 만들어 주는 메서드들.

    def get_progress_label(self, input_data: dict[str, Any]) -> str:
        # 도구가 실행되는 동안 화면에 표시할 진행 문구(예: "Searching **/*.py").
        return f"Searching {input_data.get('pattern', '...')}"

    def get_input_summary(self, input_data: dict[str, Any]) -> str:
        # 도구 호출을 한 줄로 요약할 때 쓰는 값. 여기서는 검색 패턴 그대로를 보여 준다.
        return input_data.get("pattern", "")
