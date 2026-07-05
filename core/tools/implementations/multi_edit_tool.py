"""
MultiEdit 도구 — 여러 파일을 한 번의 도구 호출로 일괄 편집한다.

[이 파일이 하는 일]
모델(LLM)이 여러 곳을 동시에 고쳐야 할 때, 편집을 하나씩 여러 번 호출하는 대신
편집 목록(edits 배열)을 한 번에 받아 순차적으로 적용한다. 각 편집은 특정 파일에서
old_string(기존 문자열)을 new_string(새 문자열)으로 바꾸는 단순 치환이다.

[핵심 설계 원칙]
- 각 편집은 EditTool과 동일한 안전 규칙을 따른다: old_string이 파일에 실제로
  존재하는지, 그리고 replace_all이 아니라면 파일 안에서 유일한지(1회만 등장)를 검증한다.
  유일성 검증은 "엉뚱한 위치를 바꾸는" 실수를 막기 위한 안전장치다.
- 실제 파일 쓰기는 _atomic_write로 원자적으로 수행한다. 쓰다가 중단돼도 원본이
  깨지지 않도록 임시 파일에 먼저 쓴 뒤 os.replace로 통째로 교체한다.
- 편집 하나가 실패해도 전체를 중단하지 않는다(부분 성공 허용). 실패한 편집의 사유는
  개별적으로 기록하고, 나머지 편집은 계속 진행한다.

[주요 구성 요소]
- MultiEditTool : BaseTool을 상속한 도구 클래스. 스키마/권한/실행 로직을 정의한다.
- _atomic_write : 모듈 하단의 원자적 파일 쓰기 유틸리티 함수.

[의존 관계]
core.tools.base의 BaseTool 계약(스키마·권한·실행 lifecycle)에 맞춰 구현되며,
도구 레지스트리에 등록되어 query_loop이 tool_calls로 호출한다.

작성자: 이현수 / 작성일: 2026-07-05
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Any

from core.tools.base import (
    BaseTool,
    PermissionBehavior,
    PermissionResult,
    ToolResult,
    ToolUseContext,
)

logger = logging.getLogger("nexus.tools.multi_edit")


class MultiEditTool(BaseTool):
    """
    다중 파일 편집 도구.

    여러 파일에 대한 편집을 배열(edits)로 받아 위에서부터 순차적으로 적용한다.
    BaseTool의 lifecycle 계약(validate_input → check_permissions → call)을 구현하며,
    아래 섹션 주석(1. Identity ~ 7. UI Hints)은 그 계약의 각 단계를 구분한 것이다.

    동작 요약:
      - 파일마다 old_string을 찾아 존재/유일성을 검증한 뒤 new_string으로 치환한다.
      - 쓰기 도구이므로 실행 전 사용자 확인(ASK)을 요구한다(fail-closed).
      - 편집별로 성공/실패를 집계해 사람이 읽을 수 있는 요약 텍스트로 반환한다.
    """

    # ═══ 1. Identity(정체성) ═══
    # 도구의 이름·설명·그룹. 레지스트리 등록과 모델 노출용 메타데이터다.

    @property
    def name(self) -> str:
        """도구의 고유 이름. 모델이 tool_calls에서 이 이름으로 도구를 지목한다."""
        return "MultiEdit"

    @property
    def description(self) -> str:
        """모델에게 보여줄 도구 설명. 언제·무엇을 하는 도구인지 짧게 알려준다."""
        return (
            "여러 파일에 대한 편집을 한 번에 수행합니다. "
            "각 편집은 old_string → new_string 치환입니다."
        )

    @property
    def group(self) -> str:
        """도구 분류. 파일 시스템을 다루므로 'filesystem' 그룹에 속한다."""
        return "filesystem"

    # ═══ 2. Schema(입력 스키마) ═══
    # 모델이 넘겨야 하는 입력의 형태를 JSON Schema로 선언한다.
    # 이 스키마로 프레임워크가 입력 구조를 1차 검증하고, 모델에게 인자 형식을 안내한다.

    @property
    def input_schema(self) -> dict[str, Any]:
        """
        입력 인자 구조 정의. 최상위는 edits 배열 하나이며, 배열의 각 원소가
        한 파일에 대한 편집이다. 편집 원소는 file_path/old_string/new_string이
        필수이고 replace_all은 선택(기본 False)이다.
        """
        return {
            "type": "object",
            "properties": {
                "edits": {
                    "type": "array",
                    "description": "편집 목록. 각 항목은 하나의 파일 편집을 정의한다.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "file_path": {
                                "type": "string",
                                "description": "편집할 파일의 절대 경로",
                            },
                            "old_string": {
                                "type": "string",
                                "description": "교체할 기존 문자열",
                            },
                            "new_string": {
                                "type": "string",
                                "description": "교체 후 문자열",
                            },
                            "replace_all": {
                                "type": "boolean",
                                "description": "True면 모든 발생을 교체 (기본 false)",
                                "default": False,
                            },
                        },
                        "required": ["file_path", "old_string", "new_string"],
                    },
                },
            },
            "required": ["edits"],
        }

    # ═══ 3. Behavior Flags(동작 플래그) ═══
    # 이 도구는 파일을 수정하는 '쓰기 도구'다. BaseTool의 fail-closed 기본값
    # (is_read_only=False, is_concurrency_safe=False 등)을 그대로 물려받는다.
    # 즉 별도 완화를 하지 않으므로 순차 실행·확인 필요로 안전하게 동작한다.

    # ═══ 5. Lifecycle(실행 수명주기) ═══
    # 도구 호출은 validate_input → check_permissions → call 순으로 진행된다.

    def validate_input(self, input_data: dict[str, Any]) -> str | None:
        """
        실제 파일을 건드리기 전, 입력이 최소한의 상식을 만족하는지 빠르게 점검한다.

        검증 항목:
          - edits 배열이 최소 1개는 있어야 한다(빈 배열이면 할 일이 없다).
          - 각 편집에 file_path가 있어야 한다.
          - old_string과 new_string이 같으면 바꿀 게 없으므로 무의미한 편집이다.

        반환: 문제가 있으면 사람이 읽을 오류 메시지(str), 이상 없으면 None.
        """
        edits = input_data.get("edits", [])
        if not edits:
            return "edits 배열이 비어 있습니다. 최소 1개의 편집이 필요합니다."
        # 편집을 하나씩 돌며 필수 필드와 무의미한 편집(old==new)을 걸러낸다.
        # enumerate의 i는 오류 메시지에서 몇 번째 편집이 문제인지 알려주는 용도다.
        for i, edit in enumerate(edits):
            if not edit.get("file_path"):
                return f"edits[{i}]: file_path가 비어 있습니다."
            if edit.get("old_string") == edit.get("new_string"):
                return f"edits[{i}]: old_string과 new_string이 동일합니다."
        return None

    async def check_permissions(
        self,
        input_data: dict[str, Any],
        context: ToolUseContext,
    ) -> PermissionResult:
        """
        권한 판정 단계. 파일을 수정하는 도구이므로 실행 전 사용자 확인(ASK)을
        요청한다. 확인 메시지에는 어떤 파일들이 바뀌는지 보여줘 사용자가
        판단할 수 있게 한다.

        반환: behavior=ASK인 PermissionResult. 최종 허용/거부는 상위
        권한 파이프라인이 사용자 응답과 모드를 종합해 결정한다.
        """
        edits = input_data.get("edits", [])
        # 편집 대상 파일 경로를 집합(set)으로 모아 중복을 제거한다. 같은 파일을
        # 여러 번 편집해도 확인 메시지에는 한 번만 나오게 하려는 것.
        files = list({e.get("file_path", "") for e in edits})
        # 메시지가 너무 길어지지 않도록 앞 5개만 나열하고 나머지는 개수로 축약한다.
        file_list = ", ".join(files[:5])
        if len(files) > 5:
            file_list += f" 외 {len(files) - 5}개"
        return PermissionResult(
            behavior=PermissionBehavior.ASK,
            message=f"MultiEdit: {file_list}",
        )

    async def call(
        self,
        input_data: dict[str, Any],
        context: ToolUseContext,
    ) -> ToolResult:
        """
        실제 편집을 수행하는 핵심 메서드. edits 배열을 위에서부터 하나씩 처리한다.

        편집 하나당 처리 순서:
          1. 파일이 실제로 존재하는 일반 파일인지 확인
          2. 파일을 UTF-8로 읽기
          3. old_string 검증 — 존재하는지, (replace_all이 아니면) 유일한지
          4. 문자열 치환 후 _atomic_write로 원자적 쓰기
          5. 편집별 성공/실패를 results 리스트에 기록하고 카운트 누적

        중요: 한 편집이 실패해도 continue로 다음 편집을 계속 처리한다(부분 성공).
        모든 편집이 실패했을 때만 전체를 오류로 반환하고, 하나라도 성공하면
        성공 결과로 반환한다.

        반환: 성공/실패 요약과 편집별 상세를 담은 ToolResult.
        """
        # validate_input을 통과했으므로 edits는 존재한다고 보고 바로 꺼낸다.
        edits = input_data["edits"]
        results: list[str] = []  # 편집별 결과 문장(사람이 읽을 로그)을 모으는 리스트
        success_count = 0
        error_count = 0

        for i, edit in enumerate(edits):
            # 편집 정의에서 필드를 꺼낸다. replace_all은 없으면 False(단일 치환).
            file_path = edit["file_path"]
            old_string = edit["old_string"]
            new_string = edit["new_string"]
            replace_all = edit.get("replace_all", False)

            path = Path(file_path)

            # (1) 대상이 실제 파일인지 확인. 디렉토리이거나 없으면 이 편집만 건너뛴다.
            if not path.exists() or not path.is_file():
                results.append(f"[{i}] 실패 — 파일 없음: {file_path}")
                error_count += 1
                continue

            # (2) 파일 내용을 UTF-8로 읽는다. 인코딩/입출력 오류는 이 편집만 실패 처리.
            try:
                content = path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError) as e:
                results.append(f"[{i}] 실패 — 읽기 오류: {file_path}: {e}")
                error_count += 1
                continue

            # (3) old_string이 몇 번 등장하는지 센다. 0번이면 바꿀 대상이 없어 실패.
            count = content.count(old_string)
            if count == 0:
                results.append(f"[{i}] 실패 — old_string을 찾을 수 없음: {file_path}")
                error_count += 1
                continue

            # replace_all이 아닌데 2번 이상 등장하면, 어느 것을 바꿀지 모호하므로
            # 안전을 위해 실패 처리한다(엉뚱한 위치 치환 방지).
            if not replace_all and count > 1:
                results.append(
                    f"[{i}] 실패 — old_string이 {count}회 존재 (유일하지 않음): {file_path}"
                )
                error_count += 1
                continue

            # (4) 실제 치환. replace_all이면 전부, 아니면 첫 1건만 바꾼다.
            if replace_all:
                new_content = content.replace(old_string, new_string)
            else:
                new_content = content.replace(old_string, new_string, 1)

            # 바뀐 내용을 원자적으로 저장한다. 쓰기 실패도 이 편집만 실패 처리.
            try:
                _atomic_write(path, new_content)
            except OSError as e:
                results.append(f"[{i}] 실패 — 쓰기 오류: {file_path}: {e}")
                error_count += 1
                continue

            # (5) 성공 기록. 몇 건을 바꿨는지(replace_all이면 count, 아니면 1) 함께 남긴다.
            replacements = count if replace_all else 1
            results.append(f"[{i}] 성공 — {file_path} ({replacements}건 교체)")
            success_count += 1

        # 모든 편집을 처리한 뒤, 요약 한 줄 + 편집별 상세를 합쳐 결과 텍스트를 만든다.
        summary = f"MultiEdit 완료: {success_count}건 성공, {error_count}건 실패"
        detail = "\n".join(results)
        result_text = f"{summary}\n\n{detail}"

        logger.info("MultiEdit: %d success, %d error", success_count, error_count)

        # 하나도 성공하지 못했고 실패가 있으면 전체를 오류로 반환한다.
        # 반대로 일부라도 성공했다면 성공 결과로 보고한다(부분 성공 허용 정책).
        if error_count > 0 and success_count == 0:
            return ToolResult.error(result_text)

        return ToolResult.success(
            result_text,
            success_count=success_count,
            error_count=error_count,
        )

    # ═══ 7. UI Hints(진행 표시 힌트) ═══

    def get_progress_label(self, input_data: dict[str, Any]) -> str:
        """
        도구 실행 중 UI에 보여줄 진행 라벨을 만든다. 몇 개의 편집을
        적용 중인지 사용자에게 알려주는 용도다.
        """
        edits = input_data.get("edits", [])
        return f"MultiEdit: {len(edits)}개 편집 적용 중..."


# ─────────────────────────────────────────────
# 유틸리티 함수
# ─────────────────────────────────────────────
def _atomic_write(path: Path, content: str) -> None:
    """
    파일을 '원자적으로' 저장하는 헬퍼.

    왜 이렇게 하나:
      곧바로 path에 덮어쓰면, 쓰는 도중 프로세스가 죽거나 오류가 나면 파일이
      절반만 써진 깨진 상태로 남을 수 있다. 이를 막기 위해 (1) 같은 디렉토리에
      임시 파일을 만들어 내용을 전부 쓴 뒤, (2) os.replace로 원본을 임시 파일로
      통째 교체한다. os.replace는 같은 파일 시스템 안에서 원자적이라, 최종 파일은
      항상 '이전 내용' 또는 '완전한 새 내용' 둘 중 하나만 보게 된다.
      (임시 파일을 같은 디렉토리에 두는 이유도 os.replace가 동일 파일 시스템에서만
      원자성을 보장하기 때문이다.)

    매개변수:
      path    : 최종적으로 쓰려는 대상 파일 경로
      content : 파일에 쓸 전체 문자열(UTF-8로 인코딩되어 저장된다)
    """
    # 임시 파일의 파일 디스크립터와 경로. 정리(finally) 단계에서 상태를 판단하려고
    # None으로 초기화해 둔다. 정상 처리되면 각각 None으로 되돌려 '이미 닫힘/이동됨'을 표시.
    tmp_fd = None
    tmp_path = None
    try:
        # 대상과 같은 디렉토리에 임시 파일 생성. prefix로 어떤 파일의 임시본인지 표시.
        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=str(path.parent),
            prefix=f".{path.name}.",
            suffix=".tmp",
        )
        # 내용을 임시 파일에 전부 쓴 뒤 닫는다. 닫고 나면 tmp_fd는 더 유효하지 않으므로
        # None으로 표시해, finally에서 중복으로 닫으려 하지 않도록 한다.
        os.write(tmp_fd, content.encode("utf-8"))
        os.close(tmp_fd)
        tmp_fd = None

        # 원본을 임시 파일로 원자적 교체. 성공하면 임시 파일은 사라졌으므로
        # tmp_path도 None으로 표시해 finally의 삭제 로직을 건너뛴다.
        os.replace(tmp_path, str(path))
        tmp_path = None
    finally:
        # 예외로 중간에 빠져나온 경우의 뒤처리:
        # 열린 채 남은 디스크립터가 있으면 닫고, 남은 임시 파일이 있으면 지워
        # 쓰레기 파일이 남지 않게 한다. 삭제 실패는 무시(이미 없거나 권한 문제).
        if tmp_fd is not None:
            os.close(tmp_fd)
        if tmp_path is not None:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
