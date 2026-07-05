"""
Notebook 도구 모음 — Jupyter Notebook (.ipynb) 파일 조작 도구.

이 파일은 Nexus 도구 시스템(Phase 2.0)에 등록되는 "노트북 계열" 도구 2개를
정의한다. 둘 다 BaseTool ABC를 상속하며, 도구 레지스트리에 등록되면 LLM이
직접 호출할 수 있는 도구로 노출된다.

제공하는 도구:
  - NotebookReadTool (이름 "NotebookRead"): .ipynb 파일을 파싱하여 각 셀의
    타입/소스/출력을 사람이 읽기 좋은 텍스트로 표시한다. 읽기 전용이라
    권한 검사에서 항상 ALLOW이며 병렬 실행도 안전하다.
  - NotebookEditTool (이름 "NotebookEdit"): .ipynb 파일의 특정 셀을 수정,
    삽입, 삭제한다. 파일을 변경하므로 쓰기 도구(fail-closed)로 취급되어
    권한 검사에서 사용자 확인(ASK)을 요구한다.

동작 배경(왜 이렇게 구현했는가):
  - Jupyter Notebook 파일(.ipynb)은 내부적으로 JSON 텍스트다. 따라서
    별도 라이브러리 없이 표준 json 모듈만으로 읽고 쓴다.
  - Nexus는 에어갭(폐쇄망) 환경에서 돌아가므로 Jupyter 커널/서버에 접속하지
    않는다. 코드를 "실행"하지 않고, 파일을 직접 파싱/편집하기만 한다.

주요 구성:
  - 모듈 레벨 헬퍼: _resolve_notebook_path, _format_cell,
    _source_to_lines, _create_empty_cell
  - 도구 클래스: NotebookReadTool, NotebookEditTool

의존:
  - core.tools.base 의 BaseTool / PermissionBehavior / PermissionResult /
    ToolResult / ToolUseContext 계약을 그대로 따른다.

작성자: 이현수 / 작성일: 2026-07-05
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from core.tools.base import (
    BaseTool,
    PermissionBehavior,
    PermissionResult,
    ToolResult,
    ToolUseContext,
)

logger = logging.getLogger("nexus.tools.notebook")


def _resolve_notebook_path(file_path: str, cwd: str) -> Path:
    """
    입력받은 노트북 파일 경로를 "절대 경로"로 정규화한다.

    도구는 LLM이 넘겨준 경로를 그대로 신뢰하지 않고 항상 절대 경로로
    바꿔 다룬다. 그래야 파일 존재/타입 검사와 로그가 일관되기 때문이다.

    Args:
        file_path: LLM이 넘긴 경로. 절대 경로일 수도, 상대 경로일 수도 있다.
        cwd: 현재 작업 디렉토리. 상대 경로를 붙일 기준점이 된다.

    Returns:
        심볼릭 링크와 ".." 등을 모두 해석한 절대 Path 객체.
    """
    # Path로 감싼 뒤, 절대 경로가 아니면 cwd를 앞에 붙여 기준을 맞춘다.
    p = Path(file_path)
    if not p.is_absolute():
        p = Path(cwd) / p
    return p.resolve()


def _format_cell(index: int, cell: dict[str, Any]) -> str:
    """
    노트북 셀 하나를 사람이 읽기 좋은 텍스트 한 덩어리로 변환한다.

    NotebookRead 도구가 셀을 화면에 보여줄 때 쓰는 핵심 포맷터다.
    셀의 소스 코드뿐 아니라, code 셀이라면 실행 출력(stdout, 결과값,
    에러 트레이스백)까지 뽑아내 함께 붙여 준다. 그래야 LLM이 노트북을
    읽을 때 "이 셀이 무엇을 하고 어떤 결과가 나왔는지"를 한눈에 안다.

    Args:
        index: 셀 번호(0부터 시작). 헤더에 표시된다.
        cell: 노트북 JSON에서 꺼낸 셀 딕셔너리 하나.

    Returns:
        "--- Cell N [타입] ---" 헤더 + 소스 + (있으면) 출력이 이어진 문자열.
    """
    cell_type = cell.get("cell_type", "unknown")
    # .ipynb 규격상 source는 "줄 단위 문자열 리스트" 또는 "단일 문자열"
    # 두 형태가 모두 가능하다. 리스트면 이어 붙이고, 아니면 문자열로 캐스팅.
    source = cell.get("source", [])
    if isinstance(source, list):
        source_text = "".join(source)
    else:
        source_text = str(source)

    # 셀 헤더 — 몇 번째 셀인지, 어떤 타입(code/markdown)인지 표시한다.
    header = f"--- Cell {index} [{cell_type}] ---"

    # 실행 출력은 code 셀에만 존재한다. markdown 셀은 출력 개념이 없어 건너뛴다.
    output_text = ""
    if cell_type == "code":
        outputs = cell.get("outputs", [])
        if outputs:
            # 한 셀에 여러 출력이 쌓일 수 있어(예: print 여러 번 + 반환값)
            # 각 출력을 순회하며 사람이 읽을 텍스트만 모은다.
            output_parts = []
            for out in outputs:
                # (1) stream/execute_result 등에서 오는 순수 텍스트 출력.
                #     text 역시 리스트 또는 문자열 형태를 모두 지원한다.
                if "text" in out:
                    text = out["text"]
                    if isinstance(text, list):
                        text = "".join(text)
                    output_parts.append(text)
                elif "data" in out:
                    # (2) MIME 번들 출력. 이미지/HTML 등 다양한 표현이 담기는데,
                    #     터미널에서 의미 있는 text/plain을 최우선으로 보여준다.
                    data = out["data"]
                    if "text/plain" in data:
                        plain = data["text/plain"]
                        if isinstance(plain, list):
                            plain = "".join(plain)
                        output_parts.append(plain)
                    else:
                        # text/plain이 없으면(예: 이미지 전용) 실제 값 대신
                        # 어떤 MIME 타입이 들어있는지 키 목록만 알려 준다.
                        mime_types = ", ".join(data.keys())
                        output_parts.append(f"[Output: {mime_types}]")
                elif "traceback" in out:
                    # (3) 셀 실행 중 예외가 났을 때의 트레이스백. 줄 리스트로
                    #     저장되므로 개행으로 이어 붙여 에러 블록으로 표시한다.
                    tb = "\n".join(out["traceback"])
                    output_parts.append(f"[Error]\n{tb}")

            # 실제로 뽑아낸 출력이 있을 때만 "[Output]" 섹션을 덧붙인다.
            if output_parts:
                output_text = "\n[Output]\n" + "\n".join(output_parts)

    # 헤더 + 소스 + (선택)출력을 하나의 문자열로 합쳐 반환한다.
    return f"{header}\n{source_text}{output_text}"


# ─────────────────────────────────────────────
# NotebookReadTool — 노트북 파일 읽기
# ─────────────────────────────────────────────
class NotebookReadTool(BaseTool):
    """
    Jupyter Notebook(.ipynb)을 읽어 셀 내용을 텍스트로 보여주는 읽기 전용 도구.

    LLM이 노트북 파일 내용을 파악할 때 사용한다. JSON 구조를 파싱해 각 셀의
    타입(code/markdown), 소스 코드, 그리고 code 셀의 실행 출력까지 함께
    포맷해서 돌려준다. cell_index를 주면 특정 셀 하나만, 생략하면 전체 셀을
    표시한다.

    권한/동작 특성:
      - is_read_only=True  → 파일을 절대 변경하지 않는다.
      - is_concurrency_safe=True → 여러 개를 동시에 병렬 실행해도 안전하다.
      - check_permissions는 언제나 ALLOW를 반환한다(읽기라 위험이 없음).

    BaseTool 계약에 따라 아래 순서로 속성/메서드를 구현한다:
      Identity → Schema → Behavior Flags → Lifecycle → UI Hints.
    """

    # ═══ 1. Identity ═══

    @property
    def name(self) -> str:
        # 레지스트리에 등록되고 LLM이 tool_calls로 호출할 때 쓰는 고유 이름.
        return "NotebookRead"

    @property
    def description(self) -> str:
        # LLM에게 이 도구가 무엇을 하는지 설명하는 문구. 모델이 도구 선택
        # 판단을 내릴 때 참고하므로 용도가 분명하게 드러나도록 적는다.
        return (
            "Jupyter Notebook (.ipynb) 파일을 읽습니다. "
            "각 셀의 타입(code/markdown), 소스 코드, 출력을 표시합니다."
        )

    @property
    def group(self) -> str:
        # UI/분류용 그룹 이름. 노트북 계열 도구를 한데 묶는다.
        return "notebook"

    # ═══ 2. Schema ═══
    # LLM에게 노출할 입력 파라미터 정의(JSON Schema). 모델은 이 스키마를 보고
    # 어떤 인자를 어떤 타입으로 넣어야 하는지 판단한다.

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": ".ipynb 파일 경로",
                },
                "cell_index": {
                    "type": "integer",
                    "description": "특정 셀만 읽기 (0부터 시작). 생략 시 전체 셀 표시",
                    "minimum": 0,
                },
            },
            "required": ["file_path"],
        }

    # ═══ 3. Behavior Flags ═══
    # BaseTool은 fail-closed가 기본(가장 제한적). 읽기 전용 도구라서 아래
    # 두 플래그만 명시적으로 완화한다.

    @property
    def is_read_only(self) -> bool:
        # 파일을 읽기만 하고 절대 바꾸지 않으므로 읽기 전용으로 표시.
        return True

    @property
    def is_concurrency_safe(self) -> bool:
        # 부작용이 없어 다른 도구와 동시에 병렬 실행해도 안전하다.
        return True

    # ═══ 5. Lifecycle ═══
    # 도구 실행 수명주기: validate_input(사전 검증) → check_permissions(권한)
    # → call(실제 실행) 순서로 호출된다.

    def validate_input(self, input_data: dict[str, Any]) -> str | None:
        """
        실행 전 입력값을 검증한다. 문제가 없으면 None, 있으면 에러 메시지
        문자열을 반환한다(반환값이 있으면 실행이 중단된다).

        파일 경로가 비어 있지 않은지, 그리고 확장자가 .ipynb인지만 확인한다.
        """
        file_path = input_data.get("file_path", "")
        if not file_path:
            return "file_path는 비어 있을 수 없습니다."
        if not file_path.endswith(".ipynb"):
            return "file_path는 .ipynb 파일이어야 합니다."
        return None

    async def check_permissions(
        self, input_data: dict[str, Any], context: ToolUseContext
    ) -> PermissionResult:
        """
        권한 파이프라인이 호출하는 도구 단위 권한 판단.
        읽기 전용이라 위험이 없으므로 사용자 확인 없이 항상 ALLOW를 준다.
        """
        return PermissionResult(behavior=PermissionBehavior.ALLOW)

    async def call(self, input_data: dict[str, Any], context: ToolUseContext) -> ToolResult:
        """
        실제 실행부: 노트북 파일을 읽어 셀 내용을 포맷한 뒤 ToolResult로 반환.

        성공/실패는 예외를 던지지 않고 ToolResult.success/.error로 표현한다
        (도구 계약: 오류도 정상 반환값으로 감싸 상위 루프에 전달).

        처리 순서:
          1. 파일 경로를 절대 경로로 해석하고 존재/파일 여부 확인
          2. 파일을 UTF-8로 읽어 JSON 파싱
          3. cells 목록 추출(비어 있으면 조기 반환)
          4. cell_index가 있으면 그 셀만, 없으면 전체 셀을 포맷
          5. 노트북 메타데이터(커널 등) 헤더를 붙여 결과 반환
        """
        file_path = input_data["file_path"]
        # cell_index는 선택 인자 — 없으면 None(전체 셀 모드)이 된다.
        cell_index = input_data.get("cell_index")

        # 1. 경로 해석 후 파일이 실제로 존재하고 파일인지 확인한다.
        resolved = _resolve_notebook_path(file_path, context.cwd)
        if not resolved.exists():
            return ToolResult.error(f"파일을 찾을 수 없습니다: {resolved}")
        if not resolved.is_file():
            return ToolResult.error(f"파일이 아닙니다: {resolved}")

        # 2. 파일을 읽어 JSON으로 파싱. 깨진 JSON과 IO 오류를 각각 처리한다.
        try:
            content = resolved.read_text(encoding="utf-8")
            notebook = json.loads(content)
        except json.JSONDecodeError as e:
            return ToolResult.error(f"노트북 JSON 파싱 실패: {e}")
        except OSError as e:
            return ToolResult.error(f"파일 읽기 실패: {e}")

        # 3. 셀 목록을 꺼낸다. 셀이 하나도 없으면 곧바로 안내 후 종료.
        cells = notebook.get("cells", [])
        if not cells:
            return ToolResult.success("노트북에 셀이 없습니다.", cell_count=0)

        # 4-a. cell_index가 주어지면 해당 셀 하나만 읽는다.
        if cell_index is not None:
            # 인덱스가 범위를 벗어나면 유효 범위를 알려 주며 에러 반환.
            if cell_index < 0 or cell_index >= len(cells):
                return ToolResult.error(
                    f"셀 인덱스 {cell_index}이(가) 범위를 벗어났습니다. "
                    f"총 셀 수: {len(cells)} (0~{len(cells) - 1})"
                )
            formatted = _format_cell(cell_index, cells[cell_index])
            return ToolResult.success(formatted, cell_count=1, total_cells=len(cells))

        # 4-b. cell_index가 없으면 모든 셀을 포맷해 빈 줄 두 개로 이어 붙인다.
        formatted_cells = [_format_cell(i, cell) for i, cell in enumerate(cells)]
        result_text = "\n\n".join(formatted_cells)

        # 5. 노트북 상단에 파일명/커널/셀 개수를 요약한 헤더를 만든다.
        #    커널 정보는 metadata.kernelspec.display_name에 들어 있다.
        kernel_info = notebook.get("metadata", {}).get("kernelspec", {})
        kernel_name = kernel_info.get("display_name", "unknown")
        header = f"Notebook: {resolved.name} | Kernel: {kernel_name} | Cells: {len(cells)}\n"
        header += "=" * 60

        logger.debug("NotebookRead: %s, %d cells", resolved, len(cells))
        return ToolResult.success(
            f"{header}\n\n{result_text}",
            cell_count=len(cells),
        )

    # ═══ 7. UI Hints ═══
    # 실행 중/목록 표시에 쓰이는 짧은 라벨을 제공하는 보조 메서드들.

    def get_progress_label(self, input_data: dict[str, Any]) -> str:
        # 실행 중 진행 표시줄에 뜨는 문구(예: "Reading foo.ipynb...").
        return f"Reading {input_data.get('file_path', 'notebook')}..."

    def get_input_summary(self, input_data: dict[str, Any]) -> str:
        # 도구 호출을 한 줄로 요약할 때 쓰는 문자열(대상 파일 경로).
        return input_data.get("file_path", "")


# ─────────────────────────────────────────────
# NotebookEditTool — 노트북 셀 수정
# ─────────────────────────────────────────────
class NotebookEditTool(BaseTool):
    """
    Jupyter Notebook(.ipynb)의 특정 셀을 편집하는 쓰기 도구.

    operation 인자에 따라 세 가지 작업을 수행한다:
      - edit  : cell_index 위치의 기존 셀 소스를 new_source로 교체
      - insert: cell_index 위치에 새 셀(cell_type/new_source)을 끼워넣기
      - delete: cell_index 위치의 셀을 제거

    편집이 끝나면 노트북 전체를 다시 JSON으로 직렬화해 파일에 덮어쓴다.

    권한/동작 특성:
      - 파일을 변경하는 쓰기 도구라 BaseTool 기본값(fail-closed)을 그대로
        유지한다. is_read_only/is_concurrency_safe를 완화하지 않는다.
      - check_permissions는 ASK를 반환해 실행 전 사용자 확인을 요구한다.
    """

    # ═══ 1. Identity ═══

    @property
    def name(self) -> str:
        # 레지스트리 등록 및 LLM 호출에 쓰이는 고유 도구 이름.
        return "NotebookEdit"

    @property
    def description(self) -> str:
        # 모델이 도구 용도를 이해하도록 하는 설명. 수정/삽입/삭제를 명시.
        return (
            "Jupyter Notebook (.ipynb)의 특정 셀을 수정합니다. "
            "셀 내용 변경, 새 셀 삽입, 셀 삭제를 지원합니다."
        )

    @property
    def group(self) -> str:
        # NotebookRead와 같은 "notebook" 그룹으로 묶는다.
        return "notebook"

    # ═══ 2. Schema ═══
    # 편집에 필요한 입력 파라미터 정의. file_path와 cell_index는 필수이고,
    # new_source/cell_type/operation은 작업 종류에 따라 선택적으로 쓰인다.

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": ".ipynb 파일 경로",
                },
                "cell_index": {
                    "type": "integer",
                    "description": "수정할 셀 번호 (0부터 시작)",
                    "minimum": 0,
                },
                "new_source": {
                    "type": "string",
                    "description": "새 셀 소스 코드/텍스트",
                },
                "cell_type": {
                    "type": "string",
                    "description": "셀 타입 (code 또는 markdown). 삽입 시 사용",
                    "enum": ["code", "markdown"],
                },
                "operation": {
                    "type": "string",
                    "description": "수행할 작업: edit(수정), insert(삽입), delete(삭제)",
                    "enum": ["edit", "insert", "delete"],
                    "default": "edit",
                },
            },
            "required": ["file_path", "cell_index"],
        }

    # ═══ 3. Behavior Flags ═══
    # 쓰기 도구이므로 BaseTool의 fail-closed 기본값을 그대로 둔다.
    # (is_read_only=False, is_concurrency_safe=False 등을 오버라이드하지 않음)

    # ═══ 5. Lifecycle ═══

    def validate_input(self, input_data: dict[str, Any]) -> str | None:
        """
        실행 전 입력값 검증. 문제가 없으면 None, 있으면 에러 문자열을 반환.

        확인 항목:
          - file_path가 비어 있지 않고 .ipynb 확장자인지
          - edit/insert 작업이면 new_source가 제공됐는지
            (단, 빈 문자열 ""은 "내용을 비우는" 의도로 허용한다)
        """
        file_path = input_data.get("file_path", "")
        if not file_path:
            return "file_path는 비어 있을 수 없습니다."
        if not file_path.endswith(".ipynb"):
            return "file_path는 .ipynb 파일이어야 합니다."

        operation = input_data.get("operation", "edit")
        # edit(수정)과 insert(삽입)는 넣을 내용이 있어야 한다. 다만 빈 문자열
        # ""은 유효한 값(빈 셀)이므로 통과시키고, None/미제공만 걸러낸다.
        if operation in ("edit", "insert"):
            if not input_data.get("new_source") and input_data.get("new_source") != "":
                return f"'{operation}' 작업에는 new_source가 필요합니다."
        return None

    async def check_permissions(
        self, input_data: dict[str, Any], context: ToolUseContext
    ) -> PermissionResult:
        """
        쓰기 도구라 파일을 바꾸기 전에 사용자 확인이 필요하다. 따라서 항상
        ASK를 반환하고, 어떤 작업을 어떤 셀에 할지 message로 알려 준다.
        """
        file_path = input_data.get("file_path", "")
        operation = input_data.get("operation", "edit")
        cell_index = input_data.get("cell_index", 0)
        return PermissionResult(
            behavior=PermissionBehavior.ASK,
            message=f"Notebook {operation}: cell {cell_index} in {file_path}",
        )

    async def call(self, input_data: dict[str, Any], context: ToolUseContext) -> ToolResult:
        """
        실제 실행부: 노트북 셀을 수정/삽입/삭제한 뒤 파일에 저장한다.

        읽기 도구와 마찬가지로 예외를 던지지 않고 오류도 ToolResult.error로
        감싸 반환한다.

        처리 순서:
          1. 파일 경로 해석 및 존재 확인
          2. 파일을 읽어 JSON 파싱
          3. operation(edit/insert/delete)에 맞춰 셀 목록을 변경
             (각 작업마다 cell_index 범위를 먼저 검증)
          4. 변경된 노트북을 다시 JSON으로 직렬화해 파일에 덮어쓰기
        """
        file_path = input_data["file_path"]
        cell_index = input_data["cell_index"]
        new_source = input_data.get("new_source", "")
        # cell_type은 insert에서 새 셀 종류를 정할 때 쓰인다(기본 code).
        cell_type = input_data.get("cell_type", "code")
        # operation 미지정 시 기본 동작은 edit(수정).
        operation = input_data.get("operation", "edit")

        # 1. 경로를 절대 경로로 바꾸고 파일이 있는지 확인한다.
        resolved = _resolve_notebook_path(file_path, context.cwd)
        if not resolved.exists():
            return ToolResult.error(f"파일을 찾을 수 없습니다: {resolved}")

        # 2. 파일을 읽어 JSON 파싱(깨진 JSON/IO 오류를 구분해 처리).
        try:
            content = resolved.read_text(encoding="utf-8")
            notebook = json.loads(content)
        except json.JSONDecodeError as e:
            return ToolResult.error(f"노트북 JSON 파싱 실패: {e}")
        except OSError as e:
            return ToolResult.error(f"파일 읽기 실패: {e}")

        cells = notebook.get("cells", [])

        # 3. operation에 따라 cells 리스트를 직접 수정한다. 성공 안내 문구는
        #    msg에 담아 두었다가 저장 후 결과로 돌려준다.
        if operation == "edit":
            # 기존 셀의 소스만 교체한다. 인덱스는 [0, len) 범위여야 한다.
            if cell_index < 0 or cell_index >= len(cells):
                return ToolResult.error(
                    f"셀 인덱스 {cell_index}이(가) 범위를 벗어났습니다. 총 셀 수: {len(cells)}"
                )
            # 새 소스를 Jupyter 표준(줄 단위 리스트) 형식으로 바꿔 넣는다.
            cells[cell_index]["source"] = _source_to_lines(new_source)
            msg = f"셀 {cell_index}의 내용을 수정했습니다."

        elif operation == "insert":
            # 새 셀을 끼워넣기. 맨 끝(len 위치) 삽입도 허용하므로 상한이
            # len(cells)까지다(edit/delete와 범위 상한이 다른 점에 주의).
            if cell_index < 0 or cell_index > len(cells):
                return ToolResult.error(
                    f"삽입 위치 {cell_index}이(가) 범위를 벗어났습니다. 허용 범위: 0~{len(cells)}"
                )
            new_cell = _create_empty_cell(cell_type, new_source)
            cells.insert(cell_index, new_cell)
            msg = f"셀 {cell_index} 위치에 새 {cell_type} 셀을 삽입했습니다."

        elif operation == "delete":
            # 지정 셀을 제거한다. pop으로 빼내면서 삭제된 셀 타입도 알려 준다.
            if cell_index < 0 or cell_index >= len(cells):
                return ToolResult.error(
                    f"셀 인덱스 {cell_index}이(가) 범위를 벗어났습니다. 총 셀 수: {len(cells)}"
                )
            deleted = cells.pop(cell_index)
            deleted_type = deleted.get("cell_type", "unknown")
            msg = f"셀 {cell_index} ({deleted_type})을 삭제했습니다."

        else:
            # 스키마 enum 밖의 값이 들어온 경우의 방어 코드.
            return ToolResult.error(f"알 수 없는 작업: {operation}")

        # 4. 변경한 cells를 노트북에 되꽂고 파일에 저장한다.
        #    ensure_ascii=False로 한글이 그대로 저장되게 하고, indent=1은
        #    Jupyter가 쓰는 표준 들여쓰기 폭이라 diff 노이즈를 줄인다.
        notebook["cells"] = cells
        try:
            resolved.write_text(
                json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
                encoding="utf-8",
            )
        except OSError as e:
            return ToolResult.error(f"파일 저장 실패: {e}")

        logger.info("NotebookEdit: %s cell %d in %s", operation, cell_index, resolved)
        return ToolResult.success(msg, cell_count=len(cells))

    # ═══ 7. UI Hints ═══
    # 진행 표시/호출 요약용 짧은 라벨.

    def get_progress_label(self, input_data: dict[str, Any]) -> str:
        # 실행 중 표시 문구(예: "Notebook insert...").
        operation = input_data.get("operation", "edit")
        return f"Notebook {operation}..."

    def get_input_summary(self, input_data: dict[str, Any]) -> str:
        # 호출을 한 줄로 요약: 대상 파일과 셀 번호.
        return f"{input_data.get('file_path', '')} cell {input_data.get('cell_index', '')}"


# ─────────────────────────────────────────────
# 유틸리티 함수
# ─────────────────────────────────────────────
def _source_to_lines(source: str) -> list[str]:
    """
    소스 문자열을 Jupyter 표준 형식인 "줄 단위 리스트"로 변환한다.

    .ipynb 규격에서 셀 source는 각 원소가 개행(\\n)으로 끝나는 문자열
    리스트로 저장하는 것이 관례다(마지막 줄만 개행 없이). 이 형식을
    지켜야 다른 Jupyter 도구에서 diff가 깔끔하게 나온다. edit/insert 시
    사용자가 준 통짜 문자열을 이 형식으로 맞춰 주는 헬퍼.

    예) "a\\nb"  → ["a\\n", "b"]
        "a\\nb\\n"→ ["a\\n", "b\\n"]  (끝 개행이면 마지막 원소도 개행 포함)

    Args:
        source: 셀에 넣을 원본 문자열.

    Returns:
        Jupyter 규격의 줄 단위 문자열 리스트(빈 문자열이면 빈 리스트).
    """
    if not source:
        return []
    lines = source.split("\n")
    # 마지막 줄을 뺀 나머지 줄에는 split으로 사라진 \n을 도로 붙여 준다.
    result = [line + "\n" for line in lines[:-1]]
    # 마지막 조각이 빈 문자열이면(원본이 \n으로 끝난 경우) 추가하지 않아
    # 불필요한 빈 원소가 생기지 않게 한다. 내용이 있으면 개행 없이 추가.
    if lines[-1]:
        result.append(lines[-1])
    return result


def _create_empty_cell(cell_type: str, source: str = "") -> dict[str, Any]:
    """
    새 노트북 셀 딕셔너리를 규격에 맞게 만들어 준다.

    NotebookEdit의 insert 작업에서 새 셀을 끼워넣을 때 사용한다. 모든
    셀은 cell_type/metadata/source를 갖고, code 셀은 추가로 실행 관련
    필드(execution_count, outputs)를 가진다.

    Args:
        cell_type: "code" 또는 "markdown".
        source: 셀에 넣을 초기 소스(기본값은 빈 셀).

    Returns:
        .ipynb 규격을 따르는 셀 딕셔너리.
    """
    cell: dict[str, Any] = {
        "cell_type": cell_type,
        "metadata": {},
        "source": _source_to_lines(source),
    }
    if cell_type == "code":
        # code 셀은 아직 실행 전이므로 execution_count는 None, 출력은 빈 리스트.
        cell["execution_count"] = None
        cell["outputs"] = []
    return cell
