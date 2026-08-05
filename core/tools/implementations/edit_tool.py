"""
Edit 도구 — 파일 안의 특정 문자열을 정확히 찾아 다른 문자열로 바꾸는 도구.

[이 파일이 하는 일]
사용자(또는 모델)가 지정한 파일을 열어서, old_string 과 완전히 똑같은
부분을 찾아 new_string 으로 교체한 뒤 파일을 다시 저장한다.
즉 "부분 수정" 전용 도구다. 파일 전체를 새로 쓰는 Write 도구와 달리,
기존 파일의 일부만 안전하게 바꾸고 싶을 때 사용한다.

[핵심 안전장치]
- 유일성 검증: replace_all=False(기본값)일 때는 old_string 이 파일 안에
  정확히 1회만 존재해야 한다. 여러 곳에 있으면 어디를 바꿀지 모호하므로
  에러를 돌려주고 아무것도 바꾸지 않는다(엉뚱한 곳 수정 방지).
- 원자적 쓰기: 같은 폴더에 임시 파일을 먼저 만들어 새 내용을 다 쓴 다음,
  os.replace 로 원본과 한 번에 교체한다. 도중에 프로그램이 죽더라도
  원본이 반쯤 망가진 상태로 남지 않는다(전부 성공 아니면 전부 실패).

[주요 구성요소]
- EditTool 클래스: BaseTool 을 상속한 실제 도구 구현체. 스키마, 권한,
  입력 검증, 실행(call) 로직을 담는다.
- _atomic_write 함수: 위에서 설명한 안전한 파일 교체를 담당하는 헬퍼.

[다른 모듈과의 관계]
- core.tools.base 의 BaseTool/ToolResult/권한 타입을 사용한다.
- 도구 레지스트리에 "Edit" 라는 이름으로 등록되어, 오케스트레이터의
  쿼리 루프가 모델의 tool_calls 를 받아 이 도구의 call()을 호출한다.

작성자: 이현수 / 작성일: 2026-07-05
"""

from __future__ import annotations

import difflib
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

logger = logging.getLogger("nexus.tools.edit")


class EditTool(BaseTool):
    """
    파일 편집 도구(BaseTool 구현체).

    파일에서 old_string 을 찾아 new_string 으로 교체한다.
    기본(replace_all=False)에서는 old_string 이 파일 안에서 유일해야 하며,
    유일하지 않으면 실행을 거부해 잘못된 위치를 수정하는 사고를 막는다.

    BaseTool 이 정한 도구의 생애주기(입력검증 → 권한확인 → 실행)에 맞춰
    validate_input / check_permissions / call 을 각각 구현한다.
    """

    # ═══ 1. Identity(도구 식별 정보) ═══
    # 아래 세 property 는 이 도구가 "무엇인지"를 알려주는 메타데이터다.
    # 레지스트리 등록, 모델에게 노출할 도구 이름/설명, 분류 그룹에 쓰인다.

    @property
    def name(self) -> str:
        # 레지스트리와 모델의 tool_calls 에서 이 도구를 지칭하는 고유 이름.
        return "Edit"

    @property
    def description(self) -> str:
        # 모델에게 보여줄 짧은 설명. "파일 안의 정확한 문자열을 교체한다".
        return "Replace exact string in a file."

    @property
    def group(self) -> str:
        # 도구 분류용 그룹명. 파일시스템 계열 도구로 묶인다.
        return "filesystem"

    # ═══ 2. Schema(입력 스키마) ═══
    # 모델이 이 도구를 호출할 때 넘겨야 하는 인자들의 JSON Schema 정의.
    # 오케스트레이터는 이 스키마로 모델 입력을 검증하고, 모델에게도 그대로
    # 노출해 어떤 인자가 필요한지 알려준다.

    @property
    def input_schema(self) -> dict[str, Any]:
        # file_path/old_string/new_string 은 필수, replace_all 은 선택(기본 false).
        return {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Absolute file path",
                },
                "old_string": {
                    "type": "string",
                    "description": "Exact string to find",
                },
                "new_string": {
                    "type": "string",
                    "description": "Replacement string",
                },
                "replace_all": {
                    "type": "boolean",
                    "description": "True면 모든 발생을 교체. 기본 false (유일성 검증)",
                    "default": False,
                },
            },
            "required": ["file_path", "old_string", "new_string"],
        }

    # ═══ 3. Behavior Flags(동작 플래그) ═══
    # 이 도구는 파일을 "쓰는" 도구다. 따라서 BaseTool 이 정한 fail-closed
    # 기본값(읽기전용 아님, 동시실행 안전 아님 등)을 그대로 물려받는다.
    # 즉 별도로 완화하지 않으므로 병렬 실행되지 않고 확인 절차를 거친다.

    # ═══ 5. Lifecycle(생애주기 메서드) ═══
    # 도구 실행은 validate_input → check_permissions → call 순서로 진행된다.

    def validate_input(self, input_data: dict[str, Any]) -> str | None:
        """
        실행 전에 입력값이 말이 되는지 빠르게 검사한다.

        여기서 걸러내는 것:
          - file_path 가 비었거나 공백뿐이면 대상 파일을 특정할 수 없다.
          - old_string 과 new_string 이 같으면 바꿀 내용이 없어 무의미하다.

        문제가 있으면 사용자에게 보여줄 한글 에러 메시지(문자열)를 반환하고,
        문제가 없으면 None 을 반환해 "통과"를 알린다.
        """
        # 대상 파일 경로가 비어 있는지 확인(공백만 있는 경우도 무효로 처리).
        file_path = input_data.get("file_path", "")
        if not file_path or not file_path.strip():
            return "file_path는 비어 있을 수 없습니다."

        # 찾을 문자열과 바꿀 문자열이 동일하면 실질적인 변경이 없다.
        old_string = input_data.get("old_string", "")
        new_string = input_data.get("new_string", "")
        if old_string == new_string:
            return "old_string과 new_string이 동일합니다. 변경할 내용이 없습니다."

        # 모든 검사를 통과 — None 을 돌려 다음 단계로 진행하게 한다.
        return None

    async def check_permissions(
        self,
        input_data: dict[str, Any],
        context: ToolUseContext,
    ) -> PermissionResult:
        """
        실행 권한을 판정한다. 이 도구는 파일을 수정하는 쓰기 작업이므로
        곧바로 실행하지 않고 사용자에게 확인을 요청(ASK)한다.

        어떤 파일을 편집할지 메시지에 담아 사용자가 판단할 수 있게 한다.
        """
        file_path = input_data.get("file_path", "")
        # ASK: 상위 권한 파이프라인이 사용자에게 승인 여부를 묻도록 지시.
        return PermissionResult(
            behavior=PermissionBehavior.ASK,
            message=f"Edit: {file_path}",
        )

    async def call(
        self,
        input_data: dict[str, Any],
        context: ToolUseContext,
    ) -> ToolResult:
        """
        실제 편집을 수행하는 핵심 메서드.
        파일을 읽어 old_string 을 new_string 으로 바꾼 뒤 다시 저장한다.

        처리 순서(단계별로 아래 코드에 주석을 달아 두었다):
          1. 파일이 실제로 존재하고 "파일"인지 확인한다.
          2. 파일을 UTF-8 로 읽어 old_string 이 몇 번 나오는지 센다.
          3. replace_all=False 면 유일성(정확히 1회)을 검증한다.
          4. 조건에 맞게 문자열을 치환해 새 내용을 만든다.
          5. 원자적 쓰기로 파일을 안전하게 교체한다.

        반환: 성공하면 ToolResult.success(교체 건수 포함), 문제가 있으면
        각 단계에서 ToolResult.error(한글 메시지)로 조기 반환한다.
        """
        # 필수 인자는 스키마로 이미 보장되므로 대괄호로 바로 꺼낸다.
        file_path = input_data["file_path"]
        old_string = input_data["old_string"]
        new_string = input_data["new_string"]
        # replace_all 은 선택 인자 — 없으면 기본값 False(유일성 검증 모드).
        replace_all = input_data.get("replace_all", False)

        # 문자열 경로를 다루기 쉬운 Path 객체로 변환한다.
        path = Path(file_path)

        # 1단계: 대상이 존재하며 "파일"인지 확인(폴더 등을 잘못 지정한 경우 차단).
        if not path.exists():
            return ToolResult.error(f"파일을 찾을 수 없습니다: {file_path}")
        if not path.is_file():
            return ToolResult.error(f"파일이 아닙니다: {file_path}")

        # 파일 내용을 UTF-8 로 읽는다. 인코딩/입출력 오류는 각각 구분해 처리한다.
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            # 바이너리이거나 다른 인코딩이라 UTF-8 로 해석 불가한 경우.
            return ToolResult.error(f"파일을 UTF-8로 읽을 수 없습니다: {file_path}")
        except OSError as e:
            # 권한 문제, 잠금 등 파일시스템 레벨의 읽기 실패.
            return ToolResult.error(f"파일을 읽을 수 없습니다: {e}")

        # 2단계: old_string 이 파일 안에 몇 번 등장하는지 센다.
        count = content.count(old_string)
        if count == 0:
            # 정확 매칭 실패 — 폴백으로 "공백 정규화 매칭"을 시도한다.
            # (실측 배경: A.X-4.0이 들여쓰기·공백을 정확히 재현하지 못해
            #  Edit이 연속 실패하고 작업을 포기하는 사례가 관찰됐다. 줄 단위로
            #  공백을 정규화해 비교하면 이런 실수를 흡수할 수 있다.
            #  단, 정규화 매치가 "정확히 1곳"일 때만 적용한다 — 2곳 이상이면
            #  어디를 바꿀지 모호하므로 실패로 처리한다(fail-closed).)
            span = find_whitespace_fuzzy_span(content, old_string)
            if span is not None:
                start, end = span
                new_content = content[:start] + new_string + content[end:]
                try:
                    _atomic_write(path, new_content)
                except OSError as e:
                    return ToolResult.error(f"파일 쓰기에 실패했습니다: {e}")
                logger.info("Edit %s: 1 replacement (fuzzy whitespace)", file_path)
                return ToolResult.success(
                    f"편집 완료: {file_path} (1건 교체 — 공백 정규화 매칭. "
                    "old_string의 들여쓰기/공백이 파일과 달랐지만 내용이 유일하게 "
                    "일치해 해당 부분을 교체했습니다)",
                    file_path=file_path,
                    replacements=1,
                    fuzzy=True,
                )
            # 폴백도 실패 — 가장 비슷한 부분을 힌트로 실어, 모델이 다음 시도에서
            # old_string을 바로잡거나 Write 전체 재작성으로 전환하도록 돕는다.
            hint = closest_match_hint(content, old_string)
            return ToolResult.error(
                "old_string을 파일에서 찾을 수 없습니다. "
                "정확한 문자열(공백, 들여쓰기 포함)을 확인해주세요."
                + (f"\n[가장 비슷한 부분]\n{hint}" if hint else "")
                + "\n(팁: Edit이 반복 실패하면 Read로 파일을 다시 읽고, "
                "Write로 파일 전체를 재작성하세요.)"
            )

        # 3단계: 유일성 검증 — replace_all 이 아닌데 2회 이상이면 모호하다.
        # 어디를 바꿀지 알 수 없으므로 실행을 거부하고 대안을 안내한다.
        if not replace_all and count > 1:
            return ToolResult.error(
                f"old_string이 파일에 {count}회 존재합니다. "
                f"더 많은 주변 컨텍스트를 포함하여 유일하게 만들거나, "
                f"replace_all=true를 사용하세요."
            )

        # 4단계: 실제 문자열 치환으로 바뀐 새 내용을 만든다(원본은 아직 유지).
        if replace_all:
            # 모든 발생을 한꺼번에 교체.
            new_content = content.replace(old_string, new_string)
        else:
            # 유일하게 검증된 첫 번째 발생만 교체(세 번째 인자 1 = 최대 1회).
            new_content = content.replace(old_string, new_string, 1)

        # 5단계: 원자적 쓰기로 파일을 교체. 실패 시 원본은 그대로 보존된다.
        try:
            _atomic_write(path, new_content)
        except OSError as e:
            return ToolResult.error(f"파일 쓰기에 실패했습니다: {e}")

        # 실제 교체 건수: replace_all 이면 발견된 count, 아니면 1건.
        logger.info("Edit %s: %d replacements", file_path, count if replace_all else 1)

        replacements = count if replace_all else 1
        # 성공 결과. 메시지와 함께 파일 경로/교체 건수를 메타데이터로 실어 보낸다.
        return ToolResult.success(
            f"편집 완료: {file_path} ({replacements}건 교체)",
            file_path=file_path,
            replacements=replacements,
        )

    # ═══ 7. UI Hints(사용자 인터페이스 표시용 힌트) ═══
    # 진행 상황 표시나 요약 라벨 등, CLI/웹에서 사용자에게 보여줄 문구를 만든다.

    def get_progress_label(self, input_data: dict[str, Any]) -> str:
        # 실행 중 표시할 라벨. 예: "Editing /path/to/file".
        return f"Editing {input_data.get('file_path', '...')}"

    def get_input_summary(self, input_data: dict[str, Any]) -> str:
        # 도구 호출을 한 줄로 요약할 때 쓰는 값 — 여기서는 편집 대상 경로.
        return input_data.get("file_path", "")


# ─────────────────────────────────────────────
# 유틸리티 함수
# ─────────────────────────────────────────────
def _norm_line(line: str) -> str:
    """줄 하나의 공백을 정규화한다 — 앞뒤 공백 제거 + 내부 연속 공백을 1칸으로.

    "들여쓰기 4칸 vs 2칸", "탭 vs 스페이스", "줄 끝 공백" 같은 차이를 전부
    흡수해 "내용이 같은 줄"인지만 비교할 수 있게 만든다.
    """
    return " ".join(line.split())


def find_whitespace_fuzzy_span(content: str, old_string: str) -> tuple[int, int] | None:
    """공백 정규화 기준으로 old_string과 일치하는 유일한 구간의 원본 오프셋을 찾는다.

    동작 방식(줄 단위 슬라이딩 윈도우):
      1. content와 old_string을 줄로 나누고 각 줄을 _norm_line으로 정규화한다.
      2. old_string의 줄 수만큼의 윈도우를 content 위에서 한 줄씩 밀며,
         정규화된 줄들이 전부 일치하는 위치를 찾는다.
      3. 일치 위치가 "정확히 1곳"일 때만 그 구간의 (시작, 끝) 문자 오프셋을
         반환한다. 0곳이거나 2곳 이상(모호)이면 None — fail-closed.

    반환 오프셋 규약: 시작은 첫 줄의 시작 위치, 끝은 마지막 줄의 개행 "직전"
    위치다. 즉 교체 시 개행 구조는 그대로 보존된다.

    Edit(정확 매칭 실패 시 폴백)과 MultiEdit이 공유하는 공용 헬퍼다.
    """
    old_lines = [_norm_line(line) for line in old_string.splitlines()]
    # 정규화 결과가 전부 빈 줄이면(공백뿐인 old_string) 오매칭 위험만 크므로 포기.
    if not old_lines or all(not line for line in old_lines):
        return None

    # content를 줄로 나누되, 각 줄의 시작 오프셋을 함께 기록한다(교체 위치 계산용).
    lines: list[str] = content.splitlines()
    offsets: list[int] = []
    pos = 0
    for line in lines:
        offsets.append(pos)
        pos += len(line) + 1  # +1 = 개행 문자(\n). 마지막 줄은 개행이 없어도 무해.

    norm_lines = [_norm_line(line) for line in lines]
    window = len(old_lines)

    matches: list[int] = []  # 일치하는 윈도우의 시작 줄 번호들
    for i in range(len(lines) - window + 1):
        if norm_lines[i : i + window] == old_lines:
            matches.append(i)
            if len(matches) > 1:
                return None  # 2곳 이상 — 모호하므로 즉시 포기(fail-closed)

    if len(matches) != 1:
        return None

    start_line = matches[0]
    end_line = start_line + window - 1
    start = offsets[start_line]
    end = offsets[end_line] + len(lines[end_line])  # 마지막 줄 개행 직전까지
    return (start, end)


def closest_match_hint(
    content: str, old_string: str, context_lines: int = 3, max_chars: int = 500
) -> str:
    """old_string과 가장 비슷한 파일 부분을 찾아 힌트 문자열로 만든다.

    왜 필요한가: "찾을 수 없습니다"만 돌려주면 모델이 같은 실수를 반복하다
    포기한다(실측). 실제 파일에서 가장 근접한 원문을 보여주면 다음 시도에서
    old_string을 바로잡을 수 있다.

    구현: old_string의 첫 번째 비어있지 않은(정규화) 줄을 앵커로 삼아,
    difflib.get_close_matches로 가장 비슷한 파일 줄을 찾고 그 주변
    ±context_lines 줄의 "원문"(공백 그대로)을 돌려준다. 못 찾으면 빈 문자열.
    """
    anchor = next(
        (line for line in (_norm_line(x) for x in old_string.splitlines()) if line),
        "",
    )
    if not anchor:
        return ""

    lines = content.splitlines()
    norm_lines = [_norm_line(line) for line in lines]
    close = difflib.get_close_matches(anchor, norm_lines, n=1, cutoff=0.5)
    if not close:
        return ""

    idx = norm_lines.index(close[0])
    lo = max(0, idx - context_lines)
    hi = min(len(lines), idx + context_lines + 1)
    snippet = "\n".join(lines[lo:hi])
    if len(snippet) > max_chars:
        snippet = snippet[:max_chars] + "…"
    return snippet


def _atomic_write(path: Path, content: str) -> None:
    """
    파일을 "원자적으로" 안전하게 덮어쓰는 헬퍼.

    [왜 이렇게 하나?]
    파일에 직접 이어 쓰는 도중 프로그램이 죽거나 디스크가 꽉 차면,
    원본이 절반만 쓰인 채 망가질 수 있다. 이를 막기 위해:
      1) 대상과 같은 폴더에 임시 파일을 만들어 새 내용을 전부 쓰고,
      2) os.replace 로 임시 파일을 원본 자리로 한 번에 옮긴다.
    os.replace 는 (같은 볼륨에서) 원자적 연산이라 "전부 성공" 또는
    "전부 실패" 만 존재한다. 같은 폴더에 만드는 이유도 볼륨을 맞춰
    원자성을 보장하기 위함이다.

    매개변수:
      path    : 최종적으로 덮어쓸 대상 파일 경로.
      content : 파일에 저장할 새 문자열 내용(UTF-8 로 인코딩해 기록).
    반환값: 없음. 실패 시 OSError 를 그대로 올려보낸다(호출부가 처리).
    """
    # tmp_fd/tmp_path 는 정리(finally)에서 참조하므로 미리 None 으로 초기화.
    # "아직 정리할 것이 없음"을 None 으로 표시하는 방식이다.
    tmp_fd = None
    tmp_path = None
    try:
        # 대상과 같은 폴더에 임시 파일 생성(원자적 교체를 위해 같은 볼륨 필요).
        # prefix/suffix 로 어떤 파일의 임시본인지 알아보기 쉽게 이름을 짓는다.
        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=str(path.parent),
            prefix=f".{path.name}.",
            suffix=".tmp",
        )
        # 새 내용을 UTF-8 바이트로 임시 파일에 기록한 뒤 닫는다.
        os.write(tmp_fd, content.encode("utf-8"))
        os.close(tmp_fd)
        # 닫았으므로 finally 에서 다시 닫지 않도록 None 으로 표시.
        tmp_fd = None

        # 임시 파일을 원본 자리로 원자적으로 이동 = 실제 교체가 일어나는 순간.
        os.replace(tmp_path, str(path))
        # 교체 성공 — 임시 파일은 이미 원본이 되었으니 정리 대상에서 제외.
        tmp_path = None
    finally:
        # 아래는 예외로 중간에 빠져나왔을 때 남은 자원을 반드시 정리하는 부분.
        # 열린 채 남은 파일 디스크립터가 있으면 닫는다.
        if tmp_fd is not None:
            os.close(tmp_fd)
        # 교체까지 못 가고 남은 임시 파일이 있으면 삭제해 쓰레기를 남기지 않는다.
        if tmp_path is not None:
            try:
                os.unlink(tmp_path)
            except OSError:
                # 정리 실패는 치명적이지 않으므로 조용히 무시한다.
                pass
