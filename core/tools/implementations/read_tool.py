"""
Read 도구 — 로컬 파일 읽기 구현.

이 파일은 Nexus 도구 시스템의 24개 도구 중 하나인 "Read" 도구를 구현한다.
모델(에이전트)이 로컬 파일 시스템의 텍스트 파일을 읽고 싶을 때 호출하며,
파일 내용을 `cat -n`처럼 각 줄 앞에 라인 번호를 붙인 형태로 돌려준다.

핵심 특징:
  - 바이너리 파일 감지: NULL 바이트 비율을 보고 텍스트가 아니면 크기만 알려준다.
  - 한국어 인코딩 폴백: utf-8 실패 시 euc-kr, cp949, latin-1 순으로 시도한다.
  - offset/limit: 큰 파일에서 원하는 라인 구간만 잘라 읽을 수 있다.
  - 읽기 전용이라 안전하므로 동시 실행(병렬) 가능으로 완화한다.

구성 요소:
  - ReadTool 클래스: BaseTool ABC를 상속한 도구 본체.
  - 모듈 하단 유틸 함수 3개: _is_binary, _decode_with_fallback, _format_size.

의존: core.tools.base(BaseTool, PermissionResult 등 표준 계약 타입).
읽은 시각을 context.read_file_timestamps에 기록하여 Write/Edit 도구의
"읽은 뒤 바뀌었는지" 충돌 감지에 활용된다.

작성자: 이현수 / 작성일: 2026-07-05
"""

from __future__ import annotations

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

# 이 도구 전용 로거. 규칙상 "nexus.{module}" 네임스페이스를 사용한다.
logger = logging.getLogger("nexus.tools.read")

# 바이너리 판정을 위해 파일 앞부분에서 읽어볼 바이트 수(8KB).
# 전체를 다 검사하지 않고 앞 8KB 표본만 봐서 빠르게 판단한다.
_BINARY_CHECK_SIZE = 8192
# NULL 바이트 비율 임계값. 표본 중 1%를 넘는 NULL(\x00)이 있으면
# 텍스트가 아니라 바이너리 파일로 간주한다. (텍스트에는 NULL이 거의 없음)
_NULL_THRESHOLD = 0.01  # 1% 이상 NULL 바이트가 있으면 바이너리로 판정

# 텍스트 디코딩을 시도할 인코딩 순서.
# utf-8(표준) → euc-kr, cp949(한국어 레거시) → latin-1 순으로 시도한다.
# latin-1은 모든 바이트를 매핑할 수 있어 절대 실패하지 않는 최종 안전망이다.
_ENCODING_FALLBACKS = ("utf-8", "euc-kr", "cp949", "latin-1")


class ReadTool(BaseTool):
    """
    파일 읽기 도구(BaseTool 구현체).

    지정한 파일을 읽어서 각 줄 앞에 라인 번호를 붙여 반환한다.
    offset(시작 라인)과 limit(읽을 줄 수)으로 원하는 구간만 읽을 수 있어
    수천 줄짜리 큰 파일도 일부만 잘라서 안전하게 볼 수 있다.

    BaseTool이 요구하는 다섯 축(Identity/Schema/Behavior/Permission/Lifecycle)을
    아래에 순서대로 구현한다. 각 섹션은 ═══ 구분선으로 나눠 두었다.
    """

    # ═══ 1. Identity(도구 식별 정보) ═══
    # 레지스트리 등록·모델 노출에 쓰이는 이름/설명/그룹을 정의한다.

    @property
    def name(self) -> str:
        # 모델이 tool_calls에서 사용하는 고유 이름. 변경하면 호출이 깨진다.
        return "Read"

    @property
    def description(self) -> str:
        # 모델에게 노출되는 한 줄 설명(도구 선택 판단 근거).
        return "Read file contents with optional line range."

    @property
    def group(self) -> str:
        # UI 그룹핑/분류용 카테고리. 파일 시스템 계열 도구로 묶는다.
        return "filesystem"

    # ═══ 2. Schema(입력 JSON 스키마) ═══
    # 모델이 넘겨야 하는 인자의 형태·기본값·필수 여부를 JSON Schema로 정의한다.

    @property
    def input_schema(self) -> dict[str, Any]:
        # file_path만 필수이며, offset/limit은 생략 시 기본값(0, 2000)이 적용된다.
        return {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Absolute file path",
                },
                "offset": {
                    "type": "integer",
                    "description": "Start line number (0-based)",
                    "default": 0,
                    "minimum": 0,
                },
                "limit": {
                    "type": "integer",
                    "description": "Number of lines to read",
                    "default": 2000,
                    "minimum": 1,
                },
            },
            "required": ["file_path"],
        }

    # ═══ 3. Behavior Flags(동작 특성 플래그) ═══
    # BaseTool의 기본값은 fail-closed(가장 제한적)이다.
    # Read는 파일을 건드리지 않는 읽기 전용이라 안전하므로 두 플래그를 완화한다.

    @property
    def is_read_only(self) -> bool:
        # 파일 시스템 상태를 바꾸지 않음 → 읽기 전용 True.
        return True

    @property
    def is_concurrency_safe(self) -> bool:
        # 부작용이 없으므로 여러 Read를 병렬 실행해도 안전 → True.
        return True

    # ═══ 5. Lifecycle(생명주기 메서드) ═══
    # validate_input → check_permissions → call 순서로 실행된다.

    def validate_input(self, input_data: dict[str, Any]) -> str | None:
        """
        실제 파일을 열기 전에 입력값을 가볍게 검증한다.

        여기서는 file_path가 비었거나 공백뿐인지만 확인한다.
        문제가 있으면 사용자에게 보여줄 오류 메시지(str)를 반환하고,
        정상이면 None을 반환하여 다음 단계로 진행하게 한다.
        """
        file_path = input_data.get("file_path", "")
        # 빈 문자열이거나 공백만 있으면 읽을 대상이 없으므로 거부한다.
        if not file_path or not file_path.strip():
            return "file_path는 비어 있을 수 없습니다."
        return None

    async def check_permissions(
        self,
        input_data: dict[str, Any],
        context: ToolUseContext,
    ) -> PermissionResult:
        """
        이 도구 자체의 권한 판단. 읽기 전용이므로 항상 ALLOW를 반환한다.

        주의: 여기서 ALLOW를 줘도 실제 접근 차단은 상위 PermissionPipeline의
        다른 레이어(경로 검증 등)에서 이루어진다. 이 메서드는 "도구 관점의
        기본 판단"만 담당하며 권한 레이어를 대체하지 않는다.
        """
        return PermissionResult(behavior=PermissionBehavior.ALLOW)

    async def call(
        self,
        input_data: dict[str, Any],
        context: ToolUseContext,
    ) -> ToolResult:
        """
        도구의 실제 본체. 파일을 읽어 cat -n 형식(라인 번호 + 탭 + 내용)으로 반환한다.

        매개변수:
          input_data — 모델이 넘긴 인자 dict. file_path(필수),
                       offset(시작 라인, 0-기반), limit(읽을 줄 수)을 담는다.
          context    — 실행 컨텍스트. 읽은 파일의 수정 시각을 기록해 두는
                       read_file_timestamps 딕셔너리를 제공한다.

        반환: ToolResult. 성공 시 라인 번호가 붙은 텍스트를, 실패 시 오류 메시지를 담는다.

        처리 순서:
          1. 파일 존재 여부 확인 (없거나 디렉토리면 에러)
          2. 앞부분 8KB로 바이너리 파일인지 검사
          3. 인코딩 폴백으로 전체를 텍스트로 디코딩
          4. offset/limit에 맞게 라인 슬라이싱
          5. 각 줄에 라인 번호를 붙여 반환하고 읽은 시각을 기록
        """
        # 모델이 넘긴 인자를 꺼낸다. offset/limit은 없으면 스키마 기본값을 사용.
        file_path = input_data["file_path"]
        offset = input_data.get("offset", 0)
        limit = input_data.get("limit", 2000)

        path = Path(file_path)

        # 1단계: 파일이 실제로 존재하는지, 그리고 디렉토리가 아닌지 확인한다.
        if not path.exists():
            return ToolResult.error(f"파일을 찾을 수 없습니다: {file_path}")

        if not path.is_file():
            # 경로는 있지만 파일이 아닌 경우(예: 디렉토리)를 걸러낸다.
            return ToolResult.error(f"파일이 아닙니다 (디렉토리일 수 있습니다): {file_path}")

        # 2단계: 앞 8KB만 읽어 바이너리 여부를 판정한다.
        # 전체를 읽기 전에 표본으로 먼저 검사해 불필요한 대용량 로드를 피한다.
        try:
            raw_head = path.read_bytes()[:_BINARY_CHECK_SIZE]
        except OSError as e:
            # 권한 문제 등 OS 레벨 읽기 실패는 그대로 오류로 감싸 반환한다.
            return ToolResult.error(f"파일을 읽을 수 없습니다: {e}")

        if _is_binary(raw_head):
            # 바이너리 파일은 텍스트로 보여줄 수 없으므로 크기 정보만 알려준다.
            size = path.stat().st_size
            return ToolResult.success(
                f"바이너리 파일입니다 ({_format_size(size)}). 텍스트로 표시할 수 없습니다.",
                binary=True,
                file_path=file_path,
                size=size,
            )

        # 3단계: 텍스트로 확인됐으니 전체를 읽어 인코딩 폴백으로 디코딩한다.
        raw_bytes = path.read_bytes()
        text = _decode_with_fallback(raw_bytes)

        # 4단계: 줄 단위로 나눈 뒤 offset부터 limit개만큼만 잘라낸다.
        lines = text.splitlines()
        total_lines = len(lines)
        sliced = lines[offset : offset + limit]

        # 5단계: 잘라낸 각 줄 앞에 라인 번호(1-기반)와 탭을 붙인다.
        # enumerate의 start를 offset+1로 줘서 실제 파일 줄 번호와 맞춘다.
        numbered_lines = []
        for i, line in enumerate(sliced, start=offset + 1):
            numbered_lines.append(f"{i}\t{line}")

        result_text = "\n".join(numbered_lines)

        # 요청 구간 뒤에 더 남은 줄이 있으면, 잘렸다는 안내 문구를 덧붙인다.
        if offset + limit < total_lines:
            result_text += (
                f"\n\n... (총 {total_lines}줄 중 {offset + 1}~{offset + len(sliced)}줄 표시)"
            )

        # 방금 읽은 파일의 수정 시각을 컨텍스트에 기록한다.
        # 이후 Write/Edit 도구가 "읽은 뒤 파일이 바뀌었는지"를 이 값으로 판단한다.
        context.read_file_timestamps[file_path] = os.path.getmtime(file_path)

        # 디버깅용 로그. 어떤 파일을 어떤 범위로 읽었는지 남긴다.
        logger.debug(
            "Read %s: offset=%d, limit=%d, total=%d",
            file_path,
            offset,
            limit,
            total_lines,
        )

        # 성공 결과를 반환. 본문 텍스트와 함께 메타데이터(전체/표시 줄 수)를 담는다.
        return ToolResult.success(
            result_text,
            file_path=file_path,
            total_lines=total_lines,
            lines_shown=len(sliced),
        )

    # ═══ 7. UI Hints(사용자 인터페이스 표시용 힌트) ═══
    # 도구 실행 중/후에 CLI·웹 UI가 보여줄 짧은 문구를 만들어 준다.

    def get_progress_label(self, input_data: dict[str, Any]) -> str:
        # 실행 중 진행 표시줄에 뜨는 라벨(예: "Reading /path/to/file").
        return f"Reading {input_data.get('file_path', '...')}"

    def get_input_summary(self, input_data: dict[str, Any]) -> str:
        # 이 도구 호출을 한 줄로 요약. 여기서는 대상 파일 경로만 보여준다.
        return input_data.get("file_path", "")


# ─────────────────────────────────────────────
# 유틸리티 함수 — 클래스 밖의 순수 함수들.
# 상태를 갖지 않아 테스트하기 쉽고, call() 내부에서 재사용된다.
# ─────────────────────────────────────────────
def _is_binary(data: bytes) -> bool:
    """
    주어진 바이트 표본이 바이너리인지 판별한다.

    판단 기준: 전체 바이트 중 NULL(\\x00)의 비율이 임계값(_NULL_THRESHOLD)을
    넘으면 바이너리로 본다. 일반 텍스트에는 NULL이 거의 없고, 실행 파일·이미지
    등 바이너리에는 NULL이 흔하다는 점을 이용한 간단하고 빠른 휴리스틱이다.
    빈 데이터는 텍스트(False)로 취급한다.
    """
    if not data:
        return False
    null_count = data.count(b"\x00")
    return (null_count / len(data)) > _NULL_THRESHOLD


def _decode_with_fallback(data: bytes) -> str:
    """
    바이트를 여러 인코딩으로 차례로 시도해 문자열로 디코딩한다.

    utf-8을 먼저 시도하고, 실패하면 한국어 레거시 인코딩인 euc-kr, cp949를
    시도한다. 마지막 latin-1은 모든 바이트 값을 매핑할 수 있어 절대 실패하지
    않으므로 최종 안전망 역할을 한다. 따라서 어떤 입력이 들어와도 문자열을
    돌려주며, 디코딩 실패로 예외가 밖으로 새어 나가지 않는다.
    """
    for encoding in _ENCODING_FALLBACKS:
        try:
            return data.decode(encoding)
        except (UnicodeDecodeError, LookupError):
            # 해당 인코딩으로 실패하면 다음 후보로 넘어간다.
            continue
    # 루프 안에서 latin-1이 항상 성공하므로 여기까진 오지 않지만,
    # 만약을 대비한 명시적 안전장치로 한 번 더 latin-1 디코딩을 시도한다.
    return data.decode("latin-1")


def _format_size(size: int) -> str:
    """
    바이트 수를 사람이 읽기 좋은 단위(B/KB/MB/GB/TB)로 변환한다.

    1024 미만이 될 때까지 단위를 한 칸씩 올리며 나눈다. 바이트(B) 단위는
    소수점이 무의미하므로 정수로, 그 이상은 소수점 한 자리로 표시한다.
    (예: 512 → "512B", 2048 → "2.0KB")
    """
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.1f}{unit}" if unit != "B" else f"{size}{unit}"
        size /= 1024
    # GB까지 넘어설 만큼 큰 값은 TB로 표시한다.
    return f"{size:.1f}TB"
