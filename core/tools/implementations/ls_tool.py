"""
LS 도구 — 지정한 디렉토리의 파일/폴더 목록을 사람이 읽기 좋은 텍스트로 반환한다.

[이 파일이 하는 일]
사용자(또는 모델)가 넘긴 절대 경로의 디렉토리를 열어, 그 안의 항목들을
"디렉토리 먼저 → 이름순"으로 정렬한 뒤, 각 항목의 유형(d/-), 수정 시각,
크기를 정렬된 표 형태의 문자열로 만들어 ToolResult로 돌려준다.
Unix의 `ls -l` 과 비슷한 결과를 Nexus 도구 규격에 맞춰 구현한 것이다.

[주요 구성 요소]
  - LSTool          : BaseTool ABC를 상속한 실제 도구 구현체 (모델에 노출되는 "LS")
  - _DirEntry       : 항목 하나의 메타데이터(이름/유형/크기/수정시각)를 담는 내부 클래스
  - _format_size    : 바이트 정수를 B/KB/MB/GB/TB 표기로 변환하는 헬퍼
  - _format_mtime   : POSIX timestamp를 "YYYY-MM-DD HH:MM" 문자열로 변환하는 헬퍼

[도구 성격]
디스크를 읽기만 하고 아무것도 변경하지 않으므로 읽기 전용(is_read_only=True)이며,
동시에 여러 번 호출해도 부작용이 없어 병렬 실행 안전(is_concurrency_safe=True)으로
완화한다. Fail-closed 기본값(둘 다 False)을 이 도구에 한해 명시적으로 푸는 것이다.

[의존/연동]
core.tools.base 의 BaseTool·ToolResult·Permission 계약을 따르며, Registry에
등록되면 도구 실행 파이프라인(executor)이 validate_input → check_permissions →
call 순서로 호출한다.

작성자: 이현수 / 작성일: 2026-07-05
"""

from __future__ import annotations

import logging
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from core.tools.base import (
    BaseTool,
    PermissionBehavior,
    PermissionResult,
    ToolResult,
    ToolUseContext,
)

# 이 모듈 전용 로거. 프로젝트 규칙상 "nexus.{module}" 네이밍을 사용한다.
# 목록 조회 결과 요약(dir/file 개수)을 debug 레벨로 남긴다.
logger = logging.getLogger("nexus.tools.ls")


class LSTool(BaseTool):
    """
    디렉토리 목록 표시 도구 (모델에 "LS"라는 이름으로 노출).

    한 번의 호출로 대상 디렉토리 바로 아래(1단계)의 파일/폴더를 훑어,
    각 항목의 이름·유형·크기·수정 시각을 정렬된 텍스트로 만들어 반환한다.
    하위 디렉토리를 재귀적으로 파고들지는 않는다 (얕은 목록만 제공).

    BaseTool의 표준 수명주기(validate_input → check_permissions → call)를
    구현하며, 아래 코드는 크게 다음 5개 영역으로 나뉜다:
      1. Identity   — 도구 이름/설명/그룹 (모델 프롬프트에 실리는 메타데이터)
      2. Schema     — 입력 인자 JSON Schema (여기서는 path 하나만 받음)
      3. Behavior   — 읽기전용/병렬안전 등 동작 플래그
      5. Lifecycle  — 실제 검증·권한·실행 로직
      7. UI Hints   — CLI/웹에서 진행 상황을 보여줄 라벨 문자열
    """

    # ═══ 1. Identity ═══
    # 도구의 신원 정보. Registry는 name으로 도구를 찾고, description은
    # 모델이 "언제 이 도구를 쓸지" 판단하는 근거로 시스템 프롬프트에 실린다.

    @property
    def name(self) -> str:
        # Registry 등록 키이자 모델이 tool_calls에서 지정하는 고유 이름.
        return "LS"

    @property
    def description(self) -> str:
        # 모델에게 보여줄 한 줄 설명. 모델 프롬프트는 영어 기준이라 영문으로 둔다.
        return "List directory contents."

    @property
    def group(self) -> str:
        # 도구 분류 태그. 파일시스템 계열 도구(Read/Write/Glob 등)와 함께 묶인다.
        return "filesystem"

    # ═══ 2. Schema ═══
    # 모델이 이 도구를 호출할 때 넘겨야 하는 인자의 JSON Schema.
    # 여기서는 "path"(절대 디렉토리 경로) 하나만 필수로 받는다.

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute directory path",
                },
            },
            "required": ["path"],
        }

    # ═══ 3. Behavior Flags ═══
    # BaseTool의 동작 플래그는 fail-closed(가장 안전한 값)가 기본이다.
    # LS는 디스크를 읽기만 하고 변경이 없으므로 아래 두 플래그를 명시적으로 완화한다.

    @property
    def is_read_only(self) -> bool:
        # 파일 시스템을 변경하지 않는다 → 권한 파이프라인에서 READONLY로 취급된다.
        return True

    @property
    def is_concurrency_safe(self) -> bool:
        # 부작용이 없어 여러 LS 호출을 동시에 실행해도 안전하다 → 병렬 실행 허용.
        return True

    # ═══ 5. Lifecycle ═══
    # 도구 실행 파이프라인이 호출하는 3단계 훅. 순서는 항상
    # validate_input(형식 검증) → check_permissions(권한) → call(실제 실행) 이다.

    def validate_input(self, input_data: dict[str, Any]) -> str | None:
        """
        입력 형식 검증 훅. path 인자가 실질적으로 비어 있지 않은지만 확인한다.

        여기서는 "존재하는 경로인지"까지는 보지 않는다. 존재 여부·디렉토리 여부는
        실제 디스크 접근이 필요하므로 call() 안에서 검사한다. 이 단계는 값이
        아예 없거나 공백뿐인 명백한 잘못된 호출을 빠르게 걸러내는 역할만 한다.

        반환:
          - 문제가 있으면 사용자에게 보여줄 오류 메시지(str)
          - 정상이면 None (검증 통과를 의미)
        """
        path = input_data.get("path", "")
        # 값이 없거나(falsy) 앞뒤 공백을 제거하면 빈 문자열인 경우를 모두 거른다.
        if not path or not path.strip():
            return "path는 비어 있을 수 없습니다."
        return None

    async def check_permissions(
        self,
        input_data: dict[str, Any],
        context: ToolUseContext,
    ) -> PermissionResult:
        """
        권한 판정 훅. LS는 읽기 전용이라 경로에 상관없이 항상 실행을 허용한다.

        쓰기/삭제 계열 도구라면 여기서 경로 검증이나 사용자 확인(ASK)을 걸지만,
        목록 조회는 위험이 없어 곧바로 ALLOW를 반환한다. (경로 접근 자체가
        OS 권한으로 막히는 경우는 call() 안에서 PermissionError로 처리한다.)
        """
        return PermissionResult(behavior=PermissionBehavior.ALLOW)

    async def call(
        self,
        input_data: dict[str, Any],
        context: ToolUseContext,
    ) -> ToolResult:
        """
        실제 실행 훅 — 대상 디렉토리를 열어 목록 텍스트를 만들어 반환한다.

        전체 흐름을 5단계로 나눠 처리한다:
          1. 경로가 실제로 존재하는 디렉토리인지 확인 (아니면 오류 반환)
          2. os.scandir로 바로 아래 항목들을 한 번에 읽어들임
          3. 항목마다 유형/크기/수정시각을 stat으로 수집 (_DirEntry로 보관)
          4. "디렉토리 먼저 → 이름순(대소문자 무시)"으로 정렬
          5. 표 형태 텍스트로 포맷팅 + 헤더/요약을 붙여 ToolResult로 반환

        매개변수:
          - input_data : validate_input을 통과한 입력. "path" 키를 반드시 가진다.
          - context    : 도구 실행 컨텍스트(여기서는 직접 사용하지 않지만 계약상 받음).

        반환:
          - 성공 시 ToolResult.success(목록 텍스트, 개수 메타데이터)
          - 실패 시 ToolResult.error(사유 메시지)
        """
        # 입력 경로 문자열을 꺼내고, 조작이 쉬운 pathlib.Path로도 준비한다.
        dir_path = input_data["path"]
        path = Path(dir_path)

        # 1단계: 존재 여부와 "디렉토리인지"를 확인한다.
        #         존재하지 않거나 파일/기타 유형이면 여기서 즉시 오류로 끝낸다.
        if not path.exists():
            return ToolResult.error(f"경로를 찾을 수 없습니다: {dir_path}")
        if not path.is_dir():
            return ToolResult.error(f"디렉토리가 아닙니다: {dir_path}")

        # 2단계: 디렉토리 항목을 읽는다. os.scandir는 이름과 stat 정보를
        #         함께 얻을 수 있어 항목별 stat 재호출 비용을 줄여준다.
        #         결과를 리스트로 소진해 스캐너 핸들을 곧바로 닫는다.
        try:
            entries = list(os.scandir(dir_path))
        except PermissionError:
            # OS 레벨 접근 권한이 없어 디렉토리를 열지 못한 경우.
            return ToolResult.error(f"디렉토리 접근 권한이 없습니다: {dir_path}")
        except OSError as e:
            # 그 외 입출력 오류(경로가 사라짐, 장치 오류 등)를 포괄 처리.
            return ToolResult.error(f"디렉토리를 읽을 수 없습니다: {e}")

        # 항목이 하나도 없으면 빈 디렉토리임을 알리고 total=0으로 성공 반환.
        if not entries:
            return ToolResult.success(
                f"빈 디렉토리입니다: {dir_path}",
                total=0,
            )

        # 3단계: 항목마다 필요한 메타데이터(유형/크기/수정시각)를 수집한다.
        #         정렬·포맷팅에 쓰기 좋도록 가벼운 _DirEntry 객체로 담는다.
        items: list[_DirEntry] = []
        for entry in entries:
            try:
                # stat()으로 크기·수정시각을 얻는다. 디렉토리는 크기 개념이
                # 의미 없으므로 size는 0으로 둔다.
                st = entry.stat()
                is_dir = entry.is_dir()
                size = st.st_size if not is_dir else 0
                mtime = st.st_mtime
                items.append(
                    _DirEntry(
                        name=entry.name,
                        is_dir=is_dir,
                        size=size,
                        mtime=mtime,
                    )
                )
            except OSError:
                # 깨진 심볼릭 링크처럼 stat이 실패하는 항목도 목록에서
                # 빠뜨리지 않는다. error=True로 표시해 "접근 불가"로 노출한다.
                items.append(
                    _DirEntry(
                        name=entry.name,
                        is_dir=False,
                        size=0,
                        mtime=0.0,
                        error=True,
                    )
                )

        # 4단계: 정렬. 튜플 키 (not is_dir, name.lower())를 쓰면
        #         디렉토리(False=0)가 파일(True=1)보다 앞서고, 같은 그룹
        #         안에서는 대소문자 무시 이름순으로 정렬된다.
        items.sort(key=lambda e: (not e.is_dir, e.name.lower()))

        # 5단계: 각 항목을 한 줄 텍스트로 만든다. 유형 기호(d/-/?),
        #         수정시각, 크기, 이름을 열 맞춤(정렬 폭)해 표처럼 보이게 한다.
        lines: list[str] = []
        dir_count = 0   # 디렉토리 개수 집계(요약/메타데이터용)
        file_count = 0  # 파일 개수 집계

        for item in items:
            if item.error:
                # stat 실패 항목: 물음표 기호와 함께 접근 불가로 표시.
                lines.append(f"  ?  {item.name} (접근 불가)")
                continue

            if item.is_dir:
                # 디렉토리: 유형 기호 'd', 이름 끝에 '/'를 붙여 구분한다.
                # 크기 칸은 공백으로 두어 파일 행과 열을 맞춘다.
                mtime_str = _format_mtime(item.mtime)
                lines.append(f"  d  {mtime_str}  {'':>10}  {item.name}/")
                dir_count += 1
            else:
                # 파일: 유형 기호 '-', 사람이 읽기 쉬운 크기와 수정시각을 표시.
                mtime_str = _format_mtime(item.mtime)
                size_str = _format_size(item.size)
                lines.append(f"  -  {mtime_str}  {size_str:>10}  {item.name}")
                file_count += 1

        # 목록 위에 대상 경로 헤더를, 아래에 개수 요약을 붙여 최종 텍스트 완성.
        header = f"디렉토리: {dir_path}"
        summary = f"합계: {dir_count}개 디렉토리, {file_count}개 파일"
        result_text = f"{header}\n\n" + "\n".join(lines) + f"\n\n{summary}"

        # 운영 중 추적용 디버그 로그. 어떤 경로를 몇 개 훑었는지 남긴다.
        logger.debug("LS %s: %d dirs, %d files", dir_path, dir_count, file_count)

        # 사람이 읽을 텍스트와 함께, 프로그램이 쓰기 좋은 개수 메타데이터를 반환.
        return ToolResult.success(
            result_text,
            total=dir_count + file_count,
            directories=dir_count,
            files=file_count,
        )

    # ═══ 7. UI Hints ═══
    # CLI/웹에서 도구 실행 중 보여줄 짧은 안내 문자열. 실행 로직과는 무관하다.

    def get_progress_label(self, input_data: dict[str, Any]) -> str:
        # 실행 중 진행 표시줄 등에 노출할 라벨. 예: "Listing /home/user".
        return f"Listing {input_data.get('path', '...')}"

    def get_input_summary(self, input_data: dict[str, Any]) -> str:
        # 도구 호출을 한눈에 요약할 때 쓰는 문자열(대상 경로만 보여줌).
        return input_data.get("path", "")


# ─────────────────────────────────────────────
# 내부 데이터 클래스
# ─────────────────────────────────────────────
class _DirEntry:
    """
    디렉토리 항목 하나의 메타데이터를 담는 가벼운 내부 전용 클래스.

    call() 안에서 os.scandir 결과를 정렬·포맷팅하기 좋은 형태로 옮겨 담는다.
    __slots__를 지정해 인스턴스마다 __dict__를 만들지 않으므로, 항목이
    수천 개인 디렉토리에서도 메모리 사용과 속성 접근이 효율적이다.

    필드:
      - name   : 항목 이름(파일/폴더 이름)
      - is_dir : 디렉토리이면 True, 파일이면 False
      - size   : 파일 크기(바이트). 디렉토리는 0
      - mtime  : 마지막 수정 시각(POSIX timestamp). 알 수 없으면 0.0
      - error  : stat 실패 등으로 정보를 못 읽은 항목이면 True
    """

    __slots__ = ("name", "is_dir", "size", "mtime", "error")

    def __init__(
        self,
        name: str,
        is_dir: bool,
        size: int,
        mtime: float,
        error: bool = False,
    ) -> None:
        # 전달받은 값을 그대로 슬롯 속성에 저장하는 단순 생성자.
        self.name = name
        self.is_dir = is_dir
        self.size = size
        self.mtime = mtime
        self.error = error


# ─────────────────────────────────────────────
# 유틸리티 함수
# ─────────────────────────────────────────────
def _format_size(size: int) -> str:
    """
    바이트 크기(정수)를 사람이 읽기 쉬운 단위 문자열로 변환한다.

    1024 미만이면 현재 단위로 표기하고, 그 이상이면 1024로 나누며 다음
    단위로 올라간다. B는 소수점 없이(예: "512B"), KB 이상은 소수점 한
    자리로 표기한다(예: "1.5KB"). GB를 넘어서면 마지막에 TB로 표기한다.
    """
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            # B 단위는 정수 그대로, 그 외 단위는 소수 첫째 자리까지 보여준다.
            if unit == "B":
                return f"{size}{unit}"
            return f"{size:.1f}{unit}"
        # 아직 현재 단위로 표기하기엔 크다 → 1024로 나눠 다음 단위로.
        size /= 1024
    # 위 루프(GB까지)를 다 통과할 만큼 큰 값은 TB로 표기한다.
    return f"{size:.1f}TB"


def _format_mtime(mtime: float) -> str:
    """
    수정 시각(POSIX timestamp)을 "YYYY-MM-DD HH:MM" 문자열로 변환한다.

    mtime이 0.0이면 시각을 알 수 없는(예: stat 실패) 항목이므로, 목록의
    열 정렬이 깨지지 않도록 실제 날짜 문자열과 같은 폭(16자)의 공백을
    돌려준다. 시간대는 프로젝트 일관성을 위해 UTC 기준으로 표기한다.
    """
    if mtime == 0.0:
        return "                "  # 16자리 공백 (정렬용)
    dt = datetime.fromtimestamp(mtime, tz=UTC)
    return dt.strftime("%Y-%m-%d %H:%M")
