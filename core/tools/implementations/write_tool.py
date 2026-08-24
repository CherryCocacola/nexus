"""
Write 도구 — 파일 쓰기 (원자적 저장 구현).

이 파일은 Nexus의 24개 도구 중 하나인 "Write" 도구를 정의한다.
모델(에이전트)이 새 파일을 만들거나 기존 파일을 통째로 덮어쓸 때 사용한다.

핵심 특징 — "원자적 쓰기(atomic write)":
  - 대상 경로에 바로 쓰지 않고, 같은 디렉토리에 임시 파일을 먼저 만들어 내용을 쓴 뒤
    os.replace로 한 번에 교체한다.
  - 이렇게 하면 쓰기 도중 프로세스가 죽거나 정전이 나도, 대상 파일이 반쯤 쓰이다 만
    "손상된 상태"로 남지 않는다. 교체는 성공 아니면 실패, 둘 중 하나만 발생한다.
  - 부모 디렉토리가 없으면 자동으로 만들어 준다.

주요 클래스:
  - WriteTool: BaseTool을 상속한 도구 구현체. registry에 "Write"라는 이름으로 등록된다.

의존:
  - core.tools.base 의 BaseTool(추상 클래스)과 권한/결과 관련 타입들을 사용한다.
  - 표준 라이브러리 os, tempfile, pathlib로 실제 파일 입출력을 수행한다.

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
from core.tools.validation.syntax_validator import rejection_message, syntax_error

# 이 모듈 전용 로거. "nexus.tools.write" 네임스페이스로 로그를 남겨
# 전체 로깅 설정에서 도구별로 필터링/레벨 조정이 가능하도록 한다.
logger = logging.getLogger("nexus.tools.write")


class WriteTool(BaseTool):
    """
    파일 쓰기 도구.

    지정한 절대 경로에 내용을 원자적으로 쓴다. 기존 파일이 있으면 덮어쓰고,
    부모 디렉토리가 없으면 자동으로 생성한다.

    BaseTool의 계약에 따라 아래 네 가지 축을 구현한다.
      1. Identity  — 도구 이름/설명/그룹 (레지스트리 등록·모델 노출용)
      2. Schema    — 입력 JSON Schema (모델이 어떤 인자를 줘야 하는지)
      3. Behavior  — 동작 성격 플래그 (fail-closed 기본값, 여기선 파괴적 쓰기만 표시)
      5. Lifecycle — validate_input → check_permissions → call 순으로 실제 실행
      7. UI Hints  — CLI/웹에서 진행 상황을 표시하기 위한 짧은 라벨
    """

    # ═══ 1. Identity ═══

    @property
    def name(self) -> str:
        # 레지스트리에 등록되는 도구의 고유 이름. 모델은 이 이름으로 도구를 호출한다.
        return "Write"

    @property
    def description(self) -> str:
        # 모델에게 노출되는 설명. 도구 목록/프롬프트에 들어간다.
        #
        # 2026-08-07: "기존 파일은 Edit 를 쓰라"를 설명에 넣었다.
        #   실측에서 모델이 파일 전문을 Write 인자로 넘기다 내용이 손상됐다
        #   (단어 중간 공백, 줄바꿈 소실 → SyntaxError). 긴 리터럴을 통째로
        #   재현하는 것이 그 자체로 취약한 동작이라, 조각 치환이 더 안전하다.
        #   구문 검사가 .py/.json 손상은 잡지만 .md·.txt 는 잡을 수 없으므로,
        #   애초에 통짜 쓰기를 덜 하도록 유도한다.
        return (
            "Create a new file, or overwrite one completely. "
            "To change part of an existing file, prefer Edit — rewriting a whole "
            "file risks corrupting content that was already correct."
        )

    @property
    def group(self) -> str:
        # 도구 분류 그룹. Read/Edit 등과 함께 "filesystem" 계열로 묶인다.
        return "filesystem"

    # ═══ 2. Schema ═══

    @property
    def input_schema(self) -> dict[str, Any]:
        # 모델이 이 도구를 호출할 때 넘겨야 하는 입력의 JSON Schema.
        # file_path와 content 두 개가 모두 필수이며, 둘 다 문자열이어야 한다.
        return {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Absolute file path",
                },
                "content": {
                    "type": "string",
                    "description": "File content to write",
                },
            },
            "required": ["file_path", "content"],
        }

    # ═══ 3. Behavior Flags ═══
    # 쓰기 도구 — fail-closed 기본값 유지, is_destructive만 True로 설정
    # (읽기전용/동시성안전/확인필요 등은 BaseTool의 가장 제한적인 기본값을 그대로 사용한다.)

    @property
    def is_destructive(self) -> bool:
        # 기존 파일을 덮어써 되돌릴 수 없으므로 "파괴적" 도구로 표시한다.
        # 이 플래그는 권한/UI 쪽에서 위험 도구를 구분하는 데 쓰인다.
        return True

    # ═══ 5. Lifecycle ═══

    def validate_input(self, input_data: dict[str, Any]) -> str | None:
        """
        입력값 사전 검증.

        file_path가 비어 있거나 공백뿐이면 오류 메시지 문자열을 돌려주고,
        문제가 없으면 None을 돌려준다. None이 반환돼야 다음 단계로 넘어간다.
        (content는 빈 문자열도 허용되므로 여기서 검사하지 않는다.)
        """
        file_path = input_data.get("file_path", "")
        if not file_path or not file_path.strip():
            return "file_path는 비어 있을 수 없습니다."
        return None

    async def check_permissions(
        self,
        input_data: dict[str, Any],
        context: ToolUseContext,
    ) -> PermissionResult:
        """
        권한 판정.

        Write는 파일을 변경하는 파괴적 도구이므로 항상 사용자에게 확인을 요청한다
        (ASK). 상위 권한 파이프라인이 이 결과를 받아 모드에 따라 최종 처리한다.
        """
        file_path = input_data.get("file_path", "")
        return PermissionResult(
            behavior=PermissionBehavior.ASK,
            message=f"Write to: {file_path}",
        )

    async def call(
        self,
        input_data: dict[str, Any],
        context: ToolUseContext,
    ) -> ToolResult:
        """
        파일을 원자적으로 쓴다.

        "임시 파일에 먼저 쓰고 os.replace로 교체" 하는 방식으로, 쓰기 도중 문제가
        생겨도 대상 파일이 손상되지 않도록 보장한다.

        처리 순서:
          1. 부모 디렉토리 생성 (없으면)
          2. 같은 디렉토리에 임시 파일 생성 후 내용 쓰기
          3. os.replace로 원자적 교체 (같은 파일시스템이어야 원자적)
          4. 실패 시 임시 파일 정리

        반환: 성공 시 작성한 줄 수/바이트 수를 담은 ToolResult.success,
             실패 시 사유 메시지를 담은 ToolResult.error.
        """
        file_path = input_data["file_path"]
        content = input_data["content"]

        path = Path(file_path)

        # 0단계: 구문 검사 — 디스크에 닿기 전에 막는다.
        #   깨진 내용을 그대로 저장하고 성공이라 보고하면, 기존 파일은 이미 사라지고
        #   아무도 손상을 모른다. 여기서 걸면 모델이 그 자리에서 다시 쓴다.
        detail = syntax_error(file_path, content)
        if detail is not None:
            logger.warning("Write 거부(구문 오류) %s: %s", file_path, detail)
            return ToolResult.error(rejection_message(detail))

        # 1단계: 부모 디렉토리 자동 생성
        # parents=True로 중간 경로까지 한 번에 만들고, exist_ok=True로 이미
        # 존재해도 예외를 내지 않는다. 권한/디스크 문제 등은 OSError로 잡아 보고한다.
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            return ToolResult.error(f"디렉토리를 생성할 수 없습니다: {e}")

        # 1.5단계: 무변화 감지 (2026-08-23)
        # 왜 필요한가(실측): 모델이 **똑같은 내용을 11회 다시 쓰며** 30턴을 공전한
        # 사고가 있었다. 매번 "파일을 작성했습니다"가 성공으로 돌아오니 모델 입장에서
        # 진전이 없다는 단서가 어디에도 없었다. 결과 문구로 그 사실을 알려 주면
        # 모델이 다음 턴에 다른 행동을 고를 수 있다(in-band 신호).
        #
        # 오케스트레이터에 상태를 더하지 않는 것이 핵심이다 — pytest 반복 실행처럼
        # 같은 호출이 정상인 워크플로와 충돌하지 않는다. 여기서는 사실만 보고한다.
        #
        # 쓰기를 막지는 않는다. 내용이 같으면 결과도 같으므로 해가 없고, 파일이
        # 실제로 존재하게 만드는 것이 이 도구의 계약이기 때문이다.
        unchanged = False
        if path.exists():
            try:
                unchanged = path.read_text(encoding="utf-8") == content
            except (OSError, UnicodeDecodeError):
                # 읽기 실패는 무시한다 — 이 비교는 안내용이라 쓰기를 막으면 안 된다.
                unchanged = False

        # 2단계: 임시 파일에 먼저 쓰기 (원자적 쓰기 보장)
        # tmp_fd(파일 디스크립터), tmp_path(임시 파일 경로)를 미리 None으로 두어,
        # finally 정리 블록에서 "아직 안 만들어졌는지 / 이미 정리됐는지"를 판별한다.
        tmp_fd = None
        tmp_path = None
        try:
            # 같은 디렉토리에 임시 파일 생성 (os.replace 원자성 보장)
            # os.replace는 동일 파일시스템 안에서만 원자적이므로, 대상과 같은
            # 디렉토리(path.parent)에 임시 파일을 만드는 것이 핵심이다.
            tmp_fd, tmp_path = tempfile.mkstemp(
                dir=str(path.parent),
                prefix=f".{path.name}.",
                suffix=".tmp",
            )
            # 파일 디스크립터로 직접 쓰기
            # 내용을 UTF-8 바이트로 변환해 그대로 기록한다.
            os.write(tmp_fd, content.encode("utf-8"))
            os.close(tmp_fd)
            tmp_fd = None  # close 완료 표시

            # 3단계: 원자적 교체
            # 임시 파일을 대상 경로로 옮겨 덮어쓴다. 이 한 번의 호출로 교체가
            # 완결되므로 중간에 깨진 파일이 남지 않는다.
            os.replace(tmp_path, file_path)
            tmp_path = None  # replace 완료 표시

        except OSError as e:
            return ToolResult.error(f"파일 쓰기에 실패했습니다: {e}")
        finally:
            # 4단계: 실패 시 정리
            # 위 try 도중 예외가 나면 열린 디스크립터나 남은 임시 파일이 있을 수
            # 있으므로 여기서 확실히 닫고 지운다. 성공 시엔 None이라 아무 일도 안 한다.
            if tmp_fd is not None:
                os.close(tmp_fd)
            if tmp_path is not None:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    # 임시 파일 삭제 실패는 치명적이지 않으므로 조용히 넘어간다.
                    pass

        # 결과 통계 계산
        # 줄 수: 개행 문자 개수에, 마지막 줄이 개행으로 끝나지 않으면 1을 더한다
        # (빈 내용은 0줄). 바이트 수: UTF-8 인코딩 기준 실제 크기.
        line_count = content.count("\n") + (1 if content and not content.endswith("\n") else 0)
        byte_count = len(content.encode("utf-8"))

        # 감사/디버깅용 로그. 어떤 파일에 몇 줄·몇 바이트를 썼는지 기록한다.
        logger.info("Write %s: %d lines, %d bytes", file_path, line_count, byte_count)

        # 성공 결과 반환. 사람이 읽을 요약 문자열과 함께, 후속 처리에 쓸 수 있도록
        # file_path/lines/bytes를 메타데이터로 함께 담아 준다.
        # 무변화면 그 사실을 문구에 실어 준다. 모델이 "썼다"만 보고 진전이 있다고
        # 오인한 채 같은 내용을 반복하는 것을 막기 위한 유일한 단서다.
        summary = f"파일을 작성했습니다: {file_path} ({line_count}줄, {byte_count}바이트)"
        if unchanged:
            summary += " — 기존 내용과 동일합니다(변경 없음). 같은 내용을 다시 쓰지 마세요."

        return ToolResult.success(
            summary,
            file_path=file_path,
            lines=line_count,
            bytes=byte_count,
            unchanged=unchanged,
        )

    # ═══ 7. UI Hints ═══

    def get_progress_label(self, input_data: dict[str, Any]) -> str:
        # 도구 실행 중 CLI/웹에 표시할 진행 라벨 (예: "Writing /path/to/file").
        return f"Writing {input_data.get('file_path', '...')}"

    def get_input_summary(self, input_data: dict[str, Any]) -> str:
        # 도구 호출을 한눈에 보여줄 때 쓰는 짧은 요약. 여기선 대상 파일 경로.
        return input_data.get("file_path", "")
