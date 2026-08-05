# RenderPreview 도구 — 로컬 HTML을 헤드리스 브라우저로 렌더해 스크린샷을 만든다.
"""
RenderPreview 도구 — 생성한 웹 결과물을 "실제로 렌더해서" 눈으로 확인하는 도구.

[왜 필요한가 — 자가 검증 루프의 첫 단추]
  실측(2026-08-04 홈페이지 e2e): 모델이 HTML/CSS를 생성만 하고 렌더 확인을
  하지 않아, 네비게이션 스타일 미적용·카드 세로나열 같은 문제를 스스로 알지
  못했다. 이 도구는 로컬 HTML 파일을 헤드리스 Chrome/Edge로 렌더해 스크린샷
  PNG를 만들고, 그 경로를 돌려준다. 모델은 이어서 AnalyzeImage(비전 VLM)로
  스크린샷을 검토해 레이아웃 문제를 발견·수정하는 루프를 돌 수 있다.

  RenderPreview(렌더→스크린샷) → AnalyzeImage(스크린샷→문제 목록) → Edit/Write(수정)

[에어갭 준수]
  - 브라우저는 "로컬에 설치된" Chrome/Edge/Chromium 실행 파일만 사용한다.
    외부 네트워크 호출이 없다(file:// URL 렌더). 페이지가 CDN을 참조하면
    에어갭에서는 로드되지 않은 모습 그대로 찍힌다 — 이는 버그가 아니라
    "에어갭 배포에서 이 페이지가 어떻게 보일지"의 정직한 미리보기다.
  - 브라우저가 없는 환경(서버 컨테이너 등)에서는 명확한 오류 메시지로
    비활성임을 알린다(fail-soft — 도구 부재가 다른 작업을 막지 않는다).

[스크린샷 저장 위치 — 업로드 샌드박스]
  AnalyzeImage는 보안상 업로드 샌드박스({tempdir}/nexus_uploads) 하위만
  읽는다. 연계가 목적이므로 스크린샷도 같은 폴더(resolve_uploads_dir 단일
  소스)에 저장한다 — 별도 경로 승인 없이 바로 AnalyzeImage에 넘길 수 있다.

작성자: 이현수 / 작성일: 2026-08-04
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Any

from core.tools.base import (
    BaseTool,
    PermissionBehavior,
    PermissionResult,
    ToolResult,
    ToolUseContext,
)
from core.tools.implementations.analyze_image_tool import resolve_uploads_dir

logger = logging.getLogger("nexus.tools.render_preview")

# 렌더 대기 예산(ms) — JS(Vue 등) 실행이 끝날 시간을 가상 시간으로 준다.
DEFAULT_VIRTUAL_TIME_MS = 5000
# 브라우저 프로세스 전체 타임아웃(초) — 행 방지.
DEFAULT_TIMEOUT_SEC = 45
# 기본 뷰포트. 세로를 길게 잡아 전체 페이지가 한 장에 담기게 한다.
DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 2200

# 렌더를 허용하는 확장자(fail-closed — 임의 파일을 브라우저에 넘기지 않는다).
_ALLOWED_SUFFIXES = {".html", ".htm"}

# 흔한 브라우저 실행 파일 이름(PATH 탐색용)과 Windows 표준 설치 경로.
_BROWSER_NAMES = (
    "chrome",
    "google-chrome",
    "chromium",
    "chromium-browser",
    "msedge",
)
_WINDOWS_BROWSER_PATHS = (
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
)


def find_browser(override: str | None = None) -> str | None:
    """사용 가능한 헤드리스 브라우저 실행 파일 경로를 찾는다.

    우선순위: ①명시 override(options/환경변수) ②PATH의 표준 이름들
    ③Windows 표준 설치 경로. 전부 없으면 None(도구가 오류로 안내).
    설정값이 아니라 "환경 탐지"이므로 yaml 강제 대상이 아니다 — 단
    options["browser_path"]로 언제든 명시 지정할 수 있게 열어 둔다.
    """
    if override:
        return override if Path(override).is_file() else None
    for name in _BROWSER_NAMES:
        found = shutil.which(name)
        if found:
            return found
    for path in _WINDOWS_BROWSER_PATHS:
        if Path(path).is_file():
            return path
    return None


class RenderPreviewTool(BaseTool):
    """로컬 HTML 파일을 헤드리스 브라우저로 렌더해 스크린샷 PNG를 만드는 도구."""

    # ═══ 1. Identity ═══

    @property
    def name(self) -> str:
        return "RenderPreview"

    @property
    def description(self) -> str:
        # 모델에게 "만들고 끝"이 아니라 "찍어서 확인"을 유도하는 설명.
        return (
            "Render a local HTML file in a headless browser and save a screenshot "
            "PNG. Use this after creating or editing a web page, then pass the "
            "returned screenshot path to AnalyzeImage to visually check layout "
            "problems (unstyled elements, overlaps, missing sections)."
        )

    @property
    def group(self) -> str:
        return "web"

    # ═══ 2. Schema ═══

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Absolute path of the local .html file to render",
                },
                "width": {
                    "type": "integer",
                    "description": f"Viewport width px (default {DEFAULT_WIDTH})",
                    "default": DEFAULT_WIDTH,
                },
                "height": {
                    "type": "integer",
                    "description": f"Viewport height px (default {DEFAULT_HEIGHT})",
                    "default": DEFAULT_HEIGHT,
                },
            },
            "required": ["file_path"],
        }

    # ═══ 3. Behavior Flags ═══
    # 스크린샷 파일을 쓰므로 is_read_only는 기본값 False를 유지한다(정직).
    # 동시 실행도 브라우저 프로세스 자원 문제로 기본값(순차)을 유지한다.

    # ═══ 5. Lifecycle ═══

    def validate_input(self, input_data: dict[str, Any]) -> str | None:
        """file_path가 비어 있지 않고 확장자가 html인지 빠르게 검사한다."""
        file_path = str(input_data.get("file_path", ""))
        if not file_path.strip():
            return "file_path는 비어 있을 수 없습니다."
        if Path(file_path).suffix.lower() not in _ALLOWED_SUFFIXES:
            return "html/htm 파일만 렌더할 수 있습니다."
        return None

    async def check_permissions(
        self,
        input_data: dict[str, Any],
        context: ToolUseContext,
    ) -> PermissionResult:
        """
        ALLOW로 판정한다(확인 프롬프트 없음).

        근거: ①읽는 대상은 명시된 로컬 html 1개 ②쓰는 대상은 업로드 샌드박스
        하위 스크린샷 PNG뿐 ③외부 네트워크 없음. "만들고 → 찍고 → 고치는"
        자가 검증 루프가 매 반복마다 승인에 막히면 도구의 존재 이유가 없어,
        위험이 제한적인 이 도구만 명시적으로 완화한다(P6 — 명시적 완화).
        """
        return PermissionResult(
            behavior=PermissionBehavior.ALLOW,
            message=f"RenderPreview: {input_data.get('file_path', '')}",
        )

    async def call(
        self,
        input_data: dict[str, Any],
        context: ToolUseContext,
    ) -> ToolResult:
        """
        헤드리스 브라우저로 렌더해 스크린샷을 만든다.

        처리 순서:
          1. 대상 html 존재 확인 + 브라우저 실행 파일 탐지.
          2. 업로드 샌드박스에 출력 PNG 경로 확보(AnalyzeImage 연계 위치).
          3. 브라우저를 --headless=new --screenshot 으로 실행(가상 시간 예산
             으로 JS 렌더 대기, 전체 타임아웃으로 행 방지).
          4. 생성된 PNG 경로와 "다음 단계(AnalyzeImage) 안내"를 결과로 반환.
        """
        file_path = str(input_data["file_path"])
        width = int(input_data.get("width") or DEFAULT_WIDTH)
        height = int(input_data.get("height") or DEFAULT_HEIGHT)

        path = Path(file_path)
        if not path.is_file():
            return ToolResult.error(f"HTML 파일을 찾을 수 없습니다: {file_path}")

        browser = find_browser(
            context.options.get("browser_path") or os.environ.get("NEXUS_BROWSER")
        )
        if browser is None:
            return ToolResult.error(
                "이 환경에서 헤드리스 브라우저(Chrome/Edge/Chromium)를 찾지 못해 "
                "RenderPreview를 사용할 수 없습니다. 렌더 확인 없이 코드 검토로 "
                "대신하세요."
            )

        # 출력 PNG — AnalyzeImage가 읽을 수 있는 업로드 샌드박스에 저장한다.
        uploads_dir = resolve_uploads_dir(context.options.get("uploads_dir"))
        out_path = uploads_dir / f"render_{path.stem}_{int(time.time())}.png"

        cmd = [
            browser,
            "--headless=new",
            "--disable-gpu",
            "--hide-scrollbars",
            f"--window-size={width},{height}",
            f"--screenshot={out_path}",
            f"--virtual-time-budget={DEFAULT_VIRTUAL_TIME_MS}",
            path.resolve().as_uri(),  # file:// URL — 네트워크 접근 없음
        ]
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            try:
                await asyncio.wait_for(proc.wait(), timeout=DEFAULT_TIMEOUT_SEC)
            except TimeoutError:
                proc.kill()
                return ToolResult.error(
                    f"렌더 타임아웃({DEFAULT_TIMEOUT_SEC}초) — 페이지가 너무 무겁거나 "
                    "브라우저가 응답하지 않습니다."
                )
        except OSError as e:
            return ToolResult.error(f"브라우저 실행 실패: {e}")

        if not out_path.is_file() or out_path.stat().st_size == 0:
            return ToolResult.error(
                "스크린샷 생성에 실패했습니다(출력 파일 없음). 브라우저 버전이 "
                "--headless=new 를 지원하는지 확인하세요."
            )

        logger.info("RenderPreview %s -> %s", file_path, out_path)
        # 안내에는 "짧은 파일명"만 노출한다(전체 경로는 metadata에). 실측 근거:
        # 모델이 긴 절대 경로를 그대로 복사하지 못하고 환각 경로를 만들어
        # AnalyzeImage가 반복 실패했다. AnalyzeImage는 파일명만 받으면 업로드
        # 샌드박스 기준으로 해석하므로 파일명만으로 충분하다.
        return ToolResult.success(
            f"렌더 완료: {file_path}\n"
            f"스크린샷 파일명: {out_path.name}\n"
            f"다음 단계: AnalyzeImage(image_path=\"{out_path.name}\", "
            "question=\"이 웹페이지 스크린샷의 레이아웃 문제(스타일 미적용, 겹침, "
            "정렬 깨짐, 빈 영역)를 구체적으로 지적해줘\")로 화면을 검토하고, "
            "발견된 문제를 수정하세요. image_path에는 위 파일명을 그대로 쓰세요"
            "(경로를 새로 만들지 마세요).",
            screenshot_path=str(out_path),
            screenshot_name=out_path.name,
            source_html=file_path,
        )

    # ═══ 7. UI Hints ═══

    def get_progress_label(self, input_data: dict[str, Any]) -> str:
        return f"Rendering {input_data.get('file_path', '...')}"

    def get_input_summary(self, input_data: dict[str, Any]) -> str:
        return input_data.get("file_path", "")
