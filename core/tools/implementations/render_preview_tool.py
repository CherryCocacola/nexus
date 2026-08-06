# RenderPreview 도구 — 로컬 HTML을 헤드리스 브라우저로 렌더해 스크린샷을 만든다.
"""
RenderPreview 도구 — 생성한 웹 결과물을 "실제로 렌더해서" 눈으로 확인하는 도구.

[왜 필요한가 — 자가 검증 루프의 첫 단추]
  실측(2026-08-04 홈페이지 e2e): 모델이 HTML/CSS를 생성만 하고 렌더 확인을
  하지 않아, 네비게이션 스타일 미적용·카드 세로나열 같은 문제를 스스로 알지
  못했다. 이 도구는 로컬 HTML 파일을 헤드리스 Chrome/Edge로 렌더해 스크린샷
  PNG를 만들고, 그 경로를 돌려준다. 모델은 이어서 AnalyzeImage(비전 VLM)로
  화면을 눈으로 확인하고, 필요하면 고친 뒤 다시 렌더한다.

  RenderPreview(렌더→스크린샷+자산 점검) → AnalyzeImage(화면 묘사) → Edit/Write(수정)

  단 "결함이냐"의 판정 근거는 비전이 아니라 **자산 점검**이다. 아래 라운드 상한
  주석과 VERIFY_QUESTION 주석에 실측 근거를 적어 두었다.

[라운드 상한 — 프롬프트가 아니라 도구가 강제한다 (2026-08-07)]
  실측: 시스템 프롬프트에 "최대 2회 수정 후 보고"라고 써 뒀는데도 모델이 4회를
  돌았다. 비전 모델이 "폰트를 키워라" 같은 취향 개선을 끝없이 제안하기 때문이다.
  지시문은 상한이 아니라 권고로 읽힌다 — 그래서 도구가 직접 센다.

  같은 파일을 같은 세션에서 MAX_RENDER_ROUNDS 회 렌더하면, 그 다음 호출은
  렌더하지 않고 오류로 막는다. 오류(tool_use_error)로 돌려주는 이유는 성공
  결과에 "그만"이라고 적어 봐야 스크린샷이 함께 오면 또 검토하기 때문이다.
  받을 게 없어야 루프가 실제로 끝난다.

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
import re
import shutil
import time
from collections import OrderedDict
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

# 한 파일당 허용 렌더 횟수. 첫 렌더 1회 + 수정 후 재렌더 2회 = 3.
# 시스템 프롬프트의 "at most 2 fix-and-rerender rounds"와 같은 수를 가리킨다.
MAX_RENDER_ROUNDS = 3
# 라운드 카운터가 무한히 쌓이지 않게 하는 상한(오래된 항목부터 버린다).
# 도구 인스턴스는 레지스트리에 상주하므로 세션이 계속 바뀌면 키가 계속 는다.
_ROUND_CACHE_MAX = 256

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


# AnalyzeImage에 넘길 질문 — "판정"이 아니라 "묘사"를 요구한다.
#
# 실측(2026-08-07, Gemma 3 27B, 정상/CSS제거 두 스크린샷 × 5개 질문 형태 ≈30회):
#   - "레이아웃 문제를 지적해줘"(구 질문) → 검증 템플릿인데도 겹침·정렬붕괴 같은
#     없는 결함을 매번 지어냈다. 이게 모델이 4라운드를 돈 원인이다.
#   - "치명 결함만, 없으면 '결함 없음'" → 파손본 3/3 오통과(손쉬운 탈출구).
#   - 번호 체크리스트로 항목별 판정 → 파손본 3/3 오통과. 심지어 "링크는 파란
#     밑줄, 목록은 검은 점"이라 정확히 관찰하고도 결론은 '정상'이었다.
#   - "CSS가 적용됐나? 예/아니오" → 6/6 '예'(정상·파손 구분 못 함).
#   - **평가를 빼고 묘사만 시키면 두 화면을 정확히 구별해 묘사했다.**
#
# 결론: 이 등급 VLM은 보기는 하지만 판정은 못 한다(긍정 편향). 그래서 VLM에는
# 눈 역할만 맡기고, 판단은 본 모델이 한다. 다만 그 판단도 완전하지 않으므로
# CSS 미적용은 아래 자산 점검이 모델 없이 확정한다 — 눈은 참고, 판정은 코드.
VERIFY_QUESTION = (
    "이 스크린샷을 보이는 그대로 묘사해라. 배경색, 링크 색과 밑줄 유무, "
    "목록 앞의 기호, 글꼴, 카드나 버튼 같은 디자인 요소가 보이는지, 그리고 "
    "글자가 서로 겹치거나 잘리거나 크게 비어 있는 곳이 있는지를 말해라. "
    "판단이나 평가, 개선 제안은 하지 마라."
)

# ── 자산 점검(모델 없이 확정) ──────────────────────────────
# 자가 검증 루프를 만들게 한 최초 사고가 "네비게이션 스타일 미적용"이었다. 그런데
# 그건 눈으로 볼 필요가 없다 — HTML이 가리키는 css/js 파일이 실제로 있는지 보면
# 확정된다. 비전 모델의 의견보다 이쪽이 정확하고 빠르다.
_STYLESHEET_RE = re.compile(r"<link\b[^>]*>", re.IGNORECASE)
_HREF_RE = re.compile(r"""\bhref\s*=\s*["']([^"']+)["']""", re.IGNORECASE)
_REL_STYLESHEET_RE = re.compile(r"""\brel\s*=\s*["']?[^"'>]*stylesheet""", re.IGNORECASE)
_SRC_RE = re.compile(r"""<(?:script|img)\b[^>]*?\bsrc\s*=\s*["']([^"']+)["']""", re.IGNORECASE)
# 로컬 파일이 아니어서 존재 확인 대상이 아닌 참조들.
_EXTERNAL_PREFIXES = ("http://", "https://", "//", "data:", "#", "mailto:", "javascript:")


def check_local_assets(html_path: Path) -> tuple[list[str], int]:
    """HTML이 참조하는 로컬 자산 중 실제로 없는 것과, 스타일시트 참조 개수를 센다.

    반환: (없는 자산 경로 목록, 스타일시트 <link> 개수)

    쿼리스트링(`styles.css?v=2`)과 앵커는 떼고 본다. 외부 URL·data URI 는
    에어갭에서 어차피 로드되지 않으며 파일 존재 여부를 물을 대상도 아니다.
    """
    try:
        html = html_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return [], 0

    refs: list[str] = []
    stylesheet_count = 0
    for tag in _STYLESHEET_RE.findall(html):
        if not _REL_STYLESHEET_RE.search(tag):
            continue
        stylesheet_count += 1
        href = _HREF_RE.search(tag)
        if href:
            refs.append(href.group(1))
    refs.extend(_SRC_RE.findall(html))

    missing: list[str] = []
    base = html_path.parent
    for ref in refs:
        if ref.lower().startswith(_EXTERNAL_PREFIXES):
            continue
        clean = ref.split("?", 1)[0].split("#", 1)[0].strip()
        if not clean:
            continue
        if not (base / clean).exists() and clean not in missing:
            missing.append(clean)
    return missing, stylesheet_count


def _asset_note(html_path: Path) -> str:
    """자산 점검 결과를 결과 문자열에 실을 한 줄로 만든다."""
    missing, stylesheet_count = check_local_assets(html_path)
    if missing:
        return (
            "⚠ 자산 누락(확정): " + ", ".join(missing) + " — 이 파일이 없어 페이지가 "
            "스타일·스크립트 없이 렌더됐습니다. 화면 인상보다 이것부터 해결하세요.\n"
        )
    if stylesheet_count == 0:
        return (
            '⚠ 스타일시트 참조 없음(확정) — <link rel="stylesheet">가 하나도 없어 '
            "브라우저 기본 스타일로 렌더됐습니다. 이것부터 해결하세요.\n"
        )
    return "자산 점검(확정): 참조된 로컬 css/js/img가 모두 존재합니다.\n"


def _round_advice(round_no: int) -> str:
    """회차에 따라 다음 행동을 좁혀 주는 안내 문구를 만든다."""
    if round_no >= MAX_RENDER_ROUNDS:
        return (
            f"이번이 마지막 렌더({MAX_RENDER_ROUNDS}/{MAX_RENDER_ROUNDS})입니다. "
            "다시 렌더할 수 없으니, 남은 사항은 사용자에게 보고한 뒤 마치세요."
        )
    return (
        "위 자산 점검이 정상이고 묘사에서 겹침·잘림이 보이지 않으면 더 렌더하지 말고 "
        "완료 보고하세요. 묘사는 참고 자료일 뿐이며, 폰트·색상·여백 같은 인상은 "
        "결함이 아니므로 그것 때문에 다시 렌더하지 마세요(남은 렌더 "
        f"{MAX_RENDER_ROUNDS - round_no}회)."
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

    def __init__(self) -> None:
        # (세션, 파일) 별 렌더 횟수. 파일이 다르면 카운터도 별개이고,
        # 세션이 다르면 같은 파일이라도 처음부터 다시 센다.
        # OrderedDict 인 이유: 상한을 넘으면 가장 오래된 항목부터 버리기 위해.
        self._round_counts: OrderedDict[tuple[str, str], int] = OrderedDict()

    def _next_round(self, session_id: str, resolved_path: str) -> int:
        """이 (세션, 파일) 조합의 렌더 회차를 1부터 세어 돌려준다."""
        key = (session_id, resolved_path)
        count = self._round_counts.pop(key, 0) + 1
        self._round_counts[key] = count  # pop→재삽입으로 최근 사용 순서 갱신
        while len(self._round_counts) > _ROUND_CACHE_MAX:
            self._round_counts.popitem(last=False)  # 가장 오래된 것부터 제거
        return count

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

        # 라운드 상한 — 렌더를 시작하기 전에 막는다(브라우저를 띄우지도 않는다).
        # 상한을 넘긴 호출에 스크린샷을 돌려주면 모델이 또 검토하므로, 줄 것을
        # 주지 않는 것이 유일하게 확실한 종료 조건이다.
        round_no = self._next_round(context.session_id, str(path.resolve()))
        if round_no > MAX_RENDER_ROUNDS:
            return ToolResult.error(
                f"렌더 검증 상한({MAX_RENDER_ROUNDS}회)에 도달했습니다 — 이 파일은 "
                "더 이상 렌더하지 않습니다. 남은 개선점을 계속 고치지 말고, 지금까지 "
                "무엇을 고쳤고 무엇이 남았는지 사용자에게 정직하게 보고하고 마치세요."
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

        logger.info("RenderPreview %s -> %s (round %d)", file_path, out_path, round_no)
        # 안내에는 "짧은 파일명"만 노출한다(전체 경로는 metadata에). 실측 근거:
        # 모델이 긴 절대 경로를 그대로 복사하지 못하고 환각 경로를 만들어
        # AnalyzeImage가 반복 실패했다. AnalyzeImage는 파일명만 받으면 업로드
        # 샌드박스 기준으로 해석하므로 파일명만으로 충분하다.
        missing, stylesheet_count = check_local_assets(path)
        return ToolResult.success(
            f"렌더 완료: {file_path} ({round_no}/{MAX_RENDER_ROUNDS}회차)\n"
            + _asset_note(path)
            + f"스크린샷 파일명: {out_path.name}\n"
            + f'다음 단계: AnalyzeImage(image_path="{out_path.name}", '
            f'question="{VERIFY_QUESTION}")로 화면을 눈으로 확인하세요. '
            "image_path에는 위 파일명을 그대로 쓰세요(경로를 새로 만들지 마세요).\n"
            + _round_advice(round_no),
            screenshot_path=str(out_path),
            screenshot_name=out_path.name,
            source_html=file_path,
            render_round=round_no,
            max_render_rounds=MAX_RENDER_ROUNDS,
            missing_assets=missing,
            stylesheet_count=stylesheet_count,
        )

    # ═══ 7. UI Hints ═══

    def get_progress_label(self, input_data: dict[str, Any]) -> str:
        return f"Rendering {input_data.get('file_path', '...')}"

    def get_input_summary(self, input_data: dict[str, Any]) -> str:
        return input_data.get("file_path", "")
