# ScaffoldWeb 도구 — 검증된 프론트엔드 템플릿을 "결정적 파일 복사"로 스캐폴드한다.
"""
ScaffoldWeb 도구 — 웹 페이지 제작을 템플릿 복사에서 시작하게 하는 도구.

[왜 필요한가 — CHI 2026 실증 + 자체 실측]
  실측(2026-08-04~05): A.X-4.0은 긴 HTML/CSS를 처음부터 작성하면 디자인 품질이
  낮고, 긴 내용을 재현/재작성하는 과정에서 출력이 붕괴(degeneration)한다.
  CHI 2026 연구(Design System-Compliant UI Generation)에서도 검증된 컴포넌트
  레지스트리 기반 조립(95.08% 준수)이 프롬프트 주입을 압도했다.

  이 도구는 사람이 미리 검증한 고품질 템플릿을 **모델의 텍스트 출력을 거치지
  않고 Python 파일 복사로** 대상 폴더에 스캐폴드한다. 모델의 역할은:
    ① ScaffoldWeb()           → 카탈로그 확인(어떤 템플릿이 있나)
    ② ScaffoldWeb(template=…, target_dir=…) → 스캐폴드(결정적 복사)
    ③ app.js의 SITE 객체 등 "작은 슬롯"만 Edit  → 내용 교체
    ④ RenderPreview → AnalyzeImage             → 검증
  즉 모델이 긴 코드를 생성·재현할 일이 구조적으로 사라진다.

[에어갭 준수]
  템플릿에는 Vue 3 로컬 번들(vendor/)이 포함되어 CDN 없이 동작한다.
  이 도구 자체는 로컬 파일 복사만 하며 네트워크 접근이 없다.

[템플릿 위치]
  {repo_root}/assets/frontend_templates/ — catalog.json이 단일 카탈로그 소스.
  options["frontend_templates_dir"]로 재지정 가능(배포 레이아웃 대응).

작성자: 이현수 / 작성일: 2026-08-05
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any

from core.tools.base import (
    BaseTool,
    PermissionBehavior,
    PermissionResult,
    ToolResult,
    ToolUseContext,
)

logger = logging.getLogger("nexus.tools.scaffold_web")

# 기본 템플릿 루트 — 이 파일(core/tools/implementations/) 기준 리포 루트의 assets/.
_DEFAULT_TEMPLATES_DIR = Path(__file__).resolve().parents[3] / "assets" / "frontend_templates"


def _resolve_templates_dir(context: ToolUseContext) -> Path:
    """템플릿 루트 디렉토리를 결정한다(options 주입 우선, 없으면 리포 기본 경로)."""
    configured = context.options.get("frontend_templates_dir")
    return Path(configured) if configured else _DEFAULT_TEMPLATES_DIR


def _load_catalog(templates_dir: Path) -> list[dict[str, Any]]:
    """catalog.json을 읽어 템플릿 목록을 반환한다. 없으면 빈 목록."""
    catalog_path = templates_dir / "catalog.json"
    if not catalog_path.is_file():
        return []
    data = json.loads(catalog_path.read_text(encoding="utf-8"))
    return list(data.get("templates", []))


class ScaffoldWebTool(BaseTool):
    """검증된 프론트엔드 템플릿을 대상 폴더로 복사(스캐폴드)하는 도구."""

    # ═══ 1. Identity ═══

    @property
    def name(self) -> str:
        return "ScaffoldWeb"

    @property
    def description(self) -> str:
        # 모델에게 "직접 작성 대신 템플릿에서 시작"을 유도하는 설명.
        return (
            "Scaffold a verified, professionally designed web template into a "
            "target directory (deterministic file copy — no code generation). "
            "Call with no arguments to list available templates. ALWAYS use this "
            "when asked to build a web page/site; then edit only the SITE data "
            "object in app.js instead of writing HTML/CSS from scratch."
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
                "template": {
                    "type": "string",
                    "description": (
                        "Template name from the catalog (e.g. 'landing-vue'). "
                        "Omit to list available templates."
                    ),
                },
                "target_dir": {
                    "type": "string",
                    "description": "Directory to scaffold into (created if missing)",
                },
                "overwrite": {
                    "type": "boolean",
                    "description": "True to overwrite existing files (default false)",
                    "default": False,
                },
            },
            "required": [],
        }

    # ═══ 3. Behavior Flags ═══
    # 파일을 쓰는 도구 — fail-closed 기본값 유지(is_read_only=False, 순차 실행).

    # ═══ 5. Lifecycle ═══

    def validate_input(self, input_data: dict[str, Any]) -> str | None:
        """template을 줬는데 target_dir이 없으면 스캐폴드 대상이 없다."""
        if input_data.get("template") and not str(input_data.get("target_dir", "")).strip():
            return "template을 지정했으면 target_dir도 필요합니다."
        return None

    async def check_permissions(
        self,
        input_data: dict[str, Any],
        context: ToolUseContext,
    ) -> PermissionResult:
        """
        카탈로그 조회(인자 없음)는 읽기뿐이라 ALLOW, 스캐폴드는 파일 쓰기라 ASK.

        accept_edits 모드에서는 CLI 핸들러가 FILE_WRITE 분류(categorize_tool_name)
        를 보고 자동 승인한다 — Write/Edit과 같은 정책.
        """
        template = input_data.get("template")
        if not template:
            return PermissionResult(behavior=PermissionBehavior.ALLOW, message="카탈로그 조회")
        return PermissionResult(
            behavior=PermissionBehavior.ASK,
            message=f"Scaffold '{template}' -> {input_data.get('target_dir', '')}",
        )

    async def call(
        self,
        input_data: dict[str, Any],
        context: ToolUseContext,
    ) -> ToolResult:
        """
        카탈로그 조회 또는 템플릿 스캐폴드를 수행한다.

        스캐폴드 안전 규칙(fail-closed):
          - target_dir은 반드시 작업 디렉토리(context.cwd) 하위여야 한다(순회 차단).
          - 기존 파일과 충돌하면 overwrite=true 없이는 아무것도 복사하지 않는다.
        """
        templates_dir = _resolve_templates_dir(context)
        try:
            catalog = _load_catalog(templates_dir)
        except (OSError, json.JSONDecodeError) as e:
            return ToolResult.error(f"템플릿 카탈로그를 읽을 수 없습니다: {e}")

        template_name = input_data.get("template")

        # ── 모드 1: 카탈로그 조회 ──
        if not template_name:
            if not catalog:
                return ToolResult.error(
                    f"등록된 템플릿이 없습니다(경로: {templates_dir}). "
                    "이 환경에서는 직접 작성으로 대신하세요."
                )
            lines = ["사용 가능한 템플릿:"]
            for t in catalog:
                lines.append(f"- {t['name']}: {t['description']}")
                lines.append(f"  커스터마이징: {t.get('customize', '')}")
            lines.append(
                "다음 단계: ScaffoldWeb(template=\"이름\", target_dir=\"대상 폴더\")"
            )
            return ToolResult.success("\n".join(lines), templates=[t["name"] for t in catalog])

        # ── 모드 2: 스캐폴드 ──
        entry = next((t for t in catalog if t["name"] == template_name), None)
        if entry is None:
            names = ", ".join(t["name"] for t in catalog) or "(없음)"
            return ToolResult.error(
                f"템플릿 '{template_name}'이 카탈로그에 없습니다. 사용 가능: {names}"
            )

        src_dir = templates_dir / template_name
        if not src_dir.is_dir():
            return ToolResult.error(f"템플릿 폴더가 없습니다: {src_dir}")

        # 대상 경로 확정 + 작업 디렉토리 하위 검증(경로 순회 차단, fail-closed).
        cwd = Path(context.cwd or ".").resolve()
        target = Path(str(input_data["target_dir"]))
        if not target.is_absolute():
            target = cwd / target
        target = target.resolve()
        if not target.is_relative_to(cwd):
            return ToolResult.error(
                f"target_dir은 작업 디렉토리 하위여야 합니다: {target} (cwd={cwd})"
            )

        # 충돌 검사 — overwrite=false면 기존 파일이 하나라도 있으면 전체 중단.
        files: list[str] = list(entry.get("files", []))
        overwrite = bool(input_data.get("overwrite", False))
        collisions = [f for f in files if (target / f).exists()]
        if collisions and not overwrite:
            return ToolResult.error(
                f"대상에 이미 파일이 있습니다: {', '.join(collisions)}. "
                "덮어쓰려면 overwrite=true를 지정하세요."
            )

        # 복사 실행 — 모델 텍스트를 거치지 않는 결정적 복사(품질·무결성 보장).
        try:
            for rel in files:
                src = src_dir / rel
                dst = target / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
        except OSError as e:
            return ToolResult.error(f"템플릿 복사 실패: {e}")

        logger.info("ScaffoldWeb %s -> %s (%d files)", template_name, target, len(files))
        return ToolResult.success(
            f"스캐폴드 완료: {template_name} → {target}\n"
            f"생성 파일: {', '.join(files)}\n"
            f"커스터마이징 방법: {entry.get('customize', '')}\n"
            "다음 단계: ①Read로 app.js를 읽고 SITE 객체를 요구사항에 맞게 Edit "
            "②색상 변경이 필요하면 styles.css 맨 위 :root 변수만 Edit "
            "③RenderPreview로 렌더 확인. index.html 구조는 수정하지 마세요.",
            template=template_name,
            target_dir=str(target),
            files=files,
        )

    # ═══ 7. UI Hints ═══

    def get_progress_label(self, input_data: dict[str, Any]) -> str:
        t = input_data.get("template")
        return f"Scaffolding {t}" if t else "Listing templates"

    def get_input_summary(self, input_data: dict[str, Any]) -> str:
        return input_data.get("template", "(list)")
