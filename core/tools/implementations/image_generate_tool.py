# ImageGenerate 도구 — 프롬프트(텍스트)를 확산 모델로 PNG 이미지로 "생성"한다.
"""
ImageGenerate 도구 — 텍스트 프롬프트를 이미지(PNG) 파일로 "생성"한다.

[이 파일이 하는 일]
  사용자가 요청한 장면·그림을 LAN 내부의 이미지 서버(FLUX 등 확산 모델)에
  HTTP로 요청해 PNG로 받아오고, 그 결과를 사용자가 실제로 내려받을 수 있는
  이미지 "파일"로 저장한다. 저장 후에는 다운로드/미리보기 URL을 돌려주고,
  웹에서는 그 URL(/v1/download/...)로 이미지를 받아 미리 볼 수 있다.
  DocumentExport(문서 생성) 도구의 이미지 버전 — 생성물→저장→URL 반환이 대칭이다.

[아키텍처 규칙 — 왜 GPU 직접 호출이 아닌가]
  anti-pattern #2에 따라 Machine A(오케스트레이터)의 도구 코드는 GPU/CUDA를
  직접 만지지 않는다. 확산 모델 추론은 전적으로 Machine B(이미지 서버)의
  OpenAI 유사 HTTP API(POST {image_url}/v1/images/generate)로만 수행한다.
  요청은 반드시 LAN 주소(127.0.0.1 터널 등)로만 보낸다(에어갭 준수).

[안전 설계 — DocumentExport와 동일]
  이 도구도 저장 경로를 모델이 정하지 못한다. 저장 위치는 DocumentExport와
  똑같은 exports 샌드박스 디렉토리로 고정하고, 파일명은 정화 + uuid로 고유화한다.
  경로 순회(../..)나 기존 파일 덮어쓰기가 구조적으로 불가능하므로
  check_permissions 를 ALLOW 로 둔다.

[이미지 서버 계약]
  요청: POST {image_url}/v1/images/generate
        body = {"prompt": str, "width": int, "height": int, "steps": int, "seed": int|null}
  응답: {"image_base64": "<png base64>", "width": int, "height": int, "seed": int, "model": str}

작성자: 이현수 / 작성일: 2026-07-09
"""

from __future__ import annotations

import base64
import binascii
import logging
from typing import Any

import httpx

from core.tools.base import (
    BaseTool,
    PermissionBehavior,
    PermissionResult,
    ToolResult,
    ToolUseContext,
)

# DocumentExport와 저장 디렉토리/파일명 정화 로직을 공유한다 — 두 도구가 같은
# exports 샌드박스에 저장하고 같은 /v1/download 라우트로 내려주기 위함이다.
from core.tools.implementations.document_export_tool import (
    _safe_filename,
    resolve_exports_dir,
)

# 이 모듈 전용 로거. 규칙에 따라 "nexus.{module}" 네임스페이스를 사용한다.
logger = logging.getLogger("nexus.tools.image_generate")

# 이미지 서버 기본 주소 — config(gpu_server.image_url)나 context.options 로
# 주입되지 않았을 때의 폴백. 하드코딩 금지 원칙에 따라 실제 운영값은 yaml에서
# 온다(여기 값은 어떤 진입점에서도 동작이 깨지지 않게 하는 최후 폴백일 뿐).
DEFAULT_IMAGE_URL = "http://127.0.0.1:8003"

# FLUX schnell 계열의 권장 스텝 수. 확산 모델은 스텝이 많을수록 느려지므로
# schnell(빠른 변형)은 4스텝이 품질/속도의 표준 절충점이다. context.options 로
# 덮어쓸 수 있으나 기본은 이 값을 쓴다.
DEFAULT_STEPS = 4

# size 문자열("WxH") → (width, height) 매핑. 입력 스키마의 enum과 1:1 대응한다.
_SIZE_MAP: dict[str, tuple[int, int]] = {
    "1024x1024": (1024, 1024),
    "1024x1536": (1024, 1536),
    "1536x1024": (1536, 1024),
}


class ImageGenerateTool(BaseTool):
    """
    텍스트 프롬프트를 PNG 이미지로 생성하는 도구(BaseTool 구현체).

    [수명주기(BaseTool 계약)]
      validate_input()  → 입력을 미리 검증(prompt 존재, size 지원 여부).
      check_permissions() → 실행 허가 판단(항상 ALLOW, 위 안전설계 참조).
      call()            → 이미지 서버에 HTTP 요청 → PNG 저장 → 다운로드 URL 반환.
      그 외 get_*()      → 진행 표시 등 UI 힌트.

    저장/URL 방식은 DocumentExport와 완전히 대칭이다(같은 exports 폴더, 같은
    /v1/download/{filename} 라우트).
    """

    # ═══ 1. Identity(도구 식별 정보) ═══

    @property
    def name(self) -> str:
        # 모델이 tool_calls에서 호출할 때 쓰는 고유 이름(레지스트리 키).
        return "ImageGenerate"

    @property
    def description(self) -> str:
        # 모델에게 "이 도구를 언제 써야 하는지" 알려주는 설명(프롬프트에 노출됨).
        return (
            "Generate a downloadable PNG image from a text prompt using the "
            "local (LAN) diffusion image server. Use when the user asks to draw, "
            "create, or generate an image/picture/illustration. "
            "이미지·그림·일러스트 생성 요청에 사용한다."
        )

    @property
    def aliases(self) -> list[str]:
        # 모델이 다른 이름으로 부를 때도 이 도구로 연결되도록 하는 별칭 목록.
        return ["GenerateImage", "DrawImage", "CreateImage"]

    @property
    def group(self) -> str:
        # DocumentExport와 같은 "file" 그룹 계열 — 생성물(파일)을 내놓는 도구다.
        return "file"

    # ═══ 2. Schema(입력 스키마) ═══

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": (
                        "Description of the image to generate. "
                        "영어 프롬프트 권장(확산 모델은 영어 학습이 주력이라 "
                        "영어 프롬프트가 더 정확한 결과를 낸다)."
                    ),
                },
                "size": {
                    "type": "string",
                    "enum": ["1024x1024", "1024x1536", "1536x1024"],
                    "description": "Output image size (width x height). Default 1024x1024.",
                },
                "seed": {
                    "type": "integer",
                    "description": "Optional seed for reproducible generation.",
                },
            },
            "required": ["prompt"],
        }

    # ═══ 3. Behavior Flags (fail-closed 기본값 유지) ═══
    # BaseTool 기본값(is_read_only=False, is_destructive=False, requires_confirmation=False)을
    # 그대로 쓰되, 동시성만 명시적으로 재확인한다. 아래 is_concurrency_safe=False 는
    # 사실 BaseTool 기본값과 같지만, "GPU가 무거우므로 병렬 실행하지 않는다"는
    # 의도를 코드로 분명히 남기기 위해 명시한다.

    @property
    def is_concurrency_safe(self) -> bool:
        # 이미지 생성은 GPU를 무겁게 점유하므로 다른 도구와 병렬 실행하지 않는다.
        # (fail-closed 기본값 False와 동일하지만 의도를 명시적으로 표시.)
        return False

    # ═══ 5. Lifecycle(수명주기 메서드) ═══

    def validate_input(self, input_data: dict[str, Any]) -> str | None:
        """
        실제 실행 전에 입력을 사전 검증한다.

        무엇을 검사하나:
          - prompt 가 비어 있지 않은지(빈 프롬프트로 무의미한 생성 방지).
          - size 가 지원하는 값(_SIZE_MAP)에 속하는지. 미지정은 허용(기본값 사용).

        반환:
          문제가 없으면 None, 문제가 있으면 사용자에게 보여줄 오류 메시지(str).
        """
        prompt = input_data.get("prompt", "")
        if not prompt or not str(prompt).strip():
            return "prompt는 비어 있을 수 없습니다."
        # size는 선택 필드 — 주어졌을 때만 지원 목록에 있는지 확인한다.
        size = input_data.get("size")
        if size is not None and str(size) not in _SIZE_MAP:
            return f"지원하지 않는 size입니다: '{size}'. 지원: {', '.join(_SIZE_MAP)}"
        return None

    async def check_permissions(
        self, input_data: dict[str, Any], context: ToolUseContext
    ) -> PermissionResult:
        """
        이 도구의 실행 허가 여부를 판단한다.

        [왜 항상 ALLOW인가]
          DocumentExport와 동일한 근거다. 저장 위치가 exports 샌드박스로 고정되고
          파일명도 정화·uuid 고유화되어 모델이 저장 경로를 조작할 방법이 없다.
          따라서 임의 경로 쓰기 위험이 없으므로 별도 확인 없이 허용한다.

        반환:
          behavior=ALLOW 인 PermissionResult.
        """
        return PermissionResult(behavior=PermissionBehavior.ALLOW)

    async def call(self, input_data: dict[str, Any], context: ToolUseContext) -> ToolResult:
        """
        이미지 서버에 생성을 요청하고, PNG로 저장한 뒤 다운로드 URL을 반환한다.
        이 도구의 핵심 메서드다.

        처리 순서:
          1. prompt/size/seed 입력값을 꺼내고 size를 width/height로 파싱한다.
          2. 이미지 서버 URL·steps를 context.options(웹이 config에서 주입) 우선으로 결정.
          3. httpx로 POST {image_url}/v1/images/generate 요청(타임아웃 넉넉히 120초).
             연결/타임아웃/HTTP 오류는 anti-pattern #8대로 구체 예외로 잡아 래핑한다.
          4. 응답 image_base64 를 디코드해 PNG 바이트를 얻는다.
          5. DocumentExport와 동일하게 exports 폴더에 .png 저장 + /v1/download URL 구성.
          6. ToolResult.success — metadata에 download_url/preview_url/width/height/seed/model.

        매개변수:
          input_data — prompt(필수), size(선택), seed(선택).
          context    — 실행 컨텍스트. options["exports_dir"]로 저장 폴더,
                       options["image_url"]로 이미지 서버 주소를 주입받는다.
        반환:
          성공 시 ToolResult.success(안내문 + download_url 등 메타데이터),
          실패 시 ToolResult.error(사유 메시지).
        """
        # 1) 입력값 추출. size는 기본 1024x1024, seed는 선택(없으면 None → 서버 랜덤).
        prompt = str(input_data["prompt"])
        size = str(input_data.get("size", "1024x1024"))
        width, height = _SIZE_MAP.get(size, _SIZE_MAP["1024x1024"])
        seed = input_data.get("seed")  # int 또는 None

        # 2) 이미지 서버 주소·스텝 결정. options 로 주입되면 그 값을, 없으면 폴백 상수.
        #    (웹은 config.gpu_server.image_url 을 base_options["image_url"]로 주입한다.)
        image_url = str(context.options.get("image_url") or DEFAULT_IMAGE_URL).rstrip("/")
        steps = int(context.options.get("image_steps") or DEFAULT_STEPS)

        # 이미지 서버 계약에 맞춘 요청 본문. seed는 None이면 그대로 null로 전달.
        payload: dict[str, Any] = {
            "prompt": prompt,
            "width": width,
            "height": height,
            "steps": steps,
            "seed": seed,
        }

        # 3) HTTP 요청. GPU 확산 추론은 오래 걸릴 수 있어 타임아웃을 120초로 넉넉히 둔다.
        #    bare except 금지 — 예외 종류별로 구체적으로 잡아 원인을 구분해 안내한다.
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    f"{image_url}/v1/images/generate", json=payload
                )
                response.raise_for_status()
                body = response.json()
        except httpx.ConnectError as e:
            # 이미지 서버에 연결 자체가 안 됨(서버 미기동/주소 오류 등).
            logger.warning("이미지 서버 연결 실패: %s (%s)", image_url, e)
            return ToolResult.error(
                f"이미지 서버에 연결할 수 없습니다({image_url}): {e}"
            )
        except httpx.TimeoutException as e:
            # 응답이 제한 시간(120초) 안에 오지 않음.
            logger.warning("이미지 서버 응답 타임아웃: %s (%s)", image_url, e)
            return ToolResult.error(f"이미지 생성이 시간 내에 끝나지 않았습니다: {e}")
        except httpx.HTTPStatusError as e:
            # 서버가 4xx/5xx 상태 코드로 응답(요청 형식 오류/서버 내부 오류 등).
            logger.warning("이미지 서버 HTTP 오류: %s", e)
            return ToolResult.error(f"이미지 서버가 오류를 반환했습니다: {e}")
        except httpx.HTTPError as e:
            # 그 밖의 httpx 통신 오류(응답 파싱 등). 위 구체 예외에 안 걸린 나머지.
            logger.warning("이미지 서버 통신 오류: %s", e)
            return ToolResult.error(f"이미지 서버 통신에 실패했습니다: {e}")

        # 4) 응답에서 base64 PNG를 꺼내 디코드한다. 값이 없거나 손상되면 오류로 래핑.
        image_b64 = body.get("image_base64")
        if not image_b64:
            return ToolResult.error("이미지 서버 응답에 image_base64 가 없습니다.")
        try:
            png_bytes = base64.b64decode(image_b64)
        except (binascii.Error, ValueError) as e:
            return ToolResult.error(f"이미지 서버 응답(base64) 디코드에 실패했습니다: {e}")

        # 5) DocumentExport와 동일한 저장 방식 — exports 샌드박스에 .png 로 저장.
        #    exports_dir 은 웹이 base_options 로 주입(config 값). 미주입 시 tempdir 폴백.
        exports_dir = resolve_exports_dir(context.options.get("exports_dir"))
        # 파일명은 프롬프트 앞부분을 바탕으로 정화 + uuid 고유화(경로 순회 원천 차단).
        filename = _safe_filename(prompt[:40] or "image", "png")
        out_path = exports_dir / filename
        out_path.write_bytes(png_bytes)

        # 서버가 실제로 사용한 값(응답 기준)을 메타데이터에 담는다. 없으면 요청값으로 폴백.
        out_width = int(body.get("width", width))
        out_height = int(body.get("height", height))
        out_seed = body.get("seed", seed)
        model = str(body.get("model", ""))
        size_bytes = out_path.stat().st_size

        # 6) 다운로드/미리보기 URL 구성 — DocumentExport와 동일한 /v1/download 라우트.
        #    이미지는 그 URL로 바로 미리보기가 가능하므로 preview_url=download_url 이다.
        download_url = f"/v1/download/{filename}"
        preview_url = download_url
        logger.info(
            "ImageGenerate %s (%d bytes, %dx%d, seed=%s) → %s",
            filename,
            size_bytes,
            out_width,
            out_height,
            out_seed,
            out_path,
        )

        # 결과 본문에는 URL을 "한 번만" 넣는다(서버가 _DOWNLOAD_URL_RE로 추출 → UI에 자동 첨부).
        # 모델에게는 URL/파일명을 재현하지 말라고 지시한다 — 긴 UUID 문자열을 FP8 모델이
        # 반복 재현하다 degeneration(반복 붕괴, 특수토큰 누수)에 빠지는 것을 막기 위함이다.
        # 미리보기·다운로드 링크는 서버가 구조화 데이터(_collect_downloads)로 주입하므로
        # 모델 텍스트에 의존하지 않는다(파일 상단 주석 및 web/app.py:115 참조).
        return ToolResult.success(
            f"이미지 생성 완료: {filename} ({size_bytes:,} bytes, {out_width}x{out_height}).\n"
            f"[시스템] 미리보기·다운로드 링크는 서버가 사용자 화면에 자동으로 첨부합니다. "
            f"답변에는 파일명·URL·마크다운 링크를 다시 쓰지 말고, "
            f"'요청하신 이미지를 생성했습니다.' 같은 짧은 한 줄만 작성하세요.\n"
            f"(서버 링크 추출용, 사용자에게 노출 금지: {download_url})",
            download_url=download_url,
            preview_url=preview_url,
            filename=filename,
            width=out_width,
            height=out_height,
            seed=out_seed,
            model=model,
            bytes=size_bytes,
        )

    # ═══ 7. UI Hints(진행 표시용 힌트) ═══

    def get_progress_label(self, input_data: dict[str, Any]) -> str:
        # 실행 중 표시 문구. 예: "Generating image".
        return "Generating image"

    def get_input_summary(self, input_data: dict[str, Any]) -> str:
        # 도구 호출을 한 줄로 요약할 때 쓰는 값(프롬프트 앞부분).
        return str(input_data.get("prompt", ""))[:100]
