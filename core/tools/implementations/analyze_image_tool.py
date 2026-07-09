# AnalyzeImage 도구 — 업로드된 이미지를 VLM 서버에 위임해 "텍스트(분석 결과)"로 이해한다.
"""
AnalyzeImage 도구 — 이미지(파일)를 비전 언어 모델(VLM)로 분석해 텍스트를 반환한다.

[이 파일이 하는 일]
  사용자가 첨부(업로드)한 이미지를 LAN 내부의 VLM 서버(Gemma 4 12B 등)에
  OpenAI 비전 형식(HTTP)으로 보내, "이 이미지에 무엇이 있는지 / OCR / 차트 해석 /
  객체 설명" 같은 질문의 답(텍스트)을 받아온다. ImageGenerate(텍스트→이미지)의
  정반대 방향, 즉 이미지→텍스트(분석) 도구다. 두 도구는 config 로드·httpx 요청·
  오류 래핑·ToolResult 반환 방식이 대칭이다.

[아키텍처 규칙 — 왜 GPU 직접 호출이 아닌가]
  anti-pattern #2에 따라 Machine A(오케스트레이터)의 도구 코드는 GPU/CUDA를
  직접 만지지 않는다. 비전 추론은 전적으로 Machine B(VLM 서버)의 OpenAI 호환
  HTTP API(POST {vision_url}/v1/chat/completions)로만 수행한다. 요청은 반드시
  LAN 주소(127.0.0.1 터널 등)로만 보낸다(에어갭 준수, P4).

[안전 설계 — 업로드 디렉토리 하위만 읽는다]
  이 도구는 모델이 준 image_path 를 그대로 읽지 않는다. 웹 업로드 라우트
  (POST /v1/upload)가 저장하는 업로드 샌드박스({tempdir}/nexus_uploads) 하위
  경로만 허용한다. resolve() 로 실제 경로를 확정한 뒤 그 부모가 업로드
  디렉토리 하위인지 검사하므로, 경로 순회(../../etc/passwd)나 임의 파일 읽기가
  구조적으로 불가능하다(fail-closed). 확장자도 png/jpg/jpeg/webp 로 제한한다.

[VLM 서버 계약]
  요청: POST {vision_url}/v1/chat/completions
        body = {
          "model": <config vision 모델명, 기본 "gemma-4-12b">,
          "messages": [{"role": "user", "content": [
              {"type": "text", "text": <question>},
              {"type": "image_url", "image_url": {"url": "data:image/<mime>;base64,<b64>"}}
          ]}],
          "max_tokens": <config 또는 1024>, "temperature": 0.2
        }
  응답: OpenAI 표준 choices[0].message.content → 이 텍스트가 분석 결과.

작성자: 이현수 / 작성일: 2026-07-09
"""

from __future__ import annotations

import base64
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

import httpx

from core.tools.base import (
    BaseTool,
    PermissionBehavior,
    PermissionResult,
    ToolResult,
    ToolUseContext,
)

# 이 모듈 전용 로거. 규칙에 따라 "nexus.{module}" 네임스페이스를 사용한다.
logger = logging.getLogger("nexus.tools.analyze_image")

# VLM 서버 기본 주소 — config(gpu_server.vision_url)나 context.options 로 주입되지
# 않았을 때의 폴백. 하드코딩 금지 원칙에 따라 실제 운영값은 yaml에서 온다
# (여기 값은 어떤 진입점에서도 동작이 깨지지 않게 하는 최후 폴백일 뿐).
DEFAULT_VISION_URL = "http://127.0.0.1:8004"

# VLM served-model-name 기본값. vLLM이 Gemma 4 12B 를 이 이름으로 서빙한다고 본다
# (중국 모델 배제 제약 + Apache 2.0 상업 라이선스로 Gemma 4 채택).
# config(gpu_server.vision_model)/options 로 덮어쓸 수 있으며, 없으면 이 값을 쓴다.
DEFAULT_VISION_MODEL = "gemma-4-12b"

# VLM 응답 최대 토큰. 이미지 설명/OCR 은 어느 정도 길이가 필요하므로 1024로 둔다.
# options["vision_max_tokens"]로 덮어쓸 수 있다.
DEFAULT_MAX_TOKENS = 1024

# 이미지에 대해 물을 기본 질문. 사용자가 question 을 안 주면 "자세히 설명" 을 요청한다.
DEFAULT_QUESTION = "이 이미지를 자세히 설명해줘."

# 분석 가능한 이미지 파일의 최대 크기(바이트). 큰 이미지는 base64 전송(+33%)과
# VLM 토큰 폭증으로 요청 지연·타임아웃을 유발하므로 상한을 둔다(fail-closed).
# 필요 시 options["vision_max_image_mb"]로 완화할 수 있다.
DEFAULT_MAX_IMAGE_MB = 10

# 허용 확장자 → data URL 의 MIME 서브타입 매핑. 이 목록에 없는 확장자는 거부한다
# (fail-closed). jpg/jpeg 는 둘 다 image/jpeg 로 매핑된다.
_MIME_BY_EXT: dict[str, str] = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
}


def resolve_uploads_dir(configured: str | None = None) -> Path:
    """
    업로드된 첨부 파일이 저장되는 디렉토리를 결정하고, 없으면 만들어 반환한다.

    [왜 이 함수가 따로 필요한가]
      파일을 "저장하는" 쪽(웹의 /v1/upload 라우트)과 "읽어서 분석하는" 쪽
      (이 도구)이 반드시 같은 폴더를 가리켜야 한다. 그래서 경로 결정 로직을 이
      함수 하나로 모아 두 곳이 공유한다(DocumentExport 의 resolve_exports_dir 과
      같은 관례). 경로를 여기저기 하드코딩하면 어긋나기 쉽기 때문이다.

    [경로 결정 우선순위]
      1) configured 설정값이 있으면 그대로 사용(context.options["uploads_dir"] 주입).
      2) 비어 있으면 시스템 임시폴더 아래 nexus_uploads 폴더로 폴백
         (업로드 라우트가 쓰는 {tempdir}/nexus_uploads 와 동일).

    매개변수:
      configured — 설정에서 온 저장 경로 문자열. None/빈문자열이면 폴백을 쓴다.
    반환:
      실제로 존재가 보장된(mkdir 완료된) 업로드 디렉토리의 Path(resolve 완료).
    """
    base = (configured or "").strip() or os.path.join(tempfile.gettempdir(), "nexus_uploads")
    path = Path(base)
    # 상위 폴더까지 한 번에 생성. 이미 있으면 조용히 넘어간다(exist_ok=True).
    path.mkdir(parents=True, exist_ok=True)
    return path.resolve()


class AnalyzeImageTool(BaseTool):
    """
    업로드된 이미지를 VLM 서버로 분석해 텍스트를 반환하는 도구(BaseTool 구현체).

    [수명주기(BaseTool 계약)]
      validate_input()  → 입력을 미리 검증(image_path 존재).
      check_permissions() → 실행 허가 판단(업로드 디렉토리 하위 경로만 ALLOW).
      call()            → 이미지 base64 인코딩 → VLM 서버에 HTTP 요청 → 분석 텍스트 반환.
      map_result()      → 분석 텍스트를 그대로 tool_result 로 모델에 전달(BaseTool 기본).

    ImageGenerate 와 대칭이다(같은 config 로드·httpx·오류 래핑 패턴, 방향만 반대).
    """

    # ═══ 1. Identity(도구 식별 정보) ═══

    @property
    def name(self) -> str:
        # 모델이 tool_calls에서 호출할 때 쓰는 고유 이름(레지스트리 키).
        return "AnalyzeImage"

    @property
    def description(self) -> str:
        # 모델에게 "이 도구를 언제 써야 하는지" 알려주는 설명(프롬프트에 노출됨).
        return (
            "Analyze an uploaded image with the local (LAN) vision model (VLM). "
            "Use when the user attaches an image and asks to describe it, read text "
            "(OCR), interpret a chart/table, or identify objects. Pass the uploaded "
            "attachment's '서버 경로'(server path) as image_path. "
            "이미지 첨부의 '서버 경로'를 받아 설명·OCR·차트 해석에 사용한다."
        )

    @property
    def aliases(self) -> list[str]:
        # 모델이 다른 이름으로 부를 때도 이 도구로 연결되도록 하는 별칭 목록.
        return ["DescribeImage", "VisionAnalyze", "ReadImage"]

    @property
    def group(self) -> str:
        # 부작용 없이 이미지를 "읽어" 분석하는 읽기 계열 도구다.
        return "readonly"

    # ═══ 2. Schema(입력 스키마) ═══

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "image_path": {
                    "type": "string",
                    "description": (
                        "Server path of the uploaded image file to analyze. "
                        "업로드 첨부의 '서버 경로'(예: /tmp/nexus_uploads/foo.png)를 넣으라. "
                        "png/jpg/jpeg/webp 만 지원한다."
                    ),
                },
                "question": {
                    "type": "string",
                    "description": (
                        "What to ask about the image (OCR, chart reading, objects, etc.). "
                        f"미지정 시 기본값 '{DEFAULT_QUESTION}' 을 사용한다."
                    ),
                },
            },
            "required": ["image_path"],
        }

    # ═══ 3. Behavior Flags (fail-closed 기본값 — 필요한 것만 명시적 완화) ═══

    @property
    def is_read_only(self) -> bool:
        # 이미지를 읽어 분석만 할 뿐 파일을 만들거나 바꾸지 않는다(부작용 없음).
        return True

    @property
    def is_concurrency_safe(self) -> bool:
        # 비전 추론은 GPU를 무겁게 점유하므로 다른 도구와 병렬 실행하지 않는다.
        # (fail-closed 기본값 False와 동일하지만 의도를 명시적으로 표시.)
        return False

    # ═══ 5. Lifecycle(수명주기 메서드) ═══

    def validate_input(self, input_data: dict[str, Any]) -> str | None:
        """
        실제 실행 전에 입력을 사전 검증한다.

        무엇을 검사하나:
          - image_path 가 비어 있지 않은지(빈 경로로 무의미한 호출 방지).

        (업로드 디렉토리 하위 여부·존재·확장자 검사는 실제 경로 해석이 필요하므로
         check_permissions/call 에서 수행한다.)

        반환:
          문제가 없으면 None, 문제가 있으면 사용자에게 보여줄 오류 메시지(str).
        """
        image_path = input_data.get("image_path", "")
        if not image_path or not str(image_path).strip():
            return "image_path는 비어 있을 수 없습니다."
        return None

    def _resolve_uploaded_image(
        self, image_path: str, context: ToolUseContext
    ) -> tuple[Path | None, str | None]:
        """
        image_path 를 업로드 샌드박스 하위의 안전한 이미지 파일 경로로 확정한다.

        검증 단계(fail-closed):
          1) 업로드 디렉토리를 resolve() 로 확정(options["uploads_dir"] 우선).
          2) image_path 를 resolve() 로 확정한다(../ 등 상대 성분 정규화).
          3) 확정 경로가 업로드 디렉토리 "하위"인지 확인(is_relative_to) — 아니면 거부.
          4) 실제 파일로 존재하는지 확인.
          5) 확장자가 허용 목록(png/jpg/jpeg/webp)인지 확인.

        반환:
          (Path, None) 검증 통과 시 확정된 경로,
          (None, str)  실패 시 오류 메시지.
        """
        uploads_dir = resolve_uploads_dir(context.options.get("uploads_dir"))
        try:
            target = Path(image_path).resolve()
        except (OSError, ValueError) as e:
            return None, f"이미지 경로를 해석할 수 없습니다: {e}"

        # 업로드 샌드박스 하위가 아니면 거부(경로 순회·임의 파일 읽기 차단).
        if not target.is_relative_to(uploads_dir):
            return None, (
                f"허용된 업로드 디렉토리 하위 경로만 분석할 수 있습니다: {image_path}"
            )
        if not target.is_file():
            return None, f"이미지 파일을 찾을 수 없습니다: {image_path}"

        ext = target.suffix.lower()
        if ext not in _MIME_BY_EXT:
            return None, (
                f"지원하지 않는 이미지 형식입니다: '{ext}'. "
                f"지원: {', '.join(sorted(_MIME_BY_EXT))}"
            )

        # 크기 상한 검사(fail-closed). 큰 이미지는 전송 지연·토큰 폭증·타임아웃을
        # 유발하므로 여기서 미리 막는다. 상한은 options 로 완화 가능.
        max_mb = int(context.options.get("vision_max_image_mb") or DEFAULT_MAX_IMAGE_MB)
        size = target.stat().st_size
        if size > max_mb * 1024 * 1024:
            return None, (
                f"이미지 파일이 너무 큽니다({size / 1024 / 1024:.1f}MB). "
                f"최대 {max_mb}MB 까지 지원합니다."
            )
        return target, None

    async def check_permissions(
        self, input_data: dict[str, Any], context: ToolUseContext
    ) -> PermissionResult:
        """
        이 도구의 실행 허가 여부를 판단한다.

        [읽기 계열 + 경로 샌드박싱]
          이미지를 읽어 분석만 하므로 쓰기 위험은 없다. 다만 임의 파일 읽기를
          막기 위해, image_path 가 업로드 샌드박스({tempdir}/nexus_uploads) 하위인지
          여기서 먼저 검사한다. 하위가 아니거나 존재/확장자가 어긋나면 DENY 한다
          (fail-closed). 통과하면 ALLOW.

        반환:
          통과 시 behavior=ALLOW, 실패 시 behavior=DENY(사유 메시지 포함).
        """
        image_path = str(input_data.get("image_path", ""))
        _, error = self._resolve_uploaded_image(image_path, context)
        if error is not None:
            return PermissionResult(behavior=PermissionBehavior.DENY, message=error)
        return PermissionResult(behavior=PermissionBehavior.ALLOW)

    async def call(self, input_data: dict[str, Any], context: ToolUseContext) -> ToolResult:
        """
        이미지를 VLM 서버에 보내 분석 텍스트를 받아 반환한다. 이 도구의 핵심 메서드다.

        처리 순서:
          1. image_path 를 업로드 샌드박스 하위의 안전한 이미지로 확정(존재/확장자 검증).
          2. 파일 바이트를 읽어 base64 인코딩 + MIME(data URL) 판별.
          3. VLM 서버 URL·모델명·max_tokens 를 context.options(웹이 config에서 주입) 우선 결정.
          4. httpx로 POST {vision_url}/v1/chat/completions(OpenAI 비전 형식, 타임아웃 120초).
             연결/타임아웃/HTTP 오류는 anti-pattern #8대로 구체 예외로 잡아 래핑한다.
          5. 응답 choices[0].message.content 를 꺼내 ToolResult.success(분석 텍스트).
             metadata 에 model/image_path/question 을 담는다.

        매개변수:
          input_data — image_path(필수), question(선택).
          context    — 실행 컨텍스트. options["uploads_dir"]로 업로드 디렉토리,
                       options["vision_url"]/["vision_model"]로 VLM 서버를 주입받는다.
        반환:
          성공 시 ToolResult.success(분석 텍스트 + 메타데이터),
          실패 시 ToolResult.error(사유 메시지).
        """
        # 1) 업로드 샌드박스 하위의 안전한 이미지로 확정. 실패 시 즉시 오류.
        image_path = str(input_data["image_path"])
        target, error = self._resolve_uploaded_image(image_path, context)
        if error is not None or target is None:
            return ToolResult.error(error or "이미지 경로 검증에 실패했습니다.")

        question = str(input_data.get("question") or DEFAULT_QUESTION)

        # 2) 파일 바이트 → base64 + MIME(data URL). 읽기 오류는 구체 예외로 래핑.
        try:
            image_bytes = target.read_bytes()
        except OSError as e:
            return ToolResult.error(f"이미지 파일을 읽을 수 없습니다: {e}")
        mime = _MIME_BY_EXT[target.suffix.lower()]
        image_b64 = base64.b64encode(image_bytes).decode("ascii")
        data_url = f"data:{mime};base64,{image_b64}"

        # 3) VLM 서버 주소·모델명·출력 토큰 결정. options 주입이 있으면 그 값을, 없으면 폴백.
        #    (웹은 config.gpu_server.vision_url/vision_model 을 base_options로 주입한다.)
        vision_url = str(context.options.get("vision_url") or DEFAULT_VISION_URL).rstrip("/")
        vision_model = str(context.options.get("vision_model") or DEFAULT_VISION_MODEL)
        max_tokens = int(context.options.get("vision_max_tokens") or DEFAULT_MAX_TOKENS)

        # OpenAI 비전 형식 요청 본문. temperature 는 사실 위주 분석이므로 낮게(0.2).
        payload: dict[str, Any] = {
            "model": vision_model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
            "max_tokens": max_tokens,
            "temperature": 0.2,
        }

        # 4) HTTP 요청. VLM 추론은 오래 걸릴 수 있어 타임아웃을 120초로 넉넉히 둔다.
        #    bare except 금지 — 예외 종류별로 구체적으로 잡아 원인을 구분해 안내한다.
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    f"{vision_url}/v1/chat/completions", json=payload
                )
                response.raise_for_status()
                body = response.json()
        except httpx.ConnectError as e:
            # VLM 서버에 연결 자체가 안 됨(서버 미기동/주소 오류 등).
            logger.warning("VLM 서버 연결 실패: %s (%s)", vision_url, e)
            return ToolResult.error(f"VLM 서버에 연결할 수 없습니다({vision_url}): {e}")
        except httpx.TimeoutException as e:
            # 응답이 제한 시간(120초) 안에 오지 않음.
            logger.warning("VLM 서버 응답 타임아웃: %s (%s)", vision_url, e)
            return ToolResult.error(f"이미지 분석이 시간 내에 끝나지 않았습니다: {e}")
        except httpx.HTTPStatusError as e:
            # 서버가 4xx/5xx 상태 코드로 응답(요청 형식 오류/서버 내부 오류 등).
            logger.warning("VLM 서버 HTTP 오류: %s", e)
            return ToolResult.error(f"VLM 서버가 오류를 반환했습니다: {e}")
        except httpx.HTTPError as e:
            # 그 밖의 httpx 통신 오류(응답 파싱 등). 위 구체 예외에 안 걸린 나머지.
            logger.warning("VLM 서버 통신 오류: %s", e)
            return ToolResult.error(f"VLM 서버 통신에 실패했습니다: {e}")

        # 5) 응답에서 분석 텍스트(choices[0].message.content)를 꺼낸다. 없으면 오류로 래핑.
        try:
            content = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            logger.warning("VLM 서버 응답 형식이 예상과 다름: %s", e)
            return ToolResult.error(f"VLM 서버 응답에서 분석 결과를 찾을 수 없습니다: {e}")
        if not content or not str(content).strip():
            return ToolResult.error("VLM 서버가 빈 분석 결과를 반환했습니다.")

        analysis = str(content).strip()
        logger.info(
            "AnalyzeImage %s (%d bytes, model=%s) → %d chars",
            target.name,
            len(image_bytes),
            vision_model,
            len(analysis),
        )

        # 분석 텍스트를 그대로 도구 결과로 넘긴다(map_result 기본 = data 문자열 전달).
        # 모델(A.X-4.0)이 이 텍스트를 근거로 사용자에게 답하도록 한다.
        return ToolResult.success(
            analysis,
            model=vision_model,
            image_path=str(target),
            question=question,
        )

    # ═══ 7. UI Hints(진행 표시용 힌트) ═══

    def get_progress_label(self, input_data: dict[str, Any]) -> str:
        # 실행 중 표시 문구. 예: "Analyzing image".
        return "Analyzing image"

    def get_input_summary(self, input_data: dict[str, Any]) -> str:
        # 도구 호출을 한 줄로 요약할 때 쓰는 값(이미지 파일명 + 질문 앞부분).
        path = str(input_data.get("image_path", ""))
        name = Path(path).name if path else ""
        question = str(input_data.get("question", ""))[:60]
        return f"{name} — {question}" if question else name
