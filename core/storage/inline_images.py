# OpenAI 규격 메시지에 인라인으로 실려 온 이미지를 검증·저장하고 핸들 문구로 바꾼다.
"""
인라인 이미지 수용 — `content` 배열의 `image_url`(data URL)을 파일로 내린다.

[왜 필요한가 — 2026-08-12]
    코딩 API(`/v1/chat/completions`)는 `content` 가 문자열일 때만 받았다. OpenAI 비전
    규격(`content: [{type:"text"...},{type:"image_url"...}]`)을 보내면 Pydantic 검증에서
    막혀 **422** 가 났다. 실측 결과 막히는 이유는 이미지가 아니라 **배열 형식**이었다 —
    텍스트만 든 배열도 똑같이 422 였다.

    NOVA 의 주 모델(A.X-4.0)은 텍스트 전용이고 비전은 별도 서버(Gemma3-27B)에 있다.
    그래서 이미지 블록을 그대로 흘려보낼 수 없고, **파일로 내린 뒤 그 경로를 대화에
    남겨** AnalyzeImage 도구가 보게 하는 중계가 필요하다. 웹이 이미 같은 방식으로
    동작하므로(업로드 → 서버 경로 → 도구), 문구 규약도 웹과 맞춘다.

[★해시로 저장하는 이유 — 이게 없으면 디스크가 터진다]
    코딩 API 는 **무상태**다. 요청마다 새 세션이고, 클라이언트가 히스토리를 통째로
    다시 보낸다. 그래서 같은 이미지가 **매 턴 다시 도착한다.** 파일명을 새로 만들면
    30턴 대화에서 같은 20MB 가 30벌 쌓인다(600MB).

    파일명을 내용의 SHA-256 으로 정하면 두 번째부터는 쓰지 않는다(멱등). 덤으로,
    보존 기간이 지나 파일이 지워져도 다음 요청에서 **같은 경로로 되살아난다.**

[검증 순서 — 싼 것부터]
    ① base64 문자열 길이로 먼저 자른다. 디코딩 후에 재면 이미 메모리를 썼다.
    ② data URL 형식·MIME 화이트리스트
    ③ 매직 바이트 — 선언한 MIME 과 실제 내용이 같은지(확장자만 믿지 않는다)

작성자: 이현수 / 작성일: 2026-08-12
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import re
from pathlib import Path

# AnalyzeImage 가 실제로 다루는 형식과 **일치시킨다**.
# (analyze_image_tool._MIME_BY_EXT — 여기서 받아 놓고 도구가 거부하면 의미가 없다)
MIME_TO_EXT: dict[str, str] = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/webp": ".webp",
}

# 매직 바이트(파일 시그니처). MIME 선언은 클라이언트가 만든 값이라 믿지 않는다.
#   webp 는 "RIFF....WEBP" 라 앞 4바이트만으로는 부족해 8~12바이트를 함께 본다.
_SIGNATURES: dict[str, tuple[bytes, ...]] = {
    "image/png": (b"\x89PNG\r\n\x1a\n",),
    "image/jpeg": (b"\xff\xd8\xff",),
}

_DATA_URL_RE = re.compile(r"^data:(?P<mime>[\w.+-]+/[\w.+-]+)?;base64,(?P<b64>.*)$", re.S)

# base64 는 원본 대비 약 4/3 로 부풀고 개행/패딩이 더 붙는다. 여유를 조금 둔다.
_B64_OVERHEAD = 1.4


class InlineImageError(ValueError):
    """인라인 이미지를 받아들일 수 없을 때. 메시지는 그대로 사용자에게 보인다."""


def _check_signature(mime: str, raw: bytes) -> None:
    """선언한 MIME 과 실제 내용이 같은지 본다(확장자·MIME 위조 차단)."""
    if mime == "image/webp":
        # RIFF<4바이트 크기>WEBP
        if not (raw[:4] == b"RIFF" and raw[8:12] == b"WEBP"):
            raise InlineImageError("webp 로 선언됐지만 실제 내용이 webp 가 아닙니다.")
        return
    sigs = _SIGNATURES.get(mime, ())
    if sigs and not any(raw.startswith(s) for s in sigs):
        raise InlineImageError(f"{mime} 로 선언됐지만 실제 내용이 일치하지 않습니다.")


def parse_data_url(url: str, max_bytes: int) -> tuple[str, bytes]:
    """`data:image/png;base64,...` 를 (mime, 원본 바이트)로 푼다.

    Args:
        url: image_url.url 값.
        max_bytes: 허용 최대 원본 크기.

    Returns:
        (mime, raw)

    Raises:
        InlineImageError: 형식·크기·MIME·시그니처 중 하나라도 어긋나면.
    """
    if not isinstance(url, str) or not url.startswith("data:"):
        # ★원격 URL 은 받지 않는다 — 에어갭 위반이고 SSRF 통로가 된다.
        raise InlineImageError(
            "이미지는 data URL 로만 보낼 수 있습니다(원격 URL 은 지원하지 않습니다)."
        )
    m = _DATA_URL_RE.match(url)
    if not m:
        raise InlineImageError("data URL 형식이 아닙니다(base64 인코딩이 필요합니다).")

    mime = (m.group("mime") or "").lower()
    if mime not in MIME_TO_EXT:
        raise InlineImageError(
            f"지원하지 않는 이미지 형식입니다: {mime or '(형식 없음)'}. "
            f"지원: {', '.join(sorted(MIME_TO_EXT))}"
        )

    b64 = m.group("b64") or ""
    # ①디코딩 **전**에 자른다. 디코딩 후에 재면 이미 메모리를 다 썼다.
    if len(b64) > int(max_bytes * _B64_OVERHEAD):
        raise InlineImageError(
            f"이미지가 너무 큽니다. 최대 {max_bytes // (1024 * 1024)}MB 까지 지원합니다."
        )

    try:
        raw = base64.b64decode(b64, validate=True)
    except (binascii.Error, ValueError) as e:
        raise InlineImageError(f"base64 디코딩에 실패했습니다: {e}") from e

    if not raw:
        raise InlineImageError("이미지 내용이 비어 있습니다.")
    if len(raw) > max_bytes:
        raise InlineImageError(
            f"이미지가 너무 큽니다({len(raw) / 1024 / 1024:.1f}MB). "
            f"최대 {max_bytes // (1024 * 1024)}MB 까지 지원합니다."
        )

    _check_signature(mime, raw)
    return mime, raw


def save_inline_image(raw: bytes, mime: str, uploads_dir: Path) -> Path:
    """내용 해시를 파일명으로 저장한다. 같은 내용이면 다시 쓰지 않는다(멱등).

    파일명은 **서버가 만든다** — 클라이언트가 준 이름은 쓰지 않는다(경로 주입 차단).
    접두사 `img-` 는 업로드 라우트의 `upload-` 와 구분하기 위한 것이다(정리·조회 시
    와일드카드가 서로를 물지 않게).
    """
    digest = hashlib.sha256(raw).hexdigest()[:32]
    path = uploads_dir / f"img-{digest}{MIME_TO_EXT[mime]}"
    if not path.exists():
        uploads_dir.mkdir(parents=True, exist_ok=True)
        # 같은 요청이 동시에 들어와도 반쪽 파일이 남지 않게 임시로 쓰고 옮긴다.
        tmp = path.with_suffix(path.suffix + ".part")
        tmp.write_bytes(raw)
        tmp.replace(path)
    return path


def build_image_handle(path: Path, index: int) -> str:
    """대화에 남길 핸들 문구. 웹(`readFileContents`)과 같은 규약을 쓴다.

    문구가 갈리면 모델이 웹에서 배운 방식대로 도구를 부르지 못한다.
    """
    return (
        f"사용자가 이미지 파일을 업로드했습니다.\n"
        f"파일명: 이미지 #{index}\n"
        f"서버 경로: {path.as_posix()}\n"
        "이 이미지의 내용을 분석·설명·요약·읽기(OCR)해 달라는 요청이면 "
        "AnalyzeImage 도구로 분석하세요. 이미 분석한 이미지라도 **새로운 질문이면 "
        "다시 호출하세요** — 앞선 요약에는 그 질문의 답이 없을 수 있습니다."
    )
