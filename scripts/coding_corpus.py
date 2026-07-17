# 코딩 코퍼스(Stack Overflow 등) 정제 공용 모듈 — 코드 인지 청킹부터 시작한다.
"""
코딩 코퍼스 적재를 위한 공용 정제 유틸(prepare_stackoverflow.py 등이 공유).

── 이 파일이 하는 일 (한눈에) ─────────────────────────────
Stack Overflow·OKKY·GitHub 같은 "코드가 섞인 텍스트"를 tb_knowledge RAG나 학습
데이터로 넣기 전에, 임베딩하기 좋은 크기의 청크로 나눈다. 일반 산문 청커
(core.rag.knowledge_store.split_into_chunks)를 코드에 그대로 쓰면 두 가지 사고가
난다 — 그래서 별도의 "코드 인지 청커"가 필요하다.

── 왜 기존 청커를 못 쓰나 (M2 결함) ──────────────────────
1. 코드 블록 파괴: split_into_chunks는 빈 줄(\\n\\n)로 먼저 자른다. 코드에는
   함수 사이 빈 줄이 흔해 코드 한 덩어리가 여러 조각으로 찢긴다.
2. 앞부분 무언 폐기(데이터 유실): 문장부호가 없는 긴 코드 블록은 하나의
   "문장"으로 취급되고, `buf = s[-max_chars:]`가 **뒤 max_chars만 남기고 앞을
   버린다**. 코드의 핵심인 import·함수 시그니처가 조용히 사라진다.

── 이 모듈의 청커가 지키는 불변식 ────────────────────────
  (A) 코드 블록(```...```)은 원자 단위 — max_chars 이하면 절대 중간에서 안 쪼갠다.
  (B) 무손실 — 어떤 청크 조합에도 원문의 비공백 라인이 전부 살아 있다(앞부분 폐기 없음).
  (C) 코드가 max_chars를 넘으면 **라인 경계**로 나누고 각 조각을 다시 펜스로 감싸
      유효한 코드 블록으로 유지한다(글자 중간 절단 금지, 앞→뒤 순서 보존).
  (D) 설명문 + 인접 코드가 한 청크에 함께 들어갈 수 있으면 붙여 둔다(Q&A 맥락 유지).
  (E) 모든 청크는 max_chars 이하(단일 라인이 max_chars를 넘는 극단만 예외적으로 하드분할).

── 향후 이 모듈에 추가될 것(별도 단계) ──────────────────
  - html_to_markdown(): SO Body의 <pre><code>·<p> 등을 마크다운 펜스로 변환
  - build_provenance(): source/url/license/score 등 metadata 구성
  - dedup_key()/detect_languages(): 중복 제거 키, 프로그래밍 언어 태깅
지금 단계에서는 (A)~(E)를 보장하는 chunk_code_aware()만 구현·검증한다.

작성자: Nexus 팀 / 작성일: 2026-07-17
"""

from __future__ import annotations

import re
from collections.abc import Iterator

# 펜스 코드 블록: ```lang\n ... \n``` (여러 줄, 언어 표기는 선택).
# non-greedy(.*?)로 가장 가까운 닫는 펜스까지만 한 블록으로 잡는다.
# 닫히지 않은 ```는 매치되지 않아 그냥 산문으로 처리된다(안전 — 크래시 없음).
_FENCE_RE = re.compile(r"```[^\n]*\n.*?```", re.DOTALL)

# 문장 경계: 마침표/느낌표/물음표/。 뒤의 공백. (산문 재분할용)
_SENT_RE = re.compile(r"(?<=[.!?。])\s+")

# 문단 경계: 빈 줄. (산문 1차 분할용)
_PARA_RE = re.compile(r"\n\s*\n")


def _iter_segments(text: str) -> Iterator[tuple[str, str]]:
    """텍스트를 순서대로 ("prose"|"code", 조각) 세그먼트로 방출한다.

    펜스 코드 블록과 그 사이의 산문을 원래 순서 그대로 잘라 낸다.
    이렇게 분리해야 청킹 단계에서 "코드는 원자, 산문은 재분할" 규칙을
    각기 다르게 적용할 수 있다.
    """
    pos = 0
    for m in _FENCE_RE.finditer(text):
        # 코드 블록 앞의 산문 조각(있으면).
        if m.start() > pos:
            prose = text[pos : m.start()]
            if prose.strip():
                yield ("prose", prose)
        # 코드 블록 자체(원자 단위).
        yield ("code", m.group())
        pos = m.end()
    # 마지막 코드 블록 뒤에 남은 산문.
    if pos < len(text):
        tail = text[pos:]
        if tail.strip():
            yield ("prose", tail)


def _hard_split(s: str, max_chars: int, overlap: int) -> list[str]:
    """max_chars보다 긴 한 덩어리를 글자 단위로, **앞에서 뒤로** 나눈다(무손실).

    기존 split_into_chunks의 `s[-max_chars:]`(뒤만 남김) 버그를 바로잡는 지점이다.
    앞부분부터 max_chars씩 잘라 나가되, 이웃 조각이 overlap만큼 겹치게 해 경계
    문맥을 살린다. 전체 문자열이 빠짐없이 어느 조각엔가 반드시 포함된다.
    """
    if len(s) <= max_chars:
        return [s]
    step = max(1, max_chars - overlap)  # 다음 조각 시작 위치(겹침만큼 되감는다)
    pieces: list[str] = []
    start = 0
    while start < len(s):
        pieces.append(s[start : start + max_chars])
        if start + max_chars >= len(s):
            break  # 끝까지 커버했으면 종료
        start += step
    return pieces


def _split_prose(prose: str, max_chars: int, overlap: int) -> list[str]:
    """산문 세그먼트를 각 조각이 max_chars 이하가 되도록 나눈다(무손실).

    문단 → (넘치면) 문장 → (그래도 넘치면) 글자 하드분할 순으로 잘게 쪼갠다.
    각 조각은 max_chars 이하임이 보장되어, 상위 패킹 로직이 안전하게 담을 수 있다.
    """
    out: list[str] = []
    for para in _PARA_RE.split(prose):
        para = para.strip()
        if not para:
            continue
        if len(para) <= max_chars:
            out.append(para)
            continue
        # 문단이 너무 길면 문장 단위로 재분할하며 max_chars까지 모은다.
        buf = ""
        for sent in _SENT_RE.split(para):
            sent = sent.strip()
            if not sent:
                continue
            if len(sent) > max_chars:
                # 문장 하나가 한도를 넘으면(긴 URL·표 등) 버퍼를 비우고 하드분할.
                if buf:
                    out.append(buf)
                    buf = ""
                out.extend(_hard_split(sent, max_chars, overlap))
            elif len(buf) + len(sent) + 1 <= max_chars:
                buf = f"{buf} {sent}".strip() if buf else sent
            else:
                out.append(buf)
                buf = sent
        if buf:
            out.append(buf)
    return out


def _split_code_block(block: str, max_chars: int) -> list[str]:
    """max_chars를 넘는 코드 블록을 **라인 경계**로 나누고 각 조각을 재-펜스한다.

    글자 중간에서 자르지 않으므로 각 조각이 문법적으로 온전한 라인들로 구성된다.
    언어 표기(```python 등)를 모든 조각에 유지해 각 조각이 유효한 코드 블록이 된다.
    단일 라인이 max_chars를 넘는 극단(압축된 minified 코드 등)은 그 라인만 글자
    하드분할하되 어느 조각에도 빠짐없이 담는다(무손실 우선).
    """
    lines = block.split("\n")
    # 첫 줄 = 여는 펜스(```lang), 마지막 줄 = 닫는 펜스(```). 내부만 재분할한다.
    fence_open = lines[0] if lines and lines[0].startswith("```") else "```"
    inner = lines[1:-1] if len(lines) >= 2 and lines[-1].strip().startswith("```") else lines
    lang = fence_open[3:].strip()  # ```python → "python"
    reopen = f"```{lang}" if lang else "```"
    overhead = len(reopen) + 1 + 1 + 3  # 여는펜스+개행 + 개행+닫는펜스 대략치

    chunks: list[str] = []
    buf: list[str] = []
    buf_len = 0
    for line in inner:
        # 한 라인이 예산을 통째로 넘으면 그 라인만 글자 하드분할해 담는다(무손실).
        if len(line) + overhead > max_chars:
            if buf:
                chunks.append(f"{reopen}\n" + "\n".join(buf) + "\n```")
                buf, buf_len = [], 0
            for part in _hard_split(line, max_chars - overhead, 0):
                chunks.append(f"{reopen}\n{part}\n```")
            continue
        # 이 라인을 더하면 예산 초과 → 현재 버퍼를 조각으로 확정하고 새로 시작.
        if buf_len + len(line) + 1 + overhead > max_chars and buf:
            chunks.append(f"{reopen}\n" + "\n".join(buf) + "\n```")
            buf, buf_len = [], 0
        buf.append(line)
        buf_len += len(line) + 1
    if buf:
        chunks.append(f"{reopen}\n" + "\n".join(buf) + "\n```")
    return chunks


def chunk_code_aware(
    text: str,
    max_chars: int = 1500,
    overlap: int = 150,
) -> list[str]:
    """코드가 섞인 텍스트를 코드 블록을 보존하며 청크로 나눈다(무손실).

    임베딩 모델(e5-large)의 입력 창(~512토큰)을 고려해 max_chars 기본값을 1500으로
    둔다. 코드 블록은 원자 단위로 다루고, 넘칠 때만 라인 경계로 나눈다.

    Args:
        text: 코드 펜스(```)가 섞여 있을 수 있는 원문. 비면 빈 리스트.
        max_chars: 한 청크의 목표 최대 글자 수(임베딩 창 근사 상한).
        overlap: 산문 하드분할 시 이웃 조각이 겹칠 글자 수(코드에는 적용 안 함).

    Returns:
        청크 문자열 리스트. 불변식 (A)~(E)를 만족한다(모듈 docstring 참조).
    """
    if not text or not text.strip():
        return []

    chunks: list[str] = []
    buf = ""  # 현재 채우는 중인 청크(산문+짧은 코드가 함께 쌓일 수 있다)

    def flush() -> None:
        # 버퍼에 쌓인 내용을 청크로 확정하고 비운다(빈 버퍼는 무시).
        nonlocal buf
        if buf.strip():
            chunks.append(buf.strip())
        buf = ""

    for kind, seg in _iter_segments(text):
        if kind == "code":
            if len(seg) <= max_chars:
                # 짧은 코드: 현재 버퍼에 붙일 수 있으면 붙이고(설명+코드 동거),
                # 아니면 버퍼를 비우고 이 코드로 새 버퍼를 시작한다(원자성 유지).
                if buf and len(buf) + len(seg) + 2 <= max_chars:
                    buf = f"{buf}\n\n{seg}"
                else:
                    flush()
                    buf = seg
            else:
                # 긴 코드: 버퍼를 비운 뒤 라인 경계로 나눠 각각 독립 청크로.
                flush()
                chunks.extend(_split_code_block(seg, max_chars))
        else:  # prose
            for piece in _split_prose(seg, max_chars, overlap):
                if buf and len(buf) + len(piece) + 2 <= max_chars:
                    buf = f"{buf}\n\n{piece}"
                else:
                    flush()
                    buf = piece
    flush()
    return chunks
