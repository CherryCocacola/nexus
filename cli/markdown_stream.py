# 스트리밍으로 도착하는 모델 답변을 "완성된 블록 단위"로 마크다운 렌더링한다.
"""
CLI 답변 본문 마크다운 렌더러 (2026-08-18).

■ 왜 필요한가
    모델은 답변을 마크다운으로 쓴다(`## 제목`, `**굵게**`, `1.` 목록, ``` 코드블록).
    그런데 CLI 는 이 원문을 **그대로** 흘려보내고 있었다. 이유가 있었다 — 답변에
    들어 있는 대괄호(`[INFO]`, `list[int]`)를 Rich 가 콘솔 markup 으로 오해해
    글자를 먹거나 스트림이 끊기는 사고가 있었고, 그래서 리터럴 `Text` 로 감쌌다
    (B-4, 2026-07-29). 안전해졌지만 대신 화면이 기호 범벅이 됐다.

    이 모듈은 그 둘을 동시에 만족시킨다. Rich `Markdown` 렌더러블은 내용을
    markup 으로 재해석하지 않으므로 B-4 의 사고가 재발하지 않으면서, 제목·굵기·
    목록·코드블록이 제대로 그려진다.

■ 왜 "블록 단위"인가 (핵심 설계)
    마크다운은 문서 전체를 봐야 의미가 정해진다. 토큰이 올 때마다 다시 그리면
    (Rich `Live`) 긴 답변에서 화면이 통째로 깜빡이고 터미널 스크롤이 깨진다.
    그래서 **완성된 블록만** 확정해 출력한다. 마크다운의 블록 경계는 빈 줄이므로,
    빈 줄을 만나면 그 앞까지를 하나의 블록으로 확정한다.

    단 ``` 코드펜스 안의 빈 줄은 경계가 아니다. 펜스 상태를 따로 추적하는 이유다.

■ 지연 상한 (_MAX_PENDING_LINES)
    빈 줄 없이 길게 쓰는 답변은 경계가 안 잡혀 끝까지 아무것도 안 보인다.
    사용자에겐 멈춘 것처럼 보이므로, 펜스 밖에서 줄이 일정 수 이상 쌓이면
    거기까지 강제로 확정한다. 문단이 둘로 나뉘어도 렌더 결과는 자연스럽다.

작성자: 이현수 / 작성일: 2026-08-18
"""

from __future__ import annotations

from typing import Any

from rich.markdown import Markdown
from rich.text import Text

# 빈 줄이 오지 않아도 이만큼 쌓이면 확정해 출력한다(체감 지연 상한).
_MAX_PENDING_LINES = 20


class MarkdownStreamRenderer:
    """스트리밍 텍스트 조각을 받아 완성된 마크다운 블록만 내보내는 버퍼.

    사용법:
        r = MarkdownStreamRenderer()
        for block in r.feed(delta):   # 확정된 블록이 있으면 리스트로 반환
            console.print(block)
        for block in r.flush():       # 스트림이 끝나면 남은 것을 마저 뱉는다
            console.print(block)

    enabled=False 면 렌더링하지 않고 원문을 리터럴 Text 로 그대로 돌려준다
    (`/markdown` 토글 off — 원문을 봐야 하는 디버깅 상황용).
    """

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self._buf = ""

    def reset(self) -> None:
        """턴이 끝나거나 취소됐을 때 남은 찌꺼기를 버린다."""
        self._buf = ""

    @property
    def pending(self) -> bool:
        """아직 출력하지 않고 들고 있는 내용이 있는가."""
        return bool(self._buf)

    def feed(self, text: str) -> list[Any]:
        """텍스트 조각을 넣고, 이번에 '확정된' 블록들을 돌려준다."""
        if not text:
            return []
        if not self.enabled:
            # 렌더링 없이 원문 그대로 — 단 markup 오해석은 막는다(B-4).
            return [Text(text, end="")]
        self._buf += text
        return self._drain(final=False)

    def flush(self) -> list[Any]:
        """스트림 종료 — 버퍼에 남은 것을 마지막 블록으로 확정한다."""
        if not self._buf:
            return []
        if not self.enabled:
            out = [Text(self._buf, end="")]
            self._buf = ""
            return out
        return self._drain(final=True)

    # ─── 내부 ───

    def _drain(self, final: bool) -> list[Any]:
        """확정 가능한 블록을 가능한 만큼 꺼낸다."""
        out: list[Any] = []
        while True:
            block = self._take_block(final)
            if block is None:
                break
            if block.strip():
                out.append(Markdown(block))
        return out

    def _take_block(self, final: bool) -> str | None:
        """버퍼 앞에서 완성된 블록 하나를 떼어낸다. 없으면 None."""
        if not self._buf:
            return None

        if final:
            block, self._buf = self._buf, ""
            return block

        # 마지막 줄은 아직 오는 중일 수 있다 → 개행으로 끝난 줄만 판단 대상으로 삼는다.
        cut = self._buf.rfind("\n")
        if cut == -1:
            return None
        complete, tail = self._buf[: cut + 1], self._buf[cut + 1 :]

        # complete 는 항상 개행으로 끝나므로 split 결과의 마지막 ""는 버린다.
        # ★이걸 안 버리면 끝 개행을 '빈 줄'로 오인해 모든 줄이 즉시 확정된다
        #   (2026-08-18 단위테스트가 잡아낸 버그).
        lines = complete.split("\n")[:-1]
        if not lines:
            return None

        def _rest(from_idx: int) -> str:
            """남은 줄들을 개행을 되살려 버퍼 문자열로 되돌린다."""
            return "".join(line + "\n" for line in lines[from_idx:]) + tail

        in_fence = False
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("```"):
                # 펜스 여닫기. 닫는 순간 그 줄까지가 하나의 블록이다.
                in_fence = not in_fence
                if not in_fence:
                    self._buf = _rest(i + 1)
                    return "\n".join(lines[: i + 1])
                continue
            if in_fence:
                continue
            if stripped == "":
                # 펜스 밖 빈 줄 = 마크다운 블록 경계.
                self._buf = _rest(i + 1)
                return "\n".join(lines[:i])

        # 경계가 없다 — 너무 오래 들고 있지 않도록 상한에서 끊는다.
        if not in_fence and len(lines) >= _MAX_PENDING_LINES:
            self._buf = _rest(len(lines) - 1)
            return "\n".join(lines[:-1])

        return None
