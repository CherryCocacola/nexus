"""
Go 심볼 파서 — 정규식 경량 구현 (Phase 10.0 확장, 2026-04-21).

패턴:
  함수:        func Name(args) returnType { ... }
  메서드:      func (receiver Type) Name(args) returnType { ... }
  구조체:      type Name struct { ... }
  인터페이스:  type Name interface { ... }
  타입 별칭:   type Name = ...  또는  type Name Underlying

에어갭 환경이라 `go/parser` 같은 Go 툴체인을 쓰지 않고 정규식으로 처리한다.
심볼 검색 목적으로는 충분 — 정확한 타입 추론이 필요한 도구는 별도 영역.
"""
from __future__ import annotations

import logging
import re

from core.rag.parsers.base import BaseParser, ParsedSymbol

logger = logging.getLogger("nexus.rag.parsers.go")

_RE_FUNC = re.compile(
    r"^func\s+"
    r"(?:\((?P<receiver>[^)]+)\)\s+)?"
    r"(?P<name>[A-Za-z_][\w]*)"
    r"\s*\((?P<args>[^)]*)\)"
    r"\s*(?P<ret>[^{]*?)\{",
    re.MULTILINE,
)
_RE_TYPE = re.compile(
    r"^type\s+(?P<name>[A-Za-z_][\w]*)"
    r"\s+(?P<kind>struct|interface)\s*\{",
    re.MULTILINE,
)
_RE_TYPE_ALIAS = re.compile(
    r"^type\s+(?P<name>[A-Za-z_][\w]*)\s*(?:=\s*)?(?P<base>[A-Za-z_][\w.\[\]*]+)\s*$",
    re.MULTILINE,
)

# Go의 doc comment은 바로 위 `// ...` 연속 줄
_RE_DOC_COMMENT_ABOVE = re.compile(r"((?:^[ \t]*//[^\n]*\n)+)\s*$", re.MULTILINE)
_RE_LINE_COMMENT = re.compile(r"//[^\n]*")
_RE_BLOCK_COMMENT = re.compile(r"/\*[\s\S]*?\*/")
_RE_STRING_DQ = re.compile(r'"(?:\\.|[^"\\])*"')
_RE_STRING_BT = re.compile(r"`[^`]*`")


def _scrub(source: str) -> str:
    """주석/문자열을 공백으로 치환 (라인 보존)."""
    def repl(m: re.Match[str]) -> str:
        return "".join(c if c == "\n" else " " for c in m.group(0))
    out = _RE_BLOCK_COMMENT.sub(repl, source)
    out = _RE_LINE_COMMENT.sub(repl, out)
    out = _RE_STRING_DQ.sub(repl, out)
    out = _RE_STRING_BT.sub(repl, out)
    return out


def _line_ends(text: str) -> list[int]:
    return [i for i, c in enumerate(text) if c == "\n"]


def _line_of(offset: int, line_ends: list[int]) -> int:
    import bisect
    return bisect.bisect_right(line_ends, offset) + 1


def _match_braces(text: str, start: int) -> int:
    depth = 0
    i = start
    n = len(text)
    while i < n:
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return n - 1


def _doc_comment_above(source: str, offset: int) -> str:
    prefix = source[:offset]
    m = _RE_DOC_COMMENT_ABOVE.search(prefix)
    if not m:
        return ""
    block = m.group(1)
    lines = [re.sub(r"^\s*//\s?", "", ln).rstrip() for ln in block.splitlines()]
    return "\n".join(ln for ln in lines if ln).strip()


def _receiver_type(receiver: str) -> str:
    """'(s *Server)' → 'Server', '(s Server)' → 'Server'."""
    parts = receiver.strip().split()
    if not parts:
        return ""
    t = parts[-1].lstrip("*")
    # 제네릭 타입 파라미터 제거 `Server[T]` → `Server`
    return re.split(r"[\[\s]", t, maxsplit=1)[0]


class GoParser(BaseParser):
    language = "go"
    extensions = (".go",)

    def parse(self, source: str, file_path: str) -> list[ParsedSymbol]:
        scrubbed = _scrub(source)
        lines = _line_ends(source)
        out: list[ParsedSymbol] = []

        for m in _RE_FUNC.finditer(scrubbed):
            name = m.group("name")
            receiver = m.group("receiver") or ""
            args = (m.group("args") or "").strip()
            ret = (m.group("ret") or "").strip()
            sig = f"({args})"
            if ret:
                sig += f" {ret}"

            brace_open = scrubbed.find("{", m.end() - 1)
            brace_close = _match_braces(scrubbed, brace_open) if brace_open != -1 else m.end()
            line_start = _line_of(m.start(), lines)
            line_end = _line_of(brace_close, lines)
            doc = _doc_comment_above(source, m.start())

            if receiver:
                recv_type = _receiver_type(receiver)
                qual = f"{recv_type}.{name}" if recv_type else name
                kind = "method"
            else:
                qual = name
                kind = "function"

            out.append(ParsedSymbol(
                kind=kind,
                name=name,
                qualified_name=qual,
                signature=sig,
                docstring=doc,
                line_start=line_start,
                line_end=line_end,
                language=self.language,
                extra_tags=("exported",) if name[:1].isupper() else (),
            ))

        for m in _RE_TYPE.finditer(scrubbed):
            name = m.group("name")
            kind = "struct" if m.group("kind") == "struct" else "interface"
            brace_open = scrubbed.find("{", m.end() - 1)
            brace_close = _match_braces(scrubbed, brace_open) if brace_open != -1 else m.end()
            line_start = _line_of(m.start(), lines)
            line_end = _line_of(brace_close, lines)
            out.append(ParsedSymbol(
                kind=kind,
                name=name,
                qualified_name=name,
                signature="",
                docstring=_doc_comment_above(source, m.start()),
                line_start=line_start,
                line_end=line_end,
                language=self.language,
                extra_tags=("exported",) if name[:1].isupper() else (),
            ))

        for m in _RE_TYPE_ALIAS.finditer(scrubbed):
            name = m.group("name")
            base = m.group("base")
            line_start = _line_of(m.start(), lines)
            out.append(ParsedSymbol(
                kind="type",
                name=name,
                qualified_name=name,
                signature=f"= {base}" if base else "",
                docstring=_doc_comment_above(source, m.start()),
                line_start=line_start,
                line_end=line_start,
                language=self.language,
                extra_tags=("exported",) if name[:1].isupper() else (),
            ))

        return out
