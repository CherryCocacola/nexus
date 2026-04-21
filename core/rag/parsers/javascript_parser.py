"""
JavaScript/TypeScript 심볼 파서 — 정규식 경량 구현 (Phase 10.0 확장, 2026-04-21).

에어갭 환경이라 `esprima`/`tree-sitter` 같은 외부 파서를 쓰지 않는다.
정규식으로 실용 범위 내 패턴만 처리한다:

  함수:
    function foo(args) { ... }
    async function foo(args) { ... }
    export [default] function foo(...) { ... }
    const foo = (args) => { ... }
    const foo = async (args) => { ... }
  클래스:
    class Foo [extends Bar] { ... }
    export [default] class Foo { ... }
  메서드 (클래스 body 내부):
    methodName(args) { ... }
    async methodName(args) { ... }
    static methodName(args) { ... }
    get / set 접근자는 함수와 동일 처리
  TypeScript 전용 (TypeScriptParser만 인식):
    interface Foo { ... }
    type Foo = ...

한계 (의도적 범위 제한):
  - 중첩 함수는 모듈 top-level만 추출 (클래스 내부 메서드 제외는 '깊이>1')
  - arrow function은 `const/let/var` 선언 형태만
  - 문자열/주석 안의 키워드는 간단한 cleanup 후 잡을 수 있음 (오탐 허용)
  - JSDoc은 심볼 바로 위 `/** ... */`만 최대 3줄 연결
"""
from __future__ import annotations

import logging
import re

from core.rag.parsers.base import BaseParser, ParsedSymbol

logger = logging.getLogger("nexus.rag.parsers.javascript")


# ─────────────────────────────────────────────
# 정규식 — 멀티라인(MULTILINE) 모드로 라인 시작 매칭
# ─────────────────────────────────────────────
_RE_LINE_COMMENT = re.compile(r"//[^\n]*")
_RE_BLOCK_COMMENT = re.compile(r"/\*[\s\S]*?\*/")
_RE_STRING_DQ = re.compile(r'"(?:\\.|[^"\\])*"')
_RE_STRING_SQ = re.compile(r"'(?:\\.|[^'\\])*'")
_RE_STRING_BT = re.compile(r"`(?:\\.|[^`\\])*`")

_RE_FUNCTION = re.compile(
    r"^(?P<indent>[ \t]*)(?P<exp>export\s+(?:default\s+)?)?"
    r"(?P<async>async\s+)?function\s*\*?\s*(?P<name>[A-Za-z_$][\w$]*)"
    r"\s*\((?P<args>[^)]*)\)",
    re.MULTILINE,
)
_RE_ARROW = re.compile(
    r"^(?P<indent>[ \t]*)(?P<exp>export\s+(?:default\s+)?)?"
    r"(?:const|let|var)\s+(?P<name>[A-Za-z_$][\w$]*)\s*=\s*"
    r"(?P<async>async\s+)?\((?P<args>[^)]*)\)\s*(?::\s*[^=]+?)?=>",
    re.MULTILINE,
)
_RE_CLASS = re.compile(
    r"^(?P<indent>[ \t]*)(?P<exp>export\s+(?:default\s+)?)?"
    r"class\s+(?P<name>[A-Za-z_$][\w$]*)"
    r"(?:\s+extends\s+(?P<base>[A-Za-z_$][\w$.]*))?",
    re.MULTILINE,
)
# 클래스 body 내부 메서드 — 들여쓰기 2칸 이상일 때만 (top-level 함수와 구분)
_RE_METHOD = re.compile(
    r"^(?P<indent>[ \t]{2,})"
    r"(?P<mods>(?:static\s+|public\s+|private\s+|protected\s+|readonly\s+|async\s+|get\s+|set\s+|#)*)?"
    r"(?P<name>[A-Za-z_$#][\w$]*)\s*\((?P<args>[^)]*)\)\s*(?::\s*[^{]+?)?\s*\{",
    re.MULTILINE,
)

# TypeScript 전용
_RE_INTERFACE = re.compile(
    r"^(?P<indent>[ \t]*)(?P<exp>export\s+)?interface\s+(?P<name>[A-Za-z_$][\w$]*)",
    re.MULTILINE,
)
_RE_TYPE = re.compile(
    r"^(?P<indent>[ \t]*)(?P<exp>export\s+)?type\s+(?P<name>[A-Za-z_$][\w$]*)\s*=",
    re.MULTILINE,
)

_RE_JSDOC_ABOVE = re.compile(
    r"/\*\*(?P<body>[\s\S]*?)\*/\s*$",
    re.MULTILINE,
)


def _scrub_for_scan(source: str) -> str:
    """주석·문자열을 공백으로 대체하여 키워드 오탐을 줄인다.

    라인 번호 보존을 위해 같은 길이의 공백으로 대체한다.
    (헤더에서만 키워드를 찾으므로 완벽하지 않아도 실용적으로 충분.)
    """
    def repl(m: re.Match[str]) -> str:
        text = m.group(0)
        # 줄 수 유지를 위해 개행은 남긴다
        return "".join(c if c == "\n" else " " for c in text)

    out = _RE_BLOCK_COMMENT.sub(repl, source)
    out = _RE_LINE_COMMENT.sub(repl, out)
    out = _RE_STRING_DQ.sub(repl, out)
    out = _RE_STRING_SQ.sub(repl, out)
    out = _RE_STRING_BT.sub(repl, out)
    return out


def _line_of(offset: int, line_ends: list[int]) -> int:
    """offset이 속한 줄 번호 (1-indexed). line_ends는 각 줄 끝 인덱스의 sorted list."""
    import bisect
    return bisect.bisect_right(line_ends, offset) + 1


def _compute_line_ends(text: str) -> list[int]:
    return [i for i, c in enumerate(text) if c == "\n"]


def _match_braces(text: str, start: int) -> int:
    """start 위치의 '{'부터 짝 맞는 '}' 위치(offset)를 반환. 실패 시 len(text)."""
    depth = 0
    i = start
    n = len(text)
    while i < n:
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return n - 1


def _jsdoc_above(source: str, line_start_offset: int) -> str:
    """심볼 선언 바로 위 JSDoc(`/** ... */`)이 있으면 내용을 반환."""
    prefix = source[:line_start_offset]
    m = _RE_JSDOC_ABOVE.search(prefix)
    if not m:
        return ""
    body = m.group("body")
    # 각 줄의 앞 '*' 제거
    lines = [re.sub(r"^\s*\*\s?", "", ln).rstrip() for ln in body.splitlines()]
    return "\n".join(ln for ln in lines if ln).strip()


# ─────────────────────────────────────────────
# JS/TS 공통 파서
# ─────────────────────────────────────────────
class _JsBase(BaseParser):
    """JS/TS 공통 로직 — 언어 식별자와 확장자만 하위에서 바꾼다."""

    # 하위 클래스에서 set
    _includes_typescript: bool = False

    def parse(self, source: str, file_path: str) -> list[ParsedSymbol]:
        scrubbed = _scrub_for_scan(source)
        line_ends = _compute_line_ends(source)
        out: list[ParsedSymbol] = []

        # 1) top-level function 선언
        for m in _RE_FUNCTION.finditer(scrubbed):
            if self._is_inside_class(scrubbed, m.start()):
                # 클래스 본문 안에 떨어진 function은 메서드로 잡힌다 → 여기선 skip
                continue
            is_async = bool(m.group("async"))
            name = m.group("name")
            args = (m.group("args") or "").strip()
            sig = f"({args})"
            line_start = _line_of(m.start(), line_ends)
            brace_open = scrubbed.find("{", m.end())
            brace_close = (
                _match_braces(scrubbed, brace_open) if brace_open != -1 else m.end()
            )
            line_end = _line_of(brace_close, line_ends)
            out.append(ParsedSymbol(
                kind="async_function" if is_async else "function",
                name=name, qualified_name=name,
                signature=sig,
                docstring=_jsdoc_above(source, m.start()),
                line_start=line_start, line_end=line_end,
                language=self.language,
            ))

        # 2) arrow function (const/let/var)
        for m in _RE_ARROW.finditer(scrubbed):
            if self._is_inside_class(scrubbed, m.start()):
                continue
            is_async = bool(m.group("async"))
            name = m.group("name")
            args = (m.group("args") or "").strip()
            sig = f"({args})"
            line_start = _line_of(m.start(), line_ends)
            # arrow의 body 끝 탐지는 까다로우므로 최소 +5줄로 근사 (vector 검색엔 영향 미미)
            line_end = line_start + 5
            out.append(ParsedSymbol(
                kind="async_function" if is_async else "function",
                name=name, qualified_name=name,
                signature=sig,
                docstring=_jsdoc_above(source, m.start()),
                line_start=line_start, line_end=line_end,
                language=self.language,
            ))

        # 3) class 및 내부 메서드
        for m in _RE_CLASS.finditer(scrubbed):
            name = m.group("name")
            base = m.group("base") or ""
            sig = f"({base})" if base else ""
            line_start = _line_of(m.start(), line_ends)
            brace_open = scrubbed.find("{", m.end())
            if brace_open == -1:
                line_end = line_start
                body_text = ""
                body_off = m.end()
            else:
                brace_close = _match_braces(scrubbed, brace_open)
                line_end = _line_of(brace_close, line_ends)
                body_text = scrubbed[brace_open + 1: brace_close]
                body_off = brace_open + 1
            out.append(ParsedSymbol(
                kind="class",
                name=name, qualified_name=name,
                signature=sig,
                docstring=_jsdoc_above(source, m.start()),
                line_start=line_start, line_end=line_end,
                language=self.language,
            ))

            # 3-b) 클래스 본문 내부 메서드
            for mm in _RE_METHOD.finditer(body_text):
                # 예약어/제어문 제외
                mname = mm.group("name")
                if mname in {"if", "for", "while", "switch", "catch", "return"}:
                    continue
                # 생성자는 별도 이름으로
                mods = (mm.group("mods") or "").strip()
                is_async = "async" in mods
                is_static = "static" in mods
                kind = "async_method" if is_async else "method"
                margs = (mm.group("args") or "").strip()
                msig = f"({margs})"
                m_abs_start = body_off + mm.start()
                m_line_start = _line_of(m_abs_start, line_ends)
                # 메서드 본문 끝
                body_brace_open = body_text.find("{", mm.end())
                if body_brace_open != -1:
                    brace_abs = body_off + body_brace_open
                    brace_abs_close = _match_braces(scrubbed, brace_abs)
                    m_line_end = _line_of(brace_abs_close, line_ends)
                else:
                    m_line_end = m_line_start
                tags = tuple(t for t in ("static",) if is_static)
                out.append(ParsedSymbol(
                    kind=kind,
                    name=mname,
                    qualified_name=f"{name}.{mname}",
                    signature=msig,
                    docstring=_jsdoc_above(source, m_abs_start),
                    line_start=m_line_start, line_end=m_line_end,
                    language=self.language,
                    extra_tags=tags,
                ))

        # 4) TypeScript 전용 — interface / type
        if self._includes_typescript:
            for m in _RE_INTERFACE.finditer(scrubbed):
                name = m.group("name")
                line_start = _line_of(m.start(), line_ends)
                brace_open = scrubbed.find("{", m.end())
                if brace_open == -1:
                    line_end = line_start
                else:
                    brace_close = _match_braces(scrubbed, brace_open)
                    line_end = _line_of(brace_close, line_ends)
                out.append(ParsedSymbol(
                    kind="interface",
                    name=name, qualified_name=name,
                    signature="",
                    docstring=_jsdoc_above(source, m.start()),
                    line_start=line_start, line_end=line_end,
                    language=self.language,
                ))
            for m in _RE_TYPE.finditer(scrubbed):
                name = m.group("name")
                line_start = _line_of(m.start(), line_ends)
                line_end = line_start
                out.append(ParsedSymbol(
                    kind="type",
                    name=name, qualified_name=name,
                    signature="",
                    docstring=_jsdoc_above(source, m.start()),
                    line_start=line_start, line_end=line_end,
                    language=self.language,
                ))

        return out

    @staticmethod
    def _is_inside_class(scrubbed: str, offset: int) -> bool:
        """
        offset 위치가 가장 가까운 class 본문 {...} 내부인지 heuristic 확인.
        class 내부 function 선언은 top-level이 아니다.
        """
        # 단순 heuristic — 직전 class 키워드를 찾고, 그 이후 { 까지 범위 안인지 확인
        last_class = scrubbed.rfind("class ", 0, offset)
        if last_class == -1:
            return False
        brace_open = scrubbed.find("{", last_class)
        if brace_open == -1 or brace_open >= offset:
            return False
        # brace_open..close 안에 offset이 있으면 class 내부
        depth = 0
        i = brace_open
        n = len(scrubbed)
        while i < n:
            c = scrubbed[i]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return brace_open <= offset <= i
            i += 1
        return False


class JavaScriptParser(_JsBase):
    language = "javascript"
    extensions = (".js", ".jsx", ".mjs", ".cjs")
    _includes_typescript = False


class TypeScriptParser(_JsBase):
    language = "typescript"
    extensions = (".ts", ".tsx")
    _includes_typescript = True
