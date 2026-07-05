"""
JavaScript/TypeScript 심볼 파서 — 정규식 기반 경량 구현 (Phase 10.0 확장, 2026-04-21).

[이 파일이 하는 일]
소스 코드(.js/.ts 등) 한 파일의 문자열을 받아, 그 안에 정의된
함수·클래스·메서드·인터페이스·타입 같은 "심볼"을 뽑아내는 파서다.
결과는 언어-독립 표현인 ``ParsedSymbol`` 리스트로 반환되고,
상위의 SymbolProjectIndexer가 이를 ``SymbolEntry``로 바꿔 DB(pgvector)에
적재한다. 즉 코드 검색(RAG)에서 "어떤 파일 몇 번째 줄에 무슨 함수가
있는지"를 색인하기 위한 전처리 단계다.

[왜 정규식인가]
Nexus는 에어갭(폐쇄망) 환경이라 ``esprima``/``tree-sitter`` 같은 외부
파서 라이브러리를 설치·사용할 수 없다. 그래서 완벽한 AST 파싱 대신
정규식으로 "실용적으로 충분한" 패턴만 잡는다. 정확도 100%가 목표가
아니라, 벡터 검색에서 쓸 심볼 목록을 빠르고 안정적으로 만드는 게 목표다.

[처리하는 패턴]
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

[한계 — 의도적으로 범위를 좁힌 부분]
  - 중첩 함수는 모듈 top-level만 추출 (클래스 내부 메서드 제외는 '깊이>1')
  - arrow function은 `const/let/var` 선언 형태만
  - 문자열/주석 안의 키워드는 간단한 cleanup 후 잡을 수 있음 (오탐 허용)
  - JSDoc은 심볼 바로 위 `/** ... */`만 최대 3줄 연결

[주요 구성]
  - 모듈 상단 정규식 상수(_RE_*): 함수/클래스/메서드/인터페이스/타입 매칭
  - 헬퍼 함수: _scrub_for_scan(주석·문자열 제거), _line_of/_compute_line_ends
    (오프셋→줄 번호 변환), _match_braces(중괄호 짝 찾기), _jsdoc_above(JSDoc 추출)
  - _JsBase: JS/TS 공통 파싱 로직 (parse 구현)
  - JavaScriptParser / TypeScriptParser: 확장자·언어명·TS 인식 여부만 다른 하위 클래스

작성자: 이현수 / 작성일: 2026-07-05
"""
from __future__ import annotations

import logging
import re

from core.rag.parsers.base import BaseParser, ParsedSymbol

# 모듈 전용 로거. 프로젝트 규칙상 "nexus.{module}" 네임스페이스를 따른다.
logger = logging.getLogger("nexus.rag.parsers.javascript")


# ─────────────────────────────────────────────
# 정규식 — 멀티라인(MULTILINE) 모드로 라인 시작 매칭
# ─────────────────────────────────────────────
# 아래 4개는 "주석/문자열을 지워내기(scrub)" 위한 패턴이다.
# 코드 헤더에서 키워드를 찾을 때, 주석이나 문자열 리터럴 안의 단어가
# 함수/클래스로 오인되지 않도록 먼저 이들을 공백으로 치환한다.
_RE_LINE_COMMENT = re.compile(r"//[^\n]*")            # 한 줄 주석  // ...
_RE_BLOCK_COMMENT = re.compile(r"/\*[\s\S]*?\*/")     # 블록 주석  /* ... */
_RE_STRING_DQ = re.compile(r'"(?:\\.|[^"\\])*"')      # 큰따옴표 문자열 "..."
_RE_STRING_SQ = re.compile(r"'(?:\\.|[^'\\])*'")      # 작은따옴표 문자열 '...'
_RE_STRING_BT = re.compile(r"`(?:\\.|[^`\\])*`")      # 백틱 템플릿 리터럴 `...`

# top-level 함수 선언 매칭.
# 캡처 그룹: indent(들여쓰기), exp(export/default 여부), async(비동기 여부),
#            name(함수 이름), args(괄호 안 인자). `function*`(제너레이터)도 허용.
_RE_FUNCTION = re.compile(
    r"^(?P<indent>[ \t]*)(?P<exp>export\s+(?:default\s+)?)?"
    r"(?P<async>async\s+)?function\s*\*?\s*(?P<name>[A-Za-z_$][\w$]*)"
    r"\s*\((?P<args>[^)]*)\)",
    re.MULTILINE,
)
# 화살표 함수 매칭 — `const/let/var 이름 = (인자) => ...` 형태만 잡는다.
# TypeScript 반환 타입 주석 `: Type`이 `=>` 앞에 올 수 있어 선택적으로 흡수한다.
_RE_ARROW = re.compile(
    r"^(?P<indent>[ \t]*)(?P<exp>export\s+(?:default\s+)?)?"
    r"(?:const|let|var)\s+(?P<name>[A-Za-z_$][\w$]*)\s*=\s*"
    r"(?P<async>async\s+)?\((?P<args>[^)]*)\)\s*(?::\s*[^=]+?)?=>",
    re.MULTILINE,
)
# 클래스 선언 매칭. base 그룹은 `extends 부모클래스`가 있을 때만 채워진다.
_RE_CLASS = re.compile(
    r"^(?P<indent>[ \t]*)(?P<exp>export\s+(?:default\s+)?)?"
    r"class\s+(?P<name>[A-Za-z_$][\w$]*)"
    r"(?:\s+extends\s+(?P<base>[A-Za-z_$][\w$.]*))?",
    re.MULTILINE,
)
# 클래스 body 내부 메서드 — 들여쓰기 2칸 이상일 때만 (top-level 함수와 구분)
# mods 그룹으로 static/public/private/protected/readonly/async/get/set/# 접두어를
# 흡수한다. TS 반환 타입 `: Type`도 `{` 앞에서 선택적으로 흡수한다.
_RE_METHOD = re.compile(
    r"^(?P<indent>[ \t]{2,})"
    r"(?P<mods>(?:static\s+|public\s+|private\s+|protected\s+|readonly\s+|async\s+|get\s+|set\s+|#)*)?"
    r"(?P<name>[A-Za-z_$#][\w$]*)\s*\((?P<args>[^)]*)\)\s*(?::\s*[^{]+?)?\s*\{",
    re.MULTILINE,
)

# TypeScript 전용 — interface / type 선언. TypeScriptParser에서만 사용한다.
_RE_INTERFACE = re.compile(
    r"^(?P<indent>[ \t]*)(?P<exp>export\s+)?interface\s+(?P<name>[A-Za-z_$][\w$]*)",
    re.MULTILINE,
)
_RE_TYPE = re.compile(
    r"^(?P<indent>[ \t]*)(?P<exp>export\s+)?type\s+(?P<name>[A-Za-z_$][\w$]*)\s*=",
    re.MULTILINE,
)

# 심볼 선언 "바로 위"에 붙은 JSDoc(`/** ... */`)을 잡는 패턴.
# 선언 줄 앞부분(prefix)의 맨 끝($)에 붙은 블록만 매칭하도록 구성했다.
_RE_JSDOC_ABOVE = re.compile(
    r"/\*\*(?P<body>[\s\S]*?)\*/\s*$",
    re.MULTILINE,
)


def _scrub_for_scan(source: str) -> str:
    """주석·문자열을 공백으로 대체하여 키워드 오탐을 줄인다.

    왜 필요한가: 정규식은 코드 구조를 이해하지 못하므로, 주석이나 문자열
    안에 들어 있는 `function`/`class` 같은 단어를 진짜 선언으로 착각할 수
    있다. 스캔 전에 그런 영역을 미리 지워 오탐을 줄인다.

    줄 번호 보존: 심볼의 line_start/line_end는 offset을 줄 번호로 환산해
    구하므로, 대체 시 원본과 길이·개행 위치가 어긋나면 안 된다. 그래서
    지우는 문자를 삭제하지 않고 "같은 길이의 공백"으로 바꾸고 개행은 남긴다.
    (헤더에서만 키워드를 찾으므로 완벽하지 않아도 실용적으로 충분.)

    반환: 주석·문자열이 공백으로 치환된, 원본과 동일 길이의 소스 문자열.
    """
    def repl(m: re.Match[str]) -> str:
        # 매칭된 구간 전체를 받아, 개행(\n)만 보존하고 나머지는 공백으로 바꾼다.
        text = m.group(0)
        # 줄 수 유지를 위해 개행은 남긴다
        return "".join(c if c == "\n" else " " for c in text)

    # 블록 주석 → 라인 주석 → 문자열 순으로 차례로 지운다. 앞 단계에서 지워진
    # 영역은 이미 공백이라 뒤 단계 정규식에 다시 걸리지 않는다.
    out = _RE_BLOCK_COMMENT.sub(repl, source)
    out = _RE_LINE_COMMENT.sub(repl, out)
    out = _RE_STRING_DQ.sub(repl, out)
    out = _RE_STRING_SQ.sub(repl, out)
    out = _RE_STRING_BT.sub(repl, out)
    return out


def _line_of(offset: int, line_ends: list[int]) -> int:
    """문자 offset이 속한 줄 번호를 1부터 세어 반환한다.

    line_ends는 _compute_line_ends가 만든 "각 개행 문자의 인덱스" 정렬
    리스트다. offset보다 작은 개행이 k개면 그 앞에 k줄이 끝난 것이므로
    현재 줄은 k+1번째다. 이진 탐색(bisect_right)으로 k를 O(log n)에 구한다.
    """
    import bisect
    return bisect.bisect_right(line_ends, offset) + 1


def _compute_line_ends(text: str) -> list[int]:
    """소스 전체에서 개행 문자('\\n')의 인덱스를 순서대로 모은 리스트를 만든다.

    _line_of가 이 리스트를 이진 탐색해 offset→줄 번호 변환에 쓴다.
    파일당 한 번만 계산해 재사용하므로 반복 스캔 비용을 아낀다.
    """
    return [i for i, c in enumerate(text) if c == "\n"]


def _match_braces(text: str, start: int) -> int:
    """start 위치의 '{'부터 짝이 맞는 '}'의 위치(offset)를 반환한다.

    중괄호 깊이(depth)를 세며 앞으로 훑는다. 여는 괄호마다 +1, 닫는
    괄호마다 -1 하다가 depth가 0이 되는 지점이 짝이 맞는 닫는 괄호다.
    이를 이용해 함수/클래스/메서드 본문의 끝 줄을 알아낸다.
    실패(짝을 못 찾음) 시 문자열 끝 인덱스를 반환해 안전하게 마무리한다.
    """
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
                # depth가 0으로 돌아온 지점 = 처음 '{'와 짝이 맞는 '}'.
                return i
        i += 1
    return n - 1


def _jsdoc_above(source: str, line_start_offset: int) -> str:
    """심볼 선언 바로 위에 붙은 JSDoc(`/** ... */`)이 있으면 본문을 정리해 반환한다.

    동작: 선언 줄 시작 오프셋 앞부분(prefix)에서 맨 끝에 붙은 JSDoc 블록을
    찾고, 각 줄 앞에 관례적으로 붙는 ` * ` 장식을 제거한 뒤 빈 줄을 걸러
    이어 붙인다. JSDoc이 없으면 빈 문자열을 반환한다.

    매개변수:
      source            : 원본 소스 전체 (scrub 전 원문 — 주석 내용을 살려야 하므로)
      line_start_offset : 심볼 선언 줄의 시작 오프셋
    반환: 정리된 doc 문자열(없으면 "").
    """
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
    """JS/TS 공통 파싱 로직을 담은 베이스 클래스.

    JavaScript와 TypeScript는 심볼 추출 방식이 거의 같고, 차이는
    "확장자·언어명·TS 전용 문법(interface/type) 인식 여부"뿐이다. 그래서
    공통 로직을 여기 두고, 하위 클래스에서 클래스 속성만 바꾼다.
    (언더스코어 접두 `_JsBase`는 외부에 직접 노출하지 않는 내부용 표시.)
    """

    # 하위 클래스에서 set
    # TS 전용 문법(interface/type)을 파싱할지 여부. TypeScriptParser만 True.
    _includes_typescript: bool = False

    def parse(self, source: str, file_path: str) -> list[ParsedSymbol]:
        """소스 한 파일에서 심볼을 추출해 ParsedSymbol 리스트로 반환한다.

        전체 흐름:
          0) scrub — 주석·문자열을 공백으로 지운 사본(scrubbed)을 만든다.
             줄 번호 계산용 line_ends도 원본 기준으로 미리 구한다.
          1) top-level 함수 선언 추출
          2) top-level 화살표 함수(const/let/var = ...) 추출
          3) 클래스 + 그 본문 내부 메서드 추출
          4) (TS일 때만) interface / type 추출
        각 심볼의 line_start/line_end는 offset을 _line_of로 환산해 채운다.
        docstring은 원본(source)에서 선언 위 JSDoc을 찾아 넣는다.

        매개변수:
          source    : 파일 소스 전체 문자열
          file_path : 파일 경로 (현재 로직에선 미사용이나 인터페이스상 유지)
        반환: 추출된 ParsedSymbol 리스트 (없으면 빈 리스트).
        """
        # scrubbed: 키워드 오탐을 막기 위해 주석·문자열을 공백 처리한 사본.
        # 위치(offset)는 원본과 1:1로 대응하므로 줄 번호 계산에 그대로 쓸 수 있다.
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
            # 함수 이름 뒤 첫 '{'를 본문 시작으로 보고, 짝 맞는 '}'까지의 줄을 끝으로.
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
            # 부모 클래스가 있으면 시그니처에 (Base) 형태로 표기, 없으면 빈 문자열.
            sig = f"({base})" if base else ""
            line_start = _line_of(m.start(), line_ends)
            brace_open = scrubbed.find("{", m.end())
            if brace_open == -1:
                # 본문 '{'를 못 찾으면(선언만 있는 비정상 케이스) 한 줄짜리로 처리.
                line_end = line_start
                body_text = ""
                body_off = m.end()
            else:
                # 클래스 본문 범위를 구해, 내부 메서드 스캔 대상 문자열(body_text)과
                # 그 시작 오프셋(body_off)을 확보한다. body_off는 나중에 메서드의
                # body 내 상대 위치를 원본 절대 오프셋으로 되돌릴 때 쓴다.
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
            #      body_text(클래스 본문 사본) 안에서만 메서드를 찾는다.
            for mm in _RE_METHOD.finditer(body_text):
                # 예약어/제어문 제외
                # if(...) / for(...) 같은 제어문도 "이름(인자){" 형태라 메서드로
                # 오인될 수 있어, 대표 키워드는 이름 단계에서 걸러낸다.
                mname = mm.group("name")
                if mname in {"if", "for", "while", "switch", "catch", "return"}:
                    continue
                # 생성자는 별도 이름으로
                # mods 문자열에서 async/static 여부를 판별한다.
                mods = (mm.group("mods") or "").strip()
                is_async = "async" in mods
                is_static = "static" in mods
                kind = "async_method" if is_async else "method"
                margs = (mm.group("args") or "").strip()
                msig = f"({margs})"
                # body_text 내 상대 시작 위치를 원본 절대 오프셋으로 환산한다.
                m_abs_start = body_off + mm.start()
                m_line_start = _line_of(m_abs_start, line_ends)
                # 메서드 본문 끝
                # body_text 기준으로 '{'를 찾은 뒤, 절대 오프셋으로 바꿔 원본
                # scrubbed에서 짝 맞는 '}'를 찾아 끝 줄을 구한다.
                body_brace_open = body_text.find("{", mm.end())
                if body_brace_open != -1:
                    brace_abs = body_off + body_brace_open
                    brace_abs_close = _match_braces(scrubbed, brace_abs)
                    m_line_end = _line_of(brace_abs_close, line_ends)
                else:
                    m_line_end = m_line_start
                # static 메서드면 태그로 표시(검색·필터 용). 이름은 "클래스.메서드"로 한정.
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
        #    JavaScript 파일에는 없는 문법이라 TS일 때(_includes_typescript)만 스캔.
        if self._includes_typescript:
            for m in _RE_INTERFACE.finditer(scrubbed):
                name = m.group("name")
                line_start = _line_of(m.start(), line_ends)
                # interface 본문 { ... }의 끝을 찾아 끝 줄로. 본문이 없으면 한 줄 처리.
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
                # type 별칭은 보통 한 줄이므로 끝 줄을 시작 줄과 동일하게 둔다.
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
        offset 위치가 가장 가까운 class 본문 {...} 내부인지 heuristic으로 확인한다.

        왜 필요한가: 클래스 내부에 있는 function 선언은 "top-level 함수"가
        아니라 메서드/내부 함수이므로, 1)·2) 단계에서 중복으로 잡지 않도록
        걸러내야 한다.

        방식(휴리스틱): offset 앞쪽에서 가장 가까운 `class ` 키워드를 찾고,
        그 뒤 첫 '{'부터 짝 맞는 '}'까지의 범위 안에 offset이 들어오는지
        중괄호 깊이를 세며 판정한다. 정규식만으로는 완벽히 알 수 없어
        근사적으로 판단한다(오탐 일부 허용).
        반환: 클래스 본문 내부이면 True, 아니면 False.
        """
        # 단순 heuristic — 직전 class 키워드를 찾고, 그 이후 { 까지 범위 안인지 확인
        last_class = scrubbed.rfind("class ", 0, offset)
        if last_class == -1:
            # 앞쪽에 class가 전혀 없으면 클래스 내부일 수 없다.
            return False
        brace_open = scrubbed.find("{", last_class)
        if brace_open == -1 or brace_open >= offset:
            # 클래스 본문 '{'가 없거나, 그 '{'가 offset보다 뒤에 있으면 내부 아님.
            return False
        # brace_open..close 안에 offset이 있으면 class 내부
        # 여는 '{'부터 깊이를 세어 짝 맞는 '}'를 찾고, 그 사이에 offset이 있는지 본다.
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
    """JavaScript 파일용 파서 — .js/.jsx/.mjs/.cjs를 담당. TS 문법은 인식하지 않는다."""

    language = "javascript"
    extensions = (".js", ".jsx", ".mjs", ".cjs")
    _includes_typescript = False


class TypeScriptParser(_JsBase):
    """TypeScript 파일용 파서 — .ts/.tsx를 담당. interface/type까지 함께 추출한다."""

    language = "typescript"
    extensions = (".ts", ".tsx")
    _includes_typescript = True
