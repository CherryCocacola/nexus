"""
Go 심볼 파서 — 정규식 기반 경량 구현 (Phase 10.0 확장, 2026-04-21).

[이 파일이 하는 일]
Go 소스 코드(.go) 한 파일을 문자열로 받아, 그 안에 정의된 심볼
(함수 / 메서드 / 구조체 / 인터페이스 / 타입)을 뽑아 언어-독립 표현인
`ParsedSymbol` 리스트로 반환한다. 이 결과는 상위의 SymbolProjectIndexer가
`SymbolEntry`로 변환해 RAG용 심볼 DB에 적재한다. 즉, "코드 안에 어떤
심볼이 어디(줄 번호)에 있는지"를 색인하기 위한 파서다.

[인식하는 패턴]
  함수:        func Name(args) returnType { ... }
  메서드:      func (receiver Type) Name(args) returnType { ... }
  구조체:      type Name struct { ... }
  인터페이스:  type Name interface { ... }
  타입 별칭:   type Name = ...  또는  type Name Underlying

[왜 정규식인가]
에어갭(폐쇄망) 환경이라 `go/parser` 같은 정식 Go 툴체인을 설치·실행할 수
없다. 그래서 완전한 구문 분석 대신 정규식으로 "심볼 이름과 위치"만 뽑는
경량 방식을 쓴다. 심볼 검색/색인 목적에는 충분하며, 정확한 타입 추론이
필요한 도구는 이 파서의 책임 범위가 아니다.

[핵심 구성]
  - GoParser: BaseParser를 상속한 실제 파서 클래스 (parse() 진입점)
  - _scrub(): 주석/문자열을 지워 정규식 오탐을 방지하는 전처리
  - _match_braces(): 중괄호 짝을 세어 블록의 끝 위치를 찾음
  - _doc_comment_above(): 심볼 바로 위의 `// ...` 문서 주석 추출
  - _line_of() / _line_ends(): 문자 오프셋 → 1-based 줄 번호 변환

의존: core.rag.parsers.base 의 BaseParser, ParsedSymbol

작성자: 이현수 / 작성일: 2026-07-05
"""
from __future__ import annotations

import logging
import re

from core.rag.parsers.base import BaseParser, ParsedSymbol

# 모듈 전용 로거. 프로젝트 규칙상 "nexus.{module}" 네임스페이스를 따른다.
logger = logging.getLogger("nexus.rag.parsers.go")

# ---------------------------------------------------------------------------
# 심볼 추출용 정규식들
# 공통적으로 re.MULTILINE을 써서 `^`가 각 줄의 시작에 매칭되도록 한다.
# (Go의 선언은 대부분 줄 맨 앞에서 시작하기 때문)
# ---------------------------------------------------------------------------

# 함수/메서드 선언 매칭.
#   ^func\s+                              : 줄 시작의 func 키워드
#   (?:\((?P<receiver>[^)]+)\)\s+)?       : (옵션) 메서드의 리시버 `(s *Server)`
#   (?P<name>[A-Za-z_][\w]*)              : 함수/메서드 이름
#   \s*\((?P<args>[^)]*)\)                : 괄호 안 인자 목록
#   \s*(?P<ret>[^{]*?)\{                  : (옵션) 반환 타입 뒤 여는 중괄호 `{`
# receiver 그룹이 있으면 메서드, 없으면 일반 함수로 판별한다.
_RE_FUNC = re.compile(
    r"^func\s+"
    r"(?:\((?P<receiver>[^)]+)\)\s+)?"
    r"(?P<name>[A-Za-z_][\w]*)"
    r"\s*\((?P<args>[^)]*)\)"
    r"\s*(?P<ret>[^{]*?)\{",
    re.MULTILINE,
)
# 구조체/인터페이스 선언 매칭: `type Name struct {` 또는 `type Name interface {`.
# kind 그룹으로 struct / interface 를 구분한다.
_RE_TYPE = re.compile(
    r"^type\s+(?P<name>[A-Za-z_][\w]*)"
    r"\s+(?P<kind>struct|interface)\s*\{",
    re.MULTILINE,
)
# 타입 별칭/정의 매칭: `type Name = Base` 또는 `type Name Base`.
# base 그룹에는 밑줄 타입 이름(패키지 경로 `.`, 슬라이스 `[]`, 포인터 `*` 포함)이 들어온다.
# 주의: struct/interface 는 위 _RE_TYPE 가 먼저 처리하고, 이건 그 외 단순 타입만 잡는다.
_RE_TYPE_ALIAS = re.compile(
    r"^type\s+(?P<name>[A-Za-z_][\w]*)\s*(?:=\s*)?(?P<base>[A-Za-z_][\w.\[\]*]+)\s*$",
    re.MULTILINE,
)

# Go의 doc comment은 심볼 선언 바로 위에 붙은 `// ...` 연속 줄이다.
# 아래 정규식은 "문자열 끝($) 직전에 오는 // 주석 줄 묶음"을 잡는다.
# (심볼 시작 전까지의 소스만 잘라 넣고 search 하기 때문에 "바로 위 주석"이 된다.)
_RE_DOC_COMMENT_ABOVE = re.compile(r"((?:^[ \t]*//[^\n]*\n)+)\s*$", re.MULTILINE)
# 한 줄 주석 `// ...`
_RE_LINE_COMMENT = re.compile(r"//[^\n]*")
# 블록 주석 `/* ... */` (여러 줄 가능)
_RE_BLOCK_COMMENT = re.compile(r"/\*[\s\S]*?\*/")
# 큰따옴표 문자열 리터럴 (백슬래시 이스케이프 처리 포함)
_RE_STRING_DQ = re.compile(r'"(?:\\.|[^"\\])*"')
# 백틱 raw 문자열 리터럴 (개행 포함 가능)
_RE_STRING_BT = re.compile(r"`[^`]*`")


def _scrub(source: str) -> str:
    """주석/문자열을 같은 길이의 공백으로 치환한다 (줄 구조는 보존).

    왜 필요한가:
      주석이나 문자열 안에 `func ...` 같은 텍스트가 들어 있으면 심볼 정규식이
      이를 진짜 선언으로 오인할 수 있다. 그래서 파싱 전에 주석/문자열 내용을
      싹 지워(=공백으로 덮어) 오탐을 막는다.

    핵심 트릭:
      내용을 지우되 개행(\n)은 그대로 남기고 나머지 문자만 공백으로 바꾼다.
      이렇게 하면 전체 문자열의 길이와 줄 번호가 원본과 100% 동일하게 유지되어,
      매칭된 오프셋을 원본 소스의 줄 번호로 그대로 환산할 수 있다.

    처리 순서(중요): 블록 주석 → 한 줄 주석 → 큰따옴표 문자열 → 백틱 문자열.
    반환: 주석/문자열이 공백으로 치환된, 원본과 같은 길이의 소스 문자열.
    """
    def repl(m: re.Match[str]) -> str:
        # 매칭된 구간에서 개행만 남기고 나머지는 전부 공백으로 바꾼다.
        return "".join(c if c == "\n" else " " for c in m.group(0))
    out = _RE_BLOCK_COMMENT.sub(repl, source)
    out = _RE_LINE_COMMENT.sub(repl, out)
    out = _RE_STRING_DQ.sub(repl, out)
    out = _RE_STRING_BT.sub(repl, out)
    return out


def _line_ends(text: str) -> list[int]:
    """텍스트 안 모든 개행(\n) 문자의 오프셋 위치 리스트를 만든다.

    이 리스트는 _line_of()가 "문자 오프셋 → 줄 번호"를 빠르게 계산하는 데 쓰는
    사전 인덱스다. 오프셋은 오름차순으로 정렬되어 있어 이진 탐색이 가능하다.
    """
    return [i for i, c in enumerate(text) if c == "\n"]


def _line_of(offset: int, line_ends: list[int]) -> int:
    """문자 오프셋을 1-based 줄 번호로 변환한다.

    line_ends(개행 위치 목록)에서 offset보다 작거나 같은 개행이 몇 개인지 세면
    그 앞에 지나온 줄 수가 나오고, 여기에 +1 을 하면 현재 줄 번호가 된다.
    bisect_right 로 이진 탐색해 O(log n)으로 처리한다.
    """
    import bisect
    return bisect.bisect_right(line_ends, offset) + 1


def _match_braces(text: str, start: int) -> int:
    """start 위치의 여는 중괄호 `{`에 대응하는 닫는 `}`의 오프셋을 찾는다.

    동작: start부터 문자를 훑으며 `{`를 만나면 깊이를 +1, `}`를 만나면 -1 한다.
    깊이가 다시 0이 되는 지점이 짝이 맞는 닫는 중괄호이며 그 오프셋을 반환한다.
    (중첩 블록도 깊이 계산 덕분에 올바르게 건너뛴다.)

    주의: 이 함수는 이미 _scrub()으로 주석/문자열이 지워진 텍스트에 쓰는 것을
    전제로 한다. 그래야 문자열 안의 `{`, `}`에 속지 않는다.
    짝을 못 찾으면(비정상 소스) 마지막 문자 오프셋(n-1)을 안전값으로 반환한다.
    """
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
                # 깊이가 0으로 복귀 → 짝이 맞는 닫는 중괄호를 찾음
                return i
        i += 1
    # 짝을 끝내 못 찾은 경우의 방어적 반환값
    return n - 1


def _doc_comment_above(source: str, offset: int) -> str:
    """심볼 선언 바로 위에 붙은 Go 문서 주석(`// ...` 묶음)을 추출한다.

    매개변수:
      source: 원본 소스 (주석이 지워지지 않은 원문 — 주석 내용을 읽어야 하므로)
      offset: 심볼 선언이 시작되는 문자 오프셋

    흐름:
      1) 선언 시작 지점까지의 앞부분(prefix)만 잘라낸다.
      2) 그 끝($)에 붙어 있는 연속된 `//` 주석 줄 묶음을 정규식으로 찾는다.
      3) 각 줄에서 앞쪽 `//`와 공백을 벗겨내고, 빈 줄은 버린 뒤 개행으로 합친다.
    문서 주석이 없으면 빈 문자열을 반환한다.
    """
    prefix = source[:offset]
    m = _RE_DOC_COMMENT_ABOVE.search(prefix)
    if not m:
        return ""
    block = m.group(1)
    # 각 줄에서 선행 `//`(+공백 한 칸)를 제거하고 오른쪽 공백을 정리한다.
    lines = [re.sub(r"^\s*//\s?", "", ln).rstrip() for ln in block.splitlines()]
    # 내용이 있는 줄만 남겨 개행으로 이어 붙인다.
    return "\n".join(ln for ln in lines if ln).strip()


def _receiver_type(receiver: str) -> str:
    """메서드 리시버 문자열에서 순수 타입 이름만 뽑는다.

    예: '(s *Server)'의 내부 's *Server' → 'Server', 's Server' → 'Server'.

    처리:
      1) 공백으로 나눠 마지막 토큰을 타입으로 본다 (앞 토큰은 리시버 변수명).
      2) 포인터 표시 '*'를 앞에서 제거한다.
      3) 제네릭 타입 파라미터를 떼어낸다: 'Server[T]' → 'Server'
         (여는 대괄호 '[' 또는 공백을 만나는 지점에서 잘라 앞부분만 취함).
    리시버가 비어 있으면 빈 문자열을 반환한다.
    """
    parts = receiver.strip().split()
    if not parts:
        return ""
    t = parts[-1].lstrip("*")
    # 제네릭 타입 파라미터 제거 `Server[T]` → `Server`
    return re.split(r"[\[\s]", t, maxsplit=1)[0]


class GoParser(BaseParser):
    """Go 소스에서 심볼을 추출하는 파서.

    BaseParser를 상속하며, 확장자 `.go` 파일을 담당한다. ParserRegistry가
    확장자로 이 파서를 찾아 parse()를 호출한다. 실제 추출 로직은 parse()에
    모두 들어 있고, 이 클래스 자체는 상태를 갖지 않는다(재사용 안전).
    """

    language = "go"
    extensions = (".go",)

    def parse(self, source: str, file_path: str) -> list[ParsedSymbol]:
        """Go 소스 문자열 하나에서 모든 심볼을 뽑아 리스트로 반환한다.

        매개변수:
          source: Go 파일 전체 내용(문자열)
          file_path: 원본 파일 경로 (현재 로직에선 직접 쓰지 않지만 인터페이스 통일용)

        전체 흐름:
          1) _scrub()으로 주석/문자열을 지운 버전(scrubbed)을 만든다. 심볼 정규식은
             이 버전에 돌려 오탐을 막는다. (문서 주석 추출만은 원본 source를 쓴다.)
          2) 개행 위치 인덱스(lines)를 만들어 오프셋→줄번호 변환을 준비한다.
          3) 함수/메서드 → 구조체/인터페이스 → 타입 별칭 순으로 각각 정규식을 돌려
             ParsedSymbol을 만들어 out 리스트에 쌓는다.

        반환: 발견된 ParsedSymbol들의 리스트 (없으면 빈 리스트).
        BaseParser 계약상 파싱 중 예외를 밖으로 던지지 않는 것을 지향한다
        (Indexer가 수백 개 파일을 순회하므로 한 파일 실패가 전체를 막으면 안 됨).
        """
        scrubbed = _scrub(source)
        lines = _line_ends(source)
        out: list[ParsedSymbol] = []

        # --- (1) 함수 및 메서드 ---
        for m in _RE_FUNC.finditer(scrubbed):
            name = m.group("name")
            # 리시버가 있으면 메서드, 없으면 빈 문자열
            receiver = m.group("receiver") or ""
            args = (m.group("args") or "").strip()
            ret = (m.group("ret") or "").strip()
            # 사람이 읽기 좋은 시그니처 문자열을 조립한다: "(args) ret"
            sig = f"({args})"
            if ret:
                sig += f" {ret}"

            # 본문 블록의 시작 `{`와 그에 대응하는 `}` 위치를 찾아 끝 줄을 계산한다.
            brace_open = scrubbed.find("{", m.end() - 1)
            brace_close = _match_braces(scrubbed, brace_open) if brace_open != -1 else m.end()
            line_start = _line_of(m.start(), lines)
            line_end = _line_of(brace_close, lines)
            # 문서 주석은 원본 source(주석이 살아 있는)에서 뽑아야 한다.
            doc = _doc_comment_above(source, m.start())

            if receiver:
                # 메서드: 리시버 타입을 접두어로 붙여 "Type.Method"로 정규화한다.
                recv_type = _receiver_type(receiver)
                qual = f"{recv_type}.{name}" if recv_type else name
                kind = "method"
            else:
                # 일반 함수: 이름 그대로가 정규화 이름.
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
                # Go 규칙: 이름 첫 글자가 대문자면 패키지 외부로 공개(exported)됨.
                extra_tags=("exported",) if name[:1].isupper() else (),
            ))

        # --- (2) 구조체 / 인터페이스 ---
        for m in _RE_TYPE.finditer(scrubbed):
            name = m.group("name")
            kind = "struct" if m.group("kind") == "struct" else "interface"
            # 함수와 마찬가지로 중괄호 블록의 끝 줄을 찾는다.
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

        # --- (3) 타입 별칭 / 단순 타입 정의 ---
        for m in _RE_TYPE_ALIAS.finditer(scrubbed):
            name = m.group("name")
            base = m.group("base")
            # 별칭은 보통 한 줄짜리라 시작=끝 줄로 둔다.
            line_start = _line_of(m.start(), lines)
            out.append(ParsedSymbol(
                kind="type",
                name=name,
                qualified_name=name,
                # 밑줄 타입이 있으면 "= Base" 형태로 시그니처에 표기.
                signature=f"= {base}" if base else "",
                docstring=_doc_comment_above(source, m.start()),
                line_start=line_start,
                line_end=line_start,
                language=self.language,
                extra_tags=("exported",) if name[:1].isupper() else (),
            ))

        return out
