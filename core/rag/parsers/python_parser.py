"""
Python 심볼 파서 — 파이썬 표준 라이브러리 `ast`(추상 구문 트리) 기반 파서.

[이 파일이 하는 일]
RAG(검색 증강 생성) 시스템은 프로젝트 소스코드를 "심볼" 단위로 색인해 둔다.
여기서 심볼이란 함수/클래스/메서드 같은, 코드에서 이름을 가진 정의를 말한다.
이 파일은 파이썬 소스 문자열 하나를 받아, 그 안에 정의된 top-level 함수/클래스와
클래스 내부 메서드를 모두 찾아내 `ParsedSymbol` 리스트로 돌려준다.
정규식이 아니라 파이썬 공식 `ast` 파서를 쓰기 때문에, 문법적으로 정확하게
함수/클래스 경계·시그니처·docstring·줄 범위를 추출할 수 있다.

[역사적 배경]
원래 `symbol_indexer.extract_symbols_from_source()` 안에 흩어져 있던 파싱 로직을,
언어별 파서를 갈아끼울 수 있는 `BaseParser` 인터페이스 형태로 재포장한 것이다
(Phase 10.0 확장, 2026-04-21). 추출 동작 자체는 이전과 동일하다 —
top-level function/class + 내부 메서드, async 변형(async def)까지 포함.

[주요 구성]
- PythonParser: BaseParser 구현체. parse()가 진입점.
- _visit(): AST 노드 리스트를 재귀 순회하며 함수/클래스를 골라내는 내부 헬퍼.
- _mk_func(): 함수/메서드 노드 하나를 ParsedSymbol로 변환하는 내부 헬퍼.

[의존 관계]
- core.rag.parsers.base 의 BaseParser(상속용)와 ParsedSymbol(결과 자료형)에 의존.
- 상위에서는 SymbolProjectIndexer가 이 파서를 호출해 결과를 DB에 적재한다.

작성자: 이현수 / 작성일: 2026-07-05
"""
from __future__ import annotations

import ast
import logging

from core.rag.parsers.base import BaseParser, ParsedSymbol

# 이 모듈 전용 로거. 계층형 이름을 써서 로그 필터링/레벨 제어를 쉽게 한다.
logger = logging.getLogger("nexus.rag.parsers.python")


class PythonParser(BaseParser):
    """파이썬 소스에서 심볼을 뽑아내는 파서 — 정확한 AST 기반 구현.

    BaseParser를 상속하며, 클래스 속성 두 개(language, extensions)로
    "나는 파이썬을 담당하고 .py 파일을 처리한다"는 것을 레지스트리에 알린다.
    실제 파싱은 parse() 하나가 전담하고, 세부 처리는 _visit / _mk_func로 나눈다.
    """

    # 언어 식별자 — ParsedSymbol.language 태그로도 그대로 쓰인다.
    language = "python"
    # 이 파서가 담당하는 파일 확장자 (소문자, 점 포함). ParserRegistry가 참조.
    extensions = (".py",)

    def parse(self, source: str, file_path: str) -> list[ParsedSymbol]:
        """소스 문자열 하나를 파싱해 심볼 리스트를 돌려준다 (이 클래스의 진입점).

        매개변수:
            source: 파이썬 소스코드 전체 문자열.
            file_path: 원본 파일 경로. 에러 메시지/AST 위치 표기용으로만 쓰인다.
        반환:
            추출된 ParsedSymbol 리스트. 문법 오류가 나면 빈 리스트.

        핵심 흐름:
            1) ast.parse로 소스를 AST(구문 트리)로 변환한다.
            2) 문법 오류(SyntaxError)가 나면 예외를 위로 올리지 않고 빈 리스트를
               반환한다. Indexer가 수백 개 파일을 순회하므로, 깨진 파일 하나가
               전체 색인 작업을 중단시켜선 안 되기 때문이다(부분 실패 허용).
            3) 최상위 노드들(tree.body)부터 _visit로 재귀 순회를 시작한다.
        """
        try:
            # 소스를 AST로 파싱. filename은 오류 위치 표기에만 사용된다.
            tree = ast.parse(source, filename=file_path)
        except SyntaxError as e:
            # 문법이 깨진 파일 — 조용히 건너뛴다. debug 레벨로만 기록.
            logger.debug("Python SyntaxError (%s): %s", file_path, e)
            return []

        # 원본을 줄 단위로 나눠 둔다(현재 시그니처만으로 충분하지만, 향후 줄 기반
        # 추가 정보 추출을 위해 함께 넘겨 두는 값이다).
        source_lines = source.splitlines()
        # 최종 결과가 담길 리스트. _visit가 여기에 계속 append 한다.
        out: list[ParsedSymbol] = []
        # 최상위 노드부터 순회 시작. 부모가 없으므로 qual=""·kind=None.
        self._visit(tree.body, "", None, source_lines, out)
        return out

    # ─────────────────────────────────────────────
    # 내부 헬퍼
    # ─────────────────────────────────────────────
    def _visit(
        self,
        body: list[ast.stmt],
        parent_qual: str,
        parent_kind: str | None,
        source_lines: list[str],
        out: list[ParsedSymbol],
    ) -> None:
        """AST 노드 리스트를 훑으며 함수/클래스를 골라 out에 채우는 재귀 순회기.

        같은 로직을 최상위와 "클래스 내부"에서 반복 사용하기 위해 재귀로 만들었다.
        클래스를 만나면 그 안으로 다시 _visit를 호출해 메서드를 뽑는다.

        매개변수:
            body: 순회할 AST 문(statement) 리스트. 최상위면 tree.body,
                  클래스 내부면 node.body.
            parent_qual: 부모의 정규화 이름(qualified name). 최상위면 "".
                         자식 이름 앞에 "부모.자식" 형태로 붙여 계층을 표현한다.
            parent_kind: 부모의 종류. 클래스 내부를 순회할 때만 "class"가 들어오며,
                         이 값으로 함수가 "메서드"인지 "일반 함수"인지 판정한다.
            source_lines: parse()에서 받은 줄 단위 소스(현재는 _mk_func로 전달만).
            out: 결과 누적 리스트. 이 함수는 반환값 없이 out을 직접 채운다.

        주의: 함수 정의는 재귀로 더 파고들지 않는다. 즉 함수 안에 중첩 정의된
              내부 함수(closure)는 심볼로 추출하지 않는다 — 색인 대상은 top-level
              함수/클래스와 클래스의 메서드까지로 한정한다.
        """
        for node in body:
            # (1) 함수 정의 — 일반(def)·비동기(async def) 모두 처리.
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # 함수 노드를 ParsedSymbol로 변환해 바로 결과에 추가.
                out.append(self._mk_func(node, parent_qual, parent_kind, source_lines))
            # (2) 클래스 정의 — 클래스 자체를 심볼로 만들고, 내부로 다시 파고든다.
            elif isinstance(node, ast.ClassDef):
                # 정규화 이름: 부모가 있으면 "부모.클래스", 없으면 클래스 이름만.
                qual = f"{parent_qual}.{node.name}" if parent_qual else node.name
                # 클래스 docstring 추출(없으면 빈 문자열).
                doc = ast.get_docstring(node) or ""
                # 정의 시작 줄. lineno가 없거나 0이면 0으로 안전 처리.
                line_start = getattr(node, "lineno", 0) or 0
                # 정의 끝 줄. end_lineno가 없으면 시작 줄로 대체(1줄짜리 취급).
                line_end = getattr(node, "end_lineno", line_start) or line_start
                # 상속 베이스 목록을 "Base1, Base2" 형태 문자열로 복원한다.
                # ast.unparse가 드물게 실패할 수 있어 방어적으로 감싸고, 실패 시
                # 빈 문자열로 떨어뜨려 파싱 전체가 중단되지 않게 한다.
                try:
                    bases = ", ".join(ast.unparse(b) for b in node.bases)
                except Exception:
                    bases = ""
                # 클래스 "시그니처"는 상속 목록을 괄호로 감싼 형태. 베이스가 없으면 "".
                sig = f"({bases})" if bases else ""
                # 추출한 정보로 클래스 심볼을 만들어 결과에 추가.
                out.append(
                    ParsedSymbol(
                        kind="class",
                        name=node.name,
                        qualified_name=qual,
                        signature=sig,
                        docstring=doc,
                        line_start=line_start,
                        line_end=line_end,
                        language=self.language,
                    )
                )
                # 클래스 내부로 재귀. parent_kind="class"를 넘겨, 안쪽 함수들이
                # "메서드"로 판정되도록 한다.
                self._visit(node.body, qual, "class", source_lines, out)

    def _mk_func(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        parent_qual: str,
        parent_kind: str | None,
        source_lines: list[str],
    ) -> ParsedSymbol:
        """함수/메서드 AST 노드 하나를 ParsedSymbol로 변환한다.

        _visit가 함수 노드를 만날 때마다 호출하는 헬퍼. 종류(kind) 판정,
        시그니처 복원, docstring·줄 범위 추출을 담당한다.

        매개변수:
            node: 변환할 함수 노드(FunctionDef 또는 AsyncFunctionDef).
            parent_qual: 부모 정규화 이름. 메서드면 소속 클래스 이름이 들어온다.
            parent_kind: 부모 종류. "class"면 이 함수는 메서드로 분류된다.
            source_lines: 줄 단위 소스(현재 로직에서는 사용하지 않지만 시그니처
                          일관성을 위해 함께 받는다).
        반환:
            완성된 ParsedSymbol 하나.
        """
        # 비동기 함수(async def)인지 먼저 구분해 둔다.
        is_async = isinstance(node, ast.AsyncFunctionDef)
        # 종류 판정: 부모가 클래스면 메서드, 아니면 일반 함수.
        # 여기에 async 여부를 조합해 4가지(method/async_method/function/
        # async_function) 중 하나로 확정한다.
        if parent_kind == "class":
            kind = "async_method" if is_async else "method"
        else:
            kind = "async_function" if is_async else "function"

        # 시그니처 — 인자 목록(args) + 반환 타입 어노테이션(return)을 복원한다.
        # 인자 부분을 소스 문자열로 되살린다. unparse 실패 시 "..."로 대체해
        # 심볼 자체는 살린다(시그니처 없어도 이름/위치 정보는 유효하므로).
        try:
            args = ast.unparse(node.args)
        except Exception:
            args = "..."
        ret = ""
        # 반환 타입 어노테이션(`-> X`)이 있을 때만 복원한다.
        if node.returns is not None:
            try:
                ret = " -> " + ast.unparse(node.returns)
            except Exception as e:
                # 반환 어노테이션 복원만 실패한 경우 — 로그만 남기고 무시.
                logger.debug("return annotation unparse 실패 (무시): %s", e)
        # 최종 시그니처 문자열: "(인자들) -> 반환타입" 형태.
        sig = f"({args}){ret}"

        # 정규화 이름: 부모가 있으면 "부모.함수", 없으면 함수 이름만.
        qual = f"{parent_qual}.{node.name}" if parent_qual else node.name
        # 함수 docstring(없으면 빈 문자열).
        doc = ast.get_docstring(node) or ""
        # 정의 시작/끝 줄 — 클래스 처리와 동일한 안전 규칙 적용.
        line_start = getattr(node, "lineno", 0) or 0
        line_end = getattr(node, "end_lineno", line_start) or line_start

        # 수집한 정보를 담아 심볼 하나를 만들어 돌려준다.
        return ParsedSymbol(
            kind=kind,
            name=node.name,
            qualified_name=qual,
            signature=sig,
            docstring=doc,
            line_start=line_start,
            line_end=line_end,
            language=self.language,
        )
