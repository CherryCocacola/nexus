"""
Python 심볼 파서 — `ast` 표준 라이브러리 기반 (Phase 10.0 확장, 2026-04-21).

기존 `symbol_indexer.extract_symbols_from_source()`의 파싱 로직을 BaseParser
인터페이스로 재포장했다. 동작은 동일 (top-level function/class + 내부 메서드,
async 변형 포함).
"""
from __future__ import annotations

import ast
import logging

from core.rag.parsers.base import BaseParser, ParsedSymbol

logger = logging.getLogger("nexus.rag.parsers.python")


class PythonParser(BaseParser):
    """Python 파서 — 정확한 AST 기반."""

    language = "python"
    extensions = (".py",)

    def parse(self, source: str, file_path: str) -> list[ParsedSymbol]:
        try:
            tree = ast.parse(source, filename=file_path)
        except SyntaxError as e:
            logger.debug("Python SyntaxError (%s): %s", file_path, e)
            return []

        source_lines = source.splitlines()
        out: list[ParsedSymbol] = []
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
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                out.append(self._mk_func(node, parent_qual, parent_kind, source_lines))
            elif isinstance(node, ast.ClassDef):
                qual = f"{parent_qual}.{node.name}" if parent_qual else node.name
                doc = ast.get_docstring(node) or ""
                line_start = getattr(node, "lineno", 0) or 0
                line_end = getattr(node, "end_lineno", line_start) or line_start
                try:
                    bases = ", ".join(ast.unparse(b) for b in node.bases)
                except Exception:
                    bases = ""
                sig = f"({bases})" if bases else ""
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
                # 클래스 내부 — 부모 kind="class"로 메서드 판정
                self._visit(node.body, qual, "class", source_lines, out)

    def _mk_func(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        parent_qual: str,
        parent_kind: str | None,
        source_lines: list[str],
    ) -> ParsedSymbol:
        is_async = isinstance(node, ast.AsyncFunctionDef)
        if parent_kind == "class":
            kind = "async_method" if is_async else "method"
        else:
            kind = "async_function" if is_async else "function"

        # 시그니처 — args + return
        try:
            args = ast.unparse(node.args)
        except Exception:
            args = "..."
        ret = ""
        if node.returns is not None:
            try:
                ret = " -> " + ast.unparse(node.returns)
            except Exception as e:
                logger.debug("return annotation unparse 실패 (무시): %s", e)
        sig = f"({args}){ret}"

        qual = f"{parent_qual}.{node.name}" if parent_qual else node.name
        doc = ast.get_docstring(node) or ""
        line_start = getattr(node, "lineno", 0) or 0
        line_end = getattr(node, "end_lineno", line_start) or line_start

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
