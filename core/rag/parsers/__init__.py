"""
언어별 심볼 파서 패키지 (Phase 10.0 확장, 2026-04-21).

지원 언어:
  - Python: `ast` 표준 라이브러리 (정확한 AST)
  - JavaScript/TypeScript: 정규식 (에어갭에서 tree-sitter/esprima 미사용)
  - Go: 정규식

확장 방법:
  `BaseParser`를 상속하고 `build_default_registry()`에 등록한다.
"""
from core.rag.parsers.base import BaseParser, ParsedSymbol, ParserRegistry
from core.rag.parsers.go_parser import GoParser
from core.rag.parsers.javascript_parser import JavaScriptParser, TypeScriptParser
from core.rag.parsers.python_parser import PythonParser


def build_default_registry() -> ParserRegistry:
    """기본 파서가 등록된 ParserRegistry를 생성한다."""
    reg = ParserRegistry()
    reg.register(PythonParser())
    reg.register(JavaScriptParser())
    reg.register(TypeScriptParser())
    reg.register(GoParser())
    return reg


__all__ = [
    "BaseParser",
    "GoParser",
    "JavaScriptParser",
    "ParsedSymbol",
    "ParserRegistry",
    "PythonParser",
    "TypeScriptParser",
    "build_default_registry",
]
