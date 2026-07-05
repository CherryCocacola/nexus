"""
언어별 심볼 파서 패키지 (Phase 10.0 확장, 2026-04-21).

이 패키지는 소스 코드 파일을 읽어 그 안에 들어 있는 "심볼"
(함수, 클래스, 메서드 등 코드 구조의 최소 단위)을 뽑아내는
파서 모음이다. RAG(검색 증강 생성)가 코드베이스를 이해하고
검색할 수 있도록, 각 언어의 소스를 구조화된 심볼 목록으로
변환하는 것이 이 패키지의 역할이다.

지원 언어:
  - Python: `ast` 표준 라이브러리 (정확한 AST)
      * 파이썬은 표준 라이브러리만으로 구문 트리를 얻을 수 있어
        정규식보다 훨씬 정확하게 심볼을 추출한다.
  - JavaScript/TypeScript: 정규식 (에어갭에서 tree-sitter/esprima 미사용)
      * 폐쇄망(에어갭) 환경이라 외부 파서 패키지를 설치할 수 없어
        정규식 기반으로 함수/클래스 선언을 근사 추출한다.
  - Go: 정규식
      * 위와 같은 이유로 Go 역시 정규식 기반 파서를 사용한다.

이 패키지가 밖으로 노출하는 핵심 요소:
  - BaseParser: 모든 파서가 상속하는 추상 기반 클래스
  - ParsedSymbol: 추출된 심볼 하나를 표현하는 데이터 모델
  - ParserRegistry: 확장자별로 파서를 골라주는 등록소
  - PythonParser / JavaScriptParser / TypeScriptParser / GoParser:
    언어별 실제 파서 구현체
  - build_default_registry(): 기본 파서들이 미리 등록된 레지스트리 팩토리

확장 방법:
  `BaseParser`를 상속해 새 언어 파서를 만든 뒤,
  아래 `build_default_registry()`에 `reg.register(...)`로 추가한다.

작성자: 이현수 / 작성일: 2026-07-05
"""
# 하위 모듈에서 공개 API를 끌어와 패키지 최상단에서 바로 쓸 수 있게 재노출한다.
# (사용하는 쪽은 core.rag.parsers 만 import 하면 되도록 편의를 제공)
from core.rag.parsers.base import BaseParser, ParsedSymbol, ParserRegistry
from core.rag.parsers.go_parser import GoParser
from core.rag.parsers.javascript_parser import JavaScriptParser, TypeScriptParser
from core.rag.parsers.python_parser import PythonParser


def build_default_registry() -> ParserRegistry:
    """기본 파서가 등록된 ParserRegistry를 생성한다.

    이 프로젝트가 기본으로 지원하는 4개 언어 파서(Python, JavaScript,
    TypeScript, Go)를 모두 등록한 레지스트리를 만들어 돌려준다.
    RAG 인덱싱 파이프라인은 이 함수를 호출해 파서 세트를 확보한 뒤,
    파일 확장자에 맞는 파서를 골라 심볼을 추출한다.

    왜 팩토리 함수로 두는가:
      전역 싱글턴 대신 호출 시점마다 새 레지스트리를 만들어,
      테스트나 특수 상황에서 파서 구성을 자유롭게 바꿀 수 있도록 하기 위함.

    반환값:
      ParserRegistry — 4개 기본 파서가 등록된 새 레지스트리 인스턴스.
    """
    # 빈 레지스트리를 먼저 만들고, 지원 언어 파서를 하나씩 등록한다.
    reg = ParserRegistry()
    reg.register(PythonParser())      # .py — ast 기반 정확 추출
    reg.register(JavaScriptParser())  # .js — 정규식 기반 추출
    reg.register(TypeScriptParser())  # .ts — 정규식 기반 추출
    reg.register(GoParser())          # .go — 정규식 기반 추출
    return reg


# 이 패키지를 `from core.rag.parsers import *` 형태로 가져올 때
# 노출할 공개 이름 목록. 외부에 제공하는 안정적 API 표면을 명시한다.
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
