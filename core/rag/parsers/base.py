"""
심볼 파서 공통 인터페이스 (Phase 10.0 확장, 2026-04-21).

각 언어는 `BaseParser`를 상속해 `parse(source, file_path)`를 구현한다.
결과는 `ParsedSymbol` 리스트로 반환되며, SymbolProjectIndexer가 이를
`SymbolEntry`로 변환해 DB에 적재한다.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ParsedSymbol:
    """언어-독립 심볼 표현.

    Indexer가 이 객체를 `SymbolEntry`로 매핑한다. language 필드는 태그용.
    """

    # function/class/method/async_function/async_method/interface/struct/type 등
    kind: str
    name: str                    # 단일 이름 (예: "Foo")
    qualified_name: str          # 네임스페이스 포함 (예: "Outer.Foo")
    signature: str = ""          # 시그니처 문자열 (언어별 자연 표현)
    docstring: str = ""          # docstring / doc comment
    line_start: int = 0
    line_end: int = 0
    language: str = ""           # "python", "javascript" 등 (Indexer가 태그로 사용)
    extra_tags: tuple[str, ...] = field(default_factory=tuple)  # "export", "public" 등


class BaseParser(ABC):
    """언어별 심볼 파서."""

    # 언어 식별자 ("python", "javascript", ...)
    language: str = ""
    # 이 파서가 담당할 파일 확장자 (소문자, 점 포함)
    extensions: tuple[str, ...] = ()

    @abstractmethod
    def parse(self, source: str, file_path: str) -> list[ParsedSymbol]:
        """단일 파일 소스에서 심볼을 추출한다.

        파싱 에러가 있어도 예외를 올리지 말고 빈 리스트 혹은 부분 결과를 반환한다.
        Indexer가 수백 파일을 순회하기 때문에 단일 파일의 실패가 전체를 망쳐선
        안 된다.
        """


class ParserRegistry:
    """확장자 → Parser 매핑.

    같은 확장자가 여러 파서에 등록되면 마지막에 등록된 파서가 우선한다
    (예: `.ts`는 TypeScriptParser가 JavaScriptParser를 덮어쓴다).
    """

    def __init__(self) -> None:
        self._by_ext: dict[str, BaseParser] = {}

    def register(self, parser: BaseParser) -> None:
        for ext in parser.extensions:
            self._by_ext[ext.lower()] = parser

    def for_path(self, path: str | Path) -> BaseParser | None:
        """경로의 확장자로 파서를 찾는다. 지원 안 하면 None."""
        suffix = Path(path).suffix.lower()
        return self._by_ext.get(suffix)

    def supported_extensions(self) -> list[str]:
        return sorted(self._by_ext.keys())

    def supported_languages(self) -> list[str]:
        return sorted({p.language for p in self._by_ext.values() if p.language})
