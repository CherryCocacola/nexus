"""
심볼 파서 공통 인터페이스 (Phase 10.0 확장, 2026-04-21).

이 파일은 "소스 코드에서 심볼(함수/클래스/메서드 등)을 뽑아내는" 여러 언어별
파서들이 공통으로 지켜야 할 뼈대를 정의한다. 언어마다 문법이 다르지만, 위쪽
계층(인덱서)은 언어와 상관없이 동일한 방식으로 결과를 다루고 싶어 한다.
그래서 이 모듈이 "언어-독립 표준 계약"을 제공한다.

핵심 구성 요소:
- `ParsedSymbol` : 언어에 상관없이 심볼 하나를 표현하는 불변 데이터 객체.
- `BaseParser`   : 각 언어 파서가 상속해야 하는 추상 베이스 클래스(ABC).
                   실제 파싱 로직은 언어별 서브클래스가 `parse()`에 구현한다.
- `ParserRegistry`: 파일 확장자(.py, .ts 등)를 알맞은 파서로 연결해 주는 등록소.

동작 흐름(호출 관계):
각 언어 파서가 `BaseParser`를 상속해 `parse(source, file_path)`를 구현한다.
결과는 `ParsedSymbol` 리스트로 반환되며, SymbolProjectIndexer가 이를
`SymbolEntry`로 변환해 DB에 적재한다. 즉 이 파일은 인덱서와 언어별 파서
사이의 "약속(인터페이스)" 역할만 하고, 무거운 파싱은 서브클래스가 맡는다.

작성자: 이현수 / 작성일: 2026-07-05
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ParsedSymbol:
    """언어에 독립적인 "심볼 한 개"를 표현하는 불변 데이터 객체.

    함수, 클래스, 메서드 같은 코드 구성 요소 하나를 담는 그릇이다. 파이썬,
    자바스크립트 등 언어마다 문법은 다르지만, 각 언어 파서는 결과를 모두 이
    형태로 통일해서 돌려준다. 그러면 위쪽 인덱서는 언어를 신경 쓰지 않고
    동일하게 처리할 수 있다.

    `frozen=True`라서 생성 후에는 필드를 바꿀 수 없다(불변). 이렇게 하면
    실수로 값이 뒤바뀌는 사고를 막고, 안전하게 여기저기 전달할 수 있다.

    Indexer가 이 객체를 `SymbolEntry`로 매핑한다. language 필드는 태그용.
    """

    # 심볼의 종류. function/class/method/async_function/async_method/
    # interface/struct/type 등이 들어간다. 언어별로 표현 가능한 값이 다르다.
    kind: str
    name: str                    # 단일 이름 (예: "Foo")
    qualified_name: str          # 네임스페이스 포함 (예: "Outer.Foo")
    signature: str = ""          # 시그니처 문자열 (언어별 자연 표현)
    docstring: str = ""          # docstring / doc comment
    # 심볼이 시작/끝나는 소스 코드의 줄 번호(1부터). 0이면 정보 없음.
    line_start: int = 0
    line_end: int = 0
    language: str = ""           # "python", "javascript" 등 (Indexer가 태그로 사용)
    # 추가 표식들. "export", "public" 같은 가시성/속성 정보를 담는다.
    # 불변 객체이므로 리스트가 아닌 tuple을 기본값으로 쓴다.
    extra_tags: tuple[str, ...] = field(default_factory=tuple)  # "export", "public" 등


class BaseParser(ABC):
    """언어별 심볼 파서가 반드시 상속해야 하는 추상 베이스 클래스.

    이 클래스 자체는 직접 사용하지 않는다. PythonParser, JavaScriptParser처럼
    언어마다 구체적인 서브클래스를 만들고, 그 안에서 `parse()`를 구현한다.
    공통 규격(어떤 언어인지, 어떤 확장자를 맡는지, 어떻게 파싱하는지)을 여기서
    강제함으로써, 인덱서가 모든 파서를 같은 방식으로 다룰 수 있게 한다.
    """

    # 언어 식별자 ("python", "javascript", ...). 서브클래스가 값을 채운다.
    language: str = ""
    # 이 파서가 담당할 파일 확장자 (소문자, 점 포함)
    extensions: tuple[str, ...] = ()

    @abstractmethod
    def parse(self, source: str, file_path: str) -> list[ParsedSymbol]:
        """단일 파일 소스에서 심볼을 추출한다.

        매개변수:
            source: 파싱할 파일의 전체 텍스트 내용.
            file_path: 원본 파일 경로(로그/디버깅 및 참고용).
        반환값:
            추출된 `ParsedSymbol`들의 리스트. 심볼이 없으면 빈 리스트.

        중요한 계약(반드시 지킬 것):
        파싱 에러가 있어도 예외를 올리지 말고 빈 리스트 혹은 부분 결과를 반환한다.
        Indexer가 수백 파일을 순회하기 때문에 단일 파일의 실패가 전체를 망쳐선
        안 된다.
        """


class ParserRegistry:
    """파일 확장자를 알맞은 파서로 연결해 주는 등록소(레지스트리).

    인덱서는 파일 경로만 알고 있고 어떤 파서를 써야 할지는 모른다. 이 레지스트리에
    파서들을 미리 등록해 두면, 확장자(.py, .ts 등)를 보고 담당 파서를 찾아 준다.

    확장자 → Parser 매핑.

    같은 확장자가 여러 파서에 등록되면 마지막에 등록된 파서가 우선한다
    (예: `.ts`는 TypeScriptParser가 JavaScriptParser를 덮어쓴다).
    """

    def __init__(self) -> None:
        # 확장자(소문자) → 파서 인스턴스로 이어지는 내부 매핑 딕셔너리.
        self._by_ext: dict[str, BaseParser] = {}

    def register(self, parser: BaseParser) -> None:
        # 파서가 선언한 모든 확장자를 순회하며 매핑에 등록한다.
        # 키는 항상 소문자로 저장해, 나중에 조회할 때 대소문자 차이로
        # 놓치는 일이 없게 한다. 이미 있던 확장자면 새 파서로 덮어쓴다.
        for ext in parser.extensions:
            self._by_ext[ext.lower()] = parser

    def for_path(self, path: str | Path) -> BaseParser | None:
        """경로의 확장자로 파서를 찾는다. 지원 안 하면 None.

        문자열 경로든 Path 객체든 모두 받아 처리한다. 확장자를 소문자로
        정규화한 뒤 매핑에서 찾는다.
        """
        # Path.suffix는 마지막 확장자(예: ".py")를 준다. 소문자로 맞춰서 조회.
        suffix = Path(path).suffix.lower()
        return self._by_ext.get(suffix)

    def supported_extensions(self) -> list[str]:
        # 현재 등록된 모든 확장자를 정렬해 반환한다(어떤 파일을 다룰 수 있는지 확인용).
        return sorted(self._by_ext.keys())

    def supported_languages(self) -> list[str]:
        # 등록된 파서들이 담당하는 언어 이름을 중복 없이 정렬해 반환한다.
        # 빈 문자열 언어는 제외한다(언어를 지정하지 않은 파서 걸러내기).
        return sorted({p.language for p in self._by_ext.values() if p.language})
