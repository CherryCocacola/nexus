"""
다언어(多言語) 심볼 추출·인덱싱 모듈 (Phase 10.0 + 확장, 2026-04-21).

[이 파일이 하는 일]
프로젝트의 소스 코드 파일을 훑어서, 그 안에 정의된 "심볼"(함수/클래스/메서드 등)을
하나씩 뽑아낸 뒤, 검색이 가능한 형태로 DB(SymbolStore)에 적재한다. 이렇게 만들어 둔
심볼 인덱스는 나중에 RAG(검색 증강 생성) 단계에서 "이 기능을 하는 함수가 어디 있지?"
같은 코드 검색 질의에 답하기 위해 쓰인다.

[지원 언어와 파서]
- Python: 표준 라이브러리 ast(추상 구문 트리)로 정확히 파싱.
- JavaScript / TypeScript, Go: 외부 라이브러리 없이 정규식으로 근사 파싱.
언어별 구체 파서는 `core/rag/parsers/`에 있고, 본 모듈은 직접 파싱하지 않는다.
대신 ParserRegistry(파서 등록소)에 파일 확장자를 넘겨서 알맞은 파서를 골라 쓴다.

[각 심볼의 summary(임베딩 대상 텍스트) 구성 규칙]
  "{kind} {qualified_name}{signature}
  {docstring}
  source:
  {최대 5줄 발췌}"
이 summary 문자열이 임베딩 벡터로 변환되어 의미 기반 검색의 재료가 된다.

[주요 구성 요소]
- iter_source_files(): 인덱싱 대상 소스 파일을 걸러서 하나씩 내주는 제너레이터.
- extract_symbols_from_source(): 파일 하나의 소스 텍스트에서 심볼 목록을 추출.
- SymbolProjectIndexer: 프로젝트 전체를 순회하며 SymbolStore에 적재하는 오케스트레이터.
- background_index(): bootstrap에서 fire-and-forget으로 부르는 예외 삼킴 래퍼.

[설계 결정]
  - 에어갭 준수 — 외부 파서 라이브러리 없이 표준 ast / 정규식만 사용.
  - 파일 단위 트랜잭션: `delete_by_path()` → `add_many()` 순서로 재인덱싱을 멱등하게.
    (같은 파일을 다시 인덱싱해도 중복 없이 항상 최신 상태로 교체된다.)
  - 임베딩이 실패(임베딩 서버 다운 등)해도 본문과 메타데이터는 그대로 저장 →
    벡터 검색은 못 해도 텍스트(키워드) 검색은 여전히 가능하도록 안전하게 설계.

작성자: 이현수 / 작성일: 2026-07-05
"""

from __future__ import annotations

import ast  # noqa: F401 — 하위 호환(extract_symbols_from_source 이전 import 참조)
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from core.rag.parsers import (
    BaseParser,
    ParsedSymbol,
    ParserRegistry,
    build_default_registry,
)
from core.rag.symbol_store import SymbolEntry, SymbolStore

logger = logging.getLogger("nexus.rag.symbol_indexer")


# 인덱싱에서 제외할 디렉토리 이름 목록 (indexer.py와 같은 규칙을 공유한다).
# 버전 관리·캐시·의존성·빌드 산출물·대용량 데이터 폴더는 코드 심볼이 없거나
# 인덱싱해봤자 노이즈만 늘어나므로 통째로 건너뛴다.
EXCLUDED_DIRS = {
    ".git", "__pycache__", "node_modules", ".venv", "venv",
    ".mypy_cache", ".pytest_cache", ".ruff_cache",
    "dist", "build", "egg-info",
    "models", "checkpoints", "data", "logs",
    ".nexus", "kowiki_ingest",
}

# 단일 파일 최대 크기(바이트). 이 값을 넘는 파일은 인덱싱하지 않는다.
# 자동 생성된 초대형 파일(번들·마이그레이션 등)이 인덱싱을 느리게 하거나
# 파서를 폭주시키는 것을 막기 위한 보호 장치. 현재 200KB.
MAX_FILE_SIZE = 200 * 1024

# 임베딩을 요청할 때 한 번에 묶어 보내는 심볼 개수(배치 크기).
# 너무 크면 임베딩 서버 요청이 무거워지고, 너무 작으면 왕복 횟수가 늘어난다.
EMBED_BATCH_SIZE = 16


# ─────────────────────────────────────────────
# AST 추출
# ─────────────────────────────────────────────
def iter_python_files(root: Path) -> Iterator[Path]:
    """인덱싱 대상 .py 파일만 순회하는 제너레이터 (하위 호환용 얇은 래퍼).

    내부적으로는 확장자를 `.py`로 고정한 채 `iter_source_files()`에 위임한다.
    다언어 지원 이전에 쓰이던 호출부를 깨뜨리지 않기 위해 남겨 둔 함수.

    Args:
        root: 스캔을 시작할 루트 디렉토리 경로.

    Yields:
        조건을 통과한 .py 파일의 Path 객체.
    """
    yield from iter_source_files(root, extensions=(".py",))


def iter_source_files(
    root: Path,
    extensions: tuple[str, ...] | None = None,
) -> Iterator[Path]:
    """인덱싱 대상 소스 파일을 하나씩 걸러 내주는 제너레이터.

    루트 아래를 재귀적으로 걸으면서 (1) 제외 디렉토리를 쳐내고, (2) 허용 확장자만
    통과시키고, (3) 최대 크기를 넘지 않는 파일만 내준다. 실제 순회는 지연 평가라
    호출부에서 필요한 만큼만 소비하면 되어 메모리에 파일 목록을 다 쌓지 않는다.

    Args:
        root: 스캔 시작 디렉토리.
        extensions: 허용 확장자 튜플 (소문자, 점 포함). None이면 ParserRegistry
            기본 확장자(py/js/jsx/mjs/cjs/ts/tsx/go)를 사용한다.

    Yields:
        인덱싱 조건을 모두 통과한 파일의 Path.
    """
    # 확장자 인자가 없으면 기본 파서 레지스트리가 지원하는 확장자를 그대로 쓴다.
    if extensions is None:
        extensions = tuple(build_default_registry().supported_extensions())
    # 확장자 비교는 대소문자를 무시하기 위해 미리 소문자로 정규화해 둔다.
    exts = tuple(e.lower() for e in extensions)

    for current, dirs, files in _walk(root):
        # dirs를 제자리(in-place)에서 수정해야 os.walk가 하위 순회를 건너뛴다.
        # 제외 목록에 있거나 점(.)으로 시작하는 숨김 디렉토리는 아예 내려가지 않는다.
        dirs[:] = [d for d in dirs if d not in EXCLUDED_DIRS and not d.startswith(".")]
        for f in files:
            p = current / f
            # 확장자가 허용 목록에 없으면 건너뛴다.
            if p.suffix.lower() not in exts:
                continue
            try:
                # 최대 크기 이하인 파일만 내준다. stat 접근이 실패하면 조용히 skip.
                if p.stat().st_size <= MAX_FILE_SIZE:
                    yield p
            except OSError:
                continue


def _walk(root: Path):
    """os.walk를 감싼 내부 헬퍼 — 반환 타입을 Path로 명확히 하기 위한 래퍼.

    os.walk는 경로를 문자열로 돌려주는데, 여기서 곧바로 Path로 감싸 주면
    호출부(iter_source_files)에서 경로 조작이 일관되고 타입 힌트도 깔끔해진다.

    Yields:
        (현재 디렉토리 Path, 하위 디렉토리명 리스트, 파일명 리스트) 튜플.
    """
    import os
    for dirpath, dirnames, filenames in os.walk(root):
        yield Path(dirpath), dirnames, filenames


def _source_excerpt(source_lines: list[str], start: int, end: int, n: int = 5) -> str:
    """심볼 정의의 첫 몇 줄을 소스에서 발췌한다 (summary에 붙일 코드 미리보기).

    임베딩 요약에 "실제 코드가 이렇게 생겼다"는 힌트를 담기 위해 정의 시작 지점부터
    최대 n줄을 잘라 온다. 전체 본문을 넣으면 요약이 너무 길어지므로 앞부분만 쓴다.

    Args:
        source_lines: 파일 전체를 줄 단위로 분해한 리스트.
        start: 심볼이 시작하는 1-indexed 줄 번호(파서가 알려준 값).
        end: 심볼이 끝나는 줄 번호(현재 구현에서는 상한 계산에 직접 쓰지 않음).
        n: 발췌할 최대 줄 수.

    Returns:
        발췌한 코드 문자열. 소스가 비었거나 start가 유효하지 않으면 빈 문자열.
    """
    if not source_lines or start <= 0:
        return ""
    # 파서가 주는 줄 번호는 1부터 시작하므로 리스트 인덱스(0부터)로 보정한다.
    start_idx = max(0, start - 1)
    # 시작 지점부터 n줄까지, 단 파일 끝을 넘지 않도록 상한을 건다.
    end_idx = min(len(source_lines), start_idx + n)
    return "\n".join(source_lines[start_idx:end_idx])


def _build_summary(
    kind: str, qualified: str, signature: str, docstring: str,
    excerpt: str,
) -> str:
    """임베딩·검색에 쓸 심볼 요약 텍스트를 조립한다.

    모듈 docstring에 적힌 규칙대로 "종류 + 정규화 이름 + 시그니처", 그 아래 docstring,
    그 아래 실제 코드 발췌를 빈 줄로 구분해 이어 붙인다. 이 문자열이 의미 검색의
    재료가 되므로 사람이 읽어도 이해되는 형태로 만드는 것이 핵심이다.

    Args:
        kind: 심볼 종류(function/class/method 등).
        qualified: 정규화된 전체 이름(예: 모듈.클래스.메서드).
        signature: 시그니처 문자열(인자 목록 등).
        docstring: 심볼의 docstring(있으면 앞 600자만 사용).
        excerpt: `_source_excerpt()`로 얻은 코드 발췌.

    Returns:
        빈 줄로 구분해 조립한 요약 텍스트.
    """
    # 첫 줄은 항상 "종류 이름+시그니처" 형태로 만든다.
    parts = [f"{kind} {qualified}{signature}"]
    # docstring이 있으면 양끝 공백을 제거하고 과도한 길이를 600자로 제한해 붙인다.
    if docstring:
        parts.append(docstring.strip()[:600])
    # 코드 발췌가 있으면 "source:" 라벨과 함께 붙인다.
    if excerpt:
        parts.append("source:\n" + excerpt)
    return "\n\n".join(parts)


def _parsed_to_entry(
    sym: ParsedSymbol,
    source_text: str,
    *, path: str, module: str, project_source: str,
) -> SymbolEntry:
    """파서가 뱉은 ParsedSymbol을 DB 적재 모델 SymbolEntry로 변환한다.

    파싱 결과(구조 정보)에 더해 summary(임베딩 대상 텍스트)와 태그를 얹어서,
    SymbolStore가 바로 저장할 수 있는 완성된 엔트리로 만드는 다리 역할.

    Args:
        sym: 파서가 추출한 단일 심볼 정보.
        source_text: 해당 파일의 전체 소스(코드 발췌를 뽑기 위해 필요).
        path: 프로젝트 루트 기준 상대 경로(저장·조회 키로 사용).
        module: `a.b.c` 형태의 모듈 경로.
        project_source: 이 심볼이 속한 프로젝트/소스 식별자(예: "nexus").

    Returns:
        summary와 태그까지 채워진 SymbolEntry (embedding은 아직 비어 있음).
    """
    # 코드 발췌를 뽑으려면 소스를 줄 단위로 나눠 둬야 한다.
    source_lines = source_text.splitlines()
    excerpt = _source_excerpt(source_lines, sym.line_start, sym.line_end, n=6)
    summary = _build_summary(
        sym.kind, sym.qualified_name, sym.signature, sym.docstring, excerpt,
    )
    # 언어 태그와 파서가 준 추가 태그를 합치되, 빈 값(None/"")은 걸러 낸다.
    tags = tuple(filter(None, (sym.language, *sym.extra_tags)))
    return SymbolEntry(
        source=project_source,
        path=path,
        module=module,
        kind=sym.kind,
        name=sym.name,
        qualified_name=sym.qualified_name,
        signature=sym.signature,
        docstring=sym.docstring,
        summary=summary,
        line_start=sym.line_start,
        line_end=sym.line_end,
        tags=tags,
    )


def extract_symbols_from_source(
    source_text: str,
    path: str,
    module: str,
    project_source: str = "nexus",
    parser: BaseParser | None = None,
    registry: ParserRegistry | None = None,
) -> list[SymbolEntry]:
    """파일 하나의 소스 텍스트에서 심볼 목록을 추출한다 (임베딩은 여기서 안 함).

    임베딩 벡터 채우기는 적재 단계(SymbolProjectIndexer._embed_entries)에서 따로
    처리하므로, 이 함수는 순수하게 "소스 → 구조 심볼 목록" 변환만 담당한다.

    파서 선택 우선순위:
      1. parser 인자가 명시되면 → 그 파서를 그대로 사용.
      2. registry가 주어지면 → 파일 확장자로 알맞은 파서를 조회.
      3. 둘 다 없으면 → 확장자가 `.py`일 때만 PythonParser 자동 사용(하위 호환).

    지원하지 않는 언어(적합한 파서가 없는 경우)는 예외 없이 조용히 빈 목록을 반환한다.

    Args:
        source_text: 파일의 전체 소스 텍스트.
        path: 프로젝트 루트 기준 상대 경로(확장자 판별·저장 키로 사용).
        module: `a.b.c` 형태의 모듈 경로.
        project_source: 프로젝트/소스 식별자. 기본값 "nexus".
        parser: 강제로 사용할 파서(선택).
        registry: 확장자→파서 매핑 레지스트리(선택).

    Returns:
        추출된 SymbolEntry 리스트. 파서가 없으면 빈 리스트.
    """
    if parser is None:
        if registry is None:
            # 하위 호환 경로: 명시적 설정이 하나도 없으면 Python 파일만 처리한다.
            if not path.lower().endswith(".py"):
                return []
            # 순환 import를 피하기 위해 여기서 지연(lazy) import 한다.
            from core.rag.parsers.python_parser import PythonParser
            parser = PythonParser()
        else:
            # 레지스트리가 있으면 확장자에 맞는 파서를 찾는다. 없으면 skip.
            parser = registry.for_path(path)
            if parser is None:
                return []

    # 선택된 파서로 소스를 파싱하고, 각 심볼을 SymbolEntry로 변환해 리스트로 반환.
    parsed = parser.parse(source_text, path)
    return [
        _parsed_to_entry(
            s, source_text,
            path=path, module=module, project_source=project_source,
        )
        for s in parsed
    ]


def module_name_for(path: Path, root: Path) -> str:
    """파일 경로를 `a.b.c` 형태의 점(.) 구분 모듈 경로로 변환한다.

    루트를 기준으로 상대 경로를 구한 뒤 확장자를 떼고 경로 구분자를 점으로 바꾼다.
    파일이 루트 밖에 있어 상대 경로를 만들 수 없으면 원본 경로를 그대로 쓴다.
    `__init__` 파일은 패키지 자체를 가리키므로 마지막 조각을 떼어 낸다.

    Args:
        path: 대상 파일 경로.
        root: 모듈 경로 계산의 기준이 되는 루트 디렉토리.

    Returns:
        점으로 구분된 모듈 경로 문자열(예: "core.rag.symbol_indexer").
    """
    try:
        rel = path.resolve().relative_to(root.resolve())
    except ValueError:
        # 루트 바깥 경로 등 상대화가 불가능한 경우 원본 경로를 그대로 사용.
        rel = path
    # 확장자를 제거하고 경로 조각 리스트로 분해한다.
    parts = list(rel.with_suffix("").parts)
    # 마지막 조각이 __init__이면 패키지를 뜻하므로 이름에서 뺀다.
    if parts and parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


# ─────────────────────────────────────────────
# 프로젝트 인덱서
# ─────────────────────────────────────────────
class SymbolProjectIndexer:
    """프로젝트 전체 소스를 훑어 SymbolStore에 적재하는 오케스트레이터 (다언어).

    앞서 정의한 순회·추출·변환 함수들을 엮어서 "루트 디렉토리 하나 → DB 적재 완료"
    까지의 전체 흐름을 담당한다. 파일마다 delete + add_many로 교체하므로 몇 번을
    다시 돌려도 결과가 같은(멱등) 재인덱싱이 된다.

    구성 요소:
        embedder: 임베딩 배치 함수 (list[str] → list[list[float]]). None이면
            임베딩 없이 구조 메타데이터만 적재한다(텍스트 검색은 여전히 가능).
        registry: 언어별 Parser 레지스트리(확장자 라우팅). None이면 기본 등록.
    """

    def __init__(
        self,
        store: SymbolStore,
        embedder: Any | None = None,
        project_source: str = "nexus",
        registry: ParserRegistry | None = None,
    ) -> None:
        """인덱서를 초기화한다.

        Args:
            store: 심볼을 저장·조회할 SymbolStore(DB 접근 계층).
            embedder: summary를 벡터로 바꾸는 비동기 배치 함수. None이면 임베딩 생략.
            project_source: 적재되는 심볼에 붙일 프로젝트/소스 식별자.
            registry: 확장자→파서 레지스트리. None이면 기본 레지스트리를 생성해 사용.
        """
        self._store = store
        self._embedder = embedder
        self._source = project_source
        self._registry = registry or build_default_registry()

    async def index_project(self, root: str | Path) -> dict[str, int]:
        """루트 아래 지원 확장자 파일을 전부 다시 인덱싱한다 (파일 단위 멱등 교체).

        각 파일마다: 읽기 → 심볼 추출 → (선택)임베딩 → 기존 것 삭제 → 새로 삽입.
        delete_by_path와 add_many를 짝지어 쓰기 때문에 같은 파일을 여러 번 인덱싱해도
        중복이 쌓이지 않고 항상 최신 상태로 덮어써진다.

        Args:
            root: 인덱싱할 프로젝트 루트 경로(문자열 또는 Path).

        Returns:
            처리 결과 집계 딕셔너리 {"files": 처리 파일 수, "symbols": 적재 심볼 수}.
        """
        root_path = Path(root).resolve()
        # 적재 전에 저장소 스키마(테이블 등)가 준비돼 있는지 보장한다.
        await self._store.ensure_schema()

        files_done = 0
        symbols_done = 0
        extensions = tuple(self._registry.supported_extensions())
        for file_path in iter_source_files(root_path, extensions=extensions):
            try:
                # 인코딩 오류가 나도 멈추지 않도록 errors="replace"로 안전하게 읽는다.
                text = file_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                # 읽을 수 없는 파일(권한·삭제 등)은 건너뛴다.
                continue

            # 저장 키로 쓸 상대 경로. OS 구분자 차이를 없애려 항상 슬래시로 통일한다.
            rel = str(file_path.resolve().relative_to(root_path)).replace("\\", "/")
            module = module_name_for(file_path, root_path)
            entries = extract_symbols_from_source(
                text, path=rel, module=module, project_source=self._source,
                registry=self._registry,
            )
            # 심볼이 하나도 없는 파일은 삭제·삽입할 것도 없으니 넘어간다.
            if not entries:
                continue

            # 임베더가 설정돼 있으면 summary를 벡터로 채운 새 엔트리로 교체한다.
            if self._embedder is not None:
                entries = await self._embed_entries(entries)

            # 파일 단위 교체(멱등): 같은 경로의 기존 심볼을 지우고 새로 넣는다.
            await self._store.delete_by_path(self._source, rel)
            await self._store.add_many(entries)

            files_done += 1
            symbols_done += len(entries)
            # 진행 상황을 50개 파일마다 로그로 남겨 장시간 작업의 상태를 알 수 있게 한다.
            if files_done % 50 == 0:
                logger.info(
                    "심볼 인덱싱 진행: files=%d, symbols=%d",
                    files_done, symbols_done,
                )

        logger.info(
            "심볼 인덱싱 완료: files=%d, symbols=%d (source=%s)",
            files_done, symbols_done, self._source,
        )
        return {"files": files_done, "symbols": symbols_done}

    # ─── 내부 ─────────────────────────────────────
    async def _embed_entries(self, entries: list[SymbolEntry]) -> list[SymbolEntry]:
        """엔트리들의 summary를 배치 임베딩해 embedding 필드를 채운 새 리스트를 반환.

        SymbolEntry는 불변(immutable) 모델로 다루므로, 기존 객체를 수정하지 않고
        embedding만 채운 새 객체를 만들어 돌려준다. 임베딩 도중 예외가 나면 실패로
        중단하지 않고 원본(embedding 없는) 엔트리를 그대로 반환한다 — 이렇게 하면
        임베딩 서버가 죽어 있어도 구조/텍스트 검색용 데이터는 손실 없이 저장된다.

        Args:
            entries: 임베딩을 채워 넣을 심볼 엔트리 리스트.

        Returns:
            embedding이 채워진 새 엔트리 리스트. 임베딩 실패 시 원본 리스트 그대로.
        """
        texts = [e.summary for e in entries]
        vectors: list[list[float]] = []
        try:
            # 긴 목록을 EMBED_BATCH_SIZE 단위로 잘라 여러 번 나눠 임베딩 요청한다.
            for i in range(0, len(texts), EMBED_BATCH_SIZE):
                batch = texts[i:i + EMBED_BATCH_SIZE]
                vecs = await self._embedder(batch)
                vectors.extend(vecs)
        except Exception as e:
            # 임베딩 실패는 치명적 오류가 아니라 "구조만 저장"으로 우아하게 강등한다.
            logger.warning("심볼 임베딩 실패 (구조만 저장): %s", e)
            return entries  # embedding=None 그대로 저장

        updated: list[SymbolEntry] = []
        # 엔트리와 벡터를 짝지어, embedding만 채운 새 SymbolEntry를 만든다.
        # strict=False라 길이가 어긋나면 짧은 쪽 기준으로 멈춘다(방어적 처리).
        for e, v in zip(entries, vectors, strict=False):
            updated.append(
                SymbolEntry(
                    source=e.source, path=e.path, module=e.module,
                    kind=e.kind, name=e.name, qualified_name=e.qualified_name,
                    signature=e.signature, docstring=e.docstring,
                    summary=e.summary, line_start=e.line_start, line_end=e.line_end,
                    tags=e.tags, metadata=e.metadata,
                    embedding=tuple(v),
                )
            )
        return updated


# ─────────────────────────────────────────────
# 편의 — bootstrap에서 fire-and-forget 호출
# ─────────────────────────────────────────────
# ivfflat 벡터 인덱스를 빌드하기 위한 최소 행수 임계치.
# 사양서 Ch 12 권장(1000+ rows)을 따른다. 데이터가 너무 적을 때 인덱스를 만들면
# lists=100 설정 대비 통계 표본이 부족해, 오히려 검색 품질이 떨어질 수 있다.
# 인덱스가 이미 있으면 IF NOT EXISTS로 아무 일도 하지 않는다(no-op).
_SYMBOL_VECTOR_INDEX_MIN_ROWS = 1000


async def background_index(indexer: SymbolProjectIndexer, root: str | Path) -> None:
    """예외를 삼키는 백그라운드 인덱싱 래퍼 — bootstrap의 create_task와 함께 쓴다.

    앱 기동(bootstrap) 시 `asyncio.create_task()`로 던져 두면, 서비스는 곧바로
    응답을 시작하고 인덱싱은 뒤에서 조용히 돈다. 어떤 예외가 나도 앱 기동을 막지
    않도록 여기서 모두 잡아 WARNING 로그만 남긴다.

    초기 인덱싱이 끝난 뒤 저장된 행수가 임계치를 넘으면 ivfflat 벡터 인덱스도
    빌드한다. 데이터가 충분히 쌓인 뒤 늦게 빌드해야 데이터 분포가 잡혀 검색
    품질이 좋아지기 때문에, 인덱싱 직후 이 시점에 만드는 것이다.

    Args:
        indexer: 실제 인덱싱을 수행할 SymbolProjectIndexer.
        root: 인덱싱할 프로젝트 루트 경로.
    """
    try:
        await indexer.index_project(root)
    except Exception as e:
        # 인덱싱 자체가 실패해도 앱은 계속 떠 있어야 하므로 무시하고 종료.
        logger.warning("심볼 백그라운드 인덱싱 실패 (무시): %s", e)
        return

    # 인덱싱이 성공했을 때만, 행수 조건을 만족하면 벡터 인덱스를 빌드한다.
    try:
        store = indexer._store  # noqa: SLF001 — 같은 모듈 내부 협력
        row_count = await store.count()
        if row_count >= _SYMBOL_VECTOR_INDEX_MIN_ROWS:
            await store.build_vector_index()
            logger.info(
                "tb_symbols ivfflat 인덱스 빌드 완료 (rows=%d)", row_count,
            )
        else:
            # 아직 데이터가 부족하면 빌드를 미룬다(다음 인덱싱 때 재평가).
            logger.debug(
                "tb_symbols 행수 %d < %d — ivfflat 빌드 보류",
                row_count, _SYMBOL_VECTOR_INDEX_MIN_ROWS,
            )
    except Exception as e:
        # 인덱스 빌드 실패는 검색 품질 저하일 뿐 본류 응답은 가능 — WARNING만 남긴다.
        logger.warning("tb_symbols 벡터 인덱스 빌드 실패 (무시): %s", e)
