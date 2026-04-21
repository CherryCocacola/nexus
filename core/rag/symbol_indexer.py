"""
다언어 심볼 추출기 (Phase 10.0 + 확장, 2026-04-21).

지원 언어: Python(ast), JavaScript/TypeScript(정규식), Go(정규식).
언어별 구체 파서는 `core/rag/parsers/`에 위치하고, 본 모듈은 ParserRegistry
경유로 확장자를 라우팅한다.

각 심볼의 summary(임베딩 대상 텍스트)는 다음 규칙으로 구성:
  "{kind} {qualified_name}{signature}
  {docstring}
  source:
  {최대 5줄 발췌}"

설계 결정:
  - 에어갭 준수 — 외부 파서 라이브러리 없이 표준 ast / 정규식만 사용
  - 파일 단위 트랜잭션: `delete_by_path()` → `add_many()` 순으로 재인덱싱 멱등
  - 임베딩 실패(서버 다운 등)해도 본문과 메타데이터는 저장 → 텍스트 검색은 가능
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


# 인덱싱 제외 디렉토리 (indexer.py와 동일 규칙 공유)
EXCLUDED_DIRS = {
    ".git", "__pycache__", "node_modules", ".venv", "venv",
    ".mypy_cache", ".pytest_cache", ".ruff_cache",
    "dist", "build", "egg-info",
    "models", "checkpoints", "data", "logs",
    ".nexus", "kowiki_ingest",
}

# 단일 파일 최대 크기 — 초과 시 건너뜀 (생성된 대형 파일 보호)
MAX_FILE_SIZE = 200 * 1024

# 임베딩 배치 크기
EMBED_BATCH_SIZE = 16


# ─────────────────────────────────────────────
# AST 추출
# ─────────────────────────────────────────────
def iter_python_files(root: Path) -> Iterator[Path]:
    """인덱싱 대상 .py 파일을 순회한다 (하위 호환 함수)."""
    yield from iter_source_files(root, extensions=(".py",))


def iter_source_files(
    root: Path,
    extensions: tuple[str, ...] | None = None,
) -> Iterator[Path]:
    """
    인덱싱 대상 소스 파일을 순회한다.

    Args:
        root: 스캔 시작 디렉토리
        extensions: 허용 확장자 튜플 (소문자, 점 포함). None이면 ParserRegistry
            기본 확장자(py/js/jsx/mjs/cjs/ts/tsx/go).
    """
    if extensions is None:
        extensions = tuple(build_default_registry().supported_extensions())
    exts = tuple(e.lower() for e in extensions)

    for current, dirs, files in _walk(root):
        dirs[:] = [d for d in dirs if d not in EXCLUDED_DIRS and not d.startswith(".")]
        for f in files:
            p = current / f
            if p.suffix.lower() not in exts:
                continue
            try:
                if p.stat().st_size <= MAX_FILE_SIZE:
                    yield p
            except OSError:
                continue


def _walk(root: Path):
    """os.walk 래퍼 — 타입 힌트 명확화 목적. Path를 돌려준다."""
    import os
    for dirpath, dirnames, filenames in os.walk(root):
        yield Path(dirpath), dirnames, filenames


def _source_excerpt(source_lines: list[str], start: int, end: int, n: int = 5) -> str:
    """AST/정규식에서 얻은 소스의 앞부분 몇 줄을 발췌한다."""
    if not source_lines or start <= 0:
        return ""
    # 1-indexed
    start_idx = max(0, start - 1)
    end_idx = min(len(source_lines), start_idx + n)
    return "\n".join(source_lines[start_idx:end_idx])


def _build_summary(
    kind: str, qualified: str, signature: str, docstring: str,
    excerpt: str,
) -> str:
    """임베딩에 넣을 요약 텍스트를 조립한다."""
    parts = [f"{kind} {qualified}{signature}"]
    if docstring:
        parts.append(docstring.strip()[:600])
    if excerpt:
        parts.append("source:\n" + excerpt)
    return "\n\n".join(parts)


def _parsed_to_entry(
    sym: ParsedSymbol,
    source_text: str,
    *, path: str, module: str, project_source: str,
) -> SymbolEntry:
    """ParsedSymbol → SymbolEntry 변환 (summary + 태그 포함)."""
    source_lines = source_text.splitlines()
    excerpt = _source_excerpt(source_lines, sym.line_start, sym.line_end, n=6)
    summary = _build_summary(
        sym.kind, sym.qualified_name, sym.signature, sym.docstring, excerpt,
    )
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
    """
    단일 파일의 소스에서 심볼을 추출한다 (임베딩은 적재 단계에서 따로).

    파서 선택:
      - parser가 명시되면 그 파서 사용
      - registry가 주어지면 path.suffix로 파서 조회
      - 둘 다 없으면 확장자 `.py`일 때만 PythonParser로 자동 (하위 호환)

    파서가 없으면 빈 리스트 반환 (지원 안 하는 언어는 조용히 skip).
    """
    if parser is None:
        if registry is None:
            # 하위 호환: 명시적 설정 없으면 Python만
            if not path.lower().endswith(".py"):
                return []
            from core.rag.parsers.python_parser import PythonParser
            parser = PythonParser()
        else:
            parser = registry.for_path(path)
            if parser is None:
                return []

    parsed = parser.parse(source_text, path)
    return [
        _parsed_to_entry(
            s, source_text,
            path=path, module=module, project_source=project_source,
        )
        for s in parsed
    ]


def module_name_for(path: Path, root: Path) -> str:
    """파일 경로를 `a.b.c` 모듈 경로로 변환한다."""
    try:
        rel = path.resolve().relative_to(root.resolve())
    except ValueError:
        rel = path
    parts = list(rel.with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


# ─────────────────────────────────────────────
# 프로젝트 인덱서
# ─────────────────────────────────────────────
class SymbolProjectIndexer:
    """프로젝트 전체 소스 파일 → SymbolStore 적재기 (다언어 지원).

    embedder는 임베딩 배치 함수 (list[str] → list[list[float]]). None이면
    임베딩 없이 구조 메타데이터만 적재 (텍스트 검색은 여전히 가능).

    registry는 언어별 Parser 레지스트리 (확장자 라우팅). None이면 기본 등록.
    """

    def __init__(
        self,
        store: SymbolStore,
        embedder: Any | None = None,
        project_source: str = "nexus",
        registry: ParserRegistry | None = None,
    ) -> None:
        self._store = store
        self._embedder = embedder
        self._source = project_source
        self._registry = registry or build_default_registry()

    async def index_project(self, root: str | Path) -> dict[str, int]:
        """
        루트 디렉토리 아래 지원 확장자 파일을 전부 재인덱싱한다
        (파일 단위 delete + add_many로 멱등).
        """
        root_path = Path(root).resolve()
        await self._store.ensure_schema()

        files_done = 0
        symbols_done = 0
        extensions = tuple(self._registry.supported_extensions())
        for file_path in iter_source_files(root_path, extensions=extensions):
            try:
                text = file_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue

            rel = str(file_path.resolve().relative_to(root_path)).replace("\\", "/")
            module = module_name_for(file_path, root_path)
            entries = extract_symbols_from_source(
                text, path=rel, module=module, project_source=self._source,
                registry=self._registry,
            )
            if not entries:
                continue

            # 임베딩 배치
            if self._embedder is not None:
                entries = await self._embed_entries(entries)

            # 파일 단위 교체 (멱등)
            await self._store.delete_by_path(self._source, rel)
            await self._store.add_many(entries)

            files_done += 1
            symbols_done += len(entries)
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
        """entry.summary를 배치 임베딩하여 embedding 필드를 채운 새 리스트를 반환."""
        texts = [e.summary for e in entries]
        vectors: list[list[float]] = []
        try:
            for i in range(0, len(texts), EMBED_BATCH_SIZE):
                batch = texts[i:i + EMBED_BATCH_SIZE]
                vecs = await self._embedder(batch)
                vectors.extend(vecs)
        except Exception as e:
            logger.warning("심볼 임베딩 실패 (구조만 저장): %s", e)
            return entries  # embedding=None 그대로 저장

        updated: list[SymbolEntry] = []
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
async def background_index(indexer: SymbolProjectIndexer, root: str | Path) -> None:
    """예외 삼킴 래퍼 — bootstrap의 `asyncio.create_task()`와 함께 쓴다."""
    try:
        await indexer.index_project(root)
    except Exception as e:
        logger.warning("심볼 백그라운드 인덱싱 실패 (무시): %s", e)
