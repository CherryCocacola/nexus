"""
Python AST 기반 심볼 추출기 (Phase 10.0, 2026-04-21).

프로젝트 내 `.py` 파일을 순회하며 다음 심볼을 추출하여 SymbolStore에 적재:
  - 모듈 top-level 함수 / 비동기 함수
  - 클래스
  - 클래스의 메서드 / 비동기 메서드

각 심볼의 summary(임베딩 대상 텍스트)는 다음 규칙으로 구성:
  "{kind} {qualified_name}{signature}
  {docstring}
  source:
  {최대 5줄 발췌}"

설계 결정:
  - 외부 의존성 없이 표준 `ast` 모듈만 사용 — 에어갭 준수
  - 파일 단위 트랜잭션: `delete_by_path()` → `add_many()` 순으로 재인덱싱 멱등
  - 임베딩 실패(서버 다운 등)해도 본문과 메타데이터는 저장 → 텍스트 검색은 가능
"""

from __future__ import annotations

import ast
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

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
    """인덱싱 대상 .py 파일을 순회한다."""
    for current, dirs, files in _walk(root):
        # 제외 디렉토리 즉시 가지치기
        dirs[:] = [d for d in dirs if d not in EXCLUDED_DIRS and not d.startswith(".")]
        for f in files:
            if f.endswith(".py"):
                p = current / f
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


def _signature_of(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    """함수/메서드의 시그니처 문자열을 재구성한다."""
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
    return f"({args}){ret}"


def _source_excerpt(source_lines: list[str], start: int, end: int, n: int = 5) -> str:
    """AST 노드의 source에서 앞부분 몇 줄을 발췌한다."""
    if not source_lines or start <= 0:
        return ""
    # ast line은 1-indexed
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
        # 긴 docstring은 너무 커지면 의미가 희석되므로 제한
        parts.append(docstring.strip()[:600])
    if excerpt:
        parts.append("source:\n" + excerpt)
    return "\n\n".join(parts)


def extract_symbols_from_source(
    source_text: str,
    path: str,
    module: str,
    project_source: str = "nexus",
) -> list[SymbolEntry]:
    """
    단일 파일의 소스에서 심볼을 추출한다 (임베딩은 적재 단계에서 따로).

    추출 대상:
      - top-level FunctionDef / AsyncFunctionDef
      - top-level ClassDef (클래스 자체 + 내부 메서드)
      - 중첩 클래스도 재귀 순회

    private 식별자(_leading_underscore)는 기본 포함 — 필요 시 호출자가 필터.
    """
    try:
        tree = ast.parse(source_text, filename=path)
    except SyntaxError as e:
        logger.debug("SyntaxError (%s): %s", path, e)
        return []

    source_lines = source_text.splitlines()
    out: list[SymbolEntry] = []

    def _mk_func(
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        parent_qual: str,
        parent_kind: str | None,  # 'class'이면 이 함수는 메서드
    ) -> SymbolEntry:
        is_async = isinstance(node, ast.AsyncFunctionDef)
        if parent_kind == "class":
            kind = "async_method" if is_async else "method"
        else:
            kind = "async_function" if is_async else "function"
        sig = _signature_of(node)
        doc = ast.get_docstring(node) or ""
        qual = f"{parent_qual}.{node.name}" if parent_qual else node.name
        line_start = getattr(node, "lineno", 0) or 0
        line_end = getattr(node, "end_lineno", line_start) or line_start
        excerpt = _source_excerpt(source_lines, line_start, line_end, n=6)
        summary = _build_summary(kind, qual, sig, doc, excerpt)
        return SymbolEntry(
            source=project_source,
            path=path,
            module=module,
            kind=kind,
            name=node.name,
            qualified_name=qual,
            signature=sig,
            docstring=doc,
            summary=summary,
            line_start=line_start,
            line_end=line_end,
        )

    def _visit(body: list[ast.stmt], parent_qual: str, parent_kind: str | None) -> None:
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                out.append(_mk_func(node, parent_qual, parent_kind))
            elif isinstance(node, ast.ClassDef):
                qual = f"{parent_qual}.{node.name}" if parent_qual else node.name
                doc = ast.get_docstring(node) or ""
                line_start = getattr(node, "lineno", 0) or 0
                line_end = getattr(node, "end_lineno", line_start) or line_start
                excerpt = _source_excerpt(source_lines, line_start, line_end, n=6)
                # 클래스 시그니처 = 베이스 클래스 목록
                try:
                    bases = ", ".join(ast.unparse(b) for b in node.bases)
                except Exception:
                    bases = ""
                sig = f"({bases})" if bases else ""
                summary = _build_summary("class", qual, sig, doc, excerpt)
                out.append(
                    SymbolEntry(
                        source=project_source,
                        path=path,
                        module=module,
                        kind="class",
                        name=node.name,
                        qualified_name=qual,
                        signature=sig,
                        docstring=doc,
                        summary=summary,
                        line_start=line_start,
                        line_end=line_end,
                    )
                )
                # 클래스 내부도 재귀 — 부모 kind='class'
                _visit(node.body, qual, "class")

    # tree.body는 top-level. 부모 없음.
    _visit(tree.body, "", None)
    return out


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
    """프로젝트 전체 Python 파일 → SymbolStore 적재기.

    embedder는 임베딩 배치 함수 (list[str] → list[list[float]]). None이면
    임베딩 없이 구조 메타데이터만 적재 (텍스트 검색은 여전히 가능).
    """

    def __init__(
        self,
        store: SymbolStore,
        embedder: Any | None = None,
        project_source: str = "nexus",
    ) -> None:
        self._store = store
        self._embedder = embedder
        self._source = project_source

    async def index_project(self, root: str | Path) -> dict[str, int]:
        """루트 디렉토리 아래 .py 파일을 전부 재인덱싱한다 (파일 단위 delete + add_many)."""
        root_path = Path(root).resolve()
        await self._store.ensure_schema()

        files_done = 0
        symbols_done = 0
        for file_path in iter_python_files(root_path):
            try:
                text = file_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue

            rel = str(file_path.resolve().relative_to(root_path)).replace("\\", "/")
            module = module_name_for(file_path, root_path)
            entries = extract_symbols_from_source(
                text, path=rel, module=module, project_source=self._source
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
