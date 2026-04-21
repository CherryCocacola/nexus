"""
SymbolSearch 도구 — tb_symbols(Phase 10.0)에서 심볼 검색 (2026-04-21).

사용 시나리오:
  - "query_loop 함수 어디 있어?"           → 이름 정확 매칭
  - "Config 관련 클래스 찾아줘"             → trigram 부분매칭
  - "토큰 추정하는 함수 있나?"              → 벡터 의미 검색 (이름 모를 때)

Scout 서브에이전트에 노출되어 프로젝트 탐색 속도를 대폭 끌어올린다.
Worker에게도 SymbolSearch가 노출되면 "심볼 명시적 탐색" 1단계로 활용 가능.

검색 모드 자동 선택:
  - 입력이 Python 식별자 모양(isidentifier) 또는 dotted path → 이름 매칭 우선
  - 그 외 자연어 → 벡터 검색 우선 (임베딩 프로바이더가 있을 때)
  - 이름 매칭 0건이고 임베딩 가능하면 벡터로 폴백
"""

from __future__ import annotations

import logging
import re
from typing import Any

from core.tools.base import (
    BaseTool,
    PermissionBehavior,
    PermissionResult,
    ToolResult,
    ToolUseContext,
)

logger = logging.getLogger("nexus.tools.symbol_search")

# 파이썬 식별자 또는 dotted path 검사
_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*$")


def _looks_like_identifier(q: str) -> bool:
    """입력이 식별자/qualified name 형태이면 True."""
    return bool(_IDENT_RE.match(q.strip())) if q else False


def _format_rows(rows: list[dict[str, Any]]) -> str:
    """검색 결과를 Worker가 읽기 좋은 텍스트로 포매팅한다."""
    if not rows:
        return "매칭된 심볼이 없습니다."
    lines: list[str] = []
    for r in rows:
        head = (
            f"{r['kind']} {r['qualified_name']}{r.get('signature', '')}  "
            f"({r['path']}:{r['line_start']}-{r['line_end']}, sim={r.get('similarity', 0):.2f})"
        )
        lines.append(head)
        doc = (r.get("docstring") or "").strip()
        if doc:
            first_line = doc.splitlines()[0][:160]
            lines.append(f"  → {first_line}")
    return "\n".join(lines)


class SymbolSearchTool(BaseTool):
    """
    프로젝트 심볼(함수/클래스/메서드) 검색 도구.

    tb_symbols 인덱스에서:
      1) 이름/qualified_name 정확/부분 매칭 (pg_trgm)
      2) (옵션) 벡터 유사도 검색 — 의미 기반

    query는 Python 식별자 또는 자연어 모두 허용. SymbolStore와 임베딩
    프로바이더는 `context.options["symbol_store"]` / `["model_provider"]`에서
    찾는다.
    """

    # ═══ 1. Identity ═══

    @property
    def name(self) -> str:
        return "SymbolSearch"

    @property
    def description(self) -> str:
        return (
            "Search indexed Python symbols (functions/classes/methods) by exact name, "
            "partial name (trigram), or natural-language meaning (vector). "
            "Returns file path + line range + signature + first docstring line. "
            "Use this BEFORE Grep/Glob when looking for a function or class definition."
        )

    @property
    def group(self) -> str:
        return "rag"

    @property
    def aliases(self) -> list[str]:
        return ["FindSymbol", "LookupSymbol"]

    # ═══ 2. Schema ═══

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Symbol name (e.g. 'query_loop'), qualified path "
                                   "(e.g. 'KnowledgeStore.add'), or natural-language "
                                   "description.",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Max results (default 10, cap 30)",
                    "default": 10,
                    "minimum": 1,
                    "maximum": 30,
                },
                "kind": {
                    "type": "string",
                    "description": "Filter by kind (function/class/method/async_function/"
                                   "async_method). Omit for all.",
                },
            },
            "required": ["query"],
        }

    # ═══ 3. Behavior Flags ═══

    @property
    def is_read_only(self) -> bool:
        return True

    @property
    def is_concurrency_safe(self) -> bool:
        return True

    # ═══ 5. Lifecycle ═══

    def validate_input(self, input_data: dict[str, Any]) -> str | None:
        q = (input_data.get("query") or "").strip()
        if not q:
            return "query는 비어 있을 수 없습니다."
        if len(q) > 500:
            return "query는 500자 이하여야 합니다."
        return None

    async def check_permissions(
        self, input_data: dict[str, Any], context: ToolUseContext
    ) -> PermissionResult:
        return PermissionResult(behavior=PermissionBehavior.ALLOW)

    async def call(
        self, input_data: dict[str, Any], context: ToolUseContext
    ) -> ToolResult:
        query = input_data["query"].strip()
        max_results = min(int(input_data.get("max_results", 10) or 10), 30)
        kind_filter = input_data.get("kind")

        store = context.options.get("symbol_store")
        if store is None:
            return ToolResult.error(
                "SymbolStore가 context.options에 없습니다. bootstrap 연결 확인 필요."
            )

        results: list[dict[str, Any]] = []

        # 1) 이름 경로 우선 시도 (식별자 또는 qualified name)
        if _looks_like_identifier(query):
            try:
                results = await store.search_by_name(query, top_k=max_results)
            except Exception as e:
                logger.warning("SymbolSearch 이름 검색 실패: %s", e)

        # 2) 결과 없거나 자연어면 벡터 검색 시도
        if not results:
            provider = context.options.get("model_provider")
            if provider is not None and hasattr(provider, "embed"):
                try:
                    vecs = await provider.embed([query])
                    if vecs and vecs[0]:
                        results = await store.search_by_vector(
                            embedding=vecs[0], top_k=max_results,
                        )
                except Exception as e:
                    logger.debug("SymbolSearch 벡터 검색 실패 (무시): %s", e)

        # 3) 벡터도 없으면 이름 부분매칭 마지막 시도
        if not results:
            try:
                results = await store.search_by_name(query, top_k=max_results)
            except Exception as e:
                logger.debug("SymbolSearch 이름 폴백 검색 실패 (무시): %s", e)

        # kind 필터 (클라이언트 측)
        if kind_filter:
            results = [r for r in results if r.get("kind") == kind_filter]

        if not results:
            return ToolResult.success(
                f"'{query}'에 매칭되는 심볼이 없습니다.", count=0,
            )

        text = _format_rows(results[:max_results])
        logger.info("SymbolSearch: '%s' → %d", query[:60], len(results))
        return ToolResult.success(text, count=len(results))
