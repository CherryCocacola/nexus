"""
Memory 도구 모음 — 메모리 검색 및 저장.

2개의 메모리 도구를 제공한다:
  - MemoryRead: 장기/단기 메모리에서 관련 정보를 검색 (읽기 전용)
  - MemoryWrite: 새 정보를 메모리에 저장

Nexus의 메모리 시스템은 2계층으로 구성된다:
  - 단기 메모리: Redis (세션 내 빠른 접근)
  - 장기 메모리: PostgreSQL + pgvector (의미 검색)

실제 MemoryManager는 Phase 5.0에서 구현되므로,
현재는 context.options.get("memory_manager")로 접근하되
없으면 간단한 인메모리 딕셔너리 폴백을 사용한다.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from core.tools.base import (
    BaseTool,
    PermissionBehavior,
    PermissionResult,
    ToolResult,
    ToolUseContext,
)

logger = logging.getLogger("nexus.tools.memory")

# 인메모리 폴백 저장소 (MemoryManager가 없을 때 사용)
# 키: memory_id, 값: {"content": str, "tags": list, "created_at": float}
_fallback_memory: dict[str, dict[str, Any]] = {}

# 메모리 ID 카운터
_memory_counter = 0


def _get_memory_manager(context: ToolUseContext) -> Any:
    """
    context.options에서 memory_manager를 가져온다.
    없으면 None을 반환하여 인메모리 폴백을 사용하게 한다.
    """
    return context.options.get("memory_manager")


# ─────────────────────────────────────────────
# MemoryReadTool — 메모리 검색
# ─────────────────────────────────────────────
class MemoryReadTool(BaseTool):
    """
    메모리에서 관련 정보를 검색하는 도구.
    키워드 기반 검색 또는 태그 필터링을 지원한다.
    실제 MemoryManager가 있으면 벡터 유사도 검색(pgvector)도 가능하다.
    읽기 전용이므로 병렬 실행이 안전하다.
    """

    # ═══ 1. Identity ═══

    @property
    def name(self) -> str:
        return "MemoryRead"

    @property
    def description(self) -> str:
        return (
            "메모리에서 관련 정보를 검색합니다. "
            "키워드 검색, 태그 필터링을 지원하며, "
            "MemoryManager 연동 시 의미(semantic) 검색도 가능합니다."
        )

    @property
    def group(self) -> str:
        return "memory"

    # ═══ 2. Schema ═══

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "검색 쿼리 (키워드 또는 자연어)",
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "태그 필터 (지정된 태그를 가진 메모리만 검색)",
                },
                "max_results": {
                    "type": "integer",
                    "description": "최대 결과 수 (기본: 10)",
                    "default": 10,
                    "minimum": 1,
                    "maximum": 50,
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
        """query가 비어 있는지 검증한다."""
        query = input_data.get("query", "")
        if not query or not query.strip():
            return "query는 비어 있을 수 없습니다."
        return None

    async def check_permissions(
        self, input_data: dict[str, Any], context: ToolUseContext
    ) -> PermissionResult:
        """읽기 전용이므로 항상 허용한다."""
        return PermissionResult(behavior=PermissionBehavior.ALLOW)

    async def call(self, input_data: dict[str, Any], context: ToolUseContext) -> ToolResult:
        """
        메모리를 검색한다.

        처리 순서:
          1. MemoryManager가 있으면 벡터 유사도 검색 위임
          2. 없으면 인메모리 폴백에서 키워드 매칭
          3. 태그 필터 적용
          4. 결과 포맷하여 반환
        """
        query = input_data["query"]
        tags = input_data.get("tags", [])
        max_results = input_data.get("max_results", 10)

        manager = _get_memory_manager(context)

        # TODO(nexus): Phase 5.0 MemoryManager 구현 후 벡터 검색 연동
        if manager and hasattr(manager, "search"):
            results = await manager.search(query=query, tags=tags, limit=max_results)
            if not results:
                return ToolResult.success("검색 결과가 없습니다.", count=0)
            # MemoryManager의 결과 형식에 따라 포맷 (추후 구현)
            return ToolResult.success(str(results), count=len(results))

        # 인메모리 폴백: 단순 키워드 매칭
        query_lower = query.lower()
        matches = []

        for mem_id, mem in _fallback_memory.items():
            content = mem.get("content", "")
            mem_tags = mem.get("tags", [])

            # 태그 필터: 지정된 태그가 모두 포함되어야 함
            if tags and not all(t in mem_tags for t in tags):
                continue

            # 키워드 매칭: 쿼리 단어 중 하나라도 content에 포함
            if query_lower in content.lower():
                matches.append((mem_id, mem))

        if not matches:
            return ToolResult.success(f"'{query}'에 대한 검색 결과가 없습니다.", count=0)

        # 최신순 정렬
        matches.sort(key=lambda x: x[1].get("created_at", 0), reverse=True)
        matches = matches[:max_results]

        # 결과 포맷
        lines = []
        for mem_id, mem in matches:
            tags_str = ", ".join(mem.get("tags", [])) if mem.get("tags") else "없음"
            lines.append(f"[{mem_id}] (태그: {tags_str})\n  {mem['content']}")

        logger.debug("MemoryRead: query='%s', %d results", query, len(matches))
        return ToolResult.success("\n\n".join(lines), count=len(matches))

    # ═══ 7. UI Hints ═══

    def get_progress_label(self, input_data: dict[str, Any]) -> str:
        return "Searching memory..."

    def get_input_summary(self, input_data: dict[str, Any]) -> str:
        return input_data.get("query", "")[:80]


# ─────────────────────────────────────────────
# MemoryWriteTool — 메모리 저장
# ─────────────────────────────────────────────
class MemoryWriteTool(BaseTool):
    """
    새 정보를 메모리에 저장하는 도구.
    텍스트 내용과 태그를 함께 저장할 수 있다.
    실제 MemoryManager가 있으면 벡터 임베딩도 함께 생성된다.
    """

    # ═══ 1. Identity ═══

    @property
    def name(self) -> str:
        return "MemoryWrite"

    @property
    def description(self) -> str:
        return (
            "새 정보를 메모리에 저장합니다. "
            "태그를 지정하여 분류할 수 있으며, "
            "MemoryManager 연동 시 벡터 임베딩도 자동 생성됩니다."
        )

    @property
    def group(self) -> str:
        return "memory"

    # ═══ 2. Schema ═══

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "저장할 내용 (텍스트)",
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "메모리 태그 목록 (분류/검색용)",
                },
            },
            "required": ["content"],
        }

    # ═══ 3. Behavior Flags ═══
    # 쓰기 도구 — 기본 fail-closed 유지

    # ═══ 5. Lifecycle ═══

    def validate_input(self, input_data: dict[str, Any]) -> str | None:
        """content가 비어 있는지 검증한다."""
        content = input_data.get("content", "")
        if not content or not content.strip():
            return "content는 비어 있을 수 없습니다."
        return None

    async def check_permissions(
        self, input_data: dict[str, Any], context: ToolUseContext
    ) -> PermissionResult:
        """쓰기 도구이므로 사용자에게 확인을 요청한다."""
        content = input_data.get("content", "")
        preview = content[:50] + "..." if len(content) > 50 else content
        return PermissionResult(
            behavior=PermissionBehavior.ASK,
            message=f"메모리 저장: {preview}",
        )

    async def call(self, input_data: dict[str, Any], context: ToolUseContext) -> ToolResult:
        """
        메모리에 새 정보를 저장한다.

        처리 순서:
          1. MemoryManager가 있으면 위임 (벡터 임베딩 포함)
          2. 없으면 인메모리 폴백에 저장
          3. 저장된 메모리 ID 반환
        """
        content = input_data["content"]
        tags = input_data.get("tags", [])

        manager = _get_memory_manager(context)

        # TODO(nexus): Phase 5.0 MemoryManager 구현 후 연동
        if manager and hasattr(manager, "store"):
            mem_id = await manager.store(content=content, tags=tags)
            logger.info("MemoryWrite: stored via manager, id=%s", mem_id)
            return ToolResult.success(f"메모리를 저장했습니다. (ID: {mem_id})", memory_id=mem_id)

        # 인메모리 폴백 저장
        global _memory_counter  # noqa: PLW0603 — 모듈 레벨 카운터, 간단한 폴백용
        _memory_counter += 1
        mem_id = f"mem-{_memory_counter:04d}"

        _fallback_memory[mem_id] = {
            "content": content,
            "tags": tags,
            "created_at": time.time(),
        }

        tags_str = ", ".join(tags) if tags else "없음"
        logger.info("MemoryWrite: stored in fallback, id=%s, tags=%s", mem_id, tags_str)
        return ToolResult.success(
            f"메모리를 저장했습니다. (ID: {mem_id}, 태그: {tags_str})\n"
            f"참고: 인메모리 폴백 모드입니다. 프로세스 종료 시 데이터가 사라집니다.",
            memory_id=mem_id,
        )

    # ═══ 7. UI Hints ═══

    def get_progress_label(self, input_data: dict[str, Any]) -> str:
        return "Saving to memory..."

    def get_input_summary(self, input_data: dict[str, Any]) -> str:
        return input_data.get("content", "")[:80]
