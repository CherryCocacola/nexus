"""
도구 레지스트리 — 도구 등록, 조회, 필터링, 어셈블리.

Claude Code의 tools.ts (registry & assembler)를 재구현한다.

핵심 기능:
  1. 도구 등록 (register) + alias 등록
  2. 이름/alias 조회 (find_tool)
  3. deny 패턴 기반 필터링 (fnmatch)
  4. cache-stable 정렬 (이름순 — prompt cache 무효화 방지)
  5. 도구 풀 어셈블리 (assemble_tool_pool)

왜 이름순 정렬인가: Claude Code에서 도구 순서가 바뀌면
vLLM/OpenAI prompt cache가 무효화된다. 항상 같은 순서를 보장해야 한다.
"""

from __future__ import annotations

import fnmatch
import logging
from typing import Any

from core.tools.base import BaseTool

logger = logging.getLogger("nexus.tools.registry")


class ToolRegistry:
    """
    도구 레지스트리.
    모든 사용 가능한 도구를 관리하고, 쿼리 시점에 적절한 도구 풀을 어셈블한다.
    """

    def __init__(self):
        self._tools: dict[str, BaseTool] = {}
        self._alias_map: dict[str, str] = {}  # alias(소문자) → 정규 이름

    # ─── 등록 ───

    def register(self, tool: BaseTool) -> None:
        """
        도구를 등록한다. 동일 이름이면 나중 것이 우선한다 (override).
        alias도 자동 등록된다.
        """
        if tool.name in self._tools:
            logger.warning(f"도구 '{tool.name}' 이미 등록됨, 덮어쓰기")

        self._tools[tool.name] = tool

        # alias 등록
        for alias in tool.aliases:
            self._alias_map[alias.lower()] = tool.name

        logger.debug(f"도구 등록: {tool}")

    def register_many(self, tools: list[BaseTool]) -> None:
        """여러 도구를 일괄 등록한다."""
        for tool in tools:
            self.register(tool)

    def unregister(self, name: str) -> BaseTool | None:
        """도구를 등록 해제한다."""
        tool = self._tools.pop(name, None)
        if tool:
            for alias in tool.aliases:
                self._alias_map.pop(alias.lower(), None)
        return tool

    # ─── 조회 ───

    def get_all_tools(self) -> list[BaseTool]:
        """
        모든 등록된 도구를 반환한다 (활성/비활성 무관).
        이름순 정렬 (cache stability).
        """
        return sorted(self._tools.values(), key=lambda t: t.name)

    def find_tool(self, name: str) -> BaseTool | None:
        """
        이름 또는 alias로 도구를 조회한다.

        조회 순서:
          1. 정확한 이름 매칭
          2. alias 매칭 (대소문자 무시)
          3. 없으면 None
        """
        # 정확한 이름
        if name in self._tools:
            return self._tools[name]

        # alias (대소문자 무시)
        canonical = self._alias_map.get(name.lower())
        if canonical and canonical in self._tools:
            return self._tools[canonical]

        return None

    # ─── 필터링 & 어셈블리 ───

    def get_tools(
        self,
        deny_patterns: list[str] | None = None,
        only_enabled: bool = True,
        only_groups: list[str] | None = None,
    ) -> list[BaseTool]:
        """
        필터링된 도구 목록을 반환한다.

        Args:
            deny_patterns: 제외할 도구 이름 패턴 (fnmatch 지원)
            only_enabled: True면 비활성 도구 제외
            only_groups: 특정 그룹만 포함 (None이면 전체)

        Returns:
            필터링 + 이름순 정렬된 도구 목록
        """
        tools = list(self._tools.values())

        if only_enabled:
            tools = [t for t in tools if t.is_enabled]

        if only_groups:
            tools = [t for t in tools if t.group in only_groups]

        if deny_patterns:
            tools = self._filter_by_deny_rules(tools, deny_patterns)

        # 이름순 정렬 (cache stability)
        tools.sort(key=lambda t: t.name)
        return tools

    def assemble_tool_pool(
        self,
        deny_patterns: list[str] | None = None,
        only_enabled: bool = True,
    ) -> tuple[list[BaseTool], list[dict[str, Any]]]:
        """
        모델에 전달할 도구 풀을 어셈블한다.

        Returns:
            (tools, schemas): 도구 객체 리스트 + 모델 스키마 리스트
        """
        tools = self.get_tools(deny_patterns=deny_patterns, only_enabled=only_enabled)
        schemas = [tool.to_schema() for tool in tools]
        return tools, schemas

    @staticmethod
    def _filter_by_deny_rules(
        tools: list[BaseTool],
        patterns: list[str],
    ) -> list[BaseTool]:
        """
        deny 패턴으로 도구를 필터링한다.
        fnmatch 문법 지원: "Bash" (정확), "File*" (와일드카드), "*" (전체).
        """
        result = []
        for tool in tools:
            denied = any(fnmatch.fnmatch(tool.name, p) for p in patterns)
            if not denied:
                result.append(tool)
            else:
                logger.debug(f"도구 '{tool.name}' deny 패턴에 의해 제외됨")
        return result

    # ─── 통계 ───

    @property
    def tool_count(self) -> int:
        return len(self._tools)

    def __len__(self) -> int:
        return len(self._tools)

    def summary(self) -> str:
        """레지스트리 요약."""
        groups: dict[str, list[str]] = {}
        for t in self._tools.values():
            groups.setdefault(t.group, []).append(t.name)
        lines = [f"Tool Registry: {len(self._tools)}개 도구"]
        for group, names in sorted(groups.items()):
            lines.append(f"  [{group}]: {', '.join(sorted(names))}")
        return "\n".join(lines)
