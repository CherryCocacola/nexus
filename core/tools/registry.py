"""
도구 레지스트리(ToolRegistry) — 도구의 등록·조회·필터링·어셈블리를 담당하는 중앙 저장소.

이 파일이 하는 일 개요:
  Nexus에서 사용 가능한 모든 "도구(BaseTool 구현체)"를 한곳에 모아 두고,
  실제 쿼리(query)가 들어올 때 상황에 맞는 도구 묶음(도구 풀)을 골라
  모델에게 넘겨줄 수 있도록 준비한다.
  Claude Code의 tools.ts(registry & assembler 역할)를 파이썬으로 재구현한 것이다.

핵심 기능(왜 필요한지 함께):
  1. 도구 등록(register) + alias(별칭) 등록
     - 여러 도구를 한 저장소에 모아 이름으로 빠르게 찾기 위함.
  2. 이름/alias 조회(find_tool)
     - 모델이 부른 도구 이름이 정식 이름이든 별칭이든 찾아내기 위함.
  3. deny 패턴 기반 필터링(fnmatch)
     - 권한/정책상 특정 도구를 모델에게 아예 노출하지 않기 위함.
  4. cache-stable 정렬(이름순) — prompt cache 무효화 방지
     - 항상 같은 순서를 보장해 캐시를 살리기 위함.
  5. 도구 풀 어셈블리(assemble_tool_pool)
     - 도구 객체 목록과 모델용 스키마 목록을 한 번에 만들어 주기 위함.

왜 이름순 정렬인가:
  Claude Code에서 도구 순서가 바뀌면 vLLM/OpenAI의 prompt cache가 무효화된다.
  즉 매 요청마다 같은 도구를 넣어도 순서만 달라지면 캐시를 못 쓰게 되어
  지연·비용이 커진다. 그래서 항상 이름순으로 고정된 순서를 보장한다.

주요 클래스:
  - ToolRegistry: 위 5가지 기능을 모두 제공하는 단일 클래스.

의존:
  - core.tools.base.BaseTool (등록·관리 대상이 되는 도구의 추상 기반 클래스)

작성자: 이현수 / 작성일: 2026-07-05
"""

from __future__ import annotations

import fnmatch
import logging
from typing import Any

from core.tools.base import BaseTool

# 이 모듈 전용 로거. 규칙상 "nexus.{module}" 네임스페이스를 사용한다.
logger = logging.getLogger("nexus.tools.registry")


class ToolRegistry:
    """
    도구 레지스트리 — 모든 사용 가능한 도구를 보관·관리하는 저장소.

    역할:
      등록된 도구를 이름·별칭으로 조회하고, 쿼리 시점에 정책(deny 패턴·활성
      여부·그룹)에 맞춰 필요한 도구만 골라 "도구 풀"로 어셈블한다.

    내부 상태:
      - _tools: {정규 이름 → BaseTool} 매핑. 실제 도구 객체의 원본 저장소.
      - _alias_map: {소문자 별칭 → 정규 이름} 매핑. 별칭으로 원본을 찾기 위한
        인덱스. 별칭은 대소문자를 무시하려고 항상 소문자로 저장한다.

    사용 흐름:
      register/register_many로 도구를 채운 뒤, get_tools 또는
      assemble_tool_pool로 모델에 넘길 도구 묶음을 만들어 사용한다.
    """

    def __init__(self):
        # 정규 이름을 키로 도구 객체를 저장하는 기본 저장소.
        self._tools: dict[str, BaseTool] = {}
        # 별칭 → 정규 이름 인덱스. 키는 항상 소문자(대소문자 무시 조회용).
        self._alias_map: dict[str, str] = {}  # alias(소문자) → 정규 이름

    # ─── 등록 ───

    def register(self, tool: BaseTool) -> None:
        """
        도구 하나를 레지스트리에 등록한다.

        동작:
          - 같은 이름이 이미 있으면 경고 로그를 남기고 나중 것으로 덮어쓴다
            (override). 즉 마지막에 등록한 도구가 최종적으로 유효하다.
          - 도구가 가진 모든 별칭(aliases)도 함께 _alias_map에 등록한다.

        Args:
            tool: 등록할 BaseTool 구현체.

        Returns:
            없음(부수효과로 내부 상태만 갱신).
        """
        # 같은 이름이 이미 있으면 실수로 덮어쓰는 상황을 알 수 있게 경고를 남긴다.
        if tool.name in self._tools:
            logger.warning(f"도구 '{tool.name}' 이미 등록됨, 덮어쓰기")

        # 정규 이름을 키로 도구 객체를 저장(또는 덮어쓰기).
        self._tools[tool.name] = tool

        # alias 등록 — 각 별칭을 소문자로 바꿔 정규 이름과 연결한다.
        for alias in tool.aliases:
            self._alias_map[alias.lower()] = tool.name

        logger.debug(f"도구 등록: {tool}")

    def register_many(self, tools: list[BaseTool]) -> None:
        """
        여러 도구를 한 번에 등록한다. 내부적으로 register를 반복 호출한다.

        Args:
            tools: 등록할 BaseTool 목록.
        """
        for tool in tools:
            self.register(tool)

    def unregister(self, name: str) -> BaseTool | None:
        """
        이름으로 도구를 등록 해제한다.

        동작:
          - _tools에서 해당 도구를 제거하고, 그 도구가 가진 별칭들도
            _alias_map에서 함께 지워 인덱스가 어긋나지 않게 한다.

        Args:
            name: 제거할 도구의 정규 이름.

        Returns:
            제거된 BaseTool 객체. 존재하지 않으면 None.
        """
        # pop으로 꺼내면서 동시에 제거. 없으면 None이 반환된다.
        tool = self._tools.pop(name, None)
        if tool:
            # 도구가 실제로 있었을 때만 그 별칭들을 인덱스에서 제거한다.
            for alias in tool.aliases:
                self._alias_map.pop(alias.lower(), None)
        return tool

    # ─── 조회 ───

    def get_all_tools(self) -> list[BaseTool]:
        """
        등록된 모든 도구를 반환한다(활성/비활성 여부와 무관).

        정렬:
          이름순으로 정렬해 반환한다. 이유는 모듈 상단 설명 참고 —
          도구 순서를 고정해 prompt cache 무효화를 막기 위함(cache stability).

        Returns:
            이름순으로 정렬된 BaseTool 목록.
        """
        return sorted(self._tools.values(), key=lambda t: t.name)

    def find_tool(self, name: str) -> BaseTool | None:
        """
        이름 또는 별칭(alias)으로 도구 하나를 찾는다.

        조회 순서(먼저 맞는 것을 반환):
          1. 정규 이름 정확 매칭
          2. 별칭 매칭(대소문자 무시)
          3. 둘 다 없으면 None

        Args:
            name: 찾을 도구의 이름 또는 별칭.

        Returns:
            찾은 BaseTool. 없으면 None.
        """
        # 1) 정규 이름으로 정확히 일치하는지 먼저 본다(가장 흔한 경로).
        if name in self._tools:
            return self._tools[name]

        # 2) 별칭으로 조회 — 저장 시 소문자였으므로 조회도 소문자로 맞춘다.
        canonical = self._alias_map.get(name.lower())
        # 별칭이 가리키는 정규 이름이 아직 등록돼 있을 때만 유효.
        if canonical and canonical in self._tools:
            return self._tools[canonical]

        # 3) 어디에도 없으면 None.
        return None

    # ─── 필터링 & 어셈블리 ───

    def get_tools(
        self,
        deny_patterns: list[str] | None = None,
        only_enabled: bool = True,
        only_groups: list[str] | None = None,
    ) -> list[BaseTool]:
        """
        정책 조건으로 걸러낸 도구 목록을 반환한다.

        필터는 아래 순서로 차례대로 적용된다:
          1) only_enabled: 활성화된 도구만 남긴다.
          2) only_groups: 지정한 그룹에 속한 도구만 남긴다.
          3) deny_patterns: 제외 패턴에 걸리는 도구를 뺀다.
        마지막에 항상 이름순으로 정렬한다(cache stability).

        Args:
            deny_patterns: 제외할 도구 이름 패턴 목록(fnmatch 문법 지원).
            only_enabled: True면 비활성(is_enabled=False) 도구를 제외한다.
            only_groups: 이 그룹들에 속한 도구만 포함(None이면 그룹 제한 없음).

        Returns:
            필터링 후 이름순으로 정렬된 BaseTool 목록.
        """
        # 원본 저장소를 건드리지 않도록 값들의 얕은 복사 리스트로 시작한다.
        tools = list(self._tools.values())

        # 1) 활성 도구만 남기기.
        if only_enabled:
            tools = [t for t in tools if t.is_enabled]

        # 2) 지정 그룹에 속한 도구만 남기기.
        if only_groups:
            tools = [t for t in tools if t.group in only_groups]

        # 3) deny 패턴에 걸리는 도구 제거(별도 헬퍼로 위임).
        if deny_patterns:
            tools = self._filter_by_deny_rules(tools, deny_patterns)

        # 이름순 정렬 (cache stability) — 순서 고정으로 캐시를 살린다.
        tools.sort(key=lambda t: t.name)
        return tools

    def assemble_tool_pool(
        self,
        deny_patterns: list[str] | None = None,
        only_enabled: bool = True,
    ) -> tuple[list[BaseTool], list[dict[str, Any]]]:
        """
        모델에 전달할 "도구 풀"을 조립한다.

        get_tools로 조건에 맞는 도구를 고른 뒤, 각 도구의 to_schema()를
        호출해 모델이 이해하는 JSON 스키마 목록을 함께 만든다. 두 리스트는
        같은 순서로 대응된다(같은 인덱스 = 같은 도구).

        Args:
            deny_patterns: 제외할 도구 이름 패턴 목록(fnmatch 문법).
            only_enabled: True면 비활성 도구를 제외한다.

        Returns:
            (tools, schemas) 튜플.
              - tools: 실제 실행에 쓸 BaseTool 객체 목록.
              - schemas: 모델에 넘길 스키마(dict) 목록.
        """
        # 정책에 맞는 도구를 먼저 고르고(이미 이름순 정렬됨),
        tools = self.get_tools(deny_patterns=deny_patterns, only_enabled=only_enabled)
        # 각 도구를 모델용 스키마로 변환한다(순서는 tools와 1:1 대응).
        schemas = [tool.to_schema() for tool in tools]
        return tools, schemas

    @staticmethod
    def _filter_by_deny_rules(
        tools: list[BaseTool],
        patterns: list[str],
    ) -> list[BaseTool]:
        """
        deny 패턴으로 도구를 걸러낸다(내부 헬퍼).

        fnmatch 문법을 지원하므로 다음처럼 유연하게 지정할 수 있다:
          - "Bash"  : 이름이 정확히 Bash인 도구
          - "File*" : File로 시작하는 모든 도구(와일드카드)
          - "*"     : 모든 도구(전체 차단)

        Args:
            tools: 필터링 대상 도구 목록.
            patterns: 제외할 이름 패턴 목록.

        Returns:
            어떤 패턴에도 걸리지 않은 도구만 남긴 목록.
        """
        result = []
        for tool in tools:
            # 패턴들 중 하나라도 매칭되면 이 도구는 차단(denied) 대상이다.
            denied = any(fnmatch.fnmatch(tool.name, p) for p in patterns)
            if not denied:
                # 어떤 패턴에도 안 걸린 경우에만 결과에 포함한다.
                result.append(tool)
            else:
                # 어떤 도구가 왜 빠졌는지 추적할 수 있게 디버그 로그를 남긴다.
                logger.debug(f"도구 '{tool.name}' deny 패턴에 의해 제외됨")
        return result

    # ─── 통계 ───

    @property
    def tool_count(self) -> int:
        """등록된 도구 개수(별칭 제외, 정규 도구 기준)."""
        return len(self._tools)

    def __len__(self) -> int:
        """len(registry)로도 도구 개수를 얻을 수 있게 한다."""
        return len(self._tools)

    def summary(self) -> str:
        """
        레지스트리 상태를 사람이 읽기 쉬운 여러 줄 문자열로 요약한다.

        구성:
          - 첫 줄: 전체 도구 개수.
          - 이후 줄: 그룹별로 묶어 해당 그룹에 속한 도구 이름을 나열.
        그룹명과 도구 이름 모두 정렬해 출력이 항상 일정하게 보이도록 한다.

        Returns:
            요약 텍스트(줄바꿈으로 연결된 문자열).
        """
        # 그룹 이름 → 그 그룹에 속한 도구 이름 목록으로 모은다.
        groups: dict[str, list[str]] = {}
        for t in self._tools.values():
            # setdefault로 그룹 키가 없으면 빈 리스트를 만들고 이름을 추가.
            groups.setdefault(t.group, []).append(t.name)
        lines = [f"Tool Registry: {len(self._tools)}개 도구"]
        # 그룹은 그룹명 순, 각 그룹 안의 도구는 이름순으로 정렬해 출력.
        for group, names in sorted(groups.items()):
            lines.append(f"  [{group}]: {', '.join(sorted(names))}")
        return "\n".join(lines)
