"""
Memory 도구 모음 — 에이전트의 "기억"을 검색하고 저장하는 도구 2종.

이 파일은 LLM 에이전트가 자신의 장기/단기 기억을 다루기 위해 호출하는
두 개의 도구(Tool)를 정의한다. 둘 다 BaseTool ABC를 상속하며,
Nexus의 표준 도구 수명주기(validate_input → check_permissions → call)를 따른다.

제공하는 도구:
  - MemoryReadTool  (도구 이름: "MemoryRead")
      기억에서 관련 정보를 찾아 읽는다. 읽기 전용이라 병렬 실행이 안전하다.
  - MemoryWriteTool (도구 이름: "MemoryWrite")
      새 정보를 기억에 저장한다. 쓰기 작업이라 실행 전 사용자 확인을 받는다.

Nexus 메모리 시스템은 2계층 구조다:
  - 단기 메모리: Redis        — 세션 안에서 빠르게 읽고 쓰는 임시 기억
  - 장기 메모리: PostgreSQL + pgvector — 의미(벡터) 유사도 기반 검색이 가능한 영구 기억

동작 방식(중요):
  실제 저장/검색 로직을 담당하는 MemoryManager는 Phase 5.0 산출물이라,
  런타임에 주입될 수도 있고 아직 없을 수도 있다. 그래서 두 도구 모두
  context.options.get("memory_manager")로 매니저를 먼저 찾아보고,
  없으면 이 모듈 안의 간단한 인메모리 딕셔너리(_fallback_memory)로 폴백한다.
  폴백 저장소는 프로세스 메모리에만 존재하므로 프로세스가 끝나면 사라진다.

의존:
  core.tools.base 의 BaseTool / PermissionBehavior / PermissionResult /
  ToolResult / ToolUseContext 만 사용한다. (MemoryManager는 직접 import하지 않고
  런타임 context를 통해 느슨하게 연결해, 매니저 미구현 상태에서도 동작하게 한다.)

작성자: 이현수 / 작성일: 2026-07-05
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

# 이 모듈 전용 로거. 규칙상 "nexus.{module}" 네임스페이스를 사용한다.
logger = logging.getLogger("nexus.tools.memory")

# 인메모리 폴백 저장소 — 진짜 MemoryManager가 주입되지 않았을 때만 사용한다.
# 모듈 레벨 전역 딕셔너리라 프로세스가 살아있는 동안만 유지되고,
# 프로세스를 종료하면 내용이 전부 사라진다(영속성 없음).
# 구조: 키 = memory_id(str), 값 = {"content": str, "tags": list, "created_at": float}
_fallback_memory: dict[str, dict[str, Any]] = {}

# 폴백 저장소에서 memory_id를 만들 때 쓰는 단조 증가 카운터.
# 저장할 때마다 1씩 올려 "mem-0001", "mem-0002" 형태의 ID를 만든다.
_memory_counter = 0


def _get_memory_manager(context: ToolUseContext) -> Any:
    """
    현재 실행 컨텍스트에서 MemoryManager 인스턴스를 꺼내는 헬퍼.

    두 도구가 공통으로 쓰는 조회 지점이다. 도구는 MemoryManager를 직접
    import하지 않고, 오케스트레이터가 context.options에 넣어 준 것을 그대로 쓴다.
    이렇게 하면 매니저가 아직 구현/주입되지 않아도 도구가 죽지 않는다.

    매개변수:
        context: 현재 도구 호출의 실행 컨텍스트(옵션 딕셔너리를 포함).

    반환:
        주입돼 있으면 MemoryManager 객체, 없으면 None.
        None이면 호출부에서 인메모리 폴백 경로를 타게 된다.
    """
    return context.options.get("memory_manager")


# ─────────────────────────────────────────────
# MemoryReadTool — 메모리 검색
# ─────────────────────────────────────────────
class MemoryReadTool(BaseTool):
    """
    메모리에서 관련 정보를 찾아 읽어 오는 읽기 전용 도구("MemoryRead").

    역할:
        에이전트가 과거에 저장해 둔 기억 중 지금 질의(query)와 관련된 것을
        찾아 텍스트로 돌려준다. 태그(tags)로 범위를 좁힐 수도 있다.

    검색 방식(우선순위):
        1) MemoryManager가 주입돼 있으면 → 벡터 유사도 검색(pgvector) 위임.
           의미가 비슷한 기억까지 찾아 주므로 가장 좋은 경로다.
        2) 매니저가 없거나 검색이 실패하면 → 인메모리 폴백에서 단순 키워드 매칭.

    동작 특성:
        읽기 전용(is_read_only=True)이라 상태를 바꾸지 않는다. 따라서
        동시 실행이 안전(is_concurrency_safe=True)해 다른 읽기 도구와 병렬 실행된다.
    """

    # ═══ 1. Identity ═══

    @property
    def name(self) -> str:
        # 레지스트리에 등록되고 모델이 호출할 때 쓰는 도구의 고유 이름.
        return "MemoryRead"

    @property
    def description(self) -> str:
        # 모델(LLM)에게 이 도구가 무엇을 하는지 알려 주는 설명문.
        # 모델은 이 문장을 보고 언제 도구를 호출할지 판단하므로 명확히 적는다.
        return (
            "메모리에서 관련 정보를 검색합니다. "
            "키워드 검색, 태그 필터링을 지원하며, "
            "MemoryManager 연동 시 의미(semantic) 검색도 가능합니다."
        )

    @property
    def group(self) -> str:
        # 도구 분류용 그룹명. 메모리 관련 도구는 "memory" 그룹으로 묶는다.
        return "memory"

    # ═══ 2. Schema ═══

    @property
    def input_schema(self) -> dict[str, Any]:
        # 모델이 넘겨야 하는 입력의 JSON Schema.
        # query는 필수, tags/max_results는 선택이다. max_results는 1~50으로 제한한다.
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
        # 상태를 바꾸지 않는 검색 전용 도구라 읽기 전용으로 명시한다.
        return True

    @property
    def is_concurrency_safe(self) -> bool:
        # 읽기만 하므로 다른 읽기 도구와 동시에 실행해도 안전하다.
        return True

    # ═══ 5. Lifecycle ═══

    def validate_input(self, input_data: dict[str, Any]) -> str | None:
        """
        도구 실행 전에 입력을 미리 검증한다(수명주기 1단계).

        query가 없거나 공백뿐이면 검색이 무의미하므로 오류 메시지를 돌려준다.
        문제가 없으면 None을 반환해 다음 단계로 넘어가게 한다.

        반환:
            검증 실패 시 사용자에게 보여줄 오류 문자열, 통과 시 None.
        """
        query = input_data.get("query", "")
        if not query or not query.strip():
            return "query는 비어 있을 수 없습니다."
        return None

    async def check_permissions(
        self, input_data: dict[str, Any], context: ToolUseContext
    ) -> PermissionResult:
        """
        권한 검사(수명주기 2단계). 이 도구는 읽기 전용이라 무조건 허용한다.

        읽기만 하므로 사용자 확인 없이 바로 실행해도 안전하다.
        따라서 항상 ALLOW를 반환한다.
        """
        return PermissionResult(behavior=PermissionBehavior.ALLOW)

    async def call(self, input_data: dict[str, Any], context: ToolUseContext) -> ToolResult:
        """
        실제 검색을 수행한다(수명주기 3단계, 도구의 핵심 로직).

        처리 순서:
          1. MemoryManager가 있으면 벡터 유사도 검색에 위임(가장 좋은 경로).
          2. 매니저가 없거나 검색 도중 예외가 나면 인메모리 폴백으로 넘어간다.
          3. 폴백에서는 태그 필터 + 단순 키워드 매칭으로 후보를 고른다.
          4. 최신순으로 정렬하고 max_results개만 잘라 사람이 읽기 좋게 포맷해 반환.

        매개변수:
            input_data: query(필수), tags(선택), max_results(선택) 를 담은 딕셔너리.
            context: MemoryManager 조회 등에 쓰는 실행 컨텍스트.

        반환:
            검색 결과를 담은 ToolResult.success(문자열 + count 메타데이터).
            결과가 없을 때도 성공으로 처리하되 count=0으로 알린다.
        """
        # 입력 값 추출. query는 필수라 [] 없이 바로 꺼내고, 나머지는 기본값을 준다.
        query = input_data["query"]
        tags = input_data.get("tags", [])
        max_results = input_data.get("max_results", 10)

        # 먼저 진짜 매니저가 주입돼 있는지 확인한다.
        manager = _get_memory_manager(context)

        # ── 경로 1: MemoryManager 위임 ──
        # 매니저가 있고 search_relevant 메서드를 지원하면 벡터+텍스트 검색을 맡긴다.
        if manager and hasattr(manager, "search_relevant"):
            try:
                # top_k만큼 유사도 상위 기억을 받아 온다.
                entries = await manager.search_relevant(query=query, top_k=max_results)
                if not entries:
                    # 결과가 없어도 오류가 아니라 "검색 결과 없음"으로 정상 반환한다.
                    return ToolResult.success(f"'{query}'에 대한 검색 결과가 없습니다.", count=0)
                # 반환된 MemoryEntry들을 한 줄씩 사람이 읽기 좋은 문자열로 만든다.
                lines = []
                for entry in entries:
                    # tags 속성이 없거나 None일 수 있어 안전하게 기본 []로 처리한다.
                    entry_tags = getattr(entry, "tags", []) or []
                    tags_str = ", ".join(entry_tags) if entry_tags else "없음"
                    lines.append(f"[{entry.id}] (태그: {tags_str})\n  {entry.content}")
                return ToolResult.success("\n\n".join(lines), count=len(entries))
            except Exception as e:
                # 매니저 검색이 깨져도 도구 전체를 실패시키지 않고 폴백으로 이어간다.
                logger.warning("MemoryManager 검색 실패, 폴백 사용: %s", e)

        # ── 경로 2: 인메모리 폴백(단순 키워드 매칭) ──
        # 대소문자 구분 없이 비교하려고 쿼리를 소문자로 만들어 둔다.
        query_lower = query.lower()
        matches = []

        # 폴백 저장소를 전부 훑으며 조건에 맞는 기억을 모은다.
        for mem_id, mem in _fallback_memory.items():
            content = mem.get("content", "")
            mem_tags = mem.get("tags", [])

            # 태그 필터: 요청한 태그가 하나라도 빠지면 제외(모두 포함해야 통과).
            if tags and not all(t in mem_tags for t in tags):
                continue

            # 키워드 매칭: 쿼리 문자열이 content 안에 부분 문자열로 있으면 후보로 채택.
            if query_lower in content.lower():
                matches.append((mem_id, mem))

        if not matches:
            # 폴백에서도 못 찾으면 결과 없음으로 반환한다.
            return ToolResult.success(f"'{query}'에 대한 검색 결과가 없습니다.", count=0)

        # created_at 기준 내림차순 정렬 → 가장 최근에 저장한 기억이 위로 온다.
        matches.sort(key=lambda x: x[1].get("created_at", 0), reverse=True)
        # 요청한 최대 개수만큼만 남긴다.
        matches = matches[:max_results]

        # 최종 결과를 MemoryManager 경로와 같은 형태의 문자열로 포맷한다.
        lines = []
        for mem_id, mem in matches:
            tags_str = ", ".join(mem.get("tags", [])) if mem.get("tags") else "없음"
            lines.append(f"[{mem_id}] (태그: {tags_str})\n  {mem['content']}")

        logger.debug("MemoryRead: query='%s', %d results", query, len(matches))
        return ToolResult.success("\n\n".join(lines), count=len(matches))

    # ═══ 7. UI Hints ═══

    def get_progress_label(self, input_data: dict[str, Any]) -> str:
        # 도구 실행 중 UI에 표시할 진행 문구.
        return "Searching memory..."

    def get_input_summary(self, input_data: dict[str, Any]) -> str:
        # UI에서 이 호출을 요약해 보여줄 때 쓰는 짧은 문자열(query 앞 80자).
        return input_data.get("query", "")[:80]


# ─────────────────────────────────────────────
# MemoryWriteTool — 메모리 저장
# ─────────────────────────────────────────────
class MemoryWriteTool(BaseTool):
    """
    새 정보를 메모리에 저장하는 쓰기 도구("MemoryWrite").

    역할:
        에이전트가 나중에 다시 참고할 내용을 기억에 남긴다.
        텍스트 본문(content)과 분류용 태그(tags)를 함께 저장할 수 있다.

    저장 방식(우선순위):
        1) MemoryManager가 있으면 add_semantic으로 장기 메모리에 저장하고,
           이때 벡터 임베딩까지 자동 생성돼 나중에 의미 검색이 가능해진다.
        2) 매니저가 없거나 저장이 실패하면 인메모리 폴백에 넣는다(휘발성).

    주의:
        쓰기 작업이라 fail-closed 원칙에 따라 실행 전 사용자 확인(ASK)을 받는다.
        (behavior flag를 별도로 완화하지 않아 기본값인 쓰기/순차/미확인 규칙을 따른다.)
    """

    # ═══ 1. Identity ═══

    @property
    def name(self) -> str:
        # 레지스트리 등록 및 모델 호출 시 사용하는 도구 고유 이름.
        return "MemoryWrite"

    @property
    def description(self) -> str:
        # 모델에게 이 도구의 기능을 알려 주는 설명문.
        return (
            "새 정보를 메모리에 저장합니다. "
            "태그를 지정하여 분류할 수 있으며, "
            "MemoryManager 연동 시 벡터 임베딩도 자동 생성됩니다."
        )

    @property
    def group(self) -> str:
        # 메모리 관련 도구 그룹.
        return "memory"

    # ═══ 2. Schema ═══

    @property
    def input_schema(self) -> dict[str, Any]:
        # 입력 JSON Schema. content는 필수, tags는 선택(문자열 배열)이다.
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
    # 여기서는 behavior flag를 재정의하지 않는다. BaseTool의 fail-closed 기본값
    # (is_read_only=False, is_concurrency_safe=False)을 그대로 물려받아,
    # 쓰기 도구답게 순차 실행 + 확인 요청 방식으로 동작하도록 둔다.

    # ═══ 5. Lifecycle ═══

    def validate_input(self, input_data: dict[str, Any]) -> str | None:
        """
        저장 전에 입력을 검증한다(수명주기 1단계).

        content가 없거나 공백뿐이면 저장할 내용이 없으므로 오류를 반환한다.
        정상이면 None을 반환해 다음 단계로 넘어간다.
        """
        content = input_data.get("content", "")
        if not content or not content.strip():
            return "content는 비어 있을 수 없습니다."
        return None

    async def check_permissions(
        self, input_data: dict[str, Any], context: ToolUseContext
    ) -> PermissionResult:
        """
        권한 검사(수명주기 2단계). 쓰기 도구라 실행 전 사용자 확인을 받는다.

        저장될 내용을 사용자가 미리 알 수 있도록 앞부분 미리보기(preview)를
        만들어 확인 메시지에 담는다. 50자를 넘으면 뒤를 "..."로 줄인다.

        반환:
            behavior=ASK 인 PermissionResult(사용자 승인 후에만 call이 실행됨).
        """
        content = input_data.get("content", "")
        preview = content[:50] + "..." if len(content) > 50 else content
        return PermissionResult(
            behavior=PermissionBehavior.ASK,
            message=f"메모리 저장: {preview}",
        )

    async def call(self, input_data: dict[str, Any], context: ToolUseContext) -> ToolResult:
        """
        메모리에 새 정보를 실제로 저장한다(수명주기 3단계, 핵심 로직).

        처리 순서:
          1. MemoryManager가 있으면 add_semantic으로 위임(벡터 임베딩 자동 생성).
          2. 매니저가 없거나 저장 중 예외가 나면 인메모리 폴백에 저장.
          3. 어느 경로든 저장된 메모리 ID를 결과에 담아 반환.

        매개변수:
            input_data: content(필수) 와 tags(선택) 를 담은 딕셔너리.
            context: MemoryManager 조회에 쓰는 실행 컨텍스트.

        반환:
            저장 성공 메시지와 memory_id 메타데이터를 담은 ToolResult.success.
        """
        # 입력 값 추출. content는 필수라 바로 꺼내고, tags는 없으면 빈 리스트.
        content = input_data["content"]
        tags = input_data.get("tags", [])

        # 진짜 매니저가 주입돼 있는지 확인.
        manager = _get_memory_manager(context)

        # ── 경로 1: MemoryManager 위임(시맨틱 메모리, 벡터 임베딩 포함) ──
        if manager and hasattr(manager, "add_semantic"):
            try:
                # key는 현재 시각(epoch 초)으로 고유하게 만들고, value에 본문을 넣는다.
                # tags가 비어 있으면 None으로 넘겨 매니저 쪽 기본 처리에 맡긴다.
                mem_id = await manager.add_semantic(
                    key=f"user_stored_{int(time.time())}",
                    value=content,
                    tags=tags or None,
                )
                logger.info("MemoryWrite: stored via MemoryManager, id=%s", mem_id)
                tags_str = ", ".join(tags) if tags else "없음"
                return ToolResult.success(
                    f"메모리를 저장했습니다. (ID: {mem_id}, 태그: {tags_str})",
                    memory_id=mem_id,
                )
            except Exception as e:
                # 매니저 저장이 실패해도 도구를 실패시키지 않고 폴백으로 이어간다.
                logger.warning("MemoryManager 저장 실패, 폴백 사용: %s", e)

        # ── 경로 2: 인메모리 폴백 저장(휘발성) ──
        # 모듈 레벨 카운터를 1 올려 새 ID를 만든다. noqa는 전역 변수 수정 경고를
        # 의도적으로 허용한다는 표시다(간단한 폴백 용도라 전역 카운터로 충분).
        global _memory_counter  # noqa: PLW0603 — 모듈 레벨 카운터, 간단한 폴백용
        _memory_counter += 1
        mem_id = f"mem-{_memory_counter:04d}"

        # 본문/태그/저장 시각을 딕셔너리로 묶어 폴백 저장소에 넣는다.
        _fallback_memory[mem_id] = {
            "content": content,
            "tags": tags,
            "created_at": time.time(),
        }

        tags_str = ", ".join(tags) if tags else "없음"
        logger.info("MemoryWrite: stored in fallback, id=%s, tags=%s", mem_id, tags_str)
        # 폴백은 프로세스 종료 시 사라지므로, 사용자에게 휘발성이라는 점을 함께 알린다.
        return ToolResult.success(
            f"메모리를 저장했습니다. (ID: {mem_id}, 태그: {tags_str})\n"
            f"참고: 인메모리 폴백 모드입니다. 프로세스 종료 시 데이터가 사라집니다.",
            memory_id=mem_id,
        )

    # ═══ 7. UI Hints ═══

    def get_progress_label(self, input_data: dict[str, Any]) -> str:
        # 도구 실행 중 UI에 표시할 진행 문구.
        return "Saving to memory..."

    def get_input_summary(self, input_data: dict[str, Any]) -> str:
        # UI 요약용 짧은 문자열(저장할 content 앞 80자).
        return input_data.get("content", "")[:80]
