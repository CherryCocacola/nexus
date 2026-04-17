"""
턴 상태 외부화 — raw messages 누적 대신 턴 요약을 저장한다.

v7.0 핵심 변경: 기존 query_loop은 messages[]를 매 턴 누적하여
8,192 토큰 환경에서 2~3턴 만에 컨텍스트가 포화된다.
TurnState는 각 턴의 핵심 정보만 추출하여 외부에 저장하고,
다음 턴에는 raw messages 대신 이 요약만 컨텍스트에 넣는다.

왜 이 방식인가:
  - messages 10개 누적 → ~3,000토큰 소비
  - TurnState 요약 10턴분 → ~300~500토큰으로 동일 맥락 유지
  - TIER_S(8K)에서 수십 턴 대화가 가능해진다
  - TIER_M/L에서는 메타데이터 용도로만 사용 (기존 messages 누적 유지)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("nexus.orchestrator.turn_state")


@dataclass(frozen=True)
class TurnState:
    """
    한 턴의 핵심 정보를 요약한 불변 객체.

    raw messages 대신 이 요약을 다음 턴에 전달한다.
    frozen=True — 생성 후 수정 불가 (StreamEvent과 동일 원칙).
    """

    # 확인된 사실 (파일 존재, 내용 요약 등)
    facts: tuple[str, ...] = ()

    # 남은 할 일
    todo: tuple[str, ...] = ()

    # 접근한 파일 목록
    touched_files: tuple[str, ...] = ()

    # 미해결 문제
    unresolved_issues: tuple[str, ...] = ()

    # 직전 도구 실행 결과 요약
    last_tool_results: tuple[str, ...] = ()

    # Scout의 계획 (TIER_S에서만 사용)
    scout_plan: str | None = None

    # 턴 번호
    turn_number: int = 0

    # 사용자 원래 요청 (첫 턴에서 기록)
    user_request: str = ""

    def to_context_string(self) -> str:
        """
        TurnState를 컨텍스트 문자열로 변환한다.
        이 문자열이 시스템 프롬프트에 주입된다.

        왜 문자열인가: 모델은 JSON보다 자연어를 더 잘 이해하며,
        구조화된 자연어가 토큰 대비 정보 밀도가 높다.
        """
        parts: list[str] = []

        if self.facts:
            parts.append("Facts:\n" + "\n".join(f"- {f}" for f in self.facts))
        if self.todo:
            parts.append("TODO:\n" + "\n".join(f"- {t}" for t in self.todo))
        if self.touched_files:
            parts.append("Files: " + ", ".join(self.touched_files))
        if self.unresolved_issues:
            parts.append("Issues: " + ", ".join(self.unresolved_issues))
        if self.last_tool_results:
            parts.append(
                "Last results:\n" + "\n".join(f"- {r}" for r in self.last_tool_results)
            )
        if self.scout_plan:
            parts.append(f"Scout plan: {self.scout_plan}")

        return "\n\n".join(parts)

    def estimated_tokens(self) -> int:
        """요약의 토큰 수를 추정한다 (문자수 / 3, 보수적)."""
        return len(self.to_context_string()) // 3


class TurnStateStore:
    """
    TurnState를 세션 단위로 저장/조회한다.

    인메모리 딕셔너리로 구현한다.
    Redis 확장이 필요하면 이 클래스만 교체하면 된다.
    """

    def __init__(self) -> None:
        # session_id → 턴별 상태 리스트
        self._states: dict[str, list[TurnState]] = {}

    def save(self, session_id: str, state: TurnState) -> None:
        """턴 상태를 저장한다."""
        if session_id not in self._states:
            self._states[session_id] = []
        self._states[session_id].append(state)
        logger.debug(
            "TurnState 저장: session=%s, turn=%d, facts=%d, todo=%d",
            session_id,
            state.turn_number,
            len(state.facts),
            len(state.todo),
        )

    def get_latest(self, session_id: str) -> TurnState | None:
        """가장 최근 턴 상태를 반환한다."""
        states = self._states.get(session_id, [])
        return states[-1] if states else None

    def get_all(self, session_id: str) -> list[TurnState]:
        """세션의 모든 턴 상태를 반환한다 (복사본)."""
        return list(self._states.get(session_id, []))

    def get_context(self, session_id: str, max_tokens: int = 1000) -> str:
        """
        토큰 예산 내에서 최근 턴 상태들을 컨텍스트 문자열로 반환한다.

        최신 것부터 역순으로 예산이 허용하는 만큼 포함한다.
        왜 역순인가: 최근 턴이 가장 중요하고,
        오래된 턴은 컨텍스트에서 빠져도 무방하다.
        """
        states = self._states.get(session_id, [])
        if not states:
            return ""

        result_parts: list[str] = []
        used_tokens = 0

        for state in reversed(states):
            text = state.to_context_string()
            tokens = state.estimated_tokens()
            if used_tokens + tokens > max_tokens:
                break
            result_parts.append(f"[Turn {state.turn_number}]\n{text}")
            used_tokens += tokens

        result_parts.reverse()
        return "\n---\n".join(result_parts)

    def clear(self, session_id: str) -> None:
        """세션의 모든 턴 상태를 삭제한다."""
        self._states.pop(session_id, None)

    @property
    def session_count(self) -> int:
        """저장된 세션 수를 반환한다."""
        return len(self._states)


def extract_turn_state(
    turn_number: int,
    user_request: str,
    assistant_text: str,
    tool_use_blocks: list[dict[str, Any]],
    tool_results: list[str] | None = None,
) -> TurnState:
    """
    턴의 raw 데이터에서 TurnState를 추출한다.

    이 함수는 query_loop의 Phase 3 끝에서 호출된다.
    assistant_text와 tool_use_blocks에서 핵심 정보만 추출하여
    TurnState로 변환한다.

    왜 모델을 호출하지 않는가:
      - 8K 환경에서 요약을 위한 추가 모델 호출은 토큰 낭비
      - 규칙 기반 추출이 빠르고 결정적(deterministic)
      - 도구 호출 결과는 구조화되어 있어 파싱이 쉽다
    """
    facts: list[str] = []
    todo: list[str] = []
    touched_files: list[str] = []
    tool_summaries: list[str] = []

    # 도구 호출에서 정보 추출
    for block in tool_use_blocks:
        tool_name = block.get("name", "")
        tool_input = block.get("input", {})

        # 접근한 파일 기록
        file_path = tool_input.get("file_path") or tool_input.get("path", "")
        if file_path:
            touched_files.append(file_path)

        # 도구별 요약 생성
        if tool_name == "Read":
            facts.append(f"Read {file_path}")
        elif tool_name == "Write":
            facts.append(f"Wrote {file_path}")
        elif tool_name == "Edit":
            facts.append(f"Edited {file_path}")
        elif tool_name == "Bash":
            cmd = tool_input.get("command", "")[:80]
            facts.append(f"Ran: {cmd}")
        elif tool_name in ("Glob", "Grep"):
            pattern = tool_input.get("pattern", "")
            facts.append(f"{tool_name}: {pattern}")
        elif tool_name == "LS":
            facts.append(f"Listed {file_path}")
        elif tool_name in ("GitLog", "GitDiff", "GitStatus"):
            facts.append(f"{tool_name} executed")
        elif tool_name == "DocumentProcess":
            facts.append(f"Parsed document: {file_path}")
        else:
            facts.append(f"Used {tool_name}")

    # 도구 결과 요약
    if tool_results:
        for result in tool_results:
            # 결과가 길면 첫 100자만
            summary = result[:100] + "..." if len(result) > 100 else result
            tool_summaries.append(summary)

    # assistant 텍스트에서 TODO 패턴 추출
    if assistant_text:
        # 텍스트를 줄 단위로 검사하여 할 일 패턴 추출
        for line in assistant_text.split("\n"):
            line_stripped = line.strip()
            # "다음에", "해야", "필요" 등의 패턴을 감지
            if any(
                kw in line_stripped
                for kw in ["TODO", "todo", "다음에", "해야", "필요", "should", "need to"]
            ):
                if len(line_stripped) > 10:
                    todo.append(line_stripped[:100])

        # 짧은 응답 요약을 fact로 추가
        if len(assistant_text) < 200:
            facts.append(f"Response: {assistant_text[:100]}")

    return TurnState(
        facts=tuple(facts),
        todo=tuple(todo),
        touched_files=tuple(dict.fromkeys(touched_files)),  # 중복 제거, 순서 유지
        last_tool_results=tuple(tool_summaries[:5]),  # 최대 5개
        turn_number=turn_number,
        user_request=user_request[:200],  # 최대 200자
    )
