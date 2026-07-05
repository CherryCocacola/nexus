"""
턴 상태 외부화(externalization) — raw messages 누적 대신 "턴 요약"을 저장한다.

[이 파일이 하는 일]
에이전트가 사용자와 여러 턴(turn)에 걸쳐 대화할 때, 기존 방식은 매 턴의
원본 메시지(raw messages)를 계속 배열에 쌓아 다음 턴 컨텍스트로 넘긴다.
이 파일은 그 대신 각 턴에서 "핵심 정보만" 뽑아 작은 요약 객체(TurnState)로
저장하고, 다음 턴에는 원본 대신 이 요약만 컨텍스트에 주입한다.
그 결과 적은 토큰으로도 대화 맥락을 오래 유지할 수 있다.

[v7.0 핵심 변경 배경]
기존 query_loop은 messages[]를 매 턴 누적하기 때문에, 8,192 토큰짜리
작은 컨텍스트 환경에서는 2~3턴 만에 컨텍스트가 포화(overflow)된다.
TurnState는 각 턴의 핵심만 추출·외부 저장하여 이 포화 문제를 완화한다.

[왜 이 방식이 이득인가]
  - raw messages 10개 누적 → 대략 3,000토큰 소비
  - TurnState 요약 10턴분 → 대략 300~500토큰으로 동일 맥락 유지
  - TIER_S(8K 컨텍스트)에서도 수십 턴 대화가 가능해진다
  - TIER_M / TIER_L(더 큰 컨텍스트)에서는 요약을 "메타데이터" 용도로만
    쓰고, 기존 messages 누적 방식을 그대로 유지한다

[주요 구성 요소]
  - TurnState        : 한 턴의 요약을 담는 불변(frozen) 데이터 객체
  - TurnStateStore   : 세션 단위로 TurnState들을 저장/조회하는 저장소
  - extract_turn_state: 턴의 원본 데이터에서 TurnState를 규칙 기반으로 추출

[호출 관계]
extract_turn_state()는 query_loop의 Phase 3 마지막에서 호출되고,
반환된 TurnState는 TurnStateStore.save()로 세션에 적재된다.
다음 턴 시작 시 TurnStateStore.get_context()가 시스템 프롬프트에
주입할 요약 문자열을 만들어 준다.

작성자: 이현수 / 작성일: 2026-07-05
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

# 이 모듈 전용 로거. 로그를 볼 때 "nexus.orchestrator.turn_state" 네임스페이스로
# 필터링하면 턴 상태 저장 흐름만 골라 확인할 수 있다.
logger = logging.getLogger("nexus.orchestrator.turn_state")


@dataclass(frozen=True)
class TurnState:
    """
    한 턴(turn)의 핵심 정보를 요약해 담는 불변(immutable) 객체.

    [역할] 원본 메시지(raw messages) 전체 대신, 다음 턴에 넘길 "요약본"이다.
    필드는 모두 tuple(불변 시퀀스)로 정의되어 있어, 생성 후 내용이 바뀌지
    않도록 강제한다.

    [왜 frozen=True인가]
    StreamEvent과 동일한 원칙으로, 한번 만든 턴 요약은 이후 아무도 수정하지
    못하게 잠근다. 이렇게 하면 여러 곳에서 같은 객체를 공유해도 부작용
    (side effect)으로 값이 바뀔 위험이 없다.

    [필드 개요]
    facts / todo / touched_files / unresolved_issues / last_tool_results는
    각각 "확인된 사실 / 남은 할 일 / 접근한 파일 / 미해결 문제 / 직전 도구
    결과"를 담는다. scout_plan은 TIER_S 전용 계획 문자열, turn_number는 턴
    번호, user_request는 사용자의 원래 요청이다.
    """

    # 이번 턴에서 확인된 사실들 (예: 파일이 존재함, 내용 요약, 실행한 명령 등).
    facts: tuple[str, ...] = ()

    # 아직 처리하지 못해 다음 턴으로 넘길 "남은 할 일" 목록.
    todo: tuple[str, ...] = ()

    # 이번 턴에서 읽거나 쓴 파일 경로 목록 (중복 없이 접근 순서 유지).
    touched_files: tuple[str, ...] = ()

    # 아직 해결되지 못한 문제/에러 등을 기록해 다음 턴이 이어받게 한다.
    unresolved_issues: tuple[str, ...] = ()

    # 직전 도구(tool) 실행 결과의 짧은 요약 (원문이 길면 잘라서 보관).
    last_tool_results: tuple[str, ...] = ()

    # Scout(사전 정찰 단계)가 세운 계획. 작은 컨텍스트인 TIER_S에서만 사용한다.
    scout_plan: str | None = None

    # 몇 번째 턴인지 나타내는 번호 (0부터 시작).
    turn_number: int = 0

    # 사용자가 처음에 요청한 원문 (첫 턴에서 기록해 이후 맥락 유지에 사용).
    user_request: str = ""

    def to_context_string(self) -> str:
        """
        TurnState를 시스템 프롬프트에 넣을 "컨텍스트 문자열"로 변환한다.

        [흐름] 값이 채워진 필드만 골라 사람이 읽기 좋은 섹션(Facts / TODO /
        Files / Issues / Last results / Scout plan)으로 조립한 뒤, 섹션 사이를
        빈 줄로 구분해 하나의 문자열로 이어 붙인다.

        [왜 JSON이 아니라 문자열인가]
        모델은 JSON 구조보다 자연어를 더 잘 이해하며, 잘 구조화된 자연어가
        토큰 대비 정보 밀도(정보량 / 토큰 수)가 더 높기 때문이다.

        반환값: 컨텍스트에 주입할 요약 문자열. 채워진 필드가 없으면 빈 문자열.
        """
        # 각 섹션 문자열을 순서대로 담아 두었다가 마지막에 합친다.
        parts: list[str] = []

        # 확인된 사실이 있으면 "Facts:" 섹션으로, 항목마다 "- " 불릿을 붙인다.
        if self.facts:
            parts.append("Facts:\n" + "\n".join(f"- {f}" for f in self.facts))
        # 남은 할 일이 있으면 "TODO:" 섹션으로 동일하게 불릿 나열한다.
        if self.todo:
            parts.append("TODO:\n" + "\n".join(f"- {t}" for t in self.todo))
        # 접근한 파일은 한 줄에 콤마로 이어 간결하게 표기한다.
        if self.touched_files:
            parts.append("Files: " + ", ".join(self.touched_files))
        # 미해결 문제도 한 줄로 콤마 구분해 표기한다.
        if self.unresolved_issues:
            parts.append("Issues: " + ", ".join(self.unresolved_issues))
        # 직전 도구 결과 요약은 다시 불릿 목록으로 펼친다.
        if self.last_tool_results:
            parts.append(
                "Last results:\n" + "\n".join(f"- {r}" for r in self.last_tool_results)
            )
        # Scout 계획이 있으면 한 줄로 덧붙인다 (TIER_S 전용).
        if self.scout_plan:
            parts.append(f"Scout plan: {self.scout_plan}")

        # 섹션들 사이를 빈 줄(\n\n)로 띄워 가독성을 높인 최종 문자열을 만든다.
        return "\n\n".join(parts)

    def estimated_tokens(self) -> int:
        """
        이 요약이 대략 몇 토큰인지 추정한다.

        정확한 토크나이저를 돌리지 않고, "문자 수 // 3"이라는 간단한 근사식을
        쓴다. 3으로 나누는 것은 토큰을 넉넉히 잡는 보수적(conservative) 추정으로,
        토큰 예산을 초과하지 않도록 안전하게 판단하기 위함이다.
        """
        return len(self.to_context_string()) // 3


class TurnStateStore:
    """
    TurnState들을 "세션(session) 단위"로 저장하고 조회하는 저장소.

    [역할] 세션 ID별로 그동안 쌓인 TurnState 리스트를 관리한다. 다음 턴에
    넘길 맥락 요약을 만들거나, 최근 상태를 꺼내오는 창구 역할을 한다.

    [구현 메모] 현재는 단순한 인메모리 딕셔너리로 구현되어 있다. 나중에
    프로세스 재시작 후에도 유지되는 저장이 필요하면(예: Redis) 이 클래스의
    내부 구현만 교체하면 되고, 바깥에서 쓰는 메서드 시그니처는 그대로 둘 수
    있도록 설계했다.
    """

    def __init__(self) -> None:
        # 핵심 자료구조: session_id 문자열을 키로, 해당 세션의 TurnState들을
        # 시간순(오래된 → 최신)으로 담은 리스트를 값으로 가진다.
        self._states: dict[str, list[TurnState]] = {}

    def save(self, session_id: str, state: TurnState) -> None:
        """
        한 턴의 요약(state)을 지정한 세션에 추가 저장한다.

        해당 세션이 처음이면 빈 리스트를 먼저 만든 뒤 뒤에 덧붙인다. 저장
        직후에는 디버그 로그로 어떤 세션의 몇 번째 턴이 몇 개의 fact/todo를
        담았는지 남겨, 이후 문제 추적을 돕는다.
        """
        # 이 세션의 첫 저장이면 빈 리스트로 초기화한다.
        if session_id not in self._states:
            self._states[session_id] = []
        # 시간순으로 가장 뒤에 이번 턴 상태를 덧붙인다.
        self._states[session_id].append(state)
        # 저장 결과를 디버그 로그로 남긴다 (운영 시 흐름 추적용).
        logger.debug(
            "TurnState 저장: session=%s, turn=%d, facts=%d, todo=%d",
            session_id,
            state.turn_number,
            len(state.facts),
            len(state.todo),
        )

    def get_latest(self, session_id: str) -> TurnState | None:
        """
        해당 세션에서 가장 최근(마지막) 턴 상태를 돌려준다.

        저장된 상태가 하나도 없으면 None을 반환한다.
        """
        # 세션이 없으면 빈 리스트를 기본값으로 받아 안전하게 처리한다.
        states = self._states.get(session_id, [])
        # 리스트가 비어 있지 않으면 마지막 원소(최신)를, 비었으면 None을 반환.
        return states[-1] if states else None

    def get_all(self, session_id: str) -> list[TurnState]:
        """
        해당 세션의 모든 턴 상태를 시간순 리스트로 돌려준다.

        내부 리스트를 그대로 넘기면 바깥에서 변형될 수 있으므로, list()로
        감싼 "복사본"을 반환해 저장소 원본을 보호한다.
        """
        return list(self._states.get(session_id, []))

    def get_context(self, session_id: str, max_tokens: int = 1000) -> str:
        """
        토큰 예산(max_tokens) 안에서, 최근 턴 요약들을 이어 붙인 컨텍스트 문자열을 만든다.

        [흐름]
          1) 세션의 턴 상태들을 최신 → 과거 방향(역순)으로 훑는다.
          2) 각 턴의 추정 토큰을 누적하다가 예산을 넘기면 멈춘다.
          3) 담긴 조각들을 다시 시간순으로 뒤집어, 구분선(---)으로 이어 반환한다.

        [왜 최신부터 역순으로 담는가]
        가장 최근 턴이 현재 맥락에 제일 중요하고, 오래된 턴은 예산이 모자라면
        컨텍스트에서 빠져도 대화 진행에 큰 지장이 없기 때문이다. 즉 예산이
        부족할 때 "오래된 것부터 버린다".

        반환값: 시스템 프롬프트에 주입할 요약 묶음 문자열 (없으면 빈 문자열).
        """
        # 세션이 없으면 빈 리스트로 받아, 아래에서 곧바로 빈 문자열을 반환한다.
        states = self._states.get(session_id, [])
        if not states:
            return ""

        # 예산 안에 들어온 턴 조각들을 모을 리스트와, 지금까지 쓴 토큰 누계.
        result_parts: list[str] = []
        used_tokens = 0

        # 최신 턴부터 과거로 거슬러 올라가며 예산이 허용하는 만큼 담는다.
        for state in reversed(states):
            text = state.to_context_string()
            tokens = state.estimated_tokens()
            # 이번 턴을 더하면 예산 초과라면, 더 오래된 턴은 볼 필요 없이 중단.
            if used_tokens + tokens > max_tokens:
                break
            # 어느 턴인지 알 수 있게 "[Turn N]" 머리표를 붙여 담는다.
            result_parts.append(f"[Turn {state.turn_number}]\n{text}")
            used_tokens += tokens

        # 역순으로 담았으므로, 다시 뒤집어 실제 시간순(과거 → 최신)으로 복원한다.
        result_parts.reverse()
        # 턴 사이를 "---" 구분선으로 나눠 하나의 문자열로 합쳐 반환한다.
        return "\n---\n".join(result_parts)

    def clear(self, session_id: str) -> None:
        """
        해당 세션에 저장된 모든 턴 상태를 삭제한다.

        pop의 두 번째 인자로 None을 줘, 세션이 없어도 오류 없이 조용히 넘어간다.
        """
        self._states.pop(session_id, None)

    @property
    def session_count(self) -> int:
        """현재 저장소가 관리 중인 세션(대화)의 개수를 돌려준다."""
        return len(self._states)


def extract_turn_state(
    turn_number: int,
    user_request: str,
    assistant_text: str,
    tool_use_blocks: list[dict[str, Any]],
    tool_results: list[str] | None = None,
) -> TurnState:
    """
    한 턴의 원본(raw) 데이터에서 요약본 TurnState를 만들어 낸다.

    [언제 호출되나] query_loop의 Phase 3(턴 마무리) 끝에서 호출된다. 즉
    모델의 응답과 도구 실행이 끝난 뒤, 그 결과를 압축해 저장하기 직전이다.

    [무엇을 하나] 어시스턴트 응답 텍스트(assistant_text)와 도구 호출
    블록들(tool_use_blocks)을 훑어 핵심 정보(접근 파일, 실행 사실, 할 일 등)만
    뽑아 TurnState로 변환한다.

    [왜 요약에 별도 모델 호출을 하지 않는가]
      - 8K 같은 작은 컨텍스트에서 요약 목적의 추가 모델 호출은 토큰 낭비다.
      - 규칙 기반(rule-based) 추출은 빠르고, 같은 입력이면 항상 같은 결과가
        나오는 결정적(deterministic) 동작이라 재현·디버깅이 쉽다.
      - 도구 호출 결과는 이미 구조화되어 있어 규칙만으로도 쉽게 파싱된다.

    매개변수:
      turn_number     : 이번 턴 번호.
      user_request    : 사용자의 원래 요청 (최대 200자까지만 보관).
      assistant_text  : 이번 턴에서 모델이 생성한 응답 텍스트.
      tool_use_blocks : 이번 턴의 도구 호출 목록 (각 원소는 name/input dict).
      tool_results    : 도구 실행 결과 문자열 목록 (없을 수 있어 기본 None).

    반환: 위 정보를 요약해 담은 TurnState 객체.
    """
    # 추출 과정에서 항목들을 모을 임시 리스트들 (마지막에 tuple로 굳힌다).
    facts: list[str] = []
    todo: list[str] = []
    touched_files: list[str] = []
    tool_summaries: list[str] = []

    # ── 1) 도구 호출 블록에서 정보 추출 ──────────────────────────────
    for block in tool_use_blocks:
        # 도구 이름과 입력 인자를 꺼낸다 (키가 없으면 안전한 기본값 사용).
        tool_name = block.get("name", "")
        tool_input = block.get("input", {})

        # 파일 경로는 도구마다 "file_path" 또는 "path" 키로 들어오므로 둘 다 본다.
        file_path = tool_input.get("file_path") or tool_input.get("path", "")
        if file_path:
            touched_files.append(file_path)

        # 도구 종류별로 사람이 읽기 좋은 한 줄 사실(fact)을 만든다.
        if tool_name == "Read":
            facts.append(f"Read {file_path}")
        elif tool_name == "Write":
            facts.append(f"Wrote {file_path}")
        elif tool_name == "Edit":
            facts.append(f"Edited {file_path}")
        elif tool_name == "Bash":
            # 명령어가 길 수 있으니 앞 80자만 잘라 요약에 담는다.
            cmd = tool_input.get("command", "")[:80]
            facts.append(f"Ran: {cmd}")
        elif tool_name in ("Glob", "Grep"):
            # 검색 계열 도구는 검색 패턴을 기록한다.
            pattern = tool_input.get("pattern", "")
            facts.append(f"{tool_name}: {pattern}")
        elif tool_name == "LS":
            facts.append(f"Listed {file_path}")
        elif tool_name in ("GitLog", "GitDiff", "GitStatus"):
            # Git 조회류는 실행됐다는 사실만 남긴다.
            facts.append(f"{tool_name} executed")
        elif tool_name == "DocumentProcess":
            facts.append(f"Parsed document: {file_path}")
        else:
            # 위에서 다루지 않은 그 밖의 도구는 이름만 일반적으로 기록한다.
            facts.append(f"Used {tool_name}")

    # ── 2) 도구 실행 결과 요약 ───────────────────────────────────────
    if tool_results:
        for result in tool_results:
            # 결과 원문이 길면 앞 100자만 남기고 "..."을 붙여 축약한다.
            summary = result[:100] + "..." if len(result) > 100 else result
            tool_summaries.append(summary)

    # ── 3) 어시스턴트 텍스트에서 TODO(할 일) 패턴 추출 ───────────────
    if assistant_text:
        # 응답을 줄 단위로 검사하며 "할 일"을 암시하는 키워드가 있는지 본다.
        for line in assistant_text.split("\n"):
            line_stripped = line.strip()
            # 한/영 혼용 키워드로 다음 턴에 이어질 작업 신호를 감지한다.
            if any(
                kw in line_stripped
                for kw in ["TODO", "todo", "다음에", "해야", "필요", "should", "need to"]
            ):
                # 너무 짧은(의미 없는) 줄은 제외하고, 길면 100자까지만 담는다.
                if len(line_stripped) > 10:
                    todo.append(line_stripped[:100])

        # 응답이 짧으면(200자 미만) 그 자체를 사실(fact)로도 요약해 남긴다.
        if len(assistant_text) < 200:
            facts.append(f"Response: {assistant_text[:100]}")

    # ── 4) 수집한 정보로 최종 TurnState를 조립해 반환 ───────────────
    return TurnState(
        facts=tuple(facts),
        todo=tuple(todo),
        touched_files=tuple(dict.fromkeys(touched_files)),  # 중복 제거, 순서 유지
        last_tool_results=tuple(tool_summaries[:5]),  # 최대 5개까지만 보관
        turn_number=turn_number,
        user_request=user_request[:200],  # 최대 200자까지만 보관
    )
