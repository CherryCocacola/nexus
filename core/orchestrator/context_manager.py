"""
컨텍스트 관리자 — 하드웨어 티어별 전략으로 컨텍스트 윈도우를 관리한다.

Claude Code의 컨텍스트 압축 시스템을 재구현한다 (Ch.6).
로컬 모델의 짧은 컨텍스트 윈도우(4K~32K)에서도
안정적인 멀티턴 대화를 유지하기 위한 핵심 모듈이다.

v7.0 Part 5 Ch 6 (2026-04-21): 하드웨어 티어별 전략을 공식화한다.

전략 매핑:
  TIER_S (RTX 5090, 8K)      → TurnStateStrategy 경유
    - apply_all / auto_compact_if_needed / emergency_compact가 no-op
    - QueryEngine의 TurnStateStore가 이전 턴 요약을 effective_system_prompt로 주입
    - raw messages는 현재 턴만 유지되므로 압축 파이프라인이 불필요
  TIER_M (H100, 32K) / TIER_L (H200+, 128K) → CompressionStrategy 경유
    - 기존 4단계 압축 파이프라인을 그대로 수행
    - tier=None도 동일하게 CompressionStrategy (하위 호환)

4단계 압축 파이프라인 (TIER_M/L):
  ① apply_tool_result_budget: 도구 결과의 토큰 예산 적용
  ② snip_compact: 오래된 턴을 1줄 요약으로 교체
  ③ micro_compact: 도구 결과 내부 미세 압축
  ④ auto_compact: 모델을 사용한 전체 요약 (비동기)

설계 원칙:
  1. 최신 정보 보존: 최근 턴은 항상 원본 유지
  2. 점진적 압축: 갑작스러운 정보 손실 방지
  3. 비용 절약: 모델 호출은 최후의 수단 (auto_compact만)
  4. 투명성: 압축 이벤트를 로깅하여 디버깅 지원
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any

from core.message import (
    Message,
    ToolUseBlock,
)
from core.model.inference import ModelProvider

if TYPE_CHECKING:
    from core.model.hardware_tier import HardwareTier

logger = logging.getLogger("nexus.orchestrator.context_manager")


class ContextManager:
    """
    4단계 컨텍스트 압축 파이프라인.

    query_loop의 매 턴 시작에 apply_all()이 호출되고,
    긴급 상황에서 auto_compact_if_needed()와 emergency_compact()가 사용된다.
    """

    def __init__(
        self,
        model_provider: ModelProvider,
        max_context_tokens: int = 8192,
        tool_result_budget: int = 2048,
        snip_threshold: float = 0.7,
        auto_compact_threshold: float = 0.9,
        preserve_recent_turns: int = 3,
        preserve_recent_tool_results: int = 2,
        tier: HardwareTier | None = None,
    ):
        """
        Args:
            model_provider: 요약에 사용할 모델 프로바이더
            max_context_tokens: 최대 컨텍스트 토큰 수 (GPU 티어에 따라 자동 설정)
            tool_result_budget: 개별 도구 결과의 최대 토큰 수
            snip_threshold: snip_compact 시작 임계치 (max_tokens 대비 비율)
            auto_compact_threshold: auto_compact 시작 임계치
            preserve_recent_turns: 항상 보존할 최근 턴 수
            preserve_recent_tool_results: 항상 보존할 최근 도구 결과 수
            tier: 하드웨어 티어. TIER_S이면 압축 파이프라인을 비활성(pass-through)
                으로 두어 TurnStateStore에 컨텍스트 관리를 위임한다.
                TIER_M/L 또는 None(하위 호환)이면 기존 4단계 파이프라인을 수행.
        """
        self.model_provider = model_provider
        self.max_tokens = max_context_tokens
        self.tool_result_budget = tool_result_budget
        self.snip_threshold = snip_threshold
        self.auto_compact_threshold = auto_compact_threshold
        self.preserve_recent_turns = preserve_recent_turns
        self.preserve_recent_tool_results = preserve_recent_tool_results
        self._tier = tier

        # TIER_S는 TurnStateStore가 요약 담당 → 이 ContextManager는 pass-through
        # 타입 이름 비교로 import 사이클 회피 (HardwareTier는 TYPE_CHECKING에만 import)
        tier_name = getattr(tier, "name", None) or getattr(tier, "value", None)
        self._passthrough = tier_name in ("TIER_S", "small")
        if self._passthrough:
            logger.info(
                "ContextManager: TIER_S 감지 — pass-through 모드 "
                "(TurnStateStore가 컨텍스트 관리 담당)"
            )

        # 상태
        self._compact_boundary: int = 0  # 이 인덱스 이전은 요약으로 대체됨
        self._compact_summary: str | None = None  # 이전 대화의 요약
        self._total_compactions: int = 0  # 총 압축 횟수 (디버깅용)

    # ═══════════════════════════════════════════
    # Public API
    # ═══════════════════════════════════════════

    def apply_all(self, messages: list[Message]) -> list[Message]:
        """
        query_loop의 매 턴 시작에 호출된다.

        1~3단계를 순차 적용한다 (동기, 모델 호출 없음).
        auto_compact는 비동기이므로 별도 호출한다.

        TIER_S pass-through: 메시지를 변경 없이 반환한다.
        TurnStateStore가 이미 이전 턴 요약을 effective_system_prompt로 주입하고
        raw messages는 현재 턴만 유지되므로 압축 파이프라인이 중복·불필요.

        Args:
            messages: 원본 메시지 리스트

        Returns:
            전처리된 메시지 리스트
        """
        if self._passthrough:
            return messages

        # compact_boundary 이후의 활성 메시지만 처리
        active = messages[self._compact_boundary :]

        # 이전 요약이 있으면 맨 앞에 삽입
        result: list[Message] = []
        if self._compact_summary:
            result.append(
                Message.system(f"[대화 요약]\n{self._compact_summary}\n[요약 끝 — 여기서부터 계속]")
            )

        # 1단계: 도구 결과 토큰 예산 적용
        processed = self._apply_tool_result_budget(active)

        # 2단계: 오래된 턴 스니핑
        processed = self._snip_compact(processed)

        # 3단계: 미세 압축
        processed = self._micro_compact(processed)

        result.extend(processed)
        return result

    async def auto_compact_if_needed(
        self,
        messages: list[Message],
        force: bool = False,
    ) -> list[Message]:
        """
        4단계: 토큰 한도에 근접하면 모델을 사용하여 전체 요약한다.

        force=True이면 임계치를 무시하고 강제 압축한다
        (에러 복구 시 사용, Transition 2: reactive_compact_retry).

        TIER_S pass-through: 메시지를 변경 없이 반환한다 (TurnStateStore 담당).

        Args:
            messages: 현재 메시지 리스트
            force: 강제 압축 여부

        Returns:
            압축된 메시지 리스트
        """
        if self._passthrough:
            return messages

        token_count = self._estimate_tokens(messages)

        # 임계치 미달이고 강제가 아니면 그대로 반환
        if not force and token_count < self.max_tokens * self.auto_compact_threshold:
            return messages

        logger.warning(
            f"Auto-compact 시작: {token_count} 토큰 "
            f"(임계치: {self.max_tokens * self.auto_compact_threshold:.0f}, "
            f"force={force})"
        )

        try:
            # 모델에 요약 요청
            summary = await self._get_model_summary(messages)

            # 최근 턴 보존
            recent = self._extract_recent_turns(messages, self.preserve_recent_turns)

            # compact_boundary 업데이트
            self._compact_boundary = len(messages) - len(recent)
            self._compact_summary = summary
            self._total_compactions += 1

            # 요약 + 최근 턴으로 구성
            result = [
                Message.system(f"[대화 요약]\n{summary}\n[요약 끝 — 여기서부터 계속]"),
                *recent,
            ]

            new_count = self._estimate_tokens(result)
            logger.info(
                f"Auto-compact 완료: {token_count} → {new_count} 토큰 "
                f"(절약: {token_count - new_count})"
            )

            return result

        except Exception as e:
            logger.error(f"Auto-compact 실패: {e}")
            # 폴백: 모델 없이 강제 스니핑
            return self._force_snip(messages)

    async def emergency_compact(self, messages: list[Message]) -> list[Message]:
        """
        긴급 압축 — query_loop의 collapse_drain_retry (Transition 1)에서 사용.

        최근 1개 턴만 보존하고 나머지를 1줄 요약으로 대체한다.
        모델 호출 없이 규칙 기반으로 수행한다 (빠르고 안전).

        TIER_S pass-through: TurnStateStore가 이미 이전 맥락을 요약으로 대체
        하므로 긴급 압축도 불필요. 최근 1개 턴만 추출하여 반환한다.

        Args:
            messages: 현재 메시지 리스트

        Returns:
            긴급 압축된 메시지 리스트
        """
        if self._passthrough:
            return self._extract_recent_turns(messages, 1)

        logger.warning("긴급 압축: 최근 1개 턴만 보존")
        self._total_compactions += 1

        recent = self._extract_recent_turns(messages, 1)
        summary = self._rule_based_summary(messages)

        self._compact_summary = summary
        self._compact_boundary = len(messages) - len(recent)

        return [
            Message.system(f"[긴급 요약] {summary}"),
            *recent,
        ]

    # ═══════════════════════════════════════════
    # 1단계: 도구 결과 토큰 예산 (Tool Result Budget)
    # ═══════════════════════════════════════════

    def _apply_tool_result_budget(self, messages: list[Message]) -> list[Message]:
        """
        도구 결과에 토큰 예산을 적용한다.

        오래된 도구 결과부터 먼저 잘라내고,
        최근 N개의 도구 결과는 원본을 보존한다.
        잘라낸 부분은 "head + ... (잘림) + tail" 형태로 교체한다.
        """
        # tool_result 메시지의 인덱스를 수집
        tr_indices = [i for i, m in enumerate(messages) if _is_tool_result(m)]

        if not tr_indices:
            return messages

        result = list(messages)  # 복사

        # 최근 N개는 보존, 나머지에만 예산 적용
        budget_indices = (
            tr_indices[: -self.preserve_recent_tool_results]
            if len(tr_indices) > self.preserve_recent_tool_results
            else []
        )

        for idx in budget_indices:
            msg = result[idx]
            content = str(msg.content)
            content_tokens = self._estimate_tokens_text(content)

            if content_tokens > self.tool_result_budget:
                # head + tail만 보존
                char_budget = self.tool_result_budget * 3  # 토큰당 ~3자
                head_size = char_budget // 2
                tail_size = char_budget // 4

                truncated = (
                    f"{content[:head_size]}\n\n"
                    f"... ({len(content):,}자 전체, "
                    f"~{content_tokens:,} 토큰, 예산 초과로 잘림) ...\n\n"
                    f"{content[-tail_size:]}"
                )

                result[idx] = Message.tool_result(
                    msg.tool_use_id or "",
                    truncated,
                    msg.is_error or False,
                )

        return result

    # ═══════════════════════════════════════════
    # 2단계: 스닙 압축 (Snip Compact)
    # ═══════════════════════════════════════════

    def _snip_compact(self, messages: list[Message]) -> list[Message]:
        """
        오래된 턴을 1줄 요약 마커로 교체한다.

        총 토큰 수가 max_tokens * snip_threshold를 초과할 때 시작하고,
        최근 preserve_recent_turns개 턴은 항상 보존한다.
        """
        token_count = self._estimate_tokens(messages)

        # 임계치 미달이면 그대로 반환
        if token_count < self.max_tokens * self.snip_threshold:
            return messages

        turns = self._split_into_turns(messages)
        if len(turns) <= self.preserve_recent_turns:
            return messages

        result: list[Message] = []
        preserve_start = len(turns) - self.preserve_recent_turns
        target = self.max_tokens * (self.snip_threshold - 0.1)

        for i, turn in enumerate(turns):
            if i < preserve_start and token_count > target:
                # 이 턴을 1줄 요약으로 교체
                summary = self._summarize_turn_rule_based(turn)
                snip_msg = Message.system(f"[스닙된 턴 {i + 1}: {summary}]")
                result.append(snip_msg)

                # 토큰 수 갱신
                saved = self._estimate_tokens(turn) - self._estimate_tokens([snip_msg])
                token_count -= saved
            else:
                result.extend(turn)

        logger.debug(
            f"스닙 압축: {len(turns)}개 턴 → "
            f"{sum(1 for m in result if '스닙된' in str(m.content))}개 스닙, "
            f"남은 ~{token_count} 토큰"
        )

        return result

    # ═══════════════════════════════════════════
    # 3단계: 미세 압축 (Micro Compact)
    # ═══════════════════════════════════════════

    def _micro_compact(self, messages: list[Message]) -> list[Message]:
        """
        도구 결과 내부의 미세 압축을 수행한다.

        적용 규칙:
          - 연속 빈 줄 → 한 줄로 축소
          - 연속 공백 4개 이상 → 4개로 축소
          - 100줄 초과 → head 80줄 + "... (N줄 생략)" + tail 10줄
          - 중복 라인 압축
          - 바이너리 콘텐츠 감지
        """
        result: list[Message] = []

        for msg in messages:
            if _is_tool_result(msg):
                content = str(msg.content)

                # 연속 빈 줄 제거
                content = re.sub(r"\n{3,}", "\n\n", content)

                # 연속 공백 축소
                content = re.sub(r" {4,}", "    ", content)

                # 긴 출력 잘라내기
                lines = content.split("\n")
                if len(lines) > 100:
                    head = lines[:80]
                    tail = lines[-10:]
                    content = (
                        "\n".join(head)
                        + f"\n\n... ({len(lines) - 90}줄 생략) ...\n\n"
                        + "\n".join(tail)
                    )

                # 중복 라인 감지 및 압축
                content = self._compress_duplicate_lines(content)

                # 바이너리 감지
                if self._looks_binary(content):
                    content = f"[바이너리 콘텐츠, {len(content):,}바이트]"

                result.append(
                    Message.tool_result(
                        msg.tool_use_id or "",
                        content,
                        msg.is_error or False,
                    )
                )
            else:
                result.append(msg)

        return result

    # ═══════════════════════════════════════════
    # 내부 헬퍼
    # ═══════════════════════════════════════════

    def _estimate_tokens(self, messages: list[Message]) -> int:
        """
        메시지 목록의 토큰 수를 추정한다.

        정확한 토크나이저 없이 휴리스틱으로 계산한다.
        Message.estimated_tokens()를 합산한다.
        """
        return sum(m.estimated_tokens() for m in messages)

    @staticmethod
    def _estimate_tokens_text(text: str) -> int:
        """
        텍스트의 토큰 수를 추정한다.

        한국어: 글자당 ~2 토큰, 영어: 단어당 ~1.3 토큰.
        """
        korean_chars = sum(1 for c in text if "\uac00" <= c <= "\ud7a3")
        ascii_words = len(text.encode("ascii", "ignore").split())
        return int(ascii_words * 1.3 + korean_chars * 2.0 + len(text) * 0.1)

    async def _get_model_summary(self, messages: list[Message]) -> str:
        """
        모델을 사용한 대화 요약 (4단계).

        최근 20개 메시지만 요약 입력으로 사용한다
        (요약 입력 자체도 컨텍스트 제한이 있으므로).
        """
        # 요약 프롬프트 구성
        conversation_text: list[str] = []
        for msg in messages[-20:]:
            role = msg.role if isinstance(msg.role, str) else msg.role.value
            content = str(msg.content)[:200]
            conversation_text.append(f"[{role}]: {content}")

        prompt = (
            "다음 대화를 500자 이내로 요약해주세요.\n"
            "핵심 사실, 결정사항, 진행 중인 작업만 포함하세요.\n"
            "도구 실행 결과의 세부 내용은 생략하세요.\n\n" + "\n".join(conversation_text)
        )

        summary_parts: list[str] = []
        async for event in self.model_provider.stream(
            messages=[Message.user(prompt)],
            system_prompt="당신은 대화 요약 전문가입니다. 간결하고 사실적으로 요약하세요.",
            tools=None,
            max_tokens=512,
            temperature=0.3,
        ):
            # event.type이 enum이거나 문자열일 수 있음
            event_type = event.type if isinstance(event.type, str) else event.type.value
            if event_type == "text_delta" and event.text:
                summary_parts.append(event.text)

        return "".join(summary_parts) or "[요약 생성 실패]"

    def _force_snip(self, messages: list[Message]) -> list[Message]:
        """
        폴백: 모델 호출 없이 강제 스니핑한다.
        auto_compact 실패 시 사용한다.
        """
        recent = self._extract_recent_turns(messages, 2)
        summary = self._rule_based_summary(messages)
        return [
            Message.system(f"[강제 스닙 요약] {summary}"),
            *recent,
        ]

    def _rule_based_summary(self, messages: list[Message]) -> str:
        """
        규칙 기반 대화 요약 (모델 호출 없음).

        사용자 주제와 사용된 도구 이름을 추출하여 요약한다.
        긴급 압축과 폴백에서 사용한다.
        """
        user_topics: list[str] = []
        tool_names: set[str] = set()

        for msg in messages:
            role = msg.role if isinstance(msg.role, str) else msg.role.value
            if role == "user":
                text = str(msg.content)[:50]
                user_topics.append(text)
            elif role == "assistant" and isinstance(msg.content, list):
                for block in msg.content:
                    if isinstance(block, ToolUseBlock):
                        tool_names.add(block.name)
                    elif isinstance(block, dict) and block.get("type") == "tool_use":
                        tool_names.add(block.get("name", ""))

        topics_str = "; ".join(user_topics[-3:]) if user_topics else "주제 없음"
        tools_str = ", ".join(sorted(tool_names)) if tool_names else "없음"
        return f"주제: [{topics_str}]. 사용된 도구: [{tools_str}]."

    def _extract_recent_turns(self, messages: list[Message], n: int) -> list[Message]:
        """
        최근 n개 턴의 메시지를 추출한다.

        턴은 user → assistant + tool_result 묶음이다.
        뒤에서부터 user 메시지를 세어 n개 턴을 추출한다.
        """
        if n <= 0 or not messages:
            return []

        # 뒤에서부터 user 메시지를 세어 시작 인덱스를 결정
        user_count = 0
        start_idx = len(messages)
        for i in range(len(messages) - 1, -1, -1):
            role = messages[i].role if isinstance(messages[i].role, str) else messages[i].role.value
            if role == "user":
                user_count += 1
                if user_count >= n:
                    start_idx = i
                    break

        return messages[start_idx:]

    def _split_into_turns(self, messages: list[Message]) -> list[list[Message]]:
        """
        메시지를 턴 단위로 분할한다.

        턴은 user 메시지로 시작하여 다음 user 메시지 전까지의 묶음이다.
        시스템 메시지가 맨 앞에 있으면 첫 턴에 포함한다.
        """
        turns: list[list[Message]] = []
        current: list[Message] = []

        for msg in messages:
            role = msg.role if isinstance(msg.role, str) else msg.role.value
            if role == "user" and current:
                turns.append(current)
                current = []
            current.append(msg)

        if current:
            turns.append(current)

        return turns

    def _summarize_turn_rule_based(self, turn: list[Message]) -> str:
        """
        규칙 기반으로 단일 턴을 요약한다.

        사용자 질문과 사용된 도구를 추출하여 한 줄로 만든다.
        snip_compact에서 사용한다.
        """
        user_text = ""
        tool_names: list[str] = []

        for msg in turn:
            role = msg.role if isinstance(msg.role, str) else msg.role.value
            if role == "user":
                user_text = str(msg.content)[:40]
            elif role == "assistant" and isinstance(msg.content, list):
                for block in msg.content:
                    if isinstance(block, ToolUseBlock):
                        tool_names.append(block.name)
                    elif isinstance(block, dict) and block.get("type") == "tool_use":
                        tool_names.append(block.get("name", ""))

        tools_str = ", ".join(tool_names) if tool_names else "없음"
        return f"질문: {user_text} / 도구: {tools_str}"

    @staticmethod
    def _compress_duplicate_lines(content: str) -> str:
        """
        중복 라인을 압축한다.

        같은 줄이 3번 이상 연속되면 "... (N번 반복)" 으로 교체한다.
        로그 출력이나 반복되는 경고를 압축하는 데 효과적이다.
        """
        lines = content.split("\n")
        if len(lines) < 3:
            return content

        result: list[str] = []
        prev_line = ""
        repeat_count = 0

        for line in lines:
            stripped = line.strip()
            if stripped == prev_line and stripped:
                repeat_count += 1
            else:
                if repeat_count >= 2:
                    result.append(f"... ({repeat_count + 1}번 반복)")
                elif repeat_count == 1:
                    result.append(prev_line)
                prev_line = stripped
                repeat_count = 0
                result.append(line)

        # 마지막 반복 처리
        if repeat_count >= 2:
            result.append(f"... ({repeat_count + 1}번 반복)")
        elif repeat_count == 1:
            result.append(prev_line)

        return "\n".join(result)

    @staticmethod
    def _looks_binary(content: str) -> bool:
        """
        콘텐츠가 바이너리인지 감지한다.

        NULL 바이트나 비인쇄 문자의 비율이 높으면 바이너리로 판단한다.
        """
        if not content:
            return False

        # NULL 바이트가 있으면 바이너리
        if "\x00" in content:
            return True

        # 비인쇄 문자 비율 확인 (10% 초과하면 바이너리)
        non_printable = sum(
            1
            for c in content[:1000]  # 처음 1000자만 검사 (성능)
            if ord(c) < 32 and c not in "\n\r\t"
        )

        return non_printable > len(content[:1000]) * 0.1

    @property
    def stats(self) -> dict[str, Any]:
        """압축 통계를 반환한다 (디버깅용)."""
        tier_name = (
            getattr(self._tier, "name", None)
            or getattr(self._tier, "value", None)
            or "unset"
        )
        return {
            "tier": tier_name,
            "passthrough": self._passthrough,
            "total_compactions": self._total_compactions,
            "compact_boundary": self._compact_boundary,
            "has_summary": self._compact_summary is not None,
            "max_tokens": self.max_tokens,
        }

    @property
    def passthrough(self) -> bool:
        """TIER_S pass-through 모드 여부 (외부 관측용)."""
        return self._passthrough


# ─────────────────────────────────────────────
# 모듈 레벨 헬퍼
# ─────────────────────────────────────────────
def _is_tool_result(msg: Message) -> bool:
    """메시지가 tool_result인지 확인한다."""
    role = msg.role if isinstance(msg.role, str) else msg.role.value
    return role == "tool_result"
