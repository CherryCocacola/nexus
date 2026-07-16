"""
컨텍스트 관리자 — 하드웨어 티어별 전략으로 컨텍스트 윈도우를 관리한다.

[이 파일이 하는 일]
로컬 LLM은 한 번에 볼 수 있는 토큰(컨텍스트 윈도우)이 4K~32K로 매우 짧다.
대화가 길어지면 과거 메시지 전체가 이 한도를 넘어서고, 그러면 모델이
입력을 받지 못하거나 앞부분을 통째로 잃어버린다. 이 모듈은 그 사이에서
"오래되고 덜 중요한 내용을 점진적으로 줄여" 한도 안에 맞추는 역할을 한다.
즉 Claude Code의 컨텍스트 압축(compaction) 시스템을 로컬 환경에 맞게
재구현한 것이다 (사양서 Ch.6).

[핵심 클래스]
  - ContextManager: 압축 파이프라인 본체. query_loop이 매 턴 시작에 호출한다.
[모듈 레벨 헬퍼]
  - _is_tool_result(msg): 메시지가 도구 실행 결과인지 판별.

[누가 호출하나]
  Tier 2의 query_loop이 매 턴 apply_all()을 부르고, 컨텍스트가 넘칠 위기에서
  auto_compact_if_needed() / emergency_compact()를 부른다. 실제 요약이
  필요할 때는 생성자에 주입된 ModelProvider(GPU 서버)로 모델 호출을 위임한다.

v7.0 Part 5 Ch 6 (2026-04-21): 하드웨어 티어별 전략을 공식화한다.

[티어별 전략 매핑 — 왜 갈라지나]
GPU가 좋을수록 컨텍스트 윈도우가 넓어 압축 방식이 달라진다.
  TIER_S (RTX 5090, 8K)      → TurnStateStrategy 경유
    - apply_all / auto_compact_if_needed / emergency_compact가 사실상 no-op
    - 대신 QueryEngine의 TurnStateStore가 "이전 턴 요약"을
      effective_system_prompt(시스템 프롬프트)로 주입한다
    - 그래서 이 관리자에 들어오는 raw messages는 현재 턴만 남아 있어
      추가 압축 파이프라인이 중복·불필요 → pass-through로 그냥 통과시킴
  TIER_M (H100, 32K) / TIER_L (H200+, 128K) → CompressionStrategy 경유
    - 아래 4단계 압축 파이프라인을 그대로 수행
    - tier=None도 동일하게 CompressionStrategy로 취급 (구버전 하위 호환)

[4단계 압축 파이프라인 (TIER_M/L에서만 동작)]
  ① apply_tool_result_budget: 개별 도구 결과가 예산을 넘으면 앞뒤만 남기고 잘라냄
  ② snip_compact: 오래된 턴 전체를 1줄 요약 마커로 교체
  ③ micro_compact: 도구 결과 내부의 빈 줄·중복·긴 출력 등을 미세하게 정리
  ④ auto_compact: 그래도 넘치면 모델을 실제로 호출해 전체를 요약 (비동기, 최후수단)

[설계 원칙 — 압축 순서를 이렇게 잡은 이유]
  1. 최신 정보 보존: 최근 턴은 항상 원본 그대로 둔다 (직전 맥락이 가장 중요)
  2. 점진적 압축: 값싼 단계(①②③)를 먼저, 손실 큰 단계(④)를 나중에 → 급격한 손실 방지
  3. 비용 절약: 모델 호출은 느리고 비싸므로 auto_compact에서만 최후의 수단으로 사용
  4. 투명성: 모든 압축 이벤트를 로깅해 나중에 "왜 정보가 사라졌는지" 추적 가능

작성자: 이현수 / 작성일: 2026-07-05
"""

# __future__ 임포트: 타입 힌트를 문자열로 지연 평가(PEP 563)해서
# "X | None" 같은 신문법과 순환 참조를 안전하게 쓰게 해준다. 반드시 파일 최상단.
from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any

# core.message: 대화의 기본 단위인 Message와, 도구 호출 블록 ToolUseBlock.
#   Message.system()/user()/tool_result() 같은 팩토리 메서드로만 생성한다.
from core.message import (
    Message,
    ToolUseBlock,
)

# core.model.inference: GPU 서버(Machine B)와 통신하는 모델 프로바이더.
#   auto_compact 단계에서 실제 요약 생성을 이 프로바이더에 위임한다.
from core.model.inference import ModelProvider

# HardwareTier는 런타임에는 필요 없고 타입 체크에만 쓴다. TYPE_CHECKING 블록에
# 넣어 실제 import를 생략함으로써 import 사이클(순환 의존)을 피한다.
if TYPE_CHECKING:
    from core.model.hardware_tier import HardwareTier

# 모듈 전용 로거. 규칙상 "nexus.{모듈경로}" 네임스페이스를 사용한다.
logger = logging.getLogger("nexus.orchestrator.context_manager")


class ContextManager:
    """
    4단계 컨텍스트 압축 파이프라인의 본체.

    [사용 흐름]
      - query_loop은 매 턴을 시작하기 전에 apply_all()로 메시지를 다듬는다
        (동기, 모델 호출 없음 — 값싼 1~3단계).
      - 그래도 토큰이 임계치에 근접하면 auto_compact_if_needed()로 모델을 불러
        전체 요약을 만든다(비동기, 4단계).
      - 컨텍스트가 이미 터지기 직전인 에러 복구 상황에서는 emergency_compact()로
        최근 1개 턴만 남기고 규칙 기반 요약으로 즉시 줄인다.

    [상태 보존]
      압축이 일어나면 _compact_boundary(경계 인덱스)와 _compact_summary(누적 요약)를
      인스턴스에 기억해 둔다. 다음 apply_all()이 이 경계 이후의 메시지만 활성으로
      다루고, 요약을 맨 앞에 다시 붙여 과거 맥락을 한 줄로 유지한다.

    [티어 분기]
      생성 시 tier가 TIER_S(또는 "small")면 _passthrough=True가 되어 모든 공개
      메서드가 입력을 거의 그대로 돌려준다(위 모듈 docstring의 티어 전략 참고).
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
        ContextManager를 초기화한다. 모든 임계치·보존 개수는 설정(YAML)에서 읽어
        주입하는 것이 원칙이며, 여기 기본값은 RTX 5090(8K) 기준의 안전값이다.

        Args:
            model_provider: 4단계 auto_compact에서 요약을 생성할 모델 프로바이더.
            max_context_tokens: 이 관리자가 지키려는 최대 컨텍스트 토큰 수.
                GPU 티어에 따라 상위 계층이 자동으로 넣어 준다.
            tool_result_budget: 개별 도구 결과 하나가 가질 수 있는 최대 토큰 수.
                이를 넘으면 1단계에서 앞뒤만 남기고 잘라낸다.
            snip_threshold: 2단계 snip_compact를 시작하는 임계 비율(max_tokens 대비).
                예: 0.7이면 전체가 max의 70%를 넘을 때부터 오래된 턴을 스니핑.
            auto_compact_threshold: 4단계 auto_compact를 시작하는 임계 비율.
                snip보다 높게(예: 0.9) 두어 값싼 단계를 먼저 소진하도록 한다.
            preserve_recent_turns: 어떤 압축에서도 원본으로 지킬 최근 턴 수.
            preserve_recent_tool_results: 예산 적용에서 제외하고 원본으로 둘
                최근 도구 결과 수(최근 결과일수록 대화에 더 중요하므로).
            tier: 하드웨어 티어. TIER_S(또는 "small")이면 파이프라인을 통째로
                pass-through(무동작)로 두고 컨텍스트 관리를 TurnStateStore에 위임한다.
                TIER_M/L 또는 None(하위 호환)이면 기존 4단계 파이프라인을 수행한다.
        """
        self.model_provider = model_provider
        self.max_tokens = max_context_tokens
        self.tool_result_budget = tool_result_budget
        self.snip_threshold = snip_threshold
        self.auto_compact_threshold = auto_compact_threshold
        self.preserve_recent_turns = preserve_recent_turns
        self.preserve_recent_tool_results = preserve_recent_tool_results
        self._tier = tier

        # TIER_S에서는 TurnStateStore가 요약을 전담하므로 이 관리자는 pass-through.
        # HardwareTier를 직접 import하지 않고 "이름 문자열"로만 비교해 import
        # 사이클을 피한다. tier가 Enum이면 .name, 문자열 값이면 .value가 잡힌다.
        tier_name = getattr(tier, "name", None) or getattr(tier, "value", None)
        # 두 표기("TIER_S" enum 이름, "small" 문자열 값) 모두를 pass-through로 인정.
        self._passthrough = tier_name in ("TIER_S", "small")
        if self._passthrough:
            logger.info(
                "ContextManager: TIER_S 감지 — pass-through 모드 "
                "(TurnStateStore가 컨텍스트 관리 담당)"
            )

        # ── 압축 상태(인스턴스에 걸쳐 누적) ──
        # _compact_boundary: 이 인덱스보다 앞선 메시지는 이미 요약으로 대체됨.
        self._compact_boundary: int = 0
        # _compact_summary: 경계 이전 대화를 압축한 텍스트(없으면 None).
        self._compact_summary: str | None = None
        # _total_compactions: 지금까지 수행한 총 압축 횟수(통계/디버깅용).
        self._total_compactions: int = 0
        # _last_compaction: 직전 공개 메서드 호출에서 "실제로" 압축이 일어났을 때
        # (메시지/토큰이 실제로 줄었을 때만) 사람이 읽을 요약 문구를 담는 최소 훅.
        # None이면 이번 호출은 no-op(압축 없음)이라는 뜻. query_loop이 압축 직후
        # take_last_compaction()으로 꺼내(consume) CONTEXT_COMPACT 이벤트 발생 여부를
        # 판단한다. no-op 통과에서는 계속 None이라 매 턴 표시가 뜨지 않는다.
        self._last_compaction: str | None = None

    # ═══════════════════════════════════════════
    # Public API
    # ═══════════════════════════════════════════

    def apply_all(self, messages: list[Message]) -> list[Message]:
        """
        매 턴 시작에 호출되는 "값싼" 전처리 — 1~3단계를 순서대로 적용한다.

        여기서는 모델을 호출하지 않는다(전부 동기·규칙 기반). 모델을 쓰는 4단계
        auto_compact는 비용이 크므로 apply_all에 넣지 않고 query_loop이 필요할 때만
        auto_compact_if_needed()로 따로 부른다.

        [TIER_S pass-through]
        TurnStateStore가 이미 이전 턴 요약을 effective_system_prompt로 주입하고,
        들어오는 raw messages는 현재 턴만 담고 있어 압축이 중복·불필요하다.
        그래서 입력을 그대로 반환한다.

        Args:
            messages: 아직 압축되지 않은 원본 메시지 리스트.

        Returns:
            (필요 시) 과거 요약이 맨 앞에 붙고 1~3단계로 다듬어진 메시지 리스트.
        """
        if self._passthrough:
            return messages

        # 이전에 압축한 적이 있다면 그 경계(_compact_boundary) 이후만 "활성"으로 본다.
        # 경계 이전은 이미 _compact_summary 한 덩어리로 대체됐기 때문이다.
        active = messages[self._compact_boundary :]

        # 과거 요약이 있으면, 잘린 맥락을 모델이 알 수 있도록 맨 앞에 시스템 메시지로 붙인다.
        result: list[Message] = []
        if self._compact_summary:
            result.append(
                Message.system(f"[대화 요약]\n{self._compact_summary}\n[요약 끝 — 여기서부터 계속]")
            )

        # 1단계: 개별 도구 결과가 예산을 넘으면 앞뒤만 남기고 잘라낸다.
        processed = self._apply_tool_result_budget(active)

        # 2단계: 전체가 임계치를 넘으면 오래된 턴을 1줄 요약 마커로 교체한다.
        processed = self._snip_compact(processed)

        # 3단계: 남은 도구 결과 내부의 빈 줄·중복·긴 출력을 미세하게 정리한다.
        processed = self._micro_compact(processed)

        # 요약(있으면) + 다듬어진 활성 메시지를 이어 붙여 최종 리스트를 만든다.
        result.extend(processed)
        return result

    async def auto_compact_if_needed(
        self,
        messages: list[Message],
        force: bool = False,
    ) -> list[Message]:
        """
        4단계: 토큰이 한도에 근접하면 모델을 실제로 호출해 전체를 요약한다.

        1~3단계로도 부족할 때만 쓰는 최후의 수단이다(모델 호출은 느리고 비쌈).
        force=True이면 임계치를 무시하고 무조건 압축한다 — 에러 복구 경로에서
        사용한다(query_loop의 Transition 2: reactive_compact_retry).

        [TIER_S pass-through] 입력을 그대로 반환한다(TurnStateStore가 담당).

        [실패 시 폴백] 모델 요약이 예외로 실패하면 모델 없이 규칙 기반으로 강제
        스니핑(_force_snip)해서라도 크기를 줄인다 → 절대 그냥 터지지 않도록.

        Args:
            messages: 현재 전체 메시지 리스트.
            force: True면 임계치와 무관하게 강제로 압축.

        Returns:
            요약 1건 + 최근 턴으로 재구성된(또는 폴백으로 스니핑된) 메시지 리스트.
        """
        if self._passthrough:
            return messages

        # 현재 대화의 대략적 토큰 수를 추정한다(정확한 토크나이저 없이 휴리스틱).
        token_count = self._estimate_tokens(messages)

        # 아직 임계치에 못 미치고 강제도 아니면 압축할 필요가 없으므로 그대로 반환.
        if not force and token_count < self.max_tokens * self.auto_compact_threshold:
            return messages

        logger.warning(
            f"Auto-compact 시작: {token_count} 토큰 "
            f"(임계치: {self.max_tokens * self.auto_compact_threshold:.0f}, "
            f"force={force})"
        )

        try:
            # 모델에게 지금까지의 대화를 짧게 요약해 달라고 요청한다(비동기 스트림).
            summary = await self._get_model_summary(messages)

            # 최근 preserve_recent_turns개 턴은 요약하지 않고 원본으로 지킨다.
            recent = self._extract_recent_turns(messages, self.preserve_recent_turns)

            # 경계를 "최근 턴 시작 지점"으로 옮기고, 그 앞은 요약으로 대체한다.
            self._compact_boundary = len(messages) - len(recent)
            self._compact_summary = summary
            self._total_compactions += 1

            # 최종 형태: [요약 시스템 메시지] + [보존한 최근 턴들].
            result = [
                Message.system(f"[대화 요약]\n{summary}\n[요약 끝 — 여기서부터 계속]"),
                *recent,
            ]

            # 얼마나 줄었는지 로깅해 두면 나중에 압축 효과를 추적하기 쉽다.
            new_count = self._estimate_tokens(result)
            logger.info(
                f"Auto-compact 완료: {token_count} → {new_count} 토큰 "
                f"(절약: {token_count - new_count})"
            )

            # 모델 호출로 전체를 요약한 "가장 큰" 압축 — 표시 문구를 남긴다
            # (앞 단계(예산/스닙)가 남긴 문구가 있어도 이걸로 덮어쓴다).
            self._last_compaction = "대화 요약 생성(모델 호출)"

            return result

        except Exception as e:
            # 모델 요약이 실패해도 대화를 멈추면 안 되므로, 모델 없는 규칙 기반
            # 강제 스니핑으로 최소한의 축소는 보장한다(fail-safe).
            logger.error(f"Auto-compact 실패: {e}")
            return self._force_snip(messages)

    async def emergency_compact(self, messages: list[Message]) -> list[Message]:
        """
        긴급 압축 — 컨텍스트가 이미 터지기 직전일 때 쓰는 가장 공격적인 축소.

        query_loop의 collapse_drain_retry(Transition 1) 경로에서 호출된다.
        모델을 부를 여유조차 없는 상황이므로, 최근 1개 턴만 남기고 나머지는
        규칙 기반 한 줄 요약으로 즉시 대체한다(빠르고, 절대 실패하지 않음).

        [TIER_S pass-through] TurnStateStore가 이미 이전 맥락을 요약으로 대체해
        두었으므로 긴급 압축도 불필요 → 최근 1개 턴만 추출해 그대로 반환한다.

        Args:
            messages: 현재 전체 메시지 리스트.

        Returns:
            [긴급 요약 한 줄] + [최근 1개 턴]으로 구성된 최소 메시지 리스트.
        """
        if self._passthrough:
            return self._extract_recent_turns(messages, 1)

        logger.warning("긴급 압축: 최근 1개 턴만 보존")
        self._total_compactions += 1

        # 직전 턴 1개만 원본 보존, 나머지 전체는 도구·주제만 뽑은 한 줄 요약으로 압축.
        recent = self._extract_recent_turns(messages, 1)
        summary = self._rule_based_summary(messages)

        # 다음 apply_all이 이 경계 이후만 다루도록 상태를 갱신한다.
        self._compact_summary = summary
        self._compact_boundary = len(messages) - len(recent)

        # 컨텍스트가 터지기 직전의 가장 공격적인 축소 — 표시 문구를 남긴다.
        self._last_compaction = "긴급 압축 — 최근 대화만 보존"

        return [
            Message.system(f"[긴급 요약] {summary}"),
            *recent,
        ]

    # ═══════════════════════════════════════════
    # 1단계: 도구 결과 토큰 예산 (Tool Result Budget)
    # ═══════════════════════════════════════════

    def _apply_tool_result_budget(self, messages: list[Message]) -> list[Message]:
        """
        [1단계] 개별 도구 결과에 토큰 예산을 적용해 지나치게 큰 결과를 잘라낸다.

        도구 결과(파일 내용, 명령 출력 등)는 종종 수천 토큰에 달해 컨텍스트를
        가장 많이 잡아먹는다. 그래서 오래된 결과부터 예산(tool_result_budget)을
        넘는 것만 "앞부분 + 잘림 안내 + 끝부분"으로 축약한다. 최근 N개
        (preserve_recent_tool_results)는 대화에 가장 중요하므로 원본을 지킨다.
        """
        # 먼저 전체 메시지 중 도구 결과인 것들의 위치(인덱스)만 모은다.
        tr_indices = [i for i, m in enumerate(messages) if _is_tool_result(m)]

        # 도구 결과가 하나도 없으면 손댈 게 없다.
        if not tr_indices:
            return messages

        result = list(messages)  # 원본을 건드리지 않도록 얕은 복사 후 수정한다.

        # 최근 N개를 제외한 "오래된" 도구 결과 인덱스만 예산 적용 대상으로 삼는다.
        # 개수가 N 이하면 전부 최근으로 보고 아무것도 자르지 않는다(빈 리스트).
        budget_indices = (
            tr_indices[: -self.preserve_recent_tool_results]
            if len(tr_indices) > self.preserve_recent_tool_results
            else []
        )

        truncated_count = 0  # 실제로 잘라낸 도구 결과 수(표시 문구 판단용).
        for idx in budget_indices:
            msg = result[idx]
            content = str(msg.content)
            content_tokens = self._estimate_tokens_text(content)

            # 토큰→글자 환산은 대략 토큰당 3자로 잡는다.
            char_budget = self.tool_result_budget * 3
            # 예산을 넘는 결과만 축약 대상이다. 단 토큰 추정(한글 2자/토큰)과 글자
            # 환산(3자/토큰)이 어긋나, "토큰은 예산 초과이나 글자 수는 char_budget
            # 이하"인 한글 결과를 잘못 축약하면 head(content 전부) + tail(끝 일부)이
            # 중복돼 오히려 메시지가 팽창하는 버그가 있었다. 그래서 토큰·글자 양쪽 모두
            # 예산을 넘을 때만 축약한다(글자 수가 이미 예산 이하면 잘라도 줄지 않는다).
            if content_tokens > self.tool_result_budget and len(content) > char_budget:
                truncated_count += 1
                # 앞부분을 더 많이(1/2), 끝부분을 조금(1/4) 남긴다.
                # 보통 결과의 시작 쪽에 핵심(경로·헤더·요약)이 있기 때문이다.
                head_size = char_budget // 2
                # 끝부분은 head가 가져간 뒤 남은 글자 안에서만 취해 head와 겹치지
                # 않게 한다(len(content) > char_budget 가드로 tail_size ≥ 0 보장).
                tail_size = min(char_budget // 4, len(content) - head_size)

                # 잘린 자리에는 "얼마나 잘렸는지"를 사람이 읽을 수 있게 표기한다.
                truncated = (
                    f"{content[:head_size]}\n\n"
                    f"... ({len(content):,}자 전체, "
                    f"~{content_tokens:,} 토큰, 예산 초과로 잘림) ...\n\n"
                    f"{content[-tail_size:]}"
                )

                # 축약된 내용으로 같은 tool_use_id·에러여부를 유지한 새 결과로 교체.
                # (Message는 불변이므로 수정이 아니라 새 객체를 만들어 갈아끼운다.)
                result[idx] = Message.tool_result(
                    msg.tool_use_id or "",
                    truncated,
                    msg.is_error or False,
                )

        # 실제로 하나라도 잘랐으면 표시 문구를 남긴다(뒤 단계가 더 큰 압축을 하면
        # 덮어씀 — 우선순위: 도구결과 예산 < 턴 스닙 < 모델/긴급 요약).
        if truncated_count > 0:
            self._last_compaction = f"긴 도구 결과 정리 ({truncated_count}건)"

        return result

    # ═══════════════════════════════════════════
    # 2단계: 스닙 압축 (Snip Compact)
    # ═══════════════════════════════════════════

    def _snip_compact(self, messages: list[Message]) -> list[Message]:
        """
        [2단계] 오래된 턴 전체를 규칙 기반 1줄 요약 마커로 통째 교체한다.

        1단계가 결과 하나씩을 줄였다면, 여기서는 "턴 단위"로 과감히 접는다.
        전체 토큰이 max_tokens * snip_threshold를 넘을 때만 시작하고,
        최근 preserve_recent_turns개 턴은 언제나 원본으로 남긴다.
        모델 호출 없이 진행하므로 빠르다.
        """
        token_count = self._estimate_tokens(messages)

        # 아직 스닙 임계치에 못 미치면 손대지 않는다.
        if token_count < self.max_tokens * self.snip_threshold:
            return messages

        # 메시지를 user 시작 기준의 턴 리스트로 나눈다.
        turns = self._split_into_turns(messages)
        # 보존해야 할 턴 수 이하로만 있으면 스닙할 대상이 없다.
        if len(turns) <= self.preserve_recent_turns:
            return messages

        result: list[Message] = []
        # 이 인덱스부터(=뒤쪽 최근 턴들)는 무조건 원본 보존한다.
        preserve_start = len(turns) - self.preserve_recent_turns
        # 목표치: 임계치보다 조금(0.1) 더 낮은 수준까지 줄이면 스닙을 멈춘다.
        target = self.max_tokens * (self.snip_threshold - 0.1)

        snipped_count = 0  # 실제로 접은(스닙한) 턴 수(표시 문구 판단용).
        for i, turn in enumerate(turns):
            # 보존 구간 이전이고, 아직 목표치보다 크면 이 턴을 접는다.
            if i < preserve_start and token_count > target:
                # 이 턴을 질문+도구 이름만 담은 한 줄로 요약해 교체.
                summary = self._summarize_turn_rule_based(turn)
                snip_msg = Message.system(f"[스닙된 턴 {i + 1}: {summary}]")
                result.append(snip_msg)
                snipped_count += 1

                # 교체로 절약된 토큰만큼 현재 추정치를 깎아 목표 도달을 판정한다.
                saved = self._estimate_tokens(turn) - self._estimate_tokens([snip_msg])
                token_count -= saved
            else:
                # 최근 턴이거나 이미 목표에 도달했으면 원본 그대로 유지.
                result.extend(turn)

        logger.debug(
            f"스닙 압축: {len(turns)}개 턴 → {snipped_count}개 스닙, "
            f"남은 ~{token_count} 토큰"
        )

        # 실제로 턴을 접었을 때만 표시 문구를 남긴다(1단계 도구결과 예산보다 상위 —
        # 있으면 덮어씀). 접은 게 없으면 no-op라 이전 문구를 건드리지 않는다.
        if snipped_count > 0:
            self._last_compaction = f"이전 대화 {snipped_count}턴 요약"

        return result

    # ═══════════════════════════════════════════
    # 3단계: 미세 압축 (Micro Compact)
    # ═══════════════════════════════════════════

    def _micro_compact(self, messages: list[Message]) -> list[Message]:
        """
        [3단계] 남아 있는 도구 결과 "내부"를 미세하게 정리해 군더더기를 줄인다.

        턴을 통째로 접는 2단계와 달리, 여기서는 내용은 살리되 낭비되는 공간만
        걷어낸다(무손실에 가까운 정리). 도구 결과가 아닌 메시지는 손대지 않는다.

        적용 규칙:
          - 연속 빈 줄 3개 이상 → 2개로 축소
          - 연속 공백 4개 이상 → 4개로 정규화
          - 100줄 초과 → 앞 80줄 + "... (N줄 생략)" + 끝 10줄
          - 같은 줄 반복 → "... (N번 반복)"으로 압축
          - 바이너리로 보이면 통째로 안내 문구로 대체

        [최근성 면제] 최근 N개(preserve_recent_tool_results) 도구 결과는 대화에 가장
        중요하므로(특히 방금 읽은 문서 본문) 손실 압축(100줄 접기·중복 줄 압축)을
        건너뛴다. 이 면제가 없으면 DocumentProcess의 문서 통짜 반환이 다음 턴 apply_all
        에서 90줄로 접혀 모델이 전문을 보지 못한다(대용량 문서 분석 품질 열화의 원인).
        무손실 정리(빈 줄·공백 정규화)와 바이너리 대체는 최근 결과에도 그대로 적용한다.
        """
        # 도구 결과들의 위치를 모아 최근 N개 인덱스를 면제 집합으로 만든다.
        # preserve_recent_tool_results가 0이면 면제 없음(빈 집합) — [-0:]가 전체를
        # 뜻하는 슬라이스 함정을 피하려 0을 명시적으로 걸러낸다.
        tr_indices = [i for i, m in enumerate(messages) if _is_tool_result(m)]
        preserve_idx = set(
            tr_indices[-self.preserve_recent_tool_results :]
            if self.preserve_recent_tool_results > 0
            else []
        )

        result: list[Message] = []

        for i, msg in enumerate(messages):
            # 도구 결과에만 정리 규칙을 적용한다.
            if _is_tool_result(msg):
                content = str(msg.content)

                # 3개 이상 연속된 빈 줄을 2줄로 줄여 세로 공백을 절약한다(무손실).
                content = re.sub(r"\n{3,}", "\n\n", content)

                # 4칸 이상 연속 공백을 4칸으로 정규화한다(들여쓰기 낭비 제거, 무손실).
                content = re.sub(r" {4,}", "    ", content)

                # 최근 도구 결과가 아닐 때만 손실 압축을 적용한다(최근 것은 원본 보존).
                if i not in preserve_idx:
                    # 줄 수가 너무 많으면 앞뒤만 남기고 가운데를 생략 표기로 접는다.
                    lines = content.split("\n")
                    if len(lines) > 100:
                        head = lines[:80]
                        tail = lines[-10:]
                        content = (
                            "\n".join(head)
                            + f"\n\n... ({len(lines) - 90}줄 생략) ...\n\n"
                            + "\n".join(tail)
                        )

                    # 로그처럼 같은 줄이 반복되는 경우를 "N번 반복"으로 접는다.
                    content = self._compress_duplicate_lines(content)

                # 이미지·바이너리가 텍스트로 흘러들어온 경우는 통째로 안내로 대체한다.
                if self._looks_binary(content):
                    content = f"[바이너리 콘텐츠, {len(content):,}바이트]"

                # 정리된 내용으로 새 도구 결과를 만들어 넣는다(불변 객체이므로 재생성).
                result.append(
                    Message.tool_result(
                        msg.tool_use_id or "",
                        content,
                        msg.is_error or False,
                    )
                )
            else:
                # 도구 결과가 아니면 원본 그대로 통과.
                result.append(msg)

        return result

    # ═══════════════════════════════════════════
    # 내부 헬퍼
    # ═══════════════════════════════════════════

    def _estimate_tokens(self, messages: list[Message]) -> int:
        """
        메시지 목록 전체의 대략적 토큰 수를 추정한다.

        실제 모델 토크나이저를 돌리면 정확하지만 느리다. 압축 판단은 "대충 넘치나"만
        알면 되므로, 각 Message가 스스로 계산한 estimated_tokens()를 단순 합산한다.
        """
        return sum(m.estimated_tokens() for m in messages)

    @staticmethod
    def _estimate_tokens_text(text: str) -> int:
        """
        임의 텍스트 한 덩어리의 토큰 수를 휴리스틱으로 추정한다(도구 결과 예산용).

        경험칙: 한국어 글자는 토큰이 많이 들고(글자당 ~2), 영어는 단어당 ~1.3,
        그 밖의 문자·기호는 전체 길이의 0.1 정도로 보정한다. 정확값이 아니라
        "예산 초과 여부"를 싸게 가늠하기 위한 근사식이다.
        korean_chars는 한글 음절 범위(U+AC00 '가' ~ U+D7A3 '힣') 글자 수,
        ascii_words는 ASCII 단어 수이며, 두 값에 길이 보정을 더해 합산한다.
        """
        korean_chars = sum(1 for c in text if "\uac00" <= c <= "\ud7a3")
        ascii_words = len(text.encode("ascii", "ignore").split())
        return int(ascii_words * 1.3 + korean_chars * 2.0 + len(text) * 0.1)

    async def _get_model_summary(self, messages: list[Message]) -> str:
        """
        [4단계 실동작] 모델(GPU 서버)에 실제로 요청해 대화를 짧게 요약한다.

        요약 입력 자체도 컨텍스트 제한을 받으므로, 최근 20개 메시지만 넣고
        각 메시지는 200자로 잘라 프롬프트를 만든다. 스트림으로 받은 text_delta를
        이어 붙여 최종 요약 문자열을 반환한다. 아무것도 못 받으면 실패 표시를 낸다.
        """
        # 최근 20개 메시지를 "[역할]: 내용(최대 200자)" 형태의 줄로 만든다.
        conversation_text: list[str] = []
        for msg in messages[-20:]:
            # role은 Enum일 수도 문자열일 수도 있어 양쪽을 모두 처리한다.
            role = msg.role if isinstance(msg.role, str) else msg.role.value
            content = str(msg.content)[:200]
            conversation_text.append(f"[{role}]: {content}")

        prompt = (
            "다음 대화를 500자 이내로 요약해주세요.\n"
            "핵심 사실, 결정사항, 진행 중인 작업만 포함하세요.\n"
            "도구 실행 결과의 세부 내용은 생략하세요.\n\n" + "\n".join(conversation_text)
        )

        # 모델 스트림에서 텍스트 조각(text_delta)만 모아 요약을 조립한다.
        # temperature=0.3으로 낮춰 사실적이고 안정적인 요약을 유도한다.
        summary_parts: list[str] = []
        async for event in self.model_provider.stream(
            messages=[Message.user(prompt)],
            system_prompt="당신은 대화 요약 전문가입니다. 간결하고 사실적으로 요약하세요.",
            tools=None,
            max_tokens=512,
            temperature=0.3,
        ):
            # StreamEvent.type도 Enum/문자열 양쪽일 수 있으므로 정규화한다.
            event_type = event.type if isinstance(event.type, str) else event.type.value
            if event_type == "text_delta" and event.text:
                summary_parts.append(event.text)

        # 조각이 하나도 없으면(스트림 비정상 등) 실패 표시를 반환한다.
        return "".join(summary_parts) or "[요약 생성 실패]"

    def _force_snip(self, messages: list[Message]) -> list[Message]:
        """
        폴백 압축: 모델 없이(규칙 기반) 강제로 크기를 줄인다.

        auto_compact의 모델 요약이 예외로 실패했을 때 호출된다. 최근 2개 턴만
        남기고 나머지는 주제·도구만 뽑은 한 줄 요약으로 대체한다. 느린 모델에
        의존하지 않으므로 "무슨 일이 있어도 줄이는" 최종 안전망 역할을 한다.
        """
        recent = self._extract_recent_turns(messages, 2)
        summary = self._rule_based_summary(messages)
        # 모델 요약 실패 폴백도 실제로 대화를 줄이는 압축 — 표시 문구를 남긴다.
        self._last_compaction = "대화 압축(요약 폴백)"
        return [
            Message.system(f"[강제 스닙 요약] {summary}"),
            *recent,
        ]

    def _rule_based_summary(self, messages: list[Message]) -> str:
        """
        규칙 기반 대화 요약(모델 호출 전혀 없음) — 긴급 압축·폴백에서 사용.

        전체 대화를 훑어 (1) 사용자가 던진 주제 텍스트와 (2) assistant가 사용한
        도구 이름을 모은다. 모델 요약처럼 매끄럽지는 않지만 즉시·확실하게 만들 수
        있어, 컨텍스트가 터지기 직전에도 최소한의 맥락을 한 줄로 남길 수 있다.
        """
        user_topics: list[str] = []
        tool_names: set[str] = set()

        for msg in messages:
            role = msg.role if isinstance(msg.role, str) else msg.role.value
            if role == "user":
                # 사용자 메시지는 앞 50자만 주제 후보로 모은다.
                text = str(msg.content)[:50]
                user_topics.append(text)
            elif role == "assistant" and isinstance(msg.content, list):
                # assistant 블록에서 도구 호출을 찾아 이름을 수집한다.
                # 도구 블록은 ToolUseBlock 객체 또는 dict 형태 둘 다 가능하다.
                for block in msg.content:
                    if isinstance(block, ToolUseBlock):
                        tool_names.add(block.name)
                    elif isinstance(block, dict) and block.get("type") == "tool_use":
                        tool_names.add(block.get("name", ""))

        # 최근 주제 3개만, 도구는 이름순 정렬해 사람이 읽기 쉬운 한 줄로 만든다.
        topics_str = "; ".join(user_topics[-3:]) if user_topics else "주제 없음"
        tools_str = ", ".join(sorted(tool_names)) if tool_names else "없음"
        return f"주제: [{topics_str}]. 사용된 도구: [{tools_str}]."

    def _extract_recent_turns(self, messages: list[Message], n: int) -> list[Message]:
        """
        메시지 리스트의 끝에서 최근 n개 "턴"에 해당하는 부분만 잘라 반환한다.

        여기서 한 턴은 user 메시지로 시작해 그에 딸린 assistant/tool_result까지의
        묶음이다. user 메시지의 개수가 곧 턴의 개수이므로, 뒤에서부터 user를 세어
        n번째 user가 나온 위치를 시작점으로 잡고 그 뒤 전체를 돌려준다.
        """
        # n이 0 이하이거나 메시지가 없으면 남길 것이 없다.
        if n <= 0 or not messages:
            return []

        # 뒤에서 앞으로 훑으며 user 메시지를 센다. n번째를 만나면 그 인덱스가 시작점.
        user_count = 0
        start_idx = len(messages)
        for i in range(len(messages) - 1, -1, -1):
            role = messages[i].role if isinstance(messages[i].role, str) else messages[i].role.value
            if role == "user":
                user_count += 1
                if user_count >= n:
                    start_idx = i
                    break

        # 주의: user 메시지가 n개 미만이면 break 없이 끝나 start_idx가 그대로
        # len(messages)로 남고, 결과가 빈 리스트가 된다(호출 측이 이를 감안).
        return messages[start_idx:]

    def _split_into_turns(self, messages: list[Message]) -> list[list[Message]]:
        """
        메시지 리스트를 턴(turn) 단위의 리스트로 나눈다.

        규칙: user 메시지를 만나면 새 턴을 시작한다(단, 앞에 쌓인 게 있을 때만
        경계를 끊는다). 따라서 맨 앞의 시스템 메시지 등은 첫 user와 함께 첫 턴에
        묶인다. snip_compact가 "오래된 턴부터" 접기 위해 이 분할을 사용한다.
        """
        turns: list[list[Message]] = []
        current: list[Message] = []  # 지금 모으고 있는 턴의 버퍼

        for msg in messages:
            role = msg.role if isinstance(msg.role, str) else msg.role.value
            # 새 user가 등장하고 버퍼에 이미 내용이 있으면, 직전 턴을 확정한다.
            if role == "user" and current:
                turns.append(current)
                current = []
            current.append(msg)

        # 루프가 끝난 뒤 버퍼에 남은 마지막 턴을 마저 담는다.
        if current:
            turns.append(current)

        return turns

    def _summarize_turn_rule_based(self, turn: list[Message]) -> str:
        """
        단일 턴 하나를 규칙 기반 한 줄로 요약한다(모델 호출 없음).

        해당 턴 안의 사용자 질문(앞 40자)과 assistant가 부른 도구 이름들을 뽑아
        "질문: ... / 도구: ..." 형태로 만든다. snip_compact가 오래된 턴을 이
        한 줄 마커로 교체할 때 사용한다.
        """
        user_text = ""
        tool_names: list[str] = []

        for msg in turn:
            role = msg.role if isinstance(msg.role, str) else msg.role.value
            if role == "user":
                # 마지막 user 텍스트가 이 턴의 대표 질문이 된다(앞 40자).
                user_text = str(msg.content)[:40]
            elif role == "assistant" and isinstance(msg.content, list):
                # assistant 블록에서 도구 호출 이름을 순서대로 모은다.
                # 도구 블록은 ToolUseBlock 객체 또는 dict 두 형태 모두 가능.
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
        연속으로 중복되는 라인을 "... (N번 반복)"으로 접어 압축한다.

        같은 내용의 줄이 여러 번 이어지면(로그·반복 경고 등) 토큰만 낭비된다.
        여기서는 직전 줄과 같은 줄이 연속될 때 repeat_count를 세다가, 다른 줄을
        만나는 순간(또는 끝) 반복이 충분히 많으면 요약 마커 한 줄로 대체한다.
        줄이 3개 미만이면 압축 이득이 없어 원본을 그대로 반환한다.
        """
        lines = content.split("\n")
        if len(lines) < 3:
            return content

        result: list[str] = []
        prev_line = ""  # 직전에 본(공백 제거된) 줄
        repeat_count = 0  # 직전 줄이 몇 번 더 반복됐는지

        for line in lines:
            stripped = line.strip()
            # 빈 줄이 아니고 직전 줄과 같으면 반복으로 카운트만 늘린다.
            if stripped == prev_line and stripped:
                repeat_count += 1
            else:
                # 반복이 끊겼다. 쌓인 반복을 정리해서 결과에 반영한다.
                # 3회 이상(count>=2) 반복이면 마커 한 줄로 접는다.
                if repeat_count >= 2:
                    result.append(f"... ({repeat_count + 1}번 반복)")
                # 딱 2회(count==1)면 접을 이득이 적어 원본 한 줄만 복원한다.
                elif repeat_count == 1:
                    result.append(prev_line)
                # 기준을 새 줄로 갱신하고, 현재 줄을 결과에 추가한다.
                prev_line = stripped
                repeat_count = 0
                result.append(line)

        # 루프가 끝났는데 아직 반복이 진행 중이었다면 동일 규칙으로 마무리한다.
        if repeat_count >= 2:
            result.append(f"... ({repeat_count + 1}번 반복)")
        elif repeat_count == 1:
            result.append(prev_line)

        return "\n".join(result)

    @staticmethod
    def _looks_binary(content: str) -> bool:
        """
        콘텐츠가 (텍스트가 아니라) 바이너리로 보이는지 판별한다.

        도구가 이미지·실행파일 등을 텍스트로 흘려보내면 압축해도 의미가 없고 오히려
        토큰만 낭비된다. 그래서 NULL 바이트가 있거나, 제어문자(개행/캐리지리턴/탭
        제외)의 비율이 높으면 바이너리로 보고 통째로 안내 문구로 대체하게 한다.
        """
        if not content:
            return False

        # NULL 바이트는 텍스트에 거의 없으므로 하나만 있어도 바이너리로 확정한다.
        if "\x00" in content:
            return True

        # 성능을 위해 앞 1000자만 표본으로 검사한다. ord<32이면서 개행류가 아닌
        # 제어문자를 비인쇄로 카운트한다.
        non_printable = sum(
            1
            for c in content[:1000]
            if ord(c) < 32 and c not in "\n\r\t"
        )

        # 표본에서 비인쇄 비율이 10%를 넘으면 바이너리로 판정한다.
        return non_printable > len(content[:1000]) * 0.1

    @property
    def stats(self) -> dict[str, Any]:
        """
        현재 압축 상태를 한눈에 보는 통계 딕셔너리를 반환한다(디버깅·모니터링용).

        티어, pass-through 여부, 누적 압축 횟수, 현재 경계 인덱스, 요약 보유 여부,
        최대 토큰을 담는다. 로그나 진단 엔드포인트에서 상태를 관측할 때 쓴다.
        """
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
        """
        이 관리자가 TIER_S pass-through(무동작) 모드인지 외부에서 읽기 위한 속성.

        상위 계층이 "압축을 실제로 수행하는 관리자인지"를 판단할 때 참조한다.
        """
        return self._passthrough

    def take_last_compaction(self) -> str | None:
        """
        직전 압축 호출에서 "실제로" 압축이 일어났으면 사람이 읽을 요약 문구를 1회
        반환하고 내부 상태를 비운다(consume). 이번엔 압축이 없었으면 None.

        [왜 consume(꺼내면 지움)인가]
          query_loop이 매 턴 apply_all/auto_compact 직후 이 값을 꺼내 CONTEXT_COMPACT
          이벤트를 UI에 1회만 흘려보내기 위해서다. 꺼내면 즉시 None으로 되돌려,
          압축이 없는 다음 턴에서 같은 문구가 다시 뜨지 않게 한다(중복 표시 방지).

        [무엇이 "실제 압축"인가]
          1단계(도구 결과 예산 절단)·2단계(오래된 턴 스닙)·4단계(모델/폴백 요약)·
          긴급 압축이 실제로 메시지를 줄였을 때만 _last_compaction이 채워진다.
          임계치 미달로 아무것도 안 바뀐 no-op 통과에서는 계속 None이다.
        """
        phrase = self._last_compaction
        self._last_compaction = None
        return phrase


# ─────────────────────────────────────────────
# 모듈 레벨 헬퍼
# ─────────────────────────────────────────────
def _is_tool_result(msg: Message) -> bool:
    """
    주어진 Message가 도구 실행 결과(tool_result)인지 판별한다.

    role은 Enum(RoleEnum)일 수도, 이미 문자열일 수도 있어 양쪽을 모두 처리한 뒤
    "tool_result" 문자열과 비교한다. 여러 압축 단계에서 도구 결과만 골라낼 때 쓴다.
    """
    role = msg.role if isinstance(msg.role, str) else msg.role.value
    return role == "tool_result"
