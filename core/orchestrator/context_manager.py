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
    ThinkingBlock,
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

# ─── 요약 예산 (2026-08-27) ───────────────────────────────
# 예전 값은 "최근 20개 × 200자 = 4,000자 입력, 500자 출력, max_tokens=512" 였다.
# 압축은 55,296 토큰(≈13만 자 이상)에서 걸리는데 요약이 볼 수 있는 것이 4,000자면,
# 요약은 대화의 3% 만 보고 나머지를 대표해야 한다. 실측에서 6~8만 토큰이 234~1,061
# 토큰으로 떨어진 것은 요약 품질 문제가 아니라 **입력 자체가 없었던 것**이다.
#
# 입력 예산의 **상한**이다. 실제로는 창 크기에 맞춰 더 줄인다
# (`_summary_input_budget()`). 한글은 실측 약 0.8 토큰/자이므로 24,000자는
# 약 19,000 토큰이다 — 운영 창(61,440)에서는 안전하지만, 좁은 창 프로파일
# (model_profiles.yaml 의 h200 16,384 / h100 8,192)에서는 요약 호출 자체가
# 창을 넘는다. 그런데 그 호출이 일어나는 자리가 바로 컨텍스트 초과에서
# 회복하려는 자리라, 고정값으로 두면 복구가 복구를 막는다.
_SUMMARY_INPUT_BUDGET = 24_000
# 입력 예산이 창의 몇 배(글자/토큰 환산 포함)까지 차지해도 되는가.
# 한글 최악 1.25자/토큰을 가정해 max_tokens 의 절반(글자)까지만 쓴다.
_SUMMARY_INPUT_WINDOW_RATIO = 0.5
# 출력도 늘린다. 버리는 양이 많을수록 요약이 담아야 할 것도 많다.
_SUMMARY_TARGET_CHARS = 1_500
# 목표 1,500자를 한글 0.8 토큰/자로 환산하면 약 1,200 토큰이다. 그런데 모델이
# 목표를 넘기는 일이 흔해(실측 2,172자 = 약 1,740 토큰) 1,536 으로는 문장
# 중간에서 잘렸다. 요약의 끝은 "진행 중인 작업"이 놓이는 자리라 손실이 크다.
_SUMMARY_MAX_TOKENS = 2_560
# 요약을 만들 값이 있는가 — "버리는 것이 붙이는 것의 몇 배는 돼야 한다".
# 총량 비례(예: 전체의 20%)로 잡으면 대화가 커질수록 과도하게 보수적이 된다.
_SUMMARY_WORTH_RATIO = 2.0


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
        # 설정값 원본. GPU OOM 복구가 max_tokens 를 0.7배로 줄이는데 복원 코드가
        # 없어, CLI 처럼 관리자가 프로세스 수명 내내 사는 표면에서는 0.7ⁿ 로 계속
        # 작아졌다(2026-08-26 리뷰 지적). 되돌릴 기준값을 들고 있는다.
        self._configured_max_tokens = max_context_tokens
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
        # _compact_boundary_id: 경계 지점(=보존한 첫 메시지)의 Message.id.
        # 인덱스만으로는 다른 리스트에 잘못 적용된다 — _resolve_boundary 주석 참조.
        self._compact_boundary_id: str | None = None
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
        # 무효 압축 latch(2026-08-26 리뷰 지적). 자동 압축이 "줄지 않았다"로 끝나면
        # 토큰 수는 임계치 위에 그대로 남고, query_loop 은 매 턴 압축을 다시 시도한다.
        # 그러면 대화가 끝날 때까지 **턴마다 모델 요약 왕복이 한 번씩 버려진다.**
        # 무효였던 시점의 메시지 수를 기억해, 대화가 그보다 늘기 전에는 건너뛴다.
        self._useless_compact_at: int | None = None
        # _result_replaced: 직전 압축 호출이 **새 리스트**를 돌려줬는가.
        # 호출부는 반환값으로 리스트를 교체한 뒤 mark_result_adopted() 를 부르는데,
        # 압축이 일어나지 않아 원본이 그대로 돌아온 경우까지 부르면 경계·요약만
        # 지워져 다음 apply_all 이 경계 이전 원문을 전부 되살린다(2026-08-27).
        # 호출부를 고치는 대신 여기서 알고 있게 해 어느 호출부에서도 안전하게 한다.
        self._result_replaced: bool = False

    def for_session(self) -> ContextManager:
        """설정은 그대로 두고 **압축 상태만 새로 만든** 인스턴스를 돌려준다.

        ■ 왜 필요한가 (2026-08-25 장애)
          웹은 요청마다 세션 전용 엔진을 조립하면서 ContextManager 는 기동 시 만든
          **하나를 공유**하고 있었다(`web/app.py`의 `parts["context_manager"]`).
          그런데 이 객체에는 `_compact_boundary`·`_compact_summary` 라는 대화별
          상태가 있다.

          그래서 어떤 요청 하나가 auto-compact 를 유발하면 그 경계가 인스턴스에
          남고, **이후 모든 요청**이 `messages[_compact_boundary:]` 로 잘렸다.
          플러그인 요청은 메시지가 1~2개뿐이라 통째로 사라졌고, 모델에는 시스템
          프롬프트만 남아 prompt_tokens 가 입력 크기와 무관한 고정값이 됐다.

            08-24 03:07 압축 → 이후 모든 요청 +138 토큰 고정(요약만 남은 경우)
            08-25 08:24 압축 → 이후 모든 요청 5,504·4,688 고정(입력 유실)

          재기동하면 인스턴스가 새로 생겨 증상이 사라지므로 원인을 찾기 어려웠다.

        ■ 왜 설정을 다시 읽지 않고 복제하는가
          호출부(web)가 생성 인자를 다시 조립하면 bootstrap 과 두 벌이 되어 언젠가
          어긋난다. 설정의 단일 출처는 bootstrap 이 만든 이 객체 하나로 둔다.

        CLI 는 프로세스당 세션이 하나라 공유가 곧 격리이므로 이 메서드를 쓰지
        않아도 된다(무회귀).
        """
        return ContextManager(
            model_provider=self.model_provider,
            # **현재값이 아니라 설정값**을 넘긴다. GPU OOM 복구가 max_tokens 를
            # 줄여 놓은 상태에서 clone 하면 그 축소가 새 세션까지 따라간다
            # (2026-08-27). 새 요청은 설정값에서 시작해야 한다.
            max_context_tokens=self._configured_max_tokens,
            tool_result_budget=self.tool_result_budget,
            snip_threshold=self.snip_threshold,
            auto_compact_threshold=self.auto_compact_threshold,
            preserve_recent_turns=self.preserve_recent_turns,
            preserve_recent_tool_results=self.preserve_recent_tool_results,
            tier=self._tier,
        )

    def restore_max_tokens(self) -> None:
        """GPU OOM 복구가 줄여 놓은 컨텍스트 상한을 설정값으로 되돌린다.

        복구 경로(`query_loop` 의 OOM 처리)는 다음 시도에서 메모리를 확보하려고
        `max_tokens` 를 0.7배로 줄인다. 그런데 **되돌리는 코드가 없었다.**

        웹은 이제 요청마다 새 인스턴스를 만들어(for_session) 자연히 초기화되지만,
        CLI 는 관리자가 프로세스 수명 내내 살아 있어 OOM 이 날 때마다 0.7ⁿ 로
        누적 축소됐다. 한 번 줄면 되돌릴 수단이 없어, 이후 모든 대화가 좁아진
        창에서 돌았다.

        축소의 목적은 "이번 요청을 통과시키는 것"이므로 요청 경계에서 되돌린다.
        """
        self.max_tokens = self._configured_max_tokens

    def mark_result_adopted(self) -> None:
        """압축 **결과로 메시지 리스트를 교체한** 호출부가 부른다.

        ■ 왜 필요한가 (2026-08-26)
          `_compact_boundary` 는 **교체 전** 리스트의 인덱스다. 그런데 아래 세 곳은
          압축 결과를 그대로 채택해 리스트를 바꾼다.

            query_loop.py  emergency_compact  → state.messages = 결과
            query_loop.py  force=True 재압축   → state.messages = 결과
            cli/repl.py    /compact           → messages[:] = 결과

          채택 뒤에도 경계가 남아 있으면 다음 턴 `apply_all` 이 **짧아진** 리스트에
          옛 인덱스를 다시 적용해 한 번 더 잘라낸다. 게다가 `_compact_summary` 도
          남아 있어 요약이 **두 번** 앞에 붙는다 — 반환 리스트 0번에 이미 들어 있는데
          apply_all 이 또 붙이기 때문이다.

          반대로 `query_loop.py:821` 은 `api_messages` 라는 **임시** 리스트에만
          결과를 담고 `state.messages` 는 원본을 유지한다. 거기서는 경계가 유효하다.
          그래서 "리스트를 교체했는가"를 아는 호출부가 직접 알려 주는 형태로 둔다.

        요약은 버리지 않는다 — 채택된 리스트의 첫 원소로 이미 들어가 있다.

        ■ 압축이 안 일어났으면 아무것도 하지 않는다 (2026-08-27 리뷰 지적)
          호출부 세 곳 모두 **무조건** 이 메서드를 부른다. 압축이 일어나지 않아
          원본이 그대로 돌아온 경우까지 경계·요약을 지우면, 다음 apply_all 이
          경계 이전 원문을 전부 되살린다 — 줄이러 온 자리에서 오히려 늘어난다.
          호출부를 하나씩 고치는 대신 여기서 막아 어느 호출부에서도 안전하게 한다.
        """
        if not self._result_replaced:
            return
        self._compact_boundary = 0
        self._compact_boundary_id = None
        self._compact_summary = None
        # 리스트가 교체됐으면 옛 메시지 개수를 기준으로 잡은 latch 는 무조건 무효다.
        self._useless_compact_at = None
        self._result_replaced = False

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

        # 이전에 압축한 적이 있다면 그 경계 이후만 "활성"으로 본다.
        # 경계 이전은 이미 _compact_summary 한 덩어리로 대체됐기 때문이다.
        active = messages[self._resolve_boundary(messages) :]

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
            self._result_replaced = False
            return messages

        # 현재 대화의 대략적 토큰 수를 추정한다(정확한 토크나이저 없이 휴리스틱).
        token_count = self._estimate_tokens(messages)

        # 아직 임계치에 못 미치고 강제도 아니면 압축할 필요가 없으므로 그대로 반환.
        if not force and token_count < self.max_tokens * self.auto_compact_threshold:
            self._result_replaced = False
            return messages

        # ── 무효 압축 latch (2026-08-26) ──
        # 직전에 "줄지 않아서 포기"로 끝났다면, 대화가 그때보다 늘기 전에는 다시
        # 시도해도 같은 결과다. 그런데 토큰 수는 임계치 위에 남아 있으므로
        # query_loop 은 매 턴 여기로 들어온다 → 턴마다 모델 요약 왕복이 버려진다.
        # 메시지가 늘면 요약 재료가 달라지므로 latch 를 푼다.
        if (
            not force
            and self._useless_compact_at is not None
            and len(messages) <= self._useless_compact_at
        ):
            logger.debug(
                "Auto-compact 건너뜀 — 직전 시도가 무효였고 대화가 늘지 않았다 "
                "(메시지 %d개)",
                len(messages),
            )
            self._result_replaced = False
            return messages

        logger.warning(
            f"Auto-compact 시작: {token_count} 토큰 "
            f"(임계치: {self.max_tokens * self.auto_compact_threshold:.0f}, "
            f"force={force})"
        )

        try:
            # 최근 preserve_recent_turns개 턴은 요약하지 않고 원본으로 지킨다.
            # **요약보다 먼저 계산한다** — 요약이 대상으로 삼아야 하는 것은 지켜지는
            # 최근 구간이 아니라 **사라지는 앞부분**이기 때문이다(2026-08-27).
            recent = self._extract_recent_turns(messages, self.preserve_recent_turns)
            dropped = messages[: len(messages) - len(recent)]

            if not dropped:
                # ── force 라도 "실제로 넘칠 때"만 더 내려간다 ──
                # force 는 두 자리에서 온다. query_loop 의 컨텍스트 초과 복구는
                # "무슨 수를 써서라도 줄여라"이지만, CLI `/compact` 는 사용자가
                # 정리를 요청한 것이라 "줄일 수 있으면 줄여라"다. 둘을 같은 플래그로
                # 묶어 두고 무조건 내려가면, 짧은 대화에서 `/compact` 한 번이
                # 질문·답변을 통째로 날린다(실측 6개 45토큰 → 2개 54토큰: 줄지도
                # 않으면서 5개가 사라졌다). 그래서 임계치 초과를 조건에 넣는다.
                oversized = token_count >= int(self.max_tokens * self.auto_compact_threshold)
                split = self._message_level_split(messages) if (force and oversized) else 0

                # 짝을 지키느라 버릴 것이 껍데기만 남는 경우가 있다 — 질문 하나 +
                # assistant(tool_calls) + 거대한 도구 결과들이면 split 이 1이라
                # 버리는 것이 user 메시지 하나뿐이다. 그러면 요약이 붙어 오히려
                # 커지는데 force 는 무효 검증(new_count >= token_count)을 건너뛰므로
                # 그 커진 결과가 그대로 나간다. 무게를 먼저 재고, 가벼우면 요약
                # 왕복을 아예 하지 않고 본문 절단으로 간다.
                #
                # 기준은 총량 비례가 아니라 **버리는 것이 붙이는 것보다 큰가**다
                # (2026-08-27 3차 검증). 총량 비례로 잡으면 대화가 커질수록
                # 과도하게 보수적이 된다 — 84만 토큰에서 4만을 버릴 수 있는데도
                # "20% 미만"이라 건너뛰었다. 요약이 차지할 자리(_SUMMARY_MAX_TOKENS)
                # 의 배수로 잡으면 상수를 바꿔도 자동으로 따라온다.
                dropped_tokens = self._estimate_tokens(messages[:split]) if split else 0
                if split > 0 and dropped_tokens < _SUMMARY_MAX_TOKENS * _SUMMARY_WORTH_RATIO:
                    logger.warning(
                        "Auto-compact(force) — 버릴 구간(%d 토큰)이 요약이 차지할 "
                        "자리보다 작다. 도구 결과 본문 절단으로 간다.",
                        dropped_tokens,
                    )
                    split = 0

                if split > 0:
                    recent = messages[split:]
                    dropped = messages[:split]
                    logger.warning(
                        "Auto-compact(force) — 턴 경계로 자를 수 없어 메시지 단위로 "
                        "내려간다(%d개 중 뒤 %d개 보존).",
                        len(messages),
                        len(recent),
                    )
                elif force and oversized:
                    # 메시지 단위로도 못 자른다 = 도구 결과 하나하나가 크다는 뜻이다.
                    # 메시지를 버리는 대신 **본문만** 잘라 낸다. 짝(assistant
                    # tool_calls ↔ tool_result)을 건드리지 않으면서 실제로 줄어드는
                    # 유일한 수단이다.
                    return self._hard_truncate_tool_results(messages, token_count)
                else:
                    # 자동 경로: 버릴 것이 없다 = 전부 보존 대상이다. 요약해도 얻을
                    # 게 없고, 오히려 요약 한 덩어리가 더 붙어 커진다
                    # (실측 -262/-333 의 형태).
                    logger.warning(
                        "Auto-compact 생략 — 보존 대상이 전체다(메시지 %d개). "
                        "요약해도 줄지 않는다.",
                        len(messages),
                    )
                    self._useless_compact_at = len(messages)
                    self._result_replaced = False
                    return messages

            # 모델에게 **버려질 앞부분**을 요약해 달라고 요청한다(비동기 스트림).
            summary = await self._get_model_summary(dropped)

            # 최종 형태: [요약 시스템 메시지] + [보존한 최근 턴들].
            result = [
                Message.system(f"[대화 요약]\n{summary}\n[요약 끝 — 여기서부터 계속]"),
                *recent,
            ]
            new_count = self._estimate_tokens(result)

            # ── 결과 검증 후에만 상태를 커밋한다 (2026-08-25) ──
            # 예전에는 검증 **전에** _compact_boundary/_compact_summary 를 썼다.
            # 그래서 요약이 원본보다 큰 경우에도 경계가 남았다. 실측 로그:
            #   08-24 03:07  60,953 → 61,215 (절약 -262)
            #   08-25 08:22  60,976 → 61,309 (절약 -333)
            # 줄지 않은 압축은 이득이 없으면서 상태만 남긴다. 남은 경계는 이후
            # 모든 요청의 메시지를 잘라내므로(_prepare 의 messages[boundary:]),
            # 이득 없는 압축이 **입력 유실**로 이어졌다.
            # force=True 는 제외한다. 그 경로는 query_loop 의 에러 복구(컨텍스트
            # 초과)에서 "무슨 수를 써서라도 줄여라"로 부르는 자리다. 여기서 원본을
            # 돌려주면 호출자가 같은 초과로 다시 실패한다. 실측 사고 2건은 모두
            # force=False(임계치 자동 압축)였다.
            if new_count >= token_count:
                if force:
                    # force 는 원본을 돌려줄 수 없다 — 호출자가 같은 초과로 다시
                    # 실패한다. 대신 짝을 건드리지 않는 본문 절단으로 내려간다.
                    logger.warning(
                        "Auto-compact(force) 무효 — 요약이 더 크다: %d → %d 토큰. "
                        "도구 결과 본문 절단으로 대체한다.",
                        token_count,
                        new_count,
                    )
                    return self._hard_truncate_tool_results(messages, token_count)

                logger.warning(
                    "Auto-compact 무효 — 결과가 더 크거나 같다: %d → %d 토큰. "
                    "원본을 유지하고 압축 상태를 남기지 않는다.",
                    token_count,
                    new_count,
                )
                # 대화가 늘기 전에는 같은 낭비를 반복하지 않는다.
                self._useless_compact_at = len(messages)
                self._result_replaced = False
                return messages

            # 경계를 "최근 턴 시작 지점"으로 옮기고, 그 앞은 요약으로 대체한다.
            # 인덱스와 함께 **보존한 첫 메시지의 id** 를 남긴다. 다음 턴에는 다른
            # 리스트가 들어오므로 id 로 다시 찾아야 한다(_resolve_boundary).
            self._compact_boundary = len(messages) - len(recent)
            self._compact_boundary_id = recent[0].id if recent else None
            self._compact_summary = summary
            self._total_compactions += 1
            # 압축이 성공했으니 무효 latch 를 푼다. 안 풀면 성공 압축이 리스트를
            # 크게 줄인 뒤 len(messages) 가 옛 latch 값 아래에 오래 머물러,
            # 그동안 자동 압축이 계속 건너뛰어진다(2026-08-27 리뷰 지적).
            self._useless_compact_at = None

            # 얼마나 줄었는지 로깅해 두면 나중에 압축 효과를 추적하기 쉽다.
            logger.info(
                f"Auto-compact 완료: {token_count} → {new_count} 토큰 "
                f"(절약: {token_count - new_count})"
            )

            # 모델 호출로 전체를 요약한 "가장 큰" 압축 — 표시 문구를 남긴다
            # (앞 단계(예산/스닙)가 남긴 문구가 있어도 이걸로 덮어쓴다).
            self._last_compaction = "대화 요약 생성(모델 호출)"

            self._result_replaced = True
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
            self._result_replaced = False
            return self._extract_recent_turns(messages, 1)

        logger.warning("긴급 압축: 최근 1개 턴만 보존")
        self._total_compactions += 1
        # 리스트를 최대로 줄이는 경로다. latch 는 옛 메시지 개수 기준이라 무효다
        # (성공 경로에만 해제를 넣어 두면 긴급 압축 뒤 자동 압축이 오래 막힌다).
        self._useless_compact_at = None
        self._result_replaced = True

        # 직전 턴 1개만 원본 보존, 나머지 전체는 도구·주제만 뽑은 한 줄 요약으로 압축.
        recent = self._extract_recent_turns(messages, 1)
        summary = self._rule_based_summary(messages)

        # 다음 apply_all이 이 경계 이후만 다루도록 상태를 갱신한다.
        # 인덱스와 함께 보존한 첫 메시지의 id 를 남긴다(_resolve_boundary 참조).
        self._compact_summary = summary
        self._compact_boundary = len(messages) - len(recent)
        self._compact_boundary_id = recent[0].id if recent else None

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

    def _resolve_boundary(self, messages: list[Message]) -> int:
        """경계를 **이번에 넘어온 리스트 기준**으로 다시 찾는다.

        ■ 왜 인덱스만으로는 안 되는가 (2026-08-27 3차 검증)
          `_compact_boundary` 는 `auto_compact_if_needed` 에 넘어온 리스트 기준
          인덱스다. 그런데 `query_loop.py:837-838` 은 이렇게 부른다.

              api_messages = context_manager.apply_all(state.messages)
              api_messages = await context_manager.auto_compact_if_needed(api_messages)

          즉 경계는 **apply_all 출력** 기준으로 잡히는데, 다음 턴의 apply_all 은
          그 인덱스를 **state.messages** 에 적용한다. 두 리스트는 길이가 다르다 —
          apply_all 이 요약을 앞에 붙이고 snip 이 여러 턴을 마커 하나로 접기
          때문이다. 실측(운영과 같은 흐름).

              턴2  state= 9  apply_all→7  boundary=1
              턴5  state=18  apply_all→7  boundary=4  → state[4] 는 assistant

          경계 앞의 user 질문이 요약에도 없이 조용히 잘려 나간다. 08-24·08-25 에
          관측된 입력 유실과 같은 계열이다.

        ■ 그래서 메시지 id 로 잡는다
          압축 시점에 "보존한 첫 메시지"의 id 를 같이 기억해 두고, 여기서 그 id 를
          이번 리스트에서 찾는다. 리스트가 달라도 같은 메시지를 가리킨다.

          못 찾으면 **자르지 않는다**(0). 맥락이 중복될 수는 있어도 입력이 사라지지는
          않는다 — 둘 중에는 중복이 낫다. 다음 압축에서 경계가 새로 잡힌다.
        """
        if not self._compact_boundary_id:
            # id 를 모르는 상태(구버전 상태 복원 등)에서는 종전 인덱스를 쓰되,
            # 범위를 벗어나면 자르지 않는다.
            return self._compact_boundary if self._compact_boundary < len(messages) else 0

        for i, msg in enumerate(messages):
            if msg.id == self._compact_boundary_id:
                return i

        logger.debug(
            "압축 경계 메시지를 이번 리스트에서 찾지 못했다(id=%s, 메시지 %d개). "
            "자르지 않는다 — 입력 유실보다 맥락 중복이 낫다.",
            self._compact_boundary_id,
            len(messages),
        )
        return 0

    def _message_level_split(self, messages: list[Message]) -> int:
        """턴 경계로 못 자를 때 쓸 **메시지 단위** 분할 지점을 찾는다.

        ■ 짝을 깨면 안 된다 (2026-08-27 리뷰 지적 N1)
          단순히 `messages[-1:]` 만 남기면, 마지막이 tool_result 일 때 짝이 되는
          assistant(tool_calls) 가 dropped 로 사라져 **고아 tool_result** 가 남는다.
          `inference.py` 가 이를 `{"role":"tool", "tool_call_id":...}` 로 변환하므로
          선행 assistant 없이 `role:"tool"` 만 있는 payload 가 나간다 — OpenAI 규약상
          무효한 순서다. 하필 컨텍스트 초과 복구 자리라, 컨텍스트 오류를 템플릿
          오류로 바꿔 버린다.

          그리고 이 폴백이 발동하는 상황에서 마지막 메시지는 실전에서 대개
          tool_result 다 — 도구를 실행해 결과를 붙인 직후 프롬프트가 넘치기 때문이다.

        Returns:
            dropped/recent 경계 인덱스. 0이면 "메시지 단위로도 자를 수 없다".
        """
        split = len(messages) - 1
        # 뒤에서부터 tool_result 를 지나 짝이 되는 assistant 까지 끌어온다.
        while split > 0 and _is_tool_result(messages[split]):
            split -= 1
        return max(0, split)

    def _hard_truncate_tool_results(
        self, messages: list[Message], token_count: int
    ) -> list[Message]:
        """최후 수단 — 메시지는 그대로 두고 **도구 결과 본문만** 잘라 낸다.

        메시지 단위로도 자를 수 없는 형태(질문 하나 + 거대한 도구 결과들)에서
        쓴다. 메시지를 버리면 짝이 깨지므로, 짝을 건드리지 않고 줄일 수 있는
        유일한 방법이 본문 절단이다.
        """
        # 1단계(도구 결과 예산)보다 **반드시 공격적이어야** 한다 — 1단계를 이미
        # 거치고도 넘쳤기 때문에 여기까지 온 것이다. 그래서 같은 설정값에서
        # 유도한다(1단계는 tool_result_budget * 3 글자, 여기는 그 1/4).
        # 하드코딩하면 YAML 을 바꿔도 이 경로만 따로 놀아 관계가 깨진다.
        hard_budget = max(500, self.tool_result_budget * 3 // 4)  # 글자. 앞 2/3 + 뒤 1/3.
        result: list[Message] = []
        truncated = 0
        for msg in messages:
            content = str(msg.content) if isinstance(msg.content, str) else msg.text_content
            if _is_tool_result(msg) and len(content) > hard_budget:
                head = hard_budget * 2 // 3
                tail = hard_budget - head
                result.append(
                    Message.tool_result(
                        msg.tool_use_id or "",
                        f"{content[:head]}\n\n... ({len(content):,}자 전체, "
                        f"긴급 절단) ...\n\n{content[-tail:]}",
                        msg.is_error or False,
                    )
                )
                truncated += 1
            else:
                result.append(msg)

        new_count = self._estimate_tokens(result)
        if truncated == 0 or new_count >= token_count:
            # 줄일 것이 없었다. 원본을 그대로 둔다 — 호출부가 상태를 지우지
            # 않도록 "교체 안 함"을 남긴다.
            logger.warning(
                "Auto-compact(force) — 도구 결과 절단으로도 줄지 않았다: %d → %d 토큰.",
                token_count,
                new_count,
            )
            self._result_replaced = False
            return messages

        logger.warning(
            "Auto-compact(force) — 메시지 단위로도 못 잘라 도구 결과 %d개를 "
            "본문 절단했다: %d → %d 토큰.",
            truncated,
            token_count,
            new_count,
        )
        self._total_compactions += 1
        self._useless_compact_at = None
        self._last_compaction = "긴급 절단 — 도구 결과 본문 축약"
        # ── 여기서는 "교체"를 표시하지 않는다 (2026-08-27 3차 검증 N4) ──
        # 이 경로는 리스트를 **교체한 것이 아니라 내용만 줄인 것**이다. 메시지
        # 개수·순서가 그대로라 기존 경계는 여전히 유효하고, 반환 리스트에는
        # 기존 _compact_summary 가 들어 있지 않다. 그런데 _result_replaced 를
        # 세우면 호출부의 mark_result_adopted() 가 경계와 요약을 지운다 —
        # 누적된 압축 상태가 사라져 다음 턴에 경계 이전 원문이 되살아난다.
        # 복구 자리에서 컨텍스트를 되돌리는 셈이라 재초과를 부른다.
        self._result_replaced = False
        return result

    def _summary_input_budget(self) -> int:
        """요약 프롬프트에 넣을 수 있는 글자 수 — 창 크기에 맞춰 정한다.

        상한 24,000자를 고정으로 쓰면 좁은 창 프로파일에서 요약 호출 자체가 창을
        넘는다(model_profiles.yaml 의 h200 16,384 / h100 8,192 토큰). 그런데 그
        호출이 일어나는 자리가 바로 컨텍스트 초과에서 회복하려는 자리다 — 넘치면
        예외로 떨어져 `_force_snip` 폴백을 타고, 그 폴백은 같은 형태의 입력에서
        줄이지 못한다. 복구가 복구를 막는 구조라 창에 종속시킨다.

        출력 토큰을 **먼저 뺀다** — 요약 호출도 입력 + 출력이 같은 창을 쓴다.
        빼지 않으면 rtx5090 프로파일(4,096)에서 하한 2,000자(≈1,600토큰)에
        출력 2,560을 더해 창을 넘는다(2026-08-27 리뷰 지적).
        그 뒤 한글 최악 1.25자/토큰을 가정해 남은 창의 절반(글자)까지만 쓴다.
        """
        usable = max(0, self.max_tokens - _SUMMARY_MAX_TOKENS)
        by_window = int(usable * _SUMMARY_INPUT_WINDOW_RATIO)
        # 하한은 두되, 창이 정말 좁으면 하한도 창을 넘지 않도록 같이 눌러 준다.
        floor = min(2_000, max(0, by_window))
        return max(floor, min(_SUMMARY_INPUT_BUDGET, by_window))

    async def _get_model_summary(self, messages: list[Message]) -> str:
        """
        [4단계 실동작] 모델(GPU 서버)에 실제로 요청해 대화를 짧게 요약한다.

        ■ 무엇을 요약해야 하는가 (2026-08-27 수정)
          예전에는 `messages[-20:]` 을 각 200자로 잘라 넣었다. 두 가지가 잘못이었다.

          ① **버려질 부분이 아니라 보존될 부분을 요약했다.** 압축은 앞쪽을 요약으로
             바꾸고 최근 턴은 원본으로 지킨다. 그런데 요약 입력이 "최근 20개"라,
             어차피 원본으로 남는 구간과 겹쳤다. 정작 사라지는 앞부분에 대해서는
             요약이 아무 말도 하지 않았다 — 요약의 존재 이유가 사라진 구조다.

          ② **입력이 대화의 3% 였다.** 압축은 55,296 토큰(≈13만 자 이상)에서
             걸리는데 요약 입력 상한은 20×200=4,000자였다. 실측에서 6~8만 토큰이
             234~1,061 토큰으로 떨어진 것은 "요약이 성기다"가 아니라 **요약이 볼
             수 있는 내용 자체가 없었다**는 뜻이다.

          그래서 호출부가 **버릴 구간만** 넘기고(dropped), 여기서는 문자 예산 안에서
          가능한 한 많이 담는다. 긴 메시지는 앞만 자르지 않고 앞뒤를 남긴다 —
          도구 결과는 끝부분에 결론이 오는 경우가 많다.

        Args:
            messages: 요약 대상. 압축에서 **버려질 구간**을 넘겨야 한다.
        """
        conversation_text: list[str] = []
        budget = self._summary_input_budget()
        # 메시지 하나에 줄 예산. 총 예산을 개수로 나누되, 너무 잘게 쪼개지지
        # 않도록 하한을 둔다.
        per_msg = max(400, budget // max(1, len(messages)))
        for msg in messages:
            role = msg.role if isinstance(msg.role, str) else msg.role.value
            # str(msg.content) 를 쓰면 assistant 블록이 파이썬 repr 로 들어간다.
            # 그 repr 이 요약을 무너뜨렸다 — _readable_text 주석 참조.
            content = _readable_text(msg)
            if len(content) > per_msg:
                # 앞뒤를 남긴다. 도구 결과·긴 답변은 결론이 끝에 오는 경우가 많아
                # 앞만 남기면 "무엇을 하려 했는지"만 남고 "어떻게 됐는지"가 사라진다.
                head = per_msg * 2 // 3
                tail = per_msg - head
                content = f"{content[:head]}\n…(중략)…\n{content[-tail:]}"
            conversation_text.append(f"[{role}]: {content}")

        joined = "\n".join(conversation_text)
        if len(joined) > budget:
            # ── 앞에서만 자르면 안 된다 (2026-08-27 리뷰 지적) ──
            # 메시지 하나에는 "앞만 자르지 않는다"를 적용해 놓고 합친 뒤에는 앞만
            # 남기면, 빠지는 것은 **보존 구간 바로 앞** 구간이다. 이후 턴이 가장
            # 많이 참조할 자리이고, 요약을 버릴 구간으로 옮긴 이번 수정의 취지가
            # 거기서 절반 무효화된다. 개별 메시지와 같은 규칙으로 가운데를 접는다.
            head = budget * 2 // 3
            tail = budget - head
            joined = f"{joined[:head]}\n…(중략 — 분량 초과로 일부 생략)…\n{joined[-tail:]}"

        prompt = (
            f"다음은 대화의 앞부분이며, 이제 원문 대신 이 요약으로 대체됩니다.\n"
            f"{_SUMMARY_TARGET_CHARS}자 이내로 요약하세요.\n"
            "포함할 것: 사용자의 요구사항, 확정된 결정과 그 이유, 진행 중인 작업,\n"
            "  파일 경로·식별자처럼 뒤에서 다시 참조될 구체값.\n"
            "생략할 것: 도구 실행 결과의 원문, 반복된 확인 대화.\n\n" + joined
        )

        # 모델 스트림에서 텍스트 조각(text_delta)만 모아 요약을 조립한다.
        # temperature=0.3으로 낮춰 사실적이고 안정적인 요약을 유도한다.
        summary_parts: list[str] = []
        truncated = False
        async for event in self.model_provider.stream(
            messages=[Message.user(prompt)],
            system_prompt="당신은 대화 요약 전문가입니다. 간결하고 사실적으로 요약하세요.",
            tools=None,
            max_tokens=_SUMMARY_MAX_TOKENS,
            temperature=0.3,
        ):
            # StreamEvent.type도 Enum/문자열 양쪽일 수 있으므로 정규화한다.
            event_type = event.type if isinstance(event.type, str) else event.type.value
            if event_type == "text_delta" and event.text:
                summary_parts.append(event.text)
            elif event_type == "message_stop":
                stop = event.stop_reason
                truncated = (stop.value if hasattr(stop, "value") else stop) == "max_tokens"

        summary = "".join(summary_parts)
        # 상한에 물려 문장 중간에서 잘렸으면 남긴다. 요약의 끝은 "진행 중인 작업"이
        # 놓이는 자리라 조용히 잘리면 손실이 크다 — 예전에는 stop_reason 을 아예
        # 보지 않아 잘려도 알 수 없었다(2026-08-27 리뷰 지적).
        if truncated:
            logger.warning(
                "요약이 출력 상한(%d 토큰)에 물려 잘렸다 — %d자 생성. "
                "_SUMMARY_MAX_TOKENS 를 올리거나 목표 길이를 줄여야 한다.",
                _SUMMARY_MAX_TOKENS,
                len(summary),
            )

        # 조각이 하나도 없으면(스트림 비정상 등) 실패 표시를 반환한다.
        return summary or "[요약 생성 실패]"

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
        self._useless_compact_at = None
        self._result_replaced = True
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
                text = _readable_text(msg)[:50]
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

        # ── user 메시지가 n개 미만인 경우 (2026-08-26 수정) ──
        # 예전에는 break 없이 끝나 start_idx 가 len(messages) 로 남아 **빈 리스트**를
        # 돌려줬다. 주석에는 "호출 측이 이를 감안"이라 적혀 있었지만 감안하는 호출부가
        # 하나도 없었다(이 파일 안 4곳 전부).
        #
        # 그래서 auto_compact 가 이렇게 동작했다.
        #   recent = []  →  result = [요약] 하나뿐  →  boundary = len(messages)
        #   = **사용자의 현재 질문까지 통째로 버리고 요약만 모델에 보낸다.**
        # 그 요약도 최근 20개 메시지를 각 200자로 잘라 만든 것이라, 8만 자 입력의
        # 요약은 사실상 앞부분 몇 천 자의 요약이다. 실측 로그에서 6~8만 토큰이
        # 234~1,061 토큰으로 떨어진 항목들이 이 경로다.
        #
        # "최근 n턴을 남긴다"는 요청에 턴이 n개보다 적다면 답은 0개가 아니라
        # **있는 것 전부**다. user 를 하나도 못 찾은 경우도 마찬가지 — 턴 경계를
        # 모를 뿐이지 버리라는 뜻이 아니다.
        if user_count < n:
            return list(messages)
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
                user_text = _readable_text(msg)[:40]
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


def _readable_text(msg: Message) -> str:
    """
    Message를 "사람이 읽는 텍스트"로 바꾼다 — 요약 프롬프트에 넣을 때 쓴다.

    ■ 왜 필요한가 (2026-08-27, 실모델 검증에서 발견)
      `str(msg.content)` 를 쓰면 안 된다. assistant 메시지의 content는 항상
      `list[ContentBlock]` 이라(`Message.assistant()` 가 그렇게 정규화한다)
      `str()` 이 **파이썬 repr** 을 만든다.

          [TextBlock(type='text', text='중간 작업 27 결과입니다. ...')]

      이 repr이 그대로 요약 프롬프트에 들어갔다. 실모델(A.X-4.0) 검증에서
      모델이 요약 대신 이 구조를 그대로 되뱉었고, 앞부분의 파일 경로·식별자는
      요약에 하나도 살아남지 못했다. 요약 품질 저하의 실제 원인 중 하나다.

      텍스트만 뽑으면 도구 호출 사실이 사라지므로, 도구 이름은 한 줄로 덧붙인다
      ("진행 중인 작업"은 요약이 반드시 담아야 할 항목이다).
    """
    text = msg.text_content

    # 도구 호출 이름 수집 — 역직렬화 시점에 따라 dict 형태일 수도 있다.
    names: list[str] = []
    thinking: list[str] = []
    if isinstance(msg.content, list):
        for block in msg.content:
            if isinstance(block, ToolUseBlock):
                names.append(block.name)
            elif isinstance(block, ThinkingBlock):
                thinking.append(block.thinking)
            elif isinstance(block, dict) and block.get("type") == "tool_use":
                names.append(block.get("name", ""))
            elif isinstance(block, dict) and block.get("type") == "thinking":
                thinking.append(block.get("thinking", ""))

    # thinking 만 있는 메시지는 text_content 가 빈 문자열이라 요약 입력에서
    # 통째로 사라진다. 현재 배선에서는 도달 불가지만(core/thinking/ 미배선),
    # TIER_S 재사용 시 되살아나므로 폴백을 둔다. text 가 있으면 쓰지 않는다.
    if not text and thinking:
        text = "\n".join(t for t in thinking if t)

    if names:
        marker = f"(도구 호출: {', '.join(n for n in names if n)})"
        return "\n".join([text, marker]) if text else marker
    return text
