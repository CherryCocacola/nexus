"""
쿼리 라우팅 결정 로직 — v7.0 Part 2.5 + 멀티테넌시 통합 (2026-04-21 리팩토링).

[이 파일이 하는 일]
사용자가 보낸 질의 한 건을 보고 "이 질의를 어떤 성격으로 처리할지"를 정하는
모듈이다. 성격은 세 가지 라벨로 나뉜다.
  - KNOWLEDGE : 일반 지식 질문 → 지식베이스(KB) RAG를 붙여 답변
  - TOOL      : 도구 호출·프로젝트 작업 → 도구 실행 경로로 흐름
  - CHAT      : 인사·잡담 → KB를 붙이지 않고 가볍게 응답
이 라벨에 더해, 어떤 모델을 쓸지(model_override)·온도·최대 토큰·샘플링
파라미터까지 한 번에 묶어 결정한다. 그 결과 묶음이 `RoutingDecision`이다.

[왜 별도 모듈인가]
원래 `QueryEngine.submit_message()` 안에 라우팅 로직이 한 덩어리로 섞여
있었다. 그걸 이 모듈로 떼어내 테스트하기 쉽고 읽기 쉽게 만들었다. 기존
동작은 100% 그대로 유지하면서 아래 두 축을 제공한다:

  - `RoutingDecision` (frozen dataclass) — 라우팅 결과의 불변 객체
  - `RoutingResolver` — 분류(classify) + 프로필 선택 + 테넌트 override 통합

[주요 구성 요소]
  - QueryClassifier / HeuristicClassifier : 질의 → 라벨 분류 (전략 패턴)
  - build_classifier() : 설정값(classifier_type)에 맞는 분류기 생성
  - RoutingDecision : Tier 2 이하로 넘길 라우팅 파라미터 묶음
  - RoutingResolver : 위 전부를 조립해 최종 결정을 내리는 집결 지점
  - classify_query() / _resolve_profile() : 구버전 호환용 모듈 함수

[호출 관계]
`QueryEngine.submit_message()`가 RoutingResolver.resolve()를 호출하고,
반환된 RoutingDecision을 dispatcher/query_loop에 전달한다. 여기서 정한
샘플링 파라미터는 query_loop → model_dispatcher → inference.stream()을
거쳐 최종 vLLM payload까지 그대로 흘러간다(passthrough).

설계 원칙:
  - 순수 함수/클래스 — 네트워크·DB 호출 없음, 단위 테스트 용이
  - Pydantic·Config 외부 의존 최소화
  - 기존 `classify_query()` / `_resolve_profile()`는 그대로 유지(하위 호환)

작성자: 이현수 / 작성일: 2026-07-05
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from core.config import RoutingConfig, SelfConsistencyConfig, TenantConfig

logger = logging.getLogger("nexus.orchestrator.routing")


# ─────────────────────────────────────────────
# 자기일관성(Self-Consistency) G3 사실형 패턴 게이트 (Point 4.3)
# ─────────────────────────────────────────────
# "짧은 사실 답"을 기대하는 질문 패턴을 사전 컴파일 정규식으로 판정한다
# (HeuristicClassifier와 동일한 방식). 서술형 질문은 majority 합의가 본질적으로
# 어려우므로(설계 §3.3), 이 패턴에 매치되고 길이가 짧을 때만 SC를 태운다.
# 이 게이트가 실질적 비용 방어선이다 — 오탐 시 토큰만 3배 쓰고 이득이 없다.
_SC_FACTUAL_PATTERN = re.compile(
    r"언제|몇\s*(?:년|명|개|살|km|%|위|번|시|월|일)"
    r"|누구|어디|얼마|무슨\s*(?:년도|색)|수도는|이름은"
    r"|when|who|where|how\s+many|how\s+much",
    re.IGNORECASE,
)


# ─────────────────────────────────────────────
# 분류기 인터페이스 (권장 4 — 전략 패턴)
# ─────────────────────────────────────────────
# HeuristicClassifier는 기본 구현. 향후 LLM 분류기·학교별 커스터마이즈 등을
# 플러그인할 수 있도록 ABC로 열어둔다.
class QueryClassifier:
    """질의 → ("KNOWLEDGE" | "TOOL" | "CHAT") 분류기 인터페이스.

    실제 분류 알고리즘은 하위 구현체(HeuristicClassifier 등)가 담당한다.
    이 ABC는 "어떤 분류기를 쓰든 classify(텍스트) → 라벨" 계약만 강제한다.
    """

    def classify(self, user_input: str) -> str:
        # 인터페이스만 정의 — 반드시 하위 클래스에서 구현해야 한다.
        # 미구현 분류기를 실수로 그대로 쓰면 NotImplementedError로 빨리 터뜨려
        # "조용히 잘못된 라우팅"이 발생하지 않게 한다(fail-fast).
        raise NotImplementedError


class HeuristicClassifier(QueryClassifier):
    """
    기본 휴리스틱 분류기.

    규칙 (우선순위 순, Part 2.5.9 v0.14.6):
      1. `routing.enabled=False` → 항상 "TOOL" (비상 스위치)
      2. `len(input) >= long_input_threshold` → "TOOL" (문서 첨부 간주)
      3. `tool_keywords`(substring) / `tool_word_patterns`(word boundary) /
         `tool_regex_patterns`(정규식) 중 하나라도 매치 → "TOOL"
      4. `len(input) <= chat_max_length` AND `chat_keywords` 중 하나 매치 →
         "CHAT" (인사·잡담, KB RAG 미주입)
      5. 그 외 → "KNOWLEDGE"

    매칭 타입 (2026-04-22 리팩토링 6):
      - substring: 가장 빠름, 한국어에 적합. 'file'이 'filename'에 오탐될 수 있음.
      - word: 단어 경계(\\b) 매칭. 영어 용어에 권장. 'file' → "filename" 오탐 방지.
      - regex: 자유 정규식. 복잡한 패턴(예: `Read\\s*\\(`)에 사용.

    CHAT은 길이 임계와 chat_keywords의 AND 조건으로만 트리거된다 — 긴 질의
    안에 우연히 "안녕"이 들어가도 CHAT이 되지 않는다. TOOL 키워드가 우선
    검사되므로 "이 파일 안녕히 처리해줘" 같은 입력은 TOOL로 정확히 분류.
    """

    def __init__(self, routing: RoutingConfig) -> None:
        # 분류에 쓰는 모든 키워드/패턴을 생성자에서 "한 번만" 전처리해 둔다.
        # classify()는 질의마다 호출되는 hot path이므로, 소문자 변환·정규식
        # 컴파일 같은 비싼 작업을 매 호출이 아니라 여기서 미리 끝내 둔다.
        self._routing = routing
        # substring 키워드는 소문자로 미리 변환해 매 호출마다 반복 변환 제거.
        self._lowered_keywords = tuple(
            kw.lower() for kw in (routing.tool_keywords or [])
        )
        # CHAT 키워드도 동일하게 lower-case 사전 변환 (Part 2.5.9).
        chat_kws = getattr(routing, "chat_keywords", []) or []
        self._lowered_chat_keywords = tuple(kw.lower() for kw in chat_kws)
        self._chat_max_length = int(getattr(routing, "chat_max_length", 0) or 0)
        # word-boundary 패턴 사전 컴파일 — 각 단어 패턴을 `\bpattern\b` 형태로.
        # re.IGNORECASE로 대소문자 무시, 잘못된 패턴은 건너뛰며 경고 로그.
        word_patterns = getattr(routing, "tool_word_patterns", []) or []
        self._word_res: tuple[re.Pattern[str], ...] = tuple(
            p for p in (
                self._compile_word(w) for w in word_patterns
            ) if p is not None
        )
        # 정규식 패턴 사전 컴파일 — 잘못된 regex는 건너뛰며 경고 로그.
        regex_patterns = getattr(routing, "tool_regex_patterns", []) or []
        self._regex_res: tuple[re.Pattern[str], ...] = tuple(
            p for p in (
                self._compile_regex(r) for r in regex_patterns
            ) if p is not None
        )

    @staticmethod
    def _compile_word(word: str) -> re.Pattern[str] | None:
        """단어를 `\\bword\\b` 형태로 컴파일. 실패 시 None."""
        if not word:
            return None
        try:
            return re.compile(rf"\b{re.escape(word)}\b", re.IGNORECASE)
        except re.error as e:
            logger.warning("tool_word_patterns 컴파일 실패 (%r): %s", word, e)
            return None

    @staticmethod
    def _compile_regex(pattern: str) -> re.Pattern[str] | None:
        """자유 정규식 컴파일. 실패 시 None."""
        if not pattern:
            return None
        try:
            return re.compile(pattern, re.IGNORECASE)
        except re.error as e:
            logger.warning("tool_regex_patterns 컴파일 실패 (%r): %s", pattern, e)
            return None

    def classify(self, user_input: str) -> str:
        """질의 텍스트 하나를 받아 "KNOWLEDGE"/"TOOL"/"CHAT" 중 하나로 분류한다.

        클래스 docstring에 적힌 우선순위 규칙을 그대로 코드로 옮긴 것이다.
        위에서 아래로 검사하다가 처음 매치되는 규칙의 라벨을 즉시 반환한다
        (early return) — 그래서 검사 순서 자체가 곧 우선순위다.
        """
        # 0) 비상 스위치 — 라우팅 기능을 통째로 끈 경우. 분류를 시도하지 않고
        #    무조건 TOOL로 보내 도구 호출 경로(가장 안전한 기본)로 흐르게 한다.
        if not self._routing.enabled:
            return "TOOL"
        # 1) 너무 긴 입력은 문서를 통째로 붙여넣은 것으로 간주 — 도구 작업일
        #    가능성이 높으니 TOOL. (KB RAG에 거대 입력을 태우지 않으려는 의도도 있음)
        if len(user_input) >= self._routing.long_input_threshold:
            return "TOOL"
        lowered = user_input.lower()
        # 1) substring 매칭 — 가장 빠름. TOOL 키워드는 CHAT보다 항상 우선
        #    (예: "이 파일 안녕히 처리해줘" → TOOL).
        for kw in self._lowered_keywords:
            if kw and kw in lowered:
                return "TOOL"
        # 2) word-boundary 매칭 — 영어 용어 오탐 방지
        for wre in self._word_res:
            if wre.search(user_input):
                return "TOOL"
        # 3) regex 매칭 — 복잡 패턴
        for rre in self._regex_res:
            if rre.search(user_input):
                return "TOOL"
        # 4) CHAT 분류 (Part 2.5.9 v0.14.6)
        #    짧은 입력 + 인사 어휘 — 둘 중 하나라도 안 맞으면 KNOWLEDGE.
        #    chat_max_length가 0이면 CHAT 분류 자체를 비활성 (운영자가 끌 수 있음).
        if (
            self._chat_max_length > 0
            and len(user_input.strip()) <= self._chat_max_length
        ):
            stripped_lower = user_input.strip().lower()
            for kw in self._lowered_chat_keywords:
                if kw and kw in stripped_lower:
                    return "CHAT"
        return "KNOWLEDGE"


# ─────────────────────────────────────────────
# 분류기 레지스트리 — 전략 패턴의 확장 지점
# ─────────────────────────────────────────────
# `RoutingConfig.classifier_type`의 문자열 값으로 어떤 QueryClassifier 구현체를
# 사용할지 결정한다. 새 분류기를 추가하려면 여기에 등록만 하면 된다 — 호출
# 지점(QueryEngine, tests 등)을 건드리지 않아도 된다.
#
# 향후 후보:
#   - "llm_classifier": 경량 LLM(Scout 4B)에게 질의 분류 위임
#   - "embedding_classifier": 임베딩 유사도 기반 KNN 분류
_CLASSIFIER_REGISTRY: dict[str, type[QueryClassifier]] = {
    "heuristic": HeuristicClassifier,
}


def build_classifier(routing: RoutingConfig) -> QueryClassifier:
    """RoutingConfig.classifier_type에 맞는 분류기 인스턴스를 생성한다.

    알 수 없는 타입이 지정되면 경고 로그 후 HeuristicClassifier로 폴백한다
    (운영 중 config 오타로 서버가 죽지 않도록 방어).
    """
    cls_type = (routing.classifier_type or "heuristic").lower()
    klass = _CLASSIFIER_REGISTRY.get(cls_type)
    if klass is None:
        logger.warning(
            "알 수 없는 classifier_type=%r — heuristic으로 폴백", cls_type
        )
        klass = HeuristicClassifier
    return klass(routing)


# ─────────────────────────────────────────────
# RoutingDecision — 한 턴의 라우팅 결과 불변 객체
# ─────────────────────────────────────────────
@dataclass(frozen=True)
class RoutingDecision:
    """`Tier 2` 이하에 전달할 라우팅 파라미터를 한 번에 묶는다.

    `query_class`는 관측·로깅용이며, `model_override` 등이 실제 실행에 쓰인다.
    `allowed_knowledge_sources`는 KNOWLEDGE 질의일 때만 의미 있다 (KB RAG 필터).

    query_class 값:
      - "KNOWLEDGE" : 일반 지식 QA, KB RAG 주입
      - "TOOL"      : 도구 호출 / 프로젝트 작업
      - "CHAT"      : 인사·잡담, KB RAG 미주입 (Part 2.5.9, v0.14.6)
    """

    query_class: str                                   # "KNOWLEDGE" | "TOOL" | "CHAT"
    model_override: str | None
    temperature: float
    max_tokens_cap: int | None
    enable_thinking: bool | None
    allowed_knowledge_sources: list[str] | None        # KB 필터용 (None = 전체)
    tenant_id: str | None = None                       # 로깅/감사용
    profile_name: str = ""                             # 프로필 이름 (debug)
    # ── 샘플링 파라미터 (degeneration/무한 반복 방지) ──────────────────
    # 왜 여기에: 이 dataclass는 frozen이라 기본값 있는 필드를 무기본값 필드 뒤에
    # 배치해야 한다(파이썬 dataclass 제약). 그래서 profile_name 뒤에 둔다.
    # 기본값은 전부 "비활성"값 — enabled=False 폴백 경로처럼 신규 필드를 생략해도
    # 동작이 바뀌지 않도록 보장한다(하위 호환).
    top_p: float = 1.0
    repetition_penalty: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    # ── 자기일관성(Self-Consistency) 파라미터 (Point 4.3) ──────────────────
    # 기본값은 전부 "비활성"(sc_n=1). 3중 게이트(G1~G3)를 모두 통과할 때만
    # RoutingResolver.resolve()가 sc_n=config.n으로 채운다. sc_n<=1이면 Tier2/Tier3가
    # SC 로직을 통째로 우회하므로 기존 동작이 1비트도 바뀌지 않는다(무회귀).
    sc_n: int = 1                          # 1 = SC 비활성(하위 호환 기본값)
    sc_min_agreement: int = 2              # 최소 합의 표(N=3의 과반)
    sc_short_answer_max_chars: int = 80    # 이 이하면 exact majority, 초과 시 임베딩 클러스터
    sc_similarity_threshold: float = 0.90  # 서술형 임베딩 클러스터 임계(폴백 경로)

    @property
    def routing_enabled(self) -> bool:
        """라우팅이 실제로 적용됐는지 (model_override 여부로 판단)."""
        return self.model_override is not None

    @property
    def inject_knowledge_rag(self) -> bool:
        """KB RAG 주입 여부 — KNOWLEDGE만 True, CHAT/TOOL은 False (Part 2.5.9)."""
        return self.query_class == "KNOWLEDGE"

    @property
    def self_consistency_active(self) -> bool:
        """이번 턴에 자기일관성(SC) 다중표본 경로가 발동하는지 (Point 4.3).

        sc_n>1일 때만 True — Tier 2(query_loop)가 이 값으로 SC 버퍼링/합의를 켠다.
        기본 sc_n=1(비활성)이면 항상 False라 기존 단일 경로 그대로다(무회귀).
        """
        return self.sc_n > 1


# ─────────────────────────────────────────────
# RoutingResolver — 모든 로직의 집결 지점
# ─────────────────────────────────────────────
class RoutingResolver:
    """분류 + 프로필 선택 + 테넌트 override를 한 곳에서 수행한다.

    `QueryEngine.submit_message()`에서 호출되어 `RoutingDecision`을 돌려준다.
    호출자는 이 결과만 보고 dispatcher/query_loop에 파라미터를 전달한다.
    """

    # KNOWLEDGE/CHAT 질의에만 테넌트 model_override를 적용한다. TOOL 질의는 Phase
    # LoRA(nexus-phaseN)가 tool_call XML 포맷 학습을 담고 있어 override하면 도구 호출이
    # 깨지기 때문 (2026-04-21 멀티테넌시 설계 결정). CHAT은 KNOWLEDGE와 동일한
    # 베이스 모델 경로를 쓰므로 같은 분기에 포함 (Part 2.5.9 v0.14.6).
    _OVERRIDE_ON_CLASSES = frozenset({"KNOWLEDGE", "CHAT"})

    def __init__(
        self,
        routing: RoutingConfig,
        classifier: QueryClassifier | None = None,
    ) -> None:
        self._routing = routing
        # classifier를 명시 주입하지 않으면 config.classifier_type에 따라 자동 생성
        self._classifier = classifier or build_classifier(routing)

    def resolve(
        self,
        user_input: str,
        tenant: TenantConfig | None = None,
    ) -> RoutingDecision:
        """
        사용자 입력 + (선택) 테넌트를 받아 최종 RoutingDecision을 반환한다.

        [전체 흐름]
          1) 라우팅이 꺼져 있으면(enabled=False) 곧바로 기본값 결정 반환
          2) 분류기로 질의 라벨(KNOWLEDGE/TOOL/CHAT) 결정
          3) 라벨에 맞는 RoutingProfile 선택(모델·온도·샘플링 묶음)
          4) 테넌트가 있으면 model_override / allowed_sources 덮어쓰기
          5) 위 값을 모두 담아 RoutingDecision 생성

        매개변수:
          - user_input : 사용자가 보낸 원문 질의 텍스트
          - tenant     : 멀티테넌시 컨텍스트(없으면 단일 테넌트로 동작)
        반환:
          - RoutingDecision : Tier 2 이하로 넘길 라우팅 파라미터 묶음

        `routing.enabled=False`면 프로필 치환 없이 프로바이더 기본 설정을 쓴다
        (model_override=None, temperature=0.7 기본, allowed_sources=None).
        """
        if not self._routing.enabled:
            return RoutingDecision(
                query_class="TOOL",     # enabled=False일 때는 분류 의미 없음 — TOOL로 통일
                model_override=None,
                temperature=0.7,
                max_tokens_cap=None,
                enable_thinking=False,
                allowed_knowledge_sources=None,
                tenant_id=tenant.id if tenant else None,
                profile_name="disabled",
            )

        # 먼저 질의를 분류한다. 이 라벨이 아래의 모든 분기(프로필 선택, 테넌트
        # override 적용 여부, KB 필터 적용 여부)를 결정하는 기준이 된다.
        query_class = self._classifier.classify(user_input)
        # 분류 결과에 대응하는 RoutingProfile을 고른다. 각 프로필은 model 이름과
        # temperature/max_tokens/샘플링 파라미터 묶음을 담고 있다.
        # CHAT 프로필은 v0.14.6 신규 — 구버전 RoutingConfig 객체에 chat_mode가
        # 없을 수도 있으므로 getattr로 안전하게 폴백한다 (없으면 KNOWLEDGE 사용).
        if query_class == "KNOWLEDGE":
            profile = self._routing.knowledge_mode
        elif query_class == "CHAT":
            profile = (
                getattr(self._routing, "chat_mode", None)
                or self._routing.knowledge_mode
            )
        else:
            profile = self._routing.tool_mode

        # 기본 모델은 프로필이 지정한 것 — 아래 테넌트 override가 있으면 덮어쓴다.
        model = profile.model
        # KB 검색 소스 필터. KNOWLEDGE 질의 + 테넌트 제한이 있을 때만 채워진다.
        allowed_sources: list[str] | None = None

        # 테넌트 override — KNOWLEDGE/CHAT에 적용 (TOOL은 Phase LoRA 보존)
        if (
            tenant is not None
            and query_class in self._OVERRIDE_ON_CLASSES
            and getattr(tenant, "model_override", None)
        ):
            model = tenant.model_override
            logger.info(
                "라우팅(tenant): tenant=%s → model=%s",
                tenant.id, tenant.model_override,
            )

        # 테넌트 allowed_sources — KNOWLEDGE에서만 KB 필터로 쓰인다.
        # CHAT은 PromptAssembler가 KB 단계 자체를 스킵하므로 의미 없음.
        if tenant is not None and query_class == "KNOWLEDGE":
            src = getattr(tenant, "allowed_knowledge_sources", None) or []
            allowed_sources = list(src) if src else None

        # ── 자기일관성(SC) 게이팅 판정 (Point 4.3, G1~G3) ──────────────────
        # 기본 샘플링 값은 선택된 프로필 값. 3중 게이트를 모두 통과하면 SC 전용
        # 온도/top_p/max_tokens로 치환하고 sc_n을 config.n으로 채운다(설계 §5.3).
        temperature = profile.temperature
        top_p = profile.top_p
        max_tokens_cap: int | None = profile.max_tokens
        sc_n = 1
        sc_min_agreement = 2
        sc_short = 80
        sc_sim = 0.90
        sc = self._resolve_sc_gate(user_input, query_class)
        if sc is not None:
            # 게이트 통과 — SC 발동. 샘플링을 SC 전용값으로 치환(표본 다양성 확보).
            sc_n = sc.n
            temperature = sc.temperature
            top_p = sc.top_p
            max_tokens_cap = sc.max_tokens
            sc_min_agreement = sc.min_agreement
            sc_short = sc.short_answer_max_chars
            sc_sim = sc.similarity_threshold
            logger.info(
                "라우팅(SC): class=%s → n=%d, temp=%.2f, max_tokens=%d (사실 교차검증)",
                query_class, sc_n, temperature, max_tokens_cap,
            )

        return RoutingDecision(
            query_class=query_class,
            model_override=model,
            temperature=temperature,
            max_tokens_cap=max_tokens_cap,
            enable_thinking=profile.enable_thinking,
            allowed_knowledge_sources=allowed_sources,
            tenant_id=tenant.id if tenant else None,
            profile_name=query_class.lower(),
            # 선택된 프로필의 샘플링 파라미터를 그대로 실어 보낸다.
            # 여기서부터 query_loop → model_dispatcher → inference.stream()까지
            # passthrough로 흘러가 최종 vLLM payload에 반영된다(degeneration 방지).
            top_p=top_p,
            repetition_penalty=profile.repetition_penalty,
            frequency_penalty=profile.frequency_penalty,
            presence_penalty=profile.presence_penalty,
            # SC 파라미터 — 게이트 미통과 시 sc_n=1(비활성)이라 무회귀.
            sc_n=sc_n,
            sc_min_agreement=sc_min_agreement,
            sc_short_answer_max_chars=sc_short,
            sc_similarity_threshold=sc_sim,
        )

    def _resolve_sc_gate(
        self, user_input: str, query_class: str
    ) -> SelfConsistencyConfig | None:
        """자기일관성 3중 게이트(G1~G3)를 검사해 통과 시 SC 설정을, 아니면 None 반환.

        게이트(설계 §1.2 — 모두 통과해야 SC 경로 진입):
          G1. 설정      : config.self_consistency.enabled == True (기본 False)
          G2. 질의 클래스 : query_class ∈ apply_classes (기본 ["KNOWLEDGE"])
          G3. 사실형 패턴 : factual_gate on이면, 질문 길이 상한 이하 + 사실형 정규식 매치

        구버전 RoutingConfig 객체에 self_consistency 필드가 없을 수 있어 getattr로
        안전하게 폴백한다(없으면 SC 미발동 — fail-closed).
        """
        sc = getattr(self._routing, "self_consistency", None)
        # G1: 마스터 스위치 — 미설정/비활성이면 즉시 미발동(fail-closed).
        if sc is None or not sc.enabled:
            return None
        # G2: 질의 클래스 제한 — TOOL/CHAT은 원리적으로 부적합해 제외.
        if query_class not in sc.apply_classes:
            return None
        # G3: 사실형 패턴 게이트(선택). 서술형은 majority 합의가 어려워 차단한다.
        if sc.factual_gate:
            # 긴 복합 질문은 서술형 가능성이 높아 길이 상한으로 먼저 거른다.
            if len(user_input) > sc.factual_max_question_chars:
                return None
            # 짧은 사실 답을 기대하는 패턴이 아니면 SC 미발동(비용 방어선).
            if not _SC_FACTUAL_PATTERN.search(user_input):
                return None
        return sc


# ─────────────────────────────────────────────
# 하위 호환 — 기존 모듈 레벨 함수 유지
# ─────────────────────────────────────────────
def classify_query(user_input: str, routing: RoutingConfig) -> str:
    """하위 호환용 — 모듈 함수 스타일 분류기.

    기존 테스트와 외부 호출이 직접 이 함수를 쓰므로 계속 export한다.
    내부는 HeuristicClassifier로 구현이 통일됐다.
    """
    return HeuristicClassifier(routing).classify(user_input)


def _resolve_profile(query_class: str, routing: RoutingConfig) -> Any:
    """하위 호환용 — 분류 결과 → RoutingProfile 직접 반환.

    CHAT 분류는 v0.14.6 신규. 구버전 routing 객체에 chat_mode가 없으면
    knowledge_mode로 폴백한다 (안전).
    """
    if query_class == "KNOWLEDGE":
        return routing.knowledge_mode
    if query_class == "CHAT":
        return getattr(routing, "chat_mode", None) or routing.knowledge_mode
    return routing.tool_mode
