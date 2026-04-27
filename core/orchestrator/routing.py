"""
쿼리 라우팅 결정 로직 — v7.0 Part 2.5 + 멀티테넌시 통합 (2026-04-21 리팩토링).

`QueryEngine.submit_message()`에서 한 덩어리로 섞여있던 라우팅 로직을 독립
모듈로 분리한다. 기존 동작을 100% 유지하면서 다음을 제공한다:

  - `RoutingDecision` (frozen dataclass) — 라우팅 결과의 불변 객체
  - `RoutingResolver` — classify + 프로필 선택 + 테넌트 override 통합

설계 원칙:
  - 순수 함수/클래스 — 네트워크·DB 호출 없음, 단위 테스트 용이
  - Pydantic·Config 외부 의존 최소화
  - 기존 `classify_query()` / `_resolve_profile()`는 그대로 유지(하위 호환)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from core.config import RoutingConfig, TenantConfig

logger = logging.getLogger("nexus.orchestrator.routing")


# ─────────────────────────────────────────────
# 분류기 인터페이스 (권장 4 — 전략 패턴)
# ─────────────────────────────────────────────
# HeuristicClassifier는 기본 구현. 향후 LLM 분류기·학교별 커스터마이즈 등을
# 플러그인할 수 있도록 ABC로 열어둔다.
class QueryClassifier:
    """질의 → ("KNOWLEDGE" | "TOOL") 분류기 인터페이스."""

    def classify(self, user_input: str) -> str:
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
        if not self._routing.enabled:
            return "TOOL"
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

    @property
    def routing_enabled(self) -> bool:
        """라우팅이 실제로 적용됐는지 (model_override 여부로 판단)."""
        return self.model_override is not None

    @property
    def inject_knowledge_rag(self) -> bool:
        """KB RAG 주입 여부 — KNOWLEDGE만 True, CHAT/TOOL은 False (Part 2.5.9)."""
        return self.query_class == "KNOWLEDGE"


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

        query_class = self._classifier.classify(user_input)
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

        model = profile.model
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

        return RoutingDecision(
            query_class=query_class,
            model_override=model,
            temperature=profile.temperature,
            max_tokens_cap=profile.max_tokens,
            enable_thinking=profile.enable_thinking,
            allowed_knowledge_sources=allowed_sources,
            tenant_id=tenant.id if tenant else None,
            profile_name=query_class.lower(),
        )


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
