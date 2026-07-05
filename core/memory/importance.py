"""
중요도 평가기 — 메모리 한 건의 "중요도"를 0.0~1.0 숫자로 매긴다.

[이 파일이 하는 일 — 3줄 요약]
  - 대화/작업 중 생성된 메모리(텍스트)가 얼마나 중요한지 점수로 환산한다.
  - 그 점수를 근거로 "단기 메모리(Redis)에만 둘지, 장기 메모리(PG)로
    올릴지"를 판단한다. 즉 메모리 승격 게이트의 판정 기준을 제공한다.

[중요도를 정하는 두 축]
  1. 콘텐츠 키워드: 텍스트에 특정 키워드가 들어 있으면 가/감점한다.
  2. 메모리 타입: 타입별로 기본 중요도를 살짝 보정한다.

[승격 규칙 한 줄]
  중요도 > 0.6 이면(그 외 몇 가지 조건 포함) 단기→장기 메모리로 승격된다.

[키워드 분류 — 대략적인 점수대]
  - 높은 중요도(0.7~0.9): error, fix, bug, architecture, decision,
    security, 장애 등 (문제·해결·설계·보안 신호)
  - 중간 중요도(0.4~0.6): config, update, refactor, test, 설정 등
  - 낮은 중요도(0.1~0.3): ls, cat, status, cd, pwd, 확인 등 (루틴 명령)

[주요 구성 요소]
  - HIGH/MEDIUM/LOW_IMPORTANCE_KEYWORDS: 가·감점 키워드 사전(모듈 상수)
  - TYPE_IMPORTANCE_BIAS: 메모리 타입별 기본 보정값
  - ImportanceAssessor.assess():    콘텐츠+타입 → 0.0~1.0 점수
  - ImportanceAssessor.should_promote(): 메모리 항목 → 승격 여부(bool)

[의존 관계]
  - core.memory.types 의 MemoryEntry / MemoryType 를 입력 타입으로 사용.
  - 이 평가기는 보통 메모리 매니저(단기↔장기 승격 로직)에서 호출한다.

[설계 결정 — 왜 이렇게 만들었나]
  - LLM 호출 없이 규칙 기반으로 평가: 에어갭(폐쇄망) 환경에서 외부
    모델 없이도 빠르고 결과가 항상 일정(결정론적)하게 나온다.
  - 키워드 기반 휴리스틱: 단순하지만 실전에서 충분히 효과적.
  - 향후 LLM 기반 평가로 확장 가능(ModelProvider 주입 여지를 남겨둠).

작성자: 이현수 / 작성일: 2026-07-05
"""

from __future__ import annotations

import logging
import re

from core.memory.types import MemoryEntry, MemoryType

logger = logging.getLogger("nexus.memory.importance")


# ─────────────────────────────────────────────
# 중요도 키워드 사전
# ─────────────────────────────────────────────
# 아래 세 리스트는 "이 단어가 들어 있으면 가점/감점" 하는 규칙 테이블이다.
# assess()에서 소문자 부분 문자열 매칭(kw in content_lower)으로 사용하므로,
# 예를 들어 "deprecat"는 deprecated/deprecation 을 모두 걸러낸다(어간 매칭).
# 한글/영문을 함께 넣어 두 언어 콘텐츠 모두 커버한다.

# 높은 중요도 키워드 — 이 키워드가 포함되면 중요도가 크게 올라간다.
# 문제 발생·해결·설계 판단·보안·중대 변경처럼 "나중에 꼭 다시 볼" 신호들.
HIGH_IMPORTANCE_KEYWORDS: list[str] = [
    # 에러/장애 관련
    "error",
    "exception",
    "traceback",
    "failure",
    "crash",
    "bug",
    "장애",
    "에러",
    "오류",
    "버그",
    # 수정/해결
    "fix",
    "hotfix",
    "patch",
    "resolve",
    "workaround",
    "수정",
    "해결",
    "고침",
    # 아키텍처/설계 결정
    "architecture",
    "decision",
    "design",
    "tradeoff",
    "trade-off",
    "아키텍처",
    "설계",
    "결정",
    "판단",
    # 보안
    "security",
    "vulnerability",
    "permission",
    "auth",
    "보안",
    "취약점",
    "권한",
    # 중요 변경
    "breaking",
    "migration",
    "deprecat",
    "critical",
    "마이그레이션",
    "중대한",
]

# 중간 중요도 키워드 — 설정/변경/테스트/성능/배포처럼 참고 가치는 있으나
# 최상위 신호까지는 아닌 작업들. 높은 키워드보다 가점 폭이 작다.
MEDIUM_IMPORTANCE_KEYWORDS: list[str] = [
    "config",
    "configuration",
    "setting",
    "update",
    "upgrade",
    "refactor",
    "test",
    "testing",
    "coverage",
    "performance",
    "optimization",
    "cache",
    "deploy",
    "deployment",
    "설정",
    "업데이트",
    "리팩터",
    "테스트",
    "성능",
    "배포",
]

# 낮은 중요도 키워드 — 조회/출력 위주의 반복적인 루틴 작업.
# 여기 걸리면 감점하여, 사소한 명령 기록이 장기 메모리를 채우지 않게 막는다.
LOW_IMPORTANCE_KEYWORDS: list[str] = [
    "ls",
    "cat",
    "head",
    "tail",
    "pwd",
    "cd",
    "echo",
    "status",
    "log",
    "list",
    "show",
    "display",
    "print",
    "확인",
    "조회",
    "목록",
    "출력",
]

# 타입별 기본 중요도 보정값 — 같은 내용이라도 메모리 "종류"에 따라
# 보존 가치가 다르므로, 타입별로 점수를 약간 더하거나 그대로 둔다.
# assess() 마지막 단계에서 이 값을 점수에 합산한다(감점은 없음).
TYPE_IMPORTANCE_BIAS: dict[MemoryType, float] = {
    MemoryType.EPISODIC: 0.0,  # 기본값 유지
    MemoryType.SEMANTIC: 0.1,  # 지식은 약간 높음
    MemoryType.PROCEDURAL: 0.05,  # 절차는 약간 높음
    MemoryType.USER_PROFILE: 0.15,  # 사용자 프로필은 더 높음
    MemoryType.FEEDBACK: 0.1,  # 피드백은 약간 높음
}


class ImportanceAssessor:
    """
    메모리의 중요도를 0.0~1.0으로 평가하는 규칙 기반 평가기.

    상태(state)를 갖지 않는 순수 계산 객체다. 인스턴스를 하나 만들어
    여러 번 재사용해도 되고, 호출 간 부작용이 없어 스레드/코루틴에서
    나눠 써도 안전하다.

    제공 메서드:
      - assess(content, memory_type): 텍스트+타입 → 0.0~1.0 점수
      - should_promote(entry):        메모리 항목 → 단기→장기 승격 여부

    규칙 기반 휴리스틱으로 빠르게 평가하며, LLM 호출 없이 에어갭
    환경에서도 항상 동일한 결과를 내며 안정적으로 동작한다.
    """

    def assess(self, content: str, memory_type: MemoryType) -> float:
        """
        콘텐츠와 타입을 근거로 중요도 스코어(0.0~1.0)를 계산한다.

        [핵심 흐름]
          기본 점수 0.3에서 출발해, 키워드 가·감점 → 길이 보너스 →
          기술적 내용 보너스 → 타입 보정 순으로 점수를 누적한 뒤,
          마지막에 0.0~1.0 범위로 잘라(clamp) 소수 둘째 자리로 반올림.

        [계산 방법 — 단계별]
          1. 기본 점수: 0.3
          2. 높은 키워드 매칭: 하나당 +0.15 (합산 상한 +0.45)
          3. 중간 키워드 매칭: 하나당 +0.08 (합산 상한 +0.24)
          4. 낮은 키워드 매칭: 하나당 -0.1 (합산 하한 -0.2, 감점)
          5. 콘텐츠 길이 보너스: 긴 텍스트일수록 약간 가산
          6. 코드/스택 트레이스 감지: 기술적 내용이면 +0.1
          7. 타입별 보정값 합산
          8. 최종 범위 제한: 0.0 ~ 1.0 으로 클램핑 후 반올림

        Args:
            content: 중요도를 매길 메모리 콘텐츠 텍스트.
            memory_type: 메모리 타입(에피소딕/시맨틱/프로필 등).

        Returns:
            0.0~1.0 범위의 중요도 스코어(소수 둘째 자리 반올림).
            보통 이 값이 0.6을 넘으면 장기 메모리 승격 후보가 된다.
        """
        # 키워드 매칭은 대소문자를 무시해야 하므로 미리 소문자로 통일해 둔다.
        content_lower = content.lower()
        score = 0.3  # 어떤 메모리든 출발점으로 삼는 기본 점수

        # 1. 높은 중요도 키워드 매칭 — 포함된 키워드 "개수"를 센다.
        #    개수 × 0.15 를 더하되, 한 건에 너무 쏠리지 않게 +0.45로 상한.
        high_matches = sum(1 for kw in HIGH_IMPORTANCE_KEYWORDS if kw.lower() in content_lower)
        score += min(high_matches * 0.15, 0.45)

        # 2. 중간 중요도 키워드 매칭 — 가점 폭(0.08)과 상한(0.24)이 더 작다.
        medium_matches = sum(1 for kw in MEDIUM_IMPORTANCE_KEYWORDS if kw.lower() in content_lower)
        score += min(medium_matches * 0.08, 0.24)

        # 3. 낮은 중요도 키워드 매칭 — 루틴 작업 신호라 오히려 감점한다.
        #    개수 × 0.1 을 빼되, 과도한 감점을 막기 위해 -0.2까지만 반영.
        low_matches = sum(1 for kw in LOW_IMPORTANCE_KEYWORDS if kw.lower() in content_lower)
        score -= min(low_matches * 0.1, 0.2)

        # 4. 콘텐츠 길이 보너스
        #    긴 텍스트는 보통 더 많은 맥락/정보를 담으므로 약간 가산한다.
        #    100자 이하: 0, 100~500자: +0.05, 500자 초과: +0.1
        content_len = len(content)
        if content_len > 500:
            score += 0.1
        elif content_len > 100:
            score += 0.05

        # 5. 코드 블록/스택 트레이스 감지 — 이런 기술적 내용은 재참조 가치가
        #    높다고 보고 +0.1. 마크다운 코드펜스(```), 파이썬 예외 헤더
        #    "Traceback", 트레이스백의 파일 표시 'File "' 중 하나만 걸려도 인정.
        if re.search(r"```|Traceback|File \"", content):
            score += 0.1

        # 6. 타입별 보정 — 사전에 없는 타입이면 0.0(보정 없음)으로 처리.
        bias = TYPE_IMPORTANCE_BIAS.get(memory_type, 0.0)
        score += bias

        # 7. 범위 제한: 누적 과정에서 0 미만/1 초과가 될 수 있으므로 0.0~1.0로 클램핑.
        final_score = max(0.0, min(1.0, score))

        logger.debug(
            "중요도 평가: type=%s, score=%.2f (high=%d, mid=%d, low=%d)",
            memory_type,
            final_score,
            high_matches,
            medium_matches,
            low_matches,
        )
        return round(final_score, 2)

    def should_promote(self, entry: MemoryEntry) -> bool:
        """
        단기(Redis)→장기(PG) 메모리 승격 여부를 판단한다.

        [왜 필요한가]
          단기 메모리는 용량이 제한적이고 휘발적이다. 그중 오래 남길
          가치가 있는 항목만 골라 장기 저장소로 올려야 한다. 그 "골라내는"
          기준을 이 메서드가 제공한다. 보통 메모리 매니저가 승격 시점마다
          각 항목을 이 함수에 넣어 True인 것만 장기로 옮긴다.

        [승격 조건 — 하나라도 충족하면 승격(OR 조건)]
          1. importance > 0.6      : assess()가 매긴 중요도가 임계 초과
          2. access_count >= 3     : 자주 다시 찾은 메모리(재사용성 높음)
          3. USER_PROFILE 타입      : 사용자 프로필은 무조건 장기 보존

        Args:
            entry: 승격 여부를 판단할 MemoryEntry(중요도·접근횟수·타입 보유).

        Returns:
            True이면 장기 메모리로 승격 대상, False이면 단기에 유지.
        """
        # 조건 1: 중요도 기준 — 내용 자체가 충분히 중요하면 바로 승격.
        if entry.importance > 0.6:
            logger.debug(
                "승격 결정: id=%s (importance=%.2f > 0.6)",
                entry.id,
                entry.importance,
            )
            return True

        # 조건 2: 접근 빈도 기준 — 중요도는 낮아도 반복해서 찾은 메모리는
        #         실사용 가치가 있다고 보고 승격한다(3회 이상).
        if entry.access_count >= 3:
            logger.debug(
                "승격 결정: id=%s (access_count=%d >= 3)",
                entry.id,
                entry.access_count,
            )
            return True

        # 조건 3: 사용자 프로필은 항상 장기 저장 — 개인화의 근간이 되는
        #         정보라 중요도·접근횟수와 무관하게 무조건 보존한다.
        if entry.memory_type == MemoryType.USER_PROFILE:
            logger.debug("승격 결정: id=%s (USER_PROFILE 타입)", entry.id)
            return True

        # 위 세 조건 모두 해당 없음 → 승격하지 않고 단기 메모리에 그대로 둔다.
        return False
