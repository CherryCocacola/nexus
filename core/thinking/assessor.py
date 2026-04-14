"""
복잡도 평가기 — 사용자 입력의 복잡도를 0.0~1.0 스코어로 평가한다.

키워드 매칭, 메시지 길이, 코드 블록 존재, 증폭 패턴을 조합하여
입력이 얼마나 복잡한 사고를 필요로 하는지 판단한다.

이 스코어는 ThinkingStrategy 선택에 사용된다:
  - < 0.3  → DIRECT (단순 응답)
  - < 0.6  → HIDDEN_COT (2-pass 분석)
  - < 0.8  → SELF_REFLECT (3-pass 검증)
  - >= 0.8 → MULTI_AGENT (다중 에이전트)
"""

from __future__ import annotations

import logging
import re

from core.message import Message

logger = logging.getLogger("nexus.thinking.assessor")


class ComplexityAssessor:
    """
    사용자 입력의 복잡도를 0.0~1.0 스코어로 평가한다.

    평가 기준:
      1. 키워드 매칭 — 특정 작업 유형 키워드의 가중치 합산
      2. 메시지 길이 — 긴 메시지는 복잡한 요구사항일 가능성이 높다
      3. 코드 블록 — 코드가 포함되면 기술적 분석이 필요하다
      4. 증폭 패턴 — "multi-file", "across modules" 등 복합 작업 표현
      5. 대화 컨텍스트 — 이전 대화가 길수록 복잡한 맥락이다
    """

    # ─── 키워드 사전: 작업 유형별 복잡도 가중치 ───
    COMPLEXITY_KEYWORDS: dict[str, float] = {
        # 높은 복잡도 (0.3~0.4) — 구조적 변경, 설계 작업
        "architecture": 0.4,
        "아키텍처": 0.4,
        "refactor": 0.3,
        "리팩토링": 0.3,
        "리팩터링": 0.3,
        "debug": 0.3,
        "디버그": 0.3,
        "디버깅": 0.3,
        "migrate": 0.3,
        "마이그레이션": 0.3,
        "optimize": 0.3,
        "최적화": 0.3,
        "design": 0.3,
        "설계": 0.3,
        "performance": 0.3,
        "성능": 0.3,
        # 중간 복잡도 (0.2) — 구현 작업
        "implement": 0.2,
        "구현": 0.2,
        "fix": 0.2,
        "수정": 0.2,
        "버그": 0.2,
        "bug": 0.2,
        "integrate": 0.2,
        "통합": 0.2,
        "test": 0.2,
        "테스트": 0.2,
        "security": 0.2,
        "보안": 0.2,
        "concurrent": 0.2,
        "동시성": 0.2,
        "async": 0.2,
        "비동기": 0.2,
        # 낮은 복잡도 (0.1) — 단순 작업
        "create": 0.1,
        "생성": 0.1,
        "add": 0.1,
        "추가": 0.1,
        "update": 0.1,
        "변경": 0.1,
        "remove": 0.1,
        "삭제": 0.1,
        "rename": 0.1,
        "explain": 0.1,
        "설명": 0.1,
        "read": 0.05,
        "확인": 0.05,
    }

    # ─── 증폭 패턴: 복합 작업을 나타내는 표현 ───
    # (정규식 패턴, 증폭 배수)
    AMPLIFIER_PATTERNS: list[tuple[str, float]] = [
        (r"multi[- ]?file", 1.4),
        (r"여러\s*파일", 1.4),
        (r"across\s+(modules?|files?|packages?)", 1.3),
        (r"전체\s*(모듈|파일|패키지)", 1.3),
        (r"integration", 1.3),
        (r"end[- ]?to[- ]?end", 1.3),
        (r"from\s+scratch", 1.2),
        (r"처음부터", 1.2),
        (r"전면\s*재작성", 1.5),
        (r"rewrite", 1.4),
        (r"모든\s*(파일|모듈|컴포넌트)", 1.3),
        (r"시스템\s*전체", 1.4),
        (r"backward[- ]?compat", 1.2),
        (r"하위\s*호환", 1.2),
        (r"thread[- ]?safe", 1.2),
        (r"스레드\s*안전", 1.2),
    ]

    # ─── 메시지 길이 기반 스코어 보정 상수 ───
    # 긴 메시지일수록 복잡한 요구사항이다
    _LENGTH_THRESHOLDS: list[tuple[int, float]] = [
        (500, 0.15),  # 500자 이상: +0.15
        (200, 0.10),  # 200자 이상: +0.10
        (100, 0.05),  # 100자 이상: +0.05
    ]

    def assess(
        self,
        message: str,
        context: list[Message] | None = None,
    ) -> float:
        """
        복잡도 스코어(0.0~1.0)를 반환한다.

        Args:
            message: 사용자 입력 텍스트
            context: 이전 대화 메시지 목록 (선택)

        Returns:
            0.0(매우 단순) ~ 1.0(매우 복잡) 사이의 스코어

        평가 절차:
          1. 키워드 매칭 → 기본 스코어 합산
          2. 메시지 길이 보정
          3. 코드 블록 존재 시 가산
          4. 증폭 패턴 적용 (곱셈)
          5. 대화 컨텍스트 보정
          6. 0.0~1.0 범위로 클램프
        """
        # 빈 메시지는 최소 복잡도
        if not message or not message.strip():
            return 0.0

        # 소문자로 통일하여 키워드 매칭 (한글은 대소문자가 없지만 영어 혼용 대비)
        lower_msg = message.lower()

        # ── 1단계: 키워드 매칭 → 기본 스코어 ──
        score = self._score_keywords(lower_msg)

        # ── 2단계: 메시지 길이 보정 ──
        score += self._score_length(message)

        # ── 3단계: 코드 블록 존재 시 +0.1 ──
        if self._has_code_block(message):
            score += 0.1

        # ── 4단계: 증폭 패턴 적용 (곱셈) ──
        amplifier = self._calculate_amplifier(lower_msg)
        score *= amplifier

        # ── 5단계: 대화 컨텍스트 보정 ──
        if context:
            score += self._score_context(context)

        # ── 6단계: 0.0~1.0 클램프 ──
        result = max(0.0, min(1.0, score))

        logger.debug(
            f"복잡도 평가: score={result:.2f}, "
            f"keywords={self._score_keywords(lower_msg):.2f}, "
            f"length={self._score_length(message):.2f}, "
            f"amplifier={amplifier:.2f}, "
            f"context_bonus={self._score_context(context) if context else 0:.2f}"
        )

        return result

    def _score_keywords(self, lower_msg: str) -> float:
        """
        키워드 매칭으로 기본 스코어를 계산한다.
        동일 키워드가 여러 번 등장해도 한 번만 카운트한다.
        """
        score = 0.0
        matched: set[str] = set()
        for keyword, weight in self.COMPLEXITY_KEYWORDS.items():
            # 이미 매칭된 키워드는 건너뛴다 (중복 방지)
            if keyword in matched:
                continue
            if keyword in lower_msg:
                score += weight
                matched.add(keyword)
        return score

    def _score_length(self, message: str) -> float:
        """
        메시지 길이에 따라 보정 스코어를 반환한다.
        긴 메시지는 복잡한 요구사항일 가능성이 높다.
        """
        length = len(message)
        for threshold, bonus in self._LENGTH_THRESHOLDS:
            if length >= threshold:
                return bonus
        return 0.0

    @staticmethod
    def _has_code_block(message: str) -> bool:
        """
        메시지에 코드 블록(```...```)이 포함되어 있는지 확인한다.
        코드가 포함되면 기술적 분석이 필요하므로 복잡도가 올라간다.
        """
        return "```" in message

    def _calculate_amplifier(self, lower_msg: str) -> float:
        """
        증폭 패턴을 검사하여 최대 증폭 배수를 반환한다.
        여러 패턴이 매칭되면 가장 높은 배수를 사용한다.
        """
        max_amplifier = 1.0
        for pattern, multiplier in self.AMPLIFIER_PATTERNS:
            if re.search(pattern, lower_msg, re.IGNORECASE):
                max_amplifier = max(max_amplifier, multiplier)
        return max_amplifier

    @staticmethod
    def _score_context(context: list[Message]) -> float:
        """
        대화 컨텍스트 길이에 따라 보정 스코어를 반환한다.
        이전 대화가 길수록 복잡한 맥락에서 작업 중이다.

        - 5턴 이상: +0.1
        - 10턴 이상: +0.15
        - 20턴 이상: +0.2
        """
        turn_count = len(context)
        if turn_count >= 20:
            return 0.2
        elif turn_count >= 10:
            return 0.15
        elif turn_count >= 5:
            return 0.1
        return 0.0
