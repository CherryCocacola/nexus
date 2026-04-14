"""
중요도 평가기 — 메모리의 중요도를 0.0~1.0으로 평가한다.

메모리의 중요도는 두 가지 기준으로 결정된다:
  1. 콘텐츠 키워드: 특정 키워드가 포함되면 중요도가 올라간다
  2. 메모리 타입: 타입별 기본 중요도 보정

중요도 > 0.6이면 단기→장기 메모리로 승격된다.

키워드 분류:
  - 높은 중요도 (0.7~0.9): error, fix, bug, architecture, decision, security, 장애
  - 중간 중요도 (0.4~0.6): config, update, refactor, test, 설정
  - 낮은 중요도 (0.1~0.3): ls, cat, status, cd, pwd, 확인

설계 결정:
  - LLM 호출 없이 규칙 기반 평가: 에어갭 환경에서 빠르고 안정적
  - 키워드 기반 휴리스틱: 단순하지만 충분히 효과적
  - 향후 LLM 기반 평가로 확장 가능 (ModelProvider 주입)
"""

from __future__ import annotations

import logging
import re

from core.memory.types import MemoryEntry, MemoryType

logger = logging.getLogger("nexus.memory.importance")


# ─────────────────────────────────────────────
# 중요도 키워드 사전
# ─────────────────────────────────────────────

# 높은 중요도 키워드 — 이 키워드가 포함되면 중요도가 크게 올라간다
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

# 중간 중요도 키워드
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

# 낮은 중요도 키워드 — 루틴 작업
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

# 타입별 기본 중요도 보정값
TYPE_IMPORTANCE_BIAS: dict[MemoryType, float] = {
    MemoryType.EPISODIC: 0.0,  # 기본값 유지
    MemoryType.SEMANTIC: 0.1,  # 지식은 약간 높음
    MemoryType.PROCEDURAL: 0.05,  # 절차는 약간 높음
    MemoryType.USER_PROFILE: 0.15,  # 사용자 프로필은 더 높음
    MemoryType.FEEDBACK: 0.1,  # 피드백은 약간 높음
}


class ImportanceAssessor:
    """
    메모리의 중요도를 0.0~1.0으로 평가한다.

    규칙 기반 휴리스틱으로 빠르게 평가하며,
    LLM 호출 없이 에어갭 환경에서 안정적으로 동작한다.
    """

    def assess(self, content: str, memory_type: MemoryType) -> float:
        """
        콘텐츠와 타입 기반으로 중요도 스코어를 계산한다.

        계산 방법:
          1. 기본 점수: 0.3
          2. 높은 키워드 매칭: +0.15 (각, 최대 0.45)
          3. 중간 키워드 매칭: +0.08 (각, 최대 0.24)
          4. 낮은 키워드 매칭: -0.1 (각, 최대 -0.2)
          5. 콘텐츠 길이 보너스: 긴 텍스트일수록 약간 높음
          6. 타입별 보정값 적용
          7. 최종 범위: 0.0 ~ 1.0으로 클램핑

        Args:
            content: 메모리 콘텐츠 텍스트
            memory_type: 메모리 타입

        Returns:
            0.0~1.0 범위의 중요도 스코어
        """
        content_lower = content.lower()
        score = 0.3  # 기본 점수

        # 1. 높은 중요도 키워드 매칭
        high_matches = sum(1 for kw in HIGH_IMPORTANCE_KEYWORDS if kw.lower() in content_lower)
        score += min(high_matches * 0.15, 0.45)

        # 2. 중간 중요도 키워드 매칭
        medium_matches = sum(1 for kw in MEDIUM_IMPORTANCE_KEYWORDS if kw.lower() in content_lower)
        score += min(medium_matches * 0.08, 0.24)

        # 3. 낮은 중요도 키워드 매칭 (감점)
        low_matches = sum(1 for kw in LOW_IMPORTANCE_KEYWORDS if kw.lower() in content_lower)
        score -= min(low_matches * 0.1, 0.2)

        # 4. 콘텐츠 길이 보너스
        #    긴 텍스트는 더 많은 정보를 포함하므로 약간 가산
        #    100자 이하: 0, 100~500자: +0.05, 500자 이상: +0.1
        content_len = len(content)
        if content_len > 500:
            score += 0.1
        elif content_len > 100:
            score += 0.05

        # 5. 코드 블록/스택 트레이스 감지 — 기술적 내용은 중요도 높음
        if re.search(r"```|Traceback|File \"", content):
            score += 0.1

        # 6. 타입별 보정
        bias = TYPE_IMPORTANCE_BIAS.get(memory_type, 0.0)
        score += bias

        # 7. 범위 제한: 0.0 ~ 1.0
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
        단기→장기 메모리 승격 여부를 판단한다.

        승격 조건 (하나라도 충족하면 승격):
          1. importance > 0.6
          2. access_count >= 3 (자주 접근되는 메모리)
          3. USER_PROFILE 타입 (항상 장기 저장)

        Args:
            entry: 판단할 MemoryEntry

        Returns:
            True이면 장기 메모리로 승격
        """
        # 조건 1: 중요도 기준
        if entry.importance > 0.6:
            logger.debug(
                "승격 결정: id=%s (importance=%.2f > 0.6)",
                entry.id,
                entry.importance,
            )
            return True

        # 조건 2: 접근 빈도 기준
        if entry.access_count >= 3:
            logger.debug(
                "승격 결정: id=%s (access_count=%d >= 3)",
                entry.id,
                entry.access_count,
            )
            return True

        # 조건 3: 사용자 프로필은 항상 장기 저장
        if entry.memory_type == MemoryType.USER_PROFILE:
            logger.debug("승격 결정: id=%s (USER_PROFILE 타입)", entry.id)
            return True

        return False
