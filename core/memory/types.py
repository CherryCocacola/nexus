"""
메모리 타입 시스템 — 메모리의 모든 데이터 구조를 정의한다.

Nexus의 메모리 시스템은 5가지 타입을 지원한다:
  - EPISODIC: 특정 이벤트/대화 기억 (반감기 7일)
  - SEMANTIC: 일반 지식/사실 (반감기 90일)
  - PROCEDURAL: 절차적 지식, 도구 사용 패턴 (반감기 30일)
  - USER_PROFILE: 사용자 선호/스타일 (반감기 365일)
  - FEEDBACK: 피드백 기록 (반감기 14일)

각 타입마다 시간 감쇠 반감기(DECAY_HALF_LIFE)가 다르다.
오래되고 접근 빈도가 낮은 메모리는 자동으로 감쇠하여 정리된다.

설계 결정:
  - Pydantic v2 BaseModel 사용: 직렬화 + 검증 자동화
  - importance (0.0~1.0): 메모리의 중요도. 단기→장기 승격 기준에 사용
  - embedding: pgvector 벡터 검색용 임베딩 (Optional)
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ─────────────────────────────────────────────
# 메모리 타입 열거형
# ─────────────────────────────────────────────
class MemoryType(str, Enum):
    """
    메모리의 종류를 구분한다.
    각 타입마다 감쇠 반감기(DECAY_HALF_LIFE)가 다르다.
    """

    EPISODIC = "episodic"  # 특정 이벤트/대화 기억
    SEMANTIC = "semantic"  # 일반 지식/사실
    PROCEDURAL = "procedural"  # 절차적 지식 (도구 사용 패턴)
    USER_PROFILE = "user_profile"  # 사용자 선호/스타일
    FEEDBACK = "feedback"  # 피드백 기록


# ─────────────────────────────────────────────
# 메모리 엔트리 (하나의 메모리 단위)
# ─────────────────────────────────────────────
class MemoryEntry(BaseModel):
    """
    메모리 저장소의 기본 단위.

    하나의 '기억'을 나타내며, 단기(Redis) 및 장기(PostgreSQL) 저장소 모두에서 사용한다.
    importance 값에 따라 단기→장기 승격 여부가 결정된다.
    embedding은 pgvector 기반 벡터 유사도 검색에 사용된다.
    """

    # 고유 식별자 — UUID 앞 12자리 (충분히 고유하면서 짧다)
    id: str = Field(default_factory=lambda: str(uuid.uuid4().hex[:12]))

    # 메모리 종류 (EPISODIC, SEMANTIC, PROCEDURAL, USER_PROFILE, FEEDBACK)
    memory_type: MemoryType

    # 실제 기억 내용 (텍스트)
    content: str

    # 검색/조회용 키 (예: "user_preference_language", "tool_pattern_bash")
    key: str = ""

    # 분류/검색용 태그 (예: ["architecture", "decision"])
    tags: list[str] = Field(default_factory=list)

    # 중요도 (0.0~1.0) — 0.6 이상이면 장기 메모리로 승격
    importance: float = Field(default=0.5, ge=0.0, le=1.0)

    # 접근 횟수 — 자주 접근되는 메모리는 감쇠가 느려진다
    access_count: int = 0

    # 생성 시각
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    # 마지막 접근 시각 — 감쇠 계산의 기준
    last_accessed: datetime = Field(default_factory=lambda: datetime.now(UTC))

    # 벡터 임베딩 (pgvector용, e5-large 등으로 생성)
    embedding: list[float] | None = None

    # 추가 메타데이터 (세션 ID, 출처 도구 이름 등)
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = {"use_enum_values": True}


# ─────────────────────────────────────────────
# 타입별 감쇠 반감기 (일 단위)
# ─────────────────────────────────────────────
# 왜 반감기를 사용하는가:
#   지수 감쇠(exponential decay)로 오래된 메모리의 유효 중요도가 자연스럽게 감소한다.
#   반감기가 긴 타입(USER_PROFILE=365일)은 오래 유지되고,
#   짧은 타입(EPISODIC=7일)은 빨리 잊혀진다.
DECAY_HALF_LIFE: dict[MemoryType, float] = {
    MemoryType.EPISODIC: 7.0,  # 대화 기억은 1주일 반감기
    MemoryType.SEMANTIC: 90.0,  # 지식/사실은 3개월 반감기
    MemoryType.PROCEDURAL: 30.0,  # 절차 지식은 1개월 반감기
    MemoryType.USER_PROFILE: 365.0,  # 사용자 프로필은 1년 반감기
    MemoryType.FEEDBACK: 14.0,  # 피드백은 2주 반감기
}


# ─────────────────────────────────────────────
# 메모리 검색 결과 (점수 포함)
# ─────────────────────────────────────────────
class MemorySearchResult(BaseModel):
    """
    메모리 검색 결과 한 건.
    entry와 함께 유사도 점수(score)를 포함한다.
    """

    entry: MemoryEntry
    score: float = 0.0  # 유사도 또는 관련도 점수 (높을수록 관련성 높음)
