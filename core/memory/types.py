"""
메모리 타입 시스템 — Nexus 메모리 시스템의 모든 데이터 구조를 한곳에서 정의한다.

[이 파일이 하는 일]
Nexus는 대화·지식·습관 등을 '기억'으로 저장한다. 이 모듈은 그 기억을
표현하는 공통 자료구조(스키마)를 정의하는 곳으로, 단기 저장소(Redis)와
장기 저장소(PostgreSQL + pgvector) 양쪽에서 똑같이 재사용된다.
실제 저장/조회 로직은 여기 없고, 오직 "데이터의 모양"만 규정한다.

[메모리 5가지 타입 — 각각 시간 감쇠 반감기가 다름]
  - EPISODIC     : 특정 이벤트/대화 기억            (반감기 7일)
  - SEMANTIC     : 일반 지식/사실                    (반감기 90일)
  - PROCEDURAL   : 절차적 지식, 도구 사용 패턴        (반감기 30일)
  - USER_PROFILE : 사용자 선호/스타일                (반감기 365일)
  - FEEDBACK     : 피드백 기록                       (반감기 14일)

각 타입은 아래 DECAY_HALF_LIFE에 정의된 반감기를 가진다. 오래되고 접근
빈도가 낮은 기억은 유효 중요도가 자연스럽게 줄어들어 정리 대상이 된다.

[주요 구성 요소]
  - MemoryType         : 메모리 종류를 나타내는 문자열 Enum
  - MemoryEntry        : 기억 한 건을 표현하는 핵심 모델(저장 단위)
  - DECAY_HALF_LIFE    : 타입별 감쇠 반감기(일 단위) 매핑 테이블
  - MemorySearchResult : 검색 결과 한 건(기억 + 유사도 점수)

[설계 결정]
  - Pydantic v2 BaseModel 채택: 직렬화(JSON) + 입력 검증을 자동으로 처리
  - importance(0.0~1.0): 기억의 중요도. 단기→장기 승격 판단 기준으로 쓰임
  - embedding: pgvector 벡터 유사도 검색을 위한 임베딩 값(없을 수 있어 Optional)

작성자: 이현수 / 작성일: 2026-07-05
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
    메모리의 종류를 구분하는 열거형.

    str을 함께 상속하므로 각 멤버는 곧 문자열 값이기도 하다. 덕분에 JSON
    직렬화나 DB 저장 시 별도 변환 없이 "episodic" 같은 문자열로 다뤄진다.
    타입마다 아래 DECAY_HALF_LIFE 테이블에서 서로 다른 감쇠 반감기를 가진다.
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
    메모리 저장소의 기본 단위 — '기억 한 건'을 표현하는 핵심 모델.

    단기(Redis)와 장기(PostgreSQL) 저장소 양쪽에서 동일하게 사용된다. 즉
    저장 위치와 무관하게 기억의 데이터 모양은 이 모델 하나로 통일된다.

    핵심 필드의 역할:
      - importance: 이 값에 따라 단기→장기 승격 여부가 결정된다(0.6 이상 승격).
      - embedding : pgvector 기반 벡터 유사도 검색에 쓰이는 임베딩 벡터.
      - access_count / last_accessed: 시간 감쇠 계산의 입력값으로 쓰인다.

    참고: model_config의 use_enum_values=True 때문에, memory_type에 Enum
    멤버를 넣어도 실제 저장되는 값은 "episodic" 같은 문자열이 된다.
    """

    # 고유 식별자 — UUID4의 16진수 문자열 앞 12자리만 사용한다.
    # 왜 잘라 쓰나: 전체 UUID는 너무 길다. 12자리면 충돌 확률이 매우 낮으면서
    # 로그/DB에서 다루기 짧다. default_factory라 엔트리마다 자동 생성된다.
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

    # 접근 횟수 — 이 기억이 조회된 누적 횟수. 자주 접근될수록 실제 감쇠를
    # 완화하는 근거로 쓰인다(중요한 기억은 오래 살아남게 하려는 의도).
    access_count: int = 0

    # 생성 시각
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    # 마지막 접근 시각 — 감쇠 계산의 기준
    last_accessed: datetime = Field(default_factory=lambda: datetime.now(UTC))

    # 벡터 임베딩 (pgvector용, e5-large 등으로 생성)
    embedding: list[float] | None = None

    # 추가 메타데이터 (세션 ID, 출처 도구 이름 등)
    metadata: dict[str, Any] = Field(default_factory=dict)

    # Pydantic 설정: Enum 필드를 저장할 때 멤버 객체 대신 그 값(문자열)을
    # 사용하도록 한다. 덕분에 memory_type이 "episodic"처럼 순수 문자열로
    # 직렬화되어 Redis/PG/JSON 어디에 넣어도 일관성이 유지된다.
    model_config = {"use_enum_values": True}


# ─────────────────────────────────────────────
# 타입별 감쇠 반감기 (일 단위)
# ─────────────────────────────────────────────
# 왜 반감기를 사용하는가:
#   사람의 기억처럼, 오래된 기억일수록 영향력을 서서히 잃게 만들고 싶다.
#   이를 지수 감쇠(exponential decay)로 구현하는데, '반감기'는 유효 중요도가
#   절반으로 줄어드는 데 걸리는 시간(일)을 뜻한다. 반감기가 길수록 천천히,
#   짧을수록 빨리 잊혀진다.
#   예) USER_PROFILE=365일 → 아주 오래 유지 / EPISODIC=7일 → 빨리 잊힘.
# 사용처: 실제 감쇠 점수 계산 로직(메모리 매니저)이 이 테이블을 참조한다.
DECAY_HALF_LIFE: dict[MemoryType, float] = {
    MemoryType.EPISODIC: 7.0,  # 대화 기억은 1주일 반감기
    MemoryType.SEMANTIC: 90.0,  # 지식/사실은 3개월 반감기
    MemoryType.PROCEDURAL: 30.0,  # 절차 지식은 1개월 반감기
    MemoryType.USER_PROFILE: 365.0,  # 사용자 프로필은 1년 반감기
    MemoryType.FEEDBACK: 14.0,  # 피드백은 2주 반감기
}


# ─────────────────────────────────────────────
# 코드 RAG 청크 판별 (대화 기억과 구분)
# ─────────────────────────────────────────────
# 왜 필요한가 (2026-08-06 실측):
#   tb_memories 한 테이블에 두 종류가 섞여 산다.
#     · 코드 RAG 인덱서가 넣은 소스코드 청크 — key="rag:파일:chunk_N", tags에 "rag",
#       metadata.source="rag_indexer". 실측 113만 건(전체의 99.98%).
#     · 실제 대화 기억 — key="turn:...". 실측 174건.
#   RAG 리트리버는 자기 것만 골라 쓰지만(core/rag/retriever.py), 대화 회상 쪽에는
#   반대 방향 필터가 없었다. 그대로 회상을 켜면 코드 청크가 컨텍스트를 덮는다
#   (RAG 청크는 importance=0.8로 높게 매겨져 중요도 정렬에서 앞자리를 차지한다).
#   판별 기준은 리트리버와 동일하게 유지해야 양쪽이 어긋나지 않는다.
_RAG_SOURCE = "rag_indexer"
_RAG_TAG = "rag"


def is_rag_chunk(entry: MemoryEntry) -> bool:
    """이 기억이 '코드 RAG 인덱서가 넣은 청크'인지 판별한다.

    core/rag/retriever.py 가 자기 결과를 고를 때 쓰는 기준과 같은 술어다.
    대화 회상 쪽은 이 함수가 True인 항목을 제외해 코드 조각이 섞이지 않게 한다.
    """
    return entry.metadata.get("source") == _RAG_SOURCE or _RAG_TAG in entry.tags


# ─────────────────────────────────────────────
# 메모리 검색 결과 (점수 포함)
# ─────────────────────────────────────────────
class MemorySearchResult(BaseModel):
    """
    메모리 검색 결과 한 건을 담는 모델.

    검색을 하면 '어떤 기억이(entry)' + '얼마나 관련 있는지(score)'를 함께
    알아야 한다. 이 모델은 그 두 정보를 한 쌍으로 묶어 반환하기 위한 것이다.
    보통 검색 함수는 이 결과들의 리스트를 score 내림차순으로 돌려준다.
    """

    # 검색으로 찾아낸 실제 기억 본문
    entry: MemoryEntry
    # 유사도 또는 관련도 점수 (높을수록 질의와 더 관련성이 높음)
    score: float = 0.0
