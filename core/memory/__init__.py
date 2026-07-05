"""
메모리 시스템 패키지 — 단기(Redis) + 장기(PostgreSQL+pgvector) 통합 메모리.

이 파일은 `core.memory` 패키지의 "현관(public API)" 역할을 한다.
패키지 내부의 여러 하위 모듈에 흩어져 있는 핵심 클래스/상수를 한곳에 모아
바깥 코드가 `from core.memory import MemoryManager` 처럼 짧게 가져다 쓰도록
재노출(re-export)한다. 즉 실제 구현 로직은 여기 없고, 하위 모듈을 가리키는
"입구"만 정의한다.

Nexus 메모리 시스템의 큰 그림 (2계층 구조):
  - 단기 메모리 (ShortTermMemory): Redis 기반 세션 캐시. 지금 진행 중인 대화의
    최근 기억을 빠르게 넣고 뺀다. 휘발성이 강하다.
  - 장기 메모리 (LongTermMemory): PostgreSQL + pgvector 기반 영구 저장소.
    중요도가 높은 기억을 오래 보관하고, 임베딩 벡터로 의미 기반 검색을 한다.
  - MemoryManager: 위 두 계층을 하나로 묶어 관리하는 오케스트레이터.
    턴(turn) 생명주기에 맞춰 단기→장기 승격, 회수(recall) 등을 자동 수행한다.
  - ImportanceAssessor: 각 기억의 중요도(0.0~1.0)를 평가한다.
    이 점수가 단기 메모리를 장기로 승격시킬지 판단하는 기준이 된다.
  - MemoryDecayManager: 시간 감쇠(반감기) 규칙에 따라 오래되고 잘 안 쓰이는
    기억을 점차 약화·정리한다.

노출(공개)하는 것들:
  - 데이터 타입: MemoryType(기억 종류 열거형), MemoryEntry(기억 한 개 단위),
    DECAY_HALF_LIFE(타입별 감쇠 반감기 상수 테이블)
  - 동작 클래스: ShortTermMemory, LongTermMemory, MemoryManager,
    ImportanceAssessor, MemoryDecayManager

작성자: 이현수 / 작성일: 2026-07-05
"""

# ── 하위 모듈에서 공개 심볼을 끌어와 패키지 최상단으로 재노출한다 ──
# 아래 import들은 "core.memory.xxx 에 정의된 것을 core.memory 이름으로도 쓸 수
# 있게" 연결하는 역할이다. 실제 구현은 각 하위 모듈 파일에 있다.

# 시간 감쇠 관리자 — 오래되고 접근이 뜸한 기억을 반감기에 따라 약화/정리한다.
from core.memory.decay import MemoryDecayManager

# 중요도 평가기 — 기억의 중요도(0.0~1.0)를 산정, 단기→장기 승격 판단에 쓰인다.
from core.memory.importance import ImportanceAssessor

# 장기 메모리 — PostgreSQL + pgvector 기반 영구 저장 및 벡터 유사도 검색.
from core.memory.long_term import LongTermMemory

# 메모리 매니저 — 단기/장기 두 계층을 통합 관리하는 상위 오케스트레이터.
from core.memory.manager import MemoryManager

# 단기 메모리 — Redis 기반 세션 캐시. 최근 기억을 빠르게 저장/조회한다.
from core.memory.short_term import ShortTermMemory

# 데이터 타입/상수 — 기억의 종류(MemoryType), 한 개 단위(MemoryEntry),
# 타입별 감쇠 반감기 테이블(DECAY_HALF_LIFE).
from core.memory.types import DECAY_HALF_LIFE, MemoryEntry, MemoryType

# `from core.memory import *` 로 가져올 때 외부에 노출할 공개 심볼 목록.
# 여기에 나열된 이름만 이 패키지의 "공식 공개 API"로 취급한다.
__all__ = [
    "MemoryType",  # 기억의 종류를 구분하는 열거형(episodic/semantic 등)
    "MemoryEntry",  # 기억 한 개를 표현하는 Pydantic 모델
    "DECAY_HALF_LIFE",  # 메모리 타입별 감쇠 반감기 상수 테이블
    "ShortTermMemory",  # Redis 기반 단기 메모리 저장소
    "LongTermMemory",  # PostgreSQL+pgvector 기반 장기 메모리 저장소
    "MemoryManager",  # 단기/장기 계층을 통합 관리하는 오케스트레이터
    "ImportanceAssessor",  # 기억 중요도 평가기(승격 기준 산정)
    "MemoryDecayManager",  # 시간 감쇠 기반 기억 정리 관리자
]
