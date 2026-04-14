"""
메모리 시스템 — 단기(Redis) + 장기(PostgreSQL+pgvector) 통합 메모리.

Nexus의 메모리 시스템은 2계층으로 구성된다:
  - 단기 메모리 (ShortTermMemory): Redis 기반 세션 캐시
  - 장기 메모리 (LongTermMemory): PostgreSQL + pgvector 기반 영구 저장

MemoryManager가 두 계층을 통합 관리하며,
턴(turn) 생명주기에 맞춰 자동으로 메모리를 관리한다.
"""

from core.memory.decay import MemoryDecayManager
from core.memory.importance import ImportanceAssessor
from core.memory.long_term import LongTermMemory
from core.memory.manager import MemoryManager
from core.memory.short_term import ShortTermMemory
from core.memory.types import DECAY_HALF_LIFE, MemoryEntry, MemoryType

__all__ = [
    "MemoryType",
    "MemoryEntry",
    "DECAY_HALF_LIFE",
    "ShortTermMemory",
    "LongTermMemory",
    "MemoryManager",
    "ImportanceAssessor",
    "MemoryDecayManager",
]
