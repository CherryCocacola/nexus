# 응답 사후 검증(post-verification) 모듈 모음.
"""
core.verification — 모델이 낸 답변을 근거 자료와 대조하는 결정론적 검증기들.

왜 필요한가:
  프롬프트로 "정확히 옮겨라"라고 지시해도 모델이 긴 숫자를 안정적으로 복사하지
  못하는 경우가 있다(A.X-4.0 실측: 같은 질문 5회 중 0~5회 정확, 실행마다 달라짐).
  지시로 통제되지 않는 오류는 코드로 잡아야 한다. 이 패키지는 "설득"이 아니라
  "대조"로 문제를 다룬다.

지연 재노출(PEP 562)을 쓰는 이유는 core.ingest 와 같다 — 하위 모듈 하나를 import
할 때 나머지까지 끌려 들어가지 않게 하기 위함이다.

작성자: 이현수 / 작성일: 2026-08-06
"""

from __future__ import annotations

import importlib
from typing import Any

_LAZY_EXPORTS: dict[str, str] = {
    "UncitedNumber": "core.verification.number_citation",
    "find_uncited_numbers": "core.verification.number_citation",
    "build_number_warning": "core.verification.number_citation",
    "find_execution_claims": "core.verification.execution_claim",
    "build_execution_warning": "core.verification.execution_claim",
    "build_answer_warnings": "core.verification.post_check",
    "collect_tool_result_texts": "core.verification.post_check",
}


def __getattr__(name: str) -> Any:
    """요청받은 이름이 속한 하위 모듈만 그때 import 한다(PEP 562)."""
    module_path = _LAZY_EXPORTS.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module_path), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted([*globals().keys(), *_LAZY_EXPORTS])


__all__ = list(_LAZY_EXPORTS)
