# CLI(TIER_M/L) 도구 풀에 문서 파싱·생성 도구가 실린 상태를 고정하는 구조 테스트.
"""
CLI 문서 도구 배선 테스트 (2026-08-19).

무엇을 지키는가:
  옛 TIER_S 구조에서 문서 파싱은 Scout 전담이었다. TIER_L 로 오며 Read/Glob/Grep/LS
  는 CLI 풀로 옮겨왔는데 DocumentProcess/DocumentExport 만 따라오지 않아, 웹은
  업로드 문서를 읽는데 CLI 사용자는 자기 PC 의 xlsx 를 못 읽는 비대칭이 있었다.

왜 값이 아니라 구조를 보는가:
  프롬프트 폴백 상수(_DEFAULT_EXPANDED_TOOLS)와 실제 레지스트리가 어긋나면
  "있는 도구를 숨기거나 없는 도구를 안내"하는 조용한 버그가 된다 —
  과거 `알 수 없는 도구: 'Agent'` 사고와 같은 형태다. 그래서 둘을 직접 대조한다.
"""

from __future__ import annotations

from core.bootstrap import (
    _DEFAULT_EXPANDED_TOOLS,
    _build_expanded_system_prompt,
    _create_tool_registry,
)


def _registry_names() -> set[str]:
    """풀세트 레지스트리에 실제 등록된 도구 이름 집합."""
    return {t.name for t in _create_tool_registry().get_all_tools()}


def test_cli_registry_includes_document_tools():
    """CLI 풀에 DocumentProcess(파싱)와 DocumentExport(생성)가 모두 있어야 한다."""
    names = _registry_names()
    assert "DocumentProcess" in names, "CLI 에서 로컬 문서를 읽지 못한다"
    assert "DocumentExport" in names, "CLI 에서 문서를 만들지 못한다"


def test_expanded_tools_fallback_matches_registry():
    """프롬프트 폴백 상수 == 실제 등록 이름. 도구를 추가하고 상수를 잊으면 여기서 깨진다."""
    assert set(_DEFAULT_EXPANDED_TOOLS) == _registry_names()
    # 중복 없이 나열되어야 한다(집합 비교만으로는 중복을 못 잡는다).
    assert len(_DEFAULT_EXPANDED_TOOLS) == len(set(_DEFAULT_EXPANDED_TOOLS))


def test_expanded_prompt_lists_document_tools():
    """레지스트리 이름을 넘기면 프롬프트 도구 목록에 문서 도구가 실제로 나와야 한다."""
    prompt = _build_expanded_system_prompt(_registry_names())
    assert "- DocumentProcess\n" in prompt
    assert "- DocumentExport\n" in prompt
