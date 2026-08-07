# 웹 표면의 도구 노출 경계를 고정하는 회귀 테스트 (2026-08-07 보안 결정).
"""
웹 Worker 도구 풀에 **명령 실행·파일 탐색 도구가 없어야 한다**는 계약을 고정한다.

[왜 생겼나 — 실측한 사고]
    유효한 테넌트 키 하나(coding)로 Bash 도구를 호출해 아래를 전부 읽어냈다.
      · /app/config/tenants.yaml → **모든 테넌트의 API 키 평문**
      · env                      → NEXUS_PG_PASSWORD / NEXUS_REDIS_PASSWORD
      · /app/data/exports        → 다른 테넌트가 만든 산출물
    세션 cwd(/app/.nexus/sessions/{id}/workspace)는 격리돼 있었지만 Bash 가 그 안에
    갇혀 있지 않아 격리가 형식뿐이었다. 결과적으로 테넌트 → 테넌트 권한 상승이며,
    같은 날 /v1/memories 에 붙인 IDOR 방어가 이 옆문 때문에 무의미해졌다.

[이 테스트가 막는 회귀]
    "웹에서도 명령을 실행할 수 있으면 편하겠다"는 이유로 Bash 가 다시 추가되는 것.
    되돌리려면 먼저 Bash 를 세션 workspace 에 가두는 격리(별도 컨테이너·bwrap 등)를
    구현해야 하고, 그때 이 테스트를 의도적으로 고쳐야 한다.
    CLI 는 로컬 운영자 도구라 이 제한의 대상이 아니다(아래 대조 테스트로 확인).
"""

from __future__ import annotations

from core.bootstrap import _create_tool_registry, _create_web_tool_registry
from core.model.hardware_tier import HardwareTier

# 웹 표면에서 절대 노출되면 안 되는 도구들.
#   Bash  — 임의 명령 실행(위 사고의 직접 경로)
#   Read/Glob/Grep/LS — 임의 파일 탐색(2026-07-08 결정, 같은 이유로 유지)
FORBIDDEN_ON_WEB = {"Bash", "Read", "Glob", "Grep", "LS"}


def _names(registry) -> set[str]:
    return {t.name for t in registry.get_all_tools()}


def test_web_pool_has_no_command_execution_on_any_tier():
    """★모든 티어에서 Bash 가 없어야 한다 — 티어 확장으로 새어 들어오는 것을 막는다."""
    for tier in (None, HardwareTier.TIER_S, HardwareTier.TIER_M, HardwareTier.TIER_L):
        names = _names(_create_web_tool_registry(tier))
        leaked = names & FORBIDDEN_ON_WEB
        assert not leaked, f"tier={tier} 에서 금지 도구가 노출됨: {sorted(leaked)}"


def test_web_pool_still_provides_the_useful_tools():
    """차단이 과해져 웹이 무력해지지 않았는지 확인한다(과잉 차단 방지)."""
    names = _names(_create_web_tool_registry(HardwareTier.TIER_L))

    # 문서·이미지·편집·심볼 검색은 그대로 있어야 한다.
    for expected in (
        "DocumentProcess",
        "DocumentExport",
        "AnalyzeImage",
        "ImageGenerate",
        "Edit",
        "Write",
        "SymbolSearch",
        "GitDiff",
    ):
        assert expected in names, f"웹에 있어야 할 도구가 사라짐: {expected}"


def test_cli_pool_keeps_bash():
    """CLI 는 로컬 운영자 도구라 Bash 를 유지한다 — 이 제한은 웹 표면 전용이다."""
    assert "Bash" in _names(_create_tool_registry())


def test_web_prompt_does_not_advertise_removed_tools():
    """프롬프트가 없는 도구를 안내하면 모델이 헛호출로 턴을 낭비한다.

    도구 풀과 프롬프트는 같이 움직여야 한다(코드베이스가 반복해서 겪은 불일치).
    """
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent.parent
    for name in ("worker_system.md", "worker_system_full.md"):
        text = (root / "web" / "prompts" / name).read_text(encoding="utf-8")
        assert "- Bash:" not in text, f"{name} 이 아직 Bash 를 도구로 안내한다"


# ─────────────────────────────────────────────
# 코드 자가검증 규칙 (2026-08-07)
# ─────────────────────────────────────────────
#
# 실측: 같은 코딩 과제 6개를 CLI(도구 26개)로 돌렸더니 전부 tool_use=0 이었다.
# Bash·Write 를 쥐고도 한 턴에 답만 쓰고 끝냈고, 테스트 작성은 기대값을 머릿속으로
# 계산해 적어 실패했다. 기존 verify_rule 은 "서버 기동·설치" 같은 상태 변경만
# 다뤄 코드에는 걸리지 않았다.


def test_cli_prompt_tells_model_to_run_code_it_writes():
    """CLI 프롬프트에 '쓴 코드를 실행해 확인하라'가 들어 있어야 한다."""
    from core.bootstrap import _build_expanded_system_prompt

    prompt = _build_expanded_system_prompt()

    assert "actually RUN it with Bash" in prompt
    # 머릿속 계산으로 기대값을 적지 말라는 지시가 함께 있어야 한다(실패의 직접 원인).
    assert "computed in your head" in prompt


def test_code_verify_rule_bounds_the_loop():
    """루프 상한이 명시돼야 한다 — 같은 날 렌더 루프가 종료 조건 없이 4회를 돌았다."""
    from core.bootstrap import _build_expanded_system_prompt

    assert "at most 2 fix-and-rerun cycles" in _build_expanded_system_prompt()


def test_code_verify_rule_absent_without_tools():
    """Bash·Write 가 없는 도구 풀에서는 이 지시를 넣지 않는다(없는 도구 호출 방지)."""
    from core.bootstrap import _build_expanded_system_prompt

    prompt = _build_expanded_system_prompt(tool_names={"Read", "Glob"})

    assert "actually RUN it with Bash" not in prompt
