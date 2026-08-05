# 세션 정리 도구 검증 — TTL 분류·사용자 대화 보호·메타 폴더 제외.
"""
scripts/cleanup_sessions.py 의 안전 계약을 고정한다 (2026-08-05).

핵심 계약(깨지면 사용자 대화가 날아간다):
  - web/cli 채널은 기본적으로 정리 대상이 아니다(명시 옵션이 있을 때만).
  - `_`로 시작하는 메타 폴더(_instructions/_projects/_prompts)는 절대 대상 아님.
  - 보존기간을 넘긴 것만 대상이 된다(경계값 포함).
  - apply_plan은 sessions_dir 하위가 아닌 경로를 지우지 않는다(순회 방어).
"""

from __future__ import annotations

import os
import time
from pathlib import Path

from scripts.cleanup_sessions import apply_plan, build_plan

NOW = 1_800_000_000.0  # 고정 기준 시각(테스트 결정성)
DAY = 86400.0


def _mk_session(root: Path, rel: str, age_days: float, *, transcript: bool = True) -> Path:
    """지정 경과일을 가진 세션 디렉토리를 만든다."""
    d = root / rel
    d.mkdir(parents=True, exist_ok=True)
    if transcript:
        f = d / "transcript.jsonl"
        f.write_text('{"role":"user"}\n', encoding="utf-8")
        ts = NOW - age_days * DAY
        os.utime(f, (ts, ts))
    else:
        (d / "workspace").mkdir(exist_ok=True)
    ts = NOW - age_days * DAY
    os.utime(d, (ts, ts))
    return d


def test_api_sessions_beyond_ttl_are_targeted(tmp_path: Path) -> None:
    """api 채널은 보존일을 넘기면 대상이 된다(무상태라 재사용되지 않음)."""
    _mk_session(tmp_path, "api/old", age_days=10)
    _mk_session(tmp_path, "api/fresh", age_days=1)

    plan = build_plan(tmp_path, api_days=7, flat_days=30, workspace_days=3, user_days=None, now=NOW)

    names = {p.name for p in plan.api}
    assert names == {"old"}


def test_user_channels_protected_by_default(tmp_path: Path) -> None:
    """web/cli는 아무리 오래돼도 기본값에서는 대상이 아니다(사용자 대화 보호)."""
    _mk_session(tmp_path, "web/ancient", age_days=999)
    _mk_session(tmp_path, "cli/ancient", age_days=999)

    plan = build_plan(tmp_path, api_days=7, flat_days=30, workspace_days=3, user_days=None, now=NOW)

    assert plan.user == []
    assert plan.all_targets() == []


def test_user_channels_included_when_explicitly_requested(tmp_path: Path) -> None:
    """명시적으로 보존일을 주면 web/cli도 정리 대상이 된다."""
    _mk_session(tmp_path, "web/ancient", age_days=999)
    _mk_session(tmp_path, "web/recent", age_days=1)

    plan = build_plan(tmp_path, api_days=7, flat_days=30, workspace_days=3, user_days=90, now=NOW)

    assert {p.name for p in plan.user} == {"ancient"}


def test_meta_dirs_never_targeted(tmp_path: Path) -> None:
    """_instructions/_projects/_prompts 등 메타 저장소는 절대 삭제 대상이 아니다."""
    for meta in ("_instructions", "_projects", "_prompts"):
        d = tmp_path / meta
        d.mkdir()
        (d / "data.json").write_text("{}", encoding="utf-8")
        ts = NOW - 999 * DAY
        os.utime(d, (ts, ts))

    plan = build_plan(tmp_path, api_days=1, flat_days=1, workspace_days=1, user_days=1, now=NOW)

    assert plan.all_targets() == []


def test_legacy_flat_and_workspace_are_classified(tmp_path: Path) -> None:
    """루트 직속 항목은 transcript 유무로 레거시 대화/작업 디렉토리로 나뉜다."""
    _mk_session(tmp_path, "legacy-chat", age_days=60)  # transcript 있음
    _mk_session(tmp_path, "ws-only", age_days=5, transcript=False)  # workspace만
    _mk_session(tmp_path, "ws-fresh", age_days=1, transcript=False)

    plan = build_plan(tmp_path, api_days=7, flat_days=30, workspace_days=3, user_days=None, now=NOW)

    assert {p.name for p in plan.flat} == {"legacy-chat"}
    assert {p.name for p in plan.workspace} == {"ws-only"}


def test_ttl_boundary_is_inclusive(tmp_path: Path) -> None:
    """정확히 보존일에 도달한 항목도 대상에 포함된다(경계값 고정)."""
    _mk_session(tmp_path, "api/exact", age_days=7)

    plan = build_plan(tmp_path, api_days=7, flat_days=30, workspace_days=3, user_days=None, now=NOW)

    assert {p.name for p in plan.api} == {"exact"}


def test_apply_plan_deletes_only_inside_root(tmp_path: Path) -> None:
    """계획에 루트 밖 경로가 섞여 있어도 지우지 않는다(순회 방어)."""
    root = tmp_path / "sessions"
    root.mkdir()
    inside = _mk_session(root, "api/old", age_days=10)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "keep.txt").write_text("keep", encoding="utf-8")

    plan = build_plan(root, api_days=7, flat_days=30, workspace_days=3, user_days=None, now=NOW)
    plan.flat.append(outside)  # 루트 밖 경로를 강제로 주입

    ok, fail = apply_plan(plan, root)

    assert not inside.exists()  # 정상 대상은 삭제됨
    assert outside.exists()  # 루트 밖은 보존됨
    assert ok == 1 and fail == 1


def test_empty_or_missing_dir_is_safe(tmp_path: Path) -> None:
    """세션 루트가 없거나 비어 있어도 예외 없이 빈 계획을 돌려준다."""
    assert build_plan(tmp_path / "nope", 7, 30, 3, None, now=NOW).all_targets() == []
    assert build_plan(tmp_path, 7, 30, 3, None, now=time.time()).all_targets() == []
