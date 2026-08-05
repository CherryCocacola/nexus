# 세션 저장소 정리 도구 — 채널별 보존기간(TTL) 기반으로 오래된 세션을 지운다.
"""
세션 디렉토리 정리 스크립트 (2026-08-05 신설).

[왜 필요한가]
  OpenAI 호환 API(`/v1/chat/completions`)는 세션 개념이 없어 **요청마다 새 UUID
  세션**을 만든다(무상태). 그래서 외부 소비자(VSCode 플러그인·AgentHub)가 쓰면
  `{sessions_dir}/api/` 아래에 세션 디렉토리가 무한히 쌓인다. 도구 작업 디렉토리
  (`workspace/`)와 채널 격리 이전의 레거시 flat 세션도 같이 누적된다.
  이 스크립트가 보존기간을 넘긴 것만 골라 정리한다.

[안전 설계 — 사용자 대화는 기본적으로 건드리지 않는다]
  - 기본 동작은 **dry-run**이다. 실제 삭제는 `--apply`를 명시해야 한다.
  - `web`/`cli` 채널(사람이 실제로 쓰는 대화)은 **기본 제외**한다. 지우려면
    `--include-user-channels`를 명시해야 한다(그래도 TTL은 적용된다).
  - 삭제 전 `--backup <파일>`로 tar.gz 백업을 만들 수 있다(권장).
  - 경로는 sessions_dir 하위로만 제한한다(순회 방지).

[사용 예]
  # 무엇이 지워질지만 확인(기본 dry-run)
  python scripts/cleanup_sessions.py --sessions-dir /app/.nexus/sessions

  # api 7일·workspace 3일·레거시 flat 30일 초과분을 백업 후 실제 삭제
  python scripts/cleanup_sessions.py --sessions-dir /app/.nexus/sessions \
      --api-days 7 --workspace-days 3 --flat-days 30 \
      --backup /app/.nexus/sessions_backup.tar.gz --apply

작성자: 이현수 / 작성일: 2026-08-05
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# 채널 하위 폴더 이름 — 이 이름들은 "세션"이 아니라 채널 컨테이너다.
_CHANNEL_DIRS = {"web", "cli", "api"}
# 밑줄로 시작하는 폴더는 메타 저장소(_instructions/_projects/_prompts)다 — 절대 삭제 금지.
_META_PREFIX = "_"
# 사람이 실제로 쓰는 대화 채널 — 기본적으로 정리 대상에서 제외한다.
_USER_CHANNELS = {"web", "cli"}


@dataclass
class Plan:
    """정리 계획 — 어떤 경로를 왜 지우는지 담는다."""

    api: list[Path] = field(default_factory=list)
    flat: list[Path] = field(default_factory=list)
    workspace: list[Path] = field(default_factory=list)
    user: list[Path] = field(default_factory=list)

    def all_targets(self) -> list[Path]:
        return [*self.api, *self.flat, *self.workspace, *self.user]


def _age_days(path: Path, now: float) -> float:
    """디렉토리의 마지막 수정 시각 기준 경과 일수를 돌려준다.

    transcript.jsonl이 있으면 그 파일의 시각을 쓴다(디렉토리 mtime은 하위 파일
    생성으로도 갱신되어 실제 마지막 대화 시점과 어긋날 수 있기 때문).
    """
    target = path / "transcript.jsonl"
    stat_path = target if target.is_file() else path
    try:
        return (now - stat_path.stat().st_mtime) / 86400.0
    except OSError:
        return 0.0


def build_plan(
    sessions_dir: Path,
    api_days: float,
    flat_days: float,
    workspace_days: float,
    user_days: float | None,
    now: float | None = None,
) -> Plan:
    """보존기간을 넘긴 삭제 대상 목록을 만든다(실제 삭제는 하지 않는다).

    Args:
        sessions_dir: 세션 루트({sessions_dir}).
        api_days: api 채널 세션 보존일. 이보다 오래되면 대상.
        flat_days: 채널 격리 이전 루트(flat) 세션 보존일.
        workspace_days: 대화 없이 workspace만 있는 작업 디렉토리 보존일.
        user_days: web/cli 채널 보존일. None이면 이 채널은 건드리지 않는다(기본).
        now: 기준 시각(테스트 주입용). None이면 현재 시각.

    Returns:
        Plan — 종류별 삭제 대상 경로 목록.
    """
    now = time.time() if now is None else now
    plan = Plan()
    if not sessions_dir.is_dir():
        return plan

    for entry in sorted(sessions_dir.iterdir()):
        if not entry.is_dir():
            continue
        name = entry.name
        # 메타 저장소는 절대 손대지 않는다.
        if name.startswith(_META_PREFIX):
            continue

        # ── 채널 컨테이너(web/cli/api) — 그 하위 세션을 검사한다 ──
        if name in _CHANNEL_DIRS:
            for sess in sorted(entry.iterdir()):
                if not sess.is_dir() or sess.name.startswith(_META_PREFIX):
                    continue
                age = _age_days(sess, now)
                if name == "api":
                    if age >= api_days:
                        plan.api.append(sess)
                elif name in _USER_CHANNELS and user_days is not None and age >= user_days:
                    plan.user.append(sess)
            continue

        # ── 루트 직속 항목 — 레거시 flat 세션 또는 작업 디렉토리 ──
        age = _age_days(entry, now)
        has_transcript = (entry / "transcript.jsonl").is_file()
        if has_transcript:
            if age >= flat_days:
                plan.flat.append(entry)
        else:
            # transcript 없이 workspace만 있는 도구 작업 디렉토리.
            if age >= workspace_days:
                plan.workspace.append(entry)
    return plan


def _dir_size_mb(paths: list[Path]) -> float:
    """대상 경로들의 총 용량(MB)을 어림 계산한다(표시용)."""
    total = 0
    for p in paths:
        for f in p.rglob("*"):
            try:
                if f.is_file():
                    total += f.stat().st_size
            except OSError:
                continue
    return total / 1024 / 1024


def make_backup(targets: list[Path], sessions_dir: Path, backup_path: Path) -> bool:
    """삭제 대상들을 tar.gz로 묶어 백업한다. 성공하면 True.

    되돌릴 수 없는 삭제 앞에 두는 안전장치다. tar가 없는 환경이면 False를 돌려
    호출부가 삭제를 중단하도록 한다(백업 없이 지우지 않는다).
    """
    if not targets:
        return True
    rel = [str(p.relative_to(sessions_dir)) for p in targets]
    try:
        # 인자는 전부 우리가 만든 경로다(사용자 입력 문자열을 셸에 넘기지 않음).
        # shell=False 리스트 형태라 셸 확장·인젝션이 발생하지 않는다.
        subprocess.run(  # noqa: S603 — 신뢰 가능한 인자, shell 미사용
            ["tar", "-czf", str(backup_path), "-C", str(sessions_dir), *rel],  # noqa: S607
            check=True,
            capture_output=True,
            timeout=600,
        )
        return backup_path.is_file()
    except (OSError, subprocess.SubprocessError):
        return False


def apply_plan(plan: Plan, sessions_dir: Path) -> tuple[int, int]:
    """계획된 경로들을 실제로 삭제한다.

    안전을 위해 삭제 직전 각 경로가 sessions_dir 하위인지 다시 확인한다
    (계획 생성과 실행 사이에 경로가 바뀌는 상황 방어 — fail-closed).

    Returns:
        (삭제 성공 수, 실패 수)
    """
    ok = fail = 0
    root = sessions_dir.resolve()
    for path in plan.all_targets():
        try:
            if not path.resolve().is_relative_to(root):
                fail += 1
                continue
            shutil.rmtree(path)
            ok += 1
        except OSError:
            fail += 1
    return ok, fail


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="세션 저장소 정리(기본 dry-run)")
    parser.add_argument("--sessions-dir", required=True, help="세션 루트 경로")
    parser.add_argument("--api-days", type=float, default=7, help="api 채널 보존일(기본 7)")
    parser.add_argument("--flat-days", type=float, default=30, help="레거시 flat 보존일(기본 30)")
    parser.add_argument(
        "--workspace-days", type=float, default=3, help="작업 디렉토리 보존일(기본 3)"
    )
    parser.add_argument(
        "--include-user-channels",
        type=float,
        default=None,
        metavar="DAYS",
        help="web/cli 채널도 이 보존일 기준으로 정리(기본: 건드리지 않음)",
    )
    parser.add_argument("--backup", default=None, help="삭제 전 tar.gz 백업 경로")
    parser.add_argument("--apply", action="store_true", help="실제 삭제(미지정 시 계획만 출력)")
    args = parser.parse_args(argv)

    sessions_dir = Path(args.sessions_dir)
    plan = build_plan(
        sessions_dir,
        api_days=args.api_days,
        flat_days=args.flat_days,
        workspace_days=args.workspace_days,
        user_days=args.include_user_channels,
    )

    targets = plan.all_targets()
    print(f"세션 루트: {sessions_dir}")
    print(f"  api 채널({args.api_days}일 초과)      : {len(plan.api)}개")
    print(f"  레거시 flat({args.flat_days}일 초과)  : {len(plan.flat)}개")
    print(f"  작업 디렉토리({args.workspace_days}일 초과): {len(plan.workspace)}개")
    if args.include_user_channels is not None:
        print(f"  web/cli({args.include_user_channels}일 초과) : {len(plan.user)}개")
    else:
        print("  web/cli                       : 제외(기본 보호)")
    if not targets:
        print("정리 대상 없음.")
        return 0
    print(f"  총 {len(targets)}개 / 약 {_dir_size_mb(targets):.1f}MB")

    if not args.apply:
        print("\n[dry-run] 실제로 지우려면 --apply 를 붙이세요. 예시(앞 5개):")
        for p in targets[:5]:
            print(f"  - {p}")
        return 0

    if args.backup:
        print(f"\n백업 생성 중: {args.backup}")
        if not make_backup(targets, sessions_dir, Path(args.backup)):
            print("백업 실패 — 안전을 위해 삭제를 중단합니다.")
            return 1
        print("백업 완료.")

    ok, fail = apply_plan(plan, sessions_dir)
    print(f"\n삭제 완료: {ok}개 성공, {fail}개 실패")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
