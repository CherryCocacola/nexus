# 경로 차단 사유가 '진짜 순회'와 '단순히 작업 디렉토리 밖'을 구분하는지 검증한다.
"""
`PathGuard.is_path_safe` 의 **차단 사유 문구** 테스트.

왜 문구를 테스트하는가:
    이 문구는 사람이 보는 안내가 아니라 **모델이 읽고 다음 행동을 정하는 신호**다.
    2026-08-19 실측에서 A.X-4.0 이 긴 절대경로를 틀리게 재현했는데
    (`nexus-b200` → `nexus-b2200`), 돌아온 말이 "경로 순회 공격" 뿐이라
    경로가 틀렸다는 사실을 알지 못했고 같은 경로를 반복 호출했다.
    그래서 "무엇이 잘못됐는지"가 문구에 들어 있어야 한다.

차단 자체는 두 경우 모두 유지된다(fail-closed). 바뀌는 것은 사유뿐이다.
"""

from __future__ import annotations

from core.security.path_guard import PathGuard


def test_real_traversal_still_reports_attack(tmp_path) -> None:
    """`..` 로 상위를 타는 진짜 순회는 종전대로 '순회'로 보고한다."""
    pg = PathGuard()
    safe, reason = pg.is_path_safe(str(tmp_path / ".." / ".." / "etc" / "shadow"), str(tmp_path))
    assert safe is False
    assert "순회" in reason


def test_outside_cwd_without_dotdot_is_not_called_an_attack(tmp_path) -> None:
    """★오타로 엉뚱한 절대경로를 낸 경우를 '공격'이라 부르지 않는다."""
    pg = PathGuard()
    outside = str(tmp_path.parent / "some-other-project" / "app.py")
    safe, reason = pg.is_path_safe(outside, str(tmp_path))
    assert safe is False, "차단은 그대로여야 한다(fail-closed)"
    assert "순회" not in reason, "오타를 공격으로 부르면 모델이 경로를 고칠 생각을 못 한다"


def test_outside_cwd_reason_tells_the_working_directory(tmp_path) -> None:
    """모델이 스스로 고칠 수 있도록 현재 작업 디렉토리를 알려 준다."""
    pg = PathGuard()
    outside = str(tmp_path.parent / "typo-dir" / "app.py")
    _, reason = pg.is_path_safe(outside, str(tmp_path))
    assert str(tmp_path.resolve()) in reason
    assert "상대경로" in reason


def test_inside_cwd_still_allowed(tmp_path) -> None:
    """정상 경로는 그대로 통과한다(무회귀)."""
    f = tmp_path / "ok.txt"
    f.write_text("x", encoding="utf-8")
    pg = PathGuard()
    safe, reason = pg.is_path_safe(str(f.resolve()), str(tmp_path.resolve()))
    assert safe is True, reason
