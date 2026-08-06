# 업로드 샌드박스 보존기간 정리(cleanup_expired_uploads) 단위 테스트.
"""
core.storage.uploads.cleanup_expired_uploads 의 정상·경계·안전 동작 검증.

무엇을 지키나:
    - 보존기간을 넘긴 파일만 지우고, 최근 파일은 남긴다.
    - 이름 규칙과 무관하게 지운다(upload-* / render_* / 레거시 한글명 모두 임시 산출물).
    - 하위 디렉토리는 재귀하지 않고 그대로 둔다(보수적).
    - retention_hours 가 0 이하이면 아무 것도 지우지 않는다("0=전부 삭제" 사고 방지).
    - 디렉토리가 없어도 예외 없이 (0, 0).
    - 파일 하나가 안 지워져도 나머지 정리를 계속한다(fail-soft).

테스트 전략(.claude/rules/testing.md 준수):
    - 파일시스템은 tmp_path fixture 만 사용한다.
    - 시각은 now 인자로 주입해 실제 대기(time.sleep) 없이 경계를 검증한다.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

from core.storage.uploads import cleanup_expired_uploads


def _write_at(path: Path, content: bytes, age_hours: float, now: float) -> Path:
    """지정한 '나이'를 갖도록 mtime을 조작해 파일을 만든다."""
    path.write_bytes(content)
    stamp = now - age_hours * 3600.0
    os.utime(path, (stamp, stamp))
    return path


def test_cleanup_removes_only_expired_files(tmp_path: Path):
    """보존기간을 넘긴 파일만 지우고 최근 파일은 남긴다."""
    now = time.time()
    old = _write_at(tmp_path / "upload-old.docx", b"x" * 100, age_hours=48, now=now)
    fresh = _write_at(tmp_path / "upload-new.pptx", b"y" * 50, age_hours=1, now=now)

    deleted, freed = cleanup_expired_uploads(tmp_path, retention_hours=24, now=now)

    assert deleted == 1
    assert freed == 100
    assert not old.exists()
    assert fresh.exists()


def test_cleanup_boundary_exactly_at_retention_is_kept(tmp_path: Path):
    """정확히 보존기간과 같은 나이는 아직 만료가 아니다(경계값)."""
    now = time.time()
    exact = _write_at(tmp_path / "upload-exact.pdf", b"z", age_hours=24, now=now)

    deleted, _ = cleanup_expired_uploads(tmp_path, retention_hours=24, now=now)

    assert deleted == 0
    assert exact.exists()


def test_cleanup_covers_all_filename_patterns(tmp_path: Path):
    """upload-* 뿐 아니라 render_* 스크린샷과 레거시 한글명도 함께 정리된다."""
    now = time.time()
    names = ["upload-abc123.docx", "render_page_1712345678.png", "강원대학교_제안서.docx"]
    for name in names:
        _write_at(tmp_path / name, b"data", age_hours=72, now=now)

    deleted, _ = cleanup_expired_uploads(tmp_path, retention_hours=24, now=now)

    assert deleted == 3
    assert list(tmp_path.iterdir()) == []


def test_cleanup_does_not_recurse_into_subdirectories(tmp_path: Path):
    """하위 디렉토리와 그 안의 파일은 오래됐어도 건드리지 않는다."""
    now = time.time()
    sub = tmp_path / "workspace"
    sub.mkdir()
    inner = _write_at(sub / "old.txt", b"keep", age_hours=999, now=now)
    _write_at(tmp_path / "upload-old.txt", b"gone", age_hours=999, now=now)

    deleted, _ = cleanup_expired_uploads(tmp_path, retention_hours=24, now=now)

    assert deleted == 1  # 루트의 파일 하나만
    assert sub.is_dir()
    assert inner.exists()


def test_cleanup_zero_retention_is_noop(tmp_path: Path):
    """retention_hours 가 0 이하면 아무 것도 지우지 않는다(전부 삭제 사고 방지)."""
    now = time.time()
    f = _write_at(tmp_path / "upload-ancient.docx", b"x", age_hours=10000, now=now)

    for retention in (0, -1):
        deleted, freed = cleanup_expired_uploads(tmp_path, retention_hours=retention, now=now)
        assert (deleted, freed) == (0, 0)
    assert f.exists()


def test_cleanup_missing_directory_returns_zero(tmp_path: Path):
    """디렉토리가 없어도 예외 없이 (0, 0)."""
    assert cleanup_expired_uploads(tmp_path / "nope", retention_hours=24) == (0, 0)


def test_cleanup_continues_after_single_file_failure(tmp_path: Path, monkeypatch):
    """파일 하나 삭제가 실패해도 나머지는 계속 정리한다(fail-soft)."""
    now = time.time()
    for name in ("upload-a.txt", "upload-locked.txt", "upload-c.txt"):
        _write_at(tmp_path / name, b"data", age_hours=48, now=now)

    real_unlink = Path.unlink

    def flaky_unlink(self: Path, *args, **kwargs):
        if self.name == "upload-locked.txt":
            raise PermissionError("잠긴 파일")
        return real_unlink(self, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", flaky_unlink)

    deleted, _ = cleanup_expired_uploads(tmp_path, retention_hours=24, now=now)

    assert deleted == 2  # 실패한 1건을 빼고 나머지는 삭제
    assert (tmp_path / "upload-locked.txt").exists()
