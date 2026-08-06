# web/app.py 업로드 라우트 — 확장자 allowlist·크기 상한 검증 단위 테스트.
"""
web.app.upload_file 의 입력 검증 단위 테스트 (W8 업로드 확장).

정책:
    - 확장자가 config.upload.allowed_extensions 밖이면 415 로 거절하고 파일을
      디스크에 남기지 않는다.
    - 크기가 config.upload.max_size_bytes 를 넘으면 413 으로 거절하고, 이미 쓰인
      부분 파일을 지운다(찌꺼기 방지).
    - 허용 범위 안이면 ASCII-safe 이름(upload-<uuid>.<ext>)으로 저장하고 실제
      기록된 바이트 수를 돌려준다.
    - allowed_extensions 가 비어 있으면 확장자 검사를 건너뛴다(임시 완화 탈출구).

테스트 격리:
    실제 HTTP 없이 upload_file 을 직접 await 한다. 업로드 디렉토리는 tmp_path 로
    monkeypatch 하고(_app_state 오염 방지), UploadFile 은 filename/read(size) 만
    갖춘 최소 객체로 흉내 낸다. asyncio_mode="auto" 라 데코레이터 없이 돈다.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import HTTPException

from web.app import _app_state, _uploads_cleanup_loop, upload_file


class _FakeUpload:
    """UploadFile 중 라우트가 실제로 쓰는 filename/read(size) 만 흉내 낸 스텁."""

    def __init__(self, filename: str, content: bytes) -> None:
        self.filename = filename
        self._buf = content
        self._pos = 0

    async def read(self, size: int = -1) -> bytes:
        """조각 단위 읽기 — 라우트가 1MB씩 끊어 읽는 동작을 그대로 재현한다."""
        if size is None or size < 0:
            chunk, self._pos = self._buf[self._pos :], len(self._buf)
            return chunk
        chunk = self._buf[self._pos : self._pos + size]
        self._pos += len(chunk)
        return chunk


@pytest.fixture
def uploads(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """업로드 디렉토리를 tmp_path 로 바꾸고, 끝나면 _app_state 를 원복한다."""
    target = tmp_path / "uploads"
    target.mkdir()

    # 라우트가 호출 시점에 import 하므로 원본 모듈의 심볼을 갈아끼우면 된다.
    monkeypatch.setattr(
        "core.tools.implementations.analyze_image_tool.resolve_uploads_dir",
        lambda *a, **k: target,
    )

    saved = _app_state.get("config")
    yield target
    if saved is None:
        _app_state.pop("config", None)
    else:
        _app_state["config"] = saved


def _set_upload_config(**kwargs: Any) -> None:
    """_app_state 의 config.upload 를 테스트용 값으로 세팅한다."""
    _app_state["config"] = SimpleNamespace(upload=SimpleNamespace(**kwargs))


# ─────────────────────────────────────────────
# 확장자 allowlist
# ─────────────────────────────────────────────


async def test_upload_rejects_extension_outside_allowlist(uploads: Path):
    """허용 목록 밖 확장자(.exe)는 415로 거절되고 파일이 저장되지 않는다."""
    _set_upload_config(max_size_bytes=1024, allowed_extensions=[".pdf", ".hwpx"])

    with pytest.raises(HTTPException) as exc:
        await upload_file(_FakeUpload("payload.exe", b"MZ\x00\x00"))

    assert exc.value.status_code == 415
    assert list(uploads.iterdir()) == []  # 디스크에 아무것도 남지 않는다


async def test_upload_accepts_extension_in_allowlist(uploads: Path):
    """허용 목록 안 확장자(.hwpx)는 저장되고 서버 경로·크기를 돌려준다."""
    _set_upload_config(max_size_bytes=1024, allowed_extensions=[".pdf", ".hwpx"])

    result = await upload_file(_FakeUpload("보고서.hwpx", b"PK\x03\x04payload"))

    assert result["status"] == "ok"
    assert result["file_name"] == "보고서.hwpx"
    assert result["size_bytes"] == len(b"PK\x03\x04payload")
    saved = Path(result["file_path"])
    assert saved.parent == uploads
    # 저장 이름은 ASCII-safe 여야 한다(모델이 경로를 되받아 적기 때문).
    assert saved.name.startswith("upload-") and saved.name.endswith(".hwpx")
    assert saved.name.isascii()
    assert saved.read_bytes() == b"PK\x03\x04payload"


async def test_upload_skips_extension_check_when_allowlist_empty(uploads: Path):
    """allowed_extensions 가 비면 확장자 검사를 건너뛴다(운영 중 임시 완화)."""
    _set_upload_config(max_size_bytes=1024, allowed_extensions=[])

    result = await upload_file(_FakeUpload("anything.weird", b"data"))

    assert result["status"] == "ok"


# ─────────────────────────────────────────────
# 크기 상한
# ─────────────────────────────────────────────


async def test_upload_rejects_oversize_and_leaves_no_partial_file(uploads: Path):
    """상한을 넘으면 413으로 거절하고 반쯤 쓰인 파일을 남기지 않는다."""
    _set_upload_config(max_size_bytes=10, allowed_extensions=[".txt"])

    with pytest.raises(HTTPException) as exc:
        await upload_file(_FakeUpload("big.txt", b"x" * 5000))

    assert exc.value.status_code == 413
    assert list(uploads.iterdir()) == []  # 부분 파일 찌꺼기 없음


async def test_upload_accepts_size_at_limit(uploads: Path):
    """정확히 상한과 같은 크기는 통과한다(경계값)."""
    _set_upload_config(max_size_bytes=10, allowed_extensions=[".txt"])

    result = await upload_file(_FakeUpload("exact.txt", b"x" * 10))

    assert result["status"] == "ok"
    assert result["size_bytes"] == 10


async def test_upload_falls_back_to_defaults_without_config(uploads: Path):
    """config 가 없어도(경량 경로) UploadConfig 기본값으로 동작한다."""
    _app_state["config"] = None

    result = await upload_file(_FakeUpload("memo.txt", b"hello"))

    assert result["status"] == "ok"
    assert result["size_bytes"] == 5


# ─────────────────────────────────────────────
# 업로드 정리 잡 루프
# ─────────────────────────────────────────────


async def test_cleanup_loop_sweeps_immediately_and_repeats(monkeypatch: pytest.MonkeyPatch):
    """정리 루프는 기동 직후 1회 쓸고, 이후 주기마다 반복한다."""
    calls: list[tuple[Any, float]] = []

    def fake_cleanup(uploads_dir, retention_hours, *a, **k):
        calls.append((uploads_dir, retention_hours))
        return 0, 0

    monkeypatch.setattr("core.storage.uploads.cleanup_expired_uploads", fake_cleanup)

    # 주기를 아주 짧게(0.6ms) 줘서 반복이 실제로 일어나는지 확인한다.
    task = asyncio.create_task(_uploads_cleanup_loop(Path("/tmp/x"), 24, 0.00001))
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert len(calls) >= 2, f"반복되지 않음(호출 {len(calls)}회)"
    assert calls[0] == (Path("/tmp/x"), 24)


async def test_cleanup_loop_survives_sweep_failure(monkeypatch: pytest.MonkeyPatch):
    """한 번의 정리 실패가 루프를 죽이지 않는다(fail-soft)."""
    calls: list[int] = []

    def flaky_cleanup(uploads_dir, retention_hours, *a, **k):
        calls.append(1)
        raise OSError("디스크 오류")

    monkeypatch.setattr("core.storage.uploads.cleanup_expired_uploads", flaky_cleanup)

    task = asyncio.create_task(_uploads_cleanup_loop(Path("/tmp/x"), 24, 0.00001))
    await asyncio.sleep(0.05)
    alive = not task.done()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert alive, "정리 실패로 루프가 종료됨"
    assert len(calls) >= 2, f"실패 후 재시도하지 않음(호출 {len(calls)}회)"
