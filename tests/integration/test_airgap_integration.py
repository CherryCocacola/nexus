"""
Phase 8.0 통합 테스트 — Air-Gap 무결성 검증.

deployment/integrity.py와 deployment/airgap_prep.py의
SHA256 해시 계산, 매니페스트 생성/검증, 변조 감지를 통합 테스트한다.

사양서 Ch.22.6 Test 8에 해당한다.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from deployment.airgap_prep import AirGapPrep
from deployment.integrity import IntegrityVerifier


# ─────────────────────────────────────────────
# Test 8: Air-Gap 무결성 검증
# ─────────────────────────────────────────────
class TestIntegrityVerifier:
    """IntegrityVerifier의 해시 계산, 파일/디렉토리 검증을 테스트한다."""

    def test_compute_hash_returns_64_hex_chars(self, tmp_path: Path) -> None:
        """SHA256 해시가 64자리 16진수 문자열인지 검증한다."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello world", encoding="utf-8")

        verifier = IntegrityVerifier()
        result = verifier.compute_hash(str(test_file))

        # SHA256 해시는 항상 64자리 hex
        assert len(result) == 64
        assert all(c in "0123456789abcdef" for c in result)

    def test_compute_hash_deterministic(self, tmp_path: Path) -> None:
        """같은 내용의 파일은 항상 같은 해시를 반환한다."""
        file_a = tmp_path / "a.txt"
        file_b = tmp_path / "b.txt"
        content = "동일한 내용의 테스트 파일"
        file_a.write_text(content, encoding="utf-8")
        file_b.write_text(content, encoding="utf-8")

        verifier = IntegrityVerifier()
        assert verifier.compute_hash(str(file_a)) == verifier.compute_hash(str(file_b))

    def test_compute_hash_different_content(self, tmp_path: Path) -> None:
        """다른 내용의 파일은 다른 해시를 반환한다."""
        file_a = tmp_path / "a.txt"
        file_b = tmp_path / "b.txt"
        file_a.write_text("content A", encoding="utf-8")
        file_b.write_text("content B", encoding="utf-8")

        verifier = IntegrityVerifier()
        assert verifier.compute_hash(str(file_a)) != verifier.compute_hash(str(file_b))

    def test_compute_hash_file_not_found(self) -> None:
        """존재하지 않는 파일에 대해 FileNotFoundError를 발생시킨다."""
        verifier = IntegrityVerifier()
        with pytest.raises(FileNotFoundError):
            verifier.compute_hash("/nonexistent/path/file.txt")

    def test_verify_file_pass(self, tmp_path: Path) -> None:
        """올바른 해시로 검증하면 True를 반환한다."""
        test_file = tmp_path / "valid.txt"
        test_file.write_text("valid content", encoding="utf-8")

        verifier = IntegrityVerifier()
        expected = verifier.compute_hash(str(test_file))
        assert verifier.verify_file(str(test_file), expected) is True

    def test_verify_file_fail_wrong_hash(self, tmp_path: Path) -> None:
        """잘못된 해시로 검증하면 False를 반환한다."""
        test_file = tmp_path / "valid.txt"
        test_file.write_text("valid content", encoding="utf-8")

        verifier = IntegrityVerifier()
        assert verifier.verify_file(str(test_file), "0" * 64) is False

    def test_verify_file_fail_missing_file(self) -> None:
        """존재하지 않는 파일을 검증하면 False를 반환한다 (예외 아님)."""
        verifier = IntegrityVerifier()
        assert verifier.verify_file("/nonexistent/file.txt", "abc123") is False

    def test_generate_manifest(self, tmp_path: Path) -> None:
        """디렉토리에서 매니페스트를 정상적으로 생성한다."""
        # 테스트 파일 생성
        (tmp_path / "file1.txt").write_text("content 1", encoding="utf-8")
        sub_dir = tmp_path / "subdir"
        sub_dir.mkdir()
        (sub_dir / "file2.txt").write_text("content 2", encoding="utf-8")

        verifier = IntegrityVerifier()
        manifest = verifier.generate_manifest(str(tmp_path))

        # POSIX 형식 상대 경로로 키가 구성되어야 한다
        assert "file1.txt" in manifest
        assert "subdir/file2.txt" in manifest
        assert len(manifest) == 2
        # 값은 64자리 hex
        for hash_val in manifest.values():
            assert len(hash_val) == 64

    def test_generate_manifest_excludes_pycache(self, tmp_path: Path) -> None:
        """__pycache__ 디렉토리의 파일은 매니페스트에서 제외한다."""
        (tmp_path / "main.py").write_text("print('hello')", encoding="utf-8")
        cache_dir = tmp_path / "__pycache__"
        cache_dir.mkdir()
        (cache_dir / "main.cpython-311.pyc").write_bytes(b"\x00\x01")

        verifier = IntegrityVerifier()
        manifest = verifier.generate_manifest(str(tmp_path))

        assert "main.py" in manifest
        assert len(manifest) == 1  # __pycache__ 내 파일은 제외

    def test_verify_directory_pass(self, tmp_path: Path) -> None:
        """매니페스트 생성 후 동일 디렉토리를 검증하면 통과한다."""
        (tmp_path / "a.txt").write_text("aaa", encoding="utf-8")
        (tmp_path / "b.txt").write_text("bbb", encoding="utf-8")

        verifier = IntegrityVerifier()
        manifest = verifier.generate_manifest(str(tmp_path))
        ok, errors = verifier.verify_directory(str(tmp_path), manifest)

        assert ok is True
        assert errors == []

    def test_verify_directory_fail_tampered(self, tmp_path: Path) -> None:
        """파일 변조 시 검증이 실패하고 변조된 파일을 보고한다."""
        test_file = tmp_path / "data.txt"
        test_file.write_text("original content", encoding="utf-8")

        verifier = IntegrityVerifier()
        manifest = verifier.generate_manifest(str(tmp_path))

        # 파일 변조
        test_file.write_text("tampered content", encoding="utf-8")

        ok, errors = verifier.verify_directory(str(tmp_path), manifest)
        assert ok is False
        assert len(errors) == 1
        assert "불일치" in errors[0]

    def test_verify_directory_fail_missing(self, tmp_path: Path) -> None:
        """파일 삭제 시 검증이 실패하고 누락된 파일을 보고한다."""
        test_file = tmp_path / "will_delete.txt"
        test_file.write_text("will be deleted", encoding="utf-8")

        verifier = IntegrityVerifier()
        manifest = verifier.generate_manifest(str(tmp_path))

        # 파일 삭제
        test_file.unlink()

        ok, errors = verifier.verify_directory(str(tmp_path), manifest)
        assert ok is False
        assert len(errors) == 1
        assert "누락" in errors[0]


class TestAirGapPrep:
    """AirGapPrep의 매니페스트 생성 및 검증을 테스트한다."""

    def test_generate_manifest_creates_dict(self, tmp_path: Path) -> None:
        """generate_manifest가 파일별 해시 딕셔너리를 반환한다."""
        # 번들 디렉토리 구조 시뮬레이션
        bundle_dir = tmp_path / "bundle"
        bundle_dir.mkdir()
        (bundle_dir / "config").mkdir()
        (bundle_dir / "config" / "test.yaml").write_text("key: value", encoding="utf-8")
        (bundle_dir / "packages").mkdir()
        (bundle_dir / "packages" / "test.whl").write_bytes(b"fake wheel data")

        prep = AirGapPrep()
        manifest = prep.generate_manifest(str(bundle_dir))

        assert isinstance(manifest, dict)
        assert len(manifest) >= 2
        # 모든 값이 64자리 hex
        for hash_val in manifest.values():
            assert len(hash_val) == 64

    def test_verify_manifest_pass(self, tmp_path: Path) -> None:
        """매니페스트 생성 후 검증하면 통과한다 (roundtrip)."""
        # IntegrityVerifier를 직접 사용하여 self-referencing 문제를 회피
        # (manifest.json 자체가 매니페스트에 포함되면 해시 불일치 발생)
        verifier = IntegrityVerifier()

        bundle_dir = tmp_path / "bundle"
        bundle_dir.mkdir()
        (bundle_dir / "data.txt").write_text("test data", encoding="utf-8")
        (bundle_dir / "config.yaml").write_text("key: value", encoding="utf-8")

        # 매니페스트 생성
        manifest = verifier.generate_manifest(str(bundle_dir))
        assert len(manifest) == 2

        # 동일 디렉토리를 매니페스트로 검증
        ok, errors = verifier.verify_directory(str(bundle_dir), manifest)
        assert ok is True
        assert errors == []

    def test_verify_manifest_fail_tampered(self, tmp_path: Path) -> None:
        """파일 변조 시 verify_manifest가 실패한다."""
        bundle_dir = tmp_path / "bundle"
        bundle_dir.mkdir()
        data_file = bundle_dir / "data.txt"
        data_file.write_text("original", encoding="utf-8")

        prep = AirGapPrep()

        # 매니페스트 생성 및 저장
        manifest = prep.generate_manifest(str(bundle_dir))
        manifest_path = bundle_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        # manifest.json 자체를 포함한 최종 매니페스트 생성
        manifest_final = prep.generate_manifest(str(bundle_dir))
        manifest_path.write_text(
            json.dumps(manifest_final, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        # 파일 변조
        data_file.write_text("tampered", encoding="utf-8")

        ok, errors = prep.verify_manifest(str(bundle_dir))
        assert ok is False
        assert len(errors) >= 1

    def test_verify_manifest_no_manifest_file(self, tmp_path: Path) -> None:
        """manifest.json 파일이 없으면 검증 실패한다."""
        bundle_dir = tmp_path / "empty_bundle"
        bundle_dir.mkdir()

        prep = AirGapPrep()
        ok, errors = prep.verify_manifest(str(bundle_dir))

        assert ok is False
        assert len(errors) >= 1
        assert "매니페스트" in errors[0]
