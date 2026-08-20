# 웹 표면이 경로 권한 검사를 실제로 거치는지, 허용 디렉토리가 기능을 살리는지 검증한다.
"""
웹 경로 권한 배선 테스트 (2026-08-20).

무엇을 막는가 (실증된 결함):
    웹 `base_options` 에 `permission_pipeline` 키가 없어 executor 가 Layer 2 경로
    검사를 통째로 건너뛰었다. 그 결과 **API 키 하나로** `/app/config/tenants.yaml`
    (전 테넌트 키)과 `/etc/passwd` 가 읽혔다. 테넌트 키가 새면 세션 격리도 우회된다.
    과거 "웹 Bash 제거" 사고와 같은 계열이다.

왜 허용 디렉토리가 필요한가:
    웹은 세션별 샌드박스를 cwd 로 쓴다. 업로드 파일·생성물은 그 밖에 있어서,
    경로 검사를 켜기만 하면 문서 첨부 분석과 다운로드가 통째로 죽는다.
    **막는 것과 살리는 것을 함께 검증한다** — 차단만 보는 테스트는 절반짜리다.
"""

from __future__ import annotations

import inspect

from core.security.path_guard import PathGuard

SANDBOX = "/app/.nexus/sessions/session-abc/workspace"
UPLOADS = "/app/data/uploads"
EXPORTS = "/app/data/exports"


def _guard() -> PathGuard:
    return PathGuard(allowed_dirs=[UPLOADS, EXPORTS])


# ─────────────────────────────────────────────
# 차단 — 민감 경로
# ─────────────────────────────────────────────
def test_tenant_key_file_is_blocked():
    """테넌트 키 파일이 읽히면 그 키로 다른 테넌트 세션까지 열린다."""
    ok, reason = _guard().is_path_safe("/app/config/tenants.yaml", SANDBOX)
    assert ok is False, "테넌트 키 파일이 열렸다"
    assert reason


def test_system_files_are_blocked():
    for path in ("/etc/passwd", "/etc/shadow", "/proc/1/environ"):
        ok, _ = _guard().is_path_safe(path, SANDBOX)
        assert ok is False, f"시스템 파일이 열렸다: {path}"


def test_source_code_is_blocked():
    ok, _ = _guard().is_path_safe("/app/web/app.py", SANDBOX)
    assert ok is False


def test_other_session_sandbox_is_blocked():
    """다른 세션의 작업 디렉토리도 밖이다 — 세션 간 파일 열람을 막는다."""
    ok, _ = _guard().is_path_safe("/app/.nexus/sessions/other/workspace/note.txt", SANDBOX)
    assert ok is False


# ─────────────────────────────────────────────
# 허용 — 기능이 살아 있어야 한다
# ─────────────────────────────────────────────
def test_uploaded_document_is_readable():
    """업로드 문서를 못 읽으면 첨부 분석이 통째로 죽는다."""
    ok, reason = _guard().is_path_safe(f"{UPLOADS}/upload-abc123.txt", SANDBOX)
    assert ok is True, f"업로드 파일이 막혔다: {reason}"


def test_generated_artifact_is_readable():
    ok, reason = _guard().is_path_safe(f"{EXPORTS}/document-abc.docx", SANDBOX)
    assert ok is True, f"생성물이 막혔다: {reason}"


def test_own_sandbox_is_readable():
    ok, _ = _guard().is_path_safe(f"{SANDBOX}/note.txt", SANDBOX)
    assert ok is True


def test_secrets_inside_allowed_dir_are_still_protected():
    """허용 디렉토리라고 무조건 통과가 아니다 — 보호 패턴은 그대로 적용된다."""
    ok, reason = _guard().is_path_safe(f"{UPLOADS}/.env", SANDBOX)
    assert ok is False and "보호" in reason


def test_traversal_from_allowed_dir_is_blocked():
    """허용 디렉토리를 발판으로 밖으로 나가려는 시도도 막는다."""
    ok, _ = _guard().is_path_safe(f"{UPLOADS}/../../config/tenants.yaml", SANDBOX)
    assert ok is False


# ─────────────────────────────────────────────
# 무회귀
# ─────────────────────────────────────────────
def test_without_allowed_dirs_behavior_is_unchanged():
    """허용 목록 미지정이면 종전 그대로 — cwd 밖은 전부 차단."""
    plain = PathGuard()
    assert plain.is_path_safe(f"{UPLOADS}/u.txt", SANDBOX)[0] is False
    assert plain.is_path_safe(f"{SANDBOX}/n.txt", SANDBOX)[0] is True


def test_broken_allowed_dir_does_not_break_guard():
    """허용 경로가 이상해도 가드 전체가 죽지 않는다."""
    guard = PathGuard(allowed_dirs=["", "\x00bad"])
    assert guard.is_path_safe(f"{SANDBOX}/n.txt", SANDBOX)[0] is True
    assert guard.is_path_safe("/etc/passwd", SANDBOX)[0] is False


# ─────────────────────────────────────────────
# 배선 — 값이 아니라 경로가 연결됐는가
# ─────────────────────────────────────────────
def test_web_base_options_carries_permission_pipeline():
    """이 키가 빠지면 executor 가 경로 검사를 건너뛴다 — 실제로 그래서 유출됐다."""
    from web import app as web_app

    src = inspect.getsource(web_app)
    i = src.index("base_options = {")
    block = src[i : i + 4000]
    assert '"permission_pipeline"' in block, "웹 base_options 에 권한 파이프라인이 없다"


def test_bootstrap_injects_allowed_dirs():
    """부트스트랩이 업로드·생성물을 허용 목록으로 넘겨야 웹 기능이 산다."""
    from core import bootstrap

    src = inspect.getsource(bootstrap)
    assert "PathGuard(allowed_dirs=" in src
    assert "resolve_uploads_dir" in src and "resolve_exports_dir" in src
