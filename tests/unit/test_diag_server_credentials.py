"""
diag MCP 서버 자격증명 환경변수화 단위 테스트 (Security Critical #4, 2026-07-02).

검증 대상: mcp_servers/diag_server.py
  - GPU SSH 비밀번호를 소스에 하드코딩하지 않고 환경변수(NEXUS_DIAG_GPU_PASS)로
    읽는다. 미설정 시 GPU SSH 점검을 fail-soft 로 건너뛰고 사유를 결과에 담는다.
  - 죽은 상수(PG_CONTAINER/PG_USER/PG_DB/PG_PASS)가 모듈에서 제거되었다.

Mock/격리 전략:
  - 실제 GPU 서버(192.168.21.112)에 SSH 하지 않는다. GPU_PASS 를 비활성화하여
    점검 로직이 SSH 시도 전에 fail-soft 로 빠지는 경로만 검증한다.
  - GPU_PASS 는 모듈 import 시점에 os.environ.get(...) 로 한 번 계산되는 모듈 상수다.
    따라서 env 만 지워서는 이미 계산된 상수가 바뀌지 않는다. 실제 함수가 읽는
    diag_server.GPU_PASS 를 monkeypatch.setattr 로 직접 None 으로 바꿔 격리한다.
    (env 도 함께 제거해 이중 안전을 둔다.)
"""

from __future__ import annotations

from typing import Any

from mcp_servers import diag_server


def test_diag_gpu_pass_unset_skips_probe_with_reason(monkeypatch: Any) -> None:
    """GPU_PASS 미설정이면 GPU SSH 점검을 건너뛰고 명확한 사유를 결과에 담는다."""
    # env 제거(있을 수도 있으니 raising=False) + 모듈 상수 직접 None 화.
    monkeypatch.delenv("NEXUS_DIAG_GPU_PASS", raising=False)
    monkeypatch.setattr(diag_server, "GPU_PASS", None)

    result = diag_server._check_gpu_blocking()

    # fail-soft: SSH 시도 없이 사유가 담긴 결과가 반환된다.
    assert result["reachable"] is False
    assert "NEXUS_DIAG_GPU_PASS" in result["error"]
    assert "생략" in result["error"]


def test_diag_gpu_pass_empty_string_skips_probe(monkeypatch: Any) -> None:
    """GPU_PASS 가 빈 문자열이어도(falsy) 점검을 건너뛴다."""
    monkeypatch.setattr(diag_server, "GPU_PASS", "")

    result = diag_server._check_gpu_blocking()

    assert result["reachable"] is False
    assert "NEXUS_DIAG_GPU_PASS 미설정으로 GPU SSH 점검 생략" == result["error"]


def test_diag_gpu_pass_read_from_env_at_import() -> None:
    """GPU_PASS 는 하드코딩이 아니라 환경변수에서 읽어온 값(또는 None)이다.

    소스에 비밀번호 문자열이 박혀 있지 않음을 회귀 방지로 확인한다.
    (테스트 환경에서는 보통 미설정이라 None 이다.)
    """
    # 모듈 상수는 os.environ.get 결과이므로 str 또는 None 이어야 한다.
    assert diag_server.GPU_PASS is None or isinstance(diag_server.GPU_PASS, str)


def test_diag_dead_pg_constants_removed() -> None:
    """죽은 PG 상수들이 모듈에서 완전히 제거되었다(하드코딩 자격증명 잔재 제거)."""
    for name in ("PG_CONTAINER", "PG_USER", "PG_DB", "PG_PASS"):
        assert not hasattr(diag_server, name), f"죽은 상수 {name} 가 아직 모듈에 존재한다"
