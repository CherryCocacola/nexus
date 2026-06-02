"""
scripts/prepare_documents.py 배치 인제스트 스크립트의 단위 테스트 (v7.3 단계 10).

검증 의도:
  이 스크립트는 "데이터 준비" 용도의 배치 적재기다. 실제 PG/임베딩 서버/문서 적재는
  운영 환경에서만 일어나므로, 단위 테스트에서는 외부 의존성을 전부 mock/monkeypatch 해
  순수 제어 흐름(argparse 매핑, 파일 순회, 포맷 정규화, dry-run, 파일별 격리,
  스택 자동 선택, 리소스 정리)만 결정론적으로 검증한다.

mock 전략 (testing.md):
  - DocumentIngestPipeline / ModelProvider / KnowledgeStore / asyncpg 풀: 전부 가짜 객체.
  - 스크립트가 함수 내부에서 lazy import 하는 심볼은 원본 소스 모듈 위치에서 patch 한다
    (예: from core.ingest.pipeline import DocumentIngestPipeline →
    core.ingest.pipeline.DocumentIngestPipeline).
  - 스크립트 모듈 자체에 정의된 헬퍼(_build_model_provider/_build_parser_registry/
    _create_pg_pool/_gpu_available)는 scripts.prepare_documents 모듈 속성으로 patch 한다.
  - 실 임베딩 서버(:8002)/PostgreSQL(:5440)/GPU 에 절대 접근하지 않는다.

iter_input_files 는 외부 의존성이 없는 순수 함수이므로 tmp_path 로 직접 검증한다.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

import scripts.prepare_documents as pd


# ─────────────────────────────────────────────
# 공용 fake 헬퍼
# ─────────────────────────────────────────────
class _FakeRegistry:
    """ParserRegistry 대역 — run_ingest 가 호출하는 supported_extensions() 만 제공."""

    def __init__(self, exts: set[str]) -> None:
        self._exts = exts

    def supported_extensions(self) -> set[str]:
        return self._exts


class _FakePipeline:
    """
    DocumentIngestPipeline 대역.

    ingest_file 호출을 모두 기록하고, per-file 결과를 시나리오로 주입할 수 있다.
    결과가 Exception 인스턴스이면 그 예외를 raise 해 "파일별 격리"를 시험한다.
    """

    def __init__(self, results: list[Any] | None = None) -> None:
        # 호출 인자 기록: (Path, dry_run) 튜플 리스트
        self.calls: list[tuple[Path, bool]] = []
        # 파일별 반환/예외 시나리오. 부족하면 마지막 값을 반복 사용.
        self._results = results or [_ok_summary()]

    async def ingest_file(self, path: Path, dry_run: bool = False) -> dict[str, Any]:
        self.calls.append((path, dry_run))
        idx = min(len(self.calls) - 1, len(self._results) - 1)
        result = self._results[idx]
        if isinstance(result, Exception):
            raise result
        return result


def _ok_summary(
    *,
    chunk_count: int = 3,
    ingested: int = 3,
    warning_count: int = 0,
    warnings: list[str] | None = None,
    errors: list[str] | None = None,
    title: str = "문서",
) -> dict[str, Any]:
    """ingest_file 반환 dict 의 모든 필수 키를 채운 정상 요약(run_ingest 가 인덱싱하는 키들)."""
    return {
        "chunk_count": chunk_count,
        "ingested": ingested,
        "warning_count": warning_count,
        "warnings": warnings or [],
        "errors": errors or [],
        "title": title,
    }


class _SpyModelProvider:
    """LocalModelProvider 대역 — finally 블록의 close() 호출 여부를 감시한다."""

    def __init__(self) -> None:
        self.closed = False
        self.close = AsyncMock(side_effect=self._mark_closed)

    async def _mark_closed(self) -> None:
        self.closed = True


class _FakeKnowledgeStore:
    """KnowledgeStore 대역 — ensure_schema 만 비동기로 노출."""

    def __init__(self, *, pg_pool: Any = None) -> None:
        self.pg_pool = pg_pool
        self.ensure_schema = AsyncMock()


def _make_args(**overrides: Any) -> argparse.Namespace:
    """run_ingest 가 읽는 args 의 기본값을 채운 Namespace (개별 키만 override)."""
    base = {
        "input": ".",
        "formats": "",
        "embed_url": "",
        "pg": "",
        "stack": "light",
        "dry_run": False,
        "limit": 0,
        "build_index": False,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


@pytest.fixture
def patched_ingest(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """
    run_ingest 의 외부 의존성을 전부 가짜로 교체하는 공용 fixture.

    반환 dict 로 테스트가 spy 객체(pipeline/provider/store/created_pool)에 접근한다.
    스택/파서/GPU 는 기본적으로 light 고정·확장자 {.pdf,.pptx} 로 단순화한다.
    """
    state: dict[str, Any] = {}

    # config: 속성 접근만 하므로 MagicMock 으로 충분(실제 값은 model_provider 가 mock 이라 무의미).
    # run_ingest 는 함수 내부에서 core.config 를 lazy import 하므로 원본 모듈을 patch 한다.
    import core.config

    monkeypatch.setattr(core.config, "load_and_validate_config", lambda: MagicMock())

    # 파서 레지스트리 — light 스택, 확장자 두 종.
    registry = _FakeRegistry({".pdf", ".pptx"})
    monkeypatch.setattr(pd, "_build_parser_registry", lambda stack: registry)
    state["registry"] = registry

    # 모델 프로바이더(close spy).
    provider = _SpyModelProvider()
    monkeypatch.setattr(pd, "_build_model_provider", lambda config, embed_url: provider)
    state["provider"] = provider

    # PG 풀 생성 감시 — _create_pg_pool 호출 인자(dsn)를 기록하고 가짜 풀을 돌려준다.
    created_pool = AsyncMock()  # 반환되는 가짜 풀(close 추적)
    state["created_pool"] = created_pool
    state["pool_create_calls"] = []

    async def _fake_create_pool(config: Any, dsn: str | None) -> Any:
        state["pool_create_calls"].append(dsn)
        return created_pool

    monkeypatch.setattr(pd, "_create_pg_pool", _fake_create_pool)

    # KnowledgeStore lazy import 원본 patch.
    import core.rag.knowledge_store

    monkeypatch.setattr(core.rag.knowledge_store, "KnowledgeStore", _FakeKnowledgeStore)

    # DocumentIngestPipeline lazy import 원본 patch — 테스트별로 results 주입 가능.
    def _install_pipeline(results: list[Any] | None = None) -> _FakePipeline:
        pipeline = _FakePipeline(results=results)
        import core.ingest.pipeline

        monkeypatch.setattr(
            core.ingest.pipeline,
            "DocumentIngestPipeline",
            lambda **kwargs: pipeline,
        )
        state["pipeline"] = pipeline
        return pipeline

    state["install_pipeline"] = _install_pipeline
    return state


# ═════════════════════════════════════════════
# 1. argparse — main() 인자 파싱/검증/플래그 매핑
# ═════════════════════════════════════════════
def test_main_missing_input_without_build_index_raises_systemexit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--input 도 --build-index 도 없으면 parser.error → SystemExit (한글: 입력 누락 거부)."""
    monkeypatch.setattr("sys.argv", ["prepare_documents.py"])
    with pytest.raises(SystemExit):
        pd.main()


def test_main_invalid_stack_value_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--stack 은 choices(auto/light/high) 외 값을 거부한다(SystemExit)."""
    monkeypatch.setattr("sys.argv", ["prepare_documents.py", "--input", ".", "--stack", "ultra"])
    with pytest.raises(SystemExit):
        pd.main()


def test_main_flags_mapped_to_run_ingest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--limit/--dry-run/--input 이 run_ingest 의 args 로 그대로 전달되는지 검증."""
    captured: dict[str, Any] = {}

    async def _fake_run_ingest(args: argparse.Namespace) -> int:
        captured["args"] = args
        return 0

    monkeypatch.setattr(pd, "run_ingest", _fake_run_ingest)
    # main() 은 실제 asyncio.run 을 그대로 사용한다. async fixture 루프 스코프가
    # function 이므로(pyproject.toml) 이 호출이 다른 테스트의 루프를 깨지 않는다.
    monkeypatch.setattr(
        "sys.argv",
        ["prepare_documents.py", "--input", "/docs", "--dry-run", "--limit", "5"],
    )
    rc = pd.main()
    assert rc == 0
    args = captured["args"]
    assert args.input == "/docs"
    assert args.dry_run is True
    assert args.limit == 5
    assert args.build_index is False


def test_main_build_index_routes_to_run_build_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--build-index 가 있으면 --input 없이도 run_build_index 로 분기한다."""
    captured: dict[str, Any] = {}

    async def _fake_run_build_index(args: argparse.Namespace) -> int:
        captured["called"] = True
        captured["args"] = args
        return 0

    monkeypatch.setattr(pd, "run_build_index", _fake_run_build_index)
    # 실제 asyncio.run 사용 — function 스코프 루프라 전역 루프 파괴 위험 없음.
    monkeypatch.setattr("sys.argv", ["prepare_documents.py", "--build-index"])
    rc = pd.main()
    assert rc == 0
    assert captured["called"] is True
    assert captured["args"].build_index is True


# ═════════════════════════════════════════════
# 2. iter_input_files — 순수 함수(tmp_path)
# ═════════════════════════════════════════════
def test_iter_input_files_collects_recursively(tmp_path: Path) -> None:
    """rglob 재귀 — 중첩 하위 디렉토리의 파일도 수집한다."""
    (tmp_path / "a.pdf").write_text("x")
    sub = tmp_path / "sub" / "deep"
    sub.mkdir(parents=True)
    (sub / "b.pdf").write_text("x")

    files = pd.iter_input_files(tmp_path, {".pdf"})
    names = {p.name for p in files}
    assert names == {"a.pdf", "b.pdf"}


def test_iter_input_files_uppercase_extension_matches(tmp_path: Path) -> None:
    """확장자 비교는 소문자 정규화 — .PDF 도 .pdf 허용 집합에 매칭된다."""
    (tmp_path / "UPPER.PDF").write_text("x")
    files = pd.iter_input_files(tmp_path, {".pdf"})
    assert [p.name for p in files] == ["UPPER.PDF"]


def test_iter_input_files_disallowed_extension_excluded(tmp_path: Path) -> None:
    """허용 집합에 없는 확장자(.txt)는 제외한다."""
    (tmp_path / "keep.pdf").write_text("x")
    (tmp_path / "skip.txt").write_text("x")
    files = pd.iter_input_files(tmp_path, {".pdf"})
    assert [p.name for p in files] == ["keep.pdf"]


def test_iter_input_files_hidden_files_excluded(tmp_path: Path) -> None:
    """점(.)으로 시작하는 숨김/임시 파일은 확장자가 맞아도 제외한다."""
    (tmp_path / "visible.pdf").write_text("x")
    (tmp_path / ".hidden.pdf").write_text("x")
    files = pd.iter_input_files(tmp_path, {".pdf"})
    assert [p.name for p in files] == ["visible.pdf"]


def test_iter_input_files_deterministic_sorted_order(tmp_path: Path) -> None:
    """sorted 로 처리 순서가 결정론적 — 생성 순서와 무관하게 사전순."""
    for name in ["c.pdf", "a.pdf", "b.pdf"]:
        (tmp_path / name).write_text("x")
    files = pd.iter_input_files(tmp_path, {".pdf"})
    assert files == sorted(files)
    assert [p.name for p in files] == ["a.pdf", "b.pdf", "c.pdf"]


# ═════════════════════════════════════════════
# 3. --formats 정규화 (run_ingest 내부)
# ═════════════════════════════════════════════
async def test_formats_normalizes_dot_and_intersects_registered(
    tmp_path: Path,
    patched_ingest: dict[str, Any],
) -> None:
    """
    --formats "pptx,.pdf" → {".pptx",".pdf"} 로 정규화하고, 등록 확장자와의
    교집합만 처리한다. (registry 는 {.pdf,.pptx} 등록)
    """
    pipeline = patched_ingest["install_pipeline"]()
    (tmp_path / "a.pdf").write_text("x")
    (tmp_path / "b.pptx").write_text("x")

    args = _make_args(input=str(tmp_path), formats="pptx,.pdf", dry_run=True)
    rc = await pd.run_ingest(args)

    assert rc == 0
    processed = {p.suffix for (p, _dry) in pipeline.calls}
    assert processed == {".pdf", ".pptx"}


async def test_formats_unregistered_extension_ignored(
    tmp_path: Path,
    patched_ingest: dict[str, Any],
) -> None:
    """
    등록 파서가 없는 확장자(.docx)는 교집합에서 빠져 무시된다 — .pdf 만 처리.
    registry 는 {.pdf,.pptx} 만 등록.
    """
    pipeline = patched_ingest["install_pipeline"]()
    (tmp_path / "a.pdf").write_text("x")
    (tmp_path / "b.docx").write_text("x")

    args = _make_args(input=str(tmp_path), formats="pdf,docx", dry_run=True)
    rc = await pd.run_ingest(args)

    assert rc == 0
    processed = {p.suffix for (p, _dry) in pipeline.calls}
    assert processed == {".pdf"}


async def test_formats_empty_intersection_returns_error(
    tmp_path: Path,
    patched_ingest: dict[str, Any],
) -> None:
    """요청 확장자가 모두 미등록이면 처리할 확장자가 없어 rc=1 로 조기 종료한다."""
    patched_ingest["install_pipeline"]()
    (tmp_path / "a.pdf").write_text("x")

    args = _make_args(input=str(tmp_path), formats="docx,xlsx", dry_run=True)
    rc = await pd.run_ingest(args)
    assert rc == 1


# ═════════════════════════════════════════════
# 4. dry-run — PG 풀 미생성 + ingest_file(dry_run=True)
# ═════════════════════════════════════════════
async def test_dry_run_skips_pg_pool_and_passes_dry_run_true(
    tmp_path: Path,
    patched_ingest: dict[str, Any],
) -> None:
    """
    dry_run=True 면 _create_pg_pool 을 호출하지 않고(불필요한 DB 접속 회피),
    ingest_file 에 dry_run=True 가 전달된다.
    """
    pipeline = patched_ingest["install_pipeline"]()
    (tmp_path / "a.pdf").write_text("x")

    args = _make_args(input=str(tmp_path), dry_run=True)
    rc = await pd.run_ingest(args)

    assert rc == 0
    # PG 풀 생성 호출이 한 번도 없어야 한다.
    assert patched_ingest["pool_create_calls"] == []
    # 모든 ingest_file 호출에 dry_run=True 가 전달돼야 한다.
    assert pipeline.calls and all(dry is True for (_p, dry) in pipeline.calls)


async def test_non_dry_run_creates_pg_pool_and_passes_dry_run_false(
    tmp_path: Path,
    patched_ingest: dict[str, Any],
) -> None:
    """dry_run 이 아니면 PG 풀을 생성(_create_pg_pool 호출)하고 dry_run=False 를 전달한다."""
    pipeline = patched_ingest["install_pipeline"]()
    (tmp_path / "a.pdf").write_text("x")

    args = _make_args(input=str(tmp_path), dry_run=False, pg="postgresql://x")
    rc = await pd.run_ingest(args)

    assert rc == 0
    # dsn 이 _create_pg_pool 로 전달됐는지(한 번 호출).
    assert patched_ingest["pool_create_calls"] == ["postgresql://x"]
    assert pipeline.calls and all(dry is False for (_p, dry) in pipeline.calls)


# ═════════════════════════════════════════════
# 5. 파일별 격리 — 한 파일 예외 시 루프 계속 + failed 카운트
# ═════════════════════════════════════════════
async def test_per_file_isolation_runtime_error_does_not_break_loop(
    tmp_path: Path,
    patched_ingest: dict[str, Any],
) -> None:
    """
    가운데 파일에서 RuntimeError 가 나도 try/except 로 격리해 나머지 파일을 계속 처리한다.
    실패는 격리되고(예외가 run_ingest 밖으로 전파되지 않음), rc=0 으로 정상 종료한다.

    iter_input_files 는 사전순 정렬 → a.pdf, b.pdf, c.pdf 순으로 처리되며,
    b.pdf(2번째) 에서 예외를 던지도록 results 시나리오를 구성한다.
    """
    for name in ["a.pdf", "b.pdf", "c.pdf"]:
        (tmp_path / name).write_text("x")

    results = [_ok_summary(), RuntimeError("손상 파일"), _ok_summary()]
    pipeline = patched_ingest["install_pipeline"](results)

    args = _make_args(input=str(tmp_path), dry_run=True)
    rc = await pd.run_ingest(args)

    assert rc == 0
    # 예외에도 불구하고 3개 파일 모두 ingest_file 시도가 일어났다.
    assert [p.name for (p, _d) in pipeline.calls] == ["a.pdf", "b.pdf", "c.pdf"]


async def test_per_file_isolation_os_error_absorbed(
    tmp_path: Path,
    patched_ingest: dict[str, Any],
) -> None:
    """OSError(파일 손상/IO) 역시 포착 대상 — 첫 파일이 죽어도 다음 파일을 처리한다."""
    for name in ["a.pdf", "b.pdf"]:
        (tmp_path / name).write_text("x")

    results = [OSError("읽기 실패"), _ok_summary()]
    pipeline = patched_ingest["install_pipeline"](results)

    args = _make_args(input=str(tmp_path), dry_run=True)
    rc = await pd.run_ingest(args)

    assert rc == 0
    assert len(pipeline.calls) == 2


# ═════════════════════════════════════════════
# 6. --stack auto — GPU 감지로 high/light 분기
# ═════════════════════════════════════════════
async def test_stack_auto_with_gpu_builds_high(
    tmp_path: Path,
    patched_ingest: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--stack auto + _gpu_available()=True → _build_parser_registry 에 'high' 전달."""
    patched_ingest["install_pipeline"]()
    (tmp_path / "a.pdf").write_text("x")
    monkeypatch.setattr(pd, "_gpu_available", lambda: True)

    seen: dict[str, Any] = {}

    def _capture_registry(stack: str) -> Any:
        # _build_parser_registry 가 받은 stack 인자를 기록하고 가짜 레지스트리를 돌려준다.
        seen["stack"] = stack
        return patched_ingest["registry"]

    monkeypatch.setattr(pd, "_build_parser_registry", _capture_registry)

    args = _make_args(input=str(tmp_path), stack="auto", dry_run=True)
    rc = await pd.run_ingest(args)
    assert rc == 0
    assert seen["stack"] == "high"


async def test_stack_auto_without_gpu_builds_light(
    tmp_path: Path,
    patched_ingest: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--stack auto + _gpu_available()=False → _build_parser_registry 에 'light' 전달."""
    patched_ingest["install_pipeline"]()
    (tmp_path / "a.pdf").write_text("x")
    monkeypatch.setattr(pd, "_gpu_available", lambda: False)

    seen: dict[str, Any] = {}

    def _capture_registry(stack: str) -> Any:
        # _build_parser_registry 가 받은 stack 인자를 기록하고 가짜 레지스트리를 돌려준다.
        seen["stack"] = stack
        return patched_ingest["registry"]

    monkeypatch.setattr(pd, "_build_parser_registry", _capture_registry)

    args = _make_args(input=str(tmp_path), stack="auto", dry_run=True)
    rc = await pd.run_ingest(args)
    assert rc == 0
    assert seen["stack"] == "light"


# ═════════════════════════════════════════════
# 7. 리소스 정리 — finally 에서 model_provider.close()
# ═════════════════════════════════════════════
async def test_cleanup_provider_close_on_success(
    tmp_path: Path,
    patched_ingest: dict[str, Any],
) -> None:
    """정상 종료 경로의 finally 에서 model_provider.close() 가 호출된다."""
    patched_ingest["install_pipeline"]()
    (tmp_path / "a.pdf").write_text("x")

    args = _make_args(input=str(tmp_path), dry_run=True)
    rc = await pd.run_ingest(args)

    assert rc == 0
    assert patched_ingest["provider"].closed is True
    patched_ingest["provider"].close.assert_awaited_once()


async def test_cleanup_provider_close_even_when_all_files_fail(
    tmp_path: Path,
    patched_ingest: dict[str, Any],
) -> None:
    """
    모든 파일이 예외로 실패해도 finally 가 보장돼 provider.close() 가 호출된다.
    (파일 예외는 격리되므로 run_ingest 자체는 정상 종료한다)
    """
    for name in ["a.pdf", "b.pdf"]:
        (tmp_path / name).write_text("x")

    results = [RuntimeError("x"), RuntimeError("y")]
    patched_ingest["install_pipeline"](results)

    args = _make_args(input=str(tmp_path), dry_run=True)
    rc = await pd.run_ingest(args)

    assert rc == 0
    assert patched_ingest["provider"].closed is True


async def test_cleanup_pg_pool_closed_on_non_dry_run(
    tmp_path: Path,
    patched_ingest: dict[str, Any],
) -> None:
    """non-dry-run 에서 생성한 PG 풀도 finally 에서 close() 된다."""
    patched_ingest["install_pipeline"]()
    (tmp_path / "a.pdf").write_text("x")

    args = _make_args(input=str(tmp_path), dry_run=False)
    rc = await pd.run_ingest(args)

    assert rc == 0
    patched_ingest["created_pool"].close.assert_awaited_once()


# ═════════════════════════════════════════════
# 보강: 입력 디렉토리 검증 / 빈 디렉토리 / limit
# ═════════════════════════════════════════════
async def test_run_ingest_input_not_a_directory_returns_rc1(
    tmp_path: Path,
    patched_ingest: dict[str, Any],
) -> None:
    """--input 경로가 존재하지 않으면(디렉토리 아님) rc=1 로 조기 종료한다."""
    patched_ingest["install_pipeline"]()
    missing = tmp_path / "nope"
    args = _make_args(input=str(missing), dry_run=True)
    rc = await pd.run_ingest(args)
    assert rc == 1


async def test_run_ingest_no_matching_files_returns_rc0(
    tmp_path: Path,
    patched_ingest: dict[str, Any],
) -> None:
    """디렉토리는 있으나 허용 확장자 파일이 없으면 rc=0 으로 조용히 종료한다."""
    pipeline = patched_ingest["install_pipeline"]()
    (tmp_path / "ignore.txt").write_text("x")
    args = _make_args(input=str(tmp_path), dry_run=True)
    rc = await pd.run_ingest(args)
    assert rc == 0
    assert pipeline.calls == []


async def test_limit_processes_only_top_n_files(
    tmp_path: Path,
    patched_ingest: dict[str, Any],
) -> None:
    """--limit 2 면 정렬된 상위 2개 파일만 ingest_file 로 넘어간다."""
    for name in ["a.pdf", "b.pdf", "c.pdf"]:
        (tmp_path / name).write_text("x")
    pipeline = patched_ingest["install_pipeline"]()

    args = _make_args(input=str(tmp_path), dry_run=True, limit=2)
    rc = await pd.run_ingest(args)
    assert rc == 0
    assert [p.name for (p, _d) in pipeline.calls] == ["a.pdf", "b.pdf"]


# ═════════════════════════════════════════════
# _gpu_available — torch 부재/예외 fail-soft
# ═════════════════════════════════════════════
def test_gpu_available_returns_false_when_torch_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    torch import 가 실패(에어갭/미설치)하면 예외를 흡수해 False(=GPU 없음)를 반환한다.
    builtins.__import__ 를 가로채 'torch' 만 ImportError 로 만든다.
    """
    import builtins

    real_import = builtins.__import__

    def _fake_import(name: str, *a: Any, **k: Any) -> Any:
        if name == "torch":
            raise ImportError("no torch (airgap)")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    assert pd._gpu_available() is False
