"""
mcp_servers/docingest_server.py — GPU 유무에 따른 PDF 파서 라우팅 단위 테스트.

검증 대상 (후속 구현 C — 레지스트리):
  build_app 이 _gpu_available() 결과에 따라 ParserRegistry 를 다르게 구성하는지.
    - GPU 있음: DoclingParser(.pdf, priority=10)를 등록 → .pdf 가 Docling 으로 라우팅.
    - GPU 없음: DoclingParser 미등록 → .pdf 는 pdfplumber(경량)로 라우팅.

운영 의존성/모델/네트워크 우회 (핵심):
  build_app 은 LocalModelProvider/asyncpg/KnowledgeStore 를 실제로 조립하지만,
  이 테스트가 검증하려는 것은 "파서 레지스트리 라우팅 결정" 하나뿐이다. 따라서:
    - _gpu_available 을 mock 해 GPU 분기를 통제한다.
    - asyncpg.create_pool 을 mock 해 ImportError/실패 → 인메모리 폴백(실 PG 미접속,
      ensure_schema DDL 미실행)로 흘린다.
    - LocalModelProvider 를 더미로 대체해 임베딩 서버 접속 가능성을 차단한다.
    - DocumentIngestPipeline 을 가로채(capture) 생성자에 전달된 parser_registry 를
      포착한다 — 이 레지스트리의 get_for_path 로 .pdf 라우팅을 직접 검증한다.
  DoclingParser/PdfPlumberParser 는 import 만으로 모델을 적재하지 않으므로
  (변환/추론은 parse() 시점) 실제 파서 인스턴스를 그대로 쓴다(에어갭/오프라인).
"""

from __future__ import annotations

from unittest.mock import patch

import mcp_servers.docingest_server as docingest_mod
from core.ingest.parsers.docling_layout import DoclingParser
from core.ingest.parsers.pdf_plumber import PdfPlumberParser


# ─────────────────────────────────────────────
# 더미 임베딩 프로바이더 — 네트워크 차단
# ─────────────────────────────────────────────
class _DummyModelProvider:
    """build_app 이 만드는 LocalModelProvider 자리 — 어떤 인자든 받고 아무것도 안 한다."""

    def __init__(self, *args, **kwargs) -> None:
        pass

    async def close(self) -> None:
        """리소스 정리(가짜)."""


# ─────────────────────────────────────────────
# DocumentIngestPipeline 가로채기 — parser_registry 포착
# ─────────────────────────────────────────────
class _CapturingPipeline:
    """
    DocumentIngestPipeline 자리에 끼워 생성자 인자(parser_registry)를 포착하는 가짜.

    build_app 은 pipeline 을 DocParseTool/DocIngestTool 에 넘기지만, 이 테스트는
    pipeline 의 동작이 아니라 "어떤 레지스트리로 조립됐는지"만 본다. 클래스 변수에
    마지막으로 받은 레지스트리를 저장해 테스트가 꺼내 본다.
    """

    last_registry = None

    def __init__(self, parser_registry, model_provider, knowledge_store) -> None:
        type(self).last_registry = parser_registry
        self._registry = parser_registry


async def _build_app_capture_registry(gpu: bool):
    """
    _gpu_available 을 gpu 로 고정하고 build_app 을 돌려, 조립에 쓰인
    ParserRegistry 를 돌려준다(네트워크/모델/실 PG 미접속).
    """
    _CapturingPipeline.last_registry = None
    # build_app 은 함수 본문에서 from ... import 로 심볼을 끌어오므로, import 대상
    # 모듈(core.model.inference / core.ingest.pipeline)을 직접 patch 해야 한다.
    with (
        patch.object(docingest_mod, "_gpu_available", return_value=gpu),
        patch("asyncpg.create_pool", side_effect=ConnectionError("테스트: PG 미접속")),
        patch("core.model.inference.LocalModelProvider", _DummyModelProvider),
        patch("core.ingest.pipeline.DocumentIngestPipeline", _CapturingPipeline),
    ):
        await docingest_mod.build_app(api_key="local-key")

    assert _CapturingPipeline.last_registry is not None
    return _CapturingPipeline.last_registry


# ─────────────────────────────────────────────
# GPU 있음 → Docling 우선 라우팅
# ─────────────────────────────────────────────
class TestRoutingWithGpu:
    """GPU 호스트에서 .pdf 가 DoclingParser 로 라우팅되는지 검증한다."""

    async def test_pdf_routes_to_docling_when_gpu_present(self, tmp_path):
        """GPU 있음: DoclingParser(priority=10)가 pdfplumber 보다 우선 선택된다."""
        registry = await _build_app_capture_registry(gpu=True)

        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(b"%PDF-1.7\n...")
        parser = registry.get_for_path(pdf)

        assert isinstance(parser, DoclingParser)

    async def test_pdf_extension_registered_with_gpu(self, tmp_path):
        """GPU 있음: .pdf 확장자가 레지스트리에 등록되어 있어야 한다."""
        registry = await _build_app_capture_registry(gpu=True)
        assert ".pdf" in registry.supported_extensions()


# ─────────────────────────────────────────────
# GPU 없음 → pdfplumber 폴백
# ─────────────────────────────────────────────
class TestRoutingWithoutGpu:
    """GPU 없는 호스트에서 .pdf 가 pdfplumber 경량 파서로 폴백되는지 검증한다."""

    async def test_pdf_routes_to_pdfplumber_when_no_gpu(self, tmp_path):
        """GPU 없음: DoclingParser 미등록 → .pdf 는 PdfPlumberParser 로 라우팅된다."""
        registry = await _build_app_capture_registry(gpu=False)

        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(b"%PDF-1.7\n...")
        parser = registry.get_for_path(pdf)

        assert isinstance(parser, PdfPlumberParser)

    async def test_docling_not_registered_when_no_gpu(self, tmp_path):
        """GPU 없음: 어떤 .pdf 경로로도 DoclingParser 가 선택되지 않는다."""
        registry = await _build_app_capture_registry(gpu=False)

        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(b"%PDF-1.7\n...")
        parser = registry.get_for_path(pdf)

        assert not isinstance(parser, DoclingParser)
