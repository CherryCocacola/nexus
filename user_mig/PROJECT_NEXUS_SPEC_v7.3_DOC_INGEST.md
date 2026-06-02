# Project Nexus — 기술 사양서 v7.3 (신규 기능 설계)

## 문서 양식(레이아웃) 인식 임베딩 파이프라인

**버전**: 7.3 (Document Layout-Aware Ingestion — Feature Design)
**기준 문서**: PROJECT_NEXUS_SPEC_v7.2_AMENDMENT.md (MCP), v7.1_AMENDMENT.md (운영 정합화)
**작성일**: 2026-06-01
**상태**: 제품 설계 검토본 (구현 전)

---

## 개정 개요

### 왜 v7.3인가

v7.0~v7.2는 **운영 정합화·MCP 통합** 등 기존 구조의 확장이었다. v7.3은
그와 성격이 다른 **신규 기능 설계 문서**다. HWP/PDF/PPT 등 사내 문서를
"양식(그림·도표·문서 템플릿)을 인식해" 파싱하고, **문서의 구조를 보존한
채 청킹**하여 벡터 임베딩(tb_knowledge)으로 적재하는 파이프라인을 정의한다.

이 기능은 v7.2 MCP 방향과 정합한다. 무거운 파싱·레이아웃 모델·OCR은
Machine A(오케스트레이터) 본류에 두지 않고 **LAN MCP 서버("문서 인제스트
MCP 서버")로 떼어내** v7.2의 `McpToolAdapter`로 호출하거나, `prepare_kowiki.py`
류의 **배치 인제스트** 경로로 적재한다.

### v7.3가 하는 것 / 하지 않는 것

| 한다 | 하지 않는다 |
|---|---|
| `DocumentParser` 추상 인터페이스 + 포맷별 구현체 설계 | 4-Tier 체인·권한·Hook 구조 변경 |
| 구조 트리(섹션→소제목→[문단\|표\|그림캡션]) 기반 계층형 청킹 | 기존 `DocumentProcessTool` 즉시 폐기(공존·점진 이관) |
| GPU 티어별(경량/고품질) 스택 런타임 분기 | 외부 SaaS/클라우드 OCR·파싱 API 사용 |
| 청정(MIT/Apache) 기본 + 상용 라이선스 어댑터 슬롯 | AGPL/GPL 라이브러리를 기본 스택에 포함 |
| tb_knowledge `metadata` jsonb에 구조 메타 저장 | tb_knowledge 스키마 컬럼 변경(기존 jsonb 활용) |

### 3가지 불변 전제 (v7.0~v7.2 그대로 유지)

1. Claude Code 설계안 그대로 — 4-Tier 체인, 5계층 권한, Hook, Thinking, Memory 유지
2. GPU 업그레이드 시 성능 향상 — 인제스트 품질도 GPU 티어와 직교적으로 향상
3. 에어갭 절대 준수 — 외부 네트워크·클라우드 API 금지, 모든 wheel/모델은
   오프라인 사전 번들(`deployment/`), LAN URL만

### 제품화 전제 (사용자 확정)

- **GPU 티어 진화 경로**: 최초 RTX 5090(32GB) → 자체 H200(141GB, ~2026-09)
  → **고객 배포 타겟 A100(80GB)**. 제품 검증 기준선은 **A100**.
- **라이선스 정책**: 청정(MIT/Apache) 기본 구현 + 상용 라이선스 구매 시
  교체 가능한 **어댑터 슬롯**. AGPL/GPL은 기본 스택 배제.

> **코드 확인 주의 (구현 완료, 2026-06-02)**: `core/model/gpu_detector.py::GPUTier`
> 에 **`A100 = "a100"`이 추가되었다**(`:32`). 멤버는 이제
> `RTX_5090 / A100 / H100 / H200 / MULTI_GPU` 5종이다. A100/H100 은 VRAM 이 같아
> 감지 로직(`:161~169`)이 `vram_gb > 60` 구간에서 GPU 이름에 "a100"이 포함되면
> A100, 아니면 H100 으로 판정한다. A100 프로파일(`:225~262`)은 BF16 풀 프리시전
> (Ampere — **FP8 미지원**)으로 정의되어 있다. 상세는 Part 4.4 참조.

---

## Part 0: 변경 없는 챕터 (재확인)

문서 인제스트는 도구·배치 스크립트·(선택적) MCP 서버를 추가하는 것일 뿐,
그 산출물이 거치는 파이프라인은 기존과 동일하다.

| 챕터/시스템 | 이유 |
|---|---|
| 4-Tier AsyncGenerator 체인 | 인제스트 도구도 일반 도구처럼 query_loop을 통과. 체인 무관 |
| 5계층 Permission Pipeline | 신규 도구도 BaseTool로 동일 파이프라인 통과 |
| Hook 시스템 | 동일 Hook 파이프라인 적용 |
| Thinking / Memory(tb_memories) | 인제스트는 도구·RAG(tb_knowledge) 계층. 무관 |
| Training Pipeline | 학습 대상 모델 불변 |
| tb_knowledge 스키마 | 컬럼 추가 없음 — 기존 `metadata jsonb` 활용 (Part 2.5) |
| 임베딩 서버 `/v1/embed` 계약 | v7.1 Part 2 그대로 — 인제스트도 동일 계약 사용 |

---

## Part 1: 개요 및 목표

### 1.1 문제 정의 — "충돌 없는 임베딩"이란

사용자 요구의 핵심은 **"소제목 안 텍스트 영역이 다른 항목과 충돌 없이"**
임베딩되는 것이다. 이를 설계 용어로 정의하면:

> **구조 보존 청킹(structure-preserving chunking)**: 문서의 시각적·논리적
> 경계(섹션, 소제목, 표 셀, 그림 캡션, 슬라이드, 텍스트박스)를 인식하여,
> **서로 다른 논리 영역의 텍스트가 한 청크에 뒤섞이지 않도록** 분할하는 것.

"충돌"의 구체 사례:
- 소제목 A 본문과 인접한 소제목 B 본문이 한 청크에 합쳐져 검색 시 오답 유발
- 표의 셀 텍스트가 본문 단락과 섞여 행/열 의미가 소실
- 슬라이드의 좌측 텍스트박스와 우측 텍스트박스가 읽기 순서 없이 직렬화
- 그림 캡션이 본문 문장 중간에 끼어 의미 단절

### 1.2 기존 DocumentProcessTool의 한계 (확인된 사실)

`core/tools/implementations/document_tool.py`(실측)는 다음 한계를 가진다:

| 항목 | 현 DocumentProcessTool | 한계 |
|---|---|---|
| 지원 포맷 | PDF(pypdf), DOCX(python-docx), XLSX(openpyxl) | **HWP/HWPX/PPTX 미지원** |
| 추출 방식 | 텍스트만 평면 추출 (`extract_text()`) | **레이아웃·읽기순서·표 구조 소실** |
| 청킹 | `CHUNK_SIZE=2500자` 단순 글자수 분할(`_split_chunks`) | **소제목/표/그림 경계 무시 → 충돌 발생** |
| 표 처리 | DOCX만 ` \| ` 단순 연결(`_parse_docx`) | 행/열 구조·헤더 의미 소실 |
| 스캔 PDF | 텍스트 없으면 `"스캔 이미지일 수 있음"` 문자열 반환 | **OCR 없음** |
| 용도 | Worker가 **읽기**용으로 호출(읽기 전용, 8K 컨텍스트 청크) | RAG **적재**용 아님 |

핵심: **DocumentProcessTool은 "Worker가 한 문서를 읽는" 실시간 도구**이고,
v7.3 인제스트 파이프라인은 **"문서를 구조 보존 청킹해 tb_knowledge에 적재하는"
배치/검색 기반(RAG)** 이다. 둘은 목적이 다르므로 **즉시 폐기하지 않고 공존**
시키되, 인제스트 산출물이 RAG에 쌓이면 DocumentProcessTool의 호출 빈도는
자연 감소한다 (Part 9 로드맵).

### 1.3 목표

1. **구조 인식**: 포맷별 네이티브 구조(슬라이드/도형/표/헤딩/문단)를 인식
2. **구조 보존 청킹**: 논리 경계에서 분할, heading_path로 부모-자식 관계 유지
3. **계층형 검색**: 검색용 소청크(128~256t) + 컨텍스트용 부모청크(512~1536t)
4. **에어갭·청정 라이선스**: 모든 의존성 오프라인 번들 + MIT/Apache 기본
5. **GPU 티어 적응**: 경량(저VRAM/CPU) ↔ 고품질(GPU 레이아웃 모델) 런타임 분기
6. **제품 안전성**: 고객 배포 시 라이선스 리스크 0 (AGPL/GPL 배제)

---

## Part 2: 아키텍처

### 2.1 전체 파이프라인

```
[문서 입력: PDF/PPTX/HWPX/HWP/스캔]
        │
        ▼
┌──────────────────────────────────────────────┐
│  DocumentParser (추상 인터페이스)               │   ← 포맷별 구현체로 분기
│   PdfPlumberParser / PptxParser / HwpxParser   │
│   / DoclingParser / OcrParser ...              │
└──────────────────────────────────────────────┘
        │ 출력: DocumentTree (구조 트리)
        ▼
┌──────────────────────────────────────────────┐
│  구조 트리 (DocumentTree)                       │
│   Section → Subheading → [Paragraph|Table|     │
│                           FigureCaption|...]    │
│   각 노드: element_type, heading_path, page,    │
│            bbox(선택), order                     │
└──────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────┐
│  StructureAwareChunker (계층형 청킹)            │
│   - 소제목/표/슬라이드 경계에서만 분할          │
│   - 부모청크(컨텍스트) + 소청크(검색) 동시 생성 │
└──────────────────────────────────────────────┘
        │ 출력: list[DocumentChunk]
        ▼
┌──────────────────────────────────────────────┐
│  임베딩 (v7.1 Part 2 계약 그대로)               │
│   POST /v1/embed {"texts":["passage: ..."]}    │
│   → 192.168.22.28:8002 (e5-large, 1024차원)     │
└──────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────┐
│  적재 (KnowledgeStore.add — 기존 코드)          │
│   tb_knowledge 1행 + metadata jsonb에 구조 메타 │
└──────────────────────────────────────────────┘
```

### 2.2 의존성 방향 (P2 준수)

신규 모듈 위치: `core/ingest/` (★ = 구현 완료, ○ = 후속/어댑터 슬롯)

```
core/ingest/                        ← 신규 디렉토리 (구현됨)
  types.py            ★ DocumentTree, DocumentNode, DocumentChunk (Pydantic v2/frozen)
  parser_base.py      ★ DocumentParser (ABC), ParserRegistry (priority 기반 선택/폴백)
  parsers/            — 포맷별 구현체
    pdf_plumber.py    ★ PdfPlumberParser (pdfplumber, MIT — PDF 경량)
    pptx.py           ★ PptxParser (python-pptx, MIT)
    hwpx.py           ★ HwpxParser (python-hwpx 1차 + zipfile/xml.etree 폴백, OWPML)
    docling_layout.py ○ DoclingParser (Docling, GPU 권장) — 미구현(어댑터 슬롯)
    ocr_paddle.py     ○ PaddleOcrParser (PaddleOCR/PP-Structure, GPU) — 미구현(슬롯)
    ocr_tesseract.py  ○ TesseractParser (Tesseract5, CPU) — 미구현(슬롯)
    hwp_libreoffice.py○ HwpViaLibreOffice (.hwp→.docx subprocess) — 미구현(후속)
  chunker.py          ★ StructureAwareChunker (계층형 청킹)
  pipeline.py         ★ DocumentIngestPipeline (parse→chunk→embed→적재 오케스트레이션)
```

> `parsers/__init__.py`가 현재 export 하는 것은 `HwpxParser`/`PdfPlumberParser`/
> `PptxParser` 3종이다. docling/ocr/hwp 파서는 ParserRegistry 의 priority 슬롯에
> 더 높은 우선순위로 끼우면 동일 확장자에서 자동 우선되고, 미등록 시 청정 기본
> 파서로 폴백된다(과설계 없이 후속 확장 가능한 구조).

의존성 방향 (단방향만):
```
core/ingest/ → core/rag/(KnowledgeStore), core/model/(embed, gpu_detector), core/config
core/ingest/ ← (없음 — core/rag·core/model이 ingest를 import하지 않음)
scripts/(배치) → core/ingest/        (prepare_kowiki.py 패턴)
core/tools/mcp/ 서버측 → core/ingest/  (MCP 서버 구현이 ingest 재사용 — Part 5)
```

`core/ingest/`는 **무거운 파싱·OCR·레이아웃 모델 의존성을 격리**한다. 본류
Machine A는 이 의존성을 직접 짊어지지 않고, MCP 서버 또는 배치 스크립트
쪽에서만 import한다 (에어갭·VRAM·기동 시간 부담 최소화).

### 2.3 DocumentParser 추상 인터페이스 (제안 — 발췌)

```python
# core/ingest/parser_base.py (제안 — 인터페이스 발췌, 한글 주석)
class DocumentParser(ABC):
    """
    문서 1개를 구조 트리(DocumentTree)로 파싱하는 추상 인터페이스.

    포맷별 구현체(PDF/PPTX/HWPX/스캔)가 이 계약을 따른다.
    라이선스 정책상 기본 구현은 MIT/Apache 라이브러리만 사용하며,
    상용 라이브러리(예: 구매한 PDF SDK)는 같은 인터페이스를 구현하는
    "어댑터 슬롯"으로 교체할 수 있다 (Part 6).
    """

    @property
    @abstractmethod
    def supported_extensions(self) -> tuple[str, ...]:
        """이 파서가 처리하는 확장자 (예: ('.pdf',))."""

    @property
    @abstractmethod
    def requires_gpu(self) -> bool:
        """GPU 레이아웃/OCR 모델을 쓰는지 — 티어 분기 판단에 사용 (Part 4)."""

    @abstractmethod
    def can_parse(self, path: Path) -> bool:
        """확장자·매직바이트로 처리 가능 여부 판단 (fail-closed: 불확실하면 False)."""

    @abstractmethod
    async def parse(self, path: Path) -> "DocumentTree":
        """
        문서를 구조 트리로 파싱한다.
        실패는 예외가 아닌 부분 트리 + 경고 노드로 표현 가능(fail-soft 파싱).
        외부 네트워크 호출 절대 금지 (에어갭).
        """
```

```python
# core/ingest/types.py (제안 — 발췌)
class ElementType(str, Enum):
    """구조 트리 노드의 종류 — metadata.element_type로 저장된다."""
    SECTION = "section"            # 큰 제목 (=h1)
    SUBHEADING = "subheading"      # 소제목 (=h2/h3) — "충돌 방지"의 핵심 경계
    PARAGRAPH = "paragraph"        # 본문 단락
    TABLE = "table"                # 표 (행/열 구조 보존)
    FIGURE_CAPTION = "figure_caption"  # 그림/도표 캡션
    LIST_ITEM = "list_item"        # 목록 항목
    SLIDE = "slide"                # PPTX 슬라이드 단위
    TEXTBOX = "textbox"            # PPTX/HWP 텍스트박스

class DocumentNode(BaseModel):
    """구조 트리의 한 노드 (frozen)."""
    model_config = ConfigDict(frozen=True)
    element_type: ElementType
    text: str
    heading_path: tuple[str, ...] = ()   # 루트→현재까지의 소제목 경로
    page: int | None = None              # PDF/스캔 페이지 (PPTX는 슬라이드 번호)
    order: int = 0                       # 읽기 순서 (PPTX는 (top,left) 정렬 결과)
    children: tuple["DocumentNode", ...] = ()

class DocumentChunk(BaseModel):
    """청킹 후 임베딩 단위 — KnowledgeEntry로 변환되어 적재된다."""
    content: str                         # 임베딩 대상 텍스트
    heading_path: tuple[str, ...] = ()
    element_type: ElementType
    page: int | None = None
    parent_chunk_id: str | None = None   # 계층형: 소청크 → 부모청크 참조
    is_parent: bool = False              # True=컨텍스트용 부모, False=검색용 소청크
```

### 2.4 query_loop·MCP 계약 정합 (P3 유지)

인제스트가 **실시간 도구**로 노출될 때(Part 5의 MCP 또는 로컬 도구)는
기존 `BaseTool` 계약을 그대로 따른다. query_loop은 OpenAI `tool_calls`
형식만 보고, 파싱·OCR·레이아웃 모델의 존재를 전혀 모른다. MCP 경로일 경우
JSON-RPC는 v7.2 `McpToolAdapter.call()` 아래에만 존재한다.

### 2.5 적재 — tb_knowledge metadata jsonb 매핑 (스키마 변경 없음)

`core/rag/knowledge_store.py`(실측)의 `tb_knowledge`는 이미 `metadata jsonb`
컬럼과 `KnowledgeEntry.metadata: dict` 필드를 가진다. 구조 메타를 **컬럼 추가
없이** jsonb로 저장한다:

```python
# DocumentChunk → KnowledgeEntry 변환 (제안)
KnowledgeEntry(
    source="docingest",                 # kowiki와 구분되는 소스
    title=document_title,
    section=" > ".join(chunk.heading_path) or None,  # 기존 section 컬럼 활용
    content=chunk.content,
    chunk_index=idx,
    total_chunks=total,
    embedding=tuple(embedding),
    metadata={                          # ← 구조 메타는 전부 jsonb로
        "element_type": chunk.element_type.value,
        "heading_path": list(chunk.heading_path),
        "page": chunk.page,
        "parent_chunk_id": chunk.parent_chunk_id,
        "is_parent": chunk.is_parent,
        "ingested_by": "doc_ingest_pipeline",
        "doc_format": "pptx",           # 출처 포맷
    },
)
```

- `KnowledgeEntry.id`는 source/title/section/chunk_index의 SHA-256(실측) →
  **재적재 멱등성** 자동 보장 (UPSERT, `prepare_kowiki.py`와 동일).
- 검색 시 `search_by_vector`의 반환 dict에 `metadata`가 이미 포함되므로
  (실측 라인 234), 호출측이 heading_path/page로 출처를 표시·재구성 가능.
- 계층형 검색: 소청크로 검색 → `metadata.parent_chunk_id`로 부모청크를
  추가 조회해 컨텍스트 확장 (Part 2.6).

> **인덱스 정책(v7.1 Part 1.3 준수)**: 신규 source `docingest`가 tb_knowledge에
> 합산되어 행수가 변하지만, tb_knowledge는 이미 ivfflat(lists=100, 105만행)
> 이다. 1M행 초과·검색 품질 이슈 시 v7.1 정책상 hnsw 전환 검토 대상(미확정).

### 2.6 계층형 청킹 — 검색/컨텍스트 분리

2025~26 RAG 표준(parent-child / hierarchical)을 따른다:

| 청크 종류 | 크기(권장) | 용도 | 저장 |
|---|---|---|---|
| 소청크(child) | 128~256 토큰 | **검색**(임베딩 대상, 정밀 매칭) | tb_knowledge 행, `is_parent=false` |
| 부모청크(parent) | 512~1536 토큰 | **컨텍스트 주입**(검색 후 확장) | tb_knowledge 행, `is_parent=true` |

규칙:
1. 소제목(SUBHEADING) 경계를 넘어 소청크를 합치지 않는다 → **충돌 방지의 핵심**
2. 표(TABLE)는 한 청크로 유지(분할 금지), 행이 너무 많으면 헤더를 각 분할에 반복
3. 그림 캡션(FIGURE_CAPTION)은 인접 본문과 분리해 독립 노드 유지
4. 부모청크는 같은 heading_path를 공유하는 소청크들의 상위 묶음
5. overlap은 기존 `split_into_chunks(overlap=100)`(실측)과 달리 **논리 경계
   기반**이므로 문자 단위 overlap 대신 heading_path 공유로 문맥 연결

---

## Part 3: 포맷별 파서 전략

### 3.1 포맷별 파서 매트릭스 (research-assistant 보고 기반)

| 포맷 | 청정 라이브러리 (라이선스) | 레이아웃 인식 수준 | CPU/GPU | 권장도 |
|---|---|---|---|---|
| **PPTX** | python-pptx (MIT) | 슬라이드/도형/텍스트박스/표/그림 **네이티브 구조**. 읽기순서=(top,left) 좌표 정렬로 복원 | CPU | **상** (가장 쉬움) |
| **PDF (텍스트)** | pdfplumber (MIT) | 정밀 표·좌표·문자 위치. 느림 | CPU | 중상 (경량 스택) |
| **PDF (레이아웃)** | Docling (MIT) — DocLayNet RT-DETR 레이아웃 + TableFormer 표. 에어갭 명시 지원 | 레이아웃 블록·표 구조 분리 | GPU 권장(~8GB) | **상** (고품질 스택) |
| **HWPX (신포맷)** | zipfile+xml.etree (표준 라이브러리) 직접 파싱 / python-hwpx (라이선스 검증 필요) | ZIP+OWPML XML 구조 직접 파싱 | CPU | 중상 (검증 필요) |
| **HWP (구포맷)** | LibreOffice subprocess 변환(.hwp→.docx) 경유 → python-docx | 변환 품질에 의존 | CPU | 중 (바이너리 사전 배포 필요) |
| **스캔(이미지 PDF)** — 경량 | Tesseract5 (Apache, kor) | OCR 텍스트만 (레이아웃 약함) | CPU | 중 (저VRAM 폴백) |
| **스캔(이미지 PDF)** — 고품질 | PaddleOCR PP-OCRv4 + PP-Structure (Apache) | OCR + **표 + 레이아웃 통합** | GPU | **상** (고품질 스택) |

### 3.2 배제 라이브러리 (라이선스 리스크 — 기본 스택 금지)

| 라이브러리 | 라이선스 | 조치 |
|---|---|---|
| PyMuPDF (fitz) | **AGPL** | 기본 배제. 상용 라이선스 구매 시 어댑터 슬롯으로만 |
| pyhwp | **AGPL** | 기본 배제 → .hwp는 LibreOffice 변환 경유 |
| marker-pdf | **GPL + OpenRAIL-M** | **사용 금지** (제품 배포 불가) |
| surya 모델 | **OpenRAIL-M (매출 제한)** | 제품 매출 제한 조항 — 배제 |

> **PP-OCRv5 주의**: research 보고상 PP-OCRv5는 **한국어 누락 이슈**가 있어
> **PP-OCRv4 권장**. 고품질 스택의 OCR은 PP-OCRv4 + PP-Structure로 고정.

### 3.3 PPTX 파싱 상세 (대표 사례 — 가장 명확)

python-pptx(MIT)는 슬라이드/도형/텍스트박스/표/그림을 네이티브 객체로 제공한다.
"충돌 없는 임베딩"의 핵심인 **읽기 순서 복원**은 각 도형의 `(top, left)` 좌표를
정렬하여 수행한다.

```python
# core/ingest/parsers/pptx.py (제안 — 읽기순서 복원 발췌)
def _ordered_shapes(slide) -> list:
    """
    슬라이드의 도형을 (top, left) 좌표순으로 정렬해 읽기 순서를 복원한다.
    좌표가 없는 도형(레이아웃 placeholder 등)은 뒤로 보낸다.
    이렇게 하면 좌측 텍스트박스와 우측 텍스트박스가 뒤섞이지 않는다(충돌 방지).
    """
    def key(shape):
        top = getattr(shape, "top", None)
        left = getattr(shape, "left", None)
        # None 좌표는 매우 큰 값으로 처리 → 뒤로 정렬
        return (top if top is not None else 10**9,
                left if left is not None else 10**9)
    return sorted(slide.shapes, key=key)
```

각 슬라이드 → SLIDE 노드, 텍스트박스 → TEXTBOX/PARAGRAPH 노드, 표 → TABLE
노드(행/열 보존)로 매핑. heading_path는 슬라이드 제목 placeholder에서 추출.

---

## Part 4: 티어별 스택 (런타임 GPU 감지 분기)

### 4.1 두 스택 정의

| 스택 | 대상 환경 | 파서 조합 | VRAM | 품질 |
|---|---|---|---|---|
| **경량(Lightweight)** | RTX 5090 부하 시 / CPU 전용 | pdfplumber + Tesseract5 + 자체 트리 파서 + python-pptx + HWPX 직접파싱 | ~0(CPU) | 중 (레이아웃 휴리스틱) |
| **고품질(High-Fidelity)** | A100/H200 (GPU 여유) | Docling(레이아웃) + PaddleOCR PP-Structure(스캔/표) + python-pptx + HWPX | ~8GB+ | 상 (모델 기반 레이아웃) |

PPTX/HWPX는 두 스택 공통(구조가 네이티브라 GPU 불필요). 분기는 **PDF
레이아웃**과 **스캔 OCR**에서만 발생한다.

### 4.2 런타임 분기 로직 (제안)

```python
# core/ingest/pipeline.py (제안 — 티어 분기 발췌)
def _select_pdf_parser(self, tier_has_gpu: bool, vram_free_gb: float) -> DocumentParser:
    """
    GPU 가용 여부·여유 VRAM에 따라 PDF 파서를 선택한다.
    - GPU 여유 충분 → Docling(레이아웃 모델, 고품질)
    - 그 외 → pdfplumber(CPU, 경량)
    fail-closed: 모델 로드 실패 시 경량 스택으로 자동 강등(품질↓, 가용성 유지).
    """
    if tier_has_gpu and vram_free_gb >= self.cfg.docling_min_vram_gb:  # 제안 기본 8.0
        try:
            return self._registry.get("docling")
        except Exception as e:        # 모델 로드 실패 등은 격리 → 경량 강등
            logger.warning("Docling 로드 실패, pdfplumber로 강등: %s", e)
    return self._registry.get("pdfplumber")
```

GPU 감지는 기존 `core/model/gpu_detector.py`의 `GPUTier`를 재사용한다(실측:
RTX_5090/H100/H200/MULTI_GPU). 단, **인제스트는 주로 MCP 서버/배치 측에서
실행**되므로 해당 호스트의 GPU를 감지한다 (Machine A 본류가 아님).

### 4.3 제품 검증 기준 = A100, 권장 조합

| 환경 | 권장 PDF | 권장 스캔 | 비고 |
|---|---|---|---|
| RTX 5090 (32GB, 추론 공유) | pdfplumber(경량) | Tesseract5 | 추론과 VRAM 경합 → 경량 우선 |
| **A100 (80GB, 고객 타겟)** | **Docling** | **PaddleOCR PP-Structure** | **제품 검증 기준선** |
| H200 (141GB, 자체) | Docling | PaddleOCR PP-Structure | 배치 대량 처리 여유 |

### 4.4 A100 티어 추가 (구현 완료)

`GPUTier.A100 = "a100"`이 추가되었다(`core/model/gpu_detector.py:32`).

- **판정**: A100 80GB 와 H100 80GB 는 VRAM 이 동일해 용량만으로 구분 불가하므로,
  감지 로직(`:161~169`)이 `vram_gb > 60` 구간에서 **GPU 이름에 "a100"이 포함되면
  A100**, 아니면 H100 으로 판정한다(이름 기반 분기).
- **프로파일**(`get_tier_config`, `:225~262`): primary `qwen3.5-27b` /
  auxiliary `exaone-32b` 모두 **BF16 풀 프리시전(QuantizationMethod.NONE)**,
  `max_model_len=8192`, `max_num_seqs=4`. training 은 `lora`(rank 64).
- **H100 과의 차이**: A100 은 Hopper 가 아니라 **Ampere 라 FP8/Transformer
  Engine 미지원** — 양자화 없이 BF16 으로만 서빙. HBM2e 대역폭이 낮아
  `max_num_seqs` 를 보수적으로 둔다. `notes="A100 80GB(Ampere). 58+32>80GB이므로
  hot-swap 필요. FP8 미지원."`
- **VRAM 80GB → TIER_M 매핑**: 본 인제스트의 PDF 고품질 분기(Part 4.2)는
  `vram_free_gb >= docling_min_vram_gb(8.0)` 기준으로 Docling 을 선택하므로,
  A100(80GB)은 고품질 스택을 쓸 여유가 충분하다(제품 검증 기준선 — Part 4.3).

---

## Part 5: MCP 서버로의 분리 (v7.2 정합)

### 5.1 두 실행 경로 — 실시간 도구 vs 대량 배치

문서 인제스트는 **성격이 다른 두 경로**로 동작한다.

| 경로 | 용도 | 트리거 | 구현 |
|---|---|---|---|
| **실시간 도구 (MCP)** | Worker가 "이 문서를 색인해줘" 류 요청 시 1건 처리 | query_loop 도구 호출 | `mcp__docingest__*` (v7.2 McpToolAdapter) |
| **대량 배치** | 사내 문서 수천 건 일괄 적재 | 운영자 CLI 실행 | `scripts/prepare_documents.py` (prepare_kowiki 패턴) |

### 5.2 문서 인제스트 MCP 서버 (제안)

v7.2 Part 6의 PoC 4종(db/diag/docutil/kowiki)에 **5번째 대상**으로 추가
제안한다. 무거운 파싱·OCR·레이아웃 모델을 LAN MCP 서버 호스트(GPU 보유)에
격리하여, Machine A 본류의 VRAM·기동 시간 부담을 없앤다.

```yaml
# config/nexus_config.yaml (제안 — v7.2 mcp.servers에 추가)
mcp:
  servers:
    - name: "docingest"
      transport: "http_sse"
      base_url: "http://192.168.22.28:8814"   # GPU 호스트 (LAN) — placeholder 포트
      enabled: false                            # fail-closed
      trust: { read_only: false }               # 적재(쓰기) 가능 → 보수적
```

> **포트 8814는 제안 placeholder** (v7.2가 8810~8813을 제안했으므로 그 다음).
> 운영 확정 전 미정.

노출 도구(제안):

| 도구 이름 | read-only | 동작 |
|---|---|---|
| `mcp__docingest__parse` | ✓ | 문서 1건 파싱 → 구조 트리(JSON) 반환(적재 없음, 미리보기) |
| `mcp__docingest__ingest` | ✗ | 파싱→청킹→임베딩→tb_knowledge **적재** (쓰기) |
| `mcp__docingest__search` | ✓ | 적재된 문서 청크 벡터 검색 (kowiki search와 유사) |

권한 정합(v7.2 Part 4 그대로): 이름이 `mcp__`로 시작 → `ToolCategory.MCP`
자동 분류. `mcp__docingest__ingest`는 **쓰기**이므로 `is_read_only=False`
유지(fail-closed) → DEFAULT 모드 ASK, PLAN 모드 DENY(Layer 5 write 보정).

### 5.3 배치 경로 (prepare_documents.py — 제안)

`scripts/prepare_kowiki.py`(실측)와 동일 패턴:
- 외부 네트워크 없음 — 로컬 파일 디렉토리 입력
- 임베딩은 LAN `/v1/embed`(:8002) 배치 호출
- `KnowledgeStore.add` UPSERT로 멱등 적재
- `--dry-run`(파싱/청킹까지만), `--limit N`(스모크), `--build-index`(ivfflat) 지원

```bash
# 제안 사용 예
python scripts/prepare_documents.py \
  --input /opt/nexus-gpu/corpora/docs/ \
  --formats "pptx,pdf,hwpx" \
  --stack high          # high=Docling+Paddle, light=pdfplumber+Tesseract
  --embed-url http://192.168.22.28:8002 \
  --pg "postgresql://nexus:idino%4012@192.168.10.39:5440/nexus"
```

### 5.4 역할 구분 요약

- **MCP `mcp__docingest__ingest`**: Worker가 자율적으로 소량(1~수건) 색인.
  실시간, 권한 파이프라인 통과, 쓰기라 ASK.
- **배치 `prepare_documents.py`**: 운영자가 대량(수천건) 일괄 색인. 비실시간,
  권한 무관(운영자 직접 실행), GPU 호스트에서 야간 실행 적합.

---

## Part 6: 라이선스 매트릭스 (청정 기본 vs 상용 슬롯)

### 6.1 기본(청정) 스택 — 제품 배포 안전

| 컴포넌트 | 라이브러리 | 라이선스 | 제품 배포 |
|---|---|---|---|
| PPTX 파서 | python-pptx | MIT | ✓ 안전 |
| PDF 텍스트/표 | pdfplumber | MIT | ✓ 안전 |
| PDF 레이아웃 | Docling | MIT | ✓ 안전 (모델 라이선스 별도 검증 필요) |
| HWPX | zipfile/xml.etree (표준) | PSF | ✓ 안전 |
| OCR 경량 | Tesseract5 | Apache-2.0 | ✓ 안전 |
| OCR 고품질 | PaddleOCR / PP-Structure | Apache-2.0 | ✓ 안전 (PP-OCRv4) |
| DOCX (.hwp 변환 후) | python-docx | MIT | ✓ 안전 |
| XLSX | openpyxl | MIT | ✓ 안전 |

### 6.2 배제 (AGPL/GPL/제한 라이선스)

| 라이브러리 | 라이선스 | 사유 |
|---|---|---|
| PyMuPDF | AGPL-3.0 | 네트워크 배포 시 소스 공개 의무 → 상용 슬롯으로만 |
| pyhwp | AGPL-3.0 | 동일 → .hwp는 LibreOffice 변환 경유 |
| marker-pdf | GPL + OpenRAIL-M | **사용 금지** |
| surya 모델 | OpenRAIL-M | 매출 제한 조항 |

### 6.3 어댑터 슬롯 (상용 라이선스 구매 시 교체)

`DocumentParser` 인터페이스 덕분에, 상용 라이선스를 구매하면 **인터페이스만
구현하여 교체**할 수 있다. 예: 고객이 PyMuPDF 상용 라이선스를 구매하면
`PyMuPdfParser(DocumentParser)`를 추가하고 `ParserRegistry`에서 우선순위만
올리면 된다 (기본 스택은 그대로 청정 유지).

```python
# ParserRegistry 우선순위 (제안)
# 기본: pdfplumber/docling (청정)
# 상용 슬롯이 등록되면 해당 포맷에서 우선 사용 (config로 on/off)
```

### 6.4 미확정 (검증 필요)

- **python-hwpx 라이선스**: research 미확인 → 확정 전까지 **zipfile+xml.etree
  직접 파싱**을 1차로 채택(라이선스 무위험).
- **Docling 한국어 정확도**: research 미확인 → A100 PoC에서 한국어 PDF 벤치 필요.
- **Docling 모델(DocLayNet RT-DETR/TableFormer) 라이선스**: 코드(MIT)와 별개로
  모델 가중치 라이선스 별도 검증 필요.

---

## Part 7: 에어갭 번들 (deployment/)

모든 의존성은 오프라인 사전 번들. 외부 네트워크 호출·런타임 설치 금지(P10).

### 7.1 wheel 번들 (개략)

| 분류 | 패키지(개략) | 대상 |
|---|---|---|
| 공통 파서 | python-pptx, pdfplumber, python-docx, openpyxl, lxml | 경량+고품질 |
| 고품질 레이아웃 | docling, docling 의존 (onnxruntime-gpu 등) | A100/H200 |
| OCR 경량 | pytesseract (Tesseract5 바이너리는 별도) | CPU |
| OCR 고품질 | paddleocr, paddlepaddle-gpu | GPU |

### 7.2 모델·바이너리 번들 (개략)

| 항목 | 위치(제안) | 비고 |
|---|---|---|
| Docling 레이아웃/표 모델 | `./models/docling/` | 사전 다운로드, 라이선스 검증 후 |
| PaddleOCR PP-OCRv4 + PP-Structure 모델 | `./models/paddleocr/` | **v4 고정**(v5 한국어 누락) |
| Tesseract5 바이너리 + kor.traineddata | `deployment/bin/tesseract/` | OS별 바이너리 |
| LibreOffice (headless) | `deployment/bin/libreoffice/` | .hwp 변환용, 용량 큼 |

> **구포맷 .hwp 비중이 높음(2026-06-01 사용자 확정)** → LibreOffice headless는
> **필수 번들**이다. 용량·OS 의존성 부담이 있으나 선택 사항이 아니며,
> deployment 무결성·OS별 바이너리 검증 대상에 포함한다.

### 7.3 무결성 검증

기존 `deployment/integrity` 패턴에 따라 wheel·모델 해시 검증. 모델 가중치는
SHA-256 매니페스트로 변조 탐지.

---

## Part 8: (Part 9로 통합)

---

## Part 9: 단계별 로드맵 (PoC 아닌 제품 품질 기준)

development-workflow.md 준수. 하위 단계 완성 전 상위 단계 미착수.

| 단계 | 작업 | 검증 기준 (제품 품질) | 상태 (2026-06-02) |
|---|---|---|---|
| 1 | `core/ingest/types.py` + `parser_base.py` (DocumentTree/DocumentParser ABC, Pydantic v2/frozen) | 단위 테스트: 트리 직렬화·heading_path 전파 | **구현 완료** |
| 2 | **PPTX 파서** (python-pptx, 좌표 읽기순서 복원) | 실제 .pptx 5종에서 슬라이드/표/텍스트박스 충돌 0 | **구현 완료** (`parsers/pptx.py`) |
| 3 | `StructureAwareChunker` (계층형, 소제목 경계 분할) | 인접 소제목 본문이 한 청크에 섞이지 않음 검증 | **구현 완료** |
| 4 | `DocumentIngestPipeline` + tb_knowledge 적재 (metadata jsonb) | embed→add→search 왕복, metadata에 heading_path/page 보존 | **구현 완료** |
| 5 | **PDF 경량** (pdfplumber, 표·좌표) | 텍스트 PDF에서 표 행/열 보존, 본문 분리 | **구현 완료** (`parsers/pdf_plumber.py`) |
| 6 | **PDF 고품질** (Docling, A100) + 티어 분기 | A100에서 Docling, 5090에서 pdfplumber 자동 선택 | **미착수(후속)** — 어댑터 슬롯만 예약 |
| 7 | **HWPX** (python-hwpx, zipfile+xml.etree 폴백) | OWPML XML에서 헤딩/문단/표 구조 추출 | **구현 완료** (`parsers/hwpx.py`) |
| 8 | **스캔 OCR** — Tesseract(경량) → PaddleOCR PP-Structure(고품질) | 스캔 PDF 한국어 OCR + 표 인식, v4 사용 확인 | **미착수(후속)** — 어댑터 슬롯만 예약 |
| 9 | **.hwp 변환** (LibreOffice subprocess → .docx) | .hwp→.docx→트리, 변환 실패 fail-soft | **미착수(후속)** |
| 10 | **MCP 서버화** (`mcp__docingest__*`) + 배치 스크립트 | v7.2 권한(ingest=ASK) 정합, 대량 배치 멱등 | **구현 완료** (MCP 서버) — 배치 스크립트는 후속 |

우선순위 근거: **PPTX(네이티브 구조, 가장 쉬움) → PDF → HWPX → 스캔 → .hwp**.
PoC가 아니라 각 단계가 제품 품질(충돌 0, 멱등, 에어갭, 라이선스 청정)을
만족해야 다음 단계로 이동.

> **우선순위 조정(2026-06-01 사용자 확정)**: 고객 문서의 **구포맷 .hwp 비중이
> 높으므로**, 단계 9(.hwp/LibreOffice 변환)를 HWPX(단계 7) 직후로 **상향**하는
> 것을 권장한다. .hwp 변환은 기술 난이도가 아니라 고객 데이터 분포가 우선순위를
> 결정한다(LibreOffice 필수 번들 — Part 7.2).

### 9.1 구현 현황 (초판 "신규 제안" → 구현 반영, 2026-06-02)

| # | 항목 | 상태 |
|---|---|---|
| 1 | `core/ingest/` 신규 디렉토리 (types/parser_base/pipeline/chunker/parsers) | **구현 완료** |
| 2 | `GPUTier.A100` 추가 + A100 ModelSpec/EmbeddingSpec | **구현 완료** (`gpu_detector.py:32, 225~262`, 이름 기반 판정) |
| 3 | tb_knowledge source="docingest" 운영 정책 | **구현 완료** (스키마 변경 없음, source 필터 검색) |
| 4 | `mcp__docingest__*` MCP 서버 | **구현 완료** (`mcp_servers/docingest_server.py`, parse/ingest/search) |
| 5 | `scripts/prepare_documents.py` 배치 | **미착수(후속)** |
| 6 | tb_knowledge 1M행 초과 시 hnsw 전환 | 미확정 (v7.1 Part 1.3 정책 대상) |
| 7 | PDF 고품질(Docling) / 스캔 OCR(Tesseract·PaddleOCR) 고품질 파서 | **미착수(후속)** — 어댑터 슬롯만 예약(ParserRegistry priority 로 끼움) |

> **HWPX 파서 실제 구현(정정)**: 로드맵 초안은 HWPX 1차를 "zipfile+xml.etree 직접
> 파싱"으로 적었으나, 실제 구현(`parsers/hwpx.py`)은 **python-hwpx(OWPML, 2.9.0
> 기준)를 1차 파서로 사용**하고, 열기 실패 시 **zipfile+xml.etree 로 폴백**한다.
> 표는 `table.iter_grid()`가 일부 표에서 예외를 던질 수 있어
> `row_count/column_count + cell(r,c)` 직접 순회로 견고하게 처리한다.

> **라이브러리 설치 현황(개발 환경)**: pdfplumber / python-hwpx / pytesseract 와
> docling / paddleocr 가 **개발 환경에 설치되어 있다**. 단, 현재 구현이 실제로
> 사용하는 것은 **pdfplumber(PDF 경량)·python-hwpx(HWPX)·python-pptx(PPTX)** 뿐이며,
> docling/paddleocr/tesseract 고품질 파서는 **아직 구현되지 않은 어댑터 슬롯**이다.
> 에어갭 원칙은 **배포물(wheel 번들)에만** 적용된다 — 개발 중 라이브러리 설치는
> 정상이며, 배포 시 오프라인 wheel 로 번들한다(Part 7).

---

## Part 10: 미확정 / 검증 필요 (research 미확인 인용)

| # | 항목 | 현 입장 / 필요 검증 |
|---|---|---|
| 1 | **python-hwpx 라이선스** | **사용 채택** — HWPX 1차 파서로 python-hwpx(2.9.0, OWPML) 채택, zipfile+xml.etree 폴백 유지(`parsers/hwpx.py`). 배포 wheel 번들 시 라이선스 최종 확인 필요 |
| 2 | **Docling 한국어 정확도** | research 미확인 → A100 PoC에서 한국어 PDF 벤치 필수 |
| 3 | **Docling 모델 가중치 라이선스** | 코드 MIT와 별개 → 별도 검증 |
| 4 | **.hwp / .hwpx 고객 비율** | **구포맷 .hwp 비중 높음(확정)** → LibreOffice **필수** 번들, 로드맵 우선순위 상향(단계 9→HWPX 직후 검토) |
| 5 | **OCR 엔진 한국어 벤치마크** | Tesseract vs PaddleOCR PP-OCRv4 한국어 정확도 비교 미수행 |
| 6 | **PP-OCRv5 한국어 누락** | research 보고 → **v4 고정**으로 회피 |
| 7 | **MCP 포트 8814** | **확정** — `mcp_servers/run.py::_SERVERS` 에서 docingest 기본 포트 8814 로 확정(db 8810/diag 8811/kowiki 8813) |
| 8 | **A100 ModelSpec 값** | **정의 완료** — `gpu_detector.py:225~262` (BF16 NONE, max_model_len 8192, max_num_seqs 4, lora rank 64, FP8 미지원) |
| 9 | **계층형 청크 크기(128~256/512~1536)** | 한국어 e5-large 기준 튜닝 필요 |

---

## Part 11: 검증 — 설계 일관성

문서 인제스트는 도구·배치·(선택)MCP 서버를 추가할 뿐, 다음 무결성을 깨지 않는다.

| 영역 | 영향 | 근거 |
|---|---|---|
| 4-Tier AsyncGenerator 체인 | **영향 없음** | 인제스트 도구도 query_loop→executor 통과 |
| 표준 내부 계약 (OpenAI tool_calls) | **영향 없음** | 파싱/OCR은 도구/MCP 어댑터 아래에만 존재 |
| 5계층 권한 | **영향 없음** | 신규 도구도 BaseTool로 동일 파이프라인. ingest=쓰기→ASK |
| Hook / Thinking / Memory | **영향 없음** | 도구·RAG 계층 |
| tb_knowledge 스키마 | **영향 없음** | metadata jsonb 활용, 컬럼 추가 없음 |
| 임베딩 서버 계약 | **영향 없음** | v7.1 `/v1/embed` 그대로 |
| Air-gap | **유지** | wheel/모델 오프라인 번들, LAN URL만, 외부 호출 없음 |
| 라이선스(제품) | **강화** | AGPL/GPL 배제 + 청정 기본 + 상용 어댑터 슬롯 |
| 기존 DocumentProcessTool | **공존** | 읽기용 도구 유지, 인제스트와 목적 분리 |

---

## 부록 A: 코드 위치 매트릭스 (이미 존재 vs 신규 제안)

| 항목 | 코드 위치 | 상태 |
|---|---|---|
| tb_knowledge + metadata jsonb | `core/rag/knowledge_store.py` | **이미 존재** (실측) |
| KnowledgeEntry / KnowledgeStore.add (UPSERT) | `core/rag/knowledge_store.py:84, 124` | **이미 존재** |
| split_into_chunks (단순 청킹) | `core/rag/knowledge_store.py:348` | **이미 존재** (인제스트는 별도 청커) |
| 임베딩 `/v1/embed` 호출 | `core/model/inference.py::embed` (v7.1 Part 2) | **이미 존재** |
| GPUTier (RTX_5090/A100/H100/H200/MULTI_GPU) | `core/model/gpu_detector.py:28` | **이미 존재** (A100 추가됨) |
| DocumentProcessTool (읽기용, 평면 텍스트) | `core/tools/implementations/document_tool.py` | **이미 존재** (한계 Part 1.2) |
| 배치 인제스트 패턴 (prepare_kowiki) | `scripts/prepare_kowiki.py` | **이미 존재** (참조 패턴) |
| McpToolAdapter / McpConnectionManager | `core/tools/mcp/*` | **구현됨** (v7.2) |
| `core/ingest/` (DocumentParser/트리/청커/파이프라인) | `core/ingest/{types,parser_base,chunker,pipeline}.py` | **구현됨** |
| PPTX/PDF/HWPX 파서 | `core/ingest/parsers/{pptx,pdf_plumber,hwpx}.py` | **구현됨** |
| Docling/OCR(Paddle·Tesseract)/.hwp 파서 | `core/ingest/parsers/` | **미구현(후속·어댑터 슬롯)** |
| `GPUTier.A100` + A100 ModelSpec | `core/model/gpu_detector.py:32, 225~262` | **구현됨** (이름 기반 판정) |
| `mcp__docingest__*` MCP 서버 (parse/ingest/search) | `mcp_servers/docingest_server.py` (포트 8814) | **구현됨** |
| 인제스트 테스트 | `tests/unit/test_{pptx,pdf_plumber,hwpx}_parser.py`, `test_ingest_*`, `tests/integration/test_ingest_pipeline.py` | **구현됨** |
| `scripts/prepare_documents.py` (배치) | `scripts/` | **미착수(후속)** |

---

*작성일: 2026-06-01 (구현 반영 갱신: 2026-06-02)*
*코드 확인 기준(2026-06-02): core/ingest/{types,parser_base,chunker,pipeline}.py,*
*core/ingest/parsers/{pptx,pdf_plumber,hwpx,__init__}.py, mcp_servers/docingest_server.py,*
*core/model/gpu_detector.py(GPUTier.A100), core/rag/knowledge_store.py,*
*tests/unit/test_{pptx,pdf_plumber,hwpx}_parser.py, tests/integration/test_ingest_pipeline.py*
*테스트 실측(2026-06-02): 전체 회귀 1091 passed·1 skipped (1 failed=test_gpu_e2e, 무관)*
*조사 근거: research-assistant 보고 (라이선스·라이브러리 레이아웃 인식 수준)*
