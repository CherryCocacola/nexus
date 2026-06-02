"""
문서 인제스트 데이터 모델 — 구조 트리(DocumentTree)와 청크(DocumentChunk).

역할:
  파서(parser)가 문서를 읽어 만들어내는 "구조 트리"와, 청커(chunker)가
  그 트리를 잘라 만들어내는 "임베딩 단위 청크"를 정의한다.

왜 이렇게 나누는가 (v7.3 Part 1.1 — 충돌 없는 임베딩):
  문서는 (섹션 → 소제목 → [문단|표|그림캡션]) 처럼 논리적 경계가 있다.
  이 경계를 "구조 트리"로 먼저 보존해 두면, 청킹 단계에서 서로 다른 논리
  영역의 텍스트가 한 청크에 뒤섞이는 것("충돌")을 막을 수 있다.

설계 원칙:
  - 모든 모델은 Pydantic v2 BaseModel. 트리/노드는 frozen(불변) — 생성 후
    수정 불가. 파싱이 끝난 트리를 누구도 몰래 바꾸지 못하게 하기 위함이다.
  - DocumentChunk는 청킹 과정에서 parent_chunk_id 등을 채워야 하므로
    frozen으로 두지 않는다(가변). 단 외부에서 함부로 바꾸지 않는 약속.
  - 외부 네트워크/무거운 의존성 없음 — 순수 데이터 모델만 둔다(에어갭).
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


# ─────────────────────────────────────────────
# ElementType — 구조 트리 노드의 종류
# ─────────────────────────────────────────────
class ElementType(str, Enum):
    """
    구조 트리 노드의 종류. 적재 시 metadata.element_type 로 저장된다.

    왜 str Enum 인가: 값(value)이 그대로 문자열이라 JSON(jsonb) 직렬화가
    자연스럽고, 검색 결과에서 사람이 읽기도 쉽다.

    SUBHEADING 은 "충돌 방지"의 핵심 경계다 — 청커는 이 경계를 넘어
    텍스트를 합치지 않는다(types 정의만으로는 강제되지 않으나, chunker가
    이 규칙을 구현한다).
    """

    SECTION = "section"  # 큰 제목 (= h1)
    SUBHEADING = "subheading"  # 소제목 (= h2/h3) — 청킹 경계의 핵심
    PARAGRAPH = "paragraph"  # 본문 단락
    TABLE = "table"  # 표 (행/열 구조 보존, 한 청크로 유지)
    FIGURE_CAPTION = "figure_caption"  # 그림/도표 캡션 (독립 노드 유지)
    LIST_ITEM = "list_item"  # 목록 항목
    SLIDE = "slide"  # PPTX 슬라이드 단위 (컨테이너)
    TEXTBOX = "textbox"  # PPTX/HWP 텍스트박스


# ─────────────────────────────────────────────
# DocumentNode — 구조 트리의 한 노드 (불변)
# ─────────────────────────────────────────────
class DocumentNode(BaseModel):
    """
    구조 트리의 한 노드. 자기 자신을 children 으로 가질 수 있는 재귀 구조다.

    예시 (PPTX):
      SLIDE 노드(슬라이드 1)
        └ children: [TEXTBOX(제목), PARAGRAPH(본문), TABLE(표) ...]

    필드 설명:
      - element_type: 노드 종류 (위 ElementType)
      - text: 이 노드가 담는 텍스트. 컨테이너성 노드(SLIDE 등)는 빈 문자열일
        수 있고, 실제 내용은 children 에 들어간다.
      - heading_path: 루트→현재까지의 소제목 경로. 예) ("3장 보안", "3.1 권한").
        청커가 이 경로를 청크 metadata 로 전파해 "어느 맥락의 텍스트인지"를
        검색 시 알 수 있게 한다(문맥 연결).
      - page: PDF/스캔의 페이지 번호. PPTX 는 슬라이드 번호(1부터).
      - order: 같은 부모 안에서의 읽기 순서. PPTX 는 (top,left) 좌표 정렬 결과.
      - children: 자식 노드들 (tuple — 불변).
    """

    # frozen=True: 파싱 완료 후 노드를 수정 불가능하게 고정한다(불변 우선).
    model_config = ConfigDict(frozen=True)

    element_type: ElementType
    text: str = ""
    heading_path: tuple[str, ...] = ()
    page: int | None = None
    order: int = 0
    children: tuple[DocumentNode, ...] = ()


# 자기참조(children: tuple["DocumentNode", ...]) 타입을 Pydantic 이 완전히
# 해석하도록 모델을 재빌드한다. from __future__ import annotations 와 함께
# 쓰면 문자열 전방참조가 남으므로 명시적으로 rebuild 한다.
DocumentNode.model_rebuild()


# ─────────────────────────────────────────────
# DocumentTree — 문서 1개의 루트 컨테이너 (불변)
# ─────────────────────────────────────────────
class DocumentTree(BaseModel):
    """
    문서 1개 전체를 담는 루트 컨테이너. 파서 parse() 의 반환 타입이다.

    필드 설명:
      - title: 문서 제목 (파일명 또는 첫 슬라이드 제목 등에서 추출).
      - source_path: 원본 파일 경로 (추적/디버깅용 문자열).
      - nodes: 최상위 노드들 (PPTX 의 경우 SLIDE 노드 목록).
      - doc_format: 출처 포맷 식별자 ("pptx" 등). 적재 metadata.doc_format 으로
        그대로 저장된다.
      - warnings: fail-soft 경고 수집 통로. 파싱 중 일부 도형/표를 읽지 못해도
        예외로 중단하지 않고, 이 목록에 사람이 읽을 수 있는 경고를 쌓는다
        (anti-pattern #8 — bare except 금지, 부분 트리+경고로 표현).

    왜 frozen 인가: 파싱이 끝난 트리는 이후 단계(청킹/적재)에서 읽기만 하므로,
    실수로 변형되는 것을 막기 위해 불변으로 고정한다.
    """

    model_config = ConfigDict(frozen=True)

    title: str
    source_path: str
    nodes: tuple[DocumentNode, ...] = ()
    doc_format: str = ""
    warnings: tuple[str, ...] = ()


# ─────────────────────────────────────────────
# DocumentChunk — 임베딩 단위 (가변)
# ─────────────────────────────────────────────
class DocumentChunk(BaseModel):
    """
    청킹 후의 임베딩 단위. KnowledgeEntry 로 변환되어 tb_knowledge 에 적재된다.

    계층형(parent-child) 청킹 (v7.3 Part 2.6):
      - 소청크(child, is_parent=False): 작게 잘라 정밀 "검색"에 쓴다.
      - 부모청크(parent, is_parent=True): 같은 heading_path 의 소청크들을 묶은
        큰 청크. 검색 후 "컨텍스트 주입"에 쓴다.
      - 소청크의 parent_chunk_id 가 부모청크를 가리켜, 검색→컨텍스트 확장이
        가능하다.

    필드 설명:
      - content: 임베딩 대상 텍스트.
      - heading_path: 이 청크가 속한 소제목 경로 (문맥 연결).
      - element_type: 청크의 대표 종류 (소청크는 원본 노드 종류, 부모청크는
        보통 SUBHEADING/SECTION).
      - page: 페이지(또는 슬라이드) 번호.
      - parent_chunk_id: 소청크가 가리키는 부모청크의 안정적 식별자.
        부모청크 자신은 None.
      - is_parent: True=컨텍스트용 부모청크, False=검색용 소청크.

    왜 frozen 이 아닌가: 청커가 부모청크를 먼저 만들고 id 를 계산한 뒤
    소청크들의 parent_chunk_id 를 채우는 식으로 단계적으로 구성하기 때문에
    가변이 편하다. 외부에서는 읽기 전용처럼 다룬다.
    """

    content: str
    heading_path: tuple[str, ...] = ()
    element_type: ElementType
    page: int | None = None
    parent_chunk_id: str | None = None
    is_parent: bool = False

    # Pydantic 기본 동작 유지(가변). Field 는 향후 검증 확장 여지를 위해 import 유지.


# Field 는 향후 확장(예: min_length 검증)을 위해 import 했으나 현재 미사용.
# ruff 의 F401(미사용 import) 회피용 명시적 참조.
_ = Field
