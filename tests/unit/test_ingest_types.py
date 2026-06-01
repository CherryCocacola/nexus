"""
core.ingest.types 단위 테스트 (v7.3 문서 인제스트).

검증 의도:
  1. DocumentNode / DocumentTree 가 frozen(불변)인지 — 파싱 완료 후 누구도 몰래
     트리를 바꾸지 못해야 한다(설계 원칙: 불변 우선).
  2. children 자기참조(재귀) 구조가 올바르게 직렬화/역직렬화되는지.
  3. heading_path 가 tuple 로 보존되는지(리스트로 흘러들어가도 tuple 로 강제).
  4. DocumentChunk 는 가변이어야 한다 — 청커가 parent_chunk_id 를 나중에 채우기
     때문(설계상 의도된 가변).

모두 순수 데이터 모델 테스트라 외부 의존성(네트워크/GPU/DB)이 전혀 없다.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from core.ingest.types import (
    DocumentChunk,
    DocumentNode,
    DocumentTree,
    ElementType,
)


# ─────────────────────────────────────────────
# DocumentNode — 불변성
# ─────────────────────────────────────────────
def test_document_node_is_frozen_immutable() -> None:
    """DocumentNode 의 필드를 생성 후 수정하면 ValidationError 가 나야 한다(frozen)."""
    node = DocumentNode(element_type=ElementType.PARAGRAPH, text="본문")
    # frozen=True 인 Pydantic 모델은 속성 재할당 시 ValidationError 를 던진다.
    with pytest.raises(ValidationError):
        node.text = "변경 시도"  # type: ignore[misc]


def test_document_node_default_fields_are_safe() -> None:
    """기본값 검증 — text="", heading_path=(), page=None, order=0, children=()."""
    node = DocumentNode(element_type=ElementType.SLIDE)
    assert node.text == ""
    assert node.heading_path == ()
    assert node.page is None
    assert node.order == 0
    assert node.children == ()


def test_document_node_heading_path_coerced_to_tuple() -> None:
    """heading_path 에 list 를 넣어도 tuple 로 강제되어 불변성이 유지되는지."""
    node = DocumentNode(
        element_type=ElementType.PARAGRAPH,
        text="x",
        heading_path=["3장 보안", "3.1 권한"],  # type: ignore[arg-type]
    )
    assert node.heading_path == ("3장 보안", "3.1 권한")
    assert isinstance(node.heading_path, tuple)


# ─────────────────────────────────────────────
# DocumentNode — 자기참조(재귀) children
# ─────────────────────────────────────────────
def test_document_node_recursive_children_roundtrip() -> None:
    """SLIDE → [TEXTBOX, PARAGRAPH] 같은 재귀 구조가 직렬화/역직렬화 가능한지."""
    child_a = DocumentNode(element_type=ElementType.TEXTBOX, text="제목", order=0)
    child_b = DocumentNode(element_type=ElementType.PARAGRAPH, text="본문", order=1)
    slide = DocumentNode(
        element_type=ElementType.SLIDE,
        page=1,
        order=1,
        children=(child_a, child_b),
    )

    # 재귀 직렬화 — model_dump 가 children 까지 내려간다.
    dumped = slide.model_dump()
    assert dumped["element_type"] == "slide"
    assert len(dumped["children"]) == 2
    assert dumped["children"][0]["text"] == "제목"

    # 역직렬화 — 동일 구조로 복원되는지(자기참조 rebuild 가 올바른지 확인).
    restored = DocumentNode.model_validate(dumped)
    assert restored == slide
    assert restored.children[1].element_type == ElementType.PARAGRAPH


def test_document_node_children_are_tuple_not_list() -> None:
    """children 은 list 로 줘도 tuple 로 강제되어야 한다(불변 컨테이너)."""
    node = DocumentNode(
        element_type=ElementType.SLIDE,
        children=[DocumentNode(element_type=ElementType.PARAGRAPH, text="a")],  # type: ignore[arg-type]
    )
    assert isinstance(node.children, tuple)


# ─────────────────────────────────────────────
# DocumentTree — 불변성 + 필드
# ─────────────────────────────────────────────
def test_document_tree_is_frozen_immutable() -> None:
    """DocumentTree 도 frozen — title 재할당 시 ValidationError."""
    tree = DocumentTree(title="문서", source_path="/x/y.pptx")
    with pytest.raises(ValidationError):
        tree.title = "다른 제목"  # type: ignore[misc]


def test_document_tree_warnings_and_nodes_are_tuples() -> None:
    """warnings/nodes 는 tuple 로 강제 — fail-soft 경고 통로가 불변이어야 한다."""
    tree = DocumentTree(
        title="문서",
        source_path="/x/y.pptx",
        nodes=[DocumentNode(element_type=ElementType.SLIDE)],  # type: ignore[arg-type]
        doc_format="pptx",
        warnings=["슬라이드 2: 표 파싱 실패"],  # type: ignore[arg-type]
    )
    assert isinstance(tree.nodes, tuple)
    assert isinstance(tree.warnings, tuple)
    assert tree.doc_format == "pptx"


# ─────────────────────────────────────────────
# DocumentChunk — 가변(설계상 의도)
# ─────────────────────────────────────────────
def test_document_chunk_is_mutable_for_parent_linking() -> None:
    """
    DocumentChunk 는 frozen 이 아니어야 한다.

    이유: 청커가 부모청크 id 를 먼저 계산한 뒤 소청크들의 parent_chunk_id 를
    단계적으로 채우기 때문(설계 원칙). 따라서 재할당이 허용돼야 한다.
    """
    chunk = DocumentChunk(content="본문", element_type=ElementType.PARAGRAPH)
    assert chunk.parent_chunk_id is None
    # 나중에 부모 id 를 채우는 동작이 예외 없이 가능해야 한다.
    chunk.parent_chunk_id = "deadbeef"
    assert chunk.parent_chunk_id == "deadbeef"


def test_document_chunk_defaults() -> None:
    """기본값 — heading_path=(), page=None, parent_chunk_id=None, is_parent=False."""
    chunk = DocumentChunk(content="x", element_type=ElementType.TABLE)
    assert chunk.heading_path == ()
    assert chunk.page is None
    assert chunk.parent_chunk_id is None
    assert chunk.is_parent is False


def test_element_type_values_are_serializable_strings() -> None:
    """ElementType 은 str Enum — value 가 그대로 문자열이라 jsonb 직렬화에 안전."""
    assert ElementType.SUBHEADING.value == "subheading"
    assert ElementType.TABLE.value == "table"
    # str 서브클래스라 문자열 비교가 자연스럽다.
    assert ElementType.SLIDE == "slide"
