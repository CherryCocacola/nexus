"""
core.ingest.chunker — StructureAwareChunker 단위 테스트 (v7.3 문서 인제스트).

핵심 검증 의도 — "충돌 0"이 실제로 보장되는가:
  1. 서로 다른 소제목(SUBHEADING/슬라이드)의 본문이 같은 청크에 절대 섞이지
     않아야 한다(예: "보안" 텍스트와 "네트워크" 텍스트가 한 청크에 공존 금지).
  2. TABLE 은 분할되지 않고 한 청크로 유지(행/열 보존).
  3. 부모청크(is_parent=True) + 소청크(False) 동시 생성, 소청크의
     parent_chunk_id 가 같은 그룹 부모를 가리켜야 한다.
  4. 같은 heading_path 를 부모/소청크가 공유해야 한다(문맥 연결).

청커는 DocumentTree(구조 트리)를 입력으로 받으므로, 여기서는 PPTX 없이
트리를 직접 조립해 경계 분리 로직만 정밀하게 검증한다.
"""

from __future__ import annotations

from core.ingest.chunker import StructureAwareChunker, _estimate_tokens
from core.ingest.types import DocumentNode, DocumentTree, ElementType


# ─────────────────────────────────────────────
# 헬퍼 — 슬라이드(컨테이너) 노드를 만든다.
# ─────────────────────────────────────────────
def _slide(heading: str, *leaves: DocumentNode, page: int = 1) -> DocumentNode:
    """heading 을 자식들의 heading_path 로 채운 SLIDE 컨테이너 노드를 만든다."""
    children = tuple(
        leaf.model_copy(update={"heading_path": (heading,), "page": page}) for leaf in leaves
    )
    return DocumentNode(
        element_type=ElementType.SLIDE,
        page=page,
        order=page,
        children=children,
    )


def _para(text: str) -> DocumentNode:
    return DocumentNode(element_type=ElementType.PARAGRAPH, text=text)


def _table(text: str) -> DocumentNode:
    return DocumentNode(element_type=ElementType.TABLE, text=text)


# ─────────────────────────────────────────────
# 1) 충돌 0 — 서로 다른 소제목 본문이 섞이지 않는다
# ─────────────────────────────────────────────
def test_chunk_different_subheadings_never_mixed() -> None:
    """
    '보안' 슬라이드 본문과 '네트워크' 슬라이드 본문이 한 청크에 공존하면 안 된다.

    이것이 v7.3 "충돌 없는 임베딩"의 핵심 보장이다.
    """
    tree = DocumentTree(
        title="정책",
        source_path="/x.pptx",
        nodes=(
            _slide("보안", _para("접근 통제는 최소 권한을 따른다."), page=1),
            _slide("네트워크", _para("방화벽은 기본 차단 정책이다."), page=2),
        ),
    )
    chunks = StructureAwareChunker().chunk(tree)

    # 모든 청크(부모+소청크)에서 두 슬라이드 텍스트가 같이 들어간 청크가 없어야 한다.
    for c in chunks:
        has_security = "최소 권한" in c.content
        has_network = "방화벽" in c.content
        assert not (has_security and has_network), (
            f"충돌 발생 — 한 청크에 보안+네트워크가 섞임: {c.content!r}"
        )


def test_chunk_groups_partition_by_heading_path() -> None:
    """heading_path 가 다르면 부모청크가 분리되어 각 그룹마다 1개씩 생긴다."""
    tree = DocumentTree(
        title="정책",
        source_path="/x.pptx",
        nodes=(
            _slide("보안", _para("보안 본문"), page=1),
            _slide("네트워크", _para("네트워크 본문"), page=2),
        ),
    )
    chunks = StructureAwareChunker().chunk(tree)

    parents = [c for c in chunks if c.is_parent]
    parent_paths = {c.heading_path for c in parents}
    # 두 개의 서로 다른 heading_path 그룹 → 부모청크도 2개.
    assert parent_paths == {("보안",), ("네트워크",)}
    assert len(parents) == 2


# ─────────────────────────────────────────────
# 2) TABLE 은 분할되지 않고 한 청크 유지
# ─────────────────────────────────────────────
def test_chunk_table_kept_as_single_child_chunk() -> None:
    """표 노드는 분할/병합 없이 독립 소청크 1개로 유지되어야 한다."""
    table_text = "이름 | 부서\n홍길동 | 보안팀\n이순신 | 해군"
    tree = DocumentTree(
        title="명부",
        source_path="/x.pptx",
        nodes=(_slide("명단", _table(table_text), page=1),),
    )
    chunks = StructureAwareChunker().chunk(tree)

    table_children = [
        c for c in chunks if (not c.is_parent) and c.element_type == ElementType.TABLE
    ]
    assert len(table_children) == 1
    # 표 내용이 통째로(분할 없이) 보존됐는지.
    assert table_children[0].content == table_text


def test_chunk_table_not_merged_with_adjacent_paragraph() -> None:
    """표 소청크는 인접 본문 소청크와 한 청크로 합쳐지지 않아야 한다."""
    tree = DocumentTree(
        title="문서",
        source_path="/x.pptx",
        nodes=(
            _slide(
                "현황",
                _para("아래는 인원 현황이다."),
                _table("이름 | 부서\n홍길동 | 보안팀"),
                _para("이상으로 현황을 마친다."),
                page=1,
            ),
        ),
    )
    chunks = StructureAwareChunker().chunk(tree)
    children = [c for c in chunks if not c.is_parent]

    # 표 소청크는 본문 텍스트를 포함하지 않아야 한다(독립 유지).
    table_child = next(c for c in children if c.element_type == ElementType.TABLE)
    assert "현황" not in table_child.content
    assert "홍길동" in table_child.content


# ─────────────────────────────────────────────
# 3) 부모/소청크 동시 생성 + parent_chunk_id 연결
# ─────────────────────────────────────────────
def test_chunk_creates_parent_and_children_linked() -> None:
    """한 그룹에서 부모청크 1개 + 소청크 N개가 생성되고 id 로 연결되는지."""
    tree = DocumentTree(
        title="문서",
        source_path="/x.pptx",
        nodes=(_slide("보안", _para("문장 하나"), _para("문장 둘"), page=1),),
    )
    chunks = StructureAwareChunker().chunk(tree)

    parents = [c for c in chunks if c.is_parent]
    children = [c for c in chunks if not c.is_parent]
    assert len(parents) == 1
    assert len(children) >= 1

    parent = parents[0]
    # 부모는 parent_chunk_id 가 None(자기 자신이 루트).
    assert parent.parent_chunk_id is None
    assert parent.is_parent is True

    # 모든 소청크는 같은 부모를 가리켜야 한다.
    parent_id = StructureAwareChunker._chunk_id(parent)
    for c in children:
        assert c.parent_chunk_id == parent_id


def test_chunk_child_parent_id_is_not_none() -> None:
    """소청크의 parent_chunk_id 는 채워져 있어야 한다(연결 누락 방지)."""
    tree = DocumentTree(
        title="문서",
        source_path="/x.pptx",
        nodes=(_slide("개요", _para("개요 본문입니다."), page=1),),
    )
    chunks = StructureAwareChunker().chunk(tree)
    children = [c for c in chunks if not c.is_parent]
    assert children, "소청크가 최소 1개는 있어야 한다"
    assert all(c.parent_chunk_id is not None for c in children)


# ─────────────────────────────────────────────
# 4) 같은 heading_path 공유
# ─────────────────────────────────────────────
def test_chunk_parent_and_children_share_heading_path() -> None:
    """같은 그룹의 부모/소청크는 동일한 heading_path 를 공유해야 한다."""
    tree = DocumentTree(
        title="문서",
        source_path="/x.pptx",
        nodes=(_slide("3.1 권한", _para("권한은 역할 기반이다."), page=2),),
    )
    chunks = StructureAwareChunker().chunk(tree)
    assert chunks, "청크가 생성돼야 한다"
    for c in chunks:
        assert c.heading_path == ("3.1 권한",)


def test_chunk_empty_tree_returns_empty_list() -> None:
    """노드가 없는 트리는 빈 청크 목록을 반환(경계 케이스)."""
    tree = DocumentTree(title="빈문서", source_path="/x.pptx", nodes=())
    assert StructureAwareChunker().chunk(tree) == []


def test_chunk_whitespace_only_nodes_are_skipped() -> None:
    """공백/빈 텍스트 노드는 청크 후보에서 제외된다."""
    tree = DocumentTree(
        title="문서",
        source_path="/x.pptx",
        nodes=(_slide("섹션", _para("   "), _para(""), page=1),),
    )
    assert StructureAwareChunker().chunk(tree) == []


# ─────────────────────────────────────────────
# 부가 — 토큰 추정 휴리스틱 / 큰 그룹 분할
# ─────────────────────────────────────────────
def test_estimate_tokens_minimum_one() -> None:
    """빈/짧은 문자열도 최소 1 토큰으로 추정(0 division/0 토큰 방지)."""
    assert _estimate_tokens("") == 1
    assert _estimate_tokens("ab") == 1


def test_chunk_large_group_splits_children_by_target() -> None:
    """
    한 소제목 안에 큰 본문이 많으면 소청크가 여러 개로 분할되지만,

    그래도 경계(heading_path)는 유지되어 충돌이 없어야 한다.
    """
    # child_target_tokens 를 작게(=10) 두어 강제로 다수 소청크가 나오게 한다.
    chunker = StructureAwareChunker(child_target_tokens=10, parent_max_tokens=1536)
    long_paras = [_para("가" * 40) for _ in range(5)]  # 각 ~20토큰 추정
    tree = DocumentTree(
        title="문서",
        source_path="/x.pptx",
        nodes=(_slide("긴섹션", *long_paras, page=1),),
    )
    chunks = chunker.chunk(tree)
    children = [c for c in chunks if not c.is_parent]
    # 여러 소청크로 분할됐어야 한다.
    assert len(children) >= 2
    # 분할됐어도 모두 같은 heading_path(경계 유지).
    assert all(c.heading_path == ("긴섹션",) for c in children)
