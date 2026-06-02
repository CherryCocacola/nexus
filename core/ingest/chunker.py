"""
구조 인식 청킹 — StructureAwareChunker.

역할:
  구조 트리(DocumentTree)를 "임베딩 단위 청크(DocumentChunk)" 목록으로
  변환한다. 핵심은 "충돌 없는 청킹"(v7.3 Part 1.1, 2.6): 서로 다른 논리
  영역(소제목/표/그림캡션)의 텍스트가 한 청크에 뒤섞이지 않도록 한다.

계층형(parent-child) 청킹 (v7.3 Part 2.6):
  - 소청크(child): 작게 잘라 정밀 "검색"용 (128~256 토큰 상당).
  - 부모청크(parent): 같은 heading_path 의 소청크들을 묶은 큰 청크
    (512~1536 토큰 상당). 검색 후 "컨텍스트 주입"용.
  - 소청크의 parent_chunk_id 가 부모청크를 가리켜 검색→컨텍스트 확장이 가능.

청킹 규칙 (v7.3 Part 2.6):
  ① SUBHEADING/SLIDE 등 경계를 넘어 소청크를 합치지 않는다 (충돌 방지 핵심).
  ② TABLE 은 한 청크로 유지(분할 금지).
  ③ FIGURE_CAPTION 은 인접 본문과 분리해 독립 청크.
  ④ 부모청크(is_parent=True) + 소청크(is_parent=False) 동시 생성, parent_chunk_id 연결.
  ⑤ 문자 단위 overlap 대신 heading_path 공유로 문맥을 연결한다.

토큰 추정 (v7.3 Part 10 #9 — 한국어 e5-large 기준 튜닝 필요):
  정확한 토크나이저 없이 글자수 기반 휴리스틱을 쓴다. 한국어는 글자당 토큰
  비율이 높으므로 보수적으로 추정한다(아래 _estimate_tokens 참조).

의존성 방향 (P2): core.ingest.types 만 의존(역방향/순환 없음).
"""

from __future__ import annotations

import hashlib

from core.ingest.types import DocumentChunk, DocumentNode, DocumentTree, ElementType

# ─────────────────────────────────────────────
# 크기 상수 (v7.3 Part 2.6 권장값, 토큰 상당)
# ─────────────────────────────────────────────
# 소청크 목표/최대 토큰. 검색 정밀도를 위해 작게 유지한다.
CHILD_TARGET_TOKENS = 256
# 부모청크 최대 토큰. 컨텍스트 주입용으로 크게.
PARENT_MAX_TOKENS = 1536

# 글자수→토큰 추정 계수. 한국어/영어 혼합 문서를 보수적으로 추정하기 위한
# 휴리스틱(한 토큰 ≈ 2 글자로 가정). 정확한 토크나이저는 후속 튜닝 대상.
_CHARS_PER_TOKEN = 2.0

# 청킹 경계 정책: SUBHEADING/SECTION/SLIDE 같은 논리 경계는 heading_path 에
# 반영되어 있으므로(파서가 슬라이드 제목/소제목을 heading_path 로 채움),
# _group_by_heading() 이 heading_path 변화를 경계로 삼아 충돌을 막는다.
# 별도의 경계 타입 집합을 두지 않고 heading_path 단일 기준으로 통일한다.

# 독립 청크로 유지해야 하는 노드 종류 — 다른 텍스트와 절대 합치지 않는다.
_STANDALONE_TYPES = frozenset({ElementType.TABLE, ElementType.FIGURE_CAPTION})


def _estimate_tokens(text: str) -> int:
    """
    글자수 기반 토큰 수 추정(휴리스틱).

    왜 휴리스틱인가: 정확한 토크나이저(예: e5-large tokenizer)는 무거운
    의존성이며, 청킹 단계에서는 대략적인 크기 제어만 필요하다. 한국어는
    글자당 토큰 비율이 높으므로 보수적으로(글자수/2) 잡는다.
    """
    return max(1, int(len(text) / _CHARS_PER_TOKEN))


class StructureAwareChunker:
    """
    구조 트리를 계층형 청크 목록으로 변환하는 청커.

    크기 상수는 모듈 기본값을 쓰되, 생성자 인자로 덮어쓸 수 있다(튜닝 편의).
    """

    def __init__(
        self,
        child_target_tokens: int = CHILD_TARGET_TOKENS,
        parent_max_tokens: int = PARENT_MAX_TOKENS,
    ) -> None:
        """
        Args:
            child_target_tokens: 소청크 목표 토큰(이 크기를 넘기지 않게 분할).
            parent_max_tokens: 부모청크 최대 토큰(넘으면 부모를 더 쪼갠다).
        """
        self.child_target_tokens = child_target_tokens
        self.parent_max_tokens = parent_max_tokens

    # ─── 공개 진입점 ───

    def chunk(self, tree: DocumentTree) -> list[DocumentChunk]:
        """
        구조 트리를 청크 목록으로 변환한다.

        절차:
          1. 트리를 평탄화해 "리프 노드(실제 텍스트를 가진 노드)" 목록을 얻는다.
             읽기 순서(order)와 heading_path 가 보존된다.
          2. 같은 heading_path 끼리 묶어 "논리 그룹"을 만든다(경계 분할).
          3. 그룹마다 부모청크 1개 + 소청크 N개를 생성하고 parent_chunk_id 연결.
        """
        leaves = self._flatten(tree.nodes)
        groups = self._group_by_heading(leaves)

        chunks: list[DocumentChunk] = []
        for heading_path, group_nodes in groups:
            chunks.extend(self._chunk_group(heading_path, group_nodes))
        return chunks

    # ─── 1) 평탄화 ───

    def _flatten(self, nodes: tuple[DocumentNode, ...]) -> list[DocumentNode]:
        """
        트리를 깊이 우선으로 순회해 "텍스트를 가진 리프 노드"만 모은다.

        컨테이너성 노드(SLIDE 등, 자식이 있는 노드)는 그 자체로는 청크가
        되지 않고, 자식들이 heading_path 를 통해 맥락을 물려받는다.
        읽기 순서(order)는 트리 순회 순서로 자연스럽게 보존된다.
        """
        leaves: list[DocumentNode] = []
        for node in nodes:
            if node.children:
                # 컨테이너 — 자식을 재귀로 평탄화.
                leaves.extend(self._flatten(node.children))
                # 컨테이너 자신이 텍스트도 가진 경우는 드물지만, 있으면 무시한다
                # (SLIDE 의 text 는 제목 표시용이며 heading_path 로 이미 반영됨).
            elif node.text.strip():
                # 리프 + 실제 텍스트 — 청크 후보.
                leaves.append(node)
        return leaves

    # ─── 2) heading_path 별 그룹화 (경계 분할) ───

    @staticmethod
    def _group_by_heading(
        leaves: list[DocumentNode],
    ) -> list[tuple[tuple[str, ...], list[DocumentNode]]]:
        """
        리프 노드를 heading_path 기준으로 묶는다.

        규칙 ① (충돌 방지 핵심): heading_path 가 바뀌면 새 그룹을 시작한다.
        즉 서로 다른 소제목/슬라이드의 텍스트는 절대 같은 그룹(=같은 부모청크)에
        섞이지 않는다. 같은 heading_path 가 연속될 때만 한 그룹으로 묶는다.

        반환: [(heading_path, [노드...]), ...] — 읽기 순서 유지.
        """
        groups: list[tuple[tuple[str, ...], list[DocumentNode]]] = []
        current_path: tuple[str, ...] | None = None
        current: list[DocumentNode] = []

        for node in leaves:
            if node.heading_path != current_path:
                # 경계 전환 — 이전 그룹을 닫고 새 그룹 시작.
                if current:
                    groups.append((current_path or (), current))
                current_path = node.heading_path
                current = [node]
            else:
                current.append(node)

        if current:
            groups.append((current_path or (), current))
        return groups

    # ─── 3) 그룹 → 부모청크 + 소청크 ───

    def _chunk_group(
        self,
        heading_path: tuple[str, ...],
        nodes: list[DocumentNode],
    ) -> list[DocumentChunk]:
        """
        한 논리 그룹(같은 heading_path)을 부모청크 + 소청크들로 변환한다.

        절차:
          1. 부모청크: 그룹 전체 텍스트를 묶은 컨텍스트용 청크.
             parent_max_tokens 를 넘으면 여러 부모로 분할한다.
          2. 소청크: 규칙 ②③ 에 따라 표/그림캡션은 독립 청크로, 그 외 텍스트는
             child_target_tokens 단위로 모아 분할한다.
          3. 각 소청크에 자신이 속한 부모청크의 id 를 parent_chunk_id 로 연결.
        """
        # ── 1) 부모청크 생성 (컨텍스트용) ──
        parent_chunks = self._build_parents(heading_path, nodes)

        # ── 2) 소청크 생성 (검색용) ──
        child_chunks = self._build_children(heading_path, nodes)

        # ── 3) 소청크를 부모청크에 연결 ──
        # 같은 그룹이므로 모든 소청크는 (분할된 부모 중) 첫 부모에 매핑한다.
        # 부모가 여러 개로 분할된 경우에도 같은 heading_path 의 컨텍스트이므로
        # 첫 부모를 대표로 연결한다(검색 후 부모 확장 시 충분한 맥락 제공).
        if parent_chunks:
            primary_parent_id = self._chunk_id(parent_chunks[0])
            for c in child_chunks:
                c.parent_chunk_id = primary_parent_id

        return [*parent_chunks, *child_chunks]

    def _build_parents(
        self,
        heading_path: tuple[str, ...],
        nodes: list[DocumentNode],
    ) -> list[DocumentChunk]:
        """
        그룹 텍스트를 부모청크(들)로 묶는다.

        TABLE/FIGURE_CAPTION 도 부모청크 안에 포함시켜 "이 소제목 아래 전체
        맥락"을 제공한다(소청크 단계에서만 독립 분리). parent_max_tokens 를
        넘으면 여러 부모로 쪼갠다(드문 경우 — 보통 한 소제목은 한 부모).
        """
        parents: list[DocumentChunk] = []
        buffer: list[str] = []
        buffer_tokens = 0
        part_index = 0

        def flush() -> None:
            nonlocal buffer, buffer_tokens, part_index
            if not buffer:
                return
            content = "\n\n".join(buffer)
            parents.append(
                DocumentChunk(
                    content=content,
                    heading_path=heading_path,
                    # 부모청크의 대표 종류는 SUBHEADING(맥락 단위) 의미를 갖는다.
                    element_type=ElementType.SUBHEADING,
                    page=nodes[0].page if nodes else None,
                    parent_chunk_id=None,
                    is_parent=True,
                )
            )
            buffer = []
            buffer_tokens = 0
            part_index += 1

        for node in nodes:
            t = _estimate_tokens(node.text)
            # 한 노드가 부모 한도를 통째로 넘기면 단독 부모로 처리.
            if buffer_tokens + t > self.parent_max_tokens and buffer:
                flush()
            buffer.append(node.text)
            buffer_tokens += t

        flush()
        return parents

    def _build_children(
        self,
        heading_path: tuple[str, ...],
        nodes: list[DocumentNode],
    ) -> list[DocumentChunk]:
        """
        그룹을 검색용 소청크들로 분할한다.

        규칙:
          ② TABLE 은 분할하지 않고 독립 소청크 1개로 유지(행/열 보존).
          ③ FIGURE_CAPTION 도 독립 소청크로 유지.
          그 외(PARAGRAPH/LIST_ITEM/TEXTBOX): child_target_tokens 단위로 인접
          노드를 모아 분할하되, 경계는 이미 그룹 단계에서 분리되었으므로
          여기서는 같은 맥락 안의 본문만 모은다(충돌 없음).
        """
        children: list[DocumentChunk] = []
        buffer: list[str] = []
        buffer_tokens = 0
        # 버퍼에 모인 본문의 대표 페이지(첫 노드 기준).
        buffer_page: int | None = None

        def flush() -> None:
            nonlocal buffer, buffer_tokens, buffer_page
            if not buffer:
                return
            children.append(
                DocumentChunk(
                    content="\n".join(buffer),
                    heading_path=heading_path,
                    element_type=ElementType.PARAGRAPH,
                    page=buffer_page,
                    parent_chunk_id=None,  # 이후 _chunk_group 에서 채움
                    is_parent=False,
                )
            )
            buffer = []
            buffer_tokens = 0
            buffer_page = None

        for node in nodes:
            # ②③ 독립 유지 종류 — 버퍼를 먼저 비우고 단독 소청크로 추가.
            if node.element_type in _STANDALONE_TYPES:
                flush()
                children.append(
                    DocumentChunk(
                        content=node.text,
                        heading_path=heading_path,
                        element_type=node.element_type,
                        page=node.page,
                        parent_chunk_id=None,
                        is_parent=False,
                    )
                )
                continue

            t = _estimate_tokens(node.text)
            # 목표 토큰을 넘기면 현재 버퍼를 비우고 새로 시작(인접 본문만 묶음).
            if buffer_tokens + t > self.child_target_tokens and buffer:
                flush()
            if buffer_page is None:
                buffer_page = node.page
            buffer.append(node.text)
            buffer_tokens += t

        flush()
        return children

    # ─── 안정적 청크 식별자 ───

    @staticmethod
    def _chunk_id(chunk: DocumentChunk) -> str:
        """
        부모청크의 안정적 식별자를 생성한다(parent_chunk_id 로 사용).

        heading_path + page + 내용 해시 조합의 SHA-256(앞 32자). 같은 입력이면
        같은 id 가 나와 재적재 시에도 소청크→부모 연결이 안정적으로 유지된다.
        """
        key = (
            " > ".join(chunk.heading_path)
            + f"|{chunk.page}|"
            + hashlib.sha256(chunk.content.encode("utf-8")).hexdigest()[:16]
        )
        return hashlib.sha256(key.encode("utf-8")).hexdigest()[:32]
