"""
구조 인식 청킹 — StructureAwareChunker.

■ 이 파일이 하는 일 (한눈에)
  파서가 만든 "문서 구조 트리(DocumentTree)"를 받아, 벡터 임베딩과 검색에
  쓰기 좋은 크기의 "청크(DocumentChunk)" 목록으로 잘라준다. RAG 파이프라인의
  전처리 단계로, 여기서 잘린 청크가 이후 임베딩→pgvector 적재→검색으로 흘러간다.

  가장 중요한 목표는 "충돌 없는 청킹"(v7.3 Part 1.1, 2.6)이다. 즉 서로 다른
  논리 영역(소제목/표/그림캡션)의 텍스트가 한 청크 안에 뒤섞이지 않게 한다.
  섞이면 검색 품질이 나빠지고(엉뚱한 문맥이 함께 검색됨) 답변 정확도가 떨어진다.

■ 계층형(parent-child) 청킹 (v7.3 Part 2.6)
  검색 정밀도와 문맥 충분성을 동시에 잡기 위해 두 종류의 청크를 함께 만든다.
  - 소청크(child): 작게 잘라 정밀 "검색"용 (128~256 토큰 상당). 질의와 잘 맞는
    좁은 조각을 찾는 데 유리하다.
  - 부모청크(parent): 같은 heading_path 의 소청크들을 묶은 큰 청크
    (512~1536 토큰 상당). 검색으로 소청크를 찾은 뒤, 모델에게 넣어줄 "넓은
    문맥"을 제공하는 용도다.
  - 소청크의 parent_chunk_id 가 자기 부모청크를 가리킨다. 그래서 "좁게 검색 →
    부모로 확장" 이라는 2단 검색 전략이 가능하다.

■ 청킹 규칙 (v7.3 Part 2.6)
  ① SUBHEADING/SLIDE 등 논리 경계를 넘어 소청크를 합치지 않는다(충돌 방지 핵심).
  ② TABLE 은 한 청크로 유지한다(표를 쪼개면 행/열 의미가 깨지므로 분할 금지).
  ③ FIGURE_CAPTION 은 인접 본문과 분리해 독립 청크로 둔다.
  ④ 부모청크(is_parent=True) 와 소청크(is_parent=False) 를 동시에 만들고
     parent_chunk_id 로 연결한다.
  ⑤ 문자 단위 overlap(겹치기) 대신 heading_path 공유로 문맥을 잇는다. 같은
     소제목 아래 조각들은 heading_path 가 같아, 굳이 글자를 겹치지 않아도 이어진다.

■ 토큰 추정 (v7.3 Part 10 #9 — 한국어 e5-large 기준 튜닝 필요)
  청킹 단계에서는 정확한 토크나이저를 쓰지 않고 글자수 기반 휴리스틱으로 크기를
  가늠한다. 한국어는 글자당 토큰 비율이 높아 보수적으로 추정한다(_estimate_tokens 참조).

■ 주요 구성요소
  - _estimate_tokens(): 글자수 → 토큰 수 대략 추정.
  - StructureAwareChunker: 실제 청킹 로직을 담은 클래스(공개 진입점은 chunk()).

■ 의존성 방향 (P2)
  core.ingest.types 만 의존한다(역방향/순환 import 없음).

작성자: 이현수 / 작성일: 2026-07-05
"""

from __future__ import annotations

import hashlib

from core.ingest.types import DocumentChunk, DocumentNode, DocumentTree, ElementType

# ─────────────────────────────────────────────
# 크기 상수 (v7.3 Part 2.6 권장값, 토큰 상당)
# ─────────────────────────────────────────────
# 소청크 목표 토큰. 검색 정밀도를 위해 작게 유지한다. 이 값을 넘으면 버퍼를
# 비우고 새 소청크를 시작한다(_build_children 참조).
CHILD_TARGET_TOKENS = 256
# 부모청크 최대 토큰. 컨텍스트 주입용이라 크게 잡는다. 이 값을 넘으면 부모청크를
# 여러 개로 쪼갠다(_build_parents 참조).
PARENT_MAX_TOKENS = 1536

# 글자수 → 토큰 수 추정 계수. 한국어/영어 혼합 문서를 보수적으로 추정하기 위한
# 휴리스틱으로, "한 토큰 ≈ 2 글자" 라고 가정한다. 즉 토큰수 = 글자수 / 2.
# 정확한 토크나이저 연동은 후속 튜닝 대상(v7.3 Part 10 #9).
_CHARS_PER_TOKEN = 2.0

# ── 청킹 경계 정책(중요) ──
# SUBHEADING/SECTION/SLIDE 같은 논리 경계는 이미 heading_path 에 반영되어 있다.
# (파서가 슬라이드 제목·소제목을 각 노드의 heading_path 로 채워 넣는다.)
# 따라서 _group_by_heading() 은 heading_path 가 바뀌는 지점을 경계로 삼아
# 그룹을 나누기만 하면 충돌(서로 다른 영역의 텍스트가 섞이는 것)을 막을 수 있다.
# 별도의 "경계 타입 집합"을 두지 않고 heading_path 단일 기준으로 통일한 이유다.

# 독립 청크로 유지해야 하는 노드 종류 — 다른 텍스트와 절대 합치지 않는다.
# 표는 쪼개면 행/열 구조가 깨지고, 그림 캡션은 본문과 섞이면 의미가 흐려진다.
_STANDALONE_TYPES = frozenset({ElementType.TABLE, ElementType.FIGURE_CAPTION})


def _estimate_tokens(text: str) -> int:
    """
    글자수 기반으로 텍스트의 토큰 수를 대략 추정한다(휴리스틱).

    왜 휴리스틱인가: 정확한 토크나이저(예: e5-large tokenizer)는 무거운
    의존성이며, 청킹 단계에서는 대략적인 크기 제어만 필요하다. 한국어는
    글자당 토큰 비율이 높으므로 보수적으로(글자수/2) 잡는다.

    Args:
        text: 토큰 수를 추정할 문자열.

    Returns:
        추정 토큰 수(정수). 빈 문자열이라도 최소 1을 반환한다(0으로 나눔·0
        크기 청크 같은 경계 문제를 피하기 위해 max(1, ...) 처리).
    """
    return max(1, int(len(text) / _CHARS_PER_TOKEN))


class StructureAwareChunker:
    """
    구조 트리(DocumentTree)를 계층형 청크(부모+소청크) 목록으로 변환하는 청커.

    사용법: 인스턴스를 만든 뒤 chunk(tree) 를 호출하면 DocumentChunk 목록을
    돌려준다. 내부는 (1) 트리 평탄화 → (2) heading_path 별 그룹화 →
    (3) 그룹마다 부모청크·소청크 생성 의 3단계로 이어진다.

    크기 상수(소청크/부모청크 토큰 한도)는 모듈 기본값을 쓰되, 문서 종류나
    임베딩 모델에 맞춰 생성자 인자로 덮어쓸 수 있다(튜닝 편의).
    """

    def __init__(
        self,
        child_target_tokens: int = CHILD_TARGET_TOKENS,
        parent_max_tokens: int = PARENT_MAX_TOKENS,
    ) -> None:
        """
        청커를 초기화하고 청크 크기 한도를 인스턴스에 저장한다.

        Args:
            child_target_tokens: 소청크 목표 토큰. 이 크기를 넘기지 않도록
                본문을 모아 분할한다(검색용, 작게).
            parent_max_tokens: 부모청크 최대 토큰. 이 크기를 넘으면 부모를 더
                여러 개로 쪼갠다(컨텍스트 주입용, 크게).
        """
        self.child_target_tokens = child_target_tokens
        self.parent_max_tokens = parent_max_tokens

    # ─── 공개 진입점 ───

    def chunk(self, tree: DocumentTree) -> list[DocumentChunk]:
        """
        구조 트리를 청크 목록으로 변환한다. 이 클래스의 유일한 공개 진입점이다.

        절차:
          1. 트리를 평탄화해 "리프 노드(실제 텍스트를 가진 노드)" 목록을 얻는다.
             읽기 순서(order)와 heading_path 가 보존된다.
          2. 같은 heading_path 끼리 묶어 "논리 그룹"을 만든다(경계 분할).
          3. 그룹마다 부모청크 1개 이상 + 소청크 N개를 만들고 parent_chunk_id 연결.

        Args:
            tree: 파서가 만든 문서 구조 트리(DocumentTree).

        Returns:
            부모청크와 소청크가 섞인 DocumentChunk 리스트. 그룹 순서(=읽기 순서)를
            따르며, 각 그룹 안에서는 부모청크들이 먼저, 소청크들이 뒤에 온다.
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
        트리를 깊이 우선(DFS)으로 순회해 "텍스트를 가진 리프 노드"만 모은다.

        왜 리프만 모으나: 컨테이너성 노드(SLIDE 등, 자식이 있는 노드)는 그 자체로는
        청크가 되지 않는다. 자식들이 heading_path 를 통해 맥락(어느 슬라이드/소제목
        아래인지)을 이미 물려받았기 때문이다. 읽기 순서(order)는 재귀 순회 순서로
        자연스럽게 보존되므로 별도 정렬이 필요 없다.

        Args:
            nodes: 순회할 노드 튜플(최초 호출은 tree.nodes, 재귀 시 자식들).

        Returns:
            실제 텍스트를 가진 리프 노드들의 리스트(읽기 순서 유지).
        """
        leaves: list[DocumentNode] = []
        for node in nodes:
            if node.children:
                # 컨테이너 노드 — 자식들을 재귀로 평탄화해 이어붙인다.
                leaves.extend(self._flatten(node.children))
                # 컨테이너 자신이 텍스트도 함께 가진 경우는 드물지만, 있어도 무시한다.
                # (예: SLIDE 의 text 는 제목 표시용이며 이미 자식의 heading_path 로
                #  반영되어 있어, 청크로 또 만들면 중복이 된다.)
            elif node.text.strip():
                # 리프 노드이면서 공백을 제외한 실제 텍스트가 있으면 청크 후보로 채택.
                leaves.append(node)
        return leaves

    # ─── 2) heading_path 별 그룹화 (경계 분할) ───

    @staticmethod
    def _group_by_heading(
        leaves: list[DocumentNode],
    ) -> list[tuple[tuple[str, ...], list[DocumentNode]]]:
        """
        리프 노드들을 heading_path 기준으로 연속 구간별로 묶는다(경계 분할).

        규칙 ① (충돌 방지 핵심): heading_path 가 바뀌는 순간 새 그룹을 시작한다.
        즉 서로 다른 소제목/슬라이드의 텍스트는 절대 같은 그룹(= 같은 부모청크)에
        섞이지 않는다. 같은 heading_path 가 연속으로 이어질 때만 한 그룹으로 묶는다.

        주의: heading_path 가 "값이 같은지"만 비교하므로, 문서 뒤쪽에서 같은
        heading_path 가 떨어져 다시 나오면 별개의 그룹이 된다(연속 구간 기준).

        Args:
            leaves: _flatten() 이 만든 리프 노드 리스트(읽기 순서).

        Returns:
            [(heading_path, [노드...]), ...] 형태의 리스트. 읽기 순서를 유지한다.
            heading_path 가 비어 있을 수 있어 None 대신 빈 튜플 () 로 채운다.
        """
        groups: list[tuple[tuple[str, ...], list[DocumentNode]]] = []
        # current_path: 지금 모으는 중인 그룹의 heading_path(아직 없으면 None).
        current_path: tuple[str, ...] | None = None
        # current: 지금 모으는 중인 그룹의 노드 버퍼.
        current: list[DocumentNode] = []

        for node in leaves:
            if node.heading_path != current_path:
                # 경계 전환 — 지금까지 모은 그룹을 닫고, 새 그룹을 시작한다.
                if current:
                    # current_path 가 None 이면 빈 튜플로 대체해 타입을 맞춘다.
                    groups.append((current_path or (), current))
                current_path = node.heading_path
                current = [node]
            else:
                # 같은 heading_path 가 이어짐 — 현재 그룹에 계속 누적.
                current.append(node)

        # 루프 종료 후 마지막으로 모으던 그룹이 남아 있으면 마저 닫는다.
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

        Args:
            heading_path: 이 그룹이 속한 소제목 경로(모든 청크가 공유).
            nodes: 그룹에 속한 리프 노드들(읽기 순서).

        Returns:
            [부모청크들..., 소청크들...] 순서의 DocumentChunk 리스트.
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

        소청크와 달리 표/그림캡션(TABLE/FIGURE_CAPTION)도 부모청크 안에 그대로
        포함시킨다. 부모청크는 "이 소제목 아래 전체 맥락"을 통째로 제공하는 게
        목적이라, 여기서는 분리하지 않는다(독립 분리는 소청크 단계에서만 한다).
        parent_max_tokens 를 넘으면 여러 부모로 쪼갠다(드문 경우 — 보통 한
        소제목은 부모 한 개로 담긴다).

        Args:
            heading_path: 부모청크에 붙일 소제목 경로.
            nodes: 그룹의 리프 노드들(읽기 순서).

        Returns:
            부모 DocumentChunk 리스트(보통 1개, 길면 여러 개).
        """
        parents: list[DocumentChunk] = []
        # buffer: 아직 부모청크로 확정하지 않고 모으는 중인 텍스트 조각들.
        buffer: list[str] = []
        # buffer_tokens: 현재 버퍼에 쌓인 추정 토큰 수(한도 비교용).
        buffer_tokens = 0
        # part_index: 한 그룹이 여러 부모로 쪼개질 때의 순번(현재 로직상 참고용).
        part_index = 0

        def flush() -> None:
            # 현재 버퍼 내용을 하나의 부모청크로 확정하고 버퍼를 비운다.
            nonlocal buffer, buffer_tokens, part_index
            if not buffer:
                return
            # 부모청크 본문은 문단 사이를 빈 줄(\n\n)로 이어 가독성을 살린다.
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
            # 이 노드를 더하면 부모 한도를 넘고, 버퍼에 이미 내용이 있으면
            # 지금까지 모은 것을 먼저 부모청크로 확정한 뒤 새로 담기 시작한다.
            # (버퍼가 비어 있는데 노드 하나가 한도를 넘는 경우는 그냥 담아 단독
            #  부모가 되게 한다 — 텍스트를 임의로 잘라 의미를 깨지 않기 위함.)
            if buffer_tokens + t > self.parent_max_tokens and buffer:
                flush()
            buffer.append(node.text)
            buffer_tokens += t

        # 루프가 끝나고 버퍼에 남은 마지막 조각을 마저 부모청크로 확정한다.
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
          ② TABLE 은 분할하지 않고 독립 소청크 1개로 유지한다(행/열 구조 보존).
          ③ FIGURE_CAPTION 도 독립 소청크로 유지한다.
          그 외(PARAGRAPH/LIST_ITEM/TEXTBOX): child_target_tokens 단위로 인접
          노드를 모아 분할한다. 논리 경계는 이미 그룹 단계(_group_by_heading)에서
          갈라졌으므로, 여기서는 같은 맥락 안의 본문만 모은다(충돌 걱정 없음).

        Args:
            heading_path: 소청크에 붙일 소제목 경로.
            nodes: 그룹의 리프 노드들(읽기 순서).

        Returns:
            소청크 DocumentChunk 리스트. 여기서는 parent_chunk_id 를 아직 채우지
            않고(None), 상위 _chunk_group() 이 부모청크 id 로 연결한다.
        """
        children: list[DocumentChunk] = []
        # buffer: 아직 소청크로 확정하지 않고 모으는 중인 본문 조각들.
        buffer: list[str] = []
        # buffer_tokens: 현재 버퍼의 추정 토큰 수(목표치 비교용).
        buffer_tokens = 0
        # buffer_page: 버퍼에 처음 담긴 노드의 페이지 번호(소청크 대표 페이지).
        buffer_page: int | None = None

        def flush() -> None:
            # 현재 버퍼를 소청크 하나로 확정하고 버퍼 상태를 초기화한다.
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
            # 규칙 ②③ — 표/그림캡션은 독립 유지 대상. 지금까지 모으던 본문 버퍼를
            # 먼저 소청크로 확정(flush)한 뒤, 이 노드를 단독 소청크로 추가한다.
            # 이렇게 해야 표/캡션이 앞뒤 본문과 한 청크로 섞이지 않는다.
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
            # 일반 본문 노드 — 이 노드를 더하면 목표 토큰을 넘고 버퍼에 이미
            # 내용이 있으면, 지금까지 모은 것을 소청크로 확정하고 새로 시작한다.
            if buffer_tokens + t > self.child_target_tokens and buffer:
                flush()
            # 버퍼가 비어 있던 상태라면(=이 노드가 새 소청크의 첫 노드) 이 노드의
            # 페이지를 소청크 대표 페이지로 기록한다.
            if buffer_page is None:
                buffer_page = node.page
            buffer.append(node.text)
            buffer_tokens += t

        # 루프 종료 후 버퍼에 남은 마지막 본문을 소청크로 확정한다.
        flush()
        return children

    # ─── 안정적 청크 식별자 ───

    @staticmethod
    def _chunk_id(chunk: DocumentChunk) -> str:
        """
        부모청크의 안정적(deterministic) 식별자를 생성한다(parent_chunk_id 로 사용).

        구성: heading_path + page + 내용 해시 를 이어붙인 키를 SHA-256 으로 해싱해
        앞 32자를 쓴다. 무작위 UUID 가 아니라 "내용에서 파생된" id 라서, 같은
        문서를 다시 적재해도 같은 id 가 나온다. 덕분에 재적재 시에도 소청크 →
        부모 연결이 흔들리지 않고 그대로 유지된다.

        Args:
            chunk: id 를 만들 부모 DocumentChunk.

        Returns:
            32자 16진수 문자열 식별자.
        """
        key = (
            " > ".join(chunk.heading_path)
            + f"|{chunk.page}|"
            + hashlib.sha256(chunk.content.encode("utf-8")).hexdigest()[:16]
        )
        return hashlib.sha256(key.encode("utf-8")).hexdigest()[:32]
