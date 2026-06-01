"""
문서 파서 추상 인터페이스(DocumentParser ABC)와 파서 레지스트리.

역할:
  "문서 1개 → 구조 트리(DocumentTree)" 변환을 담당하는 파서의 공통 계약을
  정의한다. 포맷별 구현체(PPTX/PDF/HWPX/OCR ...)는 모두 이 계약을 따른다.

왜 추상 인터페이스인가 (v7.3 Part 2.3, Part 6.3 — 어댑터 슬롯):
  라이선스 정책상 기본 구현은 청정(MIT/Apache) 라이브러리만 쓰고, 상용
  라이브러리(예: 구매한 PDF SDK)는 같은 인터페이스를 구현하는 "어댑터
  슬롯"으로 끼워 넣을 수 있게 한다. 인터페이스만 맞추면 파이프라인은 어떤
  파서를 쓰는지 몰라도 된다.

의존성 방향 (P2):
  이 모듈은 core.ingest.types 만 의존한다. core/rag·core/model 을 import 하지
  않으므로 역방향/순환 위험이 없다.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from core.ingest.types import DocumentTree


# ─────────────────────────────────────────────
# DocumentParser — 추상 인터페이스
# ─────────────────────────────────────────────
class DocumentParser(ABC):
    """
    문서 1개를 구조 트리(DocumentTree)로 파싱하는 추상 인터페이스.

    구현체가 지켜야 할 계약:
      1. can_parse(): 확장자·매직바이트로 처리 가능 여부를 fail-closed 로
         판단한다(불확실하면 False — 함부로 처리하지 않는다).
      2. parse(): 실패를 예외로 던지지 않고, 읽을 수 있는 만큼의 부분 트리 +
         DocumentTree.warnings 로 표현한다(fail-soft). 단, 파일 자체가
         없거나 열 수 없는 치명적 상황은 예외를 허용한다(파이프라인이 처리).
      3. 외부 네트워크 호출 절대 금지(에어갭).
    """

    @property
    @abstractmethod
    def supported_extensions(self) -> tuple[str, ...]:
        """
        이 파서가 처리하는 확장자 목록 (소문자, 점 포함). 예: (".pptx",).

        레지스트리가 확장자→파서 매핑을 만들 때 사용한다.
        """

    @property
    @abstractmethod
    def requires_gpu(self) -> bool:
        """
        GPU 레이아웃/OCR 모델을 사용하는지 여부.

        티어 분기(v7.3 Part 4)에서 "이 파서를 이 호스트에서 돌릴 수 있는가"를
        판단하는 데 쓴다. PPTX 같은 네이티브 구조 파서는 False.
        """

    @abstractmethod
    def can_parse(self, path: Path) -> bool:
        """
        주어진 경로를 이 파서가 처리할 수 있는지 판단한다.

        fail-closed 원칙: 확장자가 맞지 않거나, 매직바이트 확인 중 의심스러우면
        False 를 반환한다(불확실하면 거부).
        """

    @abstractmethod
    async def parse(self, path: Path) -> DocumentTree:
        """
        문서를 구조 트리로 파싱한다.

        fail-soft: 일부 요소(도형/표 등) 파싱에 실패해도 예외로 중단하지 않고,
        성공한 부분만 트리에 담고 실패 사유를 DocumentTree.warnings 에 모은다.
        """


# ─────────────────────────────────────────────
# ParserRegistry — 확장자→파서 등록/조회
# ─────────────────────────────────────────────
class ParserRegistry:
    """
    파서 등록소. 확장자(예: ".pptx")로 적절한 파서를 찾아준다.

    우선순위 규칙 (v7.3 Part 6.3 — 어댑터 슬롯):
      같은 확장자에 여러 파서가 등록될 수 있다(예: 기본 pdfplumber +
      상용 PyMuPDF 어댑터). priority 가 높은 파서가 우선한다. 기본 청정
      파서는 낮은 priority, 상용 슬롯은 높은 priority 로 등록하면 상용이
      우선 사용되고, 미등록 시 자동으로 청정 기본으로 폴백된다.

    fail-closed: 등록되지 않은 확장자는 None 을 반환한다(임의 처리 금지).
    """

    def __init__(self) -> None:
        # 확장자 → [(priority, 등록순번, parser)] 목록.
        # 등록순번(tie-breaker)은 priority 가 같을 때 먼저 등록된 것을 우선시켜
        # 정렬을 결정론적으로 만든다.
        self._by_ext: dict[str, list[tuple[int, int, DocumentParser]]] = {}
        self._seq: int = 0  # 등록 순번 카운터

    def register(self, parser: DocumentParser, *, priority: int = 0) -> None:
        """
        파서를 등록한다.

        Args:
            parser: 등록할 DocumentParser 구현체.
            priority: 우선순위(클수록 우선). 상용 어댑터 슬롯은 큰 값으로.
        """
        for ext in parser.supported_extensions:
            # 확장자는 항상 소문자·점 포함으로 정규화해 매칭을 일관되게 한다.
            key = ext.lower()
            self._by_ext.setdefault(key, []).append((priority, self._seq, parser))
            # priority 내림차순, 같은 priority 면 등록순번 오름차순(먼저 등록 우선).
            self._by_ext[key].sort(key=lambda t: (-t[0], t[1]))
        self._seq += 1

    def get_for_path(self, path: Path) -> DocumentParser | None:
        """
        경로의 확장자에 맞는 최우선 파서를 반환한다.

        실제 처리 가능 여부는 파서의 can_parse() 로 한 번 더 확인한다
        (확장자는 맞아도 매직바이트가 다를 수 있으므로 fail-closed).
        후보를 우선순위 순으로 훑어 can_parse() 가 True 인 첫 파서를 쓴다.
        없으면 None.
        """
        candidates = self._by_ext.get(path.suffix.lower())
        if not candidates:
            return None
        for _priority, _seq, parser in candidates:
            if parser.can_parse(path):
                return parser
        return None

    def supported_extensions(self) -> tuple[str, ...]:
        """현재 등록된 모든 확장자를 정렬해 반환한다(진단/표시용)."""
        return tuple(sorted(self._by_ext.keys()))
