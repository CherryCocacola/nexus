"""
문서 파서 추상 인터페이스(DocumentParser ABC)와 파서 레지스트리(ParserRegistry).

■ 이 파일이 하는 일 (한 줄 요약)
  "문서 파일 1개 → 구조 트리(DocumentTree)"로 바꾸는 파서들이 공통으로
  지켜야 할 규칙(계약)을 정의하고, 확장자별로 어떤 파서를 쓸지 골라주는
  등록소를 제공한다.

■ 왜 필요한가
  Nexus 인제스트(ingest) 파이프라인은 PPTX·PDF·HWPX·OCR 등 서로 다른
  포맷을 받아들인다. 포맷마다 파싱 방법은 완전히 다르지만, 파이프라인
  입장에서는 "아무 문서나 넣으면 DocumentTree가 나온다"는 한 가지 모습만
  보이길 원한다. 그래서 모든 포맷별 파서가 똑같은 인터페이스
  (DocumentParser)를 구현하도록 강제하고, 파이프라인은 그 인터페이스에만
  의존한다. 새 포맷이 생겨도 파이프라인 코드는 손대지 않는다.

■ 이 파일이 노출하는 것
  - DocumentParser (ABC): 포맷별 파서 구현체가 상속해야 하는 추상 클래스.
  - ParserRegistry: 확장자(예: ".pptx")를 주면 알맞은 파서를 찾아주는 등록소.

■ 왜 "추상 인터페이스 + 어댑터 슬롯"인가 (v7.3 Part 2.3, Part 6.3)
  라이선스 정책상 기본 구현에는 청정(MIT/Apache) 라이브러리만 쓴다. 반면
  성능이 더 좋은 상용 라이브러리(예: 구매한 PDF SDK)는 같은 인터페이스를
  구현하는 "어댑터 슬롯"으로 나중에 끼워 넣을 수 있게 열어 둔다. 인터페이스
  모양만 맞추면 파이프라인은 어떤 파서가 실제로 도는지 알 필요가 없다.
  상용 슬롯을 등록하면 그게 우선 쓰이고, 없으면 자동으로 청정 기본 파서로
  폴백된다(자세한 규칙은 ParserRegistry 참고).

■ 의존성 방향 (아키텍처 규칙 P2)
  이 모듈은 core.ingest.types 만 import 한다. core/rag·core/model 등 상위
  레이어를 import 하지 않으므로 역방향 의존이나 순환 import 위험이 없다.
  즉 이 파일은 인제스트 레이어의 "바닥(토대)" 계약에 해당한다.

작성자: 이현수 / 작성일: 2026-07-05
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

    포맷별 실제 파서(예: PptxParser, PdfParser 등)는 모두 이 클래스를 상속하고
    아래 프로퍼티/메서드를 구현해야 한다. ABC(추상 베이스 클래스)이므로 직접
    인스턴스화할 수 없고, 추상 멤버를 하나라도 빼먹으면 인스턴스 생성 시점에
    TypeError 가 난다 — 계약 누락을 조기에 잡아 주는 안전장치다.

    구현체가 반드시 지켜야 할 계약(중요, 3가지):
      1. can_parse(): 이 파일을 내가 처리할 수 있는지 fail-closed 로 판단한다.
         확장자·매직바이트를 보되, 조금이라도 불확실하면 False 를 낸다
         (애매하면 처리하지 않는다 = 잘못된 파서가 파일을 망치는 것을 방지).
      2. parse(): 부분 실패를 예외로 터뜨리지 말고 fail-soft 로 다룬다. 도형
         하나·표 하나 파싱에 실패해도 성공한 부분은 트리에 담고, 실패 사유는
         DocumentTree.warnings 에 모아 둔다. 다만 파일이 아예 없거나 열 수조차
         없는 치명적 상황은 예외를 던져도 된다(그건 파이프라인 상위가 처리).
      3. 외부 네트워크 호출 절대 금지 — Nexus 는 에어갭(폐쇄망) 시스템이다.
    """

    @property
    @abstractmethod
    def supported_extensions(self) -> tuple[str, ...]:
        """
        이 파서가 처리하는 확장자 목록. 반드시 소문자 + 점 포함 형식이어야
        한다. 예: (".pptx",) 또는 (".pdf", ".PDF" 금지 → ".pdf" 만).

        ParserRegistry.register() 가 이 값을 읽어 "확장자 → 파서" 매핑을
        만든다. 그래서 여기 형식이 어긋나면 등록소가 파서를 못 찾는다.
        튜플로 두는 이유: 불변(수정 불가)이라 실수로 바뀔 일이 없기 때문.
        """

    @property
    @abstractmethod
    def requires_gpu(self) -> bool:
        """
        이 파서가 GPU(레이아웃 분석·OCR 모델 등)를 필요로 하는지 여부.

        왜 필요한가: 파이프라인의 티어 분기(v7.3 Part 4)에서 "이 파서를 지금
        이 호스트에서 돌려도 되는가"를 판단하는 데 쓴다. GPU 없는 노드에
        GPU 파서를 배정하면 안 되기 때문이다.
        - True  : OCR/딥러닝 레이아웃 파서처럼 GPU 가 있어야 도는 경우.
        - False : PPTX 처럼 파일 내부 구조를 그대로 읽는 네이티브 구조 파서.
        """

    @abstractmethod
    def can_parse(self, path: Path) -> bool:
        """
        주어진 파일 경로를 이 파서가 실제로 처리할 수 있는지 판단한다.

        확장자만으로는 부족할 수 있다(확장자는 .pdf 인데 내용은 깨진 파일일
        수 있음). 그래서 구현체는 확장자에 더해 파일 앞부분의 매직바이트
        (파일 종류를 나타내는 시그니처) 등을 함께 확인하는 것이 좋다.

        Args:
            path: 검사할 파일 경로(Path 객체).

        Returns:
            처리 가능하면 True, 아니면 False.

        fail-closed 원칙: 확장자가 안 맞거나 매직바이트 확인 중 조금이라도
        의심스러우면 무조건 False(불확실하면 거부). 이 판단은 레지스트리의
        get_for_path() 가 후보 파서를 고를 때 최종 확인용으로 호출한다.
        """

    @abstractmethod
    async def parse(self, path: Path) -> DocumentTree:
        """
        문서를 실제로 읽어 구조 트리(DocumentTree)로 변환한다.

        async 인 이유: 파일 I/O 나 (GPU 파서의 경우) 원격 추론 호출처럼 시간이
        걸리는 작업을 이벤트 루프를 막지 않고 처리하기 위해서다. 구현체는
        무거운 동기 작업을 스레드로 오프로딩하는 등으로 협조하는 것이 좋다.

        Args:
            path: 파싱할 문서 파일 경로. 호출 전에 can_parse() 로 걸러졌다는
                  전제이지만, 구현체는 방어적으로 다시 확인해도 된다.

        Returns:
            DocumentTree — 문서 구조를 담은 트리. 부분 실패가 있었다면 그
            사유들이 DocumentTree.warnings 에 함께 담겨 온다.

        fail-soft 원칙: 도형/표 같은 개별 요소 파싱이 실패해도 예외로 전체를
        중단하지 말고, 성공한 부분만 트리에 담고 실패 사유는 warnings 에
        모은다. 파일이 없거나 열 수 없는 치명적 상황만 예외를 허용한다.
        """


# ─────────────────────────────────────────────
# ParserRegistry — 확장자→파서 등록/조회
# ─────────────────────────────────────────────
class ParserRegistry:
    """
    파서 등록소(레지스트리). 확장자(예: ".pptx")를 주면 그 파일에 알맞은
    파서를 골라 준다. 파이프라인은 이 클래스만 알면 되고, 개별 파서 구현체를
    직접 몰라도 된다(느슨한 결합).

    우선순위 규칙 (v7.3 Part 6.3 — 어댑터 슬롯):
      같은 확장자에 여러 파서가 등록될 수 있다(예: 기본 pdfplumber +
      나중에 붙이는 상용 PyMuPDF 어댑터). 이때 priority 가 높은 파서가
      우선한다. 기본 청정 파서는 낮은 priority, 상용 슬롯은 높은 priority 로
      등록하면 → 상용이 있으면 상용이 쓰이고, 없으면 자동으로 청정 기본으로
      폴백된다. 코드 수정 없이 등록만으로 파서를 교체할 수 있는 구조다.

    fail-closed: 등록되지 않은 확장자에는 None 을 돌려준다. "모르는 파일은
    함부로 건드리지 않는다"는 안전 우선 원칙이다.
    """

    def __init__(self) -> None:
        # 내부 저장 구조: 확장자(소문자) → [(priority, 등록순번, parser), ...].
        # 한 확장자에 여러 파서가 쌓일 수 있어 리스트로 보관한다.
        # 두 번째 필드 '등록순번'은 tie-breaker(동점 처리)다: priority 가 같을
        # 때 먼저 등록된 파서를 앞에 오게 해 정렬 결과를 항상 결정론적으로
        # (= 실행할 때마다 동일하게) 만든다.
        self._by_ext: dict[str, list[tuple[int, int, DocumentParser]]] = {}
        self._seq: int = 0  # 등록할 때마다 1씩 증가하는 순번 카운터

    def register(self, parser: DocumentParser, *, priority: int = 0) -> None:
        """
        파서를 레지스트리에 등록한다.

        파서가 지원한다고 밝힌 모든 확장자(supported_extensions)에 대해 자기
        자신을 후보로 집어넣고, 그 확장자의 후보 목록을 우선순위대로 다시
        정렬해 둔다. 그래서 조회 시점(get_for_path)에는 정렬 없이 앞에서부터
        훑기만 하면 된다.

        Args:
            parser: 등록할 DocumentParser 구현체.
            priority: 우선순위(클수록 먼저 시도). 상용 어댑터 슬롯은 큰 값으로
                      주어 기본 청정 파서보다 앞서게 한다. 기본값 0.
        """
        for ext in parser.supported_extensions:
            # 확장자는 항상 소문자로 정규화한다. ".PDF"/".pdf" 가 다른 키로
            # 갈라지지 않게 해 조회를 일관되게 만든다(점 포함은 그대로 유지).
            key = ext.lower()
            # 이 확장자 칸이 없으면 빈 리스트로 만들고, 거기에 후보를 추가.
            self._by_ext.setdefault(key, []).append((priority, self._seq, parser))
            # 정렬 기준: priority 내림차순(-t[0]) → 높은 우선순위가 앞으로.
            # 동점이면 등록순번 오름차순(t[1]) → 먼저 등록된 파서가 앞으로.
            self._by_ext[key].sort(key=lambda t: (-t[0], t[1]))
        # 순번은 파서 1개 등록 = 1 증가(그 파서의 여러 확장자는 같은 순번 공유).
        self._seq += 1

    def get_for_path(self, path: Path) -> DocumentParser | None:
        """
        주어진 경로에 가장 알맞은 파서 하나를 골라 반환한다.

        동작 순서:
          1) 경로의 확장자(소문자)로 후보 목록을 찾는다. 없으면 None.
          2) 후보를 우선순위 순서대로(등록 시 이미 정렬돼 있음) 훑으면서,
             각 파서의 can_parse() 로 실제 처리 가능 여부를 확인한다.
          3) 처음으로 can_parse() 가 True 인 파서를 반환한다.

        왜 can_parse() 로 한 번 더 확인하나: 확장자는 맞아도 파일 내용
        (매직바이트)이 다를 수 있기 때문이다. 확장자 매칭만 믿지 않는
        fail-closed 방어다. 모든 후보가 거부하면 None(처리 불가).

        Args:
            path: 파서를 찾을 대상 파일 경로.

        Returns:
            사용할 DocumentParser, 또는 알맞은 파서가 없으면 None.
        """
        candidates = self._by_ext.get(path.suffix.lower())
        if not candidates:
            return None
        # 이미 (우선순위 높은 순 → 먼저 등록된 순)으로 정렬돼 있으므로,
        # 앞에서부터 첫 번째로 처리 가능하다는 파서를 그대로 쓰면 된다.
        for _priority, _seq, parser in candidates:
            if parser.can_parse(path):
                return parser
        return None

    def supported_extensions(self) -> tuple[str, ...]:
        """
        현재 레지스트리에 등록된 모든 확장자를 정렬해 튜플로 반환한다.

        진단/상태 표시용(예: "이 노드가 지원하는 포맷 목록")으로 쓴다.
        정렬해 주는 이유: 호출할 때마다 순서가 같아 출력이 일관되기 때문.
        """
        return tuple(sorted(self._by_ext.keys()))
