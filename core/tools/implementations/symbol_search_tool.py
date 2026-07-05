"""
SymbolSearch 도구 — tb_symbols(Phase 10.0)에서 프로젝트 심볼을 검색한다.

[이 파일이 하는 일]
Nexus 프로젝트 소스코드에서 미리 인덱싱해 둔 "심볼"(함수/클래스/메서드)을
빠르게 찾아 주는 도구다. 심볼 인덱스는 tb_symbols 테이블에 적재돼 있고, 이
도구는 그 인덱스를 3가지 방식(정확 이름 / 부분 이름 / 의미 벡터)으로 조회한다.
AI 에이전트가 "그 함수 어디 있지?"를 Grep으로 파일 전체를 훑기 전에, 훨씬
저렴하게 위치를 특정하도록 돕는 것이 목적이다.

[사용 시나리오]
  - "query_loop 함수 어디 있어?"      → 이름 정확 매칭
  - "Config 관련 클래스 찾아줘"        → trigram(pg_trgm) 부분매칭
  - "토큰 추정하는 함수 있나?"         → 벡터 의미 검색 (이름을 모를 때)

[누구에게 노출되나]
Scout 서브에이전트에 노출되어 프로젝트 탐색 속도를 대폭 끌어올린다.
Worker에게도 SymbolSearch가 노출되면 "심볼 명시적 탐색" 1단계로 활용 가능하다.

[검색 모드 자동 선택 규칙]
  - 입력이 Python 식별자 모양(isidentifier) 또는 dotted path → 이름 매칭 우선
  - 그 외 자연어 → 벡터 검색 우선 (임베딩 프로바이더가 있을 때)
  - 이름 매칭 0건이고 임베딩 가능하면 벡터로 폴백

[주요 구성 요소]
  - SymbolSearchTool : BaseTool을 상속한 도구 본체 (검색 로직 담당)
  - _looks_like_identifier() : 입력이 식별자/qualified name인지 판별하는 헬퍼
  - _format_rows() : DB 조회 결과(dict 목록)를 사람이 읽기 좋은 텍스트로 변환

[의존 관계]
  - core.tools.base 의 BaseTool/ToolResult/권한 타입에 의존한다.
  - 실제 DB 조회는 직접 하지 않고, 부트스트랩이 주입한 SymbolStore(및 임베딩
    프로바이더)를 context.options 에서 꺼내 위임한다. (도구는 얇은 어댑터 역할)

작성자: 이현수 / 작성일: 2026-07-05
"""

from __future__ import annotations

import logging
import re
from typing import Any

from core.tools.base import (
    BaseTool,
    PermissionBehavior,
    PermissionResult,
    ToolResult,
    ToolUseContext,
)

# 이 모듈 전용 로거. "nexus.tools.symbol_search" 네임스페이스로 로그를 남긴다.
logger = logging.getLogger("nexus.tools.symbol_search")

# 입력 문자열이 "파이썬 식별자" 또는 "점으로 이어진 qualified name"인지 판별하는 정규식.
# 예) "query_loop", "KnowledgeStore", "KnowledgeStore.add" 는 매칭되고,
#     "토큰 추정하는 함수" 같은 자연어 문장은 매칭되지 않는다.
# 패턴 의미: 첫 글자는 영문/밑줄, 이후 영문/숫자/밑줄 반복, 그리고 ".식별자"가 0회 이상.
_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*$")


def _looks_like_identifier(q: str) -> bool:
    """입력이 식별자/qualified name 형태인지 판별한다.

    검색 모드를 고르는 첫 관문이다. 여기서 True가 나오면 "이름 매칭"을 먼저
    시도하고, False(자연어로 보임)면 벡터 의미 검색을 먼저 시도한다.

    매개변수:
        q: 사용자/에이전트가 넘긴 검색어. 앞뒤 공백은 제거하고 검사한다.

    반환:
        식별자/dotted path 모양이면 True, 빈 문자열이거나 자연어면 False.
    """
    # q가 빈 문자열/None이면 굳이 정규식을 돌리지 않고 바로 False 처리한다.
    return bool(_IDENT_RE.match(q.strip())) if q else False


def _format_rows(rows: list[dict[str, Any]]) -> str:
    """DB에서 받은 심볼 목록(dict 리스트)을 사람이 읽기 좋은 텍스트로 변환한다.

    Worker/Scout 에이전트에게 그대로 보여줄 최종 출력 문자열을 만든다.
    각 심볼은 한 줄 요약(종류·이름·시그니처·위치·유사도)으로 뽑고, docstring이
    있으면 그 첫 줄을 들여쓰기해 부가 설명으로 덧붙인다.

    매개변수:
        rows: 각 원소가 심볼 한 개를 나타내는 dict. kind/qualified_name/path/
              line_start/line_end 등의 키를 가지며, signature/similarity/
              docstring 은 없을 수도 있어 .get()으로 안전하게 접근한다.

    반환:
        줄바꿈으로 이어붙인 표시용 문자열. 결과가 비면 안내 문구를 반환한다.
    """
    # 조회 결과가 아예 없으면 빈 텍스트 대신 명시적 안내 문구를 돌려준다.
    if not rows:
        return "매칭된 심볼이 없습니다."
    lines: list[str] = []
    for r in rows:
        # 한 줄 헤더: "<종류> <정규화된 이름><시그니처>  (경로:시작줄-끝줄, sim=유사도)"
        # signature/similarity는 없을 수 있으므로 기본값(''/0)을 두어 KeyError를 피한다.
        head = (
            f"{r['kind']} {r['qualified_name']}{r.get('signature', '')}  "
            f"({r['path']}:{r['line_start']}-{r['line_end']}, sim={r.get('similarity', 0):.2f})"
        )
        lines.append(head)
        # docstring이 있으면 첫 줄만(그리고 160자까지만) 잘라 부연 설명으로 붙인다.
        # 여러 줄 docstring 전체를 쏟아내 출력이 길어지는 것을 막으려는 의도.
        doc = (r.get("docstring") or "").strip()
        if doc:
            first_line = doc.splitlines()[0][:160]
            lines.append(f"  → {first_line}")
    return "\n".join(lines)


class SymbolSearchTool(BaseTool):
    """프로젝트 심볼(함수/클래스/메서드) 검색 도구.

    BaseTool을 상속해 표준 도구 인터페이스(정체성/스키마/동작 플래그/수명주기)를
    구현한다. 실제 검색은 tb_symbols 인덱스를 감싼 SymbolStore에 위임하며, 이
    클래스는 "어떤 검색 모드를 어떤 순서로 시도할지"를 결정하는 오케스트레이션만
    담당한다.

    tb_symbols 인덱스에서 지원하는 조회:
      1) 이름/qualified_name 정확·부분 매칭 (pg_trgm 트라이그램)
      2) (옵션) 벡터 유사도 검색 — 의미 기반 (임베딩 프로바이더 필요)

    query는 Python 식별자 또는 자연어 문장 모두 허용한다. 검색에 필요한
    SymbolStore와 임베딩 프로바이더는 부트스트랩이 미리 넣어 둔 값을
    `context.options["symbol_store"]` / `context.options["model_provider"]`
    에서 꺼내 쓴다. (도구 자체는 DB/모델 연결을 직접 소유하지 않는다.)
    """

    # ═══ 1. Identity — 도구를 식별하는 이름/설명/그룹/별칭 ═══

    @property
    def name(self) -> str:
        # 레지스트리 등록 및 모델이 호출할 때 쓰는 공식 도구 이름.
        return "SymbolSearch"

    @property
    def description(self) -> str:
        # 모델(LLM)에게 전달되는 도구 설명. 영문으로 작성돼 있으며, "함수/클래스
        # 정의를 찾을 때 Grep/Glob보다 먼저 이 도구를 쓰라"는 사용 지침을 담는다.
        return (
            "Search indexed Python symbols (functions/classes/methods) by exact name, "
            "partial name (trigram), or natural-language meaning (vector). "
            "Returns file path + line range + signature + first docstring line. "
            "Use this BEFORE Grep/Glob when looking for a function or class definition."
        )

    @property
    def group(self) -> str:
        # 도구 분류. RAG(검색 증강) 계열 도구로 묶인다.
        return "rag"

    @property
    def aliases(self) -> list[str]:
        # 모델이 다른 이름으로 불러도 매칭되도록 하는 별칭 목록.
        return ["FindSymbol", "LookupSymbol"]

    # ═══ 2. Schema — 입력 파라미터를 정의하는 JSON Schema ═══

    @property
    def input_schema(self) -> dict[str, Any]:
        # query(필수), max_results(선택, 1~30), kind(선택 필터)를 정의한다.
        # 이 스키마는 모델에게 그대로 노출되어 도구 호출 인자 형식을 안내한다.
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Symbol name (e.g. 'query_loop'), qualified path "
                                   "(e.g. 'KnowledgeStore.add'), or natural-language "
                                   "description.",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Max results (default 10, cap 30)",
                    "default": 10,
                    "minimum": 1,
                    "maximum": 30,
                },
                "kind": {
                    "type": "string",
                    "description": "Filter by kind (function/class/method/async_function/"
                                   "async_method). Omit for all.",
                },
            },
            "required": ["query"],
        }

    # ═══ 3. Behavior Flags — 도구의 실행 특성(읽기전용/동시성) ═══

    @property
    def is_read_only(self) -> bool:
        # DB를 조회만 하고 아무것도 변경하지 않으므로 읽기 전용(True)이다.
        return True

    @property
    def is_concurrency_safe(self) -> bool:
        # 부작용이 없는 순수 조회라 병렬로 여러 개 실행해도 안전(True)하다.
        return True

    # ═══ 5. Lifecycle — 입력 검증 → 권한 확인 → 실행(call) ═══

    def validate_input(self, input_data: dict[str, Any]) -> str | None:
        """호출 전 입력을 가볍게 검증한다.

        문제가 있으면 사람이 읽을 오류 메시지(문자열)를, 정상이면 None을
        반환한다. 실제 검색 로직에 들어가기 전에 명백히 잘못된 입력을 걸러낸다.
        """
        q = (input_data.get("query") or "").strip()
        # query가 비어 있으면 검색 자체가 불가능하므로 거부한다.
        if not q:
            return "query는 비어 있을 수 없습니다."
        # 지나치게 긴 입력은 임베딩/DB 비용을 낭비하므로 500자로 상한을 둔다.
        if len(q) > 500:
            return "query는 500자 이하여야 합니다."
        return None

    async def check_permissions(
        self, input_data: dict[str, Any], context: ToolUseContext
    ) -> PermissionResult:
        # 읽기 전용 심볼 조회라 위험이 없으므로 항상 즉시 허용(ALLOW)한다.
        # 별도의 경로/명령어 검증이 필요 없는 안전한 도구다.
        return PermissionResult(behavior=PermissionBehavior.ALLOW)

    async def call(
        self, input_data: dict[str, Any], context: ToolUseContext
    ) -> ToolResult:
        """실제 심볼 검색을 수행하는 도구의 핵심 진입점.

        전체 흐름(3단계 폴백):
          1) 입력이 식별자 모양이면 이름 매칭(search_by_name)을 먼저 시도한다.
          2) 결과가 없거나 자연어면 임베딩→벡터 검색(search_by_vector)을 시도한다.
          3) 그래도 없으면 마지막으로 이름 부분매칭을 한 번 더 시도한다.
        이후 kind 필터를 적용하고, 결과를 표시용 텍스트로 만들어 반환한다.

        매개변수:
            input_data: query(필수) / max_results / kind 를 담은 도구 입력 dict.
            context: 실행 컨텍스트. options 에서 SymbolStore·임베딩 프로바이더를 꺼낸다.

        반환:
            ToolResult.success(포매팅된 텍스트) 또는 ToolResult.error(설정 누락 시).
        """
        query = input_data["query"].strip()
        # max_results는 기본 10, 최대 30으로 상한을 둔다. 0/None이면 10으로 보정.
        max_results = min(int(input_data.get("max_results", 10) or 10), 30)
        kind_filter = input_data.get("kind")

        # 실제 DB 조회를 담당하는 SymbolStore를 컨텍스트에서 꺼낸다.
        # 부트스트랩이 주입하지 않았다면 검색이 불가능하므로 오류로 안내한다.
        store = context.options.get("symbol_store")
        if store is None:
            return ToolResult.error(
                "SymbolStore가 context.options에 없습니다. bootstrap 연결 확인 필요."
            )

        results: list[dict[str, Any]] = []

        # 1) 이름 경로 우선 시도 (식별자 또는 qualified name일 때)
        #    이름을 정확히 아는 경우가 가장 흔하고 저렴하므로 맨 먼저 시도한다.
        #    실패해도 다음 단계로 넘어가야 하므로 예외는 경고 로그만 남기고 삼킨다.
        if _looks_like_identifier(query):
            try:
                results = await store.search_by_name(query, top_k=max_results)
            except Exception as e:
                logger.warning("SymbolSearch 이름 검색 실패: %s", e)

        # 2) 결과가 없거나(1단계 실패) 애초에 자연어면 벡터 의미 검색을 시도한다.
        #    임베딩 프로바이더가 있고 embed()를 지원할 때만 가능하다.
        if not results:
            provider = context.options.get("model_provider")
            if provider is not None and hasattr(provider, "embed"):
                try:
                    # 검색어를 임베딩 벡터로 변환한 뒤, 그 벡터로 유사 심볼을 찾는다.
                    vecs = await provider.embed([query])
                    if vecs and vecs[0]:
                        results = await store.search_by_vector(
                            embedding=vecs[0], top_k=max_results,
                        )
                except Exception as e:
                    # 벡터 검색은 보조 경로라 실패해도 치명적이지 않다 → debug로만 기록.
                    logger.debug("SymbolSearch 벡터 검색 실패 (무시): %s", e)

        # 3) 앞의 두 경로가 모두 빈손이면, 이름 부분매칭(trigram)을 마지막으로 시도한다.
        #    (예: 자연어처럼 보였지만 사실은 부분 이름이었던 경우를 건진다.)
        if not results:
            try:
                results = await store.search_by_name(query, top_k=max_results)
            except Exception as e:
                logger.debug("SymbolSearch 이름 폴백 검색 실패 (무시): %s", e)

        # kind 필터 (클라이언트 측에서 후처리)
        # DB가 아니라 여기서 걸러 준다. 예: function/class/method 등으로 좁히기.
        if kind_filter:
            results = [r for r in results if r.get("kind") == kind_filter]

        # 세 경로 모두 매칭이 없으면 실패가 아닌 "결과 없음" 성공으로 반환한다.
        # (에러가 아니라 정상적으로 0건이라는 뜻이므로 count=0으로 알린다.)
        if not results:
            return ToolResult.success(
                f"'{query}'에 매칭되는 심볼이 없습니다.", count=0,
            )

        # 상한(max_results)까지만 잘라 표시용 텍스트로 포매팅한 뒤 성공으로 반환한다.
        text = _format_rows(results[:max_results])
        logger.info("SymbolSearch: '%s' → %d", query[:60], len(results))
        return ToolResult.success(text, count=len(results))
