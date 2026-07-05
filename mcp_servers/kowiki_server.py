"""
kowiki MCP 서버 — tb_knowledge(한국어 위키 등) 벡터 검색을 도구로 노출하는 파일.

■ 이 파일이 하는 일 (한눈에)
  Nexus는 여러 MCP(Model Context Protocol) 서버를 두고, 각 서버가 "도구(tool)"를
  하나 이상 노출한다. 이 파일은 그중 하나로, 적재된 지식 베이스(PostgreSQL의
  tb_knowledge 테이블 — 한국어 위키 덤프 등)를 "의미 기반(벡터) 검색"할 수 있게
  해 주는 search 도구를 FastAPI 앱 형태로 만들어 제공한다.
  즉, 사용자의 자연어 질문을 임베딩(숫자 벡터)으로 바꾼 뒤, 미리 임베딩해 둔
  문서 청크들과 코사인 유사도를 비교해 가장 비슷한 상위 청크들을 돌려준다.

■ 주요 구성 요소
  - KowikiSearchTool : 실제 검색 로직을 담은 도구 클래스(name="search").
  - build_app()      : 설정을 읽어 임베딩 프로바이더 + 지식 저장소를 조립하고,
                       위 도구를 등록한 FastAPI 앱을 반환하는 팩토리 함수.

■ 노출 도구
  search(query, top_k?)
    - query : 검색할 자연어 질의(필수).
    - top_k : 반환할 결과 개수(선택, 기본 5, 최대 50).
    질의를 임베딩한 뒤 KnowledgeStore.search_by_vector 로 코사인 유사도 검색을
    수행하고 상위 결과(content/제목/섹션/유사도점수/메타데이터)를 반환한다.

■ 처리 흐름 (core 모듈을 그대로 재사용)
  ModelProvider.embed(["query: " + 질의]) → 1024차원 임베딩 벡터
    → KnowledgeStore.search_by_vector(embedding, top_k)
    → [{source, title, section, content, similarity, metadata, tags}, ...]

■ 왜 "query:" 접두사를 붙이는가
  임베딩 모델 e5-large(multilingual)의 권장 규약이다. 검색용 질의에는 "query:",
  적재해 두는 문서 청크에는 "passage:" 접두사를 붙이면 두 종류의 텍스트를
  모델이 구분해 임베딩 품질(검색 정확도)이 올라간다.
  (같은 규약을 쓰는 곳: core/model/inference.py, RAG pipeline.py 참고.)

■ read-only(읽기 전용) 서버
  이 서버는 "검색"만 제공한다. 지식을 새로 넣거나 수정하는 적재 기능은 별도의
  docingest 서버가 담당한다. 여기서는 DB를 절대 쓰지(write) 않는다.

■ 에어갭(폐쇄망) 준수
  임베딩 서버 URL과 DB 접속 정보는 모두 core/config 에서 LAN 주소로만 로드한다.
  이 파일에는 외부 인터넷을 호출하는 코드가 없다.

작성자: 이현수 / 작성일: 2026-07-05
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI

from mcp_servers.framework import McpServerTool, create_mcp_app

# 이 모듈 전용 로거. "nexus.mcp_servers.kowiki" 네임스페이스로 로그를 남겨
# 나중에 로그에서 이 서버의 메시지만 골라 볼 수 있게 한다.
logger = logging.getLogger("nexus.mcp_servers.kowiki")

# 검색 질의 앞에 붙이는 임베딩 접두사(e5-large 규약). 위 모듈 docstring의
# "왜 query: 접두사인가" 설명 참고. 반드시 뒤에 공백까지 포함해 "query: " 이다.
_QUERY_PREFIX = "query: "

# top_k(반환 결과 개수)의 기본값과 상한.
# - 기본값: 사용자가 top_k 를 지정하지 않았을 때 쓰는 값.
# - 상한: 너무 많은 결과를 돌려주면 LLM 컨텍스트 토큰을 낭비하므로 최대 50개로 제한.
_DEFAULT_TOP_K = 5
_MAX_TOP_K = 50


class KowikiSearchTool(McpServerTool):
    """
    tb_knowledge 벡터 검색 도구. MCP 프레임워크의 McpServerTool 을 상속한다.

    이 도구는 스스로 임베딩 모델이나 DB를 만들지 않고, 생성 시점에 "임베딩
    프로바이더"와 "지식 저장소"를 밖에서 주입(의존성 주입)받아 사용한다.
    이렇게 하면 실제 구현(core의 LocalModelProvider/KnowledgeStore)을 그대로
    재사용하면서, 테스트 때는 가짜 객체를 끼워 넣기도 쉬워진다.

    한 번의 검색은 call() 안에서 (1) 질의 임베딩 → (2) 벡터 검색 두 단계로
    이뤄진다. 자세한 규약(name/description/input_schema)은 아래 프로퍼티 참고.
    """

    def __init__(self, model_provider: Any, knowledge_store: Any) -> None:
        """
        도구를 생성하면서 필요한 두 협력 객체를 주입받아 보관한다.

        Args:
            model_provider: embed(texts) -> list[list[float]] 를 제공하는
                임베딩 프로바이더. 질의 텍스트를 벡터로 바꾸는 데 쓴다.
            knowledge_store: search_by_vector(...) 를 제공하는 KnowledgeStore.
                임베딩 벡터로 tb_knowledge 에서 유사 청크를 찾는 데 쓴다.
        """
        # 밑줄(_) 접두사는 "외부에서 직접 건드리지 말라"는 내부 필드 관례.
        self._model = model_provider
        self._store = knowledge_store

    @property
    def name(self) -> str:
        # MCP 클라이언트/LLM이 이 도구를 호출할 때 쓰는 이름. 이 서버 안에서
        # 유일해야 한다. 여기서는 단순히 "search".
        return "search"

    @property
    def description(self) -> str:
        # LLM이 "언제 이 도구를 써야 하는지" 판단하는 근거가 되는 설명문.
        # 어떤 데이터를 검색하고 무엇을 돌려주는지 명확히 적어 준다.
        return (
            "한국어 위키 등 적재된 지식 베이스(tb_knowledge)에서 의미 기반 검색을 수행한다. "
            "질의를 임베딩해 코사인 유사도가 높은 상위 청크를 반환한다(content/제목/섹션/점수)."
        )

    @property
    def input_schema(self) -> dict[str, Any]:
        # 이 도구가 받는 인자를 JSON Schema 로 선언한다. LLM은 이 스키마를 보고
        # 어떤 인자를 어떤 타입으로 넘겨야 하는지 알아낸다. query 는 필수,
        # top_k 는 선택(생략 시 기본값 적용)이다.
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "검색할 자연어 질의.",
                },
                "top_k": {
                    "type": "integer",
                    "description": f"반환할 결과 수(기본 {_DEFAULT_TOP_K}, 최대 {_MAX_TOP_K}).",
                },
            },
            "required": ["query"],
        }

    async def call(self, arguments: dict[str, Any]) -> Any:
        """
        도구의 실제 실행부. 질의를 임베딩해 벡터 검색을 수행하고 결과를 돌려준다.

        MCP 프레임워크가 클라이언트 요청을 받으면 이 메서드를 호출하며, 인자는
        input_schema 에 맞춘 dict(arguments) 형태로 전달된다. 처리는 크게
        (1) 입력 검증 → (2) 질의 임베딩 → (3) 벡터 검색 → (4) 결과 조립 순서다.

        Args:
            arguments: {"query": str, "top_k"?: int} 형태의 호출 인자 딕셔너리.

        Returns:
            {
              "query": str,       # 원본 질의(접두사 없는 사용자 입력 그대로)
              "count": int,       # 반환한 결과 개수
              "results": [
                {"title","section","content","source","similarity","tags","metadata"},
                ...
              ],
            }

        예외:
            query 누락/빈 값, top_k 가 1 미만 → ValueError (MCP 에러코드 -32602,
                즉 "잘못된 인자"에 대응).
            임베딩 또는 벡터 검색 실패 → RuntimeError (MCP 에러코드 -32603,
                즉 "내부 오류"에 대응).
        """
        # 1) query 검증 — 문자열이 아니거나 공백만 있으면 잘못된 입력으로 거부한다.
        query = arguments.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("필수 인자 'query'(비어 있지 않은 문자열)가 없습니다.")

        # 2) top_k 검증 — 미지정일 때만 기본값을 쓴다.
        #   주의: `arguments.get("top_k") or _DEFAULT_TOP_K` 같은 `or` 관용구는
        #   0 을 falsy 로 보고 조용히 기본값으로 바꿔 버린다. 하지만 top_k=0/음수는
        #   "기본값으로 대체할 값"이 아니라 "잘못된 입력"이므로 거부해야 한다.
        #   그래서 get(키, 기본값) 으로 값을 받은 뒤 아래에서 명시적으로 검증한다.
        top_k = arguments.get("top_k", _DEFAULT_TOP_K)
        #   isinstance(top_k, bool) 도 막는 이유: 파이썬에서 bool 은 int 의 하위
        #   타입이라 True/False 가 정수 검사(isinstance int)를 통과해 버린다.
        #   top_k 로 True(=1) 같은 값이 들어오는 것을 방지한다.
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
            raise ValueError("top_k 는 1 이상의 정수여야 합니다.")
        #   상한 적용 — 너무 큰 값이 와도 _MAX_TOP_K 로 잘라 토큰 낭비를 막는다.
        top_k = min(top_k, _MAX_TOP_K)

        # 3) 질의 임베딩 — 앞에 "query:" 접두사를 붙여 e5 규약을 따른다.
        #    embed 는 리스트를 받아 리스트를 돌려주므로, 질의 하나만 담아 호출한다.
        try:
            embeddings = await self._model.embed([f"{_QUERY_PREFIX}{query}"])
        except Exception as e:  # noqa: BLE001 — embed 는 httpx 등 다양한 예외 → RuntimeError 로 정규화
            #   임베딩 서버 호출은 httpx 타임아웃 등 여러 예외를 낼 수 있다.
            #   호출자가 처리하기 쉽도록 하나의 RuntimeError 로 정규화한다.
            raise RuntimeError(f"임베딩 실패: {type(e).__name__}: {e}") from e

        #   결과가 비었거나 첫 벡터가 비면 검색을 진행할 수 없으니 오류로 처리.
        if not embeddings or not embeddings[0]:
            raise RuntimeError("임베딩 서버가 빈 결과를 반환했습니다.")

        # 4) 벡터 검색 — KnowledgeStore.search_by_vector 를 그대로 재사용한다.
        #    embeddings[0] 이 방금 만든 질의 벡터. top_k 개의 유사 청크를 받는다.
        try:
            results = await self._store.search_by_vector(embeddings[0], top_k=top_k)
        except Exception as e:  # noqa: BLE001 — asyncpg 등 예외 → RuntimeError 로 정규화
            #   DB 검색도 asyncpg 등 여러 예외를 낼 수 있어 RuntimeError 로 정규화.
            raise RuntimeError(f"벡터 검색 실패: {type(e).__name__}: {e}") from e

        # 5) 결과 조립 — 원본 질의, 개수, 검색 결과 목록을 묶어 반환한다.
        return {"query": query, "count": len(results), "results": results}


async def build_app(api_key: str = "local-key") -> FastAPI:
    """
    kowiki MCP 서버의 FastAPI 앱을 조립해서 반환하는 팩토리(조립) 함수.

    서버 부팅 시 한 번 호출되어, 검색 도구가 동작하는 데 필요한 협력 객체들을
    설정에서 읽어 구성하고, 그것을 KowikiSearchTool 에 주입한 뒤 앱으로 감싼다.

    Args:
        api_key: MCP 클라이언트 인증에 쓰는 키. 폐쇄망 기본값은 "local-key".

    동작 순서:
      1) core/config 로드 → 임베딩 서버 URL, PostgreSQL 접속 정보 확보.
      2) LocalModelProvider(임베딩 전용) + KnowledgeStore(pg_pool) 구성.
         pg_pool 연결에 실패하면 KnowledgeStore 가 인메모리 폴백으로 동작한다
         (fail-soft: DB가 없어도 서버 자체는 뜬다).
      3) KowikiSearchTool 을 등록한 FastAPI 앱을 반환.

    Returns:
        FastAPI: 검색 도구가 붙은, 바로 실행 가능한 MCP 서버 앱.
    """
    # 지연(lazy) import: 이 함수가 실제로 호출될 때만 core 모듈을 불러온다.
    # 무거운 의존성을 모듈 로드 시점이 아니라 앱 조립 시점으로 미뤄 두는 것.
    from core.config import load_and_validate_config
    from core.model.inference import LocalModelProvider
    from core.rag.knowledge_store import KnowledgeStore

    # 1) 설정 로드 및 검증 — GPU/임베딩 서버 URL, PG 접속 정보 등을 담고 있다.
    config = load_and_validate_config()

    # 2) 임베딩 전용 프로바이더 생성.
    #    이 서버는 텍스트 생성(stream())은 쓰지 않고 embed() 만 사용하지만,
    #    LocalModelProvider 생성자는 생성용/임베딩용 설정을 함께 받으므로
    #    둘 다 넘겨 준다.
    model_provider = LocalModelProvider(
        base_url=config.gpu_server.url,
        api_key=config.scout.api_key,
        model_id=config.model.primary_model,
        embedding_model_id=config.model.embedding_model,
        embedding_base_url=config.gpu_server.embedding_url,
    )

    # 3) PostgreSQL 커넥션 풀 생성 — 실패해도 서버는 떠야 하므로 예외를 흡수한다.
    #    성공하면 풀 객체, 실패하면 None 이 되고, None 이면 KnowledgeStore 가
    #    인메모리 폴백으로 동작한다(검색 결과는 비지만 서버는 살아 있음).
    pg_pool: Any | None = None
    try:
        # asyncpg 도 지연 import — 미설치 환경에서도 이 파일 자체는 로드되도록.
        import asyncpg

        # 풀 크기(min/max)와 연결 타임아웃은 가벼운 검색용으로 작게 잡았다.
        pg_pool = await asyncpg.create_pool(
            host=config.postgresql.host,
            port=config.postgresql.port,
            database=config.postgresql.database,
            user=config.postgresql.user,
            password=config.postgresql.password,
            min_size=1,
            max_size=4,
            timeout=10.0,
        )
        logger.info("kowiki MCP 서버: PostgreSQL 연결 성공")
    except ImportError:
        # asyncpg 자체가 설치돼 있지 않은 경우 — 폴백으로 계속 진행.
        logger.warning("kowiki MCP 서버: asyncpg 미설치 — 인메모리 폴백")
    except Exception as e:  # noqa: BLE001 — 연결 실패는 인메모리 폴백으로 흡수
        # DB 주소 오류/타임아웃 등 어떤 연결 실패든 서버를 죽이지 않고 폴백한다.
        logger.warning("kowiki MCP 서버: PostgreSQL 연결 실패(인메모리 폴백): %s", e)

    # 4) 지식 저장소 생성. pg_pool 이 None 이면 내부적으로 인메모리 모드로 뜬다.
    knowledge_store = KnowledgeStore(pg_pool=pg_pool)

    # 서버 종료 시 호출될 정리(cleanup) 훅. 아래 create_mcp_app 의
    # shutdown_hooks 로 등록되며, 열어 둔 리소스를 깔끔히 닫아 누수를 막는다.
    async def _cleanup() -> None:
        # 임베딩 프로바이더가 내부에 들고 있는 httpx 클라이언트를 닫는다.
        await model_provider.close()
        # PG 풀이 실제로 열려 있을 때만 닫는다(폴백이면 None 이므로 건너뜀).
        if pg_pool is not None:
            await pg_pool.close()
        logger.info("kowiki MCP 서버: 리소스 정리 완료")

    # 5) 도구를 등록한 MCP FastAPI 앱을 만들어 반환한다.
    return create_mcp_app(
        tools=[KowikiSearchTool(model_provider, knowledge_store)],
        api_key=api_key,
        title="Nexus Kowiki RAG MCP Server",
        shutdown_hooks=[_cleanup],
    )
