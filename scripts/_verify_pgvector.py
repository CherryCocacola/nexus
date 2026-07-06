"""pgvector 동적 라이브러리 상태 verify (2026-06-01).

이 스크립트는 PostgreSQL에 설치된 pgvector 확장이 정상 동작하는지
점검하는 "운영 진단(diagnostic)" 도구다. Nexus의 지식 RAG와 장기 메모리는
모두 pgvector의 벡터 연산(코사인/유클리드 거리)에 의존하기 때문에,
확장이 깨지면 검색 기능 전체가 멈춘다. 그래서 배포/복구 전후로 상태를
빠르게 확인할 수 있는 단독 실행 스크립트가 필요하다.

- 복구 전: pgvector가 어디서 깨졌는지(확장 미등록 / .so 누락 등) 확인
- 복구 후: 정상화 여부 + 벡터 테이블/인덱스 데이터 무결성까지 검증

이 파일은 읽기 전용(read-only) 진단이다. 어떤 경우에도 스키마나 데이터를
생성·수정·삭제하지 않으며, 오직 SELECT 조회만 수행한다.

주요 함수 구성:
    - section()                : 진단 섹션 제목을 예쁘게 출력하는 헬퍼
    - check_extension()        : PG 버전 + pgvector 확장 등록 상태 확인
    - check_basic()            : 벡터 캐스팅/거리 연산 3종이 되는지 확인
    - check_tables()           : 벡터 컬럼을 가진 테이블의 행 수·인덱스 확인
    - check_knowledge_search() : 임베딩 서버 → 실제 벡터 검색까지 end-to-end
    - main()                   : 위 점검들을 순서대로 실행하고 종료 코드 반환

사용:
    python scripts/_verify_pgvector.py
    python scripts/_verify_pgvector.py --search "양자역학"   # 실 KNOWLEDGE 검색까지

DB 접속 자격(호스트/포트/사용자/비밀번호)은 config/nexus_config.yaml의
postgresql 섹션과 동일한 값을 아래 PG_DSN 상수에 하드코딩해 두었다.

작성자: 이현수 / 작성일: 2026-07-05
"""

from __future__ import annotations

import argparse
import asyncio
import sys

import asyncpg  # PostgreSQL 비동기 드라이버 — asyncio 기반 커넥션 풀 사용

# 한글 출력이 콘솔에서 깨지지 않도록 표준출력 인코딩을 UTF-8로 강제한다.
# (특히 Windows 기본 콘솔은 cp949라서 한글/기호가 깨질 수 있음)
sys.stdout.reconfigure(encoding="utf-8")

# 접속할 PostgreSQL DSN. config/nexus_config.yaml의 postgresql 섹션과 동일한 값이다.
# 비밀번호의 '@'는 URL 인코딩되어 '%40'으로 표기됨(idino@12 → idino%4012).
PG_DSN = "postgresql://nexus:idino%4012@192.168.10.39:5440/nexus"

# pgvector의 vector 타입 컬럼을 가진, 점검 대상 테이블 목록.
# tb_knowledge=지식 RAG, tb_memories=장기 메모리, tb_symbols=코드 심볼 인덱스.
# 아래 check_tables()에서 이 목록을 순회하며 행 수와 인덱스 상태를 확인한다.
VECTOR_TABLES = ["tb_knowledge", "tb_memories", "tb_symbols"]


def section(title: str) -> None:
    """진단 섹션 제목을 한 줄로 출력하는 헬퍼.

    각 점검 단계를 '=== 제목 ===' 형태로 구분해 찍어서, 운영자가 콘솔 출력만
    훑어봐도 어느 단계에서 문제가 생겼는지 한눈에 알 수 있게 한다.

    매개변수:
        title: 섹션 제목 문자열.
    반환값: 없음(출력만 수행).
    """
    print(f"\n=== {title} ===")


async def check_basic(conn: asyncpg.Connection) -> dict:
    """가장 기본적인 vector 연산 3종을 시도해 pgvector 라이브러리가 로드되는지 판별한다.

    pgvector 확장이 pg_extension에 등록되어 있어도, 실제 동적 라이브러리(.so)가
    없으면 벡터 연산 시점에 에러가 난다. 그래서 "실제로 실행되는지"를 직접
    시도해 보는 것이 가장 확실한 판별법이다.

    3가지 케이스:
        - type_cast   : 문자열을 vector 타입으로 캐스팅
        - cosine_dist : 코사인 거리(<=>) 연산 — RAG 검색에서 실제 쓰는 연산자
        - l2_dist     : 유클리드(L2) 거리(<->) 연산

    실패 메시지에 '$libdir/vector'가 보이면 .so 파일 누락이 원인 —
    즉 PostgreSQL 컨테이너 이미지 자체의 문제다.

    매개변수:
        conn: 열려 있는 asyncpg 커넥션.
    반환값:
        {연산이름: 성공여부(bool)} 딕셔너리. 호출 측(main)은 이 값이 모두
        True인지로 pgvector 전체 정상 여부를 판단한다.
    """
    section("기본 vector 연산 점검")
    results: dict[str, bool] = {}

    # 점검할 SQL 3종. (이름, 실행할 SQL) 튜플 리스트로 정의해 순회한다.
    cases = [
        ("type_cast", "SELECT '[1,2,3]'::vector"),
        ("cosine_dist", "SELECT '[1,2,3]'::vector <=> '[1,2,4]'::vector"),
        ("l2_dist", "SELECT '[1,2,3]'::vector <-> '[1,2,4]'::vector"),
    ]
    for name, sql in cases:
        try:
            # 한 건만 조회하면 되므로 fetchval로 스칼라 결과를 받는다.
            r = await conn.fetchval(sql)
            print(f"  {name:14s} OK   ({r})")
            results[name] = True
        except Exception as e:
            # 예외 메시지가 여러 줄일 수 있어 첫 줄만 뽑아 간결히 보여준다.
            msg = str(e).splitlines()[0]
            print(f"  {name:14s} FAIL ({msg})")
            results[name] = False
    return results


async def check_extension(conn: asyncpg.Connection) -> None:
    """PostgreSQL 서버 버전과 pgvector 확장의 등록 상태를 확인한다.

    pg_extension 시스템 카탈로그를 조회해 'vector' 확장이 등록되어 있는지,
    등록됐다면 어떤 버전인지 출력한다. 여기서 미등록으로 나오면 애초에
    CREATE EXTENSION vector 부터 실행해야 한다는 뜻이다.

    매개변수:
        conn: 열려 있는 asyncpg 커넥션.
    반환값: 없음(출력만 수행).
    """
    section("PostgreSQL / pgvector 등록 상태")
    # 서버 버전 문자열(예: "PostgreSQL 17.x ...")을 그대로 출력한다.
    pg_ver = await conn.fetchval("SELECT version()")
    print(f"  PG: {pg_ver}")
    # pg_extension 카탈로그에서 'vector' 확장 등록 여부와 버전을 조회한다.
    ext = await conn.fetch(
        "SELECT extname, extversion FROM pg_extension WHERE extname='vector'"
    )
    if ext:
        # 등록되어 있으면 확장 이름과 버전을 출력한다.
        for r in ext:
            print(f"  extension: {r['extname']} v{r['extversion']} (pg_extension에 등록됨)")
    else:
        # 등록이 아예 안 된 상태 — 확장 설치부터 필요하다는 안내를 남긴다.
        print("  extension: vector (등록 안 됨) — CREATE EXTENSION 부터 필요")


async def check_tables(conn: asyncpg.Connection, vector_ok: bool) -> None:
    """벡터 컬럼이 있는 3개 테이블의 대략적 행 수와 인덱스 상태를 점검한다.

    VECTOR_TABLES에 정의된 테이블들을 순회하면서 각 테이블의 행 수와,
    벡터 검색용 인덱스(ivfflat/hnsw)가 걸려 있는지를 출력한다. 인덱스가
    없으면 검색이 풀스캔으로 느려지므로, 데이터 적재 후 인덱스 존재
    여부를 확인하는 것이 이 함수의 핵심 목적이다.

    매개변수:
        conn: 열려 있는 asyncpg 커넥션.
        vector_ok: check_basic()이 모두 통과했는지 여부. True일 때만
            정확한 COUNT(*)까지 시도한다(벡터 lib가 정상일 때만 안전).
    반환값: 없음(출력만 수행).
    """
    section("벡터 테이블 상태")
    for tab in VECTOR_TABLES:
        # 행 수는 정확한 COUNT(*) 대신 pg_class.reltuples(통계 기반 추정치)를
        # 쓴다. 벡터 라이브러리가 깨진 상황에서도 안전하게 조회되고, 대용량
        # 테이블에서도 즉시 반환되기 때문이다.
        approx = await conn.fetchval(
            "SELECT reltuples::bigint FROM pg_class WHERE relname = $1", tab
        )
        if approx is None:
            # pg_class에 해당 이름이 없으면 테이블 자체가 존재하지 않는 것.
            print(f"  {tab:14s} 테이블 없음")
            continue

        # 이 테이블에 걸린 인덱스의 이름과 접근 방식(amname)만 조회한다.
        # 인덱스 정의(indexdef) 원문은 건드리지 않고 메타데이터만 읽는다.
        idx = await conn.fetch(
            """
            SELECT i.relname AS name, am.amname AS method
            FROM pg_index x
            JOIN pg_class t ON t.oid = x.indrelid
            JOIN pg_class i ON i.oid = x.indexrelid
            JOIN pg_am am   ON am.oid = i.relam
            WHERE t.relname = $1
            ORDER BY i.relname
            """,
            tab,
        )
        # 인덱스 접근 방식 중 벡터 전용(ivfflat 또는 hnsw)이 하나라도 있는지.
        has_vec_idx = any(r["method"] in ("ivfflat", "hnsw") for r in idx)
        # 테이블별 요약 한 줄: 추정 행 수 + 벡터 인덱스 유무(O/X).
        print(f"  {tab:14s} approx={approx:>10,d}  vector_idx={'O' if has_vec_idx else 'X'}")
        # 이 테이블에 걸린 모든 인덱스를 이름/방식과 함께 나열한다.
        for r in idx:
            print(f"    - {r['name']:30s} {r['method']}")

        # 기본 벡터 연산이 모두 정상일 때만 정확한 COUNT(*)를 시도한다.
        # (COUNT는 vector 컬럼이 포함된 행도 스캔하므로, lib가 깨진 상태면
        #  괜히 에러를 낼 수 있어 vector_ok=True인 경우로 한정한다.)
        # SQL의 테이블명은 VECTOR_TABLES 화이트리스트에서만 오므로 인젝션
        # 위험이 없어 S608 경고를 억제한다.
        if vector_ok:
            try:
                exact = await conn.fetchval(
                    f"SELECT COUNT(*) FROM {tab}"  # noqa: S608 — 화이트리스트
                )
                print(f"    exact COUNT(*) = {exact:,d}")
            except Exception as e:
                print(f"    COUNT 실패: {e}")


async def check_knowledge_search(conn: asyncpg.Connection, query: str) -> None:
    """실제 지식 RAG 검색 경로를 그대로 흉내 내는 end-to-end 점검.

    운영 시 KNOWLEDGE 라우팅은 (1) 질의 문장을 임베딩 서버로 보내 벡터로
    바꾸고 (2) 그 벡터로 tb_knowledge를 코사인 거리 정렬해 상위 문서를
    가져온다. 이 함수는 그 흐름을 똑같이 재현해서, 임베딩 서버와 pgvector가
    함께 정상 동작하는지를 실제 검색 결과로 확인한다.

    통과 조건: 임베딩 서버(192.168.21.112:8002)가 살아 있고, pgvector 벡터
    검색도 정상이어야 한다. 둘 중 하나라도 죽어 있으면 실패로 표시된다.

    매개변수:
        conn: 열려 있는 asyncpg 커넥션.
        query: 검색할 자연어 질의 문자열(--search 인자로 전달됨).
    반환값: 없음(결과를 콘솔에 출력).
    """
    section(f"실 KNOWLEDGE 검색 — '{query}'")
    try:
        # httpx는 임베딩 호출에만 필요하다. 이 진단 스크립트 실행 환경에
        # 없을 수도 있으므로, 없으면 조용히 이 단계만 건너뛴다.
        import httpx
    except ImportError:
        print("  httpx 미설치 — 임베딩 호출 건너뜀")
        return

    # 임베딩 서버 호출 — nexus 본체와 동일한 커스텀 계약을 따른다.
    # 엔드포인트 /v1/embed 에 {"texts": [...]} 형태로 POST 한다.
    # (표준 OpenAI 임베딩 API가 아니라 자체 계약이라는 점에 주의)
    # 자세한 계약은 core/model/inference.py::ModelProvider.embed() 참조.
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                "http://192.168.21.112:8002/v1/embed",
                # 'query:' 접두사는 e5 계열 임베딩 모델의 질의용 규약이다.
                json={"texts": [f"query: {query}"]},
            )
            resp.raise_for_status()  # 4xx/5xx면 예외를 던져 아래 except로 보냄
            payload = resp.json()
            embedding = payload["embeddings"][0]  # 첫 번째 텍스트의 임베딩 벡터
        print(f"  임베딩 OK (dim={len(embedding)})")
    except Exception as e:
        # 서버 다운/타임아웃/응답 형식 오류 등 — 여기서 멈추고 원인 출력.
        print(f"  임베딩 서버 실패: {e}")
        return

    # 파이썬 float 리스트를 pgvector가 이해하는 '[0.1,0.2,...]' 문자열로 변환.
    # 소수점 6자리로 고정해 SQL 파라미터로 넘긴다.
    vec_str = "[" + ",".join(f"{x:.6f}" for x in embedding) + "]"
    try:
        # 코사인 거리(<=>) 오름차순 정렬로 가장 가까운 문서 5건을 가져온다.
        rows = await conn.fetch(
            """
            SELECT title, section, (embedding <=> $1::vector) AS dist
            FROM tb_knowledge
            ORDER BY embedding <=> $1::vector
            LIMIT 5
            """,
            vec_str,
        )
        for r in rows:
            # 코사인 거리(dist)를 유사도(sim)로 환산해 보여준다(1 - 거리).
            # 값이 1에 가까울수록 질의와 더 비슷한 문서라는 뜻이다.
            sim = 1.0 - float(r["dist"])
            print(f"  sim={sim:.3f} | {r['title']} {r['section'] or ''}")
        if not rows:
            # 검색은 성공했지만 테이블이 비어 결과가 없는 경우.
            print("  (결과 0건)")
    except Exception as e:
        # 벡터 검색 자체가 실패 — 인덱스/타입 문제 등일 수 있다.
        print(f"  벡터 검색 실패: {e}")


async def main(args: argparse.Namespace) -> int:
    """모든 점검을 순서대로 실행하고, 최종 판정에 따라 종료 코드를 반환한다.

    실행 순서는 "가벼운 것 → 무거운 것"이다. 먼저 확장 등록 상태를 보고,
    기본 벡터 연산을 시도한 뒤, 테이블/인덱스를 확인한다. --search가 주어진
    경우에만 임베딩 서버까지 부르는 실 검색을 마지막에 수행한다.

    매개변수:
        args: argparse로 파싱된 CLI 인자(현재는 args.search만 사용).
    반환값:
        프로세스 종료 코드. 0=pgvector 정상, 1=문제 있음. 호출 측에서
        sys.exit()로 그대로 넘겨 셸/CI가 성공·실패를 판단할 수 있게 한다.
    """
    # 진단용이라 커넥션은 1개면 충분하다(min=max=1). 명령 타임아웃 30초로
    # 서버가 응답 없이 멈춰도 스크립트가 무한 대기하지 않도록 한다.
    pool = await asyncpg.create_pool(PG_DSN, min_size=1, max_size=1, command_timeout=30)
    try:
        async with pool.acquire() as conn:
            # 1) 확장 등록 상태 → 2) 기본 벡터 연산 → 3) 테이블/인덱스 순으로 점검.
            await check_extension(conn)
            basic = await check_basic(conn)
            # basic이 모두 통과했을 때만 정확한 COUNT까지 시도하도록 전달한다.
            await check_tables(conn, vector_ok=all(basic.values()))
            if args.search:
                # 기본 연산이 전부 정상일 때만 실제 검색을 수행한다.
                if all(basic.values()):
                    await check_knowledge_search(conn, args.search)
                else:
                    # 벡터가 깨진 상태면 실 검색은 의미가 없으므로 건너뛴다.
                    section("실 KNOWLEDGE 검색 건너뜀")
                    print("  vector 기본 연산이 실패 — 복구 후 재실행")
    finally:
        # 예외가 나든 안 나든 커넥션 풀은 반드시 닫아 리소스를 정리한다.
        await pool.close()

    # 최종 판정: 기본 벡터 연산 3종이 전부 통과해야만 정상으로 본다.
    section("결론")
    if all(basic.values()):
        print("  ✓ pgvector 정상 — 운영 RAG/장기메모리 사용 가능")
        return 0
    # 하나라도 실패면 대개 컨테이너 이미지의 .so 누락이 원인이다.
    print("  ✗ pgvector .so 누락 — docutil-postgres 컨테이너 복구 필요")
    print("    → project_pgvector_outage.md 참조")
    return 1


# 스크립트로 직접 실행될 때만 CLI 인자를 파싱하고 이벤트 루프를 돌린다.
# (다른 모듈에서 import하는 경우엔 이 블록이 실행되지 않는다.)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="pgvector verify (2026-06-01)")
    # --search: 지정 시 임베딩 서버 호출 + 실제 벡터 검색까지 수행한다.
    parser.add_argument(
        "--search", type=str, default=None,
        help="실 KNOWLEDGE 검색까지 수행 (임베딩 서버 호출). 예: --search '양자역학'",
    )
    # asyncio.run으로 async main을 실행하고, 그 반환값을 프로세스 종료 코드로 넘긴다.
    sys.exit(asyncio.run(main(parser.parse_args())))
