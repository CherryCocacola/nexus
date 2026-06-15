"""pgvector 동적 라이브러리 상태 verify (2026-06-01).

복구 전: 어디서 깨졌는지 확인
복구 후: 정상화 확인 + 데이터 무결성 검증

읽기 전용 — 스키마/데이터 변경 없음.

사용:
    python scripts/_verify_pgvector.py
    python scripts/_verify_pgvector.py --search "양자역학"   # 실 KNOWLEDGE 검색까지

자격은 config/nexus_config.yaml의 postgresql 섹션을 따른다.
"""

from __future__ import annotations

import argparse
import asyncio
import sys

import asyncpg

sys.stdout.reconfigure(encoding="utf-8")

# config/nexus_config.yaml의 postgresql 섹션과 동일
PG_DSN = "postgresql://nexus:idino%4012@192.168.10.39:5440/nexus"

# 점검할 vector 컬럼을 가진 테이블
VECTOR_TABLES = ["tb_knowledge", "tb_memories", "tb_symbols"]


def section(title: str) -> None:
    """진단 섹션 헤더 출력 — 운영자가 한눈에 어디가 깨졌는지 보게 한다."""
    print(f"\n=== {title} ===")


async def check_basic(conn: asyncpg.Connection) -> dict:
    """가장 기본적인 vector 연산 3종을 시도해 라이브러리 로드 가능 여부를 가른다.

    실패 메시지에 '$libdir/vector'가 보이면 .so 누락 — 컨테이너 이미지 문제.
    """
    section("기본 vector 연산 점검")
    results: dict[str, bool] = {}

    cases = [
        ("type_cast", "SELECT '[1,2,3]'::vector"),
        ("cosine_dist", "SELECT '[1,2,3]'::vector <=> '[1,2,4]'::vector"),
        ("l2_dist", "SELECT '[1,2,3]'::vector <-> '[1,2,4]'::vector"),
    ]
    for name, sql in cases:
        try:
            r = await conn.fetchval(sql)
            print(f"  {name:14s} OK   ({r})")
            results[name] = True
        except Exception as e:
            msg = str(e).splitlines()[0]
            print(f"  {name:14s} FAIL ({msg})")
            results[name] = False
    return results


async def check_extension(conn: asyncpg.Connection) -> None:
    """pg_extension 등록 상태와 PG 버전 확인."""
    section("PostgreSQL / pgvector 등록 상태")
    pg_ver = await conn.fetchval("SELECT version()")
    print(f"  PG: {pg_ver}")
    ext = await conn.fetch(
        "SELECT extname, extversion FROM pg_extension WHERE extname='vector'"
    )
    if ext:
        for r in ext:
            print(f"  extension: {r['extname']} v{r['extversion']} (pg_extension에 등록됨)")
    else:
        print("  extension: vector (등록 안 됨) — CREATE EXTENSION 부터 필요")


async def check_tables(conn: asyncpg.Connection, vector_ok: bool) -> None:
    """벡터 컬럼이 있는 3개 테이블의 행 수와 인덱스 상태."""
    section("벡터 테이블 상태")
    for tab in VECTOR_TABLES:
        # 행 수: vector lib 안 거치도록 reltuples 사용
        approx = await conn.fetchval(
            "SELECT reltuples::bigint FROM pg_class WHERE relname = $1", tab
        )
        if approx is None:
            print(f"  {tab:14s} 테이블 없음")
            continue

        # 인덱스 메타데이터 (이름/방식만 — indexdef 안 건드림)
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
        has_vec_idx = any(r["method"] in ("ivfflat", "hnsw") for r in idx)
        print(f"  {tab:14s} approx={approx:>10,d}  vector_idx={'O' if has_vec_idx else 'X'}")
        for r in idx:
            print(f"    - {r['name']:30s} {r['method']}")

        # vector_ok=True면 실제 COUNT(*)도 가능 (vector 컬럼 포함 행이라도)
        if vector_ok:
            try:
                exact = await conn.fetchval(
                    f"SELECT COUNT(*) FROM {tab}"  # noqa: S608 — 화이트리스트
                )
                print(f"    exact COUNT(*) = {exact:,d}")
            except Exception as e:
                print(f"    COUNT 실패: {e}")


async def check_knowledge_search(conn: asyncpg.Connection, query: str) -> None:
    """KNOWLEDGE 라우팅을 모사한 실 검색 — 임베딩 서버 → 벡터 검색.

    임베딩 서버(192.168.21.112:8002)가 살아있고 pgvector도 정상이어야 통과.
    """
    section(f"실 KNOWLEDGE 검색 — '{query}'")
    try:
        import httpx
    except ImportError:
        print("  httpx 미설치 — 임베딩 호출 건너뜀")
        return

    # 임베딩 서버 호출 — nexus 본체와 동일한 커스텀 계약(/v1/embed + {"texts":[]})
    # core/model/inference.py::ModelProvider.embed() 참조
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                "http://192.168.21.112:8002/v1/embed",
                json={"texts": [f"query: {query}"]},
            )
            resp.raise_for_status()
            payload = resp.json()
            embedding = payload["embeddings"][0]
        print(f"  임베딩 OK (dim={len(embedding)})")
    except Exception as e:
        print(f"  임베딩 서버 실패: {e}")
        return

    vec_str = "[" + ",".join(f"{x:.6f}" for x in embedding) + "]"
    try:
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
            sim = 1.0 - float(r["dist"])
            print(f"  sim={sim:.3f} | {r['title']} {r['section'] or ''}")
        if not rows:
            print("  (결과 0건)")
    except Exception as e:
        print(f"  벡터 검색 실패: {e}")


async def main(args: argparse.Namespace) -> int:
    pool = await asyncpg.create_pool(PG_DSN, min_size=1, max_size=1, command_timeout=30)
    try:
        async with pool.acquire() as conn:
            await check_extension(conn)
            basic = await check_basic(conn)
            await check_tables(conn, vector_ok=all(basic.values()))
            if args.search:
                if all(basic.values()):
                    await check_knowledge_search(conn, args.search)
                else:
                    section("실 KNOWLEDGE 검색 건너뜀")
                    print("  vector 기본 연산이 실패 — 복구 후 재실행")
    finally:
        await pool.close()

    section("결론")
    if all(basic.values()):
        print("  ✓ pgvector 정상 — 운영 RAG/장기메모리 사용 가능")
        return 0
    print("  ✗ pgvector .so 누락 — docutil-postgres 컨테이너 복구 필요")
    print("    → project_pgvector_outage.md 참조")
    return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="pgvector verify (2026-06-01)")
    parser.add_argument(
        "--search", type=str, default=None,
        help="실 KNOWLEDGE 검색까지 수행 (임베딩 서버 호출). 예: --search '양자역학'",
    )
    sys.exit(asyncio.run(main(parser.parse_args())))
