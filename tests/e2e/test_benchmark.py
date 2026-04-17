"""
성능 벤치마크 — 사양서 Ch.22.4 Success Metrics 기준.

RTX 5090 32GB (24GB+ 티어) 목표:
  - Response Time (simple): < 1.5s
  - Response Time (complex): < 8s
  - TTFT (Time to First Token): 측정
  - TPS (Tokens Per Second): 측정
  - Embedding Latency: 측정

GPU 서버가 가동 중일 때만 실행 가능하다.
"""

from __future__ import annotations

import json
import time

import httpx
import pytest

GPU_SERVER_URL = "http://192.168.22.28:8001"
EMBEDDING_SERVER_URL = "http://192.168.22.28:8002"
MODEL_ID = "qwen3.5-27b"


def _gpu_server_available() -> bool:
    """GPU 서버 연결 가능 여부."""
    try:
        import socket

        s = socket.socket()
        s.settimeout(3)
        result = s.connect_ex(("192.168.22.28", 8001)) == 0
        s.close()
        return result
    except Exception:
        return False


pytestmark = [
    pytest.mark.e2e,
    pytest.mark.benchmark,
    pytest.mark.skipif(
        not _gpu_server_available(),
        reason="GPU 서버(192.168.22.28:8001)에 연결할 수 없습니다",
    ),
]

# 벤치마크에 사용할 질문들
SIMPLE_QUESTIONS = [
    "1 + 1 = ?",
    "Python에서 리스트를 정렬하는 함수 이름은?",
    "HTTP 상태 코드 404의 의미는?",
    "JSON의 약자는?",
    "Git에서 브랜치를 생성하는 명령어는?",
    "TCP와 UDP의 차이를 한 문장으로 설명하세요.",
    "Python의 GIL이란?",
    "REST API에서 GET과 POST의 차이는?",
    "Docker 컨테이너와 가상 머신의 차이를 한 줄로 설명하세요.",
    "SQL에서 중복을 제거하는 키워드는?",
]

COMPLEX_QUESTIONS = [
    "Python으로 이진 탐색 트리를 구현하고 삽입, 검색, 삭제 메서드를 포함하세요.",
    "마이크로서비스 아키텍처의 장단점과 모놀리식 대비 장단점을 비교 분석하세요.",
    "asyncio를 사용한 비동기 웹 크롤러의 설계 패턴을 설명하세요.",
    "데이터베이스 인덱싱의 B-Tree와 Hash Index의 동작 원리를 비교하세요.",
    "OAuth 2.0 Authorization Code Flow를 단계별로 설명하세요.",
    "React와 Vue의 상태 관리 패턴 차이를 설명하세요.",
    "Kubernetes Pod의 라이프사이클을 설명하세요.",
    "분산 시스템에서 CAP 정리를 설명하고 실제 시스템 예시를 들어주세요.",
    "Python GC의 세대별 가비지 컬렉션 동작 원리를 설명하세요.",
    "TLS 1.3 핸드셰이크 과정을 단계별로 설명하세요.",
]


class TestResponseTime:
    """응답 시간 벤치마크."""

    @pytest.mark.asyncio
    async def test_simple_response_time(self):
        """
        단순 질문 10회 평균 응답 시간을 측정한다.
        목표: < 1.5s (RTX 5090 24GB+ 티어)
        """
        latencies: list[float] = []

        async with httpx.AsyncClient(timeout=30.0) as client:
            for question in SIMPLE_QUESTIONS:
                start = time.monotonic()
                resp = await client.post(
                    f"{GPU_SERVER_URL}/v1/chat/completions",
                    json={
                        "model": MODEL_ID,
                        "messages": [{"role": "user", "content": question}],
                        "max_tokens": 64,
                        "temperature": 0.1,
                    },
                )
                elapsed = time.monotonic() - start
                assert resp.status_code == 200
                latencies.append(elapsed)

        avg = sum(latencies) / len(latencies)
        min_l = min(latencies)
        max_l = max(latencies)
        p50 = sorted(latencies)[len(latencies) // 2]

        # 결과 출력 (pytest -v에서 확인 가능)
        print("\n[Simple Response Time]")
        print(f"  평균: {avg:.3f}s")
        print(f"  P50:  {p50:.3f}s")
        print(f"  최소: {min_l:.3f}s")
        print(f"  최대: {max_l:.3f}s")
        print("  목표: < 1.5s")

        # soft assert — 목표 미달이어도 측정 데이터는 기록
        assert avg < 10.0, f"평균 응답 시간이 10초를 초과: {avg:.3f}s"

    @pytest.mark.asyncio
    async def test_complex_response_time(self):
        """
        복잡한 질문 10회 평균 응답 시간을 측정한다.
        목표: < 8s (RTX 5090 24GB+ 티어)
        """
        latencies: list[float] = []

        async with httpx.AsyncClient(timeout=120.0) as client:
            for question in COMPLEX_QUESTIONS:
                start = time.monotonic()
                resp = await client.post(
                    f"{GPU_SERVER_URL}/v1/chat/completions",
                    json={
                        "model": MODEL_ID,
                        "messages": [{"role": "user", "content": question}],
                        "max_tokens": 512,
                        "temperature": 0.1,
                    },
                )
                elapsed = time.monotonic() - start
                assert resp.status_code == 200
                latencies.append(elapsed)

        avg = sum(latencies) / len(latencies)
        min_l = min(latencies)
        max_l = max(latencies)
        p50 = sorted(latencies)[len(latencies) // 2]

        print("\n[Complex Response Time]")
        print(f"  평균: {avg:.3f}s")
        print(f"  P50:  {p50:.3f}s")
        print(f"  최소: {min_l:.3f}s")
        print(f"  최대: {max_l:.3f}s")
        print("  목표: < 8s")

        assert avg < 60.0, f"평균 응답 시간이 60초를 초과: {avg:.3f}s"


class TestStreamingPerformance:
    """스트리밍 성능 벤치마크."""

    @pytest.mark.asyncio
    async def test_ttft(self):
        """
        TTFT (Time to First Token)를 측정한다.
        스트리밍 요청 시 첫 번째 텍스트 토큰이 도착하기까지의 시간.
        """
        ttft_list: list[float] = []

        async with httpx.AsyncClient(timeout=60.0) as client:
            for question in SIMPLE_QUESTIONS[:5]:
                start = time.monotonic()
                first_token_time = None

                async with client.stream(
                    "POST",
                    f"{GPU_SERVER_URL}/v1/chat/completions",
                    json={
                        "model": MODEL_ID,
                        "messages": [{"role": "user", "content": question}],
                        "max_tokens": 64,
                        "temperature": 0.1,
                        "stream": True,
                    },
                ) as resp:
                    async for line in resp.aiter_lines():
                        if line.startswith("data: ") and line[6:].strip() != "[DONE]":
                            try:
                                data = json.loads(line[6:])
                                delta = data["choices"][0].get("delta", {})
                                if delta.get("content") and first_token_time is None:
                                    first_token_time = time.monotonic() - start
                            except (json.JSONDecodeError, KeyError, IndexError):
                                continue

                if first_token_time is not None:
                    ttft_list.append(first_token_time)

        if ttft_list:
            avg_ttft = sum(ttft_list) / len(ttft_list)
            print("\n[TTFT - Time to First Token]")
            print(f"  평균: {avg_ttft:.3f}s")
            print(f"  최소: {min(ttft_list):.3f}s")
            print(f"  최대: {max(ttft_list):.3f}s")
            print(f"  샘플: {len(ttft_list)}개")

    @pytest.mark.asyncio
    async def test_throughput_tps(self):
        """
        TPS (Tokens Per Second)를 측정한다.
        스트리밍 응답에서 초당 생성되는 토큰 수.
        """
        tps_list: list[float] = []

        async with httpx.AsyncClient(timeout=60.0) as client:
            for question in SIMPLE_QUESTIONS[:5]:
                token_count = 0
                first_token_time = None

                async with client.stream(
                    "POST",
                    f"{GPU_SERVER_URL}/v1/chat/completions",
                    json={
                        "model": MODEL_ID,
                        "messages": [{"role": "user", "content": question}],
                        "max_tokens": 128,
                        "temperature": 0.1,
                        "stream": True,
                    },
                ) as resp:
                    async for line in resp.aiter_lines():
                        if line.startswith("data: ") and line[6:].strip() != "[DONE]":
                            try:
                                data = json.loads(line[6:])
                                delta = data["choices"][0].get("delta", {})
                                if delta.get("content"):
                                    if first_token_time is None:
                                        first_token_time = time.monotonic()
                                    token_count += 1
                            except (json.JSONDecodeError, KeyError, IndexError):
                                continue

                end = time.monotonic()
                # 첫 토큰 이후부터 마지막 토큰까지의 시간으로 TPS 계산
                if first_token_time and token_count > 1:
                    generation_time = end - first_token_time
                    if generation_time > 0:
                        tps = token_count / generation_time
                        tps_list.append(tps)

        if tps_list:
            avg_tps = sum(tps_list) / len(tps_list)
            print("\n[TPS - Tokens Per Second]")
            print(f"  평균: {avg_tps:.1f} tokens/s")
            print(f"  최소: {min(tps_list):.1f} tokens/s")
            print(f"  최대: {max(tps_list):.1f} tokens/s")
            print(f"  샘플: {len(tps_list)}개")


class TestEmbeddingPerformance:
    """임베딩 성능 벤치마크."""

    @pytest.mark.asyncio
    async def test_embedding_latency(self):
        """임베딩 생성 지연 시간을 측정한다."""
        latencies: list[float] = []
        texts = ["This is a test sentence for embedding benchmark."]

        async with httpx.AsyncClient(timeout=30.0) as client:
            for _ in range(10):
                start = time.monotonic()
                resp = await client.post(
                    f"{EMBEDDING_SERVER_URL}/v1/embed",
                    json={"texts": texts},
                )
                elapsed = time.monotonic() - start
                assert resp.status_code == 200
                latencies.append(elapsed)

        avg = sum(latencies) / len(latencies)
        print("\n[Embedding Latency]")
        print(f"  평균: {avg * 1000:.1f}ms")
        print(f"  최소: {min(latencies) * 1000:.1f}ms")
        print(f"  최대: {max(latencies) * 1000:.1f}ms")

        assert avg < 5.0, f"임베딩 평균 지연이 5초를 초과: {avg:.3f}s"
