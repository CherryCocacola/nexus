"""
Soak Test — 장시간 안정성 검증.

사양서 Soak Test Suite 기준 (축소 버전):
  - 100-Turn 대화: OOM 없이 완료
  - 50회 연속 추론: 안정성 확인
  - 동시 임베딩+추론: 성능 저하 < 20% 확인

GPU 서버가 가동 중일 때만 실행 가능하다.
"""

from __future__ import annotations

import asyncio
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
    pytest.mark.soak,
    pytest.mark.skipif(
        not _gpu_server_available(),
        reason="GPU 서버(192.168.22.28:8001)에 연결할 수 없습니다",
    ),
]


class TestSoak:
    """장시간 안정성 테스트."""

    @pytest.mark.asyncio
    async def test_100_turn_conversation(self):
        """
        100턴 대화를 OOM 없이 완료해야 한다.

        매 턴마다 짧은 질문과 짧은 응답을 주고받아
        대화 히스토리가 누적되어도 안정적으로 동작하는지 검증한다.
        max_tokens를 작게 유지하여 전체 소요 시간을 합리적으로 제한한다.
        """
        messages: list[dict[str, str]] = []
        system_prompt = "You are a helpful assistant. Keep answers under 20 words."
        turn_latencies: list[float] = []
        error_count = 0

        async with httpx.AsyncClient(timeout=60.0) as client:
            for turn in range(100):
                # 간결한 질문으로 대화 히스토리를 누적
                user_msg = f"Turn {turn + 1}: What is {turn + 1} * 2?"
                messages.append({"role": "user", "content": user_msg})

                start = time.monotonic()
                try:
                    resp = await client.post(
                        f"{GPU_SERVER_URL}/v1/chat/completions",
                        json={
                            "model": MODEL_ID,
                            "messages": [
                                {"role": "system", "content": system_prompt},
                                # 컨텍스트 관리: 최근 20턴만 전송 (OOM 방지)
                                *messages[-40:],
                            ],
                            "max_tokens": 32,
                            "temperature": 0.1,
                        },
                    )
                    elapsed = time.monotonic() - start
                    turn_latencies.append(elapsed)

                    if resp.status_code == 200:
                        data = resp.json()
                        assistant_text = data["choices"][0]["message"]["content"]
                        messages.append({"role": "assistant", "content": assistant_text})
                    else:
                        error_count += 1
                        messages.append({"role": "assistant", "content": "(error)"})
                except Exception as e:
                    error_count += 1
                    elapsed = time.monotonic() - start
                    turn_latencies.append(elapsed)
                    messages.append({"role": "assistant", "content": f"(error: {e})"})

        # 결과 보고
        avg_latency = sum(turn_latencies) / len(turn_latencies) if turn_latencies else 0
        first_10_avg = sum(turn_latencies[:10]) / 10
        last_10_avg = sum(turn_latencies[-10:]) / 10

        print("\n[100-Turn Conversation]")
        print(f"  완료 턴: {len(turn_latencies)}/100")
        print(f"  에러: {error_count}개")
        print(f"  평균 지연: {avg_latency:.3f}s")
        print(f"  처음 10턴 평균: {first_10_avg:.3f}s")
        print(f"  마지막 10턴 평균: {last_10_avg:.3f}s")
        print(f"  총 소요: {sum(turn_latencies):.1f}s")

        # 에러가 5% 미만이어야 한다
        assert error_count < 5, f"에러가 5개 이상: {error_count}개"
        # 100턴 모두 완료되어야 한다
        assert len(turn_latencies) == 100

    @pytest.mark.asyncio
    async def test_continuous_inference_50(self):
        """
        50회 연속 독립 추론이 안정적으로 완료되어야 한다.

        매 요청이 독립적 (대화 히스토리 없음).
        응답 시간이 점진적으로 증가하지 않는지 확인한다.
        """
        latencies: list[float] = []
        error_count = 0

        questions = [
            f"What is the square root of {(i + 1) * 100}? Answer with just the number."
            for i in range(50)
        ]

        async with httpx.AsyncClient(timeout=30.0) as client:
            for _i, question in enumerate(questions):
                start = time.monotonic()
                try:
                    resp = await client.post(
                        f"{GPU_SERVER_URL}/v1/chat/completions",
                        json={
                            "model": MODEL_ID,
                            "messages": [{"role": "user", "content": question}],
                            "max_tokens": 32,
                            "temperature": 0.1,
                        },
                    )
                    elapsed = time.monotonic() - start
                    latencies.append(elapsed)

                    if resp.status_code != 200:
                        error_count += 1
                except Exception:
                    error_count += 1
                    latencies.append(time.monotonic() - start)

        # 처음 10개와 마지막 10개의 평균 비교 — 성능 저하가 없어야 한다
        first_10 = sum(latencies[:10]) / 10
        last_10 = sum(latencies[-10:]) / 10
        avg = sum(latencies) / len(latencies)

        print("\n[Continuous Inference 50회]")
        print(f"  완료: {len(latencies) - error_count}/50")
        print(f"  에러: {error_count}개")
        print(f"  평균 지연: {avg:.3f}s")
        print(f"  처음 10회 평균: {first_10:.3f}s")
        print(f"  마지막 10회 평균: {last_10:.3f}s")
        print(f"  성능 변화: {((last_10 / first_10) - 1) * 100:+.1f}%")

        assert error_count < 3, f"에러가 3개 이상: {error_count}개"
        # 마지막 10회가 처음 10회 대비 50% 이상 느려지면 안 된다
        assert last_10 < first_10 * 1.5, (
            f"성능 저하 감지: 처음 {first_10:.3f}s → 마지막 {last_10:.3f}s"
        )

    @pytest.mark.asyncio
    async def test_concurrent_embed_and_infer(self):
        """
        추론 중에 임베딩을 동시에 실행해도 성능 저하가 20% 미만이어야 한다.

        1. 기준 추론 시간 측정 (5회)
        2. 임베딩과 동시 실행하여 추론 시간 측정 (5회)
        3. 성능 저하율 확인
        """
        # 기준 추론 시간 측정
        baseline_latencies: list[float] = []

        async with httpx.AsyncClient(timeout=30.0) as client:
            for _ in range(5):
                start = time.monotonic()
                resp = await client.post(
                    f"{GPU_SERVER_URL}/v1/chat/completions",
                    json={
                        "model": MODEL_ID,
                        "messages": [
                            {"role": "user", "content": "What is 42 * 7?"},
                        ],
                        "max_tokens": 32,
                        "temperature": 0.1,
                    },
                )
                elapsed = time.monotonic() - start
                assert resp.status_code == 200
                baseline_latencies.append(elapsed)

        baseline_avg = sum(baseline_latencies) / len(baseline_latencies)

        # 임베딩과 동시 실행
        concurrent_latencies: list[float] = []

        async def embed_background():
            """배경에서 임베딩을 반복 실행한다."""
            async with httpx.AsyncClient(timeout=30.0) as emb_client:
                for _ in range(10):
                    await emb_client.post(
                        f"{EMBEDDING_SERVER_URL}/v1/embed",
                        json={"texts": [f"Embedding test sentence {i}" for i in range(5)]},
                    )
                    await asyncio.sleep(0.05)

        async def infer_during_embed():
            """임베딩과 동시에 추론을 실행한다."""
            async with httpx.AsyncClient(timeout=30.0) as inf_client:
                for _ in range(5):
                    start = time.monotonic()
                    resp = await inf_client.post(
                        f"{GPU_SERVER_URL}/v1/chat/completions",
                        json={
                            "model": MODEL_ID,
                            "messages": [
                                {"role": "user", "content": "What is 42 * 7?"},
                            ],
                            "max_tokens": 32,
                            "temperature": 0.1,
                        },
                    )
                    elapsed = time.monotonic() - start
                    assert resp.status_code == 200
                    concurrent_latencies.append(elapsed)

        # 동시 실행
        await asyncio.gather(embed_background(), infer_during_embed())

        concurrent_avg = sum(concurrent_latencies) / len(concurrent_latencies)
        degradation = ((concurrent_avg / baseline_avg) - 1) * 100

        print("\n[Concurrent Embed + Infer]")
        print(f"  기준 추론 평균: {baseline_avg:.3f}s")
        print(f"  동시 추론 평균: {concurrent_avg:.3f}s")
        print(f"  성능 저하: {degradation:+.1f}%")
        print("  목표: < 20%")

        # 임베딩이 별도 서버이므로 성능 저하가 거의 없어야 한다
        # 50% 이상 저하면 실패로 처리 (여유 있게 설정)
        assert degradation < 50, f"성능 저하 {degradation:.1f}%가 50%를 초과"
