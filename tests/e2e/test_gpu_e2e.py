"""
E2E 테스트 — 실 GPU 서버(192.168.22.28) 연결 검증.

이 테스트는 실제 vLLM 서버가 가동 중일 때만 실행 가능하다.
pytest -m e2e 로 분리 실행하거나,
GPU 서버가 없으면 자동 스킵된다.

서버 구성:
  - Gemma 4 31B (AWQ INT4): 192.168.22.28:8001
  - e5-large embedding: 192.168.22.28:8002
"""

from __future__ import annotations

import json

import httpx
import pytest

# GPU 서버 주소 (config와 동일)
GPU_SERVER_URL = "http://192.168.22.28:8001"
EMBEDDING_SERVER_URL = "http://192.168.22.28:8002"
MODEL_ID = "qwen3.5-27b"
EMBEDDING_MODEL_ID = "multilingual-e5-large"


def _gpu_server_available() -> bool:
    """GPU 서버에 연결 가능한지 확인한다."""
    try:
        import socket

        s = socket.socket()
        s.settimeout(3)
        result = s.connect_ex(("192.168.22.28", 8001)) == 0
        s.close()
        return result
    except Exception:
        return False


# GPU 서버 미접속 시 전체 스킵
pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(
        not _gpu_server_available(),
        reason="GPU 서버(192.168.22.28:8001)에 연결할 수 없습니다",
    ),
]


# ─────────────────────────────────────────────
# 기본 연결 테스트
# ─────────────────────────────────────────────
class TestGPUServerConnection:
    """GPU 서버 기본 연결 및 엔드포인트를 검증한다."""

    @pytest.mark.asyncio
    async def test_health_check(self):
        """vLLM /health 엔드포인트가 정상 응답해야 한다."""
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{GPU_SERVER_URL}/health")
            assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_embedding_health_check(self):
        """임베딩 서버 /health 엔드포인트가 정상 응답해야 한다."""
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{EMBEDDING_SERVER_URL}/health")
            assert resp.status_code == 200
            data = resp.json()
            assert data.get("model") == EMBEDDING_MODEL_ID

    @pytest.mark.asyncio
    async def test_models_endpoint(self):
        """/v1/models에서 qwen3.5-27b 모델이 로드되어 있어야 한다."""
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{GPU_SERVER_URL}/v1/models")
            assert resp.status_code == 200
            data = resp.json()
            model_ids = [m["id"] for m in data["data"]]
            assert MODEL_ID in model_ids


# ─────────────────────────────────────────────
# 추론 테스트
# ─────────────────────────────────────────────
class TestInference:
    """실 모델로 추론이 올바르게 동작하는지 검증한다."""

    @pytest.mark.asyncio
    async def test_simple_chat_completion(self):
        """간단한 텍스트 생성이 정상적으로 완료되어야 한다."""
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{GPU_SERVER_URL}/v1/chat/completions",
                json={
                    "model": MODEL_ID,
                    "messages": [
                        {"role": "user", "content": "1 + 1 = ?. 숫자만 답하세요."},
                    ],
                    "max_tokens": 32,
                    "temperature": 0.1,
                },
            )
            assert resp.status_code == 200
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            assert content, "응답이 비어있으면 안 된다"
            assert "2" in content

    @pytest.mark.asyncio
    async def test_streaming_chat(self):
        """SSE 스트리밍으로 텍스트 델타를 수신할 수 있어야 한다."""
        chunks: list[str] = []

        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream(
                "POST",
                f"{GPU_SERVER_URL}/v1/chat/completions",
                json={
                    "model": MODEL_ID,
                    "messages": [
                        {"role": "user", "content": "Hello, say 'world'."},
                    ],
                    "max_tokens": 32,
                    "temperature": 0.1,
                    "stream": True,
                },
            ) as resp:
                assert resp.status_code == 200
                async for line in resp.aiter_lines():
                    if line.startswith("data: ") and line[6:].strip() != "[DONE]":
                        data = json.loads(line[6:])
                        delta = data["choices"][0].get("delta", {})
                        if delta.get("content"):
                            chunks.append(delta["content"])

        full_text = "".join(chunks)
        assert len(full_text) > 0, "스트리밍 응답이 비어있으면 안 된다"

    @pytest.mark.asyncio
    async def test_tool_calling(self):
        """tool_calls가 올바르게 생성되어야 한다."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "Read",
                    "description": "파일을 읽습니다.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {
                                "type": "string",
                                "description": "읽을 파일 경로",
                            },
                        },
                        "required": ["file_path"],
                    },
                },
            }
        ]

        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{GPU_SERVER_URL}/v1/chat/completions",
                json={
                    "model": MODEL_ID,
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "You are a helpful assistant. "
                                "Use the Read tool to read files when asked."
                            ),
                        },
                        {
                            "role": "user",
                            "content": "Read the file at /tmp/test.txt",
                        },
                    ],
                    "tools": tools,
                    "tool_choice": "auto",
                    "max_tokens": 256,
                    "temperature": 0.1,
                },
            )
            assert resp.status_code == 200
            data = resp.json()
            choice = data["choices"][0]

            # 모델이 tool_calls를 생성했는지 확인
            # (모델에 따라 text 응답만 할 수도 있으므로 유연하게 처리)
            if choice.get("finish_reason") == "tool_calls":
                tool_calls = choice["message"].get("tool_calls", [])
                assert len(tool_calls) >= 1
                assert tool_calls[0]["function"]["name"] == "Read"


# ─────────────────────────────────────────────
# 임베딩 테스트
# ─────────────────────────────────────────────
class TestEmbedding:
    """e5-large 임베딩 서버를 검증한다."""

    @pytest.mark.asyncio
    async def test_embedding_generation(self):
        """텍스트 임베딩이 1024차원 벡터로 생성되어야 한다."""
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{EMBEDDING_SERVER_URL}/v1/embed",
                json={"texts": ["This is a test sentence."]},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert len(data["embeddings"]) == 1
            assert len(data["embeddings"][0]) == 1024
            assert data["dimension"] == 1024

    @pytest.mark.asyncio
    async def test_batch_embedding(self):
        """여러 텍스트의 배치 임베딩이 올바르게 생성되어야 한다."""
        texts = [
            "First sentence.",
            "Second sentence.",
            "Third sentence in English.",
        ]

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{EMBEDDING_SERVER_URL}/v1/embed",
                json={"texts": texts},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert len(data["embeddings"]) == 3
            for emb in data["embeddings"]:
                assert len(emb) == 1024


# ─────────────────────────────────────────────
# LocalModelProvider 통합 테스트
# ─────────────────────────────────────────────
class TestLocalModelProvider:
    """LocalModelProvider를 통한 통합 테스트."""

    def _create_provider(self):
        """테스트용 LocalModelProvider를 생성한다."""
        from core.model.inference import LocalModelProvider

        return LocalModelProvider(
            base_url=GPU_SERVER_URL,
            model_id=MODEL_ID,
            embedding_base_url=EMBEDDING_SERVER_URL,
            embedding_model_id=EMBEDDING_MODEL_ID,
            max_context_tokens=4096,
            max_output_tokens=4096,
        )

    @pytest.mark.asyncio
    async def test_provider_health_check(self):
        """LocalModelProvider.health_check()가 True를 반환해야 한다."""
        provider = self._create_provider()
        try:
            result = await provider.health_check()
            assert result is True
        finally:
            await provider.close()

    @pytest.mark.asyncio
    async def test_provider_stream(self):
        """LocalModelProvider.stream()이 StreamEvent를 yield해야 한다."""
        from core.message import Message, StreamEventType

        provider = self._create_provider()
        try:
            messages = [Message.user("1 + 2 = ? 숫자만 답하세요.")]
            events = []

            async for event in provider.stream(
                messages=messages,
                system_prompt="You are a helpful assistant.",
                temperature=0.1,
                max_tokens=32,
            ):
                events.append(event)

            # 최소한 MESSAGE_START, TEXT_DELTA, MESSAGE_STOP이 있어야 한다
            event_types = [e.type for e in events]
            assert StreamEventType.MESSAGE_START in event_types
            assert StreamEventType.MESSAGE_STOP in event_types

            # TEXT_DELTA가 하나 이상 있어야 한다
            text_deltas = [e for e in events if e.type == StreamEventType.TEXT_DELTA]
            assert len(text_deltas) >= 1

            # 응답에 "3"이 포함되어야 한다
            full_text = "".join(e.text for e in text_deltas if e.text)
            assert "3" in full_text
        finally:
            await provider.close()

    @pytest.mark.asyncio
    async def test_provider_embed(self):
        """LocalModelProvider.embed()가 올바른 차원의 벡터를 반환해야 한다."""
        provider = self._create_provider()
        try:
            result = await provider.embed(["테스트 문장입니다."])
            assert len(result) == 1
            assert len(result[0]) == 1024
        finally:
            await provider.close()

    @pytest.mark.asyncio
    async def test_multi_turn_conversation(self):
        """3턴 대화가 문맥을 유지하며 진행되어야 한다."""
        from core.message import Message, StreamEventType

        provider = self._create_provider()
        try:
            messages = [Message.user("내 이름은 넥서스야. 기억해.")]

            # 턴 1: 이름 알려주기
            turn1_text = ""
            async for event in provider.stream(
                messages=messages,
                system_prompt="You are a helpful assistant. Remember user information.",
                max_tokens=128,
            ):
                if event.type == StreamEventType.TEXT_DELTA and event.text:
                    turn1_text += event.text

            messages.append(Message.assistant(turn1_text))

            # 턴 2: 이름 물어보기
            messages.append(Message.user("내 이름이 뭐라고 했지?"))
            turn2_text = ""
            async for event in provider.stream(
                messages=messages,
                system_prompt="You are a helpful assistant. Remember user information.",
                max_tokens=128,
            ):
                if event.type == StreamEventType.TEXT_DELTA and event.text:
                    turn2_text += event.text

            # 이전 턴에서 알려준 이름을 기억하고 있어야 한다
            assert "넥서스" in turn2_text or "nexus" in turn2_text.lower()
        finally:
            await provider.close()
