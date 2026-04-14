"""
Phase 8.0 통합 테스트 — Training Pipeline.

부트스트랩 데이터 생성, 상호작용 데이터 수집(PII 마스킹),
체크포인트 관리 라이프사이클을 통합 테스트한다.

사양서 Ch.22.6 Test 6에 해당한다.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from training.bootstrap_generator import BootstrapGenerator
from training.checkpoint_manager import CheckpointManager
from training.data_collector import DataCollector


# ─────────────────────────────────────────────
# Test 6: Training Pipeline 통합 테스트
# ─────────────────────────────────────────────
@pytest.mark.asyncio
class TestBootstrapGenerator:
    """부트스트랩 합성 데이터 생성기를 테스트한다."""

    async def test_generate_creates_valid_jsonl(self, tmp_path: Path) -> None:
        """generate()가 유효한 JSONL 파일을 생성하고 요약 정보를 반환한다."""
        gen = BootstrapGenerator(seed=42)
        output_path = str(tmp_path / "bootstrap")
        result = await gen.generate(count=20, output_path=output_path)

        # 반환값 검증
        assert result["total"] == 20
        assert "tool_samples" in result
        assert "reasoning_samples" in result
        # 도구 70%, 추론 30% 비율
        assert result["tool_samples"] == 14  # 20 * 0.7
        assert result["reasoning_samples"] == 6  # 20 * 0.3

    async def test_generate_output_file_exists(self, tmp_path: Path) -> None:
        """생성된 JSONL 파일이 실제로 디스크에 존재하고 내용이 유효하다."""
        gen = BootstrapGenerator(seed=123)
        output_path = str(tmp_path / "bootstrap_data")
        result = await gen.generate(count=5, output_path=output_path)

        # 출력 파일이 존재하는지 확인
        output_file = Path(result.get("output_file", output_path + "/bootstrap_data.jsonl"))
        if not output_file.exists():
            # output_path 디렉토리 내의 JSONL 파일 탐색
            jsonl_files = list(Path(output_path).rglob("*.jsonl"))
            assert len(jsonl_files) > 0, "JSONL 파일이 생성되지 않았다"
            output_file = jsonl_files[0]

        # 각 줄이 유효한 JSON인지 검증
        lines = output_file.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 5
        for line in lines:
            data = json.loads(line)
            assert "messages" in data
            assert "metadata" in data

    async def test_generate_tool_sample_format(self, tmp_path: Path) -> None:
        """도구 사용 샘플의 형식이 OpenAI tool_calls 호환인지 검증한다."""
        gen = BootstrapGenerator(seed=42)
        # 내부 메서드 직접 호출하여 샘플 형식 검증
        sample = gen._generate_tool_sample()

        assert "messages" in sample
        assert "metadata" in sample
        assert sample["metadata"]["source"] == "bootstrap"
        # 첫 번째 메시지는 user, 두 번째는 assistant (tool_calls 포함)
        assert len(sample["messages"]) >= 2
        assert sample["messages"][0]["role"] == "user"

    async def test_generate_deterministic_with_seed(self, tmp_path: Path) -> None:
        """같은 seed로 생성하면 동일한 결과를 반환한다."""
        gen1 = BootstrapGenerator(seed=42)
        gen2 = BootstrapGenerator(seed=42)

        sample1 = gen1._generate_tool_sample()
        sample2 = gen2._generate_tool_sample()

        # 같은 seed이므로 동일한 샘플이 생성되어야 한다
        assert sample1["metadata"] == sample2["metadata"]


@pytest.mark.asyncio
class TestDataCollector:
    """상호작용 데이터 수집기의 PII 마스킹과 수집 기능을 테스트한다."""

    async def test_collect_turn_stores_in_buffer(self, tmp_path: Path) -> None:
        """collect_turn()이 데이터를 버퍼에 정상적으로 저장한다."""
        collector = DataCollector(
            storage_dir=str(tmp_path / "collected"),
            max_buffer_size=100,
        )

        await collector.collect_turn(
            messages=[
                {"role": "user", "content": "안녕하세요"},
                {"role": "assistant", "content": "반갑습니다!"},
            ],
        )

        assert collector.buffer_size == 1

    async def test_pii_masking_email(self, tmp_path: Path) -> None:
        """이메일 주소가 [EMAIL]로 마스킹된다."""
        collector = DataCollector(
            storage_dir=str(tmp_path / "collected"),
            max_buffer_size=100,
        )

        await collector.collect_turn(
            messages=[
                {"role": "user", "content": "내 이메일은 user@test.com 이야"},
            ],
        )

        # 버퍼에서 마스킹 확인
        record = collector._buffer[0]
        user_content = record["messages"][0]["content"]
        assert "user@test.com" not in user_content
        assert "[EMAIL]" in user_content

    async def test_pii_masking_phone(self, tmp_path: Path) -> None:
        """전화번호가 [PHONE]으로 마스킹된다."""
        collector = DataCollector(
            storage_dir=str(tmp_path / "collected"),
            max_buffer_size=100,
        )

        await collector.collect_turn(
            messages=[
                {"role": "user", "content": "전화번호는 010-1234-5678 입니다"},
            ],
        )

        record = collector._buffer[0]
        user_content = record["messages"][0]["content"]
        assert "010-1234-5678" not in user_content
        assert "[PHONE]" in user_content

    async def test_sensitive_path_filtering(self, tmp_path: Path) -> None:
        """민감한 경로(.env 등)가 포함된 도구 결과가 필터링된다."""
        collector = DataCollector(
            storage_dir=str(tmp_path / "collected"),
            max_buffer_size=100,
        )

        await collector.collect_turn(
            messages=[{"role": "user", "content": "환경 변수 확인"}],
            tool_results=[
                {
                    "tool_name": "Read",
                    "data": "SECRET_KEY=abc123",
                    "path": "/project/.env",
                }
            ],
        )

        # 민감 경로 포함된 도구 결과가 필터링되었는지 확인
        if collector.buffer_size > 0:
            record = collector._buffer[0]
            # tool_results에서 .env 경로가 필터링되어야 한다
            if "tool_results" in record and record["tool_results"]:
                for result in record["tool_results"]:
                    data_str = str(result.get("data", ""))
                    path_str = str(result.get("path", ""))
                    # .env 경로의 민감 데이터가 제거되었는지 확인
                    assert "SECRET_KEY=abc123" not in data_str or ".env" not in path_str

    async def test_collector_stats_tracking(self, tmp_path: Path) -> None:
        """수집 통계가 올바르게 추적된다."""
        collector = DataCollector(
            storage_dir=str(tmp_path / "collected"),
            max_buffer_size=100,
        )

        await collector.collect_turn(
            messages=[{"role": "user", "content": "hi"}],
        )
        await collector.collect_turn(
            messages=[{"role": "user", "content": "contact me at a@b.com"}],
        )

        stats = collector.stats
        assert stats["total_collected"] >= 2


@pytest.mark.asyncio
class TestCheckpointManager:
    """체크포인트 관리자의 목록 조회, 활성화, 롤백을 테스트한다."""

    def test_list_checkpoints_empty(self, tmp_path: Path) -> None:
        """빈 디렉토리에서 빈 목록을 반환한다."""
        mgr = CheckpointManager(checkpoints_dir=str(tmp_path / "ckpts"))
        checkpoints = mgr.list_checkpoints()
        assert checkpoints == []

    def test_list_checkpoints_with_metadata(self, tmp_path: Path) -> None:
        """metadata.json이 있는 체크포인트를 올바르게 조회한다."""
        ckpts_dir = tmp_path / "ckpts"
        ckpts_dir.mkdir()

        # 체크포인트 2개 수동 생성
        for name in ["ckpt-001", "ckpt-002"]:
            ckpt_dir = ckpts_dir / name
            ckpt_dir.mkdir()
            metadata = {
                "created_at": "2026-04-14T10:00:00+00:00",
                "eval_accuracy": 0.85 if name == "ckpt-002" else 0.78,
                "training_loss": 0.5,
            }
            (ckpt_dir / "metadata.json").write_text(
                json.dumps(metadata, ensure_ascii=False), encoding="utf-8"
            )

        mgr = CheckpointManager(checkpoints_dir=str(ckpts_dir))
        checkpoints = mgr.list_checkpoints()

        assert len(checkpoints) == 2
        names = [c["name"] for c in checkpoints]
        assert "ckpt-001" in names
        assert "ckpt-002" in names

    def test_active_checkpoint_initially_none(self, tmp_path: Path) -> None:
        """초기 상태에서 활성 체크포인트가 None이다."""
        mgr = CheckpointManager(checkpoints_dir=str(tmp_path / "ckpts"))
        assert mgr.active_checkpoint is None

    async def test_activate_with_mock_gpu_server(self, tmp_path: Path) -> None:
        """activate()가 GPU 서버에 LoRA 로드 요청을 보내고 상태를 업데이트한다."""
        ckpts_dir = tmp_path / "ckpts"
        ckpt_dir = ckpts_dir / "ckpt-001"
        ckpt_dir.mkdir(parents=True)
        (ckpt_dir / "metadata.json").write_text(
            json.dumps({"created_at": "2026-04-14T10:00:00+00:00"}),
            encoding="utf-8",
        )

        mgr = CheckpointManager(checkpoints_dir=str(ckpts_dir))

        # httpx.AsyncClient를 mock — raise_for_status()와 json()을 동기 메서드로
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()  # 동기 메서드
        mock_response.json.return_value = {"status": "loaded"}

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch("training.checkpoint_manager.httpx.AsyncClient") as mock_cls:
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            await mgr.activate("ckpt-001", gpu_server_url="http://192.168.22.28:8000")

        # 활성 체크포인트가 업데이트되었는지 확인
        assert mgr.active_checkpoint == "ckpt-001"

    async def test_rollback_with_mock_gpu_server(self, tmp_path: Path) -> None:
        """rollback()이 GPU 서버에 LoRA 언로드 요청을 보내고 상태를 초기화한다."""
        ckpts_dir = tmp_path / "ckpts"
        ckpts_dir.mkdir(parents=True)
        mgr = CheckpointManager(checkpoints_dir=str(ckpts_dir))
        # 활성 체크포인트를 수동 설정
        mgr._active_checkpoint = "ckpt-001"

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"status": "unloaded"}

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with patch("training.checkpoint_manager.httpx.AsyncClient") as mock_cls:
            mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            await mgr.rollback(gpu_server_url="http://192.168.22.28:8000")

        # 활성 체크포인트가 초기화되었는지 확인
        assert mgr.active_checkpoint is None
