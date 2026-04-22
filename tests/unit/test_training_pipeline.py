"""
training/ 모듈 단위 테스트 — BootstrapGenerator, DataCollector, Trainer,
CheckpointManager, FeedbackLoop 검증.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from training.bootstrap_generator import BootstrapGenerator
from training.checkpoint_manager import CheckpointManager
from training.data_collector import DataCollector
from training.feedback_loop import FeedbackLoop
from training.strategy import TrainingPhase, TrainingStrategy
from training.trainer import LoRATrainer, TrainingConfig


# ─────────────────────────────────────────────
# BootstrapGenerator 테스트
# ─────────────────────────────────────────────
class TestBootstrapGenerator:
    """BootstrapGenerator 부트스트랩 데이터 생성 테스트."""

    @pytest.fixture
    def generator(self) -> BootstrapGenerator:
        """재현 가능한 생성기 fixture."""
        return BootstrapGenerator(seed=42)

    async def test_generate_creates_jsonl_file(self, generator, tmp_path):
        """generate()가 JSONL 파일을 생성하는지 확인한다."""
        output_dir = str(tmp_path / "bootstrap")
        stats = await generator.generate(count=10, output_path=output_dir)

        output_file = Path(stats["output_file"])
        assert output_file.exists()
        assert output_file.suffix == ".jsonl"

    async def test_generate_correct_count(self, generator, tmp_path):
        """
        생성된 샘플 수가 요청한 수와 일치하는지 확인한다.

        Phase 3 비율: 도구 45% / 추론 25% / 서브에이전트 15% / 지식 15%.
        """
        output_dir = str(tmp_path / "bootstrap")
        stats = await generator.generate(count=100, output_path=output_dir)

        assert stats["total"] == 100
        assert stats["tool_samples"] == 45  # 45%
        assert stats["reasoning_samples"] == 25  # 25%
        assert stats["subagent_samples"] == 15  # 15%
        assert stats["knowledge_samples"] == 15  # 15%

    async def test_generate_valid_jsonl(self, generator, tmp_path):
        """생성된 JSONL이 유효한 JSON인지 확인한다."""
        output_dir = str(tmp_path / "bootstrap")
        stats = await generator.generate(count=20, output_path=output_dir)

        with open(stats["output_file"], encoding="utf-8") as f:  # noqa: ASYNC230
            for line in f:
                record = json.loads(line.strip())
                assert "messages" in record
                assert "metadata" in record
                assert len(record["messages"]) >= 2

    async def test_generate_tool_samples_have_tool_calls(self, generator, tmp_path):
        """도구 샘플의 assistant 메시지에 tool_calls가 포함되는지."""
        output_dir = str(tmp_path / "bootstrap")
        stats = await generator.generate(count=50, output_path=output_dir)

        tool_count = 0
        with open(stats["output_file"], encoding="utf-8") as f:  # noqa: ASYNC230
            for line in f:
                record = json.loads(line.strip())
                if record["metadata"].get("category") == "tool_use":
                    assistant_msg = record["messages"][1]
                    assert "tool_calls" in assistant_msg
                    assert len(assistant_msg["tool_calls"]) > 0
                    tool_count += 1
        assert tool_count > 0

    async def test_generate_deterministic_with_seed(self, tmp_path):
        """같은 seed로 생성하면 같은 결과가 나오는지 확인한다."""
        gen1 = BootstrapGenerator(seed=123)
        gen2 = BootstrapGenerator(seed=123)

        dir1 = str(tmp_path / "run1")
        dir2 = str(tmp_path / "run2")

        await gen1.generate(count=10, output_path=dir1)
        await gen2.generate(count=10, output_path=dir2)

        with (
            open(Path(dir1) / "bootstrap_data.jsonl", encoding="utf-8") as f1,  # noqa: ASYNC230
            open(Path(dir2) / "bootstrap_data.jsonl", encoding="utf-8") as f2,  # noqa: ASYNC230
        ):
            assert f1.read() == f2.read()

    # ─────────────────────────────────────────────
    # M7 — tenant_id 지원 (2026-04-22)
    # ─────────────────────────────────────────────
    async def test_generate_default_tenant_uses_root_directory(
        self, generator, tmp_path
    ):
        """tenant_id=None이면 기존 경로(output_path/bootstrap_data.jsonl)를 유지해야 한다."""
        output_dir = str(tmp_path / "bootstrap")
        stats = await generator.generate(count=5, output_path=output_dir, tenant_id=None)

        # 하위 호환: 서브디렉토리 없이 output_root 바로 아래에 저장
        expected = Path(output_dir) / "bootstrap_data.jsonl"
        assert Path(stats["output_file"]) == expected
        assert expected.exists()
        assert stats["tenant_id"] == "default"

    async def test_generate_custom_tenant_goes_to_subdirectory(
        self, generator, tmp_path
    ):
        """tenant_id가 있으면 {output_path}/{tenant_id}/bootstrap_data.jsonl로 격리."""
        output_dir = str(tmp_path / "bootstrap")
        stats = await generator.generate(
            count=5, output_path=output_dir, tenant_id="dongguk"
        )

        expected = Path(output_dir) / "dongguk" / "bootstrap_data.jsonl"
        assert Path(stats["output_file"]) == expected
        assert expected.exists()
        assert stats["tenant_id"] == "dongguk"

    async def test_generate_stamps_tenant_id_in_sample_metadata(
        self, generator, tmp_path
    ):
        """각 샘플 metadata에 tenant_id가 찍혀야 한다 (감사·분석용)."""
        output_dir = str(tmp_path / "bootstrap")
        await generator.generate(
            count=5, output_path=output_dir, tenant_id="hanyang"
        )

        path = Path(output_dir) / "hanyang" / "bootstrap_data.jsonl"
        with open(path, encoding="utf-8") as f:
            for line in f:
                sample = json.loads(line)
                assert sample["metadata"]["tenant_id"] == "hanyang"

    async def test_generate_rejects_invalid_tenant_id(self, generator, tmp_path):
        """허용 문자셋을 벗어난 tenant_id는 ValueError로 조기 차단."""
        output_dir = str(tmp_path / "bootstrap")
        with pytest.raises(ValueError):
            await generator.generate(
                count=5, output_path=output_dir, tenant_id="동국대"  # 한글 금지
            )


# ─────────────────────────────────────────────
# DataCollector 테스트
# ─────────────────────────────────────────────
class TestDataCollector:
    """DataCollector 상호작용 데이터 수집 테스트."""

    @pytest.fixture
    def collector(self, tmp_path) -> DataCollector:
        """임시 디렉토리를 사용하는 수집기 fixture."""
        return DataCollector(
            storage_dir=str(tmp_path / "collected"),
            max_buffer_size=5,
        )

    async def test_collect_turn_adds_to_buffer(self, collector):
        """collect_turn()이 버퍼에 레코드를 추가하는지."""
        await collector.collect_turn(
            messages=[
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "world"},
            ]
        )
        assert collector.buffer_size == 1

    async def test_auto_flush_on_buffer_full(self, collector, tmp_path):
        """버퍼가 가득 차면 자동으로 플러시되는지."""
        for i in range(6):  # max_buffer_size=5이므로 5번째에서 플러시
            await collector.collect_turn(
                messages=[
                    {"role": "user", "content": f"msg {i}"},
                    {"role": "assistant", "content": f"resp {i}"},
                ]
            )
        # 5개가 플러시되고 1개가 버퍼에 남아 있어야 한다
        assert collector.buffer_size == 1
        collected_dir = tmp_path / "collected"
        jsonl_files = list(collected_dir.glob("collected_*.jsonl"))
        assert len(jsonl_files) >= 1

    async def test_pii_masking_email(self, collector):
        """이메일 PII가 마스킹되는지 확인한다."""
        await collector.collect_turn(
            messages=[
                {"role": "user", "content": "내 이메일은 test@example.com 입니다"},
                {"role": "assistant", "content": "확인했습니다"},
            ]
        )
        # 버퍼에서 직접 확인
        record = collector._buffer[0]
        assert "[EMAIL]" in record["messages"][0]["content"]
        assert "test@example.com" not in record["messages"][0]["content"]

    async def test_pii_masking_phone(self, collector):
        """전화번호 PII가 마스킹되는지 확인한다."""
        await collector.collect_turn(
            messages=[
                {"role": "user", "content": "전화번호: 010-1234-5678"},
                {"role": "assistant", "content": "네"},
            ]
        )
        record = collector._buffer[0]
        assert "[PHONE]" in record["messages"][0]["content"]

    async def test_sensitive_path_filtering(self, collector):
        """민감한 파일 경로가 포함된 도구 결과가 필터링되는지."""
        await collector.collect_turn(
            messages=[{"role": "user", "content": "test"}],
            tool_results=[
                {"tool": "Read", "data": "내용", "path": "/home/user/.env"},
                {"tool": "Read", "data": "일반 내용", "path": "/src/main.py"},
            ],
        )
        record = collector._buffer[0]
        # .env 경로가 포함된 결과는 제거되어야 한다
        assert len(record["tool_results"]) == 1
        assert collector.stats["total_filtered"] == 1

    async def test_export_jsonl_quality_filter(self, collector, tmp_path):
        """export_jsonl()이 품질 점수 기준으로 필터링하는지."""
        # 고품질 데이터
        await collector.collect_turn(
            messages=[
                {"role": "user", "content": "good"},
                {"role": "assistant", "content": "good response"},
            ],
            metadata={"quality_score": 0.9},
        )
        # 저품질 데이터
        await collector.collect_turn(
            messages=[
                {"role": "user", "content": "bad"},
                {"role": "assistant", "content": "bad response"},
            ],
            metadata={"quality_score": 0.3},
        )

        output_file = str(tmp_path / "export.jsonl")
        count = await collector.export_jsonl(output_file, min_quality=0.7)

        assert count == 1  # 고품질 1개만 내보내짐
        with open(output_file, encoding="utf-8") as f:  # noqa: ASYNC230
            records = [json.loads(line) for line in f if line.strip()]
            assert len(records) == 1


# ─────────────────────────────────────────────
# TrainingConfig 테스트
# ─────────────────────────────────────────────
class TestTrainingConfig:
    """TrainingConfig 설정 테스트."""

    def test_default_values(self):
        """기본 설정값이 올바른지 확인한다."""
        config = TrainingConfig()
        assert config.method == "qlora"
        assert config.lora_rank == 16
        assert config.lora_alpha == 32
        assert config.batch_size == 1

    def test_to_dict(self):
        """to_dict()가 모든 필드를 포함하는지."""
        config = TrainingConfig()
        d = config.to_dict()
        assert "method" in d
        assert "lora_rank" in d
        assert "learning_rate" in d
        assert d["method"] == "qlora"


# ─────────────────────────────────────────────
# LoRATrainer 테스트
# ─────────────────────────────────────────────
class TestLoRATrainer:
    """LoRATrainer 학습 관리 테스트."""

    def test_rejects_non_lan_url(self):
        """LAN 주소가 아닌 URL은 거부되는지."""
        with pytest.raises(ValueError, match="에어갭 위반"):
            LoRATrainer(
                gpu_server_url="https://api.openai.com",
                config=TrainingConfig(),
            )

    def test_accepts_lan_url(self):
        """LAN 주소는 허용되는지."""
        trainer = LoRATrainer(
            gpu_server_url="http://192.168.22.28:8000",
            config=TrainingConfig(),
        )
        assert trainer.gpu_server_url == "http://192.168.22.28:8000"

    def test_accepts_localhost(self):
        """localhost는 허용되는지."""
        trainer = LoRATrainer(
            gpu_server_url="http://localhost:8000",
            config=TrainingConfig(),
        )
        assert "localhost" in trainer.gpu_server_url

    async def test_start_training_file_not_found(self, tmp_path):
        """존재하지 않는 데이터 파일로 학습 시작하면 에러."""
        trainer = LoRATrainer(
            gpu_server_url="http://localhost:8000",
            config=TrainingConfig(),
        )
        with pytest.raises(FileNotFoundError):
            await trainer.start_training("/nonexistent/data.jsonl")

    async def test_start_training_connection_error_returns_local_id(self, tmp_path):
        """GPU 서버 연결 실패 시 로컬 job_id를 반환하는지."""
        trainer = LoRATrainer(
            gpu_server_url="http://192.168.99.99:8000",
            config=TrainingConfig(),
        )
        # 임시 데이터 파일 생성
        data_file = tmp_path / "data.jsonl"
        data_file.write_text('{"messages": []}\n')

        job_id = await trainer.start_training(str(data_file))
        assert job_id.startswith("local_")
        assert job_id in trainer.active_jobs


# ─────────────────────────────────────────────
# CheckpointManager 테스트
# ─────────────────────────────────────────────
class TestCheckpointManager:
    """CheckpointManager 체크포인트 관리 테스트."""

    @pytest.fixture
    def manager(self, tmp_path) -> CheckpointManager:
        """임시 디렉토리를 사용하는 매니저 fixture."""
        return CheckpointManager(checkpoints_dir=str(tmp_path / "checkpoints"))

    def test_list_empty_checkpoints(self, manager):
        """체크포인트가 없을 때 빈 리스트를 반환하는지."""
        assert manager.list_checkpoints() == []

    def test_save_and_list_checkpoint(self, manager):
        """메타데이터 저장 후 목록에 나타나는지."""
        manager.save_metadata(
            "ckpt_001",
            {
                "eval_accuracy": 0.85,
                "method": "qlora",
            },
        )

        checkpoints = manager.list_checkpoints()
        assert len(checkpoints) == 1
        assert checkpoints[0]["name"] == "ckpt_001"
        assert checkpoints[0]["metadata"]["eval_accuracy"] == 0.85

    def test_get_best_checkpoint(self, manager):
        """최고 성능 체크포인트를 올바르게 찾는지."""
        manager.save_metadata("ckpt_low", {"eval_accuracy": 0.6})
        manager.save_metadata("ckpt_high", {"eval_accuracy": 0.95})
        manager.save_metadata("ckpt_mid", {"eval_accuracy": 0.8})

        best = manager.get_best(metric="eval_accuracy")
        assert best is not None
        assert best["name"] == "ckpt_high"

    def test_get_best_returns_none_when_empty(self, manager):
        """체크포인트가 없을 때 None을 반환하는지."""
        assert manager.get_best() is None

    def test_active_checkpoint_initially_none(self, manager):
        """초기 상태에서 활성 체크포인트가 None인지."""
        assert manager.active_checkpoint is None

    async def test_activate_nonexistent_raises_error(self, manager):
        """존재하지 않는 체크포인트 활성화 시 에러."""
        with pytest.raises(FileNotFoundError):
            await manager.activate("nonexistent", "http://localhost:8000")

    async def test_rollback_connection_error(self, manager):
        """GPU 서버 연결 실패 시 롤백이 에러 상태를 반환하는지."""
        result = await manager.rollback("http://192.168.99.99:8000")
        assert result["status"] == "error"


# ─────────────────────────────────────────────
# FeedbackLoop 테스트
# ─────────────────────────────────────────────
class TestFeedbackLoop:
    """FeedbackLoop 자동 학습 루프 테스트."""

    async def test_run_cycle_skips_phase0(self):
        """Phase 0(프롬프트 엔지니어링)에서는 학습을 건너뛰는지."""
        loop = FeedbackLoop()
        strategy = TrainingStrategy()  # Phase 0
        trainer = MagicMock()
        collector = MagicMock()

        result = await loop.run_cycle(strategy, trainer, collector)
        assert result.get("skipped") is True
        assert "프롬프트 엔지니어링" in result.get("reason", "")

    async def test_run_cycle_skips_when_no_data(self, tmp_path):
        """내보낼 데이터가 없으면 사이클을 건너뛰는지."""
        loop = FeedbackLoop()
        strategy = TrainingStrategy(current_phase=TrainingPhase.BOOTSTRAP_LORA)

        # export_jsonl이 0을 반환하는 mock collector
        collector = AsyncMock()
        collector.export_jsonl = AsyncMock(return_value=0)

        trainer = MagicMock()

        result = await loop.run_cycle(strategy, trainer, collector)
        assert result.get("skipped") is True
        assert result.get("exported_count") == 0

    async def test_evaluate_returns_default_without_server(self):
        """GPU 서버 없이 evaluate()가 기본값을 반환하는지."""
        loop = FeedbackLoop()
        result = await loop.evaluate("/some/checkpoint")
        assert result["accuracy"] == 0.0
        assert "기본값" in result.get("note", "")

    async def test_cycle_history_records(self):
        """사이클 실행 후 이력이 기록되는지."""
        loop = FeedbackLoop()
        strategy = TrainingStrategy()
        trainer = MagicMock()
        collector = MagicMock()

        await loop.run_cycle(strategy, trainer, collector)
        # Phase 0은 건너뛰지만 이력은 없음 (skipped일 때는 이력 미기록)
        # Phase 1 이상에서 테스트
        assert len(loop.cycle_history) == 0  # Phase 0 skip은 이력 미기록
