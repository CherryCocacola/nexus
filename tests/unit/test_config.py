"""
core/config.py 단위 테스트.

NexusConfig의 기본값, YAML 로딩, 에어갭 검증을 테스트한다.
"""

from __future__ import annotations

import warnings

import yaml

from core.config import (
    GPUServerConfig,
    NexusConfig,
    RedisConfig,
    load_and_validate_config,
)


class TestGPUServerConfig:
    """GPU 서버 설정 테스트."""

    def test_default_url(self):
        """기본 URL이 localhost인지 확인한다."""
        config = GPUServerConfig()
        assert config.url == "http://localhost:8000"

    def test_local_url_passes_validation(self):
        """LAN 주소가 검증을 통과하는지 확인한다."""
        config = GPUServerConfig(url="http://192.168.22.28:8000")
        assert config.url == "http://192.168.22.28:8000"

    def test_external_url_warns(self):
        """외부 주소가 경고를 발생시키는지 확인한다."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            GPUServerConfig(url="http://api.example.com:8000")
            assert len(w) == 1
            assert "외부 주소" in str(w[0].message)


class TestNexusConfig:
    """메인 설정 테스트."""

    def test_default_values(self):
        """기본값이 올바르게 설정되는지 확인한다."""
        config = NexusConfig()
        assert config.air_gap_mode is True
        assert config.log_level == "INFO"
        assert config.debug is False
        assert config.model.primary_model == "qwen3.5-27b"

    def test_redis_property_accessors(self):
        """편의 property가 올바른 값을 반환하는지 확인한다."""
        config = NexusConfig(redis=RedisConfig(host="10.0.0.1", port=6380))
        assert config.redis_host == "10.0.0.1"
        assert config.redis_port == 6380

    def test_air_gap_validation_warns_on_external_url(self):
        """에어갭 모드에서 외부 GPU URL이 경고를 발생시키는지 확인한다."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            NexusConfig(
                air_gap_mode=True,
                gpu_server=GPUServerConfig(url="http://cloud.example.com:8000"),
            )
            # GPU 서버 URL 경고 + 에어갭 경고
            assert len(w) >= 1

    def test_security_defaults_are_restrictive(self):
        """보안 기본값이 제한적(fail-closed)인지 확인한다."""
        config = NexusConfig()
        assert config.security.sandbox_enabled is True
        assert len(config.security.bash_deny_patterns) > 0
        assert config.security.max_file_size_bytes == 10 * 1024 * 1024


class TestLoadAndValidateConfig:
    """설정 파일 로딩 테스트."""

    def test_load_from_yaml(self, tmp_path):
        """YAML 파일에서 설정을 로드할 수 있는지 확인한다."""
        config_file = tmp_path / "test_config.yaml"
        config_data = {
            "gpu_server": {"url": "http://192.168.1.100:8000"},
            "log_level": "DEBUG",
            "debug": True,
        }
        config_file.write_text(yaml.dump(config_data), encoding="utf-8")

        config = load_and_validate_config(str(config_file))
        assert config.gpu_server_url == "http://192.168.1.100:8000"
        assert config.log_level == "DEBUG"
        assert config.debug is True

    def test_load_without_file_uses_defaults(self):
        """설정 파일이 없으면 기본값을 사용하는지 확인한다."""
        config = load_and_validate_config("/nonexistent/path.yaml")
        assert config.gpu_server_url == "http://localhost:8000"

    def test_load_with_nested_config(self, tmp_path):
        """중첩 설정이 올바르게 로드되는지 확인한다."""
        config_file = tmp_path / "nested.yaml"
        config_data = {
            "redis": {"host": "10.0.0.5", "port": 6380},
            "model": {"primary_model": "custom-model"},
        }
        config_file.write_text(yaml.dump(config_data), encoding="utf-8")

        config = load_and_validate_config(str(config_file))
        assert config.redis.host == "10.0.0.5"
        assert config.redis.port == 6380
        assert config.model.primary_model == "custom-model"
