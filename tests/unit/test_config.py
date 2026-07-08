"""
core/config.py 단위 테스트.

NexusConfig의 기본값, YAML 로딩, 에어갭 검증을 테스트한다.
"""

from __future__ import annotations

import warnings

import pytest
import yaml

from core.config import (
    CitationConfig,
    GPUServerConfig,
    KnowledgeRagConfig,
    NexusConfig,
    RedisConfig,
    RoutingConfig,
    RoutingProfile,
    load_and_validate_config,
)


class TestGPUServerConfig:
    """GPU 서버(Machine B) 접속 설정의 기본값과 에어갭 검증을 다룬다."""

    def test_default_url(self):
        """설정을 비워두면 안전한 로컬 주소(localhost)로 시작하는지 확인한다.

        에어갭 환경이므로 기본값이 외부로 새는 일이 없어야 한다.
        """
        config = GPUServerConfig()
        assert config.url == "http://localhost:8000"

    def test_local_url_passes_validation(self):
        """LAN 사설 대역(192.168.x.x)은 경고 없이 그대로 통과해야 한다."""
        config = GPUServerConfig(url="http://192.168.21.112:8000")
        assert config.url == "http://192.168.21.112:8000"

    def test_external_url_warns(self):
        """공인 도메인을 넣으면 에어갭 위반 신호로 경고가 한 번 떠야 한다.

        catch_warnings(record=True)로 경고를 가로채 잡고, simplefilter("always")로
        중복 억제를 풀어 매번 기록되게 한 뒤, 경고 문구에 '외부 주소'가 담겼는지 본다.
        값을 막지는 않고 경고만 띄우는 정책이므로 예외가 아니라 warning을 검증한다.
        """
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            GPUServerConfig(url="http://api.example.com:8000")
            assert len(w) == 1
            assert "외부 주소" in str(w[0].message)


class TestNexusConfig:
    """최상위 NexusConfig의 기본값·편의 property·에어갭/보안 정책을 묶어 검증한다."""

    def test_default_values(self):
        """아무 인자 없이 만든 설정이 운영 기본값과 일치하는지 확인한다.

        에어갭 모드 ON, 로그 INFO, 디버그 OFF, 기본 모델 qwen3.5-27b가
        '안전하고 운영에 적합한' 출발점이다. 누가 기본값을 바꾸면 이 테스트가 잡는다.
        """
        config = NexusConfig()
        assert config.air_gap_mode is True
        assert config.log_level == "INFO"
        assert config.debug is False
        assert config.model.primary_model == "qwen3.5-27b"

    def test_redis_property_accessors(self):
        """config.redis_host / redis_port 단축 property가 중첩 설정을 그대로 비추는지 본다.

        호출부가 config.redis.host 대신 짧은 별칭을 써도 같은 값을 얻어야 한다.
        """
        config = NexusConfig(redis=RedisConfig(host="10.0.0.1", port=6380))
        assert config.redis_host == "10.0.0.1"
        assert config.redis_port == 6380

    def test_air_gap_validation_warns_on_external_url(self):
        """에어갭 ON 상태에서 공인 GPU URL을 주면 경고가 적어도 한 번 떠야 한다.

        GPUServerConfig 자체의 외부 주소 경고와 NexusConfig의 에어갭 경고가
        겹쳐 발생할 수 있으므로 정확한 개수가 아니라 '>= 1'로 느슨하게 확인한다.
        """
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            NexusConfig(
                air_gap_mode=True,
                gpu_server=GPUServerConfig(url="http://cloud.example.com:8000"),
            )
            # GPU 서버 URL 경고 + 에어갭 경고
            assert len(w) >= 1

    def test_security_defaults_are_restrictive(self):
        """보안 기본값이 fail-closed(가장 제한적)인지 확인한다.

        샌드박스는 켜진 채로, bash 금지 패턴 목록은 비어 있지 않게, 파일 크기
        상한은 10MB로 잡혀 있어야 한다. 명시적 완화 없이 느슨해지면 안 된다(P6).
        """
        config = NexusConfig()
        assert config.security.sandbox_enabled is True
        assert len(config.security.bash_deny_patterns) > 0
        assert config.security.max_file_size_bytes == 10 * 1024 * 1024


class TestLoadAndValidateConfig:
    """load_and_validate_config()의 YAML 로딩·폴백·중첩 병합 동작을 검증한다.

    실제 파일을 다루지만 pytest의 tmp_path fixture로 임시 디렉토리에만 쓰므로
    프로젝트 설정 파일은 절대 건드리지 않는다.
    """

    def test_load_from_yaml(self, tmp_path):
        """YAML로 적어둔 값(URL/로그레벨/디버그)이 그대로 설정 객체에 실리는지 확인한다."""
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
        """존재하지 않는 경로를 줘도 예외 없이 기본값으로 폴백하는지 확인한다.

        설정 파일을 깜빡해도 서버가 죽지 않고 안전한 기본값으로 떠야 한다.
        """
        config = load_and_validate_config("/nonexistent/path.yaml")
        assert config.gpu_server_url == "http://localhost:8000"

    def test_load_with_nested_config(self, tmp_path):
        """redis/model 같은 하위 섹션(중첩 dict)이 각 하위 설정 객체로 잘 풀리는지 확인한다."""
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


# ─────────────────────────────────────────────
# RoutingProfile / RoutingConfig 샘플링 파라미터 (degeneration 버그 수정, 2026-06-18)
# ─────────────────────────────────────────────
# vLLM payload에 누락됐던 4개 샘플링 파라미터의 클래스별 기본값을 검증한다.
# 이 값들이 잘못되면 동일 문장 무한 반복(degeneration) 또는 도구 호출 파싱
# 실패가 재발하므로, 회귀 가드로 명시적으로 고정한다.
class TestRoutingProfileSamplingDefaults:
    """RoutingProfile 자체의 비활성 기본값을 검증한다(하위 호환의 근거)."""

    def test_routing_profile_sampling_defaults_are_inactive(self):
        """RoutingProfile()의 4개 샘플링 파라미터 기본값은 모두 '비활성'값이다.

        top_p=1.0 / repetition_penalty=1.0 / frequency=0.0 / presence=0.0은
        vLLM이 사실상 무시하는 중립값이라 신규 필드를 생략해도 동작이 안 바뀐다.
        """
        profile = RoutingProfile(model="some-model")
        assert profile.top_p == pytest.approx(1.0)
        assert profile.repetition_penalty == pytest.approx(1.0)
        assert profile.frequency_penalty == pytest.approx(0.0)
        assert profile.presence_penalty == pytest.approx(0.0)


class TestRoutingConfigClassDefaults:
    """KNOWLEDGE/CHAT/TOOL 프로필별 샘플링 파라미터 운영 기본값을 검증한다."""

    def test_knowledge_mode_sampling_values(self):
        """KNOWLEDGE 프로필 — 사실 위주라 반복 페널티를 가장 강하게 건다.

        지식 답변은 같은 문장을 되풀이하는 degeneration이 특히 잘 보이므로
        repetition_penalty=1.15로 가장 세게 억제한다. 네 값을 한꺼번에 고정한다.
        """
        cfg = RoutingConfig()
        assert cfg.knowledge_mode.top_p == pytest.approx(0.95)
        assert cfg.knowledge_mode.repetition_penalty == pytest.approx(1.15)
        assert cfg.knowledge_mode.frequency_penalty == pytest.approx(0.3)
        assert cfg.knowledge_mode.presence_penalty == pytest.approx(0.0)

    def test_chat_mode_sampling_values(self):
        """CHAT 프로필 — 잡담 다양성을 위해 top_p=0.9, 중간 반복 페널티(1.1)."""
        cfg = RoutingConfig()
        assert cfg.chat_mode.top_p == pytest.approx(0.9)
        assert cfg.chat_mode.repetition_penalty == pytest.approx(1.1)
        assert cfg.chat_mode.frequency_penalty == pytest.approx(0.2)
        assert cfg.chat_mode.presence_penalty == pytest.approx(0.0)

    def test_tool_mode_sampling_values(self):
        """TOOL 프로필 — repetition_penalty=1.0(비활성)이 포맷 보존의 핵심이다."""
        cfg = RoutingConfig()
        assert cfg.tool_mode.top_p == pytest.approx(0.95)
        assert cfg.tool_mode.repetition_penalty == pytest.approx(1.0)
        assert cfg.tool_mode.frequency_penalty == pytest.approx(0.0)
        assert cfg.tool_mode.presence_penalty == pytest.approx(0.0)

    def test_knowledge_repetition_penalty_suppression_enabled(self):
        """회귀 가드 — KNOWLEDGE는 repetition_penalty > 1.0(반복 억제 켜짐)."""
        cfg = RoutingConfig()
        assert cfg.knowledge_mode.repetition_penalty > 1.0

    def test_tool_repetition_penalty_format_preserved(self):
        """회귀 가드 — TOOL은 repetition_penalty == 1.0(tool_call 포맷 보존).

        왜: tool_call JSON/XML은 같은 키/괄호를 반복해야 하는데 반복 페널티를
        걸면 필수 토큰이 왜곡되어 파싱 실패를 유발한다.
        """
        cfg = RoutingConfig()
        assert cfg.tool_mode.repetition_penalty == pytest.approx(1.0)


# ─────────────────────────────────────────────
# KnowledgeRagConfig — 지식 RAG 유사도 게이팅 설정 (2026-06-18)
# ─────────────────────────────────────────────
# 게이팅 임계값의 단일 소스는 yaml(knowledge_rag)이며, 이 클래스 기본값은
# yaml 누락 시 폴백이다. 운영 게이팅 기본값(0.84/0.03/0.75/5)을 회귀 가드로
# 고정하고, yaml 로드 시 값이 정상 반영되는지 검증한다.
class TestKnowledgeRagConfigDefaults:
    """KnowledgeRagConfig 기본값과 NexusConfig 연동을 검증한다."""

    def test_knowledge_rag_config_default_values(self):
        """KnowledgeRagConfig 4개 필드의 운영 기본값을 고정한다."""
        cfg = KnowledgeRagConfig()
        assert cfg.top_k == 5
        assert cfg.min_similarity == pytest.approx(0.75)
        assert cfg.abs_threshold == pytest.approx(0.84)
        assert cfg.relevance_margin == pytest.approx(0.03)

    def test_nexus_config_knowledge_rag_default_factory(self):
        """NexusConfig()가 knowledge_rag를 기본 게이팅 값으로 채우는지 확인한다."""
        config = NexusConfig()
        assert config.knowledge_rag.abs_threshold == pytest.approx(0.84)
        assert config.knowledge_rag.relevance_margin == pytest.approx(0.03)
        assert config.knowledge_rag.min_similarity == pytest.approx(0.75)
        assert config.knowledge_rag.top_k == 5


class TestKnowledgeRagConfigYamlLoad:
    """yaml에서 knowledge_rag 게이팅 값이 반영되는지 검증한다."""

    def test_load_knowledge_rag_from_yaml_overrides_defaults(self, tmp_path):
        """yaml에 명시한 게이팅 값이 기본값을 덮어쓰는지 확인한다."""
        config_file = tmp_path / "knowledge_rag.yaml"
        config_data = {
            "knowledge_rag": {
                "top_k": 8,
                "min_similarity": 0.7,
                "abs_threshold": 0.9,
                "relevance_margin": 0.05,
            },
        }
        config_file.write_text(yaml.dump(config_data), encoding="utf-8")

        config = load_and_validate_config(str(config_file))
        assert config.knowledge_rag.top_k == 8
        assert config.knowledge_rag.min_similarity == pytest.approx(0.7)
        assert config.knowledge_rag.abs_threshold == pytest.approx(0.9)
        assert config.knowledge_rag.relevance_margin == pytest.approx(0.05)

    def test_load_without_knowledge_rag_uses_gating_defaults(self, tmp_path):
        """yaml에 knowledge_rag가 없으면 게이팅 기본값(0.84/0.03)으로 폴백한다."""
        config_file = tmp_path / "no_knowledge_rag.yaml"
        config_file.write_text(yaml.dump({"log_level": "DEBUG"}), encoding="utf-8")

        config = load_and_validate_config(str(config_file))
        assert config.knowledge_rag.abs_threshold == pytest.approx(0.84)
        assert config.knowledge_rag.relevance_margin == pytest.approx(0.03)


# ─────────────────────────────────────────────
# CitationConfig — 지식 RAG 출처 인용 설정 (Point 4-2, 2026-07-08)
# ─────────────────────────────────────────────
class TestCitationConfigDefaults:
    """CitationConfig 기본값(무회귀: enabled=False)과 KnowledgeRagConfig 연동을 검증한다."""

    def test_citation_config_defaults_off(self):
        """출처 인용은 기본 OFF여야 한다(무회귀). 나머지 기본값도 고정한다."""
        cfg = CitationConfig()
        assert cfg.enabled is False
        assert cfg.label == "출처"
        assert cfg.max_sources == 5
        assert cfg.expose_in_response is True
        assert cfg.strip_invalid_labels is True

    def test_knowledge_rag_config_citation_default_factory(self):
        """KnowledgeRagConfig()가 citation을 기본 OFF로 채우는지 확인한다."""
        cfg = KnowledgeRagConfig()
        assert cfg.citation.enabled is False
        assert cfg.citation.label == "출처"

    def test_nexus_config_citation_default_off(self):
        """NexusConfig()의 knowledge_rag.citation도 기본 OFF다(전역 무회귀)."""
        config = NexusConfig()
        assert config.knowledge_rag.citation.enabled is False


class TestCitationConfigYamlLoad:
    """yaml에서 knowledge_rag.citation 값이 반영되는지 검증한다."""

    def test_citation_yaml_load_overrides_defaults(self, tmp_path):
        """yaml에 명시한 citation 값이 기본값을 덮어쓴다."""
        config_file = tmp_path / "citation.yaml"
        config_data = {
            "knowledge_rag": {
                "citation": {
                    "enabled": True,
                    "label": "Source",
                    "max_sources": 3,
                    "expose_in_response": False,
                    "strip_invalid_labels": False,
                },
            },
        }
        config_file.write_text(yaml.dump(config_data), encoding="utf-8")

        config = load_and_validate_config(str(config_file))
        assert config.knowledge_rag.citation.enabled is True
        assert config.knowledge_rag.citation.label == "Source"
        assert config.knowledge_rag.citation.max_sources == 3
        assert config.knowledge_rag.citation.expose_in_response is False
        assert config.knowledge_rag.citation.strip_invalid_labels is False

    def test_citation_absent_uses_off_default(self, tmp_path):
        """yaml에 citation이 없으면 기본 OFF로 폴백한다(무회귀)."""
        config_file = tmp_path / "no_citation.yaml"
        config_file.write_text(
            yaml.dump({"knowledge_rag": {"top_k": 7}}), encoding="utf-8"
        )

        config = load_and_validate_config(str(config_file))
        assert config.knowledge_rag.citation.enabled is False
