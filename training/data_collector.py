"""
상호작용 데이터 수집기 — 실 사용 데이터를 학습용으로 수집한다.

Phase 2(Self-Data QLoRA)에서 사용할 실 상호작용 데이터를 수집한다.
사용자와 모델 간의 대화, 도구 호출 결과, 평가 메타데이터를
JSONL 형식으로 저장한다.

보안 고려사항:
  - PII(개인식별정보)를 자동으로 마스킹한다
  - 민감한 파일 경로(.env, credentials 등)를 필터링한다
  - 수집된 데이터는 로컬에만 저장한다 (에어갭 준수)

의존성: training → core (Message, ToolResult 타입 사용)
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger("nexus.training.data_collector")


# ─────────────────────────────────────────────
# PII 마스킹 패턴
# ─────────────────────────────────────────────
_PII_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # 이메일 주소
    (re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b"), "[EMAIL]"),
    # 한국 전화번호 (010-1234-5678, 01012345678 등)
    (re.compile(r"\b\d{2,3}[-.]?\d{3,4}[-.]?\d{4}\b"), "[PHONE]"),
    # 주민등록번호 (000000-0000000)
    (re.compile(r"\b\d{6}[-]?\d{7}\b"), "[SSN]"),
    # IPv4 주소 (로컬/LAN 제외)
    (re.compile(r"\b(?!(?:10|127|172|192)\.)(\d{1,3}\.){3}\d{1,3}\b"), "[IP]"),
    # 신용카드 번호 (16자리)
    (re.compile(r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b"), "[CARD]"),
    # API 키 패턴 (sk-, pk-, key_ 등)
    (re.compile(r"\b(sk|pk|key|token|secret|api)[_-][A-Za-z0-9]{16,}\b"), "[API_KEY]"),
]

# 민감한 파일 경로 패턴 — 이 패턴이 포함된 경로는 수집하지 않는다
_SENSITIVE_PATH_PATTERNS: list[str] = [
    ".env",
    "credentials",
    "secret",
    "password",
    ".pem",
    ".key",
    "id_rsa",
    "id_ed25519",
    ".ssh/",
]


class DataCollector:
    """
    실 사용 중 상호작용 데이터를 수집한다.

    사용자 대화의 각 턴에서 메시지, 도구 호출 결과, 품질 점수를
    수집하여 Phase 2 QLoRA 학습에 사용할 데이터셋을 구축한다.
    """

    def __init__(
        self,
        storage_dir: str = "data/collected/",
        max_buffer_size: int = 1000,
    ) -> None:
        """
        Args:
            storage_dir: 수집된 데이터를 저장할 디렉토리 경로
            max_buffer_size: 메모리 버퍼 최대 크기 — 초과 시 자동 플러시
        """
        self._storage_dir = Path(storage_dir)
        self._storage_dir.mkdir(parents=True, exist_ok=True)
        self._buffer: list[dict[str, Any]] = []
        self._max_buffer_size = max_buffer_size
        # 수집 통계
        self._stats = {
            "total_collected": 0,
            "total_filtered": 0,
            "total_pii_masked": 0,
        }

    async def collect_turn(
        self,
        messages: list[dict[str, Any]],
        tool_results: list[dict[str, Any]] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        한 턴의 데이터를 수집한다.

        PII 마스킹과 민감 경로 필터링을 적용한 후 버퍼에 저장한다.
        버퍼가 가득 차면 자동으로 디스크에 플러시한다.

        Args:
            messages: 대화 메시지 목록 (user, assistant 역할)
            tool_results: 도구 실행 결과 목록 (선택)
            metadata: 추가 메타데이터 (quality_score, session_id 등)
        """
        # 민감한 경로가 포함된 도구 결과는 수집하지 않는다
        if tool_results:
            filtered_results = []
            for result in tool_results:
                if self._contains_sensitive_path(result):
                    self._stats["total_filtered"] += 1
                    continue
                filtered_results.append(result)
            tool_results = filtered_results

        # 메시지 내용에 PII 마스킹을 적용한다
        masked_messages = [self._mask_message(msg) for msg in messages]
        masked_results = [self._mask_tool_result(r) for r in tool_results] if tool_results else []

        # 수집 레코드 구성
        record: dict[str, Any] = {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.now(UTC).isoformat(),
            "messages": masked_messages,
            "tool_results": masked_results,
            "metadata": metadata or {},
        }

        self._buffer.append(record)
        self._stats["total_collected"] += 1

        # 버퍼가 가득 차면 디스크에 플러시
        if len(self._buffer) >= self._max_buffer_size:
            await self._flush_buffer()

    def _mask_pii(self, text: str) -> str:
        """
        텍스트에서 PII를 마스킹한다.

        정규 표현식으로 이메일, 전화번호, 주민등록번호 등을
        플레이스홀더로 치환한다. 원본 데이터는 복구 불가능하다.
        """
        masked = text
        for pattern, replacement in _PII_PATTERNS:
            if pattern.search(masked):
                masked = pattern.sub(replacement, masked)
                self._stats["total_pii_masked"] += 1
        return masked

    def _mask_message(self, message: dict[str, Any]) -> dict[str, Any]:
        """메시지의 텍스트 필드에 PII 마스킹을 적용한다."""
        masked = dict(message)
        if "content" in masked and isinstance(masked["content"], str):
            masked["content"] = self._mask_pii(masked["content"])
        return masked

    def _mask_tool_result(self, result: dict[str, Any]) -> dict[str, Any]:
        """도구 결과의 텍스트 필드에 PII 마스킹을 적용한다."""
        masked = dict(result)
        if "data" in masked and isinstance(masked["data"], str):
            masked["data"] = self._mask_pii(masked["data"])
        if "error_message" in masked and isinstance(masked["error_message"], str):
            masked["error_message"] = self._mask_pii(masked["error_message"])
        return masked

    def _contains_sensitive_path(self, result: dict[str, Any]) -> bool:
        """도구 결과에 민감한 파일 경로가 포함되어 있는지 확인한다."""
        result_str = json.dumps(result, default=str).lower()
        return any(pattern in result_str for pattern in _SENSITIVE_PATH_PATTERNS)

    async def _flush_buffer(self) -> None:
        """
        메모리 버퍼를 디스크(JSONL)에 저장한다.

        파일명에 타임스탬프를 포함하여 이전 파일과 겹치지 않게 한다.
        """
        if not self._buffer:
            return

        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        output_file = self._storage_dir / f"collected_{timestamp}.jsonl"

        # 버퍼 플러시는 소량 데이터이므로 동기 I/O로 충분하다
        with open(output_file, "w", encoding="utf-8") as f:  # noqa: ASYNC230
            for record in self._buffer:
                f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

        logger.info("버퍼 플러시: %d개 레코드 → %s", len(self._buffer), output_file)
        self._buffer.clear()

    async def export_jsonl(
        self,
        output_path: str,
        min_quality: float = 0.7,
    ) -> int:
        """
        수집된 데이터를 학습용 JSONL로 내보낸다.

        품질 점수(quality_score)가 min_quality 이상인 레코드만
        내보내어 학습 데이터의 품질을 보장한다.

        Args:
            output_path: 출력 JSONL 파일 경로
            min_quality: 최소 품질 점수 (0.0 ~ 1.0)

        Returns:
            내보낸 레코드 수
        """
        # 먼저 버퍼에 남은 데이터를 플러시한다
        await self._flush_buffer()

        # 수집된 모든 JSONL 파일을 읽어서 필터링 후 내보낸다
        exported_count = 0
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # 학습 데이터 내보내기 — 파일 크기가 작으므로 동기 I/O 사용
        with open(output_file, "w", encoding="utf-8") as out_f:  # noqa: ASYNC230
            # 저장 디렉토리의 모든 JSONL 파일을 순회한다
            for jsonl_file in sorted(self._storage_dir.glob("collected_*.jsonl")):
                with open(jsonl_file, encoding="utf-8") as in_f:  # noqa: ASYNC230
                    for line in in_f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            record = json.loads(line)
                        except json.JSONDecodeError:
                            logger.warning("잘못된 JSON 라인 건너뜀: %s", jsonl_file)
                            continue

                        # 품질 점수 필터링
                        quality = record.get("metadata", {}).get("quality_score", 1.0)
                        if quality < min_quality:
                            continue

                        # 학습 형식으로 변환: messages만 추출
                        training_record = {
                            "messages": record.get("messages", []),
                            "metadata": {
                                "source": "collected",
                                "quality_score": quality,
                                "original_id": record.get("id", ""),
                            },
                        }
                        out_f.write(json.dumps(training_record, ensure_ascii=False) + "\n")
                        exported_count += 1

        logger.info(
            "데이터 내보내기 완료: %d개 레코드 (min_quality=%.2f) → %s",
            exported_count,
            min_quality,
            output_path,
        )

        return exported_count

    @property
    def stats(self) -> dict[str, int]:
        """수집 통계의 복사본을 반환한다."""
        return dict(self._stats)

    @property
    def buffer_size(self) -> int:
        """현재 버퍼에 있는 레코드 수를 반환한다."""
        return len(self._buffer)
