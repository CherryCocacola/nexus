"""
상호작용 데이터 수집기 — 실제 사용 중 발생하는 대화 데이터를 학습용으로 모은다.

[이 파일이 하는 일]
Nexus를 실제로 운영하면 사용자와 모델이 매 턴 대화를 주고받는다. 이 파일은
그 대화(메시지), 도구 실행 결과, 품질 평가 메타데이터를 한 건씩 받아서
JSONL(한 줄에 JSON 하나) 파일로 차곡차곡 쌓아 둔다. 나중에 이 데이터를
Phase 2(Self-Data QLoRA) 단계에서 모델을 우리 도메인에 맞게 재학습시키는
학습 데이터셋으로 사용한다. 즉 "운영 로그 → 학습 재료" 로 바꿔 주는 다리 역할.

[핵심 구성]
  - DataCollector 클래스: 수집·마스킹·필터링·플러시·내보내기를 담당하는 본체
  - collect_turn(): 대화 한 턴을 받아 버퍼에 쌓는다 (외부에서 매 턴 호출)
  - export_jsonl(): 쌓인 데이터 중 품질 좋은 것만 골라 학습용 파일로 내보낸다
  - _PII_PATTERNS / _SENSITIVE_PATH_PATTERNS: 개인정보·민감경로 차단 규칙

[보안 고려사항]
  - PII(개인식별정보: 이메일·전화·주민번호 등)를 정규식으로 자동 마스킹한다
  - 민감한 파일 경로(.env, credentials, id_rsa 등)가 섞인 결과는 아예 버린다
  - 수집된 데이터는 오직 로컬 디스크에만 저장한다 (에어갭 준수, 외부 전송 없음)

[의존성 방향]
  training → core 방향만 허용된다. 여기서는 core의 Message/ToolResult 형태를
  dict 로 받아 다루며, core 가 training 을 역참조하는 일은 절대 없다.

작성자: 이현수 / 작성일: 2026-07-05
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
# (정규식, 치환문자열) 쌍의 목록이다. 텍스트에서 각 정규식에 걸리는 부분을
# 뒤의 플레이스홀더로 바꿔치기해 개인정보를 지운다. 위에서 아래로 순서대로 적용된다.
# 원본 값은 저장 전에 사라지므로 마스킹 후에는 원래 값 복구가 불가능하다.
_PII_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # 이메일 주소 — 아이디@도메인 형태를 [EMAIL] 로 치환
    (re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b"), "[EMAIL]"),
    # 한국 전화번호 (010-1234-5678, 01012345678 등) — 구분자 유무 모두 대응
    (re.compile(r"\b\d{2,3}[-.]?\d{3,4}[-.]?\d{4}\b"), "[PHONE]"),
    # 주민등록번호 (000000-0000000) — 하이픈이 있거나 없어도 매칭
    (re.compile(r"\b\d{6}[-]?\d{7}\b"), "[SSN]"),
    # IPv4 주소 — 단, 사내/로컬 대역(10·127·172·192)은 부정 전방탐색으로 제외한다.
    # LAN 주소는 개인정보가 아니고 디버깅에 필요하므로 일부러 남겨 둔다.
    (re.compile(r"\b(?!(?:10|127|172|192)\.)(\d{1,3}\.){3}\d{1,3}\b"), "[IP]"),
    # 신용카드 번호 (16자리) — 4자리씩 끊고 공백/하이픈 구분자 허용
    (re.compile(r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b"), "[CARD]"),
    # API 키/토큰 패턴 (sk-, pk-, key_, token_, secret_, api_ 뒤 16자 이상 영숫자)
    (re.compile(r"\b(sk|pk|key|token|secret|api)[_-][A-Za-z0-9]{16,}\b"), "[API_KEY]"),
]

# 민감한 파일 경로 패턴 — 도구 결과 문자열에 이 조각 중 하나라도 들어 있으면
# 그 결과는 통째로 수집 대상에서 제외한다 (자격증명·키 파일이 학습셋에 새는 것 방지).
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
    실제 사용 중 발생하는 상호작용 데이터를 모아 학습 데이터셋을 만드는 수집기.

    [동작 요약]
    외부(오케스트레이터 등)에서 대화 한 턴이 끝날 때마다 collect_turn() 을 호출하면,
    이 클래스는 PII 마스킹과 민감경로 필터링을 거친 레코드를 메모리 버퍼에 쌓는다.
    버퍼가 max_buffer_size 만큼 차면 자동으로 JSONL 파일에 플러시(저장)한다.
    학습을 시작할 때는 export_jsonl() 로 품질 기준을 통과한 레코드만 뽑아 내보낸다.

    [왜 버퍼를 쓰나]
    매 턴마다 파일을 여닫으면 I/O 비용이 크므로, 일정량을 메모리에 모았다가
    한 번에 기록해 성능을 확보한다. 그래서 종료 시점에는 export_jsonl() 내부의
    _flush_buffer() 로 남은 버퍼를 반드시 비워 주어야 데이터가 유실되지 않는다.
    """

    def __init__(
        self,
        storage_dir: str = "data/collected/",
        max_buffer_size: int = 1000,
    ) -> None:
        """
        수집기를 초기화하고 저장 디렉토리를 준비한다.

        Args:
            storage_dir: 수집된 데이터를 저장할 디렉토리 경로. 없으면 생성한다.
            max_buffer_size: 메모리 버퍼 최대 크기. 이 개수에 도달하면 자동 플러시.
        """
        # 저장 경로를 Path 로 감싸고, 부모 디렉토리까지 한 번에 만든다.
        # exist_ok=True 라 이미 있어도 에러가 나지 않는다(멱등).
        self._storage_dir = Path(storage_dir)
        self._storage_dir.mkdir(parents=True, exist_ok=True)
        # 아직 디스크에 쓰지 않은 레코드를 담아 두는 메모리 버퍼.
        self._buffer: list[dict[str, Any]] = []
        self._max_buffer_size = max_buffer_size
        # 수집 통계 — 얼마나 모았고/걸렀고/마스킹했는지 누적 카운터.
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
        대화 한 턴의 데이터를 받아 정제한 뒤 버퍼에 저장한다.

        처리 순서: (1) 민감 경로가 섞인 도구 결과 제거 → (2) 메시지·결과에 PII
        마스킹 적용 → (3) id·타임스탬프를 붙인 레코드로 만들어 버퍼에 추가 →
        (4) 버퍼가 가득 찼으면 디스크로 플러시. 외부에서 매 턴 호출하는 진입점이다.

        Args:
            messages: 대화 메시지 목록 (user, assistant 역할의 dict 들)
            tool_results: 도구 실행 결과 목록 (없을 수 있어 선택 인자)
            metadata: 추가 메타데이터 (quality_score, session_id 등). 내보내기
                단계의 품질 필터링에 quality_score 가 쓰인다.
        """
        # (1) 민감한 파일 경로가 포함된 도구 결과는 학습셋에서 통째로 배제한다.
        #     걸러낸 개수는 통계에 누적해 나중에 얼마나 필터됐는지 확인할 수 있게 한다.
        if tool_results:
            filtered_results = []
            for result in tool_results:
                if self._contains_sensitive_path(result):
                    self._stats["total_filtered"] += 1
                    continue  # 민감 경로 결과는 건너뛴다(수집 안 함)
                filtered_results.append(result)
            tool_results = filtered_results

        # (2) 살아남은 메시지·도구 결과의 텍스트에 PII 마스킹을 적용한다.
        #     tool_results 가 None/빈 값이면 빈 리스트로 둔다.
        masked_messages = [self._mask_message(msg) for msg in messages]
        masked_results = [self._mask_tool_result(r) for r in tool_results] if tool_results else []

        # (3) 저장용 레코드 구성 — 고유 id(uuid)와 UTC 타임스탬프를 부여한다.
        record: dict[str, Any] = {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.now(UTC).isoformat(),
            "messages": masked_messages,
            "tool_results": masked_results,
            "metadata": metadata or {},
        }

        # (3-끝) 레코드를 버퍼에 넣고 수집 카운터를 1 올린다.
        self._buffer.append(record)
        self._stats["total_collected"] += 1

        # (4) 버퍼가 정해진 최대 크기에 도달하면 즉시 디스크로 비운다.
        if len(self._buffer) >= self._max_buffer_size:
            await self._flush_buffer()

    def _mask_pii(self, text: str) -> str:
        """
        문자열 하나에서 개인정보(PII)를 찾아 플레이스홀더로 치환한다.

        _PII_PATTERNS 를 순서대로 돌면서, 매칭되는 패턴이 있으면 해당 부분을
        [EMAIL] 같은 표식으로 바꾼다. 실제로 치환이 일어난 패턴마다 통계 카운터를
        올려 얼마나 마스킹이 발생했는지 추적한다. 마스킹 후 원본 값은 복구 불가.
        """
        masked = text
        for pattern, replacement in _PII_PATTERNS:
            # search 로 먼저 존재 여부를 확인 — 걸릴 때만 sub 하고 통계를 센다.
            if pattern.search(masked):
                masked = pattern.sub(replacement, masked)
                self._stats["total_pii_masked"] += 1
        return masked

    def _mask_message(self, message: dict[str, Any]) -> dict[str, Any]:
        """메시지 dict 를 복사한 뒤 문자열 content 필드에만 PII 마스킹을 적용한다."""
        # 원본 dict 를 바꾸지 않도록 얕은 복사 후 수정한다(부작용 방지).
        masked = dict(message)
        if "content" in masked and isinstance(masked["content"], str):
            masked["content"] = self._mask_pii(masked["content"])
        return masked

    def _mask_tool_result(self, result: dict[str, Any]) -> dict[str, Any]:
        """도구 결과 dict 를 복사한 뒤 data·error_message 텍스트에 PII 마스킹을 적용한다."""
        # 여기도 얕은 복사로 원본 보호. 문자열인 필드에만 마스킹을 건다.
        masked = dict(result)
        if "data" in masked and isinstance(masked["data"], str):
            masked["data"] = self._mask_pii(masked["data"])
        if "error_message" in masked and isinstance(masked["error_message"], str):
            masked["error_message"] = self._mask_pii(masked["error_message"])
        return masked

    def _contains_sensitive_path(self, result: dict[str, Any]) -> bool:
        """도구 결과 전체를 JSON 문자열로 펼쳐 민감 경로 조각이 들어있는지 검사한다."""
        # dict 를 통째로 JSON 문자열화하고 소문자로 낮춰 대소문자 무시 비교를 한다.
        # default=str 은 직렬화 불가한 값(예: 객체)을 문자열로 안전 변환하기 위함.
        result_str = json.dumps(result, default=str).lower()
        # 민감 패턴 중 하나라도 포함되면 True(=수집 제외 대상).
        return any(pattern in result_str for pattern in _SENSITIVE_PATH_PATTERNS)

    async def _flush_buffer(self) -> None:
        """
        메모리 버퍼에 쌓인 레코드를 JSONL 파일로 저장하고 버퍼를 비운다.

        파일명에 초 단위 타임스탬프(collected_YYYYMMDD_HHMMSS.jsonl)를 넣어
        이전 플러시 결과와 파일이 겹치지 않게 한다. 버퍼가 비어 있으면 아무 것도
        하지 않고 즉시 반환한다.
        """
        # 비어 있으면 파일을 만들 필요가 없으므로 조기 반환.
        if not self._buffer:
            return

        # 타임스탬프로 겹치지 않는 출력 파일명을 만든다.
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        output_file = self._storage_dir / f"collected_{timestamp}.jsonl"

        # 버퍼 플러시는 소량 데이터이므로 동기 I/O로 충분하다.
        # async 함수 안의 동기 open 은 의도된 것이라 ASYNC230 경고를 억제한다.
        with open(output_file, "w", encoding="utf-8") as f:  # noqa: ASYNC230
            # 레코드 하나를 JSON 한 줄로 직렬화해 기록(JSONL 형식).
            # ensure_ascii=False 로 한글이 깨지지 않게, default=str 로 직렬화 예외 방지.
            for record in self._buffer:
                f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

        logger.info("버퍼 플러시: %d개 레코드 → %s", len(self._buffer), output_file)
        # 디스크에 안전히 옮겼으니 메모리 버퍼를 비운다.
        self._buffer.clear()

    async def export_jsonl(
        self,
        output_path: str,
        min_quality: float = 0.7,
    ) -> int:
        """
        지금까지 수집한 데이터 중 품질 기준을 통과한 것만 학습용 JSONL로 내보낸다.

        저장 디렉토리에 쌓인 collected_*.jsonl 파일들을 전부 훑어, 각 레코드의
        quality_score 가 min_quality 이상인 것만 골라 학습 표준 형식으로 다시
        기록한다. 이렇게 하면 저품질 대화가 학습셋에 섞이는 것을 막을 수 있다.

        Args:
            output_path: 내보낼 결과 JSONL 파일 경로. 부모 디렉토리는 자동 생성.
            min_quality: 최소 품질 점수 (0.0 ~ 1.0). 이 값 미만 레코드는 제외.

        Returns:
            실제로 내보낸(기준 통과) 레코드 수.
        """
        # 내보내기 전에 버퍼에 남아 있던 미저장 레코드를 먼저 파일로 비운다.
        # (이 호출이 없으면 마지막에 모은 데이터가 빠질 수 있다.)
        await self._flush_buffer()

        # 내보낸 레코드 수를 셀 카운터와 출력 경로를 준비한다.
        exported_count = 0
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # 학습 데이터 내보내기 — 파일 크기가 작으므로 동기 I/O 사용(ASYNC230 억제).
        with open(output_file, "w", encoding="utf-8") as out_f:  # noqa: ASYNC230
            # 저장 디렉토리의 collected_*.jsonl 파일을 이름순(=시간순)으로 순회한다.
            for jsonl_file in sorted(self._storage_dir.glob("collected_*.jsonl")):
                with open(jsonl_file, encoding="utf-8") as in_f:  # noqa: ASYNC230
                    # 파일을 한 줄(=레코드 하나)씩 읽는다.
                    for line in in_f:
                        line = line.strip()
                        if not line:
                            continue  # 빈 줄은 건너뛴다
                        try:
                            record = json.loads(line)
                        except json.JSONDecodeError:
                            # 깨진 줄은 전체를 멈추지 않고 경고만 남긴 뒤 건너뛴다.
                            logger.warning("잘못된 JSON 라인 건너뜀: %s", jsonl_file)
                            continue

                        # 품질 점수 필터링 — 메타데이터에 없으면 1.0(최상)으로 간주.
                        quality = record.get("metadata", {}).get("quality_score", 1.0)
                        if quality < min_quality:
                            continue  # 기준 미달 레코드는 내보내지 않는다

                        # 학습 형식으로 변환 — 대화(messages)만 남기고 출처·품질·
                        # 원본 id 를 메타데이터로 재구성한다. 마스킹은 수집 때 이미 완료.
                        training_record = {
                            "messages": record.get("messages", []),
                            "metadata": {
                                "source": "collected",
                                "quality_score": quality,
                                "original_id": record.get("id", ""),
                            },
                        }
                        # 변환된 레코드를 JSONL 한 줄로 기록하고 카운터를 올린다.
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
        """수집 통계(수집·필터·마스킹 누적 수)의 복사본을 반환한다.

        복사본을 돌려주므로 외부에서 이 값을 바꿔도 내부 통계에는 영향이 없다.
        """
        return dict(self._stats)

    @property
    def buffer_size(self) -> int:
        """아직 디스크에 쓰지 않고 메모리 버퍼에 남아 있는 레코드 수를 반환한다."""
        return len(self._buffer)
