"""
감사 로그 — AuditLogger.

Ch.8.9 사양서 기반. 모든 권한 결정을 JSONL 형식으로 기록한다.

특징:
  - JSONL 형식 (한 줄에 하나의 JSON 객체)
  - 10MB 로테이션 (설정 가능)
  - 최근 로그 조회 기능
  - 스레드 안전 (asyncio lock 사용)

왜 JSONL인가:
  - 한 줄씩 append하므로 쓰기 충돌이 적다
  - grep/jq로 쉽게 분석할 수 있다
  - 구조화된 로그이므로 자동 파싱이 가능하다
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections import deque
from pathlib import Path

from core.permission.types import PermissionAuditEntry

logger = logging.getLogger("nexus.security")


class AuditLogger:
    """
    JSONL 형식 감사 로그.

    권한 결정, 도구 실행, 보안 이벤트를 기록한다.
    파일 크기가 max_size_bytes를 초과하면 로테이션한다.
    """

    def __init__(
        self,
        log_path: str = "logs/audit.log",
        max_size_bytes: int = 10 * 1024 * 1024,  # 10MB
        rotation_count: int = 5,
    ) -> None:
        """
        AuditLogger를 초기화한다.

        Args:
            log_path: 로그 파일 경로
            max_size_bytes: 로테이션 기준 파일 크기 (기본 10MB)
            rotation_count: 보관할 백업 파일 수 (기본 5개)
        """
        self._log_path = Path(log_path)
        self._max_size_bytes = max_size_bytes
        self._rotation_count = rotation_count
        # 비동기 쓰기를 위한 락 (동시 쓰기 방지)
        self._lock = asyncio.Lock()
        # 최근 로그를 메모리에 보관 (빠른 조회용)
        self._recent: deque[PermissionAuditEntry] = deque(maxlen=500)
        # 초기화 시 로그 디렉토리 생성
        self._ensure_directory()

    def _ensure_directory(self) -> None:
        """로그 파일의 부모 디렉토리가 없으면 생성한다."""
        self._log_path.parent.mkdir(parents=True, exist_ok=True)

    def log_decision(self, entry: PermissionAuditEntry) -> None:
        """
        권한 결정을 JSONL로 기록한다 (동기 버전).

        파일에 한 줄 추가하고, 메모리 캐시에도 저장한다.
        파일 크기 초과 시 로테이션을 수행한다.

        Args:
            entry: 감사 로그 엔트리
        """
        # 메모리 캐시에 저장
        self._recent.append(entry)

        # JSONL로 직렬화
        line = entry.model_dump_json() + "\n"

        try:
            # 로테이션 필요 여부 확인
            if self._log_path.exists():
                size = self._log_path.stat().st_size
                if size + len(line.encode("utf-8")) > self._max_size_bytes:
                    self._rotate()

            # 파일에 추가
            with open(self._log_path, "a", encoding="utf-8") as f:
                f.write(line)

        except OSError as e:
            # 파일 쓰기 실패해도 프로세스를 중단하지 않는다
            logger.error("감사 로그 쓰기 실패: %s", e)

    async def async_log_decision(self, entry: PermissionAuditEntry) -> None:
        """
        권한 결정을 JSONL로 기록한다 (비동기 버전).

        asyncio lock으로 동시 쓰기를 방지한다.
        """
        async with self._lock:
            self.log_decision(entry)

    def get_recent(self, n: int = 50) -> list[PermissionAuditEntry]:
        """
        최근 n개 감사 로그를 반환한다.

        메모리 캐시에서 반환하므로 파일 I/O가 없다.

        Args:
            n: 반환할 최대 엔트리 수

        Returns:
            최근 감사 로그 목록 (최신이 마지막)
        """
        items = list(self._recent)
        return items[-n:]

    def get_all_from_file(self) -> list[PermissionAuditEntry]:
        """
        로그 파일에서 모든 엔트리를 읽어 반환한다.

        파일이 크면 느릴 수 있으므로 주의한다.

        Returns:
            모든 감사 로그 목록
        """
        if not self._log_path.exists():
            return []

        entries: list[PermissionAuditEntry] = []
        try:
            with open(self._log_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        entries.append(PermissionAuditEntry(**data))
                    except (json.JSONDecodeError, ValueError) as e:
                        logger.warning("감사 로그 파싱 실패: %s", e)
                        continue
        except OSError as e:
            logger.error("감사 로그 읽기 실패: %s", e)

        return entries

    def _rotate(self) -> None:
        """
        로그 파일을 로테이션한다.

        audit.log → audit.log.1 → audit.log.2 → ... → audit.log.N (삭제)
        """
        try:
            # 가장 오래된 백업 삭제
            oldest = Path(f"{self._log_path}.{self._rotation_count}")
            if oldest.exists():
                oldest.unlink()

            # 기존 백업을 한 칸씩 뒤로 이동
            for i in range(self._rotation_count - 1, 0, -1):
                src = Path(f"{self._log_path}.{i}")
                dst = Path(f"{self._log_path}.{i + 1}")
                if src.exists():
                    src.rename(dst)

            # 현재 로그를 .1로 이동
            if self._log_path.exists():
                self._log_path.rename(Path(f"{self._log_path}.1"))

            logger.info("감사 로그 로테이션 완료: %s", self._log_path)
        except OSError as e:
            logger.error("감사 로그 로테이션 실패: %s", e)

    def clear(self) -> None:
        """메모리 캐시를 초기화한다 (테스트용)."""
        self._recent.clear()
