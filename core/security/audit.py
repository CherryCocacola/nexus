"""
감사 로그(Audit Log) 모듈 — AuditLogger.

이 파일은 Nexus 권한 시스템(5계층 PermissionPipeline)이 내린 모든 결정을
파일에 영구 기록하는 "감사 추적(audit trail)" 기능을 담당한다.
누가 어떤 도구를 언제 호출했고, 허용/거부/확인 중 무엇으로 판정됐는지를
남겨두면 나중에 보안 사고 조사나 동작 디버깅에 쓸 수 있다.

핵심 클래스:
  - AuditLogger: 감사 엔트리를 JSONL 파일에 기록하고, 최근 기록을
    메모리에서 빠르게 조회하며, 파일이 커지면 로테이션(교체)한다.

주요 특징:
  - JSONL 형식 — 한 줄에 JSON 객체 하나. append 방식이라 쓰기 충돌이 적다.
  - 크기 기반 로테이션 — 기본 10MB를 넘으면 백업 파일로 밀어낸다(설정 가능).
  - 최근 로그 조회 — 최근 500건을 메모리(deque)에 들고 있어 파일 I/O 없이 조회.
  - 동시 쓰기 보호 — 비동기 경로는 asyncio.Lock으로 직렬화한다.

의존 관계:
  - core.permission.types.PermissionAuditEntry(Pydantic 모델)를 입출력 단위로 쓴다.
  - 권한 파이프라인(core/permission)이 이 로거를 호출하는 소비자 쪽이다.

왜 JSONL인가:
  - 한 줄씩 append하므로 쓰기 충돌이 적고, 중간이 깨져도 다른 줄은 살아있다.
  - grep/jq 같은 표준 도구로 즉시 분석할 수 있다.
  - 구조화된 로그라서 프로그램으로 자동 파싱하기 쉽다.

작성자: 이현수 / 작성일: 2026-07-05
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
    JSONL 형식으로 감사 기록을 남기는 로거.

    권한 결정, 도구 실행, 보안 이벤트 같은 "일어난 일"을 한 줄씩 파일에 쌓는다.
    파일 크기가 max_size_bytes를 넘으면 자동으로 로테이션(백업 파일로 교체)한다.

    두 가지 쓰기 경로를 제공한다:
      - log_decision(): 동기 버전. 일반 호출에서 바로 파일에 쓴다.
      - async_log_decision(): 비동기 버전. asyncio.Lock으로 감싸 동시 쓰기를 막는다.

    조회 경로는 두 가지다:
      - get_recent(): 메모리 캐시(deque)에서 최근 N건을 빠르게 반환(파일 I/O 없음).
      - get_all_from_file(): 파일 전체를 파싱해 반환(느릴 수 있어 주의).
    """

    def __init__(
        self,
        log_path: str = "logs/audit.log",
        max_size_bytes: int = 10 * 1024 * 1024,  # 10MB
        rotation_count: int = 5,
    ) -> None:
        """
        AuditLogger를 초기화한다.

        내부 상태를 세팅하고, 로그를 쓸 디렉토리가 없으면 미리 만들어 둔다.
        (파일 자체는 첫 기록 시점에 append 모드로 열리며 자동 생성된다.)

        Args:
            log_path: 로그 파일 경로. 부모 디렉토리는 여기서 자동 생성된다.
            max_size_bytes: 로테이션 기준 파일 크기. 기본 10MB.
            rotation_count: 보관할 백업 파일 개수. 기본 5개.
        """
        # 문자열 경로를 Path 객체로 감싸 이후 파일 조작을 편하게 한다.
        self._log_path = Path(log_path)
        self._max_size_bytes = max_size_bytes
        self._rotation_count = rotation_count
        # 비동기 쓰기 경로에서 동시 접근을 직렬화하기 위한 락.
        # (여러 코루틴이 같은 파일에 동시에 쓰면 줄이 섞일 수 있어 방지.)
        self._lock = asyncio.Lock()
        # 최근 기록을 메모리에 들고 있는 링버퍼. maxlen을 넘으면
        # 가장 오래된 것부터 자동으로 밀려난다(빠른 최근 조회용).
        self._recent: deque[PermissionAuditEntry] = deque(maxlen=500)
        # 로그 파일을 쓸 부모 디렉토리를 지금 미리 만들어 둔다.
        self._ensure_directory()

    def _ensure_directory(self) -> None:
        """
        로그 파일의 부모 디렉토리가 없으면 생성한다.

        parents=True로 중간 경로까지 한 번에 만들고, exist_ok=True로
        이미 있으면 조용히 넘어간다(에러 없이 멱등하게 동작).
        """
        self._log_path.parent.mkdir(parents=True, exist_ok=True)

    def log_decision(self, entry: PermissionAuditEntry) -> None:
        """
        권한 결정 한 건을 JSONL 파일에 기록한다(동기 버전).

        흐름:
          1) 먼저 메모리 캐시(deque)에 넣어 최근 조회에 바로 반영한다.
          2) 엔트리를 JSON 한 줄로 직렬화한다(끝에 개행 추가).
          3) 이 줄을 더하면 파일이 한도를 넘는지 확인해, 넘으면 로테이션한다.
          4) append 모드로 파일에 한 줄 쓴다.

        파일 쓰기가 실패하더라도 예외를 밖으로 던지지 않는다. 감사 로그는
        부가 기능이므로, 여기서 프로세스를 죽이면 오히려 본 기능이 멈춘다.

        Args:
            entry: 기록할 감사 엔트리(PermissionAuditEntry).
        """
        # 1) 메모리 캐시에 먼저 저장 — 파일 I/O와 무관하게 최근 조회에 즉시 반영.
        self._recent.append(entry)

        # 2) Pydantic 모델을 JSON 문자열로 직렬화하고 줄 끝에 개행을 붙인다.
        #    (JSONL은 "한 줄 = 한 객체"가 규칙이므로 개행이 구분자다.)
        line = entry.model_dump_json() + "\n"

        try:
            # 3) 이번 줄을 쓰면 파일이 최대 크기를 초과하는지 미리 계산한다.
            #    len(...encode("utf-8"))로 실제 바이트 길이를 재서 비교한다.
            if self._log_path.exists():
                size = self._log_path.stat().st_size
                if size + len(line.encode("utf-8")) > self._max_size_bytes:
                    # 초과하면 기존 파일들을 백업으로 밀어내고 새 파일로 시작.
                    self._rotate()

            # 4) append("a") 모드로 열어 한 줄을 덧붙인다. with 블록이 닫으며 flush.
            with open(self._log_path, "a", encoding="utf-8") as f:
                f.write(line)

        except OSError as e:
            # 파일 쓰기가 실패해도(권한/디스크 등) 예외를 삼키고 로그만 남긴다.
            # 감사 기록 실패가 본 프로세스를 중단시키지 않도록 하기 위함.
            logger.error("감사 로그 쓰기 실패: %s", e)

    async def async_log_decision(self, entry: PermissionAuditEntry) -> None:
        """
        권한 결정 한 건을 JSONL 파일에 기록한다(비동기 버전).

        비동기 컨텍스트(query loop, tool executor 등)에서 여러 코루틴이
        동시에 로그를 남기려 할 때, asyncio.Lock으로 한 번에 하나씩만
        쓰도록 직렬화한다. 실제 쓰기 자체는 동기 log_decision()에 위임한다.

        Args:
            entry: 기록할 감사 엔트리(PermissionAuditEntry).
        """
        # 락을 잡고 있는 동안에는 다른 코루틴의 쓰기가 대기하므로 줄 섞임이 없다.
        async with self._lock:
            self.log_decision(entry)

    def get_recent(self, n: int = 50) -> list[PermissionAuditEntry]:
        """
        최근 n개의 감사 엔트리를 반환한다.

        디스크가 아니라 메모리 캐시(deque)에서 꺼내므로 파일 I/O가 없어 빠르다.
        단, 캐시는 최대 500건만 유지하므로 그보다 오래된 기록은 여기 없다.
        (전체가 필요하면 get_all_from_file()을 쓴다.)

        Args:
            n: 반환할 최대 엔트리 수.

        Returns:
            최근 감사 엔트리 목록. 오래된 것이 앞, 최신이 마지막 순서.
        """
        # deque를 리스트로 복사한 뒤 뒤에서 n개만 잘라 반환한다.
        # (음수 슬라이스 [-n:]는 n이 길이보다 커도 안전하게 전체를 준다.)
        items = list(self._recent)
        return items[-n:]

    def get_all_from_file(self) -> list[PermissionAuditEntry]:
        """
        현재 로그 파일(로테이션된 백업 제외)의 모든 엔트리를 읽어 반환한다.

        파일 전체를 한 줄씩 파싱하므로, 로그가 크면 느릴 수 있어 주의한다.
        메모리 캐시(get_recent)로 부족해 과거 기록까지 훑어야 할 때만 쓴다.

        Returns:
            파싱에 성공한 모든 감사 엔트리 목록. 파일이 없으면 빈 목록.
        """
        # 파일이 아직 없으면(기록 전) 빈 목록으로 조용히 반환.
        if not self._log_path.exists():
            return []

        entries: list[PermissionAuditEntry] = []
        try:
            # 파일을 한 줄씩 순회한다(전체를 메모리에 올리지 않아 그나마 낫다).
            with open(self._log_path, encoding="utf-8") as f:
                for line in f:
                    # 앞뒤 공백/개행 제거 후 빈 줄은 건너뛴다.
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        # JSON → dict → Pydantic 모델 순으로 복원한다.
                        data = json.loads(line)
                        entries.append(PermissionAuditEntry(**data))
                    except (json.JSONDecodeError, ValueError) as e:
                        # 깨진 줄 하나 때문에 전체 읽기를 포기하지 않는다.
                        # 해당 줄만 경고로 남기고 다음 줄로 넘어간다.
                        logger.warning("감사 로그 파싱 실패: %s", e)
                        continue
        except OSError as e:
            # 파일 열기/읽기 자체가 실패하면 에러만 남기고 지금까지 모은 걸 반환.
            logger.error("감사 로그 읽기 실패: %s", e)

        return entries

    def _rotate(self) -> None:
        """
        로그 파일이 한도를 넘겼을 때 백업 파일들을 한 칸씩 밀어낸다.

        번호가 클수록 오래된 백업이다. 세대 이동은 다음과 같다:
          audit.log → audit.log.1 → audit.log.2 → ... → audit.log.N (삭제)

        가장 큰 번호(N)를 먼저 지우고, 뒤에서 앞으로 순서대로 옮겨야
        기존 파일을 덮어써 잃는 일이 없다(그래서 range를 역순으로 돈다).
        로테이션이 끝나면 현재 로그 자리는 비고, 다음 쓰기에서 새로 만들어진다.
        """
        try:
            # 1) 가장 오래된 백업(.N)을 먼저 삭제해 자리를 비운다.
            oldest = Path(f"{self._log_path}.{self._rotation_count}")
            if oldest.exists():
                oldest.unlink()

            # 2) 남은 백업들을 뒤에서부터(.N-1 → .N) 한 칸씩 뒤로 옮긴다.
            #    역순으로 도는 이유: 앞에서부터 옮기면 다음 대상을 덮어쓰기 때문.
            for i in range(self._rotation_count - 1, 0, -1):
                src = Path(f"{self._log_path}.{i}")
                dst = Path(f"{self._log_path}.{i + 1}")
                if src.exists():
                    src.rename(dst)

            # 3) 현재 로그(audit.log)를 가장 최신 백업(.1)으로 이동한다.
            if self._log_path.exists():
                self._log_path.rename(Path(f"{self._log_path}.1"))

            logger.info("감사 로그 로테이션 완료: %s", self._log_path)
        except OSError as e:
            # 로테이션 실패도 치명적으로 다루지 않는다 — 에러만 남기고 진행.
            logger.error("감사 로그 로테이션 실패: %s", e)

    def clear(self) -> None:
        """
        메모리 캐시(deque)를 비운다.

        디스크의 로그 파일은 건드리지 않고 최근 조회용 캐시만 초기화한다.
        주로 테스트에서 상태를 깨끗이 리셋할 때 사용한다.
        """
        self._recent.clear()
