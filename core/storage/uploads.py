# 업로드 샌드박스({tempdir}/nexus_uploads) 보존기간 정리 — 파일 무한 증식 방지.
"""
업로드 파일 정리(cleanup_expired_uploads).

[왜 필요한가]
  웹 업로드 라우트(POST /v1/upload)는 첨부 파일을 업로드 샌드박스에 저장하고
  그 경로만 모델에게 넘긴다. 그런데 이 파일을 지우는 주체가 아무도 없었다.
  실제로 배포 서버에서 3주 넘은 파일까지 그대로 쌓여 있었다(29개/19MB).
  컨테이너 재시작으로 /tmp 가 비워지는 배포에서는 눈에 안 띄지만, 장기 가동이나
  영속 볼륨을 쓰는 고객 환경에서는 계속 증식한다. 이 모듈이 보존기간을 넘긴
  파일만 골라 지운다.

[무엇을 지우고 무엇을 남기나]
  - 지운다: 업로드 디렉토리 **바로 아래의 일반 파일** 중 mtime 이 보존기간을
    넘긴 것. 이름 규칙을 따지지 않는다 — 이 폴더에는 업로드본(`upload-*`),
    RenderPreview 스크린샷(`render_*`), 그리고 ASCII-safe 명명 이전의 레거시
    한글 파일명이 섞여 있고, 셋 다 똑같이 임시 산출물이기 때문이다.
  - 남긴다: 하위 디렉토리(재귀하지 않는다). 이 폴더는 평평한 임시 저장소라
    디렉토리가 있다면 의도된 것이거나 다른 용도이므로 건드리지 않는다(보수적).

[안전 설계]
  - retention_hours 가 0 이하이면 아무 것도 하지 않는다. "0=전부 삭제"로
    오작동하는 사고를 막는 안전장치다(artifacts 정리의 retention_days 와 같은 규칙).
  - 파일 하나를 못 지워도(권한/경합) 전체를 중단하지 않는다(fail-soft).
    다음 주기에 다시 시도하면 되며, 정리 실패가 서비스를 막아서는 안 된다.
  - 동기 함수다. 파일시스템 작업이라 호출부(웹 lifespan)가 asyncio.to_thread 로
    돌려 이벤트 루프를 막지 않는다.

작성자: 이현수 / 작성일: 2026-08-06
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

logger = logging.getLogger("nexus.storage.uploads")


def cleanup_expired_uploads(
    uploads_dir: str | Path,
    retention_hours: float,
    now: float | None = None,
) -> tuple[int, int]:
    """보존기간이 지난 업로드 파일을 지운다(fail-soft).

    매개변수:
      uploads_dir     — 업로드 샌드박스 경로({tempdir}/nexus_uploads 등).
      retention_hours — 보존 시간. 이보다 오래된 파일이 대상. 0 이하이면 no-op.
      now             — 기준 시각(epoch 초). 테스트 주입용이며 None이면 현재 시각.

    반환:
      (삭제한 파일 수, 확보한 바이트). 디렉토리가 없거나 no-op이면 (0, 0).
    """
    # 0 이하 보존기간은 "전부 삭제"로 오작동할 수 있어 안전하게 건너뛴다.
    if retention_hours <= 0:
        logger.debug("업로드 정리 스킵 — retention_hours=%s (0 이하)", retention_hours)
        return 0, 0

    base = Path(uploads_dir)
    if not base.is_dir():
        return 0, 0

    now = time.time() if now is None else now
    cutoff = now - retention_hours * 3600.0

    deleted = 0
    freed = 0
    try:
        entries = list(base.iterdir())
    except OSError as e:
        # 디렉토리 자체를 읽지 못하는 상황(권한/마운트 문제)은 경고만 남기고 넘어간다.
        logger.warning("업로드 디렉토리 조회 실패(무시): %s — %s", base, e)
        return 0, 0

    for entry in entries:
        try:
            # 하위 디렉토리는 재귀하지 않고 그대로 둔다(위 docstring 근거).
            if not entry.is_file():
                continue
            stat = entry.stat()
            if stat.st_mtime >= cutoff:
                continue
            size = stat.st_size
            entry.unlink()
            deleted += 1
            freed += size
        except OSError as e:
            # 파일 하나 실패가 나머지 정리를 막지 않게 한다(다음 주기에 재시도).
            logger.warning("업로드 파일 삭제 실패(무시): %s — %s", entry, e)
            continue

    if deleted:
        logger.info(
            "업로드 정리 완료: %d개 삭제, %.1fMB 확보 (보존 %.1f시간)",
            deleted,
            freed / 1024 / 1024,
            retention_hours,
        )
    return deleted, freed
