"""
체크포인트 관리자 — LoRA 어댑터 체크포인트의 목록 관리, 활성화, 롤백.

[이 파일이 하는 일]
QLoRA/LoRA 학습으로 만들어진 "어댑터"(원본 모델 위에 얹는 작은 가중치)를
체크포인트 단위로 관리한다. 하나의 체크포인트는 학습 결과물 한 벌을 뜻하며,
디스크의 한 디렉토리에 담긴다. 이 관리자는 그 디렉토리들을 훑어 목록을 만들고,
그중 하나를 GPU 서버(Machine B)에 얹거나(활성화) 다시 내려(롤백) 준다.

[핵심 구성]
- 클래스 CheckpointManager: 아래 5개 기능을 제공하는 유일한 진입점.
  * list_checkpoints() — 저장된 체크포인트를 최신순으로 나열
  * activate()         — 특정 체크포인트를 GPU 서버에 hot-load (async)
  * rollback()         — 어댑터를 내리고 기본 모델로 복원 (async)
  * get_best()         — 평가 지표 기준 최고 성능 체크포인트 선택
  * save_metadata()    — 학습 후 평가 결과/설정을 metadata.json에 기록

[다른 모듈과의 관계]
- GPU 서버와는 오직 HTTP(httpx)로만 통신한다. 즉 이 파일은 GPU/CUDA를
  직접 만지지 않는다(아키텍처 P4: Machine A는 vLLM API로만 접근).
- 엔드포인트: POST /v1/lora/load(활성화), POST /v1/lora/unload(롤백).

[체크포인트 디렉토리 구조]
  checkpoints/
    {checkpoint_name}/
      adapter_model.safetensors   — LoRA 가중치
      adapter_config.json         — LoRA 설정
      metadata.json               — 평가 결과, 학습 설정, 생성 시간

[에어갭 준수]
GPU 서버 URL은 반드시 LAN 주소(192.168.x.x / 10.x.x.x / localhost)만 넘긴다.
외부 인터넷 주소를 넘기면 안 된다(폐쇄망 규칙).

작성자: 이현수 / 작성일: 2026-07-05
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

# 모듈 전용 로거. 로그는 "nexus.training.checkpoint_manager" 이름으로 남으며,
# JSONL 로깅 설정(config)에서 이 이름으로 필터링/수집할 수 있다.
logger = logging.getLogger("nexus.training.checkpoint_manager")

# 각 체크포인트 디렉토리 안에 놓이는 메타데이터 파일 이름.
# 상수로 빼두면 오타를 막고, 파일명이 바뀌어도 한 곳만 고치면 된다.
_METADATA_FILENAME = "metadata.json"


class CheckpointManager:
    """
    체크포인트 목록 관리, 활성화, 롤백을 담당하는 관리자 클래스.

    [왜 필요한가]
    학습이 끝나면 어댑터가 여러 벌 쌓인다. 어느 것을 실제 서비스에 얹을지,
    문제가 생기면 어떻게 되돌릴지 관리할 주체가 필요하다. 이 클래스가 그 역할을
    맡아, 디스크의 체크포인트들을 조회하고 GPU 서버에 얹고 내리는 일을 한다.

    [상태]
    이 객체는 "지금 어떤 체크포인트가 활성 상태인지"를 self._active_checkpoint에
    기억한다. 활성화에 성공하면 그 이름으로 갱신하고, 롤백에 성공하면 None으로
    되돌린다. (None = 어댑터 없이 기본 모델만 쓰는 상태)

    주의: 이 상태는 이 프로세스 메모리에만 있다. 프로세스를 재시작하면
    None(기본 모델)에서 다시 시작한다.
    """

    def __init__(self, checkpoints_dir: str = "./checkpoints/") -> None:
        """
        관리자를 초기화한다.

        생성 시점에 체크포인트 디렉토리를 준비한다. 디렉토리가 없으면 부모까지
        한꺼번에 만들어 두므로(exist_ok=True), 첫 실행에서도 곧바로 저장/조회가
        가능하다.

        Args:
            checkpoints_dir: 체크포인트들이 모여 있는 최상위 디렉토리 경로.
                기본값은 현재 작업 디렉토리 기준 "./checkpoints/".
        """
        # 문자열 경로를 Path 객체로 감싸 이후 경로 조작을 안전하게 한다.
        self._checkpoints_dir = Path(checkpoints_dir)
        # 디렉토리가 없으면 부모 경로까지 생성. 이미 있으면 조용히 넘어간다.
        self._checkpoints_dir.mkdir(parents=True, exist_ok=True)
        # 현재 활성화된 체크포인트 이름. None이면 어댑터 없이 기본 모델 사용 중.
        self._active_checkpoint: str | None = None

    @property
    def checkpoints_dir(self) -> Path:
        """체크포인트 최상위 디렉토리 경로(Path)를 읽기 전용으로 노출한다."""
        return self._checkpoints_dir

    @property
    def active_checkpoint(self) -> str | None:
        """
        현재 활성화된 체크포인트 이름을 반환한다.

        어댑터가 하나도 얹혀 있지 않으면(기본 모델 상태) None을 돌려준다.
        """
        return self._active_checkpoint

    def list_checkpoints(self) -> list[dict[str, Any]]:
        """
        저장된 모든 체크포인트를 훑어 정보 목록으로 반환한다.

        [흐름]
        1) 최상위 디렉토리 안의 하위 디렉토리를 하나씩 순회한다. 파일은 건너뛰고
           디렉토리만 하나의 체크포인트로 본다.
        2) 각 디렉토리에서 metadata.json을 읽어 생성 시간과 평가 결과를 뽑는다.
        3) 메타데이터가 없거나 깨졌으면, 대신 디렉토리 수정 시간을 생성 시간으로
           쓴다(최소한 정렬은 가능하도록).
        4) 마지막에 생성 시간 내림차순(최신 우선)으로 정렬해 돌려준다.

        Returns:
            체크포인트 정보 딕셔너리의 목록. 각 항목의 키:
              - name: 체크포인트 이름 (곧 디렉토리 이름)
              - path: 디렉토리의 전체 경로 문자열
              - created_at: 생성 시간 (ISO 8601 문자열)
              - metadata: metadata.json 전체 내용(없으면 빈 dict)
              - is_active: 이 체크포인트가 현재 활성 상태인지 여부
        """
        checkpoints: list[dict[str, Any]] = []

        # 디렉토리가 아직 없으면 조회할 것도 없으니 빈 목록으로 조기 반환.
        if not self._checkpoints_dir.exists():
            return checkpoints

        # 최상위 디렉토리의 바로 아래 항목들을 순회한다.
        for ckpt_dir in self._checkpoints_dir.iterdir():
            # 파일(예: 잘못 놓인 zip 등)은 체크포인트가 아니므로 건너뛴다.
            if not ckpt_dir.is_dir():
                continue

            # 메타데이터를 읽기 전에도 확실히 알 수 있는 기본 정보부터 채운다.
            info: dict[str, Any] = {
                "name": ckpt_dir.name,
                "path": str(ckpt_dir),
                # 디렉토리 이름이 현재 활성 이름과 같으면 활성 상태로 표시.
                "is_active": ckpt_dir.name == self._active_checkpoint,
            }

            # 이 체크포인트의 메타데이터 파일 경로를 조립한다.
            metadata_file = ckpt_dir / _METADATA_FILENAME
            if metadata_file.exists():
                # 파일이 있으면 JSON으로 읽되, 깨진 파일에도 견디도록 감싼다.
                try:
                    with open(metadata_file, encoding="utf-8") as f:
                        metadata = json.load(f)
                    # created_at 키가 없을 수도 있으니 빈 문자열을 기본값으로.
                    info["created_at"] = metadata.get("created_at", "")
                    info["metadata"] = metadata
                except (json.JSONDecodeError, OSError) as e:
                    # JSON 파손/읽기 실패 시: 경고만 남기고 이 항목을 버리지 않는다.
                    # (목록 전체가 하나의 깨진 파일 때문에 실패하면 곤란하므로)
                    logger.warning("메타데이터 읽기 실패: %s (%s)", metadata_file, e)
                    info["created_at"] = ""
                    info["metadata"] = {}
            else:
                # 메타데이터 파일이 아예 없으면, 정렬 기준이라도 확보하기 위해
                # 디렉토리 수정 시각(mtime)을 생성 시간 대용으로 사용한다.
                mtime = datetime.fromtimestamp(ckpt_dir.stat().st_mtime, tz=UTC)
                info["created_at"] = mtime.isoformat()
                info["metadata"] = {}

            checkpoints.append(info)

        # created_at 문자열 기준 내림차순 정렬 → 최신 체크포인트가 맨 앞에 온다.
        # ISO 8601 문자열은 사전식 정렬만으로도 시간순이 맞는다는 점을 이용.
        checkpoints.sort(key=lambda c: c.get("created_at", ""), reverse=True)
        return checkpoints

    async def activate(
        self,
        name: str,
        gpu_server_url: str,
    ) -> dict[str, Any]:
        """
        특정 체크포인트를 GPU 서버에 활성화(LoRA hot-loading)한다.

        [무엇을 하나]
        지정한 체크포인트 경로를 GPU 서버에 알려 어댑터를 얹도록(load) 요청한다.
        성공하면 그때부터 들어오는 추론 요청에 이 어댑터가 적용된다. "hot-loading"
        이라 부르는 이유는 서버를 재시작하지 않고 얹기 때문이다.

        [비동기인 이유]
        GPU 서버와의 HTTP 왕복은 수 초가 걸릴 수 있어, 그 사이 이벤트 루프를
        막지 않도록 async 함수로 만들어 await로 호출한다.

        [에러 처리 방침]
        네트워크 문제나 서버 4xx/5xx는 예외를 밖으로 던지지 않고, status="error"
        가 담긴 결과 dict로 감싸 돌려준다. 호출 측이 흐름을 끊지 않고 상황을
        판단할 수 있게 하기 위함이다. 단, 체크포인트가 로컬에 아예 없는 경우는
        프로그래밍/입력 오류에 가깝기에 예외로 즉시 알린다.

        Args:
            name: 활성화할 체크포인트 이름(디렉토리 이름).
            gpu_server_url: GPU 서버 URL. 반드시 LAN 주소만 허용(에어갭 규칙).

        Returns:
            활성화 결과 dict. 최소한 name, status를 담으며 서버 응답(message 등)을
            펼쳐 합친다. 성공 시 status="activated", 실패 시 status="error".

        Raises:
            FileNotFoundError: 해당 이름의 체크포인트 디렉토리가 없을 때.
        """
        # 로컬에 실제로 그 체크포인트가 있는지 먼저 확인한다. 없으면 서버에
        # 요청해봐야 소용없으므로 곧바로 예외로 알린다.
        ckpt_path = self._checkpoints_dir / name
        if not ckpt_path.exists():
            raise FileNotFoundError(f"체크포인트를 찾을 수 없습니다: {ckpt_path}")

        # URL 끝의 슬래시를 떼어, 뒤에 경로를 붙일 때 "//"가 생기지 않게 한다.
        gpu_url = gpu_server_url.rstrip("/")

        try:
            # with 블록을 벗어나면 커넥션이 자동 정리된다. timeout으로 무한 대기 방지.
            async with httpx.AsyncClient(timeout=60.0) as client:
                # 서버에 어댑터 이름과 로컬 경로를 넘겨 로딩을 요청한다.
                response = await client.post(
                    f"{gpu_url}/v1/lora/load",
                    json={
                        "checkpoint_name": name,
                        "checkpoint_path": str(ckpt_path),
                    },
                )
                # 4xx/5xx면 HTTPStatusError를 일으켜 아래 except로 넘긴다.
                response.raise_for_status()
                result = response.json()
        except httpx.ConnectError as e:
            # 서버가 꺼져 있거나 네트워크가 끊긴 경우.
            logger.warning("GPU 서버 연결 실패: %s", e)
            result = {"status": "error", "message": f"GPU 서버 연결 실패: {e}"}
            return {"name": name, **result}
        except httpx.HTTPStatusError as e:
            # 서버는 응답했으나 에러 상태 코드를 돌려준 경우.
            logger.warning("GPU 서버 응답 에러: %s", e)
            result = {"status": "error", "message": f"GPU 서버 응답 에러: {e}"}
            return {"name": name, **result}

        # 여기까지 왔으면 로딩 성공. 이제서야 활성 체크포인트 상태를 갱신한다.
        # (실패했는데 상태만 바뀌는 일을 막기 위해 성공 직후에만 갱신)
        self._active_checkpoint = name
        logger.info("체크포인트 활성화: %s", name)

        return {"name": name, "status": "activated", **result}

    async def rollback(self, gpu_server_url: str) -> dict[str, Any]:
        """
        현재 얹혀 있는 LoRA 어댑터를 내리고 기본 모델 상태로 되돌린다.

        [언제 쓰나]
        새 어댑터를 활성화했는데 품질이 나쁘거나 오작동할 때, 빠르게 안전한
        기본 모델로 복귀하는 비상 스위치 역할이다. 어댑터 이름을 따로 받지
        않는다. 서버에 "지금 얹힌 것을 내려라"라고만 요청한다.

        [에러 처리]
        activate()와 동일하게, 연결 실패나 서버 에러는 예외로 던지지 않고
        status="error" 결과 dict로 감싸 돌려준다.

        Args:
            gpu_server_url: GPU 서버 URL. 반드시 LAN 주소만 허용(에어갭 규칙).

        Returns:
            롤백 결과 dict. previous_checkpoint(직전에 활성이던 이름, 없으면
            None)와 status를 담고 서버 응답을 펼쳐 합친다. 성공 시
            status="rolled_back", 실패 시 status="error".
        """
        # 되돌리기 전에 "무엇을 내렸는지" 기록용으로 직전 이름을 보관해 둔다.
        previous = self._active_checkpoint
        # activate()와 마찬가지로 끝 슬래시를 정리한다.
        gpu_url = gpu_server_url.rstrip("/")

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                # 언로드는 대상 지정이 필요 없어 본문(json) 없이 POST만 보낸다.
                response = await client.post(
                    f"{gpu_url}/v1/lora/unload",
                )
                response.raise_for_status()
                result = response.json()
        except httpx.ConnectError as e:
            # 서버 미응답/네트워크 단절.
            logger.warning("GPU 서버 연결 실패 (롤백): %s", e)
            result = {"status": "error", "message": f"GPU 서버 연결 실패: {e}"}
            return {"previous_checkpoint": previous, **result}
        except httpx.HTTPStatusError as e:
            # 서버가 에러 상태 코드로 응답.
            logger.warning("GPU 서버 응답 에러 (롤백): %s", e)
            result = {"status": "error", "message": f"GPU 서버 응답 에러: {e}"}
            return {"previous_checkpoint": previous, **result}

        # 언로드 성공 후에만 활성 상태를 None(기본 모델)으로 되돌린다.
        self._active_checkpoint = None
        # previous가 None이면 원래도 기본 모델이었다는 뜻이라 "(없음)"으로 표기.
        logger.info("롤백 완료: %s → 기본 모델", previous or "(없음)")

        return {
            "previous_checkpoint": previous,
            "status": "rolled_back",
            **result,
        }

    def get_best(self, metric: str = "eval_accuracy") -> dict[str, Any] | None:
        """
        지정한 평가 지표 기준으로 가장 성능이 좋은 체크포인트를 골라 반환한다.

        [동작]
        list_checkpoints()로 전체 목록을 얻은 뒤, 각 체크포인트의 metadata에서
        metric 키의 값을 꺼내 가장 큰 값을 가진 것을 고른다. 값이 클수록 좋은
        지표(정확도 등)를 전제로 하므로, 손실(loss)처럼 작을수록 좋은 지표에는
        그대로 쓰면 안 된다는 점에 주의.

        Args:
            metric: 비교 기준이 될 메타데이터 키. 기본값은 "eval_accuracy".

        Returns:
            최고 성능 체크포인트의 정보 dict. 체크포인트가 하나도 없거나 해당
            메트릭을 가진 것이 없으면 None.
        """
        checkpoints = self.list_checkpoints()
        # 아예 체크포인트가 없으면 비교할 대상이 없으니 None.
        if not checkpoints:
            return None

        # 지금까지 찾은 최고 후보와 그 점수. 초기 점수는 음의 무한대로 두어
        # 어떤 실제 값이라도 첫 비교에서 이기도록 한다.
        best: dict[str, Any] | None = None
        best_value = -float("inf")

        for ckpt in checkpoints:
            metadata = ckpt.get("metadata", {})
            # 해당 메트릭이 없으면 -inf로 취급해 후보에서 자연히 밀려나게 한다.
            value = metadata.get(metric, -float("inf"))
            # 숫자 타입인지 확인해, 문자열 등 잘못된 값이 비교에 끼는 것을 막는다.
            if isinstance(value, (int, float)) and value > best_value:
                best_value = value
                best = ckpt

        return best

    def save_metadata(
        self,
        name: str,
        metadata: dict[str, Any],
    ) -> None:
        """
        체크포인트 디렉토리에 메타데이터를 metadata.json으로 저장한다.

        [언제 쓰나]
        보통 학습이 끝난 직후, 평가 결과(정확도/손실 등)와 학습 설정을 남겨
        나중에 list_checkpoints()/get_best()가 읽을 수 있도록 기록할 때 쓴다.

        [부수 효과 주의]
        created_at 키가 없으면 현재 UTC 시각을 넣어준다. 이때 넘겨받은 metadata
        딕셔너리 자체를 직접 수정하므로, 호출 측에서 같은 dict를 재사용한다면
        created_at이 채워진다는 점을 알아둘 것.

        Args:
            name: 체크포인트 이름(디렉토리 이름). 없으면 새로 만든다.
            metadata: 저장할 메타데이터 딕셔너리(JSON 직렬화 가능해야 함).
        """
        # 대상 디렉토리를 확보한다. 아직 없으면 부모까지 만들어 둔다.
        ckpt_dir = self._checkpoints_dir / name
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        # 생성 시간이 비어 있으면 지금 시각(UTC, ISO 8601)을 자동으로 채운다.
        # 이 값이 있어야 list_checkpoints()의 최신순 정렬이 제대로 동작한다.
        if "created_at" not in metadata:
            metadata["created_at"] = datetime.now(UTC).isoformat()

        # JSON으로 기록. ensure_ascii=False로 한글이 깨지지 않게 그대로 저장하고,
        # indent=2로 사람이 읽기 좋게 들여쓴다.
        metadata_file = ckpt_dir / _METADATA_FILENAME
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        logger.info("메타데이터 저장: %s → %s", name, metadata_file)
