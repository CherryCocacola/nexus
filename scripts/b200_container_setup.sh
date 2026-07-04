#!/usr/bin/env bash
# =============================================================================
# Project Nexus — B200 개발 컨테이너 환경 부트스트랩
# =============================================================================
# 무엇을 하는가:
#   NHN Cloud B200 개발 컨테이너 안에서 vLLM 추론 환경을 처음부터 세팅한다.
#   영구 디렉토리 생성 → uv 설치 → venv 생성 → vLLM 설치 → B200 인식 검증.
#
# 어디서 실행하나:
#   컨테이너의 터미널(JupyterLab/VSCode 터미널)에서 실행한다.
#   (개발 머신의 Claude Code에서 실행하는 게 아니다 — 컨테이너 내부다.)
#     bash b200_container_setup.sh
#
# 왜 이 순서인가:
#   가장 큰 리스크는 "vLLM이 B200(Blackwell, sm_100)에서 실제로 도는가"이다.
#   그래서 설치 후 torch가 B200을 인식하는지를 [5/5]에서 반드시 검증하고,
#   실패해도 리포트를 볼 수 있게 set -e 대신 단계별로 진행한다.
#
# 실측 전제(2026-07-03 컨테이너 진단 기준):
#   - 영구 디스크: /NHNHOME = 400GB nvme (HF_HOME=/NHNHOME/nexus/hf_cache)
#   - GPU: NVIDIA B200 183GB, 드라이버 580.95.05, 시스템 torch CUDA 13.0 인식 OK
#   - 외부망: huggingface.co 도달 가능(HTTP 200)
#   - Python 3.12.3, uv 없음
# =============================================================================

set -uo pipefail   # -e는 일부러 뺀다: vLLM 설치가 실패해도 마지막 검증/리포트까지 보기 위함

# --- 경로(컨테이너 실측 기준). 다른 환경이면 이 두 줄만 바꾸면 된다 ---
NEXUS_ROOT="/NHNHOME/nexus"      # 영구 400GB 디스크 루트
VENV="${NEXUS_ROOT}/venv"        # vLLM용 파이썬 가상환경

echo "########################################################"
echo "# Nexus B200 컨테이너 환경 부트스트랩"
echo "# NEXUS_ROOT=${NEXUS_ROOT}"
echo "########################################################"

echo ""
echo "===== [1/5] 영구 디렉토리 생성 ====="
# hf_cache: 모델 캐시(HF_HOME) / logs: vLLM 로그 / pgdata: PostgreSQL 데이터
# checkpoints: LoRA 어댑터(Phase 3) / models: 로컬 반입용(에어갭 대비)
mkdir -p "${NEXUS_ROOT}"/{hf_cache,logs,pgdata,checkpoints,models}
ls -la "${NEXUS_ROOT}"
echo "HF_HOME=${HF_HOME:-(미설정!)}"   # 컨테이너 환경변수가 hf_cache를 가리키는지 확인

echo ""
echo "===== [2/5] uv 설치 ====="
if command -v uv >/dev/null 2>&1; then
    echo "uv 이미 있음: $(uv --version)"
else
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
# uv는 보통 ~/.local/bin 또는 ~/.cargo/bin에 설치된다 — PATH에 추가
export PATH="${HOME}/.local/bin:${HOME}/.cargo/bin:${PATH}"
uv --version || echo "[경고] uv를 PATH에서 못 찾음 — 설치 로그 확인 필요"

echo ""
echo "===== [3/5] venv 생성 (Python 3.12) ====="
if [ -d "${VENV}" ]; then
    echo "venv 이미 존재: ${VENV}"
else
    uv venv "${VENV}" --python 3.12 --seed
fi
# shellcheck disable=SC1091
source "${VENV}/bin/activate"
echo "활성 python: $(python --version) @ $(command -v python)"

echo ""
echo "===== [4/5] vLLM 설치 (B200 통합 리스크 단계) ====="
# 주의: 새 venv라 vLLM이 자체 torch를 끌어온다. 그 torch가 B200(sm_100)을
#       지원하지 않으면 아래 검증에서 avail=False로 드러난다. 그러면 중단하고
#       리포트를 붙여넣어 주면, Blackwell 호환 조합으로 다시 잡는다.
echo "(수 분 소요 — 마지막 15줄만 출력)"
uv pip install --upgrade vllm 2>&1 | tail -15

echo ""
echo "===== [5/5] 검증 (이 출력을 그대로 붙여넣어 주세요) ====="
python - <<'PY'
import sys
print("--- torch ---")
try:
    import torch
    print("torch:", torch.__version__, "| cuda:", torch.version.cuda,
          "| avail:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("device:", torch.cuda.get_device_name(0),
              "| capability(sm):", torch.cuda.get_device_capability(0))
    else:
        print("[문제] CUDA 미인식 — vLLM의 torch가 B200을 지원 안 할 수 있음")
except Exception as e:  # noqa: BLE001 (검증 리포트용 — 어떤 예외든 그대로 보고)
    print("torch 검증 실패:", e)
print("--- vllm ---")
try:
    import vllm
    print("vllm:", vllm.__version__)
except Exception as e:  # noqa: BLE001
    print("vllm import 실패:", e)
PY

echo ""
echo "########################################################"
echo "# 부트스트랩 완료. 위 [5/5] 검증 결과를 붙여넣어 주세요."
echo "#   기대값: torch avail=True / device=NVIDIA B200 / sm=(10,0) / vllm 버전 출력"
echo "#   다음 단계: 모델 다운로드 + start_vllm_local.py 로 서빙"
echo "########################################################"
