"""Phase 3 LoRA 학습을 GPU 서버에서 원격으로 실행하는 오케스트레이션 스크립트.

이 스크립트는 로컬(Machine A)에서 실행하지만, 실제 학습은 GPU 서버(Machine B)에서
돌아간다. paramiko(SSH/SFTP)로 GPU 서버에 접속해 다음 순서를 자동으로 처리한다.

  0. 로컬의 학습 스크립트를 SFTP로 GPU 서버에 업로드
  1. GPU 메모리를 점유 중인 vLLM 추론 서버(Worker)를 종료
  2. nvidia-smi로 GPU 메모리가 실제로 확보됐는지 확인
  3. nohup + setsid로 학습 프로세스를 백그라운드(데몬)로 띄움
  4. 15초 뒤 학습 로그 앞부분을 tail로 확인 (정상 시작 여부 판단)
  5. 학습 프로세스가 살아있는지 pgrep으로 최종 확인

구조상 Phase 2 실행 스크립트와 동일하며, 대상 스크립트/로그 경로 등
"Phase 3"용 상수만 다르다. 주요 함수는 run()(원격 명령 실행 헬퍼)과
main()(전체 절차 오케스트레이션) 두 개다.

주의: 접속 정보(HOST/USER/PASSWORD)가 하드코딩된 일회성 운영 유틸리티다.
정식 애플리케이션 코드가 아니라 학습 배치를 손으로 돌리기 위한 보조 스크립트다.

작성자: 이현수 / 작성일: 2026-07-05
"""

# from __future__ import: 타입 힌트를 문자열로 지연 평가 (파이썬 3.11+ 호환성 보강용)
from __future__ import annotations

import sys
import time

import paramiko  # 순수 파이썬 SSH 클라이언트 (에어갭 내 LAN 서버 접속용)

# stdout을 UTF-8로 재설정 — 한글 로그/출력이 Windows 콘솔에서 깨지지 않게 함
sys.stdout.reconfigure(encoding="utf-8")

# --- GPU 서버(Machine B) 접속 정보 --------------------------------------
# LAN 내부 주소이며 에어갭 환경 전용. 외부로 나가지 않는다.
HOST = "192.168.21.112"
USER = "idino"
PASSWORD = "dkdlelsh@12"

# --- 학습 스크립트/로그 경로 --------------------------------------------
# LOCAL_SCRIPT : 이 PC(Machine A)에 있는 원본 학습 스크립트
# REMOTE_SCRIPT: GPU 서버에 업로드될 위치 (여기서 파이썬으로 실행됨)
# LOG_PATH     : 백그라운드 학습의 표준출력/표준에러가 쌓이는 로그 파일
LOCAL_SCRIPT = "scripts/train_qwen_lora_phase3.py"
REMOTE_SCRIPT = "/opt/nexus-gpu/training/train_qwen_lora_phase3.py"
LOG_PATH = "/opt/nexus-gpu/training/train_phase3.log"


def run(ssh, cmd, timeout=20):
    """열린 SSH 세션으로 원격 명령 하나를 실행하고 출력(문자열)을 돌려준다.

    stdout과 stderr를 모두 읽어 하나의 문자열로 합쳐 반환하므로, 호출부에서
    print() 한 번으로 명령 결과와 에러 메시지를 함께 확인할 수 있다.

    매개변수:
        ssh     : 이미 connect()된 paramiko.SSHClient 객체
        cmd     : 원격에서 실행할 쉘 명령 문자열
        timeout : 명령 실행/응답 대기 제한(초). 백그라운드 실행처럼 즉시
                  끝나야 하는 명령은 호출부에서 짧게 지정한다.

    반환값:
        stdout+stderr를 합친 문자열. 채널 읽기에 실패하면 에러 사유 문자열.
    """
    # 원격 명령 실행 — stdin은 쓰지 않고 stdout/stderr만 사용
    stdin, stdout, stderr = ssh.exec_command(cmd, timeout=timeout)
    try:
        # 표준출력과 표준에러를 모두 디코딩해 이어 붙임.
        # errors="replace": 깨진 바이트가 있어도 예외 없이 대체문자로 처리
        return stdout.read().decode(errors="replace") + stderr.read().decode(errors="replace")
    except Exception as e:
        # 채널이 끊기거나 읽기 실패 시에도 스크립트가 죽지 않도록 사유만 반환
        return f"(channel err: {e})"


def main() -> None:
    """Phase 3 학습을 GPU 서버에서 시작하기까지의 전 과정을 순서대로 수행한다.

    흐름:
        SSH 접속 → 스크립트 업로드 → vLLM 종료 → GPU 확인 →
        학습 백그라운드 실행 → 시작 로그 확인 → 프로세스 확인 → 접속 종료

    각 단계 결과를 print로 콘솔에 출력해 사람이 진행 상황을 눈으로 따라갈 수
    있게 한다. 반환값은 없다(부수효과 = 원격 학습 시작).
    """
    # SSH 클라이언트 생성 및 접속
    ssh = paramiko.SSHClient()
    # 서버의 host key가 등록돼 있지 않아도 자동 수락 (사내 LAN 신뢰 환경 전제)
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(HOST, username=USER, password=PASSWORD, timeout=10)

    # --- 0단계: 학습 스크립트 업로드 -----------------------------------
    # 로컬 최신 스크립트를 GPU 서버로 덮어써, 항상 최신 코드로 학습하도록 보장
    print("=== 0. 학습 스크립트 업로드 ===")
    sftp = ssh.open_sftp()
    sftp.put(LOCAL_SCRIPT, REMOTE_SCRIPT)
    sftp.close()
    # 업로드가 실제로 됐는지 파일 크기/시각을 ls로 확인
    print(run(ssh, f"ls -lh {REMOTE_SCRIPT}"))

    # --- 1단계: vLLM Worker 중단 ---------------------------------------
    # 학습은 GPU VRAM을 크게 쓰므로, 먼저 추론 서버를 내려 메모리를 비운다.
    print("\n=== 1. vLLM Worker 중단 ===")
    # pkill로 vLLM API 서버 프로세스 종료. 없어도 실패하지 않게 '|| true'
    print(run(ssh, "pkill -f 'vllm.entrypoints.openai.api_server' || true"))
    # 최대 10회(약 20초) 폴링하며 프로세스가 완전히 사라졌는지 확인
    for i in range(10):
        time.sleep(2)  # 종료에 시간이 걸릴 수 있어 2초 간격으로 재확인
        # pgrep 결과가 비면(=프로세스 없음) '(종료됨)' 문구를 대신 출력
        chk = run(ssh, "pgrep -f 'vllm.entrypoints.openai' || echo '(종료됨)'")
        print(f"  [{i+1}/10] {chk.strip()}")
        # 종료가 확인되면 남은 대기 없이 루프 탈출
        if "(종료됨)" in chk:
            break

    # --- 2단계: GPU 메모리 확보 확인 -----------------------------------
    # 학습을 띄우기 전에 VRAM이 실제로 비었는지 사용량/총량을 눈으로 확인
    print("\n=== 2. GPU 확보 확인 ===")
    print(run(ssh, "nvidia-smi --query-gpu=memory.used,memory.total --format=csv"))

    # --- 3단계: Phase 3 학습 백그라운드 실행 ---------------------------
    print("\n=== 3. Phase 3 학습 실행 ===")
    # SSH 세션이 끊겨도 학습이 계속 돌게 하는 것이 핵심.
    #   cd            : 학습 스크립트가 있는 작업 디렉토리로 이동
    #   setsid nohup  : 새 세션 + 행업 신호 무시 → 세션 종료에도 살아남음
    #   .venv 파이썬  : GPU 서버 전용 가상환경의 파이썬 3.12로 실행
    #   </dev/null    : 표준입력 차단 (백그라운드 프로세스가 입력 대기하지 않게)
    #   >LOG 2>&1     : stdout/stderr를 모두 로그 파일로 리다이렉트
    #   & echo PID    : 백그라운드로 던지고 시작된 PID를 즉시 출력
    #   disown        : 현재 쉘의 작업 목록에서 분리 (완전한 데몬화)
    train_cmd = (
        f"cd /opt/nexus-gpu/training && "
        f"setsid nohup /opt/nexus-gpu/.venv/bin/python3.12 "
        f"{REMOTE_SCRIPT} </dev/null >{LOG_PATH} 2>&1 & "
        f"echo \"TRAIN_PID=$!\"; disown"
    )
    # 이 명령은 즉시 반환되므로 timeout을 짧게(5초) 준다 — 학습 완료를 기다리지 않음
    print(run(ssh, train_cmd, timeout=5))

    # --- 4단계: 시작 로그 확인 -----------------------------------------
    # 15초 정도 지난 뒤 로그 앞부분을 확인해 학습이 정상 시작했는지 판단
    print("\n=== 4. 시작 로그 (15초 후) ===")
    time.sleep(15)  # 모델 로딩/초기화가 로그에 찍힐 시간을 벌어줌
    print(run(ssh, f"tail -30 {LOG_PATH}"))

    # --- 5단계: 학습 프로세스 생존 확인 --------------------------------
    # 학습 프로세스가 살아있는지 pgrep으로 최종 확인 (없으면 '(없음)' 출력)
    print("\n=== 5. 프로세스 ===")
    print(run(ssh, "pgrep -af 'train_qwen_lora_phase3' || echo '(없음)'"))

    # 모든 절차가 끝났으니 SSH 연결을 닫는다.
    # (학습은 데몬으로 분리돼 있으므로 연결을 닫아도 계속 진행된다.)
    ssh.close()


if __name__ == "__main__":
    # 스크립트를 직접 실행했을 때만 main() 호출 (import 시에는 실행 안 됨)
    main()
