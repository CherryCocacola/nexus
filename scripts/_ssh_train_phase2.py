"""Phase 2 LoRA 학습 파이프라인 원격 실행 스크립트.

이 스크립트는 개발자 PC(Machine A 쪽)에서 실행하며, SSH/SFTP로 GPU 서버
(192.168.21.112)에 접속하여 Qwen LoRA "Phase 2" 학습을 원격으로 개시한다.

전체 흐름은 아래 6단계로 구성된다.
  0. 로컬 학습 스크립트(train_qwen_lora_phase2.py)를 GPU 서버로 업로드
  1. 추론용 vLLM Worker(:8001)를 종료하여 VRAM(GPU 메모리)을 확보
  2. 학습 시작 전 GPU 메모리 사용량을 확인
  3. 업로드한 학습 스크립트를 백그라운드(nohup/setsid)로 실행
  4. 15초 대기 후 학습 로그 앞부분을 조회해 정상 기동 여부를 확인
  5. 학습 프로세스가 실제로 떠 있는지 최종 점검

주요 함수:
  - run():  SSH 채널로 원격 명령을 실행하고 표준출력+표준에러를 합쳐 반환
  - main(): SSH 접속부터 학습 기동까지 위 6단계를 순서대로 수행

의존:
  - paramiko (SSH/SFTP 클라이언트 라이브러리)
  - GPU 서버의 학습 가상환경: /opt/nexus-gpu/.venv (Python 3.12)

주의: 접속 정보(HOST/USER/PASSWORD)가 코드에 하드코딩되어 있으므로 사내
폐쇄망(에어갭) 전용 운영 도구로만 사용한다.

작성자: 이현수 / 작성일: 2026-07-05
"""

from __future__ import annotations

import sys
import time

import paramiko

# 원격 명령 결과에 한글 로그가 섞여 나오므로, 콘솔 출력 인코딩을 UTF-8로 고정한다.
# (Windows 기본 콘솔 코드페이지에서 한글이 깨지는 것을 방지)
sys.stdout.reconfigure(encoding="utf-8")

# --- GPU 서버 접속 정보 (사내 폐쇄망 전용) ---
HOST = "192.168.21.112"  # 학습을 수행할 GPU 서버의 LAN 주소
USER = "idino"           # SSH 로그인 계정
PASSWORD = "dkdlelsh@12"  # SSH 로그인 비밀번호

# --- 학습 스크립트 및 로그 경로 ---
# LOCAL_SCRIPT: 개발 PC 안에 있는 원본 학습 스크립트(업로드 대상)
LOCAL_SCRIPT = "scripts/train_qwen_lora_phase2.py"
# REMOTE_SCRIPT: GPU 서버에 업로드되어 실제로 실행될 위치
REMOTE_SCRIPT = "/opt/nexus-gpu/training/train_qwen_lora_phase2.py"
# LOG_PATH: 백그라운드 학습의 표준출력/표준에러가 기록될 로그 파일 경로
LOG_PATH = "/opt/nexus-gpu/training/train_phase2.log"


def run(ssh, cmd, timeout=20):
    """원격 서버에서 명령 한 줄을 실행하고, 그 출력 문자열을 돌려준다.

    표준출력(stdout)과 표준에러(stderr)를 모두 읽어 하나의 문자열로 합쳐
    반환한다. 원격 로그와 에러 메시지를 한 번에 확인하기 위한 편의 함수.

    매개변수:
      ssh:     이미 연결된 paramiko SSHClient 객체
      cmd:     원격에서 실행할 셸 명령 문자열
      timeout: 명령 실행 제한 시간(초). 기본 20초.

    반환:
      명령의 stdout+stderr를 이어 붙인 문자열. 채널에서 예외가 나면
      "(channel error: ...)" 형태의 에러 안내 문자열을 대신 반환한다.
    """
    # 원격 명령을 실행하고 입력/출력/에러 스트림 핸들을 받는다.
    stdin, stdout, stderr = ssh.exec_command(cmd, timeout=timeout)
    try:
        # 표준출력과 표준에러를 각각 읽어 UTF-8로 디코드한 뒤 이어 붙인다.
        # errors="replace": 깨진 바이트가 있어도 예외 없이 대체문자로 처리.
        return (
            stdout.read().decode(errors="replace")
            + stderr.read().decode(errors="replace")
        )
    except Exception as e:
        # 네트워크 끊김 등으로 채널 읽기에 실패해도 스크립트가 죽지 않도록
        # 에러 내용을 문자열로 감싸 반환한다(호출부에서 그대로 출력).
        return f"(channel error: {e})"


def main() -> None:
    """SSH 접속부터 Phase 2 학습 백그라운드 기동까지 전체 절차를 수행한다.

    앞서 설명한 0~5단계를 순서대로 실행하며, 각 단계의 진행 상황과 원격
    출력을 콘솔에 그대로 찍어 준다. 함수가 반환되는 시점에는 학습이 이미
    백그라운드에서 돌고 있는 상태가 된다(이 스크립트는 학습 완료를
    기다리지 않는다).
    """
    # SSH 클라이언트를 만들고, 처음 접속하는 호스트의 키를 자동 수락한다.
    # (폐쇄망 내부 고정 서버라 known_hosts 등록 없이 AutoAdd 정책 사용)
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(HOST, username=USER, password=PASSWORD, timeout=10)

    # === 0단계: 로컬 학습 스크립트를 GPU 서버로 업로드 ===
    print("=== 0. 학습 스크립트 업로드 ===")
    # SFTP 채널을 열어 로컬 파일을 원격 경로로 복사(put)한다.
    sftp = ssh.open_sftp()
    sftp.put(LOCAL_SCRIPT, REMOTE_SCRIPT)
    sftp.close()
    # 업로드된 파일의 크기/시각을 확인해 정상 전송 여부를 눈으로 검증.
    print(run(ssh, f"ls -lh {REMOTE_SCRIPT}"))

    # === 1단계: 추론 vLLM 종료 — 학습에 쓸 VRAM 확보 ===
    print("\n=== 1. vLLM (Worker :8001) 중단 — VRAM 확보 ===")
    # 추론 서버 프로세스를 이름으로 찾아 종료한다. 프로세스가 없어도
    # 파이프라인이 멈추지 않도록 '|| true'로 실패를 흡수한다.
    print(run(ssh, "pkill -f 'vllm.entrypoints.openai.api_server' || true"))
    # 완전 종료 대기: pkill 직후에는 메모리가 아직 반환되지 않을 수 있으므로,
    # 프로세스가 실제로 사라질 때까지 2초 간격으로 최대 10회(약 20초) 확인.
    for i in range(10):
        time.sleep(2)
        # pgrep으로 남은 프로세스를 확인하고, 없으면 '(종료됨)'을 출력하게 한다.
        chk = run(ssh, "pgrep -f 'vllm.entrypoints.openai' || echo '(종료됨)'")
        print(f"  [{i+1}/10] {chk.strip()}")
        # 종료가 확인되면 남은 대기 없이 즉시 반복을 빠져나온다.
        if "(종료됨)" in chk:
            break

    # === 2단계: 학습 시작 전 GPU 메모리 상태 확인 ===
    print("\n=== 2. GPU 상태 (학습 시작 전) ===")
    # nvidia-smi로 사용/전체 메모리를 CSV로 조회 — VRAM이 충분히 비었는지 점검.
    print(run(ssh, "nvidia-smi --query-gpu=memory.used,memory.total --format=csv"))

    # === 3단계: Phase 2 학습을 백그라운드로 기동 ===
    print("\n=== 3. Phase 2 학습 백그라운드 실행 ===")
    # 원격 실행 명령을 조립한다. 핵심 포인트:
    #   - cd .../training : 학습 스크립트가 상대경로 리소스를 찾을 수 있게 이동
    #   - setsid + nohup  : 새 세션에서 실행하여 SSH 연결이 끊겨도 살아남게 함
    #   - </dev/null      : 표준입력을 막아 백그라운드 프로세스가 블록되지 않게
    #   - >{LOG_PATH} 2>&1: 표준출력/에러를 모두 로그 파일로 리다이렉트
    #   - & echo TRAIN_PID: 백그라운드로 던진 뒤 그 PID를 출력해 추적 가능하게
    #   - disown          : 셸의 자식 목록에서 떼어내 완전히 독립 실행시킴
    train_cmd = (
        f"cd /opt/nexus-gpu/training && "
        f"setsid nohup /opt/nexus-gpu/.venv/bin/python3.12 "
        f"{REMOTE_SCRIPT} </dev/null >{LOG_PATH} 2>&1 & "
        f"echo \"TRAIN_PID=$!\"; disown"
    )
    # 백그라운드 실행은 즉시 반환되므로 짧은 timeout(5초)으로 충분하다.
    print(run(ssh, train_cmd, timeout=5))

    # === 4단계: 시작 로그 확인 ===
    print("\n=== 4. 시작 로그 확인 (15초 대기 후) ===")
    # 모델/데이터 로딩에 시간이 걸리므로 15초 기다린 뒤 로그를 확인한다.
    time.sleep(15)
    # 로그 마지막 30줄을 조회해 초기화 진행 상황과 에러 유무를 파악.
    print(run(ssh, f"tail -30 {LOG_PATH}"))

    # === 5단계: 학습 프로세스 생존 여부 최종 점검 ===
    print("\n=== 5. 실행 중인 파이썬 프로세스 ===")
    # 학습 스크립트 이름으로 프로세스를 찾아 살아 있는지 확인한다.
    # 없으면 '(없음!)'을 출력해 기동 실패를 즉시 알 수 있게 한다.
    print(run(ssh, "pgrep -af 'train_qwen_lora_phase2' || echo '(없음!)'"))

    # 모든 단계가 끝나면 SSH 연결을 닫는다(학습은 원격에서 계속 진행됨).
    ssh.close()


if __name__ == "__main__":
    # 스크립트를 직접 실행할 때만 main()을 호출한다(import 시에는 실행 안 함).
    main()
