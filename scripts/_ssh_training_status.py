"""Phase 2 학습(LoRA 파인튜닝) 진행 상태를 GPU 서버에서 원격으로 1회 조회하는 진단 스크립트.

이 파일은 Machine B(GPU 서버, 192.168.21.112)에 SSH로 접속해서 아래 4가지를 한 번에
확인하고 터미널에 출력한다. 상시 감시(watch)가 아니라 "지금 이 순간의 스냅샷"만 찍는 용도다.

확인 항목:
  1. 학습 프로세스(train_qwen_lora_phase2)가 아직 살아 있는지 (pgrep)
  2. 학습 로그 파일의 마지막 15줄 — 최근 step/loss 등 진행 상황
  3. 학습 완료 산출물인 metadata.json 이 생성됐는지 (있으면 사실상 완료로 판단)
  4. GPU 메모리 사용량 — 학습이 실제로 VRAM 을 점유 중인지

구성 요소:
  - run(ssh, cmd, timeout): 원격 명령 1개를 실행하고 stdout+stderr 를 합쳐 문자열로 반환
  - main(): SSH 연결 → 위 4개 명령을 순서대로 실행 후 출력 → 연결 종료

의존성: paramiko(SSH 클라이언트). 에어갭 원칙에 따라 접속 대상은 LAN 내부 주소만 사용한다.

주의: 아래 HOST/USER/PASSWORD 는 사내 폐쇄망 GPU 서버 접속용 상수다. 운영 비밀번호가
평문으로 박혀 있으므로 이 스크립트는 내부 진단 용도로만 쓰고 외부로 공유하지 않는다.

작성자: 이현수 / 작성일: 2026-07-05
"""

from __future__ import annotations

import sys

import paramiko

# 한글 출력이 깨지지 않도록 표준출력 인코딩을 UTF-8 로 강제한다.
# (Windows 콘솔 기본 코드페이지가 cp949 라 이 설정이 없으면 한글이 깨질 수 있다.)
sys.stdout.reconfigure(encoding="utf-8")

# --- GPU 서버 접속 정보 및 확인 대상 경로 (LAN 내부 전용) ---
HOST = "192.168.21.112"  # Machine B(GPU 서버)의 LAN 주소
USER = "idino"  # SSH 로그인 계정
PASSWORD = "dkdlelsh@12"  # SSH 비밀번호 — 폐쇄망 내부 진단용
LOG = "/opt/nexus-gpu/training/train_phase2.log"  # Phase 2 학습 로그 파일 경로
META = "/opt/nexus-gpu/checkpoints/qwen35-phase2/metadata.json"  # 완료 시 생성되는 메타데이터


def run(ssh, cmd, timeout=15):
    """원격 SSH 세션에서 명령 하나를 실행하고 그 출력(표준출력+표준에러)을 문자열로 돌려준다.

    학습 상태 조회는 정상 출력(stdout)뿐 아니라 에러 메시지(stderr)도 봐야 상황 파악이
    되므로, 둘을 모두 읽어 이어 붙여 반환한다. 디코딩 실패 문자는 버리지 않고 대체 문자로
    치환(errors="replace")해서 중간에 예외로 끊기지 않게 한다.

    매개변수:
      ssh: 이미 연결된 paramiko.SSHClient 인스턴스
      cmd: 원격 서버에서 실행할 셸 명령 문자열
      timeout: 명령 실행 제한 시간(초). 기본 15초 — 서버가 멈춰도 여기 걸려 무한 대기하지 않음

    반환: stdout 문자열 뒤에 stderr 문자열을 이어 붙인 하나의 문자열
    """
    # exec_command 는 (stdin, stdout, stderr) 반환. 입력은 안 쓰므로 stdin 은 무시.
    stdin, stdout, stderr = ssh.exec_command(cmd, timeout=timeout)
    # 두 채널을 각각 읽어 디코딩한 뒤 합쳐서 반환한다.
    return stdout.read().decode(errors="replace") + stderr.read().decode(errors="replace")


def main() -> None:
    """GPU 서버에 접속해 학습 상태 4종을 순서대로 조회·출력하고 연결을 닫는 진입점.

    흐름:
      1) SSH 클라이언트를 만들고 호스트 키를 자동 수락(AutoAddPolicy)한 뒤 접속한다.
      2) 프로세스 생존 / 로그 꼬리 / 완료 산출물 / GPU 메모리를 차례로 조회해 화면에 찍는다.
      3) 마지막에 SSH 연결을 정리한다.

    반환값은 없다(None). 결과는 전부 표준출력으로 사람이 눈으로 보는 용도다.
    """
    # SSH 클라이언트 생성. 폐쇄망 서버라 known_hosts 에 키가 없을 수 있어 자동 추가 정책을 쓴다.
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    # 접속 자체가 안 되면 여기서 예외가 나며 즉시 중단된다(연결 타임아웃 10초).
    ssh.connect(HOST, username=USER, password=PASSWORD, timeout=10)

    # (1) 학습 프로세스가 살아 있는지 확인. pgrep -af 는 명령줄 전체를 매칭해 PID+커맨드로 보여준다.
    #     매칭이 없으면(=이미 끝났으면) '|| echo' 로 '(종료됨)' 을 대신 출력한다.
    print("=== 프로세스 ===")
    print(run(ssh, "pgrep -af 'train_qwen_lora_phase2' || echo '(종료됨)'"))

    # (2) 학습 로그의 마지막 15줄. 최근 step/loss 등 진행 상황을 빠르게 훑는다.
    print("\n=== 로그 꼬리 (15줄) ===")
    print(run(ssh, f"tail -15 {LOG}"))

    # (3) 완료 판단: 학습이 끝나면 metadata.json 이 생성된다. 파일이 있으면 ls 상세정보를,
    #     없으면 '(아직 생성 안 됨)' 을 출력한다. 2>/dev/null 로 파일 없음 에러 메시지는 숨긴다.
    print("\n=== 완료 여부 (metadata.json 존재?) ===")
    print(run(ssh, f"ls -la {META} 2>/dev/null || echo '(아직 생성 안 됨)'"))

    # (4) GPU 메모리 사용량. 학습이 실제로 VRAM 을 잡고 있는지 used/total 을 csv 로 확인한다.
    print("\n=== GPU 메모리 ===")
    print(run(ssh, "nvidia-smi --query-gpu=memory.used,memory.total --format=csv"))

    # 조회가 끝났으니 SSH 세션을 닫아 자원을 정리한다.
    ssh.close()


if __name__ == "__main__":
    # 스크립트를 직접 실행했을 때만 main() 을 호출한다(import 시에는 실행되지 않음).
    main()
