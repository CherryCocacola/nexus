"""GPU 서버의 Phase 3 LoRA 학습 진행 상태를 원격 SSH로 조회하는 진단 스크립트.

이 스크립트는 Machine A(오케스트레이터)에서 실행되며, paramiko로 GPU 서버
(192.168.21.112)에 SSH 접속한 뒤 여러 셸 명령을 순차 실행해 Phase 3 학습이
지금 어떤 상태인지 사람이 한눈에 파악할 수 있게 출력한다.

점검 항목(출력 순서대로):
  1) 학습 프로세스(train_qwen_lora_phase3)가 살아 있는지 여부
  2) 학습 로그 파일의 마지막 15줄(현재 스텝/loss 등 최신 진행 상황)
  3) 로그에 "Saved to" 문자열이 몇 번 나왔는지 → 체크포인트 저장(완료) 신호
  4) 최종 산출물 metadata.json 파일이 생성됐는지(있으면 학습이 끝났다는 뜻)
  5) GPU 메모리 사용량(nvidia-smi) — 실제로 GPU가 돌고 있는지 교차 확인

주요 함수:
  - run(ssh, cmd): SSH 세션에서 명령 하나를 실행하고 stdout+stderr를 합쳐 반환
  - main(): SSH 접속 → 위 5개 항목을 차례로 출력 → 접속 종료

의존: paramiko(순수 파이썬 SSH 클라이언트). GPU/CUDA를 직접 만지지 않고
오직 원격 셸 명령의 결과 텍스트만 수집하는 읽기 전용(read-only) 진단 도구다.

작성자: 이현수 / 작성일: 2026-07-05
"""
from __future__ import annotations
import sys
import paramiko

# stdout 인코딩을 UTF-8로 강제한다. 원격 명령 결과에 한글/특수문자가 섞여 있어도
# Windows 콘솔(기본 cp949)에서 UnicodeEncodeError로 깨지지 않도록 하기 위함이다.
sys.stdout.reconfigure(encoding="utf-8")

# 점검 대상 경로 상수(GPU 서버 기준 절대 경로).
# LOG : Phase 3 학습 스크립트가 실시간으로 기록하는 로그 파일.
LOG = "/opt/nexus-gpu/training/train_phase3.log"
# META : 학습이 정상 종료되면 생성되는 최종 메타데이터(JSON). 존재 여부가 곧 완료 판정 근거.
META = "/opt/nexus-gpu/checkpoints/qwen35-phase3/metadata.json"


def run(ssh, cmd, timeout=15):
    """열려 있는 SSH 세션에서 셸 명령 하나를 실행하고 결과 텍스트를 돌려준다.

    표준출력(stdout)과 표준에러(stderr)를 모두 읽어 하나의 문자열로 합쳐 반환한다.
    에러 메시지도 함께 봐야 상태 판단(예: 파일 없음)에 도움이 되기 때문이다.

    매개변수:
        ssh     : 이미 connect()된 paramiko SSHClient 객체
        cmd     : 원격에서 실행할 셸 명령 문자열
        timeout : 명령 실행 제한 시간(초). 기본 15초로, 멈춘 명령에 무한 대기하지 않게 한다

    반환:
        stdout + stderr를 이어 붙인 문자열. 디코딩 실패 문자는 errors="replace"로
        치환해(예: '?') 예외 없이 안전하게 문자열화한다.
    """
    # exec_command는 (stdin, stdout, stderr) 세 채널을 반환한다. 여기선 입력은 쓰지 않는다.
    stdin, stdout, stderr = ssh.exec_command(cmd, timeout=timeout)
    # 두 스트림을 각각 읽어 디코딩한 뒤 이어 붙인다. 깨지는 바이트는 replace로 대체한다.
    return stdout.read().decode(errors="replace") + stderr.read().decode(errors="replace")


def main() -> None:
    """GPU 서버에 SSH로 붙어 Phase 3 학습 상태 5종을 순서대로 출력하는 진입점.

    흐름:
      1) SSHClient 생성 후 호스트 키 자동 수락 정책 설정(폐쇄망 내부 서버라 신뢰 가정)
      2) GPU 서버로 접속
      3) 프로세스/로그/완료신호/메타파일/GPU 메모리를 차례로 조회해 콘솔에 출력
      4) 세션 종료

    부작용: 표준출력에 사람이 읽는 형태의 진단 리포트를 찍는다. 반환값은 없다.
    """
    # SSH 클라이언트 생성.
    ssh = paramiko.SSHClient()
    # 서버의 호스트 키가 known_hosts에 없어도 자동으로 추가·수락한다.
    # 폐쇄망(에어갭) 내부의 고정 GPU 서버를 대상으로 하므로 편의상 자동 수락을 쓴다.
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    # GPU 서버로 접속(사용자/비밀번호 인증, 접속 타임아웃 10초).
    ssh.connect("192.168.21.112", username="idino", password="dkdlelsh@12", timeout=10)

    # [1] 학습 프로세스가 살아 있는지 확인.
    # pgrep -af : 명령줄 전체에서 패턴이 포함된 프로세스를 PID와 함께 출력.
    # 매칭이 하나도 없으면 pgrep이 비정상 종료코드를 내므로 '||'로 '(종료됨)'을 대신 출력한다.
    print("=== 프로세스 ===")
    print(run(ssh, "pgrep -af 'train_qwen_lora_phase3' || echo '(종료됨)'"))
    # [2] 학습 로그의 마지막 15줄 — 현재 스텝·loss 등 최신 진행 상황을 본다.
    print("\n=== 로그 꼬리 ===")
    print(run(ssh, f"tail -15 {LOG}"))
    # [3] 로그에서 'Saved to'가 몇 번 나왔는지 카운트 — 체크포인트 저장(완료) 신호.
    # 파일이 없으면 grep이 에러를 내므로 '|| echo 0'으로 0을 대신 출력한다.
    print("\n=== Saved to 검색 (완료 신호) ===")
    print(run(ssh, f"grep -c 'Saved to' {LOG} 2>/dev/null || echo '0'"))
    # [4] 최종 산출물 metadata.json이 실제로 생성됐는지 확인(있으면 학습 완료로 판단).
    # 없으면 ls가 실패하므로 '(미생성)'을 대신 출력한다.
    print("\n=== metadata.json ===")
    print(run(ssh, f"ls -la {META} 2>/dev/null || echo '(미생성)'"))
    # [5] GPU 메모리 사용량/총량 — 학습이 실제로 GPU를 점유 중인지 교차 확인한다.
    print("\n=== GPU ===")
    print(run(ssh, "nvidia-smi --query-gpu=memory.used,memory.total --format=csv"))

    # SSH 세션을 닫아 원격 연결을 정리한다.
    ssh.close()


# 이 파일을 직접 실행했을 때만 main()을 호출한다(import될 때는 실행하지 않음).
if __name__ == "__main__":
    main()
