"""GPU 서버(192.168.21.112)의 llama.cpp Scout 서버를 Qwen3.5-4B-Q4_K_M 모델로 재기동한다.

이 스크립트는 로컬 PC(Machine A)에서 paramiko(SSH)로 GPU 서버에 접속한 뒤,
기존에 떠 있던 llama-server 프로세스를 모두 종료하고 Qwen3.5-4B 모델로 다시 띄운다.
"Scout"는 이 프로젝트에서 8003 포트로 도는 경량 보조 추론 서버를 가리키는 이름이다.

핵심 포인트: 새 서버를 그냥 nohup으로만 띄우면 SSH 세션이 끊길 때 함께 죽을 수 있어서,
`setsid` + `nohup` + `stdin을 /dev/null로` 연결해 SSH 채널과 완전히 분리(detach)한다.
덕분에 paramiko 연결이 끊겨도 GPU 서버의 추론 프로세스는 계속 살아남는다.

주요 함수:
  - run(ssh, cmd): 원격 서버에서 셸 명령 하나를 실행하고 stdout+stderr를 문자열로 반환.
  - main(): SSH 접속 → 기존 종료 → 새 기동 → 헬스체크 폴링 → 로그/상태 확인의 전체 흐름.

의존성: paramiko(SSH 클라이언트). 외부 API 노출은 없고, 운영자가 직접 돌리는 유틸 스크립트다.

작성자: 이현수 / 작성일: 2026-07-05
"""

# from __future__ import annotations: 타입 힌트를 문자열로 지연 평가해
# 순환 참조나 아직 정의되지 않은 타입 참조 문제를 피한다.
from __future__ import annotations

import sys
import time

import paramiko

# 표준 출력 인코딩을 UTF-8로 재설정한다.
# 원격 명령 결과에 한글/특수문자가 섞여 나올 수 있어, Windows 콘솔에서
# 인코딩 오류(UnicodeEncodeError) 없이 출력되도록 보장하는 목적이다.
sys.stdout.reconfigure(encoding="utf-8")

# 접속 대상 GPU 서버 정보 (LAN 내부 주소).
# 에어갭(폐쇄망) 환경 전용이라 자격증명을 스크립트에 직접 둔다.
HOST = "192.168.21.112"
USER = "idino"
PASSWORD = "dkdlelsh@12"

# 서버가 로드할 모델 파일과 실행 바이너리, 로그 경로 (모두 GPU 서버 기준 절대경로).
MODEL_PATH = "/opt/nexus-gpu/models/qwen3.5-4b-gguf/Qwen3.5-4B-Q4_K_M.gguf"
LLAMA_BIN = "/opt/nexus-gpu/llama.cpp/llama-b8808/llama-server"
LOG_PATH = "/opt/nexus-gpu/llama.cpp/scout.log"


def run(ssh: paramiko.SSHClient, cmd: str, timeout: int = 20) -> str:
    """원격 서버에서 셸 명령 하나를 실행하고 그 출력(stdout+stderr)을 문자열로 돌려준다.

    매개변수:
      - ssh: 이미 connect()가 끝난 paramiko SSH 클라이언트.
      - cmd: 원격에서 실행할 셸 명령 문자열.
      - timeout: 명령 실행 제한 시간(초). 넘으면 예외가 나며 아래에서 잡아 처리한다.

    반환: stdout과 stderr를 이어 붙인 문자열. 채널 읽기 중 오류가 나면
          "(channel read error: ...)" 형태의 안내 문자열을 대신 반환한다.
          (예외를 밖으로 던지지 않고 삼켜서, 호출부의 출력 흐름이 끊기지 않게 한다.)
    """
    # exec_command는 (stdin, stdout, stderr) 세 채널을 돌려준다. 여기선 입력은 쓰지 않는다.
    stdin, stdout, stderr = ssh.exec_command(cmd, timeout=timeout)
    try:
        # stdout과 stderr를 각각 읽어 디코딩한 뒤 이어 붙인다.
        # errors="replace": 깨진 바이트가 있어도 예외 없이 대체 문자로 넘어간다.
        return (
            stdout.read().decode(errors="replace")
            + stderr.read().decode(errors="replace")
        )
    except Exception as e:
        # 타임아웃 등 채널 읽기 실패 시, 프로그램을 죽이지 않고 오류 내용을 문자열로 반환.
        return f"(channel read error: {e})"


def main() -> None:
    """Scout 서버 재기동 전체 절차를 순서대로 수행하는 진입점.

    흐름:
      1) SSH 접속 (호스트 키는 자동 수락 정책 사용).
      2) 기존 llama-server 프로세스를 모두 종료.
      3) Qwen3.5-4B 모델로 새 서버를 SSH 세션과 분리해 기동.
      4) /v1/models 엔드포인트를 폴링하며 서버가 뜰 때까지 헬스체크(최대 60초).
      5) 로그 마지막 부분과 모델 응답, 실행 중 프로세스를 확인 출력.
    """
    ssh = paramiko.SSHClient()
    # AutoAddPolicy: 처음 보는 서버의 호스트 키를 자동으로 신뢰 목록에 추가한다.
    # 내부 폐쇄망 전용 스크립트라 known_hosts 사전 등록 없이 바로 붙기 위함이다.
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(HOST, username=USER, password=PASSWORD, timeout=10)

    # --- 1단계: 기존에 떠 있는 llama-server를 모두 죽인다 ---
    # pkill로 종료 시도 후 2초 대기하고, 남은 프로세스가 있는지 pgrep으로 재확인한다.
    # "|| true"는 죽일 프로세스가 없어 pkill이 실패해도 스크립트가 멈추지 않게 한다.
    print("=== 기존 llama-server 종료 ===")
    print(run(ssh, "pkill -f llama-server || true; sleep 2; "
                  "pgrep -af llama-server || echo '(모두 종료됨)'"))

    # --- 2단계: 새 Qwen3.5-4B Scout 서버 기동 ---
    print("\n=== 새 Qwen3.5-4B Scout 서버 기동 (setsid로 완전 분리) ===")
    # setsid + nohup + stdin /dev/null 으로 SSH 세션과 완전 분리
    # - setsid: 새 세션 리더로 띄워 제어 터미널과의 연결을 끊는다.
    # - nohup: 세션 종료(HUP) 신호를 무시하게 한다.
    # - </dev/null: 표준 입력을 비워 SSH 채널 종료의 영향을 받지 않게 한다.
    # - >LOG 2>&1: 표준출력·표준에러를 모두 로그 파일로 보낸다.
    # - "echo PID=$!": 방금 백그라운드로 띄운 프로세스의 PID를 출력해 확인용으로 남긴다.
    # - disown: 셸의 잡 목록에서 제거해 셸 종료 시 딸려 죽지 않게 한다.
    start_cmd = (
        f"setsid nohup {LLAMA_BIN} "
        f"--model {MODEL_PATH} "
        f"--host 0.0.0.0 --port 8003 "
        f"--ctx-size 4096 --threads 8 --batch-size 512 "
        f"--api-key local-key --jinja "
        f"</dev/null >{LOG_PATH} 2>&1 & "
        f"echo \"PID=$!\"; disown"
    )
    # timeout=5: 백그라운드로 분리된 명령이라 즉시 반환되므로 짧게 잡아도 충분하다.
    print(run(ssh, start_cmd, timeout=5))

    # --- 3단계: 서버가 실제로 응답할 때까지 헬스체크 폴링 ---
    # 서버 준비 폴링 (최대 60초)
    # 3초 간격으로 최대 20번 = 최대 60초 동안 /v1/models 응답 코드를 확인한다.
    # 모델 로딩에 시간이 걸리므로 곧바로 200이 나오지 않는 게 정상이다.
    print("\n=== 헬스체크 ===")
    for i in range(20):
        time.sleep(3)
        try:
            # curl로 HTTP 상태 코드만 뽑아온다(-o /dev/null 로 본문은 버림).
            # -w '%{http_code}': 응답 코드만 출력. --max-time 3: 개별 요청 3초 제한.
            health = run(
                ssh,
                "curl -s -o /dev/null -w '%{http_code}' "
                "http://localhost:8003/v1/models "
                "-H 'Authorization: Bearer local-key' --max-time 3",
                timeout=8,
            ).strip()
        except Exception as e:
            # 헬스체크 요청 자체가 실패해도 루프를 계속 돌리기 위해 오류를 문자열로 담는다.
            health = f"(err: {e})"
        print(f"  [{i+1}/20] /v1/models → {health}")
        # 200이면 서버가 정상 기동된 것이므로 폴링을 조기 종료한다.
        if health == "200":
            break

    # --- 4단계: 진단용 정보 출력 (로그 / 모델 응답 / 프로세스) ---
    # 서버 로그 마지막 30줄을 보여줘 기동 실패 시 원인 파악을 돕는다.
    print("\n=== 서버 로그 (마지막 30줄) ===")
    print(run(ssh, f"tail -30 {LOG_PATH}"))

    # 실제 /v1/models 응답 본문을 그대로 출력해 로드된 모델 정보를 확인한다.
    print("\n=== /v1/models 응답 ===")
    print(run(
        ssh,
        "curl -s http://localhost:8003/v1/models "
        "-H 'Authorization: Bearer local-key' --max-time 5",
    ))

    # 최종적으로 llama-server 프로세스가 살아 있는지 PID와 함께 확인한다.
    print("\n=== 실행 중 프로세스 확인 ===")
    print(run(ssh, "pgrep -af llama-server"))

    # SSH 연결 정리. (원격 서버 프로세스는 분리되어 있어 이 종료의 영향을 받지 않는다.)
    ssh.close()


if __name__ == "__main__":
    main()
