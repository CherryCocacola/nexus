"""llama.cpp Scout 서버를 Qwen3.5-4B-Q4_K_M로 재기동.

nohup + setsid로 완전히 분리해 paramiko 채널이 끊겨도 프로세스 유지.
"""

from __future__ import annotations

import sys
import time

import paramiko

sys.stdout.reconfigure(encoding="utf-8")

HOST = "192.168.22.28"
USER = "idino"
PASSWORD = "dkdlelsh@12"

MODEL_PATH = "/opt/nexus-gpu/models/qwen3.5-4b-gguf/Qwen3.5-4B-Q4_K_M.gguf"
LLAMA_BIN = "/opt/nexus-gpu/llama.cpp/llama-b8808/llama-server"
LOG_PATH = "/opt/nexus-gpu/llama.cpp/scout.log"


def run(ssh: paramiko.SSHClient, cmd: str, timeout: int = 20) -> str:
    stdin, stdout, stderr = ssh.exec_command(cmd, timeout=timeout)
    try:
        return (
            stdout.read().decode(errors="replace")
            + stderr.read().decode(errors="replace")
        )
    except Exception as e:
        return f"(channel read error: {e})"


def main() -> None:
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(HOST, username=USER, password=PASSWORD, timeout=10)

    print("=== 기존 llama-server 종료 ===")
    print(run(ssh, "pkill -f llama-server || true; sleep 2; "
                  "pgrep -af llama-server || echo '(모두 종료됨)'"))

    print("\n=== 새 Qwen3.5-4B Scout 서버 기동 (setsid로 완전 분리) ===")
    # setsid + nohup + stdin /dev/null 으로 SSH 세션과 완전 분리
    start_cmd = (
        f"setsid nohup {LLAMA_BIN} "
        f"--model {MODEL_PATH} "
        f"--host 0.0.0.0 --port 8003 "
        f"--ctx-size 4096 --threads 8 --batch-size 512 "
        f"--api-key local-key --jinja "
        f"</dev/null >{LOG_PATH} 2>&1 & "
        f"echo \"PID=$!\"; disown"
    )
    print(run(ssh, start_cmd, timeout=5))

    # 서버 준비 폴링 (최대 60초)
    print("\n=== 헬스체크 ===")
    for i in range(20):
        time.sleep(3)
        try:
            health = run(
                ssh,
                "curl -s -o /dev/null -w '%{http_code}' "
                "http://localhost:8003/v1/models "
                "-H 'Authorization: Bearer local-key' --max-time 3",
                timeout=8,
            ).strip()
        except Exception as e:
            health = f"(err: {e})"
        print(f"  [{i+1}/20] /v1/models → {health}")
        if health == "200":
            break

    print("\n=== 서버 로그 (마지막 30줄) ===")
    print(run(ssh, f"tail -30 {LOG_PATH}"))

    print("\n=== /v1/models 응답 ===")
    print(run(
        ssh,
        "curl -s http://localhost:8003/v1/models "
        "-H 'Authorization: Bearer local-key' --max-time 5",
    ))

    print("\n=== 실행 중 프로세스 확인 ===")
    print(run(ssh, "pgrep -af llama-server"))

    ssh.close()


if __name__ == "__main__":
    main()
