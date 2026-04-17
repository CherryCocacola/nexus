"""Phase 3 학습 파이프라인 — Phase 2와 동일 구조, 스크립트/출력만 변경."""

from __future__ import annotations

import sys
import time

import paramiko

sys.stdout.reconfigure(encoding="utf-8")

HOST = "192.168.22.28"
USER = "idino"
PASSWORD = "dkdlelsh@12"

LOCAL_SCRIPT = "scripts/train_qwen_lora_phase3.py"
REMOTE_SCRIPT = "/opt/nexus-gpu/training/train_qwen_lora_phase3.py"
LOG_PATH = "/opt/nexus-gpu/training/train_phase3.log"


def run(ssh, cmd, timeout=20):
    stdin, stdout, stderr = ssh.exec_command(cmd, timeout=timeout)
    try:
        return stdout.read().decode(errors="replace") + stderr.read().decode(errors="replace")
    except Exception as e:
        return f"(channel err: {e})"


def main() -> None:
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(HOST, username=USER, password=PASSWORD, timeout=10)

    print("=== 0. 학습 스크립트 업로드 ===")
    sftp = ssh.open_sftp()
    sftp.put(LOCAL_SCRIPT, REMOTE_SCRIPT)
    sftp.close()
    print(run(ssh, f"ls -lh {REMOTE_SCRIPT}"))

    print("\n=== 1. vLLM Worker 중단 ===")
    print(run(ssh, "pkill -f 'vllm.entrypoints.openai.api_server' || true"))
    for i in range(10):
        time.sleep(2)
        chk = run(ssh, "pgrep -f 'vllm.entrypoints.openai' || echo '(종료됨)'")
        print(f"  [{i+1}/10] {chk.strip()}")
        if "(종료됨)" in chk:
            break

    print("\n=== 2. GPU 확보 확인 ===")
    print(run(ssh, "nvidia-smi --query-gpu=memory.used,memory.total --format=csv"))

    print("\n=== 3. Phase 3 학습 실행 ===")
    train_cmd = (
        f"cd /opt/nexus-gpu/training && "
        f"setsid nohup /opt/nexus-gpu/.venv/bin/python3.12 "
        f"{REMOTE_SCRIPT} </dev/null >{LOG_PATH} 2>&1 & "
        f"echo \"TRAIN_PID=$!\"; disown"
    )
    print(run(ssh, train_cmd, timeout=5))

    print("\n=== 4. 시작 로그 (15초 후) ===")
    time.sleep(15)
    print(run(ssh, f"tail -30 {LOG_PATH}"))

    print("\n=== 5. 프로세스 ===")
    print(run(ssh, "pgrep -af 'train_qwen_lora_phase3' || echo '(없음)'"))

    ssh.close()


if __name__ == "__main__":
    main()
