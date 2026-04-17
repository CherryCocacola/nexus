"""Phase 2 학습 진행 상태 한 번 조회."""

from __future__ import annotations

import sys

import paramiko

sys.stdout.reconfigure(encoding="utf-8")

HOST = "192.168.22.28"
USER = "idino"
PASSWORD = "dkdlelsh@12"
LOG = "/opt/nexus-gpu/training/train_phase2.log"
META = "/opt/nexus-gpu/checkpoints/qwen35-phase2/metadata.json"


def run(ssh, cmd, timeout=15):
    stdin, stdout, stderr = ssh.exec_command(cmd, timeout=timeout)
    return stdout.read().decode(errors="replace") + stderr.read().decode(errors="replace")


def main() -> None:
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(HOST, username=USER, password=PASSWORD, timeout=10)

    print("=== 프로세스 ===")
    print(run(ssh, "pgrep -af 'train_qwen_lora_phase2' || echo '(종료됨)'"))

    print("\n=== 로그 꼬리 (15줄) ===")
    print(run(ssh, f"tail -15 {LOG}"))

    print("\n=== 완료 여부 (metadata.json 존재?) ===")
    print(run(ssh, f"ls -la {META} 2>/dev/null || echo '(아직 생성 안 됨)'"))

    print("\n=== GPU 메모리 ===")
    print(run(ssh, "nvidia-smi --query-gpu=memory.used,memory.total --format=csv"))

    ssh.close()


if __name__ == "__main__":
    main()
