"""Phase 3 학습 진행 상태 조회."""
from __future__ import annotations
import sys
import paramiko

sys.stdout.reconfigure(encoding="utf-8")

LOG = "/opt/nexus-gpu/training/train_phase3.log"
META = "/opt/nexus-gpu/checkpoints/qwen35-phase3/metadata.json"


def run(ssh, cmd, timeout=15):
    stdin, stdout, stderr = ssh.exec_command(cmd, timeout=timeout)
    return stdout.read().decode(errors="replace") + stderr.read().decode(errors="replace")


def main() -> None:
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect("192.168.22.28", username="idino", password="dkdlelsh@12", timeout=10)

    print("=== 프로세스 ===")
    print(run(ssh, "pgrep -af 'train_qwen_lora_phase3' || echo '(종료됨)'"))
    print("\n=== 로그 꼬리 ===")
    print(run(ssh, f"tail -15 {LOG}"))
    print("\n=== Saved to 검색 (완료 신호) ===")
    print(run(ssh, f"grep -c 'Saved to' {LOG} 2>/dev/null || echo '0'"))
    print("\n=== metadata.json ===")
    print(run(ssh, f"ls -la {META} 2>/dev/null || echo '(미생성)'"))
    print("\n=== GPU ===")
    print(run(ssh, "nvidia-smi --query-gpu=memory.used,memory.total --format=csv"))

    ssh.close()


if __name__ == "__main__":
    main()
