"""vLLM의 LoRA 로드 여부를 확인한다 (nexus-phase2 존재 확인)."""
from __future__ import annotations
import sys
import paramiko

sys.stdout.reconfigure(encoding="utf-8")


def main() -> None:
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect("192.168.22.28", username="idino", password="dkdlelsh@12", timeout=10)

    stdin, stdout, stderr = ssh.exec_command(
        "curl -s http://localhost:8001/v1/models | "
        "python3 -c \"import json,sys; d=json.load(sys.stdin); "
        "print([m['id'] for m in d['data']])\"",
        timeout=15,
    )
    print(stdout.read().decode())
    ssh.close()


if __name__ == "__main__":
    main()
