"""새 bootstrap 데이터를 GPU 서버의 학습 데이터 디렉토리로 전송."""

from __future__ import annotations

import sys

import paramiko

sys.stdout.reconfigure(encoding="utf-8")

HOST = "192.168.22.28"
USER = "idino"
PASSWORD = "dkdlelsh@12"

LOCAL_PATH = "data/bootstrap/bootstrap_data.jsonl"
REMOTE_PATH = "/opt/nexus-gpu/training/bootstrap_data.jsonl"


def main() -> None:
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(HOST, username=USER, password=PASSWORD, timeout=10)

    # 기존 파일 백업
    stdin, stdout, stderr = ssh.exec_command(
        f"cp {REMOTE_PATH} {REMOTE_PATH}.phase1.bak 2>/dev/null; "
        f"ls -lh {REMOTE_PATH}* 2>/dev/null",
        timeout=10,
    )
    print("=== 기존 파일 백업 ===")
    print(stdout.read().decode(errors="replace"))

    # SFTP 업로드
    print(f"=== 업로드: {LOCAL_PATH} → {REMOTE_PATH} ===")
    sftp = ssh.open_sftp()
    sftp.put(LOCAL_PATH, REMOTE_PATH)
    sftp.close()

    # 크기 + 첫/마지막 줄 확인
    stdin, stdout, stderr = ssh.exec_command(
        f"ls -lh {REMOTE_PATH} && echo '---' && "
        f"wc -l {REMOTE_PATH} && echo '---' && "
        f"head -1 {REMOTE_PATH} | head -c 200 && echo",
        timeout=10,
    )
    print(stdout.read().decode(errors="replace"))

    ssh.close()


if __name__ == "__main__":
    main()
