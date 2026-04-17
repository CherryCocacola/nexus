"""Qwen3.5 chat template에 enable_thinking 같은 컨트롤 파라미터가 있는지 확인."""
from __future__ import annotations
import sys
import paramiko

sys.stdout.reconfigure(encoding="utf-8")


def run(ssh, cmd, timeout=15):
    stdin, stdout, stderr = ssh.exec_command(cmd, timeout=timeout)
    return stdout.read().decode(errors="replace") + stderr.read().decode(errors="replace")


def main() -> None:
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect("192.168.22.28", username="idino", password="dkdlelsh@12", timeout=10)

    # Phase 2 체크포인트의 chat_template.jinja 검사
    print("=== Phase 2 LoRA의 chat_template.jinja에서 thinking 관련 토큰 ===")
    print(run(ssh, "grep -n 'enable_thinking\\|think' "
                  "/opt/nexus-gpu/checkpoints/qwen35-phase2/chat_template.jinja | head -20"))

    print("\n=== 원본 Worker 모델(qwen3.5-27b)의 chat template ===")
    print(run(ssh, "ls /opt/nexus-gpu/models/qwen3.5-27b/ | grep -i 'chat\\|template'"))

    print("\n=== tokenizer_config.json에 chat template 존재? ===")
    print(run(ssh, "grep -o 'chat_template.\\{100\\}' "
                  "/opt/nexus-gpu/models/qwen3.5-27b/tokenizer_config.json 2>/dev/null "
                  "| head -c 500 || echo '(없음)'"))

    print("\n=== enable_thinking 관련 문구 전체 검색 ===")
    print(run(ssh, "grep -l 'enable_thinking' "
                  "/opt/nexus-gpu/checkpoints/qwen35-phase2/* 2>/dev/null"))

    # vLLM 로그에 chat template 어떻게 인식했는지
    print("\n=== vLLM 로그에서 chat_template 관련 ===")
    print(run(ssh, "grep -Ei 'chat.template|enable_thinking|think' "
                  "/opt/nexus-gpu/vllm.log | tail -10 || echo '(없음)'"))

    ssh.close()


if __name__ == "__main__":
    main()
