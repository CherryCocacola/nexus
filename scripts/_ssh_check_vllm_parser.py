"""vLLM tool_call parser 로그 + 지원 parser 목록 확인."""
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

    print("=== vLLM 로그에서 tool_call 관련 ===")
    print(run(ssh, "grep -Ei 'tool.call|qwen3|hermes|parser' /opt/nexus-gpu/vllm.log | tail -30"))

    print("\n=== 최근 /v1/chat 요청들의 본문/tool_calls 응답 샘플 ===")
    print(run(ssh, "grep -Ei 'ParseError|unable to parse|tool.*unknown' /opt/nexus-gpu/vllm.log | tail -20"))

    print("\n=== vLLM이 지원하는 tool-call-parser 목록 ===")
    print(run(ssh, "/opt/nexus-gpu/.venv/bin/python3.12 -c "
                  "\"from vllm.entrypoints.openai.tool_parsers import ToolParserManager; "
                  "print(sorted(ToolParserManager.tool_parsers.keys()))\" 2>&1"))

    print("\n=== Phase 2 체크포인트의 chat_template에 tool_call 포맷 ===")
    print(run(ssh, "grep -n 'tool_call\\|<tool' "
                  "/opt/nexus-gpu/checkpoints/qwen35-phase2/chat_template.jinja | head -15"))

    ssh.close()


if __name__ == "__main__":
    main()
