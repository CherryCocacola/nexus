"""vLLM Worker 서버 재시작 — nexus-phase1, nexus-phase2 LoRA 모두 핫로드."""

from __future__ import annotations

import sys
import time

import paramiko

sys.stdout.reconfigure(encoding="utf-8")

HOST = "192.168.22.28"
USER = "idino"
PASSWORD = "dkdlelsh@12"

LOG_PATH = "/opt/nexus-gpu/vllm.log"
VENV = "/opt/nexus-gpu/.venv/bin/python3.12"
MODEL = "/opt/nexus-gpu/models/qwen3.5-27b-awq"
PHASE1 = "nexus-phase1=/opt/nexus-gpu/checkpoints/qwen35-phase1"
PHASE2 = "nexus-phase2=/opt/nexus-gpu/checkpoints/qwen35-phase2"
PHASE3 = "nexus-phase3=/opt/nexus-gpu/checkpoints/qwen35-phase3"


def run(ssh, cmd, timeout=15):
    stdin, stdout, stderr = ssh.exec_command(cmd, timeout=timeout)
    try:
        return stdout.read().decode(errors="replace") + stderr.read().decode(errors="replace")
    except Exception as e:
        return f"(channel err: {e})"


def main() -> None:
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(HOST, username=USER, password=PASSWORD, timeout=10)

    print("=== Phase 2 체크포인트 확인 ===")
    print(run(ssh, "ls -la /opt/nexus-gpu/checkpoints/qwen35-phase2/"))

    print("\n=== 기존 vLLM 종료 (학습 후 남은 프로세스 정리) ===")
    print(run(ssh, "pkill -f 'vllm.entrypoints.openai' || true; sleep 2; "
                  "pgrep -f vllm || echo '(없음)'"))
    # GPU 메모리 해제 대기
    for i in range(5):
        time.sleep(3)
        used = run(ssh, "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits").strip()
        print(f"  [{i+1}/5] GPU used: {used}MiB")

    print("\n=== vLLM 기동 (nexus-phase1 + nexus-phase2 LoRA) ===")
    cmd = (
        f"setsid nohup {VENV} -m vllm.entrypoints.openai.api_server "
        f"--model {MODEL} "
        f"--max-model-len 8192 --gpu-memory-utilization 0.90 "
        f"--port 8001 --host 0.0.0.0 --trust-remote-code "
        f"--served-model-name qwen3.5-27b "
        f"--enable-prefix-caching --enforce-eager "
        f"--enable-auto-tool-choice --tool-call-parser qwen3_xml "
        f"--enable-lora --max-lora-rank 16 "
        f"--lora-modules {PHASE1} {PHASE2} {PHASE3} "
        f"</dev/null >{LOG_PATH} 2>&1 & "
        f"echo \"VLLM_PID=$!\"; disown"
    )
    print(run(ssh, cmd, timeout=5))

    print("\n=== 서비스 준비 폴링 (최대 180초) ===")
    for i in range(60):
        time.sleep(3)
        try:
            code = run(
                ssh,
                "curl -s -o /dev/null -w '%{http_code}' "
                "http://localhost:8001/v1/models --max-time 3",
                timeout=8,
            ).strip()
        except Exception as e:
            code = f"(err: {e})"
        print(f"  [{i+1}/60] /v1/models → {code}")
        if code == "200":
            break

    print("\n=== /v1/models 응답 (LoRA 노출 확인) ===")
    print(run(ssh, "curl -s http://localhost:8001/v1/models | python3 -m json.tool | head -40"))

    print("\n=== vLLM 로그 (마지막 25줄) ===")
    print(run(ssh, f"tail -25 {LOG_PATH}"))

    ssh.close()


if __name__ == "__main__":
    main()
