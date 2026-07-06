"""vLLM Worker 서버 재시작 스크립트 — LoRA 어댑터 3종 핫로드 자동화.

이 스크립트는 GPU 서버(Machine B, 192.168.21.112)에 SSH로 접속해서
vLLM 추론 서버를 "완전히 껐다가 다시 켜는" 운영용 도구다. 학습(QLoRA)을
막 끝낸 뒤 새 체크포인트를 실제 서비스에 반영할 때 주로 사용한다.

전체 흐름(순서대로):
  1) SSH 접속 후 Phase 2 체크포인트 디렉토리가 실제로 있는지 눈으로 확인
  2) 이전에 떠 있던 vLLM 프로세스를 종료(pkill)하고 GPU 메모리가 빠지길 대기
  3) nexus-phase1 / phase2 / phase3 세 개의 LoRA 어댑터를 붙여 vLLM 재기동
  4) /v1/models 엔드포인트가 200을 줄 때까지 최대 180초 폴링
  5) 모델 목록(LoRA 노출 여부)과 vLLM 로그 마지막 부분을 출력해 결과 확인

주요 함수:
  - run(ssh, cmd, ...)  : 원격 셸 명령 1개를 실행하고 표준출력+표준에러를 합쳐 반환
  - main()              : 위 1~5단계를 순서대로 수행하는 진입점

의존: paramiko(SSH 클라이언트). Machine B의 vLLM(OpenAI 호환 API)만 다루며
이 스크립트 자체가 GPU/CUDA를 직접 만지지는 않는다.

작성자: 이현수 / 작성일: 2026-07-05
"""

from __future__ import annotations

import sys
import time

import paramiko

# 윈도우 콘솔에서 한글 출력이 깨지지 않도록 표준출력 인코딩을 UTF-8로 강제한다.
sys.stdout.reconfigure(encoding="utf-8")

# --- GPU 서버(Machine B) 접속 정보 -------------------------------------------
# 폐쇄망(에어갭) 내부 LAN 주소만 사용한다. 비밀번호가 코드에 박혀 있으니
# 이 스크립트 파일 자체의 접근 권한 관리에 유의할 것.
HOST = "192.168.21.112"
USER = "idino"
PASSWORD = "dkdlelsh@12"

# --- vLLM 실행에 필요한 원격 서버 경로/자원 ----------------------------------
LOG_PATH = "/opt/nexus-gpu/vllm.log"                       # vLLM 표준출력·에러 로그 파일
VENV = "/opt/nexus-gpu/.venv/bin/python3.12"               # vLLM을 돌릴 파이썬(가상환경) 실행기
MODEL = "/opt/nexus-gpu/models/qwen3.5-27b-awq"            # 베이스 모델(AWQ 양자화) 디렉토리
# 아래 3개는 "어댑터이름=체크포인트경로" 형식. vLLM --lora-modules 인자에 그대로 넘긴다.
# 각 Phase는 서로 다른 학습 단계에서 나온 LoRA 어댑터이며, 동시에 핫로드해 둔다.
PHASE1 = "nexus-phase1=/opt/nexus-gpu/checkpoints/qwen35-phase1"
PHASE2 = "nexus-phase2=/opt/nexus-gpu/checkpoints/qwen35-phase2"
PHASE3 = "nexus-phase3=/opt/nexus-gpu/checkpoints/qwen35-phase3"


def run(ssh, cmd, timeout=15):
    """원격 서버에서 셸 명령 하나를 실행하고 출력 문자열을 돌려준다.

    표준출력(stdout)과 표준에러(stderr)를 모두 읽어 하나로 합쳐 반환하므로,
    호출부는 print()만 해도 실행 결과와 에러 메시지를 한꺼번에 볼 수 있다.

    매개변수:
        ssh     : 이미 connect() 된 paramiko SSHClient 객체
        cmd     : 원격에서 실행할 셸 명령 문자열
        timeout : 명령 실행 제한 시간(초). 기본 15초.

    반환:
        명령의 stdout+stderr를 합친 문자열. 채널 읽기 중 예외가 나면
        디코딩 실패 대신 "(channel err: ...)" 형태의 안내 문자열을 반환한다.
    """
    stdin, stdout, stderr = ssh.exec_command(cmd, timeout=timeout)
    try:
        # 바이트로 읽어 UTF-8 디코딩. 깨진 바이트는 errors="replace"로 대체해
        # 예외 없이 최대한 읽어낸다(로그 확인이 목적이라 손실 허용).
        return stdout.read().decode(errors="replace") + stderr.read().decode(errors="replace")
    except Exception as e:
        # 네트워크/채널 문제로 읽기가 실패해도 스크립트 전체가 죽지 않도록 방어.
        return f"(channel err: {e})"


def main() -> None:
    """vLLM 재시작 절차 전체(접속→종료→기동→확인)를 순서대로 수행하는 진입점.

    사람이 실행하고 콘솔 출력을 눈으로 보며 상태를 판단하는 운영 스크립트라,
    각 단계마다 제목과 결과를 print로 찍는다. 실패해도 다음 단계로 넘어가며
    최종적으로 모델 목록과 로그를 보여 주어 원인 파악을 돕는다.
    """
    # SSH 클라이언트 생성 후 접속. 폐쇄망 내부 서버라 호스트 키를 자동 수락한다
    # (AutoAddPolicy). 외부망이라면 보안상 권장되지 않는 설정이니 주의.
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(HOST, username=USER, password=PASSWORD, timeout=10)

    # [1단계] 재기동 전에 Phase 2 체크포인트가 실제로 존재하는지 먼저 확인한다.
    # 학습이 덜 끝났거나 경로가 틀리면 여기서 바로 이상을 눈치챌 수 있다.
    print("=== Phase 2 체크포인트 확인 ===")
    print(run(ssh, "ls -la /opt/nexus-gpu/checkpoints/qwen35-phase2/"))

    # [2단계] 기존 vLLM 프로세스를 종료한다. 학습 직후엔 이전 서버가 GPU를
    # 점유한 채 남아 있을 수 있어, 새로 띄우기 전에 반드시 정리해야 한다.
    print("\n=== 기존 vLLM 종료 (학습 후 남은 프로세스 정리) ===")
    # pkill로 종료 시도(없으면 실패해도 || true로 무시), 2초 뒤 남은 프로세스 확인.
    print(run(ssh, "pkill -f 'vllm.entrypoints.openai' || true; sleep 2; "
                  "pgrep -f vllm || echo '(없음)'"))
    # GPU 메모리 해제 대기: 프로세스를 죽여도 VRAM이 즉시 비지 않으므로
    # 3초 간격으로 5번(약 15초) 사용량을 찍어 실제로 내려가는지 눈으로 확인한다.
    for i in range(5):
        time.sleep(3)
        used = run(ssh, "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits").strip()
        print(f"  [{i+1}/5] GPU used: {used}MiB")

    # [3단계] vLLM을 새로 기동한다. 아래 명령은 하나의 긴 셸 커맨드로,
    # setsid+nohup+disown으로 SSH 세션이 끊겨도 서버가 계속 살아 있게 만든다.
    print("\n=== vLLM 기동 (nexus-phase1 + nexus-phase2 LoRA) ===")
    cmd = (
        # setsid nohup: 터미널과 분리해 백그라운드 데몬처럼 실행
        f"setsid nohup {VENV} -m vllm.entrypoints.openai.api_server "
        f"--model {MODEL} "
        # 컨텍스트 길이 8192, GPU 메모리 90%까지 사용
        f"--max-model-len 8192 --gpu-memory-utilization 0.90 "
        # 8001 포트로 전체 인터페이스(0.0.0.0)에 개방, 원격 코드 신뢰 허용
        f"--port 8001 --host 0.0.0.0 --trust-remote-code "
        # 클라이언트가 부를 모델 이름을 qwen3.5-27b로 고정
        f"--served-model-name qwen3.5-27b "
        # 프리픽스 캐시로 프롬프트 재사용 가속, eager 모드로 그래프 컴파일 생략
        f"--enable-prefix-caching --enforce-eager "
        # 자동 tool 선택 + Qwen3 XML 형식 tool-call 파서 활성화
        f"--enable-auto-tool-choice --tool-call-parser qwen3_xml "
        # LoRA 기능 켜기, 어댑터 최대 rank 16
        f"--enable-lora --max-lora-rank 16 "
        # phase1/2/3 세 어댑터를 동시에 로드
        f"--lora-modules {PHASE1} {PHASE2} {PHASE3} "
        # 표준입력 차단, 모든 출력은 로그 파일로, 백그라운드(&) 실행
        f"</dev/null >{LOG_PATH} 2>&1 & "
        # 방금 띄운 프로세스 PID를 출력하고 셸에서 분리(disown)
        f"echo \"VLLM_PID=$!\"; disown"
    )
    # 기동 명령은 즉시 반환되므로 짧은 타임아웃(5초)이면 충분하다.
    print(run(ssh, cmd, timeout=5))

    # [4단계] 서비스 준비 폴링. vLLM은 모델·LoRA 로딩에 시간이 걸리므로
    # /v1/models가 HTTP 200을 돌려줄 때까지 3초 간격으로 최대 60번(180초) 확인.
    print("\n=== 서비스 준비 폴링 (최대 180초) ===")
    for i in range(60):
        time.sleep(3)
        try:
            # curl로 상태코드만 뽑는다(-o /dev/null로 본문 버리고 -w로 코드만).
            # 개별 요청은 3초 안에 끝나야 하며, run 자체 타임아웃은 8초로 여유.
            code = run(
                ssh,
                "curl -s -o /dev/null -w '%{http_code}' "
                "http://localhost:8001/v1/models --max-time 3",
                timeout=8,
            ).strip()
        except Exception as e:
            # 아직 서버가 안 떠서 연결이 안 되는 경우도 정상 흐름이므로
            # 예외를 문자열로 바꿔 계속 폴링한다(스크립트를 죽이지 않음).
            code = f"(err: {e})"
        print(f"  [{i+1}/60] /v1/models → {code}")
        # 200이 오면 서버 준비 완료 — 폴링을 즉시 중단한다.
        if code == "200":
            break

    # [5단계-a] 모델 목록을 조회해 LoRA 어댑터들이 실제로 노출되는지 확인한다.
    # json.tool로 예쁘게 정렬하고 앞 40줄만 보여 준다.
    print("\n=== /v1/models 응답 (LoRA 노출 확인) ===")
    print(run(ssh, "curl -s http://localhost:8001/v1/models | python3 -m json.tool | head -40"))

    # [5단계-b] vLLM 로그 마지막 25줄을 출력. 기동 실패 시 원인(에러 스택 등)이
    # 대개 여기에 남으므로 문제 진단의 1차 단서가 된다.
    print("\n=== vLLM 로그 (마지막 25줄) ===")
    print(run(ssh, f"tail -25 {LOG_PATH}"))

    # 모든 확인이 끝나면 SSH 연결을 닫는다(원격 vLLM 서버는 계속 살아 있음).
    ssh.close()


if __name__ == "__main__":
    # 스크립트를 직접 실행했을 때만 재시작 절차를 수행한다(import 시엔 실행 안 함).
    main()
