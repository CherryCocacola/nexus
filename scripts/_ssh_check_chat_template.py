"""GPU 서버의 Qwen3.5 chat template 설정을 SSH로 원격 점검하는 일회성 진단 스크립트.

이 스크립트가 하는 일:
- Machine B(GPU 서버, 192.168.21.112)에 SSH로 접속한다.
- Qwen3.5 계열 모델/체크포인트의 chat_template(Jinja) 안에
  `enable_thinking` 같은 "사고 과정(thinking) 제어 파라미터"가 들어있는지 확인한다.
- 원본 Worker 모델, Phase 2 LoRA 체크포인트, tokenizer_config.json,
  그리고 vLLM 실행 로그까지 여러 위치를 훑어 chat template 인식 상태를 종합 점검한다.

왜 필요한가:
- Qwen3.5는 chat template에 thinking 토큰/스위치가 있어야 추론 모드가 켜진다.
- 배포된 모델·LoRA가 실제로 그 설정을 갖고 있는지 원격에서 눈으로 확인하기 위한 도구.

주요 함수:
- run(): 원격 명령 1개를 실행하고 stdout+stderr를 합쳐 문자열로 돌려주는 헬퍼.
- main(): SSH 세션을 열고 점검용 명령들을 순서대로 실행해 결과를 출력.

의존:
- paramiko(SSH 클라이언트). 에어갭 규칙상 LAN 주소(192.168.x.x)로만 접속한다.
- 이 파일은 운영 로직이 아닌 손으로 돌리는 진단용 스크립트다(코어 모듈에서 import하지 않음).

작성자: 이현수 / 작성일: 2026-07-05
"""
from __future__ import annotations
import sys
import paramiko

# 한글/특수문자가 섞인 원격 출력을 콘솔에 안전하게 찍기 위해 표준출력을 UTF-8로 재설정.
sys.stdout.reconfigure(encoding="utf-8")


def run(ssh, cmd, timeout=15):
    """열려 있는 SSH 세션에서 명령 하나를 실행하고 그 출력을 문자열로 반환한다.

    매개변수:
        ssh: 이미 connect()된 paramiko SSHClient 객체.
        cmd: 원격 셸에서 실행할 명령 문자열.
        timeout: 명령 실행 제한 시간(초). 기본 15초.

    반환:
        stdout와 stderr를 이어 붙인 문자열. 진단용이라 오류 출력도 함께 보여준다.
        디코딩 실패 문자는 errors="replace"로 대체해 예외 없이 넘어간다.
    """
    # exec_command는 (stdin, stdout, stderr) 세 스트림을 돌려준다. stdin은 안 쓴다.
    stdin, stdout, stderr = ssh.exec_command(cmd, timeout=timeout)
    # 정상 출력과 오류 출력을 모두 읽어 하나로 합쳐 반환(어느 쪽에 결과가 있든 놓치지 않게).
    return stdout.read().decode(errors="replace") + stderr.read().decode(errors="replace")


def main() -> None:
    """GPU 서버에 접속해 chat template 관련 항목들을 순서대로 점검하고 출력한다.

    흐름:
        1) SSH 접속(호스트 키는 자동 수락 — 폐쇄망 내부 서버 대상).
        2) Phase 2 LoRA 체크포인트의 chat_template.jinja에서 thinking 토큰 검색.
        3) 원본 Worker 모델 디렉토리에 chat/template 파일이 있는지 확인.
        4) tokenizer_config.json 안에 chat_template이 들어있는지 확인.
        5) 체크포인트 파일들 중 enable_thinking을 담은 파일이 있는지 검색.
        6) vLLM 로그에서 chat template을 어떻게 인식했는지 확인.
        7) 세션 종료.

    반환값은 없다(결과는 모두 표준출력으로 print).
    """
    # SSH 클라이언트 생성 후 접속. 내부망 전용 서버라 host key를 자동 추가하도록 설정.
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect("192.168.21.112", username="idino", password="dkdlelsh@12", timeout=10)

    # Phase 2 체크포인트의 chat_template.jinja 검사
    # enable_thinking 또는 think 문자열이 들어간 줄을 번호와 함께 최대 20줄까지 뽑는다.
    print("=== Phase 2 LoRA의 chat_template.jinja에서 thinking 관련 토큰 ===")
    print(run(ssh, "grep -n 'enable_thinking\\|think' "
                  "/opt/nexus-gpu/checkpoints/qwen35-phase2/chat_template.jinja | head -20"))

    # 원본 Worker 모델 디렉토리에서 이름에 chat/template이 들어간 파일이 있는지 나열.
    print("\n=== 원본 Worker 모델(qwen3.5-27b)의 chat template ===")
    print(run(ssh, "ls /opt/nexus-gpu/models/qwen3.5-27b/ | grep -i 'chat\\|template'"))

    # tokenizer_config.json 안에 chat_template 키가 인라인으로 들어있는지 확인.
    # 'chat_template' 뒤 100자만 잘라 앞부분 500바이트까지 미리보기(없으면 '(없음)').
    print("\n=== tokenizer_config.json에 chat template 존재? ===")
    print(run(ssh, "grep -o 'chat_template.\\{100\\}' "
                  "/opt/nexus-gpu/models/qwen3.5-27b/tokenizer_config.json 2>/dev/null "
                  "| head -c 500 || echo '(없음)'"))

    # Phase 2 체크포인트 디렉토리의 모든 파일 중 enable_thinking을 포함한 파일명을 나열(-l).
    print("\n=== enable_thinking 관련 문구 전체 검색 ===")
    print(run(ssh, "grep -l 'enable_thinking' "
                  "/opt/nexus-gpu/checkpoints/qwen35-phase2/* 2>/dev/null"))

    # vLLM 로그에 chat template 어떻게 인식했는지
    # 대소문자 무시(-i)로 chat template/enable_thinking/think 관련 로그 마지막 10줄 확인.
    print("\n=== vLLM 로그에서 chat_template 관련 ===")
    print(run(ssh, "grep -Ei 'chat.template|enable_thinking|think' "
                  "/opt/nexus-gpu/vllm.log | tail -10 || echo '(없음)'"))

    # 점검이 끝났으면 SSH 세션을 닫아 원격 연결을 정리한다.
    ssh.close()


if __name__ == "__main__":
    # 스크립트로 직접 실행할 때만 점검 루틴을 돈다(import 시에는 실행되지 않음).
    main()
