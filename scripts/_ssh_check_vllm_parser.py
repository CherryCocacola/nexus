"""GPU 서버(Machine B)의 vLLM tool_call 파서 상태를 SSH로 원격 점검하는 진단 스크립트.

이 파일이 하는 일(개요):
    - Machine A(오케스트레이터)에서 실행하는 일회성 진단 도구다.
    - paramiko로 GPU 서버(192.168.21.112)에 SSH 접속한 뒤, 아래 4가지를 순서대로 확인한다.
        1) vLLM 로그에서 tool_call / qwen3 / hermes / parser 관련 최근 기록
        2) vLLM 로그에서 파싱 실패(ParseError, unable to parse, unknown tool) 흔적
        3) 현재 vLLM 빌드가 지원하는 tool-call-parser 목록(설치본에 직접 물어봄)
        4) Phase 2 체크포인트의 chat_template.jinja가 tool_call을 어떤 포맷으로 쓰는지

왜 필요한가:
    - 모델이 도구 호출(tool_calls)을 뱉었는데 vLLM이 제대로 파싱하지 못하면,
      오케스트레이터 쪽에서 tool_calls가 비어 보이거나 이상하게 들어온다.
    - 그 원인이 "파서 미지원"인지 "chat_template 포맷 불일치"인지 로그로 빠르게 좁히려는 목적.

주요 함수:
    - run(ssh, cmd, timeout): 원격 명령 1개를 실행하고 표준출력+표준에러를 합쳐 문자열로 반환.
    - main(): SSH 접속 → 위 4개 진단 명령 실행 → 결과 출력 → 접속 종료.

의존/주의:
    - 외부 라이브러리 paramiko(SSH) 필요. LAN 내부(192.168.21.112)로만 접속(에어갭 준수).
    - 접속 계정/비밀번호가 코드에 그대로 박혀 있는 임시 진단용 스크립트다(운영 코드 아님).

작성자: 이현수 / 작성일: 2026-07-05
"""
from __future__ import annotations
import sys
import paramiko

# 원격 명령 결과에 한글/유니코드가 섞여 나올 수 있으므로 표준출력 인코딩을 UTF-8로 강제.
# (Windows 콘솔 기본 코드페이지에서 깨지는 것을 막는다.)
sys.stdout.reconfigure(encoding="utf-8")


def run(ssh, cmd, timeout=15):
    """열려 있는 SSH 세션에서 원격 셸 명령 하나를 실행하고 그 출력을 통째로 돌려준다.

    왜 stdout과 stderr를 합치나:
        - 진단 목적이라 정상 출력이든 오류 메시지든 전부 눈으로 보고 싶기 때문.
        - 특히 파이썬 호출(3번 진단)은 오류가 stderr로 나오므로 함께 받아야 원인이 보인다.

    매개변수:
        ssh     : 이미 connect()된 paramiko SSHClient 인스턴스.
        cmd     : 원격에서 실행할 셸 명령 문자열.
        timeout : 명령 실행 제한 시간(초). 기본 15초로 걸어 hang을 방지.

    반환:
        stdout + stderr를 순서대로 이어 붙인 문자열(디코딩 실패 바이트는 replace 처리).
    """
    # exec_command는 (stdin, stdout, stderr) 세 스트림을 돌려준다. 여기선 입력은 쓰지 않는다.
    stdin, stdout, stderr = ssh.exec_command(cmd, timeout=timeout)
    # 두 스트림을 각각 읽어 디코딩한 뒤 이어 붙인다. errors="replace"로 깨진 바이트도 죽지 않게.
    return stdout.read().decode(errors="replace") + stderr.read().decode(errors="replace")


def main() -> None:
    """GPU 서버에 SSH로 붙어 vLLM 파서 관련 진단 4종을 차례로 실행하고 결과를 출력한다.

    흐름:
        1) SSHClient 생성 및 접속(미등록 호스트 키는 자동 수락).
        2) run()으로 진단 명령을 하나씩 실행하며 결과를 print.
        3) 마지막에 SSH 세션을 닫는다.
    반환값 없음(진단 결과는 표준출력으로만 보여 준다).
    """
    # SSH 클라이언트 준비. AutoAddPolicy는 처음 보는 서버 호스트 키를 묻지 않고 자동 등록한다
    # (내부망 일회성 진단이라 편의상 사용 — 운영 환경이라면 지양해야 할 정책).
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect("192.168.21.112", username="idino", password="dkdlelsh@12", timeout=10)

    # [진단 1] vLLM 로그에서 tool_call·qwen3·hermes·parser 키워드가 든 최근 30줄을 본다.
    # 파서가 실제로 어떤 이름으로 동작 중인지, 관련 경고가 있는지 훑기 위함.
    print("=== vLLM 로그에서 tool_call 관련 ===")
    print(run(ssh, "grep -Ei 'tool.call|qwen3|hermes|parser' /opt/nexus-gpu/vllm.log | tail -30"))

    # [진단 2] 파싱 실패 흔적(ParseError / unable to parse / unknown tool)만 골라 최근 20줄.
    # tool_calls가 깨져 들어올 때 vLLM이 남기는 오류를 직접 확인하는 단계.
    print("\n=== 최근 /v1/chat 요청들의 본문/tool_calls 응답 샘플 ===")
    print(run(ssh, "grep -Ei 'ParseError|unable to parse|tool.*unknown' /opt/nexus-gpu/vllm.log | tail -20"))

    # [진단 3] 설치된 vLLM에게 "네가 지원하는 tool-call-parser가 뭐냐"를 직접 물어본다.
    # ToolParserManager의 등록 키 목록을 정렬해 출력 → 우리가 쓰려는 파서명이 실제 있는지 대조용.
    # (원격 파이썬 3.12 인터프리터로 한 줄짜리 스크립트를 실행하고, 오류도 함께 보려 2>&1로 합침.)
    print("\n=== vLLM이 지원하는 tool-call-parser 목록 ===")
    print(run(ssh, "/opt/nexus-gpu/.venv/bin/python3.12 -c "
                  "\"from vllm.entrypoints.openai.tool_parsers import ToolParserManager; "
                  "print(sorted(ToolParserManager.tool_parsers.keys()))\" 2>&1"))

    # [진단 4] Phase 2 체크포인트의 chat_template.jinja에서 tool_call / <tool 등장 위치를 찾는다.
    # 모델이 학습된 도구 호출 표기 포맷과 vLLM 파서가 기대하는 포맷이 맞는지 눈으로 비교하기 위함.
    print("\n=== Phase 2 체크포인트의 chat_template에 tool_call 포맷 ===")
    print(run(ssh, "grep -n 'tool_call\\|<tool' "
                  "/opt/nexus-gpu/checkpoints/qwen35-phase2/chat_template.jinja | head -15"))

    # 진단이 끝났으니 SSH 세션을 정리한다(소켓/리소스 반환).
    ssh.close()


if __name__ == "__main__":
    # 스크립트를 직접 실행했을 때만 진단을 수행. import될 때는 실행되지 않는다.
    main()
