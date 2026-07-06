"""GPU 서버(Machine B)의 vLLM에 LoRA 어댑터가 로드됐는지 원격으로 확인하는 진단 스크립트.

이 스크립트는 개발자가 로컬 PC(Machine A)에서 실행하면, SSH로 GPU 서버에 접속한 뒤
그 안에서 vLLM의 OpenAI 호환 엔드포인트(`/v1/models`)를 curl로 조회한다.
응답에 들어 있는 모델 id 목록을 출력해서, 기대하는 LoRA 어댑터
(예: `nexus-phase2`)가 실제로 서빙 중인지 눈으로 확인하는 용도다.

주요 구성:
- main(): SSH 접속 → 원격에서 curl+python3로 모델 id 목록 추출 → 출력.

의존:
- paramiko (SSH 클라이언트). GPU 서버는 192.168.21.112, vLLM은 그 안에서 localhost:8001로 뜬다.
- 에어갭 환경이므로 접속 주소는 모두 사내 LAN 대역만 사용한다.

주의: 이 파일은 사람이 손으로 돌려보는 일회성 점검 도구이며, 4-Tier 체인과는 무관하다.

작성자: 이현수 / 작성일: 2026-07-05
"""
from __future__ import annotations
import sys
import paramiko

# 원격 명령의 결과에 한글/유니코드가 섞여도 깨지지 않도록 표준출력을 UTF-8로 강제한다.
# (Windows 콘솔의 기본 코드페이지가 cp949일 수 있어서 명시적으로 재설정한다.)
sys.stdout.reconfigure(encoding="utf-8")


def main() -> None:
    """GPU 서버에 SSH로 붙어 vLLM이 서빙 중인 모델 id 목록을 출력한다.

    흐름:
    1) paramiko SSHClient 로 GPU 서버(192.168.21.112)에 접속한다.
    2) 서버 내부에서 `curl`로 vLLM의 `/v1/models`(localhost:8001)를 조회한다.
    3) 그 JSON을 그 자리에서 python3 한 줄 스크립트로 파싱해 각 모델의 id만 추린다.
    4) 추려진 id 리스트 문자열을 로컬로 받아 그대로 화면에 찍는다.

    매개변수/반환값은 없다(부수효과로 표준출력에 결과를 인쇄).
    """
    # SSH 클라이언트 객체 생성.
    ssh = paramiko.SSHClient()
    # 서버 호스트키가 로컬 known_hosts에 없어도 자동 등록하고 접속을 진행한다.
    # (사내 폐쇄망 진단용이라 편의상 AutoAdd를 쓴다. 운영 코드였다면 더 엄격히 다뤄야 함.)
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    # GPU 서버에 아이디/비밀번호로 접속. timeout=10초 안에 못 붙으면 예외를 낸다.
    ssh.connect("192.168.21.112", username="idino", password="dkdlelsh@12", timeout=10)

    # 원격 셸에서 실행할 명령:
    #   curl 로 vLLM 모델 목록 JSON을 받고, 파이프로 python3에 넘겨서
    #   data 배열의 각 항목에서 'id'만 뽑아 리스트로 출력한다.
    # exec_command 는 (stdin, stdout, stderr) 세 개의 파일류 객체를 돌려준다.
    stdin, stdout, stderr = ssh.exec_command(
        "curl -s http://localhost:8001/v1/models | "
        "python3 -c \"import json,sys; d=json.load(sys.stdin); "
        "print([m['id'] for m in d['data']])\"",
        timeout=15,  # 원격 명령 자체가 15초를 넘기면 중단한다.
    )
    # 원격 명령의 표준출력(모델 id 리스트 문자열)을 읽어 디코딩 후 그대로 출력한다.
    print(stdout.read().decode())
    # SSH 세션을 닫아 자원을 정리한다.
    ssh.close()


if __name__ == "__main__":
    # 스크립트로 직접 실행했을 때만 main()을 호출한다(모듈 import 시엔 실행되지 않음).
    main()
