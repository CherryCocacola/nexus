"""새 bootstrap 데이터를 GPU 서버의 학습 데이터 디렉토리로 전송하는 일회성 운영 스크립트.

이 스크립트는 로컬(Machine A)에서 생성한 bootstrap 학습 데이터(JSONL)를
GPU 서버(Machine B, 192.168.21.112)의 학습 데이터 디렉토리로 SSH/SFTP를 통해
업로드한다. 업로드 전에 GPU 서버에 이미 존재하던 파일을 `.phase1.bak` 이름으로
백업해 두어, 새 데이터가 잘못됐을 때 이전 데이터로 되돌릴 수 있게 한다.

전체 흐름은 단순한 4단계다.
  1) SSH 접속 (paramiko)
  2) 원격 기존 파일 백업 (cp + ls 로 확인)
  3) SFTP 로 로컬 파일을 원격 경로에 덮어쓰기 업로드
  4) 업로드 결과 검증 (파일 크기 / 줄 수 / 첫 줄 앞부분 출력)

주요 구성:
  - main(): 위 4단계를 순서대로 수행하는 유일한 진입 함수.
  - HOST/USER/PASSWORD: GPU 서버 SSH 접속 정보(상수).
  - LOCAL_PATH/REMOTE_PATH: 업로드 대상 파일의 로컬/원격 경로(상수).

의존성: paramiko(SSH/SFTP 클라이언트). 외부는 LAN 내부 GPU 서버(192.168.x.x)에만
접속하므로 에어갭 규칙에 부합한다. 별도 모듈을 import 하지 않는 독립 실행 스크립트다.

주의: 접속 비밀번호가 소스에 평문으로 들어 있으므로, 이 파일은 내부 운영용으로만
사용하고 외부에 공유하지 않는다. 반복 운영이 필요해지면 설정/시크릿 분리를 검토한다.

작성자: 이현수 / 작성일: 2026-07-05
"""

from __future__ import annotations

import sys

import paramiko

# 표준 출력을 UTF-8 로 재설정한다. 원격 명령 결과에 한글/특수문자가 섞여도
# 콘솔에서 깨지지 않게 하기 위함이다(특히 Windows 기본 인코딩 대비).
sys.stdout.reconfigure(encoding="utf-8")

# --- GPU 서버(Machine B) SSH 접속 정보 -------------------------------------
# HOST: LAN 내부 GPU 서버 주소. USER/PASSWORD: 해당 서버 계정 자격 증명.
HOST = "192.168.21.112"
USER = "idino"
PASSWORD = "dkdlelsh@12"

# --- 업로드 대상 파일 경로 --------------------------------------------------
# LOCAL_PATH: 이 스크립트를 실행하는 로컬 작업 디렉토리 기준의 원본 파일.
# REMOTE_PATH: GPU 서버에서 학습이 실제로 참조하는 목적지 경로(덮어쓰기 대상).
LOCAL_PATH = "data/bootstrap/bootstrap_data.jsonl"
REMOTE_PATH = "/opt/nexus-gpu/training/bootstrap_data.jsonl"


def main() -> None:
    """bootstrap 데이터를 GPU 서버로 업로드하는 전체 절차를 순서대로 수행한다.

    동작 순서:
      1) paramiko 로 GPU 서버에 SSH 접속한다.
      2) 원격에 이미 있던 파일을 `.phase1.bak` 으로 복사해 백업하고, 백업/원본
         파일 목록을 출력해 눈으로 확인한다.
      3) SFTP 세션을 열어 로컬 파일을 원격 경로로 업로드(덮어쓰기)한다.
      4) 업로드된 파일의 크기·총 줄 수·첫 줄 앞 200바이트를 출력해 정상 여부를
         검증한다.

    매개변수: 없음(접속 정보와 경로는 모듈 상단 상수를 사용).
    반환값: 없음(진행 상황과 검증 결과는 표준 출력으로 보여준다).
    호출 대상: paramiko.SSHClient / SFTPClient. 별도 반환 없이 부수효과(원격
    파일 변경 + 콘솔 출력)로 동작한다.
    """
    # SSH 클라이언트를 생성한다.
    ssh = paramiko.SSHClient()
    # 원격 호스트 키가 로컬 known_hosts 에 없어도 자동으로 추가해 접속을 진행한다.
    # 내부 신뢰 네트워크의 고정 서버를 대상으로 하는 운영 스크립트라 이렇게 둔다.
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    # 실제 접속. 10초 안에 연결되지 않으면 타임아웃으로 실패시킨다.
    ssh.connect(HOST, username=USER, password=PASSWORD, timeout=10)

    # 기존 파일 백업
    # cp 로 기존 REMOTE_PATH 를 `.phase1.bak` 으로 복사한다. 파일이 없어 cp 가
    # 실패해도(첫 업로드 등) `2>/dev/null` 로 에러를 삼켜 스크립트를 계속 진행한다.
    # 이어서 ls -lh 로 원본/백업 파일 목록을 조회해 백업이 됐는지 확인한다.
    stdin, stdout, stderr = ssh.exec_command(
        f"cp {REMOTE_PATH} {REMOTE_PATH}.phase1.bak 2>/dev/null; "
        f"ls -lh {REMOTE_PATH}* 2>/dev/null",
        timeout=10,
    )
    print("=== 기존 파일 백업 ===")
    # 원격 명령의 표준 출력을 읽어 그대로 보여준다. 디코드 불가 바이트는
    # replace 로 대체해 예외 없이 출력한다.
    print(stdout.read().decode(errors="replace"))

    # SFTP 업로드
    # SSH 위에서 SFTP 서브시스템을 열어 파일 전송 채널을 만든다.
    print(f"=== 업로드: {LOCAL_PATH} → {REMOTE_PATH} ===")
    sftp = ssh.open_sftp()
    # 로컬 파일을 원격 경로에 그대로 올린다. 원격에 같은 이름이 있으면 덮어쓴다
    # (그래서 위 단계에서 미리 백업을 떠 둔 것이다).
    sftp.put(LOCAL_PATH, REMOTE_PATH)
    # SFTP 세션을 닫아 자원을 정리한다.
    sftp.close()

    # 크기 + 첫/마지막 줄 확인
    # 업로드 결과를 3가지로 검증한다.
    #   - ls -lh : 파일 크기(정상적으로 올라갔는지)
    #   - wc -l  : 전체 줄 수(JSONL 이므로 한 줄=한 샘플, 개수 확인)
    #   - head   : 첫 줄 앞 200바이트만 잘라 내용이 깨지지 않았는지 눈으로 확인
    # echo '---' 로 각 결과를 구분선으로 나눠 읽기 쉽게 한다.
    stdin, stdout, stderr = ssh.exec_command(
        f"ls -lh {REMOTE_PATH} && echo '---' && "
        f"wc -l {REMOTE_PATH} && echo '---' && "
        f"head -1 {REMOTE_PATH} | head -c 200 && echo",
        timeout=10,
    )
    print(stdout.read().decode(errors="replace"))

    # SSH 연결을 종료해 세션을 정리한다.
    ssh.close()


if __name__ == "__main__":
    # 스크립트를 직접 실행할 때만 업로드 절차를 시작한다(모듈 import 시엔 실행 안 함).
    main()
