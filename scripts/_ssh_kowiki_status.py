"""GPU 서버의 kowiki 전체 적재 진행 상태를 한 번(one-shot) 조회하는 임시 진단 스크립트.

이 파일은 무엇을 하나?
    - GPU 서버(192.168.21.112)에 SSH로 접속해서, 한국어 위키(kowiki) 덤프를
      tb_knowledge 테이블에 적재(ingest)하는 백그라운드 작업이 지금 어디까지
      진행됐는지를 한눈에 확인한다.
    - 적재 작업은 GPU 서버의 tmux 세션(`kowiki_ingest`) 안에서 장시간 돌기
      때문에, 매번 사람이 SSH로 붙어 tmux/ps/DB를 뒤지는 대신 이 스크립트가
      필요한 진단 정보를 한꺼번에 뽑아 출력한다.
    - 순수 조회 전용이다. 서버 상태를 바꾸는 명령(적재 시작/중단 등)은 없고,
      tmux 버퍼 캡처·프로세스 조회·row count 집계 같은 읽기 작업만 수행한다.

무엇을 출력하나?
    1) 현재 살아있는 tmux 세션 목록
    2) 적재 pane의 최근 로그(현재 상태) 및 전체 버퍼 앞/끝 발췌
    3) 완료/에러 키워드 grep 결과
    4) 관련 python 프로세스 존재 여부
    5) tb_knowledge 테이블의 source별 / 전체 row count
    6) 적재 프로세스의 실행시간·CPU·메모리, 원본 덤프 파일 크기

주요 함수:
    - run(ssh, cmd, timeout): 원격 명령 한 개를 실행하고 stdout+stderr를 합쳐 반환
    - main(): SSH 접속 → 위 진단 항목들을 순서대로 실행/출력 → 접속 종료

의존성:
    - paramiko (순수 파이썬 SSH 클라이언트). 에어갭 LAN 내부 주소로만 접속한다.

주의: 이건 개발/운영 편의를 위한 일회성 진단 스크립트라서 접속 정보가
    소스에 하드코딩돼 있다. 정식 모듈이 아니므로 파일명이 `_`로 시작한다.

작성자: 이현수 / 작성일: 2026-07-05
"""

from __future__ import annotations

import sys

import paramiko

# 원격 로그에 한글(예: "적재 완료")이 섞여 나오므로, 콘솔 출력 인코딩을
# UTF-8로 강제한다. Windows 기본 콘솔 코드페이지에서 UnicodeEncodeError가
# 나는 것을 막기 위한 조치.
sys.stdout.reconfigure(encoding="utf-8")

# --- 접속 대상 GPU 서버 정보 (에어갭 LAN 내부 주소) ---
# 임시 진단 스크립트라 자격증명을 상수로 박아둔다. 운영 코드에서는 절대 금지.
HOST = "192.168.21.112"
USER = "idino"
PASSWORD = "dkdlelsh@12"


def run(ssh, cmd, timeout=20):
    """원격 서버에서 명령 하나를 실행하고 표준출력+표준에러를 합쳐서 문자열로 돌려준다.

    진단용이라 stdout과 stderr를 굳이 구분하지 않고 이어 붙인다. 이렇게 하면
    에러 메시지도 결과와 함께 그대로 눈에 보여서 상태 파악이 편하다.

    매개변수:
        ssh: 이미 connect()가 끝난 paramiko SSHClient 객체
        cmd: 원격에서 실행할 셸 명령 문자열
        timeout: 명령 실행 제한 시간(초). 기본 20초.

    반환:
        stdout 디코딩 결과 + stderr 디코딩 결과를 이어 붙인 문자열.
        디코딩 실패 바이트는 errors="replace"로 대체 문자 처리한다.
    """
    _, stdout, stderr = ssh.exec_command(cmd, timeout=timeout)
    return stdout.read().decode(errors="replace") + stderr.read().decode(errors="replace")


def main() -> None:
    """SSH로 GPU 서버에 붙어 kowiki 적재 진단 항목들을 순서대로 실행/출력한다.

    흐름:
        1. SSHClient 생성 후 접속 (호스트 키는 자동 수락)
        2. tmux/로그/프로세스/DB row count 등 진단 명령을 run()으로 실행하며 출력
        3. 마지막에 SSH 연결을 닫는다

    반환값은 없다. 결과는 전부 print로 콘솔에 찍는다.
    """
    ssh = paramiko.SSHClient()
    # 폐쇄망 내부 고정 서버라 known_hosts에 없어도 자동으로 키를 받아들인다.
    # (에어갭 LAN 한정이므로 MITM 위험이 낮다는 판단.)
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(HOST, username=USER, password=PASSWORD, timeout=10)

    # [1] 지금 서버에 어떤 tmux 세션이 살아있는지 확인. 세션이 없으면 안내 문구 출력.
    print("=== tmux 세션 목록 ===")
    print(run(ssh, "tmux ls 2>&1 || echo '(tmux 세션 없음)'"))

    # [2] 적재 pane(kowiki_ingest)의 마지막 3줄만 캡처 → "지금 이 순간" 상태를 본다.
    print("\n=== kowiki_ingest pane 마지막 3줄 (현재 상태) ===")
    print(run(ssh, "tmux capture-pane -p -t kowiki_ingest -S -3 2>&1"))

    # [3] pane 스크롤백 버퍼 전체를 파일로 떠서, 시작 로그(앞 5줄)와
    #     최신 로그(끝 80줄)를 함께 본다. 진행 시작 시점과 최근 진척을 대조하기 위함.
    print("\n=== kowiki_ingest pane 전체 버퍼의 처음 5줄 + 끝 80줄 ===")
    # pane 버퍼 전체를 큰 음수로 잡아 파일로 저장한 뒤 앞/끝만 추려 본다.
    # -S -100000: 스크롤백을 최대 10만 줄까지 거슬러 올라가 통째로 캡처한다.
    dump_cmd = (
        "tmux capture-pane -p -t kowiki_ingest -S -100000 "
        "> /tmp/_kowiki_pane.log 2>&1; "
        "echo '-- HEAD --'; head -5 /tmp/_kowiki_pane.log; "
        "echo '-- TAIL --'; tail -80 /tmp/_kowiki_pane.log; "
        "echo '-- LINECOUNT --'; wc -l /tmp/_kowiki_pane.log"
    )
    print(run(ssh, dump_cmd, timeout=30))

    # [4] 위에서 저장한 로그 파일에서 완료/에러/진척 관련 키워드만 골라 최근 40건 확인.
    #     대량 로그를 다 읽지 않고도 "끝났는지 / 터졌는지"를 빠르게 판정하려는 목적.
    print("\n=== 완료/요약 메시지 grep ===")
    grep_cmd = (
        "grep -Ei '(complete|finished|done|error|traceback|총|적재 완료|rows|insert)' "
        "/tmp/_kowiki_pane.log | tail -40 || true"
    )
    print(run(ssh, grep_cmd))

    # [5] 적재 관련 python 프로세스가 실제로 돌고 있는지 확인. 없으면 안내 문구.
    #     tmux 로그가 멈춰 보일 때 "죽은 건지 / 그냥 조용한 건지" 구분하는 근거.
    print("\n=== 관련 프로세스 (python 포함) ===")
    print(run(ssh, "pgrep -af python | grep -Ei 'kowiki|ingest|prepare' || echo '(관련 python 없음)'"))

    # [6] 실제 DB에 몇 행이 들어갔는지를 GPU 서버의 python으로 직접 집계한다.
    #     로그가 아니라 tb_knowledge를 직접 세므로 가장 신뢰할 수 있는 진척 지표.
    print("\n=== tb_knowledge row count (asyncpg via python) ===")
    # GPU 서버의 실제 venv 경로는 `/opt/nexus-gpu/.venv` (dot venv).
    # 접속 정보는 실제 돌고 있는 prepare_kowiki 프로세스 커맨드라인과 동일하게 맞춤:
    #   postgresql://nexus:idino%4012@192.168.10.39:5440/nexus
    # 아래 py_cmd는 원격에서 실행될 파이썬 스니펫 문자열이다. source별 count와
    # 전체 TOTAL을 뽑아 출력한다. (문자열 안의 코드는 그대로 원격에서 실행됨)
    py_cmd = (
        "/opt/nexus-gpu/.venv/bin/python -c \"\n"
        "import asyncio, asyncpg\n"
        "async def main():\n"
        "    conn = await asyncpg.connect("
        "host='192.168.10.39', port=5440, "
        "user='nexus', password='idino@12', database='nexus')\n"
        "    rows = await conn.fetch('SELECT source, COUNT(*) AS c "
        "FROM tb_knowledge GROUP BY source ORDER BY source')\n"
        "    for r in rows: print(r['source'], r['c'])\n"
        "    total = await conn.fetchval('SELECT COUNT(*) FROM tb_knowledge')\n"
        "    print('TOTAL', total)\n"
        "    await conn.close()\n"
        "asyncio.run(main())\""
    )
    print(run(ssh, py_cmd, timeout=60))

    # [7] 적재 프로세스(PID 206672)의 실행 경과시간/CPU/메모리를 본다.
    #     해당 PID가 이미 없으면 pgrep으로 현재 prepare_kowiki 프로세스를 다시 찾는다.
    #     (PID는 특정 실행 회차의 값이라, 재시작되면 달라질 수 있음에 유의.)
    print("\n=== prepare_kowiki 프로세스의 start time / CPU / mem ===")
    print(run(ssh,
        "ps -o pid,etime,pcpu,pmem,rss,cmd -p 206672 2>&1 || "
        "pgrep -af prepare_kowiki"))

    # [8] 원본 kowiki 덤프/중간 파일들의 크기를 확인. 입력 데이터가 제자리에
    #     있는지, 얼마나 큰지 가늠하기 위함. 파일이 없으면 안내 문구.
    print("\n=== 원본 덤프 / 중간 파일 크기 ===")
    print(run(ssh, "ls -lah /opt/nexus-gpu/rag/kowiki* 2>/dev/null | head -20 || echo '(kowiki 파일 없음)'"))

    # 진단이 끝나면 SSH 연결을 정리한다.
    ssh.close()


if __name__ == "__main__":
    # 스크립트로 직접 실행할 때만 진단을 수행한다 (import 시에는 실행 안 함).
    main()
