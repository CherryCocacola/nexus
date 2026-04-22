"""kowiki 전체 적재 진행 상태 한 번 조회 (임시 진단 스크립트)."""

from __future__ import annotations

import sys

import paramiko

sys.stdout.reconfigure(encoding="utf-8")

HOST = "192.168.22.28"
USER = "idino"
PASSWORD = "dkdlelsh@12"


def run(ssh, cmd, timeout=20):
    _, stdout, stderr = ssh.exec_command(cmd, timeout=timeout)
    return stdout.read().decode(errors="replace") + stderr.read().decode(errors="replace")


def main() -> None:
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(HOST, username=USER, password=PASSWORD, timeout=10)

    print("=== tmux 세션 목록 ===")
    print(run(ssh, "tmux ls 2>&1 || echo '(tmux 세션 없음)'"))

    print("\n=== kowiki_ingest pane 마지막 3줄 (현재 상태) ===")
    print(run(ssh, "tmux capture-pane -p -t kowiki_ingest -S -3 2>&1"))

    print("\n=== kowiki_ingest pane 전체 버퍼의 처음 5줄 + 끝 80줄 ===")
    # pane 버퍼 전체를 큰 음수로 잡아 파일로 저장한 뒤 앞/끝만 추려 본다.
    dump_cmd = (
        "tmux capture-pane -p -t kowiki_ingest -S -100000 "
        "> /tmp/_kowiki_pane.log 2>&1; "
        "echo '-- HEAD --'; head -5 /tmp/_kowiki_pane.log; "
        "echo '-- TAIL --'; tail -80 /tmp/_kowiki_pane.log; "
        "echo '-- LINECOUNT --'; wc -l /tmp/_kowiki_pane.log"
    )
    print(run(ssh, dump_cmd, timeout=30))

    print("\n=== 완료/요약 메시지 grep ===")
    grep_cmd = (
        "grep -Ei '(complete|finished|done|error|traceback|총|적재 완료|rows|insert)' "
        "/tmp/_kowiki_pane.log | tail -40 || true"
    )
    print(run(ssh, grep_cmd))

    print("\n=== 관련 프로세스 (python 포함) ===")
    print(run(ssh, "pgrep -af python | grep -Ei 'kowiki|ingest|prepare' || echo '(관련 python 없음)'"))

    print("\n=== tb_knowledge row count (asyncpg via python) ===")
    # GPU 서버의 실제 venv 경로는 `/opt/nexus-gpu/.venv` (dot venv).
    # 접속 정보는 실제 돌고 있는 prepare_kowiki 프로세스 커맨드라인과 동일하게 맞춤:
    #   postgresql://nexus:idino%4012@192.168.10.39:5440/nexus
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

    print("\n=== prepare_kowiki 프로세스의 start time / CPU / mem ===")
    print(run(ssh,
        "ps -o pid,etime,pcpu,pmem,rss,cmd -p 206672 2>&1 || "
        "pgrep -af prepare_kowiki"))

    print("\n=== 원본 덤프 / 중간 파일 크기 ===")
    print(run(ssh, "ls -lah /opt/nexus-gpu/rag/kowiki* 2>/dev/null | head -20 || echo '(kowiki 파일 없음)'"))

    ssh.close()


if __name__ == "__main__":
    main()
