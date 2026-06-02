"""
MCP 서버를 DB 서버(192.168.10.39)에 배포하는 스크립트.

목적:
  Nexus 의 LAN MCP 서버(db / diag / kowiki / docingest)를 DB 서버(.39)에 올려
  기동하기 위한 paramiko 기반 배포 도구다. 기존 scripts/_ssh_*.py 패턴
  (paramiko SSH + SFTP 업로드 + 원격 명령 + nohup 기동 + 헬스체크)을 따른다.

매우 중요 — 안전:
  이 스크립트는 **운영자가 검토 후 직접 실행**하는 것을 전제로 한다. 기본 모드는
  --dry-run(원격 명령을 실행하지 않고 출력만)이다. 실제 배포는 --no-dry-run 을
  명시해야만 수행된다(fail-closed). 자동화/CI 에서 무인 실행하지 말 것.

배포 대상(사용자 확정):
  mcp_servers 를 DB 서버 192.168.10.39 에 배포한다.
    · db        — PostgreSQL 을 localhost(컨테이너 포트)로 접속(.39 로컬)
    · kowiki    — 임베딩 서버(192.168.22.28:8002, LAN)로 접속
    · docingest — 임베딩 서버(192.168.22.28:8002, LAN)로 접속 + 로컬 파서
    · diag      — 인프라 도달성/지연 점검

코드 배포 방식(가이드 MCP_DEPLOYMENT.md 와 동일 전제):
  .39 에는 DocUtil 등 다른 서비스가 도는 일반 Ubuntu 호스트이며, Nexus 레포가
  사전에 git clone 되어 있다는 보장이 없다. 따라서 기본 방식은 **SFTP 업로드**다
  (rsync 없이 paramiko 만으로 동작). 운영자가 .39 에 레포를 두고 git pull 로
  운영하려면 --code-mode pull 을 선택할 수 있다(전제: 원격에 git + 레포 존재).

에어갭:
  외부 네트워크 호출 없음. 모든 접속은 LAN 주소(192.168.x)다. 원격 의존성
  (fastapi/uvicorn/asyncpg/python-pptx/pdfplumber 등)은 오프라인 wheel 로
  사전 설치되어 있어야 한다(이 스크립트는 설치하지 않음 — anti-pattern #10).

anti-pattern #8(bare except 금지):
  원격 명령/SFTP 실패는 구체 예외로 포착해 사유와 함께 보고한다. 한 단계 실패가
  전체 배포 흐름을 모호하게 삼키지 않도록 한다.
"""

from __future__ import annotations

import argparse
import posixpath
import sys
from collections.abc import Iterable

import paramiko

# Windows 콘솔에서 한글 출력이 깨지지 않도록 stdout 을 UTF-8 로 재설정.
sys.stdout.reconfigure(encoding="utf-8")

# ─────────────────────────────────────────────
# SSH 자격 / 배포 상수 (LAN 전용 — 기존 _ssh_*.py / diag_server.py 와 동일)
# ─────────────────────────────────────────────
# .39 자격은 메모리(reference_servers)와 diag_server.py 의 DB_HOST 와 일치한다.
# 운영자가 다른 호스트/계정으로 배포하려면 argparse(--host/--user/--password)로 덮어쓴다.
DEFAULT_HOST = "192.168.10.39"
DEFAULT_USER = "idino"
DEFAULT_PASSWORD = "dkdlelsh@12"  # noqa: S105 — LAN 내부 배포용 고정 자격(에어갭). 운영자가 --password 로 덮어쓸 수 있음.

# 원격에 코드를 둘 기본 경로(업로드 모드). 운영자 홈 하위에 둔다.
DEFAULT_REMOTE_DIR = "/home/idino/nexus"

# 원격 파이썬 인터프리터. 에어갭 호스트에 사전 구성된 venv 가 있다면 그 경로로 덮어쓴다.
DEFAULT_REMOTE_PYTHON = "python3"

# MCP 서버 이름 → 기본 포트 (config/nexus_config.yaml mcp 섹션 / run.py 와 일치).
SERVER_PORTS: dict[str, int] = {
    "db": 8810,
    "diag": 8811,
    "kowiki": 8813,
    "docingest": 8814,
}

# 업로드 모드에서 .39 로 올릴 디렉토리(의존 core 코드 포함).
#   mcp_servers   — MCP 서버 본체
#   core          — db/kowiki/docingest 가 의존하는 config/rag/model/ingest 등
#   config        — nexus_config.yaml(접속 정보) 등 YAML 설정
# 주의: core 전체를 올리는 이유는 mcp_servers 가 core/config·core/rag·core/model·
#       core/ingest 를 import 하기 때문이다(개별 추림은 누락 위험이 커 디렉토리 단위로 올림).
UPLOAD_DIRS = ("mcp_servers", "core", "config")

# 업로드에서 제외할 디렉토리/파일 패턴(용량/불필요 파일 회피).
EXCLUDE_NAMES = {"__pycache__", ".pytest_cache", ".git", ".venv", "venv", ".mypy_cache"}
EXCLUDE_SUFFIXES = (".pyc", ".pyo", ".log")


# ─────────────────────────────────────────────
# 출력/명령 헬퍼
# ─────────────────────────────────────────────
def _print_header(title: str) -> None:
    """단계 구분용 헤더를 출력한다."""
    print(f"\n=== {title} ===")


def _should_skip(name: str) -> bool:
    """업로드 시 건너뛸 파일/디렉토리인지 판정한다(캐시/숨김/바이트코드 제외)."""
    if name in EXCLUDE_NAMES:
        return True
    return any(name.endswith(suffix) for suffix in EXCLUDE_SUFFIXES)


def run_remote(ssh: paramiko.SSHClient, cmd: str, *, dry_run: bool, timeout: int = 30) -> str:
    """
    원격에서 명령을 실행하거나(dry-run 이면) 출력만 한다.

    Args:
        ssh: 연결된 SSHClient.
        cmd: 실행할 셸 명령.
        dry_run: True 면 명령을 실행하지 않고 "[DRY-RUN] $ cmd" 만 출력.
        timeout: exec_command 타임아웃(초).

    Returns:
        실제 실행 시 stdout+stderr 텍스트. dry-run 이면 빈 문자열.

    예외 정책(anti-pattern #8):
        SSH 채널 오류는 구체 예외(SSHException/OSError)로 포착해 사유 문자열을
        반환한다. bare except 로 삼키지 않는다.
    """
    if dry_run:
        print(f"[DRY-RUN] $ {cmd}")
        return ""
    print(f"$ {cmd}")
    try:
        _stdin, stdout, stderr = ssh.exec_command(cmd, timeout=timeout)
        out = stdout.read().decode(errors="replace")
        err = stderr.read().decode(errors="replace")
        combined = out + err
        if combined.strip():
            print(combined.rstrip())
        return combined
    except (paramiko.SSHException, OSError) as e:
        msg = f"(원격 명령 실패: {type(e).__name__}: {e})"
        print(msg)
        return msg


# ─────────────────────────────────────────────
# 1) 코드 배포 — 업로드(SFTP) 또는 git pull
# ─────────────────────────────────────────────
def _iter_local_files(local_root: str) -> Iterable[tuple[str, str]]:
    """
    업로드 대상 (절대 로컬 경로, 루트 기준 상대 POSIX 경로) 쌍을 순회한다.

    캐시/바이트코드/숨김 디렉토리는 건너뛴다(_should_skip). 상대 경로는 원격에서
    POSIX 구분자(/)로 합쳐야 하므로 슬래시로 정규화한다.
    """
    import os

    for sub in UPLOAD_DIRS:
        base = os.path.join(local_root, sub)
        if not os.path.isdir(base):
            print(f"  (경고: 로컬 디렉토리 없음 — 건너뜀: {base})")
            continue
        for dirpath, dirnames, filenames in os.walk(base):
            # 제외 디렉토리는 walk 가 더 내려가지 않도록 in-place 로 잘라낸다.
            dirnames[:] = [d for d in dirnames if not _should_skip(d)]
            for fn in filenames:
                if _should_skip(fn):
                    continue
                abs_path = os.path.join(dirpath, fn)
                rel = os.path.relpath(abs_path, local_root).replace(os.sep, "/")
                yield abs_path, rel


def _sftp_mkdirs(sftp: paramiko.SFTPClient, remote_dir: str) -> None:
    """
    원격 디렉토리를 재귀적으로 생성한다(mkdir -p 상당).

    SFTP 에는 mkdir -p 가 없어 상위부터 하나씩 만든다. 이미 있으면 IOError 를
    흡수한다(존재 = 정상). 구체 예외(IOError/OSError)만 포착한다.
    """
    parts = remote_dir.strip("/").split("/")
    cur = ""
    for p in parts:
        cur = cur + "/" + p
        try:
            sftp.stat(cur)
        except FileNotFoundError:
            try:
                sftp.mkdir(cur)
            except (OSError, paramiko.SSHException) as e:
                print(f"  (mkdir 경고 {cur}: {type(e).__name__}: {e})")


def deploy_code_upload(
    ssh: paramiko.SSHClient, local_root: str, remote_dir: str, *, dry_run: bool
) -> None:
    """
    SFTP 로 mcp_servers/core/config 를 원격 remote_dir 에 업로드한다.

    dry-run 이면 어떤 파일이 어디로 갈지 요약만 출력하고 실제 전송은 하지 않는다.
    실제 모드에서는 원격 디렉토리를 만들고 파일을 put 한다.
    """
    _print_header(f"코드 배포(업로드) → {remote_dir}")
    files = list(_iter_local_files(local_root))
    print(f"  업로드 대상 파일: {len(files)}개 (디렉토리: {', '.join(UPLOAD_DIRS)})")

    if dry_run:
        # 처음 몇 개만 미리보기로 출력(전체 나열은 노이즈).
        for _abs, rel in files[:8]:
            print(f"[DRY-RUN] put → {posixpath.join(remote_dir, rel)}")
        if len(files) > 8:
            print(f"[DRY-RUN] ... 외 {len(files) - 8}개")
        return

    sftp = ssh.open_sftp()
    try:
        made_dirs: set[str] = set()
        for abs_path, rel in files:
            remote_path = posixpath.join(remote_dir, rel)
            remote_parent = posixpath.dirname(remote_path)
            if remote_parent not in made_dirs:
                _sftp_mkdirs(sftp, remote_parent)
                made_dirs.add(remote_parent)
            sftp.put(abs_path, remote_path)
        print(f"  업로드 완료: {len(files)}개 → {remote_dir}")
    finally:
        sftp.close()


def deploy_code_pull(ssh: paramiko.SSHClient, remote_dir: str, *, dry_run: bool) -> None:
    """
    원격 레포에서 git pull 로 최신 코드를 가져온다(전제: remote_dir 에 레포 존재).

    에어갭 주의: git pull 은 사내 git 원격(LAN)만 가능하다. 인터넷 원격이 설정돼
    있으면 실패할 수 있으며, 그 경우 업로드 모드(--code-mode upload)를 쓴다.
    """
    _print_header(f"코드 배포(git pull) → {remote_dir}")
    run_remote(
        ssh,
        f"cd {remote_dir} && git rev-parse --abbrev-ref HEAD && git pull --ff-only",
        dry_run=dry_run,
    )


# ─────────────────────────────────────────────
# 2) 원격 의존성 확인 (설치는 하지 않음 — 에어갭)
# ─────────────────────────────────────────────
def check_dependencies(
    ssh: paramiko.SSHClient, remote_dir: str, python_bin: str, *, dry_run: bool
) -> None:
    """
    원격에 필수 파이썬 패키지가 import 되는지 확인한다(설치는 하지 않음).

    에어갭: 의존성은 오프라인 wheel 로 사전 설치되어 있어야 한다. 이 함수는 누락
    여부를 조기에 드러내기 위한 점검일 뿐, pip install 코드를 절대 넣지 않는다
    (anti-pattern #10). 누락이 있으면 운영자가 wheel 로 사전 설치해야 한다.
    """
    _print_header("원격 의존성 확인(설치 안 함 — 에어갭)")
    # fastapi/uvicorn/asyncpg 는 db/kowiki/docingest 공통. python-pptx/pdfplumber 는
    # docingest 파서용(경량 CPU). 하나라도 없으면 import 단계에서 사유가 출력된다.
    mods = ["fastapi", "uvicorn", "asyncpg", "httpx", "pptx", "pdfplumber"]
    check = (
        f"cd {remote_dir} && {python_bin} - <<'PY'\n"
        "import importlib.util\n"
        f"mods = {mods!r}\n"
        "for m in mods:\n"
        "    ok = importlib.util.find_spec(m) is not None\n"
        "    print(f\"  {'OK ' if ok else 'MISSING'} {m}\")\n"
        "PY"
    )
    run_remote(ssh, check, dry_run=dry_run)


# ─────────────────────────────────────────────
# 3) MCP 서버 기동 (nohup) + 4) 헬스체크
# ─────────────────────────────────────────────
def start_server(
    ssh: paramiko.SSHClient,
    name: str,
    remote_dir: str,
    python_bin: str,
    api_key: str,
    *,
    dry_run: bool,
) -> None:
    """
    MCP 서버 1개를 nohup 으로 기동하고 기동 PID/로그 경로를 출력한다.

    기동 명령은 run.py 규약을 따른다:
        python -m mcp_servers.run <name> --host 0.0.0.0 --port <p> --api-key <key>

    왜 nohup 인가:
      가이드(MCP_DEPLOYMENT.md)는 운영 권장으로 systemd unit 을 제시한다. 이
      스크립트의 nohup 기동은 빠른 수동 기동/검증용이다. PYTHONPATH 를 remote_dir
      로 잡아 mcp_servers/core 패키지를 찾게 한다(venv 미사용 호스트 대비).

    docingest 의 GPU/파서:
      docingest 서버는 GPU 가 없으면 자동으로 경량 파서(pdfplumber/Tesseract)만
      사용한다(docingest_server.py 의 _gpu_available 폴백). .39 는 GPU 가 없으므로
      경량 파서 경로로 동작한다 — 별도 플래그 불필요.
    """
    port = SERVER_PORTS[name]
    log_path = posixpath.join(remote_dir, f"mcp_{name}.log")
    pid_path = posixpath.join(remote_dir, f"mcp_{name}.pid")
    _print_header(f"MCP 서버 기동: {name} (포트 {port})")

    # 기존 같은 서버가 떠 있으면 정리(중복 바인드 방지). pkill 패턴은 run.py 모듈명.
    run_remote(
        ssh,
        f"pkill -f 'mcp_servers.run {name}' 2>/dev/null; true",
        dry_run=dry_run,
    )

    # PYTHONPATH=remote_dir 로 두어 venv 없이도 패키지 해석. setsid+nohup 으로 SSH
    # 세션 종료 후에도 살아남게 한다(disown).
    cmd = (
        f"cd {remote_dir} && "
        f"PYTHONPATH={remote_dir} setsid nohup {python_bin} -m mcp_servers.run {name} "
        f"--host 0.0.0.0 --port {port} --api-key {api_key} "
        f"</dev/null >{log_path} 2>&1 & "
        f"echo $! > {pid_path}; echo \"PID=$(cat {pid_path})\"; disown"
    )
    run_remote(ssh, cmd, dry_run=dry_run, timeout=10)


def health_check(ssh: paramiko.SSHClient, name: str, *, dry_run: bool) -> None:
    """
    기동된 서버의 /health 를 curl 로 확인한다(인증 불필요 엔드포인트).

    framework.create_mcp_app 의 GET /health 는 {"status":"healthy", ...} 를 반환한다.
    기동 직후 잠깐의 준비 시간이 필요하므로 sleep 후 1회 curl 한다(실패해도 서버
    프로세스는 살아 있을 수 있으니 로그를 함께 확인하라는 안내를 가이드에 둠).
    """
    port = SERVER_PORTS[name]
    _print_header(f"헬스체크: {name} → http://localhost:{port}/health")
    run_remote(
        ssh,
        f"sleep 3; curl -s -m 5 http://localhost:{port}/health; echo",
        dry_run=dry_run,
        timeout=15,
    )


# ─────────────────────────────────────────────
# 엔트리포인트
# ─────────────────────────────────────────────
def _local_repo_root() -> str:
    """이 스크립트(scripts/_ssh_deploy_mcp.py)의 상위 = 레포 루트를 추정한다."""
    import os

    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def build_parser() -> argparse.ArgumentParser:
    """argparse 파서를 구성한다(별도 함수 — 테스트에서 --help/인자 파싱 검증 용이)."""
    parser = argparse.ArgumentParser(
        prog="_ssh_deploy_mcp",
        description=(
            "MCP 서버(db/diag/kowiki/docingest)를 DB 서버(.39)에 배포·기동한다. "
            "기본은 --dry-run(실행 안 함). 실제 배포는 --no-dry-run 필요."
        ),
    )
    parser.add_argument(
        "--servers",
        nargs="+",
        choices=sorted(SERVER_PORTS.keys()),
        default=["db", "diag"],
        help="기동할 MCP 서버 목록(기본: db diag). 예: --servers db kowiki docingest",
    )
    # dry-run 기본 True. 실제 실행은 --no-dry-run 으로만(fail-closed).
    parser.add_argument(
        "--dry-run",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="명령을 실행하지 않고 출력만(기본 True). 실제 배포는 --no-dry-run.",
    )
    parser.add_argument(
        "--host", default=DEFAULT_HOST, help=f"배포 대상 호스트(기본 {DEFAULT_HOST})"
    )
    parser.add_argument("--user", default=DEFAULT_USER, help=f"SSH 사용자(기본 {DEFAULT_USER})")
    parser.add_argument(
        "--password",
        default=DEFAULT_PASSWORD,
        help="SSH 비밀번호(기본: 상수). 운영 시 명시 주입 권장.",
    )
    parser.add_argument(
        "--remote-dir",
        default=DEFAULT_REMOTE_DIR,
        help=f"원격 코드 디렉토리(기본 {DEFAULT_REMOTE_DIR})",
    )
    parser.add_argument(
        "--python",
        default=DEFAULT_REMOTE_PYTHON,
        help=f"원격 파이썬 인터프리터(기본 {DEFAULT_REMOTE_PYTHON}). venv 면 그 경로 지정.",
    )
    parser.add_argument(
        "--api-key",
        default="local-key",
        help="MCP Bearer 인증 키(기본 placeholder). 운영 시 명시 주입.",
    )
    parser.add_argument(
        "--code-mode",
        choices=["upload", "pull", "skip"],
        default="upload",
        help="코드 배포 방식: upload(SFTP, 기본) / pull(원격 git pull) / skip(코드 배포 생략).",
    )
    parser.add_argument(
        "--skip-deps-check",
        action="store_true",
        help="원격 의존성 확인 단계를 건너뛴다.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """
    배포 절차를 순서대로 수행한다(코드 배포 → 의존성 확인 → 기동 → 헬스체크).

    Returns:
        0(정상). dry-run 이 기본이므로 무인 실행해도 원격을 건드리지 않는다.
    """
    args = build_parser().parse_args(argv)

    mode = "DRY-RUN(실행 안 함)" if args.dry_run else "실제 배포"
    print("=" * 60)
    print(f"MCP 배포 — 대상 {args.user}@{args.host} | 모드: {mode}")
    print(f"  서버: {', '.join(args.servers)} | 코드모드: {args.code_mode}")
    print(f"  원격 디렉토리: {args.remote_dir} | python: {args.python}")
    print("=" * 60)

    if not args.dry_run:
        # 실제 배포는 명시적 경고 — 운영자가 의도한 것인지 한 번 더 환기.
        print("\n[주의] 실제 배포 모드입니다. 원격 호스트에 파일을 올리고 프로세스를 기동합니다.\n")

    ssh = paramiko.SSHClient()
    # LAN 내부 사내 호스트(.39)만 대상이며 기존 _ssh_*.py 패턴과 동일하다(에어갭).
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())  # noqa: S507 — LAN 내부 사내 호스트 전용

    if args.dry_run:
        # dry-run 에서는 실제 SSH 접속도 하지 않는다 — 순수하게 "무엇을 할지"만 출력.
        print("[DRY-RUN] SSH 접속 생략 (실제 배포 시 paramiko 로 접속).")
    else:
        ssh.connect(args.host, username=args.user, password=args.password, timeout=10)

    try:
        # 1) 코드 배포
        if args.code_mode == "upload":
            deploy_code_upload(ssh, _local_repo_root(), args.remote_dir, dry_run=args.dry_run)
        elif args.code_mode == "pull":
            deploy_code_pull(ssh, args.remote_dir, dry_run=args.dry_run)
        else:
            _print_header("코드 배포 생략(--code-mode skip)")

        # 2) 의존성 확인
        if not args.skip_deps_check:
            check_dependencies(ssh, args.remote_dir, args.python, dry_run=args.dry_run)

        # 3) 서버 기동 + 4) 헬스체크
        for name in args.servers:
            start_server(
                ssh,
                name,
                args.remote_dir,
                args.python,
                args.api_key,
                dry_run=args.dry_run,
            )
            health_check(ssh, name, dry_run=args.dry_run)
    finally:
        if not args.dry_run:
            ssh.close()

    _print_header("배포 절차 종료")
    if args.dry_run:
        print("dry-run 이었습니다. 실제 배포하려면 --no-dry-run 을 붙이세요.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
