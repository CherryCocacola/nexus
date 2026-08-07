# 선언된 상태와 실제 돌고 있는 상태를 대조해 "다른 것만" 보고하는 점검 도구.
"""
드리프트 점검 — 재부팅을 기다리지 않고 미리 어긋난 곳을 찾는다.

[왜 만들었나 — 2026-08-07 재부팅 사고]
    112를 49일 만에 재부팅했더니 문제가 7건 쏟아졌다. 재부팅이 원인이 아니라,
    그동안 "돌고 있는 프로세스에는 적용됐지만 영속 정의에는 반영되지 않은" 변경이
    쌓여 있다가 한꺼번에 정산된 것이다. 7건 중 5건이 이 종류였다.

      · 터널 systemd 유닛 파일에는 -L 이 1개인데 실행 프로세스는 3개를 열고 있었다
        → 재시작하는 순간 비전·이미지가 조용히 끊겼을 상태
      · 설정이 B200 이관 후에도 옛 주소를 가리켜 두 도구가 죽어 있었다
      · 호스트에 방치된 PostgreSQL 이 도커와 같은 포트를 선점했다
      · 리포에 커밋했지만 컨테이너에 docker cp 하지 않은 파일이 있었다

    전부 **재부팅 전에 볼 수 있었던 것들**이다. 이 스크립트가 그걸 본다.

[원칙]
    - 다른 것만 출력한다. 아무 것도 안 나오면 정상이다.
    - 검사 하나가 실패해도 나머지는 계속한다(그 실패 자체를 드리프트로 보고한다).
    - 종료 코드 0=정상 / 1=드리프트 있음 → cron·CI 에 그대로 걸 수 있다.
    - 자격증명은 환경변수로만 받는다. 이 파일에 비밀번호를 적지 않는다.

[사용]
    NEXUS_OPS_SSH_PASSWORD=... python deployment/drift_check.py
    NEXUS_OPS_SSH_PASSWORD=... python deployment/drift_check.py --skip-b200
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

# ─────────────────────────────────────────────
# 선언된 상태(declared) — 이 블록이 "이래야 한다"의 정본이다
# ─────────────────────────────────────────────
HOST = os.environ.get("NEXUS_OPS_HOST", "192.168.21.112")
USER = os.environ.get("NEXUS_OPS_USER", "idino")

# 112에서 항상 떠 있어야 하는 컨테이너와 재시작 정책.
EXPECTED_CONTAINERS = {
    "nexus-web": "unless-stopped",
    "idino-postgres": "unless-stopped",
    "idino-redis": "unless-stopped",
}
# 112의 systemd 유닛 — enabled(재부팅 자동기동) + active(지금 동작) 둘 다 필요.
EXPECTED_UNITS = (
    "nexus-b200-tunnel",
    "nexus-devstral-tunnel",
    "nexus-embedding",
    "nexus-flux",
    "docker",
)
# 터널이 열어야 하는 로컬 포트. 유닛 파일과 실행 프로세스 양쪽에서 확인한다.
EXPECTED_TUNNEL_PORTS = ("18001", "18003", "18004", "18005")
# 도커가 잡고 있어야 하는 포트 — 다른 프로세스가 선점하면 컨테이너가 못 뜬다.
DOCKER_OWNED_PORTS = ("5440", "6340", "8600")

# B200에서 떠 있어야 하는 tmux 세션과 포트.
B200_SESSIONS = ("vllm", "embed", "image", "vision", "coder", "watchdog")
B200_PORTS = ("8001", "8002", "8003", "8004", "8005")
# LogLevel=ERROR 를 반드시 넣는다: 이게 없으면 ssh 가 매번
# "Warning: Permanently added ... to the list of known hosts." 를 뱉어 출력에 섞이고,
# 숫자를 세는 검사가 그 줄 때문에 오판한다(실제로 첫 실행에서 거짓 양성을 냈다).
B200_SSH = (
    "ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o BatchMode=yes "
    "-o LogLevel=ERROR -i /home/idino/nexus-tunnel/nexus_key -p 45702 idino_user@59.150.33.1"
)

# 배포본과 대조할 코드 확장자(파이썬 캐시·바이너리는 제외).
CODE_SUFFIXES = (".py", ".html", ".css", ".js", ".md")

# 컨테이너 안에서 실행할 해시 스크립트.
#   CR 을 지운 내용으로 md5 를 낸다 — 로컬의 content_hash() 와 같은 기준이라야
#   줄바꿈 차이가 거짓 양성이 되지 않는다(배포본에 CRLF/LF 가 섞여 있다).
REMOTE_HASH_SCRIPT = r"""cd /app || exit 1
find core web -type f ! -path '*/__pycache__/*' | while read -r f; do
  h=$(tr -d '\r' < "$f" | md5sum | cut -d' ' -f1)
  printf '%s %s\n' "$h" "$f"
done
"""


def content_hash(data: bytes) -> str:
    """줄바꿈을 정규화한 뒤 해시한다.

    Windows 개발 PC 의 작업본은 core.autocrlf 때문에 CRLF 이고 리눅스 컨테이너의
    배포본은 LF 다. 원시 바이트로 비교하면 **내용이 같아도 전부 다르다고 나온다**
    (첫 실행에서 실제로 5건이 거짓 양성이었다). 여기서 보려는 것은 코드 내용이
    어긋났는지이지 줄바꿈 형식이 아니다.
    """
    return hashlib.md5(data.replace(b"\r\n", b"\n")).hexdigest()  # noqa: S324


def first_number(text: str) -> int | None:
    """원격 출력에서 숫자만 뽑는다(ssh 경고 등 잡음 줄을 건너뛴다)."""
    for line in text.splitlines():
        s = line.strip()
        if s.isdigit():
            return int(s)
    return None


@dataclass
class Drift:
    area: str
    what: str
    declared: str
    actual: str
    hint: str = ""


@dataclass
class Report:
    drifts: list[Drift] = field(default_factory=list)
    checked: list[str] = field(default_factory=list)

    def add(self, *a, **k) -> None:
        self.drifts.append(Drift(*a, **k))

    def ok(self, name: str) -> None:
        self.checked.append(name)


# ─────────────────────────────────────────────
# 원격 실행
# ─────────────────────────────────────────────
class Remote:
    def __init__(self, password: str) -> None:
        import paramiko

        self.c = paramiko.SSHClient()
        self.c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self.c.connect(HOST, username=USER, password=password, timeout=30)

    def sh(self, cmd: str, timeout: int = 120) -> str:
        _i, o, e = self.c.exec_command(cmd, timeout=timeout)
        return (o.read() + e.read()).decode("utf-8", "replace").strip()

    def b200(self, cmd: str, timeout: int = 180) -> str:
        # 중첩 따옴표를 피하려고 base64 로 감싸 보낸다.
        b = base64.b64encode(cmd.encode()).decode()
        return self.sh(f"{B200_SSH} \"echo {b} | base64 -d | bash\"", timeout)

    def close(self) -> None:
        self.c.close()


# ─────────────────────────────────────────────
# 검사들 — 각각 오늘 실제로 터진 사고 하나에 대응한다
# ─────────────────────────────────────────────
def check_code_hashes(r: Remote, rep: Report) -> None:
    """리포에 커밋했는데 컨테이너에 안 올라간 파일을 찾는다.

    실제 사고: config.py 를 커밋만 하고 docker cp 하지 않아 배포본이 뒤처져 있었다.
    """
    root = Path(__file__).resolve().parent.parent
    git_cmd = ["git", "ls-files", "core", "web"]  # 고정 리터럴 — 사용자 입력 없음
    tracked = subprocess.run(  # noqa: S603
        git_cmd,
        cwd=root, capture_output=True, text=True, check=False,
    ).stdout.splitlines()
    local: dict[str, str] = {}
    for rel in tracked:
        if not rel.endswith(CODE_SUFFIXES):
            continue
        p = root / rel
        if p.is_file():
            local[rel] = content_hash(p.read_bytes())

    # 원격 해시도 CR 을 지운 내용으로 계산한다(위 content_hash 와 같은 기준).
    #   따옴표를 여러 겹(paramiko→bash→docker→sh) 통과시키면 반드시 깨진다.
    #   실제로 한 번 깨져 128개가 거짓 양성으로 나왔다. 그래서 스크립트를 파일로
    #   넣어 실행한다 — 셸 인용을 아예 통과시키지 않는 방법이 가장 안전하다.
    script = REMOTE_HASH_SCRIPT
    b64 = base64.b64encode(script.encode()).decode()
    out = r.sh(
        f"echo {b64} | base64 -d > /tmp/_drift_hash.sh && "
        "docker cp /tmp/_drift_hash.sh nexus-web:/tmp/_drift_hash.sh >/dev/null && "
        "docker exec nexus-web sh /tmp/_drift_hash.sh",
        300,
    )
    remote: dict[str, str] = {}
    for line in out.splitlines():
        parts = line.split(None, 1)
        if len(parts) == 2:
            remote[parts[1].strip()] = parts[0]

    stale = [f for f, h in local.items() if f in remote and remote[f] != h]
    for f in sorted(stale)[:10]:
        rep.add("112/코드", f, "리포 버전", "배포본이 다름",
                "docker cp 로 반영하고 컨테이너를 재시작하라")
    if len(stale) > 10:
        rep.add("112/코드", f"…외 {len(stale) - 10}개", "리포 버전", "배포본이 다름")
    if not stale:
        rep.ok(f"코드 해시 {len(local)}개 일치")


def check_units_vs_processes(r: Remote, rep: Report) -> None:
    """systemd 유닛 파일의 정의와 실제 실행 중인 명령이 같은지 본다.

    실제 사고: 터널 유닛 파일에는 -L 이 1개인데 실행 프로세스는 3개를 열고 있었다.
    active 였지만 재시작하면 비전·이미지가 끊기는 상태였고, is-active 로는 안 보였다.
    """
    for unit in EXPECTED_UNITS:
        state = r.sh(f"systemctl is-enabled {unit} 2>&1; systemctl is-active {unit} 2>&1")
        lines = state.splitlines()
        enabled = lines[0] if lines else "?"
        active = lines[1] if len(lines) > 1 else "?"
        if "enabled" not in enabled:
            rep.add("112/systemd", unit, "enabled", enabled,
                    "재부팅 시 자동 기동되지 않는다")
        if active != "active":
            rep.add("112/systemd", unit, "active", active)

    # 터널: 유닛 정의(재부팅 후 열릴 포트) vs 지금 실제로 열린 포트
    declared = r.sh(
        "systemctl show nexus-b200-tunnel nexus-devstral-tunnel -p ExecStart --value "
        "| grep -oE '127.0.0.1:1800[0-9]:' | grep -oE '1800[0-9]' | sort -u | tr '\\n' ' '"
    ).split()
    actual = r.sh(
        "ss -ltn | grep -oE '127.0.0.1:1800[0-9]' | grep -oE '1800[0-9]' | sort -u | tr '\\n' ' '"
    ).split()
    for port in EXPECTED_TUNNEL_PORTS:
        if port not in declared:
            rep.add("112/터널", f"포트 {port}", "유닛 파일에 존재", "유닛 파일에 없음",
                    "지금은 열려 있어도 재시작·재부팅하면 사라진다")
        if port not in actual:
            rep.add("112/터널", f"포트 {port}", "열려 있어야 함", "닫힘")
    if set(declared) >= set(EXPECTED_TUNNEL_PORTS) and set(actual) >= set(EXPECTED_TUNNEL_PORTS):
        rep.ok(f"터널 포트 {len(EXPECTED_TUNNEL_PORTS)}개 (유닛 정의=실행 상태)")


def check_port_owners(r: Remote, rep: Report) -> None:
    """도커가 잡아야 할 포트를 다른 프로세스가 선점하지 않았는지 본다.

    실제 사고: 호스트에 방치돼 있던 PostgreSQL 이 부팅 때 5440 을 4초 먼저 잡아
    idino-postgres 컨테이너가 뜨지 못했고, NOVA 는 46MB 짜리 빈 DB 에 붙었다.
    """
    for port in DOCKER_OWNED_PORTS:
        # docker-proxy 는 backlog 4096, 네이티브 서버는 대개 다른 값이다.
        owner = r.sh(f"ss -ltnp 2>/dev/null | grep ':{port} ' | head -1")
        if not owner:
            rep.add("112/포트", f"{port}", "리스닝", "아무도 안 열고 있음")
            continue
        # docker 가 아닌 프로세스가 보이면 경합 후보다.
        procs = r.sh(
            f"ss -ltnp 2>/dev/null | grep ':{port} ' | grep -oE 'users:\\(\\(\"[^\"]+\"' "
            "| grep -oE '\"[^\"]+\"' | tr -d '\"' | sort -u | tr '\\n' ' '"
        ).split()
        foreign = [p for p in procs if p not in ("docker-proxy", "docker")]
        if foreign:
            rep.add("112/포트", f"{port}", "docker 소유", f"{' '.join(foreign)} 가 점유",
                    "재부팅 시 컨테이너가 이 포트를 못 잡을 수 있다")
    rep.ok(f"포트 소유자 {len(DOCKER_OWNED_PORTS)}개")

    # 호스트에 방치된 PostgreSQL 패키지(오늘 사고의 직접 원인)
    leftover = first_number(r.sh("dpkg -l 2>/dev/null | grep -cE '^ii +postgresql-[0-9]'"))
    if leftover:
        rep.add("112/패키지", "호스트 PostgreSQL", "미설치(도커로 운영)",
                f"{leftover}개 설치됨",
                "재부팅 시 도커와 포트가 충돌할 수 있다 — apt purge 검토")


def check_containers(r: Remote, rep: Report) -> None:
    """컨테이너가 떠 있고 재시작 정책이 걸려 있는지 본다(재부팅 자동복구)."""
    running = set(r.sh("docker ps --format '{{.Names}}'").split())
    for name, policy in EXPECTED_CONTAINERS.items():
        if name not in running:
            rep.add("112/컨테이너", name, "실행 중", "정지됨")
            continue
        actual = r.sh(f"docker inspect {name} --format '{{{{.HostConfig.RestartPolicy.Name}}}}'")
        if actual != policy:
            rep.add("112/컨테이너", name, f"restart={policy}", f"restart={actual}",
                    "재부팅 후 자동으로 뜨지 않는다")
    rep.ok(f"컨테이너 {len(EXPECTED_CONTAINERS)}개")


def check_model_identity(r: Remote, rep: Report) -> None:
    """설정이 가리키는 모델 서버가 정말 그 모델을 서빙하는지 본다.

    실제 사고: B200 이관 후 vision_url·image_url 이 옛 주소를 가리켰고
    vision_model 이름도 실제 서빙명과 달랐다. 부팅 로그는 깨끗했다.
    """
    out = r.sh(
        "docker logs --since 24h nexus-web 2>&1 | grep -a '모델 서버 신원' | tail -1"
    )
    if not out:
        rep.add("112/모델", "신원 로그", "부팅 시 기록됨", "없음",
                "부팅 자가검증이 없는 구버전 배포본일 수 있다")
        return
    if "불일치" in out or "닿지않음" in out:
        rep.add("112/모델", "엔드포인트 신원", "전부 일치", out.split("신원:")[-1].strip())
    else:
        rep.ok("모델 서버 신원 5종")

    db = r.sh("docker logs --since 24h nexus-web 2>&1 | grep -a 'DB 신원' | tail -1")
    if db and ("비어 있습니다" in db or "테이블이 하나도" in db):
        rep.add("112/DB", "신원", "데이터 있는 DB", db.split("DB 신원:")[-1].strip())
    elif db:
        rep.ok("DB 신원")


def check_b200(r: Remote, rep: Report) -> None:
    """B200 — tmux 세션·포트·스크립트 해시·재부팅 자동기동 등록 여부."""
    sessions = r.b200("tmux ls 2>/dev/null | cut -d: -f1 | tr '\\n' ' '").split()
    for s in B200_SESSIONS:
        if s not in sessions:
            rep.add("B200/세션", s, "실행 중", "없음")
    ports = r.b200("ss -ltn | grep -oE ':800[0-9]' | tr -d : | sort -u | tr '\\n' ' '").split()
    for p in B200_PORTS:
        if p not in ports:
            rep.add("B200/포트", p, "리스닝", "닫힘")

    # 재부팅 자동기동(cron @reboot) 등록 여부 — 없으면 재부팅 후 전부 수동 복구다.
    cron = first_number(r.b200("crontab -l 2>/dev/null | grep -c boot_start.sh"))
    if not cron:
        rep.add("B200/자동기동", "cron @reboot", "등록됨", "없음",
                "재부팅하면 GPU 백엔드를 손으로 살려야 한다")

    # 리포의 운영 스크립트와 서버 사본이 같은지
    root = Path(__file__).resolve().parent / "b200"
    names = sorted(p.name for p in root.glob("*.sh"))
    out = r.b200("cd /NHNHOME/nexus && md5sum " + " ".join(names) + " 2>/dev/null")
    remote = {}
    for line in out.splitlines():
        parts = line.split(None, 1)
        if len(parts) == 2:
            remote[parts[1].strip()] = parts[0]
    for n in names:
        local_hash = content_hash((root / n).read_bytes())
        if n not in remote:
            rep.add("B200/스크립트", n, "서버에 배포됨", "없음")
        elif remote[n] != local_hash:
            rep.add("B200/스크립트", n, "리포와 동일", "서버 사본이 다름")
    if not any(d.area.startswith("B200") for d in rep.drifts):
        rep.ok(f"B200 세션 {len(B200_SESSIONS)}개·포트 {len(B200_PORTS)}개·스크립트 {len(names)}개")


CHECKS = (
    ("코드 해시", check_code_hashes),
    ("systemd 정의↔실행", check_units_vs_processes),
    ("포트 소유자", check_port_owners),
    ("컨테이너", check_containers),
    ("모델·DB 신원", check_model_identity),
)


def main() -> int:
    ap = argparse.ArgumentParser(description="선언된 상태와 실제를 대조한다")
    ap.add_argument("--skip-b200", action="store_true", help="B200 점검 생략(느릴 때)")
    args = ap.parse_args()

    password = os.environ.get("NEXUS_OPS_SSH_PASSWORD")
    if not password:
        print("NEXUS_OPS_SSH_PASSWORD 환경변수가 필요하다.", file=sys.stderr)
        return 2

    rep = Report()
    r = Remote(password)
    try:
        checks = list(CHECKS) + ([] if args.skip_b200 else [("B200", check_b200)])
        for name, fn in checks:
            try:
                fn(r, rep)
            except Exception as e:  # noqa: BLE001 — 검사 실패도 드리프트로 취급
                rep.add("점검", name, "정상 수행", f"{type(e).__name__}: {e}")
    finally:
        r.close()

    print("=" * 68)
    if not rep.drifts:
        print("드리프트 없음 — 선언된 상태와 실제가 일치한다")
        for c in rep.checked:
            print(f"  · {c}")
        return 0

    print(f"드리프트 {len(rep.drifts)}건")
    print("=" * 68)
    for d in rep.drifts:
        print(f"\n[{d.area}] {d.what}")
        print(f"   선언: {d.declared}")
        print(f"   실제: {d.actual}")
        if d.hint:
            print(f"   → {d.hint}")
    if rep.checked:
        print("\n정상 항목: " + ", ".join(rep.checked))
    return 1


if __name__ == "__main__":
    sys.exit(main())
