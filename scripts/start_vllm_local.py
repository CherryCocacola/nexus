"""B200 컨테이너 내부에서 vLLM 추론 서버를 기동하는 스크립트.

무엇을 하는가:
    config/vllm_launch.yaml 의 프로파일(axmodel / hyperclova)을 읽어
    vLLM OpenAI 호환 API 서버 실행 명령을 조립하고, 컨테이너 내부(localhost)에서
    백그라운드로 기동한 뒤 /v1/models 가 200을 반환할 때까지 폴링한다.

왜 필요한가:
    기존 scripts/_ssh_restart_vllm.py 는 원격 5090 서버에 SSH로 접속해 기동하는
    5090 전용 스크립트다. B200은 단일 컨테이너 안에서 localhost로 기동하므로
    SSH가 필요 없고 값(BF16·긴 컨텍스트·CUDA graph)도 달라, 별도 스크립트로 분리했다.

사용법:
    # 명령만 확인(GPU 없이도 실행 가능 — 조립 검증용)
    python scripts/start_vllm_local.py --profile axmodel --dry-run

    # 실제 기동(B200 컨테이너 안에서)
    python scripts/start_vllm_local.py --profile hyperclova

bake-off 주의:
    단일 B200 180GB에는 두 모델을 동시에 못 올린다. 한 번에 하나만 기동하고,
    비교가 끝나면 종료(--stop) 후 다른 프로파일을 기동한다.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import yaml

# 콘솔 한글 깨짐 방지(Windows/컨테이너 공통) — stdout·stderr 모두 UTF-8로 맞춘다.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# 이 스크립트(scripts/)의 부모가 프로젝트 루트. 거기서 config/를 찾는다.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
LAUNCH_CONFIG_PATH = PROJECT_ROOT / "config" / "vllm_launch.yaml"


def load_profile(profile_name: str) -> tuple[dict, dict]:
    """vllm_launch.yaml에서 공통 defaults와 지정 프로파일을 읽어 반환한다.

    Args:
        profile_name: "axmodel" 또는 "hyperclova" 등 프로파일 키.

    Returns:
        (defaults, profile) 두 dict. 프로파일이 없으면 명확한 에러로 종료.
    """
    if not LAUNCH_CONFIG_PATH.exists():
        sys.exit(f"[오류] 기동 설정 파일이 없습니다: {LAUNCH_CONFIG_PATH}")

    with LAUNCH_CONFIG_PATH.open(encoding="utf-8") as f:
        config = yaml.safe_load(f)

    defaults = config.get("defaults", {})
    profiles = config.get("profiles", {})

    if profile_name not in profiles:
        available = ", ".join(profiles.keys()) or "(없음)"
        sys.exit(f"[오류] 알 수 없는 프로파일 '{profile_name}'. 사용 가능: {available}")

    return defaults, profiles[profile_name]


def build_command(defaults: dict, profile: dict) -> list[str]:
    """프로파일 + 공통 옵션으로 vLLM 실행 명령(argv 리스트)을 조립한다.

    왜 리스트로 만드나:
        shell=False로 실행하기 위해서다. 문자열 명령을 shell로 넘기면
        경로에 공백/특수문자가 있을 때 인젝션 위험이 생긴다(anti-pattern 회피).
    """
    # 파이썬 인터프리터로 vLLM OpenAI API 서버 모듈을 직접 실행한다.
    cmd: list[str] = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        str(profile["model"]),
        "--served-model-name",
        str(profile["served_model_name"]),
        "--dtype",
        str(profile.get("dtype", "bfloat16")),
        "--max-model-len",
        str(profile["max_model_len"]),
        "--gpu-memory-utilization",
        str(profile["gpu_memory_utilization"]),
        "--port",
        str(profile["port"]),
        "--host",
        str(defaults.get("host", "127.0.0.1")),
    ]

    # 공통 불리언 플래그 — 값이 참일 때만 플래그를 붙인다.
    if defaults.get("trust_remote_code"):
        cmd.append("--trust-remote-code")
    if defaults.get("enable_prefix_caching"):
        cmd.append("--enable-prefix-caching")
    if defaults.get("enforce_eager"):
        # B200 기본값은 False라 보통 붙지 않는다(CUDA graph 사용). 5090 호환용.
        cmd.append("--enforce-eager")
    if defaults.get("enable_auto_tool_choice"):
        cmd.append("--enable-auto-tool-choice")

    # 도구 호출 파서(모델 계열마다 다름 — hermes / qwen3_xml 등)
    if profile.get("tool_call_parser"):
        cmd += ["--tool-call-parser", str(profile["tool_call_parser"])]

    # LoRA — Phase 3(핫스왑 PoC)에서만 활성. 기본은 비활성.
    if profile.get("enable_lora"):
        cmd += ["--enable-lora", "--max-lora-rank", str(profile.get("max_lora_rank", 64))]
        if profile.get("max_loras"):
            cmd += ["--max-loras", str(profile["max_loras"])]
        # lora_modules: {이름: 경로} → "이름=경로" 형태로 나열
        lora_modules = profile.get("lora_modules") or {}
        for name, path in lora_modules.items():
            cmd += ["--lora-modules", f"{name}={path}"]

    # 프로파일별 추가 인자(예: HyperCLOVA 전용 플러그인 옵션)
    for extra in profile.get("extra_args", []) or []:
        cmd.append(str(extra))

    return cmd


def wait_until_ready(port: int, host: str, max_wait_seconds: int = 300) -> bool:
    """/v1/models 가 200을 반환할 때까지 폴링한다(콜드 로딩 대기).

    Returns:
        준비되면 True, 시간 초과면 False.
    """
    url = f"http://{host}:{port}/v1/models"
    deadline = time.monotonic() + max_wait_seconds
    attempt = 0
    while time.monotonic() < deadline:
        attempt += 1
        # 기동 초기에는 연결 거부/타임아웃이 정상이다(서버가 아직 안 떴을 뿐).
        # 그래서 예외를 통째로 삼키지 않고, 예상되는 네트워크 예외만 잡아
        # 사유를 대기 메시지에 남긴다(anti-pattern #8: 에러 무시 금지).
        reason = ""
        try:
            with urllib.request.urlopen(url, timeout=3) as resp:  # noqa: S310 (localhost 고정)
                if resp.status == 200:
                    print(f"  [{attempt}] /v1/models → 200 (준비 완료)")
                    return True
                reason = f"HTTP {resp.status}"
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            reason = str(e)
        print(f"  [{attempt}] 대기 중… ({url}) {reason}")
        time.sleep(3)
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="B200 로컬 vLLM 기동")
    parser.add_argument(
        "--profile",
        required=True,
        help="기동할 프로파일 이름 (config/vllm_launch.yaml의 profiles 키, 예: axmodel)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="실제 기동 없이 조립된 vLLM 명령만 출력(GPU 불필요, 검증용)",
    )
    parser.add_argument(
        "--wait",
        type=int,
        default=300,
        help="/v1/models 준비 대기 최대 시간(초). 기본 300",
    )
    args = parser.parse_args()

    defaults, profile = load_profile(args.profile)
    cmd = build_command(defaults, profile)

    print(f"=== 프로파일: {args.profile} ({profile['served_model_name']}) ===")
    print("조립된 vLLM 명령:")
    # 사람이 읽기 쉽게 한 줄로 출력(따옴표 없이 — 실제 실행은 리스트로 shell=False)
    print("  " + " ".join(cmd))
    print()

    if args.dry_run:
        print("[dry-run] 실제 기동은 건너뜁니다.")
        return

    # 로그 디렉토리 준비
    log_dir = Path(defaults.get("log_dir", str(PROJECT_ROOT / "logs")))
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"vllm_{args.profile}.log"

    print(f"vLLM 기동 중… (로그: {log_path})")
    # 백그라운드 기동. stdout/stderr는 로그 파일로.
    with log_path.open("w", encoding="utf-8") as log_file:
        subprocess.Popen(  # noqa: S603 (cmd는 위에서 리스트로 안전하게 조립)
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            cwd=str(PROJECT_ROOT),
        )

    print("서비스 준비 폴링 시작…")
    ready = wait_until_ready(
        port=int(profile["port"]),
        host=str(defaults.get("host", "127.0.0.1")),
        max_wait_seconds=args.wait,
    )
    if ready:
        print(f"\n기동 완료: http://{defaults.get('host')}:{profile['port']}/v1/models")
    else:
        print(f"\n[경고] 시간 초과. 로그를 확인하세요: {log_path}")
        sys.exit(1)


if __name__ == "__main__":
    main()
