"""B200 컨테이너 내부에서 vLLM 추론 서버를 기동하는 스크립트.

무엇을 하는가:
    config/vllm_launch.yaml 의 프로파일(axmodel / hyperclova)을 읽어
    vLLM OpenAI 호환 API 서버 실행 명령을 조립하고, 컨테이너 내부(localhost)에서
    백그라운드로 기동한 뒤 /v1/models 가 200을 반환할 때까지 폴링한다.
    즉 "설정을 읽는다 → 실행 명령을 만든다 → 서버를 띄운다 → 준비될 때까지 기다린다"
    는 4단계 흐름을 하나의 CLI로 묶은 기동 도구다.

왜 필요한가:
    기존 scripts/_ssh_restart_vllm.py 는 원격 5090 서버에 SSH로 접속해 기동하는
    5090 전용 스크립트다. B200은 단일 컨테이너 안에서 localhost로 기동하므로
    SSH가 필요 없고 값(BF16·긴 컨텍스트·CUDA graph)도 달라, 별도 스크립트로 분리했다.
    두 스크립트를 억지로 합치면 분기가 복잡해지고 실수가 늘기 때문에 파일을 나눴다.

주요 함수(위에서 아래로 읽으면 전체 흐름이 보인다):
    load_profile()     : YAML에서 공통 defaults와 지정 프로파일을 읽어온다.
    build_command()    : defaults + profile을 합쳐 vLLM 실행 argv 리스트를 만든다.
    wait_until_ready() : /v1/models 를 폴링해 서버가 응답할 때까지 대기한다.
    main()             : CLI 인자를 파싱하고 위 세 함수를 순서대로 호출한다.

외부 의존:
    - config/vllm_launch.yaml (필수) — 프로파일과 공통 옵션의 단일 출처.
    - vllm 패키지 — 실제 추론 서버 모듈(vllm.entrypoints.openai.api_server).
      단, --dry-run 은 명령만 조립·출력하므로 vLLM/GPU 없이도 동작한다.

사용법:
    # 명령만 확인(GPU 없이도 실행 가능 — 조립 검증용)
    python scripts/start_vllm_local.py --profile axmodel --dry-run

    # 실제 기동(B200 컨테이너 안에서)
    python scripts/start_vllm_local.py --profile hyperclova

bake-off 주의:
    단일 B200 180GB에는 두 모델을 동시에 못 올린다. 한 번에 하나만 기동하고,
    비교가 끝나면 종료(--stop) 후 다른 프로파일을 기동한다.

작성자: 이현수 / 작성일: 2026-07-05
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
# reconfigure는 Python 3.7+ 표준 스트림에만 있으므로, 없는 환경(리다이렉트된
# 파이프 등)에서 AttributeError가 나지 않도록 hasattr로 존재 여부를 먼저 확인한다.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# 이 스크립트(scripts/)의 부모가 프로젝트 루트. 거기서 config/를 찾는다.
# __file__ 기준으로 경로를 계산하므로, 어느 작업 디렉토리에서 실행해도
# 항상 같은 config 파일을 가리킨다(상대경로로 인한 오작동 방지).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
LAUNCH_CONFIG_PATH = PROJECT_ROOT / "config" / "vllm_launch.yaml"


def load_profile(profile_name: str) -> tuple[dict, dict]:
    """vllm_launch.yaml에서 공통 defaults와 지정 프로파일을 읽어 반환한다.

    YAML은 크게 두 부분으로 나뉜다:
        - defaults: 모든 프로파일이 공유하는 공통 옵션(host, 로그 경로, 불리언 플래그).
        - profiles: 모델별 개별 설정(model 경로, 포트, 컨텍스트 길이 등).
    이 함수는 두 조각을 그대로 꺼내 (defaults, profile) 튜플로 넘겨주고,
    실제 조립은 build_command()가 담당한다(읽기와 조립 책임을 분리).

    설정 파일이 없거나 프로파일 이름이 틀리면 스택트레이스 대신 사람이 읽을 수 있는
    한글 오류 메시지를 찍고 sys.exit로 즉시 종료한다(운영자가 원인을 바로 알도록).

    Args:
        profile_name: "axmodel" 또는 "hyperclova" 등 프로파일 키.

    Returns:
        (defaults, profile) 두 dict. 프로파일이 없으면 명확한 에러로 종료.
    """
    # 설정 파일 존재 여부를 가장 먼저 확인 — 없으면 이후 단계가 전부 무의미하다.
    if not LAUNCH_CONFIG_PATH.exists():
        sys.exit(f"[오류] 기동 설정 파일이 없습니다: {LAUNCH_CONFIG_PATH}")

    # safe_load는 임의 파이썬 객체를 만들지 않는 안전한 YAML 파서다(코드 실행 위험 없음).
    with LAUNCH_CONFIG_PATH.open(encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 키가 없을 때를 대비해 .get으로 기본 빈 dict를 준다(KeyError로 죽지 않게).
    defaults = config.get("defaults", {})
    profiles = config.get("profiles", {})

    # 오타 등으로 없는 프로파일을 지정하면, 실제 있는 이름 목록을 함께 알려준다.
    if profile_name not in profiles:
        available = ", ".join(profiles.keys()) or "(없음)"
        sys.exit(f"[오류] 알 수 없는 프로파일 '{profile_name}'. 사용 가능: {available}")

    return defaults, profiles[profile_name]


def build_command(defaults: dict, profile: dict) -> list[str]:
    """프로파일 + 공통 옵션으로 vLLM 실행 명령(argv 리스트)을 조립한다.

    조립 순서:
        1) 모델·포트 등 프로파일 필수 인자를 먼저 넣는다.
        2) defaults의 공통 불리언 플래그를 "참일 때만" 덧붙인다.
        3) 도구 호출 파서, LoRA, 프로파일별 extra_args를 조건부로 추가한다.
    이렇게 "필수 → 공통 → 선택" 순서로 쌓으면 명령이 예측 가능해지고,
    어떤 옵션이 왜 붙었는지 추적하기 쉽다.

    Args:
        defaults: 모든 프로파일 공통 옵션(host, 불리언 플래그 등).
        profile:  선택된 모델 하나의 설정(model, port, max_model_len 등).

    Returns:
        subprocess에 그대로 넘길 수 있는 문자열 리스트(argv).

    왜 리스트로 만드나:
        shell=False로 실행하기 위해서다. 문자열 명령을 shell로 넘기면
        경로에 공백/특수문자가 있을 때 인젝션 위험이 생긴다(anti-pattern 회피).
    """
    # 파이썬 인터프리터로 vLLM OpenAI API 서버 모듈을 직접 실행한다.
    # sys.executable을 쓰면 현재 실행 중인 파이썬과 동일한 인터프리터를 보장한다
    # (가상환경/컨테이너 파이썬 불일치로 vllm import가 깨지는 것을 방지).
    # 모든 값은 str()로 감싸 YAML이 숫자/불리언으로 읽은 값도 argv 문자열로 통일한다.
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
    # (vLLM CLI는 "플래그 존재=켜짐"이라 False일 때는 아예 넣지 않아야 한다.)
    if defaults.get("trust_remote_code"):
        # 커스텀 모델 코드(HF repo의 modeling_*.py)를 신뢰해 로드. 사내 모델용.
        cmd.append("--trust-remote-code")
    if defaults.get("enable_prefix_caching"):
        # 프롬프트 앞부분(시스템 프롬프트 등) 재사용 캐시로 지연시간을 줄인다.
        cmd.append("--enable-prefix-caching")
    if defaults.get("enforce_eager"):
        # B200 기본값은 False라 보통 붙지 않는다(CUDA graph 사용). 5090 호환용.
        cmd.append("--enforce-eager")
    if defaults.get("enable_auto_tool_choice"):
        # 모델이 스스로 도구 호출 여부를 결정하도록 허용(tool_calls 자동 생성).
        cmd.append("--enable-auto-tool-choice")

    # 도구 호출 파서(모델 계열마다 다름 — hermes / qwen3_xml 등)
    # 모델이 뱉는 tool_call 표기 형식이 계열마다 달라, 프로파일에서 지정한다.
    if profile.get("tool_call_parser"):
        cmd += ["--tool-call-parser", str(profile["tool_call_parser"])]

    # LoRA — Phase 3(핫스왑 PoC)에서만 활성. 기본은 비활성.
    # enable_lora가 켜졌을 때만 관련 인자를 한 묶음으로 추가한다.
    if profile.get("enable_lora"):
        # max_lora_rank: 어댑터가 가질 수 있는 최대 rank(미지정 시 64로 안전 기본값).
        cmd += ["--enable-lora", "--max-lora-rank", str(profile.get("max_lora_rank", 64))]
        if profile.get("max_loras"):
            # 동시에 메모리에 올릴 수 있는 LoRA 어댑터 개수 상한.
            cmd += ["--max-loras", str(profile["max_loras"])]
        # lora_modules: {이름: 경로} → "이름=경로" 형태로 나열
        # vLLM은 어댑터를 "별칭=디렉토리" 문자열로 받으므로 dict를 그 형식으로 변환.
        lora_modules = profile.get("lora_modules") or {}
        for name, path in lora_modules.items():
            cmd += ["--lora-modules", f"{name}={path}"]

    # 프로파일별 추가 인자(예: HyperCLOVA 전용 플러그인 옵션)
    # 위 표준 플래그로 표현 못 하는 모델별 특수 옵션을 그대로 이어붙일 통로.
    for extra in profile.get("extra_args", []) or []:
        cmd.append(str(extra))

    return cmd


def wait_until_ready(port: int, host: str, max_wait_seconds: int = 300) -> bool:
    """/v1/models 가 200을 반환할 때까지 폴링한다(콜드 로딩 대기).

    vLLM은 프로세스가 떠도 모델 가중치를 GPU로 올리는 데(콜드 로딩) 수십 초~수 분이
    걸린다. 그동안 포트는 연결 거부되거나 응답이 없다. 그래서 즉시 성공을 기대하지 않고,
    /v1/models 엔드포인트가 200을 줄 때까지 3초 간격으로 반복 확인한다.

    Args:
        port:             확인할 서버 포트(프로파일의 port).
        host:             확인할 호스트(보통 127.0.0.1 — 컨테이너 내부 localhost).
        max_wait_seconds: 이 시간(초)을 넘기면 포기하고 False 반환. 기본 300초.

    Returns:
        준비되면 True, 시간 초과면 False.
    """
    url = f"http://{host}:{port}/v1/models"
    # time.monotonic()은 시스템 시계 변경에 영향받지 않는 단조 증가 시간이라
    # 경과 시간 측정에 적합하다(NTP 보정 등으로 뒤로 가는 일이 없음).
    deadline = time.monotonic() + max_wait_seconds
    attempt = 0
    # 마감 시각(deadline)에 도달할 때까지만 반복 — 무한 루프 방지.
    while time.monotonic() < deadline:
        attempt += 1
        # 기동 초기에는 연결 거부/타임아웃이 정상이다(서버가 아직 안 떴을 뿐).
        # 그래서 예외를 통째로 삼키지 않고, 예상되는 네트워크 예외만 잡아
        # 사유를 대기 메시지에 남긴다(anti-pattern #8: 에러 무시 금지).
        reason = ""
        try:
            # timeout=3: 한 번의 시도가 오래 매달리지 않도록 짧게 끊는다.
            with urllib.request.urlopen(url, timeout=3) as resp:  # noqa: S310 (localhost 고정)
                # 200이면 서버가 모델 로딩까지 끝내고 요청을 받을 준비가 된 상태.
                if resp.status == 200:
                    print(f"  [{attempt}] /v1/models → 200 (준비 완료)")
                    return True
                # 200이 아닌 다른 응답도 사유로 기록해 다음 폴링 로그에 남긴다.
                reason = f"HTTP {resp.status}"
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            # 연결 거부/타임아웃/기타 소켓 오류는 "아직 준비 안 됨"으로 간주하고 계속.
            reason = str(e)
        # 매 시도의 결과(성공 못 함)를 사유와 함께 출력해 진행 상황을 보여준다.
        print(f"  [{attempt}] 대기 중… ({url}) {reason}")
        # 3초 쉬고 다시 시도 — 서버에 과도한 요청을 보내지 않도록 간격을 둔다.
        time.sleep(3)
    # 여기까지 왔다면 max_wait_seconds 안에 준비되지 못한 것 → 실패로 반환.
    return False


def main() -> None:
    """CLI 진입점 — 인자를 파싱하고 설정 로드→명령 조립→기동→폴링을 순서대로 수행한다.

    흐름:
        1) --profile / --dry-run / --wait 인자를 읽는다.
        2) load_profile()로 설정을, build_command()로 실행 명령을 얻는다.
        3) --dry-run이면 명령만 출력하고 끝낸다(GPU/vLLM 불필요).
        4) 아니면 로그 파일을 열고 subprocess로 백그라운드 기동한 뒤,
           wait_until_ready()로 준비를 기다린다. 실패 시 exit(1)로 종료한다.
    """
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

    # 1) 설정 로드 → 2) 실행 명령 조립. 둘 다 GPU 없이도 동작하는 순수 로직이다.
    defaults, profile = load_profile(args.profile)
    cmd = build_command(defaults, profile)

    print(f"=== 프로파일: {args.profile} ({profile['served_model_name']}) ===")
    print("조립된 vLLM 명령:")
    # 사람이 읽기 쉽게 한 줄로 출력(따옴표 없이 — 실제 실행은 리스트로 shell=False)
    print("  " + " ".join(cmd))
    print()

    # --dry-run이면 여기서 멈춘다. 명령 조립이 맞는지 눈으로 확인하는 검증 모드.
    if args.dry_run:
        print("[dry-run] 실제 기동은 건너뜁니다.")
        return

    # 로그 디렉토리 준비 — 없으면 만들고, 있으면 그대로 사용(exist_ok=True).
    log_dir = Path(defaults.get("log_dir", str(PROJECT_ROOT / "logs")))
    log_dir.mkdir(parents=True, exist_ok=True)
    # 프로파일별로 로그 파일을 분리해 어느 모델의 기동 로그인지 바로 알 수 있게 한다.
    log_path = log_dir / f"vllm_{args.profile}.log"

    print(f"vLLM 기동 중… (로그: {log_path})")
    # 백그라운드 기동. stdout/stderr는 로그 파일로.
    # Popen은 자식 프로세스를 띄우고 곧바로 반환한다(블로킹하지 않음) — 그래서
    # 아래에서 폴링으로 준비 상태를 따로 확인한다. stderr는 STDOUT으로 합쳐 한 파일에.
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
    # 준비 완료면 접속 주소를 안내하고, 실패면 로그 위치를 알리고 비정상 종료(1)한다.
    if ready:
        print(f"\n기동 완료: http://{defaults.get('host')}:{profile['port']}/v1/models")
    else:
        # exit(1)로 종료 코드를 남겨, 상위 스크립트/CI가 실패를 감지할 수 있게 한다.
        print(f"\n[경고] 시간 초과. 로그를 확인하세요: {log_path}")
        sys.exit(1)


# 모듈로 import될 때는 실행되지 않고, 직접 실행할 때만 main()을 호출한다.
if __name__ == "__main__":
    main()
