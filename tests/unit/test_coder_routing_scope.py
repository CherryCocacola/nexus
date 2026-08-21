# 코딩 모델로 보낼 질의의 범위를 고정한다 — 구현 요청은 잡고, 일반 업무 요청은 흘린다.
"""
코더 라우팅 범위 테스트 (2026-08-20).

왜 필요한가 (실측):
    앵커(A.X)에게 "turnover_rate 함수를 추가해줘" 를 시키면 `Write` 4회가 전부
    구문 오류로 거부되고, 완성 코드를 답변에 10,723자 쏟아낸 뒤 끝났다(파일 미변경).
    같은 과제를 코딩 모델로 보내면 `Edit` 3회로 함수와 테스트를 넣고 6 passed.
    그래서 "구현/추가" 형태를 코딩 모델로 보내야 한다.

왜 부분 문자열이 아니라 정규식인가:
    NOVA 는 코딩 전용 도구가 아니다. "만들어" 를 키워드로 넣으면 "보고서 만들어줘"
    같은 일반 업무 요청이 코딩 모델로 새고, 그 모델은 한국어 문서 작성이 약하다.
    그래서 동사 단독이 아니라 **코딩 명사와 함께**일 때만 건다.

왜 배포 YAML 을 읽는가:
    패턴은 config/*.yaml 에 산다(tool_regex_patterns 와 같은 관례). 테스트가 패턴을
    복사해 두면 설정만 바뀌었을 때 조용히 어긋난다 — 그래서 실제 배포 파일을 읽는다.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from core.config import RoutingConfig
from core.orchestrator.routing import RoutingResolver

CONFIGS = [
    "config/nexus_config.pc.yaml",
    "config/nexus_config.yaml",
    "config/nexus_config.112.yaml",
]


def _routing_from(path: str, **overrides) -> RoutingConfig:
    """배포 설정의 routing 섹션을 그대로 읽어 RoutingConfig 로 만든다."""
    raw = yaml.safe_load(open(path, encoding="utf-8"))["routing"]
    raw.update(overrides)
    return RoutingConfig(**raw)


@pytest.fixture
def resolver() -> RoutingResolver:
    # 코더 판정 자체를 보는 테스트이므로 스위치는 켠 상태로 고정한다.
    return RoutingResolver(_routing_from(CONFIGS[0], coder_enabled=True))


# 코딩 모델이 받아야 하는 요청.
CODING = [
    "대화창 구현해줘",
    "로그인 기능 추가해줘",
    "파일 업로드 함수를 만들어줘",
    "결제 API 엔드포인트 작성해줘",
    "이 코드 리팩터링해줘",
    "에러 디버깅 해줘",
    "이 버그를 고쳐줘",
    # 2026-08-21 보강으로 새로 포착되는 표현들. 이전 패턴은 20종 중 6종만 잡았다.
    "korean_ratio 함수의 시그니처에 타입 힌트를 붙여줘",
    "변수명을 더 명확하게 바꿔줘",
    "이 함수에 주석을 달아줘",
    "import 문 정리해줘",
    "app.py 의 라우터 부분을 개선해줘",
    "이 부분 성능 개선해줘",
]

# 앵커가 받아야 하는 요청 — 코딩 모델로 새면 품질이 떨어진다.
NOT_CODING = [
    "보고서 만들어줘",
    "회의록 문서 작성해줘",
    "김치 만드는 법 알려줘",
    "구현 원리를 설명해줘",
    "오늘 날씨 어때?",
    "광합성이 뭐야?",
    "이 문서 요약해줘",
    # 순수 테스트 작성은 앵커가 받는다. 08-07 Devstral 실측으로 정했고,
    # 2026-08-21 Qwen3-Coder 로 재측정해 **같은 결론**을 얻었다
    #   (생성한 pytest 를 실제 실행: A.X 3/3 통과, Qwen 0/3 실패.
    #    실패 원인은 경계 조건 오독 — `q < threshold` 를 `<=` 로 읽었다).
    "이 함수 테스트 코드 짜줘",
    "write a unit test for this",
    # 2026-08-21 보강 시 가장 위험했던 표현들(동사를 넓히면 여기가 먼저 샌다).
    "발표 자료 제목을 바꿔줘",
    "회의 일정 추가해줘",
    "이메일 초안 작성해줘",
    "제안서.docx 요약해줘",
    "매출 데이터 분석해줘",
]


@pytest.mark.parametrize("text", CODING)
def test_implementation_requests_go_to_coder(resolver: RoutingResolver, text: str) -> None:
    """구현·수정 요청은 코딩 모델로 간다."""
    decision = resolver.resolve(text)
    assert decision.use_coder is True, f"코딩 모델로 안 갔다: {text}"
    # 코딩 턴에는 사내 문서 RAG 가 붙으면 안 된다.
    assert decision.query_class == "TOOL"
    assert decision.inject_knowledge_rag is False


@pytest.mark.parametrize("text", NOT_CODING)
def test_general_requests_stay_on_anchor(resolver: RoutingResolver, text: str) -> None:
    """일반 업무·지식 요청과 순수 테스트 작성은 앵커에 남는다(오탐 방지)."""
    decision = resolver.resolve(text)
    assert decision.use_coder is False, f"코딩 모델로 잘못 샜다: {text}"


def test_compound_request_with_tests_still_goes_to_coder(resolver: RoutingResolver) -> None:
    """구현이 주된 요청이면 '테스트도 추가' 문구가 있어도 코딩 모델로 간다.

    제외 규칙이 과하게 걸리면, 실제로 가장 흔한 형태("함수 추가하고 테스트도 추가")가
    통째로 앵커로 되돌아간다 — 그것이 바로 이번에 고치려던 실패 유형이다.
    """
    text = "turnover_rate 함수를 추가해줘. test_inventory.py 에 테스트도 추가해줘."
    assert resolver.resolve(text).use_coder is True


def test_debugging_tests_is_not_excluded(resolver: RoutingResolver) -> None:
    """'테스트 코드 디버깅'은 제외 대상이 아니다 — 디버깅은 동률이 확인된 범주다."""
    assert resolver.resolve("테스트 코드 디버깅해줘").use_coder is True


def test_disabled_switch_blocks_everything() -> None:
    """coder_enabled=False 면 어떤 문장도 코딩 모델로 가지 않는다(무회귀·fail-safe)."""
    resolver = RoutingResolver(_routing_from(CONFIGS[0], coder_enabled=False))
    for text in CODING:
        assert resolver.resolve(text).use_coder is False


@pytest.mark.parametrize("path", CONFIGS)
def test_every_config_ships_the_patterns(path: str) -> None:
    """설정 3본이 **같은** 패턴을 갖는다 — 한 본만 고치는 드리프트를 막는다.

    개수를 박아 두지 않는다(2026-08-21 3개→5개로 보강했다). 대신 세 본이
    서로 같은지를 본다 — 실제 사고는 "한 본만 고쳤다"에서 났다.
    """
    raw = yaml.safe_load(open(path, encoding="utf-8"))["routing"]
    patterns = raw.get("coder_regex_patterns") or []
    assert patterns, f"{path}: 패턴이 비었다"
    reference = yaml.safe_load(open(CONFIGS[0], encoding="utf-8"))["routing"]
    assert patterns == reference["coder_regex_patterns"], f"{path}: 다른 본과 어긋난다"


@pytest.mark.parametrize("path", CONFIGS)
def test_every_surface_has_coder_enabled(path: str) -> None:
    """세 표면 모두 코딩 라우팅이 켜져 있다(2026-08-21 서버까지 활성).

    ★계약 변경 기록: 08-20 에는 "112 는 꺼져 있어야 한다"였다. 배포된 컨테이너에
    코더 모델명 수정(404)이 없었기 때문이다. 배포·검증을 마치고 켰다 —
    VSCode 플러그인 find/replace 실측에서 앵커 75~83% / 코딩 모델 100%,
    속도 4~23초 → 2초. 켠 상태로 표면 e2e 34/34 통과.

    되돌리려면 이 테스트부터 바꿔야 한다 — 조용히 꺼지는 것을 막기 위함이다.
    """
    raw = yaml.safe_load(open(path, encoding="utf-8"))["routing"]
    assert raw.get("coder_enabled", False) is True, f"{path}: 코딩 라우팅이 꺼져 있다"


def test_broken_regex_does_not_break_routing() -> None:
    """패턴 하나가 깨져도 라우팅 전체가 죽지 않는다."""
    resolver = RoutingResolver(
        RoutingConfig(
            enabled=True,
            coder_enabled=True,
            coder_regex_patterns=["(unclosed", "구현\\s*해"],
        )
    )
    assert resolver.resolve("대화창 구현해줘").use_coder is True


def test_patterns_live_in_yaml_not_code_defaults() -> None:
    """코드 기본값은 비어 있어야 한다 — 운영자가 키워드를 좁히면 그대로 좁아진다."""
    assert RoutingConfig().coder_regex_patterns == []
    # 반대로 제외 패턴은 코드에 둔다(측정으로 정한 안전장치라 배포마다 흔들리면 안 된다).
    assert len(RoutingConfig().coder_exclude_patterns) > 0


def test_repo_root_is_cwd() -> None:
    """이 테스트 파일은 리포 루트 기준 상대경로로 설정을 읽는다(전제 확인)."""
    assert Path("config/nexus_config.pc.yaml").is_file()
