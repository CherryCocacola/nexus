# 구현 요청이 TOOL 로 분류되는지, 지식 질의가 새지 않는지 고정한다.
"""
`tool_regex_patterns` 회귀 테스트 (2026-08-19 신설).

배경 — 실측으로 드러난 결함:
    "대화창 구현해줘" 가 KNOWLEDGE 로 분류됐다. tool_keywords 가 전부 파일·프로젝트
    명사(파일/폴더/디렉토리...)라, 파일을 명시하지 않은 구현 요청이 통째로 새어
    지식 질의로 갔다. 그러면 셋이 한꺼번에 나빠진다.
      ① 도구를 안 쓰고 코드를 화면에만 뿌린다(사용자가 복사·붙여넣기)
      ② 사내 문서 RAG 가 불필요하게 주입된다
      ③ max_tokens 가 4096 으로 묶인다(TOOL 은 8192) — 긴 코드가 잘린다

이 테스트가 지키는 것은 둘이다. 구현 요청을 놓치지 않을 것, 그리고 **지식 질의를
TOOL 로 끌어가지 않을 것**. 후자가 더 중요하다 — 오탐이 나면 RAG 가 죽어 사내 문서
질의가 조용히 나빠지는데, 그건 눈에 잘 안 띈다.
"""

from __future__ import annotations

import pytest

from core.config import RoutingConfig
from core.orchestrator.routing import RoutingResolver

# 운영 설정과 같은 패턴(config/*.yaml 의 routing.tool_regex_patterns 와 일치시킨다).
_PATTERNS = [
    r"(구현|개발|리팩터\S*|리팩토\S*|디버깅|디버그)\s*(해|하|중|좀)",
    r"(함수|클래스|모듈|스크립트|컴포넌트|엔드포인트|API|UI|화면|페이지|서버|테스트|기능|창)"
    r"[^\n]{0,15}?(만들|작성|생성|추가|수정|짜)",
    r"코드\s*\S{0,6}\s*(짜|작성|만들|생성)",
]


def _resolver() -> RoutingResolver:
    return RoutingResolver(RoutingConfig(tool_regex_patterns=_PATTERNS))


@pytest.mark.parametrize(
    "text",
    [
        "대화창 구현해줘",
        "gb10 에 설치된 nova 와 대화하고 싶은데, 대화창 구현해줄 수 있을까",
        "로그인 기능 만들어 줄 수 있어?",
        "이 함수 리팩터링 해줄 수 있을까",
        "이 클래스 리팩토링 해줘",
        "테스트 코드 작성해줘",
        "React 컴포넌트 하나 만들어줘",
        "REST API 엔드포인트 추가해줘",
        "채팅 UI 개발해줘",
        "이 버그 디버깅 좀 해줘",
        # ★2026-08-19 2차 — 명사와 동사 사이가 벌어진 문장을 놓쳤다.
        #   "창을 chat.js 로 만들어줘" 처럼 중간에 단어가 끼면 좁은 간격 패턴이 못 넘는다.
        "WebSocket 대화창을 chat.js 로 만들어줘. 메시지 타입별 분기를 모두 처리해줘",
        "로그인 화면을 별도 파일로 분리해서 만들어줘",
        "사용자 목록 API 를 FastAPI 로 새로 작성해줘",
    ],
)
def test_implementation_request_routes_to_tool(text: str) -> None:
    """구현 요청은 질문형이든 명령형이든 TOOL 이어야 한다."""
    assert _resolver().resolve(text, None).query_class == "TOOL", text


@pytest.mark.parametrize(
    "text",
    [
        "김치는 어떻게 만들어?",
        "빵 만드는 법 알려줘",
        "회사 연차 규정 알려줘",
        "베토벤 교향곡 9번에 대해 설명해줘",
        "광합성 과정을 설명해줘",
        "조선시대 과거제도가 뭐야?",
        # ★함정 — '구현'이 지식 맥락에 등장한다. 여기서 새면 사내 문서 RAG 가 죽는다.
        "블록체인 합의 알고리즘 구현 원리를 설명해줘",
        "이 정책은 어떻게 구현되나요?",
        "OAuth 인증 방식의 차이를 설명해줘",
        "REST API 가 무엇인지 설명해줘",
        "함수형 프로그래밍의 장점을 알려줘",
        "테스트 주도 개발이 뭐야?",
    ],
)
def test_knowledge_question_stays_knowledge(text: str) -> None:
    """지식 질의가 TOOL 로 새면 안 된다(RAG 가 조용히 꺼진다)."""
    assert _resolver().resolve(text, None).query_class == "KNOWLEDGE", text


def test_patterns_match_shipped_config() -> None:
    """테스트의 패턴이 실제 배포 설정과 어긋나지 않게 고정한다.

    설정만 고치고 테스트를 안 고치면(또는 그 반대면) 이 테스트가 통과해도
    운영은 다르게 동작한다. 그래서 파일을 직접 읽어 대조한다.
    """
    import yaml

    for path in (
        "config/nexus_config.yaml",
        "config/nexus_config.pc.yaml",
        "config/nexus_config.112.yaml",
    ):
        with open(path, encoding="utf-8") as f:
            shipped = yaml.safe_load(f)["routing"]["tool_regex_patterns"]
        assert shipped == _PATTERNS, f"{path} 의 패턴이 테스트와 다르다"
