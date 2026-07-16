"""
Scout 모델 프로바이더 — llama.cpp CPU 모델 연동.

[이 파일의 역할]
Nexus는 두 개의 LLM을 역할별로 나눠 쓴다. 무거운 추론과 도구 실행은
Worker(Qwen3.5-27B, GPU)가 담당하고, 그 앞단에서 "무엇을 찾고 어떤 순서로
일할지"를 미리 정리하는 가벼운 정찰(Scout) 역할은 이 파일의 프로바이더가
맡는다. Scout는 CPU에서 도는 작은 모델(Qwen3.5-4B)이라 GPU VRAM을 전혀
쓰지 않으므로, 무거운 Worker와 동시에 상주시켜도 자원 충돌이 없다.

[핵심 구성요소]
  - ScoutModelProvider: LocalModelProvider를 상속한 Scout 전용 프로바이더.
    Scout 서버에 맞는 설정과 몇 가지 우회(override)를 얹는다.
  - create_scout_provider_if_available(): 서버가 살아있을 때만 프로바이더를
    만들어 주고, 없으면 None을 돌려 Worker 단독 모드로 넘어가게 하는 팩토리.

[히스토리 — v7.0 Phase 9.5, 2026-04-17 Qwen 소형 전환]
Scout(4B)는 파일 탐색과 계획 수립만 담당하는 작은 모델이다.
모델 변경: Gemma 4 E4B (llama.cpp tool_call 한계) → Qwen3.5-4B.
  이유: Worker(Qwen3.5-27B)와 같은 모델 패밀리로 맞추면 토크나이저,
  chat template, tool-call 문법이 일관되어 Scout → Worker 핸드오프
  호환성이 올라간다. (Scout가 뽑은 계획을 Worker가 그대로 이어받기 쉬워짐)

[구현 방식]
llama.cpp도 OpenAI 호환 API를 제공하므로, 통신 로직은 GPU용
LocalModelProvider와 완전히 동일하다. 그래서 새 HTTP 클라이언트를 짜지
않고 LocalModelProvider를 상속해 재사용하며, 아래 차이점만 덧입힌다.

[Worker와의 주요 차이점]
  - max_context_tokens: 4096 (Scout는 짧은 컨텍스트만 다룸)
  - max_output_tokens: 512 (계획서 정도의 짧은 출력만 냄)
  - 연결 실패 시 예외를 던지지 않고 None 반환 → Scout 없이도 동작(fallback)

작성자: 이현수 / 작성일: 2026-07-05
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from typing import Any

# Message/StreamEvent: 4-Tier 스트리밍 체인 전체에서 주고받는 표준 데이터 단위.
#   - Message: 대화 한 건(role + content) 표현용 Pydantic 모델
#   - StreamEvent: 토큰/도구 호출 등 스트림 조각을 담는 frozen 이벤트
from core.message import Message, StreamEvent

# Scout는 별도 백엔드가 아니라 LocalModelProvider를 상속해 재사용한다.
# (llama.cpp도 OpenAI 호환 API를 제공하므로 통신·SSE 파싱 로직이 완전히 동일)
from core.model.inference import LocalModelProvider, StructuredOutputSpec

# 모듈 전용 로거. 규칙상 "nexus.{모듈경로}" 네임스페이스를 쓴다.

logger = logging.getLogger("nexus.model.scout_provider")


class ScoutModelProvider(LocalModelProvider):
    """
    llama.cpp OpenAI 호환 API를 통한 Scout 모델 프로바이더.

    LocalModelProvider를 그대로 상속하므로, 상위 오케스트레이터 입장에서는
    Worker 프로바이더와 완전히 똑같은 인터페이스(stream, health_check 등)로
    보인다. 다만 생성자에서 Scout 서버에 맞는 값(짧은 컨텍스트/짧은 출력/짧은
    연결 타임아웃)을 채워 넣고, 아래 두 가지 Scout 전용 우회를 얹는다.

    [Scout 전용 오버라이드 1 — enable_thinking 강제 해제]
      stream()에서 enable_thinking 인자를 호출자가 뭘 넘기든 무조건 None으로
      바꿔 부모에 전달한다.
      이유: Qwen3.5-4B가 llama.cpp에서 enable_thinking=False를 받으면 빈
      <think></think> 블록 뒤에 거짓 tool_call 1개만 뱉고 28 토큰 만에 조기
      종료되는 버그가 있음(α 진단, 2026-04-21, tmp_scout_thinking 재현).
      None으로 두면 chat_template_kwargs 자체가 요청에서 빠져 llama.cpp
      기본 템플릿이 정상 동작한다. 이 버그는 Worker(27B)와 무관하므로
      Scout 경로에서만 플래그를 제거한다.

    [Scout 전용 오버라이드 2 — reasoning를 본문 텍스트로 병합]
      생성자에서 _include_reasoning_as_text=True를 켠다. Qwen3.5-4B는 실제로
      쓸모 있는 계획 내용의 상당 부분을 reasoning_content 쪽에 실어 보내는데,
      그걸 버리지 않고 본문(content)과 합쳐 TEXT_DELTA로 노출한다. (자세한
      설명은 __init__ 내부 주석 참고)
    """

    def __init__(
        self,
        base_url: str = "http://192.168.21.112:8003",
        api_key: str = "local-key",
        model_id: str = "qwen3.5-4b",
        max_context_tokens: int = 4096,
        max_output_tokens: int = 512,
        connect_timeout: float = 5.0,
        read_timeout: float = 60.0,
    ) -> None:
        """
        Scout 모델 프로바이더를 초기화한다.

        모든 인자는 기본값을 가지므로, 특별한 사정이 없으면 인자 없이
        ScoutModelProvider()로 만들면 된다. 실제 설정 주입은 부모
        LocalModelProvider.__init__()가 담당하고, 여기서는 Scout에 맞는
        기본값만 채워 넘긴다.

        Args:
            base_url: llama.cpp 서버 URL (CPU 노드, :8003 포트)
            api_key: API 키 (llama.cpp 실행 시 준 --api-key와 같아야 함)
            model_id: 모델 식별자 (로깅·요청 표시용)
            max_context_tokens: 최대 컨텍스트 (Scout는 4096으로 짧게)
            max_output_tokens: 최대 출력 토큰 (계획서만 내므로 512)
            connect_timeout: 연결 타임아웃. 짧게 잡아 서버가 죽었을 때
                빠르게 실패하고 Worker 단독 모드로 넘어가도록 한다.
            read_timeout: 읽기 타임아웃. CPU 추론은 느리므로 60초로 넉넉히.
        """
        super().__init__(
            base_url=base_url,
            api_key=api_key,
            model_id=model_id,
            max_context_tokens=max_context_tokens,
            max_output_tokens=max_output_tokens,
            connect_timeout=connect_timeout,
            read_timeout=read_timeout,
        )
        # Scout 전용 플래그 — reasoning_content를 본문과 합쳐 TEXT_DELTA로 낸다.
        # (이 플래그는 부모 LocalModelProvider의 스트림 파서가 읽어서 동작한다)
        #
        # 왜 필요한가: Qwen3.5-4B는 llama.cpp에서 Qwen chat template의
        # `<think>` 블록을 응답 본문(content)과 분리해 reasoning_content라는
        # 별도 필드로 보낸다. 그런데 Scout가 만드는 "4섹션 마크다운 리포트"의
        # 알맹이 상당 부분이 하필 그 reasoning 쪽에 실려 온다.
        # 이 내용은 뒤이어 Worker에게 그대로 넘겨야 하는 핵심 산출물이므로,
        # 버리지 않고 content와 병합해 일반 텍스트(TEXT_DELTA)로 노출시킨다.
        self._include_reasoning_as_text = True

        logger.info(
            "ScoutModelProvider 초기화: %s (ctx=%d, max_out=%d)",
            base_url,
            max_context_tokens,
            max_output_tokens,
        )

    async def stream(
        self,
        messages: list[Message],
        system_prompt: str,
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        stop_sequences: list[str] | None = None,
        model_override: str | None = None,
        enable_thinking: bool | None = False,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        structured_output: StructuredOutputSpec | None = None,
        n: int = 1,
        force_tool_choice: str | None = None,
    ) -> AsyncGenerator[StreamEvent, None]:
        """
        Scout 전용 stream — enable_thinking을 None으로 강제한다.

        시그니처는 부모 LocalModelProvider.stream()과 동일하게 맞춰, 상위
        오케스트레이터가 프로바이더 종류를 신경 쓰지 않고 호출할 수 있게 한다.

        핵심 동작: 호출자가 enable_thinking에 무엇을 넘기든(True/False/None)
        전부 무시하고 None으로 바꿔 부모에 전달한다. None이 되면 부모가
        chat_template_kwargs 자체를 요청 본문에서 빼므로, llama.cpp의 기본
        Qwen3.5 chat template이 그대로 동작해 정상 길이의 응답(마크다운 4섹션
        리포트)을 얻을 수 있다. (배경은 클래스 docstring의 오버라이드 1 참고)

        Args:
            messages: 대화 이력(Message 리스트)
            system_prompt: 시스템 프롬프트 문자열
            tools: 노출할 도구 스키마 목록(없으면 None)
            temperature~presence_penalty: 샘플링 파라미터. 여기서는 손대지
                않고 부모로 그대로 넘긴다(passthrough).
            enable_thinking: 무시됨 — 항상 None으로 덮어써 전달한다.
            structured_output: Scout(4B/llama.cpp) 경로에서는 미사용. 상위
                오케스트레이터가 프로바이더 종류를 구분하지 않도록 시그니처
                정합성만 맞추고, 값은 그대로 부모에 passthrough한다.
        Yields:
            StreamEvent: 부모가 만들어 흘려보내는 스트림 이벤트를 그대로 중계.
        """
        # 부모 stream()을 그대로 호출하되 enable_thinking만 None으로 강제 주입한다.
        # 부모가 만들어 yield하는 StreamEvent를 한 건씩 그대로 다시 흘려보내는
        # passthrough 패턴 — 4-Tier 체인을 우회하지 않고 부모 제너레이터를 감싼다.
        # (여기서 이벤트를 가로채 수정하지 않는다: StreamEvent는 frozen이며,
        #  Scout 프로바이더가 할 일은 인자 보정뿐 스트림 변형이 아니기 때문)
        async for ev in super().stream(
            messages=messages,
            system_prompt=system_prompt,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
            stop_sequences=stop_sequences,
            model_override=model_override,
            enable_thinking=None,
            # 샘플링 파라미터를 그대로 상위로 전달(passthrough) — degeneration 방지.
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            # Scout는 구조화 출력 대상이 아니지만 시그니처 정합을 위해 그대로 전달.
            structured_output=structured_output,
            # SC 표본 수도 시그니처 정합을 위해 그대로 전달(Scout는 SC 대상 아님 — 통상 1).
            n=n,
            # guided 재시도 강제 도구도 시그니처 정합을 위해 그대로 전달(passthrough).
            force_tool_choice=force_tool_choice,
        ):
            yield ev


async def create_scout_provider_if_available(
    base_url: str = "http://192.168.21.112:8003",
    api_key: str = "local-key",
) -> ScoutModelProvider | None:
    """
    Scout 서버가 실행 중이면 ScoutModelProvider를 생성하고,
    아니면 None을 반환한다 (fallback 지원 팩토리).

    왜 이 함수인가: llama.cpp Scout 서버가 아직 설치/실행되지 않았거나
    잠깐 죽어 있을 수 있다. 그래도 Nexus는 Worker 단독 모드로 계속
    돌아가야 하므로, 여기서 연결 실패를 예외로 터뜨리지 않고 None으로
    조용히 흡수한다. 호출부는 반환값이 None인지만 보고 Scout 사용
    여부를 결정하면 된다.

    동작 순서:
      1) 프로바이더를 일단 만든다(생성 자체는 네트워크를 타지 않음).
      2) health_check()로 서버가 실제로 응답하는지 확인한다.
      3) 정상이면 provider를, 응답 없음/예외면 None을 반환한다.

    Args:
        base_url: Scout llama.cpp 서버 URL (CPU 노드, :8003)
        api_key: API 키
    Returns:
        ScoutModelProvider | None: 서버가 살아있으면 프로바이더, 아니면 None.
    """
    # 프로바이더 객체 생성 — 여기서는 아직 서버에 접속하지 않는다(설정만 준비).
    provider = ScoutModelProvider(base_url=base_url, api_key=api_key)
    try:
        # 실제 접속 확인은 health_check()에서 이뤄진다(짧은 연결 타임아웃 적용).
        healthy = await provider.health_check()
        if healthy:
            logger.info("Scout 서버 연결 성공: %s", base_url)
            return provider
        # 서버가 떠 있긴 하나 정상 응답이 아닌 경우 → Scout 미사용으로 처리.
        logger.warning("Scout 서버 응답 없음: %s", base_url)
        return None
    except Exception as e:
        # 접속 거부·타임아웃 등 모든 예외를 여기서 흡수한다. Scout는 선택
        # 요소이므로 실패해도 앱을 죽이지 않고 Worker 단독 모드로 넘어간다.
        logger.warning("Scout 서버 연결 실패 (Worker 단독 모드로 전환): %s", e)
        return None
