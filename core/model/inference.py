"""
추론 엔진 — ModelProvider ABC + LocalModelProvider 구현.

이 파일은 Nexus에서 "실제로 LLM에게 말을 거는" 유일한 통로다. 상위 계층
(query_loop, QueryEngine, 도구 시스템)은 여기 정의된 ModelProvider 인터페이스만
바라보므로, 뒤에 어떤 모델(Qwen/ExaOne/OpenAI)이 붙든 상위 코드는 바뀌지 않는다.

구성 요소:
  - ModelConfig       : 모델 1개의 런타임 설정(컨텍스트 상한, 출력 상한 등) 데이터클래스
  - ModelProvider(ABC): 모든 모델 백엔드가 구현해야 하는 추상 인터페이스
                        (stream / embed / rerank / health_check / count_tokens / get_config)
  - LocalModelProvider: 에어갭용 구현체. LAN 안의 vLLM(OpenAI 호환 API)과 통신한다.

핵심 흐름(가장 중요한 stream() 기준):
  Nexus Message[] → OpenAI messages[] → HTTP POST(SSE) → SSE chunk 파싱 → StreamEvent yield

즉 상위 계층이 쓰는 도메인 타입(Message)을 vLLM이 이해하는 OpenAI 형식으로 바꿔
요청하고, 서버가 조각조각 흘려보내는 SSE 응답을 다시 Nexus의 StreamEvent로 되돌려
상위로 실시간 전달한다.

이 모듈은 4-Tier AsyncGenerator 체인에서 Tier 3~4에 해당한다:
  Tier 3: stream() — SSE 스트림 파싱 및 StreamEvent 변환
  Tier 4: httpx 클라이언트로의 실제 HTTP 왕복
    (주의: Tier 4 with_retry는 아직 이 왕복부에 배선되지 않았다. 현재 일시 오류는
     컨텍스트 초과 재시도 루프 + 연결 오류의 ERROR 이벤트 변환으로 처리하며,
     스트림 정지 재시도는 Tier 2 query_loop이 담당한다. architecture.md P1 구현 현황 참조.)

의존 방향: 이 파일은 core.message의 도메인 타입만 의존하며, 외부로는 httpx로 LAN의
vLLM 서버에만 접근한다(에어갭 규칙 준수 — 외부 인터넷 호출 없음).

작성자: 이현수 / 작성일: 2026-07-05
"""

from __future__ import annotations

# 표준 라이브러리
import json  # SSE 청크(JSON 문자열) 파싱 및 tool_calls arguments 직렬화에 사용
import logging
import time  # latency 측정용 단조 시계(time.monotonic)
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import Any

# 외부 라이브러리 — vLLM 서버와 HTTP/SSE로 통신하는 비동기 클라이언트
import httpx

# Pydantic v2 — 구조화 출력 스펙(StructuredOutputSpec)을 불변(frozen) 모델로 정의한다.
# (프로젝트 규칙 P5: 데이터 구조는 Pydantic BaseModel 또는 frozen dataclass)
from pydantic import BaseModel, ConfigDict

# 내부 도메인 모델 — 4-Tier 체인 전체에서 공유하는 데이터 타입
# (StreamEvent/StopReason 등은 frozen이므로 생성 후 수정하지 않는다)
from core.message import (
    Message,
    StopReason,
    StreamEvent,
    StreamEventType,
    TextBlock,
    TokenUsage,
    ToolUseBlock,
)

# 모듈 전용 로거 — JSONL 로깅 규칙에 맞춰 "nexus.{module}" 네이밍을 따른다
logger = logging.getLogger("nexus.model.inference")


# ─────────────────────────────────────────────
# 모델 설정 (단일 모델의 런타임 설정)
# ─────────────────────────────────────────────
@dataclass
class ModelConfig:
    """
    모델별 런타임 설정.
    GPU 티어에 따라 자동 결정되거나, 설정 파일에서 로드된다.
    """

    model_id: str  # vLLM에서 사용하는 모델 식별자
    max_context_tokens: int = 8192  # 입력+출력 합산 상한 (모델 컨텍스트 윈도우)
    max_output_tokens: int = 4096  # 한 번에 생성할 수 있는 응답 토큰 상한
    default_temperature: float = 0.7
    supports_tool_calling: bool = True  # vLLM 네이티브 tool calling 지원 여부
    supports_streaming: bool = True
    stop_sequences: list[str] = field(default_factory=list)  # 생성 중단 토큰 목록
    fallback_model_id: str | None = None  # OOM 시 대체 모델
    # 로컬 추론이라 토큰 과금이 없다. 사용량 추적 코드와의 호환을 위해 필드만 유지.
    cost_per_input_token: float = 0.0  # 로컬: 0 (전기 비용은 별도)
    cost_per_output_token: float = 0.0


# ─────────────────────────────────────────────
# 구조화 출력 스펙 (vLLM guided decoding)
# ─────────────────────────────────────────────
class StructuredOutputSpec(BaseModel):
    """
    구조화 출력(guided decoding) 요청 스펙 — 불변(frozen) 객체.

    stream() 호출자가 이 객체를 넘기면 Tier 3(LocalModelProvider.stream)가
    vLLM payload에 response_format(json_schema) 형태로 주입해, 모델이 지정한
    JSON Schema를 "토큰 생성 단계에서" 강제로 따르게 한다(사후 검증이 아니라
    디코딩 시점 예방). 용도: (a) 외부 OpenAI 클라이언트(AgentHub)의
    response_format 수용, (c) 사실 필드 형식 안정화.

    왜 dict가 아니라 Pydantic 모델인가:
      스키마 dict만 넘기면 이름(name)·strict 여부를 별도 인자로 또 늘려야 한다.
      스펙 객체 하나로 세 정보를 함께 운반하면 시그니처가 단순해지고(P5 준수),
      frozen이라 생성 후 변형될 수 없어 호출 간 오염 위험도 없다.
    """

    model_config = ConfigDict(frozen=True)

    json_schema: dict[str, Any]  # JSON Schema (draft 2020-12 부분집합)
    name: str = "nexus_structured"  # OpenAI json_schema.name 필드
    strict: bool = True  # vLLM strict 모드 (스키마 완전 준수 강제)


# ─────────────────────────────────────────────
# ModelProvider ABC (추상 인터페이스)
# ─────────────────────────────────────────────
class ModelProvider(ABC):
    """
    LLM 추상 인터페이스.
    Claude Code의 ApiClient에 대응한다.

    이 인터페이스만 구현하면 QueryEngine, Tool System, Agent System 전체가
    구체적인 모델(Qwen, ExaOne, OpenAI 등)과 독립적으로 동작한다.

    왜 ABC인가: 향후 다른 모델 백엔드(직접 torch, TGI 등)를
    추가할 때 이 인터페이스만 구현하면 된다.
    """

    @abstractmethod
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
        모델에 스트리밍 요청을 보낸다.

        Args:
            n: 자기일관성(Self-Consistency)용 표본 수(Point 4.3). 기본 1이면 기존
                단일 경로 그대로다(무회귀). n>1이면 vLLM `n` 파라미터로 한 요청 안에서
                프롬프트 KV cache를 공유하며 N개 시퀀스를 생성하고(prefill 1회 +
                디코드만 N배), 라이브 TEXT_DELTA 대신 표본별 완성 텍스트를 SC_CANDIDATE
                이벤트(text + sample_index)로 올려보낸다. 상위(query_loop)가 이를
                버퍼링해 다수결 합의를 낸다. 어떤 표본에 tool_calls가 섞이면 SC를
                포기하고 choice 0만으로 기존 단일 스트림처럼 폴백한다(설계 §2.3).
            model_override: 호출 시점에 기본 model_id를 덮어쓴다.
                v7.0 Part 2.5 쿼리 라우팅에서 LoRA ON/OFF를 런타임 전환하기 위해 사용.
                None이면 프로바이더의 model_id를 그대로 쓴다.
            top_p: nucleus 샘플링 임계. 1.0이면 비활성(전체 분포 사용).
            repetition_penalty: 반복 토큰 페널티. 1.0이 중립(비활성).
                동일 문장 무한 반복(degeneration)을 억제한다.
            frequency_penalty: 빈도 페널티. 0.0이면 비활성.
            presence_penalty: 등장 페널티. 0.0이면 비활성.
            structured_output: 구조화 출력(guided decoding) 스펙. None이면 일반
                생성(무회귀). 지정되면 payload에 response_format을 주입해 모델이
                JSON Schema를 강제로 따르게 한다. tools와는 상호 배타(구현체에서
                ValueError). 자세한 규칙은 LocalModelProvider.stream() 참조.
            enable_thinking: Qwen3.5 chat_template_kwargs.enable_thinking 인자.
                - False(기본): 빈 <think></think> 블록 주입 — Worker(27B)에서 내부 독백이
                  답변을 잡아먹는 현상 회피 목적.
                - True: thinking 블록 생성 허용.
                - None: chat_template_kwargs 자체를 요청에서 생략 (llama.cpp/vLLM 기본
                  동작 사용). Scout(Qwen3.5-4B)가 False에서 tool_call 조기 종료되는
                  이슈(α 진단, 2026-04-21)를 피하기 위해 Scout 전용 값으로 추가됨.

        Yields:
            StreamEvent — text_delta, tool_use, message_stop 등
        """
        ...
        yield  # type: ignore  # ABC에서 AsyncGenerator yield 필요

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """텍스트 임베딩을 생성한다."""
        ...

    async def rerank(self, query: str, documents: list[str]) -> list[float]:
        """(query, 각 document) 쌍의 관련도 점수(0~1)를 documents와 같은 순서로 반환한다.

        왜 @abstractmethod가 아니라 기본 구현(NotImplementedError)인가:
          기존 ModelProvider 하위 구현체(테스트용 Mock 등)가 rerank를 몰라도
          인스턴스화가 깨지지 않게 하려는 하위 호환 조치다. 리랭킹을 지원하지
          않는 백엔드에서 호출되면 예외가 나고, 호출자(KnowledgeRetriever)는
          이를 fail-safe로 받아 기존 벡터순 경로로 폴백한다.
        """
        raise NotImplementedError("이 ModelProvider는 rerank를 지원하지 않습니다.")

    @abstractmethod
    async def health_check(self) -> bool:
        """모델 서버 가용성을 확인한다."""
        ...

    @abstractmethod
    async def count_tokens(self, messages: list[Message]) -> int:
        """메시지 목록의 토큰 수를 추정한다."""
        ...

    @abstractmethod
    def get_config(self) -> ModelConfig:
        """현재 모델 설정을 반환한다."""
        ...


# ─────────────────────────────────────────────
# LocalModelProvider (에어갭 전용 — vLLM OpenAI 호환)
# ─────────────────────────────────────────────
class LocalModelProvider(ModelProvider):
    """
    vLLM OpenAI 호환 API를 통한 로컬 모델 프로바이더.

    Claude Code의 Anthropic SDK 클라이언트에 대응하지만,
    에어갭 환경에서 LAN 내 vLLM 서버와 통신한다.

    기능:
      - /v1/chat/completions SSE 스트리밍
      - Nexus Message → OpenAI message 자동 변환
      - tool schema → OpenAI function_calling 변환
      - 연결 실패 시 graceful error event yield
      - vLLM 네이티브 tool calling 지원
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_key: str = "local-key",
        model_id: str = "qwen3.5-27b",
        max_context_tokens: int = 8192,
        max_output_tokens: int = 4096,
        fallback_model_id: str | None = None,
        embedding_model_id: str = "multilingual-e5-large",
        embedding_base_url: str | None = None,
        connect_timeout: float = 10.0,
        read_timeout: float = 300.0,
        structured_output_injection_mode: str = "response_format",
    ):
        """
        vLLM 서버 연결 정보와 httpx 클라이언트를 준비한다.

        Args:
            base_url: vLLM 서버 주소 (에어갭이므로 LAN/localhost만 허용)
            api_key: vLLM --api-key와 동일해야 인증을 통과한다
            model_id: 기본으로 사용할 모델 식별자 (stream에서 override 가능)
            embedding_base_url: 임베딩 전용 서버가 따로 있을 때 지정. None이면
                채팅과 동일한 base_url을 재사용한다.
            connect_timeout: 연결 수립 제한 시간(초)
            read_timeout: 응답 본문 읽기 제한 시간(초). GPU 추론이 길 수 있어 넉넉히 둔다.
            structured_output_injection_mode: 구조화 출력 payload 주입 형태.
                "response_format"(기본, OpenAI 표준 — Phase 0 B200 실측으로 확정) 또는
                "structured_outputs"(vLLM 신형 확장, 표준형 미지원 시 폴백). 값 자체는
                config/nexus_config.yaml#structured_output.injection_mode가 단일
                소스이며, bootstrap이 그 값을 여기로 주입한다(안티패턴 #4 하드코딩 회피).
        """
        # 끝의 슬래시를 제거해 "{base_url}/v1/..." 조합 시 // 가 생기지 않게 한다
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model_id = model_id
        self.embedding_model_id = embedding_model_id
        # 임베딩 서버가 별도 포트/인스턴스인 경우 분리
        # 없으면 base_url과 동일한 서버를 사용한다
        self._embedding_base_url = (
            embedding_base_url.rstrip("/") if embedding_base_url else self.base_url
        )
        # 구조화 출력 주입 형태 — _build_response_format()이 참조한다.
        # 기본 "response_format"은 Phase 0 실측(B200 vLLM 0.24)으로 확정된 값이며,
        # 실측에서 표준형이 거부되는 티어를 만나면 config로 "structured_outputs"로
        # 전환한다(코드 무수정 대응, 리스크 R1).
        self._structured_output_injection_mode = structured_output_injection_mode

        self._config = ModelConfig(
            model_id=model_id,
            max_context_tokens=max_context_tokens,
            max_output_tokens=max_output_tokens,
            fallback_model_id=fallback_model_id,
            supports_tool_calling=True,
        )

        # httpx 비동기 클라이언트 — 커넥션 풀링 + 단계별 타임아웃 설정.
        # 클라이언트를 1개만 만들어 재사용하면 매 요청마다 TCP 핸드셰이크를
        # 반복하지 않아도 되어 LAN 추론에서 지연이 줄어든다.
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(
                connect=connect_timeout,  # 연결 수립 단계 제한
                read=read_timeout,  # 응답 읽기 단계 제한 (스트리밍 본문)
                write=30.0,  # 요청 본문 전송 제한
                pool=10.0,  # 풀에서 커넥션을 얻기까지의 대기 제한
            ),
            limits=httpx.Limits(
                max_connections=20,  # 동시에 열 수 있는 총 커넥션 수
                max_keepalive_connections=10,  # 재사용 위해 유지할 idle 커넥션 수
            ),
        )

        # 요청 통계 — stats 프로퍼티로 노출해 운영 중 latency/에러율 모니터링에 쓴다
        self._request_count: int = 0
        self._total_latency: float = 0.0
        self._error_count: int = 0

    # ─── 핵심: stream() ───

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
        vLLM /v1/chat/completions SSE 스트리밍.

        변환 흐름:
          Nexus Message[] → OpenAI messages[] → SSE chunks → StreamEvent yield

        n>1(자기일관성, Point 4.3)이면 payload에 "n"을 실어 한 요청으로 N개 시퀀스를
        생성하고, choices[i].index로 표본을 디멀티플렉스한다. 이 경우 라이브 TEXT_DELTA는
        내보내지 않고 표본별 완성 텍스트를 SC_CANDIDATE 이벤트로 올려보낸다(설계 §2.3).
        n=1(기본)이면 아래 코드 경로가 기존과 완전히 동일하다(무회귀).

        model_override/enable_thinking은 v7.0 Part 2.5 쿼리 라우팅에서 LoRA
        ON/OFF 및 thinking 모드를 런타임 전환하기 위해 사용된다.

        enable_thinking=None이면 chat_template_kwargs 자체를 요청에서 빼서
        llama.cpp/vLLM 기본 동작을 따르도록 한다(Scout 전용 경로).
        """
        # 통계용 카운터 증가 + 지연 측정 시작점 기록
        self._request_count += 1
        start_time = time.monotonic()

        # 라우팅에서 지정한 모델 ID가 있으면 기본값을 덮어쓴다.
        # 예: 일반 지식 질의 → model_override="qwen3.5-27b" (LoRA OFF)
        # 예: 도구 호출 질의 → model_override=None 또는 "nexus-phase3"
        active_model_id = model_override or self.model_id

        # 메시지 변환: Nexus → OpenAI 형식
        oai_messages = self._convert_messages(messages, system_prompt)

        # 요청 페이로드 구성
        # chat_template_kwargs로 Qwen3.5 thinking 모드를 제어한다.
        # 왜 False가 기본인가: Qwen3.5의 기본 chat template은 <think> 블록을 강제
        # 삽입한다. Worker(27B)의 도구 호출 흐름에서 내부 독백만 뱉고 답변을
        # 누락하는 사례가 관찰되어 enable_thinking=False로 빈 <think></think>를
        # 주입해 답변 생성 모드로 바로 진입시킨다.
        # 왜 None 옵션이 있는가: Scout(Qwen3.5-4B on llama.cpp)가
        # enable_thinking=False에서 28 토큰 조기 종료(α 진단, 2026-04-21) —
        # Scout 전용 경로에서는 chat_template_kwargs를 아예 보내지 않고
        # 모델의 기본 동작에 맡기는 것이 안전하다.
        payload: dict[str, Any] = {
            "model": active_model_id,
            "messages": oai_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},  # vLLM 사용량 추적
            # ── 샘플링 파라미터 (degeneration/무한 반복 방지) ──────────────
            # 왜 항상 top-level에 넣는가:
            #   이 코드는 openai SDK가 아니라 httpx raw POST(json=payload)로
            #   /v1/chat/completions에 직접 보낸다. 따라서 vLLM 확장 파라미터인
            #   repetition_penalty도 extra_body가 아니라 payload 최상위에 그대로
            #   넣어야 vLLM이 인식한다(SDK라면 extra_body가 필요하지만 여기선 아님).
            # 왜 조건부 주입 없이 항상 넣는가:
            #   기본값(top_p=1.0/repetition_penalty=1.0/freq=0.0/presence=0.0)은
            #   전부 "비활성"값이라 vLLM이 사실상 무시한다. 분기 없이 항상 보내면
            #   코드가 단순해지고 누락(=이번 버그의 원인)을 원천 차단한다.
            "top_p": top_p,
            "repetition_penalty": repetition_penalty,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
        }
        if enable_thinking is not None:
            # bool(True/False)일 때만 명시적으로 주입 — None이면 완전 생략
            payload["chat_template_kwargs"] = {"enable_thinking": enable_thinking}

        # ── 자기일관성(Self-Consistency) 표본 수 (Point 4.3) ─────────────────
        # n>1일 때만 "n"을 주입한다. n=1이면 vLLM 기본값과 같아 굳이 넣지 않아
        # payload가 기존과 바이트 단위로 동일하게 유지된다(무회귀). n개 시퀀스는
        # 프롬프트 KV cache를 공유하므로 prefill은 1회, 디코드만 N배 든다(설계 §2.1).
        sc_mode = n > 1
        if sc_mode:
            payload["n"] = n

        if stop_sequences:
            payload["stop"] = stop_sequences

        # 도구 스키마 변환: Nexus → OpenAI function_calling
        if tools:
            payload["tools"] = [self._convert_tool_schema(t) for t in tools]
            # force_tool_choice가 지정되면 해당 함수로 tool_choice를 '강제'한다.
            # 왜: hermes 파서 + tool_choice="auto"는 인자를 자유형식으로 통과시켜
            # 모델이 깨진 JSON(미이스케이프 따옴표/개행, 닫는 구조 누락)을 내도 그대로
            # 흘러간다. 반면 named tool_choice면 vLLM이 그 도구의 parameters 스키마로
            # guided decoding을 적용해 '유효 JSON'을 보장한다(B200 실서버 확증됨).
            # 상위(query_loop)가 인자 파싱 실패를 감지했을 때만 이 경로로 재시도한다.
            if force_tool_choice:
                payload["tool_choice"] = {
                    "type": "function",
                    "function": {"name": force_tool_choice},
                }
            else:
                payload["tool_choice"] = "auto"

        # ── 구조화 출력 (guided decoding) 주입 ──────────────────────────────
        # 왜 도구 블록 '직후'인가: tools와 response_format의 동시 사용은 vLLM에서
        # 의미가 충돌한다(도구 호출 문법 vs 응답 본문 문법). 같은 자리에서 상호
        # 배타를 검사해 fail-closed로 즉시 거부하면 조용한 오동작을 막는다.
        # 왜 top-level 주입인가: 이 코드는 openai SDK가 아니라 httpx raw
        # POST(json=payload)라, vLLM 확장/OpenAI 표준 필드 모두 extra_body 래핑
        # 없이 payload 최상위에 넣어야 서버가 인식한다(repetition_penalty와 동일 규칙).
        if structured_output is not None:
            if tools:
                raise ValueError(
                    "structured_output과 tools는 동시에 사용할 수 없습니다 "
                    "(guided decoding은 응답 본문 문법을 강제하므로 tool_calls "
                    "생성과 충돌). 호출자가 둘 중 하나만 지정해야 합니다."
                )
            # 주입 형태에 따라 top-level 키가 다르므로(표준=response_format,
            # 폴백=structured_outputs) 헬퍼가 만든 payload 조각을 통째로 병합한다.
            payload.update(self._build_response_format(structured_output))
            # thinking 블록(<think>…</think>)은 JSON 문법을 위반하므로, 구조화 출력
            # 모드에서는 호출자가 넘긴 enable_thinking 값을 무시하고 무조건 끈다.
            # (위에서 이미 세팅됐을 수 있는 chat_template_kwargs를 여기서 덮어쓴다.)
            payload["chat_template_kwargs"] = {"enable_thinking": False}
            logger.debug(
                "구조화 출력 활성 — enable_thinking을 False로 강제 (mode=%s, name=%s)",
                self._structured_output_injection_mode,
                structured_output.name,
            )

        # vLLM 인증 헤더 — api_key가 서버 --api-key와 일치해야 401을 피한다
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        # MESSAGE_START 이벤트 — 실제 호출된 model_id를 전달(라우팅 추적용)
        # 상위 Tier(query_loop)는 이 이벤트로 "어떤 모델이 응답을 시작했는지" 안다
        yield StreamEvent(
            type=StreamEventType.MESSAGE_START,
            model_id=active_model_id,
        )

        # 컨텍스트 초과 시 max_tokens를 줄여서 자동 재시도
        # 왜 여기서 하는가: vLLM의 실제 토큰 카운트는 BPE 기반이라
        # 사전 추정이 부정확하다. 에러 발생 후 정확한 input_tokens 값으로 재계산.
        max_retries_for_context = 3
        current_max_tokens = payload["max_tokens"]

        # 컨텍스트 초과(400) 시에만 max_tokens를 줄여 다시 도는 재시도 루프.
        # 정상 흐름이면 첫 번째 attempt에서 [DONE]을 받고 return 하므로 1회만 돈다.
        for attempt in range(max_retries_for_context):
            payload["max_tokens"] = current_max_tokens

            # ── 이번 attempt의 누적 상태 (attempt마다 새로 초기화) ──
            # tool_calls는 SSE 청크에 쪼개져 오므로 index별로 모아 합친다
            accumulated_tool_calls: dict[int, dict[str, Any]] = {}
            total_usage = TokenUsage()
            context_exceeded = False  # 이번 attempt가 컨텍스트 초과로 끝났는지 표시
            # finish_reason 이후 usage 청크를 기다리기 위한 변수
            # vLLM은 finish_reason 청크 → usage 청크 → [DONE] 순으로 전송한다.
            pending_stop: StopReason | None = None
            # 누적 tool_calls를 finalize(TOOL_USE_STOP 방출)했는지 표시. finish_reason
            # 청크 또는 [DONE] 중 먼저 오는 곳에서 정확히 한 번만 finalize하기 위한 가드.
            # (강제 tool_choice에서는 finish_reason="stop"으로 와도 tool_calls가 있으므로,
            #  finish_reason 문자열이 아니라 이 플래그로 중복 없이 finalize한다.)
            tool_calls_finalized = False

            # ── 자기일관성(SC) 전용 상태 (n>1일 때만 사용) ──────────────────
            # 표본(choice) 인덱스별 텍스트 조각을 버퍼링한다. 라이브 TEXT_DELTA는
            # 내보내지 않고(§2.3), [DONE] 시점에 SC_CANDIDATE로 한꺼번에 올려보낸다.
            # 어떤 표본이든 tool_calls가 나오면 sc_tool_seen을 세워 SC를 포기하고
            # choice 0만으로 단일 스트림처럼 폴백한다(설계 §2.3).
            sc_candidate_texts: dict[int, list[str]] = {}
            sc_tool_seen = False

            try:
                async with self._client.stream(
                    "POST",
                    f"{self.base_url}/v1/chat/completions",
                    json=payload,
                    headers=headers,
                ) as response:
                    if response.status_code != 200:
                        error_body = ""
                        async for chunk in response.aiter_bytes():
                            error_body += chunk.decode("utf-8", errors="replace")

                        # 컨텍스트 초과 에러 감지 → max_tokens 줄여서 재시도
                        if response.status_code == 400 and "context length" in error_body:
                            import re

                            m = re.search(r"contains at least (\d+) input", error_body)
                            if m:
                                actual_input = int(m.group(1))
                                # 여유 마진 300 — vLLM 재시도 시 input이 100 토큰 가량
                                # 증가하는 현상이 관찰됨(원인 미상, 아마도 prefix cache
                                # 또는 chat_template 동적 요소). 마진을 넉넉히 두어
                                # 재시도 2회까지 견딜 수 있게 한다.
                                new_max = self._config.max_context_tokens - actual_input - 300
                                # 입력만으로 컨텍스트를 초과하면 재시도 불가
                                if new_max < 256:
                                    yield StreamEvent(
                                        type=StreamEventType.ERROR,
                                        error_code="CONTEXT_OVERFLOW",
                                        message=(
                                            f"입력({actual_input} 토큰)이 "
                                            f"컨텍스트({self._config.max_context_tokens})를 "
                                            f"거의 다 사용하여 응답을 생성할 수 없습니다. "
                                            f"더 짧은 내용으로 다시 시도해 주세요."
                                        ),
                                    )
                                    return
                                current_max_tokens = new_max
                                logger.warning(
                                    "컨텍스트 초과 → 재시도 %d/%d: "
                                    "input=%d, max_tokens=%d→%d",
                                    attempt + 1, max_retries_for_context,
                                    actual_input, payload["max_tokens"],
                                    current_max_tokens,
                                )
                                context_exceeded = True
                                continue

                        yield StreamEvent(
                            type=StreamEventType.ERROR,
                            error_code=f"HTTP_{response.status_code}",
                            message=f"vLLM 서버 에러: {response.status_code} - {error_body[:500]}",
                        )
                        return

                    # SSE 라인 파싱 (async with 블록 안에서 수행).
                    # SSE 프로토콜은 "data: {json}\n\n" 형태로 청크를 흘려보낸다.
                    async for line in response.aiter_lines():
                        # "data: " 접두사가 없는 줄(빈 줄, 주석 등)은 무시
                        if not line.startswith("data: "):
                            continue

                        # "data: " (6글자) 이후가 실제 페이로드
                        data_str = line[6:].strip()
                        if data_str == "[DONE]":
                            if sc_mode:
                                # ── SC 종료 처리 (§2.3) ──
                                for _evt in self._finalize_sc_stream(
                                    sc_candidate_texts,
                                    sc_tool_seen,
                                    accumulated_tool_calls,
                                    pending_stop,
                                    total_usage,
                                ):
                                    yield _evt
                                return
                            # 아직 finalize되지 않은 누적 tool_calls가 있으면 여기서
                            # 처리한다. (finish_reason 청크가 먼저 오면 그쪽에서 이미
                            # 처리되어 tool_calls_finalized=True. finish_reason 없이
                            # [DONE]만 오는 경우엔 여기가 유일한 finalize 지점이다.)
                            if accumulated_tool_calls and not tool_calls_finalized:
                                for _evt in self._finalize_tool_calls(accumulated_tool_calls):
                                    yield _evt
                                tool_calls_finalized = True
                                # tool_calls가 있으면 종료 이유는 TOOL_USE가 맞다.
                                if pending_stop is None:
                                    pending_stop = StopReason.TOOL_USE
                            yield StreamEvent(
                                type=StreamEventType.MESSAGE_STOP,
                                stop_reason=pending_stop or StopReason.END_TURN,
                                usage=total_usage,
                            )
                            return

                        try:
                            data = json.loads(data_str)
                        except json.JSONDecodeError:
                            logger.warning("잘못된 SSE JSON: %s", data_str[:200])
                            continue

                        # usage 청크 — stream_options.include_usage 덕분에 마지막에
                        # 별도 청크로 온다. OpenAI 키 이름(prompt/completion)을
                        # Nexus TokenUsage 필드(input/output)로 매핑한다.
                        if "usage" in data and data["usage"]:
                            u = data["usage"]
                            total_usage = TokenUsage(
                                input_tokens=u.get("prompt_tokens", 0),
                                output_tokens=u.get("completion_tokens", 0),
                            )

                        # usage 전용 청크는 choices가 비어 있을 수 있으므로 건너뛴다
                        if not data.get("choices"):
                            continue

                        # ── SC 모드(n>1): 표본을 index로 디멀티플렉스 ──────────────
                        # vLLM은 청크마다 choices[i].index로 시퀀스를 구분해 보낸다.
                        # 표본별 텍스트를 버퍼링만 하고(라이브 TEXT_DELTA 억제), tool_calls는
                        # choice 0에 대해서만 라이브로 흘려 단일 폴백을 준비한다(§2.3).
                        if sc_mode:
                            for _sc_choice in data["choices"]:
                                ci = _sc_choice.get("index", 0)
                                sc_delta = _sc_choice.get("delta", {})
                                sc_finish = _sc_choice.get("finish_reason")
                                # 본문 텍스트 조각을 표본 버퍼에 누적(표출은 합의 후).
                                if sc_delta.get("content"):
                                    sc_candidate_texts.setdefault(ci, []).append(
                                        sc_delta["content"]
                                    )
                                # reasoning_content를 본문으로 합치는 프로바이더(Scout 등)면
                                # 그 조각도 같은 버퍼에 담는다(내용 유실 방지).
                                sc_reasoning = sc_delta.get("reasoning_content")
                                if sc_reasoning and getattr(
                                    self, "_include_reasoning_as_text", False
                                ):
                                    sc_candidate_texts.setdefault(ci, []).append(
                                        sc_reasoning
                                    )
                                # tool_calls가 어떤 표본에든 나오면 SC 포기 신호. 실제
                                # 누적/이벤트는 choice 0에 대해서만 수행(단일 폴백 대상).
                                if "tool_calls" in sc_delta:
                                    sc_tool_seen = True
                                    if ci == 0:
                                        for _tc in sc_delta["tool_calls"]:
                                            for _evt in self._accumulate_tool_call(
                                                accumulated_tool_calls, _tc
                                            ):
                                                yield _evt
                                # choice 0의 finish_reason만 단일 폴백의 종료 이유로 쓴다.
                                if sc_finish and ci == 0:
                                    pending_stop = {
                                        "stop": StopReason.END_TURN,
                                        "length": StopReason.MAX_TOKENS,
                                        "tool_calls": StopReason.TOOL_USE,
                                    }.get(sc_finish, StopReason.END_TURN)
                            continue

                        # 스트리밍에서는 한 청크당 choice 1개. delta에 이번 조각이 담긴다.
                        choice = data["choices"][0]
                        delta = choice.get("delta", {})
                        finish_reason = choice.get("finish_reason")

                        # 본문 텍스트 조각 → 그대로 TEXT_DELTA로 상위에 흘려보낸다
                        if delta.get("content"):
                            yield StreamEvent(
                                type=StreamEventType.TEXT_DELTA,
                                text=delta["content"],
                            )

                        # Qwen3.5가 `<think>` 블록을 reasoning_content로 분리해 보내는 경우가
                        # 있다. 프로바이더 플래그로 이 값을 어떻게 다룰지 결정한다:
                        #   - _include_reasoning_as_text=True: TEXT_DELTA로 합쳐서 yield
                        #     (Scout 전용 — 4B는 정답 대부분을 reasoning에 실어 보낸다)
                        #   - 그 외: THINKING_DELTA로 분리 yield — UI에서 선택적으로 숨김
                        reasoning = delta.get("reasoning_content")
                        if reasoning:
                            include_as_text = getattr(
                                self, "_include_reasoning_as_text", False
                            )
                            yield StreamEvent(
                                type=(
                                    StreamEventType.TEXT_DELTA
                                    if include_as_text
                                    else StreamEventType.THINKING_DELTA
                                ),
                                text=reasoning,
                            )

                        # tool_calls 조각도 청크에 쪼개져 오므로 누적 헬퍼에 위임.
                        # 헬퍼가 시작/델타 시점에 맞는 StreamEvent를 만들어 반환한다.
                        if "tool_calls" in delta:
                            for tc_delta in delta["tool_calls"]:
                                for _evt in self._accumulate_tool_call(
                                    accumulated_tool_calls, tc_delta
                                ):
                                    yield _evt

                        # finish_reason이 오면 이번 응답이 끝났다는 뜻.
                        # OpenAI finish_reason 문자열을 Nexus StopReason으로 매핑.
                        # 알 수 없는 값은 안전하게 END_TURN으로 처리(fail-closed).
                        if finish_reason:
                            stop = {
                                "stop": StopReason.END_TURN,
                                "length": StopReason.MAX_TOKENS,
                                "tool_calls": StopReason.TOOL_USE,
                            }.get(finish_reason, StopReason.END_TURN)

                            # 누적된 tool_calls가 있으면 finalize한다. finish_reason
                            # 문자열("tool_calls")에만 의존하지 않는 이유: 강제
                            # tool_choice(named)에서는 vLLM이 tool_calls를 스트리밍하면서도
                            # finish_reason="stop"을 반환한다(B200 실측). 문자열만 보면
                            # 이 호출을 텍스트 응답으로 오인해 tool_calls를 유실한다.
                            # 그래서 실제 누적분 유무로 판정하고, tool_calls가 있으면
                            # 종료 이유도 TOOL_USE로 바로잡아 상위(query_loop)가 도구
                            # 실행 턴으로 진행하게 한다.
                            if accumulated_tool_calls and not tool_calls_finalized:
                                for _evt in self._finalize_tool_calls(
                                    accumulated_tool_calls
                                ):
                                    yield _evt
                                tool_calls_finalized = True
                                # finish_reason="length"(진짜 max_tokens 절단)는 MAX_TOKENS
                                # 종료 이유를 보존한다 — query_loop의 max-output 복구 판정에
                                # 필요하기 때문. 그 외("stop": 강제 tool_choice, "tool_calls":
                                # 일반)에서만 TOOL_USE로 바로잡는다. 절단된 tool call은 어차피
                                # parse_error=True로 finalize돼 상위 guided 재생성 경로로
                                # 흡수되므로, 여기서 TOOL_USE로 덮어써 MAX_TOKENS를 지울
                                # 필요가 없다(기존 복구 경로 보존).
                                if finish_reason != "length":
                                    stop = StopReason.TOOL_USE

                            # finish_reason 이후 usage 청크가 올 수 있으므로
                            # 바로 return하지 않고 finish 정보를 저장한다.
                            # [DONE] 또는 usage 청크에서 최종 yield + return 한다.
                            pending_stop = stop

            # ── 예외 처리: 모든 통신 오류를 ERROR StreamEvent로 변환 ──
            # 예외를 위로 던지지 않고 이벤트로 감싸는 이유: 상위 Tier(query_loop)는
            # AsyncGenerator를 소비할 뿐 try/except로 감싸지 않으므로, 여기서 ERROR
            # 이벤트로 내려보내야 사용자에게 안내 메시지가 깔끔하게 전달된다.
            except httpx.ConnectError as e:
                # 서버 자체에 연결 불가 — GPU 서버 다운/네트워크/방화벽 문제
                self._error_count += 1
                yield StreamEvent(
                    type=StreamEventType.ERROR,
                    error_code="CONNECT_ERROR",
                    message=(
                        f"GPU 서버(vLLM) 연결 실패: {self.base_url}\n"
                        f"상세: {e}\n"
                        f"확인: 1) GPU 서버 실행 여부 2) 네트워크 연결 3) 방화벽 설정"
                    ),
                )
            except httpx.ReadTimeout as e:
                # 연결은 됐지만 read_timeout 내에 응답 본문이 안 옴 (추론 과부하 등)
                self._error_count += 1
                yield StreamEvent(
                    type=StreamEventType.ERROR,
                    error_code="READ_TIMEOUT",
                    message=f"GPU 서버 응답 타임아웃 ({self._client.timeout.read}s): {e}",
                )
            except httpx.HTTPStatusError as e:
                # raise_for_status 등에서 올라온 HTTP 상태 오류
                self._error_count += 1
                yield StreamEvent(
                    type=StreamEventType.ERROR,
                    error_code=f"HTTP_{e.response.status_code}",
                    message=f"vLLM HTTP 에러: {e.response.status_code} - {e.response.text[:500]}",
                )
            except Exception as e:
                # 예상 못한 모든 오류의 최종 안전망 — anti-pattern #8(bare except)을
                # 피하되, 마지막 방어선으로 구체 타입을 남기고 traceback을 로깅한다.
                self._error_count += 1
                logger.exception("stream()에서 예상치 못한 에러: %s", e)
                yield StreamEvent(
                    type=StreamEventType.ERROR,
                    error_code="UNKNOWN",
                    message=f"예상치 못한 에러: {type(e).__name__}: {e}",
                )

            # 컨텍스트 초과 재시도가 아니면(정상 종료 또는 다른 에러) 루프 탈출.
            # context_exceeded=True일 때만 max_tokens를 줄여 다음 attempt로 넘어간다.
            if not context_exceeded:
                break
        else:
            # for...else: for가 break 없이 끝났을 때만 실행된다.
            # 즉 모든 attempt가 컨텍스트 초과로 소진된 경우 = 재시도 모두 실패
            yield StreamEvent(
                type=StreamEventType.ERROR,
                error_code="CONTEXT_OVERFLOW",
                message=(
                    "입력 내용이 너무 길어 분석할 수 없습니다. "
                    "더 짧은 내용으로 다시 시도해 주세요."
                ),
            )

        # 성공/실패와 무관하게 이번 호출에 걸린 시간을 누적(평균 latency 산출용)
        self._total_latency += time.monotonic() - start_time

    # ─── embed() ───

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """
        별도 임베딩 서버의 /v1/embed 엔드포인트를 호출한다.

        임베딩 서버 API:
          요청: POST /v1/embed {"texts": ["문장1", "문장2"]}
          응답: {"embeddings": [[...], [...]], "dimension": 1024}
        """
        try:
            response = await self._client.post(
                f"{self._embedding_base_url}/v1/embed",
                json={"texts": texts},
                headers={"Authorization": f"Bearer {self.api_key}"},
            )
            response.raise_for_status()
            data = response.json()
            return data["embeddings"]
        except Exception as e:
            logger.error(f"임베딩 실패: {e}")
            raise

    # ─── rerank() ───

    async def rerank(self, query: str, documents: list[str]) -> list[float]:
        """
        임베딩 서버의 /v1/rerank 엔드포인트를 호출해 크로스인코더 리랭킹 점수를 받는다.

        임베딩 서버 API (embed()와 동일한 base_url·인증 패턴):
          요청: POST /v1/rerank {"query": "질의", "documents": ["청크1", "청크2", ...]}
          응답: {"scores": [0.97, 0.00, ...]}  — documents와 동일 순서, 0~1(sigmoid)

        반환: documents와 같은 순서의 관련도 점수 리스트.
        빈 documents면 서버를 호출하지 않고 빈 리스트를 반환한다.
        실패/미로드 시 예외를 그대로 올린다 — 호출자(KnowledgeRetriever)가 이를
        fail-safe로 받아 기존 벡터순 경로로 폴백한다.
        """
        # 빈 입력은 서버 왕복 없이 즉시 반환(불필요한 네트워크 호출 회피).
        if not documents:
            return []
        try:
            response = await self._client.post(
                f"{self._embedding_base_url}/v1/rerank",
                json={"query": query, "documents": documents},
                headers={"Authorization": f"Bearer {self.api_key}"},
            )
            response.raise_for_status()
            data = response.json()
            return data["scores"]
        except Exception as e:
            logger.error(f"리랭킹 실패: {e}")
            raise

    # ─── health_check() ───

    async def health_check(self) -> bool:
        """vLLM /health 엔드포인트로 가용성을 확인한다."""
        try:
            response = await self._client.get(
                f"{self.base_url}/health",
                timeout=5.0,
            )
            return response.status_code == 200
        except Exception:
            return False

    # ─── count_tokens() ───

    async def count_tokens(self, messages: list[Message]) -> int:
        """
        메시지의 토큰 수를 휴리스틱으로 추정한다.

        실제 vLLM의 BPE 토크나이저를 호출하지 않고 Message.estimated_tokens()의
        근사치를 합산한다. 정확도가 필요한 컨텍스트 초과 판정은 stream() 내부에서
        서버가 돌려준 실제 input_tokens로 재계산하므로, 여기선 빠른 추정으로 충분하다.
        """
        return sum(m.estimated_tokens() for m in messages)

    # ─── get_config() ───

    def get_config(self) -> ModelConfig:
        return self._config

    # ─── 내부: 메시지 변환 ───

    def _convert_messages(
        self, messages: list[Message], system_prompt: str
    ) -> list[dict[str, Any]]:
        """
        Nexus Message[] → OpenAI messages[] 변환.

        변환 규칙:
          - system → {"role": "system", "content": ...}
          - user → {"role": "user", "content": ...}
          - assistant → {"role": "assistant", "content": ..., "tool_calls": [...]}
          - tool_result → {"role": "tool", "content": ..., "tool_call_id": ...}
        """
        oai: list[dict[str, Any]] = []

        # 시스템 프롬프트
        if system_prompt:
            oai.append({"role": "system", "content": system_prompt})

        for msg in messages:
            role = msg.role if isinstance(msg.role, str) else msg.role.value

            if role == "system":
                oai.append({"role": "system", "content": msg.text_content})
            elif role == "user":
                oai.append({"role": "user", "content": msg.text_content})
            elif role == "tool_result":
                oai.append({
                    "role": "tool",
                    "content": str(msg.content),
                    "tool_call_id": msg.tool_use_id or "",
                })
            elif role == "assistant":
                entry: dict[str, Any] = {"role": "assistant"}

                if isinstance(msg.content, str):
                    entry["content"] = msg.content
                elif isinstance(msg.content, list):
                    text_parts: list[str] = []
                    tool_calls: list[dict[str, Any]] = []

                    for block in msg.content:
                        if isinstance(block, TextBlock):
                            text_parts.append(block.text)
                        elif isinstance(block, ToolUseBlock):
                            tool_calls.append({
                                "id": block.id,
                                "type": "function",
                                "function": {
                                    "name": block.name,
                                    "arguments": json.dumps(
                                        block.input, ensure_ascii=False
                                    ),
                                },
                            })

                    # 텍스트가 전혀 없으면 content를 None으로 둔다. OpenAI 규약상
                    # tool_calls만 있는 assistant 메시지는 content가 null이어도 된다.
                    entry["content"] = " ".join(text_parts) if text_parts else None
                    if tool_calls:
                        entry["tool_calls"] = tool_calls

                oai.append(entry)

        return oai

    # ─── 내부: 구조화 출력 payload 조립 ───

    def _build_response_format(
        self, spec: StructuredOutputSpec
    ) -> dict[str, Any]:
        """
        StructuredOutputSpec을 vLLM payload에 병합할 조각(dict)으로 변환한다.

        반환값은 payload.update()로 최상위에 병합된다. injection_mode에 따라
        최상위 키 자체가 달라지므로(표준=response_format, 폴백=structured_outputs)
        키를 포함한 완성된 조각을 돌려준다.

        injection_mode(생성자 주입, 단일 소스는 config)에 따라 형태가 갈린다:
          - "response_format"  : OpenAI 표준
              {"response_format": {"type": "json_schema",
                "json_schema": {"name": ..., "schema": {...}, "strict": true}}}
            Phase 0 실측(B200 vLLM 0.24)에서 순수 JSON 강제 성공을 확인한 값이라
            기본값으로 둔다.
          - "structured_outputs": vLLM 신형 확장(설계 2.2)
              {"structured_outputs": {"json": {...}}}
            ※ 미실측 폴백 — 표준형이 거부되는 티어를 만났을 때만 config로 전환한다.
              strict/name은 이 형태에 대응 필드가 없어 스키마만 전달한다.

        알 수 없는 injection_mode는 fail-closed로 ValueError를 던진다(조용한 오동작 방지).
        """
        mode = self._structured_output_injection_mode
        if mode == "response_format":
            return {
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": spec.name,
                        "schema": spec.json_schema,
                        "strict": spec.strict,
                    },
                }
            }
        if mode == "structured_outputs":
            # 미실측 폴백 형태 — Phase 0에서 검증되지 않았다.
            return {"structured_outputs": {"json": spec.json_schema}}
        raise ValueError(
            f"알 수 없는 structured_output injection_mode: {mode!r} "
            "('response_format' 또는 'structured_outputs'만 허용)"
        )

    # ─── 내부: 도구 스키마 변환 ───

    @staticmethod
    def _convert_tool_schema(tool_schema: dict[str, Any]) -> dict[str, Any]:
        """
        Nexus 도구 스키마 → OpenAI function_calling 형식 변환.

        Nexus: {"name": "Read", "description": "...", "input_schema": {...}}
        OpenAI: {"type": "function", "function": {"name": "Read", ...}}
        """
        return {
            "type": "function",
            "function": {
                "name": tool_schema["name"],
                "description": tool_schema.get("description", ""),
                "parameters": tool_schema.get(
                    "input_schema", {"type": "object", "properties": {}}
                ),
            },
        }

    # ─── 내부: tool_calls 증분 누적 ───

    @staticmethod
    def _accumulate_tool_call(
        accumulated: dict[int, dict[str, Any]],
        tc_delta: dict[str, Any],
    ) -> list[StreamEvent]:
        """
        SSE chunk에서 온 tool_call 델타를 누적하고,
        필요한 StreamEvent를 반환한다.

        왜 누적이 필요한가: vLLM은 하나의 tool_call도 여러 청크로 쪼개 보낸다.
        보통 첫 청크에 id+function.name이 오고, 이후 청크들에 arguments(JSON 문자열)가
        조금씩 이어진다. index를 키로 삼아 같은 도구 호출의 조각들을 한곳에 모은다.
        """
        events: list[StreamEvent] = []
        # 같은 tool_call의 조각을 식별하는 키. 병렬 도구 호출이면 0,1,2... 로 구분된다.
        idx = tc_delta.get("index", 0)

        # 이 index가 처음 등장하면 빈 누적 슬롯을 만든다
        if idx not in accumulated:
            accumulated[idx] = {
                "id": tc_delta.get("id", ""),
                "function": {"name": "", "arguments": ""},
            }

        tc = accumulated[idx]
        # id는 나중 청크에서 채워질 수 있으므로 들어올 때마다 갱신
        if tc_delta.get("id"):
            tc["id"] = tc_delta["id"]

        func = tc_delta.get("function", {})
        # name은 보통 한 번만 온다 → 이 시점을 "도구 호출 시작"으로 보고 알린다
        if func.get("name"):
            tc["function"]["name"] = func["name"]
            # TOOL_USE_START 이벤트 (input은 아직 비어 있음 — 곧 델타로 채워진다)
            events.append(
                StreamEvent(
                    type=StreamEventType.TOOL_USE_START,
                    tool_use=ToolUseBlock(
                        id=tc["id"],
                        name=func["name"],
                        input={},
                    ),
                )
            )
        # arguments는 JSON 문자열 조각으로 여러 번 나뉘어 온다 → 이어붙이며 누적.
        # 동시에 각 조각을 TOOL_USE_DELTA로 흘려보내 UI가 실시간 표시할 수 있게 한다.
        if func.get("arguments"):
            tc["function"]["arguments"] += func["arguments"]
            events.append(
                StreamEvent(
                    type=StreamEventType.TOOL_USE_DELTA,
                    tool_use_delta=func["arguments"],
                )
            )

        return events

    # ─── 내부: tool_calls 최종 완성 ───

    @staticmethod
    def _finalize_tool_calls(
        accumulated: dict[int, dict[str, Any]],
    ) -> list[StreamEvent]:
        """
        누적된 tool_calls를 최종 TOOL_USE_STOP 이벤트로 변환한다.

        finish_reason="tool_calls" 또는 [DONE] 시점에 호출된다. 그동안 조각으로
        모아둔 arguments 문자열을 이제 완성된 JSON으로 파싱해 도구별로 닫아준다.
        index 순서대로 처리해 병렬 도구 호출의 순서를 안정적으로 유지한다.
        """
        events: list[StreamEvent] = []
        for idx in sorted(accumulated.keys()):
            tc = accumulated[idx]
            # name이 없는 슬롯은 불완전한 호출이므로 건너뛴다
            if tc.get("function", {}).get("name"):
                # 누적된 arguments 문자열을 dict로 파싱. 깨진 JSON이면 빈 dict로
                # 폴백해 도구 실행 단계에서 스키마 검증이 처리하도록 넘긴다.
                #
                # 왜 경고 로그를 남기나(관측성): vLLM은 auto tool choice에서 tool
                # 인자에 guided decoding을 기본 적용하므로 실측상 깨진 JSON은 거의
                # 없다(스키마·enum 준수 확인됨). 그래도 복잡한 스키마·모델 부하 시
                # 드물게 깨질 수 있는데, 지금까지는 조용히 {}로 삼켜 프로덕션에서
                # 발생 여부조차 알 수 없었다. 여기서 원문(절단)과 도구명을 warning으로
                # 남겨, guided decoding이 실제로 실패하는지 감지한다(반복되면 그때
                # 명시적 guided 강제를 검토). {} 폴백은 유지 — 도구 스키마 검증이
                # 거부하면 tool_use_error가 되어 모델이 자기교정(재호출)한다.
                raw_args = tc["function"].get("arguments", "{}")
                # 파싱 실패 신호 — strict=False로도 복구 못 하면 True. 상위(query_loop)가
                # 이 신호를 보고 tool_choice를 해당 도구로 강제한 guided 재시도를 건다.
                parse_failed = False
                try:
                    args = json.loads(raw_args)
                except json.JSONDecodeError as e:
                    # 1차 실패의 실측 원인: A.X-4.0가 DocumentExport 등 긴 content
                    # 문자열 값 안에 실제 개행/탭(제어문자)을 이스케이프(\\n) 없이
                    # 그대로 넣어, json 기본(strict=True)이 "Invalid control
                    # character"로 거부한다. (해당 턴은 output_tokens가 max에 못
                    # 미치고 stop_reason=tool_use로 정상 종료 → 절단이 아님이 실측됨.)
                    # strict=False로 재파싱하면 문자열 내부의 실제 제어문자를 허용해
                    # 대부분 복구된다 → content가 살아 DocumentExport가 파일을 만든다.
                    try:
                        args = json.loads(raw_args, strict=False)
                        logger.warning(
                            "tool_call arguments 1차 파싱 실패 → strict=False 복구 성공 "
                            "(tool=%s, err=%s, len=%d)",
                            tc["function"]["name"],
                            e,
                            len(raw_args),
                        )
                    except json.JSONDecodeError as e2:
                        # strict=False로도 실패 = 다른 malformation(절단/구조 파손).
                        # 말미(tail)를 남겨 원인(절단이면 문자열 미종료)을 추적하고,
                        # {}로 폴백해 도구 스키마 검증이 거부 → tool_use_error로 모델이
                        # 스스로 재호출(자기교정)하도록 넘긴다.
                        logger.warning(
                            "tool_call arguments JSON 파싱 실패 — 빈 dict 폴백 "
                            "(tool=%s, err=%s, len=%d, tail=%.200r)",
                            tc["function"]["name"],
                            e2,
                            len(raw_args),
                            raw_args[-200:],
                        )
                        args = {}
                        parse_failed = True
                events.append(
                    StreamEvent(
                        type=StreamEventType.TOOL_USE_STOP,
                        tool_use=ToolUseBlock(
                            id=tc.get("id", f"call_{idx}"),
                            name=tc["function"]["name"],
                            input=args,
                            parse_error=parse_failed,
                        ),
                    )
                )
        return events

    # ─── 내부: 자기일관성(SC) 스트림 종료 처리 ───

    def _finalize_sc_stream(
        self,
        candidate_texts: dict[int, list[str]],
        tool_seen: bool,
        accumulated_tool_calls: dict[int, dict[str, Any]],
        pending_stop: StopReason | None,
        total_usage: TokenUsage,
    ) -> list[StreamEvent]:
        """
        SC 모드([DONE] 시점)의 종료 이벤트 목록을 만든다(설계 §2.3).

        두 갈래:
          (1) tool_seen=True → SC 포기, choice 0만으로 단일 스트림처럼 폴백.
              choice 0 버퍼 텍스트를 TEXT_DELTA로, 누적된 tool_calls를 TOOL_USE_STOP으로
              내보내고 MESSAGE_STOP으로 닫는다. 상위(query_loop)는 tool_use가 있으므로
              SC 합의를 건너뛰고 일반 도구 실행 턴으로 진행한다.
          (2) tool_seen=False → 표본별 완성 텍스트를 SC_CANDIDATE(text + sample_index)로
              index 순서대로 1건씩 올려보낸 뒤 MESSAGE_STOP으로 닫는다. 상위가 이
              후보들을 버퍼링해 다수결 합의를 낸다. usage는 표본 합산이 이미 반영된
              vLLM 값을 그대로 정직 보고한다(과소 계상 금지, §6.2).
        """
        events: list[StreamEvent] = []
        if tool_seen:
            # choice 0의 버퍼 텍스트를 단일 스트림처럼 흘려보낸다.
            text0 = "".join(candidate_texts.get(0, []))
            if text0:
                events.append(
                    StreamEvent(type=StreamEventType.TEXT_DELTA, text=text0)
                )
            # 누적된 choice 0 tool_calls를 완성 이벤트로 닫는다.
            events.extend(self._finalize_tool_calls(accumulated_tool_calls))
            events.append(
                StreamEvent(
                    type=StreamEventType.MESSAGE_STOP,
                    stop_reason=pending_stop or StopReason.END_TURN,
                    usage=total_usage,
                )
            )
            return events

        # 순수 SC — 표본을 SC_CANDIDATE로 index 순서대로 올려보낸다.
        for ci in sorted(candidate_texts.keys()):
            events.append(
                StreamEvent(
                    type=StreamEventType.SC_CANDIDATE,
                    text="".join(candidate_texts[ci]),
                    sample_index=ci,
                )
            )
        events.append(
            StreamEvent(
                type=StreamEventType.MESSAGE_STOP,
                stop_reason=pending_stop or StopReason.END_TURN,
                usage=total_usage,
            )
        )
        return events

    # ─── 통계 ───

    @property
    def stats(self) -> dict[str, Any]:
        """
        요청 통계를 반환한다 (운영 모니터링/디버깅용).

        request_count가 0일 때 0으로 나누지 않도록 avg 계산에서 max(...,1)을 쓴다.
        """
        return {
            "request_count": self._request_count,
            "error_count": self._error_count,
            "total_latency_s": round(self._total_latency, 2),
            "avg_latency_s": round(
                self._total_latency / max(self._request_count, 1), 2
            ),
        }

    async def close(self) -> None:
        """
        HTTP 클라이언트를 정리한다.

        커넥션 풀의 열린 소켓을 닫는다. 앱 종료 시 호출해 리소스 누수를 막는다.
        """
        await self._client.aclose()
