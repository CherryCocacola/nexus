"""
메시지 타입 시스템 — 대화의 모든 데이터 구조를 정의한다.

이 파일은 Nexus 대화 엔진의 "데이터 계약(schema)"을 한곳에 모아둔 모듈이다.
사용자 입력부터 모델 응답, 도구 호출/결과, 토큰 사용량까지 — 4-Tier
AsyncGenerator 체인을 흐르는 모든 값이 여기서 정의한 타입으로 표현된다.
Claude Code의 messages/ 모듈을 Python Pydantic v2로 재구현한 것이다.

한눈에 보는 전체 그림:
  - 스트리밍(모델→우리)은 StreamEvent 하나로 표현한다. (Tier 3~4에서 발생)
  - 대화 이력(우리가 저장/재전송)은 Message 리스트로 표현한다.
  - 여러 Message를 담아 압축 경계까지 관리하는 컨테이너가 Conversation이다.

주요 타입:
  - StreamEvent: 모델 스트리밍 응답의 개별 이벤트. type 필드로 종류를 구분.
    (frozen 아님 — 파싱 도중 model_id 등 메타데이터를 덧붙일 수 있어야 함)
  - ContentBlock: assistant 메시지 내부의 구조화 블록. text/tool_use/thinking를
    하나의 discriminated union으로 묶는다. (type 필드가 판별자)
  - Message: 대화의 한 턴. user / assistant / tool_result / system 역할을 가진다.
  - Conversation: Message 컨테이너. 컨텍스트 압축 경계(compact_boundary)를 관리.
  - TokenUsage: API 호출당 토큰 사용량. 세션 누적을 위해 덧셈(+)을 지원.

설계 결정(왜 이렇게 했는가):
  - Pydantic v2 사용: 직렬화 성능 + JSON Schema 자동 생성 이점.
  - Union discriminator: ContentBlock의 type 필드로 역직렬화 시 자동 dispatch.
  - Factory method 패턴: Message.user(), Message.assistant() 등으로 생성 규칙을
    한곳에 모아 호출부의 실수를 줄인다.

주로 어디서 쓰나:
  - core/orchestrator/query_loop.py — StreamEvent를 yield하고 Message를 누적.
  - core/model/inference.py — SSE를 파싱해 StreamEvent로 변환.
  - core/memory/* — Message/Conversation을 저장·복원.

작성자: 이현수 / 작성일: 2026-07-05
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


# ─────────────────────────────────────────────
# 열거형 (Enums)
# ─────────────────────────────────────────────
class Role(str, Enum):
    """
    메시지 역할. Claude Code의 role type에 대응한다.

    str을 함께 상속하므로 값이 곧 문자열("user" 등)이라 JSON 직렬화가 자연스럽다.
    이 역할값에 따라 Message.content의 실제 타입(문자열/블록 리스트)이 달라진다.
    """

    USER = "user"  # 사용자 입력
    ASSISTANT = "assistant"  # 모델 응답 (텍스트 + 도구 호출 + 사고)
    TOOL_RESULT = "tool_result"  # 도구 실행 결과를 모델에게 되돌려줄 때
    SYSTEM = "system"  # 시스템 지시 / 컨텍스트 압축 요약 등


class StopReason(str, Enum):
    """
    모델 응답 종료 이유. vLLM/OpenAI의 finish_reason에 매핑된다.

    query_loop은 이 값을 보고 "다음 턴을 계속할지"를 판단한다.
    특히 TOOL_USE면 도구를 실행한 뒤 그 결과를 넣어 다시 모델을 호출한다.
    """

    END_TURN = "end_turn"  # 정상 종료
    MAX_TOKENS = "max_tokens"  # 토큰 한도 도달
    STOP_SEQUENCE = "stop_sequence"  # 정지 시퀀스 매칭
    TOOL_USE = "tool_use"  # 도구 호출로 인한 종료


class StreamEventType(str, Enum):
    """
    스트리밍 이벤트 타입.
    Claude Code의 SDKMessage union에 대응한다.
    4-Tier 체인에서 yield되는 유일한 데이터 전달 단위이다.

    하나의 응답 스트림은 대략 다음 흐름으로 이벤트가 발생한다:
    MESSAGE_START → (TEXT_DELTA*/TOOL_USE_START·DELTA·STOP/THINKING_*) →
    USAGE_UPDATE → MESSAGE_STOP. 소비자는 type만 보고 필요한 필드를 읽으면 된다.
    """

    # 모델 응답 관련
    MESSAGE_START = "message_start"  # 새 응답(assistant 메시지) 시작
    TEXT_DELTA = "text_delta"  # 본문 텍스트 조각(증분)
    TEXT_STOP = "text_stop"  # 텍스트 블록 종료
    TOOL_USE_START = "tool_use_start"  # 도구 호출 시작(이름/ID 확정)
    TOOL_USE_DELTA = "tool_use_delta"  # 도구 JSON 증분 조각
    TOOL_USE_STOP = "tool_use_stop"  # 도구 인자 JSON 완성
    MESSAGE_STOP = "message_stop"  # 응답 전체 종료(stop_reason 포함)

    # 사고(thinking) 관련
    THINKING_START = "thinking_start"  # 사고 블록 시작
    THINKING_DELTA = "thinking_delta"  # 사고 텍스트 조각(증분)
    THINKING_STOP = "thinking_stop"  # 사고 블록 종료

    # 시스템 관련
    ERROR = "error"  # 오류 발생(error_code/message 사용)
    SYSTEM_INFO = "system_info"  # 정보성 알림
    SYSTEM_WARNING = "system_warning"  # 경고성 알림

    # 컨텍스트 압축 표시 (2026-07-09)
    # ContextManager가 "실제로" 대화를 줄였을 때(no-op 통과가 아닐 때만) Tier 2
    # (query_loop)가 1회 yield해, UI에 "🗜️ 대화 압축 중 · <요약>"을 클로드처럼
    # 보여주기 위한 전용 이벤트. message 필드에 사람이 읽을 압축 요약 문구를 담는다
    # (예: "이전 대화 3턴 요약", "긴 도구 결과 정리", "대화 요약 생성(모델 호출)").
    # 미지 타입을 무시하는 기존 소비자에는 하위 호환(신규 type 추가일 뿐 기존 이벤트
    # 수정 아님 — anti-patterns #3 준수: StreamEvent는 새 인스턴스로만 생성).
    CONTEXT_COMPACT = "context_compact"  # 실제 압축 발생 알림(message = 사람이 읽을 요약)

    # 진행 상태
    TOOL_RESULT = "tool_result"  # 도구 실행 결과를 스트림으로 전달
    STREAM_REQUEST_START = "stream_request_start"  # 모델 요청 시작 마커
    STREAM_REQUEST_END = "stream_request_end"  # 모델 요청 종료 마커

    # 지식 RAG 출처 인용 (Point 4-2, 2026-07-08)
    # KNOWLEDGE 질의에서 주입된 지식 청크의 출처 메타데이터를 상위(웹/CLI)로 1회 전달.
    # Tier 1(QueryEngine.submit_message)이 프롬프트 조립 직후 yield하며, 웹은 이를
    # ChatResponse.sources(비스트림)/SSE(스트림)로 노출한다. 미지 타입을 무시하는
    # 기존 소비자에는 하위 호환(신규 type 추가일 뿐 기존 이벤트 수정 아님).
    KNOWLEDGE_SOURCES = "knowledge_sources"  # 지식 RAG 출처 목록(knowledge_sources 필드)

    # 자기일관성(Self-Consistency) 표본 후보 (Point 4-3, 2026-07-09)
    # vLLM n>1 요청에서 얻은 개별 표본 텍스트 1건을 Tier 3(inference.stream)가
    # Tier 2(query_loop)로 올려보내는 전용 이벤트. text에 후보 원문, sample_index에
    # 표본 번호(0..n-1)를 담는다. query_loop이 이 이벤트들을 버퍼링해 다수결 합의를
    # 낸 뒤 승자만 TEXT_DELTA로 의사-스트림하므로, UI로는 흘려보내지 않는다(버퍼 전용).
    # 미지 타입을 무시하는 기존 소비자에는 하위 호환(신규 type 추가일 뿐 기존 수정 아님).
    SC_CANDIDATE = "sc_candidate"  # SC 표본 후보 텍스트(text + sample_index 필드)

    # 사용량
    USAGE_UPDATE = "usage_update"  # 토큰 사용량 갱신


# ─────────────────────────────────────────────
# SYSTEM_WARNING 의 error_code — 스트림 정합성 신호 (2026-08-23)
# ─────────────────────────────────────────────
# 왜 필요한가 (실측 사고):
#   생성 붕괴가 감지되면 query_loop 은 앞의 출력을 버리고 재생성한다. 재생성으로도
#   복구하지 못하면 **잘린 출력을 그대로** 내보낸다. 그런데 그 사실을 알리는 신호가
#   없어서, 비대화형 `nexus ask` 가 잘린 답변을 **exit 0** 으로 내보냈다. 파이프·CI
#   소비자는 성공으로 오인한다.
#
#   메시지 텍스트를 매칭해 판별하면 문구가 바뀔 때마다 조용히 깨진다. 그래서
#   StreamEvent 가 이미 갖고 있는 error_code 필드에 안정적인 코드를 싣는다.
#
# 소비자 규약:
#   - STREAM_DISCARD_RETRY : 여태 흘려보낸 본문은 **폐기 대상**이다. 버퍼링하는
#     소비자(파이프 모드)는 버퍼를 비워야 한다. 재시도 자체는 정상 복구 경로이므로
#     **실패가 아니다** — 종료 코드를 바꾸면 안 된다.
#   - STREAM_TRUNCATED : 복구에 실패해 불완전한 출력을 내보내는 중이다. 자동화
#     소비자는 이것을 **실패로 처리**해야 한다(ask 는 exit 1).
STREAM_DISCARD_RETRY = "stream_discard_retry"
STREAM_TRUNCATED = "stream_truncated"


# ─────────────────────────────────────────────
# 콘텐츠 블록 (assistant 메시지의 내부 구조)
# ─────────────────────────────────────────────
class TextBlock(BaseModel):
    """
    텍스트 콘텐츠 블록.

    assistant 메시지 본문의 일반 텍스트 한 덩어리를 나타낸다.
    type 필드는 항상 "text" 고정값이라 union 판별자로 쓰인다.
    """

    type: Literal["text"] = "text"
    text: str


class ToolUseBlock(BaseModel):
    """
    도구 호출 블록.
    Claude Code의 tool_use ContentBlock에 대응한다.
    id는 자동 생성되며, 도구 결과와 매칭하는 데 사용한다.

    필드 설명:
      - id: 이 도구 호출의 고유 식별자. 나중에 ToolResultBlock.tool_use_id와
        짝을 맞춰 "어느 호출에 대한 결과인지" 연결한다.
      - name: 실행할 도구 이름(레지스트리 키).
      - input: 도구에 넘길 인자 딕셔너리. 스트리밍 중에는 JSON 조각으로 오다가
        완성 시점(TOOL_USE_STOP)에 파싱되어 채워진다.
      - parse_error: 모델이 낸 arguments JSON이 (strict=False로도) 파싱 불가여서
        input이 빈 dict로 폴백됐음을 표시하는 신호. True이면 이 호출을 빈 인자로
        실행하지 말고(→ 스키마 검증 실패·오답), 상위(query_loop)가 tool_choice를
        해당 도구로 강제한 guided decoding 재시도로 정상 JSON을 다시 받게 한다.
    """

    type: Literal["tool_use"] = "tool_use"
    id: str = Field(default_factory=lambda: f"toolu_{uuid.uuid4().hex[:12]}")
    name: str
    input: dict[str, Any] = Field(default_factory=dict)
    parse_error: bool = False


class ToolResultBlock(BaseModel):
    """
    도구 실행 결과 블록.

    tool_use_id로 어떤 ToolUseBlock에 대한 응답인지 가리키며,
    is_error가 True면 도구 실행이 실패했음을 모델에게 알린다.
    """

    type: Literal["tool_result"] = "tool_result"
    tool_use_id: str  # 짝이 되는 ToolUseBlock.id
    content: str  # 결과 텍스트(오류 시 오류 메시지)
    is_error: bool = False  # 실패 여부 — fail 정보를 모델에 전달


class ThinkingBlock(BaseModel):
    """
    사고(thinking) 콘텐츠 블록. Ch.11에서 상세 구현한다.

    모델이 답을 내기 전 내부 추론을 담는 블록이다.
    signature는 원격 API의 사고 서명 검증용 필드로, 로컬 모델에서는 쓰지 않는다.
    """

    type: Literal["thinking"] = "thinking"
    thinking: str
    signature: str | None = None  # 로컬 모델에서는 미사용


# type 필드로 자동 dispatch되는 Discriminated union.
# assistant content 리스트에 섞여 담기는 4종 블록을 하나의 타입으로 묶는다.
# 역직렬화 시 각 dict의 "type" 값을 보고 Pydantic이 알맞은 클래스로 복원한다.
ContentBlock = TextBlock | ToolUseBlock | ToolResultBlock | ThinkingBlock


# ─────────────────────────────────────────────
# 토큰 사용량 추적
# ─────────────────────────────────────────────
class TokenUsage(BaseModel):
    """
    단일 API 호출의 토큰 사용량.
    vLLM의 stream_options.include_usage 응답에서 채워진다.

    캐시 관련 필드는 vLLM prefix caching 통계로, 프롬프트 캐시가 얼마나
    새로 만들어졌는지/재사용됐는지를 나타낸다. 세션 전체 누적은 __add__로 합산한다.
    """

    input_tokens: int = 0  # 프롬프트(입력) 토큰
    output_tokens: int = 0  # 생성(출력) 토큰
    cache_creation_input_tokens: int = 0  # vLLM prefix caching — 신규 생성분
    cache_read_input_tokens: int = 0  # vLLM prefix caching — 재사용분

    @property
    def total_tokens(self) -> int:
        """입력+출력 토큰 합계. 예산 판단 등에서 총량이 필요할 때 사용한다."""
        return self.input_tokens + self.output_tokens

    def __add__(self, other: TokenUsage) -> TokenUsage:
        """
        두 사용량을 합산한다. 세션 누적에 사용한다.

        `+` 연산자를 오버로드해 `session_usage = session_usage + call_usage`
        처럼 자연스럽게 누적할 수 있게 한다. 각 필드를 항목별로 더한다.
        """
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_creation_input_tokens=(
                self.cache_creation_input_tokens + other.cache_creation_input_tokens
            ),
            cache_read_input_tokens=(
                self.cache_read_input_tokens + other.cache_read_input_tokens
            ),
        )


# ─────────────────────────────────────────────
# 지식 RAG 출처 인용 (Point 4-2, 2026-07-08)
# ─────────────────────────────────────────────
class KnowledgeCitation(BaseModel):
    """주입된 지식 청크 1건의 출처 메타데이터 — 답변 인용·UI 노출용.

    KNOWLEDGE 질의에서 knowledge_retriever가 "실제로 프롬프트에 주입한" 청크
    하나마다 만들어지는 불변(frozen) 레코드다. 본문의 `[출처N]` 마커가 가리키는
    실체(제목/source/섹션/점수)를 담아, 모델이 아니라 "서버가 아는 진실"로 출처를
    노출한다(downloads 필드와 동일 원칙 — 모델 텍스트를 신뢰하지 않는다).

    이 모델을 core/message.py에 두는 이유(의존성 방향 P2):
      StreamEvent가 이 타입을 필드로 참조해야 하는데, StreamEvent는 core/message에
      있다. rag가 message를 import하는 것은 순방향(core/rag → core/message)이므로,
      모델을 여기 두고 rag가 가져다 쓴다(역방향 import 회피).

    필드:
      index    : 1부터 시작하는 출처 번호([출처N]의 N). 조립 시점의 '실제 주입
                 순서'로 부여 → 토큰 예산에서 잘려 주입 안 된 청크에는 번호가
                 없다("번호는 있는데 본문이 없는" 불일치 원천 차단).
      source   : 출처 계열('kowiki', 'docingest' 등).
      title    : 문서 제목.
      section  : 섹션명(없으면 None).
      score    : 신뢰도 — rerank_score 우선, 없으면 similarity.
      chunk_id : tb_knowledge.id(감사·추적용, 선택 — 없으면 None).
    """

    model_config = {"frozen": True}

    index: int
    source: str
    title: str
    section: str | None = None
    score: float = 0.0
    chunk_id: str | None = None


# ─────────────────────────────────────────────
# 스트리밍 이벤트
# ─────────────────────────────────────────────
class StreamEvent(BaseModel):
    """
    모델 스트리밍 이벤트.
    query_loop에서 AsyncGenerator[StreamEvent, None]으로 yield된다.

    Claude Code의 SDKMessage union을 하나의 모델로 통합한다.
    type 필드로 이벤트 종류를 구분하고, 해당 필드만 사용한다.

    설계 메모: 이벤트 종류마다 별도 클래스를 두지 않고 필드를 모두 Optional로
    합쳐 하나로 만들었다. 그 대신 "type에 맞는 필드만 채워진다"는 규칙을 지킨다.
    (예: TEXT_DELTA면 text만, MESSAGE_STOP이면 stop_reason/usage만 유효)
    아래 주석의 그룹핑이 "어떤 type일 때 어떤 필드를 읽어야 하는지" 지도 역할을 한다.
    """

    # use_enum_values: enum을 원시 문자열 값으로 저장(직렬화 단순화).
    # protected_namespaces=(): model_id 등 model_ 접두 필드 경고를 끈다.
    # frozen=True — StreamEvent는 4-Tier 체인 전체를 흐르는 불변 데이터 단위다
    # (domain-model.md·architecture.md P1, anti-patterns #3). 생성 후 수정 불가로
    # 두어 하위 Tier가 만든 이벤트를 상위에서 몰래 바꾸는 사고를 원천 차단한다.
    model_config = {"frozen": True, "use_enum_values": True, "protected_namespaces": ()}

    type: StreamEventType  # 이 이벤트의 종류 — 나머지 필드 해석의 기준

    # TEXT_DELTA
    text: str | None = None  # 본문 텍스트 조각

    # TOOL_USE_START / TOOL_USE_STOP
    tool_use: ToolUseBlock | None = None  # 도구 호출 정보(이름/ID/인자)
    # TOOL_USE_DELTA — JSON 증분 조각
    tool_use_delta: str | None = None  # 도구 인자 JSON의 부분 문자열

    # TOOL_RESULT
    tool_result: ToolResultBlock | None = None  # 도구 실행 결과

    # MESSAGE_STOP
    stop_reason: StopReason | None = None  # 응답 종료 이유

    # USAGE_UPDATE / MESSAGE_STOP
    usage: TokenUsage | None = None  # 토큰 사용량 스냅샷

    # ERROR / SYSTEM_INFO / SYSTEM_WARNING
    error_code: str | None = None  # 오류 분류 코드
    message: str | None = None  # 사람이 읽는 메시지 텍스트

    # THINKING
    thinking_text: str | None = None  # 사고(thinking) 텍스트 조각

    # KNOWLEDGE_SOURCES — 지식 RAG 출처 목록(주입된 청크의 출처 메타). Point 4-2.
    knowledge_sources: list[KnowledgeCitation] | None = None

    # SC_CANDIDATE — 자기일관성 표본 번호(0..n-1). Point 4-3. SC_CANDIDATE 이벤트에서만
    # 유효하며, 나머지 이벤트에서는 None(하위 호환 — 기존 소비자는 이 필드를 모른다).
    sample_index: int | None = None

    # 메타데이터
    model_id: str | None = None  # 이 이벤트를 만든 모델 식별자
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


# ─────────────────────────────────────────────
# 메시지 (대화의 한 턴)
# ─────────────────────────────────────────────
class Message(BaseModel):
    """
    대화의 한 턴을 나타내는 메시지.
    Claude Code의 messages 배열 요소에 대응한다.

    StreamEvent가 "실시간으로 흐르는" 단위라면, Message는 "저장·재전송되는"
    확정된 대화 기록 단위다. 스트리밍이 끝나면 이벤트들을 모아 Message로 굳힌다.

    역할별 content 타입:
      - user: str (사용자 입력 텍스트)
      - assistant: list[ContentBlock] (텍스트 + 도구 호출 + 사고)
      - tool_result: str (도구 실행 결과)
      - system: str (시스템 지시, 컨텍스트 압축 요약 등)

    즉 assistant만 구조화된 블록 리스트를 가지고, 나머지는 단순 문자열이다.
    생성은 아래 Factory 메서드(user/assistant/tool_result/system)로 하는 것을 권장.
    """

    id: str = Field(default_factory=lambda: f"msg_{uuid.uuid4().hex[:12]}")
    role: Role  # 이 메시지의 역할 — content 타입을 결정
    content: str | list[ContentBlock]  # 역할에 따라 문자열 또는 블록 리스트
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))

    # tool_result 전용 — 어떤 도구 호출에 대한 결과인지와 성공/실패
    tool_use_id: str | None = None
    is_error: bool | None = None

    # 메타데이터 — 자유롭게 부가정보를 담는 확장 슬롯
    metadata: dict[str, Any] = Field(default_factory=dict)

    # ─── Factory 메서드 ───
    # 생성 규칙(정규화, 기본값)을 한곳에 모아 호출부 실수를 막는다.

    @classmethod
    def user(cls, text: str, **metadata) -> Message:
        """
        사용자 메시지를 생성한다.

        추가 키워드 인자는 metadata로 그대로 저장되어, 출처·태그 등 부가정보를
        붙일 수 있다. content는 단순 문자열이다.
        """
        return cls(role=Role.USER, content=text, metadata=metadata)

    @classmethod
    def assistant(
        cls,
        text: str = "",
        tool_uses: list[dict[str, Any]] | None = None,
        thinking: str | None = None,
    ) -> Message:
        """
        assistant 메시지를 생성한다.
        content는 항상 list[ContentBlock]으로 정규화한다.

        블록 조립 순서는 thinking → text → tool_use 이며, 이 순서가 모델이
        실제로 "생각하고 → 말하고 → 도구를 부른" 흐름과 일치한다.

        매개변수:
          - text: 본문 텍스트(없으면 TextBlock을 만들지 않음).
          - tool_uses: 도구 호출들의 원시 dict 목록. 각 항목에서 id(없으면 자동
            생성)/name/input을 꺼내 ToolUseBlock으로 변환한다.
          - thinking: 사고 텍스트(있으면 맨 앞에 ThinkingBlock 추가).

        반환: 블록이 하나라도 있으면 content=블록 리스트, 전부 비면 content=text
        (빈 문자열)로 두어 최소한의 형태를 보장한다.
        """
        blocks: list[ContentBlock] = []
        # 1) 사고가 있으면 가장 먼저 담는다(추론 → 발화 → 도구 순서).
        if thinking:
            blocks.append(ThinkingBlock(thinking=thinking))
        # 2) 본문 텍스트가 있으면 텍스트 블록으로 담는다.
        if text:
            blocks.append(TextBlock(text=text))
        # 3) 도구 호출들을 각각 ToolUseBlock으로 변환해 담는다.
        if tool_uses:
            for tu in tool_uses:
                blocks.append(
                    ToolUseBlock(
                        # id가 없으면 여기서 새 식별자를 부여해 결과 매칭을 보장.
                        id=tu.get("id", f"toolu_{uuid.uuid4().hex[:12]}"),
                        name=tu["name"],
                        input=tu.get("input", {}),
                    )
                )
        return cls(
            role=Role.ASSISTANT,
            # 블록이 하나도 없을 때만 원시 text로 폴백한다.
            content=blocks if blocks else text,
        )

    @classmethod
    def tool_result(
        cls,
        tool_use_id: str,
        content: str,
        is_error: bool = False,
    ) -> Message:
        """
        도구 실행 결과 메시지를 생성한다.

        tool_use_id로 짝이 되는 도구 호출을 가리킨다. 실행이 실패했다면
        is_error=True로 만들어 모델이 오류를 인지하고 대응하도록 한다.
        """
        return cls(
            role=Role.TOOL_RESULT,
            content=content,
            tool_use_id=tool_use_id,
            is_error=is_error,
        )

    @classmethod
    def system(cls, text: str) -> Message:
        """
        시스템 메시지를 생성한다 (컨텍스트 압축 요약, 지시 등).

        주로 대화가 길어져 압축(compact)했을 때, 이전 내용을 요약한 결과를
        system 메시지로 넣어 이후 턴에서 참조하게 한다.
        """
        return cls(role=Role.SYSTEM, content=text)

    # ─── 유틸리티 ───

    @property
    def text_content(self) -> str:
        """
        content에서 순수 텍스트만 추출한다.

        content가 문자열이면 그대로 반환한다. 블록 리스트면 TextBlock의 텍스트만
        골라 이어 붙인다. 역직렬화 시점에 따라 블록이 아직 dict 형태일 수도 있어,
        dict인 경우("type"=="text")도 함께 처리한다. 여러 텍스트는 개행으로 잇는다.
        """
        if isinstance(self.content, str):
            return self.content
        texts = []
        for block in self.content:
            if isinstance(block, TextBlock):
                texts.append(block.text)
            elif isinstance(block, dict) and block.get("type") == "text":
                # 아직 Pydantic 객체로 복원되지 않은 dict 형태 방어 처리.
                texts.append(block.get("text", ""))
        return "\n".join(texts)

    @property
    def tool_use_blocks(self) -> list[ToolUseBlock]:
        """
        content에서 tool_use 블록만 추출한다.

        content가 문자열(도구 호출이 없는 역할)이면 빈 리스트를 돌려준다.
        """
        if isinstance(self.content, str):
            return []
        return [b for b in self.content if isinstance(b, ToolUseBlock)]

    @property
    def has_tool_use(self) -> bool:
        """
        도구 호출이 포함되어 있는지 확인한다.

        query_loop이 "이 턴에서 도구를 실행해야 하는가"를 빠르게 판단할 때 쓴다.
        """
        return len(self.tool_use_blocks) > 0

    model_config = {"use_enum_values": True}

    def estimated_tokens(self) -> int:
        """
        토큰 수를 대략적으로 추정한다.
        한국어: 글자당 ~2 토큰, 영어: 단어당 ~1.3 토큰.
        정확한 카운팅은 토크나이저가 필요하지만,
        컨텍스트 압축 판단에는 이 수준이면 충분하다.

        계산 방식(빠른 근사):
          1) 한글 음절 개수를 센다(유니코드 가~힣 범위).
          2) ASCII 단어 수를 센다(비ASCII 제거 후 공백 분리).
          3) 남은 기타 문자를 글자 수의 10%로 보정한다.
        토크나이저를 부르지 않으므로 매우 빠르고, 압축 트리거 판단용으로 적합하다.
        """
        # content가 문자열이면 text_content로, 블록 리스트면 문자열화해 근사한다.
        text = self.text_content if isinstance(self.content, str) else str(self.content)
        # 한글 음절(가~힣) 개수 — 글자당 약 2토큰으로 가중.
        korean_chars = sum(1 for c in text if "\uac00" <= c <= "\ud7a3")
        # ASCII 단어 수 — 비ASCII를 버리고 공백으로 나눠 센다.
        ascii_words = len(text.encode("ascii", "ignore").split())
        # 세 요소를 가중 합산해 정수 토큰 추정치를 만든다.
        return int(ascii_words * 1.3 + korean_chars * 2.0 + len(text) * 0.1)


# ─────────────────────────────────────────────
# Conversation (메시지 컨테이너)
# ─────────────────────────────────────────────
class Conversation(BaseModel):
    """
    대화 전체를 관리하는 컨테이너.
    QueryEngine이 하나의 Conversation을 소유한다.
    compact_boundary: 컨텍스트 압축 후 이 인덱스 이전의 메시지는 요약으로 대체된다.

    핵심 개념 — compact_boundary(압축 경계):
      대화가 길어져 컨텍스트 한도에 다가가면 앞부분을 요약(compact)한다. 이때
      원본 메시지를 지우지 않고, 경계 인덱스만 앞으로 옮겨 "여기부터가 활성"임을
      표시한다. get_active_messages()는 경계 이후만 반환해 모델에 보낼 분량을 줄인다.
    """

    messages: list[Message] = Field(default_factory=list)
    system_prompt: str = ""  # 대화 전체에 적용되는 시스템 프롬프트
    compact_boundary: int = 0  # 이 인덱스 이전은 압축(요약)된 것으로 간주

    def append(self, message: Message) -> None:
        """메시지를 대화 끝에 추가한다."""
        self.messages.append(message)

    def get_active_messages(self) -> list[Message]:
        """
        compact_boundary 이후의 활성 메시지만 반환한다.

        모델 호출 시 실제로 전송할 대상이며, 압축된 앞부분은 제외된다.
        """
        return self.messages[self.compact_boundary :]

    @property
    def total_estimated_tokens(self) -> int:
        """
        활성 메시지의 총 추정 토큰 수를 반환한다.

        각 메시지의 estimated_tokens()를 합산한다. 압축 트리거 판단의 입력값이다.
        """
        return sum(m.estimated_tokens() for m in self.get_active_messages())

    @property
    def turn_count(self) -> int:
        """
        user → assistant 쌍의 수를 반환한다.

        user 메시지 개수를 곧 턴 수로 본다(한 번의 사용자 발화 = 한 턴).
        """
        return sum(1 for m in self.messages if m.role == Role.USER)

    def get_last_n_turns(self, n: int) -> list[Message]:
        """
        최근 n개 턴의 메시지를 추출한다 (user → assistant + tool_result 묶음).

        동작 방식:
          - 뒤에서부터 메시지를 훑으며 임시 버퍼(current)에 앞쪽으로 쌓는다.
          - user 메시지를 만나면 "한 턴의 시작"으로 보고 그 묶음을 확정한다.
          - 확정된 턴이 n개가 되면 멈춘다.
          - 마지막에 턴들을 원래 순서대로 평탄화(flatten)해 단일 리스트로 돌려준다.
        즉 "user + 그 뒤에 딸린 assistant/tool_result"가 하나의 턴 묶음이 된다.
        """
        turns: list[list[Message]] = []  # 확정된 턴들(각 턴은 메시지 묶음)
        current: list[Message] = []  # 아직 user를 못 만난 진행 중 묶음
        for msg in reversed(self.messages):
            # 역순 순회이므로 항상 맨 앞(0)에 삽입해 원래 순서를 유지한다.
            current.insert(0, msg)
            # user를 만나면 이 묶음이 한 턴의 시작 — 턴 목록 앞에 확정한다.
            if msg.role == Role.USER:
                turns.insert(0, current)
                current = []
                # 원하는 턴 수를 채웠으면 더 볼 필요 없다.
                if len(turns) >= n:
                    break
        # 2차원(턴×메시지)을 1차원으로 펼쳐 반환한다.
        return [msg for turn in turns for msg in turn]
