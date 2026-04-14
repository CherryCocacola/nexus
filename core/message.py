"""
메시지 타입 시스템 — 대화의 모든 데이터 구조를 정의한다.

Claude Code의 messages/ 모듈을 Python Pydantic v2로 재구현한다.
4-Tier AsyncGenerator 체인 전체에서 StreamEvent와 Message를 사용한다.

주요 타입:
  - StreamEvent: 모델 스트리밍 응답의 개별 이벤트 (frozen하지 않음 — 메타데이터 추가 필요)
  - ContentBlock: assistant 메시지 내부의 구조화 블록 (text, tool_use, thinking)
  - Message: 대화의 한 턴 (user, assistant, tool_result, system)
  - Conversation: Message 컨테이너 (컨텍스트 압축 경계 관리)
  - TokenUsage: API 호출당 토큰 사용량

설계 결정:
  - Pydantic v2 사용: 직렬화 성능 + JSON Schema 자동 생성
  - Union discriminator: ContentBlock의 type 필드로 자동 dispatch
  - Factory method 패턴: Message.user(), Message.assistant() 등
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
    """메시지 역할. Claude Code의 role type에 대응한다."""

    USER = "user"
    ASSISTANT = "assistant"
    TOOL_RESULT = "tool_result"
    SYSTEM = "system"


class StopReason(str, Enum):
    """모델 응답 종료 이유. vLLM/OpenAI의 finish_reason에 매핑된다."""

    END_TURN = "end_turn"  # 정상 종료
    MAX_TOKENS = "max_tokens"  # 토큰 한도 도달
    STOP_SEQUENCE = "stop_sequence"  # 정지 시퀀스 매칭
    TOOL_USE = "tool_use"  # 도구 호출로 인한 종료


class StreamEventType(str, Enum):
    """
    스트리밍 이벤트 타입.
    Claude Code의 SDKMessage union에 대응한다.
    4-Tier 체인에서 yield되는 유일한 데이터 전달 단위이다.
    """

    # 모델 응답 관련
    MESSAGE_START = "message_start"
    TEXT_DELTA = "text_delta"
    TEXT_STOP = "text_stop"
    TOOL_USE_START = "tool_use_start"
    TOOL_USE_DELTA = "tool_use_delta"  # 도구 JSON 증분 조각
    TOOL_USE_STOP = "tool_use_stop"
    MESSAGE_STOP = "message_stop"

    # 사고(thinking) 관련
    THINKING_START = "thinking_start"
    THINKING_DELTA = "thinking_delta"
    THINKING_STOP = "thinking_stop"

    # 시스템 관련
    ERROR = "error"
    SYSTEM_INFO = "system_info"
    SYSTEM_WARNING = "system_warning"

    # 진행 상태
    TOOL_RESULT = "tool_result"
    STREAM_REQUEST_START = "stream_request_start"
    STREAM_REQUEST_END = "stream_request_end"

    # 사용량
    USAGE_UPDATE = "usage_update"


# ─────────────────────────────────────────────
# 콘텐츠 블록 (assistant 메시지의 내부 구조)
# ─────────────────────────────────────────────
class TextBlock(BaseModel):
    """텍스트 콘텐츠 블록."""

    type: Literal["text"] = "text"
    text: str


class ToolUseBlock(BaseModel):
    """
    도구 호출 블록.
    Claude Code의 tool_use ContentBlock에 대응한다.
    id는 자동 생성되며, 도구 결과와 매칭하는 데 사용한다.
    """

    type: Literal["tool_use"] = "tool_use"
    id: str = Field(default_factory=lambda: f"toolu_{uuid.uuid4().hex[:12]}")
    name: str
    input: dict[str, Any] = Field(default_factory=dict)


class ToolResultBlock(BaseModel):
    """도구 실행 결과 블록."""

    type: Literal["tool_result"] = "tool_result"
    tool_use_id: str
    content: str
    is_error: bool = False


class ThinkingBlock(BaseModel):
    """사고(thinking) 콘텐츠 블록. Ch.11에서 상세 구현한다."""

    type: Literal["thinking"] = "thinking"
    thinking: str
    signature: str | None = None  # 로컬 모델에서는 미사용


# type 필드로 자동 dispatch되는 Discriminated union
ContentBlock = TextBlock | ToolUseBlock | ToolResultBlock | ThinkingBlock


# ─────────────────────────────────────────────
# 토큰 사용량 추적
# ─────────────────────────────────────────────
class TokenUsage(BaseModel):
    """
    단일 API 호출의 토큰 사용량.
    vLLM의 stream_options.include_usage 응답에서 채워진다.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0  # vLLM prefix caching
    cache_read_input_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def __add__(self, other: TokenUsage) -> TokenUsage:
        """두 사용량을 합산한다. 세션 누적에 사용한다."""
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
# 스트리밍 이벤트
# ─────────────────────────────────────────────
class StreamEvent(BaseModel):
    """
    모델 스트리밍 이벤트.
    query_loop에서 AsyncGenerator[StreamEvent, None]으로 yield된다.

    Claude Code의 SDKMessage union을 하나의 모델로 통합한다.
    type 필드로 이벤트 종류를 구분하고, 해당 필드만 사용한다.
    """

    model_config = {"use_enum_values": True}

    type: StreamEventType

    # TEXT_DELTA
    text: str | None = None

    # TOOL_USE_START / TOOL_USE_STOP
    tool_use: ToolUseBlock | None = None
    # TOOL_USE_DELTA — JSON 증분 조각
    tool_use_delta: str | None = None

    # TOOL_RESULT
    tool_result: ToolResultBlock | None = None

    # MESSAGE_STOP
    stop_reason: StopReason | None = None

    # USAGE_UPDATE / MESSAGE_STOP
    usage: TokenUsage | None = None

    # ERROR / SYSTEM_INFO / SYSTEM_WARNING
    error_code: str | None = None
    message: str | None = None

    # THINKING
    thinking_text: str | None = None

    # 메타데이터
    model_id: str | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


# ─────────────────────────────────────────────
# 메시지 (대화의 한 턴)
# ─────────────────────────────────────────────
class Message(BaseModel):
    """
    대화의 한 턴을 나타내는 메시지.
    Claude Code의 messages 배열 요소에 대응한다.

    역할별 content 타입:
      - user: str (사용자 입력 텍스트)
      - assistant: list[ContentBlock] (텍스트 + 도구 호출 + 사고)
      - tool_result: str (도구 실행 결과)
      - system: str (시스템 지시, 컨텍스트 압축 요약 등)
    """

    id: str = Field(default_factory=lambda: f"msg_{uuid.uuid4().hex[:12]}")
    role: Role
    content: str | list[ContentBlock]
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))

    # tool_result 전용
    tool_use_id: str | None = None
    is_error: bool | None = None

    # 메타데이터
    metadata: dict[str, Any] = Field(default_factory=dict)

    # ─── Factory 메서드 ───

    @classmethod
    def user(cls, text: str, **metadata) -> Message:
        """사용자 메시지를 생성한다."""
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
        """
        blocks: list[ContentBlock] = []
        if thinking:
            blocks.append(ThinkingBlock(thinking=thinking))
        if text:
            blocks.append(TextBlock(text=text))
        if tool_uses:
            for tu in tool_uses:
                blocks.append(
                    ToolUseBlock(
                        id=tu.get("id", f"toolu_{uuid.uuid4().hex[:12]}"),
                        name=tu["name"],
                        input=tu.get("input", {}),
                    )
                )
        return cls(
            role=Role.ASSISTANT,
            content=blocks if blocks else text,
        )

    @classmethod
    def tool_result(
        cls,
        tool_use_id: str,
        content: str,
        is_error: bool = False,
    ) -> Message:
        """도구 실행 결과 메시지를 생성한다."""
        return cls(
            role=Role.TOOL_RESULT,
            content=content,
            tool_use_id=tool_use_id,
            is_error=is_error,
        )

    @classmethod
    def system(cls, text: str) -> Message:
        """시스템 메시지를 생성한다 (컨텍스트 압축 요약, 지시 등)."""
        return cls(role=Role.SYSTEM, content=text)

    # ─── 유틸리티 ───

    @property
    def text_content(self) -> str:
        """content에서 순수 텍스트만 추출한다."""
        if isinstance(self.content, str):
            return self.content
        texts = []
        for block in self.content:
            if isinstance(block, TextBlock):
                texts.append(block.text)
            elif isinstance(block, dict) and block.get("type") == "text":
                texts.append(block.get("text", ""))
        return "\n".join(texts)

    @property
    def tool_use_blocks(self) -> list[ToolUseBlock]:
        """content에서 tool_use 블록만 추출한다."""
        if isinstance(self.content, str):
            return []
        return [b for b in self.content if isinstance(b, ToolUseBlock)]

    @property
    def has_tool_use(self) -> bool:
        """도구 호출이 포함되어 있는지 확인한다."""
        return len(self.tool_use_blocks) > 0

    model_config = {"use_enum_values": True}

    def estimated_tokens(self) -> int:
        """
        토큰 수를 대략적으로 추정한다.
        한국어: 글자당 ~2 토큰, 영어: 단어당 ~1.3 토큰.
        정확한 카운팅은 토크나이저가 필요하지만,
        컨텍스트 압축 판단에는 이 수준이면 충분하다.
        """
        text = self.text_content if isinstance(self.content, str) else str(self.content)
        korean_chars = sum(1 for c in text if "\uac00" <= c <= "\ud7a3")
        ascii_words = len(text.encode("ascii", "ignore").split())
        return int(ascii_words * 1.3 + korean_chars * 2.0 + len(text) * 0.1)


# ─────────────────────────────────────────────
# Conversation (메시지 컨테이너)
# ─────────────────────────────────────────────
class Conversation(BaseModel):
    """
    대화 전체를 관리하는 컨테이너.
    QueryEngine이 하나의 Conversation을 소유한다.
    compact_boundary: 컨텍스트 압축 후 이 인덱스 이전의 메시지는 요약으로 대체된다.
    """

    messages: list[Message] = Field(default_factory=list)
    system_prompt: str = ""
    compact_boundary: int = 0

    def append(self, message: Message) -> None:
        """메시지를 추가한다."""
        self.messages.append(message)

    def get_active_messages(self) -> list[Message]:
        """compact_boundary 이후의 활성 메시지만 반환한다."""
        return self.messages[self.compact_boundary :]

    @property
    def total_estimated_tokens(self) -> int:
        """활성 메시지의 총 추정 토큰 수를 반환한다."""
        return sum(m.estimated_tokens() for m in self.get_active_messages())

    @property
    def turn_count(self) -> int:
        """user → assistant 쌍의 수를 반환한다."""
        return sum(1 for m in self.messages if m.role == Role.USER)

    def get_last_n_turns(self, n: int) -> list[Message]:
        """최근 n개 턴의 메시지를 추출한다 (user → assistant + tool_result 묶음)."""
        turns: list[list[Message]] = []
        current: list[Message] = []
        for msg in reversed(self.messages):
            current.insert(0, msg)
            if msg.role == Role.USER:
                turns.insert(0, current)
                current = []
                if len(turns) >= n:
                    break
        return [msg for turn in turns for msg in turn]
