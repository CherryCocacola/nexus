"""
프롬프트 포매터 — 모델별 chat template(대화 템플릿)을 적용하는 모듈.

[이 파일이 하는 일]
LLM은 "역할 구분자(마커)"로 감싼 하나의 긴 텍스트를 입력으로 받는다.
그런데 그 마커 문법은 모델마다 다르다. 예를 들어 어떤 모델은
<|im_start|>user 로 사용자 발화를 표시하고, 어떤 모델은 [|user|] 로 표시한다.
이 파일은 우리 내부 표준인 messages 리스트(역할+내용의 딕셔너리 목록)를
받아서, 대상 모델이 이해하는 정확한 문자열 프롬프트로 조립해 준다.

[왜 직접 구현하나]
Claude Code에서는 Anthropic SDK가 이 변환을 자동으로 처리해 준다.
하지만 Nexus는 외부 네트워크가 차단된 에어갭 환경에서 로컬 모델을 쓰므로
그 편의 기능이 없다. 따라서 우리가 직접 마커를 붙여 줘야 한다.
vLLM의 --chat-template 옵션으로도 비슷한 처리가 가능하지만, "도구 스키마를
시스템 프롬프트 안에 어떻게 심을지"는 Nexus가 직접 제어해야 하기 때문에
(모델별로 XML 도구 호출 포맷을 최적화해야 한다) 이 파일에서 통제한다.

[지원하는 형식]
  - Gemma 4: <start_of_turn>user / <start_of_turn>model 형식
  - ExaOne: [|system|] / [|user|] / [|assistant|] 형식
  - 기타(폴백): ChatML(<|im_start|>) 형식 — Qwen 등 범용 모델에 사용

[공개 진입점]
  format_chat_prompt() 하나가 외부에서 호출되는 메인 함수이고,
  나머지 _format_* / _extract_* 함수는 내부 헬퍼다.

작성자: 이현수 / 작성일: 2026-07-05
"""

from __future__ import annotations

import json

# ─────────────────────────────────────────────
# 도구 시스템 프롬프트 템플릿
# ─────────────────────────────────────────────
# 이 문자열은 "모델에게 도구 사용법을 가르치는 설명서"다.
# 도구가 있는 대화라면, 아래 각 _format_* 함수가 이 템플릿을 시스템 프롬프트
# 뒤에 이어 붙인다. {tool_descriptions} 자리에는 실제 도구 목록(_format_tool_
# descriptions의 출력)이 채워진다.
# 핵심 규약: 모델은 도구를 쓸 때 <tool_use> ... </tool_use> 블록 안에 JSON을
# 하나만 담아 응답해야 한다. 이 규약 덕분에 상위 파서가 XML 블록을 찾아
# tool_calls(내부 표준 형식)로 변환할 수 있다.
# 주의: 이 템플릿은 나중에 .format(tool_descriptions=...) 로 채워지므로,
# JSON 예시의 실제 중괄호는 이스케이프하기 위해 {{ }} 로 두 번 써야 한다.
TOOL_SYSTEM_PROMPT = """You have access to tools. To use a tool, respond with EXACTLY:
<tool_use>
{{"name": "tool_name", "input": {{"param": "value"}}}}
</tool_use>

Available tools:
{tool_descriptions}

Rules:
1. Use exactly ONE <tool_use> block per response when using tools
2. JSON inside must be valid
3. After </tool_use>, STOP generating
4. If no tool needed, respond normally with text only
5. NEVER output <tool_use> inside normal text explanation"""


# ─────────────────────────────────────────────
# 메인 포맷 함수
# ─────────────────────────────────────────────
def format_chat_prompt(
    messages: list[dict],
    system_prompt: str,
    tools: list[dict],
    model_name: str,
) -> str:
    """
    모델별 chat template을 적용하여 최종 프롬프트 문자열을 생성한다.

    이 파일의 유일한 공개 진입점이다. 모델 이름을 보고 어떤 마커 형식으로
    조립할지 결정한 뒤, 실제 조립은 알맞은 내부 _format_* 함수에 위임한다.
    (이런 구조를 흔히 "디스패처(dispatcher)"라고 부른다 — 판단만 하고
    실제 작업은 전문 함수에게 넘긴다.)

    Args:
        messages: 대화 이력. 각 원소는 역할과 내용을 담은 딕셔너리다.
            예) [{"role": "user"|"assistant"|"tool_result", "content": "..."}]
        system_prompt: 모델의 역할·규칙을 정의하는 시스템 프롬프트 텍스트.
        tools: OpenAI function schema 형식의 도구 목록. 비어 있으면 도구
            안내문을 붙이지 않는다.
        model_name: 대상 모델 이름. 소문자로 바꿔 부분 일치로 판별한다
            (예: "qwen-3.5-27b", "exaone-7.8b").

    Returns:
        대상 모델이 그대로 입력으로 받을 수 있는, 마커가 붙은 프롬프트 문자열.
    """
    # 모델 이름을 소문자로 정규화한 뒤 부분 문자열로 형식을 판별한다.
    if "qwen" in model_name.lower():
        # Qwen은 ChatML(<|im_start|>) 형식을 사용하므로 _format_chatml로 라우팅
        return _format_chatml(messages, system_prompt, tools)
    elif "exaone" in model_name.lower():
        # ExaOne 전용 [|...|] 마커 형식으로 조립한다
        return _format_exaone(messages, system_prompt, tools)
    else:
        # 어느 쪽에도 해당하지 않으면 가장 범용적인 ChatML 형식으로 폴백한다.
        # (알 수 없는 모델이라도 최소한 동작하도록 하는 안전한 기본값)
        return _format_chatml(messages, system_prompt, tools)


# ─────────────────────────────────────────────
# Gemma 4 포맷
# ─────────────────────────────────────────────
def _format_gemma(
    messages: list[dict],
    system_prompt: str,
    tools: list[dict],
) -> str:
    """
    Gemma 4 전용 chat template으로 프롬프트를 조립한다.

    Gemma가 이해하는 형식은 다음과 같다(한 턴마다 열고 닫는 마커로 감싼다):
      <start_of_turn>user
      ...본문...<end_of_turn>
      <start_of_turn>model
      ...본문...<end_of_turn>

    핵심 제약: Gemma는 별도의 system 역할을 지원하지 않는다. 그래서 시스템
    프롬프트(+도구 안내문)를 맨 앞 user 턴 안에 [System Instructions]라는
    머리표를 달아 끼워 넣는 방식으로 우회한다.

    Args:
        messages: 대화 이력(역할/내용 딕셔너리 목록).
        system_prompt: 시스템 프롬프트 텍스트.
        tools: 도구 스키마 목록. 있으면 안내문을 시스템 텍스트에 덧붙인다.

    Returns:
        Gemma 마커가 적용된 프롬프트 문자열. 맨 끝은 모델이 이어서 생성하도록
        <start_of_turn>model 로 열어 둔 채 반환한다.
    """
    # parts: 완성된 각 턴 조각을 순서대로 쌓아 두는 리스트.
    # 마지막에 개행으로 이어 붙여 하나의 문자열을 만든다.
    parts = []

    # 1) 시스템 프롬프트를 준비한다. 도구가 있으면 도구 사용 안내문을 뒤에 붙인다.
    full_system = system_prompt
    if tools:
        tool_desc = _format_tool_descriptions(tools)
        full_system += "\n\n" + TOOL_SYSTEM_PROMPT.format(tool_descriptions=tool_desc)

    # Gemma는 system 역할이 없으므로, 준비한 시스템 텍스트를 첫 user 턴에 심는다.
    parts.append(
        f"<start_of_turn>user\n[System Instructions]\n{full_system}<end_of_turn>"
    )

    # 2) 실제 대화 이력을 역할별로 마커를 붙여 하나씩 추가한다.
    for msg in messages:
        # role 키가 없으면 안전하게 "user"로 간주한다.
        role = msg.get("role", "user")
        # content는 문자열일 수도, 블록 리스트일 수도 있어 헬퍼로 평탄화한다.
        content = _extract_content(msg.get("content", ""))

        if role == "user":
            # 사용자 발화는 user 턴으로
            parts.append(f"<start_of_turn>user\n{content}<end_of_turn>")
        elif role == "assistant":
            # 모델(어시스턴트) 발화는 Gemma에서 "model" 역할로 표기한다.
            parts.append(f"<start_of_turn>model\n{content}<end_of_turn>")
        elif role == "tool_result":
            # 도구 실행 결과. Gemma엔 tool 역할이 없으니 user 턴에 넣되,
            # 성공/실패에 따라 머리표를 달고 어떤 호출의 결과인지 id를 표기한다.
            tool_id = msg.get("tool_use_id", "")
            is_error = msg.get("is_error", False)
            prefix = "[Tool Error]" if is_error else "[Tool Result]"
            parts.append(
                f"<start_of_turn>user\n{prefix} (id={tool_id})\n{content}<end_of_turn>"
            )

    # 3) 마지막에 model 턴을 "열기만" 한다. 닫지 않아야 모델이 이 지점부터
    #    답변을 생성한다(생성 시작 지점을 지정하는 관용적 패턴).
    parts.append("<start_of_turn>model")

    return "\n".join(parts)


# ─────────────────────────────────────────────
# ExaOne 포맷
# ─────────────────────────────────────────────
def _format_exaone(
    messages: list[dict],
    system_prompt: str,
    tools: list[dict],
) -> str:
    """
    ExaOne 전용 chat template으로 프롬프트를 조립한다.

    ExaOne이 이해하는 형식(각 턴은 [|endofturn|] 로 닫는다):
      [|system|]...[|endofturn|]
      [|user|]...[|endofturn|]
      [|assistant|]...[|endofturn|]

    Gemma와 달리 ExaOne은 system 역할을 정식으로 지원한다. 따라서 시스템
    프롬프트를 우회 없이 전용 [|system|] 블록에 그대로 넣을 수 있다.

    Args:
        messages: 대화 이력(역할/내용 딕셔너리 목록).
        system_prompt: 시스템 프롬프트 텍스트.
        tools: 도구 스키마 목록. 있으면 안내문을 시스템 텍스트에 덧붙인다.

    Returns:
        ExaOne 마커가 적용된 프롬프트 문자열. 끝은 [|assistant|] 로 열어 두어
        모델이 이어서 답변을 생성하게 한다.
    """
    parts = []

    # 1) 시스템 프롬프트 준비 — 도구가 있으면 사용 안내문을 뒤에 이어 붙인다.
    full_system = system_prompt
    if tools:
        tool_desc = _format_tool_descriptions(tools)
        full_system += "\n\n" + TOOL_SYSTEM_PROMPT.format(tool_descriptions=tool_desc)

    # ExaOne은 system 역할이 있으므로 전용 블록에 그대로 넣는다.
    parts.append(f"[|system|]{full_system}[|endofturn|]")

    # 2) 대화 이력을 역할별로 마커를 붙여 추가한다.
    for msg in messages:
        role = msg.get("role", "user")
        content = _extract_content(msg.get("content", ""))

        # 사용자 발화와 도구 결과는 모두 [|user|] 턴으로 표현한다.
        # (ExaOne에도 별도 tool 역할이 없어 user 쪽에 합류시킨다.)
        if role in ("user", "tool_result"):
            if role == "tool_result":
                # 도구 결과면 성공/실패 머리표를 본문 앞에 덧붙여 구분한다.
                prefix = "[Tool Error]" if msg.get("is_error") else "[Tool Result]"
                content = f"{prefix}\n{content}"
            parts.append(f"[|user|]{content}[|endofturn|]")
        elif role == "assistant":
            # 모델(어시스턴트) 발화
            parts.append(f"[|assistant|]{content}[|endofturn|]")

    # 3) 마지막에 assistant 블록을 열어 두어 생성 시작 지점을 지정한다.
    parts.append("[|assistant|]")

    return "\n".join(parts)


# ─────────────────────────────────────────────
# ChatML 범용 포맷 (폴백)
# ─────────────────────────────────────────────
def _format_chatml(
    messages: list[dict],
    system_prompt: str,
    tools: list[dict],
) -> str:
    """
    범용 ChatML 형식으로 프롬프트를 조립한다(폴백 및 Qwen용).

    ChatML은 많은 오픈소스 모델이 공유하는 사실상 표준 형식이라, 우리가
    명시적으로 지원하지 않는 모델이 와도 무난히 동작한다. Qwen 계열도 이
    형식을 쓰므로 format_chat_prompt에서 이 함수로 라우팅된다.

    형식(각 턴은 <|im_end|> 로 닫는다):
      <|im_start|>system\n...<|im_end|>
      <|im_start|>user\n...<|im_end|>
      <|im_start|>assistant\n...<|im_end|>

    Args:
        messages: 대화 이력(역할/내용 딕셔너리 목록).
        system_prompt: 시스템 프롬프트 텍스트.
        tools: 도구 스키마 목록. 있으면 안내문을 시스템 텍스트에 덧붙인다.

    Returns:
        ChatML 마커가 적용된 프롬프트 문자열. 끝은 assistant 턴을 열어 둔다.
    """
    parts = []

    # 1) 시스템 프롬프트 준비 — 도구가 있으면 사용 안내문을 이어 붙인다.
    full_system = system_prompt
    if tools:
        tool_desc = _format_tool_descriptions(tools)
        full_system += "\n\n" + TOOL_SYSTEM_PROMPT.format(tool_descriptions=tool_desc)

    # ChatML은 system 역할을 지원하므로 전용 블록으로 넣는다.
    parts.append(f"<|im_start|>system\n{full_system}<|im_end|>")

    # 2) 대화 이력을 역할별로 마커를 붙여 추가한다.
    for msg in messages:
        role = msg.get("role", "user")
        content = _extract_content(msg.get("content", ""))

        # 사용자 발화와 도구 결과는 모두 user 턴으로 합류시킨다.
        if role in ("user", "tool_result"):
            parts.append(f"<|im_start|>user\n{content}<|im_end|>")
        elif role == "assistant":
            parts.append(f"<|im_start|>assistant\n{content}<|im_end|>")

    # 3) assistant 턴을 열어 두어 모델이 이 지점부터 답변을 생성하게 한다.
    parts.append("<|im_start|>assistant")

    return "\n".join(parts)


# ─────────────────────────────────────────────
# 도구 설명 포맷 (XML)
# ─────────────────────────────────────────────
def _format_tool_descriptions(tools: list[dict]) -> str:
    """
    도구 스키마 목록을 모델이 읽기 좋은 XML 형태의 설명문으로 변환한다.

    TOOL_SYSTEM_PROMPT의 {tool_descriptions} 자리에 채워질 텍스트를 만든다.
    각 도구를 <tool name="..."> ... </tool> 블록으로 감싸, 이름·설명·파라미터
    스키마(JSON)를 보기 좋게 나열한다. 이렇게 하면 모델이 어떤 도구를 어떤
    인자로 호출할 수 있는지 프롬프트만 보고 이해할 수 있다.

    Args:
        tools: OpenAI function schema 형식의 도구 목록. 각 원소는 최상위에
            "function" 키로 감싸여 있거나(표준), 그 내용이 바로 담겨 있을 수
            있어(func = tool.get("function", tool)) 둘 다 처리한다.

    Returns:
        모든 도구 블록을 개행으로 이어 붙인 하나의 XML 설명 문자열.
    """
    parts = []
    for tool in tools:
        # OpenAI 표준은 {"function": {...}} 로 감싸지만, 감싸지 않은 스키마도
        # 허용하려고 "function"이 없으면 tool 자체를 사용한다.
        func = tool.get("function", tool)
        # 각 필드는 누락돼도 안전하도록 기본값을 준다.
        name = func.get("name", "unknown")
        description = func.get("description", "")
        params = func.get("parameters", {})

        # 한 도구를 XML 블록으로 조립한다. params는 사람이 읽기 쉽도록
        # indent=2로 들여쓰고, ensure_ascii=False로 한글이 깨지지 않게 한다.
        parts.append(
            f'<tool name="{name}">\n'
            f"  Description: {description}\n"
            f"  Parameters: {json.dumps(params, ensure_ascii=False, indent=2)}\n"
            f"</tool>"
        )
    return "\n".join(parts)


# ─────────────────────────────────────────────
# 콘텐츠 텍스트 추출
# ─────────────────────────────────────────────
def _extract_content(content) -> str:
    """
    메시지의 content를 항상 하나의 평문 문자열로 평탄화(flatten)한다.

    왜 필요한가: content의 모양이 상황마다 다르기 때문이다. 단순 문자열일
    수도 있고, Anthropic 스타일의 "블록 리스트"(텍스트 블록/도구 호출 블록의
    목록)일 수도 있다. 위쪽 _format_* 함수들은 문자열만 다루면 되도록, 이
    함수가 어떤 형태가 오든 문자열 하나로 정리해 준다.

    처리 규칙:
      - str  → 그대로 반환.
      - list → 각 블록을 순회하며 텍스트로 변환해 개행으로 이어 붙인다.
          · {"type": "text"}     → 그 text를 사용.
          · {"type": "tool_use"} → 이름/입력을 JSON으로 만들어 <tool_use>
            블록 문자열로 복원(모델이 과거에 낸 도구 호출을 이력에 남길 때).
          · 문자열 원소          → 그대로 사용.
      - 그 외 예상 못 한 타입 → str()로 방어적으로 문자열화.

    Args:
        content: 문자열, 블록 딕셔너리 리스트, 또는 기타 값.

    Returns:
        평탄화된 단일 문자열.
    """
    # 가장 흔한 경우: 이미 문자열이면 손대지 않고 바로 돌려준다.
    if isinstance(content, str):
        return content
    # 블록 리스트인 경우: 블록 종류별로 텍스트를 뽑아 texts에 모은다.
    if isinstance(content, list):
        texts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    # 일반 텍스트 블록 — text 필드를 그대로 사용.
                    texts.append(block.get("text", ""))
                elif block.get("type") == "tool_use":
                    # 도구 호출 블록 — 모델이 과거에 낸 <tool_use>를 이력에
                    # 다시 넣기 위해 이름/입력만 추려 JSON으로 직렬화한다.
                    tool_data = {
                        "name": block.get("name"),
                        "input": block.get("input"),
                    }
                    # ensure_ascii=False로 한글 입력값이 깨지지 않게 한다.
                    tool_json = json.dumps(tool_data, ensure_ascii=False)
                    texts.append(f"<tool_use>\n{tool_json}\n</tool_use>")
            elif isinstance(block, str):
                # 리스트 안에 그냥 문자열이 섞여 있으면 그대로 채택.
                texts.append(block)
        return "\n".join(texts)
    # 문자열도 리스트도 아닌 예상 밖 타입 — 최소한 깨지지 않게 문자열화한다.
    return str(content)
