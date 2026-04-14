"""
프롬프트 포매터 — 모델별 chat template을 적용한다.

Claude Code에서는 Anthropic SDK가 자동 처리하던 부분을
에어갭 환경에서 직접 구현한다.

지원 모델:
  - Gemma 4: <start_of_turn>user/model 형식
  - ExaOne: [|system|]/[|user|]/[|assistant|] 형식
  - 기타: ChatML 형식 (범용 폴백)

왜 별도 포매터인가: vLLM이 --chat-template 옵션으로 처리할 수도 있지만,
도구 스키마를 시스템 프롬프트에 주입하는 로직은 Nexus가 직접 제어해야 한다.
특히 XML 기반 도구 호출 포맷은 모델별로 최적화가 필요하다.
"""

from __future__ import annotations

import json

# ─────────────────────────────────────────────
# 도구 시스템 프롬프트 템플릿
# ─────────────────────────────────────────────
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
    모델별 chat template을 적용하여 프롬프트를 생성한다.

    Args:
        messages: [{"role": "user"|"assistant"|"tool_result", "content": "..."}]
        system_prompt: 시스템 프롬프트
        tools: OpenAI function schema 리스트
        model_name: 모델 이름 (gemma-*, exaone-*)

    Returns:
        포맷된 프롬프트 문자열
    """
    if "gemma" in model_name.lower():
        return _format_gemma(messages, system_prompt, tools)
    elif "exaone" in model_name.lower():
        return _format_exaone(messages, system_prompt, tools)
    else:
        # 범용 ChatML 형식 폴백
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
    Gemma 4 chat template.

    형식:
      <start_of_turn>user
      ...<end_of_turn>
      <start_of_turn>model
      ...<end_of_turn>

    Gemma는 system 역할을 지원하지 않으므로,
    시스템 프롬프트를 첫 번째 user 메시지에 [System Instructions]로 주입한다.
    """
    parts = []

    # 시스템 프롬프트 + 도구 정의를 첫 user 메시지로 주입
    full_system = system_prompt
    if tools:
        tool_desc = _format_tool_descriptions(tools)
        full_system += "\n\n" + TOOL_SYSTEM_PROMPT.format(tool_descriptions=tool_desc)

    parts.append(
        f"<start_of_turn>user\n[System Instructions]\n{full_system}<end_of_turn>"
    )

    # 대화 메시지
    for msg in messages:
        role = msg.get("role", "user")
        content = _extract_content(msg.get("content", ""))

        if role == "user":
            parts.append(f"<start_of_turn>user\n{content}<end_of_turn>")
        elif role == "assistant":
            parts.append(f"<start_of_turn>model\n{content}<end_of_turn>")
        elif role == "tool_result":
            tool_id = msg.get("tool_use_id", "")
            is_error = msg.get("is_error", False)
            prefix = "[Tool Error]" if is_error else "[Tool Result]"
            parts.append(
                f"<start_of_turn>user\n{prefix} (id={tool_id})\n{content}<end_of_turn>"
            )

    # 모델 응답 시작
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
    ExaOne chat template.

    형식:
      [|system|]...[|endofturn|]
      [|user|]...[|endofturn|]
      [|assistant|]...[|endofturn|]

    ExaOne은 system 역할을 지원하므로 별도 [|system|] 블록을 사용한다.
    """
    parts = []

    # 시스템 프롬프트
    full_system = system_prompt
    if tools:
        tool_desc = _format_tool_descriptions(tools)
        full_system += "\n\n" + TOOL_SYSTEM_PROMPT.format(tool_descriptions=tool_desc)

    parts.append(f"[|system|]{full_system}[|endofturn|]")

    # 대화 메시지
    for msg in messages:
        role = msg.get("role", "user")
        content = _extract_content(msg.get("content", ""))

        if role in ("user", "tool_result"):
            if role == "tool_result":
                prefix = "[Tool Error]" if msg.get("is_error") else "[Tool Result]"
                content = f"{prefix}\n{content}"
            parts.append(f"[|user|]{content}[|endofturn|]")
        elif role == "assistant":
            parts.append(f"[|assistant|]{content}[|endofturn|]")

    # assistant 응답 시작
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
    """범용 ChatML 형식. Gemma/ExaOne 이외의 모델에 사용한다."""
    parts = []

    full_system = system_prompt
    if tools:
        tool_desc = _format_tool_descriptions(tools)
        full_system += "\n\n" + TOOL_SYSTEM_PROMPT.format(tool_descriptions=tool_desc)

    parts.append(f"<|im_start|>system\n{full_system}<|im_end|>")

    for msg in messages:
        role = msg.get("role", "user")
        content = _extract_content(msg.get("content", ""))

        if role in ("user", "tool_result"):
            parts.append(f"<|im_start|>user\n{content}<|im_end|>")
        elif role == "assistant":
            parts.append(f"<|im_start|>assistant\n{content}<|im_end|>")

    parts.append("<|im_start|>assistant")

    return "\n".join(parts)


# ─────────────────────────────────────────────
# 도구 설명 포맷 (XML)
# ─────────────────────────────────────────────
def _format_tool_descriptions(tools: list[dict]) -> str:
    """도구 설명을 XML 형식으로 포맷한다."""
    parts = []
    for tool in tools:
        func = tool.get("function", tool)
        name = func.get("name", "unknown")
        description = func.get("description", "")
        params = func.get("parameters", {})

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
    """content가 str 또는 list[block]일 수 있으므로 텍스트를 추출한다."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    texts.append(block.get("text", ""))
                elif block.get("type") == "tool_use":
                    tool_data = {
                        "name": block.get("name"),
                        "input": block.get("input"),
                    }
                    tool_json = json.dumps(tool_data, ensure_ascii=False)
                    texts.append(f"<tool_use>\n{tool_json}\n</tool_use>")
            elif isinstance(block, str):
                texts.append(block)
        return "\n".join(texts)
    return str(content)
