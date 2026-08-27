# "파일에 작성했다"는 완료형 주장을, 실제 쓰기 도구 실행 기록과 대조하는 검증기.
"""
2026-08-23 실측 사고: NOVA 가 문서를 파일에 쓰지 않고 **채팅에 출력한 뒤** 이렇게
답했다.

    "docs/TABLE_DESIGN.md 파일에 내용을 작성했습니다. 파일 존재를 확인했습니다."

Write 호출 0회, 검증 0회였다. 8회 시도 중 2회 발생했고 재시도하면 성공하니
확률적 실패다. 시스템 프롬프트의 "열지 않은 것을 읽었다고 말하지 마라" 지시가
있는데도 뚫렸으므로, 프롬프트가 아니라 **코드로** 대조해야 한다.

기존 execution_claim.py 가 못 잡은 이유는 둘이다.
  ① 패턴이 "실행했다/테스트 통과/검증 완료"만 다뤄 **파일 작성 주장이 없었다.**
  ② 도구 결과 **개수**만 봐서, Read 2건이 있었다는 이유로 경고가 억제됐다.

그래서 이 모듈은 (a) 파일 작성 주장 패턴을 따로 두고, (b) **성공한 쓰기 도구
결과**만 센다. 시도가 아니라 성공을 세는 것이 중요하다 — 권한 거부나 에러로
실패한 Write 를 성공으로 치면, 정작 잡아야 할 케이스가 조용히 빠져나간다.

경고일 뿐 차단이 아니다. 판단은 사람이 한다.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from core.verification._markers import PATH_OR_FILENAME

logger = logging.getLogger("nexus.verification.file_claim")

# 파일을 만들었다고 **단정**하는 완료형 표현. 안내·계획형("작성하겠습니다",
# "작성하면")은 제외한다 — 아직 안 했다고 말하는 것은 거짓 주장이 아니다.
_CLAIM_PATTERNS = (
    r"(?:작성|생성|저장|기록)(?:해|하여|해서)?\s*(?:했|하였|됐|되었|완료)\w*",
    r"(?:만들|썼|썻)(?:었|)\w*습니다",
    r"파일(?:이|을|은)?\s*(?:성공적으로\s*)?(?:생성|작성|저장)(?:됐|되었|했|하였)\w*",
    r"(?:존재|생성)(?:을|를|여부를)?\s*확인(?:했|하였|됐|되었)\w*",
)
_CLAIM_RE = re.compile("|".join(_CLAIM_PATTERNS))

# ★파일 표지. 주장 문장 안에 이것이 있어야만 발화한다.
#   왜 필요한가: "계획서 작성해줘" 요청에 모델이 **채팅 본문에** 문서를 쓰고
#   "작성했습니다"로 맺는 것은 정상이다(쓰기 도구 0건이 맞다). 파일을 지목하지
#   않은 완료형까지 잡으면 그 흔한 정상 케이스가 전부 오탐이 된다.
_FILE_MARKER_RE = re.compile(
    # 확장자 목록은 apply_claim 과 공유한다 — 따로 두면 조용히 갈라진다(_markers 참조).
    rf"파일|디렉[터토]리|폴더|{PATH_OR_FILENAME}"
)

# 코드 블록은 검사 대상이 아니다 — 모델이 **작성한 코드** 안의 문자열은 주장이
# 아니다(execution_claim 과 같은 이유, 실서버에서 실측된 오탐).
_FENCED_BLOCK = re.compile(r"```.*?```", re.DOTALL)
_INLINE_CODE = re.compile(r"`[^`\n]+`")

# 파일을 만들 수 있는 도구들. Bash 를 반드시 포함한다 — 모델이 `python x.py`나
# 리다이렉션으로 파일을 만드는 것은 정상이고, 빼면 그 경우가 전부 오탐이 된다.
WRITE_TOOLS = frozenset(
    {
        "Write",
        "Edit",
        "MultiEdit",
        "NotebookEdit",
        "DocumentExport",
        "ScaffoldWeb",
        "ImageGenerate",
        "Bash",
    }
)

_MAX_QUOTES = 3
_QUOTE_CHARS = 80


def find_file_claims(answer: str) -> list[str]:
    """답변에서 "파일에 무엇을 했다"는 완료형 주장 문장을 찾는다.

    파일 표지가 없는 완료형은 제외한다(채팅 전용 산출을 오탐하지 않기 위함).

    Args:
        answer: 모델이 낸 최종 답변 텍스트.

    Returns:
        주장 문장들(원문, 중복 제거). 없으면 빈 목록.
    """
    if not answer:
        return []

    prose = _INLINE_CODE.sub(" ", _FENCED_BLOCK.sub("\n", answer))

    found: list[str] = []
    seen: set[str] = set()
    for raw in re.split(r"(?<=[.!?。])\s+|\n+", prose):
        sentence = raw.strip()
        if not sentence or sentence in seen:
            continue
        if _CLAIM_RE.search(sentence) and _FILE_MARKER_RE.search(sentence):
            seen.add(sentence)
            found.append(sentence)
    return found


def collect_successful_write_tools(messages: list[Any]) -> set[str]:
    """이번 턴에 **성공한** 쓰기 도구 이름을 모은다.

    왜 "시도"가 아니라 "성공"인가: Write 를 호출했지만 권한 거부·에러로 실패한
    턴에 "작성했습니다"라고 주장하는 것이 **정확히 잡아야 할 케이스**다. 호출
    기록만 세면 그 경우가 억제되어 거짓 음성이 된다.

    도구 결과 메시지에는 도구 **이름**이 없고 `tool_use_id` 만 있다. 그래서
    assistant 메시지의 ToolUseBlock 에서 id→name 맵을 만들고, 결과의
    tool_use_id 와 조인해 이름을 복원한다.

    Args:
        messages: 이번 턴 구간의 메시지들.

    Returns:
        성공한 쓰기 도구 이름 집합. 없으면 빈 집합.
    """
    id_to_name: dict[str, str] = {}
    for msg in messages:
        content = getattr(msg, "content", None)
        if not isinstance(content, list):
            continue
        for block in content:
            block_id = getattr(block, "id", None)
            block_name = getattr(block, "name", None)
            if block_id and block_name:
                id_to_name[block_id] = block_name

    succeeded: set[str] = set()
    for msg in messages:
        role = msg.role if isinstance(msg.role, str) else getattr(msg.role, "value", "")
        if role != "tool_result":
            continue
        if getattr(msg, "is_error", False):
            continue  # 실패한 실행은 "했다"의 근거가 못 된다
        name = id_to_name.get(getattr(msg, "tool_use_id", "") or "")
        if name in WRITE_TOOLS:
            succeeded.add(name)
    return succeeded


def build_file_claim_warning(claims: list[str], write_tools: set[str]) -> str:
    """경고 문구를 만든다. 붙일 것이 없으면 빈 문자열.

    Args:
        claims: find_file_claims 의 결과.
        write_tools: collect_successful_write_tools 의 결과.

    Returns:
        답변 뒤에 덧붙일 경고 마크다운. 경고할 것이 없으면 "".
    """
    if not claims or write_tools:
        return ""

    lines = [
        "",
        "---",
        "⚠️ **파일 확인 필요** — 파일을 만들었다고 했지만, 이번 답변에서 "
        "파일을 쓰는 도구가 성공적으로 실행된 기록이 없습니다.",
    ]
    for sentence in claims[:_MAX_QUOTES]:
        quote = sentence[:_QUOTE_CHARS]
        if len(sentence) > _QUOTE_CHARS:
            quote += "…"
        lines.append(f"> {quote}")
    lines.append("")
    lines.append("실제로 파일이 생겼는지 직접 확인해 보세요.")
    return "\n".join(lines)
