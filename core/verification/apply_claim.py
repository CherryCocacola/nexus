# "변경을 적용했다"는 주장을, 클라이언트가 실행한 도구 결과와 대조하는 검증기.
"""
2026-08-25 실측: VSCode 플러그인 회귀 측정에서 모델이 이렇게 답했다.

    "변경이 성공적으로 적용되었습니다.
     - backend/services/x.py 파일의 A 함수 위에 한 줄 주석이 추가되었습니다."

그런데 파일은 **한 줄도 바뀌지 않았다**(주석 행 13개 → 13개). 그 턴에 모델이 받은
도구 결과는 `read_file` 과 `search_text` 뿐이었고, 둘 다 `ok: true` 였다.
**모델이 읽기 성공을 쓰기 성공으로 오독한 것이다.**

오늘 회차에서만 1,760 세션 중 43건(2.4%)이 이 패턴이었고, 43건 전부가 코딩 전용
모델로 라우팅된 세션이었다.

■ 기존 file_claim.py 가 왜 못 잡았나 — 구멍이 둘이다

  ① 주장 패턴에 "적용·반영·추가"가 없다. `작성|생성|저장|기록|만들|썼` 만 본다.
     실측 문장은 "적용되었습니다"·"추가되었습니다" 라 통과했다.

  ② 이 표면은 **클라이언트가 도구를 실행한다.** 그래서 서버가 보는 메시지에
     `role="tool_result"` 가 아예 없다. 도구 결과는 user 메시지 본문에 JSON 으로
     실려 온다. `collect_successful_write_tools()` 는 tool_result 역할만 훑으므로
     이 표면에서는 언제나 빈 집합을 돌려준다.

■ 그래서 이 모듈이 하는 일
  (a) 적용·반영·추가 계열 완료 주장을 따로 잡고,
  (b) user 메시지에 실려 온 **클라이언트 도구 결과**를 파싱해 이름과 성공 여부를 얻고,
  (c) 성공한 도구가 **읽기 전용뿐이면** 주장의 근거가 없다고 본다.

읽기 전용만으로 판정하는 것이 핵심이다. 도구 결과가 "있다"는 것만 세면 이번 사고가
그대로 빠져나간다 — read_file 이 2건 있었으니까.

■ 차단이 아니라 경고다
  판단은 사람이 한다. 그리고 이 경고는 답변 본문에 덧붙이지 않고 응답의 `warnings`
  배열로 내보낸다. 클라이언트가 게이트로 쓸 수 있고, 정상 흐름의 사용자 화면을
  더럽히지 않는다.

작성자: 이현수 / 작성일: 2026-08-25
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger("nexus.verification.apply_claim")

# 변경을 **가했다고 단정**하는 완료형 표현. 계획형("적용하겠습니다", "추가하면")은
# 제외한다 — 아직 안 했다고 말하는 것은 거짓 주장이 아니다.
_CLAIM_PATTERNS = (
    r"(?:적용|반영|추가|삽입|수정|변경|삭제|제거)(?:이|가|을|를)?\s*"
    r"(?:성공적으로\s*)?(?:됐|되었|했|하였|완료)\w*",
    r"(?:성공적으로\s*)?(?:적용|반영)(?:됐|되었)\w*",
)
_CLAIM_RE = re.compile("|".join(_CLAIM_PATTERNS))

# 파일 표지가 문장 안에 있어야 발화한다. file_claim.py 와 같은 이유 —
# "설명을 추가했습니다" 같은 대화형 완료까지 잡으면 전부 오탐이 된다.
_FILE_MARKER_RE = re.compile(
    r"파일|디렉[터토]리|폴더|함수|클래스|주석"
    r"|[\w./\-]+\.(?:md|txt|py|js|jsx|ts|tsx|json|yaml|yml|sql|html|css|java|go|rs)"
    r"|[\w-]+/[\w./-]+"
)

# 코드 블록 안의 문자열은 주장이 아니다(file_claim.py 와 동일한 실측 근거).
_FENCED_BLOCK = re.compile(r"```.*?```", re.DOTALL)
_INLINE_CODE = re.compile(r"`[^`\n]+`")

# 클라이언트 도구 이름은 플러그인마다 다르므로 **어휘가 아니라 어간**으로 판정한다.
# 이름을 열거하면 플러그인이 도구를 하나 추가할 때마다 조용히 빠져나간다.
_WRITE_HINTS = ("write", "edit", "apply", "create", "delete", "remove", "rename",
                "patch", "insert", "replace", "save", "modify",
                # 셸 실행 계열 — 리다이렉션·스크립트로 파일을 만드는 것이 정상이라
                # 쓰기로 본다(file_claim.py 가 Bash 를 포함한 것과 같은 근거).
                "command", "terminal", "shell", "bash", "exec")
# 읽기 어간이 하나라도 걸리면 읽기로 본다(안전 방향). 다만 어간을 **토큰 경계**로
# 본다 — 부분 문자열로 보면 `search_replace`·`replace_symbol_body` 처럼 실제로
# 쓰기인 도구가 읽기로 뒤집힌다(2026-08-26 리뷰 지적).
_READ_HINTS = ("read", "search", "list", "get", "find", "symbol", "grep", "stat", "open")

_MAX_QUOTES = 3
_QUOTE_CHARS = 90


def should_check_apply_claim(answer: str) -> bool:
    """이 답변을 적용 주장 검사 대상으로 볼지 판별한다.

    ■ deny-list 가 아니라 allow-list 다 (2026-08-26)
      처음에는 `final_proposal` 만 제외했다. 그런데 액션은 셋이고
      `tool_request` 에도 같은 오탐이 남았다. 프로토콜 문서 4.1절의 `VERIFYING`
      상태가 **쓰기 직후 검증 도구를 다시 요청하는 정상 경로**이고, 그 턴의
      rationale 은 자연히 "적용했으니 확인한다"가 된다.

        {"type":"tool_request","rationale":"src/main.py 를 수정했으니 확인합니다",…}
        → 이전 구현에서 APPLY_CLAIM_UNVERIFIED 발화(오탐)

      실측 정탐 68건은 **전부 chat_response** 였다. 그래서 chat_response 일 때만
      검사한다 — 정탐 손실 0으로 오탐군 하나를 통째로 없앤다.

      액션 이름을 열거해 빼는 방식은 플러그인이 액션을 추가하면 또 같은 사고를
      낸다. 도구 이름을 열거하지 않은 것과 같은 이유다.

    구조화 출력이 아닌 평문 답변(웹 UI 표면)은 판별할 수 없으므로 **검사 대상으로
    둔다** — 그쪽은 액션 스키마 자체가 없어 기존 동작이 맞다(무회귀).
    """
    if not answer:
        return False
    try:
        obj = json.loads(answer)
    except (ValueError, TypeError):
        return True  # 평문 — 기존대로 검사한다
    if not isinstance(obj, dict) or "type" not in obj:
        return True
    return obj.get("type") == "chat_response"


def is_change_proposal(answer: str) -> bool:
    """이 답변이 "변경안 제출"인지 판별한다(적용 완료 주장이 아니다).

    ■ 왜 필요한가 (2026-08-26, 플러그인 팀 실측)
      클라이언트는 변경을 `final_proposal` 로 받아 **사용자 승인 뒤 로컬에서**
      적용한다. 그러니 정상 흐름에서도 서버는 쓰기 도구 성공을 볼 수 없다.
      그 상태에서 "적용" 단어만 보고 경고를 내면 정상 제안이 전부 오탐이 된다.

      실측: 경고 222건 중 153건(69%)이 `final_proposal` 에 붙은 오탐이었다.
      이 분기를 넣으면 정밀도가 31% → 99% 가 된다. 계획형("적용하겠습니다")을
      주장에서 제외한 것과 같은 이유다 — 아직 하지 않았다고 말하는 것은 거짓
      주장이 아니고, 제안은 완료 선언이 아니다.

    구조화 출력(JSON)일 때만 판별할 수 있다. 평문이면 False 를 돌려주어 기존
    검사를 그대로 받게 한다(무회귀).
    """
    if not answer:
        return False
    try:
        obj = json.loads(answer)
    except ValueError:
        return False
    return isinstance(obj, dict) and obj.get("type") == "final_proposal"


def find_apply_claims(answer: str) -> list[str]:
    """답변에서 "변경을 적용했다"는 완료형 주장 문장을 찾는다.

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


def collect_client_tool_results(messages: list[Any]) -> list[tuple[str, bool]]:
    """user 메시지에 실려 온 **클라이언트 실행 도구 결과**를 (이름, 성공) 으로 모은다.

    이 표면에서는 도구를 클라이언트가 실행하므로 서버 메시지에 `tool_result` 역할이
    없다. 결과는 user 본문에 아래 형태의 JSON 으로 온다.

        {"iteration": 1, "results": [{"id": "call_1", "name": "read_file", "ok": true, ...}]}

    본문에는 그 앞뒤로 안내문이 섞여 있으므로 JSON 만 잘라 파싱한다. 파싱 실패는
    조용히 넘긴다 — 형식이 바뀌어도 응답을 막으면 안 된다.

    Returns:
        [(도구이름, 성공여부)] 목록. 하나도 못 찾으면 빈 목록.
    """
    out: list[tuple[str, bool]] = []
    decoder = json.JSONDecoder()

    for msg in messages:
        role = msg.role if isinstance(msg.role, str) else getattr(msg.role, "value", "")
        if role != "user":
            continue
        text = getattr(msg, "text_content", None) or getattr(msg, "content", "")
        if not isinstance(text, str) or '"results"' not in text:
            continue

        # ── 왜 raw_decode 인가 (2026-08-26 수정) ──
        # 예전에는 `{`/`}` 를 문자 단위로 세어 균형점을 찾았다. 그 방식은 **JSON
        # 문자열 리터럴 안의 중괄호를 구분하지 못한다.** 그런데 이 표면의 도구
        # 결과는 본질적으로 코드다 — `log.info("start {")` 같은 줄이 흔하다.
        #
        # 실측: 그런 줄이 든 파일을 read_file 로 읽으면
        #   27,710자 → 642ms 소요, 같은 봉투의 apply_patch 증거는 **통째로 소실**
        # 즉 (a) async 핸들러에서 await 없이 도는 코드가 서버 루프를 수백 ms 멈추고
        #     (b) 정상적으로 적용에 성공한 턴에 "근거 없음" 오탐이 붙었다.
        #
        # raw_decode 는 문자열 인식이 공짜로 따라오고, 실패 시 첫 구조 오류에서
        # 즉시 멈춘다. 성공하면 끝 위치를 알려 주므로 그 뒤부터 이어서 훑는다.
        pos = 0
        while True:
            start = text.find("{", pos)
            if start < 0:
                break
            try:
                obj, end = decoder.raw_decode(text, start)
            except ValueError:
                pos = start + 1  # 이 위치는 객체 시작이 아니다
                continue
            results = obj.get("results") if isinstance(obj, dict) else None
            if isinstance(results, list):
                for r in results:
                    if isinstance(r, dict) and r.get("name"):
                        out.append((str(r["name"]), bool(r.get("ok"))))
            # 파싱에 성공한 만큼 건너뛴다. 이전 구현은 결과를 하나라도 얻으면
            # **메시지 순회 자체를** 끊어서, 뒤 메시지의 쓰기 증거를 놓쳤다.
            pos = end
    return out


def has_write_evidence(tool_results: list[tuple[str, bool]]) -> bool:
    """성공한 도구 중 **쓰기 성격**의 것이 하나라도 있는지.

    읽기 전용만 성공했다면 "적용했다"의 근거가 못 된다 — 이번 사고가 정확히 그
    형태였다(read_file·search_text 가 ok:true, 쓰기 0건).

    이름을 열거하지 않고 어간으로 판정한다. 읽기 어간에 걸리면 쓰기 어간이 있어도
    읽기로 본다(`read_file_and_apply` 같은 이름에 속지 않기 위함).
    """
    for name, ok in tool_results:
        if not ok:
            continue
        # 토큰 경계로 쪼갠다. 부분 문자열 매칭이면 이름 중간에 우연히 든 어간에
        # 걸린다.
        parts = set(re.split(r"[^a-z0-9]+", name.lower())) - {""}
        # ── 둘 다 걸리면 쓰기가 이긴다 (2026-08-26 수정) ──
        # 예전에는 읽기가 이겼다. `read_file_and_apply` 처럼 읽기인데 쓰기 어간이
        # 섞인 이름을 막으려던 것이다. 그런데 그 이름은 **가상**이고, 반대로
        # 아래는 전부 실재하며 전부 쓰기다.
        #     search_replace · find_and_replace · replace_symbol_body
        #     insert_after_symbol · edit_symbol
        # 읽기 우선 규칙에서는 이것들이 전부 읽기로 뒤집혀, 정상적으로 적용에
        # 성공한 턴에 "근거 없음" 오탐이 붙었다. 복합어에서 조작을 결정하는 것은
        # 쓰기 동사 쪽이다("검색해서 치환한다"는 치환이다).
        if parts & set(_WRITE_HINTS):
            return True
    return False


def build_apply_claim_warning(
    claims: list[str], tool_results: list[tuple[str, bool]]
) -> str:
    """경고 문구를 만든다. 붙일 것이 없으면 빈 문자열.

    Args:
        claims: find_apply_claims 의 결과.
        tool_results: collect_client_tool_results 의 결과.

    Returns:
        `warnings` 배열에 실을 한 줄. 경고할 것이 없으면 "".
    """
    if not claims or has_write_evidence(tool_results):
        return ""

    succeeded = sorted({n for n, ok in tool_results if ok})
    if succeeded:
        basis = "이번 대화에서 성공한 도구는 " + ", ".join(succeeded[:5]) + " 뿐입니다"
    else:
        basis = "이번 대화에 성공한 도구 실행 기록이 없습니다"

    quote = claims[0][:_QUOTE_CHARS]
    if len(claims[0]) > _QUOTE_CHARS:
        quote += "…"
    more = f" (외 {len(claims) - 1}건)" if len(claims) > 1 else ""

    return (
        f"APPLY_CLAIM_UNVERIFIED: 변경을 적용했다고 답했으나 근거가 없습니다. "
        f"{basis}. 주장: \"{quote}\"{more}"
    )
