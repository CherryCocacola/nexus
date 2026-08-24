"""
내장 훅 — Built-in Hooks (기본 제공 훅 핸들러 모음).

이 파일은 Nexus가 부팅될 때 HookManager에 기본으로 등록되는 훅(hook)
핸들러들을 정의한다. 훅이란 "도구(Tool)가 실행되기 직전/직후에 끼어들어"
관찰하거나 차단할 수 있는 콜백을 말한다. Claude Code의 PreToolUse /
PostToolUse 훅 개념을 그대로 옮긴 것으로, 사양서 Ch.10.6을 기반으로 한다.

훅이 필요한 이유:
  - 도구 실행 흐름 자체는 건드리지 않으면서, 감사(로깅)나 보안(차단)처럼
    "가로지르는 관심사(cross-cutting concern)"를 한곳에 모으기 위해서다.
  - 권한 파이프라인(Layer 4, HookPermissionChecker)이 이 훅들을 호출한다.

이 파일이 제공하는 내장 훅 2종:
  1. audit_logging_hook  — 모든 도구 실행(PRE/POST)을 로그로 남긴다.
                           절대 차단하지 않고 항상 CONTINUE를 반환한다.
  2. sensitive_path_hook — .env, .ssh 같은 민감 파일/디렉토리 접근을 감지해
                           BLOCK(차단)한다. Fail-safe 보안 장치다.

핵심 데이터 계약(core.hooks.hook_manager 에 정의됨):
  - HookInput    : 훅에 전달되는 입력(이벤트 종류, 도구 이름/입력/결과)
  - HookResult   : 훅이 돌려주는 결정(decision) + 부가 정보
  - HookEvent    : 훅 시점 열거형 — PRE_TOOL_USE / POST_TOOL_USE / STOP 등
  - HookDecision : 결정 열거형 — CONTINUE(통과) / BLOCK(차단) / APPROVE(즉시승인)

이 훅들은 HookManager에 의해 자동 등록되며, 모든 훅 핸들러는 비동기(async)
함수로 HookInput 하나를 받아 HookResult 하나를 반환하는 규약을 지킨다.

작성자: 이현수 / 작성일: 2026-07-05
"""

from __future__ import annotations

import logging
import os
from datetime import UTC, datetime

from core.hooks.hook_manager import (
    HookDecision,
    HookEvent,
    HookInput,
    HookResult,
)

# 이 모듈 전용 로거. "nexus.hooks" 네임스페이스로 통일해 로그 필터링을 쉽게 한다.
logger = logging.getLogger("nexus.hooks")

# ─────────────────────────────────────────────
# 민감 경로 패턴 — 접근을 차단할 파일/디렉토리 이름 목록
#
# sensitive_path_hook 이 이 목록을 순회하며, 도구 입력에 담긴 경로의
# 파일명(basename)이나 경로 구성요소가 이 중 하나와 일치하면 차단한다.
# 비밀번호·개인키·인증 토큰이 담긴 파일이 실수로 읽히거나 덮어써지는 것을
# 막기 위한 "블랙리스트"다. 새 민감 파일 유형이 생기면 여기에 추가하면 된다.
# ─────────────────────────────────────────────
SENSITIVE_PATH_PATTERNS: list[str] = [
    # 환경변수 파일
    ".env",
    ".env.local",
    ".env.production",
    ".env.staging",
    ".env.development",
    # SSH 관련
    ".ssh",
    "id_rsa",
    "id_ed25519",
    "authorized_keys",
    "known_hosts",
    # 인증 정보
    "credentials.json",
    "secrets.yaml",
    "secrets.yml",
    ".netrc",
    ".pgpass",
    # 인증서 / 키
    ".pem",
    ".key",
    ".p12",
    ".pfx",
    # GPG
    ".gnupg",
    # Git 인증
    ".git-credentials",
    # Docker secrets
    ".dockerconfigjson",
]


async def audit_logging_hook(hook_input: HookInput) -> HookResult:
    """
    감사 로깅 훅 (audit logging hook).

    역할:
        모든 도구 실행을 시간과 함께 로그로 남겨, 나중에 "언제 어떤 도구가
        어떤 입력으로 실행됐고 결과가 무엇이었는지"를 추적할 수 있게 한다.
        보안 감사·디버깅·사용 패턴 분석의 기초 자료가 된다.

    동작 방식:
        - 도구 실행 전(PRE_TOOL_USE)에는 도구 이름과 입력 요약을 남긴다.
        - 도구 실행 후(POST_TOOL_USE)에는 도구 이름과 결과 요약을 남긴다.
        - 이 훅은 오직 "관찰(observe)"만 한다. 실행을 막거나 입력을 바꾸지
          않으며, 어떤 경우에도 항상 CONTINUE를 반환해 다음 훅으로 넘긴다.

    Args:
        hook_input: 훅 입력 데이터. event 종류와 도구 이름/입력/결과를 담는다.

    Returns:
        HookResult: 항상 decision=CONTINUE (관찰 전용이므로 흐름을 막지 않음).
    """
    # 로그에 찍을 UTC 타임스탬프. 서버 타임존에 관계없이 일관되게 기록하려고
    # 로컬 시간이 아닌 UTC(협정 세계시)를 ISO 8601 문자열로 남긴다.
    timestamp = datetime.now(UTC).isoformat()

    if hook_input.event == HookEvent.PRE_TOOL_USE:
        # ── 도구 실행 "전" 분기 ──
        # 어떤 도구를 어떤 입력으로 호출하려는지 기록한다.
        input_summary = ""
        if hook_input.tool_input:
            # 로그가 비대해지거나 민감 정보가 통째로 노출되는 것을 막기 위해,
            # 각 값이 100자를 넘으면 앞 100자만 남기고 "..."로 잘라 요약한다.
            safe_input = {
                k: (str(v)[:100] + "..." if len(str(v)) > 100 else v)
                for k, v in hook_input.tool_input.items()
            }
            input_summary = str(safe_input)

        # 최종 로그 라인도 한 번 더 200자로 제한해 과도한 길이를 방지한다.
        logger.info(
            "[감사] PRE_TOOL_USE: tool=%s, input=%s, time=%s",
            hook_input.tool_name,
            input_summary[:200],
            timestamp,
        )

    elif hook_input.event == HookEvent.POST_TOOL_USE:
        # ── 도구 실행 "후" 분기 ──
        # 도구가 돌려준 결과를 앞 200자만 요약해 기록한다.
        result_summary = ""
        if hook_input.tool_result:
            result_summary = hook_input.tool_result[:200]

        logger.info(
            "[감사] POST_TOOL_USE: tool=%s, result=%s, time=%s",
            hook_input.tool_name,
            result_summary,
            timestamp,
        )

    # PRE/POST 어느 쪽이든(혹은 그 외 이벤트든) 항상 CONTINUE를 반환한다.
    # 이 훅은 관찰만 하므로 실행 흐름을 절대 막지 않는다.
    return HookResult(
        decision=HookDecision.CONTINUE,
        message=f"Audit logged: {hook_input.event.value}",
    )


async def sensitive_path_hook(hook_input: HookInput) -> HookResult:
    """
    민감 경로 차단 훅 (sensitive path guard hook).

    역할:
        도구가 .env, .ssh, 개인키(.pem/.key) 같은 민감 파일이나 디렉토리에
        접근하려 할 때 이를 감지해 실행을 차단(BLOCK)한다. LLM이 실수로 혹은
        악의적 유도로 비밀 정보를 읽거나 덮어쓰는 사고를 막는 보안 장치다.

    적용 범위:
        - 도구 실행 "전"(PRE_TOOL_USE)에만 동작한다. 이미 실행된 뒤(POST)에는
          막을 수 없으므로 검사할 필요가 없다.
        - 실질적으로 경로를 다루는 도구(Read, Write, Edit, Glob, Bash 등)의
          입력에서만 의미가 있다.

    검사 방법(두 가지 매칭):
        1) 파일명 매칭  : 경로의 마지막 이름(basename)이 패턴과 정확히 같거나
                          경로가 "/패턴"으로 끝나면 차단. (예: .../.env)
        2) 디렉토리 매칭: 경로를 "/"로 쪼갠 구성요소 중 하나가 패턴과 같으면
                          차단. (예: /home/user/.ssh/config → .ssh 포함)

    검사 대상 입력 키: file_path, path, pattern, 그리고 command(명령어) 안의 인자.

    Args:
        hook_input: 훅 입력 데이터. tool_input 딕셔너리에서 경로를 추출한다.

    Returns:
        HookResult: 민감 경로가 감지되면 decision=BLOCK(차단 사유 포함),
                    문제가 없으면 decision=CONTINUE.
    """
    # 실행 전 시점이 아니면 검사할 이유가 없으므로 그대로 통과시킨다.
    if hook_input.event != HookEvent.PRE_TOOL_USE:
        return HookResult(decision=HookDecision.CONTINUE)

    # 도구 입력 자체가 비어 있으면 검사할 경로가 없으니 통과.
    if not hook_input.tool_input:
        return HookResult(decision=HookDecision.CONTINUE)

    # 경로 문자열이 들어올 수 있는 대표적인 입력 키들. 여기서 값을 모아
    # paths_to_check 리스트에 쌓은 뒤 한꺼번에 민감 패턴과 대조한다.
    path_keys = ["file_path", "path", "pattern"]
    paths_to_check: list[str] = []

    for key in path_keys:
        value = hook_input.tool_input.get(key)
        # 값이 존재하고 문자열일 때만 검사 후보로 담는다(숫자/None 등은 제외).
        if value and isinstance(value, str):
            paths_to_check.append(value)

    # Bash 같은 도구는 경로를 command 문자열 안에 섞어 넘긴다. 그래서 명령어를
    # 공백으로 쪼갠 각 토큰 중 민감 패턴으로 "끝나는" 것을 골라 추가 검사한다.
    command = hook_input.tool_input.get("command", "")
    if command and isinstance(command, str):
        # 완벽한 파서가 아니라 가벼운 휴리스틱이다: 토큰이 민감 패턴 중
        # 하나로 끝나면(예: cat /root/.env → ".env"로 끝남) 후보로 담는다.
        for token in command.split():
            if any(token.endswith(p) for p in SENSITIVE_PATH_PATTERNS):
                paths_to_check.append(token)

    # 모아 둔 후보 경로들을 하나씩 민감 패턴과 대조한다.
    for path in paths_to_check:
        # OS별 경로 구분자를 통일하기 위해 역슬래시(\)를 슬래시(/)로 바꾼다.
        # 이렇게 정규화해야 Windows 경로도 동일한 로직으로 검사할 수 있다.
        normalized = path.replace("\\", "/")
        basename = os.path.basename(normalized)

        for pattern in SENSITIVE_PATH_PATTERNS:
            # (1) 파일명 매칭: 마지막 이름이 패턴과 같거나 "/패턴"으로 끝나면 차단.
            if basename == pattern or normalized.endswith(f"/{pattern}"):
                logger.warning(
                    "[보안] 민감 경로 접근 차단: tool=%s, path=%s, pattern=%s",
                    hook_input.tool_name,
                    path,
                    pattern,
                )
                return HookResult(
                    decision=HookDecision.BLOCK,
                    block_reason=(
                        f"민감한 파일 접근이 차단되었습니다: '{basename}' (패턴: {pattern})"
                    ),
                )

            # (2) 디렉토리 매칭: 경로를 "/"로 쪼갠 구성요소 중 하나가 패턴과
            # 같으면 차단. 예: /home/user/.ssh/config 에서 ".ssh"가 걸린다.
            parts = normalized.split("/")
            if pattern in parts:
                logger.warning(
                    "[보안] 민감 디렉토리 접근 차단: tool=%s, path=%s, pattern=%s",
                    hook_input.tool_name,
                    path,
                    pattern,
                )
                return HookResult(
                    decision=HookDecision.BLOCK,
                    block_reason=(
                        f"민감한 디렉토리 접근이 차단되었습니다: 경로에 '{pattern}'이(가) 포함됨"
                    ),
                )

    # 어떤 후보 경로도 민감 패턴에 걸리지 않았다 — 안전하다고 보고 통과시킨다.
    return HookResult(decision=HookDecision.CONTINUE)


# ─────────────────────────────────────────────
# 3. file_claim_stop_hook — "썼다"고 했는데 안 쓴 턴의 종료를 막는다
# ─────────────────────────────────────────────
# 왜 필요한가 (2026-08-23 실측, 재현 3회):
#   모델이 문서를 파일에 쓰지 않고 **채팅에 출력한 뒤** 이렇게 답하고 종료했다.
#     "docs/TABLE_DESIGN.md 파일에 내용을 작성했습니다. 파일 존재를 확인했습니다."
#   Write 0회, 검증 0회. 시스템 프롬프트에 "열지 않은 것을 읽었다고 말하지 마라"가
#   있는데도 뚫렸고, 8회 중 2회 발생하는 확률적 실패다.
#
#   사후 검증기(core/verification/file_claim.py)는 이 상황에 **경고만** 붙인다.
#   경고는 사람이 무시할 수 있고 자동화는 아예 못 본다. 종료 자체를 막아 모델에게
#   한 번 더 기회를 주는 것이 이 훅의 역할이다 — 탐지에서 예방으로 한 칸 올린다.
#
# 왜 STOP 시점인가:
#   PRE_TOOL_USE 로는 못 잡는다. 문제는 "도구를 부른 것"이 아니라 **안 부른 것**이라,
#   도구 호출이 없는 종료 시점에만 관측된다.

# 한 세션에서 이 훅이 종료를 막을 수 있는 최대 횟수.
#   왜 상한이 필요한가: 모델이 계속 파일을 안 쓰고 같은 주장을 반복하면 무한 루프가
#   된다. 두 번 기회를 주고도 안 되면 사람이 볼 수 있게 그냥 끝내는 편이 낫다
#   (사후 검증 경고가 답변 뒤에 붙으므로 사실이 사라지지는 않는다).
MAX_FILE_CLAIM_BLOCKS = 2


async def file_claim_stop_hook(hook_input: HookInput) -> HookResult:
    """파일을 만들었다는 주장과 실제 쓰기 도구 실행을 대조해 종료를 차단한다.

    판정 근거는 전부 호출부(query_loop)가 metadata 에 실어 준다.
      - assistant_text : 이번 턴 모델 답변 전문(주장을 찾을 대상)
      - write_tools    : 마지막 사용자 메시지 이후 **성공한** 쓰기 도구 이름들
      - block_count    : 이 세션에서 이미 차단한 횟수

    차단 조건은 셋을 모두 만족할 때다.
      ① 답변에 파일 작성 완료형 주장이 있다(파일 표지 포함 — 채팅 전용 산출 제외).
      ② 성공한 쓰기 도구 실행이 0건이다(시도가 아니라 성공. 권한 거부로 실패한
         Write 를 성공으로 치면 정작 잡아야 할 케이스가 빠진다).
      ③ 차단 상한에 아직 도달하지 않았다.

    Args:
        hook_input: STOP 이벤트 입력. metadata 에 위 세 값이 담겨 온다.

    Returns:
        BLOCK(차단 사유 포함) 또는 CONTINUE.
    """
    if hook_input.event != HookEvent.STOP:
        return HookResult(decision=HookDecision.CONTINUE)

    meta = hook_input.metadata or {}
    if meta.get("block_count", 0) >= MAX_FILE_CLAIM_BLOCKS:
        # 상한 도달 — 더 붙잡지 않는다. 사후 검증 경고가 사실을 남긴다.
        return HookResult(decision=HookDecision.CONTINUE)

    # 성공한 쓰기가 하나라도 있으면 주장이 사실일 수 있다 — 통과시킨다.
    if meta.get("write_tools"):
        return HookResult(decision=HookDecision.CONTINUE)

    # 주장 탐지는 사후 검증기와 같은 규칙을 재사용한다(두 벌 관리하면 반드시 어긋난다).
    from core.verification.file_claim import find_file_claims

    claims = find_file_claims(meta.get("assistant_text") or "")
    if not claims:
        return HookResult(decision=HookDecision.CONTINUE)

    logger.warning("STOP 차단 — 파일 작성 주장 %d건, 성공한 쓰기 0건", len(claims))
    return HookResult(
        decision=HookDecision.BLOCK,
        block_reason=(
            "파일을 만들었다고 했지만 이번 대화에서 파일을 쓰는 도구가 성공적으로 "
            f"실행된 기록이 없습니다. 주장: {claims[0][:120]}"
        ),
    )
