"""
내장 훅 — Built-in Hooks.

Ch.10.6 사양서 기반. 기본으로 제공되는 훅 핸들러들.

내장 훅 목록:
  1. audit_logging_hook: 모든 도구 실행을 감사 로그에 기록
  2. sensitive_path_hook: 민감한 경로(.env, .ssh 등) 접근을 차단

이 훅들은 HookManager에 자동으로 등록된다.
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

logger = logging.getLogger("nexus.hooks")

# ─────────────────────────────────────────────
# 민감 경로 패턴 — 접근을 차단할 파일 패턴
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
    감사 로깅 훅.

    모든 도구 실행(PRE/POST)을 로그에 기록한다.
    차단하지 않고 항상 CONTINUE를 반환한다.
    이 훅은 관찰만 하고 실행 흐름에 영향을 주지 않는다.

    Args:
        hook_input: 훅 입력 데이터

    Returns:
        항상 CONTINUE (관찰 전용)
    """
    timestamp = datetime.now(UTC).isoformat()

    if hook_input.event == HookEvent.PRE_TOOL_USE:
        # 도구 실행 전 — 어떤 도구를 어떤 입력으로 호출하는지 기록
        input_summary = ""
        if hook_input.tool_input:
            # 민감 정보 제거를 위해 값을 요약한다
            safe_input = {
                k: (str(v)[:100] + "..." if len(str(v)) > 100 else v)
                for k, v in hook_input.tool_input.items()
            }
            input_summary = str(safe_input)

        logger.info(
            "[감사] PRE_TOOL_USE: tool=%s, input=%s, time=%s",
            hook_input.tool_name,
            input_summary[:200],
            timestamp,
        )

    elif hook_input.event == HookEvent.POST_TOOL_USE:
        # 도구 실행 후 — 결과 요약 기록
        result_summary = ""
        if hook_input.tool_result:
            result_summary = hook_input.tool_result[:200]

        logger.info(
            "[감사] POST_TOOL_USE: tool=%s, result=%s, time=%s",
            hook_input.tool_name,
            result_summary,
            timestamp,
        )

    # 항상 CONTINUE — 이 훅은 관찰만 한다
    return HookResult(
        decision=HookDecision.CONTINUE,
        message=f"Audit logged: {hook_input.event.value}",
    )


async def sensitive_path_hook(hook_input: HookInput) -> HookResult:
    """
    민감 경로 차단 훅.

    도구 입력에 민감한 파일 경로가 포함되어 있으면 BLOCK한다.
    파일 관련 도구(Read, Write, Edit, Glob 등)에만 적용된다.

    검사 대상 입력 키: file_path, path, pattern, command

    Args:
        hook_input: 훅 입력 데이터

    Returns:
        민감 경로 감지 시 BLOCK, 아니면 CONTINUE
    """
    # PRE_TOOL_USE에서만 동작
    if hook_input.event != HookEvent.PRE_TOOL_USE:
        return HookResult(decision=HookDecision.CONTINUE)

    # 도구 입력이 없으면 건너뜀
    if not hook_input.tool_input:
        return HookResult(decision=HookDecision.CONTINUE)

    # 경로를 포함할 수 있는 입력 키들을 검사한다
    path_keys = ["file_path", "path", "pattern"]
    paths_to_check: list[str] = []

    for key in path_keys:
        value = hook_input.tool_input.get(key)
        if value and isinstance(value, str):
            paths_to_check.append(value)

    # command 키에서도 파일 경로를 추출한다
    command = hook_input.tool_input.get("command", "")
    if command and isinstance(command, str):
        # 명령어에서 파일 경로처럼 보이는 인자를 추출
        # 간단한 휴리스틱: / 또는 . 으로 시작하는 토큰
        for token in command.split():
            if any(token.endswith(p) for p in SENSITIVE_PATH_PATTERNS):
                paths_to_check.append(token)

    # 각 경로에 대해 민감 패턴 검사
    for path in paths_to_check:
        # 경로를 정규화 (역슬래시 → 슬래시)
        normalized = path.replace("\\", "/")
        basename = os.path.basename(normalized)

        for pattern in SENSITIVE_PATH_PATTERNS:
            # 파일 이름이 민감 패턴과 일치하는지 확인
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

            # 경로 구성요소에 민감 디렉토리가 포함되는지 확인
            # 예: /home/user/.ssh/config → .ssh가 경로에 포함
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

    # 민감 경로 없음 — 통과
    return HookResult(decision=HookDecision.CONTINUE)
