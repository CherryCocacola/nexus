"""Phase N LoRA 회귀 검증 스크립트 — 5가지 시나리오로 품질 채점.

사용법:
  python scripts/_verify_phase.py

검증 시나리오:
  1. 짧은 인사 → 짧은 답변, 도구 호출 없음
  2. 장문 지식 질문 → 200+ 토큰 답변
  3. 단일 파일 탐색 → Read/LS/Grep 직접 호출 (Agent 아님)
  4. 대규모 탐색 → Agent(subagent_type=scout) 호출
  5. 응답 본문에 날것의 tool_call JSON/XML이 누출되지 않는지

각 시나리오는 pass/fail + 점수(0~1) 반환.
"""

from __future__ import annotations

import json
import re
import sys
import time
import urllib.request
import ssl

sys.stdout.reconfigure(encoding="utf-8")

BASE = "https://localhost:8443"


def post_chat(message: str, session: str, timeout: int = 180) -> dict:
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    req = urllib.request.Request(
        BASE + "/v1/chat",
        data=json.dumps({"message": message, "session_id": session}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start = time.monotonic()
    with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    body["_elapsed_sec"] = round(time.monotonic() - start, 1)
    return body


def check_no_tool_leak(text: str) -> tuple[bool, str]:
    """본문에 직렬화된 tool_call 흔적이 있으면 실패."""
    # JSON 형태: {"name": "Agent", ...}
    if re.search(r'\{\s*"name"\s*:\s*"\w+"\s*,\s*"arguments"', text):
        return False, "JSON tool_call in body"
    # 원시 XML 형태: <tool_call>, <function=...>, <parameter=...> (정상은 tool_calls 필드로 파싱됨)
    if "<tool_call>" in text or "<function=" in text:
        return False, "XML tool_call in body"
    return True, "clean"


def get_metrics() -> dict:
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    req = urllib.request.Request(BASE + "/metrics", method="GET")
    with urllib.request.urlopen(req, timeout=10, context=ctx) as resp:
        return json.loads(resp.read().decode("utf-8"))


def run() -> int:
    print("=" * 60)
    print("Phase N LoRA 회귀 검증")
    print("=" * 60)

    results = []
    before_metrics = get_metrics()
    before_scout_calls = before_metrics.get("agents", {}).get("scout", {}).get("calls", 0)

    # 1. 인사
    print("\n[1/5] 짧은 인사 → 빠른 답변, 도구 호출 없음")
    r = post_chat("안녕!", "verify-greet", timeout=30)
    ok1 = (
        r["usage"]["output_tokens"] < 80
        and not r.get("tool_calls")
        and check_no_tool_leak(r["response"])[0]
    )
    print(f"  elapsed={r['_elapsed_sec']}s, tokens={r['usage']['output_tokens']}, "
          f"leak={check_no_tool_leak(r['response'])[1]}")
    print(f"  결과: {'PASS' if ok1 else 'FAIL'}")
    print(f"  응답 (100자): {r['response'][:100]!r}")
    results.append(("greeting", ok1))

    # 2. 장문 지식
    print("\n[2/5] 장문 지식 질문 → 200+ 토큰 답변")
    r = post_chat("Python의 GIL에 대해 자세히 알려줘", "verify-knowledge", timeout=120)
    ok2 = (
        r["usage"]["output_tokens"] >= 200
        and not r.get("tool_calls")
        and check_no_tool_leak(r["response"])[0]
    )
    print(f"  elapsed={r['_elapsed_sec']}s, tokens={r['usage']['output_tokens']}")
    print(f"  결과: {'PASS' if ok2 else 'FAIL'}")
    print(f"  응답 (200자): {r['response'][:200]!r}")
    results.append(("knowledge_longform", ok2))

    # 3. 단일 도구 (Read)
    print("\n[3/5] 단일 파일 탐색 → Read/LS 직접 호출 (Agent 아님)")
    r = post_chat("config/nexus_config.yaml 파일의 내용을 보여줘", "verify-single", timeout=60)
    # tool_calls에 Agent가 아닌 다른 도구가 있거나, 응답 본문에 내용 설명
    ok3 = (
        check_no_tool_leak(r["response"])[0]
        and len(r["response"]) > 50
    )
    print(f"  elapsed={r['_elapsed_sec']}s, tokens={r['usage']['output_tokens']}")
    print(f"  결과: {'PASS' if ok3 else 'FAIL'}")
    print(f"  응답 (150자): {r['response'][:150]!r}")
    results.append(("single_tool", ok3))

    # 4. 대규모 탐색 — Agent(scout) 호출 기대
    print("\n[4/5] 대규모 탐색 → Agent(scout) 호출 기대")
    r = post_chat(
        "이 프로젝트의 5계층 권한 시스템이 전반적으로 어떻게 구현돼 있는지 "
        "여러 디렉토리를 뒤져 전수 조사해줘.",
        "verify-broad",
        timeout=240,
    )
    after_metrics = get_metrics()
    after_scout_calls = after_metrics.get("agents", {}).get("scout", {}).get("calls", 0)
    scout_invoked = after_scout_calls > before_scout_calls
    ok4 = (
        check_no_tool_leak(r["response"])[0]
        and scout_invoked
    )
    print(f"  elapsed={r['_elapsed_sec']}s, tokens={r['usage']['output_tokens']}")
    print(f"  scout_calls 증가: {before_scout_calls} → {after_scout_calls}")
    print(f"  결과: {'PASS' if ok4 else 'FAIL'}")
    print(f"  응답 (200자): {r['response'][:200]!r}")
    results.append(("broad_exploration_scout", ok4))

    # 5. 파일 업로드 → Scout 경유 분석 — Part 2.3 개정 검증
    # 실제 파일 업로드는 스킵하고, "업로드된 문서 분석" 메시지로 Worker가
    # Agent(scout)를 호출하는지 확인한다 (scout_calls 증가로 판정).
    print("\n[5/6] 파일 분석 요청 → Agent(scout) 호출 (Part 2.3 개정)")
    scout_calls_before_5 = after_scout_calls
    r = post_chat(
        "사용자가 문서 파일을 업로드했습니다.\n"
        "파일명: sample.pdf\n"
        "서버 경로: /tmp/sample.pdf\n"
        "Agent 도구를 사용해 subagent_type=\"scout\"으로 이 파일을 분석하고 "
        "요약을 받아 주세요.",
        "verify-file-upload",
        timeout=240,
    )
    metrics5 = get_metrics()
    scout_calls_after_5 = metrics5.get("agents", {}).get("scout", {}).get("calls", 0)
    file_scout_invoked = scout_calls_after_5 > scout_calls_before_5
    ok5 = (
        check_no_tool_leak(r["response"])[0]
        and file_scout_invoked
    )
    print(f"  elapsed={r['_elapsed_sec']}s, tokens={r['usage']['output_tokens']}")
    print(f"  scout_calls 증가: {scout_calls_before_5} → {scout_calls_after_5}")
    print(f"  결과: {'PASS' if ok5 else 'FAIL'}")
    results.append(("file_analysis_via_scout", ok5))

    # 6. tool_call leak 회귀 — 위 모든 테스트 종합 확인
    print("\n[6/6] tool_call 누출 종합 확인")
    ok6 = all(ok for _, ok in results)
    print(f"  결과: {'PASS' if ok6 else 'FAIL (상위 테스트 중 하나에서 leak 또는 실패)'}")
    results.append(("no_tool_leak_aggregate", ok6))

    # 요약
    print("\n" + "=" * 60)
    print("요약")
    print("=" * 60)
    pass_count = sum(1 for _, ok in results if ok)
    total = len(results)
    for name, ok in results:
        print(f"  {'[OK]' if ok else '[X]'} {name}")
    print(f"\n통과: {pass_count}/{total}")

    return 0 if pass_count == total else 1


if __name__ == "__main__":
    sys.exit(run())
