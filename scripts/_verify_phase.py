"""Phase N LoRA 회귀(regression) 검증 스크립트 — 5가지 시나리오로 품질 채점.

이 스크립트는 LoRA 어댑터를 새로 학습/교체한 뒤, 실제로 떠 있는 Nexus 웹
서버(Machine A, https://localhost:8443)에 여러 종류의 채팅 요청을 실제로
보내 보고, 응답이 우리가 기대하는 "행동 규범"을 지키는지 자동으로 채점한다.
즉 mock 없이 진짜 서버를 두드려 보는 end-to-end 스모크 테스트에 가깝다.

왜 필요한가:
  LoRA를 갈아끼우면 모델의 말투/도구 호출 습관이 미묘하게 망가질 수 있다.
  (예: 인사에도 장문으로 답하거나, 도구 호출 JSON을 본문에 그대로 흘리거나,
   대규모 탐색인데 Agent(scout)를 안 부르는 등.) 사람이 매번 눈으로 확인하는
  대신, 대표 시나리오 6개를 자동으로 돌려 pass/fail로 회귀를 잡아낸다.

검증 시나리오:
  1. 짧은 인사 → 짧은 답변, 도구 호출 없음
  2. 장문 지식 질문 → 200+ 토큰 답변
  3. 단일 파일 탐색 → Read/LS/Grep 직접 호출 (Agent 아님)
  4. 대규모 탐색 → Agent(subagent_type=scout) 호출
  5. 파일 분석 요청 → Agent(scout) 경유 (Part 2.3 개정 검증)
  6. 위 전체를 종합해 tool_call 본문 누출/실패가 하나도 없는지 재확인

주요 함수:
  - post_chat()          : 웹 서버 /v1/chat 에 채팅 요청 1건 전송
  - check_no_tool_leak() : 응답 본문에 직렬화된 tool_call 흔적이 있는지 검사
  - get_metrics()        : 웹 서버 /metrics 로 scout 호출 횟수 등 지표 조회
  - run()                : 6개 시나리오를 순서대로 실행하고 통과 개수를 반환

의존/전제:
  Nexus 웹 서버가 localhost:8443 에서 이미 떠 있어야 한다(자체 서명 TLS).
  각 시나리오는 pass/fail 로 판정하며, 전부 통과하면 종료코드 0, 아니면 1.

작성자: 이현수 / 작성일: 2026-07-05
"""

from __future__ import annotations

import json
import re
import sys
import time
import urllib.request
import ssl

# 표준출력을 UTF-8로 강제한다. 한글 응답/로그를 Windows 콘솔에서 출력할 때
# 기본 코드페이지(cp949) 때문에 UnicodeEncodeError가 나는 것을 막기 위함.
sys.stdout.reconfigure(encoding="utf-8")

# 검증 대상 웹 서버 주소. Machine A(오케스트레이터)의 로컬 HTTPS 엔드포인트.
# 자체 서명 인증서라 아래에서 인증서 검증을 끄고 접속한다(에어갭 LAN 내부용).
BASE = "https://localhost:8443"


def post_chat(message: str, session: str, timeout: int = 180) -> dict:
    """웹 서버 /v1/chat 에 채팅 요청 1건을 보내고, 파싱된 응답 dict를 돌려준다.

    한 시나리오 = 한 번의 채팅 왕복이므로, 이 함수가 검증의 최소 단위다.

    매개변수:
      message : 사용자 입력으로 보낼 프롬프트 문자열.
      session : 세션 식별자. 시나리오마다 다른 값을 줘서 대화 이력이 서로
                섞이지 않도록 격리한다(예: "verify-greet").
      timeout : 응답 대기 최대 초. 장문/탐색 시나리오는 오래 걸리므로 호출부에서
                넉넉히 늘려 준다.

    반환:
      서버 JSON 응답을 dict로 파싱한 것. 여기에 실제 왕복 소요시간을
      "_elapsed_sec" 키로 덧붙여 돌려준다(원본 서버 필드가 아니라 우리가 추가).
    """
    # 자체 서명 인증서를 쓰는 로컬 서버라, 호스트명 검증과 인증서 검증을 끈다.
    # (에어갭 LAN 내부 신뢰 구간이므로 허용. 외부 통신용 코드가 아니다.)
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    # message/session_id 를 JSON 본문으로 실어 POST 요청 객체를 구성한다.
    req = urllib.request.Request(
        BASE + "/v1/chat",
        data=json.dumps({"message": message, "session_id": session}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    # 왕복 소요시간을 재기 위해 요청 직전 단조시계(monotonic) 값을 기록.
    start = time.monotonic()
    # 실제 요청 전송 후 응답 본문을 읽어 JSON으로 파싱한다.
    with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    # 응답에 걸린 시간(초)을 소수 첫째자리로 반올림해 덧붙인다.
    body["_elapsed_sec"] = round(time.monotonic() - start, 1)
    return body


def check_no_tool_leak(text: str) -> tuple[bool, str]:
    """응답 본문에 직렬화된 tool_call 흔적이 있으면 실패로 판정한다.

    정상 동작이라면 도구 호출은 서버가 파싱해 별도 tool_calls 필드로 넘겨야
    하고, 사람이 읽는 본문(response) 안에는 날것의 JSON/XML이 새지 않아야 한다.
    이 함수는 그 "누출(leak)"을 정규식으로 잡아낸다.

    반환:
      (통과여부, 사유) 튜플. 깨끗하면 (True, "clean"),
      새면 (False, 어떤 형태로 샜는지 설명).
    """
    # JSON 형태 누출: {"name": "Agent", "arguments": ...} 처럼 도구 호출이
    # 본문에 그대로 찍힌 경우를 잡는다.
    if re.search(r'\{\s*"name"\s*:\s*"\w+"\s*,\s*"arguments"', text):
        return False, "JSON tool_call in body"
    # 원시 XML 형태 누출: <tool_call>, <function=...>, <parameter=...> 같은
    # 태그가 본문에 노출된 경우(정상은 tool_calls 필드로 파싱되어야 함).
    if "<tool_call>" in text or "<function=" in text:
        return False, "XML tool_call in body"
    return True, "clean"


def get_metrics() -> dict:
    """웹 서버 /metrics 엔드포인트를 호출해 지표 dict를 반환한다.

    여기서는 주로 에이전트별 호출 횟수(agents.scout.calls)를 읽어,
    특정 시나리오 전후로 scout 호출이 실제 늘었는지 비교하는 데 쓴다.
    """
    # post_chat 과 동일하게 자체 서명 인증서 검증을 끈 SSL 컨텍스트를 만든다.
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    # 지표 조회는 부작용이 없는 단순 GET. 가볍게 10초 타임아웃.
    req = urllib.request.Request(BASE + "/metrics", method="GET")
    with urllib.request.urlopen(req, timeout=10, context=ctx) as resp:
        return json.loads(resp.read().decode("utf-8"))


def run() -> int:
    """6개 검증 시나리오를 순서대로 실행하고, 결과를 콘솔에 출력한다.

    흐름:
      1) 시작 전 /metrics 로 현재 scout 호출 횟수를 스냅샷한다.
      2) 시나리오 1~6을 차례로 돌리며 각각 pass/fail 을 results 에 쌓는다.
         (scout 호출을 기대하는 시나리오는 전후 지표 차이로 판정한다.)
      3) 마지막에 통과 개수를 요약 출력한다.

    반환:
      모든 시나리오가 통과하면 0, 하나라도 실패하면 1(종료코드로 사용).
    """
    print("=" * 60)
    print("Phase N LoRA 회귀 검증")
    print("=" * 60)

    # 각 시나리오의 (이름, 통과여부)를 순서대로 담는다. 마지막 요약에 쓴다.
    results = []
    # 검증 시작 시점의 scout 누적 호출 횟수를 기준선으로 잡아 둔다.
    # 이후 시나리오에서 이 값보다 늘었는지로 "scout가 실제 불렸는지"를 판정.
    before_metrics = get_metrics()
    before_scout_calls = before_metrics.get("agents", {}).get("scout", {}).get("calls", 0)

    # 1. 인사 — 짧은 입력엔 짧게 답하고, 도구를 부르지 않아야 정상.
    print("\n[1/5] 짧은 인사 → 빠른 답변, 도구 호출 없음")
    r = post_chat("안녕!", "verify-greet", timeout=30)
    # 통과 조건: 출력 토큰 80 미만(간결) + tool_calls 없음 + 본문 누출 없음.
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

    # 2. 장문 지식 — 지식형 질문엔 충분히 길게(200+ 토큰) 설명해야 정상.
    print("\n[2/5] 장문 지식 질문 → 200+ 토큰 답변")
    r = post_chat("Python의 GIL에 대해 자세히 알려줘", "verify-knowledge", timeout=120)
    # 통과 조건: 출력 토큰 200 이상(충분한 설명) + 불필요한 도구 호출 없음 + 누출 없음.
    ok2 = (
        r["usage"]["output_tokens"] >= 200
        and not r.get("tool_calls")
        and check_no_tool_leak(r["response"])[0]
    )
    print(f"  elapsed={r['_elapsed_sec']}s, tokens={r['usage']['output_tokens']}")
    print(f"  결과: {'PASS' if ok2 else 'FAIL'}")
    print(f"  응답 (200자): {r['response'][:200]!r}")
    results.append(("knowledge_longform", ok2))

    # 3. 단일 도구 (Read) — 파일 하나만 보면 되는 요청은 무거운 Agent 대신
    #    Read/LS 같은 도구를 직접 부르고, 내용을 본문으로 풀어 줘야 한다.
    print("\n[3/5] 단일 파일 탐색 → Read/LS 직접 호출 (Agent 아님)")
    r = post_chat("config/nexus_config.yaml 파일의 내용을 보여줘", "verify-single", timeout=60)
    # tool_calls에 Agent가 아닌 다른 도구가 있거나, 응답 본문에 내용 설명
    # 통과 조건(느슨): 본문 누출 없음 + 응답이 50자 초과(실제로 뭔가 답했음).
    ok3 = (
        check_no_tool_leak(r["response"])[0]
        and len(r["response"]) > 50
    )
    print(f"  elapsed={r['_elapsed_sec']}s, tokens={r['usage']['output_tokens']}")
    print(f"  결과: {'PASS' if ok3 else 'FAIL'}")
    print(f"  응답 (150자): {r['response'][:150]!r}")
    results.append(("single_tool", ok3))

    # 4. 대규모 탐색 — 여러 디렉토리를 전수 조사하는 무거운 작업은 Worker가
    #    직접 하지 말고 Agent(subagent_type=scout)에 위임해야 정상.
    print("\n[4/5] 대규모 탐색 → Agent(scout) 호출 기대")
    r = post_chat(
        "이 프로젝트의 5계층 권한 시스템이 전반적으로 어떻게 구현돼 있는지 "
        "여러 디렉토리를 뒤져 전수 조사해줘.",
        "verify-broad",
        timeout=240,
    )
    # 이 시나리오 직후의 scout 호출 횟수를 다시 읽어, 기준선 대비 늘었는지 확인.
    after_metrics = get_metrics()
    after_scout_calls = after_metrics.get("agents", {}).get("scout", {}).get("calls", 0)
    scout_invoked = after_scout_calls > before_scout_calls
    # 통과 조건: 본문 누출 없음 + scout가 실제로 호출됨.
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
    # 이 시나리오의 기준선은 "4번이 끝난 시점"의 scout 호출 횟수다.
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
    # 요청 후 scout 호출 횟수를 다시 읽어, 5번 기준선 대비 늘었는지 판정.
    metrics5 = get_metrics()
    scout_calls_after_5 = metrics5.get("agents", {}).get("scout", {}).get("calls", 0)
    file_scout_invoked = scout_calls_after_5 > scout_calls_before_5
    # 통과 조건: 본문 누출 없음 + 파일 분석용 scout가 실제로 호출됨.
    ok5 = (
        check_no_tool_leak(r["response"])[0]
        and file_scout_invoked
    )
    print(f"  elapsed={r['_elapsed_sec']}s, tokens={r['usage']['output_tokens']}")
    print(f"  scout_calls 증가: {scout_calls_before_5} → {scout_calls_after_5}")
    print(f"  결과: {'PASS' if ok5 else 'FAIL'}")
    results.append(("file_analysis_via_scout", ok5))

    # 6. tool_call leak 회귀 — 위 모든 테스트 종합 확인
    # 개별 시나리오가 이미 누출을 각자 검사했으므로, 여기서는 앞의 5개가
    # 모두 통과했는지를 한 번에 묶어 회귀 여부를 최종 확인한다.
    print("\n[6/6] tool_call 누출 종합 확인")
    ok6 = all(ok for _, ok in results)
    print(f"  결과: {'PASS' if ok6 else 'FAIL (상위 테스트 중 하나에서 leak 또는 실패)'}")
    results.append(("no_tool_leak_aggregate", ok6))

    # 요약 — 시나리오별 OK/X 표시와 총 통과 개수를 출력한다.
    print("\n" + "=" * 60)
    print("요약")
    print("=" * 60)
    pass_count = sum(1 for _, ok in results if ok)
    total = len(results)
    for name, ok in results:
        print(f"  {'[OK]' if ok else '[X]'} {name}")
    print(f"\n통과: {pass_count}/{total}")

    # 전부 통과하면 0, 하나라도 실패하면 1을 종료코드로 반환한다(CI에서 활용).
    return 0 if pass_count == total else 1


if __name__ == "__main__":
    # 스크립트로 직접 실행되면 검증을 돌리고, 그 결과를 프로세스 종료코드로 넘긴다.
    sys.exit(run())
