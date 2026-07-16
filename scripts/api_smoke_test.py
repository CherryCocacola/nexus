"""IDINO NOVA 웹 API 스모크 테스트 셋 — 112 서비스의 주요 엔드포인트를 한 번에 점검한다.

무엇을 하나:
    운영 웹(기본 192.168.21.112:8600)의 인증·기본 대화·지식질의·도구목록·문서 생성·
    업로드+분석 경로를 순서대로 호출해 200/응답/다운로드가 정상인지 빠르게 확인한다.
    배포 직후 회귀 확인, 데모 전 점검, 신규 담당자의 API 감잡기에 쓴다.

인증:
    web_auth 활성 배포라 모든 /v1/* 요청에 Authorization: Bearer <API_KEY> 헤더가 필요하다.
    API 키는 config/tenants.yaml의 테넌트 api_keys와 일치해야 한다(기본 테넌트 키를 기본값으로).

실행:
    python -m scripts.api_smoke_test
    python -m scripts.api_smoke_test --base http://192.168.21.112:8600 --key nexus-b200-test-key-001
    python -m scripts.api_smoke_test --only greet,knowledge   # 일부만

주의:
    운영 서버에 실제 요청을 보낸다(문서 생성 케이스는 exports에 파일 + tb_artifacts 행이 남는다).
    가벼운 점검이면 --only greet,knowledge 로 부작용 없는 케이스만 돌린다.
"""

from __future__ import annotations

import argparse
import io
import sys

import httpx

sys.stdout.reconfigure(encoding="utf-8")

# 기본 접속값(운영 112). CLI 인자로 덮어쓸 수 있다.
DEFAULT_BASE = "http://192.168.21.112:8600"
DEFAULT_KEY = "nexus-b200-test-key-001"  # config/tenants.yaml 기본 테넌트 키


def _headers(key: str) -> dict[str, str]:
    """모든 보호 엔드포인트에 붙일 Bearer 인증 헤더."""
    return {"Authorization": f"Bearer {key}"}


def _print_case(name: str, ok: bool, detail: str) -> None:
    """케이스 결과를 'PASS/FAIL 이름 — 상세' 한 줄로 출력한다."""
    mark = "PASS" if ok else "FAIL"
    print(f"[{mark}] {name} — {detail}")


def _small_docx() -> bytes:
    """업로드+분석 케이스에 쓸 짧은 docx 바이트를 만든다(1청크로 처리될 분량)."""
    from docx import Document

    doc = Document()
    doc.add_heading("AI 포털 도입 제안 요약", 0)
    for i in range(6):
        doc.add_heading(f"{i + 1}. 항목", level=1)
        for _ in range(4):
            doc.add_paragraph(
                "통합대학 출범 이후 학사·행정·연구 데이터 통합과 AI 포털 단일 진입점 제공을 "
                "위한 실행 방안. human-in-the-loop 원칙으로 생성만 허용하고 확정은 사람이 한다."
            )
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def run(base: str, key: str, only: set[str] | None) -> int:
    """선택된 케이스를 순서대로 실행하고 실패 수를 돌려준다(0이면 전부 통과)."""
    h = _headers(key)
    fails = 0

    def want(name: str) -> bool:
        return only is None or name in only

    with httpx.Client(timeout=420.0) as c:
        # 1) health — 인증 없이도 200. 서버 기동/버전 확인.
        if want("health"):
            try:
                r = c.get(f"{base}/metrics", headers=h)
                ok = r.status_code == 200
                _print_case("health(/metrics)", ok, f"status={r.status_code}")
                fails += 0 if ok else 1
            except Exception as e:
                _print_case("health(/metrics)", False, repr(e))
                fails += 1

        # 2) auth — 잘못된 키는 401/403으로 막혀야 한다(web_auth 동작 확인).
        if want("auth"):
            try:
                r = c.post(
                    f"{base}/v1/chat",
                    headers={"Authorization": "Bearer wrong-key"},
                    json={"message": "안녕"},
                )
                ok = r.status_code in (401, 403)
                _print_case("auth(잘못된 키 차단)", ok, f"status={r.status_code} (기대 401/403)")
                fails += 0 if ok else 1
            except Exception as e:
                _print_case("auth(잘못된 키 차단)", False, repr(e))
                fails += 1

        # 3) greet — 인사는 도구 없이 짧게 답해야 한다.
        if want("greet"):
            try:
                r = c.post(f"{base}/v1/chat", headers=h, json={"message": "안녕"})
                b = r.json()
                resp = (b.get("response") or "").strip()
                tools = [t.get("name") for t in (b.get("tool_calls") or [])]
                ok = r.status_code == 200 and len(resp) > 0 and not tools
                _print_case("greet('안녕')", ok, f"tools={tools} resp={resp[:40]!r}")
                fails += 0 if ok else 1
            except Exception as e:
                _print_case("greet('안녕')", False, repr(e))
                fails += 1

        # 4) knowledge — 자동 RAG로 사실 질의에 답한다(도구 호출 없이).
        if want("knowledge"):
            try:
                r = c.post(
                    f"{base}/v1/chat",
                    headers=h,
                    json={"message": "대한민국의 수도는 어디야? 한 단어로만."},
                )
                b = r.json()
                resp = (b.get("response") or "")
                ok = r.status_code == 200 and "서울" in resp
                _print_case("knowledge(수도=서울)", ok, f"resp={resp[:40]!r}")
                fails += 0 if ok else 1
            except Exception as e:
                _print_case("knowledge(수도=서울)", False, repr(e))
                fails += 1

        # 5) tools — Worker에게 노출된 도구 목록(mcp 포함) 조회.
        if want("tools"):
            try:
                r = c.get(f"{base}/v1/tools", headers=h)
                names = sorted(t["name"] for t in (r.json().get("tools") or []))
                ok = r.status_code == 200 and "DocumentExport" in names
                _print_case("tools(/v1/tools)", ok, f"count={len(names)}")
                fails += 0 if ok else 1
            except Exception as e:
                _print_case("tools(/v1/tools)", False, repr(e))
                fails += 1

        # 6) doc_report — 문서 생성 요청 → DocumentExport 호출 + 다운로드 링크.
        if want("doc_report"):
            try:
                r = c.post(
                    f"{base}/v1/chat",
                    headers=h,
                    json={"message": "AI 도입 효과를 정리한 짧은 보고서를 word로 작성해줘."},
                )
                b = r.json()
                dls = b.get("downloads") or []
                ok = r.status_code == 200 and len(dls) == 1 and dls[0].get("url", "").startswith("/v1/download/")
                _print_case("doc_report(DocumentExport)", ok, f"downloads={[d.get('url') for d in dls]}")
                fails += 0 if ok else 1
                # 다운로드가 실제로 내려오는지(200) 확인.
                if dls:
                    dr = c.get(f"{base}{dls[0]['url']}", headers=h)
                    ok2 = dr.status_code == 200 and len(dr.content) > 0
                    _print_case("download(파일 수신)", ok2, f"status={dr.status_code} bytes={len(dr.content)}")
                    fails += 0 if ok2 else 1
            except Exception as e:
                _print_case("doc_report(DocumentExport)", False, repr(e))
                fails += 1

        # 7) upload_analyze — 문서 업로드 → 분석→보고서(청킹 수정 회귀 확인).
        if want("upload_analyze"):
            try:
                up = c.post(
                    f"{base}/v1/upload",
                    headers=h,
                    files={
                        "file": (
                            "제안요약.docx",
                            _small_docx(),
                            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                        )
                    },
                ).json()
                path = up.get("file_path")
                r = c.post(
                    f"{base}/v1/chat",
                    headers=h,
                    json={"message": f"다음 문서를 분석해 요약 보고서를 word로 작성해줘. 문서 경로: {path}"},
                )
                b = r.json()
                tc = [t.get("name") for t in (b.get("tool_calls") or [])]
                dp = tc.count("DocumentProcess")
                dls = b.get("downloads") or []
                # 짧은 문서라 DocumentProcess 1회 + 다운로드 1개면 정상(청킹 수정 반영).
                ok = r.status_code == 200 and dp <= 2 and len(dls) == 1
                _print_case("upload_analyze(청킹 수정)", ok, f"DocumentProcess={dp} downloads={len(dls)} tools={tc}")
                fails += 0 if ok else 1
            except Exception as e:
                _print_case("upload_analyze(청킹 수정)", False, repr(e))
                fails += 1

    print("\n" + ("모든 케이스 통과 ✔" if fails == 0 else f"{fails}개 실패 ✘"))
    return fails


def main() -> None:
    """CLI 인자를 파싱해 스모크 테스트를 실행하고, 실패 수를 종료코드로 돌려준다."""
    ap = argparse.ArgumentParser(description="IDINO NOVA API 스모크 테스트 셋")
    ap.add_argument("--base", default=DEFAULT_BASE, help=f"웹 베이스 URL (기본 {DEFAULT_BASE})")
    ap.add_argument("--key", default=DEFAULT_KEY, help="Bearer API 키 (config/tenants.yaml)")
    ap.add_argument(
        "--only",
        default="",
        help="쉼표로 실행할 케이스만 지정 (health,auth,greet,knowledge,tools,doc_report,upload_analyze)",
    )
    args = ap.parse_args()
    only = {s.strip() for s in args.only.split(",") if s.strip()} or None
    print(f"=== IDINO NOVA API 스모크 테스트 — {args.base} ===")
    rc = run(args.base, args.key, only)
    sys.exit(1 if rc else 0)


if __name__ == "__main__":
    main()
