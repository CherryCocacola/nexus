# 업로드→파싱→생성→다운로드로 이어지는 문서 파이프라인을 배포 서버에서 확인한다.
"""
표면 회귀 — 문서 파이프라인.

왜 반복 질의를 하는가:
    1회 성공은 검증이 아니다. 같은 문서에 같은 질문을 여러 번 던져 **일관성**을 본다.
    실제로 지원되지 않는 형식으로 물었을 때 4회 중 1회는 전혀 다른 문서 내용을
    지어냈다(2026-08-20). 정상 형식에서는 흔들리지 않아야 한다.

프런트가 실제로 만드는 첨부 형식은 둘이다(web/static/index.html).
    ① 텍스트류   : "[첨부파일: 이름]\\n{본문}"
    ② 바이너리류 : "서버 경로: {경로}\\n위 서버 경로의 문서를 분석해 주세요."
테스트도 그 형식을 그대로 쓴다 — 형식이 다르면 결과도 다르기 때문이다.
"""

from __future__ import annotations

import pytest

from tests.e2e.conftest import requires_server

pytestmark = [pytest.mark.e2e, requires_server]

DOC_BODY = (
    "IDINO NOVA 표면 회귀 문서.\n"
    "핵심 지표: 재고 회전율 4.7회, 불량률 0.23%, 담당자 서현석.\n"
    "특이사항: 2026년 3분기 창고 이전 예정.\n"
)


@pytest.fixture(scope="module")
def uploaded(api, key_primary):
    """샘플 문서를 업로드하고 서버 경로를 돌려준다."""
    r = api.post(
        "/v1/upload",
        key=key_primary,
        files={"file": ("표면회귀샘플.txt", DOC_BODY.encode("utf-8"), "text/plain")},
        timeout=120,
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body.get("file_path"), f"업로드 응답에 경로가 없다: {body}"
    return body


def test_upload_stores_file(uploaded):
    """업로드는 저장까지만 한다 — 파싱은 모델이 도구로 한다(계약 확인)."""
    assert uploaded["file_path"].endswith(".txt")
    assert uploaded.get("size_bytes", 0) > 0


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("불량률이 몇 퍼센트야? 숫자만 답해줘.", "0.23"),
        ("담당자 이름만 답해줘.", "서현석"),
        ("재고 회전율 수치만 답해줘.", "4.7"),
    ],
)
def test_inline_attachment_is_quoted_correctly(api, key_primary, question, expected):
    """인라인 첨부 형식 — 프런트가 텍스트 파일에 쓰는 방식."""
    prefix = f"[첨부파일: 표면회귀샘플.txt]\n{DOC_BODY}"
    status, text = api.chat(f"{prefix}\n\n{question}", key_primary, max_tokens=200)
    assert status == 200, text
    assert expected in text, f"인용이 틀렸다: {text[:120]}"


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("불량률이 몇 퍼센트야? 숫자만 답해줘.", "0.23"),
        ("담당자 이름만 답해줘.", "서현석"),
    ],
)
def test_server_path_attachment_is_read_by_tool(
    api, key_primary, uploaded, question, expected
):
    """서버 경로 형식 — 모델이 DocumentProcess 로 직접 읽어야 답할 수 있다."""
    prefix = f"서버 경로: {uploaded['file_path']}\n위 서버 경로의 문서를 분석해 주세요."
    status, text = api.chat(f"{prefix}\n\n{question}", key_primary, max_tokens=250, timeout=420)
    assert status == 200, text
    assert expected in text, f"도구로 못 읽었거나 인용이 틀렸다: {text[:150]}"


def test_document_export_produces_downloadable_file(api, key_primary):
    """문서 생성 → 다운로드까지 이어져야 사용자가 결과물을 받는다."""
    status, text = api.chat(
        "아래 내용을 담은 docx 문서를 만들어줘. 제목은 '표면 회귀 보고'.\n"
        "- 재고 회전율 4.7회\n- 불량률 0.23%\n",
        key_primary,
        max_tokens=500,
        timeout=420,
    )
    assert status == 200, text

    # 다운로드 링크는 서버가 tool_result 에서 추출해 본문에 붙인다(모델 텍스트 아님).
    import re

    m = re.search(r"/v1/download/([A-Za-z0-9._가-힣-]+)", text)
    assert m, f"다운로드 링크가 본문에 없다: {text[:200]}"

    r = api.get(f"/v1/download/{m.group(1)}", key=key_primary, timeout=60)
    assert r.status_code == 200, f"다운로드 실패: {r.status_code}"
    assert len(r.content) > 500, "받은 파일이 비정상적으로 작다"
