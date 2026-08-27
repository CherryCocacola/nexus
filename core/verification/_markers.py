# 검증기들이 공유하는 "파일 표지" 정규식 — 확장자 목록이 갈라지는 것을 막는다.
"""
core.verification._markers — 파일 표지 패턴 한 벌.

■ 왜 따로 뺐는가 (2026-08-27)
  `apply_claim.py` 와 `file_claim.py` 가 같은 목적("이 문장이 파일을 가리키는가")
  으로 각자 확장자 목록을 들고 있었고, **서로 다르게 갈라져 있었다.**

      apply_claim  jsx tsx java go rs 는 있는데 csv docx xlsx pptx hwpx 가 없다
      file_claim   그 반대다

  둘 다 "모델이 파일을 건드렸다고 주장하는가"를 판정하므로, 한쪽에만 있는 확장자는
  그쪽 검증기에서만 잡히는 조용한 구멍이 된다. 목록을 한 곳에 둬서 다음에 확장자를
  추가할 때 한 번만 고치게 한다.

  각 모듈의 **다른** 표지(예: apply_claim 의 "함수|클래스|주석")는 목적이 달라
  그대로 각자 둔다 — 여기 모으는 것은 확장자 목록뿐이다.
"""

from __future__ import annotations

# 코드·문서·설정 파일 확장자. 새 형식을 지원하면 여기만 고친다.
FILE_EXTENSIONS = (
    "md|txt|py|js|jsx|ts|tsx|json|yaml|yml|csv|sql|html|css"
    "|java|go|rs|docx|xlsx|pptx|hwpx?"
)

# 경로처럼 보이는 토큰(`src/main.py`, `backend/services`) + 확장자를 가진 파일명.
PATH_OR_FILENAME = rf"[\w./\-]+\.(?:{FILE_EXTENSIONS})|[\w-]+/[\w./-]+"
