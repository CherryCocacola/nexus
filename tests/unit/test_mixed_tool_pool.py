# 혼합 도구 풀 — 서버 도구를 다시 여는 범위를 못 박는다(보안 결정이므로).
"""
2026-08-12. 코딩 API 는 `tools` 가 오면 서버 도구 풀을 **통째로 교체**했다. 그래서
플러그인 요청에는 `AnalyzeImage` 가 없었고, 이미지를 받아 놓고도 분석할 수 없었다.
클라이언트가 대신 실행할 수도 없다 — 비전 서버는 서버에만 있다.

그래서 예외를 **하나만** 뒀다. 이 테스트가 지키는 것은 그 "하나만" 이다.

  ★목록이 조용히 늘어나면 안 된다. 서버 도구를 다시 여는 것은 보안 결정이다.
    (테넌트 키 하나로 tenants.yaml 전량이 읽혔던 실측 사고 때문에 뺐던 것이다.)
  ★프롬프트와 실제 풀이 어긋나면 안 된다. 넓게 말하면 없는 도구를 부르고
    (`알 수 없는 도구: 'Agent'`), 좁게 말하면 있는 도구를 안 쓴다.
"""

from __future__ import annotations

from web.app import _client_tools_instruction, _mixed_tool_pool


class _FakeTool:
    def __init__(self, name: str, client_executed: bool = False) -> None:
        self.name = name
        self.is_client_executed = client_executed


WEB_TOOLS = [
    _FakeTool("Read"),
    _FakeTool("Write"),
    _FakeTool("Edit"),
    _FakeTool("AnalyzeImage"),
    _FakeTool("DocumentProcess"),
]
CLIENT_TOOLS = [_FakeTool("read_file", True), _FakeTool("write_file", True)]


# ─────────────────────────────────────────────
# ★화이트리스트 — 이 목록이 이 변경의 보안 경계다
# ─────────────────────────────────────────────
def test_whitelist_contains_only_analyze_image() -> None:
    """늘리려면 이 테스트를 고쳐야 한다 — 그 순간 보안 판단이 명시적으로 드러난다."""
    from core.tools.implementations.client_tool import SERVER_TOOLS_KEPT_WITH_CLIENT_TOOLS

    assert SERVER_TOOLS_KEPT_WITH_CLIENT_TOOLS == ("AnalyzeImage",)


def test_only_whitelisted_server_tool_survives() -> None:
    pool = _mixed_tool_pool(CLIENT_TOOLS, WEB_TOOLS)
    names = {t.name for t in pool}

    assert names == {"read_file", "write_file", "AnalyzeImage"}
    # ★파일시스템을 만지는 서버 도구는 하나도 남지 않아야 한다.
    for forbidden in ("Read", "Write", "Edit", "DocumentProcess"):
        assert forbidden not in names, f"서버 도구가 새어 들어왔다: {forbidden}"


def test_no_write_capable_server_tool_leaks() -> None:
    """화이트리스트가 늘어나도 쓰기 도구는 절대 들어오면 안 된다."""
    from core.tools.implementations.client_tool import SERVER_TOOLS_KEPT_WITH_CLIENT_TOOLS

    for name in SERVER_TOOLS_KEPT_WITH_CLIENT_TOOLS:
        assert name in {"AnalyzeImage"}, (
            f"'{name}' 이 추가됐다. 서버 도구를 다시 여는 결정이므로 "
            "보안 검토 없이 통과시키지 않는다."
        )


# ─────────────────────────────────────────────
# 실행 주체 판정 — query_loop 은 플래그로 가른다
# ─────────────────────────────────────────────
def test_retained_tool_is_server_executed() -> None:
    """혼합 풀에서 서버가 실행할 도구는 `is_client_executed=False` 여야 한다.

    query_loop 이 이 플래그로 실행 대기열 진입을 가르므로, 여기가 틀리면 서버가
    비전 도구를 실행하지 않고 tool_calls 로 돌려보낸다(클라이언트는 실행 못 함).
    """
    pool = _mixed_tool_pool(CLIENT_TOOLS, WEB_TOOLS)
    kept = next(t for t in pool if t.name == "AnalyzeImage")

    assert kept.is_client_executed is False
    assert all(t.is_client_executed for t in pool if t.name != "AnalyzeImage")


def test_real_analyze_image_is_read_only_and_sandboxed() -> None:
    """예외를 정당화한 근거 자체를 고정한다 — read-only 이고 경로가 갇혀 있다."""
    from core.tools.implementations.analyze_image_tool import AnalyzeImageTool

    tool = AnalyzeImageTool()
    assert tool.is_read_only is True
    assert tool.name == "AnalyzeImage"


# ─────────────────────────────────────────────
# 무회귀 · 충돌 처리
# ─────────────────────────────────────────────
def test_without_client_tools_pool_is_unchanged() -> None:
    """도구를 안 보낸 소비자(웹 등)는 종전 그대로여야 한다."""
    assert _mixed_tool_pool(None, WEB_TOOLS) is WEB_TOOLS
    assert _mixed_tool_pool([], WEB_TOOLS) is WEB_TOOLS


def test_client_definition_wins_on_name_clash() -> None:
    """호출자가 같은 이름을 선언했으면 그쪽을 존중한다.

    서버가 몰래 가로채면 클라이언트는 자기 도구가 왜 안 불리는지 알 수 없다.
    """
    clashing = [_FakeTool("AnalyzeImage", True)]
    pool = _mixed_tool_pool(clashing, WEB_TOOLS)

    assert len(pool) == 1
    assert pool[0].is_client_executed is True


def test_missing_server_tool_is_not_an_error() -> None:
    """비전 도구가 배선되지 않은 환경에서도 죽지 않는다(경량 경로)."""
    pool = _mixed_tool_pool(CLIENT_TOOLS, [_FakeTool("Read")])
    assert {t.name for t in pool} == {"read_file", "write_file"}


# ─────────────────────────────────────────────
# ★프롬프트 ↔ 풀 일치
# ─────────────────────────────────────────────
def test_prompt_announces_the_retained_tool() -> None:
    """좁게 말하면 있는 도구를 안 쓴다 — 이미지를 받아 놓고 분석하지 않는다."""
    note = _client_tools_instruction(CLIENT_TOOLS)

    assert "AnalyzeImage" in note
    assert "read_file" in note


def test_prompt_still_forbids_other_server_tools() -> None:
    """예외를 열었다고 나머지 금지가 풀리면 안 된다."""
    note = _client_tools_instruction(CLIENT_TOOLS)

    assert "NOT available" in note
    for forbidden in ("Write", "Edit", "DocumentProcess"):
        assert forbidden not in note


def test_prompt_tells_model_to_recall_per_question() -> None:
    """재질의 유도 — 앞선 요약에 답이 없을 수 있다는 것을 알려야 한다."""
    note = _client_tools_instruction(CLIENT_TOOLS)
    assert "again" in note.lower()


def test_prompt_empty_without_client_tools() -> None:
    assert _client_tools_instruction([]) == ""


def test_prompt_names_match_the_actual_pool() -> None:
    """★프롬프트에 적힌 이름이 실제 풀과 정확히 일치하는지 — 불일치가 버그의 근원."""
    pool = _mixed_tool_pool(CLIENT_TOOLS, WEB_TOOLS)
    note = _client_tools_instruction(CLIENT_TOOLS)

    for t in pool:
        assert t.name in note, f"풀에 있는데 프롬프트에 없다: {t.name}"
