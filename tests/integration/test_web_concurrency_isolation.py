"""
동시성 격리 통합 테스트 — QueryEngine 싱글톤 동시성 결함(감사 Critical #5) 회귀 방지.

배경(수정 전 결함):
  web/app.py 가 QueryEngine 을 앱 전역 싱글톤 1개로 두고, 모든 HTTP 요청이 그 하나의
  `_messages`/`_session_id`/tenant 를 락 없이 덮어썼다. 두 사용자가 동시에 채팅하면
  서로의 대화 이력·테넌트가 뒤섞였다(멀티테넌트 프로덕션 최대 블로커).

수정:
  요청/세션별로 격리된 QueryEngine 을 조립하고(_acquire_session_engine →
  _assemble_session_engine), '같은 세션'의 동시 요청만 세션별 asyncio.Lock 으로
  직렬화한다. 서로 다른 세션은 병렬을 유지한다.

이 테스트의 설계:
  실제 모델/vLLM 없이, QueryEngine 을 가짜로 치환한다. 가짜 submit_message 는
  asyncio.Barrier 로 '두 요청이 모두 바인딩을 마친 뒤'에야 자기 상태(session_id/
  tenant)를 읽어 응답에 실어 보낸다. 따라서:
    - 격리(수정본): 각 요청이 자기 전용 엔진/컨텍스트를 보므로 A→A, B→B 로 정확.
    - 공유(구 버그): 하나의 엔진을 공유하면 뒤에 바인딩한 요청이 앞 요청의 상태를
      덮어써, 배리어 이후 둘 다 '마지막에 쓰인 값'을 읽어 응답이 뒤섞인다.
  두 번째 테스트(control)가 바로 그 '공유 시 뒤섞임'을 재현해, 이 테스트 방법론이
  원래 버그를 실제로 잡아냄을 증명한다.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest

from core.message import StreamEvent, StreamEventType
from core.tools.base import ToolUseContext
from web.app import ChatRequest, _app_state, chat


# ─────────────────────────────────────────────
# 가짜 QueryEngine — 배리어 이후 자기 컨텍스트의 session/tenant 를 응답에 싣는다.
# ─────────────────────────────────────────────
class _IsolationFakeEngine:
    """chat() 핸들러가 쓰는 표면만 구현한 가짜 엔진.

    핵심: submit_message 가 클래스 공유 배리어에서 '두 요청이 모두 도착'할 때까지
    기다린 뒤, 자기 컨텍스트(options["tenant"])와 self._session_id 를 읽는다. 상태가
    세션별로 격리돼 있으면 정확한 값을, 공유돼 있으면 마지막에 덮인 값을 읽게 된다.
    """

    # 테스트가 매번 새로 세팅하는 공유 배리어(2 parties). 두 요청을 같은 시점에
    # '상태 읽기' 직전으로 모아 경합 창을 결정적으로 만든다.
    barrier: asyncio.Barrier | None = None

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # _assemble_session_engine 는 context= 로 세션 전용 컨텍스트를 넘긴다.
        self._context = kwargs.get("context")
        self._session_id = getattr(self._context, "session_id", "")
        self._messages: list = []
        self._total_turns = 0   # 실 QueryEngine 계약(생성물 turn 기록에서 참조)

    def bind_request(
        self,
        session_id: str,
        tenant: Any | None = None,
        transcript: Any | None = None,
        restore_messages: list | None = None,
        channel: str | None = None,
    ) -> None:
        """세션/테넌트를 바인딩한다(실 QueryEngine.bind_request 계약과 동일)."""
        self._session_id = session_id
        if tenant is not None and self._context is not None:
            self._context.options["tenant"] = tenant

    def clear_messages(self) -> None:
        self._messages.clear()

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def total_turns(self) -> int:
        return self._total_turns

    async def submit_message(self, message: str):
        """배리어 이후 자기 상태를 읽어 'sid=..;tid=..' TEXT_DELTA 로 흘려보낸다."""
        # 두 요청이 모두 바인딩을 마친 뒤 동시에 상태를 읽도록 정렬한다.
        if type(self).barrier is not None:
            await type(self).barrier.wait()
        tenant = self._context.options.get("tenant") if self._context else None
        tid = getattr(tenant, "id", None)
        yield StreamEvent(
            type=StreamEventType.TEXT_DELTA,
            text=f"sid={self._session_id};tid={tid}",
        )


# ─────────────────────────────────────────────
# 가짜 테넌트 레지스트리 — body.tenant_id → SimpleNamespace(id, ...) 로 해석
# ─────────────────────────────────────────────
class _FakeTenantRegistry:
    """_resolve_tenant 가 호출하는 최소 표면(get/resolve/resolve_by_api_key)."""

    def get(self, tenant_id: str | None) -> Any:
        if not tenant_id:
            return None
        return SimpleNamespace(id=tenant_id, allowed_knowledge_sources=[])

    def resolve_by_api_key(self, api_key: str) -> Any:
        return None

    def resolve(self, tenant_id: str | None) -> Any:
        # 기본 테넌트 폴백 — 미지정 시 "default".
        return SimpleNamespace(id="default", allowed_knowledge_sources=[])


def _make_parts() -> dict:
    """_assemble_session_engine 이 읽는 최소 parts. 무거운 부품은 모두 더미."""
    return {
        "tier": SimpleNamespace(value="TIER_S"),  # ModelDispatcher 가 .value 로그
        "worker_provider": object(),
        "scout_provider": None,
        "web_tools": [],
        "scout_tools": [],
        "combined_pool": [],
        "context_manager": None,
        "memory_manager": None,
        "knowledge_retriever": None,
        "system_prompt": "SYS",
        "routing_config": None,
        "budgets": None,
        "base_options": {},
        "permission_mode": "default",
        "base_cwd": ".",
        "pe_enabled": False,
        "sessions_dir": None,
    }


@pytest.fixture
def _isolate_app_state():
    """테스트가 건드리는 _app_state 키를 스냅샷 후 원복한다(테스트 독립성)."""
    keys = [
        "web_engine_parts",
        "query_engine",
        "memory_manager",
        "tenant_registry",
        "tenant_stats",
        "session_locks_store",
        "chat_histories",
        "config",
    ]
    saved = {k: _app_state.get(k) for k in keys}
    # 세션 복원 분기(Redis)를 건너뛰도록 memory_manager 는 None.
    _app_state["memory_manager"] = None
    _app_state["tenant_registry"] = _FakeTenantRegistry()
    _app_state["tenant_stats"] = {}
    # 이전 테스트 잔여 락 저장소 제거(루프 교체 대비).
    _app_state.pop("session_locks_store", None)
    yield
    for k, v in saved.items():
        if v is None:
            _app_state.pop(k, None)
        else:
            _app_state[k] = v
    _IsolationFakeEngine.barrier = None


@pytest.mark.asyncio
@pytest.mark.integration
class TestConcurrentSessionIsolation:
    """서로 다른 두 세션의 동시 요청이 이력/tenant 를 섞지 않는지 검증한다."""

    async def test_two_sessions_concurrent_do_not_cross_contaminate(
        self, _isolate_app_state
    ) -> None:
        """세션 A/B 를 asyncio.gather 로 동시 처리 → 각 응답이 자기 session/tenant 만 본다.

        parts 경로가 활성(_acquire_session_engine 이 세션별 엔진 조립)이면, 배리어로
        경합을 강제해도 A→A, B→B 로 정확해야 한다.
        """
        _app_state["web_engine_parts"] = _make_parts()
        _IsolationFakeEngine.barrier = asyncio.Barrier(2)

        # _assemble_session_engine 내부의 lazy import 대상(QueryEngine)을 가짜로 치환.
        with patch(
            "core.orchestrator.query_engine.QueryEngine", _IsolationFakeEngine
        ):
            resp_a, resp_b = await asyncio.gather(
                chat(
                    ChatRequest(message="hi", session_id="sessA", tenant_id="A"),
                    x_tenant_id=None,
                    authorization=None,
                ),
                chat(
                    ChatRequest(message="hi", session_id="sessB", tenant_id="B"),
                    x_tenant_id=None,
                    authorization=None,
                ),
            )

        # 각 응답의 session_id 는 자기 것이어야 한다.
        assert resp_a.session_id == "sessA"
        assert resp_b.session_id == "sessB"
        # 응답 본문(가짜가 실은 관측값)도 자기 session/tenant 만 담아야 한다.
        assert resp_a.response == "sid=sessA;tid=A"
        assert resp_b.response == "sid=sessB;tid=B"

    async def test_shared_engine_control_reproduces_the_bug(
        self, _isolate_app_state
    ) -> None:
        """control: 하나의 공유 엔진(구 버그)을 쓰면 응답이 뒤섞임을 재현한다.

        이 테스트는 '방법론이 원래 버그를 실제로 잡는다'는 것을 증명한다. parts 를
        제거해 싱글톤 경로로 폴백시키면 두 요청이 같은 엔진/컨텍스트를 공유하고,
        배리어 이후 둘 다 '마지막에 바인딩된' 상태를 읽어 최소 한 응답이 상대의
        session/tenant 로 오염된다.
        """
        _app_state.pop("web_engine_parts", None)  # 싱글톤 폴백 강제
        _IsolationFakeEngine.barrier = asyncio.Barrier(2)

        # 단 하나의 공유 엔진(구 설계) — 컨텍스트도 하나만 존재.
        shared_ctx = ToolUseContext(cwd=".", session_id="", permission_mode="default")
        _app_state["query_engine"] = _IsolationFakeEngine(context=shared_ctx)

        resp_a, resp_b = await asyncio.gather(
            chat(
                ChatRequest(message="hi", session_id="sessA", tenant_id="A"),
                x_tenant_id=None,
                authorization=None,
            ),
            chat(
                ChatRequest(message="hi", session_id="sessB", tenant_id="B"),
                x_tenant_id=None,
                authorization=None,
            ),
        )

        observed = {resp_a.response, resp_b.response}
        # 격리가 됐다면 {"sid=sessA;tid=A", "sid=sessB;tid=B"} 여야 한다. 공유 엔진에서는
        # 배리어 이후 둘 다 같은(마지막) tenant/session 을 읽어 이 집합이 깨진다.
        assert observed != {"sid=sessA;tid=A", "sid=sessB;tid=B"}, (
            "공유 엔진인데도 격리된 것처럼 보임 — control 이 버그를 재현하지 못함"
        )
