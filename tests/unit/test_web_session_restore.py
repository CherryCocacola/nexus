"""
web/app.py 의 _restore_messages_from_saved — 세션 대화 이력 복원 회귀 테스트.

배경(버그):
  웹 비스트리밍 /v1/chat 멀티턴에서 1턴 assistant 응답이 Redis 대화 이력에서
  유실됐다. 근본 원인은 저장 직렬화가 user content는 평문 str, assistant content는
  ContentBlock 리스트로 "비대칭"으로 저장한 것이었다. 복원 측이 리스트를
  Message.assistant 에 넘기다 ValidationError 가 났고, 그 예외가 복원 루프 전체를
  중단시켜(이력 유실의 직접 원인) 그 뒤 메시지까지 통째로 사라졌다.

수정(테스트 대상):
  _restore_messages_from_saved 는
    - saved 의 각 항목(dict)에서 role/content 를 읽어 Message.user()/assistant()로 복원,
    - content 가 리스트(과거 오염 ContentBlock)면 type=="text" 블록 text 만 join 해 평문화,
    - content 가 비면 건너뛰고,
    - 항목 단위 try/except 로 한 항목이 깨져도 나머지는 복원(부분 복원 보장)한다.

테스트 격리:
  순수 함수라 QueryEngine/Redis/PG/vLLM 미사용. web.app import 시 모듈 레벨은
  FastAPI 인스턴스/상태 dict 생성뿐이라 외부 호출 부작용이 없다.
"""

from __future__ import annotations

from core.message import Role
from web.app import _restore_messages_from_saved


# ─────────────────────────────────────────────
# _restore_messages_from_saved 테스트
# ─────────────────────────────────────────────
class TestRestoreMessagesFromSaved:
    """직렬화된 대화 이력(dict 목록)을 Message 리스트로 복원하는 동작을 검증한다."""

    def test_restore_plain_user_and_assistant_preserves_order_and_role(self):
        """평문 user + 평문 assistant 2개가 순서/내용/role 그대로 복원되는지 확인한다."""
        # 정상 저장 형식: content 가 평문 str (수정 후의 저장 계약)
        saved = [
            {"role": "user", "content": "안녕하세요"},
            {"role": "assistant", "content": "네, 반갑습니다"},
        ]

        restored = _restore_messages_from_saved(saved, "session-001")

        assert len(restored) == 2
        # 순서 보존
        assert restored[0].role == Role.USER
        assert restored[1].role == Role.ASSISTANT
        # 내용 보존(평문 텍스트)
        assert restored[0].text_content == "안녕하세요"
        assert restored[1].text_content == "네, 반갑습니다"

    def test_restore_polluted_assistant_list_content_recovers_text(self):
        """과거 오염 데이터(assistant content 가 ContentBlock 리스트)를
        평문 텍스트로 복구하는지 확인한다 — 버그 회귀 데이터 호환 검증.
        """
        # 예전 model_dump(mode="json")가 남긴 오염 형식
        saved = [
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "홍길동"}],
            },
        ]

        restored = _restore_messages_from_saved(saved, "session-001")

        # 리스트여도 예외 없이 텍스트만 추출해 복원되어야 한다.
        assert len(restored) == 1
        assert restored[0].role == Role.ASSISTANT
        assert restored[0].text_content == "홍길동"

    def test_restore_polluted_list_with_nontext_blocks_only_keeps_text(self):
        """오염 리스트에 text 외 블록(tool_use 등)이 섞여 있어도
        text 블록의 text 만 추출하는지 확인한다.
        """
        saved = [
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "앞부분"},
                    {"type": "tool_use", "name": "Read", "input": {}},
                    {"type": "text", "text": "뒷부분"},
                ],
            },
        ]

        restored = _restore_messages_from_saved(saved, "session-001")

        assert len(restored) == 1
        # text 블록만 join — tool_use 블록은 무시된다.
        assert restored[0].text_content == "앞부분뒷부분"

    def test_restore_skips_empty_content_items(self):
        """content 가 빈 항목(빈 문자열/빈 리스트)은 건너뛰는지 확인한다."""
        saved = [
            {"role": "user", "content": ""},  # 빈 문자열 → 건너뜀
            {"role": "assistant", "content": []},  # 빈 리스트 → 평문화하면 "" → 건너뜀
            {"role": "user", "content": "실제 질문"},  # 이것만 복원
        ]

        restored = _restore_messages_from_saved(saved, "session-001")

        assert len(restored) == 1
        assert restored[0].text_content == "실제 질문"

    def test_restore_ignores_non_user_assistant_roles(self):
        """role 이 user/assistant 가 아니면(system/tool_result 등) 무시하는지 확인한다."""
        saved = [
            {"role": "system", "content": "시스템 지시"},  # 무시
            {"role": "tool_result", "content": "도구 결과"},  # 무시
            {"role": "user", "content": "유효 질문"},  # 복원
        ]

        restored = _restore_messages_from_saved(saved, "session-001")

        assert len(restored) == 1
        assert restored[0].role == Role.USER
        assert restored[0].text_content == "유효 질문"

    def test_restore_partial_recovery_when_item_broken(self):
        """중간에 깨진 항목이 있어도 예외 없이 나머지가 복원되는지 확인한다(부분 복원).

        깨진 항목으로 'role 누락' 과 'content 타입이 잘못됨(정수)' 두 가지를 섞는다.
        예전 코드는 루프 전체를 하나의 try/except 로 감싸 한 항목이 깨지면 그 뒤가
        모두 유실됐지만, 수정된 헬퍼는 항목 단위로 예외를 격리해야 한다.
        """
        saved = [
            {"role": "user", "content": "첫 질문"},  # 정상 → 복원
            {"content": "role 누락 항목"},  # role 없음 → 무시(예외 아님)
            {"role": "assistant", "content": 12345},  # content 가 int → 깨짐, 건너뜀
            {"role": "assistant", "content": "마지막 답변"},  # 정상 → 복원
        ]

        restored = _restore_messages_from_saved(saved, "session-001")

        # 깨진 항목들이 있어도 정상 항목 2개는 살아남아야 한다.
        assert len(restored) == 2
        assert restored[0].text_content == "첫 질문"
        assert restored[1].text_content == "마지막 답변"
        assert restored[1].role == Role.ASSISTANT

    def test_restore_none_input_returns_empty_list(self):
        """None 입력 시 빈 리스트를 반환하는지 확인한다."""
        restored = _restore_messages_from_saved(None, "session-001")
        assert restored == []

    def test_restore_empty_list_returns_empty_list(self):
        """빈 리스트 입력 시 빈 리스트를 반환하는지 확인한다."""
        restored = _restore_messages_from_saved([], "session-001")
        assert restored == []
