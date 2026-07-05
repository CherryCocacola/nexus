"""권한 시스템 패키지 — 5계층(Layer) 권한 파이프라인 기반 도구 실행 통제.

이 패키지는 Nexus에서 "모델이 요청한 도구 호출을 실제로 실행해도 되는가"를
판정하는 권한(Permission) 시스템의 핵심 모듈들을 모아 둔 곳이다.
Query Loop(오케스트레이터)이 도구를 실행하기 직전에 이 패키지의 파이프라인을
반드시 통과시켜, 위험한 파일 접근·명령 실행·네트워크 호출 등을 사전에 차단한다.

핵심 설계 원칙은 Fail-Closed(안전 우선)다. 즉 5개 레이어를 "모두" 통과해야만
ALLOW(허용)가 되고, 하나라도 막히면 DENY(거부) 또는 ASK(사용자 확인)로 떨어진다.

5계층 파이프라인 개요(자세한 구현은 pipeline.py 참조):
  - Layer 1: DenyRuleFilter — 도구 자체를 목록에서 제거(모델이 아예 호출 불가)
  - Layer 2: File/Bash PermissionChecker — 경로 순회·위험 명령어 등 정적 분석
  - Layer 3: CanUseToolHandler — PermissionMode별 사용자 확인 처리
  - Layer 4: HookPermissionChecker — Hook 기반 승인/차단(BLOCK 시 Layer 3 생략)
  - Layer 5: apply_context_resolution() — 모드별 최종 보정(BYPASS/PLAN 등)

주요 구성 모듈:
  - types.py       — 권한 도메인 타입 정의. PermissionMode, ToolCategory,
                     PermissionRule, Allow/Deny/AskDecision, PermissionContext,
                     PermissionAuditEntry 등 Pydantic v2 모델·Enum 모음.
  - pipeline.py    — PermissionPipeline 클래스. 위 5계층을 순서대로 실행해
                     최종 PermissionBehavior(ALLOW/DENY/ASK)를 산출하는 조립부.
  - mode_mapping.py — PermissionMode ↔ ToolCategory 동작 매핑(MODE_BEHAVIOR_MAP 등).
                     "어떤 모드에서 어떤 도구 범주를 허용/거부/확인할지"의 규칙표.

의존성 방향(architecture.md P2 준수):
  core/tools/ → core/permission/ 방향으로만 의존한다. 즉 도구·오케스트레이터가
  이 패키지를 가져다 쓰며, 이 패키지가 상위 계층을 역참조하지 않는다.

참고: 이 __init__.py 자체는 별도 심볼을 re-export 하지 않는다. 사용하는 쪽은
`from core.permission.pipeline import PermissionPipeline` 처럼 하위 모듈을
직접 import 한다. (패키지 표식 + 문서 역할만 담당)

작성자: 이현수 / 작성일: 2026-07-05
"""
