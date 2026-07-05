"""훅(Hook) 시스템 패키지 — 도구 실행 전후에 끼어드는 확장 지점 관리.

이 패키지는 도구(Tool)가 실제로 실행되기 직전/직후 같은 특정 시점마다
사용자 정의 로직(훅)을 끼워 넣어 추가 검증·로깅·차단·변환을 수행하게 해준다.
훅은 도구의 핵심 로직을 건드리지 않고 부가 기능을 붙이는 "확장 지점"이며,
권한 파이프라인 Layer 4(HookPermissionChecker)가 이 패키지를 통해 동작한다.

핵심 구성 요소:
  - hook_manager.HookManager: 훅을 이벤트별로 등록하고 순서대로 실행하는 관리자.
    HookEvent(PRE_TOOL_USE / POST_TOOL_USE / STOP / NOTIFICATION) 시점마다 훅을
    호출하고, BLOCK(즉시 차단)·APPROVE(즉시 승인)·CONTINUE(다음 훅으로) 결정을 취합한다.
  - builtin_hooks: 기본 제공 훅 모음. 감사 로그 기록(audit_logging_hook)과
    민감 경로(.env, .ssh 등) 접근 차단(sensitive_path_hook)이 들어 있다.

설계 원칙(중요):
  - 훅 실행 중 에러가 나도 도구 실행을 막지 않는다 — 안전하게 CONTINUE로 흘려보낸다.
    (훅은 부가 기능이므로, 훅 오류가 본 기능을 죽이면 안 된다는 fail-open 정책)

이 __init__.py 자체는 패키지 경계만 선언하는 마커 역할이며, 별도 로직은 없다.
실제 구현은 같은 디렉토리의 hook_manager.py / builtin_hooks.py를 참고할 것.

작성자: 이현수 / 작성일: 2026-07-05
"""
