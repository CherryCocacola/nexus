# Nexus AI Agent Collaboration Rules

AI 코딩 도구(Claude Code 등)가 이 프로젝트에서 작업할 때 따라야 하는 규칙.

## 코드 작성 전

1. 수정할 파일을 반드시 먼저 읽고 기존 패턴을 파악한다
2. 해당 모듈의 의존성 방향을 확인한다 (architecture.md P2 참조)
3. 관련 테스트 파일이 있는지 확인한다
4. 3개 이상 파일 수정 시 → 한글 계획서를 먼저 제시한다

## 코드 작성 중

1. 모든 코드에 초보자도 이해할 수 있는 **한글 주석**을 작성한다
   - 함수/클래스 위에 역할과 목적 설명
   - 복잡한 로직에 단계별 설명
   - "왜(why)" 이렇게 하는지를 중심으로
2. 4-Tier AsyncGenerator 체인을 존중한다 — 우회하지 않는다
3. 데이터 모델은 Pydantic v2 BaseModel 또는 frozen dataclass로 정의한다
4. 새 도구 추가 시:
   - BaseTool ABC를 상속한다
   - `name`, `description`, `input_schema`, `check_permissions`, `call` 구현
   - behavior flag의 fail-closed 기본값을 이해하고 필요한 것만 완화
   - `core/tools/implementations/`에 배치
5. 새 권한 규칙 추가 시:
   - 해당 레이어에만 영향을 주도록 구현
   - 5개 레이어 중 어느 레이어에 속하는지 명확히 주석으로 표시
6. 설정값은 하드코딩하지 않고 `config/*.yaml`에서 로드한다
7. 로그는 `logging.getLogger("nexus.{module}")` 사용

## 코드 작성 후

1. `ruff check . && ruff format .` 실행
2. 관련 테스트 실행: `pytest tests/ -x`
3. 의존성 방향 위반 여부 확인
4. 변경 요약을 한글로 작성

## 커밋 메시지 형식

```
[module] description (Korean)

예시:
[core/tools] Read 도구 구현 — 파일 읽기 + 라인 범위 지원
[core/orchestrator] query_loop while(True) 패턴 구현
[core/permission] Layer 2 파일 경로 검증 추가
[training] QLoRA 학습 파이프라인 초기 구현
```

## 테스트 작성 규칙

- 코드를 작성/수정한 후에는 반드시 테스트를 작성하거나 갱신한다
- 테스트 파일: `tests/{unit|integration|e2e}/test_{module}.py`
- 테스트 함수: `test_{feature}_{scenario}_{expected}`
- 외부 서비스(vLLM, Redis, PostgreSQL)는 mock 처리
- AsyncGenerator 테스트는 `async for`로 모든 이벤트를 소비하여 검증

## 에어갭 준수

- 외부 네트워크를 호출하는 코드를 절대 작성하지 않는다
- pip install, npm install 등 런타임 패키지 설치 코드를 넣지 않는다
- URL은 반드시 LAN 주소(192.168.x.x, 10.x.x.x, localhost)만 사용
