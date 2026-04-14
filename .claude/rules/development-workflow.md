# Nexus Development Workflow

## 구현 순서 원칙

사양서(Chapter 22)에 정의된 8.5단계 개발 로드맵을 따른다.
하위 Phase의 코드가 완성되지 않은 상태에서 상위 Phase를 구현하지 않는다.

```
Phase 0.5: Foundation    — 프로젝트 구조, CI, 설정 시스템
Phase 1.0: Model Layer   — GPU 감지, 추론 클라이언트
Phase 2.0: Tool System   — 24개 도구, Registry, Executor
Phase 3.0: Orchestrator  — Query loop, 스트리밍, 컨텍스트 관리
Phase 4.0: Security      — 5계층 권한, sandbox
Phase 5.0: Thinking & Memory — (Phase 3.0과 병렬 가능)
Phase 6.0: Training      — QLoRA 학습 파이프라인
Phase 7.0: Interface     — CLI(Rich), 배포
Phase 8.0: Integration   — 통합 테스트, 폴리싱
```

## 모듈 구현 체크리스트

새 모듈을 구현할 때 아래를 순서대로 완료한다:

1. [ ] 사양서에서 해당 챕터를 읽고 요구사항 파악
2. [ ] Pydantic 데이터 모델(types, schemas) 먼저 정의
3. [ ] ABC/인터페이스 정의 (BaseTool, ModelProvider 등)
4. [ ] 핵심 로직 구현 (한글 주석 포함)
5. [ ] 에러 처리 추가 (tool_use_error 래핑 패턴)
6. [ ] 단위 테스트 작성
7. [ ] ruff check + format
8. [ ] 통합 테스트 (다른 모듈과의 연동)

## 코드 리뷰 기준

Pull Request 또는 코드 리뷰 시 아래 기준으로 검증한다:

### 필수 통과 (MUST)
- [ ] 4-Tier AsyncGenerator 체인 무결성 유지
- [ ] 의존성 방향 위반 없음
- [ ] StreamEvent / Message 타입 올바르게 사용
- [ ] Fail-closed 기본값 유지 (명시적 완화만 허용)
- [ ] 에어갭 규칙 준수 (외부 네트워크 호출 없음)
- [ ] 한글 주석 작성됨
- [ ] 테스트 존재 및 통과

### 권장 (SHOULD)
- [ ] Pydantic v2 모델 사용
- [ ] frozen/immutable 객체 패턴 적용
- [ ] JSONL 로깅 형식 준수
- [ ] Factory method 패턴 사용 (Message.user(), ToolResult.success() 등)

## 세션 간 일관성 유지

- 현재 구현 중인 Phase와 완료된 모듈을 명확히 기록
- 다음 세션에서 이어서 작업할 수 있도록 진행 상황을 구체적으로 남김
- 미완성 코드에는 `# TODO(nexus): 설명` 주석을 남김
- 설계 결정의 이유(why)를 주석이나 커밋 메시지에 기록
