# Project Nexus

에어갭(폐쇄망) 로컬 LLM 오케스트레이션 플랫폼.
Claude Code의 아키텍처를 Python으로 재구현하여, Qwen 3.5 27B + ExaOne 7.8B로
완전 자율 운영이 가능한 AI 시스템을 구축한다.

## 기술 스택

- **언어**: Python 3.11+ (asyncio)
- **추론 서버**: vLLM (OpenAI 호환 API)
- **모델**: Qwen 3.5 27B (primary), ExaOne 7.8B (Korean auxiliary), e5-large (embedding)
- **GPU**: RTX 5090 (32GB) — INT4 양자화, H100/H200 확장 경로
- **저장소**: Redis (단기 메모리/세션), PostgreSQL + pgvector (장기 메모리)
- **UI**: Rich/Textual (터미널)
- **데이터 모델**: Pydantic v2
- **설정**: YAML
- **학습**: QLoRA/LoRA

## 핵심 아키텍처

4-Tier AsyncGenerator Chain:
```
Tier 1: QueryEngine.submit_message()  — 세션 오케스트레이터
Tier 2: query() / query_loop()        — while(True) 에이전트 턴 루프
Tier 3: query_model_streaming()       — SSE 스트림 파싱
Tier 4: with_retry()                  — 재시도 + httpx
```

## 사양서

- 기술 사양서: `user_mig/PROJECT_NEXUS_SPEC_v6.1_EN.md` (22개 챕터, ~510KB)

## 규칙

- `.claude/rules/architecture.md` — 아키텍처 원칙 (4-Tier 체인, 디렉토리, 의존성 방향)
- `.claude/rules/anti-patterns.md` — 금지 패턴 12가지
- `.claude/rules/agent-collaboration.md` — AI 에이전트 협업 규칙
- `.claude/rules/domain-model.md` — 도메인 엔티티 및 용어 정의
- `.claude/rules/testing.md` — 테스트 전략
- `.claude/rules/development-workflow.md` — 개발 워크플로우 및 Phase 순서

## 에이전트

- `.claude/agents/qa-tester.md` — 4계층 QA 검증 에이전트

## 주요 규칙 요약

1. 모든 코드에 초보자도 이해할 수 있는 **한글 주석** 작성
2. 4-Tier AsyncGenerator 체인 절대 우회 금지
3. 의존성 방향 단방향만 허용 (순환 import 금지)
4. 표준 내부 계약: OpenAI tool_calls 형식
5. Fail-closed 기본값 유지
6. 에어갭 준수: 외부 네트워크 호출 금지
7. 설정은 YAML, 로그는 JSONL
