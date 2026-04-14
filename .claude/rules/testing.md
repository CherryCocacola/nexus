# Nexus Test Strategy

## 테스트 구조

```
tests/
  unit/              # 개별 클래스/함수 단위 테스트
  integration/       # 모듈 간 상호작용 테스트
  e2e/               # 전체 query loop → tool 실행 → 결과 반환 시나리오
```

## 반드시 테스트해야 하는 모듈 (우선순위순)

1. **core/orchestrator/query_loop.py**: while(True) 루프, 7가지 ContinueReason 전환, max-output recovery
2. **core/tools/executor.py**: 13단계 실행 파이프라인, 에러 래핑, 타임아웃
3. **core/tools/implementations/**: 24개 도구 각각의 call() + check_permissions()
4. **core/permission/pipeline.py**: 5계층 통과/거부 시나리오
5. **core/permission/layer2_bash_permissions.py**: 30+ 위험 패턴 감지
6. **core/model/inference.py**: SSE 스트림 파싱, tool_calls 추출
7. **core/tools/registry.py**: 도구 등록, alias 조회, 이름순 정렬
8. **core/memory/**: 단기(Redis) ↔ 장기(PostgreSQL+pgvector) 승격

## 테스트 실행

```bash
pytest tests/ -v                                    # 전체
pytest tests/unit/test_query_loop.py -v              # 특정 모듈
pytest tests/ --cov=core --cov-report=html           # 커버리지
pytest tests/e2e/ -v --timeout=60                    # E2E (타임아웃 확장)
```

## 네이밍 규칙

- 파일: `test_{module_name}.py`
- 함수: `test_{feature}_{scenario}_{expected_result}`
- 예시:
  - `test_query_loop_tool_use_continues_next_turn`
  - `test_bash_permission_rm_rf_denied`
  - `test_tool_executor_timeout_returns_error`
  - `test_streaming_executor_concurrent_reads_parallel`

## AsyncGenerator 테스트 패턴

```python
async def test_query_loop_yields_stream_events():
    """query_loop이 StreamEvent를 올바른 순서로 yield하는지 검증."""
    events = []
    async for event in query_loop(messages, tools, context):
        events.append(event)

    # 이벤트 순서 검증
    assert events[0].type == EventType.TURN_START
    assert any(e.type == EventType.TEXT_DELTA for e in events)
    assert events[-1].type == EventType.TURN_END
```

## Mock 전략

| 대상 | Mock 방법 | 이유 |
|---|---|---|
| vLLM (GPU 서버) | `httpx.AsyncClient` mock + SSE 응답 fixture | Machine B 없이 테스트 |
| Redis | `fakeredis.aioredis` | 단기 메모리 테스트 |
| PostgreSQL | `pytest-asyncio` + SQLite in-memory 또는 testcontainers | 장기 메모리 테스트 |
| 파일 시스템 | `tmp_path` fixture (pytest 기본) | Read/Write/Edit 도구 테스트 |
| asyncio.Event (abort) | 직접 생성하여 주입 | 취소 시그널 테스트 |

## 테스트 금지사항

- 테스트 간 실행 순서 의존성 금지 — 각 테스트는 독립적
- 실제 GPU 서버 호출 금지 (unit/integration 테스트에서)
- 실제 파일 시스템의 프로젝트 파일 수정 금지 — 항상 tmp_path 사용
- `time.sleep()` 사용 금지 — `asyncio` 기반 타이밍은 `asyncio.wait_for` 사용

## Permission 테스트 필수 시나리오

- Layer 1: deny rule로 도구가 제거되면 모델에 스키마가 전달되지 않는지
- Layer 2: 경로 순회(../../etc/passwd), UNC 경로, .env 파일 접근 차단
- Layer 2: `rm -rf /`, `curl`, `wget` 등 위험 명령어 차단
- Layer 3: 각 PermissionMode별 ToolCategory 동작 (MODE_BEHAVIOR_MAP 전체)
- Layer 4: Hook이 BLOCK 반환 시 Layer 3 건너뛰기
- Layer 5: BYPASS 모드에서 ASK→ALLOW 변환, PLAN 모드에서 쓰기 ASK→DENY 변환
- 전체: 5개 레이어 모두 통과해야만 실행 허용
