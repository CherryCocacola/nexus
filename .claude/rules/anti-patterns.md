# Nexus Forbidden Patterns

이 패턴들은 Project Nexus에서 절대 허용되지 않는다.

## 1. AsyncGenerator 체인 우회

```python
# BAD — Tier 2에서 Tier 4를 직접 호출
async def query_loop():
    response = await with_retry(request)  # Tier 3 건너뜀

# GOOD — 체인을 순서대로 통과
async def query_loop():
    async for event in query_model_streaming(messages):
        yield event
```

## 2. vLLM / GPU 직접 호출 (Machine A에서)

```python
# BAD — Orchestrator에서 GPU 직접 접근
import torch
result = model.generate(input_ids)

# GOOD — OpenAI 호환 API를 통해서만 접근
from core.model.inference import ModelClient
client = ModelClient(base_url=config.gpu_server.url)
async for event in client.stream(messages, tools):
    yield event
```

## 3. StreamEvent 수정

```python
# BAD — frozen 객체 수정 시도
event.data = new_data

# GOOD — 새 이벤트 생성
new_event = StreamEvent(type=event.type, data=new_data)
```

## 4. 하드코딩된 설정

```python
# BAD
GPU_SERVER_URL = "http://192.168.1.100:8000"
MAX_CONTEXT = 4096

# GOOD
from core.config import get_config
config = get_config()
config.gpu_server.url
config.model.max_model_len
```

## 5. 도구 스키마 순서 비고정

```python
# BAD — 순서 미보장 (prompt cache 무효화)
tools = list(registry.tools.values())

# GOOD — 이름순 정렬 (cache 안정성)
tools = sorted(registry.tools.values(), key=lambda t: t.name)
```

## 6. XML tool_calls를 query_loop에 노출

```python
# BAD — query_loop에서 XML 직접 파싱
if "<tool_use>" in response_text:
    tool_call = parse_xml(response_text)

# GOOD — ModelAdapter 내부에서만 XML→tool_calls 변환
# query_loop은 항상 OpenAI tool_calls 형식만 수신
```

## 7. 에이전트의 무한 재귀

```python
# BAD — 서브 에이전트가 Agent 도구 사용 가능
agent_tools = registry.get_all_tools()

# GOOD — DISALLOWED_TOOLS_FOR_AGENTS로 필터링
DISALLOWED = {"Agent", "TaskCreate", "TaskStop", "TrainingTool", "CheckpointTool"}
agent_tools = [t for t in tools if t.name not in DISALLOWED]
```

## 8. 에러 무시 (bare except)

```python
# BAD
try:
    result = await tool.call(input, context)
except:
    pass

# GOOD — 구체적 예외 + tool_use_error 래핑
try:
    result = await tool.call(input, context)
except asyncio.TimeoutError:
    yield Message.tool_result(
        tool_use_id=tool_use_id,
        content="<tool_use_error>Tool timed out</tool_use_error>",
        is_error=True
    )
```

## 9. training 모듈에서 core 역방향 import

```python
# BAD — core가 training을 import
# core/orchestrator/query_loop.py
from training.data_collector import collect

# GOOD — training이 core를 import
# training/data_collector.py
from core.orchestrator.query_loop import StreamEvent
```

## 10. 외부 네트워크 접근 (에어갭 위반)

```python
# BAD
import requests
resp = requests.get("https://api.openai.com/v1/models")

# BAD
pip install some-package  # 런타임 설치

# GOOD — 모든 외부 의존성은 오프라인 wheel 번들로 사전 설치
# GOOD — 모델은 ./models/ 디렉토리에 사전 다운로드
```

## 11. 권한 레이어 건너뛰기

```python
# BAD — Layer 2만 체크하고 실행
if file_checker.check(path):
    await tool.call(input, context)

# GOOD — PermissionPipeline 전체를 통과
decision = await pipeline.check_permissions(tool_name, tool_input, tool_use_id)
if decision.behavior == PermissionBehavior.ALLOW:
    await tool.call(input, context)
```

## 12. 동시성 안전하지 않은 도구의 병렬 실행

```python
# BAD — 쓰기 도구를 asyncio.gather로 병렬 실행
await asyncio.gather(edit_tool.call(...), write_tool.call(...))

# GOOD — concurrency partitioning 알고리즘 사용
# is_concurrency_safe=True인 도구만 병렬, 나머지는 순차
```
