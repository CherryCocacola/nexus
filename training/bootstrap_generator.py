"""
부트스트랩 데이터 생성기 — Phase 1용 합성 학습 데이터를 만든다.

실 사용 데이터가 없는 초기 단계에서 LoRA 학습에 사용할
합성 데이터를 생성한다. 두 가지 유형의 데이터를 생성한다:
  1. 도구 사용 데이터: Nexus 도구(Read, Write, Bash 등)의 올바른 사용 패턴
  2. 추론 데이터: 단계별 사고 과정을 포함한 문제 해결 패턴

왜 합성 데이터인가:
  - 에어갭 환경에서 외부 API(OpenAI 등)로 데이터를 생성할 수 없다
  - 템플릿 기반 생성으로 일관된 품질의 학습 데이터를 확보한다
  - 도구 사용 패턴은 정형화되어 있어 템플릿이 효과적이다
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any

logger = logging.getLogger("nexus.training.bootstrap_generator")


# ─────────────────────────────────────────────
# 도구 사용 템플릿 — Nexus 24개 도구의 대표 사용 패턴
# ─────────────────────────────────────────────
_TOOL_USE_TEMPLATES: list[dict[str, Any]] = [
    # ── Read 도구 ──
    {
        "instruction": "{path} 파일의 내용을 읽어줘.",
        "tool": "Read",
        "input": {"file_path": "{path}"},
        "paths": ["src/main.py", "config/settings.yaml", "README.md", "tests/test_app.py"],
    },
    {
        "instruction": "{path} 파일의 {start}번째 줄부터 {count}줄을 보여줘.",
        "tool": "Read",
        "input": {"file_path": "{path}", "offset": "{start}", "limit": "{count}"},
        "paths": ["core/orchestrator/query_loop.py", "core/tools/executor.py"],
        "ranges": [
            {"start": 1, "count": 50},
            {"start": 100, "count": 30},
            {"start": 50, "count": 20},
        ],
    },
    # ── Write 도구 ──
    {
        "instruction": "{path}에 새 파일을 생성해줘.",
        "tool": "Write",
        "input": {"file_path": "{path}", "content": "{content}"},
        "paths": ["src/utils/helper.py", "config/new_config.yaml"],
        "contents": [
            '"""유틸리티 헬퍼 모듈."""\n\ndef helper():\n    pass\n',
            "# 새 설정 파일\nkey: value\n",
        ],
    },
    # ── Edit 도구 ──
    {
        "instruction": "{path} 파일에서 '{old}' 부분을 '{new}'로 수정해줘.",
        "tool": "Edit",
        "input": {"file_path": "{path}", "old_string": "{old}", "new_string": "{new}"},
        "paths": ["src/app.py"],
        "edits": [
            {"old": "def old_func():", "new": "def new_func():"},
            {"old": "DEBUG = True", "new": "DEBUG = False"},
        ],
    },
    # ── Bash 도구 ──
    {
        "instruction": "{command} 명령어를 실행해줘.",
        "tool": "Bash",
        "input": {"command": "{command}"},
        "commands": [
            "ls -la",
            "git status",
            "python -m pytest tests/ -v",
            "ruff check .",
            "find . -name '*.py' | head -20",
        ],
    },
    # ── Glob 도구 ──
    {
        "instruction": "{pattern} 패턴에 맞는 파일들을 찾아줘.",
        "tool": "Glob",
        "input": {"pattern": "{pattern}"},
        "patterns": ["**/*.py", "config/*.yaml", "tests/**/test_*.py", "**/*.md"],
    },
    # ── Grep 도구 ──
    {
        "instruction": "'{query}' 문자열을 포함하는 파일을 검색해줘.",
        "tool": "Grep",
        "input": {"pattern": "{query}", "path": "."},
        "queries": [
            "async def",
            "import logging",
            "class.*BaseModel",
            "TODO",
            "def test_",
        ],
    },
    # ── Agent 도구 ──
    {
        "instruction": "{task} 작업을 서브 에이전트에게 위임해줘.",
        "tool": "Agent",
        "input": {"task": "{task}"},
        "tasks": [
            "이 모듈의 단위 테스트를 작성해줘",
            "코드 리뷰를 진행해줘",
            "버그를 분석해줘",
        ],
    },
    # ── MultiEdit 도구 ──
    # 다중 편집은 edits 배열을 직접 설정한다 (플레이스홀더 치환 불필요)
    {
        "instruction": "{path} 파일에서 여러 부분을 한 번에 수정해줘.",
        "tool": "MultiEdit",
        "input": {
            "file_path": "{path}",
            "edits": [
                {"old_string": "import os", "new_string": "import os\nimport sys"},
                {"old_string": "DEBUG = True", "new_string": "DEBUG = False"},
            ],
        },
        "paths": ["src/app.py", "core/tools/executor.py", "core/orchestrator/query_loop.py"],
    },
    {
        "instruction": "{path} 파일에서 로거와 타임아웃 설정을 동시에 변경해줘.",
        "tool": "MultiEdit",
        "input": {
            "file_path": "{path}",
            "edits": [
                {"old_string": "logger.info", "new_string": "logger.debug"},
                {"old_string": "timeout=30", "new_string": "timeout=60"},
            ],
        },
        "paths": ["core/tools/executor.py", "core/model/inference.py"],
    },
    # ── LS 도구 ──
    {
        "instruction": "{path} 디렉토리의 파일 목록을 보여줘.",
        "tool": "LS",
        "input": {"path": "{path}"},
        "paths": [".", "src/", "core/tools/", "tests/", "config/"],
    },
    # ── GitLog 도구 ──
    {
        "instruction": "최근 커밋 이력을 5개 보여줘.",
        "tool": "GitLog",
        "input": {"max_count": 5},
    },
    {
        "instruction": "최근 커밋 이력을 20개 보여줘.",
        "tool": "GitLog",
        "input": {"max_count": 20},
    },
    {
        "instruction": "{path} 파일의 Git 커밋 이력을 보여줘.",
        "tool": "GitLog",
        "input": {"max_count": 10, "file_path": "{path}"},
        "paths": ["core/orchestrator/query_loop.py", "core/tools/executor.py", "README.md"],
    },
    # ── GitDiff 도구 ──
    {
        "instruction": "현재 변경 사항을 비교해줘.",
        "tool": "GitDiff",
        "input": {},
    },
    {
        "instruction": "HEAD~3과 HEAD 사이의 차이를 보여줘.",
        "tool": "GitDiff",
        "input": {"ref1": "HEAD~3", "ref2": "HEAD"},
    },
    {
        "instruction": "main 브랜치와 현재 브랜치의 차이를 보여줘.",
        "tool": "GitDiff",
        "input": {"ref1": "main", "ref2": "HEAD"},
    },
    # ── GitStatus 도구 ──
    {
        "instruction": "현재 Git 상태를 확인해줘.",
        "tool": "GitStatus",
        "input": {},
    },
    {
        "instruction": "변경된 파일과 스테이징 상태를 보여줘.",
        "tool": "GitStatus",
        "input": {},
    },
    # ── GitCommit 도구 ──
    # 커밋 메시지는 고정 값 — commands 키를 사용하되 input.message는 직접 설정
    {
        "instruction": "'[core/tools] Read 도구 구현' 메시지로 커밋해줘.",
        "tool": "GitCommit",
        "input": {"message": "[core/tools] Read 도구 구현 — 파일 읽기 + 라인 범위 지원"},
    },
    {
        "instruction": "'[core/orchestrator] query_loop 구현' 메시지로 커밋해줘.",
        "tool": "GitCommit",
        "input": {"message": "[core/orchestrator] query_loop while(True) 패턴 구현"},
    },
    {
        "instruction": "스트리밍 파서 오류 수정 커밋을 만들어줘.",
        "tool": "GitCommit",
        "input": {"message": "[fix] 스트리밍 파서 오류 수정 — SSE 이벤트 누락 방지"},
    },
    # ── GitBranch 도구 ──
    {
        "instruction": "'feature/tool-system' 브랜치를 생성해줘.",
        "tool": "GitBranch",
        "input": {"name": "feature/tool-system", "action": "create"},
    },
    {
        "instruction": "'fix/streaming-parser' 브랜치를 생성해줘.",
        "tool": "GitBranch",
        "input": {"name": "fix/streaming-parser", "action": "create"},
    },
    {
        "instruction": "브랜치 목록을 보여줘.",
        "tool": "GitBranch",
        "input": {"action": "list"},
    },
    # ── GitCheckout 도구 ──
    {
        "instruction": "'main' 브랜치로 전환해줘.",
        "tool": "GitCheckout",
        "input": {"ref": "main"},
    },
    {
        "instruction": "'feature/tool-system' 브랜치로 전환해줘.",
        "tool": "GitCheckout",
        "input": {"ref": "feature/tool-system"},
    },
    {
        "instruction": "'develop' 브랜치로 전환해줘.",
        "tool": "GitCheckout",
        "input": {"ref": "develop"},
    },
    # ── NotebookRead 도구 ──
    {
        "instruction": "{path} 노트북 파일을 읽어줘.",
        "tool": "NotebookRead",
        "input": {"file_path": "{path}"},
        "paths": [
            "notebooks/analysis.ipynb",
            "notebooks/experiment.ipynb",
            "training/data_analysis.ipynb",
        ],
    },
    # ── NotebookEdit 도구 ──
    # 셀 인덱스와 소스를 직접 설정한다 (고정 값 사용)
    {
        "instruction": "{path} 노트북의 첫 번째 셀을 수정해줘.",
        "tool": "NotebookEdit",
        "input": {
            "file_path": "{path}",
            "cell_index": 0,
            "new_source": "import pandas as pd\nimport numpy as np\n",
        },
        "paths": ["notebooks/analysis.ipynb", "notebooks/experiment.ipynb"],
    },
    {
        "instruction": "{path} 노트북의 세 번째 셀을 데이터 전처리 코드로 바꿔줘.",
        "tool": "NotebookEdit",
        "input": {
            "file_path": "{path}",
            "cell_index": 2,
            "new_source": "# 데이터 전처리\ndf = df.dropna()\ndf = df.reset_index(drop=True)\n",
        },
        "paths": ["notebooks/analysis.ipynb", "training/data_analysis.ipynb"],
    },
    # ── TodoRead 도구 ──
    {
        "instruction": "현재 할일 목록을 보여줘.",
        "tool": "TodoRead",
        "input": {},
    },
    {
        "instruction": "미완료 할일 항목을 확인해줘.",
        "tool": "TodoRead",
        "input": {},
    },
    # ── TodoWrite 도구 ──
    {
        "instruction": "할일을 추가해줘: query_loop 단위 테스트 작성",
        "tool": "TodoWrite",
        "input": {"action": "add", "content": "query_loop 단위 테스트 작성"},
    },
    {
        "instruction": "할일을 추가해줘: Read 도구 권한 검증 로직 추가",
        "tool": "TodoWrite",
        "input": {"action": "add", "content": "Read 도구 권한 검증 로직 추가"},
    },
    {
        "instruction": "할일을 추가해줘: SSE 스트리밍 파서 에러 핸들링 개선",
        "tool": "TodoWrite",
        "input": {"action": "add", "content": "SSE 스트리밍 파서 에러 핸들링 개선"},
    },
    {
        "instruction": "1번 할일을 완료 처리해줘.",
        "tool": "TodoWrite",
        "input": {"action": "complete", "task_id": "1"},
    },
    {
        "instruction": "3번 할일을 완료 처리해줘.",
        "tool": "TodoWrite",
        "input": {"action": "complete", "task_id": "3"},
    },
    # ── Task 도구 ──
    {
        "instruction": "'대규모 코드베이스 분석' 비동기 태스크를 생성해줘.",
        "tool": "Task",
        "input": {"action": "create", "description": "대규모 코드베이스 분석"},
    },
    {
        "instruction": "'전체 테스트 스위트 실행' 비동기 태스크를 생성해줘.",
        "tool": "Task",
        "input": {"action": "create", "description": "전체 테스트 스위트 실행"},
    },
    {
        "instruction": "'데이터 마이그레이션 실행' 비동기 태스크를 생성해줘.",
        "tool": "Task",
        "input": {"action": "create", "description": "데이터 마이그레이션 실행"},
    },
    {
        "instruction": "태스크 task_001의 상태를 확인해줘.",
        "tool": "Task",
        "input": {"action": "status", "task_id": "task_001"},
    },
    {
        "instruction": "태스크 task_002의 상태를 확인해줘.",
        "tool": "Task",
        "input": {"action": "status", "task_id": "task_002"},
    },
    # ── MemoryRead 도구 ──
    {
        "instruction": "'프로젝트 아키텍처 결정 사항'에 대한 메모리를 검색해줘.",
        "tool": "MemoryRead",
        "input": {"query": "프로젝트 아키텍처 결정 사항"},
    },
    {
        "instruction": "'이전 디버깅 세션 내용'에 대한 메모리를 검색해줘.",
        "tool": "MemoryRead",
        "input": {"query": "이전 디버깅 세션 내용"},
    },
    {
        "instruction": "'코드 리뷰 피드백'에 대한 메모리를 검색해줘.",
        "tool": "MemoryRead",
        "input": {"query": "코드 리뷰 피드백"},
    },
    {
        "instruction": "'설정 변경 이력'에 대한 메모리를 검색해줘.",
        "tool": "MemoryRead",
        "input": {"query": "설정 변경 이력"},
    },
    # ── MemoryWrite 도구 ──
    {
        "instruction": "이 내용을 메모리에 저장해줘: Redis 세션 저장소로 결정",
        "tool": "MemoryWrite",
        "input": {
            "content": "Redis 세션 저장소로 결정 — TTL 기반 자동 만료",
            "tags": ["architecture", "decision"],
        },
    },
    {
        "instruction": "이 내용을 메모리에 저장해줘: CONTEXT_COMPACT 이벤트 처리 필요",
        "tool": "MemoryWrite",
        "input": {
            "content": "query_loop에서 CONTEXT_COMPACT 이벤트 처리 로직 추가 필요",
            "tags": ["todo", "orchestrator"],
        },
    },
    {
        "instruction": "이 내용을 메모리에 저장해줘: e5-large 임베딩 응답 시간 측정 결과",
        "tool": "MemoryWrite",
        "input": {
            "content": "e5-large 임베딩 모델 응답 시간: 평균 12ms",
            "tags": ["performance", "benchmark"],
        },
    },
    # ── DockerBuild 도구 ──
    {
        "instruction": "'nexus-orchestrator:latest' 태그로 Docker 이미지를 빌드해줘.",
        "tool": "DockerBuild",
        "input": {"tag": "nexus-orchestrator:latest", "dockerfile": "Dockerfile"},
    },
    {
        "instruction": "GPU용 Docker 이미지를 빌드해줘.",
        "tool": "DockerBuild",
        "input": {"tag": "nexus-gpu:latest", "dockerfile": "Dockerfile.gpu"},
    },
    {
        "instruction": "테스트용 Docker 이미지를 빌드해줘.",
        "tool": "DockerBuild",
        "input": {"tag": "nexus-test:dev", "dockerfile": "Dockerfile.test"},
    },
    # ── DockerRun 도구 ──
    {
        "instruction": "'nexus-orchestrator:latest' 이미지로 컨테이너를 실행해줘.",
        "tool": "DockerRun",
        "input": {"image": "nexus-orchestrator:latest"},
    },
    {
        "instruction": "Redis 컨테이너를 포트 6379로 실행해줘.",
        "tool": "DockerRun",
        "input": {"image": "redis:7-alpine", "ports": "6379:6379"},
    },
    {
        "instruction": "PostgreSQL 컨테이너를 포트 5432로 실행해줘.",
        "tool": "DockerRun",
        "input": {"image": "postgres:16-alpine", "ports": "5432:5432"},
    },
    {
        "instruction": "GPU 서버 컨테이너를 8080 포트로 실행해줘.",
        "tool": "DockerRun",
        "input": {"image": "nexus-gpu:latest", "ports": "8080:8080"},
    },
]

# ─────────────────────────────────────────────
# 추론 템플릿 — 단계별 사고 과정 포함
# ─────────────────────────────────────────────
_REASONING_TEMPLATES: list[dict[str, Any]] = [
    {
        "category": "debugging",
        "instruction": "이 에러를 분석하고 해결 방법을 제안해줘: {error}",
        "reasoning": (
            "1. 에러 메시지를 분석합니다: {error}\n"
            "2. 에러의 원인을 파악합니다: {cause}\n"
            "3. 관련 코드를 확인합니다.\n"
            "4. 해결 방법을 제안합니다: {solution}"
        ),
        "errors": [
            {
                "error": "ImportError: cannot import name 'foo' from 'bar'",
                "cause": "모듈에 해당 이름이 존재하지 않거나 순환 import",
                "solution": "import 경로를 확인하고, 순환 import이면 lazy import로 전환",
            },
            {
                "error": "TypeError: object NoneType can't be used in 'await' expression",
                "cause": "async 함수가 아닌 함수를 await로 호출",
                "solution": "함수 정의에 async 키워드를 추가하거나 호출 방식을 변경",
            },
            {
                "error": "asyncio.TimeoutError",
                "cause": "비동기 작업이 지정된 시간 내에 완료되지 않음",
                "solution": "타임아웃을 늘리거나, 작업을 분할하여 단계적으로 처리",
            },
        ],
    },
    {
        "category": "architecture",
        "instruction": "{question}",
        "reasoning": (
            "1. 요구사항을 분석합니다.\n"
            "2. 선택 가능한 방법을 나열합니다: {options}\n"
            "3. 각 방법의 장단점을 비교합니다.\n"
            "4. 결론: {conclusion}"
        ),
        "questions": [
            {
                "question": "Redis와 PostgreSQL 중 세션 데이터 저장에 어떤 것이 적합한가?",
                "options": "Redis(빠른 읽기/쓰기, TTL 지원), PostgreSQL(영속성, 쿼리 유연성)",
                "conclusion": "세션은 일시적이고 빠른 접근이 필요하므로 Redis가 적합합니다",
            },
            {
                "question": "AsyncGenerator를 사용하는 이유는 무엇인가?",
                "options": "콜백 패턴, Future/Promise, AsyncGenerator",
                "conclusion": (
                    "스트리밍 데이터에 자연스럽고 메모리 효율적이며 백프레셔를 지원합니다"
                ),
            },
        ],
    },
    {
        "category": "code_review",
        "instruction": "이 코드를 리뷰해줘:\n```python\n{code}\n```",
        "reasoning": (
            "1. 코드의 목적을 파악합니다.\n"
            "2. 문제점을 식별합니다: {issues}\n"
            "3. 개선 방향을 제안합니다: {improvements}"
        ),
        "reviews": [
            {
                "code": "try:\n    result = do_something()\nexcept:\n    pass",
                "issues": "bare except로 모든 예외를 무시하고 있습니다",
                "improvements": "구체적인 예외를 지정하고, 에러를 로깅하세요",
            },
            {
                "code": "data = requests.get('https://api.example.com/data')",
                "issues": "에어갭 환경에서 외부 네트워크 호출은 금지됩니다",
                "improvements": "로컬 API 서버를 통해 데이터를 가져오도록 변경하세요",
            },
        ],
    },
    # ── 보안 취약점 분석 ──
    {
        "category": "security_analysis",
        "instruction": "이 코드의 보안 취약점을 분석해줘:\n```python\n{code}\n```",
        "reasoning": (
            "1. 코드를 보안 관점에서 검토합니다.\n"
            "2. 식별된 취약점: {vulnerability}\n"
            "3. 공격 시나리오: {attack_scenario}\n"
            "4. 권장 수정 방법: {fix}"
        ),
        "analyses": [
            {
                "code": (
                    "query = f\"SELECT * FROM users WHERE name = '{user_input}'\""
                    "\ncursor.execute(query)"
                ),
                "vulnerability": "SQL Injection — 사용자 입력이 직접 쿼리에 삽입됩니다",
                "attack_scenario": (
                    "공격자가 user_input에 ' OR 1=1 -- 를 입력하면 모든 사용자 데이터가 노출됩니다"
                ),
                "fix": (
                    "파라미터화된 쿼리를 사용하세요: "
                    "cursor.execute('SELECT * FROM users WHERE name = %s', (user_input,))"
                ),
            },
            {
                "code": (
                    "file_path = os.path.join(base_dir, user_filename)\n"
                    "with open(file_path) as f:\n"
                    "    return f.read()"
                ),
                "vulnerability": (
                    "Path Traversal — 사용자가 ../../etc/passwd 같은 "
                    "경로를 입력하여 시스템 파일에 접근할 수 있습니다"
                ),
                "attack_scenario": (
                    "user_filename에 ../../../etc/shadow를 넣으면 "
                    "민감한 시스템 파일을 읽을 수 있습니다"
                ),
                "fix": (
                    "경로를 정규화한 후 base_dir 내에 있는지 검증하세요: "
                    "resolved = Path(file_path).resolve(); "
                    "assert str(resolved).startswith(str(Path(base_dir).resolve()))"
                ),
            },
            {
                "code": ("html = f'<div>환영합니다, {username}!</div>'\nreturn HTMLResponse(html)"),
                "vulnerability": (
                    "XSS (Cross-Site Scripting) — 사용자 입력이 이스케이프 없이 HTML에 삽입됩니다"
                ),
                "attack_scenario": (
                    "username에 <script>alert('XSS')</script>를 넣으면 "
                    "브라우저에서 악성 스크립트가 실행됩니다"
                ),
                "fix": (
                    "html.escape()로 사용자 입력을 이스케이프하거나 "
                    "Jinja2 템플릿의 자동 이스케이프 기능을 사용하세요"
                ),
            },
        ],
    },
    # ── 성능 최적화 분석 ──
    {
        "category": "performance",
        "instruction": (
            "이 코드의 성능 문제를 분석하고 최적화 방법을 제안해줘:\n```python\n{code}\n```"
        ),
        "reasoning": (
            "1. 코드의 성능 특성을 분석합니다.\n"
            "2. 식별된 성능 병목: {bottleneck}\n"
            "3. 영향 범위: {impact}\n"
            "4. 최적화 방법: {optimization}"
        ),
        "perf_issues": [
            {
                "code": (
                    "for user in users:\n"
                    "    orders = db.query(Order).filter(Order.user_id == user.id).all()\n"
                    "    user.orders = orders"
                ),
                "bottleneck": ("N+1 쿼리 문제 — 사용자 수만큼 추가 DB 쿼리가 발생합니다"),
                "impact": (
                    "사용자 1000명이면 1001번의 DB 쿼리가 실행되어 "
                    "응답 시간이 수 초 이상으로 증가합니다"
                ),
                "optimization": (
                    "joinedload 또는 subqueryload를 사용하여 한 번의 쿼리로 "
                    "관련 데이터를 함께 로드하세요: "
                    "db.query(User).options(joinedload(User.orders)).all()"
                ),
            },
            {
                "code": (
                    "results = []\n"
                    "for item in large_list:\n"
                    "    if item not in results:\n"
                    "        results.append(item)"
                ),
                "bottleneck": ("O(n^2) 중복 제거 — 리스트의 in 연산이 매번 전체를 순회합니다"),
                "impact": (
                    "10만 개 항목이면 최대 100억 번의 비교가 발생하여 수 분 이상 소요될 수 있습니다"
                ),
                "optimization": (
                    "set을 사용하여 O(1) 조회로 변경하세요: "
                    "seen = set(); results = [x for x in large_list "
                    "if x not in seen and not seen.add(x)]"
                ),
            },
            {
                "code": (
                    "async def get_all_data():\n"
                    "    a = await fetch_from_service_a()\n"
                    "    b = await fetch_from_service_b()\n"
                    "    c = await fetch_from_service_c()\n"
                    "    return a, b, c"
                ),
                "bottleneck": ("순차적 비동기 호출 — 독립적인 작업을 직렬로 실행합니다"),
                "impact": (
                    "각 서비스 호출이 100ms라면 총 300ms가 소요되지만, "
                    "병렬로 실행하면 100ms로 줄일 수 있습니다"
                ),
                "optimization": (
                    "asyncio.gather를 사용하여 병렬로 실행하세요: "
                    "a, b, c = await asyncio.gather("
                    "fetch_from_service_a(), fetch_from_service_b(), fetch_from_service_c())"
                ),
            },
        ],
    },
    # ── 리팩토링 제안 ──
    {
        "category": "refactoring",
        "instruction": "이 코드의 리팩토링이 필요한 부분을 지적해줘:\n```python\n{code}\n```",
        "reasoning": (
            "1. 코드 구조를 분석합니다.\n"
            "2. 식별된 코드 스멜: {smell}\n"
            "3. 문제의 근본 원인: {root_cause}\n"
            "4. 리팩토링 방법: {refactoring}"
        ),
        "smells": [
            {
                "code": (
                    "def process(data, mode):\n"
                    "    if mode == 'csv':\n"
                    "        # 50줄의 CSV 처리 로직\n"
                    "        pass\n"
                    "    elif mode == 'json':\n"
                    "        # 50줄의 JSON 처리 로직\n"
                    "        pass\n"
                    "    elif mode == 'xml':\n"
                    "        # 50줄의 XML 처리 로직\n"
                    "        pass"
                ),
                "smell": (
                    "긴 함수 + 분기 폭발 — 하나의 함수가 150줄 이상이고 "
                    "모드별 분기가 계속 증가합니다"
                ),
                "root_cause": (
                    "서로 다른 형식의 처리 로직이 하나의 함수에 결합되어 "
                    "개방-폐쇄 원칙(OCP)을 위반합니다"
                ),
                "refactoring": (
                    "전략 패턴을 적용하세요: 각 모드별 처리기를 별도 클래스로 분리하고, "
                    "딕셔너리 디스패치로 선택합니다. "
                    "processors = {'csv': CsvProcessor(), 'json': JsonProcessor(), ...}"
                ),
            },
            {
                "code": (
                    "from core.model.inference import ModelClient\n"
                    "from core.tools.executor import ToolExecutor\n"
                    "# core/tools/executor.py에서:\n"
                    "from core.model.inference import ModelClient\n"
                    "from core.orchestrator.query_loop import QueryLoop"
                ),
                "smell": ("순환 의존성 — 모듈 간 양방향 import가 발생합니다"),
                "root_cause": (
                    "계층 간 의존성 방향이 역전되어 있습니다. "
                    "core/tools가 core/orchestrator를 import하면 안 됩니다"
                ),
                "refactoring": (
                    "의존성 역전 원칙(DIP)을 적용하세요: "
                    "공통 인터페이스(ABC)를 정의하고 상위 모듈에서 구현을 주입합니다. "
                    "또는 lazy import로 런타임 순환을 회피합니다"
                ),
            },
            {
                "code": (
                    "class UserService:\n"
                    "    def create_user(self, name, email):\n"
                    "        # 유효성 검사\n"
                    "        # DB 저장\n"
                    "        # 이메일 발송\n"
                    "        # 감사 로그 기록\n"
                    "        # 캐시 갱신\n"
                    "        pass"
                ),
                "smell": (
                    "단일 책임 원칙(SRP) 위반 — 하나의 메서드가 5가지 서로 다른 관심사를 처리합니다"
                ),
                "root_cause": (
                    "횡단 관심사(이메일, 로그, 캐시)가 비즈니스 로직에 직접 결합되어 있습니다"
                ),
                "refactoring": (
                    "이벤트 기반 아키텍처로 분리하세요: "
                    "UserService는 생성만 담당하고, "
                    "이메일/로그/캐시는 이벤트 핸들러로 분리합니다. "
                    "또는 데코레이터 패턴으로 횡단 관심사를 래핑합니다"
                ),
            },
        ],
    },
]


# ─────────────────────────────────────────────
# 장문 지식 답변 템플릿 — Phase 3 추가 (2026-04-17)
# Phase 2 학습에서 "짧은 답변 샘플" 편중으로 장문 설명 능력이 퇴보하는 문제를
# 완화하기 위해 도입. 사용자가 "알려줘 / 자세히 / 설명해줘"를 요청했을 때
# 3~6개 단락의 구조화된 한국어 답변을 생성하도록 학습시킨다.
# ─────────────────────────────────────────────
_KNOWLEDGE_TEMPLATES: list[dict[str, Any]] = [
    {
        "question": "객체지향 프로그래밍의 4가지 핵심 원칙을 자세히 설명해줘.",
        "answer": (
            "객체지향 프로그래밍(OOP)의 네 가지 핵심 원칙은 다음과 같습니다.\n\n"
            "1. 캡슐화(Encapsulation): 데이터와 그것을 다루는 메서드를 하나의 객체로 묶고, "
            "외부에서 직접 접근하지 못하도록 제한합니다. public/private 접근 제어로 구현합니다.\n\n"
            "2. 상속(Inheritance): 기존 클래스의 속성과 동작을 물려받아 새로운 클래스를 정의합니다. "
            "코드 재사용과 계층 구조 표현에 사용됩니다.\n\n"
            "3. 다형성(Polymorphism): 같은 인터페이스가 객체 타입에 따라 다르게 동작합니다. "
            "메서드 오버라이딩과 오버로딩으로 구현됩니다.\n\n"
            "4. 추상화(Abstraction): 복잡한 구현 세부사항을 감추고 본질적 특성만 노출합니다. "
            "추상 클래스·인터페이스로 모델링합니다."
        ),
    },
    {
        "question": "RESTful API의 주요 원칙을 자세히 설명해줘.",
        "answer": (
            "REST(REpresentational State Transfer)는 Roy Fielding의 박사 논문에서 정의된 "
            "분산 하이퍼미디어 시스템을 위한 아키텍처 스타일입니다. 주요 원칙은 다음과 같습니다.\n\n"
            "1. 클라이언트-서버 구조: 사용자 인터페이스와 데이터 저장소를 분리합니다.\n"
            "2. 무상태성(Statelessness): 각 요청은 완결적이며 서버가 클라이언트 상태를 저장하지 않습니다.\n"
            "3. 캐시 가능성: 응답은 캐시 가능 여부를 명시합니다.\n"
            "4. 계층화 구조: 중간 서버(로드밸런서·프록시·게이트웨이)를 투명하게 배치합니다.\n"
            "5. 통일된 인터페이스: HTTP 메서드(GET/POST/PUT/DELETE)와 URI로 리소스를 식별·조작합니다.\n"
            "6. 코드 온 디맨드(선택): 서버가 클라이언트에 실행 가능한 코드를 전송할 수 있습니다.\n\n"
            "실무에서는 1~5번 원칙을 주로 따르며, URI 설계·HTTP 상태 코드·하이퍼미디어 링크를 "
            "올바르게 사용하는 것이 중요합니다."
        ),
    },
    {
        "question": "Python의 GIL(Global Interpreter Lock)에 대해 설명해줘.",
        "answer": (
            "GIL(Global Interpreter Lock)은 CPython 인터프리터에 존재하는 뮤텍스(mutex)로, "
            "하나의 프로세스 내에서 여러 스레드가 동시에 파이썬 바이트코드를 실행하지 못하도록 "
            "직렬화하는 잠금 장치입니다.\n\n"
            "**존재 이유**: CPython의 메모리 관리(특히 참조 카운팅)가 스레드 안전하지 않아, "
            "내부 자료 구조를 보호하기 위해 도입되었습니다.\n\n"
            "**영향**:\n"
            "- CPU 바운드 작업에서 멀티스레드로 성능 향상이 제한됨 (한 번에 한 스레드만 실행)\n"
            "- I/O 바운드 작업에서는 I/O 대기 중 GIL이 해제되므로 영향이 적음\n"
            "- C 확장(numpy 등)은 GIL을 해제하고 네이티브 코드 실행이 가능\n\n"
            "**우회 방법**:\n"
            "- multiprocessing 모듈로 프로세스 단위 병렬화\n"
            "- asyncio로 단일 스레드 내 비동기 처리\n"
            "- 계산 집약 작업을 C/Cython/Rust 확장으로 오프로드\n\n"
            "PEP 703(선택적 GIL 제거)이 Python 3.13부터 실험적으로 지원됩니다."
        ),
    },
    {
        "question": "트랜잭션의 ACID 속성을 자세히 알려줘.",
        "answer": (
            "ACID는 데이터베이스 트랜잭션이 신뢰성 있게 처리되기 위해 갖춰야 할 네 가지 속성을 "
            "나타내는 약어입니다.\n\n"
            "1. 원자성(Atomicity): 트랜잭션 내 모든 연산이 전부 성공하거나 전부 실패합니다. "
            "중간 실패 시 전체를 롤백하여 부분 적용이 남지 않게 합니다.\n\n"
            "2. 일관성(Consistency): 트랜잭션 전후로 데이터베이스가 정의된 규칙(제약조건·트리거·"
            "참조 무결성)을 유지합니다.\n\n"
            "3. 고립성(Isolation): 동시 실행되는 트랜잭션들이 서로의 중간 상태를 보지 못합니다. "
            "격리 수준(READ UNCOMMITTED/READ COMMITTED/REPEATABLE READ/SERIALIZABLE)으로 "
            "세밀하게 조정합니다.\n\n"
            "4. 지속성(Durability): 커밋된 트랜잭션의 결과는 시스템 장애 이후에도 유지됩니다. "
            "로그 선행 기록(WAL)과 복구 메커니즘으로 보장됩니다."
        ),
    },
    {
        "question": "HTTP/1.1과 HTTP/2의 주요 차이점을 설명해줘.",
        "answer": (
            "HTTP/1.1(1997)과 HTTP/2(2015)의 주요 차이는 다음과 같습니다.\n\n"
            "1. 프레이밍: HTTP/1.1은 텍스트 기반, HTTP/2는 바이너리 프레임 기반입니다.\n"
            "2. 멀티플렉싱: HTTP/2는 하나의 TCP 연결에서 여러 요청·응답을 병렬 처리하여 "
            "Head-of-Line Blocking을 완화합니다.\n"
            "3. 헤더 압축: HTTP/2는 HPACK 알고리즘으로 반복되는 헤더를 압축합니다.\n"
            "4. 서버 푸시: HTTP/2는 클라이언트가 요청하기 전에 서버가 리소스를 선제적으로 전송할 수 있습니다.\n"
            "5. 우선순위: HTTP/2는 스트림에 가중치를 부여해 전송 순서를 최적화합니다.\n\n"
            "후속 표준 HTTP/3는 전송 계층을 TCP에서 QUIC(UDP 기반)으로 바꿔 연결 지연과 "
            "Head-of-Line Blocking 문제를 더 근본적으로 해결합니다."
        ),
    },
    {
        "question": "CAP 정리(CAP Theorem)가 무엇인지 알려줘.",
        "answer": (
            "CAP 정리는 2000년 Eric Brewer가 제시한 분산 시스템의 근본적 제약입니다. "
            "분산 데이터 저장소는 다음 세 속성 중 **동시에 최대 두 가지만** 보장할 수 있습니다.\n\n"
            "- 일관성(Consistency): 모든 노드가 동시에 같은 데이터를 본다\n"
            "- 가용성(Availability): 모든 요청이 성공/실패 여부의 응답을 받는다\n"
            "- 분할 내성(Partition tolerance): 네트워크 장애가 있어도 시스템이 동작한다\n\n"
            "실무적 의의: 네트워크 분할은 현실에서 피할 수 없으므로 분산 시스템은 사실상 "
            "**CP(일관성·분할 내성)** 또는 **AP(가용성·분할 내성)** 중에서 선택합니다. "
            "CP 예시는 전통적인 RDBMS 기반 분산 트랜잭션, AP 예시는 DynamoDB·Cassandra입니다.\n\n"
            "이후 PACELC 정리는 '정상 상황에서의 지연 vs 일관성' 트레이드오프까지 확장합니다."
        ),
    },
    {
        "question": "TCP와 UDP의 차이점과 각각의 사용 사례를 알려줘.",
        "answer": (
            "TCP와 UDP는 전송 계층의 대표 프로토콜로, 다음과 같이 대비됩니다.\n\n"
            "**TCP (Transmission Control Protocol)**:\n"
            "- 연결 지향: 3-way handshake로 연결 수립\n"
            "- 신뢰성: 순서 보장, 재전송, 흐름 제어, 혼잡 제어\n"
            "- 오버헤드: 헤더 20바이트+\n"
            "- 사용: HTTP, HTTPS, SMTP, FTP, SSH 등 대부분의 웹 트래픽\n\n"
            "**UDP (User Datagram Protocol)**:\n"
            "- 비연결: handshake 없이 바로 전송\n"
            "- 최선형 전달: 순서·도착 보장 없음, 재전송 없음\n"
            "- 오버헤드: 헤더 8바이트로 가벼움\n"
            "- 사용: DNS, DHCP, VoIP, 게임, 실시간 스트리밍\n\n"
            "HTTP/3는 UDP 위의 QUIC 프로토콜을 사용하여 UDP의 가벼움과 TCP의 신뢰성을 결합합니다."
        ),
    },
    {
        "question": "JWT(JSON Web Token)의 구조와 사용 방식을 설명해줘.",
        "answer": (
            "JWT는 두 당사자 간에 클레임(claims)을 안전하게 전달하기 위한 "
            "Base64URL로 인코딩된 JSON 기반 토큰 표준입니다. 세 부분이 점(.)으로 연결된 형태입니다.\n\n"
            "1. Header: 토큰 타입과 서명 알고리즘(예: HS256, RS256)\n"
            "2. Payload: 클레임 집합(iss 발급자, sub 주체, exp 만료, iat 발급 시각 등)\n"
            "3. Signature: Header.Payload를 비밀키로 서명\n\n"
            "**사용 방식**:\n"
            "- 로그인 성공 시 서버가 JWT를 발급\n"
            "- 클라이언트는 이후 요청의 Authorization 헤더에 Bearer <JWT>를 포함\n"
            "- 서버는 서명을 검증하여 인증\n\n"
            "**보안 주의사항**:\n"
            "- Payload는 암호화가 아닌 인코딩이므로 민감 정보 저장 금지\n"
            "- 짧은 만료(exp) + Refresh Token 패턴 권장\n"
            "- 탈취 시 즉시 무효화가 어려우므로 블랙리스트·짧은 TTL로 보완"
        ),
    },
    {
        "question": "쿠버네티스(Kubernetes)의 기본 개념을 알려줘.",
        "answer": (
            "쿠버네티스는 구글이 내부 시스템 Borg에서 영감을 얻어 개발한 오픈소스 컨테이너 "
            "오케스트레이션 플랫폼입니다. 컨테이너화된 애플리케이션의 배포·확장·관리를 자동화합니다.\n\n"
            "**핵심 추상화**:\n"
            "- Pod: 하나 이상의 컨테이너를 묶은 최소 배포 단위\n"
            "- Service: 여러 Pod을 묶어 안정적인 네트워크 엔드포인트 제공\n"
            "- Deployment: Pod의 선언적 배포와 롤링 업데이트 관리\n"
            "- Namespace: 리소스 논리 격리\n"
            "- ConfigMap/Secret: 설정과 비밀 관리\n"
            "- Ingress: HTTP(S) 외부 노출과 라우팅\n\n"
            "**컨트롤 플레인**: API 서버, etcd(상태 저장), 스케줄러, 컨트롤러 매니저로 구성되어 "
            "선언된 상태(desired state)와 실제 상태(actual state)를 지속적으로 조정합니다.\n\n"
            "대규모 마이크로서비스 운영의 사실상 표준이며, 헬름(Helm)·아르고CD(ArgoCD)·"
            "이스티오(Istio) 등 풍부한 생태계를 갖고 있습니다."
        ),
    },
    {
        "question": "Git에서 merge와 rebase의 차이점을 자세히 알려줘.",
        "answer": (
            "git merge와 git rebase는 한 브랜치의 변경을 다른 브랜치로 합치는 두 가지 방식입니다. "
            "결과는 비슷하지만 히스토리 형태가 크게 다릅니다.\n\n"
            "**merge**: 두 브랜치의 끝을 합치는 **새로운 merge commit**을 생성합니다. "
            "원래 브랜치의 기록이 그대로 보존되어 분기가 시각적으로 남습니다.\n\n"
            "**rebase**: 현재 브랜치의 커밋들을 떼어내 대상 브랜치 끝으로 **재적용**합니다. "
            "히스토리가 선형(linear)으로 정리되지만 **원 커밋은 새 해시로 재생성**됩니다.\n\n"
            "**실무 권장**:\n"
            "- 공유 브랜치(main, develop)에 병합할 때는 merge로 컨텍스트 보존\n"
            "- 로컬 feature 브랜치를 최신 main으로 갱신할 때는 rebase로 히스토리 정돈\n"
            "- **이미 푸시된 공용 커밋은 rebase 금지** — 다른 개발자의 작업을 깨뜨림\n"
            "- Squash merge는 feature 브랜치의 여러 작은 커밋을 하나로 합쳐 main에 넣는 용도"
        ),
    },
    {
        "question": "도커(Docker)의 개념과 장점을 알려줘.",
        "answer": (
            "도커는 애플리케이션과 그 실행 환경을 **컨테이너**라는 표준 단위로 묶어 "
            "일관되게 배포·실행할 수 있게 하는 오픈소스 플랫폼입니다.\n\n"
            "**핵심 개념**:\n"
            "- 이미지(Image): 실행에 필요한 코드·라이브러리·설정을 담은 불변 스냅샷\n"
            "- 컨테이너(Container): 이미지의 실행 인스턴스, 호스트와 격리된 네임스페이스\n"
            "- Dockerfile: 이미지 빌드 절차를 기술한 선언적 스크립트\n"
            "- 레지스트리(Registry): 이미지 저장·배포용 원격 저장소(Docker Hub 등)\n\n"
            "**장점**:\n"
            "- 환경 일관성: \"내 PC에서는 되는데...\" 문제 해결\n"
            "- 가볍고 빠른 기동: VM 대비 수백 ms 단위\n"
            "- 자원 효율: 커널 공유로 메모리/디스크 절약\n"
            "- CI/CD 친화: 이미지 단위로 동일 artifact 배포\n\n"
            "단점은 보안 격리가 VM보다 약하고 스테이트풀 워크로드 설계가 복잡하다는 점입니다. "
            "운영 규모가 커지면 쿠버네티스 등 오케스트레이터와 결합합니다."
        ),
    },
    {
        "question": "SQL에서 JOIN의 종류를 예시와 함께 알려줘.",
        "answer": (
            "SQL JOIN은 여러 테이블의 행을 관계(보통 키)를 기준으로 결합하는 연산입니다. "
            "주요 종류는 다음과 같습니다.\n\n"
            "1. **INNER JOIN**: 양쪽 테이블 모두에 일치하는 행만 반환.\n"
            "   예: `SELECT u.name, o.total FROM users u INNER JOIN orders o ON u.id = o.user_id;`\n\n"
            "2. **LEFT (OUTER) JOIN**: 왼쪽 테이블의 모든 행 + 오른쪽 일치 행. 없으면 NULL.\n"
            "   주문 없는 사용자도 포함하려면 LEFT JOIN 사용.\n\n"
            "3. **RIGHT (OUTER) JOIN**: 오른쪽 테이블의 모든 행 + 왼쪽 일치 행. LEFT의 거울.\n\n"
            "4. **FULL (OUTER) JOIN**: 양쪽 모든 행. 일치 없으면 해당 쪽이 NULL.\n"
            "   MySQL은 기본 미지원 → `LEFT JOIN UNION RIGHT JOIN`으로 흉내.\n\n"
            "5. **CROSS JOIN**: 두 테이블의 데카르트 곱(모든 조합). 통상 피함.\n\n"
            "6. **SELF JOIN**: 같은 테이블을 별칭 두 개로 조인 (예: 직원·관리자 계층).\n\n"
            "성능에는 조인 키의 인덱스와 옵티마이저의 조인 순서 결정이 핵심입니다."
        ),
    },
    {
        "question": "머신러닝의 지도학습, 비지도학습, 강화학습 차이를 설명해줘.",
        "answer": (
            "머신러닝은 학습 방식에 따라 크게 세 가지로 분류됩니다.\n\n"
            "1. **지도학습(Supervised Learning)**: 입력과 정답 레이블 쌍으로 학습.\n"
            "   - 회귀(연속값 예측): 집값, 수요 예측\n"
            "   - 분류(이산 레이블): 스팸 탐지, 이미지 분류\n"
            "   - 대표 알고리즘: 선형 회귀, 결정 트리, 랜덤 포레스트, 신경망\n\n"
            "2. **비지도학습(Unsupervised Learning)**: 레이블 없이 데이터의 구조를 발견.\n"
            "   - 군집화(Clustering): 고객 세분화, 문서 그룹핑\n"
            "   - 차원 축소: PCA, t-SNE, UMAP\n"
            "   - 이상치 탐지, 밀도 추정\n\n"
            "3. **강화학습(Reinforcement Learning)**: 환경과 상호작용하며 누적 보상을 최대화.\n"
            "   - 에이전트·상태·행동·보상의 프레임워크\n"
            "   - 게임 AI(알파고), 로봇 제어, 추천 시스템\n"
            "   - Q-러닝, 정책 경사(Policy Gradient), 액터-크리틱\n\n"
            "최근에는 자기 지도 학습(Self-Supervised)이 대규모 언어 모델 사전 학습의 기반이 됩니다."
        ),
    },
    {
        "question": "유닛 테스트와 통합 테스트의 차이를 알려줘.",
        "answer": (
            "두 테스트는 다른 계층의 문제를 잡기 위한 도구입니다.\n\n"
            "**유닛 테스트(Unit Test)**:\n"
            "- 범위: 하나의 함수/클래스 등 가장 작은 코드 단위\n"
            "- 의존성: 외부 시스템(DB, API, 파일 등)을 mock 또는 stub으로 대체\n"
            "- 속도: 매우 빠름 (수 ms)\n"
            "- 목적: 알고리즘 정확성, 엣지 케이스, 내부 로직\n\n"
            "**통합 테스트(Integration Test)**:\n"
            "- 범위: 여러 모듈·외부 시스템의 상호작용\n"
            "- 의존성: 실제 DB/API/메시지 브로커 사용(또는 테스트 컨테이너)\n"
            "- 속도: 수백 ms ~ 수 초\n"
            "- 목적: 모듈 간 계약 검증, 데이터 흐름, 설정 정합성\n\n"
            "**Test Pyramid**: 유닛 테스트를 많이, 통합을 적당히, E2E는 소수. 하단이 빠르고 "
            "저렴하며 상단은 느리고 비싸지만 사용자 관점에 가깝습니다.\n\n"
            "모킹 과용은 테스트가 실제와 괴리될 위험이 있으니 통합 테스트로 주기적 검증이 필요합니다."
        ),
    },
    {
        "question": "async와 await의 의미를 자세히 설명해줘.",
        "answer": (
            "async와 await는 비동기 프로그래밍을 **동기처럼 읽히도록** 문법적으로 지원하는 키워드입니다.\n\n"
            "**async**: 함수를 **코루틴**으로 선언합니다. 호출해도 즉시 실행되지 않고 코루틴 객체를 "
            "반환하며, 이벤트 루프에 스케줄링되어야 실행됩니다.\n\n"
            "**await**: 다른 코루틴이나 Future의 완료를 **비동기로** 기다립니다. 대기 동안 이벤트 "
            "루프는 다른 코루틴을 실행하므로 전체가 블로킹되지 않습니다.\n\n"
            "**실행 모델**:\n"
            "- 싱글 스레드 기반 협력적 멀티태스킹\n"
            "- await 지점에서 제어권을 양도 (yield)\n"
            "- I/O 대기가 많은 워크로드에서 처리량 극대화\n\n"
            "**사용 지침**:\n"
            "- CPU 바운드 작업은 asyncio로 해결되지 않음 → 프로세스 풀 사용\n"
            "- 블로킹 코드(open, requests 등)를 async 함수에서 호출하면 이벤트 루프를 멈춤 → "
            "`asyncio.to_thread` 또는 aiohttp 같은 비동기 라이브러리 사용\n"
            "- Python은 asyncio, JavaScript는 Promise 기반으로 대체로 동일한 개념"
        ),
    },
    {
        "question": "GraphQL과 REST의 차이를 설명해줘.",
        "answer": (
            "GraphQL과 REST는 API 설계의 두 가지 접근 방식입니다.\n\n"
            "**REST**:\n"
            "- 리소스 중심: /users/42 같은 URI로 리소스 식별\n"
            "- 여러 엔드포인트: 각 리소스/컬렉션마다 별도\n"
            "- HTTP 메서드로 작업 표현 (GET/POST/PUT/DELETE)\n"
            "- 응답 형태가 서버에 고정 (over-fetching / under-fetching 발생 가능)\n\n"
            "**GraphQL**:\n"
            "- 단일 엔드포인트(/graphql)로 모든 쿼리 처리\n"
            "- 클라이언트가 필요한 필드를 명시 → over-fetching 해결\n"
            "- 타입 스키마가 문서 역할 + 자동 도구(GraphiQL, Apollo Studio)\n"
            "- 여러 리소스를 한 번의 요청으로 가져옴 (N+1 문제는 DataLoader로 완화)\n\n"
            "**선택 기준**:\n"
            "- 프론트엔드 다양성이 크고 over-fetch 부담이 크다면 GraphQL\n"
            "- 캐시가 중요하고 HTTP 시맨틱을 살려야 한다면 REST\n"
            "- 파일 업로드, 스트리밍은 REST가 편함\n\n"
            "최근에는 tRPC·gRPC 등 타입 안전한 대안도 활발합니다."
        ),
    },
]


# ─────────────────────────────────────────────
# 서브에이전트 판단 템플릿 — v7.0 Phase 9 (B 방식)
# Worker에게 "언제 Agent(scout)를 호출하고 언제 직접 응답할지" 학습시킨다.
#
# 각 템플릿은 "trigger" 타입에 따라 두 방향의 응답을 생성한다:
#   - "use_scout": Agent 도구로 scout를 호출하는 assistant 메시지
#   - "direct_answer": 도구 호출 없이 텍스트만 반환하는 assistant 메시지
#   - "single_tool": 특정 단일 도구(Read/Grep 등)만 호출 (scout 없이)
#
# 목표 분포: use_scout 40% / direct_answer 40% / single_tool 20%
# (부정 샘플 60%로 남용 방지를 강화 — CPU Scout는 느리므로 보수적 기본값)
# ─────────────────────────────────────────────
_SUBAGENT_TEMPLATES: list[dict[str, Any]] = [
    # ── use_scout: 대규모/다중 파일 탐색이 필요한 요청 ──
    {
        "trigger": "use_scout",
        "user_prompts": [
            # 프로젝트 구조 파악 류
            "이 프로젝트의 전체 구조를 요약해줘.",
            "core 디렉토리에 어떤 모듈들이 있는지 탐색해서 알려줘.",
            "이 저장소의 주요 기능을 전수 조사해서 정리해줘.",
            "이 프로젝트의 아키텍처를 폭넓게 분석해줘.",
            "Give me a high-level overview of the src directory.",
            "Summarize the overall architecture of this project.",
            "이 레포가 어떻게 구성돼 있는지 전반적으로 설명해줘.",
            "프로젝트 디렉토리 구조와 각 모듈의 역할을 알려줘.",
            # 다중 파일 검색/추적
            "프로젝트에서 인증 관련 코드를 전부 찾아줘.",
            "이 코드베이스에서 deprecated된 API 사용처를 다 찾아줘.",
            "Find all usages of OldClass across the codebase.",
            "여러 디렉토리에 흩어져 있는 X 관련 구현을 전부 수집해줘.",
            "이 저장소에서 A와 B가 서로 어떻게 상호작용하는지 코드로 추적해줘.",
            "프로젝트 전반에서 동일한 패턴이 반복되는 곳을 찾아줘.",
            "5계층 권한 시스템이 어느 파일들에 구현돼 있는지 전수 조사해줘.",
            # 이해/탐색
            "이 프로젝트에서 테스트 전략이 어떻게 구성돼 있는지 탐색해줘.",
            "주요 비즈니스 로직이 어디에 있는지 찾아 설명해줘.",
            "이 저장소의 진입점(entry point)부터 주요 흐름을 따라가며 해설해줘.",
            "프로젝트에서 DB 연결과 쿼리 관련 코드를 전반적으로 훑어봐줘.",
            "여러 파일에 걸쳐 있는 로깅 설정을 전체적으로 조사해줘.",
            # 리팩토링·버그 탐색
            "코드베이스 전체에서 TODO 주석이 붙은 곳을 다 찾아줘.",
            "프로젝트에서 사용되지 않는(dead) 함수가 있을지 조사해줘.",
            "모든 설정 파일이 어디에 있는지 전부 찾아줘.",
            "이 프로젝트 API 엔드포인트들이 모두 어디 정의돼 있는지 탐색해줘.",
            "코드베이스에서 동일한 SQL 쿼리가 중복되는 위치를 찾아줘.",
            # 영문 추가
            "Explore the project and summarize how authentication flows end-to-end.",
            "Find all files that depend on the deprecated `legacy_utils` module.",
            "Audit the project for every place that writes to the filesystem.",
            "Investigate the codebase and map out the tool registration points.",
            "Survey the repository for all Pydantic models and group them by domain.",
        ],
        "scout_prompt": "Explore the project and summarize findings relevant to the user's request.",
    },
    # ── direct_answer: 단순 질문/인사/일반 지식 ──
    {
        "trigger": "direct_answer",
        "user_prompts": [
            "안녕!",
            "고마워.",
            "좋은 하루 보내!",
            "Hello, how are you?",
            "Python에서 리스트와 튜플의 차이가 뭐야?",
            "async와 await의 의미를 간단히 설명해줘.",
            "What is the difference between a process and a thread?",
            "REST API란 무엇인가?",
            "SQL에서 JOIN의 종류를 알려줘.",
            "git rebase와 merge의 차이가 뭐야?",
        ],
        "answers": [
            "안녕하세요! 무엇을 도와드릴까요?",
            "천만에요. 또 필요한 게 있으면 말씀해 주세요.",
            "감사합니다. 좋은 하루 되세요!",
            "I'm doing well, thanks for asking! How can I help?",
            "리스트는 가변(mutable), 튜플은 불변(immutable)입니다. 튜플은 해시 가능하여 dict 키로 쓸 수 있습니다.",
            "async는 코루틴 함수를 선언하고, await는 다른 코루틴의 완료를 비동기로 기다립니다.",
            "A process has its own memory; threads share the parent process's memory.",
            "HTTP 메서드(GET/POST/PUT/DELETE)와 리소스 URL로 상태 없는 통신을 하는 아키텍처 스타일입니다.",
            "INNER/LEFT/RIGHT/FULL OUTER/CROSS JOIN이 있습니다.",
            "rebase는 히스토리를 선형으로 만들고, merge는 분기를 그대로 보존하는 merge commit을 만듭니다.",
        ],
    },
    # ── single_tool: 단일 파일/단일 도구로 충분한 작업 ──
    {
        "trigger": "single_tool",
        "user_prompts": [
            "config/settings.yaml 파일 읽어줘.",
            "README.md의 내용을 보여줘.",
            "src/main.py의 첫 20줄만 보여줘.",
            "Show me the content of pyproject.toml.",
        ],
        "tool": "Read",
        "tool_inputs": [
            {"file_path": "config/settings.yaml"},
            {"file_path": "README.md"},
            {"file_path": "src/main.py", "offset": 1, "limit": 20},
            {"file_path": "pyproject.toml"},
        ],
    },
    {
        "trigger": "single_tool",
        "user_prompts": [
            "현재 디렉토리의 파일 목록을 보여줘.",
            "core 폴더에 뭐가 있는지 ls해줘.",
            "List files in the tests directory.",
        ],
        "tool": "LS",
        "tool_inputs": [
            {"path": "."},
            {"path": "core"},
            {"path": "tests"},
        ],
    },
]


class BootstrapGenerator:
    """
    Phase 1용 합성 학습 데이터를 생성한다.

    도구 사용 패턴과 추론 패턴을 템플릿에서 무작위로 조합하여
    JSONL 형식의 학습 데이터를 생성한다.
    """

    def __init__(self, seed: int | None = None) -> None:
        """
        Args:
            seed: 재현성을 위한 랜덤 시드. None이면 비결정적.
        """
        self._rng = random.Random(seed)  # noqa: S311
        self._tool_templates = _TOOL_USE_TEMPLATES
        self._reasoning_templates = _REASONING_TEMPLATES
        self._subagent_templates = _SUBAGENT_TEMPLATES
        self._knowledge_templates = _KNOWLEDGE_TEMPLATES

    async def generate(
        self,
        count: int = 1000,
        output_path: str = "data/bootstrap/",
    ) -> dict[str, Any]:
        """
        부트스트랩 데이터를 JSONL로 생성한다.

        도구 사용 샘플과 추론 샘플을 지정된 비율(7:3)로 생성한다.

        Args:
            count: 생성할 총 샘플 수
            output_path: 출력 디렉토리 경로

        Returns:
            생성 통계: tool_samples, reasoning_samples, total, output_file
        """
        # 출력 디렉토리 생성
        out_dir = Path(output_path)
        out_dir.mkdir(parents=True, exist_ok=True)
        output_file = out_dir / "bootstrap_data.jsonl"

        # Phase 3 비율 (2026-04-17): 도구 45% / 추론 25% / 서브에이전트 15% / 지식 15%
        # Phase 2에서 "짧은 답변 편중" 문제를 장문 지식 카테고리 신규 추가로 완화.
        tool_count = int(count * 0.45)
        reasoning_count = int(count * 0.25)
        subagent_count = int(count * 0.15)
        knowledge_count = count - tool_count - reasoning_count - subagent_count

        samples: list[dict[str, Any]] = []

        # 도구 사용 샘플
        for _ in range(tool_count):
            sample = self._generate_tool_sample()
            if sample:
                samples.append(sample)

        # 추론 샘플
        for _ in range(reasoning_count):
            sample = self._generate_reasoning_sample()
            if sample:
                samples.append(sample)

        # 서브에이전트 판단 샘플 — v7.0 Phase 9
        for _ in range(subagent_count):
            sample = self._generate_subagent_sample()
            if sample:
                samples.append(sample)

        # 장문 지식 답변 샘플 — Phase 3 신규
        for _ in range(knowledge_count):
            sample = self._generate_knowledge_sample()
            if sample:
                samples.append(sample)

        # 순서를 셔플하여 학습 시 편향 방지
        self._rng.shuffle(samples)

        # JSONL로 저장 — 학습 데이터는 소량이므로 동기 I/O로 충분하다
        with open(output_file, "w", encoding="utf-8") as f:  # noqa: ASYNC230
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")

        stats = {
            "tool_samples": tool_count,
            "reasoning_samples": reasoning_count,
            "subagent_samples": subagent_count,
            "knowledge_samples": knowledge_count,
            "total": len(samples),
            "output_file": str(output_file),
        }

        logger.info(
            "부트스트랩 데이터 생성 완료: %d개 (도구: %d, 추론: %d, 서브에이전트: %d, 지식: %d) → %s",
            stats["total"],
            stats["tool_samples"],
            stats["reasoning_samples"],
            stats["subagent_samples"],
            stats["knowledge_samples"],
            output_file,
        )

        return stats

    def _generate_tool_sample(self) -> dict[str, Any]:
        """
        도구 사용 템플릿에서 하나의 학습 샘플을 생성한다.

        OpenAI의 tool_calls 형식에 맞춰 messages 배열을 구성한다.
        """
        template = self._rng.choice(self._tool_templates)
        tool_name = template["tool"]
        instruction = template["instruction"]
        tool_input = dict(template["input"])

        # 템플릿 변수를 구체적인 값으로 치환한다
        if "paths" in template:
            path = self._rng.choice(template["paths"])
            instruction = instruction.replace("{path}", path)
            for key, val in tool_input.items():
                if isinstance(val, str) and "{path}" in val:
                    tool_input[key] = val.replace("{path}", path)

        if "ranges" in template:
            rng_choice = self._rng.choice(template["ranges"])
            instruction = instruction.replace("{start}", str(rng_choice["start"]))
            instruction = instruction.replace("{count}", str(rng_choice["count"]))
            for key, val in tool_input.items():
                if isinstance(val, str):
                    val = val.replace("{start}", str(rng_choice["start"]))
                    val = val.replace("{count}", str(rng_choice["count"]))
                    tool_input[key] = val

        if "commands" in template:
            command = self._rng.choice(template["commands"])
            instruction = instruction.replace("{command}", command)
            tool_input["command"] = command

        if "patterns" in template:
            pattern = self._rng.choice(template["patterns"])
            instruction = instruction.replace("{pattern}", pattern)
            tool_input["pattern"] = pattern

        if "queries" in template:
            query = self._rng.choice(template["queries"])
            instruction = instruction.replace("{query}", query)
            tool_input["pattern"] = query

        if "tasks" in template:
            task = self._rng.choice(template["tasks"])
            instruction = instruction.replace("{task}", task)
            tool_input["task"] = task

        if "contents" in template:
            content = self._rng.choice(template["contents"])
            tool_input["content"] = content

        if "edits" in template:
            edit = self._rng.choice(template["edits"])
            instruction = instruction.replace("{old}", edit["old"])
            instruction = instruction.replace("{new}", edit["new"])
            tool_input["old_string"] = edit["old"]
            tool_input["new_string"] = edit["new"]

        # OpenAI tool_calls 형식의 assistant 메시지
        # 재현성을 위해 seeded RNG로 ID를 생성한다 (uuid4는 비결정적)
        tool_call_id = f"call_{self._rng.randbytes(12).hex()}"

        return {
            "messages": [
                {"role": "user", "content": instruction},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": tool_call_id,
                            "type": "function",
                            "function": {
                                "name": tool_name,
                                "arguments": json.dumps(tool_input, ensure_ascii=False),
                            },
                        }
                    ],
                },
            ],
            "metadata": {
                "source": "bootstrap",
                "category": "tool_use",
                "tool": tool_name,
            },
        }

    def _generate_reasoning_sample(self) -> dict[str, Any]:
        """
        추론 템플릿에서 하나의 학습 샘플을 생성한다.

        단계별 사고 과정(chain-of-thought)을 포함한 응답을 구성한다.
        """
        template = self._rng.choice(self._reasoning_templates)
        category = template["category"]
        instruction_tpl = template["instruction"]
        reasoning_tpl = template["reasoning"]

        # 카테고리별 구체적 값 선택 및 치환
        if category == "debugging" and "errors" in template:
            chosen = self._rng.choice(template["errors"])
            instruction = instruction_tpl.replace("{error}", chosen["error"])
            reasoning = reasoning_tpl.replace("{error}", chosen["error"])
            reasoning = reasoning.replace("{cause}", chosen["cause"])
            reasoning = reasoning.replace("{solution}", chosen["solution"])

        elif category == "architecture" and "questions" in template:
            chosen = self._rng.choice(template["questions"])
            instruction = instruction_tpl.replace("{question}", chosen["question"])
            reasoning = reasoning_tpl.replace("{options}", chosen["options"])
            reasoning = reasoning.replace("{conclusion}", chosen["conclusion"])

        elif category == "code_review" and "reviews" in template:
            chosen = self._rng.choice(template["reviews"])
            instruction = instruction_tpl.replace("{code}", chosen["code"])
            reasoning = reasoning_tpl.replace("{issues}", chosen["issues"])
            reasoning = reasoning.replace("{improvements}", chosen["improvements"])

        elif category == "security_analysis" and "analyses" in template:
            # 보안 취약점 분석 — 코드, 취약점, 공격 시나리오, 수정 방법을 치환한다
            chosen = self._rng.choice(template["analyses"])
            instruction = instruction_tpl.replace("{code}", chosen["code"])
            reasoning = reasoning_tpl.replace("{vulnerability}", chosen["vulnerability"])
            reasoning = reasoning.replace("{attack_scenario}", chosen["attack_scenario"])
            reasoning = reasoning.replace("{fix}", chosen["fix"])

        elif category == "performance" and "perf_issues" in template:
            # 성능 최적화 분석 — 코드, 병목, 영향, 최적화 방법을 치환한다
            chosen = self._rng.choice(template["perf_issues"])
            instruction = instruction_tpl.replace("{code}", chosen["code"])
            reasoning = reasoning_tpl.replace("{bottleneck}", chosen["bottleneck"])
            reasoning = reasoning.replace("{impact}", chosen["impact"])
            reasoning = reasoning.replace("{optimization}", chosen["optimization"])

        elif category == "refactoring" and "smells" in template:
            # 리팩토링 제안 — 코드, 코드 스멜, 근본 원인, 리팩토링 방법을 치환한다
            chosen = self._rng.choice(template["smells"])
            instruction = instruction_tpl.replace("{code}", chosen["code"])
            reasoning = reasoning_tpl.replace("{smell}", chosen["smell"])
            reasoning = reasoning.replace("{root_cause}", chosen["root_cause"])
            reasoning = reasoning.replace("{refactoring}", chosen["refactoring"])

        else:
            # 알 수 없는 카테고리 — 기본 처리
            instruction = instruction_tpl
            reasoning = reasoning_tpl

        return {
            "messages": [
                {"role": "user", "content": instruction},
                {"role": "assistant", "content": reasoning},
            ],
            "metadata": {
                "source": "bootstrap",
                "category": f"reasoning_{category}",
            },
        }

    def _generate_subagent_sample(self) -> dict[str, Any]:
        """
        v7.0 Phase 9: 서브에이전트 판단 학습 샘플을 생성한다.

        Worker(Qwen)가 "언제 Agent(scout)를 호출하고 언제 직접 응답할지"를
        학습할 수 있는 대조적 샘플을 만든다. 세 가지 trigger 타입:

          - use_scout: 대규모 탐색 요청 → Agent 도구 호출
          - direct_answer: 단순 질문/인사 → 텍스트만 반환 (도구 없음)
          - single_tool: 단일 파일/단일 도구로 충분 → 직접 도구 호출 (Scout 없이)

        부정 샘플(direct_answer + single_tool)이 긍정(use_scout)의 1.5배로
        많이 생성되어 Scout 남용을 억제한다.
        """
        template = self._rng.choice(self._subagent_templates)
        trigger = template["trigger"]
        user_prompt = self._rng.choice(template["user_prompts"])

        # OpenAI tool_calls 형식의 ID (재현성을 위해 seeded RNG 사용)
        tool_call_id = f"call_{self._rng.randbytes(12).hex()}"

        if trigger == "use_scout":
            # 긍정 샘플 — Agent(scout) 호출
            scout_prompt = template["scout_prompt"]
            return {
                "messages": [
                    {"role": "user", "content": user_prompt},
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": tool_call_id,
                                "type": "function",
                                "function": {
                                    "name": "Agent",
                                    "arguments": json.dumps(
                                        {
                                            "prompt": scout_prompt,
                                            "subagent_type": "scout",
                                        },
                                        ensure_ascii=False,
                                    ),
                                },
                            }
                        ],
                    },
                ],
                "metadata": {
                    "source": "bootstrap",
                    "category": "subagent_use_scout",
                    "tool": "Agent",
                },
            }

        if trigger == "direct_answer":
            # 부정 샘플 — 텍스트만 응답, 도구 호출 없음
            idx = template["user_prompts"].index(user_prompt)
            answer = template["answers"][idx]
            return {
                "messages": [
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": answer},
                ],
                "metadata": {
                    "source": "bootstrap",
                    "category": "subagent_direct_answer",
                },
            }

        if trigger == "single_tool":
            # 부정 샘플 — 단일 도구 직접 호출 (Scout 경유 안 함)
            idx = template["user_prompts"].index(user_prompt) % len(template["tool_inputs"])
            tool_input = dict(template["tool_inputs"][idx])
            return {
                "messages": [
                    {"role": "user", "content": user_prompt},
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": tool_call_id,
                                "type": "function",
                                "function": {
                                    "name": template["tool"],
                                    "arguments": json.dumps(
                                        tool_input, ensure_ascii=False
                                    ),
                                },
                            }
                        ],
                    },
                ],
                "metadata": {
                    "source": "bootstrap",
                    "category": "subagent_single_tool",
                    "tool": template["tool"],
                },
            }

        # 알 수 없는 trigger — 안전하게 None 반환 (generate에서 필터됨)
        return None  # type: ignore[return-value]

    def _generate_knowledge_sample(self) -> dict[str, Any]:
        """
        Phase 3 신규: 장문 지식/설명 답변 샘플을 생성한다.

        Phase 2 학습에서 direct_answer 샘플이 "인사/단답" 중심이라 Worker가
        긴 설명 요청 시 응답이 짧아지는 퇴보가 관찰됐다. 이를 상쇄하기 위해
        구조화된 3~6개 단락 답변 샘플을 별도 카테고리로 학습시킨다.
        """
        template = self._rng.choice(self._knowledge_templates)
        return {
            "messages": [
                {"role": "user", "content": template["question"]},
                {"role": "assistant", "content": template["answer"]},
            ],
            "metadata": {
                "source": "bootstrap",
                "category": "knowledge_explanation",
            },
        }
