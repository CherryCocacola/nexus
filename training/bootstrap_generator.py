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
            "이 프로젝트의 전체 구조를 요약해줘.",
            "core 디렉토리에 어떤 모듈들이 있는지 알려줘.",
            "이 저장소의 주요 기능을 파악해서 정리해줘.",
            "프로젝트에서 인증 관련 코드를 전부 찾아줘.",
            "이 코드베이스에서 deprecated된 API 사용처를 다 찾아줘.",
            "Give me a high-level overview of the src directory.",
            "Find all usages of OldClass across the codebase.",
            "Summarize the architecture of this project.",
            "core/orchestrator 아래 파일들의 역할을 각각 설명해줘.",
            "이 프로젝트에서 테스트 전략이 어떻게 구성돼 있는지 탐색해줘.",
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

        # v7.0: 도구 60% / 추론 30% / 서브에이전트 판단 10%
        # subagent 10%에서 다시 use_scout 40% / direct_answer 40% / single_tool 20%
        # → 부정 샘플이 전체의 60%로 Scout 남용을 억제한다.
        tool_count = int(count * 0.60)
        reasoning_count = int(count * 0.30)
        subagent_count = count - tool_count - reasoning_count

        samples: list[dict[str, Any]] = []

        # 도구 사용 샘플 생성
        for _ in range(tool_count):
            sample = self._generate_tool_sample()
            if sample:
                samples.append(sample)

        # 추론 샘플 생성
        for _ in range(reasoning_count):
            sample = self._generate_reasoning_sample()
            if sample:
                samples.append(sample)

        # 서브에이전트 판단 샘플 생성 — v7.0 Phase 9
        for _ in range(subagent_count):
            sample = self._generate_subagent_sample()
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
            "total": len(samples),
            "output_file": str(output_file),
        }

        logger.info(
            "부트스트랩 데이터 생성 완료: %d개 (도구: %d, 추론: %d, 서브에이전트: %d) → %s",
            stats["total"],
            stats["tool_samples"],
            stats["reasoning_samples"],
            stats["subagent_samples"],
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
