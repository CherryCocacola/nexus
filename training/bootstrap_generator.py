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

        # 도구 70%, 추론 30% 비율로 생성
        tool_count = int(count * 0.7)
        reasoning_count = count - tool_count

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

        # 순서를 셔플하여 학습 시 편향 방지
        self._rng.shuffle(samples)

        # JSONL로 저장 — 학습 데이터는 소량이므로 동기 I/O로 충분하다
        with open(output_file, "w", encoding="utf-8") as f:  # noqa: ASYNC230
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")

        stats = {
            "tool_samples": tool_count,
            "reasoning_samples": reasoning_count,
            "total": len(samples),
            "output_file": str(output_file),
        }

        logger.info(
            "부트스트랩 데이터 생성 완료: %d개 (도구: %d, 추론: %d) → %s",
            stats["total"],
            stats["tool_samples"],
            stats["reasoning_samples"],
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
