# Project Nexus — 개발 진행 상황

> 이 파일은 세션 간 진행 상황 공유를 위해 매 작업마다 업데이트된다.

---

## 현재 상태

- **현재 Phase**: Phase 0.5 완료, Phase 1.0 시작 예정
- **마지막 업데이트**: 2026-04-14
- **브랜치**: main

---

## Phase 0.5: Foundation (완료)

### v0.1.0 — 프로젝트 스켈레톤 (2026-04-14)
**커밋**: `1678f2e`

- `pyproject.toml` — 의존성, ruff/pytest/mypy 설정 통합
- `.gitignore` — Python, data/, models/, checkpoints/, logs/ 제외
- `requirements.txt`, `requirements-train.txt`, `requirements-dev.txt`
- `.env.example` — 환경 변수 템플릿
- 디렉토리 구조 생성 (Ch.21 기준 16개 패키지 + `__init__.py`)
- `config/*.yaml` 5종: nexus_config, tool_mappings, model_profiles, permission_rules, logging_config
- `tests/conftest.py` — 공통 fixture (mock GPU/Redis/PG)

### v0.1.1 — 핵심 부트스트랩 모듈 (2026-04-14)
**커밋**: `9746efb`

- `core/state.py` — GlobalState 싱글톤 (DAG leaf 격리, 스레드 안전 토큰 카운터)
- `core/config.py` — NexusConfig Pydantic v2 설정 시스템 (YAML 로딩, 에어갭 검증)
- `core/bootstrap.py` — Phase 1 초기화 (로깅, 종료 핸들러, GPU 사전연결, 플랫폼 감지)
- `tests/unit/test_state.py` — 11개 테스트
- `tests/unit/test_config.py` — 10개 테스트
- ruff 클린, 전체 21개 테스트 통과

---

## Phase 1.0: Model Layer (진행 중)

> GPU 감지, 추론 클라이언트, 프롬프트 포매터 구현 예정

---

## 다음 세션 시작 시 참고

1. Phase 0.5 완료 — state, config, bootstrap 구현 및 테스트 통과
2. Phase 1.0 대상 파일: `core/model/` 디렉토리 전체
3. 사양서 참조: Ch.4 (라인 2979~5148), Ch.5.2 ModelProvider (라인 5549~6174)
4. GPU 서버 주소: 192.168.22.28, DB 서버: 192.168.10.39
5. 타겟 GPU: RTX 5090 (32GB), INT4 양자화
