# NOVA CLI — Claude Code / GPT Codex 기능 이식 및 권한 인터랙션 개선 계획 (v2)

작성일 2026-08-04 · **v2 = Fable5 검토 반영** · 대상 `cli/*` + `core/permission/*`·`core/tools/executor.py`·`core/orchestrator/context_manager.py` 최소 확장
판정 **조건부 GO → 5개 조건 반영 완료**

---

## 0. 목표 (사용자 요청 5가지)

1. Claude Code / GPT Codex의 최신 CLI 기능을 NOVA CLI로 이식
2. NOVA CLI의 현 기능·디자인·실행방법 전수 조사 (§1 완료)
3. NOVA에 적용 가능한 기능만 선별 이식
4. 실행화면·표시 내용을 Claude Code와 유사하게
5. 파일 수정/프로세스 진행 시 **중간 확인·선택·피드백**(Claude의 manual/auto/accept mode) 도입

---

## 1. NOVA CLI 현황 (조사 결과)

### 1.1 구조
```
cli/
  commands.py (439)  Click 그룹: nexus {chat|ask|version|sessions|health}
  repl.py     (942)  NexusREPL — Rich + prompt-toolkit 대화 루프
  formatters.py(438) OutputFormatter — StreamEvent → Rich 렌더러블
```
- 진입점 `nexus` → 인자 없으면 `chat` 위임. `--model/--permission-mode/--resume/--log-level`.
- 4-Tier 체인은 core 담당, CLI는 표현 계층. 의존성 `cli/ → core/` 단방향.

### 1.2 실행 화면 (현재)
- **배너**: IDINO 픽셀 로고 + 둥근 Panel(모델 라우팅·인프라·세션·권한모드). 이미 Claude Code풍.
- **스피너 단계**: 요청 분석 중 → 모델 추론 중 → 응답 생성 중 → `{tool}` 실행 중. TEXT_DELTA 타자기.
- **도구 표시**: `format_tool_use`가 도구명+입력을 **JSON 전체 Panel**로(장황). 결과 녹/적 Panel(50줄 절단). Todo 체크박스 Panel.
- **슬래시 명령**: `/help /clear /exit /model /config /session /thinking` (7개).

### 1.3 권한 인터랙션 (현재) — ★Fable5 검증 반영★
- bootstrap `④-a`: 배포 config `permission_enforcement.enabled=true, mode=enforce` → PermissionPipeline 생성.
- **핵심 실측**: executor의 ASK 게이트는 **파이프라인 판정이 아니라 도구 자체 `tool.check_permissions()`**다.
  - `WriteTool.check_permissions()`는 모드 무관 **항상 ASK 반환**(`write_tool.py:123-138`). Edit/MultiEdit 동일.
  - enforce 파이프라인은 **deny만 차단**, ASK/ALLOW는 통과(`executor.py:260-262`, "ASK 강제는 P3 범위").
  - → **파이프라인 모드만 바꿔서는 Edit/Write 프롬프트가 사라지지 않는다.** (A1 재설계 근거)
- REPL `prompt_permission(tool, msg)`: `(Y)/(N)/(A)`. A는 미저장 TODO. auto/bypass/trust 모드면 ask_handler 미주입.
- 모드 이원화:
  - `PermissionModeValue`(세션/CLI): default, auto, plan, trust, bypass, headless, deny_all
  - `PermissionMode`(파이프라인): DEFAULT, **ACCEPT_EDITS**, BYPASS_PERMISSIONS, DONT_ASK, PLAN, AUTO, BUBBLE
  - `mode_mapping.py` 변환. ACCEPT_EDITS로 가는 세션값 부재.
- **기존 자산(신설 불필요)**:
  - `pipeline.update_context()` 존재(`pipeline.py:139-147`) — "세션 도중 DEFAULT→ACCEPT_EDITS 갈아끼움" 용도 명시.
  - 압축은 `auto_compact_if_needed`(임계 초과 시)·`emergency_compact`만(`context_manager.py:218,295`) — **수동 강제 경로는 신규 필요**.

### 1.4 확인된 갭
G1 런타임 모드전환 없음 · G2 accept-edits 도달 불가 · G3 diff 미리보기 없음 · G4 "항상 허용" 미저장 · G5 거부 후 피드백 경로 없음 · G6 도구 표시 장황 · G7 `!` bash 없음 · G8 슬래시 빈약(/compact /cost /resume /diff /copy /mode /save /verbose) · G9 `/` 자동완성 없음

---

## 2. Claude Code / Codex 기능 → NOVA 적용 판정

**범례** ✅이식 · 🔶조건부 · ❌제외 (Fable5 §3 전건 동의)

| 기능 | 출처 | 판정 | 사유·조정 |
|---|---|---|---|
| 권한모드 런타임 전환(Shift+Tab 순환) | Claude | ✅ | manual→accept-edits→plan. **프롬프트 경계에서만** 전환(§4-2) |
| accept-edits 모드 | Claude/Codex(Auto Edit) | ✅ | **모드 인지형 ask_handler로 구현**(A1). 파일수정 자동승인·Bash는 ASK 유지 |
| numbered 권한 프롬프트(1 예/2 항상/3 거부+피드백) | Claude | ✅ | ask_handler **v2 계약**으로 확장(A3) |
| 파일수정 diff 미리보기 후 승인 | Claude/Codex | ✅ | ASK 시 unified diff Panel. 크기·바이너리 상한(§B) |
| "이 세션 항상 허용" 저장 | Claude | ✅ | 세션 allow-set. **Bash는 명령 프리픽스 단위·DANGEROUS 제외**(§A4) |
| `!` bash 패스스루 | Claude | ✅ | CommandFilter 경유 + **감사로그** + plan/deny_all 비활성 + v1 표시전용(§D1) |
| `/compact` 수동 압축 | Claude | ✅ | **강제 압축 경로 신규**(context_manager) + 전/후 토큰 |
| `/resume` 인-REPL 세션 선택 | Claude | ✅ | **session_id 재바인드 포함**(단순 restore_messages만으론 히스토리 혼입) |
| `/diff` 작업트리 변경 | Codex | ✅ | 로컬 git diff(에어갭 안전) |
| `/save` 대화 저장 | Claude | ✅ | transcript→markdown 덤프(저비용). G8 복원 |
| `/cost`·상태줄 토큰 | Claude | 🔶 | $비용 아닌 **토큰**만. 하단 `bottom_toolbar` |
| `/copy` 마지막 응답 복사 | Claude | 🔶 | pyperclip → 실패 시 OSC52 |
| 도구표시 축약(⏺/⎿) | Claude | ✅ | G6. 1줄 요약+들여쓴 결과. 상세는 `/verbose` |
| 멀티라인 붙여넣기(bracketed paste+Alt+Enter) | Claude/Codex | ✅ | **Fable5 추가** — 코드 붙여넣기 일상 시나리오, 저비용 |
| Ctrl+R 히스토리 검색 | Claude | ✅ | **Fable5 추가** — PromptSession 옵션, 사실상 공짜 |
| `esc/Ctrl+C 중단` 힌트 | Claude | 🔶 | 취소는 이미 존재 → 힌트 표기 |
| `#` 메모리 단축(→MemoryManager) | Claude | 🔶 | Phase 3 후순위. NOVA 장기메모리와 자연 매핑 |
| `@파일` 참조 자동완성 | Claude/Codex | 🔶 | 모델이 Read 보유 → Phase 3 후순위 |
| `/model` 수동 전환 | Claude/Codex | ❌ | 질의별 라우팅 자동결정(thinking 제외와 동일 논리) |
| `/init` | Claude | ❌ | 대상(코드베이스 요약) 부재 |
| `$`비용/과금 | Claude | ❌ | 에어갭 로컬, 과금 없음(→`/cost` 토큰이 대체) |
| MCP 서버 UI | Claude | ❌ | 웹 채널 소관, CLI 범위 밖 |
| 웹검색/브라우저 | Codex | ❌ | 에어갭 위반 |
| 큐잉 입력(스트리밍 중 타이핑) | Claude | ❌ | run_in_executor 구조상 재설계 고비용·저가치(Fable5) |
| `/vim`·Esc-Esc 편집·재생성 | Claude | ❌ | 가치 낮음/웹에 존재 |

---

## 3. 수정 계획 (Phase 단위)

> 원칙 무회귀 · 한글주석 · 4-Tier 불변 · fail-closed. CLI는 core를 "사용"만.

### Phase A — 권한 모드 시스템 (요청 #5 핵심)

**A1. accept-edits 모드 — 모드 인지형 ask_handler (★치명결함 수정★)**
- enum/매핑/Choice는 표시·파이프라인 정합용으로 추가:
  - `core/state.py` `PermissionModeValue.ACCEPT_EDITS = "accept_edits"`
  - `mode_mapping.py` `ACCEPT_EDITS → PermissionMode.ACCEPT_EDITS`
  - `commands.py` `--permission-mode` Choice에 `accept_edits`
- **실효 로직은 CLI 계층에서** (core 무변경): ask_handler(v2)가 모드 인지형으로 —
  `mode == accept_edits` 이고 도구가 FILE_WRITE 부류(`Write/Edit/MultiEdit/NotebookEdit`)면 **즉시 승인 + "⏺ 자동 승인(accept edits)" 1줄 표시**. Bash 등은 종전대로 프롬프트.
  - 도구 분류는 CLI에 중복 정의하지 말고 **core의 분류를 공개 함수로 노출해 재사용**(드리프트 방지).
- (후속 분리) executor가 파이프라인 ASK/ALLOW를 실제 게이트로 승격하는 정공법은 웹·비대화형 전체 영향 → 이번 범위 밖 TODO.

**A2. 런타임 모드 전환**
- `/mode [default|accept_edits|auto|plan]` + Shift+Tab 순환(default→accept-edits→plan).
- **갱신은 단일 헬퍼 `_apply_mode_change(new_mode)`로 4개 지점 원자 갱신**:
  1. `pipeline.update_context()` + `PermissionContext.model_copy(update={"mode": ...})` — **신설 set_mode 불필요**
  2. `tool_ctx.options["ask_handler_v2"]` 추가/제거(auto/bypass는 자동허용이라 제거)
  3. `tool_ctx.permission_mode`(str)
  4. `GlobalState.permission_mode` + REPL `self._permission_mode`(배너·`/config` 표시)
- **전환은 프롬프트 경계(턴 사이)에서만** — REPL 입력은 `run_in_executor`로 턴 사이에만 읽히므로 스트리밍 중 경합 원천 없음. "스트리밍 도중 전환 미지원" 문서화.
- Shift+Tab 콜백은 상태 직접변경 대신 **버퍼에 `/mode next` 주입 후 accept** → 메인 루프에서 처리. 캡처 실패 시 `/mode`만으로 폴백.

**A3. ask_handler v2 계약 (numbered 프롬프트 + 피드백)**
- **옵셔널 인자 2회 증축 대신 v2 계약 1회 도입**:
  - `options["ask_handler_v2"]`: 요청(tool_name, message, tool_input) → 응답(approved, feedback, always_allow) 코루틴.
  - executor는 **v2 우선, 구 `ask_handler`(bool) 폴백**. 웹(미주입)·headless(DONT_ASK로 ASK 미발생)·기존 테스트 무회귀.
- 프롬프트: `1) 예  2) 예(이 세션 항상 허용)  3) 아니오 + 피드백`. 3 선택 시 텍스트 입력 → executor가 tool_use_error 대신 **피드백을 모델에 다음 턴 전달**.

**A4. 세션 allow-list ("항상 허용")**
- REPL `_session_allow`: 일반 도구는 도구명, **Bash는 명령 첫 토큰(프리픽스) 단위**, **DANGEROUS 부류는 등록 대상에서 제외**.
- 경로/프리픽스 매칭은 **realpath + NFC 정규화**(state.py 기존 로직 재사용) 후 비교 — `../` 우회 차단.
- `/config`에 현황 표시.

### Phase B — 파일 수정 diff 미리보기 (요청 #5)
- **B1. diff 렌더러** `formatters.py` `format_diff(path, old, new)` — `difflib.unified_diff` + Rich Syntax(diff), +녹/−적.
- **B2. Edit/Write/MultiEdit 승인 훅**: ask_handler v2가 tool_input 수신 → diff Panel 표시 후 승인.
  - Edit: old/new_string. Write: 기존파일 대비(부재 시 "신규"). MultiEdit: 메모리 내 순차 적용 후 단일 diff, 실패 시 edit별 블록 폴백. old_string 불일치 시 원문 블록 폴백.
  - **상한**: 파일 256KB 초과 → "diff 생략, 경로+바이트". 출력 200줄 초과 → "…외 N줄". 바이너리(NULL/디코드 실패) → "바이너리" 표기.

### Phase C — 디스플레이 Claude Code화 (요청 #4)
- **C1. 도구 표시 축약**: `⏺ {Tool}({핵심인자})` 1줄 + `⎿ {요약/줄수}`. 상세 JSON은 `/verbose` 토글.
- **C2. 상태줄**: prompt-toolkit **`bottom_toolbar`**(Rich Live는 타자기 출력과 간섭 → 회피)에 `[모드] · 토큰 in/out · 세션ID`.
- **C3. 스트리밍 중 `esc/Ctrl+C 중단` 힌트** 스피너 병기.
- **C4. 배너 힌트에 신규 명령 반영 + 권한모드 배지.**

### Phase D — 슬래시/입력 확장 (요청 #1,3)
- **D1. `!` bash 패스스루**: CommandFilter 경유 필수(bypass에서도 유지) + **감사 JSONL 기록** + **plan/deny_all 모드 비활성** + **v1 표시전용**(대화 컨텍스트 미주입, 문서 명시).
- **D2. `/compact`**: context_manager에 **강제 압축 경로 신규** + 전/후 토큰.
- **D3. `/resume`**: `list_transcript_sessions(cli)` → 번호 선택 → **session_id 재바인드 포함** 복원(히스토리 혼입 방지). 설계 확정까지 Stage 2.
- **D4. `/diff`**: 로컬 git diff Syntax Panel(리포 아니면 안내).
- **D5. `/cost`**: 세션 누적 토큰/턴(과금 없음 명시).
- **D6. `/copy`**: 마지막 응답 pyperclip→OSC52.
- **D7. `/save`**: transcript→markdown 저장.
- **D8. `/verbose`**: 도구 상세표시 토글.
- **D9. `/` 자동완성** Completer + **멀티라인 붙여넣기(bracketed paste+Alt+Enter)** + **Ctrl+R 히스토리 검색**.
- **D10. `/help` 갱신**.

### Phase E — 테스트·문서
- 단위: 모드매핑(A1) · 모드전환 4점 헬퍼(A2) · ask_handler v2/피드백(A3) · allow-list 스코프(A4) · diff렌더/상한(B) · bash필터·감사·plan비활성(D1) · 각 슬래시.
- 회귀: `test_cli_repl.py`·`test_tool_executor_ask.py`(v2 계약 폴백 검증).
- 도움말·progress.md 갱신.

---

## 4. 결정 포인트 해소 (Fable5 권고)

1. **ask_handler 계약**: v2 객체 계약 1회 도입, 구 bool 계약 폴백(§A3).
2. **mode 갱신**: 기존 `update_context()` + model_copy, 4지점 단일 헬퍼, **전환은 프롬프트 경계 한정**(경합 원천 차단).
3. **Shift+Tab**: 버퍼에 `/mode next` 주입 방식, 실패 시 `/mode` 폴백.
4. **diff 상한**: 256KB/200줄/바이너리 감지(§B2).
5. **`!` 보안**: CommandFilter 경유 + 감사로그 + plan/deny_all 비활성 + 표시전용.
6. **순서**: 2단계 분할(§6).
7. **제외 판정**: ❌ 5건 전건 타당(Fable5 동의).

## 5. 예상 변경 파일 (core 침습 3점 + CLI 다수)
- **core(최소)**: `state.py`(enum 1줄) · `mode_mapping.py`(매핑 1줄) · `orchestrator/context_manager.py`(강제 압축) · `tools/executor.py`(ask_handler_v2 폴백) · 도구 분류 공개 함수 노출. **`pipeline.py` set_mode 신설 삭제**(update_context 사용).
- **CLI**: `repl.py`(모드전환 헬퍼·v2 핸들러·allow-list·diff·bash·슬래시·상태줄·완성기) · `commands.py`(Choice) · `formatters.py`(diff·축약).
- `tests/unit/*` 다수.

## 6. Phase 순서 (2단계 분할 — Fable5 권고)
- **Stage 1 (핵심 · 요청 #5)**: A1(모드인지 ask_handler)·A2·A3(v2)·A4 + B1·B2 + C1 + D8(/verbose)·D9(자동완성/멀티라인/Ctrl+R)·D10(/help) + 테스트. 권한 UX·diff 승인이 요청 본질.
- **Stage 2 (UX 확장)**: C2(bottom_toolbar 상태줄)·C3·C4 + D1(!)·D2(/compact)·D3(/resume 재바인드)·D4~D7.

## 7. 종합
**조건부 GO의 5개 조건 전부 반영 완료** → 착수 가능. Stage 1부터 진행.
