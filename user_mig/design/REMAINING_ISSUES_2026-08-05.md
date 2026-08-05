# 미해결 과제 정리 (2026-08-05 기준)

이번 세션(08-04~05) 종료 시점의 남은 일 전체. 우선순위와 착수 지점을 함께 적는다.
해결 완료 항목은 `progress.md` 참조.

---

## P0 — 결정 필요 (다음 세션 첫 항목)

### 1. 리포 미커밋
이번 세션의 모든 변경이 커밋되지 않았다. 다음 세션에서 혼란을 피하려면 정리 필요.

- **수정** 29개: `web/app.py` · `core/bootstrap.py` · `core/orchestrator/query_engine.py` ·
  `core/tools/executor.py` · `core/tools/implementations/{edit,multi_edit,analyze_image}_tool.py` ·
  `core/permission/{pipeline,mode_mapping}.py` · `core/state.py` · `core/model/hardware_tier.py` ·
  `cli/{repl,commands}.py` · `config/nexus_config.pc.yaml` · 테스트 9종 · `progress.md` 등
- **신규**: `scripts/cleanup_sessions.py` · `core/tools/implementations/{render_preview,scaffold_web}_tool.py` ·
  `assets/frontend_templates/**` · `tests/unit/test_{edit_fuzzy,render_preview_tool,scaffold_web_tool,cleanup_sessions,web_channel_finish,permission_categorize_tool_name,cli_permission_prompt_v2}.py` ·
  `user_mig/design/{VSCODE_PLUGIN_VIBE_JSON_HANDOFF,REMAINING_ISSUES_2026-08-05}.md`
- 커밋 단위 제안(논리별 분리): ①CLI 권한 A1 ②Edit 복구력 ③렌더 자가검증 ④템플릿 스캐폴드
  ⑤채널 분리·finish_reason ⑥세션 정리 도구 ⑦문서

### 2. 112 이미지 정합성
오늘 배포는 `docker cp` + `docker commit`(`nexus-web:channelfix-20260805`, `latest` 갱신)으로
영속화했다. 컨테이너 재생성에는 견디지만 **Dockerfile 기반 정식 재빌드는 아니다.**
리포 커밋 후 정식 빌드로 한 번 정리하는 것이 안전하다.

---

## P1 — 기능·품질 (효과 큼)

### 3. CLI Stage 1 나머지 (A1만 완료)
`CLI_CLAUDE_CODEX_PORT_PLAN.md` 기준 남은 항목.

- **A2 런타임 모드 전환**: `/mode` + Shift+Tab 순환. `_apply_mode_change()` 단일 헬퍼로 4지점
  원자 갱신(pipeline.update_context + model_copy / ask_handler_v2 주입·제거 /
  tool_ctx.permission_mode / GlobalState·배너). 전환은 프롬프트 경계에서만.
  **주의**: `GlobalState.permission_mode`가 CLI 인자와 미동기(기존부터) — 여기서 함께 배선.
- **A4 세션 allow-list 확장**: 현재 도구명 단위. Bash는 명령 프리픽스 단위로,
  경로는 realpath+NFC 정규화 후 비교(`../` 우회 차단).
- **B1~B2 diff 미리보기**: `formatters.format_diff()` + ask_handler_v2에서 Edit/Write/MultiEdit
  승인 전 unified diff 표시. 상한 256KB/200줄/바이너리 감지.
- **C1 도구 표시 축약**: `⏺ Tool(핵심인자)` + `⎿ 요약`. 상세는 `/verbose` 토글.
- **D8~D10**: `/verbose` · `/` 자동완성 + 멀티라인 붙여넣기 + Ctrl+R · `/help` 갱신.

### 4. Devstral 코딩 서브모델 통합 미완
B200 8005에 서빙 중이나 **NOVA가 아직 사용하지 않는다**(수동 호출만 가능).

- mistral tool-call 파서 검증(`--tokenizer-mode mistral` / `--tool-call-parser mistral` 필요 여부)
- `config/nexus_config.*.yaml`에 `coder_url`(112=18005 터널 / B200=8005) 추가
- 라우팅에 "코드 작업 → devstral-small" 규칙(기존 routing 프로필 확장)
- 에이전트 루프 실전 비교: A.X가 붕괴했던 조건에서 Devstral이 완주하는지
- **원복 절차**: coder tmux 세션 kill → `start_all.sh`의 coder 줄 제거 →
  `run_vllm_fp8.sh.bak-util092` 복원 → vllm 재기동(util 0.92)

### 5. A.X-4.0 모델 한계 3종 (근본 원인)
완화책은 적용했으나 근본 해결은 모델 교체/LoRA.

- 긴 리터럴 재현 불가(경로 오타·스크린샷 경로 환각) → 짧은 파일명·상대 경로로 우회 중
- 장문 출력 degeneration(반복 붕괴) → 템플릿 그라운딩으로 우회 중
- 다목표 턴 조기 종료 → TodoWrite 유도 중
- 근본책: (a)코딩 서브모델 라우팅 #4 (b)코딩 LoRA(데이터 축적 후, 블로커=SO 덤프 입수)

### 6. VSCode 플러그인 적용 확인 대기
핸드오프 문서(`VSCODE_PLUGIN_VIBE_JSON_HANDOFF.md`)를 전달했으나 반영 여부 미확인.
플러그인이 해야 할 것 — ①`X-Client-Channel: api`(또는 `/v1/chat/completions`로 이전)
②`response_format` guided decoding ③changes를 find/replace 조각으로 ④`finish_reason !== "stop"`
이면 파싱 말고 잘림 안내 ⑤`X-Client-Id`로 소비자 식별(선택).

---

## P2 — 개선 (여유 있을 때)

### 7. 템플릿 커버리지 확장
현재 `landing-vue` 1종뿐. 이 유형 밖 요청(대시보드·사내 포털·문서/매뉴얼·폼 중심)은
모델이 직접 작성 폴백을 타 품질이 떨어진다. 유형별 템플릿 추가가 곧 품질이다.

**함께 고칠 것**: 색상 슬롯이 `styles.css`의 `:root`에 있어 모델이 값을 교체하지 않고
위에 삽입하는 실수를 했다(CSS 후순위 규칙상 구값이 이김). 색상도 `app.js`의 SITE로
옮겨 JS에서 CSS 변수를 주입하면 **모델이 CSS를 건드릴 일 자체가 사라진다.**

### 8. 렌더 자가검증 루프 종료 조건
시스템 프롬프트는 "최대 2회 수정 후 보고"인데 실측에서 모델이 4회 돌았다. 비전 모델이
"폰트를 키워라" 같은 일반론적 개선을 계속 제안하기 때문. 종료 기준을 "치명 결함
(스타일 미적용·겹침·빈 영역)이 없으면 종료"로 좁히고, 라운드 카운터를 도구 결과에
실어 주는 방식 검토.

---

## P3 — 기술 부채 (당장 문제 없음)

### 9. OpenAI API 세션이 요청마다 생성
OpenAI 규격에 session 개념이 없어 매 요청 `uuid4()`로 무상태 처리한다(구조적).
일일 자동 정리(cron 04:10, api 7일)로 누적은 해소했다. 세션 재사용이 필요하면
클라이언트가 `/v1/chat`(+`X-Client-Channel`)을 쓰면 된다.

### 10. `web/app.py` 기존 lint 7건
`E402`(import 위치) 6건 + `S110`(try-except-pass) 1건. **HEAD부터 있던 것**이라
이번 변경과 무관하다. 손대면 무관한 diff가 커지므로 별도 정리 커밋으로 분리 권장.

---

## 참고 — 이번 세션 산출물 위치

- 세션 백업: `112:/app/.nexus/sessions_backup_20260805.tar.gz`(475개) ·
  로컬 `scratchpad/local_sessions_backup.tar.gz`(231개)
- 홈페이지 테스트 산출물: `scratchpad/tmp_homepage_backup.tar.gz`(v1/v2/v3, 517KB)
- 자동 정리: 112 crontab `10 4 * * * /home/idino/nova_cleanup.sh`,
  로그 `/home/idino/cleanup_sessions.log`
- 112 이미지 스냅샷: `nexus-web:channelfix-20260805`(= `latest`)
