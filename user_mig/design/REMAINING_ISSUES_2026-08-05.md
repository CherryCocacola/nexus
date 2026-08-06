# 미해결 과제 정리 (2026-08-05 기준)

이번 세션(08-04~05) 종료 시점의 남은 일 전체. 우선순위와 착수 지점을 함께 적는다.
해결 완료 항목은 `progress.md` 참조.

---

## P0 — 다음에 먼저 볼 것

### 1. ~~리포 미커밋~~ · ~~112 이미지 정합성~~ — **해소 (2026-08-05)**
세션 작업 전부를 논리 단위 커밋 36개로 정리했고(작업 트리는 개인 설정 1건만 잔존),
112는 `docker commit`으로 영속화했다(최신 `nexus-web:katex-vendor-20260805` = latest).
리포 ↔ 배포본 해시 대조로 동기 확인 완료.
**단, push는 하지 않았다** — 원격 반영이 필요하면 별도 결정.

### 2. W5 KaTeX 잠금 해제 (부분 완료 상태)
벤더·배선·구분자·오인방지까지 검증됐으나 **분수·루트가 세로로 눌려 잘리는 CSS 충돌**이
남아 `NOVA_MATH_ENABLED=false`로 잠가 둔 상태다(꺼진 동안은 원문 LaTeX 표시 = 무회귀).
- 확인된 것: 폰트 20종 로드 OK, 색상 OK, DOM 구조 OK, line-height 보정은 원인 아님,
  블록/인라인 모드 무관하게 mfrac 높이 20px(2줄 조판이 1줄로 눌림).
- 착수 순서: ①`katex.min.css`의 `.vlist-t`/`.vlist-r` 규칙이 실제 적용되는지 확인
  ②`.message-body` 계열 상속 규칙을 이분 탐색으로 하나씩 비활성화
  ③최후 수단 `.katex` 하위 `all: revert` 후 KaTeX CSS만 재적용
- 재개: 콘솔에서 `window.NOVA_MATH_ENABLED = true` 후 메시지 재렌더로 즉시 실험.

## P1 — 기능·품질 (효과 큼)

### 3. ~~CLI Stage 1~~ — **전체 완료 (2026-08-05)**
A1·A2·A3·A4·B1·B2·C1·D8·D9·D10 모두 구현·검증·커밋 완료. 상세는 progress.md 참조.
- A1/A3 accept-edits 모드 + ask_handler v2(numbered·피드백) — 커밋 3f757c2
- A2 런타임 전환 `/mode`·Shift+Tab, 4지점 원자 갱신 — 커밋 34d85ea
- A4 Bash allow-list(명령 프리픽스, 메타문자·위험명령 차단) — 커밋 39ce121
- B1/B2 diff 미리보기(256KB·200줄·바이너리 상한) — 커밋 1b35aa6
- C1/D8~D10 도구 표시 축약·`/verbose`·자동완성·Alt+Enter·`/help` — 커밋 e220705

**Stage 2 — 전체 완료 (2026-08-05)**: C2 상태줄 · C3 중단 힌트 · D1 `!` bash 패스스루 ·
D2 `/compact` · D3 `/resume` 재바인드 · D4 `/diff` · D5 `/cost` · D6 `/copy` · D7 `/save`.
커밋 70ec21e(슬래시 4종) · fb2f8c6(상태줄·힌트·!bash) · 2e5f5f7(/compact·/resume).
실서버 검증 8/8. **CLI 이식 계획서의 Stage 1·2가 모두 끝났다.**

남은 CLI 후보(계획서 밖·선택): `/rewind`(diff before-image 재사용) ·
`!` 결과를 대화 맥락에 넣는 v2 · `@파일` 참조 자동완성 · `#` 메모리 단축.

### 4. ~~Devstral 코딩 서브모델 통합~~ — **완료 (2026-08-05, 커밋 8a32e19)**
config `coder_url`/`coder_model` + `routing.coder_enabled`(**기본 False**)/`coder_keywords`,
`RoutingDecision.use_coder`, `QueryEngine(coder_provider=)`가 코딩 턴에만 프로바이더 교체.
서버는 `--tool-call-parser mistral`(파서 없으면 `[TOOL_CALLS]`가 텍스트로 나옴),
`--tokenizer-mode mistral`은 금지(chat_template 400의 원인),
`LocalModelProvider(supports_chat_template_kwargs=False)`로 회피. 실서버 검증 완료.

**켜는 법**: `routing.coder_enabled: true`. 기본이 꺼진 이유는 실측상 코딩 모델이 항상
낫지 않기 때문(원샷 UI는 primary 우위, 코드 정리·리팩터링만 Devstral 우위).

**남은 실험(선택)**: 에이전트 루프 실전 비교 — A.X가 degeneration으로 붕괴했던
장컨텍스트 조건에서 Devstral이 완주하는지.

**원복 절차**: coder tmux 세션 kill → `start_all.sh`의 coder 줄 제거 →
`run_vllm_fp8.sh.bak-util092` 복원 → vllm 재기동(util 0.92).

### 4-b. 배포② web 저위험·고가치 — **2/3 완료**
- ✅ 1단계 프롬프트 주입 단일화 헬퍼(`core/system_prompt/compose.py`, 커밋 49b0663)
- ✅ 2단계 W1 응답 스타일 — 프리셋 YAML + 로더 + API + 설정 모달 UI
  (커밋 1d1ed62·a8a0208). 실측: concise가 기본 대비 42% 짧아짐.
- ⏸ 3단계 W5 KaTeX — 위 P0-2 참조(잠금)
- ❌ W4 Mermaid — **게이트 NO-GO**(방출률 20%). 재개하려면 base 프롬프트에
  다이어그램 출력 규약 1줄을 넣고 프로브를 재측정해야 한다.

### 4-c. 배포④ web 확장 — **미착수**
- **W8 업로드확장**(PDF·문서·스프레드시트 → DocumentProcess). 제품화(대학·기업)
  관점에서 가치 최상위. 미니스펙 선행 필요.
- **W3 메모리UI**(조건부) — 장기메모리가 실제 응답에 관여 중인지 실측 후. IDOR 격리 필수.

### 4-d. CLI 남은 후보 (계획서 밖·선택)
`/rewind`(diff before-image 재사용) · `/context` · `!` v2(결과를 대화 맥락에 주입) ·
`@파일` 참조 자동완성 · `#` 메모리 단축 · `/style`(웹과 같은 YAML을 읽어 말투 통일).

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
