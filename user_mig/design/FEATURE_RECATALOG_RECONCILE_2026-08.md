# NOVA 기능 재조사 — Claude Code / Codex / claude.ai 전수 카탈로그 + NOVA 대조

작성일 2026-08-04 · **방법론 교정**: 기억 기반 사전필터링 → 1차 출처(공식 docs) 전수조사 후 판단
조사 출처: code.claude.com/docs(Claude Code) · learn.chatgpt.com/docs·github.com/openai/codex(Codex) · support.claude.com·anthropic.com(claude.ai web/app)

> 배경 앞선 CLI 25개 리스트·web/app Phase1~3 로드맵이 모두 "기억으로 떠올려 미리 필터링"한 결과라 실제 기능의 일부만 잡았음. 사용자 지적으로 3개 표면을 1차 출처로 전수 재조사함. 원본 raw 카탈로그는 세션 트랜스크립트/에이전트 메모리에 보존. 본 문서는 그 위의 **NOVA 대조·판정** 계층.

**범례** ✅NOVA 보유 · 🆕신규 후보(미보유·이식가치) · 🔶조건부 · ❌제외(에어갭/모델/범위)

---

## A. claude.ai web/app 대조 (★기존 로드맵이 놓친 부분 — 사용자 우려 지점)

### A.1 이미 보유 (Phase 1/2/3 + 기존 구축)
| 기능 | NOVA |
|---|---|
| 대화 이름변경·핀·폴더 | ✅ 메타 UI |
| 코드블록 문법강조+복사 | ✅ highlight.js |
| 대화 검색 | ✅ transcript 검색 |
| 대화 내보내기 | ✅ md/txt export |
| 응답 재생성 | ✅ regenerate |
| 커스텀 인스트럭션 | ✅ 테넌트/프로젝트별 |
| 프롬프트 템플릿 | ✅ `/` 팝업 |
| HTML 프리뷰(아티팩트) | ✅ 샌드박스 iframe |
| Projects(지식·지시·소스) | ✅ Phase3 |
| 메시지 편집·분기 | ✅ fork |
| 파일 생성(docx/pptx/hwpx/xlsx/pdf) | ✅ DocumentExport |
| 이미지 생성 | ✅ FLUX/SD3.5 |
| 이미지 분석(비전) | ✅ AnalyzeImage |
| 아티팩트 저장 | ✅ tb_artifacts |
| 파일 업로드(이미지) | ✅ |

### A.2 🆕 신규 후보 — 기존 로드맵이 놓친 것
| 기능 | 설명 | 이식가치 | 비고 |
|---|---|---|---|
| **응답 스타일 프리셋** | Normal/Concise/Explanatory/Formal + 커스텀 스타일 | 高 | 시스템프롬프트 프리픽스로 구현 용이. UI 토글 |
| **Incognito 채팅** | 검색·메모리·히스토리에서 제외되는 비공개 대화 | 中 | 폐쇄망에서도 민감 대화 격리 유용 |
| **메모리(대화 간) UI 노출** | 기억 보기·편집·끄기 | 中 | NOVA 백엔드 장기메모리 있음 → UI 노출만 |
| **Mermaid 다이어그램 렌더** | 채팅/아티팩트 내 mermaid 차트 | 中 | 아티팩트가 이미 mermaid 네이티브 지원 검토 |
| **LaTeX/수식 렌더** | KaTeX 수식 표시 | 中 | backlog. A.X-4.0 LaTeX 방출 여부 선확인 |
| **"계속하기(Continue)" 버튼** | 출력 한도 시 이어쓰기 | 中 | 내부 max-output recovery 있음 → 사용자 버튼 노출 |
| **응답 중단(Stop) 버튼** | 스트리밍 중단 | 中 | 웹 UI 존재 여부 확인 필요(CLI는 Ctrl+C) |
| **파일 업로드 확장** | PDF/문서/스프레드시트 업로드 분석 | 中 | 문서분석 도구 있음 → 업로드 경로 정합 확인 |

### A.3 ❌ 제외 (에어갭/모델/범위)
웹검색 · Extended Thinking(A.X-4.0 비추론) · 음성모드/받아쓰기/카메라(모바일) · 컴퓨터 사용 · Office 애드인 · Claude in Chrome · Cowork/클라우드 · Interactive Apps · 아티팩트 퍼블릭 게시/갤러리/리믹스(폐쇄망 공유 제한) · Team/Enterprise/SSO 협업

### A.4 재검증 필요(2026-02 이후, 조사 확신도 낮음)
Adaptive Reasoning · Live Artifacts+MCP · 모델 중간전환 · Skills 마이그레이션 — 이식 판단 전 사실 재확인.

---

## B. Claude Code CLI 대조 (계획 v2 = CLI_CLAUDE_CODEX_PORT_PLAN.md)

Claude Code 실측: 슬래시 40+ · 단축키 70+ · 플래그 60+. 앞선 계획 v2에 반영된 항목 외 **전수조사로 추가 확인된 것**:

### B.1 🆕 계획 v2에 추가 검토할 항목
| 기능 | 설명 | 판정 |
|---|---|---|
| **`/rewind`·체크포인트** | 프롬프트마다 파일 스냅샷, 코드+대화 복원 | 🆕 高 — 로컬 파일 스냅샷, 에어갭 안전. 강력 |
| **`/context` 시각화** | 토큰 사용량 그리드 | 🆕 中 — `/cost` 확장 |
| **`/branch`·세션 fork** | 현 지점 분기 | 🆕 中 — web fork의 CLI판 |
| **`/rename`·`/status`·`/export`·`/copy`** | 세션 관리 | ✅ 계획 D에 포함/근접 |
| **출력 스타일(output styles)** | Default/Explanatory/Learning 등 | 🔶 A.2 응답스타일과 통합 |
| **훅(hooks) 시스템** | Pre/PostToolUse 등 라이프사이클 | 🔶 NOVA hook_manager 존재하나 미배선 → 별도 트랙 |
| **커스텀 슬래시 명령/스킬** | 사용자 정의 명령 | 🔶 후순위 |
| **비대화형 확장**(`--output-format json`, `--json-schema`) | 구조화 출력 | 🔶 `ask`에 json 모드 |
| **키바인딩 커스터마이즈** | keybindings.json | ❌ 과함 |
| **Vim 모드** | | ❌ 저가치(기존 판정 유지) |

### B.2 이미 계획 v2 반영
권한모드 런타임전환(Shift+Tab)·accept-edits·numbered 프롬프트+피드백·diff 미리보기·항상허용·`!`bash·`/compact`·`/resume`·`/diff`·`/cost`·`/copy`·`/save`·도구표시 축약·멀티라인·Ctrl+R·자동완성.

### B.3 ❌ 제외 (기존 판정 유지·확정)
`/model`(라우팅 자동) · `/init` · `$비용` · MCP UI(웹 소관) · 웹검색 · 클라우드/원격/teleport · plugins 마켓 · 아티팩트 퍼블리시 · IDE 통합.

---

## C. Codex CLI 대조 (참고 — Claude Code와 겹치는 것 위주)

Codex 실측: config.toml 150+키 · 승인모드 다층 · 훅 · 서브에이전트. Claude Code에 없고 Codex만의 것 중 참고 가치:

| 기능 | 설명 | 판정 |
|---|---|---|
| **샌드박스 모드**(read-only/workspace-write/danger-full) | 파일시스템·네트워크 접근 범위 | 🔶 NOVA 권한 파이프라인(Layer2 path_guard)과 개념 중복 — 모드명 참고 |
| **`--ask-for-approval`(untrusted/on-request/never)** | 승인 정책 3단계 | 🔶 NOVA 권한모드와 매핑 — 네이밍 참고 |
| **config 프로필(`--profile`)** | 다중 설정 세트 전환 | 🔶 NOVA YAML 3본(pc/112) 있음 → 프로필 전환 후순위 |
| **`codex exec resume`** | 비대화형 세션 재개 | 🔶 `ask` 확장 |
| **AGENTS.md 계층 탐색** | 프로젝트 지시 계층 병합 | ❌ NOVA는 CLAUDE.md/테넌트 지시로 충족 |
| **`--output-schema`** | 구조화 출력 | 🔶 B.1과 동일 |

**핵심 관찰** Codex의 승인/샌드박스 다층 구조는 NOVA가 요청 #5로 만들 권한모드 UX의 **레퍼런스**로 유용(특히 승인정책 3단계 네이밍·granular 정책). 신규 이식이라기보다 설계 참고.

---

## D. 종합 — 무엇이 바뀌었나
1. **CLI**: 계획 v2는 유효하되, `/rewind`(체크포인트)·`/context`·`/branch`를 **Stage 1.5 후보로 추가 검토** 권장(로컬·에어갭 안전, 고가치).
2. **web/app**: 기존 Phase1~3가 놓친 **8개 신규 후보**(응답스타일·incognito·메모리UI·mermaid·LaTeX·continue·stop·업로드확장) 확인 — 별도 web 로드맵 Phase4로.
3. **방법론**: 이후 모든 기능 이식은 본 문서 같은 1차출처 전수 대조를 선행. [[feedback_exhaustive_feature_research]]

## E. 다음 결정 (사용자)
- CLI 계획 v2 + B.1 추가분으로 갈지, 원안대로 갈지
- web/app 신규 8후보를 Phase4로 별도 진행할지
- 착수 순서(CLI 먼저 vs web/app 먼저 vs 병행)
