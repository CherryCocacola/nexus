# NOVA 통합 기능 이식 로드맵 (CLI + web/app) — v2 최종 (Fable5 재설계 반영)

작성일 2026-08-04 · v2 = Fable5 우선순위 재설계 + 코드 실측 반영
근거 문서: `CLI_CLAUDE_CODEX_PORT_PLAN.md`(CLI 상세) · `FEATURE_RECATALOG_RECONCILE_2026-08.md`(전수 대조)
원칙: 무회귀·에어갭(벤더 로컬 커밋)·한글주석·fail-closed·4-Tier 불변. **표면별 분리 배포.**

---

## 0. Fable5 재설계로 바뀐 핵심 (코드 실측)
| 발견 | 근거 | 조치 |
|---|---|---|
| **W7 Stop 버튼 이미 존재** | `index.html:3226-3347`(AbortController) + `app.py:1679-1848`(producer CancelledError) | 계획에서 **제거**. 부분응답 저장 QA 1건만 |
| 내부 max-output 복구 2단계(에스컬레이션+이어쓰기 3회) | `query_loop.py:100-156,1234-1276` | W6 Continue는 **잔여 케이스 전용·백로그** |
| 시스템프롬프트 주입이 이미 3지점 문자열 append | `app.py:1400,1635,1997` | **주입 단일화 헬퍼가 W1의 선행 조건** |
| `core/system_prompt/`는 빈 패키지 | Glob 실측 | 헬퍼의 자연스러운 자리(CLI 출력스타일과 공유) |

---

## 1. 배포 단위 (표면별 분리 — 무회귀 최우선)

### 배포① — CLI Stage 1 (즉시 착수, 요청 #5 본질)
= `CLI_CLAUDE_CODEX_PORT_PLAN.md` Stage 1 그대로.
- **A1~A4** 권한모드(accept-edits 모드인지 ask_handler)·런타임전환(/mode·Shift+Tab)·ask_handler **v2 계약**(numbered 1/2/3+피드백)·세션 allow-list(Bash 프리픽스·DANGEROUS 제외).
- **B1~B2** 파일수정 diff 미리보기(256KB/200줄/바이너리 상한).
- **C1 + D8~D10** 도구표시 축약(⏺/⎿)·/verbose·`/`자동완성·멀티라인·Ctrl+R.
- **배포 게이트**: 공유 접점 `executor.py`(ask_handler_v2 폴백) → 구계약 폴백 회귀 테스트 + **웹 e2e 스모크 1회**.
- **[병행] W4/W5 방출 게이트 실측**(코드 무변경 측정 — §3).

### 배포② — web 저위험·고가치
1. **시스템프롬프트 주입 단일화 헬퍼**(신규·W1 선행 필수) — `core/system_prompt/compose_system_prompt()`. **append 금지, base에서 재조립(멱등)**. 순서: base → [응답스타일] → [사용자지시] → [프로젝트지시] → [세션지시]. 스타일 섹션은 base **뒤**(캐시 프리픽스 보존). 기존 3지점 치환 → 실서버 3시나리오 회귀.
2. **W1 응답스타일**(Normal/Concise/Explanatory/Formal+커스텀) — 헬퍼 위에서만. 프리셋은 `config/` YAML로 CLI와 공유.
3. **W4 Mermaid · W5 KaTeX** — **게이트 통과분만**. 프론트+벤더 로컬 커밋, app.py 무접촉.

### 배포③ — CLI 확장 + rewind
- CLI Stage 2(C2 상태줄·C3·C4, D1 `!`bash[감사로그+plan비활성], D2 /compact, D3 /resume[session_id 재바인드], D4 /diff, D5 /cost, D6 /copy, D7 /save).
- **/rewind** — Phase B diff의 **before-image를 승인 시점에 `.nexus/checkpoints/{session}/{turn}/`로 저장**해 스냅샷 소스 재사용. 3택 복원(코드+대화/대화만/코드만). 파일 복원은 CLI 전용.
- **/context**(+/cost 동승).

### 배포④ — web 확장
- **W8 업로드확장**(P3→승격·미니스펙 선행) — PDF/문서/스프레드시트 업로드 → DocumentProcess. 기존 청킹·artifact 저장 재사용. 제품화(대학·기업 세그먼트) 실질 가치 최상위.
- **W3 메모리UI**(조건부) — 장기메모리가 실제 응답에 관여 중인지 실측 후. IDOR 격리 필수.

### 백로그·제외
- W6 Continue(잔여 케이스 전용, length 소진 빈도 실측 후 — 서버 SSE 메타 `truncated:true` 1필드 + 프론트 "계속" 버튼=일반 메시지 경로)
- /branch(fork 프리미티브 `fork_session(at_turn)` web·CLI 공통화 **후** — 단독 구현 금지)
- **`ask` JSON 출력 모드**(AgentHub .NET 연동 실수요 — 백로그 상위)
- hooks 배선(별도 트랙)
- Codex 승인정책 3단계 네이밍(A2 모드명 설계 참고)
- **W2 Incognito 제외**(폐쇄망 B2B·국가PoC는 감사추적이 요구사항 — 기록 스킵은 컴플라이언스·fail-closed와 충돌)
- **W7 제거**(이미 보유)

---

## 2. 우선순위표 (Stage별·표면 혼합)
| Stage | 항목 | 표면 | 가치/비용/리스크 |
|---|---|---|---|
| 1 | A1~A4 권한·ask_handler v2·allow-list | CLI | 최고(요청#5)/중/저 |
| 1 | B1~B2 diff 미리보기 | CLI | 고/중/저 |
| 1 | C1·D8~D10 도구축약·자동완성·멀티라인 | CLI | 고(요청#4)/저/저 |
| 1병행 | W4/W5 방출 게이트 실측 | 측정 | — (무변경) |
| 2 | 주입 단일화 헬퍼 | core | 고(부채청산+공유)/저/중 |
| 2 | W1 응답스타일 | web | 고/저/저 |
| 2 | W4 mermaid·W5 KaTeX(게이트 통과분) | web | 중/저/저 |
| 3 | CLI Stage 2(상태줄·!·/compact·/resume·/diff·/cost·/copy·/save) | CLI | 중/중/저 |
| 3 | /rewind(before-image 재사용) | CLI | 고/중/중 |
| 3 | /context | CLI | 중/저/저 |
| 4 | W8 업로드확장(미니스펙 선행) | web | 고/중/중 |
| 4 | W3 메모리UI(조건부) | web | 중/중/중 |

## 3. W4/W5 방출 게이트 (Stage 1 병행 실측)
- ① 회고: 운영 transcript에서 `\\frac|\\sum|\\int|\$\$|\\begin\{`(LaTeX), ```` ```mermaid ````(diagram) 빈도 집계.
- ② 프로브: 실서버 A.X-4.0에 수식성 질의 20~30개 + 다이어그램 질의 → 방출률·구분자 형식·파싱 성공률 측정(mock 금지).
- **정량 게이트**: 방출률 ≥70% AND 파싱 성공률 ≥80% → GO. 방출은 하나 형식 제각각이면 base 프롬프트에 출력규약 1줄(헬퍼 이후) 후 재측정.
- 렌더러: 금액 `$100` 오인 방지, 스트리밍 미완성 수식은 완료 후 렌더.

## 4. 착수 순서 결론
**CLI 먼저 + W4/W5 게이트 실측만 병행.** 근거: ①사용자 요청 핵심 #5=CLI ②CLI 계획 v2 착수 준비도 최고(조건부GO 완료) ③web은 112 hot path라 준비(헬퍼·게이트) 선행 필요 ④게이트 실측은 무변경이라 병행 무충돌.

## 5. 종합
Fable5 **조건부 GO — 5조건 반영 완료**(W7제거·W6강등 / 헬퍼 승격·W1선행 / W4·W5 게이트 / 표면분리배포 / rewind는 before-image 재사용·branch 프리미티브 후). → **배포① CLI Stage 1 즉시 착수 가능.**
