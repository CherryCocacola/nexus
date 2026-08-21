# NOVA Code API — VSCode 플러그인 통합 명세 (v1, 2026-08-21)

> **읽는 대상: 코딩 에이전트.** 이 문서의 값은 전부 NOVA 리포 코드와 배포 설정에서
> 확인했거나 실서버로 측정한 것이다. 추측한 항목은 `[미검증]` 으로 표시했다.
> 규범 표현: **MUST**(반드시) / **MUST NOT**(절대 금지) / **SHOULD**(권장).
>
> 이전 문서 `VSCODE_PLUGIN_VIBE_JSON_HANDOFF.md`(2026-08-05)를 대체한다.
> 그 문서 이후 서버가 여러 번 바뀌었다 — 특히 **§7 코딩 모델 라우팅**은 신규다.

---

## 0. TL;DR — 최소 동작 요건

1. `POST {BASE}/v1/chat/completions` (OpenAI 호환). **MUST** `Authorization: Bearer <KEY>`.
2. 코드 수정 요청은 **MUST** `response_format`(json_schema)로 스키마를 강제한다.
3. **MUST NOT** `tools` 와 `response_format` 을 같은 요청에 함께 보낸다(§8).
4. 응답을 파싱하기 전에 **MUST** `finish_reason` 을 본다. `"stop"` 이 아니면 파싱하지 말고 §6 표대로 분기한다.
5. `changes[].find` 는 **파일에 실제로 존재하는 조각**이어야 한다. 적용 알고리즘은 §9.
6. 요청 문구에 코딩 신호가 있으면 서버가 **코딩 전용 모델로 자동 전환**한다(§7). 정확도 100% vs 75~83%, 속도 2초 vs 4~23초 — 문구를 맞추는 것이 이득이 크다.

---

## 1. 엔드포인트 · 인증

| 항목 | 값 |
|---|---|
| Base URL | `http://192.168.21.112:8600` (사내망) |
| 주 엔드포인트 | `POST /v1/chat/completions` |
| 인증 | `Authorization: Bearer <TENANT_API_KEY>` |
| 키 전달 | **이 문서에 키를 싣지 않는다.** 담당자에게 별도 수령 |
| 인증 실패 | HTTP **401**, body `{"detail":"인증 실패: 유효한 API 키가 필요합니다"}` |

**MUST NOT** 웹 UI 경로(`/v1/chat`, `/v1/chat/stream`)를 쓰지 않는다. 그 경로는
채널 기본값이 `web` 이라 플러그인 대화가 사내 웹 사용자 세션 목록에 섞인다.
`/v1/chat/completions` 는 채널이 자동으로 `api` 이고 `response_format` 도 쓸 수 있다.

### 1.1 부가 엔드포인트

| 메서드 | 경로 | 용도 | 인증 |
|---|---|---|---|
| GET | `/health` | 생존 확인. `{"status":"ok","gpu_server":"healthy"}` | 불필요 |
| GET | `/v1/models` | 실제 서빙 중인 모델 목록 | 필요 |
| GET | `/v1/tools` | 서버 도구 목록(참고용) | 필요 |

`/v1/models` 응답은 **OpenAI 표준 형태가 아니다.**

```json
{"models":[{"id":"ax-4.0","name":"skt/A.X-4.0","role":"primary"}, ...],"total":3}
```

`data` 배열이 아니라 `models` 다. 표준 SDK 의 `client.models.list()` 는 깨질 수 있다. `[미검증]`

---

## 2. 요청 스키마 (`OpenAIChatCompletionRequest`)

| 필드 | 타입 | 기본 | 비고 |
|---|---|---|---|
| `messages` | `array` | **필수** | OpenAI 형식. 없으면 **422** |
| `model` | `string` | `"primary"` | **응답에 반향될 뿐 모델 선택에 쓰이지 않는다.** 실제 모델은 서버 라우팅이 정한다(§7) |
| `stream` | `bool` | `false` | true 면 SSE(§5) |
| `max_tokens` | `int?` | `null` | **하드 상한이 아니다** — §4 |
| `temperature` | `float?` | `null` | 라우팅 프로필이 우선. 무시될 수 있다 |
| `top_p` | `float?` | `null` | 위와 같음 |
| `tenant_id` | `string?` | `null` | 보통 생략(키로 결정) |
| `query_class` | `string?` | `null` | `"TOOL"` \| `"KNOWLEDGE"` \| `"CHAT"` (§3) |
| `response_format` | `object?` | `null` | 구조화 출력(§8) |
| `tools` | `array?` | `null` | **클라이언트 실행 도구**(§10). `response_format` 과 병용 금지 |
| `tool_choice` | `any?` | `null` | OpenAI 규약 |

### 2.1 헤더

| 헤더 | 필수 | 효과 |
|---|---|---|
| `Authorization: Bearer <KEY>` | **MUST** | 인증 + 테넌트 결정 |
| `Content-Type: application/json` | **MUST** | |
| `X-Request-ID: <id>` | SHOULD | 서버 로그에 남아 장애 추적이 가능해진다. **넣지 않으면 문의 시 로그 추적이 사실상 불가능하다**(실제 사례 있음) |
| `X-Client-Id: <plugin-name>` | SHOULD | 세션 메타에 소비자 식별자를 남긴다 |
| `X-Nexus-Query-Class: TOOL` | 선택 | body `query_class` 와 동일 효과 |
| `X-Tenant-ID: <id>` | 선택 | 보통 불필요 |

잘못된 `X-Request-ID`/`query_class` 값은 **400**으로 즉시 거절된다(스트리밍 시작 전).

---

## 3. 질의 클래스 (`query_class`) — 코드 요청에는 반드시 지정

서버는 질의를 3종으로 분류하고 그에 따라 **샘플링·출력 한도·사내 문서 RAG 주입**을 바꾼다.

| 클래스 | max_tokens 상한 | 사내 지식 RAG | 용도 |
|---|---|---|---|
| `TOOL` | **8192** | 주입 안 함 | 코드·도구 작업 |
| `KNOWLEDGE` | **4096** | **주입함** | 사내 문서 질의 |
| `CHAT` | 4096 | 주입 안 함 | 짧은 잡담(30자 이하) |

**MUST** 코드 요청에 `"query_class": "TOOL"` 을 넣는다. 넣지 않으면 짧은 코드 질문이
KNOWLEDGE 로 분류되어 **사내 문서가 약 5,000자 주입**된다(실측). 코드 작업에 방해이고
출력 한도도 절반으로 줄어든다.

`TOOL` 프로필 실제 값: `temperature 0.3`, `top_p 0.95`, `repetition_penalty 1.05`,
`frequency_penalty 0.15`. **`repetition_penalty` 가 특히 중요하다** — 이 값이 1.0(비활성)
이면 A.X 가 JSON 생성 중 개행을 무한 반복해 응답을 통째로 날린다(§11.1).

---

## 4. 토큰 — 반드시 이해할 것

### 4.1 컨텍스트 예산

| 항목 | 값 |
|---|---|
| `max_context_tokens` | **61440** (입력+출력 합산 예산) |
| vLLM `max-model-len` | 65536 |
| 기본 출력 한도 | 8192 |

### 4.2 `max_tokens` 는 하드 상한이 아니다

요청의 `max_tokens` 는 **시작값**이다. 엔진이 잘림을 감지하면 자동으로 올려 재시도한다.

- 에스컬레이션 단계: **`[4096, 8192]`** (배포 설정값. 코드 기본은 `[4096, 8192, 16384]`)
- 그래도 못 끝내면 "이어서 작성" 멀티턴 복구를 **최대 3회** 수행

**실증(2026-08-21)**: `max_tokens: 50` 으로 요청했는데 `completion_tokens: 1591`,
`finish_reason: "stop"` 으로 왔다. 즉 요청값의 30배가 생성됐다.
**MUST NOT** `max_tokens` 로 비용·길이를 통제하려 하지 말 것 — 통하지 않는다.

결과적으로 **웬만한 잘림은 자동 복구되어 `finish_reason: "stop"` 으로 온다.**
`"length"` 가 왔다면 **복구를 다 쓰고도 못 끝낸 것**이다 — 그때는 `max_tokens` 를
키우는 것보다 **요청을 쪼개는 것**이 맞다(파일 전문 대신 find/replace 조각).

### 4.3 한국어 토큰 배율 (실측, 2026-08-20)

| 모델 | tok/char | 비고 |
|---|---|---|
| A.X-4.0 | **0.384** | 한국어 특화 토크나이저 |
| Qwen3-Coder 30B | **0.671** | 같은 한국어 문장에 **1.75배** |

같은 파일이라도 코딩 모델로 라우팅되면 입력 토큰이 더 든다. 한국어 주석이 많은 파일을
통째로 보낼 때 예산을 그만큼 더 잡아야 한다.

### 4.4 정확한 토큰 수를 알아야 한다면

추정하지 말 것. vLLM `/tokenize` 로 세면 오차 0이다. 서버 내부에서만 접근 가능하므로
플러그인은 **보수적으로 문자수 × 0.7** 정도로 잡고, 실제 값은 응답 `usage` 로 확인한다.

```json
"usage": {"prompt_tokens": 1499, "completion_tokens": 1500, "total_tokens": 2999}
```

---

## 5. 스트리밍 (SSE)

`stream: true` 면 `text/event-stream` 으로 `chat.completion.chunk` 프레임이 온다.

```
data: {"id":"chatcmpl-...","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"..."},"finish_reason":null}]}
data: {"id":"chatcmpl-...","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}
data: [DONE]
```

- **MUST** `data: [DONE]` 을 종료 신호로 쓴다.
- 프레임은 `data: {json}\n\n` 형식. 빈 줄은 하트비트일 수 있으니 무시한다.
- `delta.content` 에 **텍스트만** 실린다. 서버 도구 사용·thinking 은 스트림에 안 나온다.
- **구조화 출력(JSON)에는 스트리밍을 SHOULD NOT.** 비스트림에서만 서버가 JSON 사전 파싱
  검사를 하고 `finish_detail` 을 실어 준다(§6). 스트리밍에는 그 검사가 없다.

---

## 6. `finish_reason` · `finish_detail` — 파싱 전에 반드시 분기

```json
{"choices":[{"finish_reason":"content_filter",
  "finish_detail":{"code":"INVALID_STRUCTURED_OUTPUT","message":"..."}}]}
```

| `finish_reason` | `finish_detail.code` | 의미 | 플러그인 조치 |
|---|---|---|---|
| `stop` | (없음) | 정상 종료 | JSON 파싱 진행 |
| `length` | `RESPONSE_TRUNCATED` | 자동 복구를 다 쓰고도 잘림 | **파싱 금지.** 요청을 쪼개 재시도 |
| `content_filter` | `INVALID_STRUCTURED_OUTPUT` | JSON 이 유효하지 않음 | **파싱 금지.** 1회 재시도, 계속되면 범위 축소 |
| `tool_calls` | (없음) | 클라이언트 도구 호출 요청 | §10 대로 실행 후 결과를 다음 요청에 |

**`content_filter` 는 콘텐츠 정책 차단이 아니다.** 이 서버에는 콘텐츠 필터가 없다.
OpenAI 규격 안에 머물기 위해 값을 재활용한 것뿐이고, 진짜 이유는 `finish_detail.code` 에 있다.
**MUST NOT** 사용자에게 "정책에 차단됐다"고 표시한다.

---

## 7. ★코딩 모델 자동 라우팅 (2026-08-21 서버 활성) — 신규

서버는 요청 문구를 보고 **코딩 전용 모델(Qwen3-Coder 30B-A3B)** 로 자동 전환한다.
플러그인이 보내는 `model` 값과 무관하다.

### 7.1 왜 중요한가 (실측, 같은 경로·같은 프롬프트·같은 스키마)

| 처리 모델 | JSON 파싱 | find/replace 적용 가능 | 속도 |
|---|---|---|---|
| 앵커(A.X-4.0) | 11~12/12 | **75~83%** | 4~23초 |
| **코딩 모델** | **12/12** | **100%** | **약 2초** |

전환되면 정확도와 속도가 모두 좋아진다. **플러그인은 전환이 걸리도록 문구를 구성하는 것이 이득이다.**

### 7.2 전환 트리거 (배포 설정 실제값)

**키워드**(부분 문자열 매칭) — 하나라도 포함되면 전환.

```
리팩터링, 리팩토링, refactor, 디버깅, debug, 스택트레이스, traceback,
컴파일, compile, 함수를 고쳐, 코드를 고쳐, 버그를 고쳐, 코드 리뷰, code review
```

**정규식** — 키워드에 안 걸릴 때 평가. (2026-08-21 보강: 3개 → 5개.
이전 패턴은 흔한 코드 수정 표현 20종 중 6종만 잡았다. 보강 후 19/20 포착·오탐 0/15.)

```regex
(구현|개발|리팩터\S*|리팩토\S*|디버깅|디버그)\s*(해|하|중|좀)
(함수|메서드|메소드|클래스|모듈|스크립트|컴포넌트|엔드포인트|API|UI|화면|페이지|기능|창|변수|변수명|타입|타입 힌트|인터페이스|상수|프로퍼티|파라미터|인자|주석|docstring|로그|예외|에러|import|반환값|시그니처|필드|라우터|헬퍼)[^\n]{0,15}?(만들|작성|생성|추가|수정|짜|붙여|바꿔|고쳐|넣어|달아|채워|분리|추출|정리|개선|최적화|지워|제거)
코드\s*\S{0,6}\s*(만들|작성|생성|추가|수정|짜|붙여|바꿔|고쳐|넣어|달아|채워|분리|추출|정리|개선|최적화|지워|제거)
\.(py|ts|tsx|js|jsx|java|go|rs|cpp|cc|c|h|hpp|cs|rb|php|kt|swift|sql|sh|ps1|yaml|yml|json|toml|xml|html|css|scss|vue|svelte)\b[^\n]{0,40}?(만들|작성|생성|추가|수정|짜|붙여|바꿔|고쳐|넣어|달아|채워|분리|추출|정리|개선|최적화|지워|제거)
(이 부분|여기|이곳)[^\n]{0,15}?(고쳐|수정|개선|최적화|바꿔)
```

**제외 패턴** — 아래에 걸리면 **앵커로 되돌린다**(키워드 매칭이 먼저면 그대로 전환).

```regex
테스트\s*코드      단위\s*테스트      unit\s*test      테스트(를|을)?\s*(짜|작성)
```

### 7.3 실무 지침

- 프롬프트 앞부분에 `"코드 수정안을 ... 수정"` 처럼 **`코드` + `수정/작성/생성`** 이 들어가면 전환된다(정규식 3번).
- 2026-08-21 보강으로 다음이 새로 걸린다 — "타입 힌트를 붙여줘", "변수명을 바꿔줘",
  "주석을 달아줘", "import 정리해줘", "app.py 의 라우터를 개선해줘", "이 부분 개선해줘".
- **파일 경로가 들어가면 대체로 걸린다**(정규식 4번이 코드 확장자를 본다).
  플러그인 요청에는 거의 항상 경로가 있으므로 유리하다.
- 여전히 안 걸리는 예: "null 체크를 보강해줘"('체크'·'보강'은 오탐 위험이 커 뺐다).
- 순수 테스트 작성 요청은 **의도적으로** 앵커가 처리한다(과거 측정에서 코딩 모델이 그 범주만 열세였다. 현 모델 기준 재측정은 미완).
- 전환 여부는 응답으로 직접 알 수 없다. `[미검증]` 필요하면 서버 로그(`라우팅: 코딩 전용 모델로 전환`)로만 확인 가능하다.

---

## 8. 구조화 출력 (`response_format`)

vLLM guided decoding 으로 **JSON 문법을 강제**한다. 서버 설정: `enabled: true`,
`strict: true`(기본), `max_schema_bytes: 65536`.

```json
{
  "model": "ax-4.0",
  "query_class": "TOOL",
  "max_tokens": 4000,
  "temperature": 0.2,
  "messages": [{"role": "user", "content": "<규칙 + 수정요청 + 파일내용>"}],
  "response_format": {
    "type": "json_schema",
    "json_schema": {
      "name": "vibe_change_plan",
      "strict": true,
      "schema": {
        "type": "object",
        "properties": {
          "summary": {"type": "string"},
          "plan": {"type": "array", "items": {"type": "string"}},
          "changes": {"type": "array", "items": {"type": "object",
            "properties": {
              "path": {"type": "string"},
              "operation": {"type": "string", "enum": ["create", "modify", "delete"]},
              "description": {"type": "string"},
              "find": {"type": "string"},
              "replace": {"type": "string"}
            },
            "required": ["path", "operation", "description", "find", "replace"]}},
          "verificationCommands": {"type": "array", "items": {"type": "string"}}
        },
        "required": ["summary", "plan", "changes", "verificationCommands"]
      }
    }
  }
}
```

### 8.1 제약과 함정

- **MUST NOT** `tools` 와 함께 보낸다. 서버가 `tools` 를 통째로 비운다(vLLM 상호배타).
  게다가 시스템 프롬프트에는 "도구 안내문"이 남아 프롬프트와 실제 도구 풀이 어긋난다(기존 결함, 미수정).
- 스키마 직렬화 **65536 바이트** 초과 → 400.
- 스키마 누락·미지원 타입 → 400.
- **guided decoding 은 문법만 보장한다.** 문자열 *내용*의 정확성(원본 조각 복사)은 보장하지 않는다 — §9 가 필요한 이유.
- **사실 값은 모델에 맡기지 말 것.** 날짜·작성자 등은 프롬프트에 명시한다(모델이 `2023-10-05` 를 환각한 실측 있음).
- **서버는 사후 검증·자동 재시도를 하지 않는다.** 비스트림에 한해 `json.loads` 를 한 번 해 보고 실패를 `finish_detail` 로 알릴 뿐이다.

---

## 9. find/replace — 적용 알고리즘과 실패 대응

### 9.1 프롬프트 규칙 (이 문구로 측정했다)

```
- changes[].find 는 파일에 "실제로 존재하는" 원본 텍스트 조각(3~8줄, 파일 내 유일)을
  글자 하나 바꾸지 말고 그대로 복사한다. 들여쓰기·공백·한글도 원문 그대로.
- changes[].replace 는 find 를 대체할 새 텍스트. 수정은 최소화한다.
- 파일 전문을 넣지 않는다. 필요한 조각만 넣는다.
- path 는 주어진 파일 경로를 그대로 쓴다.
- find 는 항상 **줄 전체**를 포함한다. 줄 중간에서 끊지 마라 — 첫 줄과 마지막 줄
  모두 개행 경계에서 시작하고 끝나야 한다.
```

### 9.2 적용 알고리즘 (TypeScript, 그대로 사용 가능)

정확 매칭(유일) → 실패 시 공백 정규화 폴백. **정규화 매치가 정확히 1곳일 때만** 적용한다
(모호하면 실패 = fail-closed).

```typescript
function normLine(line: string): string {
  return line.split(/\s+/).filter(Boolean).join(" ");
}

function findWhitespaceFuzzySpan(content: string, find: string): [number, number] | null {
  const oldLines = find.split(/\r?\n/).map(normLine);
  if (oldLines.length === 0 || oldLines.every((l) => l === "")) return null;
  const lines = content.split("\n");
  const offsets: number[] = [];
  let pos = 0;
  for (const line of lines) { offsets.push(pos); pos += line.length + 1; }
  const normLines = lines.map(normLine);
  const w = oldLines.length;
  const matches: number[] = [];
  for (let i = 0; i <= lines.length - w; i++) {
    let ok = true;
    for (let j = 0; j < w; j++) if (normLines[i + j] !== oldLines[j]) { ok = false; break; }
    if (ok) { matches.push(i); if (matches.length > 1) return null; }
  }
  if (matches.length !== 1) return null;
  const s = matches[0], e = s + w - 1;
  return [offsets[s], offsets[e] + lines[e].length];
}

function applyChange(content: string, find: string, replace: string): string | null {
  const first = content.indexOf(find);
  if (first !== -1 && content.indexOf(find, first + 1) === -1) {
    return content.slice(0, first) + replace + content.slice(first + find.length);
  }
  const span = findWhitespaceFuzzySpan(content, find);
  if (!span) return null;
  return content.slice(0, span[0]) + replace + content.slice(span[1]);
}
```

### 9.3 ★알려진 폴백 사각지대 (실측)

앵커 모델의 대표 실패 유형은 **`find` 가 줄 중간에서 잘리는 것**이다.

```
모델이 낸 find 의 마지막 줄:  "        text"
파일의 실제 줄:              "        text = user_input or \"\""
```

정확 매칭이 실패하고, **§9.2 폴백도 구제하지 못한다** — 줄 단위 비교라 마지막 줄이
다르면 매치가 아니기 때문이다.

**해결 순서 — 위에서부터 시도할 것.**

**① 코딩 모델로 라우팅되게 한다 (가장 효과적).**
이 실패 유형은 앵커(A.X)의 긴 리터럴 재현 약점에서 나온다. 코딩 모델로 가면
실측 12/12 정확 매칭이었다. §7.3 대로 요청 문구를 맞추는 것이 근본 대책이다.

**② 프롬프트 규칙에 "줄 전체" 제약을 명시한다.**
§9.1 규칙에 다음 한 줄을 더한다.

```
- find 는 항상 **줄 전체**를 포함한다. 줄 중간에서 끊지 마라.
  첫 줄과 마지막 줄 모두 개행 경계에서 시작하고 끝나야 한다.
```

증상이 "줄 중간 잘림"이므로 이것이 가장 직접적인 예방이다. `[미검증]` — 이 문구의
효과는 우리가 측정하지 못했다. 플러그인에서 A/B 로 확인해 볼 만하다.

**③ 실패하면 사용자에게 diff 확인을 받는다 (fail-closed 유지).**
자동으로 범위를 넓혀 적용하지 **않는다**. §9.4 참조.

**접두 일치(prefix match) 확장에 대하여 — 권하지 않는다.**
"마지막 줄만 접두 일치를 허용하면 구제된다"는 아이디어가 나올 수 있다. 우리도 시도했고
다음 이유로 제안하지 않기로 했다.

- **잘린 조각을 추측으로 확장하는 휴리스틱**이다. 매치가 유일해도 모델이 의도한 범위와
  다를 수 있고, 그 차이는 코드에 조용히 반영된다.
- 실제 구제 빈도를 **입증할 데이터가 없다**. 우리가 확보한 실패 샘플은 이후 파일이 바뀌어
  재현되지 않았고, 합성 케이스로는 알고리즘 동작만 볼 수 있을 뿐 빈도를 알 수 없다.
- ①을 적용하면 이 유형 자체가 크게 줄어든다. 위험한 자동 확장을 얹기 전에 원인을 없애는 편이 낫다.

굳이 넣는다면 **최소 안전장치**는 이렇다 — 마지막 줄에만 적용, 접두 길이 4자 이상,
매치가 유일할 때만, 그리고 **구제된 변경은 반드시 사용자 확인을 거칠 것**(자동 적용 금지).

### 9.4 실패 시 처리 순서

1. `applyChange` 가 `null` → **사용자에게 diff 후보를 보여주고 수동 확인**을 요청한다.
   자동으로 범위를 넓혀 적용하지 않는다(fail-closed).
2. 자동 재시도는 **1회까지**. 재시도 시 프롬프트에 실패한 `find` 를 그대로 인용하고
   "이 조각은 파일에 없다. **줄 전체**를 정확히 다시 복사하라"를 덧붙인다.
3. 같은 파일에서 2회 실패하면 **요청 범위를 줄인다**(대상 함수 하나로).
4. 3회 이상 반복되면 그 요청은 앵커로 갔을 가능성이 높다 — §7.3 대로 문구를 조정한다.

## 10. 클라이언트 실행 도구 (`tools`) — 에이전트 루프를 원한다면

`tools` 를 보내면 서버는 그 스키마를 모델에게 보여 주고 **`tool_calls` 만 돌려준다.
서버는 그 도구를 실행하지 않는다.** 루프의 주인은 플러그인이다.

```
요청 1: tools=[readFile, writeFile, runTests], messages=[user]
응답 1: finish_reason="tool_calls", tool_calls=[{function:{name:"readFile", arguments:"{\"path\":\"...\"}"}}]
        ↓ 플러그인이 자기 PC에서 실행
요청 2: messages=[user, assistant(tool_calls), tool(결과)]
응답 2: ...
```

- 이 필드가 오면 이번 요청의 도구 목록은 **클라이언트 도구로 전부 교체**된다.
  서버 도구는 하나도 노출되지 않는다(부분 혼합 없음 — 보안상 의도된 설계).
- **MUST NOT** `response_format` 과 병용(§8).
- 서버가 도구를 실행하지 않는 이유는 두 가지다. ① 보안 — 서버 도구는 서버 파일시스템을
  만진다. ② 쓸모 — 개발자의 코드는 개발자 PC 에 있고 서버의 Read 는 `/app` 을 읽는다.

**어느 방식을 쓸 것인가**

| 방식 | 장점 | 단점 |
|---|---|---|
| `response_format` + find/replace | 1회 왕복, 단순, 검증된 경로 | 큰 변경·탐색 필요 작업에 약함 |
| `tools` 클라이언트 루프 | 파일 탐색·테스트 실행까지 자율 | 왕복 다수, 구현 복잡, guided decoding 불가 |

권장: **기본은 find/replace**, 다중 파일 탐색이 필요한 작업만 `tools` 루프. `[미검증]`
— 플러그인에서 `tools` 루프의 실전 정확도는 아직 측정하지 않았다.

---

## 11. 알려진 실패 모드와 대응

### 11.1 degeneration — 개행 무한 반복

**증상**: `finish_reason: "length"`, `completion_tokens` 가 상한을 정확히 소진,
응답 끝이 개행 수백 개.

**원인**: 반복 붕괴. JSON 문법상 토큰 사이 공백은 무제한 허용이라 **guided decoding 이 막지 못한다.**

**실측**: 샘플링 방어가 없는 조건(rep_penalty 1.0)에서 A.X 는 **12/12 전부 실패**했다.
서버의 `TOOL` 프로필(rep 1.05 · freq 0.15)을 타면 같은 과제가 **83%** 로 올라간다.

**대응**: `query_class: "TOOL"` **MUST**(그래야 방어 프로필이 걸린다). 그래도 나오면
요청 범위를 줄인다. `max_tokens` 를 키우는 것은 도움이 안 된다 — 개행만 더 나온다.

### 11.2 경로 오타 환각

모델이 `nexus_config.112.yaml` 을 `nexus_config.1112.yaml` 로 바꿔 부르는 실측 사례가 있다.
**MUST** `changes[].path` 를 요청에 보낸 경로와 대조하고, 다르면 적용하지 않는다.

### 11.3 응답에 마크다운 코드펜스가 섞이는 경우

`response_format` 사용 시에는 관측되지 않았지만, 방어적으로 파싱 전
```` ```json ... ``` ```` 펜스를 제거하는 전처리를 SHOULD 둔다.

---

## 12. 서버 측 제약 (플러그인이 알아야 할 것)

| 항목 | 값 | 비고 |
|---|---|---|
| 업로드 최대 크기 | 20,971,520 B (20 MiB) | 초과 시 413 |
| 업로드 보존 | 72시간 | 이후 자동 삭제 |
| 스키마 최대 | 65,536 B | 초과 시 400 |
| 문서 단일 반환 상한 | 40,000자 | 초과 시 26,000자 청크 분할 |
| 세션 격리 | 테넌트별 | 다른 테넌트 세션 접근은 **404**(2026-08-20 적용) |
| 서버 파일 접근 | 세션 샌드박스 + uploads/exports 로 제한 | 그 밖은 차단(2026-08-20 적용) |

**세션 조회 API**(`/v1/sessions*`)는 자기 테넌트 것만 보인다. 다른 테넌트 세션 id 로
조회하면 403 이 아니라 **404** 다(존재 은닉).

---

## 13. 실전 예제

### 13.1 최소 요청 (curl)

```bash
curl -sS http://192.168.21.112:8600/v1/chat/completions \
  -H "Authorization: Bearer $NOVA_API_KEY" \
  -H "Content-Type: application/json" \
  -H "X-Request-ID: plugin-$(date +%s)" \
  -H "X-Client-Id: vscode-nova" \
  -d '{
    "model": "ax-4.0",
    "query_class": "TOOL",
    "max_tokens": 4000,
    "temperature": 0.2,
    "messages": [{"role":"user","content":"...규칙 + 수정요청 + 파일내용..."}],
    "response_format": {"type":"json_schema","json_schema":{"name":"vibe_change_plan","strict":true,"schema":{...}}}
  }'
```

### 13.2 TypeScript 통합 (권장 흐름)

```typescript
type FinishDetail = { code: string; message: string };

async function requestChangePlan(prompt: string): Promise<ChangePlan> {
  const res = await fetch(`${BASE}/v1/chat/completions`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${API_KEY}`,
      "X-Request-ID": `vscode-${Date.now()}`,
      "X-Client-Id": "vscode-nova",
    },
    body: JSON.stringify({
      model: "ax-4.0",
      query_class: "TOOL",              // MUST — 없으면 사내 문서가 주입된다
      max_tokens: 4000,
      temperature: 0.2,
      messages: [{ role: "user", content: prompt }],
      response_format: { type: "json_schema", json_schema: VIBE_SCHEMA },
      // tools 는 절대 함께 보내지 않는다
    }),
  });

  if (res.status === 401) throw new Error("NOVA 인증 실패 — API 키 확인");
  if (!res.ok) throw new Error(`NOVA HTTP ${res.status}: ${await res.text()}`);

  const body = await res.json();
  const choice = body.choices[0];
  const detail: FinishDetail | undefined = choice.finish_detail;

  // ★파싱 전에 반드시 분기
  if (choice.finish_reason !== "stop") {
    if (detail?.code === "RESPONSE_TRUNCATED") {
      throw new RetryableError("응답이 잘렸습니다. 대상 범위를 줄여 다시 시도하세요.");
    }
    if (detail?.code === "INVALID_STRUCTURED_OUTPUT") {
      // 콘텐츠 차단이 아니다 — 사용자에게 그렇게 표시하지 말 것
      throw new RetryableError("응답 JSON 이 완성되지 않았습니다. 다시 시도합니다.");
    }
    throw new RetryableError(`예상치 못한 종료: ${choice.finish_reason}`);
  }

  const plan = JSON.parse(stripFences(choice.message.content));
  for (const ch of plan.changes) {
    if (!isRequestedPath(ch.path)) throw new Error(`경로 불일치: ${ch.path}`);  // §11.2
  }
  return plan;
}
```

### 13.3 프롬프트 구성 (전환까지 노리는 형태)

```
너는 코드 수정안을 JSON 으로 내는 도구다. 규칙을 반드시 지켜라.
- changes[].find 는 파일에 실제로 존재하는 원본 텍스트 조각(3~8줄, 파일 내 유일)을
  글자 하나 바꾸지 말고 그대로 복사한다. 들여쓰기·공백·한글도 원문 그대로.
- changes[].replace 는 find 를 대체할 새 텍스트. 수정은 최소화한다.
- 파일 전문을 넣지 않는다. 필요한 조각만 넣는다.
- path 는 주어진 파일 경로를 그대로 쓴다.

[수정 요청]
<사용자 요청 — 가능하면 "…를 수정해줘 / 구현해줘" 형태로. §7.3>

[파일 경로]
<상대경로>

[파일 내용]
<파일 전문 또는 관련 구간>
```

첫 줄의 `"코드 수정안"` 이 정규식 3번(`코드\s*\S{0,6}\s*(수정)`)에 걸려 **코딩 모델로
전환된다.** 이 문구를 유지하는 것을 SHOULD.

---

## 14. 통합 체크리스트

- [ ] `Authorization` 헤더로 인증하고 401 을 사용자에게 명확히 표시한다
- [ ] `/v1/chat/completions` 를 쓴다(`/v1/chat` 아님)
- [ ] 모든 코드 요청에 `query_class: "TOOL"` 을 넣는다
- [ ] `X-Request-ID`, `X-Client-Id` 를 보낸다
- [ ] `response_format` 과 `tools` 를 함께 보내지 않는다
- [ ] 파싱 전에 `finish_reason` 을 확인하고 `finish_detail.code` 로 분기한다
- [ ] `content_filter` 를 "정책 차단"으로 표시하지 않는다
- [ ] `changes[].path` 를 요청 경로와 대조한다
- [ ] `applyChange` 정확→폴백 순서를 지키고, 실패 시 diff 확인을 요청한다
- [ ] 자동 재시도는 1회로 제한한다
- [ ] 프롬프트 첫 줄에 코딩 전환 문구를 유지한다

---

## 15. 변경 이력 · 검증 근거

| 날짜 | 내용 |
|---|---|
| 2026-08-05 | guided decoding + find/replace 확정, 실서버 e2e |
| 2026-08-13 | `finish_reason` 정직화, `finish_detail` 신설, `max_tokens` 엔진 전달 |
| 2026-08-16 | `query_class` 요청 단위 고정(body·헤더) |
| 2026-08-20 | 테넌트 세션 격리, 서버 파일 접근 제한 |
| **2026-08-21** | **코딩 모델 라우팅 서버 활성** — find/replace 정확도 75~83% → **100%**, 4~23초 → 2초 |

측정 조건: 실제 소스 4종(4.5KB~22.5KB, 한글 주석·깊은 들여쓰기 포함) × 3회,
같은 엔드포인트·같은 프롬프트·같은 스키마. 판정은 §9.2 알고리즘 그대로.

**문의 시 `X-Request-ID` 를 함께 알려주면 서버 로그에서 그 요청을 특정할 수 있다.**

---

## 16. 이 문서의 검증 상태

§15 의 측정 외에, 문서가 주장하는 **API 동작 12건을 실서버로 직접 대조**했다
(2026-08-21, `192.168.21.112:8600`). 전부 일치.

| 대조 항목 | 결과 |
|---|---|
| 무인증 401 | 일치 |
| `/v1/models` 가 `models` 키(=`data` 아님) | 일치 |
| `messages` 누락 → 422 | 일치 |
| 잘못된 `query_class` → 400 (`허용값: KNOWLEDGE, TOOL, CHAT`) | 일치 |
| `query_class` body/헤더 수용 | 일치 |
| **`max_tokens: 50` → `completion_tokens: 1591`, finish=stop** | 일치 |
| SSE `chat.completion.chunk` + `[DONE]` | 일치 |
| `tools`+`response_format` 병용 시 `tool_calls` 없음 | 일치 |
| `tools` 단독 → `tool_calls`, `finish_reason: "tool_calls"` | 일치 |
| 스키마 98,724B → 400 (`상한(65536B) 초과`) | 일치 |

`[미검증]` 으로 표시한 항목은 위 대조에 포함되지 않았다. 그 항목을 근거로 구현하기 전에
직접 확인할 것.
