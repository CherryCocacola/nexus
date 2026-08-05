# VSCode 플러그인 바이브코딩 JSON — NOVA 연동 확정안 (2026-08-05)

플러그인 세션 전달용. "AI 응답에서 바이브코딩 변경안 JSON을 해석할 수 없습니다" 문제의
원인 진단과, **실서버(112:8600 + nexus-coding-key-001)로 e2e 검증 완료된** 해법.

## 0. 2026-08-05 추가 — 서버 수정 배포됨 (플러그인 대응 필요 2건)

실측으로 두 가지가 더 드러나 **112 운영 서버에 수정 배포 완료**했다.

### (1) 플러그인 대화가 웹 히스토리에 섞이던 문제 → 헤더 한 줄로 해결

원인: 플러그인이 OpenAI 호환(`/v1/chat/completions`)이 아니라 **웹 UI용
`/v1/chat`·`/v1/chat/stream`** 을 호출하고 있었다. 이 두 경로는 `channel="web"`
하드코딩이라 플러그인 대화(`[VIBE_CODING_REQUEST]`)가 웹 사용자 목록에 그대로
보였다(web 채널에서 실측 확인).

조치(배포됨): 두 경로가 `X-Client-Channel` 헤더로 저장 채널을 정하도록 변경.

```
X-Client-Channel: api      ← 플러그인 요청에 이 헤더만 추가하면 격리된다
```

- 허용값: `web` / `app`(=web과 이력 공유) / `cli` / `api`. 그 외·미지정은 `web`(무회귀).
- 검증 완료: 헤더 있으면 api 채널에만 저장(web 불변), 헤더 없으면 종전대로 web.

**더 나은 선택지**: 아예 `/v1/chat/completions`(OpenAI 호환)로 옮기면 채널이
자동으로 `api`이고 아래 `response_format`(guided decoding)도 쓸 수 있다. 권장.

### (2) 잘린 응답을 완결로 오해하던 문제 → finish_reason 정직화

원인: OpenAI 응답의 `finish_reason`이 항상 `"stop"` 하드코딩이었다. 모델이 출력
한도로 잘려도 클라이언트는 알 수 없었다(= 깨진 JSON의 실제 원인).

조치(배포됨):
- 토큰 한도로 끝나면 `finish_reason: "length"` 를 정직하게 반환(비스트림·스트리밍 모두).
- `response_format` 요청인데 응답이 유효 JSON이 아니면 서버가 경고 로그를 남기고,
  잘림이 아닌 경우 `finish_reason: "content_filter"` 로 신호한다.
- 요청의 `max_tokens` 를 엔진까지 전달하도록 배선(이전에는 완전히 무시됐다).

**플러그인 처리 권장**: `finish_reason !== "stop"` 이면 JSON 파싱을 시도하지 말고
"응답이 잘렸다"고 사용자에게 알리고 재시도(요청 분할 또는 find/replace 축소)를 유도.

**참고 — NOVA의 잘림 자동 복구**: 엔진은 잘림을 감지하면 출력 한도를 4K→8K→16K로
올려 재시도하고, 그래도 모자라면 "이어서 작성" 멀티턴 복구를 최대 3회 수행한다.
그래서 웬만한 잘림은 자동 복구되어 `stop`으로 온다. `length`가 오는 것은 **복구를
다 쓰고도 못 끝낸 경우**이며, 그때는 요청 자체를 줄여야 한다(파일 전문 대신
find/replace 조각 — 아래 §3).

---

## 1. 원인 (진단 확정)

- 플러그인 파서 문제 아님. 모델(A.X-4.0)이 낸 JSON 자체가 깨져 있었음 —
  `content` 문자열 안에 이스케이프(`\n`) 대신 **생 줄바꿈**이 들어감(JSON 문법 위반).
- "파일 전문을 JSON 문자열로 이스케이프해 넣기"는 A.X-4.0의 긴 리터럴 재현 약점과
  정면충돌(2026-08-04~05 CLI에서 반복 실측된 동일 계열: 경로 오타·긴 CSS 붕괴).
  자유 생성에 맡기면 확률적으로 계속 깨진다.

## 2. 해법 ① — response_format으로 guided decoding 강제 (필수)

112 운영 서버는 `structured_output.enabled: true`(OpenAI 표준 `response_format` 주입)가
이미 켜져 있다. 요청에 아래를 추가하면 vLLM이 **문법 차원에서 유효한 JSON만** 생성한다.

검증 완료 요청(2026-08-05, HTTP 200·9s·json.loads 즉시 성공):

```
POST http://192.168.21.112:8600/v1/chat/completions
Authorization: Bearer nexus-coding-key-001
```

```json
{
  "model": "ax-4.0",
  "messages": [{"role": "user", "content": "<규칙+수정요청+파일내용>"}],
  "max_tokens": 2000,
  "temperature": 0.2,
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

주의:
- **`tools`와 `response_format`을 같은 요청에 함께 보내지 말 것**(vLLM 동시 사용 이슈).
- 스키마 직렬화 64KB 상한(`structured_output.max_schema_bytes`).
- 날짜·사용자명 등 사실 값은 모델에 맡기지 말고 플러그인이 프롬프트에 명시(모델이
  `2023-10-05` 환각 실측).

## 3. 해법 ② — changes를 파일 전문 대신 find/replace 조각으로 (강력 권장)

guided decoding은 문법만 보장한다. 긴 문자열 내부의 내용 붕괴(반복 루프)는 문법이 못
막으므로, `content`에 파일 전문을 넣는 설계는 파일이 길수록 위험하다. 위 스키마처럼
`find`(원본 조각, 3~8줄·파일 내 유일) / `replace`(대체 조각)로 바꿀 것.

프롬프트 규칙(검증에 사용한 문구):

```
- changes[].find 는 파일에 "실제로 존재하는" 원본 텍스트 조각(3~8줄, 파일 내 유일)을 그대로 복사한다.
- changes[].replace 는 find를 대체할 새 텍스트. 수정 최소화.
- 파일 전문을 넣지 않는다. 필요한 조각만.
```

### 적용 로직 (TypeScript 이식본 — NOVA CLI Edit 폴백과 동일 알고리즘)

정확 매칭 실패 시(모델의 공백/들여쓰기 재현 실수, CRLF/LF 불일치 포함) 공백 정규화
매칭으로 폴백한다. **정규화 매치가 정확히 1곳일 때만** 적용(모호하면 실패 = fail-closed).

```typescript
function normLine(line: string): string {
  return line.split(/\s+/).filter(Boolean).join(" ");
}

/** 공백 정규화 기준으로 find와 일치하는 "유일한" 구간의 [시작,끝) 오프셋. 없거나 모호하면 null. */
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
    if (ok) { matches.push(i); if (matches.length > 1) return null; } // 모호 → 포기
  }
  if (matches.length !== 1) return null;
  const s = matches[0], e = s + w - 1;
  return [offsets[s], offsets[e] + lines[e].length];
}

/** 정확 매칭(유일) → 공백 정규화 폴백 순서로 적용. 실패 시 null(사용자에게 diff 표시 권장). */
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

## 4. 방어선 — 파싱 실패 시 1회 재요청 (선택)

response_format 적용 후에는 이론상 불필요하지만, 방어적으로:
①응답에서 ```json 펜스 제거 후 파싱 ②실패 시 파서 오류 메시지를 붙여
"유효한 JSON만 다시 출력" 1회 재요청(NOVA DocumentExport에서 검증된 패턴).

## 5. 검증 기록

- 2026-08-05, 대상 `D:\workspace\dynamic_prompt\backend\app.py`(15KB 전문 컨텍스트),
  과제 "chat 함수 docstring에 수정일 주석 추가".
- 결과: HTTP 200(9s) → json.loads 즉시 성공 → changes 1건 정확 매칭 적용 →
  요청 주석이 정확한 위치에 반영. 검증 스크립트는 NOVA 세션 scratchpad
  `verify_vibe_json.py`.
