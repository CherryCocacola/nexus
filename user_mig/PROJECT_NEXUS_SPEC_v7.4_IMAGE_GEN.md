# Project Nexus (IDINO NOVA) — 기술 사양서 v7.4 (신규 기능 설계/구현)

## 이미지 생성 도구 (Image Generation Tool)

**버전**: 7.4 (Image Generation — Feature Design & Machine A Implementation)
**기준 문서**: PROJECT_NEXUS_SPEC_v7.3_DOC_INGEST.md, v7.2_AMENDMENT.md (MCP)
**작성일**: 2026-07-09
**상태**: Machine A(도구) 구현 완료 · 기본 비활성(enabled:false) · Machine B(FLUX 서버) 미기동

---

## 개정 개요

### 왜 v7.4인가

Claude Code(본 프로젝트가 재구현하는 원본)에는 **이미지 생성 도구가 없다**. 즉
이 기능은 패리티 복원이 아니라 **원본을 넘어서는 신규 확장**이며, 제품 차별화
요소다. 확산(diffusion) 모델로 텍스트 프롬프트→이미지를 생성하되, 기존
`DocumentExport`(생성물→저장→다운로드 URL) 패턴과 **대칭**으로 붙인다.

### v7.4가 하는 것 / 하지 않는 것

| 한다 | 하지 않는다 |
|---|---|
| `ImageGenerate` 도구(BaseTool) 추가 | 4-Tier 체인·권한·Hook 구조 변경 |
| GPU의 FLUX 이미지 서버 HTTP 계약 정의 | GPU 직접 호출(P4 준수 — HTTP만) |
| DocumentExport와 동일 다운로드/미리보기 경로 재사용 | ContentBlock(코어 모델) 변경 |
| fail-closed 기본 비활성(enabled:false) | 비전(이미지 이해) — 별도 과제 |

---

## 아키텍처 — 도구(Machine A) + 서버(Machine B)

```
사용자 "로고 그려줘"
  → A.X-4.0이 ImageGenerate 도구 호출        (Machine A, 신규 도구)
    → POST {image_url}/v1/images/generate    (Machine B, GPU / LAN·HTTP)
      → FLUX 생성 → base64 PNG 반환
    → exports 샌드박스에 .png 저장 → 다운로드 URL 반환
  → 웹이 미리보기/다운로드 노출 (DocumentExport와 동일 경로)
```

P4(2-Machine) 준수: 도구는 GPU/CUDA를 직접 만지지 않고 LAN HTTP로만 호출한다.

## 이미지 서버 계약 (Machine B — FLUX)

- `POST {image_url}/v1/images/generate`
- 요청: `{"prompt": str, "width": int, "height": int, "steps": int, "seed": int|null}`
- 응답: `{"image_base64": "<png base64>", "width": int, "height": int, "seed": int, "model": str}`
- 권장 모델: **FLUX.1 schnell** (Apache 2.0 = 상업 배포 OK, 텍스트 렌더링 우수, 1~4 스텝으로 저사양에서도 빠름). steps 기본 4.
- `image_url`: `config gpu_server.image_url`. nexus_config.yaml=`http://127.0.0.1:8003`, nexus_config.pc.yaml=`http://127.0.0.1:18003`(터널).

## ImageGenerate 도구 사양 (구현 완료)

- **name**: `ImageGenerate` (aliases: GenerateImage/DrawImage/CreateImage, group: file)
- **input_schema**: `prompt`(string, required — 영어 프롬프트 권장), `size`(enum 1024x1024/1024x1536/1536x1024, default 1024x1024), `seed`(int, optional)
- **behavior**: is_read_only=False, is_concurrency_safe=False(GPU 무거움 — 명시 override), is_destructive=False, requires_confirmation=False (fail-closed)
- **call**: size→width/height, httpx POST(타임아웃 120s) → base64 PNG 디코드 → DocumentExport의 `resolve_exports_dir()`/`_safe_filename()` 재사용해 동일 exports 샌드박스에 `.png` 저장 → `/v1/download/{filename}` URL 반환. metadata=download_url/preview_url/width/height/seed/model.
- **오류 처리**: httpx ConnectError/TimeoutException/HTTPStatusError를 구체적으로 잡아 `<tool_use_error>`로 래핑(anti-pattern #8, bare except 없음). 서버 미기동 시에도 크래시 없이 tool_use_error 반환.

## 하드웨어 (GB10 최저 기준)

FLUX schnell은 저스텝(1~4)이라 대역폭이 낮은 GB10(128GB 통합)에서도 1장 ~5–15초 예상(실측 전 추정). B200에선 ~1–2초. VRAM은 어디서든 여유.

## 에어갭 · 라이선스

- 모델 가중치는 **오프라인 사전 번들**(런타임 다운로드 금지). FLUX schnell ~24GB(fp8 ~12GB).
- 상업 배포 대상이므로 **FLUX.1 schnell(Apache 2.0)** 또는 SDXL(OpenRAIL++)만. FLUX.1 **dev(비상업)** 는 고객 배포 제외.

## 활성/비활성 (메모리 관리)

- 기본 `config/tool_mappings.yaml`에서 `ImageGenerate: enabled: false` (FLUX 서버 미기동 → fail-closed).
- 서버 기동 후 `enabled: true`로 활성. 메모리 압박 시 다시 `false` 한 줄로 즉시 비활성(레지스트리에서 제외) — 또는 권한 Layer 1 DenyRule로 제거.

## 남은 작업 (Machine B)

- B200/GB10에 FLUX schnell diffusers FastAPI 서버(위 계약) 기동 + 오프라인 가중치 번들.
- 서버 검증 후 tool_mappings `enabled: true` + 실 e2e(생성→저장→웹 미리보기) 확인.
