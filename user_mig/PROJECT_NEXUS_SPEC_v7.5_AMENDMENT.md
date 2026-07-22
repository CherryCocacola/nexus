# PROJECT NEXUS SPEC v7.5 AMENDMENT — 코딩 RAG 품질·응답속도·UI (2026-07-20)

v7.4(이미지 생성) 이후 개정. 본 세션에서 반영된 코딩 RAG 교차언어 recall, degeneration
수정, 할루시네이션 지침, 스트리밍 UI, **리랭커 GPU 이전(응답속도 근본해결)**을 명세한다.
전부 라이브 반영 + 커밋 완료. 운영 상세·실측치는 progress.md.

---

## §A. 코딩 RAG — 문서측 번역(so_ko) 교차언어 recall

**문제.** 한국어 코딩질의 ↔ 영어 SO 코퍼스는 e5·리랭커가 교차언어 매칭 실패
(리랭크 0.0~0.09). 질의번역은 한국어 코퍼스(kowiki/okky)를 파괴(실측 5중 3 손실)라
무차별 불가 → **소스-인지 문서측 번역** 채택.

**설계.** 각 SO 청크 content 앞에 **한국어 제목 글로스**(`# {ko_gloss}`)를 부착. 글로스는
임베딩·리랭커가 함께 보는 텍스트에 들어가야 게이트 통과(프로브: 본문만 0.0→글로스+본문
1.0). 신규 소스 `so_ko`로 적재(섀도, 롤백=DROP). 코딩 테넌트 `allowed_knowledge_sources`
를 so→so_ko 스왑.

**구현.** `scripts/prepare_stackoverflow.py`: `apply_gloss`(NUL 제거 포함), `translate_titles`
(A.X /v1/chat/completions EN→KO 배치), `run_glossify --reuse-embedding`(기존 임베딩 복사=
재임베딩 0, keyset 페이지네이션·캐시 이어받기). 번역기=B200 A.X(오프라인 1회, 67,218 제목).

**결과.** so_ko 255,769(=so). fetch_k=20에서 recall 3/8→5/8(지연비용 0). 8/8은 질의번역/
GPU 재임베딩 필요(보류). RAG는 보완재(수술적)로 유지, 대부분 코딩지식은 베이스 A.X.

## §B. Degeneration 수정 — 샘플링 + 스트리밍 워치독

**근본원인(반직관).** knowledge_mode의 과도한 반복페널티(rep 1.15+freq 0.3)가 긴 생성에서
종결어미·조사 등 초고빈도 토큰을 억눌러 EOS 간접붕괴→희귀어휘 폭주(finish=length). 웹
MAX_OUTPUT_RECOVERY 에스컬레이션이 이를 이어붙여 증폭(19k자·타임아웃).

**수정 1(샘플링).** knowledge_mode·chat_mode **rep 1.05·freq 0.1**(tool_mode 불변).
반복0·붕괴0 실측(config 3본 동기화 + 112 bind-mount). 페널티는 양쪽 절벽 — 부족(tool_mode,
v6.1)도 과다(knowledge)도 붕괴.

**수정 2(스트리밍 워치독).** `core/orchestrator/stream_watchdog.py` `DegenerationMonitor`
신설. 생성 중 붕괴를 감지해 **예외 아닌 graceful return**(MESSAGE_STOP 前 종료→에스컬레이션
미발동, 깨끗한 앞부분 유지) + `stream.aclose()`. 4 휴리스틱:
  - 전역 구절빈도(개행/문장부호로 끊은 ≥20자 구절이 전체에서 >6회 = **분산 반복**, kd01형)
  - window 동일라인 ≥5회 / 문자4gram 최빈>0.45 / 이모지밀도>0.15
`stream_with_watchdog(detect_degeneration=...)`, `query_loop DEGEN_GUARD_ENABLED`. 기본 off=무회귀.

## §C. 할루시네이션 방지 지침(관측 불가 정보)

`web/prompts/worker_system(_full).md` Hard rules에 규칙 추가: 로컬/실시간/비공개/미래 등
**관측 불가 정보는 구체값 단정 금지, 확인법 안내**(실패사례 예시: 버전→python --version,
실행시간→timeit, 포트→ss). 품질테스트에서 할루시네이션 3건→0.

## §D. 웹 UI — 스트리밍 마크다운 실시간 렌더

`web/static/index.html`: 스트리밍 중 원문(textContent) 노출→완료후 재정리하던 것을,
원문 누적(`_streamBodyRaw`)과 표시(innerHTML) 분리 + ~60ms throttle로 formatContent 렌더 +
미완성 코드펜스 임시닫아 렌더(팝인 방지). 처음부터 정리된 채 라이브 표시.

## §E. ★ 리랭커/임베딩 GPU 이전 — 응답속도 근본해결 (토폴로지 변경)

**진단.** TTFT 분해: 지연 전부가 첫 토큰 전(RAG), 생성 아님. 범인=**112 CPU 크로스인코더
리랭커가 긴 청크(~1300자) 20개 채점 = 25초**(짧은 문서 0.9s — 문서길이가 지배). B200/LLM 무관.

**원인.** 5090이 **유물 vLLM(qwen3.5-27b-awq, 8001)**에 29GB 점유 → 리랭커 CPU 유배. 추론은
B200(18001) 이전 완료라 8001 미사용(연결0·config 미참조 확인).

**조치(P4 토폴로지 갱신).**
  - `nexus-vllm.service` stop + **disable**(5090 32GB 확보, 복구=enable).
  - `nexus-embedding.service`(=embed_server.py, /v1/embed·/v1/rerank) env
    **EMBED_DEVICE/RERANK_DEVICE=cuda**(CUDA=fp16 조건부). GPU 3.7GB 사용.
  - **결과: 리랭크 25s→0.44s(~57배), TTFT 25~30s→~1s, 간단질문 총 30s→1.3s.**

**아키텍처 함의.** RAG 리랭킹은 **동기 핫패스 + CPU 크로스인코더**면 지연 안티패턴.
GPU 배치가 정답(설계 결함 아님, 배치 문제). 5090 여유 ~28GB → 향후 로컬 모델(Motif-12.7B
QLoRA ~18GB)·VLM·학생용 여지(단 리랭커와 compute 경합 고려). [[project_segment_model_strategy]]

---

## 배포 상태 (2026-07-20 기준, 전부 LIVE)

- **라이브 컨테이너 이미지**: `nexus-web:latest` = `nexus-web:mdstream2-20260720_074620`
  (so_ko 스왑 + 워치독 + 할루시 지침 + 스트리밍 UI 포함, docker commit 스냅샷).
- **config(112 bind-mount)**: `/home/idino/nexus-config/nexus_config.112.yaml` — 페널티 완화·
  rerank.enabled:true. **리랭커 GPU는 systemd env**(스냅샷 무관, 재부팅 영속).
- **git**: feature/b200-bakeoff, origin push 완료(462359e까지). 이후 세션 커밋은 미push 가능.
- **롤백**: 각 단계 백업(tenants.bak.pre-coding, nexus_config.bak-*, service.bak-cpu,
  index.html.bak-mdstream, 이전 스냅샷 degenfix2/hallucfix/chunkfix, 원본 source='so').
