<!-- 새 2×B200 재구축 + 분리형 아키텍처 + 비전/이미지 배포 핸드오프. 다른 세션 즉시 착수용. -->
# IDINO NOVA — B200 재구축·배포 핸드오프 (2026-07-30)

> 새 세션에서 이 문서만 읽으면 현재 상태를 파악하고 바로 이어서 작업할 수 있다.
> 요약: NHN B200을 **2 GPU로 재생성**하고 전체 스택을 **백지에서 재프로비저닝** 완료.
> 아키텍처는 **112(웹·DB) + B200(GPU 백엔드)** 분리형. 비전(Gemma 3 27B)·이미지(SD3.5) 배포 완료.

---

## 1. 현재 아키텍처 (분리형, 2026-07-30 확정)

```
사용자(브라우저/AgentHub) → 112 서비스 호스트(192.168.21.112, 온프렘)
    · nexus-web (Docker, :8600)  ← 프로덕션 웹(정본)
    · PostgreSQL 17 (:5440, db/user=nexus)  ← RAG·기억 데이터
    · Redis (:6340)
    · 로컬 임베딩 e5-large (:8002)  ← 112가 쓰는 임베딩(터널 부담 최소화)
        │
        └── SSH 터널(systemd nexus-b200-tunnel.service, 키=nova_key) ──→
              NHN Bastion 59.150.33.1:45702 → B200 컨테이너 (GPU 백엔드)
                · A.X-4.0 72B FP8   :8001 (GPU0)   ← 터널 18001→8001
                · SD 3.5 Large 이미지 :8003 (GPU1)   ← 터널 18003→8003
                · Gemma 3 27B 비전   :8004 (GPU1)   ← 터널 18004→8004
```

- **웹·DB는 112**(프로덕션 정본), **B200은 순수 GPU 백엔드**. (B200에도 web8443·PG·Redis·embed가 떠 있으나 이는 **CLI/직접접속용 잉여**로, 112 프로덕션은 안 씀.)
- 112 config(라이브)=`/home/idino/nexus-config/nexus_config.112.yaml` (nexus-web 컨테이너에 마운트). 값: `url=127.0.0.1:18001`, `image_url=127.0.0.1:18003`, `vision_url=127.0.0.1:18004`, `vision_model=gemma-3-27b`, `embedding_url=192.168.21.112:8002`(로컬).

## 2. 접속 정보

| 대상 | 방법 |
|---|---|
| **B200 SSH** | `ssh -i D:\workspace\nexus-b200\user_mig\nova_key -p 45702 idino_user@59.150.33.1` (Git Bash: `-i /d/workspace/nexus-b200/user_mig/nova_key`). PowerShell은 키 권한 조여야 함: `icacls ... /inheritance:r /grant:r "$($env:USERNAME):R"` |
| **112 SSH** | idino / `idino!@#$` (paramiko 사용 — 세션 스크래치패드 `ssh112.py`+`pw112.txt`). sudo도 같은 비번. |
| **112 웹** | `http://192.168.21.112:8600`, 헤더 `Authorization: Bearer nexus-b200-test-key-001` (default 테넌트). |
| **B200 웹(직접)** | SSH 터널 `-L 8443:127.0.0.1:8443` 후 `http://localhost:8443` (잉여, 검증용) |
| **비번/키** | B200 PG/Redis=`idino!@#$`(`/NHNHOME/nexus/.env`). HF 토큰=`/NHNHOME/nexus/.hftoken`(계정 IDINOAi). nova_key=passphrase 없음. |

## 3. B200 운영 (핵심)

- **코드/데이터 루트**: `/NHNHOME/nexus` (800GB 영구 xfs 볼륨. `/`는 휘발성 overlay). venv=`/NHNHOME/nexus/venv`.
- **전체 기동(컨테이너 rerun 후)**: `bash /NHNHOME/nexus/start_all.sh` — 멱등. PG→Redis→임베딩(8002)→A.X-4.0(8001)→비전(8004)→이미지(8003)→웹(8443) 순.
- **상태 확인**: `tmux ls` / `for p in 8001 8002 8003 8004 8443; do curl -s -o /dev/null -w "$p:%{http_code} " 127.0.0.1:$p/health; done`
- **개별 런처**: `run_vllm_fp8.sh`(A.X-4.0), `run_vision.sh`(Gemma 비전), `scripts/image_server.py`(SD3.5), `scripts/embed_server.py`(임베딩). PG=`pg_ctl -D /NHNHOME/nexus/pgdata`.
- **GPU 배치**: GPU0=A.X-4.0(FP8, ~168GB). GPU1=임베딩+Gemma27B+SD3.5(~116GB). CPU 72·RAM 2.2TB.
- **CLI**: `/NHNHOME/nexus/nova ask "질문"` 또는 `nova chat`(TTY 필요: `ssh -t`). env 자동 로드 래퍼. (`.bashrc`에 `alias nova`.)

## 4. 서비스 상세 (모델·서버)

| 서비스 | 모델 | 포트 | 서빙 | 비고 |
|---|---|---|---|---|
| LLM | **A.X-4.0 72B** (SKT, Qwen2.5기반) | 8001 | vLLM 0.26 FP8, max_len 65536, hermes 파서 | 모델 확정(아래 §5) |
| 임베딩 | intfloat/multilingual-e5-large (1024d) | 8002 | sentence-transformers, `/v1/embed` | + 리랭커 bge-reranker-v2-m3-ko(`/v1/rerank`, enabled) |
| 비전 | **google/gemma-3-27b-it** | 8004 | vLLM, served-name `gemma-3-27b`, `/v1/chat/completions`(OpenAI 비전) | 상업 안전 |
| 이미지 | **stabilityai/stable-diffusion-3.5-large** | 8003 | diffusers 0.39 커스텀 서버, `/v1/images/generate`{prompt/width/height/steps/seed}→{image_base64} | steps<20→28 보정 |
| DB | PostgreSQL 17 + pgvector 0.8.5 | 5440 | tb_knowledge 1,067,978행 + tb_memories 285,656행(kowiki+sample) | 112 백업 복원 |

## 5. 모델 선택 결정 (재론 금지 — 근거 확정)

- **LLM = A.X-4.0 유지 확정.** 2×B200(2 GPU, TP=2)이 하드 상한. 대안 전수 검토·기각:
  - Nemotron-Ultra-253B(dense): FP8 TP=4 검증·**TP=2 미검증(NAS)**→2 GPU 로드 위험.
  - Nemotron-3-Ultra-550B-A55B: FP8 550GB라 2×B200(360GB)·**4×H200(564GB)도 KV여유 부족**.
  - A.X K1(519B/33B, 국산·코딩강함): BF16 1TB라 800GB 볼륨 다운로드 불가+FP8 4 GPU 필요.
  - Mistral Large 2 123B(2×B200 FP8 유일 적합): 코딩↑지만 **한국어에서 A.X-4.0에 열세**(KMMLU78.3·CLIcK83.5·토큰33%↓).
  - **원리**: "FP8+2×B200"은 모델 ≤~280B+TP=2 가능만. 한국어 배포에서 A.X-4.0 상위는 "더 큰 국산=A.X K1"뿐인데 4 GPU 필요.
- **비전/이미지 = 상업 안전 최고.** Gemma 3 27B(상업)·SD3.5(연매출$1M미만 상업). 절대최고 Pixtral Large/FLUX.2-dev는 **비상업**이라 제품화 부적합해 배제. Molmo 72B는 무토큰 공개 대안(영어중심).

## 6. 복구 시 겪은 함정 (재발 방지)

1. **hf download**: `HF_HUB_DISABLE_XET=1` + **tmux** 필수(nohup/setsid는 SSH종료로 죽음, Xet 스톨).
2. **gcsudo**: `bash -lic "gcsudo ..."`만 됨(ctn_gcsudo 직접호출은 ENSH_HOME/make_scripts 미로드 실패). 로그인셸에서만 env(HF_HOME 등) 세팅됨.
3. **Git Bash 경로 변환**: `/home/...` 원격경로 인자를 Windows경로로 변환→sftp/exec ENOENT. `MSYS_NO_PATHCONV=1` 필수.
4. **pg_restore**: ivfflat 인덱스 빌드 실패(`maintenance_work_mem 64MB<200MB`)→ `SET maintenance_work_mem='4GB'; CREATE INDEX idx_knowledge_embed ... USING ivfflat(embedding vector_cosine_ops) WITH(lists=1000)` 수동 재빌드(8.3GB). tb_memories hnsw·btree는 정상.
5. **vLLM 0.26 `--limit-mm-per-prompt`**: `image=3` 형식 불가(JSON 필요)→제거.
6. **HF 게이트**: Gemma·SD3.5·FLUX-schnell 게이트(라이선스 수락+토큰). Molmo만 공개.

## 7. 다음 과제 — 제품화 (요청 #3, 미착수)

목표: **웹/DB 서버 별도 + 내부 패키지 배포(코드 유출 방지) + 설치·설정 GUI + CLI**. 별도 상세 설계 필요.
- **코드 보호**: Python을 Nuitka/Cython 컴파일 바이너리 또는 라이선스 잠금 Docker 이미지로 배포(소스 비포함).
- **설치·설정 GUI**: DB주소·웹포트·GPU서버(터널)주소·모델/라우팅·테넌트/API키·RAG소스 입력 → config 생성하는 설치 화면.
- **CLI 포함**: nova CLI 기반.
- 연계: `project_productization_gpu_roadmap`, `project_patent_filing`, 강원대 H200×4 목표.
- **착수 시**: 이 문서 §1~6로 현재 인프라 파악 → 제품화 스펙(아키텍처·단계·기술선택) 먼저 설계.

## 8. 열린 항목

- ⚠️ **GitHub PAT 회전**: git remote URL(`origin`)에 토큰 평문 노출됨 → 폐기 권고.
- B200 web8443·PG·Redis·embed는 112 프로덕션엔 잉여(CLI용). 정리할지 결정 가능.
- 강원대 H200×4 최종 배포는 별도(동일 DC co-locate라 터널 불필요).
