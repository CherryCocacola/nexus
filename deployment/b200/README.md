# B200 GPU 백엔드 — 재부팅 자동기동

B200(GPU 서버)에서 도는 것들을 재부팅 후 스스로 살아나게 하는 장치를 담는다.

## 배경 — 왜 systemd 가 아닌가

112(서비스 서버)는 systemd 유닛으로 정리돼 있지만, **B200에는 유닛이 하나도 없다.**
시도해 봤으나 두 가지가 막았다.

| 시도 | 결과 |
|---|---|
| 시스템 유닛(`/etc/systemd/system`) | `sudo` 에 비밀번호가 필요하다(키 인증만 가능) |
| 사용자 유닛(`systemctl --user`) | `Failed to connect to bus: No medium found` |

비root 로 재부팅 자동기동을 거는 수단이 **사용자 crontab 의 `@reboot`** 뿐이었다.

## 구성

```
@reboot → boot_start.sh → start_all.sh(멱등) → 각 tmux 세션 + watchdog
                                                    ↑
                                      watchdog.sh 가 60초마다 start_all.sh 재호출
```

### 이 디렉토리의 파일 (전부 `/NHNHOME/nexus/` 로 배포된다)

| 파일 | 역할 |
|---|---|
| `boot_start.sh` | `@reboot` 진입점. 마운트·GPU 준비를 기다렸다 `start_all.sh` 호출 |
| `start_all.sh` | **멱등** 기동기. 포트를 확인해 죽은 것만 띄운다 |
| `watchdog.sh` | 60초마다 `start_all.sh` 재호출 — 크래시 자가복구 |
| `run_vllm_fp8.sh` | A.X-4.0 FP8 (GPU0:8001, util 0.70) |
| `run_vision.sh` | Gemma 3 27B 비전 (GPU1:8004, util 0.42) |
| `run_coder.sh` | Devstral Small (GPU0:8005, util 0.20) |
| `setup_db.sh` | PG17+Redis 최초 구성(1회성) |
| `install_db.sh` / `dl_embed.sh` | 패키지 설치 / 임베딩 모델 내려받기 |
| `env.example` / `redis.conf.example` | 자격증명 템플릿 — 아래 참고 |

**서버에만 있는 것**: `.env`(600), `redis.conf`(600), `scripts/embed_server.py`,
`scripts/image_server.py`, 모델 캐시. 앞의 둘은 자격증명이라 버전관리에 올리지 않는다.

`boot_start.sh` 가 start_all.sh 를 그냥 부르지 않고 감싸는 이유는, `@reboot` 가
부팅 아주 이른 시점에 돌기 때문이다. 그때는 `/NHNHOME` 마운트나 NVIDIA 드라이버가
아직 준비되지 않았을 수 있다. 거기서 한 번 실패하고 끝나면 **watchdog 조차 뜨지
않는다**(watchdog 을 띄우는 것이 start_all.sh 이므로 자가복구가 시작되지 않는다).
그래서 준비를 기다린 뒤 부르고, 그 뒤로도 몇 번 더 시도한다.

## 설치

```bash
scp boot_start.sh idino_user@<bastion>:/NHNHOME/nexus/boot_start.sh
ssh idino_user@<bastion> 'chmod +x /NHNHOME/nexus/boot_start.sh'
# crontab 에 아래 한 줄 추가(기존 항목 보존할 것)
@reboot /usr/bin/bash /NHNHOME/nexus/boot_start.sh
```

## 재부팅 후 확인

```bash
cat /tmp/nexus_boot.log          # 언제 무엇을 기다렸고 무엇을 띄웠는지
ss -ltn | grep -E ':800[0-9]'    # 8001 A.X / 8002 임베딩 / 8003 이미지 / 8004 비전 / 8005 코딩
tmux ls                          # 6개 세션(vllm·embed·image·vision·coder·watchdog)
```

## 검증 상태 (2026-08-07)

| 항목 | 상태 |
|---|---|
| cron 최소 환경에서 `start_all.sh` 동작 | 확인 — 종료코드 0, 세션 6→6(멱등) |
| cron 최소 환경에서 `boot_start.sh` 동작 | 확인 — 로그 정상, 서비스 무영향 |
| 실행파일이 cron 기본 PATH 에 있는지 | 확인 — tmux·redis-cli·redis-server 모두 `/usr/bin` |
| cron 이 `@reboot` 를 지원하는지 | 확인 — cron 3.0pl1 (vixie-cron) |
| **`@reboot` 트리거 자체** | **미검증 — 실제 재부팅 전까지 확인 불가** |

마지막 줄이 남은 위험이다. 다음에 B200 을 재부팅하면 **가장 먼저
`/tmp/nexus_boot.log` 를 확인**할 것. 파일이 없으면 @reboot 가 돌지 않은 것이므로
수동으로 `bash /NHNHOME/nexus/start_all.sh` 를 실행하고 원인을 봐야 한다.

## 자격증명 분리 (2026-08-07)

전에는 `start_all.sh` 와 `setup_db.sh` 가 PG·Redis 비밀번호를 **본문에 평문으로** 갖고
있었다. 그래서 이 스크립트들을 버전관리에 올릴 수 없었고, 서버가 재구축되면 GPU
백엔드를 살리는 수단 자체가 사라지는 상태였다. 보안 문제이자 **복구 가능성 문제**였다.

지금은 `/NHNHOME/nexus/.env`(600)에서 읽는다.

```bash
set -a; . "$N/.env"; set +a     # 읽는 동안만 자동 export → 자식 tmux 세션이 물려받는다
```

값에 따옴표는 필요 없다. `#` 이 단어 중간에 있으면 주석이 아니기 때문이다
(`NEXUS_PG_PASSWORD=abc!@#$` 는 온전히 읽힌다 — 실측 확인). 공백이 있으면 감싸야 한다.

**두 스크립트의 실패 방식이 다르다. 의도된 것이다.**

| 스크립트 | `.env` 없을 때 | 왜 |
|---|---|---|
| `start_all.sh` | 경고만 남기고 **계속 진행** | 이게 곧 크래시 복구 수단이다. Redis 인증 하나 때문에 vLLM·비전·이미지 복구까지 멈추면 손해가 더 크다 |
| `setup_db.sh` | **즉시 중단**(fail-closed) | 빈 비밀번호로 role 을 만들면 그 순간 DB 가 무인증으로 열린다 |

`redis.conf` 는 `requirepass` 를 담으므로 서버에만 둔다. `setup_db.sh` 가 `.env` 값으로
생성하며 `umask 077` 로 600 을 보장한다(기본 umask 로 두면 644 가 된다).

### 새 서버에서

```bash
cp env.example /NHNHOME/nexus/.env && chmod 600 /NHNHOME/nexus/.env   # 값 채우기
bash install_db.sh && bash setup_db.sh && bash dl_embed.sh
bash start_all.sh
```
