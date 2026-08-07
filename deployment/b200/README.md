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

- `boot_start.sh` (이 디렉토리) — @reboot 진입점. **이 파일만 리포에 있다.**
- `start_all.sh`, `run_vllm_fp8.sh`, `run_vision.sh`, `run_coder.sh`, `watchdog.sh`
  — B200의 `/NHNHOME/nexus/` 에만 있다. 아래 "리포에 없는 이유" 참고.

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

## 리포에 없는 이유 — `start_all.sh` 의 자격증명

`start_all.sh` 는 PG·Redis 비밀번호를 **평문으로 export** 한다.

```bash
export NEXUS_PG_PASSWORD="…"
export NEXUS_REDIS_PASSWORD="…"
```

그대로 커밋하면 자격증명이 git 이력에 영구히 남는다(프로젝트 규칙: 비밀번호 커밋 금지).
그래서 이 파일은 서버에만 두었다. **후속 과제**: 비밀번호를 별도 파일(`.env`, 600)로
빼고 `start_all.sh` 가 그것을 읽게 바꾼 뒤, 스크립트 본문만 여기로 옮긴다.
그전까지 B200 이 재구축되면 이 스크립트들을 다시 만들어야 한다.
