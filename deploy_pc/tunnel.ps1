# =============================================================================
# Nexus 분리구조 — GPU 백엔드(B200 "nova") SSH 터널
# =============================================================================
# 이 스크립트는 무엇인가:
#   서비스 PC(이 컴퓨터 = Machine A/오케스트레이터)가 B200의 모델 서버들을
#   "공개 노출 없이" 이 PC의 localhost 로 안전하게 당겨오는 SSH 로컬 포워딩이다.
#
#   ※ 이 PC는 8001~8005 를 다른 Docker 스택(idino_career·docutil)이 이미 점유하고
#     있어, 로컬 바인드 포트를 18001~18005 로 옮겼다(원격 포트는 그대로).
#     config/nexus_config.pc.yaml 이 이 로컬 포트들을 가리킨다 — 짝을 맞춰야 한다.
#
# 사용법:
#   1) 이 창에서 실행 → 아무 출력 없이 멈춰 있으면 정상. 창을 "열어둔 채" 둔다.
#   2) 다른 창에서 start_web.ps1 또는 start_cli.ps1 실행.
#
# 다른 서버로 이전 시: $KeyPath, $BastionHost, $BastionPort 만 바꾸면 됨.
#
# ── 2026-08-18 수정 ──────────────────────────────────────────────────────────
#   ① 키를 nexus_key → nova_key 로 교체. 07-30 B200 재구축 때 키가 바뀌었는데
#      스크립트가 옛 키(구 리포 D:\workspace\nexus\...)를 계속 가리켜 인증이
#      항상 실패했다(Permission denied (publickey)). 실측으로 확인.
#   ② 코딩 서버 18005(Devstral) 포워딩 추가 — pc.yaml 이 요구하는데 빠져 있었다.
#   ③ "passphrase 입력 필요" 문구 제거 — 이 키는 passphrase 가 없다(07-13 확인).
# =============================================================================

# 스크립트 위치 기준으로 리포 루트를 잡는다 — 체크아웃 경로가 달라도 동작한다.
$Root        = Split-Path $PSScriptRoot -Parent
$KeyPath     = Join-Path $Root "user_mig\nova_key"
$BastionUser = "idino_user"
$BastionHost = "59.150.33.1"
$BastionPort = 45702

if (-not (Test-Path $KeyPath)) {
  Write-Host "[터널] SSH 키를 찾을 수 없습니다: $KeyPath" -ForegroundColor Red
  Write-Host "       이 키는 git 에 올라가지 않습니다(.gitignore). 담당자에게 받아 두십시오." -ForegroundColor Red
  exit 1
}

Write-Host "[터널] B200(nova) 백엔드로 연결합니다..." -ForegroundColor Cyan
Write-Host "[터널] 이 창을 닫지 마세요. 닫으면 서비스가 백엔드에 못 붙습니다." -ForegroundColor Yellow
Write-Host "[터널] 추론 18001 · 임베딩 18002 · 이미지 18003 · 비전 18004 · 코딩 18005" -ForegroundColor DarkGray

# -L 로컬포트:원격기준호스트:원격포트
#   18001 vLLM(A.X-4.0) / 18002 임베딩(e5-large) / 18003 이미지(FLUX)
#   18004 비전(Gemma) / 18005 코딩(Devstral)
#   15440·16340 은 pc.yaml 이 DB 를 LAN(192.168.21.112)으로 직결하므로 쓰지 않지만,
#   restart_web.ps1 의 사전 점검이 아직 이 포트를 보므로 호환을 위해 남긴다.
ssh -N `
  -L 18001:127.0.0.1:8001 `
  -L 18002:127.0.0.1:8002 `
  -L 18003:127.0.0.1:8003 `
  -L 18004:127.0.0.1:8004 `
  -L 18005:127.0.0.1:8005 `
  -L 15440:127.0.0.1:5440 `
  -L 16340:127.0.0.1:6340 `
  -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -o ExitOnForwardFailure=yes `
  -o StrictHostKeyChecking=no -o UserKnownHostsFile=NUL `
  -i "$KeyPath" -p $BastionPort "$BastionUser@$BastionHost"
