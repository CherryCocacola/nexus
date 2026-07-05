# =============================================================================
# Nexus 분리구조 — GPU 백엔드(B200) SSH 터널
# =============================================================================
# 이 스크립트는 무엇인가:
#   서비스 PC(이 컴퓨터 = Machine A/오케스트레이터)가 B200의 백엔드 서비스
#   (vLLM 8001 / 임베딩 8002 / PostgreSQL 5440 / Redis 6340)를 "공개 노출 없이"
#   이 PC의 localhost 로 안전하게 당겨오는 SSH 로컬 포워딩이다.
#
# 사용법:
#   1) 이 창에서 실행 → 키 passphrase 입력 → 창을 "열어둔 채" 둔다.
#   2) 다른 PowerShell 창에서 start_web.ps1 (또는 start_cli.ps1) 실행.
#
# 다른 서버로 이전 시: $KeyPath, $BastionHost, $BastionPort 만 바꾸면 됨.
# =============================================================================

$KeyPath     = "D:\workspace\nexus\user_mig\nexus_key"  # SSH 개인키(passphrase 있음)
$BastionUser = "idino_user"
$BastionHost = "59.150.33.1"
$BastionPort = 45702

Write-Host "[터널] B200 백엔드로 연결합니다 (passphrase 입력 필요)..." -ForegroundColor Cyan
Write-Host "[터널] 이 창을 닫지 마세요. 닫으면 서비스가 백엔드에 못 붙습니다." -ForegroundColor Yellow

ssh -N `
  -L 8001:127.0.0.1:8001 `
  -L 8002:127.0.0.1:8002 `
  -L 5440:127.0.0.1:5440 `
  -L 6340:127.0.0.1:6340 `
  -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -o ExitOnForwardFailure=yes `
  -o StrictHostKeyChecking=no -o UserKnownHostsFile=NUL `
  -i "$KeyPath" -p $BastionPort "$BastionUser@$BastionHost"
