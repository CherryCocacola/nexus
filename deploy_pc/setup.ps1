# =============================================================================
# Nexus 서비스 PC — 최초 1회 셋업 (Windows)
# =============================================================================
# 새 PC/서버로 옮겼을 때 이 스크립트 하나만 실행하면 오케스트레이터 실행 환경이
# 준비된다. (GPU 불필요 — 추론/임베딩은 B200 백엔드가 담당)
#
# 실행: PowerShell 에서  .\deploy_pc\setup.ps1
# =============================================================================

$Root = Split-Path -Parent $PSScriptRoot   # 프로젝트 루트(= deploy_pc 의 상위)
Set-Location $Root
Write-Host "[setup] 프로젝트 루트: $Root" -ForegroundColor Cyan

# 1) venv 생성
if (-not (Test-Path "$Root\.venv_pc")) {
  Write-Host "[setup] venv 생성(.venv_pc)..." -ForegroundColor Cyan
  python -m venv "$Root\.venv_pc"
}
$Py = "$Root\.venv_pc\Scripts\python.exe"

# 2) 의존성 설치
Write-Host "[setup] 오케스트레이터 의존성 설치..." -ForegroundColor Cyan
& $Py -m pip install --upgrade pip
& $Py -m pip install -r "$Root\deploy_pc\requirements-pc.txt"

# 3) 안내
Write-Host ""
Write-Host "[setup] 완료. 다음을 확인/수정하세요:" -ForegroundColor Green
Write-Host "  - .env                     : NEXUS_PG_PASSWORD / NEXUS_REDIS_PASSWORD (B200 백엔드 비번)"
Write-Host "  - config\nexus_config.yaml : 백엔드 호스트(127.0.0.1 = 터널). 백엔드가 원격이면 여기 수정"
Write-Host "  - deploy_pc\tunnel.ps1     : SSH 키 경로/Bastion 주소"
Write-Host ""
Write-Host "실행 순서: (1) deploy_pc\tunnel.ps1  (2) deploy_pc\start_web.ps1  또는  start_cli.ps1" -ForegroundColor Yellow
