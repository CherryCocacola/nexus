# =============================================================================
# Nexus 분리구조 — 서비스 서버(웹/오케스트레이터) 재기동 스크립트
# =============================================================================
# 무엇을 하는가:
#   서비스 서버(Machine A의 Nexus 웹, 포트 8600)가 내려갔거나 상태가 꼬였을 때
#   "기존 프로세스 종료 → 새로 분리 기동 → 헬스 확인"을 한 번에 수행한다.
#   start_web.ps1 은 포그라운드로 창을 점유하지만, 이 스크립트는 로그 파일로
#   분리(detached) 기동해 실행 후 터미널을 돌려준다.
#
# 언제 쓰나:
#   - 웹이 죽었을 때 복구.
#   - 설정/코드 변경 후 재기동(예: degeneration 수정, 도구 추가 반영).
#
# 안전장치:
#   - Nexus 웹(uvicorn web.app:app)만 종료한다. 같은 PC의 다른 프로젝트
#     (uvicorn app:app @ 8080 등)는 커맨드라인으로 구분해 절대 건드리지 않는다.
#   - 백엔드(B200) 연결용 SSH 터널 포트가 열려 있는지 사전 점검하고, 없으면
#     경고만 낸다(터널은 tunnel.ps1 이 담당 — 이 스크립트는 웹만 재기동).
#
# 사용법:
#   powershell -ExecutionPolicy Bypass -File deploy_pc\restart_web.ps1
#   (옵션) -Port 8600  -TimeoutSec 60
# =============================================================================

param(
  [int]$Port = 8600,        # 웹 리슨 포트(기본 8600)
  [int]$TimeoutSec = 60     # 기동 완료를 기다리는 최대 초
)

$ErrorActionPreference = "Continue"
$Root = "D:\workspace\nexus-b200"
$Venv = "$Root\.venv_pc\Scripts\python.exe"
Set-Location $Root

Write-Host "=== Nexus 서비스 서버(웹) 재기동 ===" -ForegroundColor Cyan

# ── 1) 기존 Nexus 웹 프로세스 종료 ──────────────────────────────────────────
# uvicorn 커맨드라인에 'web.app:app' 이 있는 python 프로세스만 고른다.
# (8080의 다른 프로젝트는 'app:app' 이라 정규식 'web\.app:app' 에 걸리지 않는다.)
$killed = @()
Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
  Where-Object { $_.CommandLine -match 'uvicorn\s+web\.app:app' } |
  ForEach-Object {
    Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
    $killed += $_.ProcessId
  }
if ($killed.Count -gt 0) {
  Write-Host "[kill] 기존 웹 프로세스 종료: PID $($killed -join ', ')" -ForegroundColor Yellow
} else {
  Write-Host "[kill] 실행 중인 웹 프로세스 없음 (새로 기동)" -ForegroundColor DarkGray
}

# 포트가 완전히 해제될 때까지 잠깐 대기(바인드 경합 방지, 최대 5초).
for ($i = 0; $i -lt 10; $i++) {
  if (-not (Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue)) { break }
  Start-Sleep -Milliseconds 500
}

# ── 2) 백엔드 터널 사전 점검(경고만) ────────────────────────────────────────
# 웹은 B200 백엔드에 터널 로컬포트로 접속한다(pc.yaml). 터널이 없으면 기동은 되나
# 추론/DB가 안 되므로, 열려 있는지 확인해 경고한다. 터널은 tunnel.ps1 이 담당한다.
$needTunnel = @(18001, 18002, 15440, 16340)   # vLLM/임베딩/PG/Redis (필수)
$missing = $needTunnel | Where-Object {
  -not (Get-NetTCPConnection -State Listen -LocalPort $_ -ErrorAction SilentlyContinue)
}
if ($missing.Count -gt 0) {
  Write-Host "[경고] 터널 포트 미개통: $($missing -join ', ') — tunnel.ps1 을 먼저 실행하세요." -ForegroundColor Red
} else {
  Write-Host "[터널] 백엔드 포트 정상(18001/18002/15440/16340). 이미지/비전은 18003/18004." -ForegroundColor DarkGray
}

# ── 3) 환경변수 로드(.env) + 설정 ──────────────────────────────────────────
# 비밀번호(NEXUS_PG_PASSWORD/NEXUS_REDIS_PASSWORD)는 .env 에서 프로세스 환경으로 로드.
if (Test-Path "$Root\.env") {
  Get-Content "$Root\.env" | ForEach-Object {
    if ($_ -match '^\s*([^#][^=]+?)\s*=\s*(.*?)\s*$') {
      $name = $matches[1]; $val = $matches[2].Trim("'").Trim('"')
      [Environment]::SetEnvironmentVariable($name, $val, "Process")
    }
  }
}
# PC 전용 설정(터널 로컬포트를 가리키는 사본)을 쓴다.
$env:NEXUS_CONFIG = "config\nexus_config.pc.yaml"
$env:PYTHONIOENCODING = "utf-8"
if (-not (Test-Path "$Root\logs")) { New-Item -ItemType Directory "$Root\logs" | Out-Null }

# ── 4) 분리(detached) 기동 + 로그 캡처 ─────────────────────────────────────
# venv python 을 직접 백그라운드로 띄우고 stdout/stderr 를 로그 파일로 남긴다.
# (래퍼 없이 직접 기동해야 이 스크립트가 끝나도 웹이 계속 살아 있는다.)
$outLog = "$Root\logs\web_stdout.log"
$errLog = "$Root\logs\web_stderr.log"
$proc = Start-Process -FilePath $Venv `
  -ArgumentList "-m", "uvicorn", "web.app:app", "--host", "0.0.0.0", "--port", "$Port", "--log-level", "info" `
  -WorkingDirectory $Root `
  -RedirectStandardOutput $outLog -RedirectStandardError $errLog `
  -WindowStyle Hidden -PassThru
Write-Host "[start] 웹 기동 시작: PID $($proc.Id) (로그: logs\web_stdout.log / web_stderr.log)" -ForegroundColor Cyan

# ── 5) 헬스 폴링(기동 완료 확인) ───────────────────────────────────────────
# 부트스트랩(PG/Redis/vLLM 연결)에 수 초 걸린다. 200 또는 401/403(인증 활성이면
# 서버는 정상)이면 성공으로 본다. 그 사이 프로세스가 죽으면 즉시 실패 처리.
$ok = $false
$elapsed = 0
while ($elapsed -lt $TimeoutSec) {
  Start-Sleep -Seconds 3
  $elapsed += 3
  # 프로세스가 죽었으면 로그를 보여주고 중단.
  if (-not (Get-Process -Id $proc.Id -ErrorAction SilentlyContinue)) {
    Write-Host "[실패] 웹 프로세스가 종료됨. stderr 마지막 20줄:" -ForegroundColor Red
    if (Test-Path $errLog) { Get-Content $errLog -Tail 20 }
    break
  }
  try {
    $r = Invoke-WebRequest "http://127.0.0.1:$Port/metrics" -TimeoutSec 4 -UseBasicParsing -ErrorAction Stop
    if ($r.StatusCode -eq 200) { $ok = $true; break }
  } catch {
    $code = $_.Exception.Response.StatusCode.value__
    if ($code -eq 401 -or $code -eq 403) { $ok = $true; break }
  }
}

if ($ok) {
  Write-Host "[성공] 웹 기동 완료 (~${elapsed}s)" -ForegroundColor Green
  Write-Host "  · 로컬:  http://127.0.0.1:$Port" -ForegroundColor Green
  Write-Host "  · LAN :  http://192.168.20.206:$Port  (Bearer API 키 필요)" -ForegroundColor Green
} else {
  Write-Host "[실패] ${TimeoutSec}초 내 기동 확인 실패. logs\web_stderr.log 를 확인하세요." -ForegroundColor Red
  exit 1
}
