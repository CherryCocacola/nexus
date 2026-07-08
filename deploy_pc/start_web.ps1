# =============================================================================
# Nexus 분리구조 — 서비스 PC에서 웹(오케스트레이터) 기동
# =============================================================================
# 전제: tunnel.ps1 이 다른 창에서 실행 중이어야 한다(B200 백엔드 연결).
# 하는 일: .env 로드 → uvicorn 으로 Nexus 웹(8443) 기동. 연산은 터널 너머 B200.
# 접속: 브라우저 http://localhost:8443  (인증 켜져 있으면 API 키 필요)
# =============================================================================

$Root = "D:\workspace\nexus-b200"
$Venv = "$Root\.venv_pc\Scripts\python.exe"
Set-Location $Root

# .env 를 프로세스 환경변수로 로드(NEXUS_PG_PASSWORD / NEXUS_REDIS_PASSWORD)
if (Test-Path "$Root\.env") {
  Get-Content "$Root\.env" | ForEach-Object {
    if ($_ -match '^\s*([^#][^=]+?)\s*=\s*(.*?)\s*$') {
      $name = $matches[1]; $val = $matches[2].Trim("'").Trim('"')
      [Environment]::SetEnvironmentVariable($name, $val, "Process")
    }
  }
}
$env:PYTHONIOENCODING = "utf-8"

# 포트 8600 사용(8443 은 Docker Desktop 등 기존 서비스와 충돌하므로 회피).
# 바인딩 0.0.0.0 — 같은 게이트웨이(LAN)의 다른 PC에서도 접속 가능하게 연다.
#   · 로컬:  http://127.0.0.1:8600
#   · LAN :  http://192.168.20.206:8600  (이 PC = Machine A)
#   · 보호:  web_auth 활성(Bearer API 키 필수) + 방화벽 규칙 "Nexus Web 8600 (LAN)"
#            (TCP/8600, Domain·Private 프로필 — 신뢰 LAN에서만 허용).
# PC 전용 설정 사용 — 터널 로컬 포트(18001/18002/15440/16340)를 가리키는 사본.
# 이렇게 분리해야 B200 co-located용 공용 config(127.0.0.1:8001…)를 훼손하지 않는다.
$env:NEXUS_CONFIG = "config\nexus_config.pc.yaml"
Write-Host "[웹] 오케스트레이터 기동: http://192.168.20.206:8600 (LAN) / http://127.0.0.1:8600 (로컬)  (백엔드=B200 터널)" -ForegroundColor Cyan
& $Venv -m uvicorn web.app:app --host 0.0.0.0 --port 8600 --log-level info
