# =============================================================================
# Nexus 분리구조 — 서비스 PC에서 대화형 CLI(REPL) 기동
# =============================================================================
# 전제: tunnel.ps1 이 다른 창에서 실행 중이어야 한다(B200 백엔드 연결).
# 하는 일: .env 로드 → 대화형 채팅 REPL. 연산은 터널 너머 B200.
# =============================================================================

$Root = "D:\workspace\nexus-b200"
$Venv = "$Root\.venv_pc\Scripts\python.exe"
Set-Location $Root

if (Test-Path "$Root\.env") {
  Get-Content "$Root\.env" | ForEach-Object {
    if ($_ -match '^\s*([^#][^=]+?)\s*=\s*(.*?)\s*$') {
      $name = $matches[1]; $val = $matches[2].Trim("'").Trim('"')
      [Environment]::SetEnvironmentVariable($name, $val, "Process")
    }
  }
}
$env:PYTHONIOENCODING = "utf-8"

Write-Host "[CLI] 대화형 REPL 시작 (백엔드=B200 터널). 종료: /exit" -ForegroundColor Cyan
& $Venv -m cli.repl
