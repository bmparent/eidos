$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$BackendDir = Join-Path $ProjectRoot "backend"
$FrontendDir = Join-Path $ProjectRoot "frontend"
$VenvPython = Join-Path $BackendDir ".venv\Scripts\python.exe"
$Python = if (Test-Path $VenvPython) { $VenvPython } else { "python" }

Write-Host "Starting Eidos Life Lab backend on http://127.0.0.1:8787"
Write-Host "Starting Eidos Life Lab frontend on http://127.0.0.1:5173"
Write-Host "Press Ctrl+C to stop both jobs."

$backendJob = Start-Job -Name "eidos-life-lab-backend" -ScriptBlock {
    param($BackendDir, $Python)
    Set-Location $BackendDir
    & $Python -m uvicorn app:app --host 127.0.0.1 --port 8787 --reload
} -ArgumentList $BackendDir, $Python

$frontendJob = Start-Job -Name "eidos-life-lab-frontend" -ScriptBlock {
    param($FrontendDir)
    Set-Location $FrontendDir
    npm run dev
} -ArgumentList $FrontendDir

try {
    while ($true) {
        Receive-Job -Job $backendJob, $frontendJob
        Start-Sleep -Milliseconds 500
    }
}
finally {
    Stop-Job -Job $backendJob, $frontendJob -ErrorAction SilentlyContinue
    Remove-Job -Job $backendJob, $frontendJob -Force -ErrorAction SilentlyContinue
}
