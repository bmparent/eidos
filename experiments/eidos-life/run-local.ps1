$repoRoot = git rev-parse --show-toplevel
Set-Location $repoRoot
Write-Host "Serving Eidos Life v0.2 at http://localhost:5173/experiments/eidos-life/"
python -m http.server 5173
