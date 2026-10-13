$ErrorActionPreference = "Stop"
$Python = if (Test-Path ".\.venv\Scripts\python.exe") { Resolve-Path ".\.venv\Scripts\python.exe" } else { "py" }
Write-Host "== Minesweeper Probabilistic Reasoning ==" -ForegroundColor Cyan
& $Python -m pip install -r requirements.txt
& $Python -m pytest -q
& $Python -m minesweeper_probabilistic.main demo
