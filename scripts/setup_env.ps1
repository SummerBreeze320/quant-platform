# Quant Platform Environment Setup (PowerShell)
# Run: .\scripts\setup_env.ps1

Write-Host "=== Quant Platform Environment Setup ===" -ForegroundColor Cyan

$venvPath = ".\venv"
if (-not (Test-Path $venvPath)) {
    Write-Host "Creating virtual environment..." -ForegroundColor Yellow
    python -m venv $venvPath
}

Write-Host "Activating venv..." -ForegroundColor Yellow
& "$venvPath\Scripts\Activate.ps1"

Write-Host "Installing dependencies..." -ForegroundColor Yellow
pip install -r requirements.txt

$dataDirs = @("data\raw", "data\qlib_bin", "data\cache", "data\reports", "data\rd_agent_workspace", "data\rd_agent_results", "logs")
foreach ($dir in $dataDirs) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
        Write-Host "Created: $dir" -ForegroundColor Green
    }
}

Write-Host "Checking Wind terminal..." -ForegroundColor Yellow
$windPath = "C:\Program Files\Wind Terminal\WindPy\WindPy.dll"
if (Test-Path $windPath) {
    Write-Host "Wind terminal found." -ForegroundColor Green
} else {
    Write-Host "Warning: Wind terminal not found at default path." -ForegroundColor Red
}

Write-Host "=== Setup complete ===" -ForegroundColor Cyan
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "  1. Edit config/settings.yaml with your MySQL password"
Write-Host "  2. Edit config/rd_agent_config.yaml with your LLM API key"
Write-Host "  3. Run: python -m src init-data --start 2015-01-01 --end 2026-08-01"
