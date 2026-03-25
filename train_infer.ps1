# Stop execution if any command fails
$ErrorActionPreference = "Stop"

Write-Host "##########################################" -ForegroundColor Yellow
Write-Host "   INITIATING FULL CYCLE" -ForegroundColor Yellow
Write-Host "##########################################" -ForegroundColor Yellow

# 1. Run Training
Write-Host "[Orchestrator] Triggering Training..." -ForegroundColor Magenta
.\train.ps1

# 2. Run Inference
Write-Host "[Orchestrator] Training finished. Triggering Inference..." -ForegroundColor Magenta
.\inference.ps1

Write-Host "##########################################" -ForegroundColor Green
Write-Host "   FULL CYCLE COMPLETED" -ForegroundColor Green
Write-Host "##########################################" -ForegroundColor Green