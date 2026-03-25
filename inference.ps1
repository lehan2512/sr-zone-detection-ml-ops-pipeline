# Stop execution if any command fails
$ErrorActionPreference = "Stop"

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "   STARTING INFERENCE PIPELINE" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

# Add current directory to PYTHONPATH
$env:PYTHONPATH = "$PWD"

# Run the inference script
python src/pipelines/inference.py

Write-Host "==========================================" -ForegroundColor Green
Write-Host "   INFERENCE COMPLETED" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Green