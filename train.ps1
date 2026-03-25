# Stop execution if any command fails
$ErrorActionPreference = "Stop"

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "   STARTING TRAINING PIPELINE" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

# Add current directory to PYTHONPATH so Python finds 'src'
$env:PYTHONPATH = "$PWD"

# Run the training script
python src/pipelines/train.py

Write-Host "==========================================" -ForegroundColor Green
Write-Host "   TRAINING COMPLETED" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Green