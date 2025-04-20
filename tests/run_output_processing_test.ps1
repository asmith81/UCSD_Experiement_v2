# PowerShell script to run output processing test
$ErrorActionPreference = "Stop"

# Get the project root directory
$projectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)

# Add src to Python path
$env:PYTHONPATH = "$projectRoot;$env:PYTHONPATH"

# Run the test
Write-Host "Running output processing test..." -ForegroundColor Cyan
python "$PSScriptRoot\test_output_processing.py"

# Check if test completed successfully
if ($LASTEXITCODE -eq 0) {
    Write-Host "Test completed successfully!" -ForegroundColor Green
} else {
    Write-Host "Test failed with exit code $LASTEXITCODE" -ForegroundColor Red
    exit $LASTEXITCODE
} 