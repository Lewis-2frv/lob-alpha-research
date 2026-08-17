param(
    [string]$InputPath = "data/raw/optiver/train.csv"
)

$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $true

function Assert-NativeSuccess {
    param([string]$Step)
    if ($LASTEXITCODE -ne 0) {
        throw "$Step failed with exit code $LASTEXITCODE."
    }
}

$ProjectPython = Join-Path ".venv" "Scripts/python.exe"
if (-not (Test-Path -LiteralPath $InputPath -PathType Leaf)) {
    throw "Missing $InputPath. Follow reports/equity_data_handoff.md first."
}

& $ProjectPython -m lob_alpha.cli equity-audit `
    --config configs/equity_close.yaml `
    --input $InputPath `
    --output data/interim/optiver_metadata_registration.json `
    --metadata-only
Assert-NativeSuccess "Target-blind metadata registration audit"

& $ProjectPython -m lob_alpha.cli equity-audit `
    --config configs/equity_close.yaml `
    --input $InputPath `
    --output data/interim/optiver_schema_audit.json
Assert-NativeSuccess "Full schema and target-finiteness audit"

& $ProjectPython -m lob_alpha.cli equity-prepare `
    --config configs/equity_close.yaml `
    --input $InputPath `
    --audit data/interim/optiver_schema_audit.json `
    --output-dir data/processed/optiver
Assert-NativeSuccess "Memory-bounded per-date Parquet preparation"

Write-Host "Audit and preparation complete. No model or performance metric was fitted."

