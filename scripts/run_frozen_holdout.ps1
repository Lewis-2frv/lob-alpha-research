param(
    [string]$RunRoot = "artifacts/real-run",
    [switch]$AcknowledgeOneShot
)

$ErrorActionPreference = "Stop"
function Assert-NativeSuccess {
    param([string]$Step)
    if ($LASTEXITCODE -ne 0) {
        throw "$Step failed with exit code $LASTEXITCODE."
    }
}
if (-not $AcknowledgeOneShot) {
    throw "Review the frozen candidate, then pass -AcknowledgeOneShot."
}

$ProjectPython = Join-Path ".venv" "Scripts/python.exe"
$TrainDir = Join-Path $RunRoot "train"
$ValidationDir = Join-Path $RunRoot "validation"
$HoldoutDir = Join-Path $RunRoot "holdout"
$ReportsDir = Join-Path $RunRoot "reports"
$FrozenPath = Join-Path $RunRoot "frozen_candidate.json"

& $ProjectPython -m lob_alpha.cli holdout-stage `
    --config configs/base.yaml `
    --catalog data/processed/catalog.json `
    --frozen-candidate $FrozenPath `
    --raw-dir data/raw/databento `
    --output-dir $HoldoutDir `
    --acknowledge-one-shot
Assert-NativeSuccess "Frozen holdout stage"
& $ProjectPython -m lob_alpha.cli build-report `
    --train-dir $TrainDir `
    --validation-dir $ValidationDir `
    --holdout-dir $HoldoutDir `
    --reports-dir $ReportsDir
Assert-NativeSuccess "Final report"

Write-Host "Holdout complete. Review the empirical report and CV evidence under $ReportsDir."
