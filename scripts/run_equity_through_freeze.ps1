param(
    [string]$RunRoot = "artifacts/equity-real-run"
)

$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $true

function Assert-NativeSuccess {
    param([string]$Step)
    if ($LASTEXITCODE -ne 0) {
        throw "$Step failed with exit code $LASTEXITCODE."
    }
}

if (Test-Path -LiteralPath $RunRoot) {
    throw "Refusing to reuse existing run root: $RunRoot"
}
$ProjectPython = Join-Path ".venv" "Scripts/python.exe"
$Manifest = "data/processed/optiver/prepared_manifest.json"
$TrainDir = Join-Path $RunRoot "train"
$ValidationDir = Join-Path $RunRoot "validation"
$FrozenPath = Join-Path $RunRoot "frozen_candidate.json"
$ReportsDir = Join-Path $RunRoot "reports"

& $ProjectPython -m lob_alpha.cli equity-train `
    --config configs/equity_close.yaml `
    --manifest $Manifest `
    --output-dir $TrainDir
Assert-NativeSuccess "Train-only expanding-window CV"

& $ProjectPython -m lob_alpha.cli equity-validate `
    --config configs/equity_close.yaml `
    --manifest $Manifest `
    --train-selection (Join-Path $TrainDir "train_selection.json") `
    --output-dir $ValidationDir
Assert-NativeSuccess "Validation model and trading selection"

& $ProjectPython -m lob_alpha.cli equity-freeze `
    --config configs/equity_close.yaml `
    --manifest $Manifest `
    --candidate (Join-Path $ValidationDir "selected_candidate.json") `
    --output $FrozenPath
Assert-NativeSuccess "Content-addressed candidate freeze"

& $ProjectPython -m lob_alpha.cli equity-report `
    --train-dir $TrainDir `
    --validation-dir $ValidationDir `
    --holdout-dir (Join-Path $RunRoot "holdout") `
    --reports-dir $ReportsDir
Assert-NativeSuccess "Pre-holdout claim-gated report"

Write-Host "Candidate frozen at $FrozenPath. The holdout has not been read."

