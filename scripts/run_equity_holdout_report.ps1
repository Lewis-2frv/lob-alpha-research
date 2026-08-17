param(
    [string]$RunRoot = "artifacts/equity-real-run",
    [string]$HoldoutAcknowledgement = ""
)

$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $true

function Assert-NativeSuccess {
    param([string]$Step)
    if ($LASTEXITCODE -ne 0) {
        throw "$Step failed with exit code $LASTEXITCODE."
    }
}

$RequiredAcknowledgement = "RELEASE OPTIVER HOLDOUT ONCE"
if ($HoldoutAcknowledgement -cne $RequiredAcknowledgement) {
    throw "Review the frozen candidate, then pass -HoldoutAcknowledgement '$RequiredAcknowledgement' exactly once."
}
$ProjectPython = Join-Path ".venv" "Scripts/python.exe"
$TrainDir = Join-Path $RunRoot "train"
$ValidationDir = Join-Path $RunRoot "validation"
$HoldoutDir = Join-Path $RunRoot "holdout"
$ReportsDir = Join-Path $RunRoot "reports"
$FrozenPath = Join-Path $RunRoot "frozen_candidate.json"

& $ProjectPython -m lob_alpha.cli equity-holdout `
    --config configs/equity_close.yaml `
    --manifest data/processed/optiver/prepared_manifest.json `
    --frozen-candidate $FrozenPath `
    --output-dir $HoldoutDir `
    --acknowledge-one-shot $HoldoutAcknowledgement
Assert-NativeSuccess "One-shot untouched equity holdout"

& $ProjectPython -m lob_alpha.cli equity-report `
    --train-dir $TrainDir `
    --validation-dir $ValidationDir `
    --holdout-dir $HoldoutDir `
    --reports-dir $ReportsDir
Assert-NativeSuccess "Final claim-gated report"

Write-Host "One-shot holdout complete. Review $ReportsDir before using any result."
