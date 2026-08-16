param(
    [Parameter(Mandatory = $true)]
    [double]$MaxDataCostUsd,
    [double]$MaxDefinitionCostUsd = 1.00,
    [string]$RunRoot = "artifacts/real-run",
    [switch]$ConfirmPaidRequest
)

$ErrorActionPreference = "Stop"
if (-not $ConfirmPaidRequest) {
    throw "Pass -ConfirmPaidRequest after reviewing the printed Databento estimate."
}
if (-not $env:DATABENTO_API_KEY) {
    throw "DATABENTO_API_KEY is not set in this PowerShell session."
}

$ProjectPython = Join-Path ".venv" "Scripts/python.exe"
if (-not (Test-Path $ProjectPython)) {
    throw "Run scripts/setup_and_verify.ps1 first."
}
if (Test-Path $RunRoot) {
    throw "RunRoot already exists. Choose a new -RunRoot to protect prior evidence."
}

& $ProjectPython -m lob_alpha.cli config-check --config configs/base.yaml
& $ProjectPython -m lob_alpha.cli estimate-cost --config configs/base.yaml

$DefinitionPath = "data/raw/databento/ESM6_20260316_definition.dbn.zst"
if (-not (Test-Path $DefinitionPath)) {
    & $ProjectPython -m lob_alpha.cli download-definitions `
        --config configs/base.yaml `
        --output $DefinitionPath `
        --max-cost-usd $MaxDefinitionCostUsd
}
& $ProjectPython -m lob_alpha.cli verify-definition `
    --config configs/base.yaml `
    --input $DefinitionPath

$MarketDataFiles = @(
    Get-ChildItem "data/raw/databento" -Recurse -File -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -like "*.dbn.zst" -and $_.Name -notlike "*definition*" }
)
if ($MarketDataFiles.Count -eq 0) {
    & $ProjectPython -m lob_alpha.cli batch-run `
        --config configs/base.yaml `
        --max-cost-usd $MaxDataCostUsd `
        --confirm-paid-request `
        --output-dir data/raw/databento
} else {
    Write-Host "Existing market-data files found; skipping paid batch submission."
}

& $ProjectPython -m lob_alpha.cli process-all `
    --config configs/base.yaml `
    --raw-dir data/raw/databento `
    --output-dir data/processed

$TrainDir = Join-Path $RunRoot "train"
$ValidationDir = Join-Path $RunRoot "validation"
$ReportsDir = Join-Path $RunRoot "reports"
$FrozenPath = Join-Path $RunRoot "frozen_candidate.json"

& $ProjectPython -m lob_alpha.cli train-stage `
    --config configs/base.yaml `
    --catalog data/processed/catalog.json `
    --output-dir $TrainDir
& $ProjectPython -m lob_alpha.cli validation-stage `
    --config configs/base.yaml `
    --catalog data/processed/catalog.json `
    --train-selection (Join-Path $TrainDir "train_selection.json") `
    --raw-dir data/raw/databento `
    --output-dir $ValidationDir
& $ProjectPython -m lob_alpha.cli freeze-candidate `
    --config configs/base.yaml `
    --catalog data/processed/catalog.json `
    --candidate (Join-Path $ValidationDir "selected_candidate.json") `
    --output $FrozenPath
& $ProjectPython -m lob_alpha.cli build-report `
    --train-dir $TrainDir `
    --validation-dir $ValidationDir `
    --holdout-dir (Join-Path $RunRoot "holdout") `
    --reports-dir $ReportsDir

Write-Host "Candidate is frozen. Review validation outputs before the one-shot holdout."
Write-Host "Then run: .\scripts\run_frozen_holdout.ps1 -RunRoot '$RunRoot' -AcknowledgeOneShot"
