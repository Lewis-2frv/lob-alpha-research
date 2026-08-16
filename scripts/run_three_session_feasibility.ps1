param(
    [Nullable[double]]$MaxDataCostUsd = $null,
    [switch]$ConfirmPaidRequest,
    [string]$RawDir = "data/raw/databento",
    [string]$ArtifactDir = "artifacts/feasibility"
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
if (-not (Test-Path -LiteralPath $ProjectPython -PathType Leaf)) {
    throw "The project .venv is missing. Run scripts/setup_and_verify.ps1 first."
}
if (-not $env:DATABENTO_API_KEY) {
    throw "DATABENTO_API_KEY is not set in this PowerShell session."
}

$PlanPath = Join-Path $ArtifactDir "session_cost_plan.json"
$AuditJson = Join-Path $ArtifactDir "resource_audit.json"
$AuditMarkdown = Join-Path $ArtifactDir "resource_audit.md"
$ProcessedDir = Join-Path $ArtifactDir "processed"
$AcquisitionManifest = "data/manifests/sample_three_session_acquisition.json"

Write-Host "STEP 1 - FREE METADATA ESTIMATE (no time-series data request)" -ForegroundColor Green
& $ProjectPython -m lob_alpha.cli estimate-session-costs `
    --config configs/sample_three_sessions.yaml `
    --output $PlanPath
Assert-NativeSuccess "Free three-session cost estimation"

if ($ConfirmPaidRequest) {
    if (
        $null -eq $MaxDataCostUsd -or
        [double]::IsNaN($MaxDataCostUsd) -or
        [double]::IsInfinity($MaxDataCostUsd) -or
        $MaxDataCostUsd -lt 0
    ) {
        throw "A finite, nonnegative -MaxDataCostUsd is required for paid downloading."
    }
    Write-Host "STEP 2 - PAID DATABENTO REQUESTS MAY OCCUR" -ForegroundColor Yellow
    Write-Host "The command will estimate every missing session and abort before downloading if their aggregate exceeds the cap."
    & $ProjectPython -m lob_alpha.cli download-sessions `
        --config configs/sample_three_sessions.yaml `
        --output-dir $RawDir `
        --manifest $AcquisitionManifest `
        --max-cost-usd $MaxDataCostUsd `
        --confirm-paid-request
    Assert-NativeSuccess "Cost-gated three-session acquisition"
} else {
    Write-Host "STEP 2 - PAID DOWNLOAD SKIPPED" -ForegroundColor Green
    Write-Host "No -ConfirmPaidRequest switch was supplied, so this script cannot request time-series data."
}

Write-Host "STEP 3 - LOCAL ENGINEERING RESOURCE AUDIT (no Databento request)" -ForegroundColor Green
& $ProjectPython -m lob_alpha.cli audit-session-resources `
    --config configs/sample_three_sessions.yaml `
    --raw-dir $RawDir `
    --processed-dir $ProcessedDir `
    --output-json $AuditJson `
    --output-markdown $AuditMarkdown `
    --overwrite
Assert-NativeSuccess "Three-session local resource audit"

Write-Host "Feasibility phase complete. Review $PlanPath and $AuditMarkdown before planning the full study."
