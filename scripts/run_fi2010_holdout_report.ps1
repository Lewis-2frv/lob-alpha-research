[CmdletBinding()]
param(
    [Parameter(Mandatory)]
    [ValidateNotNullOrEmpty()]
    [string]$Acknowledgement,
    [string]$PreparedDirectory = 'data\raw\fi2010',
    [string]$ArtifactDirectory = 'artifacts\fi2010-v060'
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$requiredPhrase = 'RELEASE FI2010 CF9 HOLDOUT ONCE'
if ($Acknowledgement -cne $requiredPhrase) {
    throw "Acknowledgement must exactly equal: $requiredPhrase"
}

$repoRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repoRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Repository virtual environment not found: $python"
}
$preparedPath = if ([IO.Path]::IsPathRooted($PreparedDirectory)) {
    $PreparedDirectory
} else {
    Join-Path $repoRoot $PreparedDirectory
}
$artifactPath = if ([IO.Path]::IsPathRooted($ArtifactDirectory)) {
    $ArtifactDirectory
} else {
    Join-Path $repoRoot $ArtifactDirectory
}
$developmentPath = Join-Path $artifactPath 'development\development_results.json'
$freezePath = Join-Path $artifactPath 'freeze'
$holdoutPath = Join-Path $artifactPath 'holdout'
# Keep the pre-release development report immutable. The final holdout report must
# use a fresh directory so consuming CF_9 cannot be followed by an avoidable
# overwrite refusal merely because a development-only report already exists.
$reportPath = Join-Path $artifactPath 'report-final'

function Invoke-RepoPython {
    param([Parameter(Mandatory)][string[]]$Arguments)
    & $python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed with exit code ${LASTEXITCODE}: $($Arguments -join ' ')"
    }
}

Push-Location $repoRoot
try {
    Invoke-RepoPython -Arguments @(
        '-m', 'lob_alpha.cli', 'fi2010-holdout',
        '--prepared-dir', $preparedPath,
        '--frozen-candidate', (Join-Path $freezePath 'frozen_candidate.json'),
        '--final-model-manifest', (Join-Path $freezePath 'final_model_manifest.json'),
        '--output-dir', $holdoutPath,
        '--acknowledgement', $Acknowledgement
    )
    Invoke-RepoPython -Arguments @(
        '-m', 'lob_alpha.cli', 'fi2010-report',
        '--prepared-dir', $preparedPath,
        '--development-results', $developmentPath,
        '--freeze-dir', $freezePath,
        '--holdout-dir', $holdoutPath,
        '--output-dir', $reportPath
    )
    try {
        Invoke-RepoPython -Arguments @(
            '-m', 'lob_alpha.cli', 'fi2010-publish',
            '--report-dir', $reportPath,
            '--repository-root', $repoRoot,
            '--output-dir', (Join-Path $repoRoot 'docs\results\fi2010')
        )
    }
    catch {
        Write-Warning "CF_9 and the final evidence report completed, but GitHub publication failed: $($_.Exception.Message)"
        Write-Warning "Rerun scripts\publish_fi2010_portfolio.ps1; do NOT rerun the holdout."
    }
}
finally {
    Pop-Location
}
