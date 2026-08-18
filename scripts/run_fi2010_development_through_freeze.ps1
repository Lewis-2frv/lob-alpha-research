[CmdletBinding()]
param(
    [string]$PreparedDirectory = 'data\raw\fi2010',
    [string]$ArtifactDirectory = 'artifacts\fi2010-v060'
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

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
$developmentPath = Join-Path $artifactPath 'development'
$freezePath = Join-Path $artifactPath 'freeze'
$resultsPath = Join-Path $developmentPath 'development_results.json'

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
        '-m', 'lob_alpha.cli', 'fi2010-audit',
        '--prepared-dir', $preparedPath
    )
    Invoke-RepoPython -Arguments @(
        '-m', 'lob_alpha.cli', 'fi2010-develop',
        '--prepared-dir', $preparedPath,
        '--output-dir', $developmentPath
    )
    Invoke-RepoPython -Arguments @(
        '-m', 'lob_alpha.cli', 'fi2010-freeze',
        '--prepared-dir', $preparedPath,
        '--development-results', $resultsPath,
        '--output-dir', $freezePath
    )
}
finally {
    Pop-Location
}
