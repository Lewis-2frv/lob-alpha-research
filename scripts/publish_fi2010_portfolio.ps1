[CmdletBinding()]
param(
    [string]$ReportDirectory = 'artifacts\fi2010\report-final',
    [switch]$DevelopmentOnly
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$repoRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repoRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Repository virtual environment not found: $python"
}
$reportPath = if ([IO.Path]::IsPathRooted($ReportDirectory)) {
    $ReportDirectory
} else {
    Join-Path $repoRoot $ReportDirectory
}

$arguments = @(
    '-m', 'lob_alpha.cli', 'fi2010-publish',
    '--report-dir', $reportPath,
    '--repository-root', $repoRoot,
    '--output-dir', (Join-Path $repoRoot 'docs\results\fi2010')
)
if ($DevelopmentOnly) {
    $arguments += '--allow-development-only'
}

Push-Location $repoRoot
try {
    & $python @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "FI-2010 portfolio publication failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}
