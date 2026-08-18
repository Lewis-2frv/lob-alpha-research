[CmdletBinding()]
param(
    [string]$ArchivePath = [IO.Path]::Combine($env:USERPROFILE, 'Downloads', 'FI-2010-official.zip'),
    [switch]$InstallDependencies
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$repoRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repoRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Repository virtual environment not found: $python"
}
if (-not (Test-Path -LiteralPath $ArchivePath -PathType Leaf)) {
    throw "FI-2010 source archive not found: $ArchivePath"
}

function Invoke-RepoPython {
    param([Parameter(Mandatory)][string[]]$Arguments)
    & $python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed with exit code ${LASTEXITCODE}: $($Arguments -join ' ')"
    }
}

Push-Location $repoRoot
try {
    if ($InstallDependencies) {
        Invoke-RepoPython -Arguments @('-m', 'pip', 'install', '-e', "${repoRoot}[dev,fi2010]")
    }
    Invoke-RepoPython -Arguments @('-c', 'import lob_alpha; print(lob_alpha.__version__)')
    Invoke-RepoPython -Arguments @('-c', 'import lightgbm; print(lightgbm.__version__)')
    Invoke-RepoPython -Arguments @(
        '-m', 'lob_alpha.cli', 'fi2010-import',
        '--archive', $ArchivePath,
        '--verify-only'
    )
    Invoke-RepoPython -Arguments @('-m', 'unittest', 'tests.test_fi2010', '-v')
}
finally {
    Pop-Location
}

