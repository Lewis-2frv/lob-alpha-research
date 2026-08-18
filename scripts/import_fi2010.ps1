[CmdletBinding()]
param(
    [string]$ArchivePath = [IO.Path]::Combine($env:USERPROFILE, 'Downloads', 'FI-2010-official.zip'),
    [string]$PreparedDirectory = 'data\raw\fi2010'
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
$preparedPath = if ([IO.Path]::IsPathRooted($PreparedDirectory)) {
    $PreparedDirectory
} else {
    Join-Path $repoRoot $PreparedDirectory
}

Push-Location $repoRoot
try {
    & $python -m lob_alpha.cli fi2010-import `
        --archive $ArchivePath `
        --prepared-dir $preparedPath
    if ($LASTEXITCODE -ne 0) {
        throw "FI-2010 import failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}
