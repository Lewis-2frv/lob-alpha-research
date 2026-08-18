[CmdletBinding()]
param(
    [string]$ArchivePath = [IO.Path]::Combine(
        $env:USERPROFILE,
        'Downloads',
        'FI-2010-official.zip'
    ),
    [string]$PreparedDirectory = 'data\raw\fi2010',
    [string]$ArtifactDirectory = 'artifacts\fi2010-v060',
    [switch]$InstallDependencies
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
}
else {
    Join-Path $repoRoot $PreparedDirectory
}

$artifactPath = if ([IO.Path]::IsPathRooted($ArtifactDirectory)) {
    $ArtifactDirectory
}
else {
    Join-Path $repoRoot $ArtifactDirectory
}

if (Test-Path -LiteralPath $artifactPath) {
    $existingItem = Get-ChildItem -LiteralPath $artifactPath -Force |
        Select-Object -First 1

    if ($null -ne $existingItem) {
        throw "Fresh artifact directory required: $artifactPath"
    }
}

function Invoke-RepoPython {
    param(
        [Parameter(Mandatory)]
        [string[]]$Arguments
    )

    & $python @Arguments

    if ($LASTEXITCODE -ne 0) {
        throw (
            "Python command failed with exit code ${LASTEXITCODE}: " +
            ($Arguments -join ' ')
        )
    }
}

Push-Location $repoRoot

try {
    #
    # Verify environment and registered FI-2010 source.
    #
    # Use a hashtable splat here so ArchivePath and InstallDependencies
    # are passed as named PowerShell parameters.
    #
    $setupArguments = @{
        ArchivePath = $ArchivePath
    }

    if ($InstallDependencies) {
        $setupArguments['InstallDependencies'] = $true
    }

    $setupScript = Join-Path $PSScriptRoot 'setup_fi2010_and_verify.ps1'
    & $setupScript @setupArguments

    #
    # Static/code-quality verification.
    #
    Invoke-RepoPython -Arguments @(
        '-m',
        'ruff',
        'check',
        '.'
    )

    & git diff --check

    if ($LASTEXITCODE -ne 0) {
        throw "git diff --check failed with exit code $LASTEXITCODE"
    }

    #
    # Import is idempotent for the exact verified source.
    #
    # Existing prepared data is reused after its size/hash manifest is
    # revalidated, so the 1.83 GB archive does not need to be downloaded
    # or expanded again for v0.6.
    #
    $importScript = Join-Path $PSScriptRoot 'import_fi2010.ps1'

    & $importScript `
        -ArchivePath $ArchivePath `
        -PreparedDirectory $preparedPath

    #
    # Run the complete target-blind CF_1-CF_8 development study,
    # select/freeze the candidate, and refit it on Train_CF_9.
    #
    # Test_CF_9 must remain unopened here.
    #
    $developmentScript = Join-Path `
        $PSScriptRoot `
        'run_fi2010_development_through_freeze.ps1'

    & $developmentScript `
        -PreparedDirectory $preparedPath `
        -ArtifactDirectory $artifactPath

    #
    # Generate the development-only report.
    #
    Invoke-RepoPython -Arguments @(
        '-m',
        'lob_alpha.cli',
        'fi2010-report',
        '--prepared-dir',
        $preparedPath,
        '--development-results',
        (Join-Path $artifactPath 'development\development_results.json'),
        '--freeze-dir',
        (Join-Path $artifactPath 'freeze'),
        '--holdout-dir',
        (Join-Path $artifactPath 'holdout'),
        '--output-dir',
        (Join-Path $artifactPath 'report-development')
    )

    Write-Host ''
    Write-Host 'PRE-HOLDOUT RELEASE CANDIDATE COMPLETE.'
    Write-Host (
        "Review: " +
        (Join-Path $artifactPath 'report-development\fi2010_evidence.md')
    )
    Write-Host 'CF_9 has not been released by this script.'
    Write-Host (
        'Commit the exact source/config state before invoking the ' +
        'one-shot holdout.'
    )
}
finally {
    Pop-Location
}