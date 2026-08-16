param(
    [string]$Python = "python",
    [string]$FixtureOutput = "artifacts/fixture-study-local"
)

$ErrorActionPreference = "Stop"

function Assert-NativeSuccess {
    param([string]$Step)
    if ($LASTEXITCODE -ne 0) {
        throw "$Step failed with exit code $LASTEXITCODE."
    }
}

if (Test-Path $FixtureOutput) {
    $FixtureOutput = "$FixtureOutput-$(Get-Date -Format 'yyyyMMdd-HHmmss')"
}

& $Python -m venv .venv
Assert-NativeSuccess "Virtual-environment creation"
$ProjectPython = Join-Path ".venv" "Scripts/python.exe"
& $ProjectPython -m pip install --upgrade pip
Assert-NativeSuccess "pip upgrade"
& $ProjectPython -m pip install -e ".[data,dev]"
Assert-NativeSuccess "Dependency installation"
& $ProjectPython -m unittest discover -s tests -v
Assert-NativeSuccess "Unit test suite"
& $ProjectPython -m lob_alpha.cli config-check --config configs/base.yaml
Assert-NativeSuccess "Configuration check"
& $ProjectPython -m lob_alpha.cli run-fixture-study `
    --config configs/fixture_study.yaml `
    --output-dir $FixtureOutput
Assert-NativeSuccess "Synthetic full-study rehearsal"

Write-Host "Setup, unit tests and the complete synthetic dry-run passed."
Write-Host "The fixture output proves mechanics only and is not empirical evidence."
