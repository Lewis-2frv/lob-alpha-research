param(
    [string]$Python = "python",
    [string]$FixtureOutput = "artifacts/equity-fixture-local"
)

$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $true

function Assert-NativeSuccess {
    param([string]$Step)
    if ($LASTEXITCODE -ne 0) {
        throw "$Step failed with exit code $LASTEXITCODE."
    }
}

if (-not (Test-Path -LiteralPath ".venv\Scripts\python.exe" -PathType Leaf)) {
    & $Python -m venv .venv
    Assert-NativeSuccess "Virtual-environment creation"
}
$ProjectPython = Join-Path ".venv" "Scripts/python.exe"
& $ProjectPython -m pip install --upgrade pip
Assert-NativeSuccess "pip upgrade"
& $ProjectPython -m pip install -e ".[data,equity,dev]"
Assert-NativeSuccess "Equity research dependency installation"
& $ProjectPython -m ruff check .
Assert-NativeSuccess "Ruff"
& $ProjectPython -m unittest discover -s tests -v
Assert-NativeSuccess "Unit tests"

if (Test-Path -LiteralPath $FixtureOutput) {
    $FixtureOutput = "$FixtureOutput-$(Get-Date -Format 'yyyyMMdd-HHmmss')"
}
& $ProjectPython -m lob_alpha.cli equity-run-synthetic `
    --config configs/equity_close_fixture.yaml `
    --output-dir $FixtureOutput
Assert-NativeSuccess "Complete synthetic equity study"

Write-Host "Equity setup and synthetic verification passed."
Write-Host "Synthetic results prove mechanics only and must never be cited as market evidence."

