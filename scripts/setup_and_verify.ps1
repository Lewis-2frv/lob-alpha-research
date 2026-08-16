param(
    [string]$Python = "python",
    [string]$FixtureOutput = "artifacts/fixture-study-local"
)

$ErrorActionPreference = "Stop"

& $Python -m venv .venv
$ProjectPython = Join-Path ".venv" "Scripts/python.exe"
& $ProjectPython -m pip install --upgrade pip
& $ProjectPython -m pip install -e ".[data,dev]"
& $ProjectPython -m unittest discover -s tests -v
& $ProjectPython -m lob_alpha.cli config-check --config configs/base.yaml
& $ProjectPython -m lob_alpha.cli run-fixture-study `
    --config configs/fixture_study.yaml `
    --output-dir $FixtureOutput

Write-Host "Setup, unit tests and the complete synthetic dry-run passed."
Write-Host "The fixture output proves mechanics only and is not empirical evidence."
