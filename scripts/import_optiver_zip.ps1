param(
    [string]$ZipPath = "",
    [string]$OutputPath = "data/raw/optiver/train.csv"
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
    throw "The project environment is missing. Run scripts/setup_equity_and_verify.ps1 first."
}

if ([string]::IsNullOrWhiteSpace($ZipPath)) {
    $SearchRoots = @((Get-Location).Path, (Join-Path ([Environment]::GetFolderPath("UserProfile")) "Downloads"))
    $Candidates = @()
    foreach ($SearchRoot in $SearchRoots) {
        if (Test-Path -LiteralPath $SearchRoot -PathType Container) {
            $Candidates += Get-ChildItem -LiteralPath $SearchRoot -File -Filter "optiver-trading-at-the-close*.zip"
        }
    }
    $Candidates = @($Candidates | Sort-Object -Property FullName -Unique)
    if ($Candidates.Count -ne 1) {
        throw "Expected exactly one Optiver ZIP in the repository root or Downloads; found $($Candidates.Count). Pass -ZipPath explicitly."
    }
    $ZipPath = $Candidates[0].FullName
}

$ResolvedZip = (Resolve-Path -LiteralPath $ZipPath).Path
& $ProjectPython -m lob_alpha.cli equity-extract-zip `
    --zip $ResolvedZip `
    --output $OutputPath
Assert-NativeSuccess "Safe Optiver train.csv extraction"

if (-not (Test-Path -LiteralPath $OutputPath -PathType Leaf)) {
    throw "Extraction returned successfully but $OutputPath does not exist."
}
Write-Host "Licensed training data is present at $OutputPath and remains ignored by Git."

