# Recruiter-facing publication workflow

Do not publish the repository as a **completed** study until the v0.6 real-data run and one-shot CF_9
release below have finished. The source tree is publication-ready; the remaining work is execution of
the frozen research protocol on the verified local FI-2010 payload.

## 1. Install the publication release

Use the supplied v0.6 ZIP as the repository working tree, or apply the accompanying patch to the
existing repository. Keep the verified `FI-2010-official.zip` in Downloads and the prepared data under
`data/raw/fi2010`; both are gitignored.

## 2. Generate the real development evidence and freeze the winner

From PowerShell in the repository root:

```powershell
.\.venv\Scripts\Activate.ps1
.\scripts\prepare_fi2010_release.ps1 `
    -InstallDependencies `
    -ArtifactDirectory 'artifacts\fi2010-v060'
```

This verifies the source identity, runs the dedicated regression suite and static checks, evaluates
the complete registered model ladder on CF_1-CF_8, selects the confidence rule, freezes the winner,
refits on `Train_CF_9`, and builds `report-development`. It does **not** open `Test_CF_9`.

Review:

```powershell
Get-Content artifacts\fi2010-v060\report-development\fi2010_evidence.md
```

Commit this exact source/configuration state before the final release.

## 3. Release CF_9 exactly once

Only after the pre-holdout report is accepted:

```powershell
.\scripts\run_fi2010_holdout_report.ps1 `
    -ArtifactDirectory 'artifacts\fi2010-v060' `
    -Acknowledgement 'RELEASE FI2010 CF9 HOLDOUT ONCE'
```

The command creates the irreversible source-side claim/seal, evaluates the frozen model once, builds
the final report, and publishes the validated small artifacts to `docs/results/fi2010/`. Never rerun
CF_9 because the retained one-shot result is part of the research design.

## 4. Final recruiter QA

Before pushing to GitHub:

```powershell
git status --short
git diff --check
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m unittest tests.test_fi2010 -v
```

Then inspect:

- `README.md` — the generated results block should show v0.6 real development + CF_9 evidence;
- `docs/results/fi2010/README.md` — charts and headline metrics;
- `docs/results/fi2010/fi2010_evidence.md` — full results and discussion;
- `docs/results/fi2010/cv_summary.md` — CV-safe bullet wording;
- `docs/results/fi2010/portfolio_metrics.json` — machine-readable evidence.

Confirm that no raw/extracted FI-2010 data, `.venv`, model binary, `artifacts/`, credentials or large
intermediate files are staged.

## 5. Suggested Git history

A clean two-commit release makes the holdout discipline visible:

1. **Pre-holdout:** `Freeze FI-2010 v0.6 research design and model ladder`
2. **Final evidence:** `Publish one-shot FI-2010 CF9 results`

Tag the completed state, for example `v1.0.0`, only after the final evidence commit.
