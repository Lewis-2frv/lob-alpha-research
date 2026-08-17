# Equity Closing-Auction Alpha Research

[![tests](https://img.shields.io/badge/tests-passing-brightgreen)](#verification)
[![status](https://img.shields.io/badge/status-pre--data%20release-blue)](#current-status)

This repository tests whether contemporaneous order-book liquidity, imbalance and closing-auction
state predict Optiver's supplied 60-second synthetic-index-relative target. It separately tests
whether those predictions can rank executable stock quote returns strongly enough to support a
spread- and fee-aware cross-sectional long/short simulation.

The primary public project uses the free Kaggle **Optiver - Trading at the Close** dataset. This is
trading research rather than a generic competition-accuracy exercise: predictive MAE and IC are
compared with executable quote-crossing returns, displayed-liquidity capacity, spread cost and
fee sensitivity. No strategy is described as profitable without an untouched real-data holdout.

## Current status

Version 0.3 is a pre-data engineering release. The complete workflow has been rehearsed on
unmistakably synthetic data, but synthetic numbers are claim-gated and are not market evidence.
Lewis must manually accept Kaggle's competition rules and data licence before using the real data.

The earlier CME E-mini/Databento pipeline remains available as an optional provider and historical
engineering path. It is documented separately below and does not run in the default equity flow.

## Registered equity study

- Licensed input: `data/raw/optiver/train.csv`, never committed or redistributed.
- Ten-second closing-auction observations and the supplied 60-second, index-relative competition
  target in basis points; the project does not reconstruct or relabel it as a raw stock return.
- Complete chronological date blocks: train `date_id` 0-329, validation 330-404, holdout 405-480.
- Target-blind metadata registration before preparation; the observed date range must match exactly.
- Order-book, auction-state, within-stock/date causal dynamics and current-`time_id`
  cross-sectional features.
- `stock_id` represented as a categorical fixed effect, never an ordinal economic quantity.
- Mandatory zero and signed-imbalance baselines, ridge, scikit-learn histogram gradient boosting,
  and optional LightGBM. Linear/LightGBM stock effects are sparse one-hot columns; histogram
  gradient boosting uses one native categorical column rather than a dense stock matrix.
- Expanding-window train-only CV with deterministic per-date tuning samples and MAE as the primary
  selection metric.
- Validation-only model and trading-rule selection followed by a content-addressed development
  refit and candidate freeze.
- Explicitly acknowledged, non-overwriting one-shot holdout with hash checks for raw data and
  metadata, prepared partitions, configuration, feature specification, fitted preprocessing and
  model.

The full design is in [`configs/equity_close.yaml`](configs/equity_close.yaml).

## Causal and execution contract

Features at a row may use only that row and earlier observations from the same `(stock_id,
date_id)`. Cross-sectional ranks and robust z-scores use only stocks sharing the current `time_id`.
`target`, future WAP, future quotes and holdout aggregates are excluded from every model feature.
Missing `near_price` and `far_price` values are preserved, imputed from development data only, and
paired with explicit missingness indicators. Rare missing supplied targets are also preserved so
their rows can contribute causal history and executable quotes, but they are excluded from every
supervised fit and predictive metric and are reported as coverage gaps.

At non-overlapping decision times, the simulator requires both positive long candidates and
negative short candidates within the same `time_id`, allocates equal gross exposure to both sides,
enters longs at the ask and shorts at the bid, and reverses at exactly aligned quotes 60 seconds
later. Invalid, missing or crossed quotes are rejected. Spread is embedded through executable
prices exactly once; separately reported fees are applied per side. These quote returns are not the
supplied predictive target. Returns are basis points on anonymised normalized prices, not invented
dollar P&L.

This is an auction-period quote-crossing simulation. It does not claim millisecond latency, queue
position, passive fills, consolidated NBBO or live deployability.

## Setup and synthetic verification

On Windows:

```powershell
.\scripts\setup_equity_and_verify.ps1
```

This installs the equity research extras, runs Ruff and all unit tests, then exercises audit,
preparation, train CV, validation selection, freeze, mechanical holdout and claim-gated reporting
on generated data.

To run only the synthetic study with an existing environment:

```powershell
.\.venv\Scripts\python.exe -m lob_alpha.cli equity-run-synthetic `
  --config configs/equity_close_fixture.yaml `
  --output-dir artifacts/equity-fixture-local
```

## Licensed data handoff

Follow [`reports/equity_data_handoff.md`](reports/equity_data_handoff.md). Manual browser download
is the default; Kaggle API credentials are neither needed nor requested.

After manually downloading the competition ZIP:

```powershell
.\scripts\import_optiver_zip.ps1 `
  -ZipPath "C:\path\to\optiver-trading-at-the-close.zip"
```

The extractor permits exactly one safe `train.csv`, rejects zip-slip paths, links, Windows-special
names and case-colliding destinations, enforces member/size/compression-ratio limits, extracts only
the expected file through a partial path, and refuses overwrite.

## Real-data stages

Audit identifiers without reading target values, then perform the full schema/finiteness audit and
memory-bounded per-date preparation:

```powershell
.\scripts\audit_prepare_equity.ps1
```

The exact target-blind registration command is:

```powershell
.\.venv\Scripts\python.exe -m lob_alpha.cli equity-audit `
  --config configs/equity_close.yaml `
  --input data/raw/optiver/train.csv `
  --output data/interim/optiver_metadata_registration.json `
  --metadata-only
```

Train through validation and freeze without reading holdout rows:

```powershell
.\scripts\run_equity_through_freeze.ps1
```

That script produces a pre-holdout report and stops. Only after independent review of the code,
audit, selection artifacts, assumptions and frozen hashes should the one-shot holdout be considered:

```powershell
.\scripts\run_equity_holdout_report.ps1 `
  -HoldoutAcknowledgement "RELEASE OPTIVER HOLDOUT ONCE"
```

The holdout command writes the stable `data/processed/optiver/HOLDOUT_STARTED.json` seal before it
opens the holdout manifest or any holdout partition. The exact phrase is case-sensitive. A crash
therefore leaves the prepared study sealed, and changing the output directory, report path, model
name or CLI invocation cannot release it again. A successful run also anchors the completion JSON
beside that seal. If a sealed run crashes, preserve the seal and partial output for review; do not
delete them or retry. Recovery means documenting the failed release and, only after an independent
decision, registering and preparing a genuinely new study—not redirecting the same holdout.

## Equity CLI

```text
equity-audit           schema/data audit; --metadata-only excludes target reads
equity-prepare         per-date causal Parquet preparation
equity-train           train-only diagnostics and expanding-window CV
equity-validate        validation model and trading-rule selection
equity-freeze          content-addressed candidate freeze
equity-holdout         explicitly acknowledged one-shot holdout
equity-report          Markdown report, figures and claim-gated CV evidence
equity-extract-zip     safe manual ZIP extraction
equity-run-synthetic   complete deterministic engineering rehearsal
```

## Evidence and reporting

Validation and holdout scoring read one date partition at a time and retain only narrow scored
tables; tuning and refit inputs have deterministic per-date row ceilings. The generated report
covers the research question, registered split, causal features, baselines,
model comparison, validation, feature ablation, daily IC, prediction deciles, stability regimes,
trading-cost frontier, capacity and falsification limitations.

Before a real holdout, `cv_evidence.md` contains only an engineering/methodology bullet. Synthetic
and validation-only values never enter it. After a real holdout it may include exact measured
numbers. A negative net result generates an evidence-focused falsification bullet rather than being
hidden or relabelled as profitable.

## Optional CME/Databento path

The v0.2 E-mini S&P 500 MBP-10 implementation remains under `configs/base.yaml` and the existing
futures CLI commands. It retains DST-safe exact-session requests, pre-download estimates, a finite
aggregate USD cap, independent paid-request confirmation, `.partial` recovery, manifests and
SHA-256 checks.

The safe estimate-first sample workflow remains:

```powershell
.\scripts\run_three_session_feasibility.ps1
```

No Databento request is part of the equity workflow. See
[`reports/manual_handoff.md`](reports/manual_handoff.md) and
[`reports/methodology.md`](reports/methodology.md) for the optional futures study.

## Data and licence safety

Kaggle/Optiver rows, raw ZIPs, prepared Parquet, fitted models and generated experiment artifacts
are ignored by Git. Do not redistribute competition data. Only small, licence-safe summaries may
be considered for version control after Lewis has reviewed the real study.

## Verification

```powershell
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
git diff --check
```

The deterministic suite covers schemas, chronological isolation, causal perturbations,
cross-sectional boundaries, train-only preprocessing, expanding folds, baselines, exact future
quote alignment, execution arithmetic, spread attribution, non-overlap, date-cluster bootstraps,
hash locking, one-shot holdout controls, synthetic claim gating, malicious ZIP paths and both full
synthetic research workflows.

## Repository map

```text
configs/                 frozen equity and optional futures registrations
data/                    ignored licensed inputs and prepared partitions
src/lob_alpha/           validation, causal research, execution and reporting logic
tests/                   deterministic leakage, financial and workflow tests
reports/                 handoffs, methodology and licence-safe report templates
scripts/                 Windows setup, import, preparation, freeze and holdout stages
```

## Limitations

The Optiver panel is competition data with anonymised normalized prices and a particular auction
sampling design. Cross-sectional dependence, regime change, normalized displayed size, missing
auction prices and selection across a small validation grid constrain interpretation. The pipeline
does not establish live fill quality or post-competition external validity. These limitations remain
visible regardless of the final result.
