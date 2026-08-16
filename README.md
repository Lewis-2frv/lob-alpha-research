# Short-Horizon Order-Flow Alpha in E-mini S&P 500 Futures

[![tests](https://img.shields.io/badge/tests-passing-brightgreen)](#verification)
[![status](https://img.shields.io/badge/status-pre--data%20release-blue)](#current-status)

An execution-aware research pipeline testing whether limit-order-book state and recent order flow predict 100–1,000 ms price changes strongly enough to survive observable depth, spread, fees and latency.

This is a quantitative trading research project, not a model-accuracy demo and not a claim of a profitable live HFT strategy.

## Current status

The v0.2 pre-data release is a staged, almost push-button study. It includes cost-gated daily acquisition, provider-hash verification, point-in-time contract checks, per-session validation and compact storage, train-only diagnostics, chronological model selection, validation-only execution selection, a hash-locked holdout, sensitivity grids and generated reports. The complete workflow has been rehearsed on synthetic books. No real-data performance result is claimed yet.

See [implementation status](reports/implementation_status.md) and the [frozen methodology](reports/methodology.md).

## Research chain

```text
CME Globex MBP-10 events
        -> validated observable book states
        -> microstructure features
        -> chronological forecasts
        -> confidence-gated decisions
        -> delayed book-sweep execution
        -> gross-to-net P&L and failure analysis
```

## Registered first release

- One exact E-mini S&P 500 front-month contract; initial candidate `ESM6`.
- US cash-market session, 09:35–15:55 America/New_York.
- Decisions every 100 ms using `ts_recv`.
- Forecast horizons: 100, 250, 500 and 1,000 ms.
- Queue/depth imbalance, microprice, OFI, signed trade flow and liquidity controls.
- Raw-signal baselines and regularized linear regression.
- Complete-session train, validation and untouched holdout partitions.
- Marketable entry/exit through displayed top-10 depth.
- Fee, size and 0–100 ms latency sensitivity.

Full MBO reconstruction, passive queue fills, nonlinear models, extra instruments and C++ are deferred until the empirical v0.1 release.

## Fastest path on Windows

Run the complete installation, unit suite and synthetic nine-session rehearsal:

```powershell
.\scripts\setup_and_verify.ps1
```

Then read [the short manual handoff](reports/manual_handoff.md). The only unavoidable manual actions are setting your own API key, approving a maximum data cost, reviewing the frozen candidate and releasing the one-shot holdout.

## Manual installation

```bash
python -m venv .venv
source .venv/bin/activate               # Windows: .venv\Scripts\activate
python -m pip install -e ".[data,dev]"
```

The small single-session fixture needs only the core dependencies:

```bash
python -m pip install -e .
lob-alpha config-check --config configs/base.yaml
lob-alpha run-fixture --output-dir artifacts/fixture
```

The full synthetic rehearsal exercises acquisition-independent processing, model selection, freeze controls, execution grids and report generation:

```bash
lob-alpha run-fixture-study --output-dir artifacts/fixture-study-local
```

Fixtures validate software mechanics only and are excluded from research evidence.

## Real-data gate

Keep the API key in the environment and never commit it:

```bash
export DATABENTO_API_KEY="db-..."       # PowerShell: $env:DATABENTO_API_KEY="db-..."
```

First estimate the full registered request without downloading:

```bash
lob-alpha estimate-cost --config configs/base.yaml
```

After reviewing the estimate, one command can submit a daily-split DBN batch, poll it, download it and verify provider-published SHA-256 hashes. It will abort if the fresh estimate exceeds your explicit ceiling:

```bash
lob-alpha batch-run \
  --config configs/base.yaml \
  --max-cost-usd 25.00 \
  --confirm-paid-request \
  --output-dir data/raw/databento
```

`25.00` is an example ceiling, not an expected price. The smaller `configs/sample_three_sessions.yaml` stream workflow remains available for a low-cost audit.

Definitions are acquired separately from one exact UTC-day snapshot:

```bash
lob-alpha download-definitions \
  --config configs/sample_three_sessions.yaml \
  --output data/raw/databento/ESM6_20260316_definition.dbn.zst \
  --max-cost-usd 1.00

lob-alpha verify-definition \
  --config configs/base.yaml \
  --input data/raw/databento/ESM6_20260316_definition.dbn.zst
```

Then process, select and freeze without touching holdout outcomes:

```bash
lob-alpha process-all --config configs/base.yaml
lob-alpha train-stage --config configs/base.yaml
lob-alpha validation-stage --config configs/base.yaml
lob-alpha freeze-candidate --config configs/base.yaml
```

After reviewing the frozen candidate, explicitly release the non-overwriting holdout and build the report:

```bash
lob-alpha holdout-stage --config configs/base.yaml --acknowledge-one-shot
lob-alpha build-report
```

Every processed output receives a manifest containing input/config hashes and the quality report. The daily catalog, stage artifacts and frozen candidate are content-addressed. Licensed raw data and generated artifacts are ignored by Git.

## Causality and execution safeguards

- Features at `t` only use receive timestamps `<= t`.
- Labels retain exact target and source timestamps.
- Stale labels and quotes are rejected.
- Complete sessions, not rows, define data splits.
- Train-only expanding windows choose ridge regularization; holdout labels are inaccessible to selection.
- Train-fitted decile edges are applied unchanged to validation.
- Costs are estimated before any provider download and actual downloads require a user-supplied USD cap.
- Paid batch submission also requires a separate confirmation flag.
- Marketable fills consume displayed depth and reject unavailable quantity.
- P&L uses executable entry and exit VWAPs; midpoint is used only as a prediction label/markout.
- Spread is embedded in fills and cannot be subtracted twice.
- The frozen candidate binds configuration, processed catalog and train selection by SHA-256.
- Holdout execution refuses any nonempty output directory.

## Verification

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

The suite covers configuration/split invariants, malformed books, tick alignment, formula checks, future-event perturbation, label timing, truncated data, dual paid-request gates, batch hashes, daily-file uniqueness, chronological folds, freeze immutability, claim-gated reporting, multi-level fills, insufficient depth and end-to-end feature/model construction.

## Repository map

```text
configs/                 frozen experiment inputs and hypotheses
data/manifests/          request/run provenance; no licensed data
src/lob_alpha/           ingestion, validation, research and execution logic
tests/                   deterministic causality and financial tests
reports/                 methodology, status and later empirical outputs
scripts/                 PowerShell setup, real-run and holdout entry points
```

## Data references

- [Databento MBP-10 schema](https://databento.com/docs/schemas-and-data-formats/mbp-10)
- [Databento instrument definitions](https://databento.com/docs/schemas-and-data-formats/instrument-definitions)
- [Databento Historical API](https://databento.com/docs/api-reference-historical/client)
- [CME E-mini S&P 500 contract page](https://www.cmegroup.com/markets/equities/sp/e-mini-sandp500.contractSpecs.html)

## Honest limitations

The planned study covers one contract, one venue and a limited period. `ts_recv` is a provider capture timestamp, not this strategy's measured live feed latency. Displayed-depth execution does not model the strategy's own market impact. Participant fees vary. Passive queue behavior is outside v0.1. These limitations will remain visible even if the holdout result is positive.
