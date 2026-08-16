# Implementation status — exact-session feasibility phase

## Completed

- Typed, validated research configuration with disjoint chronological splits.
- Pre-registered hypotheses and falsification conditions.
- Databento MBP-10 request builder and pre-download cost estimator.
- Mandatory explicit cost ceiling for actual downloads.
- Point-in-time definition download window and tick/multiplier verifier.
- DBN/CSV event loaders and canonical MBP-10 schema checks.
- Session data-quality report: ordering, duplicates, book crossing, depth ladder, nonnegative size/count and tick alignment.
- Fixed 100 ms causal decision-state sampling with quote-age filtering.
- Static book, microprice, depth, OFI, trade-flow, intensity and lagged-control features.
- Four auditable future-midpoint labels with target/source timestamps and stale-label rejection.
- Observable-depth marketable VWAP fills and exact round-trip P&L.
- Transparent ridge baseline utilities.
- Content-addressed processing manifests.
- Deterministic engineering fixture and end-to-end CLI.
- Tests for feature causality, label timing, split isolation, data quality, cost gating and execution arithmetic.
- Cost-capped daily Databento batch submission, polling, download and SHA-256 verification.
- Recursive daily-file discovery, compact per-session outputs and a hashed processed-data catalog.
- Train-only feature IC, decile diagnostics, day-cluster bootstrap and expanding-window ridge selection.
- Validation comparison against zero, microprice-only and imbalance-only baselines.
- Validation-only horizon and execution-margin selection.
- Hash-locked candidate freeze and non-overwriting, one-shot holdout command.
- Fee, latency and displayed-quantity sensitivity with one raw-file load per session.
- Generated research figures, empirical report and claim-gated CV bullet.
- PowerShell scripts for setup, the real run through freeze and the final holdout.
- A complete nine-session synthetic rehearsal of every stage.
- DST-safe planning of complete configured weekdays with exact, end-exclusive UTC bounds.
- Independent Databento metadata estimates for every intraday session and deterministic JSON plans.
- Aggregate cost rejection and an independent boolean gate before any paid session request.
- One-session-per-file compressed DBN acquisition through non-discoverable partial files and atomic renames.
- Non-overwriting resume checks tied to manifest request parameters, local paths, byte sizes and SHA-256.
- Conservative interruption handling that refuses automatic retries after a request may have incurred cost.
- Local validation and promotion of complete interrupted DBN files without another paid request.
- Serial three-session resource audit with raw bytes, decoded rows, pandas memory, wall time,
  processed rows, quality rejections, output bytes, totals and single-session maxima.
- Estimate-only-by-default PowerShell feasibility workflow with explicit free/paid step labels.

## Deliberately not claimed

- No real market data has been downloaded in this repository snapshot.
- No empirical signal or profitability result exists yet.
- The deterministic fixture is an engineering test only.
- The candidate `ESM6` window still requires provider definition/liquidity confirmation.
- No Databento estimate or paid time-series request was made while implementing this phase.
- The resource audit intentionally exposes no alpha, IC, P&L, hit rate, model selection,
  cross-validation or holdout result.

## Remaining manual/data-dependent gate

1. Configure `DATABENTO_API_KEY` locally.
2. Run `estimate-session-costs` for `configs/sample_three_sessions.yaml`; this is metadata-only.
3. If acceptable, provide both a finite cap and `-ConfirmPaidRequest` to the feasibility script.
4. Review the audit's single-session maxima before estimating or acquiring the full study.
5. Verify definition fields, generated daily quality manifests and validation outputs.
6. Freeze the candidate, review it, then explicitly acknowledge the one-shot holdout.

Databento 0.83.0 streams compressed DBN to disk and exposes chunked
`DBNStore.to_df(count=...)` iteration. The current causal feature/label pipeline still needs
one complete session DataFrame, so this implementation bounds that exposure to one session
at a time and does not claim end-to-end chunked processing.
