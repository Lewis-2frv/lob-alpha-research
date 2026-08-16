# Implementation status — v0.2 pre-data release

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

## Deliberately not claimed

- No real market data has been downloaded in this repository snapshot.
- No empirical signal or profitability result exists yet.
- The deterministic fixture is an engineering test only.
- The candidate `ESM6` window still requires provider definition/liquidity confirmation.

## Remaining manual/data-dependent gate

1. Configure `DATABENTO_API_KEY` locally.
2. Estimate the three-session MBP-10 and one-day definition requests.
3. Run the three-session audit if the full estimate is uncomfortable; otherwise use the
   cost-capped daily batch runner.
4. Verify definition fields, generated daily quality manifests and validation outputs.
5. Freeze the candidate, review it, then explicitly acknowledge the one-shot holdout.
