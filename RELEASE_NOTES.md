# v0.3 equity closing-auction pre-data release

The primary public workflow now studies Optiver's supplied 60-second synthetic-index-relative
target, separately from executable stock quote returns, using the licensed Kaggle dataset. It adds
target-blind metadata registration, bounded per-date
Parquet preparation, causal within-stock and current-time cross-sectional features, mandatory
baselines, ridge/nonlinear chronological CV, executable cross-sectional quote simulation,
content-addressed candidate freezing, a prepared-study-wide one-shot seal, claim-gated reporting,
bounded safe manual ZIP
extraction and a complete deterministic synthetic rehearsal.

The v0.2 CME/Databento pipeline remains available as an optional engineering/provider path with its
paid-request safety controls unchanged.

No Kaggle data or real-data performance claim is included.

## Earlier v0.2 release

This release moves the project from a tested feature/execution foundation to a staged empirical study that is ready for credentials and licensed market data.

Key additions:

- one-command, cost-capped daily Databento batch acquisition with an independent paid-request confirmation;
- provider SHA-256 download verification and content-addressed daily processing catalogs;
- compact daily datasets suitable for a multi-million-row 100 ms study;
- full train-only feature IC, day-block bootstrap, decile analysis and expanding-window ridge selection;
- validation comparison with zero, microprice-only and imbalance-only baselines;
- validation-only horizon and execution-margin selection;
- hash-locked candidate freezing and non-overwriting holdout execution;
- one-pass-per-day fee, latency and quantity robustness grids;
- generated figures, empirical report and claim-gated CV wording;
- PowerShell entry points and a successful full synthetic rehearsal.

No market-performance claim is included. The remaining work requires the user's Databento key, explicit cost approval and review of real-data quality/selection artifacts.
