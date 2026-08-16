# v0.2 pre-data release

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
