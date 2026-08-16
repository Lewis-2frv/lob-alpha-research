# Frozen v0.2 methodology

## Research question

Do contemporaneously observable order-book state and recent order flow predict 100–1,000 ms E-mini S&P 500 midpoint changes, and does any relationship survive chronological out-of-sample testing and executable marketable fills?

## Information clock

`ts_recv` is the primary observable clock. At decision time `t`, the state is the final complete MBP-10 event with `ts_recv <= t`. Trailing features include only events in `(t-lookback, t]`. Labels use the final event at or before the exact target time and retain that source timestamp for audit.

## Data partition

- Train: 16 March–24 April 2026.
- Validation: 27 April–8 May 2026.
- Final holdout: 11 May–5 June 2026.

All partitions are complete sessions. Feature state resets daily and labels/positions may not cross the session boundary. No row shuffling is permitted.

The contract/date proposal must be verified against point-in-time instrument definitions and liquidity before any outcome is inspected. If it is invalid, the whole contiguous window moves; contracts are not stitched in v0.1.

## Feature families

The registered families are level-one queue imbalance, weighted depth imbalance, microprice displacement, normalized OFI, signed aggressive trade imbalance, spread, depth, event intensity, lagged midpoint movement and short-window realised volatility. Every variable has a signed economic rationale and a causality test.

## Models

Zero forecast, microprice-only, queue-imbalance-only and all-feature ridge regressions are mandatory. Model scaling and regularization are fitted on training data only. Ridge alpha is chosen with contiguous expanding day blocks. Fit rows are deterministically thinned per session to bound memory and compute, while final validation and holdout metrics use every valid decision row. LightGBM is excluded from the CV-ready milestone.

## Execution

The prediction label is future midpoint movement in ticks. Strategy P&L is different: marketable orders sweep the displayed book at simulated arrival and again at liquidation. At most one position is open. There is no midpoint execution, invented depth or passive fill assumption.

Primary sensitivity grids are registered in `configs/base.yaml`. The $2.50 per-contract-per-side scenario is a deliberately conservative all-in parameter, not a statement that every participant pays that fee.

## Selection and holdout

Training uses expanding day blocks. Validation selects the registered horizon and safety margin. Before test, `freeze-candidate` writes a content-addressed `frozen_candidate.json` tying the choice to the exact configuration, processed catalog and train-selection artifact. `holdout-stage` refuses changed hashes and any nonempty output directory. Holdout outcomes cannot change the strategy. A genuine code defect requires written invalidation and a complete rerun.

## Inference

Rows are overlapping and not treated as independent observations. IC, markouts and P&L are calculated per day and uncertainty is estimated with trading-day block bootstrap and leave-one-day-out sensitivity.

## Success criterion

Success is a reproducible, causal and economically honest conclusion. Positive net P&L is not required. A stable predictive signal that fails the execution hurdle is a valid result.
