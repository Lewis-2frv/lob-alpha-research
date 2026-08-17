# Registered equity closing-auction methodology

## Question and outcome threshold

The predictive estimand is the `target` supplied by Optiver: a 60-second, synthetic-index-relative
future return in basis points. It is consumed as supplied and is not reconstructed or described as
a raw stock return. Predictive MAE and IC refer only to that target. Economic evaluation is a
separate cross-sectional simulation whose P&L is calculated only from current and exact
+60-second stock quotes. Predictive accuracy alone is not economic success.

## Frozen chronology

The expected `date_id` range is 0-480. Complete groups are assigned before results:

- train: 0-329;
- validation: 330-404;
- holdout: 405-480.

The metadata-only audit confirms schema, identifiers and this range without reading `target`.
Training CV is expanding-window and strictly future-scored. Validation chooses from the registered
model and execution grids. The holdout cannot be opened until the chosen model, preprocessing,
features and thresholds have been content-addressed. Development commands open only the
development partition manifest. The stable prepared-data seal is created before the holdout
manifest or partitions are opened, and an exact case-sensitive acknowledgement phrase is mandatory.

## Features and preprocessing

Current order-book and auction state is combined with exact-key 10-, 30- and 60-second within-stock/date
changes, short causal rolling statistics, and ranks/robust z-scores calculated only among stocks at
the current `time_id`. Missing near/far auction prices remain as rows and receive both train-fitted
median imputation and missingness indicators. Ridge and LightGBM use sparse one-hot stock effects;
histogram gradient boosting uses a native categorical stock column. No target or future quote is a
feature. Rows with a missing supplied target remain available for causal history and quote-based
execution, but are excluded from supervised fitting and predictive metrics; their count is reported.

## Models

Zero prediction and a train-fitted signed-imbalance slope are mandatory baselines. Ridge and a small
nonlinear tabular grid are selected by supplied-target-basis-point MAE. Histogram gradient boosting
is always available through scikit-learn; LightGBM is an optional research dependency.
Deterministic per-date row selection bounds tuning and refit memory without consulting targets.
Full validation and holdout scoring load and release one date partition at a time.

## Executable evaluation

At 60-second-spaced decisions, stocks are ranked within a single timestamp. A decision requires
enough strictly positive predictions for the long sleeve and strictly negative predictions for the
short sleeve; one-sided groups are skipped. Equal gross exposure is assigned to both sides. Longs
cross the current ask and exit at the exact +60-second bid;
shorts cross the current bid and cover at the exact +60-second ask. The executable-price difference
already contains spread. Fees are then applied once per entry and exit side. Invalid quotes are
rejected and displayed sizes are reported only as capacity diagnostics.

## Claims boundary

Validation supports selection but not final claims. Synthetic fixture output proves mechanics only.
Real holdout numbers may enter generated CV evidence only when every frozen hash matches and the
artifact identifies its source as real and a stable completion anchor matches every output hash.
Synthetic numeric tables and plots are suppressed. Negative net performance is retained as a
falsification and reported with predictive and cost-decomposition evidence.
