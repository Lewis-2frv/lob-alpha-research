# Implementation status — FI-2010 v0.6.0


## v0.6 portfolio completion

The pre-holdout implementation now includes an interpretable-to-nonlinear development ladder: naive
baselines, a fixed liquidity-pressure rule, from-scratch NumPy diagonal LDA/ridge/softmax classifiers,
sklearn SGD, the historical HGB candidate and a bounded LightGBM grid. It also produces richer
evidence metrics/plots, generated CV bullets, and a GitHub publisher that accepts only validated
report bundles. The historical v0.4 numerical results
remain reference diagnostics only until the real study is regenerated under v0.6 hashes.

## Pre-holdout audit status

The independent v0.4 audit found a publisher-class semantic error (`1` and `3` were named in the
wrong directions) plus several integrity/reporting hardening opportunities. The source/config/code
were corrected in v0.4.1 and retained in v0.6.0. **The real CF_9 test remains unopened.**

The v0.4 development metrics below are retained only as historical diagnostics. They are no longer
release-valid because the FI-2010 implementation/config hashes changed. Rerun the target-blind audit,
CF_1-CF_8 development, candidate freeze and Train_CF_9 refit under v0.6.0 before final holdout release.

## Implemented

- Strict registered configuration for the primary Z-score/no-auction representation, 144 features,
  five publisher labels, class mapping and primary 50-event horizon.
- Exact source-byte identity validation and safe nested-ZIP central-directory validation.
- Atomic extraction of only the required inner ZIP into a gitignored source directory.
- Separate source, development and holdout manifests; development never loads the holdout manifest
  or CF_9 test payload.
- Bounded float32 parsing and validation of 149 source rows with correct observation transposition.
- Independent paired CF_1-CF_8 evaluation and per-fold memory release.
- Fold-local class weighting, manual/from-scratch statistical baselines, bounded nonlinear model
  comparison, deterministic candidate and confidence-rule selection, robustness horizons and
  efficiency measurements.
- Candidate/config/source/implementation freeze followed by Train_CF_9-only final refit.
- Durable one-shot holdout claim, atomic seal and completion-anchor verification.
- Development-only and anchored-holdout report modes with synthetic evidence made claim-ineligible.
- Secure operator scripts, adversarial tests, methodology, handoff and attribution documentation.

## Evidence boundary

The real CF_9 test member is not opened by audit, development, freeze/refit or development reporting.
The exact official source was verified at `1,830,875,986` bytes and SHA-256
`bcc89a5aa7d8067dda98374393444eb885a4283a41fd33e323496380e057e1a6`; the imported inner archive
hashes to `cea93692a270724fa91e8f124da641db727d757e5e0f0bb85067709e9932f664`.

The full CF_1-CF_8 comparison completed without sampling. The registered nonlinear fallback was
selected at mean macro-F1 `0.553178` and worst-fold macro-F1 `0.508623`; confidence threshold `0.70`
gave mean directional precision `0.780922` at mean coverage `0.080247`. These are development
diagnostics, not a final-holdout result.

The frozen candidate was refitted on all `362,400` Train_CF_9 observations and saved with a model
SHA-256 of `1a159bbf0ae6749c60c28ef6667ab4aed45a6dac8d7e6de64fb3f9604a2e1c01`. The release stops there.
No real holdout claim, seal, completion anchor or output exists.

No executable-return claim is supported. FI-2010's anonymised normalized snapshots do not provide
the market information needed to reconstruct defensible fills, costs or profitability.

## Unsupported legacy paths

Prior restricted-data study code remains in-tree for history but is excluded from the supported
public workflow and from `reports/cv_evidence.md`.
