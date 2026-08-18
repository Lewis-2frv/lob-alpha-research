# FI-2010 methodology

## Research question and representation

The study asks whether a bounded snapshot classifier can predict the publisher's three-way
mid-price direction label out of sample under the official anchored protocol. It uses only
`NoAuction/1.NoAuction_Zscore`. Source rows 1-144 are model features; rows 145-149 are label horizons;
source columns are observations. Publisher class values are up=1, stationary=2 and down=3.

The primary horizon is the fourth label row: five sampled steps, conventionally 50 underlying LOB
events. The 1, 2, 3 and 10-step label rows are development-only label-agreement diagnostics: the
primary-horizon model predictions are rescored against those alternate publisher label rows. They are
not separately trained horizon models and cannot replace the primary result.

## Chronology and leakage controls

The official training members are cumulative. Each development fold therefore loads only
`Train_CF_n` and `Test_CF_n`, scores that pair, and releases it before the next fold. Combining
training members would count earlier observations repeatedly and is prohibited in code, manifests
and tests.

Feature snapshots are not converted into sequence windows. The release does not provide reliable
instrument/day boundary identifiers, so concatenated LSTM/CNN windows could cross unknown boundaries.

Any scaling and class weighting is fitted from the matching training member only. Test labels do not
influence preprocessing, weights, model fitting, candidate selection or threshold fitting. Other
horizons remain development-only.

## Models and selection

The candidate universe is deliberately ordered from interpretable/manual methods to flexible nonlinear
models so the results can answer both a predictive and a modelling question. Registered comparisons are:

1. always-stationary and class-prior baselines;
2. a fixed liquidity-pressure rule using the first 40 basic-book features plus the accumulated
   volume-difference feature. Because the study uses publisher Z-score inputs, this is a standardised
   depth-pressure proxy rather than a raw queue-imbalance ratio;
3. a from-scratch NumPy shared-diagonal LDA classifier. It estimates class means and pooled diagonal
   variance from the training member only, shrinks the variance vector toward its cross-feature mean,
   and uses equal class priors;
4. from-scratch NumPy class-balanced multiclass ridge regression, using fold-local centring/scaling and
   closed-form weighted normal equations over three registered L2 penalties;
5. from-scratch NumPy class-balanced multinomial softmax regression, using fold-local centring/scaling
   and deterministic mini-batch Adam;
6. sklearn multinomial SGD with log loss at two registered regularisation values;
7. the audited sklearn histogram-gradient-boosting candidate in every environment;
8. when the FI-2010 dependency set is installed, a bounded nine-point LightGBM multiclass grid over
   learning rate `{0.03, 0.05, 0.08}` and leaves `{15, 31, 63}`, with a fixed 180-tree budget and
   fixed regularisation/subsampling settings.

The directly implemented models are intentionally simple enough to derive and explain in an interview.
They are not given a privileged selection rule: every candidate is scored on exactly the same eight
anchored development pairs. Keeping the historical HGB candidate in the universe also means the v0.6
search cannot discard the earlier nonlinear benchmark unless another registered model genuinely wins.

The development selection score is mean macro-F1 over CF_1-CF_8. The first tie-break is worst-fold
macro-F1; the second is lower registered complexity; a stable serialized specification breaks any
remaining exact tie. Secondary metrics are balanced accuracy, multiclass log loss, MCC, class-level
precision/recall/F1 and the confusion matrix.

## Directional signal diagnostics

At each registered threshold, class-1 probability above threshold and greater than class-3
probability produces an up signal. The symmetric class-3 condition produces a down signal. All other cases,
including probability ties, abstain. Threshold selection uses only CF_1-CF_8 and reports directional
precision, coverage, abstention and fold stability. These labels are not executed trades.

## Freeze and holdout

After candidate and threshold selection, a frozen artifact binds the source and development
manifests, configuration, features, class mapping, primary horizon, preprocessing, model parameters,
confidence rule and FI-2010 implementation hashes. The frozen classifier is then refitted once using
Train_CF_9 and saved without opening Test_CF_9.

Final release requires the exact acknowledgement `RELEASE FI2010 CF9 HOLDOUT ONCE`. A source-side
exclusive claim and atomic seal are created before the test payload is opened. The seal is independent
of user-selected output paths and survives crashes. A successful result is reportable only if its
completion anchor validates every output hash. This task does not release the real holdout.

## Interpretation limits

FI-2010 is anonymised and normalised. The study consumes the publisher-provided Z-score representation;
the released normalized matrices do not expose the raw normalization statistics needed to independently
audit the publisher preprocessing field of view. They also omit unnormalised prices, timestamps,
instrument identities, feed/venue mechanics and reliable day boundaries needed for a defensible fill,
transaction-cost, P&L, Sharpe or live-profitability calculation. The study therefore makes predictive
and confidence/coverage claims only.

## Portfolio evidence

The reporting stage derives fold dispersion, best-baseline uplift, model ranking, per-class metrics,
confidence/coverage frontier and, after release, the retained CF_9 generalisation gap. It writes
content-addressed PNG charts, a machine-readable `portfolio_metrics.json` and a generated CV summary.
The publisher accepts only hash-valid report bundles and, by default, requires real claim-eligible
one-shot holdout evidence before updating `docs/results/fi2010/` and the README results block.
