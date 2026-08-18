# Results and discussion

## Evidence status

The current committed numerical benchmark is the historical real CF_1-CF_8 development run from
v0.4. It is retained for traceability but must be regenerated under v0.6 implementation/config hashes
before the one-shot final holdout is released. The generated v0.6 report will supersede this document
for final recruiter-facing numerical claims.

## Historical real development benchmark

The audited histogram-gradient-boosting candidate achieved mean macro-F1 **0.553178** over the eight
anchored development pairs, with **0.508623** on the worst fold and **0.586168** on the best fold.
Balanced accuracy moved in the same direction as macro-F1, while MCC rose from roughly 0.28 on the
earliest folds to 0.40 on CF_8. The development pattern therefore did not depend on a single outlier
fold.

At the development-selected confidence threshold of 0.70, the classifier produced mean directional
precision **0.780922** at mean coverage **0.080247**, abstaining on roughly 92% of snapshots. This is a
selective predictive diagnostic, not an executed-trading result.

## Why v0.6 adds the manual/statistical ladder

A boosted-tree score alone says little about where the predictive structure comes from. Version 0.6
therefore evaluates a fixed liquidity-pressure rule and directly implemented diagonal LDA, multiclass
ridge and softmax classifiers before sklearn SGD, histogram boosting and LightGBM. The final report
will show the best registered setting from each family and the difference between the best
manual/from-scratch model and the best nonlinear tree model on identical anchored folds.

Three outcomes are informative:

- if manual/statistical models are close to the winner, much of the signal is approximately linear or
  class-conditional and can be explained economically/statistically;
- if boosted trees materially outperform them, nonlinear feature interactions add measurable value;
- if the simplest model wins, the registered complexity tie-break and one-shot holdout discipline make
  that result preferable to forcing a more complicated model for presentation purposes.

## Generalisation and holdout interpretation

`Test_CF_9` remains deliberately untouched. After the v0.6 candidate is frozen and refitted on
`Train_CF_9`, CF_9 is evaluated once. The final report records the one-shot macro-F1, per-class metrics,
confidence-filtered precision/coverage and the generalisation gap relative to the CF_1-CF_8 mean.
The retained number is not rerun or tuned away if it is weaker than development performance.

## Scope and limitations

FI-2010 is suited to predictive LOB classification and controlled model comparison. The public
normalised/anonymised matrices do not provide the raw prices, timestamps, instrument/day identities
and execution mechanics needed for a defensible fill model, transaction-cost study, P&L or Sharpe
ratio. The project therefore treats predictive validity and research integrity as the deliverables
rather than implying executable profitability.
