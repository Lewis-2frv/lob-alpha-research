# FI-2010 project — CV positioning

## Safe pre-holdout wording

- Built a leakage-resistant equity limit-order-book signal research system on FI-2010, comparing a
  manual liquidity-pressure rule, from-scratch NumPy LDA/ridge/softmax classifiers and boosted-tree
  models across eight anchored walk-forward folds with fold-local fitting and a sealed final holdout.
- Historical development run achieved 0.553 mean macro-F1 (0.509 worst fold) and 78.1% directional
  precision at 8.0% coverage using confidence-based abstention; final CF_9 evidence remains excluded
  until the one-shot release is completed under the current implementation hashes.

## Final wording

After the v0.6 one-shot release, use the automatically generated
`docs/results/fi2010/cv_summary.md`. It is derived directly from the integrity-validated evidence
bundle and includes the retained CF_9 result without manual transcription.

## Interview framing

Lead with the research design rather than pretending FI-2010 is an executable strategy. The strongest
parts of the project are the cumulative-fold handling, leakage controls, interpretable-to-nonlinear
model ladder, direct statistical implementations, confidence/coverage trade-off, immutable model
freeze, and one-shot holdout discipline.
