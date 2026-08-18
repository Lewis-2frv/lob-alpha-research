# CV-ready FI-2010 project summary

- Built a leakage-resistant FI-2010 limit-order-book signal research pipeline with 8 anchored walk-forward development folds and a one-shot sealed CF_9 holdout; selected LightGBM lr=0.08, leaves=15 at 0.555 development macro-F1 and achieved 0.545 final holdout macro-F1.
- Compared a manual LOB liquidity-pressure rule and from-scratch NumPy LDA/ridge/softmax models against sklearn and boosted-tree benchmarks on identical anchored folds; implemented confidence-based abstention (77.3% development directional precision at 9.0% coverage), content-addressed model freezing, runtime/config integrity checks, and a durable single-use holdout gate.
- Best manual/from-scratch development model: NumPy ridge alpha=0.1 at 0.440 mean macro-F1; best nonlinear tree model: LightGBM lr=0.08, leaves=15 at 0.555.
- Scope: predictive LOB classification and confidence/coverage analysis only; FI-2010's normalised anonymised snapshots do not support defensible executable P&L or Sharpe claims.
