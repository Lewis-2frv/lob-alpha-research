# FI-2010 portfolio evidence

This directory is generated only from an integrity-validated FI-2010 evidence bundle. It intentionally contains no raw market data or fitted model binary.

- Selected model: **LightGBM lr=0.08, leaves=15**
- Development mean macro-F1: **0.555**
- Best manual/from-scratch development model: **NumPy ridge alpha=0.1** (**0.440** mean macro-F1)
- Development worst-fold macro-F1: **0.506**
- Development directional precision: **77.3%** at **9.0%** coverage
- One-shot CF_9 macro-F1: **0.545**
- One-shot CF_9 directional precision: **75.8%** at **6.0%** coverage

![Development stability](development_macro_f1_by_fold.png)

![Model comparison](model_comparison.png)

![Confidence frontier](confidence_precision_coverage.png)

![Development versus holdout](development_vs_holdout.png)

- [Full evidence](fi2010_evidence.md)
- [CV-ready project summary](cv_summary.md)
- [Machine-readable metrics](portfolio_metrics.json)
