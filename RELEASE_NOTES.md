# v0.6 Interpretable-to-nonlinear model ladder and publication release

- Added a fixed, label-free liquidity-pressure rule based on the registered FI-2010 10-level basic
  book layout and accumulated volume-difference feature. The rule is documented as a standardised
  depth-pressure proxy because the public study uses publisher Z-score features.
- Added three directly implemented statistical classifiers: shared-diagonal LDA with shrinkage,
  class-balanced closed-form multiclass ridge regression, and class-balanced multinomial softmax
  regression trained with NumPy/Adam.
- The registered development ladder now progresses from naive baselines -> manual microstructure rule
  -> from-scratch statistical models -> sklearn linear SGD -> histogram gradient boosting -> bounded
  LightGBM tuning, all on identical CF_1-CF_8 anchored folds.
- Added explicit model-family comparison metrics and a report discussion quantifying the development
  difference between the best manual/from-scratch candidate and the best nonlinear tree candidate.
- Changed the candidate-comparison figure to show the best registered setting from every model family,
  ensuring simple/manual methods remain visible even if several LightGBM settings rank above them.
- Expanded regression coverage for the manual directional semantics and finite/proper probabilities
  from every from-scratch statistical classifier.
- Version 0.6 remains pre-holdout: the historical v0.4 development figures are reference evidence only.
  The real CF_1-CF_8 study and Train_CF_9 refit must be regenerated under v0.6 hashes before the
  irreversible one-shot CF_9 release.

# v0.5 Portfolio-complete tuning and evidence release

- Expanded the pre-holdout candidate universe while retaining the audited sklearn HGB candidate as
  a permanent benchmark: nine bounded LightGBM settings now cover three learning rates and three
  tree-size settings.
- LightGBM regularisation/subsampling settings are fixed in code; candidate selection still uses only
  anchored CF_1-CF_8 mean macro-F1, worst-fold macro-F1 and the registered complexity tie-break.
- Added portfolio metrics including fold dispersion, best-baseline uplift, confidence/coverage
  frontier, selected-model efficiency and holdout generalisation gap.
- Added publication-quality figures for fold stability, candidate comparison, confidence filtering
  and development-versus-holdout performance.
- Added generated CV-ready project bullets and a GitHub publisher that copies only integrity-validated
  small evidence into `docs/results/fi2010/` and updates the README results block.
- The one-shot PowerShell release now builds the final report and publishes the GitHub evidence after
  CF_9 completes.
- Added FI-2010 GitHub CI, architecture/integrity documentation, MIT license and citation metadata.
- Version 0.5 does not fabricate or inherit a final CF_9 score. The real source must be rerun through
  development/freeze under the new implementation/config hashes before one deliberate final release.

# v0.4.1 FI-2010 pre-holdout audit fixes

- Corrected the publisher class semantics to `1=up`, `2=stationary`, `3=down` throughout the
  configuration, metrics, directional diagnostics and documentation.
- Revalidates the selected candidate and confidence threshold from serialized CF_1-CF_8 evidence
  before freeze/refit rather than trusting a mutable selected-candidate record.
- Binds the audit-time CF_9 holdout-manifest hash into the development/frozen evidence and rejects
  later manifest redirection before final release.
- Strengthened final-model semantic bindings to the frozen specification, source, configuration,
  implementation and Train_CF_9 member.
- Binds Python/NumPy/scikit-learn/joblib/LightGBM runtime versions across development, freeze,
  final refit and holdout so dependency drift fails closed.
- Validates development-manifest representation/member semantics before payload access and prevents
  audit-manifest rewrites after any CF_9 release marker exists.
- Makes reporting reject incomplete/crashed holdout states instead of describing a claimed CF_9 as
  untouched, and validates the seal, holdout manifest, frozen candidate and final model together.
- Added a numerically stable fallback for saturated SGD probability rows on newer scikit-learn
  releases, plus regression coverage.
- Corrected the synthetic fixture's latent up/down labels and clarified alternate-horizon outputs as
  label-agreement diagnostics rather than separately trained horizon models.
- Expanded generated evidence with model-ranking, per-class metrics and an aggregate confusion matrix.
- Changed the one-shot PowerShell release script to write the final report to `report-final`, so an
  existing development-only report cannot cause a post-release overwrite failure.
- The real CF_9 test remains unopened. Because FI-2010 implementation/config hashes changed, the
  real CF_1-CF_8 development study and Train_CF_9 refit must be rerun before release.

# v0.4 FI-2010 anchored-validation release

- Pivoted the supported public workflow to official FI-2010 equity limit-order-book snapshot
  classification.
- Added exact outer-archive size/SHA-256 verification and secure nested-ZIP handling with bounded
  metadata checks and partial-plus-atomic inner-archive import.
- Added strict 149-row float32 parsing, feature/label validation and primary-horizon selection.
- Implemented independent official paired folds CF_1-CF_8 without cumulative-fold concatenation.
- Added class-prior, always-stationary, reproducible multinomial SGD, bounded LightGBM and sklearn
  fallback classifiers with training-member-only class weights.
- Registered macro-F1 selection, deterministic tie-breaks, secondary multiclass metrics,
  confidence/coverage diagnostics and efficiency measurements.
- Added freeze-before-refit handling for Train_CF_9 and a durable, source-bound one-shot CF_9 test
  gate. This release does not open or evaluate the real CF_9 test member.
- Added integrity-gated reporting, synthetic claim exclusion, adversarial tests and four
  current-directory-independent PowerShell operator scripts.
- Added FI-2010 CC BY 4.0 attribution, methodology and data-handoff documentation.

## Legacy status

The earlier restricted-data workflow remains available only as unsupported legacy code and is not
eligible for this release's public CV evidence. The optional exact-session futures feasibility path
also remains outside the FI-2010 evidence boundary.
