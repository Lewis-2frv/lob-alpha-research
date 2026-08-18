# FI-2010 research architecture

## Design objective

The implementation treats research integrity as part of the model. The goal is not only to produce a
classification score but to make it difficult to accidentally leak future information, duplicate
cumulative observations, mutate the selected model after development, or repeatedly inspect the final
holdout.

## Data boundary

The source importer verifies the exact official outer-archive size and SHA-256, validates ZIP
structure defensively, and extracts only the registered inner benchmark archive. Development and
holdout manifests are separate. CF_1-CF_8 plus `Train_CF_9` are visible to the development workflow;
`Test_CF_9` is represented only by a sealed holdout manifest until final release.

## Anchored validation

FI-2010 training members are cumulative. The study therefore processes each pair independently:

```text
Train_CF_1 -> Test_CF_1
Train_CF_2 -> Test_CF_2
...
Train_CF_8 -> Test_CF_8
```

It never concatenates `Train_CF_1 ... Train_CF_8`, because doing so would repeatedly include earlier
observations.

For each pair, model fitting and any learned preprocessing/class weighting use only that pair's
training member. Candidate scores are aggregated only after the pair has been evaluated.

## Selection and tuning

The candidate universe is registered before the final holdout. Version 0.6 uses an explicit
complexity ladder: naive baselines, a fixed liquidity-pressure rule, from-scratch NumPy diagonal LDA,
ridge and softmax models, sklearn linear SGD, the audited histogram-gradient-boosting model, then a
bounded LightGBM grid. This lets the report show the marginal development benefit of model complexity
instead of presenting LightGBM in isolation.

All learned centring/scaling, class weighting and model parameters are fitted from the matching training
member only. Selection maximises mean macro-F1 across all eight development pairs, with worst-fold
macro-F1 as the first tie-breaker and lower complexity as the second.

The confidence threshold is selected from the same development evidence subject to a minimum coverage
constraint. It is frozen alongside the model rather than tuned on CF_9.

## Freeze/refit boundary

The frozen artifact binds:

- source and development-manifest hashes;
- holdout-manifest hash;
- model specification and confidence rule;
- feature/target/class declarations;
- configuration SHA-256;
- implementation hashes;
- Python/NumPy/scikit-learn/joblib/LightGBM runtime versions.

Only after this freeze is the selected classifier refitted on all of `Train_CF_9`.

## One-shot CF_9 release

Final evaluation requires the exact phrase `RELEASE FI2010 CF9 HOLDOUT ONCE`. Before the test payload
is opened, an exclusive source-side claim and atomic seal are created. A crash after that point remains
claimed, so changing output paths or rerunning the command cannot produce a second valid evaluation.
A successful run creates a completion anchor binding the final result hash.

## Reporting

The report layer revalidates source/config/runtime/code/freeze/holdout bindings before accepting any
result. It then produces:

- fold stability and candidate-ranking tables;
- per-class metrics and aggregate confusion matrix;
- confidence/coverage frontier;
- final holdout comparison if released;
- reproducibility hashes;
- plots and machine-readable portfolio metrics;
- CV-ready project bullets.

The publisher only moves these small validated artifacts into `docs/results/fi2010/`.
