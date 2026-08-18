# CV evidence — FI-2010

The supported public evidence is the official FI-2010 anchored walk-forward study using
`NoAuction/1.NoAuction_Zscore` snapshots.

Registered evidence rules:

1. CF_1-CF_8 use only each fold's matching cumulative training member and matching test member.
2. Cumulative training members are never concatenated.
3. The primary target is label row 4: five sampled steps / conventionally 50 underlying events.
4. Candidate selection maximises mean macro-F1, then worst-fold macro-F1, then lower complexity.
5. Confidence thresholds and every model choice use development folds only.
6. The frozen candidate is refitted on Train_CF_9 before any CF_9 test access.
7. Final holdout evidence is reportable only with a valid source-side seal and completion anchor.
8. Synthetic rehearsals and legacy restricted-data artifacts are ineligible for public CV claims.

## Historical v0.4 development result — v0.6 regeneration required

The numerical development diagnostics below were generated before the v0.4.1 pre-holdout audit.
The audit corrected the publisher class semantics and strengthened the freeze/holdout integrity
checks. Macro-F1 and the aggregate directional signal metrics are label-name invariant here, but the
implementation/config hashes changed, so these results must be regenerated before they are used as
current release evidence or a final CV bullet.


Version 0.6 retains this audited HGB specification and expands the registered development ladder with
a fixed liquidity-pressure rule, from-scratch NumPy diagonal LDA/ridge/softmax classifiers, sklearn
SGD and a bounded nine-point LightGBM grid. Every stronger candidate must win on the same CF_1-CF_8
criterion before the real CF_9 holdout is released. The final GitHub
and CV evidence is generated automatically from the v0.6 report bundle rather than manually copied.

The exact registered archive was verified and the full, unsampled CF_1-CF_8 comparison completed on
2026-08-17. LightGBM was unavailable, so the registered sklearn nonlinear fallback was evaluated.
The fallback (`learning_rate=0.08`, `max_leaf_nodes=31`, `max_iter=60`) was selected:

- Mean development macro-F1: `0.553178`.
- Worst-fold macro-F1: `0.508623`.
- Selected confidence threshold: `0.70`.
- Mean directional precision at that threshold: `0.780922`.
- Mean directional coverage: `0.080247`; mean abstention: `0.919753`.

| CF pair | Macro-F1 | Balanced accuracy | Log loss | MCC | Signal precision | Coverage |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.508623 | 0.515380 | 0.961627 | 0.285436 | 0.739808 | 0.122015 |
| 2 | 0.511113 | 0.518960 | 0.974294 | 0.284476 | 0.730848 | 0.085544 |
| 3 | 0.556653 | 0.556752 | 0.922011 | 0.335211 | 0.805520 | 0.084164 |
| 4 | 0.562801 | 0.564857 | 0.919871 | 0.344039 | 0.798295 | 0.087653 |
| 5 | 0.570066 | 0.570559 | 0.908431 | 0.356983 | 0.795050 | 0.077391 |
| 6 | 0.568641 | 0.572845 | 0.908970 | 0.354621 | 0.782764 | 0.082338 |
| 7 | 0.561359 | 0.565863 | 0.892904 | 0.359264 | 0.800459 | 0.047154 |
| 8 | 0.586168 | 0.581785 | 0.859064 | 0.403482 | 0.794634 | 0.055720 |

The full development comparison took `779.3` seconds. Maximum simultaneous input arrays occupied
`210,554,400` bytes; this is an exact NumPy-array allocation figure, not operating-system peak RAM.
The selected candidate was frozen, then refitted on all `362,400` Train_CF_9 observations in `106.3`
seconds. The final model and manifest were saved locally and content-hashed.

The real CF_9 test is intentionally untouched. The development artifact and final-refit manifest both
record `cf9_test_payload_opened=false`, and no real holdout claim, seal, completion anchor or output
exists. These are anchored walk-forward **development** results; no final-holdout result is claimed.

FI-2010 supports predictive classification and confidence/coverage analysis. It does not support a
defensible executable-return study because unnormalised prices, timestamps, instrument identities,
feed detail and reliable sequence boundaries are unavailable.
