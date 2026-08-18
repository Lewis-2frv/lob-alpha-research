# Research-integrity contract

The FI-2010 path is intentionally conservative.

1. **No cumulative-fold concatenation.** Each official anchored train/test pair is evaluated
   independently.
2. **No CF_9 test access during development.** Audit, model comparison, confidence tuning, freeze and
   final refit are all test-blind.
3. **No post-hoc holdout tuning.** The candidate, threshold, source identity, runtime and
   implementation are frozen before the holdout is opened.
4. **One final result is retained.** A durable source-side claim prevents a normal second release even
   if output paths change or the first attempt crashes.
5. **No executable-return storytelling.** FI-2010 is used for predictive classification and
   confidence/coverage analysis only.
6. **No raw-data publication.** Raw/extracted files and fitted model binaries remain outside Git.
7. **No synthetic-to-real claim leakage.** Synthetic rehearsals are marked claim-ineligible by code
   and report metadata.
