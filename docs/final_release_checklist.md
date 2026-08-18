# FI-2010 v0.6 final release checklist

The repository is designed so the final real result is produced by the code, not transcribed by hand.
Raw FI-2010 payloads and fitted models remain outside Git; small validated evidence is published after
CF_9 completes.

## Before CF_9

1. Apply the v0.6 source release and activate `.venv`.
2. Run `scripts\prepare_fi2010_release.ps1 -InstallDependencies` into a fresh artifact directory.
3. Review `report-development\fi2010_evidence.md` and `portfolio_metrics.json`.
4. Commit the exact source/config/documentation state. Do not alter code, config or runtime packages
   after development/freeze.
5. Confirm that no source-side CF_9 claim, seal or completion anchor exists.

## One-shot release

Run `scripts\run_fi2010_holdout_report.ps1` once with the exact acknowledgement phrase. The gate
validates source, configuration, implementation, runtime, frozen candidate, model artifact and
holdout-member bindings before opening the payload. A source-side claim/seal prevents output-path
reruns.

## After CF_9

The release command generates `report-final` and invokes the portfolio publisher. Commit the resulting
small `docs/results/fi2010` evidence and the README results block. Never rerun CF_9 to improve a number;
the retained result is the point of the holdout design.
