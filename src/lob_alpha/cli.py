"""Command-line interface for the staged v0.2 research pipeline."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from datetime import date, timedelta
from pathlib import Path

from .config import load_config
from .dataset import process_raw_directory
from .definitions import verify_contract_definition
from .fixture import make_mbp10_fixture
from .ingest import (
    batch_job_status,
    definition_window,
    download_batch_job,
    download_stream,
    estimate_cost,
    load_events,
    submit_batch_job,
)
from .manifest import build_run_manifest, write_json
from .pipeline import process_session, write_table
from .sampling import session_bounds
from .study import (
    freeze_candidate,
    run_holdout_stage,
    run_train_stage,
    run_validation_stage,
)


def _config_check(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    print(
        json.dumps(
            {
                "status": "valid",
                "project": config.name,
                "dataset": config.data.dataset,
                "schema": config.data.schema,
                "symbols": config.data.symbols,
                "horizons_ms": config.labels.horizons_ms,
            },
            indent=2,
        )
    )
    return 0


def _estimate(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    cost = estimate_cost(config)
    print(
        json.dumps(
            {"estimated_cost_usd": cost, "request": asdict(config.data)},
            indent=2,
            default=str,
        )
    )
    return 0


def _download(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    output, cost = download_stream(
        config,
        args.output,
        max_cost_usd=args.max_cost_usd,
        overwrite=args.overwrite,
    )
    print(json.dumps({"output": str(output), "estimated_cost_usd": cost}, indent=2))
    return 0


def _batch_submit(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    details, cost = submit_batch_job(
        config,
        max_cost_usd=args.max_cost_usd,
        confirm_paid_request=args.confirm_paid_request,
    )
    job_id = str(details["id"])
    manifest_path = Path(args.manifest or f"data/manifests/batch_{job_id}.json")
    write_json(
        manifest_path,
        {
            "job_id": job_id,
            "estimated_cost_usd": cost,
            "max_cost_usd": args.max_cost_usd,
            "provider_response": details,
        },
    )
    print(
        json.dumps(
            {"job_id": job_id, "estimated_cost_usd": cost, "manifest": str(manifest_path)},
            indent=2,
        )
    )
    return 0


def _batch_status(args: argparse.Namespace) -> int:
    print(json.dumps(batch_job_status(args.job_id), indent=2, default=str))
    return 0


def _batch_download(args: argparse.Namespace) -> int:
    downloaded, remote_files = download_batch_job(args.job_id, args.output_dir)
    manifest_path = Path(args.manifest or f"data/manifests/batch_{args.job_id}_download.json")
    write_json(
        manifest_path,
        {
            "job_id": args.job_id,
            "downloaded_files": [str(path) for path in downloaded],
            "provider_files": remote_files,
            "hash_verification": "passed",
        },
    )
    print(
        json.dumps(
            {"files": [str(path) for path in downloaded], "manifest": str(manifest_path)},
            indent=2,
        )
    )
    return 0


def _batch_run(args: argparse.Namespace) -> int:
    """Submit, poll and download a cost-capped batch in one local command."""

    if args.poll_seconds <= 0 or args.timeout_minutes <= 0:
        raise ValueError("poll seconds and timeout minutes must be positive")
    config = load_config(args.config)
    details, cost = submit_batch_job(
        config,
        max_cost_usd=args.max_cost_usd,
        confirm_paid_request=args.confirm_paid_request,
    )
    job_id = str(details["id"])
    deadline = time.monotonic() + args.timeout_minutes * 60.0
    status = details
    while status.get("state") not in {"done", "expired"}:
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"batch {job_id} did not finish within {args.timeout_minutes} minutes; "
                "resume with batch-status/batch-download"
            )
        time.sleep(args.poll_seconds)
        status = batch_job_status(job_id)
        print(json.dumps({"job_id": job_id, "state": status.get("state")}))
    if status.get("state") != "done":
        raise RuntimeError(f"batch job ended in state {status.get('state')!r}")
    downloaded, remote_files = download_batch_job(job_id, args.output_dir)
    manifest_path = Path(args.manifest or f"data/manifests/batch_{job_id}_complete.json")
    write_json(
        manifest_path,
        {
            "job_id": job_id,
            "estimated_cost_usd": cost,
            "max_cost_usd": args.max_cost_usd,
            "final_status": status,
            "downloaded_files": [str(path) for path in downloaded],
            "provider_files": remote_files,
            "hash_verification": "passed",
        },
    )
    print(
        json.dumps(
            {
                "job_id": job_id,
                "estimated_cost_usd": cost,
                "files": [str(path) for path in downloaded],
                "manifest": str(manifest_path),
            },
            indent=2,
        )
    )
    return 0


def _download_definitions(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    start, end = definition_window(config)
    output, cost = download_stream(
        config,
        args.output,
        max_cost_usd=args.max_cost_usd,
        schema="definition",
        start=start,
        end=end,
        overwrite=args.overwrite,
    )
    print(json.dumps({"output": str(output), "estimated_cost_usd": cost}, indent=2))
    return 0


def _verify_definition(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    definitions = load_events(args.input)
    verified = verify_contract_definition(
        definitions,
        symbol=args.symbol or config.data.symbols[0],
        expected=config.contract,
    )
    print(json.dumps(asdict(verified), indent=2))
    return 0


def _process(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    input_path = Path(args.input)
    events = load_events(input_path)
    session_date = date.fromisoformat(args.session_date)
    result = process_session(
        events,
        config,
        session_date=session_date,
        tick_size=args.tick_size or config.contract.expected_tick_size,
    )
    output = write_table(result.data, args.output)
    manifest = build_run_manifest(
        config_path=args.config,
        input_path=input_path,
        output_path=output,
        session_date=args.session_date,
        rows=len(result.data),
        quality=result.quality.to_dict(),
    )
    manifest_path = Path(args.manifest) if args.manifest else output.with_suffix(".manifest.json")
    write_json(manifest_path, manifest)
    print(
        json.dumps(
            {"output": str(output), "manifest": str(manifest_path), "rows": len(result.data)},
            indent=2,
        )
    )
    return 0


def _process_all(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    entries = process_raw_directory(
        config,
        raw_dir=args.raw_dir or config.data.raw_dir,
        output_dir=args.output_dir or config.data.processed_dir,
        tick_size=args.tick_size or config.contract.expected_tick_size,
        output_format=args.output_format,
        overwrite=args.overwrite,
    )
    print(
        json.dumps(
            {
                "sessions": len(entries),
                "rows": sum(entry.rows for entry in entries),
                "catalog": str(Path(args.output_dir or config.data.processed_dir) / "catalog.json"),
            },
            indent=2,
        )
    )
    return 0


def _train_stage(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    payload = run_train_stage(
        config,
        catalog_path=args.catalog,
        output_dir=args.output_dir,
    )
    print(json.dumps(payload, indent=2))
    return 0


def _validation_stage(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    payload = run_validation_stage(
        config,
        catalog_path=args.catalog,
        train_selection_path=args.train_selection,
        raw_dir=args.raw_dir or config.data.raw_dir,
        output_dir=args.output_dir,
    )
    print(json.dumps(payload, indent=2))
    return 0


def _freeze(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    payload = freeze_candidate(
        config,
        catalog_path=args.catalog,
        candidate_path=args.candidate,
        output_path=args.output,
    )
    print(json.dumps(payload, indent=2))
    return 0


def _holdout_stage(args: argparse.Namespace) -> int:
    if not args.acknowledge_one_shot:
        raise ValueError("--acknowledge-one-shot is required")
    config = load_config(args.config)
    payload = run_holdout_stage(
        config,
        catalog_path=args.catalog,
        frozen_candidate_path=args.frozen_candidate,
        raw_dir=args.raw_dir or config.data.raw_dir,
        output_dir=args.output_dir,
    )
    print(json.dumps(payload, indent=2))
    return 0


def _report(args: argparse.Namespace) -> int:
    from .reporting import build_research_report

    report, cv = build_research_report(
        train_dir=args.train_dir,
        validation_dir=args.validation_dir,
        holdout_dir=args.holdout_dir,
        reports_dir=args.reports_dir,
    )
    print(json.dumps({"report": str(report), "cv_evidence": str(cv)}, indent=2))
    return 0


def _run_fixture(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    events_path = output_dir / "engineering_fixture_events.csv.gz"
    dataset_path = output_dir / "engineering_fixture_dataset.csv.gz"
    events = make_mbp10_fixture()
    write_table(events, events_path)
    result = process_session(
        events,
        config,
        session_date=date(2026, 3, 16),
        tick_size=config.contract.expected_tick_size,
    )
    write_table(result.data, dataset_path)
    write_json(output_dir / "engineering_fixture_quality.json", result.quality.to_dict())
    print(
        json.dumps(
            {
                "warning": "engineering fixture only; not empirical evidence",
                "events": len(events),
                "processed_rows": len(result.data),
                "output": str(dataset_path),
            },
            indent=2,
        )
    )
    return 0


def _run_fixture_study(args: argparse.Namespace) -> int:
    """Exercise every stage with synthetic data, including the freeze boundary."""

    config = load_config(args.config)
    root = Path(args.output_dir)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"refusing to overwrite fixture study: {root}")
    raw_dir = root / "raw"
    processed_dir = root / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    session_dates = []
    current = config.splits.train_start
    while current <= config.splits.holdout_end:
        if current.weekday() < 5 and config.splits.split_for(current) is not None:
            session_dates.append(current)
        current += timedelta(days=1)
    for index, session_date in enumerate(session_dates):
        start, _ = session_bounds(session_date, config.session)
        events = make_mbp10_fixture(
            periods=700,
            seed=config.seed + index,
            start=start.isoformat(),
            tick_size=config.contract.expected_tick_size,
        )
        write_table(events, raw_dir / f"ESM6_{session_date.isoformat()}_mbp10.csv.gz")
    process_raw_directory(
        config,
        raw_dir=raw_dir,
        output_dir=processed_dir,
        tick_size=config.contract.expected_tick_size,
        output_format="csv.gz",
    )
    catalog = processed_dir / "catalog.json"
    train_dir = root / "train"
    validation_dir = root / "validation"
    holdout_dir = root / "holdout"
    run_train_stage(config, catalog_path=catalog, output_dir=train_dir)
    candidate = run_validation_stage(
        config,
        catalog_path=catalog,
        train_selection_path=train_dir / "train_selection.json",
        raw_dir=raw_dir,
        output_dir=validation_dir,
    )
    frozen_path = root / "frozen_candidate.json"
    freeze_candidate(
        config,
        catalog_path=catalog,
        candidate_path=validation_dir / "selected_candidate.json",
        output_path=frozen_path,
    )
    result = run_holdout_stage(
        config,
        catalog_path=catalog,
        frozen_candidate_path=frozen_path,
        raw_dir=raw_dir,
        output_dir=holdout_dir,
    )
    from .reporting import build_research_report

    report, cv = build_research_report(
        train_dir=train_dir,
        validation_dir=validation_dir,
        holdout_dir=holdout_dir,
        reports_dir=root / "reports",
        engineering_fixture=True,
    )
    print(
        json.dumps(
            {
                "warning": "engineering fixture only; never cite as empirical performance",
                "sessions": len(session_dates),
                "candidate": candidate,
                "mechanical_holdout": result,
                "report": str(report),
                "cv_evidence": str(cv),
            },
            indent=2,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lob-alpha")
    subparsers = parser.add_subparsers(dest="command", required=True)

    check = subparsers.add_parser("config-check", help="validate the research contract")
    check.add_argument("--config", default="configs/base.yaml")
    check.set_defaults(handler=_config_check)

    estimate = subparsers.add_parser(
        "estimate-cost", help="estimate Databento cost without downloading"
    )
    estimate.add_argument("--config", default="configs/sample_three_sessions.yaml")
    estimate.set_defaults(handler=_estimate)

    download = subparsers.add_parser("download", help="perform an explicitly cost-capped download")
    download.add_argument("--config", default="configs/sample_three_sessions.yaml")
    download.add_argument("--output", required=True)
    download.add_argument("--max-cost-usd", required=True, type=float)
    download.add_argument("--overwrite", action="store_true")
    download.set_defaults(handler=_download)

    batch_submit = subparsers.add_parser(
        "batch-submit", help="submit one cost-capped, daily-split Databento batch"
    )
    batch_submit.add_argument("--config", default="configs/base.yaml")
    batch_submit.add_argument("--max-cost-usd", required=True, type=float)
    batch_submit.add_argument("--confirm-paid-request", action="store_true", required=True)
    batch_submit.add_argument("--manifest")
    batch_submit.set_defaults(handler=_batch_submit)

    batch_status = subparsers.add_parser("batch-status", help="inspect a Databento batch job")
    batch_status.add_argument("--job-id", required=True)
    batch_status.set_defaults(handler=_batch_status)

    batch_fetch = subparsers.add_parser(
        "batch-download", help="download and hash-check a completed Databento batch"
    )
    batch_fetch.add_argument("--job-id", required=True)
    batch_fetch.add_argument("--output-dir", default="data/raw/databento")
    batch_fetch.add_argument("--manifest")
    batch_fetch.set_defaults(handler=_batch_download)

    batch_run = subparsers.add_parser(
        "batch-run", help="submit, wait for and hash-check one cost-capped daily batch"
    )
    batch_run.add_argument("--config", default="configs/base.yaml")
    batch_run.add_argument("--max-cost-usd", required=True, type=float)
    batch_run.add_argument("--confirm-paid-request", action="store_true", required=True)
    batch_run.add_argument("--output-dir", default="data/raw/databento")
    batch_run.add_argument("--poll-seconds", default=30.0, type=float)
    batch_run.add_argument("--timeout-minutes", default=240.0, type=float)
    batch_run.add_argument("--manifest")
    batch_run.set_defaults(handler=_batch_run)

    definitions = subparsers.add_parser(
        "download-definitions", help="download one UTC day of point-in-time definitions"
    )
    definitions.add_argument("--config", default="configs/sample_three_sessions.yaml")
    definitions.add_argument("--output", required=True)
    definitions.add_argument("--max-cost-usd", required=True, type=float)
    definitions.add_argument("--overwrite", action="store_true")
    definitions.set_defaults(handler=_download_definitions)

    verify = subparsers.add_parser(
        "verify-definition", help="verify tick size and contract multiplier"
    )
    verify.add_argument("--config", default="configs/base.yaml")
    verify.add_argument("--input", required=True)
    verify.add_argument("--symbol")
    verify.set_defaults(handler=_verify_definition)

    process = subparsers.add_parser("process-session", help="build one causal session dataset")
    process.add_argument("--config", default="configs/base.yaml")
    process.add_argument("--input", required=True)
    process.add_argument("--output", required=True)
    process.add_argument("--manifest")
    process.add_argument("--session-date", required=True)
    process.add_argument("--tick-size", type=float)
    process.set_defaults(handler=_process)

    process_all = subparsers.add_parser(
        "process-all", help="process every dated raw file and build a hashed daily catalog"
    )
    process_all.add_argument("--config", default="configs/base.yaml")
    process_all.add_argument("--raw-dir")
    process_all.add_argument("--output-dir")
    process_all.add_argument("--tick-size", type=float)
    process_all.add_argument("--output-format", choices=("parquet", "csv.gz"), default="parquet")
    process_all.add_argument("--overwrite", action="store_true")
    process_all.set_defaults(handler=_process_all)

    train_stage = subparsers.add_parser(
        "train-stage", help="run train-only diagnostics and expanding-window selection"
    )
    train_stage.add_argument("--config", default="configs/base.yaml")
    train_stage.add_argument("--catalog", default="data/processed/catalog.json")
    train_stage.add_argument("--output-dir", default="artifacts/train")
    train_stage.set_defaults(handler=_train_stage)

    validation_stage = subparsers.add_parser(
        "validation-stage", help="select one model and execution margin on validation"
    )
    validation_stage.add_argument("--config", default="configs/base.yaml")
    validation_stage.add_argument("--catalog", default="data/processed/catalog.json")
    validation_stage.add_argument(
        "--train-selection", default="artifacts/train/train_selection.json"
    )
    validation_stage.add_argument("--raw-dir")
    validation_stage.add_argument("--output-dir", default="artifacts/validation")
    validation_stage.set_defaults(handler=_validation_stage)

    freeze = subparsers.add_parser(
        "freeze-candidate", help="hash-lock the validation choice before holdout access"
    )
    freeze.add_argument("--config", default="configs/base.yaml")
    freeze.add_argument("--catalog", default="data/processed/catalog.json")
    freeze.add_argument("--candidate", default="artifacts/validation/selected_candidate.json")
    freeze.add_argument("--output", default="artifacts/frozen_candidate.json")
    freeze.set_defaults(handler=_freeze)

    holdout = subparsers.add_parser(
        "holdout-stage", help="run the frozen one-shot holdout and sensitivity grids"
    )
    holdout.add_argument("--config", default="configs/base.yaml")
    holdout.add_argument("--catalog", default="data/processed/catalog.json")
    holdout.add_argument("--frozen-candidate", default="artifacts/frozen_candidate.json")
    holdout.add_argument("--raw-dir")
    holdout.add_argument("--output-dir", default="artifacts/holdout")
    holdout.add_argument("--acknowledge-one-shot", action="store_true", required=True)
    holdout.set_defaults(handler=_holdout_stage)

    report = subparsers.add_parser(
        "build-report", help="generate figures, empirical report and CV evidence text"
    )
    report.add_argument("--train-dir", default="artifacts/train")
    report.add_argument("--validation-dir", default="artifacts/validation")
    report.add_argument("--holdout-dir", default="artifacts/holdout")
    report.add_argument("--reports-dir", default="reports")
    report.set_defaults(handler=_report)

    fixture = subparsers.add_parser("run-fixture", help="run the deterministic engineering fixture")
    fixture.add_argument("--config", default="configs/base.yaml")
    fixture.add_argument("--output-dir", default="artifacts/fixture")
    fixture.set_defaults(handler=_run_fixture)

    fixture_study = subparsers.add_parser(
        "run-fixture-study", help="dry-run the complete staged study on synthetic books"
    )
    fixture_study.add_argument("--config", default="configs/fixture_study.yaml")
    fixture_study.add_argument("--output-dir", default="artifacts/fixture-study")
    fixture_study.set_defaults(handler=_run_fixture_study)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
