"""Command-line interface for the equity-primary and optional futures workflows."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from datetime import date, timedelta
from pathlib import Path

from .acquisition import (
    download_planned_sessions,
    estimate_session_costs,
    write_session_cost_plan,
)
from .config import load_config
from .dataset import process_raw_directory
from .definitions import verify_contract_definition
from .equity_config import load_equity_config
from .equity_data import audit_optiver_csv, prepare_optiver_parquet
from .equity_fixture import write_synthetic_optiver
from .equity_reporting import build_equity_report
from .equity_study import (
    HOLDOUT_ACKNOWLEDGEMENT,
    freeze_equity_candidate,
    run_equity_holdout_stage,
    run_equity_train_stage,
    run_equity_validation_stage,
)
from .feasibility import audit_session_resources
from .fi2010_config import load_fi2010_config
from .fi2010_data import audit_inner_archive, import_inner_archive, verify_outer_archive
from .fi2010_fixture import run_synthetic_fi2010
from .fi2010_reporting import build_fi2010_report, publish_fi2010_portfolio
from .fi2010_study import (
    HOLDOUT_ACKNOWLEDGEMENT as FI2010_HOLDOUT_ACKNOWLEDGEMENT,
)
from .fi2010_study import (
    freeze_and_refit_fi2010,
    run_fi2010_development,
    run_fi2010_holdout,
)
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
from .safe_zip import extract_optiver_train_csv
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


def _estimate_sessions(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    plan = estimate_session_costs(config)
    payload = plan.to_dict(config)
    if args.output:
        output = write_session_cost_plan(args.output, plan, config=config)
        payload["planning_artifact"] = str(output.resolve())
    print(json.dumps(payload, indent=2))
    return 0


def _download_sessions(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    result = download_planned_sessions(
        config,
        args.output_dir,
        max_cost_usd=args.max_cost_usd,
        confirm_paid_request=args.confirm_paid_request,
        manifest_path=args.manifest,
    )
    print(json.dumps(result, indent=2))
    return 0


def _audit_resources(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    payload = audit_session_resources(
        config,
        raw_dir=args.raw_dir or config.data.raw_dir,
        processed_dir=args.processed_dir,
        output_json=args.output_json,
        output_markdown=args.output_markdown,
        output_format=args.output_format,
        overwrite=args.overwrite,
    )
    print(
        json.dumps(
            {
                "output_json": str(Path(args.output_json).resolve()),
                "output_markdown": str(Path(args.output_markdown).resolve()),
                "counts": payload["counts"],
                "maxima": payload["maxima"],
            },
            indent=2,
        )
    )
    return 0


def _download(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    output, cost = download_stream(
        config,
        args.output,
        max_cost_usd=args.max_cost_usd,
        confirm_paid_request=args.confirm_paid_request,
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
        confirm_paid_request=args.confirm_paid_request,
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


def _equity_audit(args: argparse.Namespace) -> int:
    config = load_equity_config(args.config)
    payload = audit_optiver_csv(
        config,
        input_path=args.input,
        output_path=args.output,
        metadata_only=args.metadata_only,
    )
    print(json.dumps(payload, indent=2))
    return 0


def _equity_prepare(args: argparse.Namespace) -> int:
    config = load_equity_config(args.config)
    payload = prepare_optiver_parquet(
        config,
        input_path=args.input,
        audit_path=args.audit,
        output_dir=args.output_dir,
    )
    print(
        json.dumps(
            {
                "prepared_manifest": str(
                    Path(args.output_dir or config.data.prepared_dir) / "prepared_manifest.json"
                ),
                "rows": payload["rows"],
                "partitions": len(payload["partitions"]),
            },
            indent=2,
        )
    )
    return 0


def _equity_train(args: argparse.Namespace) -> int:
    config = load_equity_config(args.config)
    payload = run_equity_train_stage(
        config,
        manifest_path=args.manifest,
        output_dir=args.output_dir,
    )
    print(json.dumps(payload, indent=2))
    return 0


def _equity_validate(args: argparse.Namespace) -> int:
    config = load_equity_config(args.config)
    payload = run_equity_validation_stage(
        config,
        manifest_path=args.manifest,
        train_selection_path=args.train_selection,
        output_dir=args.output_dir,
    )
    print(json.dumps(payload, indent=2))
    return 0


def _equity_freeze(args: argparse.Namespace) -> int:
    config = load_equity_config(args.config)
    payload = freeze_equity_candidate(
        config,
        manifest_path=args.manifest,
        candidate_path=args.candidate,
        output_path=args.output,
    )
    print(json.dumps(payload, indent=2))
    return 0


def _equity_holdout(args: argparse.Namespace) -> int:
    config = load_equity_config(args.config)
    payload = run_equity_holdout_stage(
        config,
        manifest_path=args.manifest,
        frozen_candidate_path=args.frozen_candidate,
        output_dir=args.output_dir,
        acknowledge_one_shot=args.acknowledge_one_shot,
    )
    print(json.dumps(payload, indent=2))
    return 0


def _equity_report(args: argparse.Namespace) -> int:
    report, cv = build_equity_report(
        train_dir=args.train_dir,
        validation_dir=args.validation_dir,
        holdout_dir=args.holdout_dir,
        reports_dir=args.reports_dir,
    )
    print(json.dumps({"report": str(report), "cv_evidence": str(cv)}, indent=2))
    return 0


def _equity_extract_zip(args: argparse.Namespace) -> int:
    output = extract_optiver_train_csv(args.zip, output_path=args.output)
    print(json.dumps({"output": str(output.resolve()), "network_access": False}, indent=2))
    return 0


def _equity_fixture_study(args: argparse.Namespace) -> int:
    config = load_equity_config(args.config)
    root = Path(args.output_dir)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"refusing to overwrite synthetic equity study: {root}")
    root.mkdir(parents=True, exist_ok=True)
    raw = write_synthetic_optiver(config, root / "synthetic_train.csv")
    audit_path = root / "schema_audit.json"
    audit_optiver_csv(
        config,
        input_path=raw,
        output_path=audit_path,
        metadata_only=False,
    )
    prepared = root / "prepared"
    prepare_optiver_parquet(
        config,
        input_path=raw,
        audit_path=audit_path,
        output_dir=prepared,
    )
    manifest = prepared / "prepared_manifest.json"
    train = root / "train"
    validation = root / "validation"
    holdout = root / "holdout"
    run_equity_train_stage(config, manifest_path=manifest, output_dir=train)
    run_equity_validation_stage(
        config,
        manifest_path=manifest,
        train_selection_path=train / "train_selection.json",
        output_dir=validation,
    )
    frozen = root / "frozen_candidate.json"
    freeze_equity_candidate(
        config,
        manifest_path=manifest,
        candidate_path=validation / "selected_candidate.json",
        output_path=frozen,
    )
    result = run_equity_holdout_stage(
        config,
        manifest_path=manifest,
        frozen_candidate_path=frozen,
        output_dir=holdout,
        acknowledge_one_shot=HOLDOUT_ACKNOWLEDGEMENT,
    )
    report, cv = build_equity_report(
        train_dir=train,
        validation_dir=validation,
        holdout_dir=holdout,
        reports_dir=root / "reports",
    )
    print(
        json.dumps(
            {
                "warning": "synthetic engineering fixture only; no empirical claims",
                "holdout_mechanics": result["stage"],
                "report": str(report),
                "cv_evidence": str(cv),
            },
            indent=2,
        )
    )
    return 0


def _fi2010_import(args: argparse.Namespace) -> int:
    config = load_fi2010_config(args.config)
    if args.verify_only:
        payload = verify_outer_archive(config, args.archive)
    else:
        payload = import_inner_archive(
            config,
            args.archive,
            prepared_dir=args.prepared_dir,
        )
    print(json.dumps(payload, indent=2))
    return 0


def _fi2010_audit(args: argparse.Namespace) -> int:
    config = load_fi2010_config(args.config)
    payload = audit_inner_archive(
        config,
        prepared_dir=args.prepared_dir,
        validate_payloads=not args.metadata_only,
    )
    print(json.dumps(payload, indent=2))
    return 0


def _fi2010_develop(args: argparse.Namespace) -> int:
    config = load_fi2010_config(args.config)
    payload = run_fi2010_development(
        config,
        prepared_dir=args.prepared_dir,
        output_dir=args.output_dir,
    )
    print(json.dumps(payload, indent=2))
    return 0


def _fi2010_freeze(args: argparse.Namespace) -> int:
    config = load_fi2010_config(args.config)
    payload = freeze_and_refit_fi2010(
        config,
        prepared_dir=args.prepared_dir,
        development_results=args.development_results,
        output_dir=args.output_dir,
    )
    print(json.dumps(payload, indent=2))
    return 0


def _fi2010_holdout(args: argparse.Namespace) -> int:
    config = load_fi2010_config(args.config)
    payload = run_fi2010_holdout(
        config,
        prepared_dir=args.prepared_dir,
        frozen_candidate=args.frozen_candidate,
        final_model_manifest=args.final_model_manifest,
        output_dir=args.output_dir,
        acknowledgement=args.acknowledgement,
    )
    print(json.dumps(payload, indent=2))
    return 0


def _fi2010_report(args: argparse.Namespace) -> int:
    config = load_fi2010_config(args.config)
    report, evidence = build_fi2010_report(
        config,
        prepared_dir=args.prepared_dir,
        development_results=args.development_results,
        freeze_dir=args.freeze_dir,
        holdout_dir=args.holdout_dir,
        output_dir=args.output_dir,
    )
    print(json.dumps({"report": str(report), "evidence": str(evidence)}, indent=2))
    return 0


def _fi2010_run_synthetic(args: argparse.Namespace) -> int:
    config = load_fi2010_config(args.config)
    payload = run_synthetic_fi2010(config, args.output_dir)
    print(json.dumps(payload, indent=2))
    return 0


def _fi2010_publish(args: argparse.Namespace) -> int:
    payload = publish_fi2010_portfolio(
        args.report_dir,
        repository_root=args.repository_root,
        output_dir=args.output_dir,
        require_claim_eligible=not args.allow_ineligible,
        require_holdout=not args.allow_development_only,
    )
    print(json.dumps(payload, indent=2))
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

    estimate_sessions = subparsers.add_parser(
        "estimate-session-costs",
        help="estimate each exact intraday session independently without downloading",
    )
    estimate_sessions.add_argument("--config", default="configs/sample_three_sessions.yaml")
    estimate_sessions.add_argument("--output")
    estimate_sessions.set_defaults(handler=_estimate_sessions)

    session_download = subparsers.add_parser(
        "download-sessions",
        help="download exact intraday sessions after aggregate cost and confirmation gates",
    )
    session_download.add_argument("--config", default="configs/sample_three_sessions.yaml")
    session_download.add_argument("--output-dir", default="data/raw/databento")
    session_download.add_argument("--manifest")
    session_download.add_argument("--max-cost-usd", required=True, type=float)
    session_download.add_argument("--confirm-paid-request", action="store_true")
    session_download.set_defaults(handler=_download_sessions)

    resource_audit = subparsers.add_parser(
        "audit-session-resources",
        help="audit downloaded sample sessions serially without research-performance claims",
    )
    resource_audit.add_argument("--config", default="configs/sample_three_sessions.yaml")
    resource_audit.add_argument("--raw-dir")
    resource_audit.add_argument(
        "--processed-dir",
        default="artifacts/feasibility/processed",
    )
    resource_audit.add_argument(
        "--output-json",
        default="artifacts/feasibility/resource_audit.json",
    )
    resource_audit.add_argument(
        "--output-markdown",
        default="artifacts/feasibility/resource_audit.md",
    )
    resource_audit.add_argument(
        "--output-format",
        choices=("parquet", "csv.gz"),
        default="parquet",
    )
    resource_audit.add_argument("--overwrite", action="store_true")
    resource_audit.set_defaults(handler=_audit_resources)

    download = subparsers.add_parser("download", help="perform an explicitly cost-capped download")
    download.add_argument("--config", default="configs/sample_three_sessions.yaml")
    download.add_argument("--output", required=True)
    download.add_argument("--max-cost-usd", required=True, type=float)
    download.add_argument("--confirm-paid-request", action="store_true")
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
    definitions.add_argument("--confirm-paid-request", action="store_true")
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

    equity_audit = subparsers.add_parser(
        "equity-audit",
        help="validate Optiver schema/identifiers without producing performance metrics",
    )
    equity_audit.add_argument("--config", default="configs/equity_close.yaml")
    equity_audit.add_argument("--input")
    equity_audit.add_argument("--output", default="data/interim/optiver_schema_audit.json")
    equity_audit.add_argument(
        "--metadata-only",
        action="store_true",
        help="exclude target from CSV reads; suitable for confirming the registration",
    )
    equity_audit.set_defaults(handler=_equity_audit)

    equity_prepare = subparsers.add_parser(
        "equity-prepare", help="stream validated Optiver CSV into causal per-date Parquet"
    )
    equity_prepare.add_argument("--config", default="configs/equity_close.yaml")
    equity_prepare.add_argument("--input")
    equity_prepare.add_argument("--audit", default="data/interim/optiver_schema_audit.json")
    equity_prepare.add_argument("--output-dir")
    equity_prepare.set_defaults(handler=_equity_prepare)

    equity_train = subparsers.add_parser(
        "equity-train", help="run train-only diagnostics and expanding-window CV"
    )
    equity_train.add_argument("--config", default="configs/equity_close.yaml")
    equity_train.add_argument("--manifest", default="data/processed/optiver/prepared_manifest.json")
    equity_train.add_argument("--output-dir", default="artifacts/equity/train")
    equity_train.set_defaults(handler=_equity_train)

    equity_validate = subparsers.add_parser(
        "equity-validate", help="select the equity model and execution rule on validation"
    )
    equity_validate.add_argument("--config", default="configs/equity_close.yaml")
    equity_validate.add_argument(
        "--manifest", default="data/processed/optiver/prepared_manifest.json"
    )
    equity_validate.add_argument(
        "--train-selection", default="artifacts/equity/train/train_selection.json"
    )
    equity_validate.add_argument("--output-dir", default="artifacts/equity/validation")
    equity_validate.set_defaults(handler=_equity_validate)

    equity_freeze = subparsers.add_parser(
        "equity-freeze", help="content-lock model, preprocessing and execution before holdout"
    )
    equity_freeze.add_argument("--config", default="configs/equity_close.yaml")
    equity_freeze.add_argument(
        "--manifest", default="data/processed/optiver/prepared_manifest.json"
    )
    equity_freeze.add_argument(
        "--candidate", default="artifacts/equity/validation/selected_candidate.json"
    )
    equity_freeze.add_argument("--output", default="artifacts/equity/frozen_candidate.json")
    equity_freeze.set_defaults(handler=_equity_freeze)

    equity_holdout = subparsers.add_parser(
        "equity-holdout", help="run the explicitly acknowledged one-shot equity holdout"
    )
    equity_holdout.add_argument("--config", default="configs/equity_close.yaml")
    equity_holdout.add_argument(
        "--manifest", default="data/processed/optiver/prepared_manifest.json"
    )
    equity_holdout.add_argument(
        "--frozen-candidate", default="artifacts/equity/frozen_candidate.json"
    )
    equity_holdout.add_argument("--output-dir", default="artifacts/equity/holdout")
    equity_holdout.add_argument(
        "--acknowledge-one-shot",
        required=True,
        metavar="EXACT_PHRASE",
        help=f"must exactly equal {HOLDOUT_ACKNOWLEDGEMENT!r}",
    )
    equity_holdout.set_defaults(handler=_equity_holdout)

    equity_report = subparsers.add_parser(
        "equity-report", help="generate the claim-gated equity report and CV evidence"
    )
    equity_report.add_argument("--train-dir", default="artifacts/equity/train")
    equity_report.add_argument("--validation-dir", default="artifacts/equity/validation")
    equity_report.add_argument("--holdout-dir", default="artifacts/equity/holdout")
    equity_report.add_argument("--reports-dir", default="artifacts/equity/reports")
    equity_report.set_defaults(handler=_equity_report)

    equity_extract = subparsers.add_parser(
        "equity-extract-zip", help="safely extract only train.csv from a manually downloaded ZIP"
    )
    equity_extract.add_argument("--zip", required=True)
    equity_extract.add_argument("--output", default="data/raw/optiver/train.csv")
    equity_extract.set_defaults(handler=_equity_extract_zip)

    equity_fixture = subparsers.add_parser(
        "equity-run-synthetic",
        help="exercise the complete equity workflow on unmistakably synthetic data",
    )
    equity_fixture.add_argument("--config", default="configs/equity_close_fixture.yaml")
    equity_fixture.add_argument("--output-dir", default="artifacts/equity-fixture")
    equity_fixture.set_defaults(handler=_equity_fixture_study)

    default_fi2010_archive = str(Path.home() / "Downloads" / "FI-2010-official.zip")
    fi_import = subparsers.add_parser(
        "fi2010-import",
        help="verify the official outer archive and atomically import only its inner ZIP",
    )
    fi_import.add_argument("--config", default="configs/fi2010.yaml")
    fi_import.add_argument("--archive", default=default_fi2010_archive)
    fi_import.add_argument("--prepared-dir")
    fi_import.add_argument("--verify-only", action="store_true")
    fi_import.set_defaults(handler=_fi2010_import)

    fi_audit = subparsers.add_parser(
        "fi2010-audit",
        help="audit paired development members and Train_CF_9 without opening Test_CF_9",
    )
    fi_audit.add_argument("--config", default="configs/fi2010.yaml")
    fi_audit.add_argument("--prepared-dir")
    fi_audit.add_argument(
        "--metadata-only",
        action="store_true",
        help="validate central metadata only; normal audit also validates development payloads",
    )
    fi_audit.set_defaults(handler=_fi2010_audit)

    fi_develop = subparsers.add_parser(
        "fi2010-develop",
        help="run independent paired anchored development folds CF_1 through CF_8",
    )
    fi_develop.add_argument("--config", default="configs/fi2010.yaml")
    fi_develop.add_argument("--prepared-dir")
    fi_develop.add_argument("--output-dir", default="artifacts/fi2010/development")
    fi_develop.set_defaults(handler=_fi2010_develop)

    fi_freeze = subparsers.add_parser(
        "fi2010-freeze",
        help="freeze the development choice and refit it on Train_CF_9 only",
    )
    fi_freeze.add_argument("--config", default="configs/fi2010.yaml")
    fi_freeze.add_argument("--prepared-dir")
    fi_freeze.add_argument(
        "--development-results",
        default="artifacts/fi2010/development/development_results.json",
    )
    fi_freeze.add_argument("--output-dir", default="artifacts/fi2010/freeze")
    fi_freeze.set_defaults(handler=_fi2010_freeze)

    fi_holdout = subparsers.add_parser(
        "fi2010-holdout",
        help="irreversibly release the source-bound one-shot CF_9 final holdout",
    )
    fi_holdout.add_argument("--config", default="configs/fi2010.yaml")
    fi_holdout.add_argument("--prepared-dir")
    fi_holdout.add_argument(
        "--frozen-candidate",
        default="artifacts/fi2010/freeze/frozen_candidate.json",
    )
    fi_holdout.add_argument(
        "--final-model-manifest",
        default="artifacts/fi2010/freeze/final_model_manifest.json",
    )
    fi_holdout.add_argument("--output-dir", default="artifacts/fi2010/holdout")
    fi_holdout.add_argument(
        "--acknowledgement",
        required=True,
        metavar="EXACT_PHRASE",
        help=f"must exactly equal {FI2010_HOLDOUT_ACKNOWLEDGEMENT!r}",
    )
    fi_holdout.set_defaults(handler=_fi2010_holdout)

    fi_report = subparsers.add_parser(
        "fi2010-report",
        help="build an integrity-gated predictive evidence report",
    )
    fi_report.add_argument("--config", default="configs/fi2010.yaml")
    fi_report.add_argument("--prepared-dir")
    fi_report.add_argument(
        "--development-results",
        default="artifacts/fi2010/development/development_results.json",
    )
    fi_report.add_argument("--freeze-dir", default="artifacts/fi2010/freeze")
    fi_report.add_argument("--holdout-dir", default="artifacts/fi2010/holdout")
    fi_report.add_argument("--output-dir", default="artifacts/fi2010/report")
    fi_report.set_defaults(handler=_fi2010_report)

    fi_fixture = subparsers.add_parser(
        "fi2010-run-synthetic",
        help="rehearse the full nested-ZIP and one-shot workflow on claim-ineligible data",
    )
    fi_fixture.add_argument("--config", default="configs/fi2010.yaml")
    fi_fixture.add_argument("--output-dir", default="artifacts/fi2010-synthetic")
    fi_fixture.set_defaults(handler=_fi2010_run_synthetic)

    fi_publish = subparsers.add_parser(
        "fi2010-publish",
        help="publish validated small FI-2010 evidence and charts into docs/results/fi2010",
    )
    fi_publish.add_argument("--report-dir", default="artifacts/fi2010/report-final")
    fi_publish.add_argument("--repository-root", default=".")
    fi_publish.add_argument("--output-dir", default="docs/results/fi2010")
    fi_publish.add_argument(
        "--allow-development-only",
        action="store_true",
        help="publish development evidence without a final holdout; not the default portfolio mode",
    )
    fi_publish.add_argument(
        "--allow-ineligible",
        action="store_true",
        help="testing only: permit synthetic or otherwise claim-ineligible evidence",
    )
    fi_publish.set_defaults(handler=_fi2010_publish)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
