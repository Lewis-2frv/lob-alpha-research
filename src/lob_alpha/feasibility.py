"""One-session-at-a-time engineering resource audit utilities."""

from __future__ import annotations

import gc
import math
from collections.abc import Callable
from pathlib import Path
from time import perf_counter
from typing import Any

import pandas as pd

from .acquisition import plan_session_requests
from .config import ResearchConfig
from .dataset import compact_session_dataset, discover_daily_raw_files
from .ingest import load_events
from .manifest import write_json
from .pipeline import process_session, write_table
from .sampling import filter_session
from .validation import DataQualityError, validate_mbp10

_REJECTION_FIELDS = (
    "exact_duplicate_rows",
    "timestamp_inversions",
    "invalid_best_quotes",
    "locked_books",
    "crossed_books",
    "negative_size_or_count",
    "ladder_errors",
    "tick_misaligned_rows",
)
_RESOURCE_FIELDS = (
    "compressed_raw_bytes",
    "decoded_event_rows",
    "dataframe_memory_bytes",
    "decode_wall_clock_seconds",
    "processing_wall_clock_seconds",
    "total_wall_clock_seconds",
    "processed_decision_rows",
    "processed_output_bytes",
)
_RESOURCE_LABELS = {
    "compressed_raw_bytes": "Compressed raw bytes",
    "decoded_event_rows": "Decoded event rows",
    "dataframe_memory_bytes": "Approximate pandas DataFrame bytes",
    "decode_wall_clock_seconds": "Decode wall-clock seconds",
    "processing_wall_clock_seconds": "Processing wall-clock seconds",
    "total_wall_clock_seconds": "Total wall-clock seconds",
    "processed_decision_rows": "Processed decision rows",
    "processed_output_bytes": "Processed output bytes",
}


def _empty_session_record(session_date: str, path: Path | None) -> dict[str, Any]:
    return {
        "session_date": session_date,
        "status": "missing" if path is None else "pending",
        "raw_path": None if path is None else str(path.resolve()),
        "processed_output_path": None,
        "compressed_raw_bytes": None if path is None else path.stat().st_size,
        "decoded_event_rows": None,
        "session_event_rows": None,
        "dataframe_memory_bytes": None,
        "decode_wall_clock_seconds": None,
        "processing_wall_clock_seconds": None,
        "total_wall_clock_seconds": None,
        "processed_decision_rows": None,
        "processed_output_bytes": None,
        "quality_accepted": False,
        "research_session_usable": False,
        "rejection_counts": None,
        "error": None,
    }


def _output_path(
    config: ResearchConfig,
    processed_dir: Path,
    session_date: str,
    output_format: str,
) -> Path:
    suffix = ".parquet" if output_format == "parquet" else ".csv.gz"
    return processed_dir / f"{config.data.symbols[0]}_{session_date}_feasibility{suffix}"


def _write_processed_atomic(frame: pd.DataFrame, output: Path) -> Path:
    lowered = output.name.lower()
    if lowered.endswith(".parquet"):
        temporary = output.with_name(output.stem + ".partial.parquet")
    else:
        temporary = output.with_name(output.name.removesuffix(".csv.gz") + ".partial.csv.gz")
    if temporary.exists():
        raise FileExistsError(f"stale feasibility output requires manual review: {temporary}")
    write_table(frame, temporary)
    temporary.replace(output)
    return output


def _aggregate(records: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    totals: dict[str, Any] = {}
    maxima: dict[str, Any] = {}
    for field in _RESOURCE_FIELDS:
        values = [record[field] for record in records if record[field] is not None]
        totals[field] = math.fsum(values) if "seconds" in field else sum(values)
        maxima[field] = max(values, default=0.0 if "seconds" in field else 0)
    return totals, maxima


def _format_mib(value: int | None) -> str:
    if value is None:
        return "-"
    return f"{value / (1024 * 1024):.2f}"


def _markdown_report(payload: dict[str, Any]) -> str:
    lines = [
        "# Three-session engineering resource audit",
        "",
        "This artifact measures acquisition-processing feasibility only. It does not calculate or "
        "report alpha, IC, P&L, hit rate, model selection, cross-validation, or holdout results.",
        "DataFrame memory values are pandas `memory_usage(deep=True)` estimates, not process "
        "peak RSS or operating-system peak RAM.",
        "",
        "| Session | Status | Raw MiB | Events | DataFrame MiB | Process s | Decisions | Quality |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for record in payload["sessions"]:
        processing_seconds = record["processing_wall_clock_seconds"]
        process_text = "-" if processing_seconds is None else f"{processing_seconds:.3f}"
        events_text = (
            record["decoded_event_rows"]
            if record["decoded_event_rows"] is not None
            else "-"
        )
        decisions_text = (
            record["processed_decision_rows"]
            if record["processed_decision_rows"] is not None
            else "-"
        )
        quality = "accepted" if record["quality_accepted"] else "not accepted"
        lines.append(
            f"| {record['session_date']} | {record['status']} | "
            f"{_format_mib(record['compressed_raw_bytes'])} | "
            f"{events_text} | "
            f"{_format_mib(record['dataframe_memory_bytes'])} | {process_text} | "
            f"{decisions_text} | "
            f"{quality} |"
        )

    counts = payload["counts"]
    lines.extend(
        [
            "",
            "## Decision-useful summary",
            "",
            f"- Planned sessions: {counts['planned_sessions']}",
            f"- Research-usable sessions: {counts['research_usable_sessions']}",
            f"- Missing session files: {counts['missing_sessions']}",
            "",
            "| Resource | Total | Single-session maximum |",
            "|---|---:|---:|",
            *[
                f"| {_RESOURCE_LABELS[field]} | {payload['totals'][field]} | "
                f"{payload['maxima'][field]} |"
                for field in _RESOURCE_FIELDS
            ],
            "",
            "Sessions are decoded and processed serially; totals are storage/work totals, while "
            "single-session maxima are the relevant laptop memory indicators.",
            "",
        ]
    )
    return "\n".join(lines)


def audit_session_resources(
    config: ResearchConfig,
    *,
    raw_dir: str | Path,
    processed_dir: str | Path,
    output_json: str | Path,
    output_markdown: str | Path,
    output_format: str = "parquet",
    overwrite: bool = False,
    loader: Callable[[str | Path], pd.DataFrame] = load_events,
    clock: Callable[[], float] = perf_counter,
) -> dict[str, Any]:
    """Audit planned sample files serially and write JSON plus concise Markdown."""

    if output_format not in {"parquet", "csv.gz"}:
        raise ValueError("output_format must be 'parquet' or 'csv.gz'")
    json_path = Path(output_json)
    markdown_path = Path(output_markdown)
    destination = Path(processed_dir)
    if not overwrite:
        for path in (json_path, markdown_path):
            if path.exists():
                raise FileExistsError(f"refusing to overwrite feasibility artifact: {path}")

    try:
        discovered = discover_daily_raw_files(raw_dir)
    except FileNotFoundError:
        discovered = {}
    planned = plan_session_requests(config)
    if not overwrite:
        for request in planned:
            output = _output_path(
                config,
                destination,
                request.session_date.isoformat(),
                output_format,
            )
            if output.exists():
                raise FileExistsError(f"refusing to overwrite processed feasibility file: {output}")

    records: list[dict[str, Any]] = []
    for request in planned:
        session_date = request.session_date.isoformat()
        raw_path = discovered.get(request.session_date)
        record = _empty_session_record(session_date, raw_path)
        records.append(record)
        if raw_path is None:
            continue

        events: pd.DataFrame | None = None
        session_events: pd.DataFrame | None = None
        compact: pd.DataFrame | None = None
        result = None
        processing_started: float | None = None
        session_started = clock()
        try:
            decode_started = clock()
            events = loader(raw_path)
            record["decode_wall_clock_seconds"] = clock() - decode_started
            record["decoded_event_rows"] = len(events)
            record["dataframe_memory_bytes"] = int(events.memory_usage(index=True, deep=True).sum())

            processing_started = clock()
            session_events = filter_session(events, request.session_date, config.session)
            record["session_event_rows"] = len(session_events)
            if session_events.empty:
                record["status"] = "no_session_data"
                record["error"] = "no events fall within the configured end-exclusive session"
            else:
                quality = validate_mbp10(
                    session_events,
                    tick_size=config.contract.expected_tick_size,
                )
                quality_dict = quality.to_dict()
                record["quality_accepted"] = quality.accepted
                record["rejection_counts"] = {
                    field: quality_dict[field] for field in _REJECTION_FIELDS
                }
                if not quality.accepted:
                    record["status"] = "rejected_quality"
                else:
                    result = process_session(
                        events,
                        config,
                        session_date=request.session_date,
                        tick_size=config.contract.expected_tick_size,
                    )
                    compact = compact_session_dataset(result.data, config)
                    record["processed_decision_rows"] = len(compact)
                    if compact.empty:
                        record["status"] = "no_processed_rows"
                    else:
                        output = _output_path(
                            config,
                            destination,
                            session_date,
                            output_format,
                        )
                        _write_processed_atomic(compact, output)
                        record["processed_output_path"] = str(output.resolve())
                        record["processed_output_bytes"] = output.stat().st_size
                        record["research_session_usable"] = True
                        record["status"] = "processed"
            record["processing_wall_clock_seconds"] = clock() - processing_started
        except (DataQualityError, KeyError, TypeError, ValueError, OSError) as exc:
            record["status"] = "error"
            record["error"] = f"{type(exc).__name__}: {exc}"
            record["research_session_usable"] = False
        finally:
            if (
                processing_started is not None
                and record["processing_wall_clock_seconds"] is None
            ):
                record["processing_wall_clock_seconds"] = clock() - processing_started
            record["total_wall_clock_seconds"] = clock() - session_started
            del result
            del compact
            del session_events
            del events
            gc.collect()

    totals, maxima = _aggregate(records)
    payload = {
        "artifact_type": "three_session_engineering_resource_audit",
        "engineering_only": True,
        "prohibited_research_outputs": [
            "alpha",
            "IC",
            "P&L",
            "hit_rate",
            "model_selection",
            "cross_validation",
            "holdout",
        ],
        "serial_processing": True,
        "sessions": records,
        "counts": {
            "planned_sessions": len(records),
            "files_found": sum(record["raw_path"] is not None for record in records),
            "quality_accepted_sessions": sum(record["quality_accepted"] for record in records),
            "research_usable_sessions": sum(
                record["research_session_usable"] for record in records
            ),
            "missing_sessions": sum(record["status"] == "missing" for record in records),
        },
        "totals": totals,
        "maxima": maxima,
    }
    write_json(json_path, payload)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_temporary = markdown_path.with_suffix(markdown_path.suffix + ".tmp")
    markdown_temporary.write_text(_markdown_report(payload), encoding="utf-8")
    markdown_temporary.replace(markdown_path)
    return payload
