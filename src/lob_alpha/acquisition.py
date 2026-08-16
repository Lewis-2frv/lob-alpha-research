"""DST-safe, per-session Databento acquisition planning and download safety."""

from __future__ import annotations

import gc
import json
import math
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from .config import ResearchConfig
from .ingest import (
    CostLimitError,
    PaidRequestConfirmationError,
    _close_dbn_store,
    _validate_cost_ceiling,
    historical_client,
)
from .manifest import sha256_file, write_json
from .sampling import session_bounds


@dataclass(frozen=True)
class SessionRequest:
    """One exact, end-exclusive Databento intraday session request."""

    session_date: date
    start_utc: pd.Timestamp
    end_utc: pd.Timestamp

    def parameters(self, config: ResearchConfig) -> dict[str, Any]:
        return {
            "dataset": config.data.dataset,
            "symbols": list(config.data.symbols),
            "schema": config.data.schema,
            "stype_in": config.data.stype_in,
            "start": self.start_utc.isoformat(),
            "end": self.end_utc.isoformat(),
        }


@dataclass(frozen=True)
class SessionCostEstimate:
    request: SessionRequest
    estimated_cost_usd: float


@dataclass(frozen=True)
class SessionCostPlan:
    """Provider estimates for all independently priced session requests."""

    estimates: tuple[SessionCostEstimate, ...]

    @property
    def total_estimated_cost_usd(self) -> float:
        return math.fsum(item.estimated_cost_usd for item in self.estimates)

    def to_dict(self, config: ResearchConfig) -> dict[str, Any]:
        return {
            "artifact_type": "databento_intraday_session_cost_plan",
            "dataset": config.data.dataset,
            "schema": config.data.schema,
            "symbols": list(config.data.symbols),
            "stype_in": config.data.stype_in,
            "session_timezone": config.session.timezone,
            "session_start_time": config.session.start_time.isoformat(),
            "session_end_time_exclusive": config.session.end_time.isoformat(),
            "sessions": [
                {
                    "session_date": item.request.session_date.isoformat(),
                    "start_utc": item.request.start_utc.isoformat(),
                    "end_utc_exclusive": item.request.end_utc.isoformat(),
                    "estimated_cost_usd": item.estimated_cost_usd,
                    "request": item.request.parameters(config),
                }
                for item in self.estimates
            ],
            "total_sessions": len(self.estimates),
            "total_estimated_cost_usd": self.total_estimated_cost_usd,
            "paid_request_made": False,
        }


def _as_utc_timestamp(value: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def plan_session_requests(config: ResearchConfig) -> tuple[SessionRequest, ...]:
    """Plan complete weekdays contained in the configured end-exclusive range."""

    configured_start = _as_utc_timestamp(config.data.start)
    configured_end = _as_utc_timestamp(config.data.end)
    if configured_start >= configured_end:
        raise ValueError("configured data start must precede its exclusive end")

    requests: list[SessionRequest] = []
    current = configured_start.date()
    final_candidate = configured_end.date()
    while current <= final_candidate:
        start_utc, end_utc = session_bounds(current, config.session)
        if (
            current.weekday() < 5
            and start_utc >= configured_start
            and end_utc <= configured_end
        ):
            requests.append(SessionRequest(current, start_utc, end_utc))
        current += timedelta(days=1)
    return tuple(requests)


def estimate_session_costs(
    config: ResearchConfig,
    *,
    client: Any | None = None,
) -> SessionCostPlan:
    """Estimate every exact intraday session independently without downloading data."""

    requests = plan_session_requests(config)
    return _estimate_requests(config, requests, client=client)


def _estimate_requests(
    config: ResearchConfig,
    requests: Sequence[SessionRequest],
    *,
    client: Any | None = None,
) -> SessionCostPlan:
    if not requests:
        return SessionCostPlan(())
    active_client = client or historical_client()
    estimates: list[SessionCostEstimate] = []
    for request in requests:
        cost = float(active_client.metadata.get_cost(**request.parameters(config)))
        if not math.isfinite(cost) or cost < 0:
            raise ValueError(
                f"provider returned an invalid cost estimate for {request.session_date}"
            )
        estimates.append(SessionCostEstimate(request=request, estimated_cost_usd=cost))
    return SessionCostPlan(tuple(estimates))


def write_session_cost_plan(
    path: str | Path,
    plan: SessionCostPlan,
    *,
    config: ResearchConfig,
) -> Path:
    """Write the stable estimate-only planning structure as sorted JSON."""

    return write_json(path, plan.to_dict(config))


def session_raw_filename(config: ResearchConfig, session_date: date) -> str:
    symbol_component = "-".join(config.data.symbols)
    return f"{symbol_component}_{session_date.isoformat()}_{config.data.schema}.dbn.zst"


def validate_local_dbn(
    path: str | Path,
    *,
    expected_schema: str,
    scan_records: bool = False,
) -> None:
    """Reopen a local DBN/Zstandard file and optionally scan every record in chunks."""

    from .ingest import _databento_module

    input_path = Path(path)
    if scan_records:
        try:
            import zstandard
        except ImportError as exc:
            raise RuntimeError("zstandard is required to validate compressed DBN files") from exc
        # Validate the complete Zstandard frame (including its checksum) before
        # asking DBNStore to open it. This also avoids an SDK file-handle leak
        # when a crash left a non-Zstandard partial file.
        with (
            input_path.open("rb") as compressed,
            zstandard.ZstdDecompressor().stream_reader(compressed) as reader,
        ):
            while reader.read(1024 * 1024):
                pass

    store = _databento_module().DBNStore.from_file(input_path)
    base_reader = store._data_source.reader
    iterator = None
    try:
        if str(store.compression) != "zstd":
            raise ValueError(f"expected Zstandard-compressed DBN file: {path}")
        if str(store.schema) != expected_schema:
            raise ValueError(
                f"DBN schema mismatch for {path}: expected {expected_schema}, got {store.schema}"
            )
        if scan_records:
            iterator = iter(store)
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                for _ in iterator:
                    pass
    finally:
        if iterator is not None:
            iterator.close()
        base_reader.close()
        del store
        gc.collect()


def _load_manifest(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("artifact_type") != "databento_intraday_session_acquisition":
        raise ValueError(f"unexpected acquisition manifest format: {path}")
    return payload


def _matching_manifest_record(
    existing: dict[str, Any] | None,
    *,
    session_date: str,
) -> dict[str, Any] | None:
    if existing is None:
        return None
    matches = [
        item
        for item in existing.get("sessions", [])
        if item.get("session_date") == session_date
    ]
    if len(matches) > 1:
        raise ValueError(f"duplicate session records in acquisition manifest: {session_date}")
    return dict(matches[0]) if matches else None


def _validate_manifest_scope(
    existing: dict[str, Any] | None,
    *,
    config: ResearchConfig,
    planned_dates: set[str],
) -> None:
    if existing is None:
        return
    expected = {
        "dataset": config.data.dataset,
        "schema": config.data.schema,
        "symbols": list(config.data.symbols),
        "stype_in": config.data.stype_in,
        "encoding": "dbn",
        "compression": "zstd",
        "session_timezone": config.session.timezone,
    }
    for field, value in expected.items():
        if existing.get(field) != value:
            raise ValueError(f"acquisition manifest {field} does not match the active config")
    existing_dates = {str(item.get("session_date")) for item in existing.get("sessions", [])}
    unexpected_dates = sorted(existing_dates - planned_dates)
    if unexpected_dates:
        raise ValueError(
            f"acquisition manifest contains sessions outside the active plan: {unexpected_dates}"
        )


def _validate_record_identity(
    record: dict[str, Any],
    *,
    request: dict[str, Any],
    final_path: Path,
    partial_path: Path,
) -> None:
    if record.get("request") != request:
        raise ValueError(f"manifest request changed for existing session file: {final_path}")
    if record.get("local_path") != str(final_path.resolve()):
        raise ValueError(f"manifest path does not match existing session file: {final_path}")
    if record.get("temporary_path") != str(partial_path.resolve()):
        raise ValueError(f"manifest temporary path does not match session file: {partial_path}")
    if record.get("encoding") != "dbn" or record.get("compression") != "zstd":
        raise ValueError(f"manifest DBN/Zstandard format changed for session file: {final_path}")


def _validated_complete_record(
    record: dict[str, Any],
    *,
    request: dict[str, Any],
    final_path: Path,
    partial_path: Path,
    expected_schema: str,
    validator: Callable[..., None],
) -> dict[str, Any]:
    _validate_record_identity(
        record,
        request=request,
        final_path=final_path,
        partial_path=partial_path,
    )
    if record.get("status") != "complete" or record.get("download_complete") is not True:
        raise FileExistsError(f"existing raw file is not recorded complete: {final_path}")
    actual_bytes = final_path.stat().st_size
    actual_sha256 = sha256_file(final_path)
    if record.get("raw_bytes") != actual_bytes or record.get("sha256") != actual_sha256:
        raise OSError(f"existing session file no longer matches its manifest: {final_path}")
    validator(final_path, expected_schema=expected_schema, scan_records=False)
    return record


def _manifest_payload(
    config: ResearchConfig,
    *,
    plan: SessionCostPlan,
    max_cost_usd: float,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "artifact_type": "databento_intraday_session_acquisition",
        "dataset": config.data.dataset,
        "schema": config.data.schema,
        "symbols": list(config.data.symbols),
        "stype_in": config.data.stype_in,
        "encoding": "dbn",
        "compression": "zstd",
        "session_timezone": config.session.timezone,
        "session_end_is_exclusive": True,
        "max_cost_usd": max_cost_usd,
        "sessions_estimated_this_run": len(plan.estimates),
        "estimated_cost_usd_this_run": plan.total_estimated_cost_usd,
        "total_sessions": len(records),
        "complete_sessions": sum(item["status"] == "complete" for item in records),
        "sessions": records,
    }


def download_planned_sessions(
    config: ResearchConfig,
    output_dir: str | Path,
    *,
    max_cost_usd: float,
    confirm_paid_request: bool,
    manifest_path: str | Path | None = None,
    client: Any | None = None,
    dbn_validator: Callable[..., None] = validate_local_dbn,
) -> dict[str, Any]:
    """Download exact sessions with aggregate cost, confirmation, and resume gates.

    All estimates and local-file preflight checks complete before the first paid
    timeseries request. A final file is resumable only when its manifest request,
    size, and SHA-256 all match. An interrupted ``downloading`` state is never
    silently retried because the earlier provider call may already have incurred cost.
    """

    _validate_cost_ceiling(max_cost_usd)
    requests = plan_session_requests(config)
    destination = Path(output_dir)
    acquisition_manifest = Path(
        manifest_path or destination / "intraday_session_acquisition.manifest.json"
    )
    existing_manifest = _load_manifest(acquisition_manifest)
    _validate_manifest_scope(
        existing_manifest,
        config=config,
        planned_dates={request.session_date.isoformat() for request in requests},
    )
    records: list[dict[str, Any]] = []
    recovery_actions: list[tuple[dict[str, Any], Path, Path]] = []

    # Resolve every local-file ambiguity before estimating or allowing a paid call.
    for session_request in requests:
        session_date = session_request.session_date.isoformat()
        request = session_request.parameters(config)
        final_path = destination / session_raw_filename(config, session_request.session_date)
        partial_path = final_path.with_name(final_path.name + ".partial")
        existing_record = _matching_manifest_record(
            existing_manifest,
            session_date=session_date,
        )
        if existing_record is not None:
            _validate_record_identity(
                existing_record,
                request=request,
                final_path=final_path,
                partial_path=partial_path,
            )
        if final_path.exists():
            if existing_record is None:
                raise FileExistsError(
                    f"refusing to overwrite or trust unmanifested raw file: {final_path}"
                )
            if partial_path.exists():
                raise FileExistsError(f"stale partial file requires manual review: {partial_path}")
            if existing_record.get("status") == "complete":
                records.append(
                    _validated_complete_record(
                        existing_record,
                        request=request,
                        final_path=final_path,
                        partial_path=partial_path,
                        expected_schema=config.data.schema,
                        validator=dbn_validator,
                    )
                )
            elif existing_record.get("status") in {"downloading", "verified"}:
                records.append(existing_record)
                recovery_actions.append((existing_record, final_path, partial_path))
            else:
                raise FileExistsError(
                    f"existing raw file has an ambiguous manifest state: {final_path}"
                )
            continue

        if partial_path.exists():
            if existing_record is None or existing_record.get("status") not in {
                "downloading",
                "verified",
            }:
                raise FileExistsError(
                    f"untrusted partial file requires manual review: {partial_path}"
                )
            records.append(existing_record)
            recovery_actions.append((existing_record, final_path, partial_path))
            continue

        if existing_record is not None:
            if existing_record.get("status") in {"complete", "downloading", "verified"}:
                raise RuntimeError(
                    f"session {session_date} may already have incurred cost; "
                    "refusing automatic retry"
                )
            existing_record["status"] = "planned"
            existing_record["download_complete"] = False
            records.append(existing_record)
        else:
            records.append(
                {
                    "session_date": session_date,
                    "request": request,
                    "encoding": "dbn",
                    "compression": "zstd",
                    "estimated_cost_usd": None,
                    "local_path": str(final_path.resolve()),
                    "temporary_path": str(partial_path.resolve()),
                    "status": "planned",
                    "download_complete": False,
                    "raw_bytes": None,
                    "sha256": None,
                }
            )

    destination.mkdir(parents=True, exist_ok=True)
    acquisition_manifest.parent.mkdir(parents=True, exist_ok=True)

    def persist(active_plan: SessionCostPlan) -> None:
        write_json(
            acquisition_manifest,
            _manifest_payload(
                config,
                plan=active_plan,
                max_cost_usd=max_cost_usd,
                records=records,
            ),
        )

    # A complete DBN left by a crash is recovered locally and never re-requested.
    recovered_sessions_this_run = 0
    for record, final_path, partial_path in recovery_actions:
        recovery_path = final_path if final_path.exists() else partial_path
        try:
            dbn_validator(
                recovery_path,
                expected_schema=config.data.schema,
                scan_records=True,
            )
        except Exception as exc:
            raise RuntimeError(
                f"interrupted session file is not a complete DBN stream; preserving it "
                f"without retry: {recovery_path}"
            ) from exc
        actual_bytes = recovery_path.stat().st_size
        actual_sha256 = sha256_file(recovery_path)
        if record.get("status") == "verified" and (
            record.get("raw_bytes") != actual_bytes or record.get("sha256") != actual_sha256
        ):
            raise OSError(f"verified interrupted file changed on disk: {recovery_path}")
        record["raw_bytes"] = actual_bytes
        record["sha256"] = actual_sha256
        record["status"] = "verified"
        record["download_complete"] = False
        persist(SessionCostPlan(()))
        if recovery_path == partial_path:
            if final_path.exists():
                raise FileExistsError(f"raw session appeared during recovery: {final_path}")
            partial_path.rename(final_path)
        dbn_validator(
            final_path,
            expected_schema=config.data.schema,
            scan_records=False,
        )
        record["status"] = "complete"
        record["download_complete"] = True
        persist(SessionCostPlan(()))
        recovered_sessions_this_run += 1

    missing_records = [record for record in records if record["status"] == "planned"]
    request_by_date = {request.session_date.isoformat(): request for request in requests}
    active_client = (client or historical_client()) if missing_records else client
    plan = _estimate_requests(
        config,
        [request_by_date[record["session_date"]] for record in missing_records],
        client=active_client,
    )
    estimate_by_date = {
        estimate.request.session_date.isoformat(): estimate for estimate in plan.estimates
    }
    for record in missing_records:
        record["estimated_cost_usd"] = estimate_by_date[
            record["session_date"]
        ].estimated_cost_usd

    # Every missing session has now been estimated; no time-series call has occurred.
    if plan.total_estimated_cost_usd > max_cost_usd:
        raise CostLimitError(
            f"aggregate missing-session estimate ${plan.total_estimated_cost_usd:.4f} exceeds "
            f"the explicit ${max_cost_usd:.4f} cap"
        )
    if plan.estimates and not confirm_paid_request:
        raise PaidRequestConfirmationError(
            "session downloads require confirm_paid_request=True in addition to the cost cap"
        )
    persist(plan)

    paid_requests_this_run = 0
    for record in records:
        if record["status"] != "planned":
            continue
        final_path = Path(record["local_path"])
        partial_path = Path(record["temporary_path"])
        record["status"] = "downloading"
        persist(plan)
        paid_requests_this_run += 1
        if active_client is None:  # pragma: no cover - impossible for a nonempty plan
            raise RuntimeError("Databento client is unavailable for a planned request")
        downloaded_store = active_client.timeseries.get_range(
            **record["request"],
            path=str(partial_path),
        )
        _close_dbn_store(downloaded_store)
        del downloaded_store
        gc.collect()
        if not partial_path.is_file() or partial_path.stat().st_size <= 0:
            raise OSError(f"Databento did not produce a nonempty session file: {partial_path}")
        dbn_validator(
            partial_path,
            expected_schema=config.data.schema,
            scan_records=True,
        )
        record["raw_bytes"] = partial_path.stat().st_size
        record["sha256"] = sha256_file(partial_path)
        record["status"] = "verified"
        record["download_complete"] = False
        persist(plan)
        if final_path.exists():
            raise FileExistsError(
                f"raw session appeared during download; preserving partial file: {final_path}"
            )
        partial_path.rename(final_path)
        dbn_validator(
            final_path,
            expected_schema=config.data.schema,
            scan_records=False,
        )
        record["status"] = "complete"
        record["download_complete"] = True
        persist(plan)

    result = _manifest_payload(
        config,
        plan=plan,
        max_cost_usd=max_cost_usd,
        records=records,
    )
    result["manifest_path"] = str(acquisition_manifest.resolve())
    result["paid_requests_this_run"] = paid_requests_this_run
    result["skipped_complete_this_run"] = len(records) - len(plan.estimates)
    result["recovered_sessions_this_run"] = recovered_sessions_this_run
    return result
