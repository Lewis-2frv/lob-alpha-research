"""Databento request gating and MBP-10 loading."""

from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path
from typing import Any

import pandas as pd

from .config import ResearchConfig


class DataDependencyError(RuntimeError):
    """Raised when the optional Databento dependency or credentials are absent."""


class CostLimitError(RuntimeError):
    """Raised when a data request would exceed an explicit user cost ceiling."""


class PaidRequestConfirmationError(RuntimeError):
    """Raised when a paid batch submission lacks an explicit confirmation flag."""


def _databento_module() -> Any:
    try:
        import databento as db
    except ImportError as exc:
        raise DataDependencyError(
            'Databento support is optional. Install with: pip install -e ".[data]"'
        ) from exc
    return db


def historical_client() -> Any:
    """Create a client that reads DATABENTO_API_KEY from the environment."""

    if not os.getenv("DATABENTO_API_KEY"):
        raise DataDependencyError("DATABENTO_API_KEY is not set")
    return _databento_module().Historical()


def request_parameters(
    config: ResearchConfig,
    *,
    schema: str | None = None,
    start: str | None = None,
    end: str | None = None,
) -> dict[str, Any]:
    return {
        "dataset": config.data.dataset,
        "symbols": list(config.data.symbols),
        "schema": schema or config.data.schema,
        "stype_in": config.data.stype_in,
        "start": start or config.data.start,
        "end": end or config.data.end,
    }


def estimate_cost(
    config: ResearchConfig,
    *,
    schema: str | None = None,
    start: str | None = None,
    end: str | None = None,
    client: Any | None = None,
) -> float:
    """Return the provider's pre-download USD cost estimate."""

    active_client = client or historical_client()
    return float(
        active_client.metadata.get_cost(
            **request_parameters(config, schema=schema, start=start, end=end)
        )
    )


def download_stream(
    config: ResearchConfig,
    output_path: str | Path,
    *,
    max_cost_usd: float,
    schema: str | None = None,
    start: str | None = None,
    end: str | None = None,
    overwrite: bool = False,
    client: Any | None = None,
) -> tuple[Path, float]:
    """Download one explicitly cost-capped request as compressed DBN.

    The caller must provide a finite cost ceiling. This prevents accidental large
    paid requests when dates or symbology are wrong.
    """

    _validate_cost_ceiling(max_cost_usd)
    path = Path(output_path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing data: {path}")

    active_client = client or historical_client()
    cost = estimate_cost(
        config,
        schema=schema,
        start=start,
        end=end,
        client=active_client,
    )
    if cost > max_cost_usd:
        raise CostLimitError(
            f"estimated request cost ${cost:.4f} exceeds the explicit ${max_cost_usd:.4f} cap"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    store = active_client.timeseries.get_range(
        **request_parameters(config, schema=schema, start=start, end=end),
        path=str(path),
    )
    if not path.exists():
        store.to_file(path, mode="w" if overwrite else "x")
    return path, cost


def _validate_cost_ceiling(max_cost_usd: float) -> None:
    if not math.isfinite(max_cost_usd) or max_cost_usd < 0:
        raise ValueError("max_cost_usd must be finite and nonnegative")


def submit_batch_job(
    config: ResearchConfig,
    *,
    max_cost_usd: float,
    confirm_paid_request: bool,
    client: Any | None = None,
) -> tuple[dict[str, Any], float]:
    """Submit one daily-split DBN batch after two independent safety gates.

    Cost is re-estimated immediately before submission. The boolean confirmation
    is deliberately separate from the numeric cap, which makes accidental paid
    calls from scripts harder.
    """

    _validate_cost_ceiling(max_cost_usd)
    if not confirm_paid_request:
        raise PaidRequestConfirmationError(
            "batch submission requires confirm_paid_request=True"
        )
    active_client = client or historical_client()
    cost = estimate_cost(config, client=active_client)
    if cost > max_cost_usd:
        raise CostLimitError(
            f"estimated request cost ${cost:.4f} exceeds the explicit ${max_cost_usd:.4f} cap"
        )
    details = active_client.batch.submit_job(
        **request_parameters(config),
        encoding="dbn",
        compression="zstd",
        split_duration="day",
        delivery="download",
    )
    return dict(details), cost


def batch_job_status(job_id: str, *, client: Any | None = None) -> dict[str, Any]:
    if not job_id.strip():
        raise ValueError("job_id cannot be empty")
    active_client = client or historical_client()
    return dict(active_client.batch.get_job_details(job_id=job_id))


def download_batch_job(
    job_id: str,
    output_dir: str | Path,
    *,
    client: Any | None = None,
) -> tuple[list[Path], list[dict[str, Any]]]:
    """Download a completed batch and verify provider-published SHA-256 hashes."""

    if not job_id.strip():
        raise ValueError("job_id cannot be empty")
    active_client = client or historical_client()
    status = batch_job_status(job_id, client=active_client)
    if status.get("state") != "done":
        raise RuntimeError(f"batch job {job_id} is not done (state={status.get('state')!r})")
    remote_files = [dict(item) for item in active_client.batch.list_files(job_id=job_id)]
    downloaded = [
        Path(path)
        for path in active_client.batch.download(job_id=job_id, output_dir=str(output_dir))
    ]
    expected_by_name = {str(item["filename"]): item for item in remote_files}
    for path in downloaded:
        details = expected_by_name.get(path.name)
        if details is None or not details.get("hash"):
            continue
        expected = str(details["hash"]).removeprefix("sha256:")
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != expected:
            raise OSError(f"SHA-256 mismatch for downloaded batch file: {path}")
    return downloaded, remote_files


def definition_window(config: ResearchConfig) -> tuple[str, str]:
    """Return the first exact UTC day, which guarantees active-definition snapshots."""

    start = pd.Timestamp(config.data.start).tz_convert("UTC").floor("D")
    end = start + pd.Timedelta(days=1)
    return start.isoformat(), end.isoformat()


def load_dbn(path: str | Path) -> pd.DataFrame:
    """Load a local DBN/DBN.ZST file as a pandas DataFrame."""

    db = _databento_module()
    store = db.DBNStore.from_file(path)
    frame = store.to_df(price_type="float", pretty_ts=True, map_symbols=True, tz="UTC")
    if frame.index.name == "ts_recv":
        frame = frame.reset_index()
    return frame


def load_events(path: str | Path) -> pd.DataFrame:
    """Load DBN or a CSV-based engineering fixture."""

    input_path = Path(path)
    lowered = input_path.name.lower()
    if lowered.endswith((".dbn", ".dbn.zst")):
        return load_dbn(input_path)
    if lowered.endswith((".csv", ".csv.gz")):
        frame = pd.read_csv(input_path)
        for column in ("ts_recv", "ts_event"):
            frame[column] = pd.to_datetime(frame[column], format="mixed", utc=True)
        return frame
    raise ValueError(f"unsupported input format: {input_path}")
