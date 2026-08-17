"""Licensed Optiver CSV validation and bounded, per-date Parquet preparation."""

from __future__ import annotations

import json
import sqlite3
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .equity_config import EquityResearchConfig
from .equity_features import (
    build_equity_features,
    feature_implementation_sha256,
    feature_specification_sha256,
)
from .manifest import sha256_file, write_json

OPTIVER_COLUMNS = (
    "stock_id",
    "date_id",
    "seconds_in_bucket",
    "imbalance_size",
    "imbalance_buy_sell_flag",
    "reference_price",
    "matched_size",
    "far_price",
    "near_price",
    "bid_price",
    "bid_size",
    "ask_price",
    "ask_size",
    "wap",
    "target",
    "time_id",
    "row_id",
)

INTEGER_COLUMNS = (
    "stock_id",
    "date_id",
    "seconds_in_bucket",
    "imbalance_buy_sell_flag",
    "time_id",
)
REQUIRED_FINITE_FLOATS = (
    "imbalance_size",
    "reference_price",
    "matched_size",
    "bid_price",
    "bid_size",
    "ask_price",
    "ask_size",
    "wap",
)


@dataclass
class _AuditState:
    last_key: tuple[int, int, int] | None = None
    rows: int = 0
    stock_ids: set[int] = field(default_factory=set)
    date_ids: set[int] = field(default_factory=set)
    time_mapping: dict[int, tuple[int, int]] = field(default_factory=dict)
    date_second_mapping: dict[tuple[int, int], int] = field(default_factory=dict)
    time_row_counts: dict[int, int] = field(default_factory=dict)
    stock_date_pairs: set[tuple[int, int]] = field(default_factory=set)
    far_missing: int = 0
    near_missing: int = 0
    target_missing: int = 0
    target_available_dates: set[int] = field(default_factory=set)


def _require_columns(path: Path) -> tuple[str, ...]:
    columns = tuple(pd.read_csv(path, nrows=0).columns)
    missing = sorted(set(OPTIVER_COLUMNS) - set(columns))
    unexpected = sorted(set(columns) - set(OPTIVER_COLUMNS))
    if missing or unexpected:
        raise ValueError(f"Optiver schema mismatch; missing={missing}, unexpected={unexpected}")
    return columns


def _finite(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    for column in columns:
        numeric = pd.to_numeric(frame[column], errors="coerce")
        if numeric.isna().any() or not np.isfinite(numeric.to_numpy(dtype=float)).all():
            raise ValueError(f"{column} must be present, numeric and finite")


def validate_optiver_frame(
    frame: pd.DataFrame,
    *,
    require_target: bool = True,
    require_order: bool = True,
) -> None:
    """Validate one complete frame without silently removing legitimate auction rows."""

    required = set(OPTIVER_COLUMNS)
    if not require_target:
        required.remove("target")
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"missing Optiver columns: {missing}")
    if frame.empty:
        raise ValueError("Optiver data is empty")
    _finite(frame, INTEGER_COLUMNS)
    for column in INTEGER_COLUMNS:
        values = pd.to_numeric(frame[column], errors="raise")
        if not np.equal(values, np.floor(values)).all():
            raise ValueError(f"{column} must contain integers")
    _finite(frame, REQUIRED_FINITE_FLOATS)
    if require_target:
        target = pd.to_numeric(frame["target"], errors="coerce")
        if (frame["target"].notna() & target.isna()).any():
            raise ValueError("present target values must be numeric")
        present_target = target.dropna()
        if not np.isfinite(present_target.to_numpy(dtype=float)).all():
            raise ValueError("present target values must be finite")
    if (
        frame["row_id"].isna().any()
        or not frame["row_id"].map(lambda value: isinstance(value, str)).all()
        or frame["row_id"].duplicated().any()
    ):
        raise ValueError("row_id must be present and unique")
    expected_row_ids = (
        frame["date_id"].astype(int).astype(str)
        + "_"
        + frame["seconds_in_bucket"].astype(int).astype(str)
        + "_"
        + frame["stock_id"].astype(int).astype(str)
    )
    normalized_row_ids = frame["row_id"].str.removeprefix("SYNTHETIC_")
    if not normalized_row_ids.equals(expected_row_ids):
        raise ValueError("row_id must encode date_id, seconds_in_bucket and stock_id")
    keys = frame.loc[:, ["date_id", "seconds_in_bucket", "stock_id"]]
    if keys.duplicated().any():
        raise ValueError("(date_id, seconds_in_bucket, stock_id) rows must be unique")
    if require_order:
        ordered = keys.sort_values(list(keys.columns), kind="stable").reset_index(drop=True)
        if not keys.reset_index(drop=True).equals(ordered):
            raise ValueError("rows must be ordered by date_id, seconds_in_bucket, stock_id")
    if not frame["imbalance_buy_sell_flag"].isin((-1, 0, 1)).all():
        raise ValueError("imbalance_buy_sell_flag must be -1, 0 or 1")
    seconds = frame["seconds_in_bucket"].astype(int)
    if seconds.lt(0).any() or seconds.ge(600).any() or seconds.mod(10).ne(0).any():
        raise ValueError("seconds_in_bucket must be a ten-second grid in [0, 600)")
    if frame[["stock_id", "date_id", "time_id"]].lt(0).any(axis=None):
        raise ValueError("stock_id, date_id and time_id must be nonnegative")
    if (frame[["bid_size", "ask_size"]] <= 0).any(axis=None):
        raise ValueError("displayed bid and ask sizes must be positive")
    if (frame[["imbalance_size", "matched_size"]] < 0).any(axis=None):
        raise ValueError("auction sizes cannot be negative")
    if (frame[["reference_price", "bid_price", "ask_price", "wap"]] <= 0).any(axis=None):
        raise ValueError("required prices must be positive")
    for column in ("near_price", "far_price"):
        numeric = pd.to_numeric(frame[column], errors="coerce")
        if (frame[column].notna() & numeric.isna()).any():
            raise ValueError(f"{column} may contain only numeric values or explicit missingness")
        present = numeric.dropna()
        if not np.isfinite(present.to_numpy(dtype=float)).all() or (present <= 0).any():
            raise ValueError(f"present {column} values must be finite and positive")
    if (frame["ask_price"] < frame["bid_price"]).any():
        raise ValueError("crossed quotes are invalid")
    if ((frame["wap"] < frame["bid_price"]) | (frame["wap"] > frame["ask_price"])).any():
        raise ValueError("WAP must lie within the displayed quote")
    mapping = frame.loc[:, ["time_id", "date_id", "seconds_in_bucket"]].drop_duplicates()
    if mapping["time_id"].duplicated().any():
        raise ValueError("each time_id must map to one date_id and seconds_in_bucket")
    reverse = mapping.loc[:, ["date_id", "seconds_in_bucket"]]
    if reverse.duplicated().any():
        raise ValueError("each (date_id, seconds_in_bucket) must map to one time_id")


def _update_global_audit(
    frame: pd.DataFrame,
    state: _AuditState,
    row_database: sqlite3.Connection,
) -> None:
    keys = frame.loc[:, ["date_id", "seconds_in_bucket", "stock_id"]].astype(int)
    first_key = tuple(int(value) for value in keys.iloc[0])
    last_key = tuple(int(value) for value in keys.iloc[-1])
    if state.last_key is not None and first_key <= state.last_key:
        raise ValueError("row ordering or uniqueness failed across CSV chunk boundary")
    state.last_key = last_key
    try:
        row_database.executemany(
            "INSERT INTO row_ids(value) VALUES (?)",
            ((str(value),) for value in frame["row_id"]),
        )
        row_database.commit()
    except sqlite3.IntegrityError as exc:
        raise ValueError("row_id must be globally unique") from exc
    for item in (
        frame.loc[:, ["time_id", "date_id", "seconds_in_bucket"]]
        .drop_duplicates()
        .itertuples(index=False)
    ):
        time_id, date_id, second = (int(item[0]), int(item[1]), int(item[2]))
        pair = (date_id, second)
        if time_id in state.time_mapping and state.time_mapping[time_id] != pair:
            raise ValueError("time_id mapping changed across chunks")
        if pair in state.date_second_mapping and state.date_second_mapping[pair] != time_id:
            raise ValueError("date/seconds mapping changed across chunks")
        state.time_mapping[time_id] = pair
        state.date_second_mapping[pair] = time_id
    state.rows += len(frame)
    state.stock_ids.update(int(value) for value in frame["stock_id"].unique())
    state.date_ids.update(int(value) for value in frame["date_id"].unique())
    for time_id, count in frame.groupby("time_id", observed=True).size().items():
        numeric_time = int(time_id)
        state.time_row_counts[numeric_time] = state.time_row_counts.get(numeric_time, 0) + int(
            count
        )
    state.stock_date_pairs.update(
        (int(item.stock_id), int(item.date_id))
        for item in frame.loc[:, ["stock_id", "date_id"]]
        .drop_duplicates()
        .itertuples(index=False)
    )
    state.far_missing += int(frame["far_price"].isna().sum())
    state.near_missing += int(frame["near_price"].isna().sum())
    if "target" in frame:
        state.target_missing += int(frame["target"].isna().sum())
        state.target_available_dates.update(
            int(value) for value in frame.loc[frame["target"].notna(), "date_id"].unique()
        )


def audit_optiver_csv(
    config: EquityResearchConfig,
    *,
    input_path: str | Path | None = None,
    output_path: str | Path,
    metadata_only: bool = False,
) -> dict[str, Any]:
    """Audit the licensed CSV; metadata-only mode never loads the target column."""

    source = Path(input_path or config.data.raw_path)
    output = Path(output_path)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite registered audit: {output}")
    columns = _require_columns(source)
    usecols = list(OPTIVER_COLUMNS)
    if metadata_only:
        usecols.remove("target")
    state = _AuditState()
    with tempfile.TemporaryDirectory(prefix="lob-alpha-row-audit-") as temporary:
        database = sqlite3.connect(Path(temporary) / "row_ids.sqlite")
        database.execute("CREATE TABLE row_ids(value TEXT PRIMARY KEY)")
        try:
            for frame in pd.read_csv(
                source, usecols=usecols, chunksize=config.data.csv_chunk_rows
            ):
                validate_optiver_frame(frame, require_target=not metadata_only)
                if config.source_kind == "real" and frame["row_id"].str.startswith(
                    "SYNTHETIC_"
                ).any():
                    raise ValueError(
                        "synthetic row identifiers cannot be audited as real Optiver data"
                    )
                _update_global_audit(frame, state, database)
        finally:
            database.close()
    expected_dates = set(
        range(config.data.expected_date_id_min, config.data.expected_date_id_max + 1)
    )
    if state.date_ids != expected_dates:
        missing = sorted(expected_dates - state.date_ids)
        unexpected = sorted(state.date_ids - expected_dates)
        raise ValueError(
            f"observed date_id range does not match registration; missing={missing}, "
            f"unexpected={unexpected}"
        )
    if not metadata_only and state.target_available_dates != expected_dates:
        missing_target_dates = sorted(expected_dates - state.target_available_dates)
        raise ValueError(f"dates without any usable target: {missing_target_dates}")
    time_counts = pd.Series(state.time_row_counts, dtype="int64")
    modal_time_count = int(time_counts.mode().iloc[0])
    payload: dict[str, Any] = {
        "artifact_type": "optiver_schema_audit",
        "source_kind": config.source_kind,
        "input_path": str(source.resolve()),
        "input_sha256": sha256_file(source),
        "config_path": str(config.source_path),
        "config_sha256": sha256_file(config.source_path),
        "target_values_read": not metadata_only,
        "target_definition": config.data.target_definition,
        "columns": list(columns),
        "rows": state.rows,
        "stock_ids": sorted(state.stock_ids),
        "date_id_min": min(state.date_ids),
        "date_id_max": max(state.date_ids),
        "date_ids": len(state.date_ids),
        "time_ids": len(state.time_mapping),
        "rows_per_time_id_min": int(time_counts.min()),
        "rows_per_time_id_max": int(time_counts.max()),
        "rows_per_time_id_mode": modal_time_count,
        "time_ids_below_modal_stock_coverage": int(time_counts.lt(modal_time_count).sum()),
        "observed_stock_date_pairs": len(state.stock_date_pairs),
        "far_price_missing_rows": state.far_missing,
        "near_price_missing_rows": state.near_missing,
        "target_missing_rows": state.target_missing if not metadata_only else None,
        "target_available_rows": (
            state.rows - state.target_missing if not metadata_only else None
        ),
        "validation": "passed",
    }
    write_json(output, payload)
    return payload


def _read_audit(path: Path, config: EquityResearchConfig, source: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("artifact_type") != "optiver_schema_audit":
        raise ValueError("unexpected audit artifact")
    if payload.get("config_sha256") != sha256_file(config.source_path):
        raise OSError("configuration changed after data audit")
    if payload.get("input_sha256") != sha256_file(source):
        raise OSError("raw Optiver CSV changed after data audit")
    if payload.get("target_definition") != config.data.target_definition:
        raise OSError("target definition changed after data audit")
    if not payload.get("target_values_read"):
        raise ValueError("full target-finiteness audit is required before preparation")
    return payload


def _write_prepared_date(
    frame: pd.DataFrame,
    *,
    config: EquityResearchConfig,
    destination: Path,
) -> dict[str, Any]:
    date_ids = frame["date_id"].unique()
    if len(date_ids) != 1:
        raise AssertionError("prepared partition must contain exactly one date_id")
    date_id = int(date_ids[0])
    featured = build_equity_features(frame, config)
    path = destination / f"date_id={date_id:04d}.parquet"
    temporary = path.with_suffix(path.suffix + ".partial")
    featured.to_parquet(temporary, index=False, compression="zstd")
    temporary.replace(path)
    return {
        "date_id": date_id,
        "split": config.splits.split_for(date_id),
        "path": str(path.resolve()),
        "rows": len(featured),
        "target_available_rows": int(featured["target"].notna().sum()),
        "sha256": sha256_file(path),
    }


def prepare_optiver_parquet(
    config: EquityResearchConfig,
    *,
    input_path: str | Path | None = None,
    audit_path: str | Path,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Prepare one date at a time, bounding memory and preserving all early rows."""

    source = Path(input_path or config.data.raw_path)
    audit = Path(audit_path)
    _read_audit(audit, config, source)
    destination = Path(output_dir or config.data.prepared_dir)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"refusing to overwrite prepared equity data: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    current_date: int | None = None
    buffered: list[pd.DataFrame] = []
    for chunk in pd.read_csv(source, chunksize=config.data.csv_chunk_rows):
        for date_id, group in chunk.groupby("date_id", sort=False):
            numeric_date = int(date_id)
            if current_date is None:
                current_date = numeric_date
            if numeric_date != current_date:
                records.append(
                    _write_prepared_date(
                        pd.concat(buffered, ignore_index=True),
                        config=config,
                        destination=destination,
                    )
                )
                buffered = []
                current_date = numeric_date
            buffered.append(group.copy())
    if buffered:
        records.append(
            _write_prepared_date(
                pd.concat(buffered, ignore_index=True),
                config=config,
                destination=destination,
            )
        )
    observed = [record["date_id"] for record in records]
    expected = list(range(config.data.expected_date_id_min, config.data.expected_date_id_max + 1))
    if observed != expected:
        raise ValueError(
            f"prepared date order/range mismatch: observed={observed}, expected={expected}"
        )
    common_manifest = {
        "source_kind": config.source_kind,
        "config_sha256": sha256_file(config.source_path),
        "feature_specification_sha256": feature_specification_sha256(config),
        "feature_implementation_sha256": feature_implementation_sha256(),
        "target_definition": config.data.target_definition,
    }
    development_records = [record for record in records if record["split"] != "holdout"]
    holdout_records = [record for record in records if record["split"] == "holdout"]
    development_manifest_path = destination / "development_manifest.json"
    holdout_manifest_path = destination / "holdout_manifest.json"
    write_json(
        development_manifest_path,
        {
            "artifact_type": "optiver_development_partitions",
            **common_manifest,
            "partitions": development_records,
            "rows": sum(int(record["rows"]) for record in development_records),
        },
    )
    write_json(
        holdout_manifest_path,
        {
            "artifact_type": "optiver_sealed_holdout_partitions",
            **common_manifest,
            "partitions": holdout_records,
            "rows": sum(int(record["rows"]) for record in holdout_records),
        },
    )
    payload: dict[str, Any] = {
        "artifact_type": "optiver_prepared_manifest",
        **common_manifest,
        "raw_path": str(source.resolve()),
        "raw_sha256": sha256_file(source),
        "raw_metadata_path": str(audit.resolve()),
        "raw_metadata_sha256": sha256_file(audit),
        "config_path": str(config.source_path),
        "development_manifest_path": str(development_manifest_path.resolve()),
        "development_manifest_sha256": sha256_file(development_manifest_path),
        "holdout_manifest_path": str(holdout_manifest_path.resolve()),
        "holdout_manifest_sha256": sha256_file(holdout_manifest_path),
        "registered_date_id_min": config.data.expected_date_id_min,
        "registered_date_id_max": config.data.expected_date_id_max,
    }
    write_json(destination / "prepared_manifest.json", payload)
    return {
        **payload,
        "partitions": records,
        "rows": sum(int(record["rows"]) for record in records),
    }


def load_prepared_manifest(
    path: str | Path,
    config: EquityResearchConfig,
    *,
    scope: str | None = "development",
    verify_partitions: bool = True,
) -> dict[str, Any]:
    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("artifact_type") != "optiver_prepared_manifest":
        raise ValueError("unexpected prepared manifest")
    if payload.get("config_sha256") != sha256_file(config.source_path):
        raise OSError("configuration does not match prepared data")
    if payload.get("feature_specification_sha256") != feature_specification_sha256(config):
        raise OSError("feature specification does not match prepared data")
    if payload.get("feature_implementation_sha256") != feature_implementation_sha256():
        raise OSError("feature implementation does not match prepared data")
    if payload.get("target_definition") != config.data.target_definition:
        raise OSError("target definition does not match prepared data")
    if scope not in {None, "development", "holdout"}:
        raise ValueError(f"unsupported prepared-manifest scope: {scope}")
    manifest_names = () if scope is None else (scope,)
    for manifest_name in manifest_names:
        manifest_file = Path(payload[f"{manifest_name}_manifest_path"])
        if (
            not manifest_file.is_file()
            or sha256_file(manifest_file) != payload[f"{manifest_name}_manifest_sha256"]
        ):
            raise OSError(f"{manifest_name} partition manifest failed hash validation")
    if scope is None:
        return payload
    scoped_path = Path(payload[f"{scope}_manifest_path"])
    scoped = json.loads(scoped_path.read_text(encoding="utf-8"))
    expected_type = (
        "optiver_development_partitions"
        if scope == "development"
        else "optiver_sealed_holdout_partitions"
    )
    if scoped.get("artifact_type") != expected_type:
        raise ValueError(f"unexpected {scope} partition manifest")
    if scoped.get("config_sha256") != payload["config_sha256"]:
        raise OSError(f"{scope} partition configuration hash mismatch")
    for key in (
        "feature_specification_sha256",
        "feature_implementation_sha256",
        "target_definition",
    ):
        if scoped.get(key) != payload.get(key):
            raise OSError(f"{scope} partition manifest disagrees on {key}")
    if scope == "development":
        expected_dates = list(
            range(config.splits.train_start, config.splits.validation_end + 1)
        )
    else:
        expected_dates = list(range(config.splits.holdout_start, config.splits.holdout_end + 1))
    partitions = scoped.get("partitions")
    if not isinstance(partitions, list):
        raise ValueError(f"{scope} partition manifest has no partition list")
    if [int(item["date_id"]) for item in partitions] != expected_dates:
        raise OSError(f"{scope} date partitions do not match registration")
    expected_splits = {"train", "validation"} if scope == "development" else {"holdout"}
    if any(item.get("split") not in expected_splits for item in partitions):
        raise OSError(f"{scope} partition manifest contains an invalid split")
    if any(int(item.get("rows", 0)) <= 0 for item in partitions):
        raise OSError(f"{scope} partition manifest contains an empty date")
    if any(
        not 0 < int(item.get("target_available_rows", 0)) <= int(item["rows"])
        for item in partitions
    ):
        raise OSError(f"{scope} partition manifest has invalid target coverage")
    rows = sum(int(item["rows"]) for item in partitions)
    if int(scoped.get("rows", -1)) != rows:
        raise OSError(f"{scope} partition row total is inconsistent")
    if verify_partitions:
        for item in partitions:
            partition = Path(item["path"])
            if not partition.is_file() or sha256_file(partition) != item["sha256"]:
                raise OSError(f"prepared partition failed hash validation: {partition}")
    return {**payload, "scope": scope, "partitions": partitions, "rows": rows}


def prepared_records_for_split(manifest: dict[str, Any], split: str) -> list[dict[str, Any]]:
    records = [dict(item) for item in manifest["partitions"] if item["split"] == split]
    if not records:
        raise ValueError(f"prepared data contains no {split} dates")
    return records
