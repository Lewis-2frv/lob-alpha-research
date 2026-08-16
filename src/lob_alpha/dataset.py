"""Daily-file processing and content-addressed processed-data catalogs."""

from __future__ import annotations

import gzip
import json
import re
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Iterable

import pandas as pd

from .config import ResearchConfig
from .features import model_feature_columns
from .ingest import load_events
from .manifest import build_run_manifest, sha256_file, write_json
from .pipeline import process_session, write_table


_DATE_PATTERNS = (
    re.compile(r"(?<!\d)(20\d{2})-(\d{2})-(\d{2})(?!\d)"),
    re.compile(r"(?<!\d)(20\d{2})(\d{2})(\d{2})(?!\d)"),
)
_DATA_SUFFIXES = (".dbn", ".dbn.zst", ".csv", ".csv.gz")


@dataclass(frozen=True)
class CatalogEntry:
    session_date: str
    split: str
    path: str
    rows: int
    sha256: str


def session_date_from_filename(path: str | Path) -> date:
    name = Path(path).name
    for pattern in _DATE_PATTERNS:
        match = pattern.search(name)
        if match:
            return date(*(int(value) for value in match.groups()))
    raise ValueError(f"filename does not contain an unambiguous session date: {name}")


def discover_daily_raw_files(raw_dir: str | Path) -> dict[date, Path]:
    """Find one provider data file per dated session, recursively."""

    root = Path(raw_dir)
    if not root.exists():
        raise FileNotFoundError(root)
    discovered: dict[date, Path] = {}
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        lowered = path.name.lower()
        if not lowered.endswith(_DATA_SUFFIXES):
            continue
        if any(token in lowered for token in ("definition", "symbology", "metadata")):
            continue
        try:
            session_date = session_date_from_filename(path)
        except ValueError:
            continue
        if session_date in discovered:
            raise ValueError(
                f"multiple raw files resolve to {session_date}: "
                f"{discovered[session_date]} and {path}"
            )
        discovered[session_date] = path
    if not discovered:
        raise FileNotFoundError(f"no dated DBN/CSV files found under {root}")
    return discovered


def compact_session_dataset(frame: pd.DataFrame, config: ResearchConfig) -> pd.DataFrame:
    """Keep audit timestamps, model inputs and registered targets; drop copied depth."""

    columns = [
        "session_date",
        "split",
        "decision_time",
        "source_ts_recv",
        "midpoint",
        *model_feature_columns(config.features),
    ]
    for horizon in config.labels.horizons_ms:
        columns.extend(
            (
                f"label_target_time_{horizon}ms",
                f"label_source_time_{horizon}ms",
                f"target_{horizon}ms_ticks",
                f"direction_{horizon}ms",
            )
        )
    unique_columns = list(dict.fromkeys(columns))
    missing = [column for column in unique_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"processed session is missing compact columns: {missing}")
    return frame.loc[:, unique_columns].copy()


def process_raw_directory(
    config: ResearchConfig,
    *,
    raw_dir: str | Path,
    output_dir: str | Path,
    tick_size: float,
    output_format: str = "parquet",
    overwrite: bool = False,
) -> list[CatalogEntry]:
    """Process all configured dated files independently and write a catalog."""

    if output_format not in {"parquet", "csv.gz"}:
        raise ValueError("output_format must be 'parquet' or 'csv.gz'")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    entries: list[CatalogEntry] = []
    for session_date, input_path in discover_daily_raw_files(raw_dir).items():
        split = config.splits.split_for(session_date)
        if split is None:
            continue
        suffix = ".parquet" if output_format == "parquet" else ".csv.gz"
        output_path = destination / f"{config.data.symbols[0]}_{session_date.isoformat()}{suffix}"
        manifest_path = destination / (
            f"{config.data.symbols[0]}_{session_date.isoformat()}.manifest.json"
        )
        if output_path.exists() and not overwrite:
            entries.append(
                CatalogEntry(
                    session_date=session_date.isoformat(),
                    split=split,
                    path=str(output_path.resolve()),
                    rows=processed_row_count(output_path),
                    sha256=sha256_file(output_path),
                )
            )
            continue
        events = load_events(input_path)
        result = process_session(
            events,
            config,
            session_date=session_date,
            tick_size=tick_size,
        )
        compact = compact_session_dataset(result.data, config)
        write_table(compact, output_path)
        write_json(
            manifest_path,
            build_run_manifest(
                config_path=config.source_path,
                input_path=input_path,
                output_path=output_path,
                session_date=session_date.isoformat(),
                rows=len(compact),
                quality=result.quality.to_dict(),
            ),
        )
        entries.append(
            CatalogEntry(
                session_date=session_date.isoformat(),
                split=split,
                path=str(output_path.resolve()),
                rows=len(compact),
                sha256=sha256_file(output_path),
            )
        )
    if not entries:
        raise ValueError("no discovered session falls inside the configured split dates")
    write_catalog(entries, destination / "catalog.json", config=config)
    return entries


def processed_row_count(path: str | Path) -> int:
    input_path = Path(path)
    lowered = input_path.name.lower()
    if lowered.endswith(".parquet"):
        try:
            import pyarrow.parquet as parquet
        except ImportError as exc:
            raise RuntimeError("pyarrow is required to inspect parquet metadata") from exc
        return int(parquet.read_metadata(input_path).num_rows)
    opener = gzip.open if lowered.endswith(".gz") else open
    with opener(input_path, "rt", encoding="utf-8") as handle:
        return max(0, sum(1 for _ in handle) - 1)


def read_processed_table(path: str | Path, *, columns: list[str] | None = None) -> pd.DataFrame:
    input_path = Path(path)
    lowered = input_path.name.lower()
    if lowered.endswith(".parquet"):
        frame = pd.read_parquet(input_path, columns=columns)
    elif lowered.endswith((".csv", ".csv.gz")):
        frame = pd.read_csv(input_path, usecols=columns)
    else:
        raise ValueError(f"unsupported processed format: {input_path}")
    for column in frame.columns:
        if column == "decision_time" or column.startswith("label_") and column.endswith("_time"):
            frame[column] = pd.to_datetime(frame[column], format="mixed", utc=True)
    return frame


def write_catalog(
    entries: Iterable[CatalogEntry],
    path: str | Path,
    *,
    config: ResearchConfig,
) -> Path:
    ordered = sorted(entries, key=lambda item: item.session_date)
    payload = {
        "config_path": str(config.source_path),
        "config_sha256": sha256_file(config.source_path),
        "entries": [asdict(entry) for entry in ordered],
        "total_rows": sum(entry.rows for entry in ordered),
    }
    return write_json(path, payload)


def load_catalog(path: str | Path, *, verify_hashes: bool = True) -> list[CatalogEntry]:
    catalog_path = Path(path)
    payload = json.loads(catalog_path.read_text(encoding="utf-8"))
    entries = [CatalogEntry(**item) for item in payload["entries"]]
    if not entries:
        raise ValueError("processed catalog is empty")
    seen: set[str] = set()
    for entry in entries:
        if entry.session_date in seen:
            raise ValueError(f"duplicate session in catalog: {entry.session_date}")
        seen.add(entry.session_date)
        data_path = Path(entry.path)
        if not data_path.exists():
            raise FileNotFoundError(data_path)
        if verify_hashes and sha256_file(data_path) != entry.sha256:
            raise IOError(f"processed file hash changed after cataloging: {data_path}")
    return sorted(entries, key=lambda item: item.session_date)
