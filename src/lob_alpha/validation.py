"""Data-quality diagnostics for exchange MBP-10 event streams."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .schema import PRICE_FIELDS, REQUIRED_FIELDS, SIZE_FIELDS


@dataclass(frozen=True)
class QualityReport:
    rows: int
    trades: int
    exact_duplicate_rows: int
    timestamp_inversions: int
    invalid_best_quotes: int
    locked_books: int
    crossed_books: int
    negative_size_or_count: int
    ladder_errors: int
    tick_misaligned_rows: int
    maximum_receive_gap_ms: float
    accepted: bool

    def to_dict(self) -> dict[str, int | float | bool]:
        return asdict(self)


class DataQualityError(ValueError):
    """Raised when a session contains fatal data-quality violations."""


def _timestamp_series(frame: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(frame["ts_recv"], utc=True, errors="coerce")


def validate_mbp10(frame: pd.DataFrame, *, tick_size: float) -> QualityReport:
    """Return deterministic diagnostics without silently repairing input."""

    missing = [column for column in REQUIRED_FIELDS if column not in frame.columns]
    if missing:
        raise DataQualityError(f"missing required MBP-10 columns: {missing}")
    if tick_size <= 0:
        raise ValueError("tick_size must be positive")

    rows = len(frame)
    timestamps = _timestamp_series(frame)
    timestamp_inversions = int((timestamps.diff() < pd.Timedelta(0)).sum())
    gaps = timestamps.sort_values(kind="stable").diff().dt.total_seconds().mul(1000.0)
    maximum_gap = float(gaps.max()) if rows > 1 else 0.0
    if not np.isfinite(maximum_gap):
        maximum_gap = 0.0

    best_bid = pd.to_numeric(frame["bid_px_00"], errors="coerce")
    best_ask = pd.to_numeric(frame["ask_px_00"], errors="coerce")
    invalid_best = int(
        (best_bid.isna() | best_ask.isna() | (best_bid <= 0) | (best_ask <= 0)).sum()
    )
    locked = int((best_bid == best_ask).sum())
    crossed = int((best_bid > best_ask).sum())

    size_values = frame.loc[:, SIZE_FIELDS].apply(pd.to_numeric, errors="coerce")
    negative_sizes = int((size_values < 0).any(axis=1).sum())

    ladder_bad = np.zeros(rows, dtype=bool)
    for level in range(9):
        bid_near = pd.to_numeric(frame[f"bid_px_{level:02d}"], errors="coerce")
        bid_far = pd.to_numeric(frame[f"bid_px_{level + 1:02d}"], errors="coerce")
        ask_near = pd.to_numeric(frame[f"ask_px_{level:02d}"], errors="coerce")
        ask_far = pd.to_numeric(frame[f"ask_px_{level + 1:02d}"], errors="coerce")
        ladder_bad |= ((bid_near <= bid_far) & bid_near.notna() & bid_far.notna()).to_numpy()
        ladder_bad |= ((ask_near >= ask_far) & ask_near.notna() & ask_far.notna()).to_numpy()

    price_values = frame.loc[:, PRICE_FIELDS].apply(pd.to_numeric, errors="coerce")
    scaled = price_values.to_numpy(dtype=float) / tick_size
    finite = np.isfinite(scaled)
    misaligned_cells = finite & ~np.isclose(scaled, np.rint(scaled), atol=1e-8, rtol=0.0)
    tick_misaligned = int(misaligned_cells.any(axis=1).sum())

    exact_duplicates = int(frame.duplicated(keep="first").sum())
    trades = int(frame["action"].astype(str).eq("T").sum())
    accepted = bool(
        rows > 0
        and timestamps.notna().all()
        and timestamp_inversions == 0
        and exact_duplicates == 0
        and invalid_best == 0
        and crossed == 0
        and negative_sizes == 0
        and not ladder_bad.any()
        and tick_misaligned == 0
    )
    return QualityReport(
        rows=rows,
        trades=trades,
        exact_duplicate_rows=exact_duplicates,
        timestamp_inversions=timestamp_inversions,
        invalid_best_quotes=invalid_best,
        locked_books=locked,
        crossed_books=crossed,
        negative_size_or_count=negative_sizes,
        ladder_errors=int(ladder_bad.sum()),
        tick_misaligned_rows=tick_misaligned,
        maximum_receive_gap_ms=maximum_gap,
        accepted=accepted,
    )


def require_usable(report: QualityReport) -> None:
    if not report.accepted:
        failures = {
            key: value
            for key, value in report.to_dict().items()
            if key != "accepted" and value
        }
        raise DataQualityError(f"MBP-10 session failed validation: {failures}")
