"""Canonical MBP-10 column definitions and normalization."""

from __future__ import annotations

import pandas as pd

DEPTH = 10
BOOK_FIELDS = tuple(
    field
    for level in range(DEPTH)
    for field in (
        f"bid_px_{level:02d}",
        f"ask_px_{level:02d}",
        f"bid_sz_{level:02d}",
        f"ask_sz_{level:02d}",
        f"bid_ct_{level:02d}",
        f"ask_ct_{level:02d}",
    )
)
PRICE_FIELDS = tuple(
    field
    for level in range(DEPTH)
    for field in (f"bid_px_{level:02d}", f"ask_px_{level:02d}")
)
SIZE_FIELDS = tuple(
    field
    for level in range(DEPTH)
    for field in (
        f"bid_sz_{level:02d}",
        f"ask_sz_{level:02d}",
        f"bid_ct_{level:02d}",
        f"ask_ct_{level:02d}",
    )
)
REQUIRED_EVENT_FIELDS = (
    "ts_recv",
    "ts_event",
    "action",
    "side",
    "sequence",
    "price",
    "size",
)
REQUIRED_FIELDS = REQUIRED_EVENT_FIELDS + BOOK_FIELDS


class SchemaError(ValueError):
    """Raised for malformed or incomplete MBP-10 input."""


def canonicalize_mbp10(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a timestamp-normalized, stably sorted MBP-10 DataFrame."""

    df = frame.copy()
    if "ts_recv" not in df.columns and df.index.name == "ts_recv":
        df = df.reset_index()
    missing = [column for column in REQUIRED_FIELDS if column not in df.columns]
    if missing:
        raise SchemaError(f"missing MBP-10 fields: {missing}")

    for column in ("ts_recv", "ts_event"):
        df[column] = pd.to_datetime(df[column], utc=True, errors="coerce")
    if df[["ts_recv", "ts_event"]].isna().any().any():
        raise SchemaError("ts_recv and ts_event must be valid timezone-aware timestamps")

    numeric_fields = ("sequence", "price", "size") + BOOK_FIELDS
    for column in numeric_fields:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    return df.sort_values(["ts_recv", "sequence"], kind="stable").reset_index(drop=True)

