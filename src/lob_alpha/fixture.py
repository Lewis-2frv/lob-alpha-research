"""Deterministic MBP-10 fixture for engineering tests only."""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_mbp10_fixture(
    *,
    periods: int = 400,
    frequency_ms: int = 20,
    seed: int = 20260815,
    start: str = "2026-03-16T13:35:00Z",
    tick_size: float = 0.25,
) -> pd.DataFrame:
    """Create a valid, non-headline order-book stream for deterministic testing.

    This fixture exists to verify mechanics and causality. It must never be used
    as empirical evidence or cited in project results.
    """

    if periods < 3:
        raise ValueError("periods must be at least 3")
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range(start, periods=periods, freq=f"{frequency_ms}ms", tz="UTC")

    rows: list[dict[str, object]] = []
    best_bid_ticks = 20_000
    for index, ts_recv in enumerate(timestamps):
        bid_size = int(20 + rng.integers(0, 31))
        ask_size = int(20 + rng.integers(0, 31))
        imbalance = (bid_size - ask_size) / (bid_size + ask_size)
        if index and index % 10 == 0:
            draw = rng.random()
            if draw < max(0.0, imbalance) * 0.45:
                best_bid_ticks += 1
            elif draw < max(0.0, -imbalance) * 0.45:
                best_bid_ticks -= 1

        best_bid = best_bid_ticks * tick_size
        best_ask = best_bid + tick_size
        action = "T" if index % 7 == 0 else ("A" if index % 3 else "C")
        side = "B" if index % 2 == 0 else "A"
        row: dict[str, object] = {
            "ts_recv": ts_recv,
            "ts_event": ts_recv - pd.Timedelta(microseconds=100),
            "rtype": 10,
            "publisher_id": 1,
            "instrument_id": 12345,
            "action": action,
            "side": side,
            "depth": 0,
            "price": best_ask if side == "B" else best_bid,
            "size": int(1 + rng.integers(0, 5)),
            "flags": 0,
            "ts_in_delta": 100_000,
            "sequence": index + 1,
            "symbol": "ESM6",
        }
        for level in range(10):
            row[f"bid_px_{level:02d}"] = best_bid - level * tick_size
            row[f"ask_px_{level:02d}"] = best_ask + level * tick_size
            row[f"bid_sz_{level:02d}"] = bid_size + level * 3
            row[f"ask_sz_{level:02d}"] = ask_size + level * 3
            row[f"bid_ct_{level:02d}"] = max(1, (bid_size + level * 3) // 5)
            row[f"ask_ct_{level:02d}"] = max(1, (ask_size + level * 3) // 5)
        rows.append(row)
    return pd.DataFrame(rows)

