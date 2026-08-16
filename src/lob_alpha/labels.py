"""Causal future-midpoint labels with auditable source timestamps."""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_labels(
    events: pd.DataFrame,
    decisions: pd.DataFrame,
    horizons_ms: tuple[int, ...],
    *,
    tick_size: float,
    maximum_age_ms: int | None = None,
) -> pd.DataFrame:
    if tick_size <= 0:
        raise ValueError("tick_size must be positive")
    right = events.loc[:, ["ts_recv", "bid_px_00", "ask_px_00"]].copy()
    right["ts_recv"] = pd.to_datetime(right["ts_recv"], utc=True)
    right = right.sort_values("ts_recv", kind="stable")
    right["future_midpoint"] = (
        pd.to_numeric(right["bid_px_00"], errors="coerce")
        + pd.to_numeric(right["ask_px_00"], errors="coerce")
    ) / 2.0
    right = right.loc[:, ["ts_recv", "future_midpoint"]]

    labels = decisions.loc[:, ["decision_time", "midpoint"]].copy()
    for horizon in horizons_ms:
        target_column = f"label_target_time_{horizon}ms"
        source_column = f"label_source_time_{horizon}ms"
        future_column = f"future_midpoint_{horizon}ms"
        tick_column = f"target_{horizon}ms_ticks"
        direction_column = f"direction_{horizon}ms"
        lookup = labels.loc[:, ["decision_time"]].copy()
        lookup[target_column] = lookup["decision_time"] + pd.to_timedelta(horizon, unit="ms")
        matched = pd.merge_asof(
            lookup.sort_values(target_column),
            right.rename(columns={"ts_recv": source_column}).sort_values(source_column),
            left_on=target_column,
            right_on=source_column,
            direction="backward",
            allow_exact_matches=True,
        ).sort_values("decision_time", kind="stable")
        labels[target_column] = matched[target_column].to_numpy()
        labels[source_column] = matched[source_column].to_numpy()
        labels[future_column] = matched["future_midpoint"].to_numpy()
        if maximum_age_ms is not None:
            age_ms = (
                labels[target_column] - labels[source_column]
            ).dt.total_seconds() * 1000.0
            stale = age_ms > maximum_age_ms
            labels.loc[stale, future_column] = np.nan
        labels[tick_column] = (labels[future_column] - labels["midpoint"]) / tick_size
        labels[direction_column] = np.select(
            [labels[tick_column] > 0, labels[tick_column] < 0],
            ["up", "down"],
            default="flat",
        )
    return labels.drop(columns=["midpoint"])
