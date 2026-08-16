"""Causal session filtering and fixed-clock decision-state sampling."""

from __future__ import annotations

from datetime import date, datetime
from zoneinfo import ZoneInfo

import pandas as pd

from .config import SessionConfig
from .schema import BOOK_FIELDS


def session_bounds(session_date: date, config: SessionConfig) -> tuple[pd.Timestamp, pd.Timestamp]:
    timezone = ZoneInfo(config.timezone)
    start_local = datetime.combine(session_date, config.start_time, tzinfo=timezone)
    end_local = datetime.combine(session_date, config.end_time, tzinfo=timezone)
    return pd.Timestamp(start_local).tz_convert("UTC"), pd.Timestamp(end_local).tz_convert("UTC")


def filter_session(
    events: pd.DataFrame,
    session_date: date,
    config: SessionConfig,
) -> pd.DataFrame:
    start, end = session_bounds(session_date, config)
    timestamps = pd.to_datetime(events["ts_recv"], utc=True)
    return events.loc[(timestamps >= start) & (timestamps <= end)].copy()


def sample_decision_states(
    events: pd.DataFrame,
    session_date: date,
    config: SessionConfig,
) -> pd.DataFrame:
    """Sample the last observable complete book on a fixed clock."""

    session_events = filter_session(events, session_date, config)
    if session_events.empty:
        raise ValueError(f"no events within configured session for {session_date}")
    session_events = session_events.sort_values(["ts_recv", "sequence"], kind="stable")
    start, configured_end = session_bounds(session_date, config)
    observed_end = pd.to_datetime(session_events["ts_recv"], utc=True).max()
    grid_end = min(configured_end, observed_end.floor(f"{config.decision_grid_ms}ms"))
    grid = pd.DataFrame(
        {
            "decision_time": pd.date_range(
                start,
                grid_end,
                freq=f"{config.decision_grid_ms}ms",
                inclusive="both",
            )
        }
    )

    right_columns = ["ts_recv", "sequence", *BOOK_FIELDS]
    right = session_events.loc[:, right_columns].rename(columns={"ts_recv": "source_ts_recv"})
    states = pd.merge_asof(
        grid.sort_values("decision_time"),
        right.sort_values("source_ts_recv"),
        left_on="decision_time",
        right_on="source_ts_recv",
        direction="backward",
        allow_exact_matches=True,
    )
    states = states.dropna(subset=["source_ts_recv", "bid_px_00", "ask_px_00"]).copy()
    states["quote_age_ms"] = (
        states["decision_time"] - states["source_ts_recv"]
    ).dt.total_seconds() * 1000.0
    states = states.loc[states["quote_age_ms"] <= config.maximum_quote_age_ms].reset_index(
        drop=True
    )
    return states
