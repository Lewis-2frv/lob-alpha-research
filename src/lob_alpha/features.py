"""Economically motivated, strictly backward-looking microstructure features."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import FeatureConfig


def _trailing_sum(
    event_times: pd.Series,
    values: np.ndarray,
    decision_times: pd.Series,
    window_ms: int,
) -> np.ndarray:
    event_ns = pd.to_datetime(event_times, utc=True).astype("int64").to_numpy()
    decision_ns = pd.to_datetime(decision_times, utc=True).astype("int64").to_numpy()
    cumulative = np.concatenate(([0.0], np.cumsum(values, dtype=float)))
    end = np.searchsorted(event_ns, decision_ns, side="right")
    start = np.searchsorted(event_ns, decision_ns - window_ms * 1_000_000, side="right")
    return cumulative[end] - cumulative[start]


def _ofi_increments(events: pd.DataFrame) -> np.ndarray:
    bid = pd.to_numeric(events["bid_px_00"], errors="coerce").to_numpy(dtype=float)
    ask = pd.to_numeric(events["ask_px_00"], errors="coerce").to_numpy(dtype=float)
    bid_size = pd.to_numeric(events["bid_sz_00"], errors="coerce").to_numpy(dtype=float)
    ask_size = pd.to_numeric(events["ask_sz_00"], errors="coerce").to_numpy(dtype=float)
    previous_bid = np.roll(bid, 1)
    previous_ask = np.roll(ask, 1)
    previous_bid_size = np.roll(bid_size, 1)
    previous_ask_size = np.roll(ask_size, 1)
    increment = (
        (bid >= previous_bid) * bid_size
        - (bid <= previous_bid) * previous_bid_size
        - (ask <= previous_ask) * ask_size
        + (ask >= previous_ask) * previous_ask_size
    )
    increment[0] = 0.0
    return increment


def _weighted_depth_imbalance(states: pd.DataFrame, levels: int, decay: float) -> np.ndarray:
    weights = np.exp(-decay * np.arange(levels, dtype=float))
    bid = np.column_stack(
        [pd.to_numeric(states[f"bid_sz_{level:02d}"], errors="coerce") for level in range(levels)]
    )
    ask = np.column_stack(
        [pd.to_numeric(states[f"ask_sz_{level:02d}"], errors="coerce") for level in range(levels)]
    )
    weighted_bid = bid @ weights
    weighted_ask = ask @ weights
    denominator = weighted_bid + weighted_ask
    return np.divide(
        weighted_bid - weighted_ask,
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )


def build_features(
    events: pd.DataFrame,
    states: pd.DataFrame,
    config: FeatureConfig,
    *,
    tick_size: float,
    decision_grid_ms: int,
) -> pd.DataFrame:
    """Build the frozen v0.1 feature family."""

    if tick_size <= 0:
        raise ValueError("tick_size must be positive")
    ordered_events = events.sort_values(["ts_recv", "sequence"], kind="stable").reset_index(
        drop=True
    )
    result = states.copy()
    bid = pd.to_numeric(result["bid_px_00"], errors="coerce")
    ask = pd.to_numeric(result["ask_px_00"], errors="coerce")
    bid_size = pd.to_numeric(result["bid_sz_00"], errors="coerce")
    ask_size = pd.to_numeric(result["ask_sz_00"], errors="coerce")
    total_top_depth = bid_size + ask_size

    result["midpoint"] = (bid + ask) / 2.0
    result["spread_ticks"] = (ask - bid) / tick_size
    result["queue_imbalance_l1"] = np.divide(
        bid_size - ask_size,
        total_top_depth,
        out=np.zeros(len(result), dtype=float),
        where=total_top_depth.to_numpy(dtype=float) > 0,
    )
    microprice = np.divide(
        ask * bid_size + bid * ask_size,
        total_top_depth,
        out=result["midpoint"].to_numpy(dtype=float).copy(),
        where=total_top_depth.to_numpy(dtype=float) > 0,
    )
    result["microprice_displacement_ticks"] = (microprice - result["midpoint"]) / tick_size
    result["log_top_depth"] = np.log1p(total_top_depth)

    for levels in config.depth_levels:
        result[f"depth_imbalance_l{levels}"] = _weighted_depth_imbalance(
            result, levels, config.depth_weight_decay
        )

    ofi = _ofi_increments(ordered_events)
    normalization = total_top_depth.clip(lower=1.0).to_numpy(dtype=float)
    for window_ms in config.ofi_lookbacks_ms:
        raw = _trailing_sum(
            ordered_events["ts_recv"], ofi, result["decision_time"], window_ms
        )
        result[f"ofi_{window_ms}ms"] = raw / normalization

    is_trade = ordered_events["action"].astype(str).eq("T").to_numpy()
    sides = ordered_events["side"].astype(str).to_numpy()
    trade_size = pd.to_numeric(ordered_events["size"], errors="coerce").fillna(0).to_numpy(float)
    signed_trade = np.where(is_trade & (sides == "B"), trade_size, 0.0) - np.where(
        is_trade & (sides == "A"), trade_size, 0.0
    )
    absolute_trade = np.where(is_trade, trade_size, 0.0)
    for window_ms in config.trade_lookbacks_ms:
        signed = _trailing_sum(
            ordered_events["ts_recv"], signed_trade, result["decision_time"], window_ms
        )
        total = _trailing_sum(
            ordered_events["ts_recv"], absolute_trade, result["decision_time"], window_ms
        )
        result[f"trade_imbalance_{window_ms}ms"] = np.divide(
            signed, total, out=np.zeros_like(signed), where=total > 0
        )

    event_count = np.ones(len(ordered_events), dtype=float)
    intensity_window = config.event_intensity_lookback_ms
    result[f"event_intensity_{intensity_window}ms"] = _trailing_sum(
        ordered_events["ts_recv"], event_count, result["decision_time"], intensity_window
    ) / (intensity_window / 1000.0)

    lag_steps = max(1, int(round(config.lagged_return_ms / decision_grid_ms)))
    result[f"lagged_mid_return_{config.lagged_return_ms}ms_ticks"] = (
        result["midpoint"] - result["midpoint"].shift(lag_steps)
    ) / tick_size
    one_step = result["midpoint"].diff() / tick_size
    result[f"realized_vol_{config.lagged_return_ms}ms_ticks"] = one_step.rolling(
        lag_steps, min_periods=lag_steps
    ).std(ddof=0)
    return result


def model_feature_columns(config: FeatureConfig) -> list[str]:
    columns = [
        "spread_ticks",
        "queue_imbalance_l1",
        "microprice_displacement_ticks",
        "log_top_depth",
    ]
    columns.extend(f"depth_imbalance_l{levels}" for levels in config.depth_levels)
    columns.extend(f"ofi_{window}ms" for window in config.ofi_lookbacks_ms)
    columns.extend(f"trade_imbalance_{window}ms" for window in config.trade_lookbacks_ms)
    columns.extend(
        (
            f"event_intensity_{config.event_intensity_lookback_ms}ms",
            f"lagged_mid_return_{config.lagged_return_ms}ms_ticks",
            f"realized_vol_{config.lagged_return_ms}ms_ticks",
            "quote_age_ms",
        )
    )
    return columns
