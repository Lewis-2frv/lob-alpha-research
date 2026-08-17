"""Causal features for the ten-second equity closing-auction panel."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .equity_config import EquityResearchConfig

BASE_FEATURE_COLUMNS = (
    "spread",
    "spread_bps",
    "midpoint",
    "wap_mid_gap_bps",
    "book_size_imbalance",
    "signed_size_imbalance",
    "bid_liquidity",
    "ask_liquidity",
    "displayed_liquidity",
    "displayed_liquidity_log",
    "microprice",
    "microprice_gap_bps",
    "reference_wap_gap_bps",
    "near_wap_gap_bps",
    "far_wap_gap_bps",
    "near_price_missing",
    "far_price_missing",
    "signed_unmatched_imbalance",
    "auction_imbalance_ratio",
    "matched_size_log",
    "imbalance_size_log",
    "flag_book_imbalance_interaction",
    "flag_wap_mid_interaction",
    "near_far_dislocation_bps",
    "time_remaining_seconds",
    "time_remaining_fraction",
    "time_remaining_squared",
    "imbalance_time_interaction",
)


def equity_model_feature_columns(config: EquityResearchConfig) -> list[str]:
    """Return the frozen model feature list; target and future quotes are excluded."""

    columns = list(BASE_FEATURE_COLUMNS)
    for lag in config.features.lag_seconds:
        columns.extend(
            (
                f"wap_return_{lag}s_bps",
                f"midpoint_return_{lag}s_bps",
                f"imbalance_size_change_{lag}s",
                f"matched_size_change_{lag}s",
                f"spread_bps_change_{lag}s",
                f"displayed_liquidity_change_{lag}s",
            )
        )
    for window in config.features.rolling_windows_seconds:
        columns.extend(
            (
                f"wap_return_10s_mean_{window}s",
                f"wap_return_10s_vol_{window}s",
                f"auction_imbalance_ratio_mean_{window}s",
                f"spread_bps_mean_{window}s",
            )
        )
    for source in config.features.cross_sectional_columns:
        columns.extend((f"{source}_cs_rank", f"{source}_cs_robust_z", f"{source}_cs_demeaned"))
    forbidden = [name for name in columns if "target" in name or name.startswith("future_")]
    if forbidden:
        raise AssertionError(f"forbidden model features: {forbidden}")
    return columns


def feature_specification(config: EquityResearchConfig) -> dict[str, object]:
    """Return the content-addressed, human-readable causal feature contract."""

    return {
        "version": 1,
        "prediction_time_information": "current and earlier rows only",
        "within_group": ["stock_id", "date_id"],
        "cross_sectional_group": "time_id",
        "lag_seconds": list(config.features.lag_seconds),
        "rolling_windows_seconds": list(config.features.rolling_windows_seconds),
        "numeric_features": equity_model_feature_columns(config),
        "categorical_features": ["stock_id"],
        "excluded": [
            "target",
            "future_bid_price",
            "future_ask_price",
            "future_wap",
            "future quotes of every kind",
            "holdout aggregates",
        ],
    }


def feature_specification_sha256(config: EquityResearchConfig) -> str:
    encoded = json.dumps(
        feature_specification(config), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def feature_implementation_sha256() -> str:
    """Hash the executable feature formulas, not only their declared names."""

    source = Path(__file__).read_text(encoding="utf-8").replace("\r\n", "\n")
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator.div(denominator.where(denominator.ne(0)))


def _within_group_rolling(
    frame: pd.DataFrame, column: str, periods: int, aggregation: str
) -> pd.Series:
    minimum = min(2, periods)
    output = pd.Series(np.nan, index=frame.index, dtype=float)
    full_grid = np.arange(0, 600, 10)
    for _, group in frame.groupby(["stock_id", "date_id"], sort=False, observed=True):
        values = pd.Series(
            group[column].to_numpy(dtype=float),
            index=group["seconds_in_bucket"].to_numpy(dtype=int),
        ).reindex(full_grid)
        if aggregation == "mean":
            rolled = values.rolling(periods, min_periods=1).mean()
        elif aggregation == "std":
            rolled = values.rolling(periods, min_periods=minimum).std(ddof=0)
        else:
            raise ValueError(f"unsupported rolling aggregation: {aggregation}")
        output.loc[group.index] = rolled.reindex(group["seconds_in_bucket"]).to_numpy()
    return output


def build_equity_features(frame: pd.DataFrame, config: EquityResearchConfig) -> pd.DataFrame:
    """Build causal features without crossing stock, date or timestamp boundaries."""

    out = frame.copy()
    expected_order = ["date_id", "seconds_in_bucket", "stock_id"]
    out = out.sort_values(expected_order, kind="stable").reset_index(drop=True)
    midpoint = (out["bid_price"] + out["ask_price"]) / 2.0
    spread = out["ask_price"] - out["bid_price"]
    liquidity = out["bid_size"] + out["ask_size"]
    out["spread"] = spread
    out["midpoint"] = midpoint
    out["spread_bps"] = _safe_ratio(spread * 10_000.0, midpoint)
    out["wap_mid_gap_bps"] = _safe_ratio((out["wap"] - midpoint) * 10_000.0, midpoint)
    out["book_size_imbalance"] = _safe_ratio(out["bid_size"] - out["ask_size"], liquidity)
    out["signed_size_imbalance"] = out["imbalance_buy_sell_flag"] * out["imbalance_size"]
    out["bid_liquidity"] = out["bid_size"]
    out["ask_liquidity"] = out["ask_size"]
    out["displayed_liquidity"] = liquidity
    out["displayed_liquidity_log"] = np.log1p(liquidity)
    out["microprice"] = _safe_ratio(
        out["ask_price"] * out["bid_size"] + out["bid_price"] * out["ask_size"],
        liquidity,
    )
    out["microprice_gap_bps"] = _safe_ratio((out["microprice"] - midpoint) * 10_000.0, midpoint)
    out["reference_wap_gap_bps"] = _safe_ratio(
        (out["reference_price"] - out["wap"]) * 10_000.0, out["wap"]
    )
    out["near_wap_gap_bps"] = _safe_ratio((out["near_price"] - out["wap"]) * 10_000.0, out["wap"])
    out["far_wap_gap_bps"] = _safe_ratio((out["far_price"] - out["wap"]) * 10_000.0, out["wap"])
    out["near_price_missing"] = out["near_price"].isna().astype("int8")
    out["far_price_missing"] = out["far_price"].isna().astype("int8")
    out["signed_unmatched_imbalance"] = out["signed_size_imbalance"]
    total_auction_volume = out["matched_size"] + out["imbalance_size"]
    out["auction_imbalance_ratio"] = _safe_ratio(
        out["signed_unmatched_imbalance"], total_auction_volume
    )
    out["matched_size_log"] = np.log1p(out["matched_size"])
    out["imbalance_size_log"] = np.log1p(out["imbalance_size"])
    out["flag_book_imbalance_interaction"] = (
        out["imbalance_buy_sell_flag"] * out["book_size_imbalance"]
    )
    out["flag_wap_mid_interaction"] = out["imbalance_buy_sell_flag"] * out["wap_mid_gap_bps"]
    out["near_far_dislocation_bps"] = _safe_ratio(
        (out["near_price"] - out["far_price"]) * 10_000.0, out["wap"]
    )
    out["time_remaining_seconds"] = config.data.closing_second - out["seconds_in_bucket"]
    out["time_remaining_fraction"] = out["time_remaining_seconds"] / config.data.closing_second
    out["time_remaining_squared"] = out["time_remaining_fraction"] ** 2
    out["imbalance_time_interaction"] = (
        out["auction_imbalance_ratio"] * out["time_remaining_fraction"]
    )

    grouped = out.groupby(["stock_id", "date_id"], sort=False, observed=True)
    for lag_seconds in config.features.lag_seconds:
        periods = lag_seconds // config.data.sample_interval_seconds
        lagged_seconds = grouped["seconds_in_bucket"].shift(periods)
        exact_lag = out["seconds_in_bucket"].sub(lagged_seconds).eq(lag_seconds)
        lagged_wap = grouped["wap"].shift(periods)
        lagged_midpoint = grouped["midpoint"].shift(periods)
        lagged_wap = lagged_wap.where(exact_lag)
        lagged_midpoint = lagged_midpoint.where(exact_lag)
        out[f"wap_return_{lag_seconds}s_bps"] = _safe_ratio(
            (out["wap"] - lagged_wap) * 10_000.0, lagged_wap
        )
        out[f"midpoint_return_{lag_seconds}s_bps"] = _safe_ratio(
            (out["midpoint"] - lagged_midpoint) * 10_000.0, lagged_midpoint
        )
        for source, output_name in (
            ("imbalance_size", "imbalance_size_change"),
            ("matched_size", "matched_size_change"),
            ("spread_bps", "spread_bps_change"),
            ("displayed_liquidity", "displayed_liquidity_change"),
        ):
            lagged = grouped[source].shift(periods).where(exact_lag)
            out[f"{output_name}_{lag_seconds}s"] = out[source] - lagged

    for window_seconds in config.features.rolling_windows_seconds:
        periods = window_seconds // config.data.sample_interval_seconds
        out[f"wap_return_10s_mean_{window_seconds}s"] = _within_group_rolling(
            out, "wap_return_10s_bps", periods, "mean"
        )
        out[f"wap_return_10s_vol_{window_seconds}s"] = _within_group_rolling(
            out, "wap_return_10s_bps", periods, "std"
        )
        out[f"auction_imbalance_ratio_mean_{window_seconds}s"] = _within_group_rolling(
            out, "auction_imbalance_ratio", periods, "mean"
        )
        out[f"spread_bps_mean_{window_seconds}s"] = _within_group_rolling(
            out, "spread_bps", periods, "mean"
        )

    cross_groups = out.groupby("time_id", sort=False, observed=True)
    for source in config.features.cross_sectional_columns:
        values = out[source]
        median = cross_groups[source].transform("median")
        absolute_deviation = (values - median).abs()
        mad = absolute_deviation.groupby(out["time_id"], sort=False).transform("median")
        scale = (1.4826 * mad).where(mad.gt(0))
        out[f"{source}_cs_rank"] = cross_groups[source].rank(pct=True, method="average")
        out[f"{source}_cs_robust_z"] = ((values - median) / scale).fillna(0.0)
        out[f"{source}_cs_demeaned"] = values - cross_groups[source].transform("mean")

    missing = [column for column in equity_model_feature_columns(config) if column not in out]
    if missing:
        raise AssertionError(f"feature construction omitted registered columns: {missing}")
    return out
