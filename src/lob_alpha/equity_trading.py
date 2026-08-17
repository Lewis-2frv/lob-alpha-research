"""Conservative 60-second executable-quote cross-sectional evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TradingRule:
    group_quantile: float
    minimum_absolute_prediction_bps: float
    maximum_spread_bps: float
    minimum_displayed_liquidity: float
    fee_per_side_bps: float


def align_future_quotes(frame: pd.DataFrame, *, horizon_seconds: int = 60) -> pd.DataFrame:
    """Attach only exact-horizon future quotes for realised evaluation."""

    keys = ["stock_id", "date_id", "seconds_in_bucket"]
    if frame.duplicated(keys).any():
        raise ValueError("future quote alignment requires unique stock/date/second rows")
    future = frame.loc[:, [*keys, "bid_price", "bid_size", "ask_price", "ask_size", "wap"]].copy()
    future["seconds_in_bucket"] -= horizon_seconds
    future = future.rename(
        columns={
            "bid_price": "future_bid_price",
            "bid_size": "future_bid_size",
            "ask_price": "future_ask_price",
            "ask_size": "future_ask_size",
            "wap": "future_wap",
        }
    )
    return frame.merge(future, on=keys, how="left", validate="one_to_one", sort=False)


def _quote_validity(frame: pd.DataFrame) -> pd.Series:
    required = [
        "bid_price",
        "ask_price",
        "bid_size",
        "ask_size",
        "future_bid_price",
        "future_ask_price",
        "future_bid_size",
        "future_ask_size",
    ]
    finite = np.isfinite(frame[required].to_numpy(dtype=float)).all(axis=1)
    positive = frame[required].gt(0).all(axis=1)
    uncrossed = (frame["ask_price"] >= frame["bid_price"]) & (
        frame["future_ask_price"] >= frame["future_bid_price"]
    )
    return pd.Series(finite, index=frame.index) & positive & uncrossed


def _execution_panel(
    frame: pd.DataFrame,
    predictions: np.ndarray,
    rule: TradingRule,
    *,
    horizon_seconds: int,
    decision_interval_seconds: int,
) -> pd.DataFrame:
    if len(frame) != len(predictions):
        raise ValueError("predictions are not aligned with evaluation rows")
    if decision_interval_seconds < horizon_seconds:
        raise ValueError("decision interval would create overlapping positions")
    aligned = align_future_quotes(frame, horizon_seconds=horizon_seconds)
    aligned["prediction_bps"] = predictions
    aligned["spread_bps_current"] = (
        (aligned["ask_price"] - aligned["bid_price"])
        / ((aligned["ask_price"] + aligned["bid_price"]) / 2.0)
        * 10_000.0
    )
    aligned["displayed_liquidity_current"] = aligned["bid_size"] + aligned["ask_size"]
    aligned["decision_time"] = aligned["seconds_in_bucket"].mod(
        decision_interval_seconds
    ).eq(0)
    aligned["valid_quotes"] = _quote_validity(aligned)
    aligned["prediction_threshold"] = (
        np.isfinite(aligned["prediction_bps"])
        & aligned["prediction_bps"].abs().ge(rule.minimum_absolute_prediction_bps)
    )
    aligned["spread_eligible"] = (
        np.isfinite(aligned["spread_bps_current"])
        & aligned["spread_bps_current"].le(rule.maximum_spread_bps)
    )
    aligned["liquidity_eligible"] = (
        np.isfinite(aligned["displayed_liquidity_current"])
        & aligned["displayed_liquidity_current"].ge(rule.minimum_displayed_liquidity)
    )
    return aligned


def _diagnostics_from_panel(
    aligned: pd.DataFrame,
    rule: TradingRule,
) -> dict[str, int]:
    future_columns = [
        "future_bid_price",
        "future_ask_price",
        "future_bid_size",
        "future_ask_size",
    ]
    decision = aligned["decision_time"]
    valid = decision & aligned["valid_quotes"]
    threshold = valid & aligned["prediction_threshold"]
    spread = threshold & aligned["spread_eligible"]
    eligible = spread & aligned["liquidity_eligible"]
    one_sided = 0
    insufficient = 0
    for _, group in aligned.loc[eligible].groupby("time_id", sort=True, observed=True):
        group_size = int(np.floor(len(group) * rule.group_quantile))
        if group_size < 1:
            insufficient += 1
        elif (
            group["prediction_bps"].gt(0).sum() < group_size
            or group["prediction_bps"].lt(0).sum() < group_size
        ):
            one_sided += 1
    return {
        "input_rows": int(len(aligned)),
        "decision_grid_rows": int(decision.sum()),
        "missing_exact_future_quote_rows": int(
            (decision & aligned[future_columns].isna().any(axis=1)).sum()
        ),
        "invalid_quote_rows": int((decision & ~aligned["valid_quotes"]).sum()),
        "prediction_threshold_rejections": int((valid & ~aligned["prediction_threshold"]).sum()),
        "spread_rejections": int((threshold & ~aligned["spread_eligible"]).sum()),
        "liquidity_rejections": int((spread & ~aligned["liquidity_eligible"]).sum()),
        "eligible_rows": int(eligible.sum()),
        "insufficient_cross_section_time_ids": insufficient,
        "one_sided_time_ids": one_sided,
    }


def execution_filter_diagnostics(
    frame: pd.DataFrame,
    predictions: np.ndarray,
    rule: TradingRule,
    *,
    horizon_seconds: int = 60,
    decision_interval_seconds: int = 60,
) -> dict[str, int]:
    """Count sequential execution filters without asserting that liquidity guarantees fills."""

    aligned = _execution_panel(
        frame,
        predictions,
        rule,
        horizon_seconds=horizon_seconds,
        decision_interval_seconds=decision_interval_seconds,
    )
    return _diagnostics_from_panel(aligned, rule)


def simulate_cross_sectional_trading(
    frame: pd.DataFrame,
    predictions: np.ndarray,
    rule: TradingRule,
    *,
    horizon_seconds: int = 60,
    decision_interval_seconds: int = 60,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cross current quotes once and exit at exact future executable quotes."""

    aligned = _execution_panel(
        frame,
        predictions,
        rule,
        horizon_seconds=horizon_seconds,
        decision_interval_seconds=decision_interval_seconds,
    )
    eligible = (
        aligned["valid_quotes"]
        & aligned["decision_time"]
        & aligned["prediction_threshold"]
        & aligned["spread_eligible"]
        & aligned["liquidity_eligible"]
    )
    leg_rows: list[dict[str, float | int | str]] = []
    decision_rows: list[dict[str, float | int]] = []
    for time_id, group in aligned.loc[eligible].groupby("time_id", sort=True, observed=True):
        ordered = group.sort_values(["prediction_bps", "stock_id"], kind="stable")
        group_size = int(np.floor(len(ordered) * rule.group_quantile))
        if group_size < 1 or len(ordered) < 2 * group_size:
            continue
        short_pool = ordered.loc[ordered["prediction_bps"].lt(0)]
        long_pool = ordered.loc[ordered["prediction_bps"].gt(0)]
        if len(short_pool) < group_size or len(long_pool) < group_size:
            continue
        short_rows = short_pool.head(group_size)
        long_rows = long_pool.tail(group_size)
        if set(short_rows.index) & set(long_rows.index):
            raise AssertionError("long and short selections overlap")
        decision_legs = []
        for side, selected in (("short", short_rows), ("long", long_rows)):
            for row in selected.itertuples(index=False):
                current_mid = (row.bid_price + row.ask_price) / 2.0
                future_mid = (row.future_bid_price + row.future_ask_price) / 2.0
                if side == "long":
                    executable = (row.future_bid_price / row.ask_price - 1.0) * 10_000.0
                    midpoint_return = (future_mid / current_mid - 1.0) * 10_000.0
                    capacity = min(row.ask_size, row.future_bid_size)
                else:
                    executable = (
                        (row.bid_price - row.future_ask_price) / row.bid_price * 10_000.0
                    )
                    midpoint_return = (current_mid - future_mid) / current_mid * 10_000.0
                    capacity = min(row.bid_size, row.future_ask_size)
                total_fee = 2.0 * rule.fee_per_side_bps
                weight = 0.5 / group_size
                leg = {
                    "date_id": int(row.date_id),
                    "time_id": int(time_id),
                    "seconds_in_bucket": int(row.seconds_in_bucket),
                    "stock_id": int(row.stock_id),
                    "side": side,
                    "weight": weight,
                    "prediction_bps": float(row.prediction_bps),
                    "midpoint_return_bps": float(midpoint_return),
                    "gross_executable_return_bps": float(executable),
                    "spread_cost_bps": float(midpoint_return - executable),
                    "fee_bps": float(total_fee),
                    "net_return_bps": float(executable - total_fee),
                    "displayed_capacity_units": float(capacity),
                }
                leg_rows.append(leg)
                decision_legs.append(leg)
        decision_rows.append(
            {
                "date_id": int(decision_legs[0]["date_id"]),
                "time_id": int(time_id),
                "seconds_in_bucket": int(decision_legs[0]["seconds_in_bucket"]),
                "stock_legs": len(decision_legs),
                "gross_return_bps": float(
                    sum(
                        float(item["weight"]) * float(item["gross_executable_return_bps"])
                        for item in decision_legs
                    )
                ),
                "net_return_bps": float(
                    sum(
                        float(item["weight"]) * float(item["net_return_bps"])
                        for item in decision_legs
                    )
                ),
                "spread_cost_bps": float(
                    sum(
                        float(item["weight"]) * float(item["spread_cost_bps"])
                        for item in decision_legs
                    )
                ),
                "fee_bps": float(
                    sum(float(item["weight"]) * float(item["fee_bps"]) for item in decision_legs)
                ),
                "minimum_displayed_capacity_units": float(
                    min(float(item["displayed_capacity_units"]) for item in decision_legs)
                ),
            }
        )
    legs = pd.DataFrame(leg_rows)
    decisions = pd.DataFrame(decision_rows)
    diagnostics = _diagnostics_from_panel(aligned, rule)
    legs.attrs["execution_filter_diagnostics"] = diagnostics
    decisions.attrs["execution_filter_diagnostics"] = diagnostics
    if not decisions.empty:
        differences = (
            decisions.sort_values(["date_id", "seconds_in_bucket"])
            .groupby("date_id", observed=True)["seconds_in_bucket"]
            .diff()
        )
        if differences.dropna().lt(horizon_seconds).any():
            raise AssertionError("overlapping cross-sectional decisions were created")
    return legs, decisions


def cluster_bootstrap_by_date(
    daily_values: pd.Series,
    *,
    repetitions: int,
    seed: int,
) -> dict[str, float | int]:
    values = pd.to_numeric(daily_values, errors="coerce").dropna().to_numpy(dtype=float)
    if len(values) < 2:
        return {
            "dates": len(values),
            "mean": float(np.mean(values)) if len(values) else None,
            "ci_low": None,
            "ci_high": None,
        }
    random = np.random.default_rng(seed)
    draws = random.choice(values, size=(repetitions, len(values)), replace=True).mean(axis=1)
    return {
        "dates": len(values),
        "mean": float(values.mean()),
        "ci_low": float(np.quantile(draws, 0.025)),
        "ci_high": float(np.quantile(draws, 0.975)),
    }


def summarize_trading(
    legs: pd.DataFrame,
    decisions: pd.DataFrame,
    *,
    repetitions: int,
    seed: int,
) -> dict[str, object]:
    if decisions.empty:
        return {
            "decisions": 0,
            "stock_legs": 0,
            "gross_mean_bps": None,
            "net_mean_bps": None,
            "net_median_bps": None,
            "long_gross_mean_bps": None,
            "long_net_mean_bps": None,
            "short_gross_mean_bps": None,
            "short_net_mean_bps": None,
            "hit_rate": None,
            "turnover_round_trip_gross": 0.0,
            "spread_cost_mean_bps": None,
            "fee_mean_bps": None,
            "break_even_additional_total_cost_bps": None,
            "minimum_displayed_capacity_units": None,
            "daily_cluster_bootstrap": cluster_bootstrap_by_date(
                pd.Series(dtype=float), repetitions=repetitions, seed=seed
            ),
            "daily_return_distribution": None,
        }
    daily = decisions.groupby("date_id", sort=True, observed=True)["net_return_bps"].mean()
    sleeve = (
        legs.groupby(["date_id", "time_id", "side"], sort=True, observed=True)
        .agg(
            gross_return_bps=("gross_executable_return_bps", "mean"),
            net_return_bps=("net_return_bps", "mean"),
        )
        .reset_index()
    )

    def sleeve_mean(side: str, column: str) -> float | None:
        selected = sleeve.loc[sleeve["side"].eq(side), column]
        return float(selected.mean()) if not selected.empty else None

    return {
        "decisions": len(decisions),
        "stock_legs": len(legs),
        "gross_mean_bps": float(decisions["gross_return_bps"].mean()),
        "net_mean_bps": float(decisions["net_return_bps"].mean()),
        "net_median_bps": float(decisions["net_return_bps"].median()),
        "long_gross_mean_bps": sleeve_mean("long", "gross_return_bps"),
        "long_net_mean_bps": sleeve_mean("long", "net_return_bps"),
        "short_gross_mean_bps": sleeve_mean("short", "gross_return_bps"),
        "short_net_mean_bps": sleeve_mean("short", "net_return_bps"),
        "hit_rate": float(decisions["net_return_bps"].gt(0).mean()),
        "turnover_round_trip_gross": float(2 * len(decisions)),
        "spread_cost_mean_bps": float(decisions["spread_cost_bps"].mean()),
        "fee_mean_bps": float(decisions["fee_bps"].mean()),
        "break_even_additional_total_cost_bps": float(
            max(0.0, decisions["net_return_bps"].mean())
        ),
        "minimum_displayed_capacity_units": float(
            decisions["minimum_displayed_capacity_units"].min()
        ),
        "daily_cluster_bootstrap": cluster_bootstrap_by_date(
            daily, repetitions=repetitions, seed=seed
        ),
        "daily_return_distribution": {
            "mean_bps": float(daily.mean()),
            "standard_deviation_bps": float(daily.std(ddof=0)),
            "minimum_bps": float(daily.min()),
            "quartile_25_bps": float(daily.quantile(0.25)),
            "median_bps": float(daily.median()),
            "quartile_75_bps": float(daily.quantile(0.75)),
            "maximum_bps": float(daily.max()),
        },
    }
