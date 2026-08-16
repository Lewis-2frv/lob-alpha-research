"""Causal fixed-horizon marketable strategy simulation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .execution import InsufficientDepthError, round_trip_pnl

TRADE_COLUMNS = [
    "decision_time",
    "position_side",
    "prediction_ticks",
    "estimated_hurdle_ticks",
    "entry_target_time",
    "entry_source_time",
    "exit_target_time",
    "exit_source_time",
    "quantity",
    "entry_price",
    "exit_price",
    "gross_pnl_usd",
    "explicit_fees_usd",
    "net_pnl_usd",
]


def _book_asof(
    events: pd.DataFrame,
    event_ns: np.ndarray,
    target_time: pd.Timestamp,
    *,
    maximum_age_ms: int,
) -> tuple[pd.Series, pd.Timestamp] | None:
    target_ns = target_time.as_unit("ns").value
    index = int(np.searchsorted(event_ns, target_ns, side="right") - 1)
    if index < 0:
        return None
    row = events.iloc[index]
    source = pd.Timestamp(row["ts_recv"])
    age_ms = (target_time - source).total_seconds() * 1000.0
    if age_ms < 0 or age_ms > maximum_age_ms:
        return None
    return row, source


def simulate_marketable_strategy(
    events: pd.DataFrame,
    signals: pd.DataFrame,
    *,
    prediction_column: str,
    horizon_ms: int,
    latency_ms: int,
    quantity: int,
    tick_size: float,
    multiplier: float,
    fee_per_contract_per_side_usd: float,
    safety_margin_ticks: float,
    maximum_quote_age_ms: int,
) -> pd.DataFrame:
    """Run one-position, fixed-horizon execution without future book access.

    The pre-trade hurdle uses the contemporaneous spread plus explicit round-trip
    fees and a registered safety margin. Actual P&L uses book-sweep VWAPs at the
    delayed entry and exit timestamps.
    """

    if horizon_ms <= 0 or latency_ms < 0 or maximum_quote_age_ms < 0:
        raise ValueError("horizon must be positive; latency and quote age must be nonnegative")
    if tick_size <= 0 or multiplier <= 0 or quantity <= 0:
        raise ValueError("tick size, multiplier and quantity must be positive")
    if fee_per_contract_per_side_usd < 0 or safety_margin_ticks < 0:
        raise ValueError("fees and safety margin must be nonnegative")
    if prediction_column not in signals.columns:
        raise ValueError(f"missing prediction column: {prediction_column}")

    ordered_events = events.sort_values(["ts_recv", "sequence"], kind="stable").reset_index(
        drop=True
    )
    ordered_events["ts_recv"] = pd.to_datetime(ordered_events["ts_recv"], utc=True)
    event_ns = pd.DatetimeIndex(ordered_events["ts_recv"]).as_unit("ns").asi8
    ordered_signals = signals.sort_values("decision_time", kind="stable").copy()
    ordered_signals["decision_time"] = pd.to_datetime(ordered_signals["decision_time"], utc=True)

    fee_ticks = 2.0 * fee_per_contract_per_side_usd / (tick_size * multiplier)
    next_available_time: pd.Timestamp | None = None
    trades: list[dict[str, object]] = []

    for signal in ordered_signals.itertuples(index=False):
        decision_time = pd.Timestamp(signal.decision_time)
        if next_available_time is not None and decision_time < next_available_time:
            continue
        prediction = float(getattr(signal, prediction_column))
        if not np.isfinite(prediction) or prediction == 0:
            continue

        decision_book_match = _book_asof(
            ordered_events,
            event_ns,
            decision_time,
            maximum_age_ms=maximum_quote_age_ms,
        )
        if decision_book_match is None:
            continue
        decision_book, _ = decision_book_match
        spread_ticks = (
            float(decision_book["ask_px_00"]) - float(decision_book["bid_px_00"])
        ) / tick_size
        hurdle_ticks = spread_ticks + fee_ticks + safety_margin_ticks
        if abs(prediction) <= hurdle_ticks:
            continue

        entry_target = decision_time + pd.to_timedelta(latency_ms, unit="ms")
        exit_target = entry_target + pd.to_timedelta(horizon_ms, unit="ms")
        entry_match = _book_asof(
            ordered_events, event_ns, entry_target, maximum_age_ms=maximum_quote_age_ms
        )
        exit_match = _book_asof(
            ordered_events, event_ns, exit_target, maximum_age_ms=maximum_quote_age_ms
        )
        if entry_match is None or exit_match is None:
            continue
        entry_book, entry_source = entry_match
        exit_book, exit_source = exit_match
        position_side = "long" if prediction > 0 else "short"
        try:
            outcome = round_trip_pnl(
                entry_book,
                exit_book,
                position_side=position_side,
                quantity=quantity,
                multiplier=multiplier,
                fee_per_contract_per_side_usd=fee_per_contract_per_side_usd,
            )
        except InsufficientDepthError:
            continue

        trades.append(
            {
                "decision_time": decision_time,
                "position_side": position_side,
                "prediction_ticks": prediction,
                "estimated_hurdle_ticks": hurdle_ticks,
                "entry_target_time": entry_target,
                "entry_source_time": entry_source,
                "exit_target_time": exit_target,
                "exit_source_time": exit_source,
                "quantity": quantity,
                "entry_price": outcome.entry_price,
                "exit_price": outcome.exit_price,
                "gross_pnl_usd": outcome.gross_pnl_usd,
                "explicit_fees_usd": outcome.explicit_fees_usd,
                "net_pnl_usd": outcome.net_pnl_usd,
            }
        )
        next_available_time = exit_target
    return pd.DataFrame(trades, columns=TRADE_COLUMNS)
