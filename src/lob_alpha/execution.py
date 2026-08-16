"""Observable-depth marketable fills and exact round-trip P&L."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


class InsufficientDepthError(RuntimeError):
    """Raised instead of inventing liquidity beyond the visible book."""


@dataclass(frozen=True)
class Fill:
    side: str
    requested_quantity: int
    filled_quantity: int
    average_price: float
    levels_used: int


@dataclass(frozen=True)
class RoundTrip:
    position_side: str
    quantity: int
    entry_price: float
    exit_price: float
    gross_pnl_usd: float
    explicit_fees_usd: float
    net_pnl_usd: float


def sweep_book(row: Mapping[str, float], *, side: str, quantity: int, depth: int = 10) -> Fill:
    """Consume visible asks for a buy or bids for a sell."""

    if side not in {"buy", "sell"}:
        raise ValueError("side must be 'buy' or 'sell'")
    if quantity <= 0:
        raise ValueError("quantity must be positive")
    remaining = quantity
    notional = 0.0
    levels_used = 0
    prefix = "ask" if side == "buy" else "bid"
    for level in range(depth):
        price = float(row[f"{prefix}_px_{level:02d}"])
        available = int(row[f"{prefix}_sz_{level:02d}"])
        if price <= 0 or available <= 0:
            continue
        fill_quantity = min(remaining, available)
        notional += fill_quantity * price
        remaining -= fill_quantity
        levels_used = level + 1
        if remaining == 0:
            break
    if remaining:
        raise InsufficientDepthError(
            f"only {quantity - remaining} of {quantity} contracts visible across {depth} levels"
        )
    return Fill(
        side=side,
        requested_quantity=quantity,
        filled_quantity=quantity,
        average_price=notional / quantity,
        levels_used=levels_used,
    )


def round_trip_pnl(
    entry_book: Mapping[str, float],
    exit_book: Mapping[str, float],
    *,
    position_side: str,
    quantity: int,
    multiplier: float,
    fee_per_contract_per_side_usd: float,
) -> RoundTrip:
    """Calculate P&L without midpoint fills or double-counting the spread."""

    if position_side not in {"long", "short"}:
        raise ValueError("position_side must be 'long' or 'short'")
    if multiplier <= 0 or fee_per_contract_per_side_usd < 0:
        raise ValueError("multiplier must be positive and fees nonnegative")
    if position_side == "long":
        entry = sweep_book(entry_book, side="buy", quantity=quantity)
        exit_fill = sweep_book(exit_book, side="sell", quantity=quantity)
        price_pnl = exit_fill.average_price - entry.average_price
    else:
        entry = sweep_book(entry_book, side="sell", quantity=quantity)
        exit_fill = sweep_book(exit_book, side="buy", quantity=quantity)
        price_pnl = entry.average_price - exit_fill.average_price
    gross = price_pnl * multiplier * quantity
    fees = 2.0 * fee_per_contract_per_side_usd * quantity
    return RoundTrip(
        position_side=position_side,
        quantity=quantity,
        entry_price=entry.average_price,
        exit_price=exit_fill.average_price,
        gross_pnl_usd=gross,
        explicit_fees_usd=fees,
        net_pnl_usd=gross - fees,
    )
