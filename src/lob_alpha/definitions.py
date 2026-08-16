"""Point-in-time futures definition verification."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import ContractConfig


class DefinitionError(ValueError):
    """Raised when a contract definition is absent or contradicts the config."""


@dataclass(frozen=True)
class VerifiedContract:
    symbol: str
    tick_size: float
    multiplier: float
    expiration: str | None
    currency: str | None
    security_type: str | None


def _scaled_value(value: object) -> float:
    result = float(value)
    # CSV/DBN integer encodings use fixed precision 1e-9; to_df(price_type="float")
    # already scales them. Supporting both here prevents silent unit errors.
    if abs(result) >= 1_000_000:
        result /= 1_000_000_000.0
    return result


def verify_contract_definition(
    definitions: pd.DataFrame,
    *,
    symbol: str,
    expected: ContractConfig,
    absolute_tolerance: float = 1e-9,
) -> VerifiedContract:
    symbol_column = "raw_symbol" if "raw_symbol" in definitions.columns else "symbol"
    if symbol_column not in definitions.columns:
        raise DefinitionError("definition data has neither raw_symbol nor symbol")
    matches = definitions.loc[definitions[symbol_column].astype(str) == symbol].copy()
    if matches.empty:
        raise DefinitionError(f"no point-in-time definition found for {symbol}")
    if "ts_recv" in matches.columns:
        matches["ts_recv"] = pd.to_datetime(matches["ts_recv"], utc=True, errors="coerce")
        matches = matches.sort_values("ts_recv", kind="stable")
    row = matches.iloc[-1]

    if "min_price_increment" not in row or "unit_of_measure_qty" not in row:
        raise DefinitionError("definition lacks tick-size or contract-size fields")
    tick_size = _scaled_value(row["min_price_increment"])
    multiplier = _scaled_value(row["unit_of_measure_qty"])
    if not np.isfinite(tick_size) or tick_size <= 0:
        raise DefinitionError("invalid min_price_increment")
    if not np.isfinite(multiplier) or multiplier <= 0:
        raise DefinitionError("invalid unit_of_measure_qty")

    if expected.require_definition_match:
        if not np.isclose(
            tick_size,
            expected.expected_tick_size,
            atol=absolute_tolerance,
            rtol=0.0,
        ):
            raise DefinitionError(
                f"definition tick {tick_size} != configured {expected.expected_tick_size}"
            )
        if not np.isclose(
            multiplier, expected.expected_multiplier, atol=absolute_tolerance, rtol=0.0
        ):
            raise DefinitionError(
                f"definition multiplier {multiplier} != configured {expected.expected_multiplier}"
            )

    expiration = None
    if "expiration" in row and pd.notna(row["expiration"]):
        expiration = str(pd.to_datetime(row["expiration"], utc=True))
    return VerifiedContract(
        symbol=symbol,
        tick_size=tick_size,
        multiplier=multiplier,
        expiration=expiration,
        currency=str(row["currency"]) if "currency" in row and pd.notna(row["currency"]) else None,
        security_type=(
            str(row["security_type"])
            if "security_type" in row and pd.notna(row["security_type"])
            else None
        ),
    )
