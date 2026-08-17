"""Unmistakably synthetic Optiver-shaped engineering data."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .equity_config import EquityResearchConfig
from .equity_data import OPTIVER_COLUMNS


def make_synthetic_optiver(
    config: EquityResearchConfig,
    *,
    stocks: int = 8,
) -> pd.DataFrame:
    """Create a deterministic panel for mechanics, never empirical evidence."""

    if config.source_kind != "synthetic":
        raise ValueError("synthetic fixture generation requires source_kind=synthetic")
    random = np.random.default_rng(config.seed)
    rows = []
    for date_id in range(config.data.expected_date_id_min, config.data.expected_date_id_max + 1):
        prices = 1.0 + np.arange(stocks) * 0.002 + date_id * 0.00005
        for seconds in range(0, config.data.closing_second, config.data.sample_interval_seconds):
            time_id = date_id * 10_000 + seconds // config.data.sample_interval_seconds
            market_shock = random.normal(0.0, 0.000015)
            for stock_id in range(stocks):
                flag = (-1, 0, 1)[(stock_id + seconds // 10 + date_id) % 3]
                imbalance = 500.0 + 35.0 * stock_id + 2.0 * seconds
                matched = 10_000.0 + 200.0 * stock_id + 25.0 * seconds
                pressure = flag * imbalance / (matched + imbalance)
                prices[stock_id] *= 1.0 + market_shock + pressure * 0.00018
                midpoint = prices[stock_id]
                half_spread = 0.00008 + 0.00001 * (stock_id % 3)
                bid_price = midpoint - half_spread
                ask_price = midpoint + half_spread
                bid_size = 900.0 + 20.0 * stock_id + 3.0 * (seconds % 50)
                ask_size = 850.0 + 15.0 * stock_id + 2.0 * ((seconds + 20) % 50)
                wap = (bid_price * ask_size + ask_price * bid_size) / (bid_size + ask_size)
                near_price = np.nan if seconds < 200 else midpoint + pressure * 0.001
                far_price = np.nan if seconds < 300 else midpoint + pressure * 0.0015
                target = pressure * 35.0 + (stock_id - (stocks - 1) / 2) * 0.015
                target += random.normal(0.0, 0.08)
                rows.append(
                    {
                        "stock_id": stock_id,
                        "date_id": date_id,
                        "seconds_in_bucket": seconds,
                        "imbalance_size": imbalance,
                        "imbalance_buy_sell_flag": flag,
                        "reference_price": midpoint + pressure * 0.0003,
                        "matched_size": matched,
                        "far_price": far_price,
                        "near_price": near_price,
                        "bid_price": bid_price,
                        "bid_size": bid_size,
                        "ask_price": ask_price,
                        "ask_size": ask_size,
                        "wap": wap,
                        "target": target,
                        "time_id": time_id,
                        "row_id": f"SYNTHETIC_{date_id}_{seconds}_{stock_id}",
                    }
                )
    return pd.DataFrame(rows, columns=OPTIVER_COLUMNS)


def write_synthetic_optiver(config: EquityResearchConfig, path: str | Path) -> Path:
    output = Path(path)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite synthetic fixture: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    make_synthetic_optiver(config).to_csv(output, index=False)
    return output
