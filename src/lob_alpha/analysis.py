"""Day-aware statistical summaries that avoid row-level independence claims."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


def spearman_correlation(left: pd.Series, right: pd.Series) -> float:
    """Return Spearman correlation without warnings for constant inputs."""

    clean = pd.concat([left, right], axis=1).dropna()
    if len(clean) < 2 or clean.iloc[:, 0].nunique() < 2 or clean.iloc[:, 1].nunique() < 2:
        return float("nan")
    return float(clean.iloc[:, 0].corr(clean.iloc[:, 1], method="spearman"))


@dataclass(frozen=True)
class BootstrapInterval:
    estimate: float
    lower: float
    upper: float
    confidence: float
    clusters: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def daily_information_coefficient(
    frame: pd.DataFrame,
    *,
    signal_column: str,
    target_column: str,
    date_column: str = "session_date",
) -> pd.DataFrame:
    """Calculate Spearman IC separately for each trading session."""

    rows = []
    for session_date, group in frame.groupby(date_column, sort=True):
        clean = group[[signal_column, target_column]].dropna()
        correlation = spearman_correlation(clean[signal_column], clean[target_column])
        rows.append({date_column: session_date, "rows": len(clean), "spearman_ic": correlation})
    return pd.DataFrame(rows)


def fit_quantile_edges(series: pd.Series, *, bins: int = 10) -> np.ndarray:
    if bins < 2:
        raise ValueError("bins must be at least 2")
    values = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
    if not len(values):
        raise ValueError("cannot fit quantiles on empty data")
    edges = np.quantile(values, np.linspace(0.0, 1.0, bins + 1))
    edges[0] = -np.inf
    edges[-1] = np.inf
    if np.unique(edges).size < 3:
        raise ValueError("signal has insufficient variation for quantile analysis")
    return np.unique(edges)


def apply_quantile_analysis(
    frame: pd.DataFrame,
    *,
    signal_column: str,
    target_column: str,
    edges: np.ndarray,
) -> pd.DataFrame:
    clean = frame[[signal_column, target_column]].dropna().copy()
    clean["signal_bin"] = pd.cut(
        clean[signal_column], bins=edges, labels=False, include_lowest=True, duplicates="drop"
    )
    return (
        clean.groupby("signal_bin", observed=True)[target_column]
        .agg([("rows", "size"), ("mean_target_ticks", "mean"), ("median_target_ticks", "median")])
        .reset_index()
    )


def cluster_bootstrap_mean(
    cluster_values: pd.Series | np.ndarray,
    *,
    repetitions: int = 2_000,
    confidence: float = 0.95,
    seed: int = 20260815,
) -> BootstrapInterval:
    values = np.asarray(cluster_values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        raise ValueError("at least two finite clusters are required")
    if repetitions <= 0 or not 0 < confidence < 1:
        raise ValueError("invalid bootstrap repetitions or confidence")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(repetitions, len(values)))
    estimates = values[indices].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return BootstrapInterval(
        estimate=float(values.mean()),
        lower=float(np.quantile(estimates, alpha)),
        upper=float(np.quantile(estimates, 1.0 - alpha)),
        confidence=confidence,
        clusters=len(values),
    )


def summarize_trades(trades: pd.DataFrame) -> dict[str, float | int]:
    if trades.empty:
        return {
            "trades": 0,
            "gross_pnl_usd": 0.0,
            "explicit_fees_usd": 0.0,
            "net_pnl_usd": 0.0,
            "mean_net_pnl_per_trade_usd": float("nan"),
            "hit_rate": float("nan"),
        }
    net = trades["net_pnl_usd"].to_numpy(dtype=float)
    return {
        "trades": len(trades),
        "gross_pnl_usd": float(trades["gross_pnl_usd"].sum()),
        "explicit_fees_usd": float(trades["explicit_fees_usd"].sum()),
        "net_pnl_usd": float(net.sum()),
        "mean_net_pnl_per_trade_usd": float(net.mean()),
        "hit_rate": float(np.mean(net > 0)),
    }
