"""Daily causal feature/label pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd

from .config import ResearchConfig
from .features import build_features, model_feature_columns
from .labels import build_labels
from .sampling import filter_session, sample_decision_states, session_bounds
from .schema import canonicalize_mbp10
from .validation import QualityReport, require_usable, validate_mbp10


@dataclass(frozen=True)
class SessionResult:
    data: pd.DataFrame
    quality: QualityReport


def process_session(
    events: pd.DataFrame,
    config: ResearchConfig,
    *,
    session_date: date,
    tick_size: float,
) -> SessionResult:
    """Validate and transform one session without crossing its boundaries."""

    # Select the complete configured session before any rolling operation so
    # pre-session/overnight events cannot leak into feature windows.
    raw_session = filter_session(events, session_date, config.session)
    quality = validate_mbp10(raw_session, tick_size=tick_size)
    require_usable(quality)
    canonical = canonicalize_mbp10(raw_session)
    states = sample_decision_states(canonical, session_date, config.session)
    features = build_features(
        canonical,
        states,
        config.features,
        tick_size=tick_size,
        decision_grid_ms=config.session.decision_grid_ms,
    )
    labels = build_labels(
        canonical,
        features,
        config.labels.horizons_ms,
        tick_size=tick_size,
        maximum_age_ms=config.session.maximum_quote_age_ms,
    )
    dataset = features.merge(labels, on="decision_time", how="left", validate="one_to_one")

    session_start, session_end = session_bounds(session_date, config.session)
    maximum_lookback = max(
        *config.features.ofi_lookbacks_ms,
        *config.features.trade_lookbacks_ms,
        config.features.event_intensity_lookback_ms,
        config.features.lagged_return_ms,
    )
    maximum_horizon = max(config.labels.horizons_ms)
    dataset = dataset.loc[
        (dataset["decision_time"] >= session_start + pd.to_timedelta(maximum_lookback, unit="ms"))
        & (dataset["decision_time"] <= session_end - pd.to_timedelta(maximum_horizon, unit="ms"))
    ].copy()
    required = model_feature_columns(config.features) + [
        f"target_{horizon}ms_ticks" for horizon in config.labels.horizons_ms
    ]
    dataset = dataset.dropna(subset=required).reset_index(drop=True)
    dataset.insert(0, "session_date", session_date.isoformat())
    dataset.insert(1, "split", config.splits.split_for(session_date))
    return SessionResult(dataset, quality)


def write_table(frame: pd.DataFrame, path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    lowered = output.name.lower()
    temporary = output.with_name(output.name + ".partial")
    try:
        if lowered.endswith(".parquet"):
            frame.to_parquet(temporary, index=False)
        elif lowered.endswith(".csv.gz"):
            frame.to_csv(
                temporary,
                index=False,
                compression={"method": "gzip", "mtime": 0},
            )
        elif lowered.endswith(".csv"):
            frame.to_csv(temporary, index=False)
        else:
            raise ValueError("output must end in .parquet, .csv, or .csv.gz")
        temporary.replace(output)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise
    return output
