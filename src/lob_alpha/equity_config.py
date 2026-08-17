"""Typed, frozen configuration for the equity closing-auction study."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class EquityConfigurationError(ValueError):
    """Raised when the registered equity study contract is invalid."""


@dataclass(frozen=True)
class EquityDataConfig:
    raw_path: Path
    prepared_dir: Path
    expected_date_id_min: int
    expected_date_id_max: int
    sample_interval_seconds: int
    target_horizon_seconds: int
    target_definition: str
    closing_second: int
    csv_chunk_rows: int


@dataclass(frozen=True)
class EquitySplitConfig:
    train_start: int
    train_end: int
    validation_start: int
    validation_end: int
    holdout_start: int
    holdout_end: int

    def split_for(self, date_id: int) -> str | None:
        if self.train_start <= date_id <= self.train_end:
            return "train"
        if self.validation_start <= date_id <= self.validation_end:
            return "validation"
        if self.holdout_start <= date_id <= self.holdout_end:
            return "holdout"
        return None


@dataclass(frozen=True)
class EquityFeatureConfig:
    lag_seconds: tuple[int, ...]
    rolling_windows_seconds: tuple[int, ...]
    cross_sectional_columns: tuple[str, ...]


@dataclass(frozen=True)
class EquityModelConfig:
    candidate_models: tuple[str, ...]
    ridge_alphas: tuple[float, ...]
    nonlinear_learning_rates: tuple[float, ...]
    nonlinear_leaf_counts: tuple[int, ...]
    nonlinear_iterations: int
    cv_folds: int
    minimum_train_dates: int
    max_tuning_rows_per_date: int
    max_refit_rows_per_date: int


@dataclass(frozen=True)
class EquityExecutionConfig:
    group_quantiles: tuple[float, ...]
    minimum_absolute_predictions_bps: tuple[float, ...]
    maximum_spreads_bps: tuple[float, ...]
    minimum_displayed_liquidity: tuple[float, ...]
    primary_fee_per_side_bps: float
    fee_sensitivity_per_side_bps: tuple[float, ...]
    decision_interval_seconds: int
    bootstrap_repetitions: int


@dataclass(frozen=True)
class EquityResearchConfig:
    name: str
    seed: int
    source_kind: str
    data: EquityDataConfig
    splits: EquitySplitConfig
    features: EquityFeatureConfig
    model: EquityModelConfig
    execution: EquityExecutionConfig
    source_path: Path


def _tuple(raw: dict[str, Any], key: str, converter: type) -> tuple[Any, ...]:
    try:
        return tuple(converter(value) for value in raw[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise EquityConfigurationError(f"invalid or missing configuration list: {key}") from exc


def load_equity_config(path: str | Path) -> EquityResearchConfig:
    """Load the equity registration without consulting data or performance."""

    source = Path(path).resolve()
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise EquityConfigurationError("configuration root must be a mapping")
    try:
        project = raw["project"]
        data = raw["data"]
        splits = raw["splits"]
        features = raw["features"]
        model = raw["model"]
        execution = raw["execution"]
        config = EquityResearchConfig(
            name=str(project["name"]),
            seed=int(project["seed"]),
            source_kind=str(project["source_kind"]),
            data=EquityDataConfig(
                raw_path=Path(data["raw_path"]),
                prepared_dir=Path(data["prepared_dir"]),
                expected_date_id_min=int(data["expected_date_id_min"]),
                expected_date_id_max=int(data["expected_date_id_max"]),
                sample_interval_seconds=int(data["sample_interval_seconds"]),
                target_horizon_seconds=int(data["target_horizon_seconds"]),
                target_definition=str(data["target_definition"]),
                closing_second=int(data["closing_second"]),
                csv_chunk_rows=int(data["csv_chunk_rows"]),
            ),
            splits=EquitySplitConfig(
                train_start=int(splits["train_start"]),
                train_end=int(splits["train_end"]),
                validation_start=int(splits["validation_start"]),
                validation_end=int(splits["validation_end"]),
                holdout_start=int(splits["holdout_start"]),
                holdout_end=int(splits["holdout_end"]),
            ),
            features=EquityFeatureConfig(
                lag_seconds=_tuple(features, "lag_seconds", int),
                rolling_windows_seconds=_tuple(features, "rolling_windows_seconds", int),
                cross_sectional_columns=_tuple(features, "cross_sectional_columns", str),
            ),
            model=EquityModelConfig(
                candidate_models=_tuple(model, "candidate_models", str),
                ridge_alphas=_tuple(model, "ridge_alphas", float),
                nonlinear_learning_rates=_tuple(model, "nonlinear_learning_rates", float),
                nonlinear_leaf_counts=_tuple(model, "nonlinear_leaf_counts", int),
                nonlinear_iterations=int(model["nonlinear_iterations"]),
                cv_folds=int(model["cv_folds"]),
                minimum_train_dates=int(model["minimum_train_dates"]),
                max_tuning_rows_per_date=int(model["max_tuning_rows_per_date"]),
                max_refit_rows_per_date=int(model["max_refit_rows_per_date"]),
            ),
            execution=EquityExecutionConfig(
                group_quantiles=_tuple(execution, "group_quantiles", float),
                minimum_absolute_predictions_bps=_tuple(
                    execution, "minimum_absolute_predictions_bps", float
                ),
                maximum_spreads_bps=_tuple(execution, "maximum_spreads_bps", float),
                minimum_displayed_liquidity=_tuple(execution, "minimum_displayed_liquidity", float),
                primary_fee_per_side_bps=float(execution["primary_fee_per_side_bps"]),
                fee_sensitivity_per_side_bps=_tuple(
                    execution, "fee_sensitivity_per_side_bps", float
                ),
                decision_interval_seconds=int(execution["decision_interval_seconds"]),
                bootstrap_repetitions=int(execution["bootstrap_repetitions"]),
            ),
            source_path=source,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise EquityConfigurationError(f"invalid equity configuration: {exc}") from exc
    validate_equity_config(config)
    return config


def _increasing(values: tuple[float | int, ...], name: str, *, lower: float = 0) -> None:
    if (
        not values
        or tuple(sorted(set(values))) != values
        or any(not math.isfinite(float(value)) or value < lower for value in values)
    ):
        raise EquityConfigurationError(f"{name} must be unique, increasing and >= {lower}")


def validate_equity_config(config: EquityResearchConfig) -> None:
    """Enforce chronology and the immutable 60-second study design."""

    data = config.data
    split = config.splits
    if config.source_kind not in {"real", "synthetic"}:
        raise EquityConfigurationError("source_kind must be real or synthetic")
    if data.expected_date_id_min < 0 or data.expected_date_id_max <= data.expected_date_id_min:
        raise EquityConfigurationError("expected date_id range is invalid")
    if data.sample_interval_seconds != 10 or data.target_horizon_seconds != 60:
        raise EquityConfigurationError(
            "the registered study requires 10-second data and 60-second targets"
        )
    expected_target = (
        "supplied_optiver_index_relative_60s_bps"
        if config.source_kind == "real"
        else "synthetic_index_relative_60s_bps_proxy"
    )
    if data.target_definition != expected_target:
        raise EquityConfigurationError(
            f"target_definition must be the registered {expected_target!r}"
        )
    if data.closing_second != 600 or data.csv_chunk_rows <= 0:
        raise EquityConfigurationError("invalid close time or CSV chunk size")
    if not (
        split.train_start
        == data.expected_date_id_min
        <= split.train_end
        < split.validation_start
        == split.train_end + 1
        <= split.validation_end
        < split.holdout_start
        == split.validation_end + 1
        <= split.holdout_end
        == data.expected_date_id_max
    ):
        raise EquityConfigurationError(
            "date_id partitions must be contiguous, ordered and exhaustive"
        )
    _increasing(config.features.lag_seconds, "lag_seconds", lower=1)
    _increasing(config.features.rolling_windows_seconds, "rolling_windows_seconds", lower=1)
    if config.features.lag_seconds != (10, 30, 60):
        raise EquityConfigurationError(
            "the registered causal lags are exactly 10, 30 and 60 seconds"
        )
    allowed_models = {"ridge", "hist_gradient_boosting", "lightgbm"}
    if (
        not config.model.candidate_models
        or not set(config.model.candidate_models) <= allowed_models
    ):
        raise EquityConfigurationError("unsupported candidate model")
    if "ridge" not in config.model.candidate_models:
        raise EquityConfigurationError("ridge is a mandatory candidate")
    _increasing(config.model.ridge_alphas, "ridge_alphas", lower=0)
    _increasing(config.model.nonlinear_learning_rates, "learning rates", lower=0)
    _increasing(config.model.nonlinear_leaf_counts, "leaf counts", lower=2)
    if config.model.cv_folds < 2 or config.model.minimum_train_dates < 3:
        raise EquityConfigurationError(
            "chronological CV requires at least two folds and three dates"
        )
    if min(config.model.max_tuning_rows_per_date, config.model.max_refit_rows_per_date) <= 0:
        raise EquityConfigurationError("row sampling bounds must be positive")
    if config.model.nonlinear_iterations <= 0 or any(
        value <= 0 for value in config.model.nonlinear_learning_rates
    ):
        raise EquityConfigurationError(
            "nonlinear iteration counts and learning rates must be positive"
        )
    _increasing(config.execution.group_quantiles, "group_quantiles", lower=0)
    if any(value <= 0 or value >= 0.5 for value in config.execution.group_quantiles):
        raise EquityConfigurationError("group quantiles must be strictly between zero and one half")
    _increasing(
        config.execution.minimum_absolute_predictions_bps,
        "minimum_absolute_predictions_bps",
    )
    _increasing(config.execution.maximum_spreads_bps, "maximum_spreads_bps")
    _increasing(config.execution.minimum_displayed_liquidity, "minimum_displayed_liquidity")
    _increasing(config.execution.fee_sensitivity_per_side_bps, "fee sensitivity")
    if (
        not math.isfinite(config.execution.primary_fee_per_side_bps)
        or config.execution.primary_fee_per_side_bps < 0
        or config.execution.primary_fee_per_side_bps
        not in config.execution.fee_sensitivity_per_side_bps
    ):
        raise EquityConfigurationError("primary fee must be registered in the sensitivity grid")
    if config.execution.decision_interval_seconds != data.target_horizon_seconds:
        raise EquityConfigurationError(
            "the registered decision interval is exactly the non-overlapping 60-second horizon"
        )
    if config.execution.bootstrap_repetitions < 100:
        raise EquityConfigurationError("bootstrap repetitions must be at least 100")
