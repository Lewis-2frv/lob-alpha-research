"""Strict registration for the public FI-2010 anchored-fold study."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class FI2010ConfigurationError(ValueError):
    """Raised when the registered study contract has been weakened or changed."""


@dataclass(frozen=True)
class FI2010Source:
    outer_archive_size: int
    outer_archive_sha256: str
    inner_member: str
    dataset_id: str
    pid: str
    title: str
    licence: str
    paper_doi: str


@dataclass(frozen=True)
class FI2010Data:
    prepared_dir: str
    representation: str
    feature_rows: tuple[int, int]
    label_rows: tuple[int, int]
    class_mapping: dict[int, str]
    horizon_sampled_steps: tuple[int, ...]
    horizon_underlying_events: tuple[int, ...]
    primary_label_row: int
    development_folds: tuple[int, ...]
    final_fold: int


@dataclass(frozen=True)
class FI2010Models:
    manual_pressure_temperature: float
    manual_pressure_stationary_logit: float
    diagonal_lda_shrinkage: float
    numpy_ridge_alphas: tuple[float, ...]
    numpy_softmax_learning_rate: float
    numpy_softmax_l2: float
    numpy_softmax_epochs: int
    numpy_softmax_batch_size: int
    sgd_alphas: tuple[float, ...]
    sgd_max_iter: int
    lightgbm_learning_rates: tuple[float, ...]
    lightgbm_num_leaves: tuple[int, ...]
    lightgbm_estimators: int
    fallback_learning_rate: float
    fallback_max_leaf_nodes: int
    fallback_max_iter: int


@dataclass(frozen=True)
class FI2010Selection:
    primary_metric: str
    first_tie_breaker: str
    second_tie_breaker: str
    confidence_thresholds: tuple[float, ...]
    minimum_directional_coverage: float


@dataclass(frozen=True)
class FI2010Config:
    path: Path
    study_id: str
    seed: int
    synthetic: bool
    source: FI2010Source
    data: FI2010Data
    models: FI2010Models
    selection: FI2010Selection


def _tuple(raw: dict[str, Any], key: str, converter: type) -> tuple[Any, ...]:
    return tuple(converter(value) for value in raw[key])


def load_fi2010_config(path: str | Path = "configs/fi2010.yaml") -> FI2010Config:
    """Load and validate the immutable public-study declaration."""

    config_path = Path(path).resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    source_raw = raw["source"]
    data_raw = raw["data"]
    models_raw = raw["models"]
    selection_raw = raw["selection"]
    config = FI2010Config(
        path=config_path,
        study_id=str(raw["study_id"]),
        seed=int(raw["seed"]),
        synthetic=bool(raw.get("synthetic", False)),
        source=FI2010Source(
            outer_archive_size=int(source_raw["outer_archive_size"]),
            outer_archive_sha256=str(source_raw["outer_archive_sha256"]).lower(),
            inner_member=str(source_raw["inner_member"]),
            dataset_id=str(source_raw["dataset_id"]),
            pid=str(source_raw["pid"]),
            title=str(source_raw["title"]),
            licence=str(source_raw["licence"]),
            paper_doi=str(source_raw["paper_doi"]),
        ),
        data=FI2010Data(
            prepared_dir=str(data_raw["prepared_dir"]),
            representation=str(data_raw["representation"]),
            feature_rows=tuple(int(value) for value in data_raw["feature_rows"]),
            label_rows=tuple(int(value) for value in data_raw["label_rows"]),
            class_mapping={
                int(key): str(value) for key, value in data_raw["class_mapping"].items()
            },
            horizon_sampled_steps=_tuple(data_raw, "horizon_sampled_steps", int),
            horizon_underlying_events=_tuple(data_raw, "horizon_underlying_events", int),
            primary_label_row=int(data_raw["primary_label_row"]),
            development_folds=_tuple(data_raw, "development_folds", int),
            final_fold=int(data_raw["final_fold"]),
        ),
        models=FI2010Models(
            manual_pressure_temperature=float(models_raw["manual_pressure_temperature"]),
            manual_pressure_stationary_logit=float(models_raw["manual_pressure_stationary_logit"]),
            diagonal_lda_shrinkage=float(models_raw["diagonal_lda_shrinkage"]),
            numpy_ridge_alphas=_tuple(models_raw, "numpy_ridge_alphas", float),
            numpy_softmax_learning_rate=float(models_raw["numpy_softmax_learning_rate"]),
            numpy_softmax_l2=float(models_raw["numpy_softmax_l2"]),
            numpy_softmax_epochs=int(models_raw["numpy_softmax_epochs"]),
            numpy_softmax_batch_size=int(models_raw["numpy_softmax_batch_size"]),
            sgd_alphas=_tuple(models_raw, "sgd_alphas", float),
            sgd_max_iter=int(models_raw["sgd_max_iter"]),
            lightgbm_learning_rates=_tuple(models_raw, "lightgbm_learning_rates", float),
            lightgbm_num_leaves=_tuple(models_raw, "lightgbm_num_leaves", int),
            lightgbm_estimators=int(models_raw["lightgbm_estimators"]),
            fallback_learning_rate=float(models_raw["fallback_learning_rate"]),
            fallback_max_leaf_nodes=int(models_raw["fallback_max_leaf_nodes"]),
            fallback_max_iter=int(models_raw["fallback_max_iter"]),
        ),
        selection=FI2010Selection(
            primary_metric=str(selection_raw["primary_metric"]),
            first_tie_breaker=str(selection_raw["first_tie_breaker"]),
            second_tie_breaker=str(selection_raw["second_tie_breaker"]),
            confidence_thresholds=_tuple(selection_raw, "confidence_thresholds", float),
            minimum_directional_coverage=float(selection_raw["minimum_directional_coverage"]),
        ),
    )
    validate_fi2010_config(config)
    return config


def validate_fi2010_config(config: FI2010Config) -> None:
    """Reject deviations from the registered representation and chronology."""

    expected = {
        "representation": "NoAuction/1.NoAuction_Zscore",
        "feature_rows": (1, 144),
        "label_rows": (145, 149),
        "class_mapping": {1: "up", 2: "stationary", 3: "down"},
        "steps": (1, 2, 3, 5, 10),
        "events": (10, 20, 30, 50, 100),
        "primary_label_row": 4,
        "development_folds": tuple(range(1, 9)),
        "final_fold": 9,
    }
    actual = {
        "representation": config.data.representation,
        "feature_rows": config.data.feature_rows,
        "label_rows": config.data.label_rows,
        "class_mapping": config.data.class_mapping,
        "steps": config.data.horizon_sampled_steps,
        "events": config.data.horizon_underlying_events,
        "primary_label_row": config.data.primary_label_row,
        "development_folds": config.data.development_folds,
        "final_fold": config.data.final_fold,
    }
    if actual != expected:
        raise FI2010ConfigurationError(f"FI-2010 registration mismatch: {actual!r}")
    if config.selection.primary_metric != "macro_f1":
        raise FI2010ConfigurationError("primary selection metric must remain macro_f1")
    if config.selection.first_tie_breaker != "worst_fold_macro_f1":
        raise FI2010ConfigurationError("first tie-breaker must remain worst-fold macro-F1")
    if config.selection.second_tie_breaker != "lower_model_complexity":
        raise FI2010ConfigurationError("second tie-breaker must remain lower complexity")
    thresholds = config.selection.confidence_thresholds
    if not thresholds or tuple(sorted(set(thresholds))) != thresholds:
        raise FI2010ConfigurationError("confidence thresholds must be unique and increasing")
    if any(not 1 / 3 < value < 1 for value in thresholds):
        raise FI2010ConfigurationError("confidence thresholds must be between 1/3 and 1")
    if not 0 <= config.selection.minimum_directional_coverage <= 1:
        raise FI2010ConfigurationError("minimum directional coverage must be in [0, 1]")
    if (
        not math.isfinite(config.models.manual_pressure_temperature)
        or config.models.manual_pressure_temperature <= 0
    ):
        raise FI2010ConfigurationError("manual pressure temperature must be finite and positive")
    if not math.isfinite(config.models.manual_pressure_stationary_logit):
        raise FI2010ConfigurationError("manual pressure stationary logit must be finite")
    if not 0 <= config.models.diagonal_lda_shrinkage <= 1:
        raise FI2010ConfigurationError("diagonal LDA shrinkage must be in [0, 1]")
    if not config.models.numpy_ridge_alphas or any(
        not math.isfinite(value) or value <= 0 for value in config.models.numpy_ridge_alphas
    ):
        raise FI2010ConfigurationError("NumPy ridge alphas must be finite and positive")
    if (
        not math.isfinite(config.models.numpy_softmax_learning_rate)
        or config.models.numpy_softmax_learning_rate <= 0
    ):
        raise FI2010ConfigurationError("NumPy softmax learning rate must be finite and positive")
    if not math.isfinite(config.models.numpy_softmax_l2) or config.models.numpy_softmax_l2 < 0:
        raise FI2010ConfigurationError("NumPy softmax L2 must be finite and nonnegative")
    if config.models.numpy_softmax_epochs <= 0 or config.models.numpy_softmax_batch_size <= 0:
        raise FI2010ConfigurationError("NumPy softmax epochs and batch size must be positive")
    if config.models.sgd_max_iter <= 0:
        raise FI2010ConfigurationError("SGD max_iter must be positive")
    if not config.models.sgd_alphas or any(
        not math.isfinite(value) or value <= 0 for value in config.models.sgd_alphas
    ):
        raise FI2010ConfigurationError("SGD alphas must be finite and positive")
    if not config.models.lightgbm_learning_rates or any(
        not math.isfinite(value) or value <= 0 for value in config.models.lightgbm_learning_rates
    ):
        raise FI2010ConfigurationError("LightGBM learning rates must be finite and positive")
    if not config.models.lightgbm_num_leaves or any(
        value < 2 for value in config.models.lightgbm_num_leaves
    ):
        raise FI2010ConfigurationError("LightGBM num_leaves values must be at least 2")
    if config.models.lightgbm_estimators <= 0:
        raise FI2010ConfigurationError("LightGBM n_estimators must be positive")
    if (
        not math.isfinite(config.models.fallback_learning_rate)
        or config.models.fallback_learning_rate <= 0
    ):
        raise FI2010ConfigurationError("fallback learning rate must be finite and positive")
    if config.models.fallback_max_leaf_nodes < 2 or config.models.fallback_max_iter <= 0:
        raise FI2010ConfigurationError("fallback tree bounds must be positive and nontrivial")
    if config.synthetic:
        return
    source = config.source
    if source.outer_archive_size != 1_830_875_986:
        raise FI2010ConfigurationError("registered outer archive size changed")
    if source.outer_archive_sha256 != (
        "bcc89a5aa7d8067dda98374393444eb885a4283a41fd33e323496380e057e1a6"
    ):
        raise FI2010ConfigurationError("registered outer archive SHA-256 changed")
    if source.inner_member != "published/BenchmarkDatasets/BenchmarkDatasets.zip":
        raise FI2010ConfigurationError("required inner archive member changed")
    if source.dataset_id != "73eb48d7-4dbc-4a10-a52a-da745b47a649":
        raise FI2010ConfigurationError("registered dataset ID changed")
    if source.pid != "urn:nbn:fi:csc-kata20170601153214969115":
        raise FI2010ConfigurationError("registered persistent identifier changed")
    if source.title != (
        "Benchmark Dataset for Mid-Price Forecasting of Limit Order Book Data with Machine "
        "Learning Methods"
    ):
        raise FI2010ConfigurationError("registered dataset title changed")
    if source.licence != "CC BY 4.0" or source.paper_doi != "10.1002/for.2543":
        raise FI2010ConfigurationError("source attribution changed")
