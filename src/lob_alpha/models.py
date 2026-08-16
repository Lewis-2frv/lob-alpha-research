"""Transparent predictive baselines for chronological research."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class RegressionMetrics:
    rows: int
    mae_ticks: float
    rmse_ticks: float
    correlation: float
    directional_accuracy_nonflat: float


def fit_ridge(
    training: pd.DataFrame,
    *,
    feature_columns: list[str],
    target_column: str,
    alpha: float,
) -> Pipeline:
    if alpha < 0:
        raise ValueError("alpha must be nonnegative")
    clean = training.dropna(subset=[*feature_columns, target_column])
    if clean.empty:
        raise ValueError("no finite training rows")
    model = Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=alpha))])
    model.fit(clean[feature_columns], clean[target_column])
    return model


def score_regression(
    model: Pipeline,
    evaluation: pd.DataFrame,
    *,
    feature_columns: list[str],
    target_column: str,
) -> tuple[np.ndarray, RegressionMetrics]:
    clean = evaluation.dropna(subset=[*feature_columns, target_column])
    if clean.empty:
        raise ValueError("no finite evaluation rows")
    actual = clean[target_column].to_numpy(dtype=float)
    predicted = model.predict(clean[feature_columns])
    return predicted, regression_metrics(actual, predicted)


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> RegressionMetrics:
    """Score aligned arrays, including honest handling of all-flat targets."""

    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if actual.shape != predicted.shape or actual.ndim != 1 or not len(actual):
        raise ValueError("actual and predicted must be nonempty aligned one-dimensional arrays")
    nonflat = actual != 0
    directional = (
        float(np.mean(np.sign(predicted[nonflat]) == np.sign(actual[nonflat])))
        if nonflat.any()
        else float("nan")
    )
    correlation = (
        float(np.corrcoef(predicted, actual)[0, 1])
        if np.std(predicted) and np.std(actual)
        else float("nan")
    )
    return RegressionMetrics(
        rows=len(actual),
        mae_ticks=float(mean_absolute_error(actual, predicted)),
        rmse_ticks=float(np.sqrt(mean_squared_error(actual, predicted))),
        correlation=correlation,
        directional_accuracy_nonflat=directional,
    )
