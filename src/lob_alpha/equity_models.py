"""Transparent baselines and bounded chronological models for the equity study."""

from __future__ import annotations

from collections.abc import Iterable
from itertools import product
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from .equity_config import EquityResearchConfig
from .equity_features import equity_model_feature_columns


def labeled_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Return rows with a finite supplied target, preserving deterministic order."""

    target = pd.to_numeric(frame["target"], errors="coerce")
    return frame.loc[target.notna() & np.isfinite(target)].copy()


def expanding_date_folds(
    date_ids: Iterable[int], *, minimum_train_dates: int, folds: int
) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    """Construct disjoint, strictly chronological expanding-window folds."""

    dates = tuple(sorted(set(int(value) for value in date_ids)))
    if len(dates) <= minimum_train_dates:
        raise ValueError("not enough dates for registered expanding-window CV")
    blocks = np.array_split(np.asarray(dates[minimum_train_dates:], dtype=int), folds)
    result: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    for block in blocks:
        validation = tuple(int(value) for value in block)
        if not validation:
            continue
        train = tuple(value for value in dates if value < validation[0])
        if not train or max(train) >= min(validation) or set(train) & set(validation):
            raise AssertionError("chronological fold isolation failed")
        result.append((train, validation))
    if len(result) < 2:
        raise ValueError("chronological CV needs at least two nonempty validation blocks")
    return result


def deterministic_sample_by_date(frame: pd.DataFrame, maximum_rows: int) -> pd.DataFrame:
    """Bound tuning memory without using target values to choose rows."""

    sampled = []
    for _, group in frame.groupby("date_id", sort=True, observed=True):
        if len(group) <= maximum_rows:
            sampled.append(group)
            continue
        locations = np.linspace(0, len(group) - 1, maximum_rows, dtype=int)
        sampled.append(group.iloc[np.unique(locations)])
    if not sampled:
        raise ValueError("cannot sample an empty model frame")
    return pd.concat(sampled, ignore_index=True)


def candidate_model_specs(config: EquityResearchConfig) -> list[dict[str, Any]]:
    """Return the small, registered grid; unavailable LightGBM is omitted explicitly."""

    specs: list[dict[str, Any]] = []
    if "ridge" in config.model.candidate_models:
        specs.extend({"model": "ridge", "alpha": value} for value in config.model.ridge_alphas)
    nonlinear = product(
        config.model.nonlinear_learning_rates,
        config.model.nonlinear_leaf_counts,
    )
    nonlinear_pairs = list(nonlinear)
    if "hist_gradient_boosting" in config.model.candidate_models:
        specs.extend(
            {
                "model": "hist_gradient_boosting",
                "learning_rate": rate,
                "max_leaf_nodes": leaves,
                "max_iter": config.model.nonlinear_iterations,
            }
            for rate, leaves in nonlinear_pairs
        )
    if "lightgbm" in config.model.candidate_models:
        try:
            import lightgbm  # noqa: F401
        except ImportError:
            pass
        else:
            specs.extend(
                {
                    "model": "lightgbm",
                    "learning_rate": rate,
                    "num_leaves": leaves,
                    "n_estimators": config.model.nonlinear_iterations,
                }
                for rate, leaves in nonlinear_pairs
            )
    return specs


def _preprocessor(
    config: EquityResearchConfig,
    *,
    scale: bool,
    ordinal_stock: bool,
    numeric_features: list[str] | None = None,
) -> ColumnTransformer:
    numeric_steps: list[tuple[str, Any]] = [
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True))
    ]
    if scale:
        numeric_steps.append(("scaler", StandardScaler()))
    stock_encoder: Any
    if ordinal_stock:
        stock_encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=np.nan,
            encoded_missing_value=np.nan,
            dtype=np.float64,
        )
    else:
        stock_encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=True,
            dtype=np.float32,
        )
    return ColumnTransformer(
        (
            (
                "numeric",
                Pipeline(numeric_steps),
                numeric_features or equity_model_feature_columns(config),
            ),
            (
                "stock_fixed_effect",
                stock_encoder,
                ["stock_id"],
            ),
        ),
        remainder="drop",
        sparse_threshold=1.0,
        verbose_feature_names_out=False,
    )


def build_model_pipeline(
    config: EquityResearchConfig,
    specification: dict[str, Any],
    *,
    numeric_features: list[str] | None = None,
) -> Pipeline:
    """Build a model where stock_id enters only as a categorical fixed effect."""

    name = specification["model"]
    features = numeric_features or equity_model_feature_columns(config)
    if name == "ridge":
        estimator: Any = Ridge(alpha=float(specification["alpha"]))
        scale = True
        ordinal_stock = False
    elif name == "hist_gradient_boosting":
        estimator = HistGradientBoostingRegressor(
            learning_rate=float(specification["learning_rate"]),
            max_leaf_nodes=int(specification["max_leaf_nodes"]),
            max_iter=int(specification["max_iter"]),
            l2_regularization=1.0,
            categorical_features=[len(features)],
            random_state=config.seed,
        )
        scale = False
        ordinal_stock = True
    elif name == "lightgbm":
        try:
            from lightgbm import LGBMRegressor
        except ImportError as exc:
            raise RuntimeError("install the equity dependency group to use LightGBM") from exc
        estimator = LGBMRegressor(
            learning_rate=float(specification["learning_rate"]),
            num_leaves=int(specification["num_leaves"]),
            n_estimators=int(specification["n_estimators"]),
            objective="regression_l1",
            random_state=config.seed,
            deterministic=True,
            force_col_wise=True,
            verbosity=-1,
        )
        scale = False
        ordinal_stock = False
    else:
        raise ValueError(f"unknown model specification: {name}")
    return Pipeline(
        (
            (
                "preprocess",
                _preprocessor(
                    config,
                    scale=scale,
                    ordinal_stock=ordinal_stock,
                    numeric_features=features,
                ),
            ),
            ("model", estimator),
        )
    )


def fit_model(
    config: EquityResearchConfig,
    specification: dict[str, Any],
    frame: pd.DataFrame,
    *,
    numeric_features: list[str] | None = None,
) -> Pipeline:
    labeled = labeled_rows(frame)
    if labeled.empty:
        raise ValueError("model fitting requires at least one finite supplied target")
    pipeline = build_model_pipeline(config, specification, numeric_features=numeric_features)
    pipeline.fit(labeled, labeled["target"].to_numpy(dtype=float))
    return pipeline


def fit_model_with_train_preprocessing(
    config: EquityResearchConfig,
    specification: dict[str, Any],
    *,
    preprocessing_frame: pd.DataFrame,
    estimator_frame: pd.DataFrame,
) -> Pipeline:
    """Freeze preprocessing on train, then refit only the estimator on development."""

    pipeline = build_model_pipeline(config, specification)
    preprocessor = pipeline.named_steps["preprocess"]
    estimator = pipeline.named_steps["model"]
    preprocessor.fit(preprocessing_frame)
    labeled_estimator = labeled_rows(estimator_frame)
    if labeled_estimator.empty:
        raise ValueError("estimator refit requires at least one finite supplied target")
    transformed = preprocessor.transform(labeled_estimator)
    estimator.fit(transformed, labeled_estimator["target"].to_numpy(dtype=float))
    return pipeline


def fit_signed_imbalance_baseline(frame: pd.DataFrame) -> float:
    """Fit one transparent train-only slope with no intercept."""

    labeled = labeled_rows(frame)
    if labeled.empty:
        raise ValueError("baseline fitting requires at least one finite supplied target")
    x = labeled["auction_imbalance_ratio"].fillna(0.0).to_numpy(dtype=float)
    y = labeled["target"].to_numpy(dtype=float)
    denominator = float(np.dot(x, x))
    return 0.0 if denominator == 0 else float(np.dot(x, y) / denominator)


def baseline_predictions(frame: pd.DataFrame, *, imbalance_slope: float) -> dict[str, np.ndarray]:
    return {
        "zero": np.zeros(len(frame), dtype=float),
        "signed_imbalance": (
            imbalance_slope * frame["auction_imbalance_ratio"].fillna(0.0).to_numpy(dtype=float)
        ),
    }


def spearman_ic(actual: pd.Series | np.ndarray, predicted: pd.Series | np.ndarray) -> float:
    table = pd.DataFrame({"actual": actual, "predicted": predicted}).dropna()
    if len(table) < 2 or table["actual"].nunique() < 2 or table["predicted"].nunique() < 2:
        return float("nan")
    return float(table["actual"].rank().corr(table["predicted"].rank()))


def predictive_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float | int]:
    if len(actual) != len(predicted) or not len(actual):
        raise ValueError("actual and predicted arrays must be nonempty and aligned")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("predictive metrics require finite actuals and predictions")
    residual = actual - predicted
    nonzero = actual != 0
    directional = (
        float(np.mean(np.sign(actual[nonzero]) == np.sign(predicted[nonzero])))
        if nonzero.any()
        else float("nan")
    )
    return {
        "rows": len(actual),
        "mae_bps": float(np.mean(np.abs(residual))),
        "spearman_ic": spearman_ic(actual, predicted),
        "directional_accuracy": directional,
    }


def daily_predictive_metrics(
    frame: pd.DataFrame, predicted: np.ndarray, *, model_name: str
) -> pd.DataFrame:
    if len(frame) != len(predicted):
        raise ValueError("daily predictions are not aligned with evaluation rows")
    scored = frame.loc[:, ["date_id", "target"]].copy()
    scored["prediction"] = predicted
    scored = scored.loc[
        pd.to_numeric(scored["target"], errors="coerce").notna()
        & np.isfinite(pd.to_numeric(scored["target"], errors="coerce"))
        & np.isfinite(scored["prediction"])
    ]
    if scored.empty:
        raise ValueError("daily predictive metrics require finite supplied targets")
    rows = []
    for date_id, group in scored.groupby("date_id", sort=True, observed=True):
        rows.append(
            {
                "date_id": int(date_id),
                "model": model_name,
                **predictive_metrics(
                    group["target"].to_numpy(dtype=float),
                    group["prediction"].to_numpy(dtype=float),
                ),
            }
        )
    return pd.DataFrame(rows)


def preprocessing_parameters(pipeline: Pipeline) -> dict[str, Any]:
    """Expose fitted imputation/scaling/categories for a hashable freeze artifact."""

    transformer: ColumnTransformer = pipeline.named_steps["preprocess"]
    numeric: Pipeline = transformer.named_transformers_["numeric"]
    imputer: SimpleImputer = numeric.named_steps["imputer"]
    payload: dict[str, Any] = {
        "numeric_columns": list(transformer.transformers_[0][2]),
        "imputer_strategy": imputer.strategy,
        "imputer_keep_empty_features": imputer.keep_empty_features,
        "imputer_statistics": [float(value) for value in imputer.statistics_],
        "categorical_columns": ["stock_id"],
        "stock_categories": [
            int(value)
            for value in transformer.named_transformers_["stock_fixed_effect"].categories_[0]
        ],
        "stock_encoding": (
            "ordinal_native_categorical"
            if isinstance(transformer.named_transformers_["stock_fixed_effect"], OrdinalEncoder)
            else "sparse_one_hot"
        ),
    }
    if "scaler" in numeric.named_steps:
        scaler: StandardScaler = numeric.named_steps["scaler"]
        payload["scaler_mean"] = [float(value) for value in scaler.mean_]
        payload["scaler_scale"] = [float(value) for value in scaler.scale_]
    else:
        payload["scaler_mean"] = None
        payload["scaler_scale"] = None
    return payload
