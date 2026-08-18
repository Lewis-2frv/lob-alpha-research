"""Bounded snapshot classifiers and registered metrics for FI-2010."""

from __future__ import annotations

import json
import pickle
import warnings
from dataclasses import dataclass
from itertools import product
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_recall_fscore_support,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .fi2010_config import FI2010Config

CLASSES = np.asarray([1, 2, 3], dtype=np.int8)


def _softmax(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    shifted = values - np.max(values, axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    totals = probabilities.sum(axis=1, keepdims=True)
    if not np.isfinite(probabilities).all() or np.any(totals <= 0):
        raise ValueError("invalid logits produced non-finite probabilities")
    return probabilities / totals


class AlwaysStationaryClassifier(ClassifierMixin, BaseEstimator):
    """Transparent publisher-class-2 baseline with a sklearn-compatible interface."""

    def fit(self, features: np.ndarray, target: np.ndarray) -> AlwaysStationaryClassifier:
        del features, target
        self.classes_ = CLASSES.copy()
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        return np.full(len(features), 2, dtype=np.int8)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        probabilities = np.zeros((len(features), 3), dtype=np.float64)
        probabilities[:, 1] = 1.0
        return probabilities


class ManualLiquidityPressureClassifier(ClassifierMixin, BaseEstimator):
    """Fixed, interpretable LOB depth-pressure rule using the publisher feature layout.

    FI-2010's first 40 features are the 10-level basic book representation in repeated
    ``ask price, ask volume, bid price, bid volume`` order. Because the public study uses
    publisher Z-score features, this is a *standardised depth-pressure proxy*, not a raw
    queue-imbalance ratio. No label is used in fitting this rule.
    """

    def __init__(self, temperature: float = 1.0, stationary_logit: float = 0.45):
        self.temperature = temperature
        self.stationary_logit = stationary_logit

    def fit(self, features: np.ndarray, target: np.ndarray) -> ManualLiquidityPressureClassifier:
        values = np.asarray(features)
        labels = np.asarray(target)
        if values.ndim != 2 or values.shape[1] < 86:
            raise ValueError(
                "manual liquidity-pressure rule requires the 144-feature FI-2010 layout"
            )
        if labels.ndim != 1 or len(values) != len(labels):
            raise ValueError("features/target shape mismatch")
        if not np.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        if not np.isfinite(self.stationary_logit):
            raise ValueError("stationary_logit must be finite")
        self.classes_ = CLASSES.copy()
        return self

    @staticmethod
    def _pressure(features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=np.float64)
        ask_volume = values[:, np.arange(1, 40, 4)]
        bid_volume = values[:, np.arange(3, 40, 4)]
        level_weights = 1.0 / np.sqrt(np.arange(1, 11, dtype=np.float64))
        level_weights /= level_weights.sum()
        depth_proxy = (bid_volume - ask_volume) @ level_weights
        # Publisher feature u5[1] is accumulated ask-volume minus bid-volume.
        # Negating it aligns positive pressure with the up class.
        accumulated_volume_proxy = -values[:, 85]
        return 0.75 * depth_proxy + 0.25 * accumulated_volume_proxy

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        pressure = np.clip(self._pressure(features) / float(self.temperature), -12.0, 12.0)
        logits = np.column_stack(
            (
                pressure,
                np.full(len(pressure), float(self.stationary_logit)),
                -pressure,
            )
        )
        return _softmax(logits)

    def predict(self, features: np.ndarray) -> np.ndarray:
        return CLASSES[np.argmax(self.predict_proba(features), axis=1)]


class NumpyDiagonalLDAClassifier(ClassifierMixin, BaseEstimator):
    """From-scratch shared-diagonal Gaussian discriminant classifier.

    The model estimates class means and a pooled diagonal covariance from the fold's
    training member only. Equal class priors make the comparison appropriate for the
    registered macro-F1 objective while shrinkage stabilises low-variance dimensions.
    """

    def __init__(self, shrinkage: float = 0.10, variance_floor: float = 1e-6):
        self.shrinkage = shrinkage
        self.variance_floor = variance_floor

    def fit(self, features: np.ndarray, target: np.ndarray) -> NumpyDiagonalLDAClassifier:
        x = np.asarray(features, dtype=np.float32)
        y = np.asarray(target, dtype=np.int8)
        if x.ndim != 2 or y.ndim != 1 or len(x) != len(y):
            raise ValueError("features/target shape mismatch")
        if not 0 <= float(self.shrinkage) <= 1:
            raise ValueError("shrinkage must be in [0, 1]")
        if not np.isfinite(self.variance_floor) or self.variance_floor <= 0:
            raise ValueError("variance_floor must be finite and positive")
        if not np.isin(y, CLASSES).all():
            raise ValueError("labels must be in {1,2,3}")
        means = []
        pooled_ss = np.zeros(x.shape[1], dtype=np.float64)
        degrees = 0
        for label in CLASSES:
            member = x[y == label]
            if len(member) < 2:
                raise ValueError("diagonal LDA requires at least two observations per class")
            mean = member.mean(axis=0, dtype=np.float64)
            variance = member.var(axis=0, dtype=np.float64, ddof=1)
            means.append(mean)
            pooled_ss += variance * (len(member) - 1)
            degrees += len(member) - 1
        variance = pooled_ss / max(degrees, 1)
        target_variance = float(np.mean(variance))
        variance = (
            (1.0 - float(self.shrinkage)) * variance
            + float(self.shrinkage) * target_variance
        )
        variance = np.maximum(variance, float(self.variance_floor))
        self.classes_ = CLASSES.copy()
        self.means_ = np.vstack(means)
        self.variance_ = variance
        self.linear_weights_ = self.means_ / self.variance_[None, :]
        self.intercepts_ = -0.5 * np.sum(self.means_ * self.linear_weights_, axis=1)
        return self

    def decision_function(self, features: np.ndarray) -> np.ndarray:
        x = np.asarray(features, dtype=np.float64)
        return x @ self.linear_weights_.T + self.intercepts_[None, :]

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        return _softmax(self.decision_function(features))

    def predict(self, features: np.ndarray) -> np.ndarray:
        return CLASSES[np.argmax(self.decision_function(features), axis=1)]


class NumpyRidgeMulticlassClassifier(ClassifierMixin, BaseEstimator):
    """From-scratch class-balanced multiclass ridge regression in closed form."""

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def fit(self, features: np.ndarray, target: np.ndarray) -> NumpyRidgeMulticlassClassifier:
        x = np.asarray(features, dtype=np.float32)
        y = np.asarray(target, dtype=np.int8)
        if x.ndim != 2 or y.ndim != 1 or len(x) != len(y):
            raise ValueError("features/target shape mismatch")
        if not np.isfinite(self.alpha) or self.alpha <= 0:
            raise ValueError("alpha must be finite and positive")
        weights = training_class_weights(y)
        self.mean_ = x.mean(axis=0, dtype=np.float64)
        self.scale_ = x.std(axis=0, dtype=np.float64)
        self.scale_[self.scale_ < 1e-8] = 1.0
        dimension = x.shape[1] + 1
        normal = np.zeros((dimension, dimension), dtype=np.float64)
        rhs = np.zeros((dimension, 3), dtype=np.float64)
        class_eye = np.eye(3, dtype=np.float64)
        chunk_size = 32768
        for start in range(0, len(x), chunk_size):
            stop = min(start + chunk_size, len(x))
            z = (x[start:stop].astype(np.float64) - self.mean_) / self.scale_
            design = np.column_stack((z, np.ones(len(z), dtype=np.float64)))
            labels = y[start:stop].astype(int) - 1
            response = class_eye[labels]
            sample_weight = np.asarray(
                [weights[int(label)] for label in y[start:stop]], dtype=np.float64
            )
            root_weight = np.sqrt(sample_weight)[:, None]
            weighted_design = design * root_weight
            normal += weighted_design.T @ weighted_design
            rhs += weighted_design.T @ (response * root_weight)
        penalty = np.eye(normal.shape[0], dtype=np.float64) * float(self.alpha)
        penalty[-1, -1] = 0.0
        self.coefficients_ = np.linalg.solve(normal + penalty, rhs)
        self.classes_ = CLASSES.copy()
        return self

    def decision_function(self, features: np.ndarray) -> np.ndarray:
        x = np.asarray(features, dtype=np.float64)
        z = (x - self.mean_) / self.scale_
        design = np.column_stack((z, np.ones(len(z), dtype=np.float64)))
        return design @ self.coefficients_

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        return _softmax(self.decision_function(features))

    def predict(self, features: np.ndarray) -> np.ndarray:
        return CLASSES[np.argmax(self.decision_function(features), axis=1)]


class NumpySoftmaxClassifier(ClassifierMixin, BaseEstimator):
    """From-scratch class-balanced multinomial logistic regression trained with Adam."""

    def __init__(
        self,
        learning_rate: float = 0.02,
        l2: float = 1e-3,
        epochs: int = 8,
        batch_size: int = 16384,
        random_state: int = 0,
    ):
        self.learning_rate = learning_rate
        self.l2 = l2
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state

    def fit(self, features: np.ndarray, target: np.ndarray) -> NumpySoftmaxClassifier:
        x = np.asarray(features, dtype=np.float32)
        y = np.asarray(target, dtype=np.int8)
        if x.ndim != 2 or y.ndim != 1 or len(x) != len(y):
            raise ValueError("features/target shape mismatch")
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if not np.isfinite(self.l2) or self.l2 < 0:
            raise ValueError("l2 must be finite and nonnegative")
        if int(self.epochs) <= 0 or int(self.batch_size) <= 0:
            raise ValueError("epochs and batch_size must be positive")
        if not np.isin(y, CLASSES).all():
            raise ValueError("labels must be in {1,2,3}")

        class_weights = training_class_weights(y)
        self.mean_ = x.mean(axis=0, dtype=np.float64)
        self.scale_ = x.std(axis=0, dtype=np.float64)
        self.scale_[self.scale_ < 1e-8] = 1.0
        parameters = np.zeros((x.shape[1] + 1, 3), dtype=np.float64)
        first_moment = np.zeros_like(parameters)
        second_moment = np.zeros_like(parameters)
        rng = np.random.default_rng(int(self.random_state))
        step = 0
        beta1, beta2 = 0.9, 0.999
        epsilon = 1e-8

        for _epoch in range(int(self.epochs)):
            order = rng.permutation(len(x))
            for start in range(0, len(order), int(self.batch_size)):
                indices = order[start : start + int(self.batch_size)]
                z = (x[indices].astype(np.float64) - self.mean_) / self.scale_
                batch_x = np.column_stack((z, np.ones(len(z), dtype=np.float64)))
                batch_y = y[indices].astype(int) - 1
                batch_weight = np.asarray(
                    [class_weights[int(label)] for label in y[indices]], dtype=np.float64
                )
                batch_weight = batch_weight / max(float(batch_weight.mean()), 1e-12)
                probabilities = _softmax(batch_x @ parameters)
                probabilities[np.arange(len(indices)), batch_y] -= 1.0
                probabilities *= batch_weight[:, None]
                gradient = batch_x.T @ probabilities / len(indices)
                gradient[:-1] += float(self.l2) * parameters[:-1]
                step += 1
                first_moment = beta1 * first_moment + (1.0 - beta1) * gradient
                second_moment = beta2 * second_moment + (1.0 - beta2) * (gradient * gradient)
                corrected_first = first_moment / (1.0 - beta1**step)
                corrected_second = second_moment / (1.0 - beta2**step)
                parameters -= float(self.learning_rate) * corrected_first / (
                    np.sqrt(corrected_second) + epsilon
                )
        self.coefficients_ = parameters
        self.classes_ = CLASSES.copy()
        return self

    def decision_function(self, features: np.ndarray) -> np.ndarray:
        x = np.asarray(features, dtype=np.float64)
        z = (x - self.mean_) / self.scale_
        design = np.column_stack((z, np.ones(len(z), dtype=np.float64)))
        return design @ self.coefficients_

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        return _softmax(self.decision_function(features))

    def predict(self, features: np.ndarray) -> np.ndarray:
        return CLASSES[np.argmax(self.decision_function(features), axis=1)]


@dataclass(frozen=True)
class FittedModel:
    estimator: Any
    specification: dict[str, Any]
    class_weights: dict[int, float]


def specification_id(specification: dict[str, Any]) -> str:
    return json.dumps(specification, sort_keys=True, separators=(",", ":"))


def candidate_specifications(config: FI2010Config) -> tuple[list[dict[str, Any]], bool]:
    """Return the registered interpretable-to-nonlinear grid and LightGBM availability."""

    specifications: list[dict[str, Any]] = [
        {"model": "always_stationary", "complexity": 0},
        {"model": "dummy_prior", "complexity": 1},
        {
            "model": "manual_liquidity_pressure",
            "temperature": config.models.manual_pressure_temperature,
            "stationary_logit": config.models.manual_pressure_stationary_logit,
            "complexity": 2,
        },
        {
            "model": "numpy_diagonal_lda",
            "shrinkage": config.models.diagonal_lda_shrinkage,
            "complexity": 3,
        },
    ]
    specifications.extend(
        {
            "model": "numpy_ridge_multiclass",
            "alpha": alpha,
            "complexity": 4,
        }
        for alpha in config.models.numpy_ridge_alphas
    )
    specifications.append(
        {
            "model": "numpy_softmax",
            "learning_rate": config.models.numpy_softmax_learning_rate,
            "l2": config.models.numpy_softmax_l2,
            "epochs": config.models.numpy_softmax_epochs,
            "batch_size": config.models.numpy_softmax_batch_size,
            "complexity": 5,
        }
    )
    specifications.extend(
        {
            "model": "sgd_log_loss",
            "alpha": alpha,
            "max_iter": config.models.sgd_max_iter,
            "complexity": 6,
        }
        for alpha in config.models.sgd_alphas
    )
    # Keep the audited sklearn nonlinear candidate in every environment. This
    # preserves a reproducible benchmark floor while allowing LightGBM to win
    # only on the registered CF_1-CF_8 evidence.
    specifications.append(
        {
            "model": "hist_gradient_boosting_fallback",
            "learning_rate": config.models.fallback_learning_rate,
            "max_leaf_nodes": config.models.fallback_max_leaf_nodes,
            "max_iter": config.models.fallback_max_iter,
            "complexity": 7,
        }
    )
    try:
        import lightgbm  # noqa: F401
    except ImportError:
        lightgbm_available = False
    else:
        lightgbm_available = True
        specifications.extend(
            {
                "model": "lightgbm_multiclass",
                "learning_rate": learning_rate,
                "num_leaves": leaves,
                "n_estimators": config.models.lightgbm_estimators,
                "complexity": 8,
            }
            for learning_rate, leaves in product(
                config.models.lightgbm_learning_rates,
                config.models.lightgbm_num_leaves,
            )
        )
    return specifications, lightgbm_available


def training_class_weights(target: np.ndarray) -> dict[int, float]:
    """Compute balanced weights from this training member and nowhere else."""

    values = np.asarray(target, dtype=np.int8)
    if values.ndim != 1 or not len(values) or not np.isin(values, CLASSES).all():
        raise ValueError("training labels must be a nonempty vector in {1,2,3}")
    counts = {int(label): int(np.sum(values == label)) for label in CLASSES}
    present = sum(count > 0 for count in counts.values())
    return {
        label: (len(values) / (present * count) if count else 0.0)
        for label, count in counts.items()
    }


def fit_classifier(
    specification: dict[str, Any],
    features: np.ndarray,
    target: np.ndarray,
    *,
    seed: int,
) -> FittedModel:
    """Fit one fold-local classifier and fold-local class weights."""

    features = np.asarray(features, dtype=np.float32)
    target = np.asarray(target, dtype=np.int8)
    weights = training_class_weights(target)
    fit_sample_weight: np.ndarray | None = None
    model_name = specification["model"]
    if model_name == "always_stationary":
        estimator: Any = AlwaysStationaryClassifier()
    elif model_name == "dummy_prior":
        estimator = DummyClassifier(strategy="prior", random_state=seed)
    elif model_name == "manual_liquidity_pressure":
        estimator = ManualLiquidityPressureClassifier(
            temperature=float(specification["temperature"]),
            stationary_logit=float(specification["stationary_logit"]),
        )
    elif model_name == "numpy_diagonal_lda":
        estimator = NumpyDiagonalLDAClassifier(shrinkage=float(specification["shrinkage"]))
    elif model_name == "numpy_ridge_multiclass":
        estimator = NumpyRidgeMulticlassClassifier(alpha=float(specification["alpha"]))
    elif model_name == "numpy_softmax":
        estimator = NumpySoftmaxClassifier(
            learning_rate=float(specification["learning_rate"]),
            l2=float(specification["l2"]),
            epochs=int(specification["epochs"]),
            batch_size=int(specification["batch_size"]),
            random_state=seed,
        )
    elif model_name == "sgd_log_loss":
        estimator = Pipeline(
            (
                ("scale", StandardScaler()),
                (
                    "classifier",
                    SGDClassifier(
                        loss="log_loss",
                        penalty="l2",
                        alpha=float(specification["alpha"]),
                        max_iter=int(specification["max_iter"]),
                        tol=1e-3,
                        early_stopping=True,
                        validation_fraction=0.1,
                        n_iter_no_change=5,
                        class_weight=weights,
                        random_state=seed,
                    ),
                ),
            )
        )
    elif model_name == "hist_gradient_boosting_fallback":
        estimator = HistGradientBoostingClassifier(
            loss="log_loss",
            learning_rate=float(specification["learning_rate"]),
            max_leaf_nodes=int(specification["max_leaf_nodes"]),
            max_iter=int(specification["max_iter"]),
            l2_regularization=1.0,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=8,
            random_state=seed,
        )
        fit_sample_weight = np.asarray([weights[int(label)] for label in target], dtype=float)
    elif model_name == "lightgbm_multiclass":
        try:
            from lightgbm import LGBMClassifier
        except ImportError as exc:
            raise RuntimeError("LightGBM became unavailable after candidate registration") from exc
        estimator = LGBMClassifier(
            objective="multiclass",
            num_class=3,
            learning_rate=float(specification["learning_rate"]),
            num_leaves=int(specification["num_leaves"]),
            n_estimators=int(specification["n_estimators"]),
            min_child_samples=40,
            reg_lambda=1.0,
            colsample_bytree=0.9,
            subsample=0.9,
            subsample_freq=1,
            class_weight=weights,
            random_state=seed,
            deterministic=True,
            force_col_wise=True,
            verbosity=-1,
            n_jobs=-1,
        )
    else:
        raise ValueError(f"unknown FI-2010 model: {model_name}")
    if fit_sample_weight is None:
        estimator.fit(features, target)
    else:
        estimator.fit(features, target, sample_weight=fit_sample_weight)
    return FittedModel(
        estimator=estimator,
        specification=dict(specification),
        class_weights=weights,
    )


def aligned_probabilities(fitted: FittedModel, features: np.ndarray) -> np.ndarray:
    """Align estimator-specific probability columns to publisher classes 1, 2, 3."""

    estimator_classes = np.asarray(fitted.estimator.classes_, dtype=int)
    if fitted.specification.get("model") == "sgd_log_loss":
        scores = np.asarray(fitted.estimator.decision_function(features), dtype=np.float64)
        if scores.ndim == 1 and len(estimator_classes) == 2:
            positive = np.exp(-np.logaddexp(0.0, -scores))
            raw = np.column_stack((1.0 - positive, positive))
        elif scores.ndim == 2 and scores.shape[1] == len(estimator_classes):
            # sklearn's multiclass SGD log-loss probabilities are normalized one-vs-rest
            # sigmoid outputs. Compute the same quantity in log-space so saturated logits
            # cannot underflow all class probabilities to zero and create 0/0 NaNs.
            log_sigmoid = -np.logaddexp(0.0, -scores)
            shifted = log_sigmoid - np.max(log_sigmoid, axis=1, keepdims=True)
            raw = np.exp(shifted)
            raw /= raw.sum(axis=1, keepdims=True)
        else:
            raise ValueError("SGD classifier produced an invalid decision-function shape")
    elif fitted.specification.get("model") == "lightgbm_multiclass":
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=(
                    "X does not have valid feature names, but LGBMClassifier was fitted "
                    "with feature names"
                ),
                category=UserWarning,
            )
            raw = np.asarray(
                fitted.estimator.predict_proba(features, validate_features=False),
                dtype=np.float64,
            )
    else:
        raw = np.asarray(fitted.estimator.predict_proba(features), dtype=np.float64)
    probabilities = np.zeros((len(features), 3), dtype=np.float64)
    for source_column, label in enumerate(estimator_classes):
        if label not in CLASSES:
            raise ValueError(f"classifier produced an unknown class: {label}")
        probabilities[:, label - 1] = raw[:, source_column]
    row_sums = probabilities.sum(axis=1)
    if not np.isfinite(probabilities).all() or np.any(row_sums <= 0):
        raise ValueError("classifier produced invalid probabilities")
    return probabilities / row_sums[:, None]


def classification_metrics(target: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    predictions = CLASSES[np.argmax(probabilities, axis=1)]
    precision, recall, per_class_f1, support = precision_recall_fscore_support(
        target,
        predictions,
        labels=CLASSES,
        zero_division=0,
    )
    per_class = {}
    names = {1: "up", 2: "stationary", 3: "down"}
    for index, label in enumerate(CLASSES):
        per_class[names[int(label)]] = {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(per_class_f1[index]),
            "support": int(support[index]),
        }
    return {
        "macro_f1": float(
            f1_score(target, predictions, labels=CLASSES, average="macro", zero_division=0)
        ),
        "balanced_accuracy": float(balanced_accuracy_score(target, predictions)),
        "multiclass_log_loss": float(log_loss(target, probabilities, labels=CLASSES)),
        "mcc": float(matthews_corrcoef(target, predictions)),
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(target, predictions, labels=CLASSES).tolist(),
    }


def directional_diagnostics(
    target: np.ndarray, probabilities: np.ndarray, threshold: float
) -> dict[str, Any]:
    """Measure confidence-filtered direction labels; these are not executed trades."""

    up_probability = probabilities[:, 0]
    down_probability = probabilities[:, 2]
    up = (up_probability >= threshold) & (up_probability > down_probability)
    down = (down_probability >= threshold) & (down_probability > up_probability)
    signal = down | up
    predicted_direction = np.full(len(target), 2, dtype=np.int8)
    predicted_direction[up] = 1
    predicted_direction[down] = 3
    correct = signal & (predicted_direction == target)

    def precision_for(mask: np.ndarray, label: int) -> float | None:
        count = int(mask.sum())
        return float(np.mean(target[mask] == label)) if count else None

    return {
        "threshold": float(threshold),
        "directional_precision": float(correct.sum() / signal.sum()) if signal.any() else None,
        "directional_coverage": float(signal.mean()),
        "abstention_rate": float(1.0 - signal.mean()),
        "signals": int(signal.sum()),
        "observations": int(len(target)),
        "up_precision": precision_for(up, 1),
        "down_precision": precision_for(down, 3),
        "tie_policy": "abstain",
    }


def serialized_model_size(fitted: FittedModel) -> int:
    return len(pickle.dumps(fitted.estimator, protocol=pickle.HIGHEST_PROTOCOL))
