"""Chronological train, validation, freeze and one-shot equity holdout stages."""

from __future__ import annotations

import hashlib
import json
import os
from itertools import product
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .equity_config import EquityResearchConfig
from .equity_data import (
    load_prepared_manifest,
    prepared_records_for_split,
)
from .equity_features import (
    equity_model_feature_columns,
    feature_implementation_sha256,
    feature_specification_sha256,
)
from .equity_models import (
    baseline_predictions,
    candidate_model_specs,
    daily_predictive_metrics,
    deterministic_sample_by_date,
    expanding_date_folds,
    fit_model,
    fit_model_with_train_preprocessing,
    fit_signed_imbalance_baseline,
    labeled_rows,
    predictive_metrics,
    preprocessing_parameters,
    spearman_ic,
)
from .equity_trading import (
    TradingRule,
    cluster_bootstrap_by_date,
    simulate_cross_sectional_trading,
    summarize_trading,
)
from .manifest import sha256_file, write_json
from .pipeline import write_table

HOLDOUT_ACKNOWLEDGEMENT = "RELEASE OPTIVER HOLDOUT ONCE"
VALIDATION_ARTIFACT_NAMES = {
    "validation_daily_metrics.csv",
    "validation_execution_filter_diagnostics.csv",
    "validation_feature_ablation.csv",
    "validation_fee_sensitivity.csv",
    "validation_prediction_deciles.csv",
    "validation_predictive_metrics.csv",
    "validation_selected_decisions.csv",
    "validation_selected_legs.csv.gz",
    "validation_selection_sample_daily_metrics.csv",
    "validation_stability.csv",
    "validation_trading_grid.csv",
}


def research_implementation_sha256() -> str:
    """Hash executable preparation, configuration, modelling, study and trading code."""

    digest = hashlib.sha256()
    directory = Path(__file__).resolve().parent
    for name in (
        "equity_config.py",
        "equity_data.py",
        "equity_features.py",
        "equity_models.py",
        "equity_study.py",
        "equity_trading.py",
    ):
        digest.update(name.encode("utf-8"))
        source = (directory / name).read_text(encoding="utf-8").replace("\r\n", "\n")
        digest.update(source.encode("utf-8"))
    return digest.hexdigest()


def _empty_destination(path: str | Path, stage: str) -> Path:
    destination = Path(path)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"refusing to overwrite {stage} output: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    return destination


def _load_records(
    records: list[dict[str, Any]], *, maximum_rows_per_date: int | None = None
) -> pd.DataFrame:
    frames = []
    for record in records:
        frame = _read_partition(record)
        if maximum_rows_per_date is not None:
            frame = deterministic_sample_by_date(frame, maximum_rows_per_date)
        frames.append(frame)
    if not frames:
        raise ValueError("no prepared dates selected")
    return pd.concat(frames, ignore_index=True)


def _read_partition(record: dict[str, Any]) -> pd.DataFrame:
    path = Path(record["path"])
    frame = pd.read_parquet(path)
    if sha256_file(path) != record["sha256"]:
        raise OSError(f"prepared partition changed while being read: {path}")
    if len(frame) != int(record["rows"]):
        raise OSError(f"prepared partition row count mismatch: {path}")
    date_ids = frame["date_id"].unique()
    if len(date_ids) != 1 or int(date_ids[0]) != int(record["date_id"]):
        raise OSError(f"prepared partition date_id mismatch: {path}")
    return frame


def _concat_nonempty(frames: list[pd.DataFrame]) -> pd.DataFrame:
    nonempty = [frame for frame in frames if not frame.empty]
    return pd.concat(nonempty, ignore_index=True) if nonempty else pd.DataFrame()


def _sum_diagnostics(
    total: dict[str, int], frame: pd.DataFrame
) -> dict[str, int]:
    for key, value in frame.attrs.get("execution_filter_diagnostics", {}).items():
        total[key] = total.get(key, 0) + int(value)
    return total


def _minimal_scored_frame(frame: pd.DataFrame, prediction: np.ndarray) -> pd.DataFrame:
    scored = frame.loc[
        :,
        [
            "stock_id",
            "date_id",
            "seconds_in_bucket",
            "time_id",
            "target",
            "time_remaining_seconds",
            "auction_imbalance_ratio",
        ],
    ].copy()
    scored["prediction_bps"] = prediction
    return scored


def _exclusive_json(path: Path, payload: dict[str, Any]) -> None:
    """Create a durable, non-overwritable seal; even a partial crash artifact remains a seal."""

    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def _specification_id(specification: dict[str, Any]) -> str:
    return json.dumps(specification, sort_keys=True, separators=(",", ":"))


def _read_stage(path: str | Path, expected: str) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("stage") != expected:
        raise ValueError(f"expected {expected!r} artifact: {path}")
    return payload


def _assert_development_hashes(
    payload: dict[str, Any],
    config: EquityResearchConfig,
    manifest_path: str | Path,
) -> None:
    if payload.get("source_kind") != config.source_kind:
        raise OSError("source classification changed after development stage")
    if payload.get("target_definition") != config.data.target_definition:
        raise OSError("target definition changed after development stage")
    if payload.get("config_sha256") != sha256_file(config.source_path):
        raise OSError("configuration changed after development stage")
    if Path(str(payload.get("prepared_manifest_path", ""))).resolve() != Path(
        manifest_path
    ).resolve():
        raise OSError("prepared manifest path changed after development stage")
    if payload.get("prepared_manifest_sha256") != sha256_file(manifest_path):
        raise OSError("prepared manifest changed after development stage")
    if payload.get("feature_specification_sha256") != feature_specification_sha256(config):
        raise OSError("feature specification changed after development stage")
    if payload.get("feature_implementation_sha256") != feature_implementation_sha256():
        raise OSError("feature implementation changed after development stage")
    if payload.get("research_implementation_sha256") != research_implementation_sha256():
        raise OSError("equity research implementation changed after development stage")


def _validate_train_selection_artifacts(
    selection: dict[str, Any],
    selection_path: str | Path,
    config: EquityResearchConfig,
) -> None:
    directory = Path(selection_path).resolve().parent
    hashes = selection.get("train_artifact_sha256")
    expected_names = {
        "train_baseline_cv.csv",
        "train_feature_daily_ic.csv",
        "train_model_cv.csv",
        "train_model_cv_summary.csv",
    }
    if not isinstance(hashes, dict) or set(hashes) != expected_names:
        raise OSError("train selection has an incomplete artifact hash inventory")
    for name, expected in hashes.items():
        if sha256_file(directory / name) != expected:
            raise OSError(f"train selection artifact changed: {name}")
    expected_dates = list(range(config.splits.train_start, config.splits.train_end + 1))
    if selection.get("train_date_ids") != expected_dates:
        raise OSError("train selection dates do not match the registration")
    registered = {
        _specification_id(specification)
        for specification in candidate_model_specs(config)
    }
    shortlist = [_specification_id(dict(item)) for item in selection.get("shortlist", [])]
    if len(shortlist) != len(set(shortlist)) or not set(shortlist) <= registered:
        raise OSError("train shortlist is not a unique subset of the registered model grid")
    summary = pd.read_csv(directory / "train_model_cv_summary.csv")
    expected_shortlist = [
        str(group.iloc[0]["specification"])
        for _, group in summary.groupby("model", sort=True, observed=True)
    ]
    if shortlist != expected_shortlist:
        raise OSError("train shortlist no longer matches the locked CV summary")


def _validate_candidate_artifacts(
    candidate: dict[str, Any], candidate_path: str | Path
) -> None:
    hashes = candidate.get("validation_artifact_sha256")
    if not isinstance(hashes, dict) or set(hashes) != VALIDATION_ARTIFACT_NAMES:
        raise OSError("validation candidate has an incomplete artifact hash inventory")
    directory = Path(
        candidate.get("validation_artifact_directory", Path(candidate_path).resolve().parent)
    ).resolve()
    for name, expected in hashes.items():
        if sha256_file(directory / name) != expected:
            raise OSError(f"validation selection artifact changed: {name}")


def run_equity_train_stage(
    config: EquityResearchConfig,
    *,
    manifest_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Run bounded train-only diagnostics and expanding-window model CV."""

    destination = _empty_destination(output_dir, "equity train")
    manifest = load_prepared_manifest(manifest_path, config)
    train_records = prepared_records_for_split(manifest, "train")
    train = _load_records(
        train_records,
        maximum_rows_per_date=config.model.max_tuning_rows_per_date,
    )
    folds = expanding_date_folds(
        train["date_id"].unique(),
        minimum_train_dates=config.model.minimum_train_dates,
        folds=config.model.cv_folds,
    )
    specifications = candidate_model_specs(config)
    if not specifications:
        raise RuntimeError("no registered candidate model is available")
    cv_rows: list[dict[str, Any]] = []
    baseline_rows: list[dict[str, Any]] = []
    for fold_number, (fit_dates, score_dates) in enumerate(folds, start=1):
        fit = train.loc[train["date_id"].isin(fit_dates)]
        score = train.loc[train["date_id"].isin(score_dates)]
        score_labeled = labeled_rows(score)
        slope = fit_signed_imbalance_baseline(fit)
        for name, prediction in baseline_predictions(
            score_labeled, imbalance_slope=slope
        ).items():
            baseline_rows.append(
                {
                    "fold": fold_number,
                    "model": name,
                    "fit_date_max": max(fit_dates),
                    "score_date_min": min(score_dates),
                    "score_date_max": max(score_dates),
                    **predictive_metrics(
                        score_labeled["target"].to_numpy(dtype=float), prediction
                    ),
                }
            )
        for specification in specifications:
            model = fit_model(config, specification, fit)
            prediction = model.predict(score_labeled)
            cv_rows.append(
                {
                    "fold": fold_number,
                    "model": specification["model"],
                    "specification": _specification_id(specification),
                    "fit_date_max": max(fit_dates),
                    "score_date_min": min(score_dates),
                    "score_date_max": max(score_dates),
                    **predictive_metrics(
                        score_labeled["target"].to_numpy(dtype=float), prediction
                    ),
                }
            )
    cv = pd.DataFrame(cv_rows)
    baselines = pd.DataFrame(baseline_rows)
    write_table(cv, destination / "train_model_cv.csv")
    write_table(baselines, destination / "train_baseline_cv.csv")
    summary = (
        cv.groupby(["model", "specification"], sort=True, observed=True)
        .agg(
            folds=("fold", "size"),
            mean_mae_bps=("mae_bps", "mean"),
            mean_spearman_ic=("spearman_ic", "mean"),
            mean_directional_accuracy=("directional_accuracy", "mean"),
        )
        .reset_index()
        .sort_values(["model", "mean_mae_bps", "specification"], kind="stable")
    )
    write_table(summary, destination / "train_model_cv_summary.csv")
    shortlist = []
    for _, group in summary.groupby("model", sort=True, observed=True):
        shortlist.append(json.loads(group.iloc[0]["specification"]))

    feature_rows = []
    for column in equity_model_feature_columns(config):
        for date_id, group in train.groupby("date_id", sort=True, observed=True):
            feature_rows.append(
                {
                    "date_id": int(date_id),
                    "feature": column,
                    "spearman_ic": spearman_ic(group[column], group["target"]),
                    "rows": int(group[[column, "target"]].dropna().shape[0]),
                }
            )
    write_table(pd.DataFrame(feature_rows), destination / "train_feature_daily_ic.csv")
    train_artifact_names = [
        "train_baseline_cv.csv",
        "train_feature_daily_ic.csv",
        "train_model_cv.csv",
        "train_model_cv_summary.csv",
    ]
    payload: dict[str, Any] = {
        "stage": "equity_train_complete",
        "source_kind": config.source_kind,
        "target_definition": config.data.target_definition,
        "config_sha256": sha256_file(config.source_path),
        "prepared_manifest_path": str(Path(manifest_path).resolve()),
        "prepared_manifest_sha256": sha256_file(manifest_path),
        "feature_specification_sha256": feature_specification_sha256(config),
        "feature_implementation_sha256": feature_implementation_sha256(),
        "research_implementation_sha256": research_implementation_sha256(),
        "train_date_ids": [int(item["date_id"]) for item in train_records],
        "holdout_rows_read": 0,
        "primary_selection_metric": "mean_absolute_error_target_bps",
        "train_artifact_sha256": {
            name: sha256_file(destination / name) for name in train_artifact_names
        },
        "shortlist": shortlist,
        "unavailable_optional_models": sorted(
            set(config.model.candidate_models) - {str(item["model"]) for item in specifications}
        ),
    }
    write_json(destination / "train_selection.json", payload)
    return payload


def _prediction_deciles(frame: pd.DataFrame, prediction: np.ndarray) -> pd.DataFrame:
    scored = frame.loc[:, ["date_id", "target"]].copy()
    scored["prediction_bps"] = prediction
    scored = scored.loc[
        pd.to_numeric(scored["target"], errors="coerce").notna()
        & np.isfinite(pd.to_numeric(scored["target"], errors="coerce"))
        & np.isfinite(scored["prediction_bps"])
    ]
    scored["prediction_decile"] = pd.qcut(
        scored["prediction_bps"], q=10, labels=False, duplicates="drop"
    )
    return (
        scored.groupby("prediction_decile", sort=True, observed=True)
        .agg(
            rows=("target", "size"),
            mean_target_bps=("target", "mean"),
            median_target_bps=("target", "median"),
        )
        .reset_index()
    )


def _stability_table(frame: pd.DataFrame, prediction: np.ndarray) -> pd.DataFrame:
    scored = frame.loc[:, ["target", "time_remaining_seconds", "auction_imbalance_ratio"]].copy()
    scored["prediction"] = prediction
    scored = scored.loc[
        pd.to_numeric(scored["target"], errors="coerce").notna()
        & np.isfinite(pd.to_numeric(scored["target"], errors="coerce"))
        & np.isfinite(scored["prediction"])
    ]
    scored["time_regime"] = pd.cut(
        scored["time_remaining_seconds"],
        bins=[-np.inf, 120, 300, np.inf],
        labels=["last_120s", "121_to_300s", "more_than_300s"],
    )
    absolute = scored["auction_imbalance_ratio"].abs()
    scored["imbalance_regime"] = pd.cut(
        absolute,
        bins=[-np.inf, 0.05, 0.2, np.inf],
        labels=["low", "medium", "high"],
    )
    rows = []
    for dimension in ("time_regime", "imbalance_regime"):
        for regime, group in scored.groupby(dimension, sort=True, observed=True):
            rows.append(
                {
                    "dimension": dimension,
                    "regime": str(regime),
                    **predictive_metrics(
                        group["target"].to_numpy(dtype=float),
                        group["prediction"].to_numpy(dtype=float),
                    ),
                }
            )
    return pd.DataFrame(rows)


def _feature_families(config: EquityResearchConfig) -> dict[str, list[str]]:
    columns = equity_model_feature_columns(config)
    cross = [column for column in columns if "_cs_" in column]
    dynamics = [
        column
        for column in columns
        if any(marker in column for marker in ("_return_", "_change_", "_mean_", "_vol_"))
        and column not in cross
    ]
    auction = [
        column
        for column in columns
        if any(marker in column for marker in ("imbalance", "matched", "near_", "far_", "time_"))
        and column not in cross
        and column not in dynamics
    ]
    order_book = [column for column in columns if column not in set(cross + dynamics + auction)]
    return {
        "order_book": order_book,
        "auction": auction,
        "within_stock_dynamics": dynamics,
        "cross_sectional": cross,
    }


def _trading_grid(config: EquityResearchConfig) -> list[TradingRule]:
    return [
        TradingRule(
            group_quantile=quantile,
            minimum_absolute_prediction_bps=minimum_prediction,
            maximum_spread_bps=maximum_spread,
            minimum_displayed_liquidity=minimum_liquidity,
            fee_per_side_bps=config.execution.primary_fee_per_side_bps,
        )
        for quantile, minimum_prediction, maximum_spread, minimum_liquidity in product(
            config.execution.group_quantiles,
            config.execution.minimum_absolute_predictions_bps,
            config.execution.maximum_spreads_bps,
            config.execution.minimum_displayed_liquidity,
        )
    ]


def _rule_payload(rule: TradingRule) -> dict[str, float]:
    return {
        "group_quantile": rule.group_quantile,
        "minimum_absolute_prediction_bps": rule.minimum_absolute_prediction_bps,
        "maximum_spread_bps": rule.maximum_spread_bps,
        "minimum_displayed_liquidity": rule.minimum_displayed_liquidity,
        "fee_per_side_bps": rule.fee_per_side_bps,
    }


def _decision_grid_summary(decisions: pd.DataFrame) -> dict[str, float | int | None]:
    if decisions.empty:
        return {
            "decisions": 0,
            "gross_mean_bps": None,
            "net_mean_bps": None,
            "spread_cost_mean_bps": None,
            "fee_mean_bps": None,
            "minimum_displayed_capacity_units": None,
        }
    return {
        "decisions": int(len(decisions)),
        "gross_mean_bps": float(decisions["gross_return_bps"].mean()),
        "net_mean_bps": float(decisions["net_return_bps"].mean()),
        "spread_cost_mean_bps": float(decisions["spread_cost_bps"].mean()),
        "fee_mean_bps": float(decisions["fee_bps"].mean()),
        "minimum_displayed_capacity_units": float(
            decisions["minimum_displayed_capacity_units"].min()
        ),
    }


def _reprice_fee(
    legs: pd.DataFrame, decisions: pd.DataFrame, fee_per_side_bps: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    repriced_legs = legs.copy()
    repriced_decisions = decisions.copy()
    total_fee = 2.0 * fee_per_side_bps
    if not repriced_legs.empty:
        repriced_legs["fee_bps"] = total_fee
        repriced_legs["net_return_bps"] = (
            repriced_legs["gross_executable_return_bps"] - total_fee
        )
    if not repriced_decisions.empty:
        repriced_decisions["fee_bps"] = total_fee
        repriced_decisions["net_return_bps"] = (
            repriced_decisions["gross_return_bps"] - total_fee
        )
    return repriced_legs, repriced_decisions


def run_equity_validation_stage(
    config: EquityResearchConfig,
    *,
    manifest_path: str | Path,
    train_selection_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Select the final model and trading rule using validation, then refit development."""

    destination = _empty_destination(output_dir, "equity validation")
    selection = _read_stage(train_selection_path, "equity_train_complete")
    _assert_development_hashes(selection, config, manifest_path)
    _validate_train_selection_artifacts(selection, train_selection_path, config)
    manifest = load_prepared_manifest(manifest_path, config, scope="development")
    train_records = prepared_records_for_split(manifest, "train")
    validation_records = prepared_records_for_split(manifest, "validation")
    train = _load_records(
        train_records,
        maximum_rows_per_date=config.model.max_refit_rows_per_date,
    )
    validation_selection = _load_records(
        validation_records,
        maximum_rows_per_date=config.model.max_tuning_rows_per_date,
    )
    validation_selection_labeled = labeled_rows(validation_selection)
    actual_selection = validation_selection_labeled["target"].to_numpy(dtype=float)
    slope = fit_signed_imbalance_baseline(train)
    overall_rows: list[dict[str, Any]] = []
    selection_daily_frames = []
    for name, prediction in baseline_predictions(
        validation_selection_labeled, imbalance_slope=slope
    ).items():
        overall_rows.append(
            {
                "model": name,
                "evaluation_scope": "deterministic_validation_selection_sample",
                **predictive_metrics(actual_selection, prediction),
            }
        )
        daily = daily_predictive_metrics(
            validation_selection_labeled, prediction, model_name=name
        )
        daily["evaluation_scope"] = "deterministic_validation_selection_sample"
        selection_daily_frames.append(daily)
    candidates: list[tuple[dict[str, Any], Any, np.ndarray, dict[str, Any]]] = []
    for specification in selection["shortlist"]:
        model = fit_model(config, dict(specification), train)
        prediction = model.predict(validation_selection_labeled)
        metrics = predictive_metrics(actual_selection, prediction)
        overall_rows.append(
            {
                "model": _specification_id(specification),
                "evaluation_scope": "deterministic_validation_selection_sample",
                **metrics,
            }
        )
        daily = daily_predictive_metrics(
            validation_selection_labeled,
            prediction,
            model_name=_specification_id(specification),
        )
        daily["evaluation_scope"] = "deterministic_validation_selection_sample"
        selection_daily_frames.append(daily)
        candidates.append((dict(specification), model, prediction, metrics))
    candidates.sort(key=lambda item: (float(item[3]["mae_bps"]), _specification_id(item[0])))
    selected_specification, selected_model, _, selected_sample_metrics = candidates[0]
    write_table(
        pd.concat(selection_daily_frames, ignore_index=True),
        destination / "validation_selection_sample_daily_metrics.csv",
    )

    rules = _trading_grid(config)
    grid_decision_frames: list[list[pd.DataFrame]] = [[] for _ in rules]
    scored_frames: list[pd.DataFrame] = []
    full_daily_frames: list[pd.DataFrame] = []
    selected_name = _specification_id(selected_specification)
    for record in validation_records:
        partition = _read_partition(record)
        prediction = selected_model.predict(partition)
        scored_frames.append(_minimal_scored_frame(partition, prediction))
        full_daily_frames.append(
            daily_predictive_metrics(
                partition,
                prediction,
                model_name=selected_name,
            )
        )
        for index, rule in enumerate(rules):
            _, decisions = simulate_cross_sectional_trading(
                partition,
                prediction,
                rule,
                horizon_seconds=config.data.target_horizon_seconds,
                decision_interval_seconds=config.execution.decision_interval_seconds,
            )
            if not decisions.empty:
                grid_decision_frames[index].append(decisions)

    scored = _concat_nonempty(scored_frames)
    if scored.empty:
        raise ValueError("validation partitions contain no rows")
    selected_prediction = scored["prediction_bps"].to_numpy(dtype=float)
    labeled_scored = labeled_rows(scored)
    selected_metrics = predictive_metrics(
        labeled_scored["target"].to_numpy(dtype=float),
        labeled_scored["prediction_bps"].to_numpy(dtype=float),
    )
    overall_rows.append(
        {
            "model": selected_name,
            "evaluation_scope": "full_validation_selected_model_only",
            **selected_metrics,
        }
    )
    full_daily = pd.concat(full_daily_frames, ignore_index=True)
    full_daily["evaluation_scope"] = "full_validation_selected_model_only"
    write_table(pd.DataFrame(overall_rows), destination / "validation_predictive_metrics.csv")
    write_table(full_daily, destination / "validation_daily_metrics.csv")
    write_table(
        _prediction_deciles(scored, selected_prediction),
        destination / "validation_prediction_deciles.csv",
    )
    write_table(
        _stability_table(scored, selected_prediction),
        destination / "validation_stability.csv",
    )
    predictive_interval = cluster_bootstrap_by_date(
        full_daily["spearman_ic"],
        repetitions=config.execution.bootstrap_repetitions,
        seed=config.seed,
    )

    all_features = equity_model_feature_columns(config)
    ablation_rows = [{"omitted_family": "none", **selected_sample_metrics}]
    for family, omitted in _feature_families(config).items():
        retained = [column for column in all_features if column not in set(omitted)]
        ablated_model = fit_model(
            config,
            selected_specification,
            train,
            numeric_features=retained,
        )
        ablated_prediction = ablated_model.predict(validation_selection_labeled)
        ablation_rows.append(
            {
                "omitted_family": family,
                **predictive_metrics(actual_selection, ablated_prediction),
            }
        )
    write_table(pd.DataFrame(ablation_rows), destination / "validation_feature_ablation.csv")

    trading_rows: list[dict[str, Any]] = []
    trading_results: list[tuple[TradingRule, pd.DataFrame, dict[str, Any]]] = []
    for rule, decision_frames in zip(rules, grid_decision_frames, strict=True):
        decisions = _concat_nonempty(decision_frames)
        summary = _decision_grid_summary(decisions)
        trading_results.append((rule, decisions, summary))
        trading_rows.append({**_rule_payload(rule), **summary})
    eligible_results = [item for item in trading_results if int(item[2]["decisions"]) > 0]
    if not eligible_results:
        raise ValueError("registered validation trading grid produced no eligible decisions")
    eligible_results.sort(
        key=lambda item: (
            -float(item[2]["net_mean_bps"]),
            json.dumps(_rule_payload(item[0]), sort_keys=True),
        )
    )
    selected_rule = eligible_results[0][0]
    write_table(pd.DataFrame(trading_rows), destination / "validation_trading_grid.csv")

    selected_leg_frames: list[pd.DataFrame] = []
    selected_decision_frames: list[pd.DataFrame] = []
    execution_diagnostics: dict[str, int] = {}
    for record in validation_records:
        partition = _read_partition(record)
        prediction = selected_model.predict(partition)
        legs, decisions = simulate_cross_sectional_trading(
            partition,
            prediction,
            selected_rule,
            horizon_seconds=config.data.target_horizon_seconds,
            decision_interval_seconds=config.execution.decision_interval_seconds,
        )
        _sum_diagnostics(execution_diagnostics, legs)
        if not legs.empty:
            selected_leg_frames.append(legs)
        if not decisions.empty:
            selected_decision_frames.append(decisions)
    selected_legs = _concat_nonempty(selected_leg_frames)
    selected_decisions = _concat_nonempty(selected_decision_frames)
    selected_trading = summarize_trading(
        selected_legs,
        selected_decisions,
        repetitions=config.execution.bootstrap_repetitions,
        seed=config.seed,
    )
    write_table(selected_legs, destination / "validation_selected_legs.csv.gz")
    write_table(selected_decisions, destination / "validation_selected_decisions.csv")
    write_table(
        pd.DataFrame(
            sorted(execution_diagnostics.items()),
            columns=["filter", "rows_or_time_ids"],
        ),
        destination / "validation_execution_filter_diagnostics.csv",
    )
    fee_rows = []
    for fee in config.execution.fee_sensitivity_per_side_bps:
        legs, decisions = _reprice_fee(
            selected_legs,
            selected_decisions,
            fee,
        )
        fee_rows.append(
            {
                "fee_per_side_bps": fee,
                **{
                    key: value
                    for key, value in summarize_trading(
                        legs,
                        decisions,
                        repetitions=config.execution.bootstrap_repetitions,
                        seed=config.seed,
                    ).items()
                    if key not in {"daily_cluster_bootstrap", "daily_return_distribution"}
                },
            }
        )
    write_table(pd.DataFrame(fee_rows), destination / "validation_fee_sensitivity.csv")

    validation_refit = _load_records(
        validation_records,
        maximum_rows_per_date=config.model.max_refit_rows_per_date,
    )
    development = pd.concat([train, validation_refit], ignore_index=True)
    frozen_model = fit_model_with_train_preprocessing(
        config,
        selected_specification,
        preprocessing_frame=train,
        estimator_frame=development,
    )
    model_path = destination / "fitted_candidate.joblib"
    model_temporary = model_path.with_suffix(model_path.suffix + ".partial")
    joblib.dump(frozen_model, model_temporary)
    model_temporary.replace(model_path)
    preprocessing_path = destination / "preprocessing_parameters.json"
    write_json(preprocessing_path, preprocessing_parameters(frozen_model))
    candidate: dict[str, Any] = {
        "stage": "equity_validation_selected",
        "source_kind": config.source_kind,
        "target_definition": config.data.target_definition,
        "config_path": str(config.source_path),
        "config_sha256": sha256_file(config.source_path),
        "prepared_manifest_path": str(Path(manifest_path).resolve()),
        "prepared_manifest_sha256": sha256_file(manifest_path),
        "raw_metadata_path": manifest["raw_metadata_path"],
        "raw_metadata_sha256": manifest["raw_metadata_sha256"],
        "raw_path": manifest["raw_path"],
        "raw_sha256": manifest["raw_sha256"],
        "feature_specification_sha256": feature_specification_sha256(config),
        "feature_implementation_sha256": feature_implementation_sha256(),
        "research_implementation_sha256": research_implementation_sha256(),
        "train_selection_path": str(Path(train_selection_path).resolve()),
        "train_selection_sha256": sha256_file(train_selection_path),
        "model_specification": selected_specification,
        "model_path": str(model_path.resolve()),
        "model_sha256": sha256_file(model_path),
        "preprocessing_parameters_path": str(preprocessing_path.resolve()),
        "preprocessing_parameters_sha256": sha256_file(preprocessing_path),
        "preprocessing_fit_partition": "train_only",
        "estimator_refit_partitions": ["train", "validation"],
        "trading_rule": _rule_payload(selected_rule),
        "validation_selection_sample_predictive_metrics": selected_sample_metrics,
        "validation_predictive_metrics": selected_metrics,
        "validation_daily_ic_bootstrap": predictive_interval,
        "validation_trading_summary": selected_trading,
        "train_date_ids": [int(item["date_id"]) for item in train_records],
        "validation_date_ids": [int(item["date_id"]) for item in validation_records],
        "validation_selection_sample_rows": int(len(validation_selection)),
        "validation_selection_sample_target_rows": int(len(validation_selection_labeled)),
        "validation_full_rows": int(len(scored)),
        "validation_full_target_rows": int(len(labeled_scored)),
        "validation_execution_filter_diagnostics": execution_diagnostics,
        "validation_artifact_directory": str(destination.resolve()),
        "validation_artifact_sha256": {
            name: sha256_file(destination / name)
            for name in sorted(VALIDATION_ARTIFACT_NAMES)
        },
        "holdout_rows_read": 0,
    }
    write_json(destination / "selected_candidate.json", candidate)
    return candidate


def freeze_equity_candidate(
    config: EquityResearchConfig,
    *,
    manifest_path: str | Path,
    candidate_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Content-address the complete development choice before holdout access."""

    output = Path(output_path)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite frozen equity candidate: {output}")
    manifest = load_prepared_manifest(
        manifest_path,
        config,
        scope=None,
        verify_partitions=False,
    )
    seal_path = Path(manifest_path).resolve().parent / "HOLDOUT_STARTED.json"
    if seal_path.exists():
        raise FileExistsError(
            f"the prepared study is already permanently sealed for holdout: {seal_path}"
        )
    candidate = _read_stage(candidate_path, "equity_validation_selected")
    _assert_development_hashes(candidate, config, manifest_path)
    _validate_candidate_artifacts(candidate, candidate_path)
    for path_key, hash_key in (
        ("raw_metadata_path", "raw_metadata_sha256"),
        ("raw_path", "raw_sha256"),
        ("train_selection_path", "train_selection_sha256"),
        ("model_path", "model_sha256"),
        ("preprocessing_parameters_path", "preprocessing_parameters_sha256"),
    ):
        if sha256_file(candidate[path_key]) != candidate[hash_key]:
            raise OSError(f"candidate input changed before freeze: {path_key}")
    selection = _read_stage(candidate["train_selection_path"], "equity_train_complete")
    _validate_train_selection_artifacts(selection, candidate["train_selection_path"], config)
    if candidate["model_specification"] not in selection["shortlist"]:
        raise OSError("selected model is not in the locked train shortlist")
    fitted_model = joblib.load(candidate["model_path"])
    estimator = fitted_model.named_steps["model"]
    specification = candidate["model_specification"]
    expected_class = {
        "ridge": "Ridge",
        "hist_gradient_boosting": "HistGradientBoostingRegressor",
        "lightgbm": "LGBMRegressor",
    }[specification["model"]]
    if type(estimator).__name__ != expected_class:
        raise OSError("fitted estimator type does not match the selected model specification")
    expected_parameters = {
        key: value for key, value in specification.items() if key != "model"
    }
    actual_parameters = estimator.get_params(deep=False)
    if any(actual_parameters.get(key) != value for key, value in expected_parameters.items()):
        raise OSError("fitted estimator parameters do not match the selected specification")
    recorded_preprocessing = json.loads(
        Path(candidate["preprocessing_parameters_path"]).read_text(encoding="utf-8")
    )
    if recorded_preprocessing != preprocessing_parameters(fitted_model):
        raise OSError("recorded preprocessing state does not match the fitted model")
    registered_rules = [_rule_payload(rule) for rule in _trading_grid(config)]
    if candidate["trading_rule"] not in registered_rules:
        raise OSError("selected trading rule is outside the registered validation grid")
    if candidate.get("train_date_ids") != list(
        range(config.splits.train_start, config.splits.train_end + 1)
    ) or candidate.get("validation_date_ids") != list(
        range(config.splits.validation_start, config.splits.validation_end + 1)
    ):
        raise OSError("candidate date partitions do not match the registration")
    payload = {
        **candidate,
        "stage": "equity_frozen_for_holdout",
        "candidate_path": str(Path(candidate_path).resolve()),
        "candidate_sha256": sha256_file(candidate_path),
        "holdout_seal_path": str(seal_path),
        "holdout_manifest_path": manifest["holdout_manifest_path"],
        "holdout_manifest_sha256": manifest["holdout_manifest_sha256"],
    }
    write_json(output, payload)
    return payload


def _validate_frozen_inputs(
    config: EquityResearchConfig,
    manifest_path: str | Path,
    frozen_path: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    frozen = _read_stage(frozen_path, "equity_frozen_for_holdout")
    _assert_development_hashes(frozen, config, manifest_path)
    if sha256_file(frozen["candidate_path"]) != frozen["candidate_sha256"]:
        raise OSError("validation candidate changed after freeze")
    candidate = _read_stage(frozen["candidate_path"], "equity_validation_selected")
    _validate_candidate_artifacts(candidate, frozen["candidate_path"])
    for key, value in candidate.items():
        if key == "stage":
            continue
        if frozen.get(key) != value:
            raise OSError(f"frozen candidate field changed after validation: {key}")
    for path_key, hash_key in (
        ("raw_metadata_path", "raw_metadata_sha256"),
        ("raw_path", "raw_sha256"),
        ("train_selection_path", "train_selection_sha256"),
        ("model_path", "model_sha256"),
        ("preprocessing_parameters_path", "preprocessing_parameters_sha256"),
    ):
        if sha256_file(frozen[path_key]) != frozen[hash_key]:
            raise OSError(f"frozen input hash mismatch: {path_key}")
    manifest = load_prepared_manifest(
        manifest_path,
        config,
        scope=None,
        verify_partitions=False,
    )
    expected_seal = Path(manifest_path).resolve().parent / "HOLDOUT_STARTED.json"
    if Path(frozen.get("holdout_seal_path", "")).resolve() != expected_seal:
        raise OSError("frozen candidate does not reference the stable prepared-data holdout seal")
    if frozen.get("holdout_manifest_path") != manifest.get("holdout_manifest_path"):
        raise OSError("frozen holdout manifest path does not match prepared manifest")
    if frozen.get("holdout_manifest_sha256") != manifest.get("holdout_manifest_sha256"):
        raise OSError("frozen holdout manifest hash does not match prepared manifest")
    return frozen, manifest


def run_equity_holdout_stage(
    config: EquityResearchConfig,
    *,
    manifest_path: str | Path,
    frozen_candidate_path: str | Path,
    output_dir: str | Path,
    acknowledge_one_shot: str,
) -> dict[str, Any]:
    """Evaluate exactly one frozen candidate; never fit or select on holdout."""

    if acknowledge_one_shot != HOLDOUT_ACKNOWLEDGEMENT:
        raise ValueError(
            "explicit one-shot holdout acknowledgement must exactly equal "
            f"{HOLDOUT_ACKNOWLEDGEMENT!r}"
        )
    frozen, _ = _validate_frozen_inputs(config, manifest_path, frozen_candidate_path)
    destination = _empty_destination(output_dir, "equity holdout")
    seal_path = Path(frozen["holdout_seal_path"])
    started = {
        "stage": "equity_holdout_started_one_shot",
        "acknowledgement": HOLDOUT_ACKNOWLEDGEMENT,
        "frozen_candidate_path": str(Path(frozen_candidate_path).resolve()),
        "frozen_candidate_sha256": sha256_file(frozen_candidate_path),
        "prepared_manifest_path": str(Path(manifest_path).resolve()),
        "prepared_manifest_sha256": sha256_file(manifest_path),
        "holdout_output_path": str(destination.resolve()),
        "selection_permitted": False,
    }
    _exclusive_json(seal_path, started)
    write_json(destination / "HOLDOUT_STARTED.json", started)
    manifest = load_prepared_manifest(
        manifest_path,
        config,
        scope="holdout",
        verify_partitions=True,
    )
    holdout_records = prepared_records_for_split(manifest, "holdout")
    model = joblib.load(frozen["model_path"])
    rule = TradingRule(**frozen["trading_rule"])
    scored_frames: list[pd.DataFrame] = []
    daily_frames: list[pd.DataFrame] = []
    leg_frames: list[pd.DataFrame] = []
    decision_frames: list[pd.DataFrame] = []
    execution_diagnostics: dict[str, int] = {}
    for record in holdout_records:
        partition = _read_partition(record)
        prediction = model.predict(partition)
        scored_frames.append(_minimal_scored_frame(partition, prediction))
        daily_frames.append(
            daily_predictive_metrics(
                partition,
                prediction,
                model_name="frozen_candidate",
            )
        )
        legs, decisions = simulate_cross_sectional_trading(
            partition,
            prediction,
            rule,
            horizon_seconds=config.data.target_horizon_seconds,
            decision_interval_seconds=config.execution.decision_interval_seconds,
        )
        _sum_diagnostics(execution_diagnostics, legs)
        if not legs.empty:
            leg_frames.append(legs)
        if not decisions.empty:
            decision_frames.append(decisions)
    holdout = _concat_nonempty(scored_frames)
    if holdout.empty:
        raise ValueError("holdout partitions contain no rows")
    prediction = holdout["prediction_bps"].to_numpy(dtype=float)
    labeled_holdout = labeled_rows(holdout)
    overall = predictive_metrics(
        labeled_holdout["target"].to_numpy(dtype=float),
        labeled_holdout["prediction_bps"].to_numpy(dtype=float),
    )
    daily = pd.concat(daily_frames, ignore_index=True)
    write_table(daily, destination / "holdout_daily_metrics.csv")
    write_table(
        _prediction_deciles(holdout, prediction),
        destination / "holdout_prediction_deciles.csv",
    )
    write_table(
        _stability_table(holdout, prediction),
        destination / "holdout_stability.csv",
    )
    prediction_output = holdout.loc[
        :, ["stock_id", "date_id", "seconds_in_bucket", "time_id", "target", "prediction_bps"]
    ].copy()
    write_table(prediction_output, destination / "holdout_predictions.csv.gz")
    legs = _concat_nonempty(leg_frames)
    decisions = _concat_nonempty(decision_frames)
    write_table(legs, destination / "holdout_legs.csv.gz")
    write_table(decisions, destination / "holdout_decisions.csv")
    write_table(
        pd.DataFrame(
            sorted(execution_diagnostics.items()),
            columns=["filter", "rows_or_time_ids"],
        ),
        destination / "holdout_execution_filter_diagnostics.csv",
    )
    trading = summarize_trading(
        legs,
        decisions,
        repetitions=config.execution.bootstrap_repetitions,
        seed=config.seed,
    )
    fee_rows = []
    for fee in config.execution.fee_sensitivity_per_side_bps:
        fee_legs, fee_decisions = _reprice_fee(legs, decisions, fee)
        fee_rows.append(
            {
                "fee_per_side_bps": fee,
                **{
                    key: value
                    for key, value in summarize_trading(
                        fee_legs,
                        fee_decisions,
                        repetitions=config.execution.bootstrap_repetitions,
                        seed=config.seed,
                    ).items()
                    if key not in {"daily_cluster_bootstrap", "daily_return_distribution"}
                },
            }
        )
    write_table(pd.DataFrame(fee_rows), destination / "holdout_fee_sensitivity.csv")
    daily_ic_interval = cluster_bootstrap_by_date(
        daily["spearman_ic"],
        repetitions=config.execution.bootstrap_repetitions,
        seed=config.seed,
    )
    artifact_names = [
        "HOLDOUT_STARTED.json",
        "holdout_daily_metrics.csv",
        "holdout_prediction_deciles.csv",
        "holdout_stability.csv",
        "holdout_predictions.csv.gz",
        "holdout_legs.csv.gz",
        "holdout_decisions.csv",
        "holdout_execution_filter_diagnostics.csv",
        "holdout_fee_sensitivity.csv",
    ]
    artifact_sha256 = {
        name: sha256_file(destination / name) for name in sorted(artifact_names)
    }
    payload: dict[str, Any] = {
        "stage": "equity_holdout_complete",
        "source_kind": config.source_kind,
        "target_definition": config.data.target_definition,
        "frozen_candidate_path": str(Path(frozen_candidate_path).resolve()),
        "frozen_candidate_sha256": sha256_file(frozen_candidate_path),
        "holdout_seal_path": str(seal_path.resolve()),
        "holdout_seal_sha256": sha256_file(seal_path),
        "prepared_manifest_path": str(Path(manifest_path).resolve()),
        "prepared_manifest_sha256": sha256_file(manifest_path),
        "holdout_manifest_path": manifest["holdout_manifest_path"],
        "holdout_manifest_sha256": manifest["holdout_manifest_sha256"],
        "raw_metadata_sha256": frozen["raw_metadata_sha256"],
        "raw_sha256": frozen["raw_sha256"],
        "config_sha256": frozen["config_sha256"],
        "feature_specification_sha256": frozen["feature_specification_sha256"],
        "feature_implementation_sha256": frozen["feature_implementation_sha256"],
        "research_implementation_sha256": frozen["research_implementation_sha256"],
        "train_selection_sha256": frozen["train_selection_sha256"],
        "model_sha256": frozen["model_sha256"],
        "preprocessing_parameters_sha256": frozen["preprocessing_parameters_sha256"],
        "holdout_date_ids": [int(item["date_id"]) for item in holdout_records],
        "predictive_metrics": overall,
        "daily_ic_cluster_bootstrap": daily_ic_interval,
        "trading_summary": trading,
        "execution_filter_diagnostics": execution_diagnostics,
        "holdout_rows": int(len(holdout)),
        "holdout_target_rows": int(len(labeled_holdout)),
        "holdout_missing_target_rows": int(len(holdout) - len(labeled_holdout)),
        "holdout_artifact_sha256": artifact_sha256,
        "cost_assumption": {
            "spread": "cross current and exact +60-second executable quotes; embedded once",
            "fee_per_side_bps": rule.fee_per_side_bps,
            "round_trip_fee_bps": 2.0 * rule.fee_per_side_bps,
            "market_impact_modeled": False,
            "displayed_liquidity_guarantees_fill": False,
        },
        "selection_performed": False,
        "claim_eligible_real_optiver": config.source_kind == "real",
    }
    completion_path = destination / "HOLDOUT_COMPLETE.json"
    write_json(completion_path, payload)
    _exclusive_json(
        seal_path.with_name("HOLDOUT_COMPLETE_ANCHOR.json"),
        {
            "stage": "equity_holdout_complete_anchor",
            "holdout_complete_path": str(completion_path.resolve()),
            "holdout_complete_sha256": sha256_file(completion_path),
            "holdout_seal_path": str(seal_path.resolve()),
            "holdout_seal_sha256": sha256_file(seal_path),
            "frozen_candidate_sha256": sha256_file(frozen_candidate_path),
            "prepared_manifest_sha256": sha256_file(manifest_path),
        },
    )
    return payload
