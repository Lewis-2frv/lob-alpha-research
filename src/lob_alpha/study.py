"""Chronological train/validation/frozen-holdout research orchestration."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from .analysis import (
    apply_quantile_analysis,
    cluster_bootstrap_mean,
    daily_information_coefficient,
    fit_quantile_edges,
    spearman_correlation,
    summarize_trades,
)
from .backtest import simulate_marketable_strategy
from .config import ResearchConfig
from .dataset import CatalogEntry, discover_daily_raw_files, load_catalog, read_processed_table
from .features import model_feature_columns
from .ingest import load_events
from .manifest import sha256_file, write_json
from .models import fit_ridge, regression_metrics, score_regression
from .pipeline import write_table


def expanding_session_folds(
    session_dates: Iterable[str], *, minimum_train_sessions: int, folds: int
) -> list[tuple[tuple[str, ...], tuple[str, ...]]]:
    """Return contiguous expanding-window folds with strictly future validation days."""

    dates = tuple(sorted(set(session_dates)))
    if len(dates) <= minimum_train_sessions:
        raise ValueError(
            f"need more than {minimum_train_sessions} train sessions for chronological CV"
        )
    future = np.array(dates[minimum_train_sessions:], dtype=object)
    blocks = [tuple(str(value) for value in block) for block in np.array_split(future, folds)]
    result = []
    for validation in blocks:
        if not validation:
            continue
        train = tuple(value for value in dates if value < validation[0])
        if set(train) & set(validation) or max(train) >= min(validation):
            raise AssertionError("chronological fold construction failed")
        result.append((train, validation))
    if len(result) < 2:
        raise ValueError("chronological CV requires at least two nonempty folds")
    return result


def _entries_for_split(entries: list[CatalogEntry], split: str) -> list[CatalogEntry]:
    selected = [entry for entry in entries if entry.split == split]
    if not selected:
        raise ValueError(f"catalog contains no {split!r} sessions")
    return selected


def _entry_by_date(entries: Iterable[CatalogEntry]) -> dict[str, CatalogEntry]:
    return {entry.session_date: entry for entry in entries}


def _sample_session(frame: pd.DataFrame, maximum_rows: int) -> pd.DataFrame:
    if len(frame) <= maximum_rows:
        return frame
    indices = np.linspace(0, len(frame) - 1, maximum_rows, dtype=int)
    return frame.iloc[np.unique(indices)]


def _load_model_frame(
    entries: Iterable[CatalogEntry],
    *,
    feature_columns: list[str],
    target_column: str,
    maximum_rows_per_session: int | None,
) -> pd.DataFrame:
    columns = ["session_date", "decision_time", *feature_columns, target_column]
    frames = []
    for entry in entries:
        frame = read_processed_table(entry.path, columns=columns)
        if maximum_rows_per_session is not None:
            frame = _sample_session(frame, maximum_rows_per_session)
        frames.append(frame)
    if not frames:
        raise ValueError("no session frames selected")
    return pd.concat(frames, ignore_index=True)


def _load_fit_pool(
    entries: Iterable[CatalogEntry],
    *,
    feature_columns: list[str],
    horizons: Iterable[int],
    maximum_rows_per_session: int,
) -> pd.DataFrame:
    """Read each daily file once and retain a deterministic bounded fit sample."""

    target_columns = [f"target_{horizon}ms_ticks" for horizon in horizons]
    columns = ["session_date", "decision_time", *feature_columns, *target_columns]
    frames = []
    for entry in entries:
        frame = read_processed_table(entry.path, columns=columns)
        frames.append(_sample_session(frame, maximum_rows_per_session))
    if not frames:
        raise ValueError("no session frames selected")
    return pd.concat(frames, ignore_index=True)


def run_train_stage(
    config: ResearchConfig,
    *,
    catalog_path: str | Path,
    output_dir: str | Path,
) -> dict[str, object]:
    """Use train sessions only for feature diagnostics and ridge-alpha selection."""

    entries = load_catalog(catalog_path)
    train_entries = _entries_for_split(entries, "train")
    feature_columns = model_feature_columns(config.features)
    folds = expanding_session_folds(
        [entry.session_date for entry in train_entries],
        minimum_train_sessions=config.research.minimum_train_sessions,
        folds=config.research.cv_folds,
    )
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    ic_rows: list[dict[str, object]] = []
    sampled_frames = []
    for entry in train_entries:
        columns = [
            "session_date",
            *feature_columns,
            *(f"target_{horizon}ms_ticks" for horizon in config.labels.horizons_ms),
        ]
        frame = read_processed_table(entry.path, columns=columns)
        sampled_frames.append(
            _sample_session(frame, config.research.max_fit_rows_per_session)
        )
        for feature in feature_columns:
            for horizon in config.labels.horizons_ms:
                target = f"target_{horizon}ms_ticks"
                value = spearman_correlation(frame[feature], frame[target])
                ic_rows.append(
                    {
                        "session_date": entry.session_date,
                        "feature": feature,
                        "horizon_ms": horizon,
                        "spearman_ic": value,
                        "rows": int(frame[[feature, target]].dropna().shape[0]),
                    }
                )
    daily_ic = pd.DataFrame(ic_rows)
    write_table(daily_ic, destination / "train_feature_daily_ic.csv")
    bootstrap_rows = []
    for (feature, horizon), group in daily_ic.groupby(["feature", "horizon_ms"], sort=True):
        finite = group["spearman_ic"].dropna()
        if len(finite) < 2:
            continue
        interval = cluster_bootstrap_mean(
            finite,
            repetitions=config.research.bootstrap_repetitions,
            seed=config.seed,
        )
        bootstrap_rows.append(
            {"feature": feature, "horizon_ms": horizon, **interval.to_dict()}
        )
    write_table(pd.DataFrame(bootstrap_rows), destination / "train_feature_ic_bootstrap.csv")

    fit_pool = pd.concat(sampled_frames, ignore_index=True)
    quantile_payload = {}
    decile_rows = []
    for feature in feature_columns:
        try:
            edges = fit_quantile_edges(fit_pool[feature], bins=10)
        except ValueError:
            continue
        quantile_payload[feature] = [float(value) for value in edges[1:-1]]
        for horizon in config.labels.horizons_ms:
            target = f"target_{horizon}ms_ticks"
            table = apply_quantile_analysis(
                fit_pool,
                signal_column=feature,
                target_column=target,
                edges=edges,
            )
            table.insert(0, "horizon_ms", horizon)
            table.insert(0, "feature", feature)
            decile_rows.append(table)
    write_table(
        pd.concat(decile_rows, ignore_index=True),
        destination / "train_feature_deciles.csv",
    )
    quantile_path = destination / "train_quantile_edges.json"
    write_json(
        quantile_path,
        {
            "stage": "train_only_quantile_edges",
            "catalog_sha256": sha256_file(catalog_path),
            "features": quantile_payload,
        },
    )
    cv_rows: list[dict[str, object]] = []
    for horizon in config.labels.horizons_ms:
        target = f"target_{horizon}ms_ticks"
        for fold_index, (train_dates, validation_dates) in enumerate(folds, start=1):
            fit_frame = fit_pool.loc[fit_pool["session_date"].isin(train_dates)]
            validation_frame = fit_pool.loc[
                fit_pool["session_date"].isin(validation_dates)
            ]
            for alpha in config.research.ridge_alphas:
                model = fit_ridge(
                    fit_frame,
                    feature_columns=feature_columns,
                    target_column=target,
                    alpha=alpha,
                )
                prediction, metrics = score_regression(
                    model,
                    validation_frame,
                    feature_columns=feature_columns,
                    target_column=target,
                )
                clean = validation_frame.dropna(subset=[*feature_columns, target]).copy()
                clean["prediction_ticks"] = prediction
                daily = daily_information_coefficient(
                    clean,
                    signal_column="prediction_ticks",
                    target_column=target,
                )
                cv_rows.append(
                    {
                        "horizon_ms": horizon,
                        "fold": fold_index,
                        "alpha": alpha,
                        "train_end": max(train_dates),
                        "validation_start": min(validation_dates),
                        "validation_end": max(validation_dates),
                        "validation_sessions": len(validation_dates),
                        "rows": metrics.rows,
                        "mae_ticks": metrics.mae_ticks,
                        "rmse_ticks": metrics.rmse_ticks,
                        "correlation": metrics.correlation,
                        "directional_accuracy_nonflat": metrics.directional_accuracy_nonflat,
                        "mean_daily_spearman_ic": float(daily["spearman_ic"].mean()),
                    }
                )
    cv = pd.DataFrame(cv_rows)
    write_table(cv, destination / "train_ridge_cv.csv")
    summary = (
        cv.groupby(["horizon_ms", "alpha"], as_index=False)
        .agg(
            mean_daily_spearman_ic=("mean_daily_spearman_ic", "mean"),
            mean_mae_ticks=("mae_ticks", "mean"),
            folds=("fold", "size"),
        )
        .sort_values(
            ["horizon_ms", "mean_daily_spearman_ic", "alpha"],
            ascending=[True, False, True],
        )
    )
    selected = {
        str(int(horizon)): float(group.iloc[0]["alpha"])
        for horizon, group in summary.groupby("horizon_ms", sort=True)
    }
    write_table(summary, destination / "train_ridge_cv_summary.csv")
    payload: dict[str, object] = {
        "stage": "train_only",
        "catalog_sha256": sha256_file(catalog_path),
        "config_sha256": sha256_file(config.source_path),
        "train_sessions": len(train_entries),
        "selected_alpha_by_horizon": selected,
        "quantile_edges_sha256": sha256_file(quantile_path),
    }
    write_json(destination / "train_selection.json", payload)
    return payload


def _read_selection(path: str | Path, *, expected_stage: str) -> dict[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("stage") != expected_stage:
        raise ValueError(f"expected {expected_stage!r} artifact, got {payload.get('stage')!r}")
    return payload


def _predict_split(
    config: ResearchConfig,
    *,
    fit_entries: list[CatalogEntry],
    evaluation_entries: list[CatalogEntry],
    alpha_by_horizon: dict[str, float],
    horizons: Iterable[int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_columns = model_feature_columns(config.features)
    prediction_frames = []
    metric_rows = []
    active_horizons = tuple(horizons) if horizons is not None else config.labels.horizons_ms
    fit_pool = _load_fit_pool(
        fit_entries,
        feature_columns=feature_columns,
        horizons=active_horizons,
        maximum_rows_per_session=config.research.max_fit_rows_per_session,
    )
    models = {}
    for horizon in active_horizons:
        target = f"target_{horizon}ms_ticks"
        model = fit_ridge(
            fit_pool,
            feature_columns=feature_columns,
            target_column=target,
            alpha=float(alpha_by_horizon[str(horizon)]),
        )
        baseline_models = {
            "single_microprice": (
                ["microprice_displacement_ticks"],
                fit_ridge(
                    fit_pool,
                    feature_columns=["microprice_displacement_ticks"],
                    target_column=target,
                    alpha=0.0,
                ),
            ),
            "single_queue_imbalance": (
                ["queue_imbalance_l1"],
                fit_ridge(
                    fit_pool,
                    feature_columns=["queue_imbalance_l1"],
                    target_column=target,
                    alpha=0.0,
                ),
            ),
        }
        models[horizon] = (model, baseline_models)

    evaluation_columns = [
        "session_date",
        "decision_time",
        *feature_columns,
        *(f"target_{horizon}ms_ticks" for horizon in active_horizons),
    ]
    for entry in evaluation_entries:
        frame = read_processed_table(entry.path, columns=evaluation_columns)
        for horizon in active_horizons:
            target = f"target_{horizon}ms_ticks"
            model, baseline_models = models[horizon]
            clean = frame.dropna(subset=[*feature_columns, target]).copy()
            predicted, metrics = score_regression(
                model,
                clean,
                feature_columns=feature_columns,
                target_column=target,
            )
            prediction_column = f"prediction_{horizon}ms_ticks"
            signals = clean[["session_date", "decision_time", target]].copy()
            signals[prediction_column] = predicted
            signals["horizon_ms"] = horizon
            prediction_frames.append(signals)
            daily_ic = spearman_correlation(signals[prediction_column], signals[target])
            metric_rows.append(
                {
                    "session_date": entry.session_date,
                    "horizon_ms": horizon,
                    "model": "ridge_all_features",
                    "alpha": float(alpha_by_horizon[str(horizon)]),
                    **asdict(metrics),
                    "spearman_ic": daily_ic,
                }
            )
            actual = clean[target].to_numpy(dtype=float)
            zero_metrics = regression_metrics(actual, np.zeros_like(actual))
            metric_rows.append(
                {
                    "session_date": entry.session_date,
                    "horizon_ms": horizon,
                    "model": "zero_forecast",
                    "alpha": np.nan,
                    **asdict(zero_metrics),
                    "spearman_ic": float("nan"),
                }
            )
            for baseline_name, (baseline_features, baseline_model) in baseline_models.items():
                baseline_prediction, baseline_metrics = score_regression(
                    baseline_model,
                    clean,
                    feature_columns=baseline_features,
                    target_column=target,
                )
                metric_rows.append(
                    {
                        "session_date": entry.session_date,
                        "horizon_ms": horizon,
                        "model": baseline_name,
                        "alpha": 0.0,
                        **asdict(baseline_metrics),
                        "spearman_ic": spearman_correlation(
                            pd.Series(baseline_prediction), pd.Series(actual)
                        ),
                    }
                )
    return pd.concat(prediction_frames, ignore_index=True), pd.DataFrame(metric_rows)


def _raw_paths_by_date(raw_dir: str | Path) -> dict[str, Path]:
    return {value.isoformat(): path for value, path in discover_daily_raw_files(raw_dir).items()}


def _backtest_scenarios(
    config: ResearchConfig,
    predictions: pd.DataFrame,
    *,
    raw_paths: dict[str, Path],
    horizon_ms: int,
    scenarios: dict[str, tuple[int, float, int, float]],
) -> dict[str, pd.DataFrame]:
    """Load each raw session once, then evaluate all requested execution scenarios."""

    prediction_column = f"prediction_{horizon_ms}ms_ticks"
    selected = (
        predictions.loc[predictions["horizon_ms"].eq(horizon_ms)]
        if "horizon_ms" in predictions
        else predictions
    )
    unique_parameters = tuple(dict.fromkeys(scenarios.values()))
    buffers: dict[tuple[int, float, int, float], list[pd.DataFrame]] = {
        parameters: [] for parameters in unique_parameters
    }
    for session_date, signals in selected.groupby("session_date", sort=True):
        raw_path = raw_paths.get(str(session_date))
        if raw_path is None:
            raise FileNotFoundError(f"no dated raw file for session {session_date}")
        events = load_events(raw_path)
        for parameters in unique_parameters:
            latency_ms, fee_usd, quantity, safety_margin_ticks = parameters
            session_trades = simulate_marketable_strategy(
                events,
                signals,
                prediction_column=prediction_column,
                horizon_ms=horizon_ms,
                latency_ms=latency_ms,
                quantity=quantity,
                tick_size=config.contract.expected_tick_size,
                multiplier=config.contract.expected_multiplier,
                fee_per_contract_per_side_usd=fee_usd,
                safety_margin_ticks=safety_margin_ticks,
                maximum_quote_age_ms=config.session.maximum_quote_age_ms,
            )
            session_trades.insert(0, "session_date", str(session_date))
            buffers[parameters].append(session_trades)
    results = {}
    for name, parameters in scenarios.items():
        session_frames = buffers[parameters]
        results[name] = (
            pd.concat(session_frames, ignore_index=True) if session_frames else pd.DataFrame()
        )
    return results


def run_validation_stage(
    config: ResearchConfig,
    *,
    catalog_path: str | Path,
    train_selection_path: str | Path,
    raw_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, object]:
    """Select the horizon and execution margin on validation, never on holdout."""

    selection = _read_selection(train_selection_path, expected_stage="train_only")
    if selection["catalog_sha256"] != sha256_file(catalog_path):
        raise OSError("catalog changed after train selection")
    if selection["config_sha256"] != sha256_file(config.source_path):
        raise OSError("configuration changed after train selection")
    entries = load_catalog(catalog_path)
    train_entries = _entries_for_split(entries, "train")
    validation_entries = _entries_for_split(entries, "validation")
    quantile_path = Path(train_selection_path).with_name("train_quantile_edges.json")
    if sha256_file(quantile_path) != selection.get("quantile_edges_sha256"):
        raise OSError("train-fitted quantile edges changed before validation")
    quantile_artifact = json.loads(quantile_path.read_text(encoding="utf-8"))
    predictions, metrics = _predict_split(
        config,
        fit_entries=train_entries,
        evaluation_entries=validation_entries,
        alpha_by_horizon={
            str(key): float(value)
            for key, value in dict(selection["selected_alpha_by_horizon"]).items()
        },
    )
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    write_table(metrics, destination / "validation_regression_daily.csv")
    aggregate = (
        metrics.groupby(["horizon_ms", "model"], as_index=False)
        .agg(
            sessions=("session_date", "size"),
            rows=("rows", "sum"),
            mean_daily_spearman_ic=("spearman_ic", "mean"),
            mean_mae_ticks=("mae_ticks", "mean"),
            mean_directional_accuracy=("directional_accuracy_nonflat", "mean"),
        )
        .sort_values(["mean_daily_spearman_ic", "horizon_ms"], ascending=[False, True])
    )
    write_table(aggregate, destination / "validation_regression_summary.csv")
    feature_columns = model_feature_columns(config.features)
    validation_deciles = []
    for entry in validation_entries:
        frame = read_processed_table(
            entry.path,
            columns=[
                "session_date",
                *feature_columns,
                *(f"target_{horizon}ms_ticks" for horizon in config.labels.horizons_ms),
            ],
        )
        for feature in quantile_artifact["features"]:
            inner = [float(value) for value in quantile_artifact["features"][feature]]
            edges = np.asarray([-np.inf, *inner, np.inf])
            for horizon in config.labels.horizons_ms:
                table = apply_quantile_analysis(
                    frame,
                    signal_column=feature,
                    target_column=f"target_{horizon}ms_ticks",
                    edges=edges,
                )
                table.insert(0, "session_date", entry.session_date)
                table.insert(1, "feature", feature)
                table.insert(2, "horizon_ms", horizon)
                validation_deciles.append(table)
    validation_decile_daily = pd.concat(validation_deciles, ignore_index=True)
    write_table(
        validation_decile_daily,
        destination / "validation_feature_deciles_daily.csv",
    )
    validation_decile_daily["weighted_target_sum"] = (
        validation_decile_daily["rows"] * validation_decile_daily["mean_target_ticks"]
    )
    validation_decile_summary = (
        validation_decile_daily.groupby(
            ["feature", "horizon_ms", "signal_bin"], as_index=False
        )
        .agg(
            rows=("rows", "sum"),
            sessions=("session_date", "nunique"),
            weighted_target_sum=("weighted_target_sum", "sum"),
        )
    )
    validation_decile_summary["mean_target_ticks"] = (
        validation_decile_summary["weighted_target_sum"]
        / validation_decile_summary["rows"]
    )
    validation_decile_summary = validation_decile_summary.drop(
        columns="weighted_target_sum"
    )
    write_table(
        validation_decile_summary,
        destination / "validation_feature_deciles_summary.csv",
    )
    ridge_aggregate = aggregate.loc[aggregate["model"].eq("ridge_all_features")]
    selected_horizon = int(ridge_aggregate.iloc[0]["horizon_ms"])
    selected_predictions = predictions.loc[
        predictions["horizon_ms"].eq(selected_horizon),
        [
            "session_date",
            "decision_time",
            f"target_{selected_horizon}ms_ticks",
            f"prediction_{selected_horizon}ms_ticks",
            "horizon_ms",
        ],
    ]
    write_table(selected_predictions, destination / "validation_selected_predictions.csv.gz")

    raw_paths = _raw_paths_by_date(raw_dir)
    scenarios = {
        str(margin): (
            config.execution.primary_latency_ms,
            config.execution.primary_fee_per_contract_per_side_usd,
            config.execution.primary_quantity,
            margin,
        )
        for margin in config.research.safety_margin_grid_ticks
    }
    scenario_trades = _backtest_scenarios(
        config,
        selected_predictions,
        raw_paths=raw_paths,
        horizon_ms=selected_horizon,
        scenarios=scenarios,
    )
    threshold_rows = []
    threshold_trades: dict[float, pd.DataFrame] = {}
    for margin in config.research.safety_margin_grid_ticks:
        trades = scenario_trades[str(margin)]
        threshold_trades[margin] = trades
        threshold_rows.append({"safety_margin_ticks": margin, **summarize_trades(trades)})
    threshold_summary = pd.DataFrame(threshold_rows).sort_values(
        ["net_pnl_usd", "safety_margin_ticks"], ascending=[False, True]
    )
    write_table(threshold_summary, destination / "validation_threshold_summary.csv")
    viable = threshold_summary.loc[threshold_summary["trades"] > 0]
    if viable.empty:
        selected_margin = config.research.primary_safety_margin_ticks
        execution_status = "no_validation_trades_at_registered_thresholds"
    else:
        selected_margin = float(viable.iloc[0]["safety_margin_ticks"])
        execution_status = "selected_by_validation_net_pnl"
        write_table(
            threshold_trades[selected_margin],
            destination / "validation_selected_trades.csv.gz",
        )
    payload: dict[str, object] = {
        "stage": "validation_selected",
        "catalog_sha256": sha256_file(catalog_path),
        "config_sha256": sha256_file(config.source_path),
        "train_selection_sha256": sha256_file(train_selection_path),
        "selected_horizon_ms": selected_horizon,
        "selected_alpha": float(
            dict(selection["selected_alpha_by_horizon"])[str(selected_horizon)]
        ),
        "selected_safety_margin_ticks": selected_margin,
        "execution_selection_status": execution_status,
        "validation_sessions": len(validation_entries),
    }
    write_json(destination / "selected_candidate.json", payload)
    return payload


def freeze_candidate(
    config: ResearchConfig,
    *,
    catalog_path: str | Path,
    candidate_path: str | Path,
    output_path: str | Path,
) -> dict[str, object]:
    """Create the immutable handoff required before any holdout access."""

    output = Path(output_path)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite frozen candidate: {output}")
    candidate = _read_selection(candidate_path, expected_stage="validation_selected")
    if candidate["catalog_sha256"] != sha256_file(catalog_path):
        raise OSError("catalog changed before freeze")
    if candidate["config_sha256"] != sha256_file(config.source_path):
        raise OSError("configuration changed before freeze")
    payload = {
        **candidate,
        "stage": "frozen_for_holdout",
        "candidate_sha256": sha256_file(candidate_path),
        "catalog_path": str(Path(catalog_path).resolve()),
        "config_path": str(config.source_path),
    }
    write_json(output, payload)
    return payload


def run_holdout_stage(
    config: ResearchConfig,
    *,
    catalog_path: str | Path,
    frozen_candidate_path: str | Path,
    raw_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, object]:
    """Evaluate the frozen candidate once; refuse overwrite or changed inputs."""

    destination = Path(output_dir)
    completion_marker = destination / "HOLDOUT_COMPLETE.json"
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"refusing to overwrite existing holdout output: {destination}")
    frozen = _read_selection(frozen_candidate_path, expected_stage="frozen_for_holdout")
    if frozen["catalog_sha256"] != sha256_file(catalog_path):
        raise OSError("catalog changed after candidate freeze")
    if frozen["config_sha256"] != sha256_file(config.source_path):
        raise OSError("configuration changed after candidate freeze")
    entries = load_catalog(catalog_path)
    development = [entry for entry in entries if entry.split in {"train", "validation"}]
    holdout = _entries_for_split(entries, "holdout")
    horizon = int(frozen["selected_horizon_ms"])
    alpha_by_horizon = {str(horizon): float(frozen["selected_alpha"])}
    predictions, metrics = _predict_split(
        config,
        fit_entries=development,
        evaluation_entries=holdout,
        alpha_by_horizon=alpha_by_horizon,
        horizons=(horizon,),
    )
    selected = predictions.loc[
        predictions[f"prediction_{horizon}ms_ticks"].notna(),
        [
            "session_date",
            "decision_time",
            f"target_{horizon}ms_ticks",
            f"prediction_{horizon}ms_ticks",
        ],
    ].copy()
    destination.mkdir(parents=True, exist_ok=True)
    write_table(selected, destination / "holdout_predictions.csv.gz")
    selected_metrics = metrics.loc[
        metrics["horizon_ms"].eq(horizon) & metrics["model"].eq("ridge_all_features")
    ].copy()
    write_table(selected_metrics, destination / "holdout_regression_daily.csv")
    raw_paths = _raw_paths_by_date(raw_dir)
    margin = float(frozen["selected_safety_margin_ticks"])
    scenario_parameters: dict[str, tuple[int, float, int, float]] = {
        "primary": (
            config.execution.primary_latency_ms,
            config.execution.primary_fee_per_contract_per_side_usd,
            config.execution.primary_quantity,
            margin,
        )
    }
    scenario_rows = []
    for value in config.execution.latency_grid_ms:
        key = f"latency_ms:{value}"
        scenario_parameters[key] = (
            int(value),
            config.execution.primary_fee_per_contract_per_side_usd,
            config.execution.primary_quantity,
            margin,
        )
        scenario_rows.append(("latency_ms", value, key))
    for value in config.execution.fee_grid_per_contract_per_side_usd:
        key = f"fee_per_side_usd:{value}"
        scenario_parameters[key] = (
            config.execution.primary_latency_ms,
            float(value),
            config.execution.primary_quantity,
            margin,
        )
        scenario_rows.append(("fee_per_side_usd", value, key))
    for value in config.execution.quantity_grid:
        key = f"quantity:{value}"
        scenario_parameters[key] = (
            config.execution.primary_latency_ms,
            config.execution.primary_fee_per_contract_per_side_usd,
            int(value),
            margin,
        )
        scenario_rows.append(("quantity", value, key))
    scenario_trades = _backtest_scenarios(
        config,
        selected,
        raw_paths=raw_paths,
        horizon_ms=horizon,
        scenarios=scenario_parameters,
    )
    primary_trades = scenario_trades["primary"]
    write_table(primary_trades, destination / "holdout_primary_trades.csv.gz")
    sensitivity_rows = [
        {"grid": grid, "value": value, **summarize_trades(scenario_trades[key])}
        for grid, value, key in scenario_rows
    ]
    write_table(pd.DataFrame(sensitivity_rows), destination / "holdout_sensitivity.csv")
    finite_ic = selected_metrics["spearman_ic"].dropna()
    interval = (
        cluster_bootstrap_mean(
            finite_ic,
            repetitions=config.research.bootstrap_repetitions,
            seed=config.seed,
        ).to_dict()
        if len(finite_ic) >= 2
        else None
    )
    payload: dict[str, object] = {
        "stage": "holdout_complete",
        "frozen_candidate_sha256": sha256_file(frozen_candidate_path),
        "holdout_sessions": len(holdout),
        "horizon_ms": horizon,
        "alpha": float(frozen["selected_alpha"]),
        "safety_margin_ticks": margin,
        "regression": {
            "mean_daily_spearman_ic": float(selected_metrics["spearman_ic"].mean()),
            "mean_mae_ticks": float(selected_metrics["mae_ticks"].mean()),
            "ic_cluster_bootstrap": interval,
        },
        "execution": summarize_trades(primary_trades),
    }
    write_json(completion_marker, payload)
    return payload
