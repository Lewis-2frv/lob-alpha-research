"""Claim-gated equity closing-auction report and diagnostic figures."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import pandas as pd

from .manifest import sha256_file

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

_TRAIN_ARTIFACT_NAMES = {
    "train_baseline_cv.csv",
    "train_feature_daily_ic.csv",
    "train_model_cv.csv",
    "train_model_cv_summary.csv",
}
_VALIDATION_ARTIFACT_NAMES = {
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


def _read_json(path: Path) -> dict[str, object] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _fmt(value: object, digits: int = 4) -> str:
    if value is None or pd.isna(value):
        return "not available"
    return f"{float(value):.{digits}f}"


def _require_hash(path: object, expected: object, label: str) -> Path:
    artifact = Path(str(path)).resolve()
    if not artifact.is_file() or sha256_file(artifact) != expected:
        raise OSError(f"{label} is missing or its SHA-256 does not match")
    return artifact


def _validate_inventory(
    directory: Path,
    hashes: object,
    expected_names: set[str],
    label: str,
) -> None:
    if not isinstance(hashes, dict) or set(hashes) != expected_names:
        raise OSError(f"{label} has an incomplete artifact hash inventory")
    for name, expected in hashes.items():
        _require_hash(directory / name, expected, f"{label} artifact {name}")


def _validate_real_development(
    candidate: dict[str, object], candidate_path: Path
) -> None:
    if candidate.get("stage") != "equity_validation_selected":
        raise ValueError("invalid validation candidate stage")
    if candidate.get("target_definition") != "supplied_optiver_index_relative_60s_bps":
        raise ValueError("real validation candidate has an invalid target definition")
    resolved: dict[str, Path] = {}
    for path_key, hash_key in (
        ("config_path", "config_sha256"),
        ("prepared_manifest_path", "prepared_manifest_sha256"),
        ("raw_metadata_path", "raw_metadata_sha256"),
        ("raw_path", "raw_sha256"),
        ("train_selection_path", "train_selection_sha256"),
        ("model_path", "model_sha256"),
        ("preprocessing_parameters_path", "preprocessing_parameters_sha256"),
    ):
        resolved[path_key] = _require_hash(
            candidate.get(path_key), candidate.get(hash_key), path_key
        )
    selection = json.loads(resolved["train_selection_path"].read_text(encoding="utf-8"))
    _validate_inventory(
        resolved["train_selection_path"].parent,
        selection.get("train_artifact_sha256"),
        _TRAIN_ARTIFACT_NAMES,
        "train selection",
    )
    _validate_inventory(
        Path(str(candidate.get("validation_artifact_directory", candidate_path.parent))).resolve(),
        candidate.get("validation_artifact_sha256"),
        _VALIDATION_ARTIFACT_NAMES,
        "validation candidate",
    )
    metadata = json.loads(Path(str(candidate["raw_metadata_path"])).read_text(encoding="utf-8"))
    if metadata.get("source_kind") != "real":
        raise ValueError("a real-data report requires raw metadata classified as real")


def _validate_real_holdout(
    holdout: Path,
    result_path: Path,
    result: dict[str, object],
) -> None:
    if result.get("stage") != "equity_holdout_complete":
        raise ValueError("invalid holdout completion stage")
    if (
        result.get("source_kind") != "real"
        or result.get("selection_performed") is not False
        or result.get("claim_eligible_real_optiver") is not True
    ):
        raise ValueError("real holdout evidence has an invalid source or selection flag")
    if result.get("target_definition") != "supplied_optiver_index_relative_60s_bps":
        raise ValueError("real holdout target definition is not the registered supplied target")
    frozen_path = _require_hash(
        result.get("frozen_candidate_path"),
        result.get("frozen_candidate_sha256"),
        "frozen candidate",
    )
    seal_path = _require_hash(
        result.get("holdout_seal_path"),
        result.get("holdout_seal_sha256"),
        "holdout seal",
    )
    if seal_path.name != "HOLDOUT_STARTED.json":
        raise ValueError("holdout seal must use the stable prepared-data filename")
    anchor_path = seal_path.with_name("HOLDOUT_COMPLETE_ANCHOR.json")
    anchor = json.loads(anchor_path.read_text(encoding="utf-8"))
    if anchor.get("stage") != "equity_holdout_complete_anchor":
        raise ValueError("holdout completion anchor is invalid")
    if Path(str(anchor.get("holdout_complete_path"))).resolve() != result_path.resolve():
        raise OSError("holdout completion anchor points to a different result")
    if anchor.get("holdout_complete_sha256") != sha256_file(result_path):
        raise OSError("holdout completion JSON changed after anchoring")
    for key in (
        "holdout_seal_sha256",
        "frozen_candidate_sha256",
        "prepared_manifest_sha256",
    ):
        if anchor.get(key) != result.get(key):
            raise OSError(f"holdout completion anchor disagrees on {key}")
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    if seal.get("stage") != "equity_holdout_started_one_shot":
        raise ValueError("holdout seal stage is invalid")
    if seal.get("frozen_candidate_sha256") != result.get("frozen_candidate_sha256"):
        raise OSError("holdout seal and completion use different frozen candidates")
    if seal.get("prepared_manifest_sha256") != result.get("prepared_manifest_sha256"):
        raise OSError("holdout seal and completion use different prepared manifests")
    if Path(str(seal.get("holdout_output_path"))).resolve() != holdout.resolve():
        raise OSError("holdout seal points to a different output directory")
    _require_hash(
        result.get("prepared_manifest_path"),
        result.get("prepared_manifest_sha256"),
        "prepared manifest",
    )
    _require_hash(
        result.get("holdout_manifest_path"),
        result.get("holdout_manifest_sha256"),
        "holdout partition manifest",
    )
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    if frozen.get("stage") != "equity_frozen_for_holdout":
        raise ValueError("frozen candidate stage is invalid")
    if (
        frozen.get("source_kind") != "real"
        or frozen.get("target_definition") != "supplied_optiver_index_relative_60s_bps"
    ):
        raise ValueError("real holdout evidence requires a real frozen target contract")
    candidate_path = _require_hash(
        frozen.get("candidate_path"), frozen.get("candidate_sha256"), "validation candidate"
    )
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    _validate_inventory(
        Path(str(candidate.get("validation_artifact_directory", candidate_path.parent))).resolve(),
        candidate.get("validation_artifact_sha256"),
        _VALIDATION_ARTIFACT_NAMES,
        "validation candidate",
    )
    for key, value in candidate.items():
        if key != "stage" and frozen.get(key) != value:
            raise OSError(f"frozen candidate field differs from validation candidate: {key}")
    for path_key, hash_key in (
        ("config_path", "config_sha256"),
        ("raw_metadata_path", "raw_metadata_sha256"),
        ("raw_path", "raw_sha256"),
        ("train_selection_path", "train_selection_sha256"),
        ("model_path", "model_sha256"),
        ("preprocessing_parameters_path", "preprocessing_parameters_sha256"),
    ):
        _require_hash(frozen.get(path_key), frozen.get(hash_key), path_key)
        if result.get(hash_key) != frozen.get(hash_key):
            raise OSError(f"holdout result disagrees with frozen {hash_key}")
    train_selection = json.loads(
        Path(str(frozen["train_selection_path"])).read_text(encoding="utf-8")
    )
    _validate_inventory(
        Path(str(frozen["train_selection_path"])).resolve().parent,
        train_selection.get("train_artifact_sha256"),
        _TRAIN_ARTIFACT_NAMES,
        "train selection",
    )
    raw_metadata = json.loads(
        Path(str(frozen["raw_metadata_path"])).read_text(encoding="utf-8")
    )
    if raw_metadata.get("source_kind") != "real":
        raise ValueError("real holdout evidence requires raw metadata classified as real")
    for hash_key in (
        "feature_specification_sha256",
        "feature_implementation_sha256",
        "research_implementation_sha256",
    ):
        if result.get(hash_key) != frozen.get(hash_key):
            raise OSError(f"holdout result disagrees with frozen {hash_key}")
    artifact_hashes = dict(result.get("holdout_artifact_sha256", {}))
    if not artifact_hashes:
        raise ValueError("holdout completion has no artifact hash inventory")
    for name, expected in artifact_hashes.items():
        if Path(name).name != name:
            raise ValueError("holdout artifact inventory contains a non-local path")
        _require_hash(holdout / name, expected, f"holdout artifact {name}")
    cost = dict(result.get("cost_assumption", {}))
    frozen_fee = float(dict(frozen["trading_rule"])["fee_per_side_bps"])
    if (
        float(cost.get("fee_per_side_bps", -1)) != frozen_fee
        or float(cost.get("round_trip_fee_bps", -1)) != 2.0 * frozen_fee
        or cost.get("market_impact_modeled") is not False
        or cost.get("displayed_liquidity_guarantees_fill") is not False
    ):
        raise ValueError("holdout cost assumptions do not match the frozen execution rule")


def _plot_daily_ic(source: Path, destination: Path) -> Path | None:
    if not source.exists():
        return None
    frame = pd.read_csv(source)
    if frame.empty:
        return None
    if "model" in frame:
        selected = frame.loc[frame["model"].eq("frozen_candidate")]
        if selected.empty:
            selected = frame.iloc[0:0]
            for _, group in frame.groupby("date_id", sort=True):
                selected = pd.concat([selected, group.tail(1)], ignore_index=True)
        frame = selected
    figure, axis = plt.subplots(figsize=(7.2, 3.8))
    axis.plot(frame["date_id"], frame["spearman_ic"], marker="o", linewidth=1)
    axis.axhline(0.0, color="#666666", linewidth=0.8)
    axis.set(title="Daily cross-sectional Spearman IC", xlabel="date_id", ylabel="IC")
    axis.grid(alpha=0.2)
    figure.tight_layout()
    output = destination / "equity_daily_ic.png"
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return output


def _plot_table_line(
    source: Path,
    destination: Path,
    *,
    x: str,
    y: str,
    filename: str,
    title: str,
    xlabel: str,
    ylabel: str,
) -> Path | None:
    if not source.exists():
        return None
    frame = pd.read_csv(source)
    if frame.empty or x not in frame or y not in frame:
        return None
    figure, axis = plt.subplots(figsize=(6.5, 3.8))
    axis.plot(frame[x], frame[y], marker="o")
    axis.axhline(0.0, color="#666666", linewidth=0.8)
    axis.set(title=title, xlabel=xlabel, ylabel=ylabel)
    axis.grid(alpha=0.2)
    figure.tight_layout()
    output = destination / filename
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return output


def _plot_feature_ablation(source: Path, destination: Path) -> Path | None:
    if not source.exists():
        return None
    frame = pd.read_csv(source)
    if frame.empty or not {"omitted_family", "mae_bps"} <= set(frame.columns):
        return None
    figure, axis = plt.subplots(figsize=(7.0, 3.8))
    axis.bar(frame["omitted_family"], frame["mae_bps"], color="#3f6f8f")
    axis.set(
        title="Validation feature-family ablation",
        xlabel="Omitted feature family",
        ylabel="MAE (bps; lower is better)",
    )
    axis.tick_params(axis="x", rotation=25)
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    output = destination / "equity_feature_ablation.png"
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return output


def _markdown_table(path: Path, columns: list[str]) -> list[str]:
    if not path.exists():
        return ["Not available.", ""]
    frame = pd.read_csv(path)
    available = [column for column in columns if column in frame]
    if frame.empty or not available:
        return ["Not available.", ""]
    lines = [
        "| " + " | ".join(available) + " |",
        "| " + " | ".join("---" for _ in available) + " |",
    ]
    for row in frame.loc[:, available].itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    lines.append("")
    return lines


def build_equity_report(
    *,
    train_dir: str | Path,
    validation_dir: str | Path,
    holdout_dir: str | Path,
    reports_dir: str | Path,
) -> tuple[Path, Path]:
    """Generate research evidence while excluding all synthetic performance claims."""

    train = Path(train_dir)
    validation = Path(validation_dir)
    holdout = Path(holdout_dir)
    destination = Path(reports_dir)
    figures = destination / "figures"
    destination.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    train_selection = _read_json(train / "train_selection.json")
    candidate = _read_json(validation / "selected_candidate.json")
    result_path = holdout / "HOLDOUT_COMPLETE.json"
    result = _read_json(result_path)
    source_kind = str((result or candidate or train_selection or {}).get("source_kind", "unknown"))
    claim_eligible = bool(result and result.get("claim_eligible_real_optiver"))
    if candidate and candidate.get("source_kind") == "real":
        _validate_real_development(candidate, validation / "selected_candidate.json")
    if result and claim_eligible:
        _validate_real_holdout(holdout, result_path, result)
    synthetic = source_kind != "real" or not claim_eligible
    metric_dir = holdout if claim_eligible else validation
    metric_prefix = "holdout" if claim_eligible else "validation"
    if source_kind == "real":
        _plot_daily_ic(metric_dir / f"{metric_prefix}_daily_metrics.csv", figures)
        _plot_table_line(
            metric_dir / f"{metric_prefix}_prediction_deciles.csv",
            figures,
            x="prediction_decile",
            y="mean_target_bps",
            filename="equity_prediction_deciles.png",
            title="Prediction-decile supplied-target spread",
            xlabel="Prediction decile",
            ylabel="Mean supplied target (bps)",
        )
        _plot_table_line(
            metric_dir / f"{metric_prefix}_fee_sensitivity.csv",
            figures,
            x="fee_per_side_bps",
            y="net_mean_bps",
            filename="equity_trading_cost_frontier.png",
            title="Executable quote return versus per-side fees",
            xlabel="Fee per side (bps)",
            ylabel="Mean net portfolio return (bps)",
        )
        _plot_feature_ablation(validation / "validation_feature_ablation.csv", figures)

    lines = [
        "# Equity closing-auction alpha study",
        "",
        "> Generated from content-addressed artifacts. Synthetic fixture values are never "
        "market evidence.",
        "",
        "## Research question",
        "",
        "Can contemporaneous order-book liquidity, order imbalance and closing-auction state "
        "predict the supplied 60-second, synthetic-index-relative Optiver target? Separately, "
        "can those predictions rank quote-based stock returns sufficiently well to support a "
        "spread- and fee-aware cross-sectional long/short simulation?",
        "",
        "## Dataset and registered split",
        "",
        (
            "The licensed Optiver - Trading at the Close training panel is sampled every ten "
            "seconds. "
            if source_kind == "real"
            else "The generated Optiver-shaped engineering panel is sampled every ten seconds. "
        ),
        "Complete date_id groups are assigned chronologically; no random row split is used.",
        "",
        f"- Train stage: {'complete' if train_selection else 'not run'}",
        f"- Validation selection/freeze input: {'complete' if candidate else 'not run'}",
        f"- One-shot holdout: {'complete' if result else 'not released'}",
        f"- Artifact source classification: `{source_kind}`",
        "",
        "## Causal feature definitions",
        "",
        "Features cover current quoted spread, midpoint/WAP pressure, displayed liquidity, signed "
        "auction imbalance, matched volume, auction-price dislocations, explicit near/far "
        "missingness, time-to-close interactions, within-stock/date lags and rolling statistics, "
        "and current-time_id cross-sectional ranks/robust z-scores. Stock identity uses sparse "
        "one-hot or native categorical encoding, never an ordinal magnitude. The supplied target "
        "and future quotes are "
        "excluded from model inputs. Predictive MAE/IC use the supplied index-relative target; "
        "execution P&L is independently calculated from current and exact +60-second quotes.",
        "",
        "## Baselines and model selection",
        "",
        "Zero and train-fitted signed-imbalance baselines are mandatory. Ridge and registered "
        "nonlinear tabular candidates are compared by expanding-window train-only MAE; validation "
        "chooses one model and one conservative execution rule.",
        "",
    ]
    if source_kind == "real":
        lines.extend(
            _markdown_table(
                validation / "validation_predictive_metrics.csv",
                [
                    "model",
                    "evaluation_scope",
                    "rows",
                    "mae_bps",
                    "spearman_ic",
                    "directional_accuracy",
                ],
            )
        )
    else:
        lines.extend(("Synthetic validation metrics are suppressed.", ""))
    if candidate and source_kind == "real":
        lines.extend(
            (
                "Selected development model: "
                f"`{json.dumps(candidate['model_specification'], sort_keys=True)}`.",
                "",
                "Selected validation execution rule: "
                f"`{json.dumps(candidate['trading_rule'], sort_keys=True)}`.",
                "",
                f"Validation coverage: {candidate['validation_full_rows']} full rows with "
                f"{candidate['validation_full_target_rows']} finite supplied targets; the "
                f"target-blind selection sample retained "
                f"{candidate['validation_selection_sample_rows']} rows and "
                f"{candidate['validation_selection_sample_target_rows']} finite targets.",
                "",
            )
        )
    lines.extend(
        (
            "## Validation results",
            "",
            "Validation is used for selection and is not presented as untouched final evidence. "
            "Directional accuracy is descriptive and does not establish economic significance.",
            "",
            (
                "![Daily IC](figures/equity_daily_ic.png)"
                if source_kind == "real"
                else "Synthetic daily-IC values and figures are suppressed."
            ),
            "",
            (
                "![Prediction deciles](figures/equity_prediction_deciles.png)"
                if source_kind == "real"
                else "Synthetic prediction-decile values and figures are suppressed."
            ),
            "",
            "## Feature ablation",
            "",
        )
    )
    if source_kind == "real":
        lines.extend(
            _markdown_table(
                validation / "validation_feature_ablation.csv",
                ["omitted_family", "mae_bps", "spearman_ic", "directional_accuracy"],
            )
        )
    else:
        lines.extend(("Synthetic ablation values are suppressed.", ""))
    if source_kind == "real":
        lines.extend(("![Feature-family ablation](figures/equity_feature_ablation.png)", ""))
    lines.extend(("## Holdout results", ""))
    if result and not synthetic:
        predictive = dict(result["predictive_metrics"])
        trading = dict(result["trading_summary"])
        diagnostics = dict(result["execution_filter_diagnostics"])
        cost = dict(result["cost_assumption"])
        predictive_interval = dict(result["daily_ic_cluster_bootstrap"])
        trading_interval = dict(trading["daily_cluster_bootstrap"])
        lines.extend(
            (
                f"The frozen model achieved MAE {_fmt(predictive['mae_bps'])} bps and overall "
                f"Spearman IC {_fmt(predictive['spearman_ic'])} on the untouched holdout.",
                "",
                f"The quote-crossing simulation produced {trading['decisions']} non-overlapping "
                f"decisions and {trading['stock_legs']} stock legs. Mean gross executable return "
                f"was {_fmt(trading['gross_mean_bps'])} bps and mean net return was "
                f"{_fmt(trading['net_mean_bps'])} bps after a registered "
                f"{_fmt(cost['fee_per_side_bps'])} bps fee per side "
                f"({_fmt(cost['round_trip_fee_bps'])} bps round trip).",
                "",
                f"Long and short sleeve mean net returns were "
                f"{_fmt(trading['long_net_mean_bps'])} and "
                f"{_fmt(trading['short_net_mean_bps'])} bps. Median portfolio return was "
                f"{_fmt(trading['net_median_bps'])} bps and hit rate was "
                f"{_fmt(trading['hit_rate'])}.",
                "",
                f"Mean spread cost was {_fmt(trading['spread_cost_mean_bps'])} bps; mean fees "
                f"were {_fmt(trading['fee_mean_bps'])} bps; round-trip gross turnover was "
                f"{_fmt(trading['turnover_round_trip_gross'], 2)}; minimum displayed capacity "
                f"was {_fmt(trading['minimum_displayed_capacity_units'], 2)} normalized units; "
                "and break-even additional total trading cost was "
                f"{_fmt(trading['break_even_additional_total_cost_bps'])} bps.",
                "",
                "Date-cluster bootstrap intervals were "
                f"[{_fmt(predictive_interval['ci_low'])}, "
                f"{_fmt(predictive_interval['ci_high'])}] for mean daily IC and "
                f"[{_fmt(trading_interval['ci_low'])}, "
                f"{_fmt(trading_interval['ci_high'])}] bps for mean daily net return.",
                "",
                f"Coverage: {result['holdout_rows']} holdout rows, including "
                f"{result['holdout_target_rows']} finite supplied targets and "
                f"{result['holdout_missing_target_rows']} missing supplied targets; "
                f"{len(result['holdout_date_ids'])} dates from "
                f"{min(result['holdout_date_ids'])} through "
                f"{max(result['holdout_date_ids'])}; "
                f"{diagnostics.get('decision_grid_rows', 0)} decision-grid rows; "
                f"{diagnostics.get('missing_exact_future_quote_rows', 0)} rejected for missing "
                "exact-horizon quotes; "
                f"{diagnostics.get('invalid_quote_rows', 0)} rejected for invalid quotes; "
                f"{diagnostics.get('spread_rejections', 0)} spread rejections; and "
                f"{diagnostics.get('liquidity_rejections', 0)} displayed-liquidity rejections.",
                "",
                f"The execution rule also skipped "
                f"{diagnostics.get('one_sided_time_ids', 0)} one-sided timestamps and "
                f"{diagnostics.get('insufficient_cross_section_time_ids', 0)} timestamps with "
                "too few eligible stocks.",
                "",
            )
        )
    elif result:
        lines.extend(
            (
                "A non-claim-eligible mechanical holdout completed, but all numerical values are "
                "suppressed. Synthetic or unanchored output cannot support a predictive or "
                "trading claim.",
                "",
            )
        )
    else:
        lines.extend(
            (
                "The final date block remains inaccessible. No holdout performance claim is made.",
                "",
            )
        )
    lines.extend(
        (
            "## Trading-cost frontier and capacity",
            "",
            "Executable returns enter longs at the ask and shorts at the bid, then reverse those "
            "quotes exactly 60 seconds later. Spread is embedded once; per-side fees are separate. "
            "Displayed sizes are capacity diagnostics, not fill guarantees. Market impact and "
            "queue position are not modelled.",
            "",
            (
                "![Trading cost frontier](figures/equity_trading_cost_frontier.png)"
                if source_kind == "real"
                else "Synthetic trading-cost values and figures are suppressed."
            ),
            "",
            "## Limitations and falsification outcome",
            "",
            "This is an auction-period quote-crossing simulation on anonymised normalized prices, "
            "not millisecond execution, queue-position modelling, passive fills, consolidated "
            "NBBO, "
            "or live deployability. A negative net result is a valid falsification of the "
            "registered "
            "economic threshold and must not be hidden; a positive result still requires caution "
            "about competition sampling, capacity and external validity.",
            "",
        )
    )
    report = destination / "equity_closing_auction_report.md"
    report.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")

    cv_lines = ["# CV evidence", ""]
    if result and not synthetic:
        predictive = dict(result["predictive_metrics"])
        trading = dict(result["trading_summary"])
        net = trading["net_mean_bps"]
        if net is not None and float(net) >= 0:
            bullet = (
                "- Built a causal equity closing-auction research pipeline with chronological "
                "date-level selection and a content-locked holdout; measured "
                f"{_fmt(predictive['mae_bps'])} "
                f"bps MAE, {_fmt(predictive['spearman_ic'])} Spearman IC and {_fmt(net)} bps mean "
                "net quote-crossing return after spread and the locked per-side fee assumption."
            )
        else:
            bullet = (
                "- Built and falsified a registered equity closing-auction strategy using a "
                "content-locked chronological holdout; quantified predictive error "
                f"({_fmt(predictive['mae_bps'])} bps MAE), rank IC "
                f"({_fmt(predictive['spearman_ic'])}) "
                f"and the execution-cost shortfall leading to {_fmt(net)} bps mean net return."
            )
        cv_lines.extend(("Evidence-backed real-data draft bullet:", "", bullet, ""))
    else:
        cv_lines.extend(
            (
                "Safe pre-holdout engineering bullet:",
                "",
                "- Engineered a tested, causal equity closing-auction pipeline with date-isolated "
                "features, expanding-window model selection, executable 60-second quote alignment, "
                "spread/fee attribution and content-addressed one-shot holdout controls.",
                "",
                "Synthetic values and validation-only numbers are intentionally excluded.",
            )
        )
    cv = destination / "cv_evidence.md"
    cv.write_text("\n".join(cv_lines).rstrip() + "\n", encoding="utf-8")
    return report, cv
