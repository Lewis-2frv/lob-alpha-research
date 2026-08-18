"""Integrity-gated predictive evidence reports and portfolio publishing for FI-2010."""

from __future__ import annotations

import json
import shutil
import statistics
from pathlib import Path
from typing import Any

from .fi2010_config import FI2010Config
from .fi2010_data import atomic_json, read_json
from .fi2010_study import (
    ANCHOR_FILENAME,
    CLAIM_FILENAME,
    SEAL_FILENAME,
    _validate_development_semantics,
    implementation_hashes,
    runtime_versions,
)
from .manifest import sha256_file

RESULTS_START = "<!-- FI2010_RESULTS_START -->"
RESULTS_END = "<!-- FI2010_RESULTS_END -->"


def _require_hash(path: str | Path, expected: str, label: str) -> Path:
    resolved = Path(path).resolve()
    if not resolved.is_file() or sha256_file(resolved) != expected:
        raise ValueError(f"{label} is missing or its SHA-256 changed: {resolved}")
    return resolved


def _metric(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.6f}"


def _percent(value: Any, digits: int = 1) -> str:
    return "n/a" if value is None else f"{100 * float(value):.{digits}f}%"


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)




def _mean_optional(values: list[Any]) -> float | None:
    numeric = [float(value) for value in values if value is not None]
    return float(statistics.fmean(numeric)) if numeric else None


def _min_optional(values: list[Any]) -> float | None:
    numeric = [float(value) for value in values if value is not None]
    return float(min(numeric)) if numeric else None

def _short_model_name(specification: dict[str, Any]) -> str:
    model = specification["model"]
    if model == "hist_gradient_boosting_fallback":
        return (
            "HistGradientBoosting"
            f" lr={specification['learning_rate']}, leaves={specification['max_leaf_nodes']}"
        )
    if model == "lightgbm_multiclass":
        return (
            "LightGBM"
            f" lr={specification['learning_rate']}, leaves={specification['num_leaves']}"
        )
    if model == "manual_liquidity_pressure":
        return "Manual liquidity-pressure rule"
    if model == "numpy_diagonal_lda":
        return f"NumPy diagonal LDA shrinkage={specification['shrinkage']}"
    if model == "numpy_ridge_multiclass":
        return f"NumPy ridge alpha={specification['alpha']}"
    if model == "numpy_softmax":
        return "NumPy softmax regression"
    if model == "sgd_log_loss":
        return f"SGD log-loss alpha={specification['alpha']}"
    if model == "always_stationary":
        return "Always stationary"
    if model == "dummy_prior":
        return "Class-prior baseline"
    return str(model)



def _model_family(model: str) -> tuple[str, int]:
    mapping = {
        "always_stationary": ("Naive stationary", 0),
        "dummy_prior": ("Class-prior baseline", 1),
        "manual_liquidity_pressure": ("Manual liquidity rule", 2),
        "numpy_diagonal_lda": ("From-scratch diagonal LDA", 3),
        "numpy_ridge_multiclass": ("From-scratch ridge", 4),
        "numpy_softmax": ("From-scratch softmax", 5),
        "sgd_log_loss": ("sklearn linear SGD", 6),
        "hist_gradient_boosting_fallback": ("sklearn histogram boosting", 7),
        "lightgbm_multiclass": ("LightGBM", 8),
    }
    return mapping.get(model, (model, 99))


def _best_family_rows(development: dict[str, Any]) -> list[dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    for item in development["model_ranking"]:
        family, order = _model_family(item["specification"]["model"])
        if family not in best:
            best[family] = {**item, "family": family, "family_order": order}
    return sorted(best.values(), key=lambda item: int(item["family_order"]))

def _selected_folds(development: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    candidate = development["selected_candidate"]
    selected_id = candidate["specification_id"]
    folds = [
        item for item in development["fold_results"] if item["specification_id"] == selected_id
    ]
    if len(folds) != 8:
        raise ValueError("selected candidate does not have exactly eight development folds")
    return candidate, sorted(folds, key=lambda item: int(item["fold"]))


def _best_baseline(development: dict[str, Any]) -> dict[str, Any] | None:
    baseline_models = {"always_stationary", "dummy_prior"}
    for item in development["model_ranking"]:
        if item["specification"].get("model") in baseline_models:
            return item
    return None


def _confidence_frontier(
    config: FI2010Config, selected_folds: list[dict[str, Any]]
) -> list[dict[str, float]]:
    frontier: list[dict[str, float]] = []
    for threshold in config.selection.confidence_thresholds:
        diagnostics = []
        for fold in selected_folds:
            match = next(
                item
                for item in fold["directional_signal_diagnostics"]
                if float(item["threshold"]) == float(threshold)
            )
            diagnostics.append(match)
        frontier.append(
            {
                "threshold": float(threshold),
                "mean_directional_precision": _mean_optional(
                    [item["directional_precision"] for item in diagnostics]
                ),
                "worst_fold_directional_precision": _min_optional(
                    [item["directional_precision"] for item in diagnostics]
                ),
                "mean_directional_coverage": float(
                    statistics.fmean(float(item["directional_coverage"]) for item in diagnostics)
                ),
                "mean_abstention_rate": float(
                    statistics.fmean(float(item["abstention_rate"]) for item in diagnostics)
                ),
            }
        )
    return frontier


def _portfolio_metrics(
    config: FI2010Config,
    development: dict[str, Any],
    final_manifest: dict[str, Any] | None,
    holdout_result: dict[str, Any] | None,
) -> dict[str, Any]:
    candidate, selected_folds = _selected_folds(development)
    scores = [float(item["metrics"]["macro_f1"]) for item in selected_folds]
    frontier = _confidence_frontier(config, selected_folds)
    threshold = float(candidate["confidence_rule"]["threshold"])
    selected_signal = next(item for item in frontier if item["threshold"] == threshold)
    baseline = _best_baseline(development)
    family_rows = _best_family_rows(development)
    manual_models = {
        "manual_liquidity_pressure",
        "numpy_diagonal_lda",
        "numpy_ridge_multiclass",
        "numpy_softmax",
    }
    manual_rows = [
        item for item in development["model_ranking"]
        if item["specification"].get("model") in manual_models
    ]
    best_manual = manual_rows[0] if manual_rows else None
    nonlinear_models = {"hist_gradient_boosting_fallback", "lightgbm_multiclass"}
    nonlinear_rows = [
        item for item in development["model_ranking"]
        if item["specification"].get("model") in nonlinear_models
    ]
    best_nonlinear = nonlinear_rows[0] if nonlinear_rows else None
    mean_fit = statistics.fmean(float(item["efficiency"]["fit_seconds"]) for item in selected_folds)
    mean_throughput = statistics.fmean(
        float(item["efficiency"]["prediction_throughput_observations_per_second"])
        for item in selected_folds
    )
    result: dict[str, Any] = {
        "stage": "fi2010_portfolio_metrics",
        "claim_eligible": bool(development["claim_eligible"]),
        "selected_model": {
            "specification": candidate["specification"],
            "specification_id": candidate["specification_id"],
            "display_name": _short_model_name(candidate["specification"]),
        },
        "model_ladder": {
            "families": [
                {
                    "family": item["family"],
                    "display_name": _short_model_name(item["specification"]),
                    "mean_macro_f1": float(item["mean_macro_f1"]),
                    "worst_fold_macro_f1": float(item["worst_fold_macro_f1"]),
                }
                for item in family_rows
            ],
            "best_manual_mean_macro_f1": (
                float(best_manual["mean_macro_f1"]) if best_manual is not None else None
            ),
            "best_manual_display_name": (
                _short_model_name(best_manual["specification"]) if best_manual is not None else None
            ),
            "best_nonlinear_mean_macro_f1": (
                float(best_nonlinear["mean_macro_f1"]) if best_nonlinear is not None else None
            ),
            "best_nonlinear_display_name": (
                _short_model_name(best_nonlinear["specification"])
                if best_nonlinear is not None
                else None
            ),
            "nonlinear_uplift_vs_best_manual_macro_f1": (
                float(best_nonlinear["mean_macro_f1"] - best_manual["mean_macro_f1"])
                if best_manual is not None and best_nonlinear is not None
                else None
            ),
        },
        "development": {
            "folds": 8,
            "mean_macro_f1": float(statistics.fmean(scores)),
            "std_macro_f1": float(statistics.pstdev(scores)),
            "worst_fold_macro_f1": float(min(scores)),
            "best_fold_macro_f1": float(max(scores)),
            "best_baseline_mean_macro_f1": (
                float(baseline["mean_macro_f1"]) if baseline is not None else None
            ),
            "uplift_vs_best_baseline_macro_f1": (
                float(statistics.fmean(scores) - float(baseline["mean_macro_f1"]))
                if baseline is not None
                else None
            ),
            "mean_fit_seconds": float(mean_fit),
            "mean_prediction_throughput_observations_per_second": float(mean_throughput),
        },
        "signal": {
            "selected_threshold": threshold,
            **selected_signal,
            "frontier": frontier,
        },
        "final_refit": (
            {
                "train_observations": int(final_manifest["train_observations"]),
                "fit_wall_seconds": float(final_manifest["fit_wall_seconds"]),
                "model_sha256": final_manifest["model_sha256"],
            }
            if final_manifest is not None
            else None
        ),
        "holdout": None,
        "executable_performance_claimed": False,
    }
    if holdout_result is not None:
        primary = holdout_result["primary_metrics"]
        holdout_signal = holdout_result["directional_signal_diagnostics"]
        result["holdout"] = {
            "observations": int(holdout_result["observations"]),
            "macro_f1": float(primary["macro_f1"]),
            "balanced_accuracy": float(primary["balanced_accuracy"]),
            "multiclass_log_loss": float(primary["multiclass_log_loss"]),
            "mcc": float(primary["mcc"]),
            "directional_precision": float(holdout_signal["directional_precision"]),
            "directional_coverage": float(holdout_signal["directional_coverage"]),
            "abstention_rate": float(holdout_signal["abstention_rate"]),
            "macro_f1_generalization_gap_vs_development_mean": float(
                primary["macro_f1"] - statistics.fmean(scores)
            ),
            "one_shot": True,
        }
    return result


def _write_figures(
    output: Path,
    development: dict[str, Any],
    metrics: dict[str, Any],
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    candidate, selected_folds = _selected_folds(development)
    selected_scores = [float(item["metrics"]["macro_f1"]) for item in selected_folds]
    folds = [int(item["fold"]) for item in selected_folds]
    baseline = _best_baseline(development)
    baseline_scores: list[float] | None = None
    if baseline is not None:
        baseline_id = baseline["specification_id"]
        baseline_rows = sorted(
            [
                item
                for item in development["fold_results"]
                if item["specification_id"] == baseline_id
            ],
            key=lambda item: int(item["fold"]),
        )
        baseline_scores = [float(item["metrics"]["macro_f1"]) for item in baseline_rows]

    paths: list[Path] = []
    fold_path = output / "development_macro_f1_by_fold.png"
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    ax.plot(folds, selected_scores, marker="o", label="Selected candidate")
    if baseline_scores is not None:
        ax.plot(folds, baseline_scores, marker="o", label="Best baseline")
    ax.axhline(
        metrics["development"]["mean_macro_f1"],
        linestyle="--",
        linewidth=1,
        label="Selected mean",
    )
    ax.set_xlabel("Anchored development fold")
    ax.set_ylabel("Macro-F1")
    ax.set_title("FI-2010 anchored development stability")
    ax.set_xticks(folds)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(fold_path, dpi=170)
    plt.close(fig)
    paths.append(fold_path)

    ranking_path = output / "model_comparison.png"
    family_rows = _best_family_rows(development)
    labels = [item["family"] for item in family_rows]
    means = [float(item["mean_macro_f1"]) for item in family_rows]
    worst = [float(item["worst_fold_macro_f1"]) for item in family_rows]
    fig_height = max(5.2, 0.52 * len(labels) + 1.8)
    fig, ax = plt.subplots(figsize=(9.6, fig_height))
    positions = list(range(len(labels)))
    ax.barh(positions, means, label="Mean macro-F1")
    ax.scatter(worst, positions, marker="|", s=120, label="Worst fold")
    ax.set_yticks(positions, labels=labels)
    ax.set_xlabel("Macro-F1")
    ax.set_title("Model ladder: best registered setting per family")
    ax.grid(axis="x", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(ranking_path, dpi=170)
    plt.close(fig)
    paths.append(ranking_path)

    frontier_path = output / "confidence_precision_coverage.png"
    frontier = metrics["signal"]["frontier"]
    plottable = [item for item in frontier if item["mean_directional_precision"] is not None]
    coverage = [float(item["mean_directional_coverage"]) for item in plottable]
    precision = [float(item["mean_directional_precision"]) for item in plottable]
    thresholds = [float(item["threshold"]) for item in plottable]
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    ax.plot(coverage, precision, marker="o")
    for x_value, y_value, threshold_value in zip(coverage, precision, thresholds, strict=True):
        ax.annotate(
            f"{threshold_value:.2f}",
            (x_value, y_value),
            textcoords="offset points",
            xytext=(5, 5),
        )
    ax.set_xlabel("Mean directional coverage")
    ax.set_ylabel("Mean directional precision")
    ax.set_title("Confidence/coverage frontier across CF_1-CF_8")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(frontier_path, dpi=170)
    plt.close(fig)
    paths.append(frontier_path)

    if metrics["holdout"] is not None:
        holdout_path = output / "development_vs_holdout.png"
        values = [
            metrics["development"]["mean_macro_f1"],
            metrics["development"]["worst_fold_macro_f1"],
            metrics["holdout"]["macro_f1"],
        ]
        fig, ax = plt.subplots(figsize=(6.8, 4.8))
        ax.bar(["Development mean", "Development worst", "CF_9 holdout"], values)
        ax.set_ylabel("Macro-F1")
        ax.set_title("Frozen model: development versus one-shot holdout")
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(holdout_path, dpi=170)
        plt.close(fig)
        paths.append(holdout_path)

    return paths


def _cv_summary(metrics: dict[str, Any]) -> str:
    selected = metrics["selected_model"]["display_name"]
    development = metrics["development"]
    signal = metrics["signal"]
    holdout = metrics["holdout"]
    if holdout is None:
        headline = (
            f"- Built a leakage-resistant FI-2010 limit-order-book signal research pipeline with "
            f"8 anchored walk-forward development folds; selected {selected} at "
            f"{development['mean_macro_f1']:.3f} mean macro-F1 "
            f"({development['worst_fold_macro_f1']:.3f} worst fold), with the final CF_9 "
            "holdout still sealed."
        )
    else:
        headline = (
            f"- Built a leakage-resistant FI-2010 limit-order-book signal research pipeline with "
            f"8 anchored walk-forward development folds and a one-shot sealed CF_9 holdout; "
            f"selected {selected} at {development['mean_macro_f1']:.3f} development macro-F1 "
            f"and achieved {holdout['macro_f1']:.3f} final holdout macro-F1."
        )
    ladder = metrics["model_ladder"]
    technical = (
        "- Compared a manual LOB liquidity-pressure rule and from-scratch NumPy "
        "LDA/ridge/softmax models against sklearn and boosted-tree benchmarks "
        "on identical anchored folds; implemented confidence-based abstention "
        f"({_percent(signal['mean_directional_precision'])} development "
        f"directional precision at {_percent(signal['mean_directional_coverage'])} "
        "coverage), content-addressed model freezing, runtime/config integrity "
        "checks, and a durable single-use holdout gate."
    )
    comparison = (
        f"- Best manual/from-scratch development model: {ladder['best_manual_display_name']} at "
        f"{ladder['best_manual_mean_macro_f1']:.3f} mean macro-F1; best nonlinear tree model: "
        f"{ladder['best_nonlinear_display_name']} at "
        f"{ladder['best_nonlinear_mean_macro_f1']:.3f}."
        if ladder["best_manual_mean_macro_f1"] is not None
        and ladder["best_nonlinear_mean_macro_f1"] is not None
        else None
    )
    caveat = (
        "- Scope: predictive LOB classification and confidence/coverage analysis only; FI-2010's "
        "normalised anonymised snapshots do not support defensible executable P&L or Sharpe claims."
    )
    bullets = [headline, technical]
    if comparison is not None:
        bullets.append(comparison)
    bullets.append(caveat)
    return "# CV-ready FI-2010 project summary\n\n" + "\n".join(bullets) + "\n"


def build_fi2010_report(
    config: FI2010Config,
    *,
    prepared_dir: str | Path | None = None,
    development_results: str | Path = "artifacts/fi2010/development/development_results.json",
    freeze_dir: str | Path = "artifacts/fi2010/freeze",
    holdout_dir: str | Path = "artifacts/fi2010/holdout",
    output_dir: str | Path = "artifacts/fi2010/report",
) -> tuple[Path, Path]:
    """Build charts and reports only from content-addressed development/holdout evidence."""

    root = Path(prepared_dir or config.data.prepared_dir).resolve()
    development_path = Path(development_results).resolve()
    development = read_json(development_path)
    if development.get("stage") != "fi2010_anchored_development":
        raise ValueError("not an FI-2010 development result")
    if development.get("config_sha256") != sha256_file(config.path):
        raise ValueError("configuration changed after development")
    _require_hash(
        development["development_manifest_path"],
        development["development_manifest_sha256"],
        "development manifest",
    )
    if development.get("implementation_hashes") != implementation_hashes():
        raise ValueError("FI-2010 implementation changed after development")
    if development.get("runtime_versions") != runtime_versions():
        raise ValueError("FI-2010 runtime dependencies changed after development")
    if development.get("cf9_test_payload_opened") is not False:
        raise PermissionError("development evidence does not prove CF_9 isolation")
    _validate_development_semantics(config, development)

    freeze = Path(freeze_dir).resolve()
    frozen_path = freeze / "frozen_candidate.json"
    final_manifest_path = freeze / "final_model_manifest.json"
    frozen: dict[str, Any] | None = None
    final_manifest: dict[str, Any] | None = None
    if frozen_path.exists() or final_manifest_path.exists():
        if not frozen_path.is_file() or not final_manifest_path.is_file():
            raise ValueError("freeze/refit evidence is incomplete")
        frozen = read_json(frozen_path)
        final_manifest = read_json(final_manifest_path)
        if frozen.get("stage") != "fi2010_frozen_candidate":
            raise ValueError("not an FI-2010 frozen candidate")
        if frozen.get("config_sha256") != sha256_file(config.path):
            raise ValueError("frozen candidate configuration binding changed")
        if frozen.get("source_manifest_sha256") != sha256_file(root / "source_manifest.json"):
            raise ValueError("frozen candidate source binding changed")
        if frozen.get("holdout_manifest_sha256") != sha256_file(root / "holdout_manifest.json"):
            raise ValueError("frozen candidate holdout-manifest binding changed")
        if frozen.get("implementation_hashes") != implementation_hashes():
            raise ValueError("frozen candidate implementation binding changed")
        if frozen.get("runtime_versions") != runtime_versions():
            raise ValueError("frozen candidate runtime binding changed")
        if frozen.get("development_results_sha256") != sha256_file(development_path):
            raise ValueError("frozen candidate is not bound to these development results")
        if final_manifest.get("frozen_candidate_sha256") != sha256_file(frozen_path):
            raise ValueError("final model manifest is not bound to the frozen candidate")
        if final_manifest.get("stage") != "fi2010_final_refit":
            raise ValueError("not an FI-2010 final-refit manifest")
        if final_manifest.get("config_sha256") != frozen.get("config_sha256"):
            raise ValueError("final model configuration binding changed")
        if final_manifest.get("implementation_hashes") != frozen.get("implementation_hashes"):
            raise ValueError("final model implementation binding changed")
        if final_manifest.get("runtime_versions") != frozen.get("runtime_versions"):
            raise ValueError("final model runtime binding changed")
        if final_manifest.get("source_manifest_sha256") != frozen.get("source_manifest_sha256"):
            raise ValueError("final model source binding changed")
        if final_manifest.get("primary_label_row") != config.data.primary_label_row:
            raise ValueError("final model target binding changed")
        if final_manifest.get("development_results_sha256") != frozen.get(
            "development_results_sha256"
        ):
            raise ValueError("final model development binding changed")
        if final_manifest.get("candidate_specification_id") != frozen["candidate"][
            "specification_id"
        ]:
            raise ValueError("final model candidate binding changed")
        if final_manifest.get("class_mapping") != frozen.get("class_mapping"):
            raise ValueError("final model class mapping changed")
        _require_hash(final_manifest["model_path"], final_manifest["model_sha256"], "final model")
        if final_manifest.get("cf9_test_payload_opened") is not False:
            raise PermissionError("final refit evidence does not prove CF_9 isolation")

    holdout = Path(holdout_dir).resolve()
    result_path = holdout / "holdout_result.json"
    holdout_result: dict[str, Any] | None = None
    claim_path = root / CLAIM_FILENAME
    seal_path = root / SEAL_FILENAME
    anchor_path = root / ANCHOR_FILENAME
    release_markers = (
        claim_path.exists(),
        seal_path.exists(),
        anchor_path.exists(),
        result_path.exists(),
    )
    if any(release_markers) and not all(release_markers):
        raise ValueError(
            "FI-2010 holdout was claimed or partially written but did not complete cleanly; "
            "it cannot be reported as untouched"
        )
    if result_path.exists():
        if frozen is None or final_manifest is None:
            raise ValueError("completed holdout cannot be reported without freeze/refit evidence")
        anchor = read_json(anchor_path)
        if anchor.get("stage") != "fi2010_cf9_holdout_completion_anchor":
            raise ValueError("invalid FI-2010 holdout completion anchor")
        _require_hash(seal_path, anchor["seal_sha256"], "holdout seal")
        if anchor.get("source_manifest_sha256") != sha256_file(root / "source_manifest.json"):
            raise ValueError("completion anchor source binding changed")
        seal = read_json(seal_path)
        if seal.get("stage") != "fi2010_cf9_holdout_started":
            raise ValueError("invalid FI-2010 holdout seal")
        if seal.get("source_manifest_sha256") != sha256_file(root / "source_manifest.json"):
            raise ValueError("holdout seal source binding changed")
        holdout_manifest_path = root / "holdout_manifest.json"
        _require_hash(
            holdout_manifest_path,
            seal.get("holdout_manifest_sha256", ""),
            "holdout manifest",
        )
        holdout_manifest = read_json(holdout_manifest_path)
        if seal.get("holdout_member") != holdout_manifest.get("member"):
            raise ValueError("holdout seal member binding changed")
        if seal.get("frozen_candidate_sha256") != sha256_file(frozen_path):
            raise ValueError("holdout seal frozen-candidate binding changed")
        if seal.get("final_model_manifest_sha256") != sha256_file(final_manifest_path):
            raise ValueError("holdout seal final-model binding changed")
        if seal.get("config_sha256") != sha256_file(config.path):
            raise ValueError("holdout seal configuration binding changed")
        if seal.get("implementation_hashes") != implementation_hashes():
            raise ValueError("holdout seal implementation binding changed")
        if seal.get("runtime_versions") != runtime_versions():
            raise ValueError("holdout seal runtime binding changed")
        outputs = anchor.get("outputs", {})
        result_key = str(result_path.resolve())
        if set(outputs) != {result_key}:
            raise ValueError("completion anchor does not bind exactly the holdout outputs")
        _require_hash(result_path, outputs[result_key], "holdout result")
        holdout_result = read_json(result_path)
        if holdout_result.get("seal_sha256") != sha256_file(seal_path):
            raise ValueError("holdout result seal binding changed")
        if holdout_result.get("stage") != "fi2010_cf9_holdout_complete":
            raise ValueError("invalid FI-2010 holdout result stage")
        if holdout_result.get("one_shot") is not True:
            raise ValueError("holdout result is not marked one-shot")
        if holdout_result.get("member") != seal.get("holdout_member"):
            raise ValueError("holdout result member binding changed")
        if bool(holdout_result.get("claim_eligible")) != bool(development.get("claim_eligible")):
            raise ValueError("holdout result claim-eligibility binding changed")
        if holdout_result.get("executable_performance_claimed") is not False:
            raise ValueError("FI-2010 holdout cannot claim executable performance")

    output = Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite FI-2010 report: {output}")
    output.mkdir(parents=True, exist_ok=True)

    candidate, selected_folds = _selected_folds(development)
    portfolio = _portfolio_metrics(config, development, final_manifest, holdout_result)
    portfolio_path = output / "portfolio_metrics.json"
    atomic_json(portfolio_path, portfolio)
    figures = _write_figures(output, development, portfolio)
    cv_path = output / "cv_summary.md"
    _atomic_text(cv_path, _cv_summary(portfolio))

    claim_eligible = bool(development["claim_eligible"])
    status = (
        "registered real anchored development evidence"
        if claim_eligible
        else "synthetic engineering rehearsal; ineligible for real CV claims"
    )
    if holdout_result is not None:
        status = (
            "registered real one-shot holdout evidence"
            if claim_eligible
            else "synthetic one-shot rehearsal; ineligible for real CV claims"
        )
    development_metrics = portfolio["development"]
    signal = portfolio["signal"]
    lines = [
        "# FI-2010 anchored walk-forward evidence",
        "",
        f"Evidence status: **{status}**.",
        "",
        "This study evaluates predictive classification and confidence-filtered directional "
        "signals only. FI-2010's normalized, anonymised snapshots do not support a defensible "
        "executable-performance reconstruction.",
        "",
        "## Executive summary",
        "",
        f"- Selected model: **{portfolio['selected_model']['display_name']}**.",
        f"- Development mean macro-F1: **{_metric(development_metrics['mean_macro_f1'])}** "
        "across eight anchored CF pairs; worst fold: "
        f"**{_metric(development_metrics['worst_fold_macro_f1'])}**.",
        "- Fold-to-fold macro-F1 standard deviation: "
        f"**{_metric(development_metrics['std_macro_f1'])}**.",
        "- Best-baseline uplift in mean macro-F1: "
        f"**{_metric(development_metrics['uplift_vs_best_baseline_macro_f1'])}**.",
        f"- Selected confidence rule: threshold **{signal['selected_threshold']:.2f}**, "
        f"mean directional precision **{_percent(signal['mean_directional_precision'])}** at "
        f"**{_percent(signal['mean_directional_coverage'])}** coverage.",
    ]
    if portfolio["holdout"] is None:
        lines.append("- Final CF_9 test: **sealed and unevaluated**.")
    else:
        holdout_metrics = portfolio["holdout"]
        lines.extend(
            [
                f"- One-shot CF_9 holdout macro-F1: **{_metric(holdout_metrics['macro_f1'])}**.",
                "- One-shot CF_9 directional precision: "
                f"**{_percent(holdout_metrics['directional_precision'])}** "
                f"at **{_percent(holdout_metrics['directional_coverage'])}** coverage.",
                "- Holdout minus development-mean macro-F1: "
                "**"
                f"{_metric(holdout_metrics['macro_f1_generalization_gap_vs_development_mean'])}"
                "**.",
            ]
        )
    lines.extend(
        [
            "",
            "![Anchored development stability](development_macro_f1_by_fold.png)",
            "",
            "## Registered protocol",
            "",
            "- Representation: `NoAuction/1.NoAuction_Zscore`; feature rows 1-144.",
            "- Publisher classes: `1=up`, `2=stationary`, `3=down`.",
            "- Primary target: label row 4, five sampled steps / 50 underlying events.",
            "- Development: matching Train/Test members for CF_1-CF_8, handled independently.",
            "- Cumulative training members were never concatenated.",
            "- Selection: mean macro-F1, then worst-fold macro-F1, then lower complexity.",
            "- Candidate and confidence-threshold tuning use development folds only.",
            "",
            "## Selected development candidate",
            "",
            f"Specification: `{candidate['specification_id']}`  ",
            f"Mean macro-F1: {_metric(candidate['selection']['mean_macro_f1'])}  ",
            f"Worst-fold macro-F1: {_metric(candidate['selection']['worst_fold_macro_f1'])}  ",
            f"Confidence threshold: {_metric(candidate['confidence_rule']['threshold'])}",
            "",
            "## Fold stability",
            "",
            "| Fold | Macro-F1 | Balanced accuracy | Log loss | MCC | Fit seconds | Pred. obs/s |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for item in selected_folds:
        item_metrics = item["metrics"]
        efficiency = item["efficiency"]
        lines.append(
            f"| {item['fold']} | {_metric(item_metrics['macro_f1'])} | "
            f"{_metric(item_metrics['balanced_accuracy'])} | "
            f"{_metric(item_metrics['multiclass_log_loss'])} | {_metric(item_metrics['mcc'])} | "
            f"{_metric(efficiency['fit_seconds'])} | "
            f"{_metric(efficiency['prediction_throughput_observations_per_second'])} |"
        )
    lines.extend(
        [
            "",
            "## Model comparison",
            "",
            "![Registered candidate comparison](model_comparison.png)",
            "",
            "| Rank | Model | Mean macro-F1 | Worst-fold macro-F1 |",
            "|---:|---|---:|---:|",
        ]
    )
    for rank, item in enumerate(development["model_ranking"], start=1):
        lines.append(
            f"| {rank} | {_short_model_name(item['specification'])} | "
            f"{_metric(item['mean_macro_f1'])} | {_metric(item['worst_fold_macro_f1'])} |"
        )
    ladder = portfolio["model_ladder"]
    lines.extend(
        [
            "",
            "### Interpretable-to-nonlinear model ladder",
            "",
            "The comparison deliberately starts with a fixed microstructure rule and "
            "from-scratch statistical models before library tree ensembles. This makes the "
            "incremental value of model complexity observable on the same anchored folds.",
            "",
            "| Family | Best registered specification | Mean macro-F1 | Worst fold |",
            "|---|---|---:|---:|",
        ]
    )
    for item in ladder["families"]:
        lines.append(
            f"| {item['family']} | {item['display_name']} | "
            f"{_metric(item['mean_macro_f1'])} | {_metric(item['worst_fold_macro_f1'])} |"
        )
    if ladder["best_manual_mean_macro_f1"] is not None:
        lines.extend(
            [
                "",
                f"Best from-scratch/manual candidate: **{ladder['best_manual_display_name']}** "
                f"at **{_metric(ladder['best_manual_mean_macro_f1'])}** mean macro-F1.",
            ]
        )
    if ladder["best_nonlinear_mean_macro_f1"] is not None:
        lines.append(
            f"Best nonlinear tree candidate: **{ladder['best_nonlinear_display_name']}** "
            f"at **{_metric(ladder['best_nonlinear_mean_macro_f1'])}** mean macro-F1."
        )
    if ladder["nonlinear_uplift_vs_best_manual_macro_f1"] is not None:
        lines.append(
            "Nonlinear-minus-best-manual development macro-F1 difference: "
            f"**{_metric(ladder['nonlinear_uplift_vs_best_manual_macro_f1'])}**. "
            "This is a development-only complexity comparison, not a holdout-tuned result."
        )
    class_names = ("up", "stationary", "down")
    lines.extend(
        [
            "",
            "## Selected-candidate class diagnostics",
            "",
            "| Class | Mean precision | Mean recall | Mean F1 | Total support |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for class_name in class_names:
        diagnostics = [item["metrics"]["per_class"][class_name] for item in selected_folds]
        lines.append(
            f"| {class_name} | "
            f"{_metric(statistics.fmean(item['precision'] for item in diagnostics))} | "
            f"{_metric(statistics.fmean(item['recall'] for item in diagnostics))} | "
            f"{_metric(statistics.fmean(item['f1'] for item in diagnostics))} | "
            f"{sum(int(item['support']) for item in diagnostics)} |"
        )
    confusion = [[0, 0, 0] for _ in range(3)]
    for item in selected_folds:
        for row in range(3):
            for column in range(3):
                confusion[row][column] += int(item["metrics"]["confusion_matrix"][row][column])
    lines.extend(
        [
            "",
            "Aggregate CF_1-CF_8 confusion matrix (rows=true, columns=predicted):",
            "",
            "| True / Pred. | Up | Stationary | Down |",
            "|---|---:|---:|---:|",
            f"| Up | {confusion[0][0]} | {confusion[0][1]} | {confusion[0][2]} |",
            f"| Stationary | {confusion[1][0]} | {confusion[1][1]} | {confusion[1][2]} |",
            f"| Down | {confusion[2][0]} | {confusion[2][1]} | {confusion[2][2]} |",
            "",
            "## Confidence/coverage frontier",
            "",
            "![Confidence precision/coverage frontier](confidence_precision_coverage.png)",
            "",
            "| Threshold | Mean precision | Worst-fold precision | Mean coverage | "
            "Mean abstention |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for item in signal["frontier"]:
        marker = " **selected**" if item["threshold"] == signal["selected_threshold"] else ""
        lines.append(
            f"| {item['threshold']:.2f}{marker} | {_metric(item['mean_directional_precision'])} | "
            f"{_metric(item['worst_fold_directional_precision'])} | "
            f"{_metric(item['mean_directional_coverage'])} | "
            f"{_metric(item['mean_abstention_rate'])} |"
        )
    lines.extend(["", "## Final holdout", ""])
    if holdout_result is None:
        lines.append(
            "CF_9 test remains sealed and unevaluated; no final-holdout result is reported."
        )
    else:
        primary = holdout_result["primary_metrics"]
        directional = holdout_result["directional_signal_diagnostics"]
        lines.extend(
            [
                "The source-side seal and completion anchor validated successfully. This is the "
                "single retained final evaluation; the release gate does not permit a rerun.",
                "",
                f"- Macro-F1: {_metric(primary['macro_f1'])}",
                f"- Balanced accuracy: {_metric(primary['balanced_accuracy'])}",
                f"- Multiclass log loss: {_metric(primary['multiclass_log_loss'])}",
                f"- MCC: {_metric(primary['mcc'])}",
                "- Directional precision at the frozen threshold: "
                f"{_metric(directional['directional_precision'])}",
                f"- Directional coverage: {_metric(directional['directional_coverage'])}",
                "",
                "![Development versus final holdout](development_vs_holdout.png)",
            ]
        )
    lines.extend(
        [
            "",
            "## Reproducibility and integrity",
            "",
            f"- Runtime versions: `{json.dumps(development['runtime_versions'], sort_keys=True)}`.",
            "- Development implementation hashes: "
            f"`{json.dumps(development['implementation_hashes'], sort_keys=True)}`.",
            "- Final Train_CF_9 refit observations: "
            + (
                str(portfolio["final_refit"]["train_observations"])
                if portfolio["final_refit"]
                else "not yet frozen"
            )
            + ".",
            "- Raw data, extracted payloads and model binaries remain gitignored; evidence "
            "artifacts are content-addressed.",
            "",
            "## Limitations",
            "",
            "FI-2010 omits unnormalised prices, timestamps, instrument identities, venue/feed "
            "details and reliable instrument/day boundaries. The study consumes publisher-provided "
            "Z-score matrices and cannot independently audit the original normalization field of "
            "view. Snapshot classification is therefore the supported design; sequence windows and "
            "executable trading claims are excluded.",
            "",
        ]
    )
    report_path = output / "fi2010_evidence.md"
    _atomic_text(report_path, "\n".join(lines))

    artifact_hashes = {
        str(path.name): sha256_file(path)
        for path in [report_path, portfolio_path, cv_path, *figures]
    }
    evidence = {
        "stage": "fi2010_evidence_report",
        "claim_eligible": claim_eligible,
        "development_results_path": str(development_path),
        "development_results_sha256": sha256_file(development_path),
        "frozen_candidate_sha256": sha256_file(frozen_path) if frozen else None,
        "final_model_manifest_sha256": sha256_file(final_manifest_path) if final_manifest else None,
        "holdout_reported": holdout_result is not None,
        "holdout_result_sha256": sha256_file(result_path) if holdout_result else None,
        "report_path": str(report_path),
        "report_sha256": sha256_file(report_path),
        "portfolio_metrics_path": str(portfolio_path),
        "portfolio_metrics_sha256": sha256_file(portfolio_path),
        "cv_summary_path": str(cv_path),
        "cv_summary_sha256": sha256_file(cv_path),
        "artifact_hashes": artifact_hashes,
        "executable_performance_claimed": False,
    }
    evidence_path = output / "fi2010_evidence.json"
    atomic_json(evidence_path, evidence)
    return report_path, evidence_path


def _read_validated_report_bundle(
    report_dir: str | Path,
    *,
    require_claim_eligible: bool,
    require_holdout: bool,
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    root = Path(report_dir).resolve()
    evidence_path = root / "fi2010_evidence.json"
    metrics_path = root / "portfolio_metrics.json"
    evidence = read_json(evidence_path)
    if evidence.get("stage") != "fi2010_evidence_report":
        raise ValueError("not an FI-2010 evidence bundle")
    if evidence.get("executable_performance_claimed") is not False:
        raise ValueError("portfolio publication cannot include executable FI-2010 claims")
    if require_claim_eligible and evidence.get("claim_eligible") is not True:
        raise PermissionError("only claim-eligible real FI-2010 evidence can be published")
    if require_holdout and evidence.get("holdout_reported") is not True:
        raise PermissionError("final portfolio publication requires the one-shot CF_9 holdout")
    hashes = evidence.get("artifact_hashes", {})
    for name, expected in hashes.items():
        _require_hash(root / name, expected, f"report artifact {name}")
    _require_hash(metrics_path, evidence["portfolio_metrics_sha256"], "portfolio metrics")
    metrics = read_json(metrics_path)
    if bool(metrics.get("claim_eligible")) != bool(evidence.get("claim_eligible")):
        raise ValueError("portfolio metrics claim status disagrees with evidence")
    if require_holdout and metrics.get("holdout") is None:
        raise ValueError("portfolio metrics do not contain final holdout evidence")
    return root, evidence, metrics


def _results_block(metrics: dict[str, Any]) -> str:
    development = metrics["development"]
    signal = metrics["signal"]
    holdout = metrics["holdout"]
    selected = metrics["selected_model"]["display_name"]
    ladder = metrics["model_ladder"]
    lines = [
        RESULTS_START,
        "## Results snapshot",
        "",
        f"- **Selected model:** {selected}",
        f"- **Anchored CF_1-CF_8 mean macro-F1:** {development['mean_macro_f1']:.3f}",
        f"- **Worst development fold:** {development['worst_fold_macro_f1']:.3f}",
        "- **Confidence-filtered development precision:** "
        f"{_percent(signal['mean_directional_precision'])} "
        f"at {_percent(signal['mean_directional_coverage'])} coverage",
    ]
    if ladder["best_manual_mean_macro_f1"] is not None:
        lines.append(
            "- **Best manual/from-scratch model:** "
            f"{ladder['best_manual_display_name']} "
            f"({ladder['best_manual_mean_macro_f1']:.3f} mean macro-F1)"
        )
    if holdout is None:
        lines.append("- **CF_9:** sealed and unevaluated")
    else:
        lines.extend(
            [
                f"- **One-shot CF_9 macro-F1:** {holdout['macro_f1']:.3f}",
                "- **One-shot CF_9 directional precision:** "
                f"{_percent(holdout['directional_precision'])} "
                f"at {_percent(holdout['directional_coverage'])} coverage",
                "",
                "Full content-addressed evidence and plots: "
                "[`docs/results/fi2010/`](docs/results/fi2010/).",
            ]
        )
    lines.append(RESULTS_END)
    return "\n".join(lines)


def publish_fi2010_portfolio(
    report_dir: str | Path,
    *,
    repository_root: str | Path = ".",
    output_dir: str | Path = "docs/results/fi2010",
    require_claim_eligible: bool = True,
    require_holdout: bool = True,
) -> dict[str, Any]:
    """Publish only validated small evidence artifacts into the GitHub-facing tree."""

    source, evidence, metrics = _read_validated_report_bundle(
        report_dir,
        require_claim_eligible=require_claim_eligible,
        require_holdout=require_holdout,
    )
    repo = Path(repository_root).resolve()
    destination = Path(output_dir)
    if not destination.is_absolute():
        destination = (repo / destination).resolve()
    docs_root = (repo / "docs").resolve()
    if docs_root not in destination.parents:
        raise ValueError("portfolio output must remain under the repository docs directory")
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)

    names = ["fi2010_evidence.md", "portfolio_metrics.json", "cv_summary.md"]
    names.extend(sorted(name for name in evidence["artifact_hashes"] if name.endswith(".png")))
    for name in names:
        source_path = source / name
        if source_path.is_file():
            shutil.copy2(source_path, destination / name)

    development = metrics["development"]
    signal = metrics["signal"]
    holdout = metrics["holdout"]
    index = [
        "# FI-2010 portfolio evidence",
        "",
        "This directory is generated only from an integrity-validated FI-2010 evidence bundle. "
        "It intentionally contains no raw market data or fitted model binary.",
        "",
        f"- Selected model: **{metrics['selected_model']['display_name']}**",
        f"- Development mean macro-F1: **{development['mean_macro_f1']:.3f}**",
        "- Best manual/from-scratch development model: "
        f"**{metrics['model_ladder']['best_manual_display_name']}** "
        f"(**{metrics['model_ladder']['best_manual_mean_macro_f1']:.3f}** mean macro-F1)",
        f"- Development worst-fold macro-F1: **{development['worst_fold_macro_f1']:.3f}**",
        "- Development directional precision: "
        f"**{_percent(signal['mean_directional_precision'])}** "
        f"at **{_percent(signal['mean_directional_coverage'])}** coverage",
    ]
    if holdout is None:
        index.append("- CF_9: **sealed and unevaluated**")
    else:
        index.extend(
            [
                f"- One-shot CF_9 macro-F1: **{holdout['macro_f1']:.3f}**",
                "- One-shot CF_9 directional precision: "
                f"**{_percent(holdout['directional_precision'])}** "
                f"at **{_percent(holdout['directional_coverage'])}** coverage",
            ]
        )
    index.extend(
        [
            "",
            "![Development stability](development_macro_f1_by_fold.png)",
            "",
            "![Model comparison](model_comparison.png)",
            "",
            "![Confidence frontier](confidence_precision_coverage.png)",
        ]
    )
    if holdout is not None and (destination / "development_vs_holdout.png").exists():
        index.extend(["", "![Development versus holdout](development_vs_holdout.png)"])
    index.extend(
        [
            "",
            "- [Full evidence](fi2010_evidence.md)",
            "- [CV-ready project summary](cv_summary.md)",
            "- [Machine-readable metrics](portfolio_metrics.json)",
            "",
        ]
    )
    index_path = destination / "README.md"
    _atomic_text(index_path, "\n".join(index))

    readme_path = repo / "README.md"
    readme = readme_path.read_text(encoding="utf-8")
    block = _results_block(metrics)
    if RESULTS_START in readme and RESULTS_END in readme:
        start = readme.index(RESULTS_START)
        end = readme.index(RESULTS_END, start) + len(RESULTS_END)
        updated = readme[:start] + block + readme[end:]
    else:
        marker = "## Current status"
        if marker not in readme:
            raise ValueError("README lacks a stable insertion point for FI-2010 results")
        updated = readme.replace(marker, block + "\n\n" + marker, 1)
    _atomic_text(readme_path, updated)

    return {
        "stage": "fi2010_portfolio_published",
        "source_report_dir": str(source),
        "output_dir": str(destination),
        "readme": str(readme_path),
        "claim_eligible": bool(evidence["claim_eligible"]),
        "holdout_reported": bool(evidence["holdout_reported"]),
        "published_files": sorted(path.name for path in destination.iterdir() if path.is_file()),
    }
