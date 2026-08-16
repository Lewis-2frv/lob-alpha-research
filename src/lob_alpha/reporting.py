"""Generate evidence-linked research reports and compact diagnostic figures."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


def _read_json(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(value: object, digits: int = 4) -> str:
    if value is None or pd.isna(value):
        return "not available"
    if isinstance(value, (float, int)):
        return f"{float(value):.{digits}f}"
    return str(value)


def _save_cv_figure(train_dir: Path, figures_dir: Path) -> Path | None:
    path = train_dir / "train_ridge_cv_summary.csv"
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    figure, axis = plt.subplots(figsize=(7.2, 4.2))
    for horizon, group in frame.groupby("horizon_ms", sort=True):
        ordered = group.sort_values("alpha")
        axis.plot(
            ordered["alpha"],
            ordered["mean_daily_spearman_ic"],
            marker="o",
            label=f"{int(horizon)} ms",
        )
    axis.axhline(0.0, color="#777777", linewidth=0.8)
    axis.set_xscale("log")
    axis.set_xlabel("Ridge alpha (log scale)")
    axis.set_ylabel("Mean daily Spearman IC")
    axis.set_title("Train-only expanding-window model selection")
    axis.legend(frameon=False, ncols=2)
    axis.grid(alpha=0.2)
    figure.tight_layout()
    output = figures_dir / "train_ridge_cv.png"
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return output


def _save_sensitivity_figure(holdout_dir: Path, figures_dir: Path) -> Path | None:
    path = holdout_dir / "holdout_sensitivity.csv"
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    grids = list(frame["grid"].drop_duplicates())
    if not grids:
        return None
    figure, axes = plt.subplots(1, len(grids), figsize=(4.2 * len(grids), 3.8), squeeze=False)
    for axis, grid in zip(axes[0], grids, strict=True):
        subset = frame.loc[frame["grid"].eq(grid)].sort_values("value")
        axis.plot(subset["value"], subset["net_pnl_usd"], marker="o", color="#315b7d")
        axis.axhline(0.0, color="#777777", linewidth=0.8)
        axis.set_title(grid.replace("_", " "))
        axis.set_xlabel("Grid value")
        axis.set_ylabel("Holdout net P&L (USD)")
        axis.grid(alpha=0.2)
    figure.suptitle("Frozen-candidate execution sensitivity")
    figure.tight_layout()
    output = figures_dir / "holdout_sensitivity.png"
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return output


def build_research_report(
    *,
    train_dir: str | Path,
    validation_dir: str | Path,
    holdout_dir: str | Path,
    reports_dir: str | Path,
    engineering_fixture: bool = False,
) -> tuple[Path, Path]:
    """Build an honest report and CV evidence block from whatever stages exist."""

    train = Path(train_dir)
    validation = Path(validation_dir)
    holdout = Path(holdout_dir)
    destination = Path(reports_dir)
    figures = destination / "figures"
    destination.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    train_selection = _read_json(train / "train_selection.json")
    candidate = _read_json(validation / "selected_candidate.json")
    result = _read_json(holdout / "HOLDOUT_COMPLETE.json")
    cv_figure = _save_cv_figure(train, figures)
    sensitivity_figure = _save_sensitivity_figure(holdout, figures)

    lines = [
        "# Empirical research report",
        "",
        "> This file is generated from content-addressed experiment artifacts. "
        "Engineering fixtures are not empirical evidence.",
        "",
        "## Stage status",
        "",
        f"- Train-only selection: {'complete' if train_selection else 'not run'}",
        f"- Validation selection: {'complete' if candidate else 'not run'}",
        f"- Frozen holdout: {'complete' if result else 'not run'}",
        "",
    ]
    if engineering_fixture:
        lines.extend(
            (
                "**ENGINEERING FIXTURE: the values below test mechanics and are not market "
                "evidence.**",
                "",
            )
        )
    if train_selection:
        lines.extend(
            (
                "## Train-only selection",
                "",
                f"Sessions: {train_selection['train_sessions']}. Selected ridge alphas by horizon: "
                f"`{json.dumps(train_selection['selected_alpha_by_horizon'], sort_keys=True)}`.",
                "",
            )
        )
        if cv_figure:
            lines.extend(("![Train ridge CV](figures/train_ridge_cv.png)", ""))
    if candidate:
        lines.extend(
            (
                "## Validation decision",
                "",
                f"Selected horizon: {candidate['selected_horizon_ms']} ms; ridge alpha: "
                f"{candidate['selected_alpha']}; execution safety margin: "
                f"{candidate['selected_safety_margin_ticks']} ticks.",
                "",
                f"Execution selection status: `{candidate['execution_selection_status']}`.",
                "",
            )
        )
    if result and not engineering_fixture:
        regression = dict(result["regression"])
        execution = dict(result["execution"])
        lines.extend(
            (
                "## Untouched holdout",
                "",
                f"The frozen candidate was evaluated across {result['holdout_sessions']} sessions. "
                f"Mean daily Spearman IC was {_fmt(regression['mean_daily_spearman_ic'])}; "
                f"mean MAE was {_fmt(regression['mean_mae_ticks'])} ticks.",
                "",
                f"The primary executable simulation produced {execution['trades']} trades, "
                f"gross P&L of ${_fmt(execution['gross_pnl_usd'], 2)}, explicit fees of "
                f"${_fmt(execution['explicit_fees_usd'], 2)}, and net P&L of "
                f"${_fmt(execution['net_pnl_usd'], 2)}.",
                "",
            )
        )
        if sensitivity_figure:
            lines.extend(("![Holdout sensitivity](figures/holdout_sensitivity.png)", ""))
    elif engineering_fixture:
        lines.extend(
            (
                "## Claims boundary",
                "",
                "A mechanical holdout was completed on generated books, but synthetic values are "
                "not evidence of prediction or trading performance.",
                "",
            )
        )
    else:
        lines.extend(
            (
                "## Claims boundary",
                "",
                "No holdout result exists yet, so this repository does not claim predictive or "
                "trading performance. The implemented and tested research system can be described; "
                "performance numbers cannot.",
                "",
            )
        )
    report_path = destination / "empirical_results.md"
    report_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")

    cv_lines = ["# CV evidence", ""]
    if result and not engineering_fixture:
        regression = dict(result["regression"])
        execution = dict(result["execution"])
        cv_lines.extend(
            (
                "Evidence-backed draft bullet:",
                "",
                "- Built a causal, execution-aware MBP-10 futures research pipeline with "
                f"day-clustered validation and a frozen {result['holdout_sessions']}-session "
                f"holdout; the selected {result['horizon_ms']} ms ridge signal achieved "
                f"{_fmt(regression['mean_daily_spearman_ic'])} mean daily Spearman IC and "
                f"${_fmt(execution['net_pnl_usd'], 2)} net simulated P&L after displayed-depth "
                "sweeps, fees and latency assumptions.",
                "",
                "Only use this bullet after reviewing the report and confirming the assumptions.",
            )
        )
    else:
        cv_lines.extend(
            (
                "Safe pre-results draft bullet:",
                "",
                "- Engineered a tested, causal MBP-10 futures research pipeline covering "
                "100–1,000 ms forecasts, chronological day-level model selection, delayed "
                "displayed-depth execution, explicit fees and frozen-holdout controls.",
                "",
                "Do not add performance figures until a real-data `HOLDOUT_COMPLETE.json` exists.",
            )
        )
    cv_path = destination / "cv_evidence.md"
    cv_path.write_text("\n".join(cv_lines).rstrip() + "\n", encoding="utf-8")
    return report_path, cv_path
