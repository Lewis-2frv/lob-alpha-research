"""Compact nested-ZIP fixture for end-to-end FI-2010 safety rehearsals."""

from __future__ import annotations

import io
import zipfile
from dataclasses import replace
from pathlib import Path

import numpy as np

from .fi2010_config import FI2010Config
from .fi2010_data import audit_inner_archive, import_inner_archive
from .fi2010_reporting import build_fi2010_report
from .fi2010_study import (
    HOLDOUT_ACKNOWLEDGEMENT,
    freeze_and_refit_fi2010,
    run_fi2010_development,
    run_fi2010_holdout,
)
from .manifest import sha256_file


def _matrix_text(features: np.ndarray, labels: np.ndarray) -> bytes:
    matrix = np.vstack((features.T, labels.T))
    stream = io.StringIO()
    np.savetxt(stream, matrix, fmt="%.7g")
    return stream.getvalue().encode("ascii")


def _synthetic_matrix(seed: int, observations: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(observations, 144)).astype(np.float32)
    base = features[:, 0] + 0.35 * features[:, 1]
    labels = np.empty((observations, 5), dtype=np.int8)
    for horizon in range(5):
        noisy = base + rng.normal(scale=0.75 + 0.1 * horizon, size=observations)
        labels[:, horizon] = np.where(noisy > 0.35, 1, np.where(noisy < -0.35, 3, 2))
    return features, labels


def write_synthetic_nested_archive(path: str | Path) -> Path:
    """Create nine paired cumulative folds with the official 149-row orientation."""

    output = Path(path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite synthetic FI-2010 archive: {output}")
    maximum_train = 90 + 9 * 15
    train_features, train_labels = _synthetic_matrix(17, maximum_train)
    inner_stream = io.BytesIO()
    prefix = "BenchmarkDatasets/NoAuction/1.NoAuction_Zscore"
    with zipfile.ZipFile(inner_stream, "w", compression=zipfile.ZIP_DEFLATED) as inner:
        for fold in range(1, 10):
            train_count = 90 + fold * 15
            train_name = (
                f"{prefix}/NoAuction_Zscore_Training/"
                f"Train_Dst_NoAuction_ZScore_CF_{fold}.txt"
            )
            inner.writestr(
                train_name,
                _matrix_text(train_features[:train_count], train_labels[:train_count]),
            )
            test_features, test_labels = _synthetic_matrix(1_000 + fold, 36)
            test_name = (
                f"{prefix}/NoAuction_Zscore_Testing/"
                f"Test_Dst_NoAuction_ZScore_CF_{fold}.txt"
            )
            inner.writestr(test_name, _matrix_text(test_features, test_labels))
    temporary = output.with_suffix(output.suffix + ".partial")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as outer:
        outer.writestr(
            "published/BenchmarkDatasets/BenchmarkDatasets.zip",
            inner_stream.getvalue(),
        )
    temporary.replace(output)
    return output


def synthetic_config(config: FI2010Config, archive: str | Path, prepared_dir: Path) -> FI2010Config:
    archive_path = Path(archive)
    source = replace(
        config.source,
        outer_archive_size=archive_path.stat().st_size,
        outer_archive_sha256=sha256_file(archive_path),
    )
    data = replace(config.data, prepared_dir=str(prepared_dir))
    # The synthetic rehearsal verifies model-family plumbing and integrity mechanics, not
    # empirical tuning. Keep one setting per expensive nonlinear family so CI remains fast
    # while the real registered configuration retains the full bounded development grid.
    models = replace(
        config.models,
        numpy_ridge_alphas=(config.models.numpy_ridge_alphas[1],),
        numpy_softmax_epochs=min(config.models.numpy_softmax_epochs, 3),
        numpy_softmax_batch_size=min(config.models.numpy_softmax_batch_size, 256),
        sgd_alphas=(config.models.sgd_alphas[0],),
        lightgbm_learning_rates=(0.05,),
        lightgbm_num_leaves=(31,),
        lightgbm_estimators=min(config.models.lightgbm_estimators, 40),
        fallback_max_iter=min(config.models.fallback_max_iter, 30),
    )
    return replace(config, synthetic=True, source=source, data=data, models=models)


def run_synthetic_fi2010(config: FI2010Config, output_dir: str | Path) -> dict[str, object]:
    """Exercise import through anchored reporting; all outputs remain claim-ineligible."""

    root = Path(output_dir).resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"refusing to overwrite FI-2010 rehearsal: {root}")
    root.mkdir(parents=True, exist_ok=True)
    archive = write_synthetic_nested_archive(root / "synthetic-fi2010.zip")
    prepared = root / "prepared"
    fixture_config = synthetic_config(config, archive, prepared)
    imported = import_inner_archive(fixture_config, archive, prepared_dir=prepared)
    audited = audit_inner_archive(fixture_config, prepared_dir=prepared)
    development_dir = root / "development"
    development = run_fi2010_development(
        fixture_config,
        prepared_dir=prepared,
        output_dir=development_dir,
    )
    freeze_dir = root / "freeze"
    frozen = freeze_and_refit_fi2010(
        fixture_config,
        prepared_dir=prepared,
        development_results=development_dir / "development_results.json",
        output_dir=freeze_dir,
    )
    holdout_dir = root / "holdout"
    held_out = run_fi2010_holdout(
        fixture_config,
        prepared_dir=prepared,
        frozen_candidate=freeze_dir / "frozen_candidate.json",
        final_model_manifest=freeze_dir / "final_model_manifest.json",
        output_dir=holdout_dir,
        acknowledgement=HOLDOUT_ACKNOWLEDGEMENT,
    )
    report, evidence = build_fi2010_report(
        fixture_config,
        prepared_dir=prepared,
        development_results=development_dir / "development_results.json",
        freeze_dir=freeze_dir,
        holdout_dir=holdout_dir,
        output_dir=root / "report",
    )
    return {
        "stage": "fi2010_synthetic_rehearsal_complete",
        "claim_eligible": False,
        "archive": str(archive),
        "imported": imported,
        "audited": audited,
        "development_results_sha256": development["development_results_sha256"],
        "frozen": frozen,
        "holdout": held_out,
        "report": str(report),
        "evidence": str(evidence),
        "warning": "synthetic engineering fixture; never real CV evidence",
    }
