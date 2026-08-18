"""Anchored development, freeze/refit, and sealed one-shot FI-2010 holdout stages."""

from __future__ import annotations

import gc
import json
import os
import platform
import time
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

import joblib
import numpy as np
import sklearn

from .fi2010_config import FI2010Config
from .fi2010_data import (
    ANCHOR_FILENAME,
    CLAIM_FILENAME,
    SEAL_FILENAME,
    atomic_json,
    available_memory_bytes,
    is_cf9_test_member,
    member_from_identity,
    parse_fi2010_member,
    read_json,
    utc_now,
    verify_prepared_source,
)
from .fi2010_models import (
    FittedModel,
    aligned_probabilities,
    candidate_specifications,
    classification_metrics,
    directional_diagnostics,
    fit_classifier,
    serialized_model_size,
    specification_id,
)
from .manifest import sha256_file

HOLDOUT_ACKNOWLEDGEMENT = "RELEASE FI2010 CF9 HOLDOUT ONCE"


def runtime_versions() -> dict[str, str | None]:
    """Bind package versions that can change fitted models or holdout predictions."""

    try:
        import lightgbm
    except ImportError:
        lightgbm_version = None
    else:
        lightgbm_version = str(lightgbm.__version__)
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "joblib": joblib.__version__,
        "lightgbm": lightgbm_version,
    }


def _registered_member_path(
    config: FI2010Config, identity: dict[str, Any], *, split: str, fold: int
) -> str:
    path = str(identity.get("path", ""))
    expected_name = (
        f"Train_Dst_NoAuction_ZScore_CF_{fold}.txt"
        if split == "train"
        else f"Test_Dst_NoAuction_ZScore_CF_{fold}.txt"
    )
    if PurePosixPath(path).name != expected_name:
        raise ValueError(f"manifest does not identify the registered {split} CF_{fold} member")
    if f"/{config.data.representation}/" not in f"/{path}":
        raise ValueError("manifested FI-2010 member is outside the registered representation")
    return path


def _validate_development_semantics(
    config: FI2010Config, development: dict[str, Any]
) -> None:
    """Recompute the frozen decision from serialized development evidence."""

    expected_folds = list(config.data.development_folds)
    if development.get("development_folds") != expected_folds:
        raise ValueError("development fold declaration changed")
    if development.get("class_mapping") != {
        str(key): value for key, value in config.data.class_mapping.items()
    }:
        raise ValueError("development class mapping changed")
    if development.get("primary_horizon") != {
        "label_row": config.data.primary_label_row,
        "sampled_steps": config.data.horizon_sampled_steps[config.data.primary_label_row - 1],
        "underlying_events": config.data.horizon_underlying_events[
            config.data.primary_label_row - 1
        ],
    }:
        raise ValueError("development primary horizon changed")

    specifications = development.get("candidate_specifications")
    fold_results = development.get("fold_results")
    if not isinstance(specifications, list) or not isinstance(fold_results, list):
        raise ValueError("development candidate evidence is incomplete")
    registered_specifications, lightgbm_available = candidate_specifications(config)
    if development.get("lightgbm_available") is not lightgbm_available:
        raise ValueError("development LightGBM availability changed")
    if specifications != registered_specifications:
        raise ValueError("development candidate specifications changed")
    expected_ids = {specification_id(specification) for specification in specifications}
    if len(expected_ids) != len(specifications):
        raise ValueError("duplicate development candidate specifications")
    for specification in specifications:
        identity = specification_id(specification)
        candidate_results = [
            item for item in fold_results if item.get("specification_id") == identity
        ]
        folds = [int(item.get("fold", -1)) for item in candidate_results]
        if folds != expected_folds:
            raise ValueError("each development candidate must contain exactly CF_1-CF_8")
        for item in candidate_results:
            if item.get("specification") != specification:
                raise ValueError("development candidate specification/result mismatch")
            fold = int(item["fold"])
            if f"CF_{fold}.txt" not in str(item.get("train_member", "")):
                raise ValueError("development train member/fold mismatch")
            if f"CF_{fold}.txt" not in str(item.get("test_member", "")):
                raise ValueError("development test member/fold mismatch")
            if is_cf9_test_member(str(item.get("test_member", ""))):
                raise PermissionError("development results contain CF_9 test evidence")
    if {item.get("specification_id") for item in fold_results} != expected_ids:
        raise ValueError("development results contain an unregistered candidate")

    selected, ranking = select_development_candidate(fold_results, specifications)
    if development.get("model_ranking") != ranking:
        raise ValueError("development model ranking does not match fold evidence")
    threshold, threshold_candidates = select_confidence_threshold(
        config, fold_results, selected["specification_id"]
    )
    if development.get("threshold_candidates") != threshold_candidates:
        raise ValueError("development threshold ranking does not match fold evidence")
    candidate = development.get("selected_candidate", {})
    if candidate.get("specification") != selected["specification"]:
        raise ValueError("selected candidate does not match recomputed development ranking")
    if candidate.get("specification_id") != selected["specification_id"]:
        raise ValueError("selected candidate identifier changed")
    selection = candidate.get("selection", {})
    if not np.isclose(
        float(selection.get("mean_macro_f1", np.nan)), selected["mean_macro_f1"]
    ) or not np.isclose(
        float(selection.get("worst_fold_macro_f1", np.nan)), selected["worst_fold_macro_f1"]
    ):
        raise ValueError("selected candidate scores do not match development folds")
    rule = candidate.get("confidence_rule", {})
    if float(rule.get("threshold", np.nan)) != float(threshold["threshold"]):
        raise ValueError("frozen confidence threshold does not match development diagnostics")


def implementation_hashes() -> dict[str, str]:
    """Content-lock every module that can affect FI-2010 evidence."""

    root = Path(__file__).resolve().parent
    names = (
        "fi2010_config.py",
        "fi2010_data.py",
        "fi2010_models.py",
        "fi2010_study.py",
        "fi2010_reporting.py",
    )
    return {name: sha256_file(root / name) for name in names if (root / name).exists()}


def _empty_destination(path: str | Path, stage: str) -> Path:
    destination = Path(path).resolve()
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"refusing to overwrite {stage} artifacts: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    return destination


def _load_development_manifest(
    config: FI2010Config, prepared_dir: str | Path | None
) -> tuple[Path, dict[str, Any], Path]:
    root = Path(prepared_dir or config.data.prepared_dir).resolve()
    path = root / "development_manifest.json"
    manifest = read_json(path)
    if manifest.get("stage") != "fi2010_development_manifest":
        raise ValueError("not an FI-2010 development manifest")
    if manifest.get("config_sha256") != sha256_file(config.path):
        raise ValueError("FI-2010 configuration changed after the archive audit")
    inner, source = verify_prepared_source(config, root)
    if manifest.get("inner_archive_sha256") != source["inner_archive_sha256"]:
        raise ValueError("development manifest source hash mismatch")
    source_path = Path(manifest["source_manifest_path"]).resolve()
    if source_path != (root / "source_manifest.json").resolve():
        raise ValueError("development manifest points outside the prepared source directory")
    if sha256_file(source_path) != manifest["source_manifest_sha256"]:
        raise ValueError("source manifest changed after the archive audit")
    if manifest.get("representation") != config.data.representation:
        raise ValueError("development manifest representation changed")
    if manifest.get("feature_rows") != list(config.data.feature_rows):
        raise ValueError("development manifest feature declaration changed")
    if manifest.get("label_rows") != list(config.data.label_rows):
        raise ValueError("development manifest label declaration changed")
    if manifest.get("primary_label_row") != config.data.primary_label_row:
        raise ValueError("development manifest primary target changed")
    records = manifest.get("members", [])
    expected_count = 2 * len(config.data.development_folds) + 1
    if len(records) != expected_count:
        raise ValueError(
            "development manifest must contain exactly CF_1-CF_8 pairs plus Train_CF_9"
        )
    for fold in config.data.development_folds:
        for split in ("train", "test"):
            _registered_member_path(
                config,
                _record_for(manifest, fold=fold, split=split),
                split=split,
                fold=fold,
            )
    _registered_member_path(
        config,
        _record_for(manifest, fold=config.data.final_fold, split="train"),
        split="train",
        fold=config.data.final_fold,
    )
    if any(is_cf9_test_member(str(record["path"])) for record in records):
        raise PermissionError("development manifest must not contain the CF_9 test identity")
    return path, manifest, inner


def _record_for(
    manifest: dict[str, Any], *, fold: int, split: str
) -> dict[str, Any]:
    records = [
        record
        for record in manifest["members"]
        if int(record["fold"]) == fold and record["split"] == split
    ]
    if len(records) != 1:
        raise ValueError(f"expected exactly one manifested {split} CF_{fold} member")
    return records[0]


def _fold_result(
    config: FI2010Config,
    specification: dict[str, Any],
    *,
    fold: int,
    train: Any,
    test: Any,
) -> dict[str, Any]:
    train_target = train.primary_target(config.data.primary_label_row)
    test_target = test.primary_target(config.data.primary_label_row)
    fit_started = time.perf_counter()
    fitted = fit_classifier(specification, train.features, train_target, seed=config.seed)
    fit_seconds = time.perf_counter() - fit_started
    prediction_started = time.perf_counter()
    probabilities = aligned_probabilities(fitted, test.features)
    prediction_seconds = time.perf_counter() - prediction_started
    metrics = classification_metrics(test_target, probabilities)
    confidence = [
        directional_diagnostics(test_target, probabilities, threshold)
        for threshold in config.selection.confidence_thresholds
    ]
    alternate_horizon_label_agreement = []
    for index, (steps, events) in enumerate(
        zip(
            config.data.horizon_sampled_steps,
            config.data.horizon_underlying_events,
            strict=True,
        )
    ):
        horizon_metrics = classification_metrics(test.labels[:, index], probabilities)
        alternate_horizon_label_agreement.append(
            {
                "label_row": index + 1,
                "sampled_steps": steps,
                "underlying_events": events,
                "macro_f1": horizon_metrics["macro_f1"],
                "development_only": index + 1 != config.data.primary_label_row,
            }
        )
    result = {
        "fold": fold,
        "specification": specification,
        "specification_id": specification_id(specification),
        "train_member": train.member,
        "test_member": test.member,
        "train_observations": int(len(train_target)),
        "test_observations": int(len(test_target)),
        "train_class_counts": {
            str(label): int(np.sum(train_target == label)) for label in (1, 2, 3)
        },
        "train_only_class_weights": {
            str(label): value for label, value in fitted.class_weights.items()
        },
        "metrics": metrics,
        "directional_signal_diagnostics": confidence,
        "alternate_horizon_label_agreement": alternate_horizon_label_agreement,
        "efficiency": {
            "fit_seconds": fit_seconds,
            "prediction_seconds": prediction_seconds,
            "prediction_latency_microseconds_per_observation": (
                prediction_seconds * 1_000_000 / len(test_target)
            ),
            "prediction_throughput_observations_per_second": (
                len(test_target) / prediction_seconds if prediction_seconds else None
            ),
            "serialized_model_bytes": serialized_model_size(fitted),
        },
    }
    del probabilities, fitted
    return result


def select_development_candidate(
    fold_results: list[dict[str, Any]], specifications: list[dict[str, Any]]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Apply mean macro-F1, worst fold, complexity, then stable-ID selection."""

    aggregates = []
    for specification in specifications:
        identity = specification_id(specification)
        results = [item for item in fold_results if item["specification_id"] == identity]
        if not results:
            continue
        scores = [float(item["metrics"]["macro_f1"]) for item in results]
        aggregates.append(
            {
                "specification": specification,
                "specification_id": identity,
                "mean_macro_f1": float(np.mean(scores)),
                "worst_fold_macro_f1": float(np.min(scores)),
                "folds": len(scores),
                "complexity": int(specification["complexity"]),
            }
        )
    if not aggregates:
        raise ValueError("no complete development candidates were evaluated")
    ranking = sorted(
        aggregates,
        key=lambda item: (
            -item["mean_macro_f1"],
            -item["worst_fold_macro_f1"],
            item["complexity"],
            item["specification_id"],
        ),
    )
    return dict(ranking[0]), ranking


def select_confidence_threshold(
    config: FI2010Config,
    fold_results: list[dict[str, Any]],
    selected_specification_id: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Select a registered confidence threshold from CF_1-CF_8 diagnostics only."""

    selected_folds = [
        item for item in fold_results if item["specification_id"] == selected_specification_id
    ]
    candidates = []
    for threshold in config.selection.confidence_thresholds:
        diagnostics = [
            next(
                item
                for item in fold["directional_signal_diagnostics"]
                if item["threshold"] == threshold
            )
            for fold in selected_folds
        ]
        precisions = [
            float(item["directional_precision"])
            if item["directional_precision"] is not None
            else 0.0
            for item in diagnostics
        ]
        coverages = [float(item["directional_coverage"]) for item in diagnostics]
        candidates.append(
            {
                "threshold": threshold,
                "mean_directional_precision": float(np.mean(precisions)),
                "worst_fold_directional_precision": float(np.min(precisions)),
                "mean_directional_coverage": float(np.mean(coverages)),
                "mean_abstention_rate": float(1.0 - np.mean(coverages)),
                "coverage_eligible": float(np.mean(coverages))
                >= config.selection.minimum_directional_coverage,
                "folds": len(diagnostics),
            }
        )
    eligible = [item for item in candidates if item["coverage_eligible"]]
    if not eligible:
        raise ValueError("no registered threshold meets minimum development coverage")
    ranking = sorted(
        eligible,
        key=lambda item: (
            -item["mean_directional_precision"],
            -item["worst_fold_directional_precision"],
            -item["mean_directional_coverage"],
            item["threshold"],
        ),
    )
    return dict(ranking[0]), candidates


def run_fi2010_development(
    config: FI2010Config,
    *,
    prepared_dir: str | Path | None = None,
    output_dir: str | Path = "artifacts/fi2010/development",
) -> dict[str, Any]:
    """Evaluate matching cumulative-train/test members independently for CF_1-CF_8."""

    destination = _empty_destination(output_dir, "FI-2010 development")
    manifest_path, manifest, inner_path = _load_development_manifest(config, prepared_dir)
    specifications, lightgbm_available = candidate_specifications(config)
    fold_results: list[dict[str, Any]] = []
    peak_array_bytes = 0
    started = time.perf_counter()
    with zipfile.ZipFile(inner_path) as bundle:
        for fold in config.data.development_folds:
            train_record = _record_for(manifest, fold=fold, split="train")
            test_record = _record_for(manifest, fold=fold, split="test")
            train_member = member_from_identity(bundle, train_record)
            test_member = member_from_identity(bundle, test_record)
            train = parse_fi2010_member(bundle, train_member)
            test = parse_fi2010_member(bundle, test_member)
            peak_array_bytes = max(
                peak_array_bytes,
                train.features.nbytes
                + train.labels.nbytes
                + test.features.nbytes
                + test.labels.nbytes,
            )
            for specification in specifications:
                fold_results.append(
                    _fold_result(
                        config,
                        specification,
                        fold=fold,
                        train=train,
                        test=test,
                    )
                )
            del train, test
            gc.collect()
    selected, model_ranking = select_development_candidate(fold_results, specifications)
    threshold, threshold_candidates = select_confidence_threshold(
        config, fold_results, selected["specification_id"]
    )
    candidate = {
        "specification": selected["specification"],
        "specification_id": selected["specification_id"],
        "selection": {
            "mean_macro_f1": selected["mean_macro_f1"],
            "worst_fold_macro_f1": selected["worst_fold_macro_f1"],
            "tie_breakers": [
                config.selection.first_tie_breaker,
                config.selection.second_tie_breaker,
                "stable_specification_id",
            ],
        },
        "confidence_rule": {
            "threshold": threshold["threshold"],
            "up": "p(class=1) >= threshold and greater than p(class=3)",
            "down": "p(class=3) >= threshold and greater than p(class=1)",
            "otherwise": "abstain",
            "tie_policy": "abstain",
            "selected_from_folds": list(config.data.development_folds),
        },
    }
    payload = {
        "stage": "fi2010_anchored_development",
        "created_utc": utc_now(),
        "claim_eligible": bool(manifest["claim_eligible"] and not config.synthetic),
        "holdout_status": "CF_9 test payload untouched",
        "cf9_test_payload_opened": False,
        "protocol": "paired anchored folds; cumulative training members never concatenated",
        "development_folds": list(config.data.development_folds),
        "primary_horizon": {
            "label_row": config.data.primary_label_row,
            "sampled_steps": config.data.horizon_sampled_steps[config.data.primary_label_row - 1],
            "underlying_events": config.data.horizon_underlying_events[
                config.data.primary_label_row - 1
            ],
        },
        "class_mapping": {str(key): value for key, value in config.data.class_mapping.items()},
        "development_manifest_path": str(manifest_path),
        "development_manifest_sha256": sha256_file(manifest_path),
        "source_manifest_sha256": manifest["source_manifest_sha256"],
        "config_path": str(config.path),
        "config_sha256": sha256_file(config.path),
        "implementation_hashes": implementation_hashes(),
        "runtime_versions": runtime_versions(),
        "lightgbm_available": lightgbm_available,
        "candidate_specifications": specifications,
        "fold_results": fold_results,
        "model_ranking": model_ranking,
        "threshold_candidates": threshold_candidates,
        "selected_candidate": candidate,
        "resource_observations": {
            "wall_seconds": time.perf_counter() - started,
            "maximum_live_input_array_bytes": peak_array_bytes,
            "memory_measurement_scope": (
                "exact NumPy input-array bytes, not operating-system peak RAM"
            ),
            "available_physical_memory_bytes_at_completion": available_memory_bytes(),
            "fold_release_policy": (
                "train, test, probabilities and estimators released after each fold"
            ),
        },
        "interpretation": "predictive classification and directional signal diagnostics only",
        "executable_performance_claimed": False,
    }
    output = destination / "development_results.json"
    atomic_json(output, payload)
    payload["development_results_path"] = str(output)
    payload["development_results_sha256"] = sha256_file(output)
    return payload


def _save_joblib_atomic(path: Path, estimator: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    if path.exists() or temporary.exists():
        raise FileExistsError(f"refusing to overwrite final fitted model: {path}")
    try:
        joblib.dump(estimator, temporary, compress=3)
        with temporary.open("r+b") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise


def freeze_and_refit_fi2010(
    config: FI2010Config,
    *,
    prepared_dir: str | Path | None = None,
    development_results: str | Path = "artifacts/fi2010/development/development_results.json",
    output_dir: str | Path = "artifacts/fi2010/freeze",
) -> dict[str, Any]:
    """Freeze the CF_1-CF_8 choice, then refit once using Train_CF_9 only."""

    destination = _empty_destination(output_dir, "FI-2010 freeze/refit")
    manifest_path, manifest, inner_path = _load_development_manifest(config, prepared_dir)
    development_path = Path(development_results).resolve()
    development = read_json(development_path)
    if development.get("stage") != "fi2010_anchored_development":
        raise ValueError("not an FI-2010 development result")
    if development.get("config_sha256") != sha256_file(config.path):
        raise ValueError("configuration changed after development")
    if development.get("development_manifest_sha256") != sha256_file(manifest_path):
        raise ValueError("development manifest changed after model selection")
    if development.get("implementation_hashes") != implementation_hashes():
        raise ValueError("FI-2010 implementation changed after development")
    if development.get("runtime_versions") != runtime_versions():
        raise ValueError("FI-2010 runtime dependencies changed after development")
    if development.get("cf9_test_payload_opened") is not False:
        raise PermissionError("development evidence does not prove CF_9 test isolation")
    _validate_development_semantics(config, development)
    holdout_manifest_sha256 = manifest.get("holdout_manifest_sha256")
    if not isinstance(holdout_manifest_sha256, str) or len(holdout_manifest_sha256) != 64:
        raise ValueError("development audit did not bind the holdout manifest")
    root = Path(prepared_dir or config.data.prepared_dir).resolve()
    if sha256_file(root / "holdout_manifest.json") != holdout_manifest_sha256:
        raise ValueError("holdout manifest changed after the target-blind audit")
    frozen = {
        "stage": "fi2010_frozen_candidate",
        "created_utc": utc_now(),
        "claim_eligible": bool(development["claim_eligible"] and not config.synthetic),
        "candidate": development["selected_candidate"],
        "feature_declaration": {
            "representation": config.data.representation,
            "feature_rows": list(config.data.feature_rows),
            "snapshot_sequence_model": False,
        },
        "primary_horizon": development["primary_horizon"],
        "class_mapping": development["class_mapping"],
        "preprocessing": "publisher Z-score representation; fold-local StandardScaler for SGD only",
        "source_manifest_sha256": manifest["source_manifest_sha256"],
        "holdout_manifest_sha256": holdout_manifest_sha256,
        "development_manifest_sha256": sha256_file(manifest_path),
        "development_results_sha256": sha256_file(development_path),
        "config_sha256": sha256_file(config.path),
        "implementation_hashes": implementation_hashes(),
        "runtime_versions": runtime_versions(),
        "holdout_status": "CF_9 test payload untouched",
    }
    frozen_path = destination / "frozen_candidate.json"
    atomic_json(frozen_path, frozen)
    train9_record = _record_for(manifest, fold=config.data.final_fold, split="train")
    started = time.perf_counter()
    with zipfile.ZipFile(inner_path) as bundle:
        train9_member = member_from_identity(bundle, train9_record)
        train9 = parse_fi2010_member(bundle, train9_member)
        target = train9.primary_target(config.data.primary_label_row)
        fitted = fit_classifier(
            frozen["candidate"]["specification"],
            train9.features,
            target,
            seed=config.seed,
        )
        train9_observations = int(len(target))
        train9_array_bytes = int(train9.features.nbytes + train9.labels.nbytes)
        class_weights = {str(key): value for key, value in fitted.class_weights.items()}
        del train9, target
    model_path = destination / "final_model.joblib"
    _save_joblib_atomic(model_path, fitted)
    del fitted
    gc.collect()
    model_manifest = {
        "stage": "fi2010_final_refit",
        "created_utc": utc_now(),
        "claim_eligible": frozen["claim_eligible"],
        "frozen_candidate_path": str(frozen_path),
        "frozen_candidate_sha256": sha256_file(frozen_path),
        "source_manifest_sha256": manifest["source_manifest_sha256"],
        "development_results_sha256": sha256_file(development_path),
        "config_sha256": sha256_file(config.path),
        "implementation_hashes": implementation_hashes(),
        "runtime_versions": runtime_versions(),
        "candidate_specification_id": frozen["candidate"]["specification_id"],
        "primary_label_row": config.data.primary_label_row,
        "class_mapping": frozen["class_mapping"],
        "model_path": str(model_path),
        "model_sha256": sha256_file(model_path),
        "train_member": train9_record,
        "train_observations": train9_observations,
        "train_only_class_weights": class_weights,
        "fit_wall_seconds": time.perf_counter() - started,
        "input_array_bytes": train9_array_bytes,
        "memory_measurement_scope": "exact NumPy input-array bytes, not operating-system peak RAM",
        "cf9_test_payload_opened": False,
        "next_action": "stop; final holdout remains sealed",
    }
    final_path = destination / "final_model_manifest.json"
    atomic_json(final_path, model_manifest)
    return {
        "frozen_candidate": str(frozen_path),
        "frozen_candidate_sha256": sha256_file(frozen_path),
        "final_model_manifest": str(final_path),
        "final_model_manifest_sha256": sha256_file(final_path),
        "cf9_test_payload_opened": False,
        "holdout_status": "sealed and untouched",
    }


def _exclusive_claim(path: Path, payload: dict[str, Any]) -> None:
    """Create an irreversible same-source claim before atomic seal promotion."""

    encoded = (json.dumps(payload, sort_keys=True, allow_nan=False) + "\n").encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        written = os.write(descriptor, encoded)
        if written != len(encoded):
            raise OSError("short write while claiming the one-shot holdout")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def run_fi2010_holdout(
    config: FI2010Config,
    *,
    prepared_dir: str | Path | None = None,
    frozen_candidate: str | Path = "artifacts/fi2010/freeze/frozen_candidate.json",
    final_model_manifest: str | Path = "artifacts/fi2010/freeze/final_model_manifest.json",
    output_dir: str | Path = "artifacts/fi2010/holdout",
    acknowledgement: str,
) -> dict[str, Any]:
    """Evaluate Test_CF_9 once, with a durable source-bound pre-open seal."""

    if acknowledgement != HOLDOUT_ACKNOWLEDGEMENT:
        raise PermissionError(f"acknowledgement must exactly equal {HOLDOUT_ACKNOWLEDGEMENT!r}")
    root = Path(prepared_dir or config.data.prepared_dir).resolve()
    destination = _empty_destination(output_dir, "FI-2010 holdout")
    inner_path, source = verify_prepared_source(config, root)
    holdout_path = root / "holdout_manifest.json"
    holdout = read_json(holdout_path)
    if holdout.get("stage") != "fi2010_holdout_manifest":
        raise ValueError("not an FI-2010 holdout manifest")
    if holdout.get("config_sha256") != sha256_file(config.path):
        raise ValueError("FI-2010 configuration changed after audit")
    if holdout.get("inner_archive_sha256") != source["inner_archive_sha256"]:
        raise ValueError("holdout source binding changed")
    frozen_path = Path(frozen_candidate).resolve()
    frozen = read_json(frozen_path)
    model_manifest_path = Path(final_model_manifest).resolve()
    model_manifest = read_json(model_manifest_path)
    if frozen.get("stage") != "fi2010_frozen_candidate":
        raise ValueError("not an FI-2010 frozen candidate")
    if frozen.get("source_manifest_sha256") != sha256_file(root / "source_manifest.json"):
        raise ValueError("frozen candidate source binding changed")
    if frozen.get("config_sha256") != sha256_file(config.path):
        raise ValueError("configuration changed after candidate freeze")
    if frozen.get("implementation_hashes") != implementation_hashes():
        raise ValueError("implementation changed after candidate freeze")
    if frozen.get("runtime_versions") != runtime_versions():
        raise ValueError("runtime dependencies changed after candidate freeze")
    if frozen.get("holdout_manifest_sha256") != sha256_file(holdout_path):
        raise ValueError("holdout manifest changed after the target-blind audit")
    if holdout.get("source_manifest_sha256") != sha256_file(root / "source_manifest.json"):
        raise ValueError("holdout manifest source binding changed")
    if holdout.get("representation") != config.data.representation:
        raise ValueError("holdout manifest representation changed")
    _registered_member_path(
        config,
        holdout["member"],
        split="test",
        fold=config.data.final_fold,
    )
    if model_manifest.get("frozen_candidate_sha256") != sha256_file(frozen_path):
        raise ValueError("final model was not fitted from this frozen candidate")
    if model_manifest.get("stage") != "fi2010_final_refit":
        raise ValueError("not an FI-2010 final-refit manifest")
    if model_manifest.get("source_manifest_sha256") != frozen.get("source_manifest_sha256"):
        raise ValueError("final model source binding changed")
    if model_manifest.get("development_results_sha256") != frozen.get(
        "development_results_sha256"
    ):
        raise ValueError("final model development binding changed")
    if model_manifest.get("config_sha256") != frozen.get("config_sha256"):
        raise ValueError("final model configuration binding changed")
    if model_manifest.get("implementation_hashes") != frozen.get("implementation_hashes"):
        raise ValueError("final model implementation binding changed")
    if model_manifest.get("runtime_versions") != frozen.get("runtime_versions"):
        raise ValueError("final model runtime binding changed")
    if model_manifest.get("candidate_specification_id") != frozen["candidate"][
        "specification_id"
    ]:
        raise ValueError("final model candidate binding changed")
    if model_manifest.get("primary_label_row") != config.data.primary_label_row:
        raise ValueError("final model target binding changed")
    if model_manifest.get("class_mapping") != frozen.get("class_mapping"):
        raise ValueError("final model class mapping changed")
    _registered_member_path(
        config,
        model_manifest["train_member"],
        split="train",
        fold=config.data.final_fold,
    )
    model_path = Path(model_manifest["model_path"])
    if model_path.resolve() != (model_manifest_path.parent / "final_model.joblib").resolve():
        raise ValueError("final model manifest points outside the frozen artifact directory")
    if sha256_file(model_path) != model_manifest["model_sha256"]:
        raise ValueError("final fitted model binary changed")
    fitted = joblib.load(model_path)
    if not isinstance(fitted, FittedModel):
        raise ValueError("final model artifact is not an FI-2010 fitted model")
    if fitted.specification != frozen["candidate"]["specification"]:
        raise ValueError("final fitted model specification changed after freeze")
    manifested_weights = {
        int(key): float(value) for key, value in model_manifest["train_only_class_weights"].items()
    }
    if fitted.class_weights != manifested_weights:
        raise ValueError("final fitted model class weights changed after refit")
    binding = {
        "stage": "fi2010_cf9_holdout_started",
        "created_utc": utc_now(),
        "acknowledgement": HOLDOUT_ACKNOWLEDGEMENT,
        "source_manifest_sha256": sha256_file(root / "source_manifest.json"),
        "holdout_manifest_sha256": sha256_file(holdout_path),
        "holdout_member": holdout["member"],
        "frozen_candidate_sha256": sha256_file(frozen_path),
        "final_model_manifest_sha256": sha256_file(model_manifest_path),
        "final_model_sha256": model_manifest["model_sha256"],
        "config_sha256": sha256_file(config.path),
        "implementation_hashes": implementation_hashes(),
        "runtime_versions": runtime_versions(),
    }
    claim_path = root / CLAIM_FILENAME
    seal_path = root / SEAL_FILENAME
    if claim_path.exists() or seal_path.exists():
        raise FileExistsError("FI-2010 CF_9 holdout has already been claimed; no repeat is allowed")
    _exclusive_claim(claim_path, binding)
    atomic_json(seal_path, binding)
    with zipfile.ZipFile(inner_path) as bundle:
        test_member = member_from_identity(bundle, holdout["member"])
        test = parse_fi2010_member(bundle, test_member, allow_cf9_test=True)
        target = test.primary_target(config.data.primary_label_row)
        probabilities = aligned_probabilities(fitted, test.features)
    threshold = float(frozen["candidate"]["confidence_rule"]["threshold"])
    result = {
        "stage": "fi2010_cf9_holdout_complete",
        "created_utc": utc_now(),
        "claim_eligible": bool(frozen["claim_eligible"] and not config.synthetic),
        "one_shot": True,
        "seal_path": str(seal_path),
        "seal_sha256": sha256_file(seal_path),
        "member": holdout["member"],
        "observations": int(len(target)),
        "primary_metrics": classification_metrics(target, probabilities),
        "directional_signal_diagnostics": directional_diagnostics(
            target, probabilities, threshold
        ),
        "interpretation": "predictive classification and directional signal diagnostics only",
        "executable_performance_claimed": False,
    }
    result_path = destination / "holdout_result.json"
    atomic_json(result_path, result)
    anchor = {
        "stage": "fi2010_cf9_holdout_completion_anchor",
        "created_utc": utc_now(),
        "source_manifest_sha256": sha256_file(root / "source_manifest.json"),
        "seal_path": str(seal_path),
        "seal_sha256": sha256_file(seal_path),
        "outputs": {
            str(result_path): sha256_file(result_path),
        },
    }
    anchor_path = root / ANCHOR_FILENAME
    atomic_json(anchor_path, anchor)
    del fitted, test, target, probabilities
    return {
        "result": str(result_path),
        "result_sha256": sha256_file(result_path),
        "completion_anchor": str(anchor_path),
        "completion_anchor_sha256": sha256_file(anchor_path),
        "one_shot_complete": True,
    }
