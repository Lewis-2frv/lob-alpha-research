from __future__ import annotations

import io
import json
import shutil
import tempfile
import unittest
import zipfile
from dataclasses import replace
from pathlib import Path
from unittest import mock

import numpy as np

from lob_alpha.fi2010_config import load_fi2010_config
from lob_alpha.fi2010_data import (
    EXPECTED_ROWS,
    audit_inner_archive,
    import_inner_archive,
    parse_fi2010_member,
    verify_outer_archive,
)
from lob_alpha.fi2010_fixture import (
    run_synthetic_fi2010,
    synthetic_config,
    write_synthetic_nested_archive,
)
from lob_alpha.fi2010_models import (
    ManualLiquidityPressureClassifier,
    NumpyDiagonalLDAClassifier,
    NumpyRidgeMulticlassClassifier,
    NumpySoftmaxClassifier,
    candidate_specifications,
    directional_diagnostics,
    training_class_weights,
)
from lob_alpha.fi2010_reporting import build_fi2010_report, publish_fi2010_portfolio
from lob_alpha.fi2010_study import (
    CLAIM_FILENAME,
    HOLDOUT_ACKNOWLEDGEMENT,
    SEAL_FILENAME,
    freeze_and_refit_fi2010,
    run_fi2010_development,
    run_fi2010_holdout,
    select_confidence_threshold,
    select_development_candidate,
)
from lob_alpha.manifest import sha256_file


def matrix_bytes(
    observations: int = 4,
    *,
    rows: int = EXPECTED_ROWS,
    bad_feature: float | None = None,
    bad_label: float | None = None,
) -> tuple[bytes, np.ndarray]:
    matrix = np.arange(rows * observations, dtype=np.float32).reshape(rows, observations)
    if rows >= EXPECTED_ROWS:
        matrix[144:, :] = 2
        matrix[144, :] = np.asarray([1, 2, 3, 1][:observations])
    if bad_feature is not None:
        matrix[0, 0] = bad_feature
    if bad_label is not None:
        matrix[144, 0] = bad_label
    stream = io.StringIO()
    np.savetxt(stream, matrix, fmt="%.7g")
    return stream.getvalue().encode("ascii"), matrix


class FI2010DataTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.base_config = load_fi2010_config("configs/fi2010.yaml")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def fixture(self) -> tuple[Path, object]:
        archive = write_synthetic_nested_archive(self.root / "fi.zip")
        config = synthetic_config(self.base_config, archive, self.root / "prepared")
        return archive, config

    def test_registered_rows_horizon_and_classes(self) -> None:
        config = self.base_config
        self.assertEqual(config.data.feature_rows, (1, 144))
        self.assertEqual(config.data.label_rows, (145, 149))
        self.assertEqual(config.data.primary_label_row, 4)
        self.assertEqual(config.data.class_mapping, {1: "up", 2: "stationary", 3: "down"})

    def test_tuned_candidate_universe_retains_manual_ladder_and_audited_hgb(self) -> None:
        specifications, lightgbm_available = candidate_specifications(self.base_config)
        models = [item["model"] for item in specifications]
        for manual in (
            "manual_liquidity_pressure",
            "numpy_diagonal_lda",
            "numpy_ridge_multiclass",
            "numpy_softmax",
        ):
            self.assertIn(manual, models)
            self.assertLess(models.index(manual), models.index("hist_gradient_boosting_fallback"))
        self.assertEqual(models.count("numpy_ridge_multiclass"), 3)
        self.assertIn("hist_gradient_boosting_fallback", models)
        self.assertEqual(models.count("hist_gradient_boosting_fallback"), 1)
        if lightgbm_available:
            lightgbm = [item for item in specifications if item["model"] == "lightgbm_multiclass"]
            self.assertEqual(len(lightgbm), 9)
            self.assertEqual(
                {item["learning_rate"] for item in lightgbm},
                {0.03, 0.05, 0.08},
            )
            self.assertEqual({item["num_leaves"] for item in lightgbm}, {15, 31, 63})
            self.assertTrue(all(item["n_estimators"] == 180 for item in lightgbm))

    def test_directional_semantics_follow_publisher_class_mapping(self) -> None:
        target = np.asarray([1, 3, 2], dtype=np.int8)
        probabilities = np.asarray(
            [
                [0.80, 0.10, 0.10],
                [0.10, 0.10, 0.80],
                [0.20, 0.60, 0.20],
            ],
            dtype=np.float64,
        )
        diagnostics = directional_diagnostics(target, probabilities, 0.70)
        self.assertEqual(diagnostics["signals"], 2)
        self.assertEqual(diagnostics["directional_precision"], 1.0)
        self.assertEqual(diagnostics["up_precision"], 1.0)
        self.assertEqual(diagnostics["down_precision"], 1.0)

    def test_manual_liquidity_pressure_rule_has_expected_direction_and_abstention(self) -> None:
        features = np.zeros((3, 144), dtype=np.float32)
        # Positive standardised bid depth and negative ask depth -> upward pressure.
        features[0, np.arange(3, 40, 4)] = 2.0
        features[0, np.arange(1, 40, 4)] = -2.0
        features[0, 85] = -2.0
        # Reverse the book-depth proxy for downward pressure.
        features[1, np.arange(3, 40, 4)] = -2.0
        features[1, np.arange(1, 40, 4)] = 2.0
        features[1, 85] = 2.0
        target = np.asarray([1, 3, 2], dtype=np.int8)
        model = ManualLiquidityPressureClassifier().fit(features, target)
        np.testing.assert_array_equal(model.predict(features), target)
        probabilities = model.predict_proba(features)
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)

    def test_from_scratch_statistical_models_produce_finite_multiclass_probabilities(self) -> None:
        rng = np.random.default_rng(41)
        target = np.repeat(np.asarray([1, 2, 3], dtype=np.int8), 30)
        features = rng.normal(scale=0.6, size=(len(target), 144)).astype(np.float32)
        features[target == 1, 0] += 2.0
        features[target == 3, 0] -= 2.0
        models = (
            NumpyDiagonalLDAClassifier(shrinkage=0.1),
            NumpyRidgeMulticlassClassifier(alpha=1.0),
            NumpySoftmaxClassifier(epochs=3, batch_size=32, random_state=9),
        )
        for model in models:
            with self.subTest(model=type(model).__name__):
                model.fit(features, target)
                probabilities = model.predict_proba(features[:12])
                self.assertEqual(probabilities.shape, (12, 3))
                self.assertTrue(np.isfinite(probabilities).all())
                np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-10)

    def test_outer_archive_requires_exact_size_and_sha256(self) -> None:
        archive, config = self.fixture()
        verified = verify_outer_archive(config, archive)
        self.assertEqual(verified["archive_sha256"], sha256_file(archive))
        with self.assertRaisesRegex(ValueError, "size mismatch"):
            verify_outer_archive(
                replace(
                    config,
                    source=replace(
                        config.source,
                        outer_archive_size=archive.stat().st_size + 1,
                    ),
                ),
                archive,
            )
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            verify_outer_archive(
                replace(config, source=replace(config.source, outer_archive_sha256="0" * 64)),
                archive,
            )

    def test_unsafe_nested_zip_is_rejected(self) -> None:
        inner_stream = io.BytesIO()
        with zipfile.ZipFile(inner_stream, "w") as inner:
            inner.writestr("../escape.txt", b"unsafe")
        archive = self.root / "unsafe.zip"
        with zipfile.ZipFile(archive, "w") as outer:
            outer.writestr(
                "published/BenchmarkDatasets/BenchmarkDatasets.zip",
                inner_stream.getvalue(),
            )
        config = synthetic_config(self.base_config, archive, self.root / "unsafe-prepared")
        import_inner_archive(config, archive)
        with self.assertRaisesRegex(ValueError, "unsafe ZIP member"):
            audit_inner_archive(config)

    def test_parser_transposes_features_and_selects_primary_horizon(self) -> None:
        content, original = matrix_bytes()
        archive = self.root / "member.zip"
        name = "safe/Train_Dst_NoAuction_ZScore_CF_1.txt"
        with zipfile.ZipFile(archive, "w") as bundle:
            bundle.writestr(name, content)
        with zipfile.ZipFile(archive) as bundle:
            parsed = parse_fi2010_member(bundle, bundle.getinfo(name))
        self.assertEqual(parsed.features.shape, (4, 144))
        self.assertEqual(parsed.labels.shape, (4, 5))
        self.assertEqual(parsed.features[2, 7], original[7, 2])
        np.testing.assert_array_equal(parsed.primary_target(4), parsed.labels[:, 3])

    def test_parser_rejects_wrong_rows_invalid_features_and_invalid_labels(self) -> None:
        cases = (
            (matrix_bytes(rows=148)[0], "149 rows"),
            (matrix_bytes(bad_feature=np.nan)[0], "NaN or infinity"),
            (matrix_bytes(bad_label=1.5)[0], "integral"),
            (matrix_bytes(bad_label=4)[0], r"\{1,2,3\}"),
        )
        for index, (content, message) in enumerate(cases):
            with self.subTest(index=index):
                archive = self.root / f"invalid-{index}.zip"
                with zipfile.ZipFile(archive, "w") as bundle:
                    bundle.writestr("safe/member.txt", content)
                with (
                    zipfile.ZipFile(archive) as bundle,
                    self.assertRaisesRegex(ValueError, message),
                ):
                    parse_fi2010_member(bundle, bundle.getinfo("safe/member.txt"))

    def test_cf9_guard_runs_before_zipfile_open(self) -> None:
        content, _ = matrix_bytes()
        archive = self.root / "holdout.zip"
        name = "safe/Test_Dst_NoAuction_ZScore_CF_9.txt"
        with zipfile.ZipFile(archive, "w") as bundle:
            bundle.writestr(name, content)
        with (
            zipfile.ZipFile(archive) as bundle,
            mock.patch.object(bundle, "open", side_effect=AssertionError("must not open")),
            self.assertRaisesRegex(PermissionError, "sealed"),
        ):
            parse_fi2010_member(bundle, bundle.getinfo(name))

    def test_audit_opens_no_cf9_test_payload(self) -> None:
        archive, config = self.fixture()
        import_inner_archive(config, archive)
        opened: list[str] = []
        original = zipfile.ZipFile.open

        def tracking(bundle: zipfile.ZipFile, name: object, *args: object, **kwargs: object):
            opened.append(name.filename if isinstance(name, zipfile.ZipInfo) else str(name))
            return original(bundle, name, *args, **kwargs)

        with mock.patch.object(zipfile.ZipFile, "open", tracking):
            result = audit_inner_archive(config)
        self.assertFalse(result["cf9_test_payload_opened"])
        self.assertFalse(
            any(path.endswith("Test_Dst_NoAuction_ZScore_CF_9.txt") for path in opened)
        )
        self.assertTrue(
            any(path.endswith("Train_Dst_NoAuction_ZScore_CF_9.txt") for path in opened)
        )


class FI2010WorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.config = load_fi2010_config("configs/fi2010.yaml")
        cls.summary = run_synthetic_fi2010(cls.config, cls.root / "complete")
        cls.complete = cls.root / "complete"

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def development(self) -> dict[str, object]:
        return json.loads(
            (self.complete / "development" / "development_results.json").read_text(
                encoding="utf-8"
            )
        )

    def test_cumulative_folds_are_paired_and_never_concatenated(self) -> None:
        development = self.development()
        self.assertEqual(
            development["protocol"],
            "paired anchored folds; cumulative training members never concatenated",
        )
        selected = development["selected_candidate"]["specification_id"]
        results = [
            result
            for result in development["fold_results"]
            if result["specification_id"] == selected
        ]
        self.assertEqual([item["fold"] for item in results], list(range(1, 9)))
        self.assertEqual(
            [item["train_observations"] for item in results],
            list(range(105, 211, 15)),
        )
        for item in results:
            self.assertIn(f"CF_{item['fold']}.txt", item["train_member"])
            self.assertIn(f"CF_{item['fold']}.txt", item["test_member"])

    def test_class_weights_use_training_labels_only(self) -> None:
        train = np.asarray([1, 1, 1, 2, 3], dtype=np.int8)
        first = training_class_weights(train)
        second = training_class_weights(train.copy())
        self.assertEqual(first, second)
        self.assertEqual(first[1], 5 / 9)
        self.assertEqual(first[2], 5 / 3)
        self.assertEqual(first[3], 5 / 3)

    def test_model_selection_is_deterministic_with_registered_ties(self) -> None:
        simple = {"model": "simple", "complexity": 1}
        complex_model = {"model": "complex", "complexity": 3}
        folds = []
        for fold in range(1, 9):
            for specification in (complex_model, simple):
                folds.append(
                    {
                        "specification_id": json.dumps(
                            specification, sort_keys=True, separators=(",", ":")
                        ),
                        "metrics": {"macro_f1": 0.5},
                        "fold": fold,
                    }
                )
        selected, _ = select_development_candidate(folds, [complex_model, simple])
        self.assertEqual(selected["specification"], simple)

    def test_freeze_recomputes_selected_candidate_before_refit(self) -> None:
        prepared = self.complete / "prepared"
        config = synthetic_config(
            self.config,
            self.complete / "synthetic-fi2010.zip",
            prepared,
        )
        original_path = self.complete / "development" / "development_results.json"
        development = json.loads(original_path.read_text(encoding="utf-8"))
        development["selected_candidate"]["selection"]["mean_macro_f1"] += 0.1
        tampered = self.complete / "tampered-development.json"
        tampered.write_text(
            json.dumps(development, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        with self.assertRaisesRegex(ValueError, "scores do not match"):
            freeze_and_refit_fi2010(
                config,
                prepared_dir=prepared,
                development_results=tampered,
                output_dir=self.complete / "tampered-freeze",
            )

    def test_confidence_threshold_uses_development_diagnostics_only(self) -> None:
        selected_id = "selected"
        folds = []
        for _fold in range(1, 9):
            folds.append(
                {
                    "specification_id": selected_id,
                    "directional_signal_diagnostics": [
                        {
                            "threshold": threshold,
                            "directional_precision": 0.5 + threshold / 10,
                            "directional_coverage": 0.25,
                        }
                        for threshold in self.config.selection.confidence_thresholds
                    ],
                }
            )
        selected, candidates = select_confidence_threshold(self.config, folds, selected_id)
        self.assertEqual(selected["threshold"], 0.8)
        self.assertTrue(all(item["folds"] == 8 for item in candidates))

    def test_synthetic_outputs_are_claim_ineligible_and_make_no_executable_claim(self) -> None:
        development = self.development()
        evidence = json.loads(
            (self.complete / "report" / "fi2010_evidence.json").read_text(encoding="utf-8")
        )
        report = (self.complete / "report" / "fi2010_evidence.md").read_text(
            encoding="utf-8"
        )
        self.assertFalse(development["claim_eligible"])
        self.assertFalse(evidence["claim_eligible"])
        self.assertFalse(evidence["executable_performance_claimed"])
        for prohibited in ("sharpe", "profitability", "transaction-cost", "p&l"):
            self.assertNotIn(prohibited, report.lower())

    def test_report_bundle_contains_portfolio_metrics_figures_and_cv_summary(self) -> None:
        report_dir = self.complete / "report"
        metrics = json.loads((report_dir / "portfolio_metrics.json").read_text(encoding="utf-8"))
        self.assertEqual(metrics["stage"], "fi2010_portfolio_metrics")
        self.assertEqual(metrics["development"]["folds"], 8)
        self.assertIsNotNone(metrics["holdout"])
        self.assertTrue((report_dir / "cv_summary.md").is_file())
        for name in (
            "development_macro_f1_by_fold.png",
            "model_comparison.png",
            "confidence_precision_coverage.png",
            "development_vs_holdout.png",
        ):
            self.assertGreater((report_dir / name).stat().st_size, 1000)

    def test_portfolio_publisher_is_fail_closed_but_testable_on_synthetic(self) -> None:
        report_dir = self.complete / "report"
        repo = self.root / "portfolio-repo"
        repo.mkdir()
        (repo / "README.md").write_text(
            "# Test\n\n## Current status\n\nSynthetic.\n", encoding="utf-8"
        )
        with self.assertRaisesRegex(PermissionError, "claim-eligible"):
            publish_fi2010_portfolio(report_dir, repository_root=repo)
        payload = publish_fi2010_portfolio(
            report_dir,
            repository_root=repo,
            require_claim_eligible=False,
            require_holdout=True,
        )
        self.assertTrue(payload["holdout_reported"])
        published = repo / "docs" / "results" / "fi2010"
        self.assertTrue((published / "README.md").is_file())
        self.assertTrue((published / "portfolio_metrics.json").is_file())
        readme = (repo / "README.md").read_text(encoding="utf-8")
        self.assertIn("FI2010_RESULTS_START", readme)
        self.assertIn("One-shot CF_9 macro-F1", readme)

    def test_completed_holdout_is_anchored_and_output_path_cannot_bypass(self) -> None:
        prepared = self.complete / "prepared"
        config = synthetic_config(
            self.config,
            self.complete / "synthetic-fi2010.zip",
            prepared,
        )
        with mock.patch("lob_alpha.fi2010_study.parse_fi2010_member") as parser:
            with self.assertRaisesRegex(FileExistsError, "already been claimed"):
                run_fi2010_holdout(
                    config,
                    prepared_dir=prepared,
                    frozen_candidate=self.complete / "freeze" / "frozen_candidate.json",
                    final_model_manifest=self.complete
                    / "freeze"
                    / "final_model_manifest.json",
                    output_dir=self.complete / "alternate-holdout",
                    acknowledgement=HOLDOUT_ACKNOWLEDGEMENT,
                )
            parser.assert_not_called()
        with self.assertRaisesRegex(FileExistsError, "release began"):
            audit_inner_archive(config, prepared_dir=prepared)
        result = self.complete / "holdout" / "holdout_result.json"
        original = result.read_bytes()
        result.write_bytes(original + b" ")
        try:
            with self.assertRaisesRegex(ValueError, "SHA-256 changed"):
                build_fi2010_report(
                    config,
                    prepared_dir=prepared,
                    development_results=self.complete
                    / "development"
                    / "development_results.json",
                    freeze_dir=self.complete / "freeze",
                    holdout_dir=self.complete / "holdout",
                    output_dir=self.complete / "tampered-report",
                )
        finally:
            result.write_bytes(original)

    def test_mutations_fail_before_access_and_crash_leaves_durable_seal(self) -> None:
        root = self.root / "crash"
        root.mkdir()
        config_path = root / "fi2010.yaml"
        shutil.copy2("configs/fi2010.yaml", config_path)
        base = load_fi2010_config(config_path)
        archive = write_synthetic_nested_archive(root / "source.zip")
        prepared = root / "prepared"
        config = synthetic_config(base, archive, prepared)
        import_inner_archive(config, archive)
        opened: list[str] = []
        original_open = zipfile.ZipFile.open

        def tracking(bundle: zipfile.ZipFile, name: object, *args: object, **kwargs: object):
            opened.append(name.filename if isinstance(name, zipfile.ZipInfo) else str(name))
            return original_open(bundle, name, *args, **kwargs)

        with mock.patch.object(zipfile.ZipFile, "open", tracking):
            audit_inner_archive(config)
            run_fi2010_development(config, output_dir=root / "development")
            freeze_and_refit_fi2010(
                config,
                development_results=root / "development" / "development_results.json",
                output_dir=root / "freeze",
            )
        self.assertFalse(
            any(path.endswith("Test_Dst_NoAuction_ZScore_CF_9.txt") for path in opened)
        )
        kwargs = {
            "prepared_dir": prepared,
            "frozen_candidate": root / "freeze" / "frozen_candidate.json",
            "final_model_manifest": root / "freeze" / "final_model_manifest.json",
            "output_dir": root / "holdout",
            "acknowledgement": HOLDOUT_ACKNOWLEDGEMENT,
        }

        def tamper_and_reject(path: Path, message: str) -> None:
            original = path.read_bytes()
            path.write_bytes(original + b"\n")
            try:
                with self.assertRaisesRegex(ValueError, message):
                    run_fi2010_holdout(config, **kwargs)
                self.assertFalse((prepared / CLAIM_FILENAME).exists())
            finally:
                path.write_bytes(original)

        tamper_and_reject(config_path, "configuration changed")
        tamper_and_reject(root / "freeze" / "frozen_candidate.json", "not fitted")
        tamper_and_reject(prepared / "holdout_manifest.json", "holdout manifest changed")
        model_manifest = json.loads(
            (root / "freeze" / "final_model_manifest.json").read_text(encoding="utf-8")
        )
        tamper_and_reject(Path(model_manifest["model_path"]), "model binary changed")
        tamper_and_reject(prepared / "BenchmarkDatasets.zip", "size changed")
        with self.assertRaisesRegex(PermissionError, "exactly equal"):
            run_fi2010_holdout(config, **{**kwargs, "acknowledgement": "yes"})
        self.assertFalse((prepared / CLAIM_FILENAME).exists())
        with (
            mock.patch(
                "lob_alpha.fi2010_study.parse_fi2010_member",
                side_effect=RuntimeError("simulated crash after seal"),
            ),
            self.assertRaisesRegex(RuntimeError, "simulated crash"),
        ):
            run_fi2010_holdout(config, **kwargs)
        self.assertTrue((prepared / CLAIM_FILENAME).is_file())
        self.assertTrue((prepared / SEAL_FILENAME).is_file())
        with mock.patch("lob_alpha.fi2010_study.parse_fi2010_member") as parser:
            with self.assertRaisesRegex(FileExistsError, "already been claimed"):
                run_fi2010_holdout(
                    config,
                    **{**kwargs, "output_dir": root / "bypass-output"},
                )
            parser.assert_not_called()
        with self.assertRaisesRegex(ValueError, "claimed or partially written"):
            build_fi2010_report(
                config,
                prepared_dir=prepared,
                development_results=root / "development" / "development_results.json",
                freeze_dir=root / "freeze",
                holdout_dir=root / "holdout",
                output_dir=root / "crash-report",
            )


if __name__ == "__main__":
    unittest.main()
