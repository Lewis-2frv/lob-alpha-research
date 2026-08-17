from __future__ import annotations

import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from lob_alpha.equity_config import load_equity_config
from lob_alpha.equity_data import (
    audit_optiver_csv,
    load_prepared_manifest,
    prepare_optiver_parquet,
    validate_optiver_frame,
)
from lob_alpha.equity_features import (
    build_equity_features,
    equity_model_feature_columns,
)
from lob_alpha.equity_fixture import make_synthetic_optiver
from lob_alpha.equity_models import (
    baseline_predictions,
    candidate_model_specs,
    expanding_date_folds,
    fit_model,
    fit_signed_imbalance_baseline,
)
from lob_alpha.equity_reporting import build_equity_report
from lob_alpha.equity_study import (
    HOLDOUT_ACKNOWLEDGEMENT,
    freeze_equity_candidate,
    run_equity_holdout_stage,
    run_equity_train_stage,
    run_equity_validation_stage,
)
from lob_alpha.equity_trading import (
    TradingRule,
    align_future_quotes,
    cluster_bootstrap_by_date,
    simulate_cross_sectional_trading,
    summarize_trading,
)
from lob_alpha.manifest import sha256_file, write_json
from lob_alpha.pipeline import write_table
from lob_alpha.safe_zip import extract_optiver_train_csv

ROOT = Path(__file__).resolve().parents[1]


class EquitySchemaAndFeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_equity_config(ROOT / "configs/equity_close_fixture.yaml")
        cls.raw = make_synthetic_optiver(cls.config)
        cls.features = build_equity_features(cls.raw, cls.config)

    def test_registered_real_split_is_explicit_and_substantial(self) -> None:
        config = load_equity_config(ROOT / "configs/equity_close.yaml")
        self.assertEqual(
            (
                config.splits.train_start,
                config.splits.train_end,
                config.splits.validation_start,
                config.splits.validation_end,
                config.splits.holdout_start,
                config.splits.holdout_end,
            ),
            (0, 329, 330, 404, 405, 480),
        )
        total = 481
        self.assertGreaterEqual(330 / total, 0.65)
        self.assertGreaterEqual(76 / total, 0.15)
        self.assertEqual(config.splits.split_for(405), "holdout")
        self.assertEqual(
            config.data.target_definition,
            "supplied_optiver_index_relative_60s_bps",
        )

    def test_schema_accepts_legitimate_auction_price_missingness(self) -> None:
        first_date = self.raw.loc[self.raw["date_id"].eq(0)].copy()
        self.assertTrue(first_date["far_price"].isna().any())
        self.assertTrue(first_date["near_price"].isna().any())
        first_date.loc[first_date.index[0], "target"] = np.nan
        validate_optiver_frame(first_date)

    def test_schema_rejects_malformed_rows(self) -> None:
        base = self.raw.loc[self.raw["date_id"].eq(0)].copy()
        mutations = {
            "duplicate row": lambda frame: frame.assign(
                row_id=frame["row_id"].mask(frame.index == frame.index[1], frame.iloc[0]["row_id"])
            ),
            "crossed quote": lambda frame: frame.assign(
                ask_price=frame["ask_price"].mask(
                    frame.index == frame.index[0], frame.iloc[0]["bid_price"] - 0.01
                )
            ),
            "negative size": lambda frame: frame.assign(
                bid_size=frame["bid_size"].mask(frame.index == frame.index[0], -1.0)
            ),
            "nonfinite target": lambda frame: frame.assign(
                target=frame["target"].mask(frame.index == frame.index[0], np.inf)
            ),
            "inconsistent time id": lambda frame: frame.assign(
                time_id=frame["time_id"].mask(
                    frame.index == frame.index[8], frame.iloc[0]["time_id"]
                )
            ),
            "out of order": lambda frame: pd.concat(
                [frame.iloc[[1, 0]], frame.iloc[2:]], ignore_index=True
            ),
            "off-grid time": lambda frame: frame.assign(
                seconds_in_bucket=frame["seconds_in_bucket"].mask(
                    frame.index == frame.index[0], 7
                )
            ),
            "nonnumeric optional price": lambda frame: frame.assign(
                far_price=frame["far_price"].astype(object).mask(
                    frame.index == frame.index[-1], "not-a-price"
                )
            ),
            "malformed row id": lambda frame: frame.assign(
                row_id=frame["row_id"].mask(frame.index == frame.index[0], "0_0_999")
            ),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name), self.assertRaises(ValueError):
                validate_optiver_frame(mutate(base.copy()))

    def test_target_and_future_quotes_are_excluded_from_features(self) -> None:
        columns = equity_model_feature_columns(self.config)
        self.assertNotIn("target", columns)
        self.assertFalse(any(column.startswith("future_") for column in columns))
        self.assertNotIn("stock_id", columns)

    def test_within_stock_date_lags_do_not_cross_boundaries(self) -> None:
        starts = self.features.loc[self.features["seconds_in_bucket"].eq(0)]
        self.assertTrue(starts["wap_return_10s_bps"].isna().all())
        row = self.features.loc[
            self.features[["date_id", "seconds_in_bucket", "stock_id"]]
            .eq([0, 10, 0])
            .all(axis=1)
        ].iloc[0]
        previous = self.features.loc[
            self.features[["date_id", "seconds_in_bucket", "stock_id"]]
            .eq([0, 0, 0])
            .all(axis=1)
        ].iloc[0]
        expected = (row["wap"] / previous["wap"] - 1.0) * 10_000.0
        self.assertAlmostEqual(row["wap_return_10s_bps"], expected)

    def test_future_perturbation_cannot_change_earlier_features(self) -> None:
        changed = self.raw.copy()
        later = changed["date_id"].eq(0) & changed["seconds_in_bucket"].gt(300)
        changed.loc[later, "imbalance_size"] *= 50
        changed.loc[later, "matched_size"] *= 10
        changed.loc[later, "wap"] += 0.00001
        rebuilt = build_equity_features(changed, self.config)
        earlier = self.features["date_id"].eq(0) & self.features["seconds_in_bucket"].le(300)
        assert_frame_equal(
            self.features.loc[earlier, equity_model_feature_columns(self.config)].reset_index(
                drop=True
            ),
            rebuilt.loc[earlier, equity_model_feature_columns(self.config)].reset_index(drop=True),
        )

    def test_future_perturbation_cannot_change_earlier_predictions(self) -> None:
        model = fit_model(
            self.config,
            {"model": "ridge", "alpha": 1.0},
            self.features.loc[self.features["date_id"].le(3)],
        )
        changed = self.raw.copy()
        later = changed["date_id"].eq(4) & changed["seconds_in_bucket"].gt(300)
        changed.loc[later, "imbalance_size"] *= 100
        changed.loc[later, "wap"] += 0.00001
        rebuilt = build_equity_features(changed, self.config)
        earlier = self.features["date_id"].eq(4) & self.features["seconds_in_bucket"].le(300)
        np.testing.assert_allclose(
            model.predict(self.features.loc[earlier]),
            model.predict(rebuilt.loc[earlier]),
        )

    def test_missing_timestamp_does_not_create_positional_exact_lag(self) -> None:
        frame = self.raw.loc[
            self.raw["date_id"].eq(0) & self.raw["stock_id"].eq(0)
        ].copy()
        frame = frame.loc[frame["seconds_in_bucket"].ne(10)]
        features = build_equity_features(frame, self.config)
        row = features.loc[features["seconds_in_bucket"].eq(60)].iloc[0]
        self.assertTrue(pd.isna(row["wap_return_60s_bps"]))

    def test_cross_sectional_features_are_isolated_by_time_id(self) -> None:
        changed = self.raw.copy()
        later_time = int(changed.loc[changed["seconds_in_bucket"].eq(400), "time_id"].iloc[0])
        changed.loc[changed["time_id"].eq(later_time), "imbalance_size"] *= 100
        rebuilt = build_equity_features(changed, self.config)
        earlier_time = int(changed.loc[changed["seconds_in_bucket"].eq(200), "time_id"].iloc[0])
        columns = [
            column
            for column in equity_model_feature_columns(self.config)
            if "_cs_" in column
        ]
        assert_frame_equal(
            self.features.loc[self.features["time_id"].eq(earlier_time), columns].reset_index(
                drop=True
            ),
            rebuilt.loc[rebuilt["time_id"].eq(earlier_time), columns].reset_index(drop=True),
        )

    def test_preprocessing_is_fitted_on_training_rows_only(self) -> None:
        train = self.features.loc[self.features["date_id"].le(3)].copy()
        train.loc[train.index[0], "target"] = np.nan
        validation = self.features.loc[self.features["date_id"].eq(4)].copy()
        validation["near_wap_gap_bps"] = 1_000_000.0
        validation["stock_id"] = 999
        model = fit_model(self.config, {"model": "ridge", "alpha": 1.0}, train)
        transformer = model.named_steps["preprocess"]
        numeric_columns = list(transformer.transformers_[0][2])
        index = numeric_columns.index("near_wap_gap_bps")
        statistic = transformer.named_transformers_["numeric"].named_steps["imputer"].statistics_[
            index
        ]
        self.assertAlmostEqual(statistic, train["near_wap_gap_bps"].median())
        categories = transformer.named_transformers_["stock_fixed_effect"].categories_[0]
        self.assertNotIn(999, categories)
        self.assertEqual(
            model.named_steps["model"].n_features_in_,
            transformer.transform(train).shape[1],
        )

    def test_stock_categorical_encoding_is_memory_bounded(self) -> None:
        train = self.features.loc[self.features["date_id"].le(3)].copy()
        ridge = fit_model(self.config, {"model": "ridge", "alpha": 1.0}, train)
        ridge_encoder = ridge.named_steps["preprocess"].named_transformers_[
            "stock_fixed_effect"
        ]
        self.assertTrue(ridge_encoder.sparse_output)
        nonlinear = fit_model(
            self.config,
            {
                "model": "hist_gradient_boosting",
                "learning_rate": 0.05,
                "max_leaf_nodes": 15,
                "max_iter": 2,
            },
            train,
        )
        nonlinear_encoder = nonlinear.named_steps["preprocess"].named_transformers_[
            "stock_fixed_effect"
        ]
        self.assertEqual(type(nonlinear_encoder).__name__, "OrdinalEncoder")
        transformed = nonlinear.named_steps["preprocess"].transform(train.iloc[:10])
        self.assertEqual(
            transformed.shape[1], len(equity_model_feature_columns(self.config)) + 1
        )

    def test_expanding_folds_are_strictly_chronological(self) -> None:
        folds = expanding_date_folds(range(7), minimum_train_dates=3, folds=2)
        self.assertEqual(len(folds), 2)
        for train_dates, validation_dates in folds:
            self.assertLess(max(train_dates), min(validation_dates))
            self.assertFalse(set(train_dates) & set(validation_dates))

    def test_optional_lightgbm_absence_is_explicit_and_nonfatal(self) -> None:
        config = load_equity_config(ROOT / "configs/equity_close.yaml")
        with patch.dict("sys.modules", {"lightgbm": None}):
            specifications = candidate_model_specs(config)
        self.assertTrue(specifications)
        self.assertFalse(any(item["model"] == "lightgbm" for item in specifications))

    def test_mandatory_baselines_are_exact(self) -> None:
        frame = pd.DataFrame(
            {"auction_imbalance_ratio": [-2.0, -1.0, 1.0, 2.0], "target": [-4.0, -2.0, 2.0, 4.0]}
        )
        slope = fit_signed_imbalance_baseline(frame)
        self.assertEqual(slope, 2.0)
        predictions = baseline_predictions(frame, imbalance_slope=slope)
        np.testing.assert_array_equal(predictions["zero"], np.zeros(4))
        np.testing.assert_allclose(predictions["signed_imbalance"], frame["target"])


def _execution_fixture() -> tuple[pd.DataFrame, np.ndarray]:
    rows = []
    for seconds, time_id in ((0, 100), (60, 101)):
        for stock_id in range(4):
            bid, ask = 99.0, 101.0
            if seconds == 60 and stock_id == 0:
                bid, ask = 97.0, 98.0
            if seconds == 60 and stock_id == 3:
                bid, ask = 102.0, 103.0
            rows.append(
                {
                    "stock_id": stock_id,
                    "date_id": 0,
                    "seconds_in_bucket": seconds,
                    "time_id": time_id,
                    "bid_price": bid,
                    "bid_size": 1_000.0,
                    "ask_price": ask,
                    "ask_size": 1_000.0,
                    "wap": (bid + ask) / 2,
                }
            )
    return pd.DataFrame(rows), np.asarray([-2.0, -1.0, 1.0, 2.0, 0.0, 0.0, 0.0, 0.0])


class EquityTradingTests(unittest.TestCase):
    def test_exact_60_second_quote_alignment(self) -> None:
        frame, _ = _execution_fixture()
        aligned = align_future_quotes(frame, horizon_seconds=60)
        selected = aligned.loc[
            aligned[["stock_id", "seconds_in_bucket"]].eq([3, 0]).all(axis=1)
        ].iloc[0]
        self.assertEqual(selected["future_bid_price"], 102.0)
        missing = align_future_quotes(
            frame.loc[~frame[["stock_id", "seconds_in_bucket"]].eq([3, 60]).all(axis=1)],
            horizon_seconds=60,
        )
        selected_missing = missing.loc[
            missing[["stock_id", "seconds_in_bucket"]].eq([3, 0]).all(axis=1)
        ].iloc[0]
        self.assertTrue(pd.isna(selected_missing["future_bid_price"]))

    def test_execution_arithmetic_and_spread_are_counted_once(self) -> None:
        frame, predictions = _execution_fixture()
        rule = TradingRule(0.25, 0.0, 250.0, 0.0, 0.5)
        legs, decisions = simulate_cross_sectional_trading(frame, predictions, rule)
        self.assertEqual(len(legs), 2)
        self.assertEqual(len(decisions), 1)
        short_gross = (99.0 - 98.0) / 99.0 * 10_000.0
        long_gross = (102.0 / 101.0 - 1.0) * 10_000.0
        expected_gross = (short_gross + long_gross) / 2.0
        self.assertAlmostEqual(decisions.iloc[0]["gross_return_bps"], expected_gross)
        self.assertAlmostEqual(decisions.iloc[0]["fee_bps"], 1.0)
        self.assertAlmostEqual(decisions.iloc[0]["net_return_bps"], expected_gross - 1.0)
        self.assertTrue(legs["spread_cost_bps"].gt(0).all())
        self.assertAlmostEqual(legs.loc[legs["side"].eq("long"), "weight"].sum(), 0.5)
        self.assertAlmostEqual(legs.loc[legs["side"].eq("short"), "weight"].sum(), 0.5)
        summary = summarize_trading(legs, decisions, repetitions=100, seed=7)
        self.assertAlmostEqual(
            summary["break_even_additional_total_cost_bps"],
            max(float(decisions.iloc[0]["net_return_bps"]), 0.0),
        )

    def test_one_sided_predictions_do_not_manufacture_long_short_trades(self) -> None:
        frame, _ = _execution_fixture()
        predictions = np.asarray([1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0])
        legs, decisions = simulate_cross_sectional_trading(
            frame,
            predictions,
            TradingRule(0.25, 0.0, 250.0, 0.0, 0.0),
        )
        self.assertTrue(legs.empty)
        self.assertTrue(decisions.empty)
        self.assertEqual(
            decisions.attrs["execution_filter_diagnostics"]["one_sided_time_ids"], 1
        )

    def test_tied_boundary_predictions_are_selected_deterministically(self) -> None:
        frame, _ = _execution_fixture()
        predictions = np.asarray([-1.0, -1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
        rule = TradingRule(0.25, 0.0, 250.0, 0.0, 0.0)
        first, _ = simulate_cross_sectional_trading(frame, predictions, rule)
        second, _ = simulate_cross_sectional_trading(frame, predictions, rule)
        self.assertEqual(first["stock_id"].tolist(), [0, 3])
        assert_frame_equal(first, second)

    def test_missing_or_crossed_future_quote_is_rejected(self) -> None:
        frame, predictions = _execution_fixture()
        frame = frame.loc[
            ~frame[["stock_id", "seconds_in_bucket"]].eq([3, 60]).all(axis=1)
        ].copy()
        predictions = np.delete(predictions, 7)
        legs, decisions = simulate_cross_sectional_trading(
            frame,
            predictions,
            TradingRule(0.25, 0.0, 250.0, 0.0, 0.0),
        )
        self.assertTrue(legs.empty)
        self.assertTrue(decisions.empty)
        self.assertGreater(
            decisions.attrs["execution_filter_diagnostics"][
                "missing_exact_future_quote_rows"
            ],
            0,
        )

    def test_overlapping_decision_interval_is_rejected(self) -> None:
        frame, predictions = _execution_fixture()
        with self.assertRaisesRegex(ValueError, "overlapping"):
            simulate_cross_sectional_trading(
                frame,
                predictions,
                TradingRule(0.25, 0.0, 250.0, 0.0, 0.0),
                decision_interval_seconds=50,
            )

    def test_date_cluster_bootstrap_is_deterministic(self) -> None:
        values = pd.Series([1.0, -0.5, 0.25, 2.0])
        first = cluster_bootstrap_by_date(values, repetitions=200, seed=7)
        second = cluster_bootstrap_by_date(values, repetitions=200, seed=7)
        self.assertEqual(first, second)
        self.assertEqual(first["dates"], 4)


class EquityAuditTests(unittest.TestCase):
    def test_metadata_only_audit_never_reads_target_values(self) -> None:
        config = load_equity_config(ROOT / "configs/equity_close_fixture.yaml")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "synthetic.csv"
            make_synthetic_optiver(config).to_csv(raw, index=False)
            calls = []
            original = pd.read_csv

            def recording_read_csv(*args, **kwargs):
                calls.append(dict(kwargs))
                return original(*args, **kwargs)

            with patch("lob_alpha.equity_data.pd.read_csv", side_effect=recording_read_csv):
                payload = audit_optiver_csv(
                    config,
                    input_path=raw,
                    output_path=root / "metadata.json",
                    metadata_only=True,
                )
            chunk_calls = [call for call in calls if "chunksize" in call]
            self.assertTrue(chunk_calls)
            self.assertTrue(all("target" not in call["usecols"] for call in chunk_calls))
            self.assertFalse(payload["target_values_read"])

    def test_incomplete_timestamp_is_reported_deterministically(self) -> None:
        config = load_equity_config(ROOT / "configs/equity_close_fixture.yaml")
        frame = make_synthetic_optiver(config)
        frame = frame.drop(frame.index[10]).reset_index(drop=True)
        frame.loc[frame.index[0], "target"] = np.nan
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "incomplete.csv"
            frame.to_csv(raw, index=False)
            payload = audit_optiver_csv(
                config,
                input_path=raw,
                output_path=root / "audit.json",
                metadata_only=False,
            )
        self.assertEqual(payload["time_ids_below_modal_stock_coverage"], 1)
        self.assertEqual(payload["rows_per_time_id_min"], 7)
        self.assertEqual(payload["rows_per_time_id_mode"], 8)
        self.assertEqual(payload["target_missing_rows"], 1)
        self.assertEqual(payload["target_available_rows"], len(frame) - 1)

    def test_missing_date_cannot_become_a_usable_partition(self) -> None:
        config = load_equity_config(ROOT / "configs/equity_close_fixture.yaml")
        frame = make_synthetic_optiver(config)
        frame = frame.loc[frame["date_id"].ne(config.data.expected_date_id_max)]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "missing-date.csv"
            frame.to_csv(raw, index=False)
            with self.assertRaisesRegex(ValueError, "date_id range"):
                audit_optiver_csv(
                    config,
                    input_path=raw,
                    output_path=root / "audit.json",
                    metadata_only=False,
                )
            self.assertFalse((root / "audit.json").exists())


class EquityWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.config = load_equity_config(ROOT / "configs/equity_close_fixture.yaml")
        cls.raw = cls.root / "synthetic_train.csv"
        make_synthetic_optiver(cls.config).to_csv(cls.raw, index=False)
        cls.audit = cls.root / "audit.json"
        audit_optiver_csv(
            cls.config,
            input_path=cls.raw,
            output_path=cls.audit,
            metadata_only=False,
        )
        cls.prepared = cls.root / "prepared"
        prepare_optiver_parquet(
            cls.config,
            input_path=cls.raw,
            audit_path=cls.audit,
            output_dir=cls.prepared,
        )
        cls.manifest = cls.prepared / "prepared_manifest.json"
        cls.train = cls.root / "train"
        cls.validation = cls.root / "validation"
        run_equity_train_stage(
            cls.config,
            manifest_path=cls.manifest,
            output_dir=cls.train,
        )
        run_equity_validation_stage(
            cls.config,
            manifest_path=cls.manifest,
            train_selection_path=cls.train / "train_selection.json",
            output_dir=cls.validation,
        )
        cls.frozen = cls.root / "frozen.json"
        freeze_equity_candidate(
            cls.config,
            manifest_path=cls.manifest,
            candidate_path=cls.validation / "selected_candidate.json",
            output_path=cls.frozen,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def test_preparation_preserves_early_missing_auction_period(self) -> None:
        first = pd.read_parquet(self.prepared / "date_id=0000.parquet")
        self.assertEqual(len(first), 8 * 60)
        early = first.loc[first["seconds_in_bucket"].lt(200)]
        self.assertFalse(early.empty)
        self.assertTrue(early["near_price_missing"].eq(1).all())
        candidate = json.loads(
            (self.validation / "selected_candidate.json").read_text(encoding="utf-8")
        )
        self.assertEqual(candidate["preprocessing_fit_partition"], "train_only")
        self.assertEqual(candidate["estimator_refit_partitions"], ["train", "validation"])

    def test_candidate_freeze_detects_hash_tampering(self) -> None:
        candidate = json.loads(
            (self.validation / "selected_candidate.json").read_text(encoding="utf-8")
        )
        candidate["model_sha256"] = "0" * 64
        tampered = self.root / "tampered_candidate.json"
        write_json(tampered, candidate)
        with self.assertRaisesRegex(OSError, "model_path"):
            freeze_equity_candidate(
                self.config,
                manifest_path=self.manifest,
                candidate_path=tampered,
                output_path=self.root / "must_not_freeze.json",
            )
        candidate = json.loads(
            (self.validation / "selected_candidate.json").read_text(encoding="utf-8")
        )
        candidate["feature_implementation_sha256"] = "0" * 64
        tampered_features = self.root / "tampered_features.json"
        write_json(tampered_features, candidate)
        with self.assertRaisesRegex(OSError, "feature implementation"):
            freeze_equity_candidate(
                self.config,
                manifest_path=self.manifest,
                candidate_path=tampered_features,
                output_path=self.root / "must_not_freeze_features.json",
            )

    def test_candidate_rejects_train_shortlist_mutation(self) -> None:
        selection = json.loads(
            (self.train / "train_selection.json").read_text(encoding="utf-8")
        )
        selection["shortlist"] = list(reversed(selection["shortlist"]))
        tampered_selection = self.train / "tampered_train_selection.json"
        write_json(tampered_selection, selection)
        candidate = json.loads(
            (self.validation / "selected_candidate.json").read_text(encoding="utf-8")
        )
        candidate["train_selection_path"] = str(tampered_selection.resolve())
        candidate["train_selection_sha256"] = sha256_file(tampered_selection)
        tampered_candidate = self.root / "candidate_with_tampered_shortlist.json"
        write_json(tampered_candidate, candidate)
        with self.assertRaisesRegex(OSError, "shortlist"):
            freeze_equity_candidate(
                self.config,
                manifest_path=self.manifest,
                candidate_path=tampered_candidate,
                output_path=self.root / "must_not_freeze_shortlist.json",
            )

    def test_development_manifest_loader_does_not_open_holdout_manifest(self) -> None:
        original = Path.read_text

        def guarded_read_text(path, *args, **kwargs):
            if Path(path).name == "holdout_manifest.json":
                raise AssertionError("development stage attempted to open holdout metadata")
            return original(path, *args, **kwargs)

        with patch.object(Path, "read_text", guarded_read_text):
            manifest = load_prepared_manifest(
                self.manifest,
                self.config,
                scope="development",
                verify_partitions=False,
            )
        self.assertEqual(manifest["scope"], "development")

    def test_frozen_execution_rule_mutation_fails_before_sealing(self) -> None:
        frozen = json.loads(self.frozen.read_text(encoding="utf-8"))
        frozen["trading_rule"]["fee_per_side_bps"] += 1.0
        tampered = self.root / "tampered_frozen.json"
        write_json(tampered, frozen)
        with self.assertRaisesRegex(OSError, "frozen candidate field"):
            run_equity_holdout_stage(
                self.config,
                manifest_path=self.manifest,
                frozen_candidate_path=tampered,
                output_dir=self.root / "tampered_holdout",
                acknowledge_one_shot=HOLDOUT_ACKNOWLEDGEMENT,
            )
        self.assertFalse((self.prepared / "HOLDOUT_STARTED.json").exists())

    def test_one_shot_holdout_and_synthetic_claim_gating(self) -> None:
        holdout = self.root / "one_shot_holdout"
        with self.assertRaisesRegex(ValueError, "acknowledgement"):
            run_equity_holdout_stage(
                self.config,
                manifest_path=self.manifest,
                frozen_candidate_path=self.frozen,
                output_dir=holdout,
                acknowledge_one_shot="yes",
            )
        self.assertFalse(holdout.exists())
        from lob_alpha import equity_study

        original_loader = equity_study.load_prepared_manifest

        def recording_loader(*args, **kwargs):
            if kwargs.get("scope") == "holdout":
                self.assertTrue((self.prepared / "HOLDOUT_STARTED.json").is_file())
            return original_loader(*args, **kwargs)

        with patch(
            "lob_alpha.equity_study.load_prepared_manifest",
            side_effect=recording_loader,
        ):
            result = run_equity_holdout_stage(
                self.config,
                manifest_path=self.manifest,
                frozen_candidate_path=self.frozen,
                output_dir=holdout,
                acknowledge_one_shot=HOLDOUT_ACKNOWLEDGEMENT,
            )
        self.assertFalse(result["selection_performed"])
        self.assertEqual(result["source_kind"], "synthetic")
        with self.assertRaises(FileExistsError):
            run_equity_holdout_stage(
                self.config,
                manifest_path=self.manifest,
                frozen_candidate_path=self.frozen,
                output_dir=self.root / "different_holdout_directory",
                acknowledge_one_shot=HOLDOUT_ACKNOWLEDGEMENT,
            )
        report, cv = build_equity_report(
            train_dir=self.train,
            validation_dir=self.validation,
            holdout_dir=holdout,
            reports_dir=self.root / "reports",
        )
        report_text = report.read_text(encoding="utf-8")
        self.assertIn("numerical values are suppressed", report_text)
        self.assertIn("generated Optiver-shaped engineering panel", report_text)
        cv_text = cv.read_text(encoding="utf-8")
        self.assertIn("Synthetic values and validation-only numbers", cv_text)
        measured = f"{float(result['predictive_metrics']['mae_bps']):.4f}"
        self.assertNotIn(measured, cv_text)
        self.assertNotIn(measured, report_text)
        self.assertFalse(any((self.root / "reports/figures").glob("*.png")))
        completion = holdout / "HOLDOUT_COMPLETE.json"
        original_completion = completion.read_text(encoding="utf-8")
        tampered_result = json.loads(original_completion)
        tampered_result["source_kind"] = "real"
        tampered_result["target_definition"] = "supplied_optiver_index_relative_60s_bps"
        tampered_result["claim_eligible_real_optiver"] = True
        tampered_result["predictive_metrics"]["mae_bps"] = 0.0
        write_json(completion, tampered_result)
        with self.assertRaisesRegex(OSError, "changed after anchoring"):
            build_equity_report(
                train_dir=self.train,
                validation_dir=self.validation,
                holdout_dir=holdout,
                reports_dir=self.root / "tampered_reports",
            )
        completion.write_text(original_completion, encoding="utf-8")


class EquityHoldoutCrashTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.config = load_equity_config(ROOT / "configs/equity_close_fixture.yaml")
        raw = cls.root / "synthetic.csv"
        make_synthetic_optiver(cls.config).to_csv(raw, index=False)
        audit = cls.root / "audit.json"
        audit_optiver_csv(
            cls.config,
            input_path=raw,
            output_path=audit,
            metadata_only=False,
        )
        cls.prepared = cls.root / "prepared"
        prepare_optiver_parquet(
            cls.config,
            input_path=raw,
            audit_path=audit,
            output_dir=cls.prepared,
        )
        cls.manifest = cls.prepared / "prepared_manifest.json"
        train = cls.root / "train"
        validation = cls.root / "validation"
        run_equity_train_stage(cls.config, manifest_path=cls.manifest, output_dir=train)
        run_equity_validation_stage(
            cls.config,
            manifest_path=cls.manifest,
            train_selection_path=train / "train_selection.json",
            output_dir=validation,
        )
        cls.frozen = cls.root / "frozen.json"
        freeze_equity_candidate(
            cls.config,
            manifest_path=cls.manifest,
            candidate_path=validation / "selected_candidate.json",
            output_path=cls.frozen,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def test_crash_after_seal_permanently_blocks_another_output_directory(self) -> None:
        first = self.root / "failed_holdout"
        with patch(
            "lob_alpha.equity_study.pd.read_parquet",
            side_effect=RuntimeError("simulated partition-read crash"),
        ) as reader:
            with self.assertRaisesRegex(RuntimeError, "simulated"):
                run_equity_holdout_stage(
                    self.config,
                    manifest_path=self.manifest,
                    frozen_candidate_path=self.frozen,
                    output_dir=first,
                    acknowledge_one_shot=HOLDOUT_ACKNOWLEDGEMENT,
                )
            self.assertEqual(reader.call_count, 1)
            with self.assertRaises(FileExistsError):
                run_equity_holdout_stage(
                    self.config,
                    manifest_path=self.manifest,
                    frozen_candidate_path=self.frozen,
                    output_dir=self.root / "second_output",
                    acknowledge_one_shot=HOLDOUT_ACKNOWLEDGEMENT,
                )
            self.assertEqual(reader.call_count, 1)
        self.assertTrue((self.prepared / "HOLDOUT_STARTED.json").is_file())
        self.assertTrue((first / "HOLDOUT_STARTED.json").is_file())
        self.assertFalse((first / "HOLDOUT_COMPLETE.json").exists())


class SafeZipTests(unittest.TestCase):
    def test_malicious_zip_path_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "malicious.zip"
            with zipfile.ZipFile(archive, "w") as bundle:
                bundle.writestr("../train.csv", "licensed rows")
            with self.assertRaisesRegex(ValueError, "unsafe ZIP"):
                extract_optiver_train_csv(archive, output_path=root / "train.csv")
            self.assertFalse((root / "train.csv").exists())

    def test_windows_unsafe_names_and_case_collisions_are_rejected(self) -> None:
        unsafe_members = (
            "..\\train.csv",
            "C:/train.csv",
            "//server/share/train.csv",
            "CON/train.csv",
            "folder/name:stream",
            "folder/trailing./train.csv",
        )
        for index, member in enumerate(unsafe_members):
            with self.subTest(member=member), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                archive = root / f"unsafe-{index}.zip"
                with zipfile.ZipFile(archive, "w") as bundle:
                    bundle.writestr(member, "data")
                with self.assertRaisesRegex(ValueError, "unsafe"):
                    extract_optiver_train_csv(archive, output_path=root / "train.csv")
                self.assertFalse((root / "train.csv").exists())

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "collision.zip"
            with zipfile.ZipFile(archive, "w") as bundle:
                bundle.writestr("folder/train.csv", "first")
                bundle.writestr("FOLDER/TRAIN.CSV", "second")
            with self.assertRaisesRegex(ValueError, "colliding"):
                extract_optiver_train_csv(archive, output_path=root / "train.csv")

    def test_zip_bomb_limits_fail_without_partial_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "ratio.zip"
            with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
                bundle.writestr("train.csv", b"0" * (1024 * 1024))
            output = root / "raw/train.csv"
            with self.assertRaisesRegex(ValueError, "compression ratio"):
                extract_optiver_train_csv(archive, output_path=output)
            self.assertFalse(output.exists())
            self.assertFalse(output.with_suffix(".csv.partial").exists())

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "members.zip"
            with zipfile.ZipFile(archive, "w") as bundle:
                bundle.writestr("train.csv", "data")
                bundle.writestr("extra.txt", "data")
            with (
                patch("lob_alpha.safe_zip.MAX_ZIP_MEMBERS", 1),
                self.assertRaisesRegex(ValueError, "member count"),
            ):
                extract_optiver_train_csv(archive, output_path=root / "train.csv")

    def test_safe_zip_extracts_only_train_and_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "competition.zip"
            with zipfile.ZipFile(archive, "w") as bundle:
                bundle.writestr("competition/train.csv", "stock_id,date_id\n0,0\n")
                bundle.writestr("competition/test.csv", "not extracted")
            output = extract_optiver_train_csv(archive, output_path=root / "raw/train.csv")
            self.assertEqual(output.read_text(encoding="utf-8"), "stock_id,date_id\n0,0\n")
            self.assertFalse((root / "raw/test.csv").exists())
            with self.assertRaises(FileExistsError):
                extract_optiver_train_csv(archive, output_path=output)


class EquitySerializationTests(unittest.TestCase):
    def test_gzip_table_serialization_has_a_stable_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "table.csv.gz"
            frame = pd.DataFrame({"value": [1.25, 2.5], "label": ["a", "b"]})
            write_table(frame, output)
            first = sha256_file(output)
            output.unlink()
            write_table(frame, output)
            self.assertEqual(sha256_file(output), first)
            self.assertFalse(output.with_name(output.name + ".partial").exists())


if __name__ == "__main__":
    unittest.main()
