from __future__ import annotations

import unittest
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from lob_alpha.config import load_config
from lob_alpha.features import model_feature_columns
from lob_alpha.fixture import make_mbp10_fixture
from lob_alpha.models import fit_ridge, score_regression
from lob_alpha.pipeline import process_session

ROOT = Path(__file__).resolve().parents[1]


class PipelineAndModelTests(unittest.TestCase):
    def test_pre_session_events_do_not_enter_features(self) -> None:
        config = load_config(ROOT / "configs/base.yaml")
        session = make_mbp10_fixture(periods=300)
        pre_session = session.iloc[:5].copy()
        pre_session["ts_recv"] = pre_session["ts_recv"] - np.timedelta64(2, "s")
        pre_session["ts_event"] = pre_session["ts_event"] - np.timedelta64(2, "s")
        pre_session["bid_sz_00"] = 1_000_000
        pre_session["ask_sz_00"] = 1
        combined = pd.concat([pre_session, session], ignore_index=True)
        clean_result = process_session(
            session,
            config,
            session_date=date(2026, 3, 16),
            tick_size=0.25,
        )
        combined_result = process_session(
            combined,
            config,
            session_date=date(2026, 3, 16),
            tick_size=0.25,
        )
        columns = model_feature_columns(config.features)
        pd.testing.assert_frame_equal(
            clean_result.data[columns], combined_result.data[columns]
        )

    def test_fixture_runs_end_to_end_and_model_scores(self) -> None:
        config = load_config(ROOT / "configs/base.yaml")
        result = process_session(
            make_mbp10_fixture(periods=600),
            config,
            session_date=date(2026, 3, 16),
            tick_size=0.25,
        )
        self.assertTrue(result.quality.accepted)
        self.assertGreater(len(result.data), 50)
        self.assertTrue(result.data["split"].eq("train").all())
        features = model_feature_columns(config.features)
        split_at = int(len(result.data) * 0.7)
        model = fit_ridge(
            result.data.iloc[:split_at],
            feature_columns=features,
            target_column="target_500ms_ticks",
            alpha=1.0,
        )
        predictions, metrics = score_regression(
            model,
            result.data.iloc[split_at:],
            feature_columns=features,
            target_column="target_500ms_ticks",
        )
        self.assertEqual(len(predictions), metrics.rows)
        self.assertTrue(np.isfinite(metrics.mae_ticks))
        self.assertTrue(np.isfinite(metrics.rmse_ticks))

    def test_microsecond_and_nanosecond_inputs_produce_identical_features(self) -> None:
        config = load_config(ROOT / "configs/base.yaml")
        events_ns = make_mbp10_fixture(periods=600)
        events_us = events_ns.copy()
        events_us["ts_recv"] = pd.Series(events_us["ts_recv"].array.as_unit("us"))
        events_us["ts_event"] = pd.Series(events_us["ts_event"].array.as_unit("us"))
        result_ns = process_session(
            events_ns,
            config,
            session_date=date(2026, 3, 16),
            tick_size=0.25,
        )
        result_us = process_session(
            events_us,
            config,
            session_date=date(2026, 3, 16),
            tick_size=0.25,
        )
        columns = model_feature_columns(config.features)
        pd.testing.assert_frame_equal(
            result_ns.data[columns],
            result_us.data[columns],
            check_dtype=False,
        )


if __name__ == "__main__":
    unittest.main()
