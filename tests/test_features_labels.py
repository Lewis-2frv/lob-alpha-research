from __future__ import annotations

import unittest
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from lob_alpha.config import load_config
from lob_alpha.features import build_features, model_feature_columns
from lob_alpha.fixture import make_mbp10_fixture
from lob_alpha.labels import build_labels
from lob_alpha.sampling import sample_decision_states
from lob_alpha.schema import canonicalize_mbp10

ROOT = Path(__file__).resolve().parents[1]


class FeatureAndLabelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_config(ROOT / "configs/base.yaml")
        self.events = canonicalize_mbp10(make_mbp10_fixture(periods=300))
        self.states = sample_decision_states(
            self.events, date(2026, 3, 16), self.config.session
        )

    def test_level_one_formulas_match_hand_calculation(self) -> None:
        features = build_features(
            self.events,
            self.states,
            self.config.features,
            tick_size=0.25,
            decision_grid_ms=100,
        )
        row = features.iloc[0]
        bid = float(row["bid_px_00"])
        ask = float(row["ask_px_00"])
        bid_size = float(row["bid_sz_00"])
        ask_size = float(row["ask_sz_00"])
        expected_qi = (bid_size - ask_size) / (bid_size + ask_size)
        expected_microprice = (ask * bid_size + bid * ask_size) / (bid_size + ask_size)
        expected_displacement = (expected_microprice - (bid + ask) / 2.0) / 0.25
        self.assertAlmostEqual(row["queue_imbalance_l1"], expected_qi)
        self.assertAlmostEqual(row["microprice_displacement_ticks"], expected_displacement)

    def test_future_events_cannot_change_past_features(self) -> None:
        cutoff = self.events.loc[149, "ts_recv"]
        original = build_features(
            self.events,
            self.states,
            self.config.features,
            tick_size=0.25,
            decision_grid_ms=100,
        )
        perturbed_events = self.events.copy()
        future = perturbed_events["ts_recv"] > cutoff
        perturbed_events.loc[future, "bid_sz_00"] *= 100
        perturbed_events.loc[future, "ask_sz_00"] = 1
        perturbed_states = sample_decision_states(
            perturbed_events, date(2026, 3, 16), self.config.session
        )
        perturbed = build_features(
            perturbed_events,
            perturbed_states,
            self.config.features,
            tick_size=0.25,
            decision_grid_ms=100,
        )
        columns = model_feature_columns(self.config.features)
        original_past = original.loc[original["decision_time"] <= cutoff, columns]
        perturbed_past = perturbed.loc[perturbed["decision_time"] <= cutoff, columns]
        pd.testing.assert_frame_equal(
            original_past.reset_index(drop=True), perturbed_past.reset_index(drop=True)
        )

    def test_labels_never_use_events_after_target_time(self) -> None:
        features = build_features(
            self.events,
            self.states,
            self.config.features,
            tick_size=0.25,
            decision_grid_ms=100,
        )
        labels = build_labels(
            self.events,
            features,
            self.config.labels.horizons_ms,
            tick_size=0.25,
            maximum_age_ms=500,
        )
        for horizon in self.config.labels.horizons_ms:
            valid = labels[f"future_midpoint_{horizon}ms"].notna()
            self.assertTrue(
                (
                    labels.loc[valid, f"label_source_time_{horizon}ms"]
                    <= labels.loc[valid, f"label_target_time_{horizon}ms"]
                ).all()
            )

    def test_truncated_future_labels_become_missing(self) -> None:
        features = build_features(
            self.events,
            self.states,
            self.config.features,
            tick_size=0.25,
            decision_grid_ms=100,
        )
        labels = build_labels(
            self.events,
            features,
            (1000,),
            tick_size=0.25,
            maximum_age_ms=100,
        )
        self.assertTrue(np.isnan(labels.iloc[-1]["target_1000ms_ticks"]))


if __name__ == "__main__":
    unittest.main()

