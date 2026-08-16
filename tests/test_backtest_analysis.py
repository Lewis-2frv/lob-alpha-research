from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from lob_alpha.analysis import (
    apply_quantile_analysis,
    cluster_bootstrap_mean,
    daily_information_coefficient,
    fit_quantile_edges,
    summarize_trades,
)
from lob_alpha.backtest import simulate_marketable_strategy
from lob_alpha.fixture import make_mbp10_fixture


class BacktestAndAnalysisTests(unittest.TestCase):
    def test_strategy_is_nonoverlapping_and_timestamp_causal(self) -> None:
        events = make_mbp10_fixture(periods=500)
        decisions = pd.DataFrame(
            {
                "decision_time": events.iloc[::5]["ts_recv"].to_numpy(),
                "prediction_ticks": np.where(np.arange(len(events.iloc[::5])) % 2, 5.0, -5.0),
            }
        )
        trades = simulate_marketable_strategy(
            events,
            decisions,
            prediction_column="prediction_ticks",
            horizon_ms=500,
            latency_ms=10,
            quantity=1,
            tick_size=0.25,
            multiplier=50.0,
            fee_per_contract_per_side_usd=2.5,
            safety_margin_ticks=0.0,
            maximum_quote_age_ms=100,
        )
        self.assertGreater(len(trades), 2)
        self.assertTrue((trades["entry_source_time"] <= trades["entry_target_time"]).all())
        self.assertTrue((trades["exit_source_time"] <= trades["exit_target_time"]).all())
        previous_exit = trades["exit_target_time"].shift(1)
        self.assertTrue(
            (
                trades.loc[previous_exit.notna(), "decision_time"] >= previous_exit.dropna()
            ).all()
        )

    def test_daily_ic_and_cluster_bootstrap(self) -> None:
        frame = pd.DataFrame(
            {
                "session_date": ["d1"] * 5 + ["d2"] * 5,
                "signal": list(range(5)) + list(range(5)),
                "target": list(range(5)) + list(reversed(range(5))),
            }
        )
        daily = daily_information_coefficient(
            frame, signal_column="signal", target_column="target"
        )
        self.assertAlmostEqual(daily.iloc[0]["spearman_ic"], 1.0)
        self.assertAlmostEqual(daily.iloc[1]["spearman_ic"], -1.0)
        interval = cluster_bootstrap_mean(daily["spearman_ic"], repetitions=100, seed=7)
        self.assertAlmostEqual(interval.estimate, 0.0)
        self.assertEqual(interval.clusters, 2)

    def test_quantile_edges_are_fitted_then_applied(self) -> None:
        training = pd.Series(np.arange(100, dtype=float))
        edges = fit_quantile_edges(training, bins=10)
        evaluation = pd.DataFrame({"signal": np.arange(100), "target": np.arange(100) / 10})
        table = apply_quantile_analysis(
            evaluation, signal_column="signal", target_column="target", edges=edges
        )
        self.assertEqual(table["rows"].sum(), 100)
        self.assertGreater(table.iloc[-1]["mean_target_ticks"], table.iloc[0]["mean_target_ticks"])

    def test_empty_trade_summary_is_explicit(self) -> None:
        summary = summarize_trades(pd.DataFrame())
        self.assertEqual(summary["trades"], 0)
        self.assertTrue(np.isnan(summary["hit_rate"]))


if __name__ == "__main__":
    unittest.main()
