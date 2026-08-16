from __future__ import annotations

import unittest
from pathlib import Path

from lob_alpha.config import load_config
from lob_alpha.ingest import CostLimitError, download_stream, estimate_cost, request_parameters


ROOT = Path(__file__).resolve().parents[1]


class _Metadata:
    def __init__(self, cost: float) -> None:
        self.cost = cost
        self.calls = []

    def get_cost(self, **kwargs):
        self.calls.append(kwargs)
        return self.cost


class _Timeseries:
    def get_range(self, **kwargs):
        raise AssertionError("download must not start when the cost cap is exceeded")


class _Client:
    def __init__(self, cost: float) -> None:
        self.metadata = _Metadata(cost)
        self.timeseries = _Timeseries()


class IngestTests(unittest.TestCase):
    def test_request_is_exact_raw_symbol_mbp10(self) -> None:
        config = load_config(ROOT / "configs/sample_three_sessions.yaml")
        parameters = request_parameters(config)
        self.assertEqual(parameters["dataset"], "GLBX.MDP3")
        self.assertEqual(parameters["schema"], "mbp-10")
        self.assertEqual(parameters["stype_in"], "raw_symbol")
        self.assertEqual(parameters["symbols"], ["ESM6"])

    def test_cost_estimation_does_not_download(self) -> None:
        config = load_config(ROOT / "configs/sample_three_sessions.yaml")
        client = _Client(1.25)
        self.assertEqual(estimate_cost(config, client=client), 1.25)
        self.assertEqual(len(client.metadata.calls), 1)

    def test_cost_cap_blocks_download(self) -> None:
        config = load_config(ROOT / "configs/sample_three_sessions.yaml")
        with self.assertRaises(CostLimitError):
            download_stream(
                config,
                ROOT / "data/raw/should-not-exist.dbn.zst",
                max_cost_usd=1.0,
                client=_Client(1.01),
            )


if __name__ == "__main__":
    unittest.main()

