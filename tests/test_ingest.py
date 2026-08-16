from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from lob_alpha.config import load_config
from lob_alpha.ingest import (
    CostLimitError,
    PaidRequestConfirmationError,
    download_stream,
    estimate_cost,
    request_parameters,
)

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


class _InterruptedTimeseries:
    def __init__(self) -> None:
        self.calls = []

    def get_range(self, **kwargs):
        self.calls.append(kwargs)
        Path(kwargs["path"]).write_bytes(b"incomplete")
        raise RuntimeError("simulated interrupted response")


class _Client:
    def __init__(self, cost: float, *, timeseries=None) -> None:
        self.metadata = _Metadata(cost)
        self.timeseries = timeseries or _Timeseries()


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
                confirm_paid_request=True,
                client=_Client(1.01),
            )

    def test_stream_download_requires_independent_confirmation(self) -> None:
        config = load_config(ROOT / "configs/sample_three_sessions.yaml")
        client = _Client(0.0)
        with self.assertRaises(PaidRequestConfirmationError):
            download_stream(
                config,
                ROOT / "data/raw/should-not-exist.dbn.zst",
                max_cost_usd=1.0,
                confirm_paid_request=False,
                client=client,
            )
        self.assertEqual(client.metadata.calls, [])

    def test_interrupted_stream_download_never_uses_final_filename(self) -> None:
        config = load_config(ROOT / "configs/sample_three_sessions.yaml")
        timeseries = _InterruptedTimeseries()
        client = _Client(0.0, timeseries=timeseries)
        with tempfile.TemporaryDirectory() as directory:
            final_path = Path(directory) / "ESM6_2026-03-16_mbp-10.dbn.zst"
            with self.assertRaisesRegex(RuntimeError, "simulated interrupted"):
                download_stream(
                    config,
                    final_path,
                    max_cost_usd=1.0,
                    confirm_paid_request=True,
                    client=client,
                )
            partial_path = final_path.with_name(f"{final_path.name}.partial")
            self.assertFalse(final_path.exists())
            self.assertEqual(partial_path.read_bytes(), b"incomplete")
            self.assertEqual(Path(timeseries.calls[0]["path"]), partial_path)

    def test_stale_stream_partial_blocks_before_cost_estimation(self) -> None:
        config = load_config(ROOT / "configs/sample_three_sessions.yaml")
        client = _Client(0.0)
        with tempfile.TemporaryDirectory() as directory:
            final_path = Path(directory) / "definitions.dbn.zst"
            partial_path = final_path.with_name(f"{final_path.name}.partial")
            partial_path.write_bytes(b"interrupted")
            with self.assertRaisesRegex(FileExistsError, "refusing to recharge"):
                download_stream(
                    config,
                    final_path,
                    max_cost_usd=1.0,
                    confirm_paid_request=True,
                    client=client,
                )
        self.assertEqual(client.metadata.calls, [])


if __name__ == "__main__":
    unittest.main()
