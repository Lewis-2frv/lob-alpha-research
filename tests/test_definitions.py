from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd

from lob_alpha.config import load_config
from lob_alpha.definitions import DefinitionError, verify_contract_definition
from lob_alpha.ingest import definition_window

ROOT = Path(__file__).resolve().parents[1]


class DefinitionTests(unittest.TestCase):
    def test_expected_definition_is_verified(self) -> None:
        config = load_config(ROOT / "configs/base.yaml")
        frame = pd.DataFrame(
            {
                "ts_recv": ["2026-03-16T00:00:00Z"],
                "raw_symbol": ["ESM6"],
                "min_price_increment": [0.25],
                "unit_of_measure_qty": [50.0],
                "expiration": ["2026-06-19T00:00:00Z"],
                "currency": ["USD"],
                "security_type": ["FUT"],
            }
        )
        verified = verify_contract_definition(
            frame, symbol="ESM6", expected=config.contract
        )
        self.assertEqual(verified.tick_size, 0.25)
        self.assertEqual(verified.multiplier, 50.0)

    def test_fixed_precision_definition_is_supported(self) -> None:
        config = load_config(ROOT / "configs/base.yaml")
        frame = pd.DataFrame(
            {
                "raw_symbol": ["ESM6"],
                "min_price_increment": [250_000_000],
                "unit_of_measure_qty": [50_000_000_000],
            }
        )
        verified = verify_contract_definition(
            frame, symbol="ESM6", expected=config.contract
        )
        self.assertEqual(verified.tick_size, 0.25)
        self.assertEqual(verified.multiplier, 50.0)

    def test_definition_mismatch_fails_closed(self) -> None:
        config = load_config(ROOT / "configs/base.yaml")
        frame = pd.DataFrame(
            {
                "raw_symbol": ["ESM6"],
                "min_price_increment": [0.5],
                "unit_of_measure_qty": [50.0],
            }
        )
        with self.assertRaises(DefinitionError):
            verify_contract_definition(frame, symbol="ESM6", expected=config.contract)

    def test_definition_window_is_exactly_one_utc_day(self) -> None:
        config = load_config(ROOT / "configs/base.yaml")
        start, end = definition_window(config)
        self.assertEqual(pd.Timestamp(end) - pd.Timestamp(start), pd.Timedelta(days=1))


if __name__ == "__main__":
    unittest.main()

