from __future__ import annotations

import unittest

from lob_alpha.execution import InsufficientDepthError, round_trip_pnl, sweep_book
from lob_alpha.fixture import make_mbp10_fixture


class ExecutionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.book = make_mbp10_fixture(periods=3).iloc[0]

    def test_single_contract_buy_uses_best_ask(self) -> None:
        fill = sweep_book(self.book, side="buy", quantity=1)
        self.assertEqual(fill.average_price, self.book["ask_px_00"])
        self.assertEqual(fill.levels_used, 1)

    def test_multi_level_vwap(self) -> None:
        book = self.book.copy()
        book["ask_sz_00"] = 1
        book["ask_sz_01"] = 2
        fill = sweep_book(book, side="buy", quantity=3)
        expected = (book["ask_px_00"] + 2 * book["ask_px_01"]) / 3
        self.assertAlmostEqual(fill.average_price, expected)
        self.assertEqual(fill.levels_used, 2)

    def test_insufficient_depth_is_rejected(self) -> None:
        with self.assertRaises(InsufficientDepthError):
            sweep_book(self.book, side="buy", quantity=1_000_000)

    def test_flat_book_round_trip_loses_spread_and_fees(self) -> None:
        result = round_trip_pnl(
            self.book,
            self.book,
            position_side="long",
            quantity=1,
            multiplier=50.0,
            fee_per_contract_per_side_usd=2.5,
        )
        self.assertAlmostEqual(result.gross_pnl_usd, -12.5)
        self.assertAlmostEqual(result.explicit_fees_usd, 5.0)
        self.assertAlmostEqual(result.net_pnl_usd, -17.5)


if __name__ == "__main__":
    unittest.main()

