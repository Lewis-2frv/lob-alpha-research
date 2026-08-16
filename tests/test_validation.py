from __future__ import annotations

import unittest

from lob_alpha.fixture import make_mbp10_fixture
from lob_alpha.validation import validate_mbp10


class ValidationTests(unittest.TestCase):
    def test_clean_fixture_is_accepted(self) -> None:
        report = validate_mbp10(make_mbp10_fixture(periods=40), tick_size=0.25)
        self.assertTrue(report.accepted)
        self.assertEqual(report.crossed_books, 0)
        self.assertEqual(report.ladder_errors, 0)

    def test_crossed_book_is_rejected(self) -> None:
        frame = make_mbp10_fixture(periods=20)
        frame.loc[5, "bid_px_00"] = frame.loc[5, "ask_px_00"] + 0.25
        report = validate_mbp10(frame, tick_size=0.25)
        self.assertFalse(report.accepted)
        self.assertEqual(report.crossed_books, 1)

    def test_duplicate_and_timestamp_inversion_are_rejected(self) -> None:
        frame = make_mbp10_fixture(periods=20)
        frame.loc[10] = frame.loc[9]
        report = validate_mbp10(frame, tick_size=0.25)
        self.assertFalse(report.accepted)
        self.assertGreaterEqual(report.exact_duplicate_rows, 1)

    def test_bad_ladder_is_rejected(self) -> None:
        frame = make_mbp10_fixture(periods=20)
        frame.loc[3, "bid_px_01"] = frame.loc[3, "bid_px_00"]
        report = validate_mbp10(frame, tick_size=0.25)
        self.assertFalse(report.accepted)
        self.assertEqual(report.ladder_errors, 1)


if __name__ == "__main__":
    unittest.main()

