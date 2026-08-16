from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml

from lob_alpha.config import ConfigurationError, load_config


ROOT = Path(__file__).resolve().parents[1]


class ConfigTests(unittest.TestCase):
    def test_base_config_is_valid_and_chronological(self) -> None:
        config = load_config(ROOT / "configs/base.yaml")
        self.assertEqual(config.data.dataset, "GLBX.MDP3")
        self.assertEqual(config.data.schema, "mbp-10")
        self.assertEqual(config.splits.split_for(config.splits.train_start), "train")
        self.assertEqual(config.splits.split_for(config.splits.validation_start), "validation")
        self.assertEqual(config.splits.split_for(config.splits.holdout_start), "holdout")

    def test_overlapping_splits_are_rejected(self) -> None:
        source = ROOT / "configs/base.yaml"
        raw = yaml.safe_load(source.read_text(encoding="utf-8"))
        raw["splits"]["validation_start"] = raw["splits"]["train_end"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.yaml"
            path.write_text(yaml.safe_dump(raw), encoding="utf-8")
            with self.assertRaises(ConfigurationError):
                load_config(path)

    def test_primary_execution_values_must_be_in_grids(self) -> None:
        source = ROOT / "configs/base.yaml"
        raw = yaml.safe_load(source.read_text(encoding="utf-8"))
        raw["execution"]["primary_latency_ms"] = 11
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.yaml"
            path.write_text(yaml.safe_dump(raw), encoding="utf-8")
            with self.assertRaises(ConfigurationError):
                load_config(path)


if __name__ == "__main__":
    unittest.main()

