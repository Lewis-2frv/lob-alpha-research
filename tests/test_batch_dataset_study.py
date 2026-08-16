from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from lob_alpha.config import load_config
from lob_alpha.dataset import (
    discover_daily_raw_files,
    processed_row_count,
    session_date_from_filename,
)
from lob_alpha.ingest import (
    CostLimitError,
    PaidRequestConfirmationError,
    download_batch_job,
    submit_batch_job,
)
from lob_alpha.manifest import sha256_file, write_json
from lob_alpha.reporting import build_research_report
from lob_alpha.study import expanding_session_folds, freeze_candidate


ROOT = Path(__file__).resolve().parents[1]


class _Metadata:
    def __init__(self, cost: float) -> None:
        self.cost = cost

    def get_cost(self, **kwargs):
        return self.cost


class _Batch:
    def __init__(self, download_path: Path | None = None) -> None:
        self.submit_calls = []
        self.download_path = download_path

    def submit_job(self, **kwargs):
        self.submit_calls.append(kwargs)
        return {"id": "GLBX-TEST", "state": "queued"}

    def get_job_details(self, *, job_id):
        return {"id": job_id, "state": "done"}

    def list_files(self, *, job_id):
        digest = hashlib.sha256(self.download_path.read_bytes()).hexdigest()
        return [{"filename": self.download_path.name, "hash": f"sha256:{digest}"}]

    def download(self, *, job_id, output_dir):
        return [self.download_path]


class _Client:
    def __init__(self, cost: float, download_path: Path | None = None) -> None:
        self.metadata = _Metadata(cost)
        self.batch = _Batch(download_path)


class BatchDatasetStudyTests(unittest.TestCase):
    def test_batch_submission_requires_boolean_confirmation(self) -> None:
        config = load_config(ROOT / "configs/sample_three_sessions.yaml")
        with self.assertRaises(PaidRequestConfirmationError):
            submit_batch_job(
                config,
                max_cost_usd=10.0,
                confirm_paid_request=False,
                client=_Client(1.0),
            )

    def test_batch_submission_rechecks_cost_and_requests_daily_dbn(self) -> None:
        config = load_config(ROOT / "configs/sample_three_sessions.yaml")
        blocked = _Client(2.0)
        with self.assertRaises(CostLimitError):
            submit_batch_job(
                config,
                max_cost_usd=1.99,
                confirm_paid_request=True,
                client=blocked,
            )
        client = _Client(1.0)
        details, cost = submit_batch_job(
            config,
            max_cost_usd=1.0,
            confirm_paid_request=True,
            client=client,
        )
        self.assertEqual(details["id"], "GLBX-TEST")
        self.assertEqual(cost, 1.0)
        call = client.batch.submit_calls[0]
        self.assertEqual(call["split_duration"], "day")
        self.assertEqual(call["encoding"], "dbn")
        self.assertEqual(call["compression"], "zstd")

    def test_batch_download_checks_provider_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "glbx-20260316.mbp-10.dbn.zst"
            path.write_bytes(b"provider bytes")
            downloaded, remote = download_batch_job(
                "GLBX-TEST", directory, client=_Client(0.0, path)
            )
            self.assertEqual(downloaded, [path])
            self.assertEqual(remote[0]["filename"], path.name)

    def test_daily_filename_discovery_is_explicit_and_unique(self) -> None:
        self.assertEqual(
            session_date_from_filename("glbx-mdp3-20260316.mbp-10.dbn.zst").isoformat(),
            "2026-03-16",
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "ESM6_2026-03-16.csv").write_text("x\n1\n", encoding="utf-8")
            discovered = discover_daily_raw_files(root)
            self.assertEqual(next(iter(discovered)).isoformat(), "2026-03-16")
            (root / "duplicate_20260316.csv.gz").write_bytes(b"x")
            with self.assertRaises(ValueError):
                discover_daily_raw_files(root)

    def test_compressed_csv_row_count_does_not_load_the_table(self) -> None:
        import gzip

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "session.csv.gz"
            with gzip.open(path, "wt", encoding="utf-8") as handle:
                handle.write("a,b\n1,2\n3,4\n")
            self.assertEqual(processed_row_count(path), 2)

    def test_expanding_folds_never_leak_future_sessions(self) -> None:
        dates = [f"2026-03-{day:02d}" for day in range(16, 26)]
        folds = expanding_session_folds(dates, minimum_train_sessions=4, folds=3)
        self.assertEqual(len(folds), 3)
        for train, validation in folds:
            self.assertLess(max(train), min(validation))
            self.assertFalse(set(train) & set(validation))

    def test_freeze_is_content_locked_and_refuses_overwrite(self) -> None:
        config = load_config(ROOT / "configs/base.yaml")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog = root / "catalog.json"
            catalog.write_text("{}\n", encoding="utf-8")
            candidate = root / "candidate.json"
            write_json(
                candidate,
                {
                    "stage": "validation_selected",
                    "catalog_sha256": sha256_file(catalog),
                    "config_sha256": sha256_file(config.source_path),
                    "selected_horizon_ms": 500,
                    "selected_alpha": 1.0,
                    "selected_safety_margin_ticks": 0.25,
                },
            )
            frozen = root / "frozen.json"
            payload = freeze_candidate(
                config,
                catalog_path=catalog,
                candidate_path=candidate,
                output_path=frozen,
            )
            self.assertEqual(payload["stage"], "frozen_for_holdout")
            with self.assertRaises(FileExistsError):
                freeze_candidate(
                    config,
                    catalog_path=catalog,
                    candidate_path=candidate,
                    output_path=frozen,
                )

    def test_report_without_holdout_makes_no_performance_claim(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report, cv = build_research_report(
                train_dir=root / "train",
                validation_dir=root / "validation",
                holdout_dir=root / "holdout",
                reports_dir=root / "reports",
            )
            self.assertIn("does not claim", report.read_text(encoding="utf-8"))
            self.assertIn("Safe pre-results", cv.read_text(encoding="utf-8"))
            json.loads(json.dumps({"path": str(report)}))

    def test_fixture_holdout_never_generates_a_performance_cv_bullet(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            holdout = root / "holdout"
            holdout.mkdir()
            write_json(holdout / "HOLDOUT_COMPLETE.json", {"stage": "holdout_complete"})
            report, cv = build_research_report(
                train_dir=root / "train",
                validation_dir=root / "validation",
                holdout_dir=holdout,
                reports_dir=root / "reports",
                engineering_fixture=True,
            )
            self.assertIn("ENGINEERING FIXTURE", report.read_text(encoding="utf-8"))
            self.assertIn("Safe pre-results", cv.read_text(encoding="utf-8"))
            self.assertNotIn("Evidence-backed", cv.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
