from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import zstandard
from databento_dbn import Metadata, Schema, SType

from lob_alpha.acquisition import (
    download_planned_sessions,
    estimate_session_costs,
    plan_session_requests,
    session_raw_filename,
    validate_local_dbn,
    write_session_cost_plan,
)
from lob_alpha.cli import build_parser
from lob_alpha.config import load_config
from lob_alpha.dataset import discover_daily_raw_files
from lob_alpha.feasibility import audit_session_resources
from lob_alpha.fixture import make_mbp10_fixture
from lob_alpha.ingest import CostLimitError, PaidRequestConfirmationError
from lob_alpha.manifest import sha256_file
from lob_alpha.sampling import filter_session, session_bounds

ROOT = Path(__file__).resolve().parents[1]


def _valid_empty_mbp10_dbn() -> bytes:
    metadata = Metadata(
        dataset="GLBX.MDP3",
        start=0,
        stype_in=SType.RAW_SYMBOL,
        stype_out=SType.INSTRUMENT_ID,
        schema=Schema.MBP_10,
        symbols=["ESM6"],
    )
    return zstandard.ZstdCompressor(write_checksum=True).compress(metadata.encode())


class _StepClock:
    def __init__(self, step: float = 0.125) -> None:
        self.value = -step
        self.step = step

    def __call__(self) -> float:
        self.value += self.step
        return self.value


class _Metadata:
    def __init__(self, costs: tuple[float, ...], events: list[tuple[str, str]]) -> None:
        self.costs = costs
        self.events = events
        self.calls: list[dict[str, object]] = []

    def get_cost(self, **kwargs):
        self.calls.append(kwargs)
        self.events.append(("estimate", str(kwargs["start"])))
        return self.costs[len(self.calls) - 1]


class _Timeseries:
    def __init__(
        self,
        *,
        events: list[tuple[str, str]] | None = None,
        fail_first: bool = False,
        corrupt_failure: bool = False,
    ) -> None:
        self.events = events if events is not None else []
        self.calls: list[dict[str, object]] = []
        self.fail_first = fail_first
        self.corrupt_failure = corrupt_failure

    def get_range(self, **kwargs):
        self.calls.append(kwargs)
        self.events.append(("download", str(kwargs["start"])))
        path = Path(str(kwargs["path"]))
        content = (
            b"truncated"
            if self.fail_first and self.corrupt_failure
            else _valid_empty_mbp10_dbn()
        )
        path.write_bytes(content)
        if self.fail_first and len(self.calls) == 1:
            raise ConnectionError("simulated interruption after partial bytes")
        return object()


class _NoDownloadTimeseries:
    def __init__(self, *, events: list[tuple[str, str]] | None = None) -> None:
        self.events = events if events is not None else []
        self.calls: list[dict[str, object]] = []

    def get_range(self, **kwargs):
        self.calls.append(kwargs)
        self.events.append(("download", str(kwargs["start"])))
        raise AssertionError("estimate-only or blocked acquisition reached the paid endpoint")


class _Client:
    def __init__(
        self,
        costs: tuple[float, ...] = (0.1, 0.2, 0.3),
        *,
        timeseries: object | None = None,
    ) -> None:
        self.api_key = "db-secret-must-never-be-persisted"
        self.events: list[tuple[str, str]] = []
        self.metadata = _Metadata(costs, self.events)
        self.timeseries = timeseries or _Timeseries(events=self.events)
        if hasattr(self.timeseries, "events"):
            self.timeseries.events = self.events


class SessionPlanningTests(unittest.TestCase):
    def test_sample_dates_and_exact_intraday_parameters(self) -> None:
        config = load_config(ROOT / "configs/sample_three_sessions.yaml")
        requests = plan_session_requests(config)
        self.assertEqual(
            [request.session_date.isoformat() for request in requests],
            ["2026-03-16", "2026-03-17", "2026-03-18"],
        )
        for request in requests:
            parameters = request.parameters(config)
            self.assertEqual(parameters["dataset"], "GLBX.MDP3")
            self.assertEqual(parameters["schema"], "mbp-10")
            self.assertEqual(parameters["symbols"], ["ESM6"])
            self.assertEqual(parameters["stype_in"], "raw_symbol")
            self.assertTrue(str(parameters["start"]).endswith("13:35:00+00:00"))
            self.assertTrue(str(parameters["end"]).endswith("19:55:00+00:00"))

    def test_dst_offsets_change_without_hard_coding_and_weekends_are_skipped(self) -> None:
        config = load_config(ROOT / "configs/sample_three_sessions.yaml")
        data = replace(
            config.data,
            start="2026-03-06T00:00:00Z",
            end="2026-03-10T00:00:00Z",
        )
        requests = plan_session_requests(replace(config, data=data))
        self.assertEqual(
            [request.session_date.isoformat() for request in requests],
            ["2026-03-06", "2026-03-09"],
        )
        self.assertEqual(requests[0].start_utc.hour, 14)
        self.assertEqual(requests[1].start_utc.hour, 13)

    def test_weekend_only_range_is_an_empty_plan_without_metadata_access(self) -> None:
        config = load_config(ROOT / "configs/sample_three_sessions.yaml")
        data = replace(
            config.data,
            start="2026-03-07T00:00:00Z",
            end="2026-03-09T00:00:00Z",
        )
        weekend_config = replace(config, data=data)
        self.assertEqual(plan_session_requests(weekend_config), ())
        plan = estimate_session_costs(weekend_config, client=object())
        self.assertEqual(plan.estimates, ())
        self.assertEqual(plan.total_estimated_cost_usd, 0.0)

    def test_configured_end_is_exclusive_when_filtering(self) -> None:
        config = load_config(ROOT / "configs/sample_three_sessions.yaml")
        requests_date = plan_session_requests(config)[0].session_date
        start, end = session_bounds(requests_date, config.session)
        events = pd.DataFrame({"ts_recv": [start, end]})
        filtered = filter_session(events, requests_date, config.session)
        self.assertEqual(filtered["ts_recv"].tolist(), [start])

    def test_estimate_only_calls_metadata_for_every_session_and_never_downloads(self) -> None:
        config = load_config(ROOT / "configs/sample_three_sessions.yaml")
        timeseries = _NoDownloadTimeseries()
        client = _Client(timeseries=timeseries)
        plan = estimate_session_costs(config, client=client)
        self.assertEqual(len(client.metadata.calls), 3)
        self.assertEqual(timeseries.calls, [])
        self.assertEqual([event[0] for event in client.events], ["estimate"] * 3)
        self.assertAlmostEqual(plan.total_estimated_cost_usd, 0.6)
        self.assertEqual(
            [call["start"] for call in client.metadata.calls],
            [
                "2026-03-16T13:35:00+00:00",
                "2026-03-17T13:35:00+00:00",
                "2026-03-18T13:35:00+00:00",
            ],
        )

    def test_estimate_artifact_is_stable_and_contains_no_credential(self) -> None:
        config = load_config(ROOT / "configs/sample_three_sessions.yaml")
        client = _Client()
        plan = estimate_session_costs(config, client=client)
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.json"
            second = Path(directory) / "second.json"
            write_session_cost_plan(first, plan, config=config)
            write_session_cost_plan(second, plan, config=config)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            self.assertNotIn(client.api_key, first.read_text(encoding="utf-8"))

    def test_databento_083_get_range_explicitly_requests_dbn_zstandard(self) -> None:
        import databento

        client = databento.Historical("unused-test-key")
        with tempfile.TemporaryDirectory() as directory:
            partial = Path(directory) / "session.dbn.zst.partial"
            with patch.object(client.timeseries, "_stream", return_value=object()) as stream:
                client.timeseries.get_range(
                    dataset="GLBX.MDP3",
                    symbols=["ESM6"],
                    schema="mbp-10",
                    stype_in="raw_symbol",
                    start="2026-03-16T13:35:00+00:00",
                    end="2026-03-16T19:55:00+00:00",
                    path=partial,
                )
            call = stream.call_args.kwargs
            self.assertEqual(call["data"]["encoding"], "dbn")
            self.assertEqual(call["data"]["compression"], "zstd")
            self.assertEqual(call["path"], partial)


class SessionDownloadSafetyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_config(ROOT / "configs/sample_three_sessions.yaml")

    def test_aggregate_cap_rejects_after_all_estimates_before_first_paid_call(self) -> None:
        timeseries = _NoDownloadTimeseries()
        client = _Client((0.5, 0.5, 0.5), timeseries=timeseries)
        with tempfile.TemporaryDirectory() as directory, self.assertRaises(
            CostLimitError
        ) as caught:
            download_planned_sessions(
                self.config,
                directory,
                max_cost_usd=1.49,
                confirm_paid_request=True,
                client=client,
            )
        self.assertEqual(len(client.metadata.calls), 3)
        self.assertEqual(timeseries.calls, [])
        self.assertEqual([event[0] for event in client.events], ["estimate"] * 3)
        self.assertNotIn(client.api_key, str(caught.exception))

    def test_nonfinite_or_negative_cap_fails_before_metadata_and_paid_calls(self) -> None:
        for invalid_cap in (float("nan"), float("inf"), float("-inf"), -0.01):
            with self.subTest(invalid_cap=invalid_cap):
                client = _Client(timeseries=_NoDownloadTimeseries())
                with tempfile.TemporaryDirectory() as directory, self.assertRaises(ValueError):
                    download_planned_sessions(
                        self.config,
                        directory,
                        max_cost_usd=invalid_cap,
                        confirm_paid_request=True,
                        client=client,
                    )
                self.assertEqual(client.metadata.calls, [])
                self.assertEqual(client.timeseries.calls, [])

    def test_confirmation_is_an_independent_paid_request_gate(self) -> None:
        timeseries = _NoDownloadTimeseries()
        client = _Client(timeseries=timeseries)
        with tempfile.TemporaryDirectory() as directory, self.assertRaises(
            PaidRequestConfirmationError
        ) as caught:
            download_planned_sessions(
                self.config,
                directory,
                max_cost_usd=1.0,
                confirm_paid_request=False,
                client=client,
            )
        self.assertEqual(len(client.metadata.calls), 3)
        self.assertEqual(timeseries.calls, [])
        self.assertEqual([event[0] for event in client.events], ["estimate"] * 3)
        self.assertNotIn(client.api_key, str(caught.exception))

    def test_atomic_files_manifest_hashes_and_verified_resume(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.json"
            client = _Client()
            result = download_planned_sessions(
                self.config,
                root / "raw",
                max_cost_usd=1.0,
                confirm_paid_request=True,
                manifest_path=manifest,
                client=client,
            )
            self.assertEqual(result["paid_requests_this_run"], 3)
            self.assertEqual(
                [event[0] for event in client.events],
                ["estimate", "estimate", "estimate", "download", "download", "download"],
            )
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertNotIn(client.api_key, manifest.read_text(encoding="utf-8"))
            self.assertEqual(payload["encoding"], "dbn")
            self.assertEqual(payload["compression"], "zstd")
            for record in payload["sessions"]:
                path = Path(record["local_path"])
                self.assertTrue(path.is_file())
                self.assertFalse(Path(record["temporary_path"]).exists())
                self.assertEqual(record["raw_bytes"], path.stat().st_size)
                self.assertEqual(record["sha256"], sha256_file(path))
                self.assertTrue(path.is_absolute())
                self.assertEqual(record["encoding"], "dbn")
                self.assertEqual(record["compression"], "zstd")
                validate_local_dbn(path, expected_schema="mbp-10", scan_records=True)
            discovered = discover_daily_raw_files(root / "raw")
            self.assertEqual(len(discovered), 3)

            resume_timeseries = _NoDownloadTimeseries()
            resume_client = _Client(timeseries=resume_timeseries)
            resumed = download_planned_sessions(
                self.config,
                root / "raw",
                max_cost_usd=1.0,
                confirm_paid_request=True,
                manifest_path=manifest,
                client=resume_client,
            )
            self.assertEqual(resumed["paid_requests_this_run"], 0)
            self.assertEqual(resumed["skipped_complete_this_run"], 3)
            self.assertEqual(resumed["estimated_cost_usd_this_run"], 0.0)
            self.assertEqual(resume_client.metadata.calls, [])
            self.assertEqual(resume_timeseries.calls, [])

            crash_state = json.loads(manifest.read_text(encoding="utf-8"))
            crash_state["sessions"][0]["status"] = "verified"
            crash_state["sessions"][0]["download_complete"] = False
            manifest.write_text(
                json.dumps(crash_state, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            post_rename_client = _Client(timeseries=_NoDownloadTimeseries())
            recovered_final = download_planned_sessions(
                self.config,
                root / "raw",
                max_cost_usd=1.0,
                confirm_paid_request=True,
                manifest_path=manifest,
                client=post_rename_client,
            )
            self.assertEqual(recovered_final["recovered_sessions_this_run"], 1)
            self.assertEqual(recovered_final["paid_requests_this_run"], 0)
            self.assertEqual(post_rename_client.metadata.calls, [])

            first_path = Path(payload["sessions"][0]["local_path"])
            first_path.write_bytes(b"tampered")
            tamper_timeseries = _NoDownloadTimeseries()
            with self.assertRaises(OSError):
                download_planned_sessions(
                    self.config,
                    root / "raw",
                    max_cost_usd=1.0,
                    confirm_paid_request=True,
                    manifest_path=manifest,
                    client=_Client(timeseries=tamper_timeseries),
                )
            self.assertEqual(tamper_timeseries.calls, [])

    def test_unknown_existing_file_is_never_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            root.mkdir(exist_ok=True)
            first_date = plan_session_requests(self.config)[0].session_date
            path = root / session_raw_filename(self.config, first_date)
            path.write_bytes(b"unknown-existing-data")
            timeseries = _NoDownloadTimeseries()
            with self.assertRaises(FileExistsError):
                download_planned_sessions(
                    self.config,
                    root,
                    max_cost_usd=1.0,
                    confirm_paid_request=True,
                    client=_Client(timeseries=timeseries),
                )
            self.assertEqual(path.read_bytes(), b"unknown-existing-data")
            self.assertEqual(timeseries.calls, [])

    def test_complete_interrupted_partial_is_recovered_without_recharge(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.json"
            with self.assertRaises(ConnectionError):
                download_planned_sessions(
                    self.config,
                    root / "raw",
                    max_cost_usd=1.0,
                    confirm_paid_request=True,
                    manifest_path=manifest,
                    client=_Client(timeseries=_Timeseries(fail_first=True)),
                )
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            interrupted = payload["sessions"][0]
            self.assertEqual(interrupted["status"], "downloading")
            self.assertTrue(Path(interrupted["temporary_path"]).is_file())
            self.assertFalse(Path(interrupted["local_path"]).exists())
            with self.assertRaises(FileNotFoundError):
                discover_daily_raw_files(root / "raw")

            retry_timeseries = _Timeseries()
            retry_client = _Client((0.2, 0.3), timeseries=retry_timeseries)
            recovered = download_planned_sessions(
                self.config,
                root / "raw",
                max_cost_usd=0.5,
                confirm_paid_request=True,
                manifest_path=manifest,
                client=retry_client,
            )
            self.assertEqual(recovered["recovered_sessions_this_run"], 1)
            self.assertEqual(recovered["paid_requests_this_run"], 2)
            self.assertEqual(len(retry_client.metadata.calls), 2)
            self.assertEqual(len(retry_timeseries.calls), 2)
            self.assertEqual(len(discover_daily_raw_files(root / "raw")), 3)

    def test_truncated_interrupted_partial_fails_closed_without_estimate_or_retry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.json"
            with self.assertRaises(ConnectionError):
                download_planned_sessions(
                    self.config,
                    root / "raw",
                    max_cost_usd=1.0,
                    confirm_paid_request=True,
                    manifest_path=manifest,
                    client=_Client(
                        timeseries=_Timeseries(fail_first=True, corrupt_failure=True)
                    ),
                )
            retry_timeseries = _NoDownloadTimeseries()
            retry_client = _Client(timeseries=retry_timeseries)
            with self.assertRaises(RuntimeError):
                download_planned_sessions(
                    self.config,
                    root / "raw",
                    max_cost_usd=1.0,
                    confirm_paid_request=True,
                    manifest_path=manifest,
                    client=retry_client,
                )
            self.assertEqual(retry_client.metadata.calls, [])
            self.assertEqual(retry_timeseries.calls, [])

    def test_resume_cost_and_metadata_include_only_missing_sessions(self) -> None:
        one_session_config = replace(
            self.config,
            data=replace(self.config.data, end="2026-03-17T00:00:00Z"),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.json"
            download_planned_sessions(
                one_session_config,
                root / "raw",
                max_cost_usd=0.4,
                confirm_paid_request=True,
                manifest_path=manifest,
                client=_Client((0.4,)),
            )

            resume_client = _Client((0.2, 0.3))
            resumed = download_planned_sessions(
                self.config,
                root / "raw",
                max_cost_usd=0.5,
                confirm_paid_request=True,
                manifest_path=manifest,
                client=resume_client,
            )
            self.assertEqual(len(resume_client.metadata.calls), 2)
            self.assertEqual(len(resume_client.timeseries.calls), 2)
            self.assertEqual(resumed["sessions_estimated_this_run"], 2)
            self.assertAlmostEqual(resumed["estimated_cost_usd_this_run"], 0.5)
            self.assertEqual(
                [call["start"] for call in resume_client.metadata.calls],
                [
                    "2026-03-17T13:35:00+00:00",
                    "2026-03-18T13:35:00+00:00",
                ],
            )


class ResourceAuditTests(unittest.TestCase):
    def test_serial_fixture_audit_aggregates_resources_and_rejects_empty_session(self) -> None:
        config = load_config(ROOT / "configs/sample_three_sessions.yaml")
        requests = plan_session_requests(config)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw"
            raw.mkdir()
            expected_raw_bytes = 0
            for index, request in enumerate(requests):
                frame = make_mbp10_fixture(
                    periods=180,
                    seed=config.seed + index,
                    start=request.start_utc.isoformat(),
                )
                if index == 2:
                    frame = frame.iloc[:0]
                path = raw / f"ESM6_{request.session_date.isoformat()}_mbp10.csv.gz"
                frame.to_csv(path, index=False)
                expected_raw_bytes += path.stat().st_size

            output_json = root / "audit.json"
            output_markdown = root / "audit.md"
            payload = audit_session_resources(
                config,
                raw_dir=raw,
                processed_dir=root / "processed",
                output_json=output_json,
                output_markdown=output_markdown,
                output_format="csv.gz",
                clock=_StepClock(),
            )
            self.assertEqual(payload["counts"]["planned_sessions"], 3)
            self.assertEqual(payload["counts"]["files_found"], 3)
            self.assertEqual(payload["counts"]["research_usable_sessions"], 2)
            self.assertEqual(payload["sessions"][2]["status"], "no_session_data")
            self.assertFalse(payload["sessions"][2]["quality_accepted"])
            self.assertEqual(payload["totals"]["compressed_raw_bytes"], expected_raw_bytes)
            self.assertGreater(payload["maxima"]["dataframe_memory_bytes"], 0)
            self.assertGreater(payload["totals"]["processed_decision_rows"], 0)
            for record in payload["sessions"][:2]:
                self.assertTrue(record["quality_accepted"])
                self.assertEqual(record["rejection_counts"]["crossed_books"], 0)
                output = Path(record["processed_output_path"])
                self.assertEqual(record["processed_output_bytes"], output.stat().st_size)
            stored = json.loads(output_json.read_text(encoding="utf-8"))
            self.assertTrue(stored["serial_processing"])
            self.assertNotIn("profitability", stored)
            markdown = output_markdown.read_text(encoding="utf-8")
            self.assertIn("engineering resource audit", markdown)
            self.assertIn("does not calculate or report alpha", markdown)
            self.assertIn("not process peak RSS", markdown)
            self.assertIn("| Resource | Total | Single-session maximum |", markdown)

            first_json = output_json.read_bytes()
            first_markdown = output_markdown.read_bytes()
            audit_session_resources(
                config,
                raw_dir=raw,
                processed_dir=root / "processed",
                output_json=output_json,
                output_markdown=output_markdown,
                output_format="csv.gz",
                overwrite=True,
                clock=_StepClock(),
            )
            self.assertEqual(output_json.read_bytes(), first_json)
            self.assertEqual(output_markdown.read_bytes(), first_markdown)

    def test_audit_distinguishes_quality_rejection_from_missing_sessions(self) -> None:
        config = load_config(ROOT / "configs/sample_three_sessions.yaml")
        first_request = plan_session_requests(config)[0]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw"
            raw.mkdir()
            frame = make_mbp10_fixture(
                periods=180,
                start=first_request.start_utc.isoformat(),
            )
            frame.loc[10, "bid_px_00"] = frame.loc[10, "ask_px_00"] + 0.25
            frame.to_csv(
                raw / f"ESM6_{first_request.session_date.isoformat()}_mbp10.csv.gz",
                index=False,
            )
            payload = audit_session_resources(
                config,
                raw_dir=raw,
                processed_dir=root / "processed",
                output_json=root / "audit.json",
                output_markdown=root / "audit.md",
                output_format="csv.gz",
                clock=_StepClock(),
            )
            self.assertEqual(payload["sessions"][0]["status"], "rejected_quality")
            self.assertEqual(payload["sessions"][0]["rejection_counts"]["crossed_books"], 1)
            self.assertEqual(
                [record["status"] for record in payload["sessions"][1:]],
                ["missing", "missing"],
            )
            self.assertEqual(payload["counts"]["research_usable_sessions"], 0)


class WindowsWorkflowTests(unittest.TestCase):
    def test_every_paid_streaming_cli_has_an_independent_confirmation_gate(self) -> None:
        parser = build_parser()
        commands = (
            ["download-sessions", "--max-cost-usd", "1.0"],
            ["download", "--output", "unused.dbn.zst", "--max-cost-usd", "1.0"],
            [
                "download-definitions",
                "--output",
                "unused.dbn.zst",
                "--max-cost-usd",
                "1.0",
            ],
        )
        for command in commands:
            with self.subTest(command=command[0]):
                self.assertFalse(parser.parse_args(command).confirm_paid_request)

    def test_powershell_workflow_is_estimate_first_and_paid_opt_in(self) -> None:
        script = (ROOT / "scripts/run_three_session_feasibility.ps1").read_text(
            encoding="utf-8"
        )
        estimate_position = script.index("estimate-session-costs")
        download_position = script.index("download-sessions")
        audit_position = script.index("audit-session-resources")
        self.assertLess(estimate_position, download_position)
        self.assertLess(download_position, audit_position)
        self.assertIn('if ($ConfirmPaidRequest)', script)
        self.assertIn("[double]::IsNaN", script)
        self.assertIn("[double]::IsInfinity", script)
        self.assertNotIn("$MaxDataCostUsd.Value", script)
        self.assertIn("$LASTEXITCODE -ne 0", script)
        self.assertIn("DATABENTO_API_KEY", script)
        self.assertIn('Join-Path ".venv" "Scripts/python.exe"', script)

        full_script = (ROOT / "scripts/run_real_through_freeze.ps1").read_text(
            encoding="utf-8"
        )
        definition_call = full_script.index("download-definitions")
        session_call = full_script.index("download-sessions")
        self.assertIn(
            "--confirm-paid-request",
            full_script[definition_call:session_call],
        )
        self.assertLess(full_script.index("freeze-candidate"), full_script.index("build-report"))
        self.assertNotIn("holdout-stage", full_script)


if __name__ == "__main__":
    unittest.main()
