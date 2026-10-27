"""Decision, numerical and failure-reporting gates for PR timing comparisons."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))
import pr_regression as benchmark


def records(ratio=1.0):
    return [{"status": "complete", "revision": revision, "input_sha256": "same-input",
             "binary_sha256": revision, "seconds": {metric: ratio if revision == "head" else 1.0
                                                      for metric in benchmark.METRICS}}
            for index in range(benchmark.ROUNDS) for revision in benchmark.order(index)]


class RegressionTest(unittest.TestCase):
    def test_signal_requires_all_rounds_and_controls(self):
        self.assertEqual(benchmark.summarize(records(1.21))["status"], "slowdown_signal")
        data = records(1.21)
        for record in data[-4:]:
            record["seconds"] = dict.fromkeys(benchmark.METRICS, 1.0)
        self.assertEqual(benchmark.summarize(data)["status"], "no_signal")

    def test_noisy_setup_does_not_hide_clean_solve_slowdown(self):
        data = records(1.5)
        data[0]["seconds"]["setup"] = 2.0
        result = benchmark.summarize(data)
        self.assertEqual(result["status"], "slowdown_signal")
        self.assertEqual(result["metrics"]["setup"]["status"], "inconclusive")
        self.assertEqual(result["metrics"]["solve"]["status"], "slowdown_signal")

    def test_noisy_fixture_does_not_hide_other_fixture_slowdown(self):
        noisy = records()
        noisy[0]["seconds"]["setup"] = 2.0
        runs = [SimpleNamespace(stdout=json.dumps(record), stderr="")
                for record in noisy + records(1.5)]
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(output=Path(directory), route="cpu", base_revision="a", head_revision="b",
                                   base_python=Path("base-python"), head_python=Path("head-python"))
            with patch.object(benchmark, "SIDES", (4, 5)), patch.object(benchmark, "hardware", return_value={}), \
                    patch.object(benchmark.os, "sched_setaffinity", create=True), \
                    patch.object(benchmark.os, "sched_getaffinity", return_value={0, 1}, create=True), \
                    patch.object(benchmark.subprocess, "run", side_effect=runs), \
                    patch.dict(benchmark.os.environ, {"GITHUB_STEP_SUMMARY": str(Path(directory) / "step.md")}):
                self.assertEqual(benchmark.campaign(args), 1)
            report = json.loads((Path(directory) / "report.json").read_text())
            self.assertEqual(report["status"], "slowdown_signal")
            self.assertEqual([case["summary"]["status"] for case in report["cases"]],
                             ["inconclusive", "slowdown_signal"])

    def test_noisy_data_never_becomes_a_performance_pass(self):
        data = records()
        data[0]["seconds"]["solve"] = 1.11
        self.assertEqual(benchmark.summarize(data)["status"], "inconclusive")

    def test_missing_failed_wrong_order_and_input_mismatch(self):
        self.assertEqual(benchmark.summarize(records()[:-1])["status"], "incomplete")
        for change, expected in (({"status": "timeout"}, "incomplete"),
                                 ({"revision": "head"}, "incomplete"),
                                 ({"input_sha256": "different"}, "invalid"),
                                 ({"binary_sha256": "different"}, "invalid")):
            data = records()
            data[0].update(change)
            self.assertEqual(benchmark.summarize(data)["status"], expected)

    def test_owner_rejects_fallback_and_bad_original_residual(self):
        fake_solver = SimpleNamespace(backend="cpu")
        fake_module = SimpleNamespace(factorize=lambda *args, **kwargs: fake_solver)
        with patch.dict(sys.modules, apxchol=fake_module):
            with self.assertRaisesRegex(ValueError, "requested gpu, selected cpu"):
                benchmark.owner(4, "gpu")
        fake_solver.solve = lambda rhs, **kwargs: SimpleNamespace(x=np.zeros_like(rhs), converged=True)
        with patch.dict(sys.modules, apxchol=fake_module):
            with self.assertRaises(ValueError):
                benchmark.owner(4, "cpu")

    def test_timeout_preserves_denominator_and_report(self):
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(output=Path(directory), route="cpu", base_revision="a", head_revision="b",
                                   base_python=Path("base-python"), head_python=Path("head-python"))
            error = benchmark.subprocess.TimeoutExpired("owner", 30, output=b"partial", stderr=b"timeout")
            with patch.object(benchmark, "SIDES", (4,)), patch.object(benchmark, "hardware", return_value={}), \
                    patch.object(benchmark.os, "sched_setaffinity", create=True), \
                    patch.object(benchmark.os, "sched_getaffinity", return_value={0, 1}, create=True), \
                    patch.object(benchmark.subprocess, "run", side_effect=error), \
                    patch.dict(benchmark.os.environ, {"GITHUB_STEP_SUMMARY": str(Path(directory) / "step.md")}):
                self.assertEqual(benchmark.campaign(args), 1)
            report = json.loads((Path(directory) / "report.json").read_text())
            self.assertEqual(report["planned_owners"], 12)
            self.assertEqual(report["completed_owners"], 0)
            self.assertEqual(report["status"], "incomplete")
            self.assertEqual((Path(directory) / "grid4-0-0-base.stderr").read_text(), "timeout")


if __name__ == "__main__":
    unittest.main()
