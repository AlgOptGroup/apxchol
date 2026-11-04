"""Tests of the documentation benchmark's numerical and reporting gates."""
import importlib.util
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

spec = importlib.util.spec_from_file_location("docs_benchmark", Path(__file__).parents[1] / "docs_benchmark.py")
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


class DocsBenchmarkTest(unittest.TestCase):
    def records(self):
        return [{"status": "passed", "seconds": {metric: 1.0 for metric in benchmark.METRICS}}
                for _ in range(2 * len(benchmark.SEEDS))]

    def test_plot_rejects_dirty_or_wrong_revision(self):
        spec = importlib.util.spec_from_file_location("performance", Path(__file__).parents[2] / "docs/performance.py")
        plot = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(plot)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = root / "report.json"
            for content in ({"dirty": True, "source": "expected"}, {"dirty": False, "source": "other"}):
                report.write_text(json.dumps(content))
                with self.assertRaisesRegex(ValueError, "clean documentation revision"):
                    plot.render([report], root / "figures", "expected")

    def test_metal_identity_is_not_inferred_from_cpu(self):
        with patch.object(benchmark.sys, "platform", "darwin"), patch.object(
                benchmark.subprocess, "check_output", side_effect=["Apple CPU\n", "3\n", "Virtual Metal device\n"]) as query:
            self.assertEqual(benchmark.hardware(True)["gpus"], ["Virtual Metal device"])
            self.assertIn("CoreGraphics", query.call_args.args[0])
            self.assertEqual(query.call_args.kwargs["timeout"], 30)
        with patch.object(benchmark.sys, "platform", "darwin"), patch.object(
                benchmark.subprocess, "check_output", side_effect=["Apple CPU\n", "3\n", "\n"]):
            with self.assertRaisesRegex(ValueError, "missing Metal device name"):
                benchmark.hardware(True)

    def test_complete_controls(self):
        report = benchmark.summarize(self.records())
        self.assertEqual(report["status"], "accepted")
        self.assertEqual(report["accepted_pairs"], 3)

    def test_one_noisy_metric_marks_comparison(self):
        records = self.records()
        records[0]["seconds"]["setup"] = 1.2
        report = benchmark.summarize(records)
        self.assertEqual(report["status"], "noisy")
        self.assertEqual(report["accepted_pairs"], 2)

    def test_reference_failure_keeps_native_coverage(self):
        records = self.records()
        for record in records:
            record["reference_cg"] = {"status": "nonconverged"}
            record["seconds"]["scipy_cg"] = None
        result = benchmark.summarize(records)
        self.assertEqual(result["status"], "accepted")
        self.assertEqual(result["reference_status"], "nonconverged")
        self.assertNotIn("scipy_cg", result["median_seconds"])

    def test_missing_or_failed_owners_are_not_accepted(self):
        self.assertEqual(benchmark.summarize(self.records()[:-1])["status"], "failed")
        records = self.records()
        records[0] = {"status": "failed"}
        self.assertEqual(benchmark.summarize(records)["status"], "failed")

    def test_nonfinite_and_nonpositive_times(self):
        for value in (float("nan"), float("inf"), 0, -1):
            with self.assertRaises(ValueError):
                benchmark.control_spread(value, 1)

    def test_independent_residual_scalar_and_batch(self):
        A = benchmark.grid(4)
        for shape in ((16,), (16, 4)):
            x = np.ones(shape)
            b = A @ x
            self.assertEqual(len(benchmark.validate(A, b, SimpleNamespace(x=x, converged=True))),
                             1 if len(shape) == 1 else 4)
            with self.assertRaises(ValueError):
                benchmark.validate(A, b, SimpleNamespace(x=x * 2, converged=True))
            with self.assertRaises(ValueError):
                benchmark.validate(A, b, SimpleNamespace(x=x, converged=False))
            with self.assertRaises(ValueError):
                benchmark.validate(A, b, SimpleNamespace(x=x * float("nan"), converged=True))


if __name__ == "__main__":
    unittest.main()
