import io
import json
import pathlib
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

BENCH = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCH))

import runner_common as rc  # noqa: E402
import stale_cells  # noqa: E402
import sweep_fair  # noqa: E402


def record(status="complete", *, sha="old", cap=None,
           family="suitesparse", mid="com-Amazon",
           solver="apxchol_v1", config="", threads=16, device="cpu"):
    cell = {
        "schema": 2,
        "cell": {
            "family": family,
            "matrix_id": mid,
            "solver": solver,
            "config": config,
            "threads": threads,
            "device": device,
        },
        "matrix_meta": {"kind": "graph"},
        "metrics": {"total_s": 1.0},
        "provenance": {"git_sha": sha},
        "status": status,
    }
    if cap is not None:
        cell["timeout_cap_s"] = cap
    return cell


class StaleAwareResumeTest(unittest.TestCase):
    def write_cell(self, store, cell):
        identity = cell["cell"]
        path = pathlib.Path(rc.cell_path(
            identity["family"], identity["matrix_id"], identity["solver"],
            identity["config"], identity["threads"], identity["device"],
        ))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(cell))
        return path

    def done(self, cell, *, no_cap=False):
        identity = cell["cell"]
        with mock.patch.object(sweep_fair, "THREADS", identity["threads"]), \
             mock.patch.object(sweep_fair, "DEVICE", identity["device"]), \
             mock.patch.object(sweep_fair, "NO_CAP", no_cap):
            return sweep_fair.cell_done(
                identity["family"], identity["matrix_id"],
                identity["solver"], identity["config"],
            )

    def test_stale_complete_is_not_done(self):
        old = record()
        with tempfile.TemporaryDirectory() as store, \
             mock.patch.object(rc, "CELLS", store), \
             mock.patch.object(stale_cells, "sha_contains", return_value=False):
            self.write_cell(store, old)
            self.assertTrue(stale_cells.cell_is_stale(old))
            self.assertFalse(self.done(old))

    def test_stale_schema2_timeout_is_not_done(self):
        old = record("timeout", cap=600)
        with tempfile.TemporaryDirectory() as store, \
             mock.patch.object(rc, "CELLS", store), \
             mock.patch.object(stale_cells, "sha_contains", return_value=False):
            self.write_cell(store, old)
            self.assertTrue(stale_cells.cell_is_stale(old))
            self.assertFalse(self.done(old))

    def test_reusable_complete_still_skips(self):
        current = record(sha="current")
        with tempfile.TemporaryDirectory() as store, \
             mock.patch.object(rc, "CELLS", store), \
             mock.patch.object(stale_cells, "sha_contains", return_value=True):
            self.write_cell(store, current)
            self.assertFalse(stale_cells.cell_is_stale(current))
            self.assertTrue(self.done(current))

    def test_no_cap_timeout_still_reruns_before_stale_lookup(self):
        timeout = record("timeout", sha="current", cap=600,
                         family="audit", mid="m", solver="amgcl")
        with tempfile.TemporaryDirectory() as store, \
             mock.patch.object(rc, "CELLS", store), \
             mock.patch.object(stale_cells, "cell_is_stale") as is_stale:
            self.write_cell(store, timeout)
            self.assertFalse(self.done(timeout, no_cap=True))
        is_stale.assert_not_called()

    def test_stale_gpu_cap_reference_is_not_reused(self):
        old = record(device="gpu", config=sweep_fair.APX_DEFAULT_CONFIG)
        with tempfile.TemporaryDirectory() as store, \
             mock.patch.object(rc, "CELLS", store), \
             mock.patch.object(sweep_fair, "THREADS", 16), \
             mock.patch.object(stale_cells, "sha_contains", return_value=False):
            self.write_cell(store, old)
            self.assertIsNone(sweep_fair.gpu_apx_total("suitesparse", "com-Amazon"))

    def test_missing_and_malformed_cells_are_not_done(self):
        wanted = record(family="audit", mid="m", solver="amgcl")
        with tempfile.TemporaryDirectory() as store, \
             mock.patch.object(rc, "CELLS", store):
            self.assertFalse(self.done(wanted))
            path = self.write_cell(store, wanted)
            path.write_text("{not json")
            self.assertFalse(self.done(wanted))
            path.write_text(json.dumps({"status": "complete"}))
            self.assertFalse(self.done(wanted))

    def test_stale_cell_is_replaced_only_after_runner_returns(self):
        old = record()
        old_text = json.dumps(old)
        replacement = {"iters": 7, "total_s": 2.0}
        with tempfile.TemporaryDirectory() as store, \
             mock.patch.object(rc, "CELLS", store), \
             mock.patch.object(stale_cells, "sha_contains", return_value=False), \
             mock.patch.object(sweep_fair, "build_toolchain", return_value={}), \
             mock.patch.object(rc, "benchmark_openmp_provenance", return_value={}):
            path = self.write_cell(store, old)

            def runner():
                self.assertEqual(path.read_text(), old_text)
                return "complete", replacement

            sweep_fair.step("suitesparse", "com-Amazon", "apxchol_v1", "",
                            runner, "apxchol")
            saved = json.loads(path.read_text())
        self.assertEqual(saved["status"], "complete")
        self.assertEqual(saved["metrics"], replacement)

    def test_failed_replacement_runner_leaves_stale_cell_untouched(self):
        old = record()
        old_text = json.dumps(old)
        with tempfile.TemporaryDirectory() as store, \
             mock.patch.object(rc, "CELLS", store), \
             mock.patch.object(stale_cells, "sha_contains", return_value=False):
            path = self.write_cell(store, old)
            with self.assertRaisesRegex(RuntimeError, "runner failed"):
                sweep_fair.step(
                    "suitesparse", "com-Amazon", "apxchol_v1", "",
                    lambda: (_ for _ in ()).throw(RuntimeError("runner failed")),
                    "apxchol",
                )
            self.assertEqual(path.read_text(), old_text)

    def test_embedded_runner_hook_is_stale_aware_and_restored(self):
        old = record(solver="parac")
        with tempfile.TemporaryDirectory() as store, \
             mock.patch.object(rc, "CELLS", store), \
             mock.patch.object(stale_cells, "sha_contains", return_value=False):
            self.write_cell(store, old)
            original = rc.cell_done
            with sweep_fair._stale_aware_embedded_resume():
                self.assertIs(rc.cell_done, sweep_fair.stored_cell_done)
                self.assertFalse(rc.cell_done(
                    "suitesparse", "com-Amazon", "parac", "", 16, "cpu",
                    terminal=frozenset({"complete"}),
                ))
            self.assertIs(rc.cell_done, original)

    def test_stale_report_keeps_the_complete_store_denominator(self):
        old = record(sha="old")
        current = record(sha="current")
        with tempfile.TemporaryDirectory() as store, \
             mock.patch.object(stale_cells, "sha_contains",
                               side_effect=lambda sha, _commit: sha == "current"), \
             mock.patch.object(sys, "argv", ["stale_cells.py", "--store", store]), \
             mock.patch("sys.stdout", new_callable=io.StringIO) as stdout:
            pathlib.Path(store, "old.json").write_text(json.dumps(old))
            pathlib.Path(store, "current.json").write_text(json.dumps(current))
            stale_cells.main()
        lines = stdout.getvalue().splitlines()
        self.assertEqual(lines[0], f"store {store}: 2 cells total")
        self.assertIn("stale: 1 (50.0%)   reusable: 1", lines[1])


class ImportBoundaryTest(unittest.TestCase):
    def test_imports_have_no_resume_hook_side_effect_or_cycle(self):
        for imports in (("stale_cells", "sweep_fair"),
                        ("sweep_fair", "stale_cells")):
            with self.subTest(imports=imports):
                code = (
                    "import sys\n"
                    f"sys.path.insert(0, {str(BENCH)!r})\n"
                    "import runner_common as rc\n"
                    "before = rc.cell_done\n"
                    f"import {imports[0]}\n"
                    f"import {imports[1]}\n"
                    "assert rc.cell_done is before\n"
                )
                completed = subprocess.run(
                    [sys.executable, "-c", code],
                    cwd=BENCH.parent,
                    text=True,
                    capture_output=True,
                    check=False,
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)
                self.assertEqual(completed.stdout, "")
                self.assertEqual(completed.stderr, "")


if __name__ == "__main__":
    unittest.main()
