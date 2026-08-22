#!/usr/bin/env python3
import json
import pathlib
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

BENCH = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCH))

import parac_runner as parac  # noqa: E402
import runner_common as rc  # noqa: E402


class ParacGpuFailureCellTest(unittest.TestCase):
    def read_cell(self, mid, solver):
        family = rc.MATRICES[mid]["family"]
        path = rc.cell_path(family, mid, solver, "", 16, "gpu")
        with open(path) as handle:
            return json.load(handle)

    def test_nnz_sort_failure_stamps_failed_and_incompatible_na(self):
        with tempfile.TemporaryDirectory() as store, \
             mock.patch.object(rc, "CELLS", store), \
             mock.patch.object(parac, "_dump", return_value=("input.mtx", "pure")), \
             mock.patch.object(parac, "_nnz_sort",
                               return_value=(None, 0.0, "fallback failed rc=1")), \
             mock.patch.object(rc, "binary_toolchain", return_value={}):
            result = parac.run_gpu("grid_500")

            graph = self.read_cell("grid_500", "parac_graph")
            physics = self.read_cell("grid_500", "parac_physics")
            self.assertEqual(graph["status"], "failed")
            self.assertIn("produced no matrix",
                          graph["matrix_meta"]["parac_prep_failure"])
            self.assertEqual(graph["matrix_meta"]["parac_prep"],
                             "fallback failed rc=1")
            self.assertEqual(physics["status"], "n/a")
            self.assertIn("FAILED(prep)", result)

    def test_preparation_timeout_stamps_timeout_with_exact_cap(self):
        timeout = subprocess.TimeoutExpired("dump", parac.TIMEOUT_GPU)
        with tempfile.TemporaryDirectory() as store, \
             mock.patch.object(rc, "CELLS", store), \
             mock.patch.object(parac, "_dump", side_effect=timeout), \
             mock.patch.object(rc, "binary_toolchain", return_value={}):
            parac.run_gpu("grid_500")

            graph = self.read_cell("grid_500", "parac_graph")
            physics = self.read_cell("grid_500", "parac_physics")
            self.assertEqual(graph["status"], "timeout")
            self.assertEqual(graph["timeout_cap_s"], parac.TIMEOUT_GPU)
            self.assertEqual(physics["status"], "n/a")

    def test_operator_failure_stamps_physics_and_graph_na(self):
        with tempfile.TemporaryDirectory() as store, \
             mock.patch.object(rc, "CELLS", store), \
             mock.patch.object(rc, "binary_toolchain", return_value={}):
            parac.record_gpu_failure("iter0010", "failed", "prep failed")

            graph = self.read_cell("iter0010", "parac_graph")
            physics = self.read_cell("iter0010", "parac_physics")
            self.assertEqual(graph["status"], "n/a")
            self.assertEqual(physics["status"], "failed")
            self.assertEqual(physics["matrix_meta"]["parac_mode"], "physics")

    def test_missing_driver_stamps_failed_instead_of_silent_skip(self):
        with tempfile.TemporaryDirectory() as store, \
             tempfile.TemporaryDirectory() as sorted_dir, \
             mock.patch.object(rc, "CELLS", store), \
             mock.patch.object(rc, "PARAC_SORTED", sorted_dir), \
             mock.patch.object(rc, "PARAC_GPU_DRIVER", "/missing/parac-graph"), \
             mock.patch.object(rc, "PARAC_GPU_DRIVER_PHYS", "/missing/parac-physics"), \
             mock.patch.object(rc, "binary_toolchain", return_value={}):
            cached = pathlib.Path(sorted_dir, "grid_500-pure-nnz-sorted.mtx")
            cached.write_text("cached")
            pathlib.Path(f"{cached}.time").write_text("0.0")
            parac.run_gpu("grid_500")

            graph = self.read_cell("grid_500", "parac_graph")
            physics = self.read_cell("grid_500", "parac_physics")
            self.assertEqual(graph["status"], "failed")
            self.assertIn("driver missing",
                          graph["matrix_meta"]["parac_prep_failure"])
            self.assertEqual(physics["status"], "n/a")

    def test_failed_fallback_reports_return_code_and_emits_no_cache_stamp(self):
        with tempfile.TemporaryDirectory() as sorted_dir, \
             mock.patch.object(rc, "PARAC_SORTED", sorted_dir), \
             mock.patch.object(parac, "_produce_upstream",
                               return_value=(None, 0.0, "upstream rc=1")), \
             mock.patch.object(parac, "sh", return_value=subprocess.CompletedProcess(
                 "julia", 9, "", "fallback exploded")):
            path, _secs, prep = parac._nnz_sort(
                "grid_500", "input.mtx", "pure")

            self.assertIsNone(path)
            self.assertIn("rc=9", prep)
            self.assertIn("fallback exploded", prep)
            self.assertFalse(any(pathlib.Path(sorted_dir).glob("*.time")))


if __name__ == "__main__":
    unittest.main()
