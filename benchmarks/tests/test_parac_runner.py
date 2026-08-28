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


class ParacRoutingTest(unittest.TestCase):
    def test_route_is_by_operator_class_not_file_kind(self):
        # ecology1 is a published operator file, but the operator is an exact
        # singular Laplacian. It must use graph mode; physics mode would trim a
        # real vertex. Full-rank operator matrices use physics as before.
        self.assertEqual(rc.MATRICES["ecology1"]["kind"], "operator")
        self.assertEqual(rc.class_of("ecology1"), "laplacian")
        self.assertFalse(parac._uses_physics("ecology1"))
        self.assertFalse(parac._uses_physics("com-Amazon"))
        self.assertTrue(parac._uses_physics("iter0040"))

        ecology_modes = {
            key: reason for key, _driver, _provenance, reason
            in parac._gpu_modes("ecology1")
        }
        self.assertIsNone(ecology_modes["parac_graph"])
        self.assertEqual(ecology_modes["parac_physics"], parac.PHYSICS_NA)


class ParacTimingContractTest(unittest.TestCase):
    def test_gpu_provenance_records_repeat_count(self):
        self.assertEqual(parac.PROV_GPU["repeat"], parac.REPS)

    def test_gpu_cell_timeout_is_one_shared_deadline(self):
        with mock.patch.object(parac.time, "monotonic", return_value=15.5):
            self.assertEqual(parac._gpu_cell_remaining(20.0), 4.5)
        with mock.patch.object(parac.time, "monotonic", return_value=20.0):
            with self.assertRaises(subprocess.TimeoutExpired):
                parac._gpu_cell_remaining(20.0)

    def test_gpu_calibration_tightens_but_never_relaxes_common_tolerance(self):
        with mock.patch.object(parac, "_run_once_gpu", return_value={"rr": "5e-8"}):
            self.assertAlmostEqual(parac._calibrate_tol_gpu("driver", "matrix", 1e-8),
                                   2e-9)
        with mock.patch.object(parac, "_run_once_gpu", return_value={"rr": "5e-9"}):
            self.assertEqual(parac._calibrate_tol_gpu("driver", "matrix", 1e-8),
                             1e-8)

    def test_cpu_setup_uses_complete_nonoverlapping_intervals(self):
        sample = {
            "factor": "1.0", "factor_setup": "2.0", "adapter": "3.0",
            "solve": "4000", "etree": "0.2", "ftree": "0.3",
            "summary": "0.4", "iters": "10", "rr": "1e-9",
            "recur": "1e-9", "n": 100, "nnz": 500, "rss_mb": 12.0,
        }
        with mock.patch.object(rc, "cell_done", return_value=False), \
             mock.patch.object(parac, "_calibrate_rel_tol", return_value=None), \
             mock.patch.object(parac, "_run_once_cpu", return_value=sample), \
             mock.patch.object(rc, "emit_cell") as emit:
            parac._measure_cpu("synthetic", "grid_500", "input.mtx", 5.0,
                               False, "parac")

        metrics = emit.call_args.args[5]
        self.assertEqual(metrics["adapter_setup_s"], 8.0)  # reorder + adapter
        self.assertEqual(metrics["native_setup_s"], 2.0)
        self.assertEqual(metrics["setup_s"], 10.0)
        self.assertEqual(metrics["solve_s"], 4.0)
        self.assertEqual(metrics["total_s"], 14.0)
        # The narrower upstream kernel timer remains diagnostic only.
        self.assertEqual(metrics["factor_kernel_s"], 1.0)

    def test_gpu_setup_and_solve_use_complete_intervals(self):
        sample = {
            "adapter": "3.0", "factor_setup": "2.0", "solver_setup": "1.0",
            "solve_total": "4.0", "factor": "500", "conv": "250",
            "spsv": "125", "solve": "3500", "etree": "0.2",
            "ftree": "0.3", "summary": "0.4", "iters": "10",
            "rr": "1e-9", "n": "100", "nnz": "500", "vram_mb": 20.0,
        }
        with tempfile.TemporaryDirectory() as store, \
             tempfile.TemporaryDirectory() as sorted_dir, \
             tempfile.TemporaryDirectory() as drivers, \
             mock.patch.object(rc, "CELLS", store), \
             mock.patch.object(rc, "PARAC_SORTED", sorted_dir), \
             mock.patch.object(rc, "PARAC_GPU_DRIVER", f"{drivers}/graph"), \
             mock.patch.object(rc, "PARAC_GPU_DRIVER_PHYS", f"{drivers}/physics"), \
             mock.patch.object(rc, "binary_toolchain", return_value={}), \
             mock.patch.object(parac, "_calibrate_tol_gpu", return_value=1e-8), \
             mock.patch.object(parac, "_run_once_gpu", return_value=sample):
            pathlib.Path(drivers, "graph").touch()
            pathlib.Path(drivers, "physics").touch()
            cached = pathlib.Path(sorted_dir, "grid_500-pure-nnz-sorted.mtx")
            cached.write_text("cached")
            pathlib.Path(f"{cached}.time").write_text("5.0")
            parac.run_gpu("grid_500")

            path = rc.cell_path(rc.MATRICES["grid_500"]["family"], "grid_500",
                                "parac_graph", "", 16, "gpu")
            with open(path) as handle:
                metrics = json.load(handle)["metrics"]

        self.assertEqual(metrics["adapter_setup_s"], 8.0)
        self.assertEqual(metrics["native_setup_s"], 3.0)
        self.assertEqual(metrics["setup_s"], 11.0)
        self.assertEqual(metrics["solve_s"], 4.0)
        self.assertEqual(metrics["total_s"], 15.0)
        self.assertEqual(metrics["pcg_kernel_s"], 3.5)


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
