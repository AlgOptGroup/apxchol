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
    def test_representative_run_is_the_real_median_total_run(self):
        runs = [{"total": 4.0, "marker": "middle"},
                {"total": 2.0, "marker": "fast"},
                {"total": 6.0, "marker": "slow"}]
        index, chosen = parac._representative_run(runs, lambda run: run["total"])
        self.assertEqual(index, 0)
        self.assertIs(chosen, runs[0])

    def test_cpu_cell_uses_one_real_repetition_not_fieldwise_medians(self):
        def sample(adapter, factor_setup, solve_ms, iters, rr):
            return {"factor": "0.25", "factor_setup": str(factor_setup),
                    "adapter": str(adapter), "solve": str(solve_ms),
                    "etree": "0.1", "ftree": "0.1", "summary": "0.1",
                    "iters": str(iters), "rr": str(rr), "recur": "1e-9",
                    "rhs_norm": "2.0",
                    "n": 100, "nnz": 500, "rss_mb": 12.0}

        # Complete totals are 4.0, 2.0, 6.0 seconds respectively. The first
        # repetition is the median-total run; medians of individual fields would
        # fabricate adapter=1, factor_setup=.5, solve=1 instead.
        runs = [sample(3.0, 0.5, 500, 11, 8e-9),
                sample(0.5, 0.5, 1000, 22, 7e-9),
                sample(1.0, 1.0, 4000, 33, 6e-9)]
        with mock.patch.object(rc, "cell_done", return_value=False), \
             mock.patch.object(parac, "_calibrate_rel_tol", return_value=None), \
             mock.patch.object(parac, "_run_once_cpu", side_effect=runs), \
             mock.patch.object(rc, "emit_cell") as emit:
            parac._measure_cpu("synthetic", "grid_500", "input.mtx", 5.0,
                               False, "parac")

        metrics = emit.call_args.args[5]
        self.assertEqual(metrics["representative_repeat"], 1)
        self.assertEqual(metrics["setup_s"], 8.5)
        self.assertEqual(metrics["solve_s"], 0.5)
        self.assertEqual(metrics["total_s"], 9.0)
        self.assertEqual(metrics["iters"], 11)
        self.assertEqual(metrics["rel_res"], 8e-9)

    def test_upstream_preprocessing_uses_complete_interval_not_kernel_print(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            src = root / "input.mtx"; src.write_text("input")
            writer = root / "write_graph.jl"; writer.write_text("producer")
            prefix = root / "case"
            (root / "case-amd.mtx").write_text("output")
            log = pathlib.Path("/tmp/parac_produce_case_amd.log")
            log.write_text("amd time: 0.125\nAPX complete preprocessing time: 4.25\n")
            try:
                with mock.patch.object(parac, "_write_graph_jl", return_value=str(writer)), \
                     mock.patch.object(parac, "sh", return_value=subprocess.CompletedProcess(
                         "julia", 0, "", "")):
                    out, seconds, error = parac._produce_upstream(
                        str(prefix), str(src), "graph", "amd", 30)
            finally:
                log.unlink(missing_ok=True)

        self.assertEqual(out, str(prefix) + "-amd.mtx")
        self.assertEqual(seconds, 4.25)
        self.assertIsNone(error)

    def test_legacy_narrow_timer_cache_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            cached = pathlib.Path(tmp, "case-amd.mtx")
            cached.write_text("matrix")
            pathlib.Path(f"{cached}.time").write_text("0.125")
            pathlib.Path(f"{cached}.prep").write_text("legacy producer")
            self.assertFalse(parac._prep_cache_valid(str(cached)))

            parac._invalidate_prep_cache(str(cached))
            self.assertFalse(cached.exists())
            self.assertFalse(pathlib.Path(f"{cached}.time").exists())
            cached.write_text("rebuilt matrix")
            parac._stamp_prep_cache(str(cached), 4.25, "complete producer")
            self.assertTrue(parac._prep_cache_valid(str(cached)))
            self.assertEqual(pathlib.Path(f"{cached}.time").read_text(), "4.25")

    def test_preprocessing_cache_is_bound_to_the_exact_input_route(self):
        with tempfile.TemporaryDirectory() as tmp:
            cached = pathlib.Path(tmp, "case-amd.mtx")
            native = pathlib.Path(tmp, "native.mtx")
            dumped = pathlib.Path(tmp, "dumped.mtx")
            cached.write_text("prepared")
            native.write_text("same operator")
            dumped.write_text("same operator")
            parac._stamp_prep_cache(
                str(cached), 1.0, "complete producer", str(native))

            self.assertTrue(parac._prep_cache_valid(str(cached), str(native)))
            self.assertFalse(parac._prep_cache_valid(str(cached), str(dumped)))

    def test_gpu_provenance_records_repeat_count(self):
        self.assertEqual(parac.PROV_GPU["repeat"], parac.REPS)

    def test_gpu_cell_timeout_is_one_shared_deadline(self):
        with mock.patch.object(parac.time, "monotonic", return_value=15.5):
            self.assertEqual(parac._gpu_cell_remaining(20.0), 4.5)
        with mock.patch.object(parac.time, "monotonic", return_value=20.0):
            with self.assertRaises(subprocess.TimeoutExpired):
                parac._gpu_cell_remaining(20.0)

    def test_cpu_cell_timeout_is_one_shared_deadline(self):
        with mock.patch.object(parac.time, "monotonic", return_value=15.5):
            self.assertEqual(parac._cpu_cell_remaining(20.0), 4.5)
        with mock.patch.object(parac.time, "monotonic", return_value=20.0):
            with self.assertRaises(subprocess.TimeoutExpired):
                parac._cpu_cell_remaining(20.0)

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
            "recur": "1e-9", "rhs_norm": "2.0",
            "n": 100, "nnz": 500, "rss_mb": 12.0,
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

    def test_cpu_graph_route_solves_a_two_vertex_component(self):
        def sample(total, rr, rhs_norm, nnz):
            return {"factor": "0.25", "factor_setup": "0.1",
                    "adapter": str(total - 0.2), "solve": "100",
                    "etree": "0.1", "ftree": "0.1", "summary": "0.1",
                    "iters": "10", "rr": str(rr), "rhs_norm": str(rhs_norm),
                    "recur": "1e-9", "n": 10, "nnz": nnz, "rss_mb": 12.0}
        # Per-component medians are repetition 2, but the coherent aggregate
        # totals [101, 4, 101] select repetition 1 (one-based). Component RHS
        # norms differ, so max(component rr) is not the global residual.
        runs = [sample(1, 1e-9, 2.0, 50), sample(2, 2e-9, 2.0, 50),
                sample(100, 3e-9, 2.0, 50),
                sample(100, 9e-9, 1.0, 5), sample(2, 8e-9, 1.0, 5),
                sample(1, 7e-9, 1.0, 5)]
        with mock.patch.object(rc, "cell_done", return_value=False), \
             mock.patch.object(parac, "_dump_component", side_effect=[
                 ("large.mtx", 10, 3, 0.2), ("pair.mtx", 2, 3, 0.3),
                 ("single.mtx", 1, 3, 0.4)]), \
             mock.patch.object(parac, "_reorder_amd",
                               side_effect=[("large-amd.mtx", 0.1, "upstream"),
                                            ("pair-amd.mtx", 0.1, "upstream")]) as reorder, \
             mock.patch.object(parac, "_calibrate_rel_tol", return_value=None), \
             mock.patch.object(parac, "_run_once_cpu", side_effect=runs), \
             mock.patch.object(rc, "emit_cell") as emit:
            parac._measure_cpu_graph_split("grids", "grid_500")

        self.assertEqual(reorder.call_count, 2)
        metrics = emit.call_args.args[5]
        self.assertEqual(metrics["n_components_solved"], 2)
        self.assertEqual(metrics["n_components_total"], 3)
        self.assertEqual(metrics["n"], rc.MATRICES["grid_500"]["n"])
        self.assertEqual(metrics["representative_repeat"], 1)
        # The final singleton was genuinely extracted and serialized before its
        # size was known, so its 0.4 s adapter interval is charged as well.
        self.assertAlmostEqual(metrics["input_dump_s"], 0.9)
        self.assertAlmostEqual(metrics["rhs_norm"], 5 ** 0.5)
        self.assertAlmostEqual(metrics["rel_res"], (97 / 5) ** 0.5 * 1e-9)

    def test_gpu_setup_and_solve_use_complete_intervals(self):
        sample = {
            "cuda_init": "0.75", "adapter": "3.0", "factor_setup": "2.0",
            "solver_setup": "1.0",
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
            component = pathlib.Path(sorted_dir, "component.mtx")
            component.write_text("component")
            cached = pathlib.Path(sorted_dir, "grid_500-pure-nnz-sorted.mtx")
            cached.write_text("cached")
            parac._stamp_prep_cache(str(cached), 5.0, "fixture", str(component))
            with mock.patch.object(
                    parac, "_dump_component",
                    return_value=(str(component), 100, 1, 0.5)):
                parac.run_gpu("grid_500")

            path = rc.cell_path(rc.MATRICES["grid_500"]["family"], "grid_500",
                                "parac_graph", "", 16, "gpu")
            with open(path) as handle:
                metrics = json.load(handle)["metrics"]

        self.assertEqual(metrics["input_dump_s"], 0.5)
        self.assertEqual(metrics["adapter_setup_s"], 8.5)
        self.assertEqual(metrics["native_setup_s"], 3.0)
        self.assertEqual(metrics["setup_s"], 11.5)
        self.assertEqual(metrics["solve_s"], 4.0)
        self.assertEqual(metrics["total_s"], 15.5)
        self.assertEqual(metrics["pcg_kernel_s"], 3.5)
        self.assertEqual(metrics["cuda_init_s"], 0.75)


class ParacGpuFailureCellTest(unittest.TestCase):
    def read_cell(self, mid, solver):
        family = rc.MATRICES[mid]["family"]
        path = rc.cell_path(family, mid, solver, "", 16, "gpu")
        with open(path) as handle:
            return json.load(handle)

    def test_nnz_sort_failure_stamps_failed_and_incompatible_na(self):
        with tempfile.TemporaryDirectory() as store, \
             tempfile.TemporaryDirectory() as sorted_dir, \
             mock.patch.object(rc, "CELLS", store), \
             mock.patch.object(rc, "PARAC_SORTED", sorted_dir), \
             mock.patch.object(parac, "_dump_component",
                               return_value=("component.mtx", 100, 1, 0.2)), \
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
        timeout = subprocess.TimeoutExpired("nnz-sort", parac.TIMEOUT_GPU)
        with tempfile.TemporaryDirectory() as store, \
             tempfile.TemporaryDirectory() as sorted_dir, \
             mock.patch.object(rc, "CELLS", store), \
             mock.patch.object(rc, "PARAC_SORTED", sorted_dir), \
             mock.patch.object(parac, "_nnz_sort", side_effect=timeout), \
             mock.patch.object(rc, "binary_toolchain", return_value={}):
            parac.run_gpu("iter0010")

            graph = self.read_cell("iter0010", "parac_graph")
            physics = self.read_cell("iter0010", "parac_physics")
            self.assertEqual(graph["status"], "n/a")
            self.assertEqual(physics["status"], "timeout")
            self.assertEqual(physics["timeout_cap_s"], parac.TIMEOUT_GPU)

    def test_file_backed_operator_goes_directly_to_parac_producer(self):
        with mock.patch.object(parac, "_dump") as dump, \
             mock.patch.object(parac, "_reorder_amd",
                               return_value=("prepared.mtx", 1.25, "upstream")) as reorder:
            prepared = parac._prep_amd("iter0010", "op", augment=True)

        dump.assert_not_called()
        self.assertEqual(reorder.call_args.args[1], rc.MATRICES["iter0010"]["spec"])
        self.assertEqual(reorder.call_args.args[2], "native-op")
        self.assertEqual(prepared, ("prepared.mtx", 1.25, "upstream", 0.0))

    def test_component_info_cache_avoids_serialization_and_reprobe(self):
        stderr = ("[component-info] 3 components: largest 17 / 20 nodes\n"
                  "APX component discovery time: 0.125\n")
        with tempfile.TemporaryDirectory() as dump_dir:
            source = pathlib.Path(dump_dir, "input.mtx")
            source.write_text("input")
            native = mock.patch.object(parac, "_native_mtx", return_value=str(source))
            run_patch = mock.patch.object(
                parac, "sh", return_value=subprocess.CompletedProcess(
                    "benchmark", 0, "", stderr))
            with native, run_patch as run:
                first = parac._component_info(
                    "com-Amazon", dump_dir, "benchmark", 30, mem_cap_gb=1)
                second = parac._component_info(
                    "com-Amazon", dump_dir, "benchmark", 30, mem_cap_gb=1)

        self.assertEqual(first, (3, 17, 0.125))
        self.assertEqual(second, first)
        self.assertEqual(run.call_count, 1)
        self.assertIn("--component-info", run.call_args.args[0])
        self.assertNotIn("--dump-mtx", run.call_args.args[0])

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
            component = pathlib.Path(sorted_dir, "component.mtx")
            component.write_text("component")
            cached = pathlib.Path(sorted_dir, "grid_500-pure-nnz-sorted.mtx")
            cached.write_text("cached")
            parac._stamp_prep_cache(str(cached), 0.0, "fixture", str(component))
            with mock.patch.object(
                    parac, "_dump_component",
                    return_value=(str(component), 100, 1, 0.0)):
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

    def test_disconnected_gpu_graph_is_explicit_na_not_an_invalid_solve(self):
        with mock.patch.object(parac, "_dump_component",
                               return_value=("giant.mtx", 100, 5, 0.2)) as dump, \
             mock.patch.object(rc, "binary_toolchain", return_value={}), \
             mock.patch.object(rc, "emit_cell") as emit:
            result = parac.run_gpu("grid_500")

        self.assertIn("parac_graph[n/a]", result)
        self.assertEqual(dump.call_args.kwargs["dump_dir"], parac.DUMP_GPU)
        self.assertIsNone(dump.call_args.kwargs["mem_cap_gb"])
        self.assertEqual(emit.call_count, 2)
        graph_call = next(call for call in emit.call_args_list
                          if call.args[2] == "parac_graph")
        self.assertEqual(graph_call.args[4], "n/a")
        self.assertIn("component-wise compatible",
                      graph_call.kwargs["matrix_meta"]["parac_na_reason"])

    def test_native_gpu_component_probe_failure_is_fail_closed(self):
        with tempfile.TemporaryDirectory() as store, \
             mock.patch.object(rc, "CELLS", store), \
             mock.patch.object(parac, "_component_info", return_value=(0, 0, 0.0)), \
             mock.patch.object(rc, "binary_toolchain", return_value={}), \
             mock.patch.object(rc, "emit_cell") as emit:
            result = parac.run_gpu("com-Amazon")

        self.assertIn("FAILED(prep)", result)
        graph_call = next(call for call in emit.call_args_list
                          if call.args[2] == "parac_graph")
        self.assertEqual(graph_call.args[4], "failed")
        self.assertIn("component-info probe produced no result",
                      graph_call.kwargs["matrix_meta"]["parac_prep_failure"])


class ParacCpuFailureCellTest(unittest.TestCase):
    def test_graph_component_reorder_failure_fails_the_whole_cell(self):
        with mock.patch.object(rc, "cell_done", return_value=False), \
             mock.patch.object(parac, "_dump_component",
                               return_value=("component.mtx", 100, 1, 0.2)), \
             mock.patch.object(parac, "_reorder_amd",
                               return_value=(None, 0.3, "producer failed")), \
             mock.patch.object(parac, "_run_once_cpu") as run, \
             mock.patch.object(rc, "emit_cell") as emit:
            result = parac._measure_cpu_graph_split("grids", "grid_500")

        self.assertEqual(result, "FAILED")
        run.assert_not_called()
        self.assertEqual(emit.call_args.args[4], "failed")
        self.assertIn("component 0",
                      emit.call_args.kwargs["matrix_meta"]["parac_prep_failure"])

    def test_operator_reorder_failure_emits_failed_physics_cell(self):
        with mock.patch.object(parac, "_prep_amd",
                               return_value=(None, 0.3, "producer failed", 0.2)), \
             mock.patch.object(rc, "emit_cell") as emit:
            result = parac._run_cpu_operator("iter0010", "ipm")

        self.assertIn("FAILED", result)
        physics = next(call for call in emit.call_args_list
                       if call.args[2] == "parac_physics")
        self.assertEqual(physics.args[4], "failed")
        self.assertIn("produced no matrix",
                      physics.kwargs["matrix_meta"]["parac_prep_failure"])


if __name__ == "__main__":
    unittest.main()
