#!/usr/bin/env python3
import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from benchmarks import runner_common as rc


class RetainedResidualTest(unittest.TestCase):
    def test_median_timing_cannot_hide_failed_retained_solve(self):
        status, _ = rc.classify({"rel_res": 1e-9, "max_repeat_rel_res": 2e-8}, 1e-8)
        self.assertEqual(status, "not_converged")

    def test_nonfinite_retained_residual_is_not_complete(self):
        status, _ = rc.classify({"rel_res": 1e-9, "max_repeat_rel_res": float("inf")}, 1e-8)
        self.assertEqual(status, "not_converged")

    def test_repeat_fields_survive_csv_parse(self):
        metrics = rc.parse_csv("Apx,grid,4,12,0.01,0.02,0.03,7,1e-9,1,2,3,-1,3,2,9e-9\n")
        self.assertEqual(metrics["retained_repeats"], 3)
        self.assertEqual(metrics["representative_repeat"], 2)
        self.assertEqual(metrics["max_repeat_rel_res"], 9e-9)
        self.assertEqual(rc.classify(metrics, 1e-8)[0], "complete")


class ExternalPathConfigTest(unittest.TestCase):
    def test_explicit_environment_wins_over_machine_default(self):
        local = SimpleNamespace(PARAC_REORD="/machine/default")
        with mock.patch.object(rc, "_paths_local", local), \
             mock.patch.dict(os.environ,
                             {"APXCHOL_PARAC_REORDER_DIR": "/campaign/cache"},
                             clear=False):
            self.assertEqual(
                rc.external_path("APXCHOL_PARAC_REORDER_DIR", "PARAC_REORD"),
                "/campaign/cache")

    def test_explicit_empty_environment_disables_machine_default(self):
        local = SimpleNamespace(PARAC_REORD="/machine/default")
        with mock.patch.object(rc, "_paths_local", local), \
             mock.patch.dict(os.environ,
                             {"APXCHOL_PARAC_REORDER_DIR": ""}, clear=False):
            self.assertEqual(
                rc.external_path("APXCHOL_PARAC_REORDER_DIR", "PARAC_REORD"), "")

    def test_machine_default_precedes_builtin_default(self):
        local = SimpleNamespace(CMG_SOLVER="/machine/cmg")
        with mock.patch.object(rc, "_paths_local", local), \
             mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("APXCHOL_CMG_SOLVER", None)
            self.assertEqual(
                rc.external_path("APXCHOL_CMG_SOLVER", "CMG_SOLVER", "/builtin"),
                "/machine/cmg")


class AffinitySpecTest(unittest.TestCase):
    def test_uses_cpus_granted_to_packed_rank(self):
        with mock.patch.object(os, "sched_getaffinity",
                               return_value=set(range(72, 144))), \
             mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("APXCHOL_BENCH_CPUSET", None)
            self.assertEqual(rc.affinity_cpus(4), [72, 73, 74, 75])
            self.assertEqual(rc.affinity_spec(4), "72-75")
            self.assertEqual(rc.taskset_prefix(4), "taskset -c 72-75")

    def test_compresses_noncontiguous_affinity(self):
        with mock.patch.object(os, "sched_getaffinity",
                               return_value={2, 3, 8, 10}):
            self.assertEqual(rc.affinity_spec(4), "2-3,8,10")

    def test_rejects_unsafe_override(self):
        with mock.patch.dict(os.environ,
                             {"APXCHOL_BENCH_CPUSET": "0-3;hostname"}):
            with self.assertRaises(ValueError):
                rc.affinity_spec(4)

    def test_override_is_expanded_and_bounded_by_thread_count(self):
        with mock.patch.dict(os.environ,
                             {"APXCHOL_BENCH_CPUSET": "8-9,12,14"}):
            self.assertEqual(rc.affinity_cpus(3), [8, 9, 12])
            self.assertEqual(rc.affinity_spec(3), "8-9,12")

    def test_rejects_duplicate_override_cpus(self):
        with mock.patch.dict(os.environ,
                             {"APXCHOL_BENCH_CPUSET": "1-2,2"}):
            with self.assertRaises(ValueError):
                rc.affinity_cpus(2)

    def test_mixed_runtime_env_is_explicit_and_rank_local(self):
        with mock.patch.object(os, "sched_getaffinity",
                               return_value={72, 73, 74, 75}), \
             mock.patch.dict(os.environ, {
                 "UNRELATED": "kept",
                 "GOMP_CPU_AFFINITY": "stale parent setting",
             }, clear=True):
            env = rc.benchmark_openmp_env(3)
            self.assertEqual(env["OMP_NUM_THREADS"], "3")
            self.assertEqual(env["OMP_DYNAMIC"], "FALSE")
            self.assertEqual(env["OMP_PROC_BIND"], "close")
            self.assertEqual(env["OMP_PLACES"], "{72},{73},{74}")
            self.assertEqual(env["KMP_AFFINITY"], "norespect")
            self.assertNotIn("GOMP_CPU_AFFINITY", env)
            self.assertEqual(env["UNRELATED"], "kept")

    def test_affinity_provenance_matches_subprocess_env(self):
        with mock.patch.object(os, "sched_getaffinity",
                               return_value={2, 3, 8, 10}):
            self.assertEqual(rc.benchmark_openmp_provenance(3), {
                "benchmark_cpuset": "2-3,8",
                "benchmark_omp_places": "{2},{3},{8}",
                "benchmark_kmp_affinity": "norespect",
            })

    def test_platform_without_affinity_runs_unpinned(self):
        with mock.patch.object(rc, "affinity_supported", return_value=False), \
             mock.patch.object(rc.os, "cpu_count", return_value=8), \
             mock.patch.dict(os.environ, {"GOMP_CPU_AFFINITY": "stale", "OMP_PLACES": "{0}",
                                          "OMP_PROC_BIND": "close", "KMP_AFFINITY": "compact"},
                             clear=True):
            self.assertIsNone(rc.affinity_cpus(4))
            self.assertEqual(rc.taskset_prefix(4), "")
            env = rc.benchmark_openmp_env(4)
            self.assertEqual(env["OMP_NUM_THREADS"], "4")
            for key in ("OMP_PLACES", "OMP_PROC_BIND", "KMP_AFFINITY", "GOMP_CPU_AFFINITY"):
                self.assertNotIn(key, env)
            self.assertEqual(rc.benchmark_openmp_provenance(4), {"benchmark_cpuset": "unpinned"})
            with self.assertRaises(RuntimeError):
                rc.affinity_cpus(9)
            os.environ["APXCHOL_BENCH_CPUSET"] = "0-3"
            with self.assertRaises(RuntimeError):
                rc.affinity_cpus(4)


class PeakRssTest(unittest.TestCase):
    def test_gnu_and_bsd_time_reports(self):
        self.assertEqual(rc.parse_peak_rss_mb("x\nAPXRSS 2048\n"), 2.0)
        bsd = ("        0.10 real         0.05 user         0.01 sys\n"
               "             3145728  maximum resident set size\n"
               "                   0  average shared memory size\n")
        self.assertEqual(rc.parse_peak_rss_mb(bsd), 3.0)
        self.assertIsNone(rc.parse_peak_rss_mb("no report\n"))

    def test_prefix_matches_the_platform_time(self):
        with mock.patch.object(rc.sys, "platform", "darwin"):
            self.assertEqual(rc.peak_rss_prefix(), "/usr/bin/time -l")
        with mock.patch.object(rc.sys, "platform", "linux"):
            self.assertEqual(rc.peak_rss_prefix(), "/usr/bin/time -f 'APXRSS %M'")


class DeviceInitTest(unittest.TestCase):
    def test_metal_init_is_reported_in_seconds(self):
        line = "[bench] metal_init (once, before any timed solver): 250.0 ms\n"
        self.assertEqual(rc.parse_metal_init(line), 0.25)
        self.assertIsNone(rc.parse_cuda_init(line))


class CsvMetricTest(unittest.TestCase):
    def test_parallel_rchol_effective_thread_count_is_preserved(self):
        output = (
            "solver,graph,n,nnz,setup_s,solve_s,total_s,iters,rel_res,"
            "fillin,us_per_nnz,solve_rss_mb,solve_vram_mb\n"
            "pRCHOL+PCG [Chen20;par] t=64,grid,10,20,1,2,3,4,5e-9,"
            "6,7,8,-1\n"
        )
        metrics = rc.parse_csv(output)
        self.assertEqual(metrics["effective_threads"], 64)

    def test_unannotated_solver_does_not_invent_effective_threads(self):
        output = (
            "solver,graph,n,nnz,setup_s,solve_s,total_s,iters,rel_res,"
            "fillin,us_per_nnz,solve_rss_mb,solve_vram_mb\n"
            "AMGCL,grid,10,20,1,2,3,4,5e-9,6,7,8,-1\n"
        )
        self.assertNotIn("effective_threads", rc.parse_csv(output))




class CompleteRouteReceiptTest(unittest.TestCase):
    def test_route_and_stop_receipts_parse_together(self):
        for route in ("cpu", "gpu", "metal"):
            row = rc.parse_csv("v1,m,4,12,1,2,3,7,1e-9,1,2,3,-1,3,2,9e-9,original-v1,2,0.25," + route + "\n")
            self.assertEqual(row["execution_route"], route)
            self.assertEqual(row["stop_contract"], "original-v1")
            self.assertEqual(row["retained_repeats"], 3)
            self.assertEqual(row["stop_check_s"], 0.25)


GPU_RECEIPT = ("[bench] gpu_memory pid=4242 bus_id=00000000:01:00.0 baseline_mib=238.0 "
               "other_processes=1 other_mib=512.0\n")


class GpuMemoryTest(unittest.TestCase):
    def test_nvml_columns_parse_with_repeat_receipts(self):
        row = rc.parse_csv("v1,m,4,12,1,2,3,7,1e-9,1,2,3,96.5,120.0,3,2,9e-9,original-v1,2,0.25,gpu\n")
        self.assertEqual(row["solve_vram_mb"], 96.5)
        self.assertEqual(row["setup_vram_mb"], 120.0)
        self.assertEqual(row["retained_repeats"], 3)
        self.assertEqual(row["max_repeat_rel_res"], 9e-9)

    def test_unavailable_vram_is_absent(self):
        row = rc.parse_csv("v1,m,4,12,1,2,3,7,1e-9,1,2,3,-1,-1,3,2,9e-9,original-v1,2,0.25,cpu\n")
        self.assertNotIn("solve_vram_mb", row)
        self.assertNotIn("setup_vram_mb", row)
        self.assertEqual(row["retained_repeats"], 3)

    def test_device_wide_vram_from_older_binaries_is_not_stored(self):
        row = rc.parse_csv("v1,m,4,12,1,2,3,7,1e-9,1,2,3,900,3,2,9e-9,original-v1,2,0.25,gpu\n")
        self.assertNotIn("solve_vram_mb", row)
        self.assertEqual(row["retained_repeats"], 3)

    def test_receipt_reports_baseline_and_other_processes(self):
        self.assertEqual(rc.parse_gpu_memory("noise\n" + GPU_RECEIPT), {
            "vram_baseline_mb": 238.0, "gpu_other_processes": 1,
            "gpu_other_mb": 512.0, "gpu_shared": True})
        self.assertEqual(rc.parse_gpu_memory("cpu route only\n"), {})

    def test_unavailable_receipt_is_none_not_zero(self):
        metrics = rc.parse_gpu_memory(
            "[bench] gpu_memory pid=7 bus_id=unknown baseline_mib=unavailable reason=mig "
            "other_processes=unknown other_mib=unknown\n")
        self.assertEqual(metrics, {
            "vram_baseline_mb": None, "vram_unavailable": "mig", "gpu_other_processes": None,
            "gpu_other_mb": None, "gpu_shared": None})

    def test_multiword_nvml_reason_keeps_sharing_metadata(self):
        metrics = rc.parse_gpu_memory(
            "[bench] gpu_memory pid=7 bus_id=00000000:01:00.0 baseline_mib=unavailable "
            "reason=Insufficient Permissions other_processes=2 other_mib=640.0\n")
        self.assertEqual(metrics, {
            "vram_baseline_mb": None, "vram_unavailable": "Insufficient Permissions",
            "gpu_other_processes": 2, "gpu_other_mb": 640.0, "gpu_shared": True})

    def test_old_rows_do_not_invent_route_and_unknown_routes_fail(self):
        old = "v1,m,4,12,1,2,3,7,1e-9,1,2,original-v1,2,0.25"
        self.assertNotIn("execution_route", rc.parse_csv(old + "\n"))
        self.assertIsNone(rc.parse_csv(old + ",hybrid\n"))
        self.assertNotIn("execution_route", rc.parse_csv(old + ",\n"))

    def test_fill_with_no_denominator_is_unavailable(self):
        row = rc.parse_csv("v1,m,1,1,1,2,3,1,0,nan,3,original-v1,1,0.25,cpu\n")
        self.assertIsNone(row["fillin"])

    def test_every_warmup_and_retained_route_is_checked(self):
        lines = ["BENCH_REPEAT phase=warmup index=1 execution_route=gpu",
                 "BENCH_REPEAT phase=retained index=1 execution_route=gpu",
                 "BENCH_REPEAT phase=retained index=2 execution_route=gpu"]
        metrics = {"execution_route": "gpu"}
        self.assertIsNone(rc.v1_route_error(metrics, "\n".join(lines), "gpu", 2, 1))
        for bad in (lines[:-1], lines + [lines[-1]],
                    [line.replace("route=gpu", "route=cpu") for line in lines],
                    [line.replace(" execution_route=gpu", "") for line in lines]):
            self.assertIsNotNone(rc.v1_route_error(metrics, "\n".join(bad), "gpu", 2, 1))
        self.assertIsNotNone(rc.v1_route_error({}, "\n".join(lines), "gpu", 2, 1))


class MatrixExportTest(unittest.TestCase):
    def test_failed_and_interrupted_exports_are_not_cached(self):
        with tempfile.TemporaryDirectory(prefix="export test ") as directory:
            calls = []
            def run(command, **kwargs):
                args = shlex.split(command)
                self.assertEqual(args[0], "/binary with spaces/benchmark")
                output = Path(args[args.index("--dump-mtx") + 1])
                output.write_text("%%MatrixMarket matrix coordinate real symmetric\n1 1 1\n1 1 1\n")
                calls.append(output)
                if len(calls) == 1:
                    return SimpleNamespace(returncode=2)
                if len(calls) == 2:
                    raise subprocess.TimeoutExpired(command, 1)
                return SimpleNamespace(returncode=0)
            with mock.patch.object(rc, "sh", side_effect=run):
                def export():
                    return rc.dump_matrix("grid_500", directory,
                                          "/binary with spaces/benchmark", 1, 10)
                self.assertIsNone(export())
                self.assertEqual(list(Path(directory).iterdir()), [])
                with self.assertRaises(subprocess.TimeoutExpired):
                    export()
                self.assertEqual(list(Path(directory).iterdir()), [])
                path = export()
                self.assertTrue(Path(path).is_file())
                self.assertEqual(export(), path)
                self.assertEqual(len(calls), 3)
                self.assertEqual(list(Path(directory).iterdir()), [Path(path)])

    def test_missing_or_malformed_export_is_not_cached(self):
        with tempfile.TemporaryDirectory() as directory:
            def malformed(command, **kwargs):
                args = shlex.split(command)
                Path(args[args.index("--dump-mtx") + 1]).write_text("partial data")
                return SimpleNamespace(returncode=0)
            for run in (lambda *a, **k: SimpleNamespace(returncode=0), malformed):
                with mock.patch.object(rc, "sh", side_effect=run):
                    self.assertIsNone(rc.dump_matrix("grid_500", directory, "benchmark", 1, 10))
                self.assertEqual(list(Path(directory).iterdir()), [])




class OriginalStoppingReceiptTest(unittest.TestCase):
    def test_cpp_and_external_csv_trailers(self):
        core = "AMGCL,m,10,20,1,2,3,4,5e-9,0,1"
        for extras in ("", ",8,-1,1,1,5e-9"):
            row = rc.parse_csv(core + extras + ",original-v1,2,0.25\n")
            self.assertEqual(row["stop_contract"], "original-v1")
            self.assertEqual(row["solve_passes"], 2)
            self.assertEqual(row["stop_check_s"], 0.25)
            self.assertEqual(row["solve_s"], 2)

    def test_checks_cannot_be_outside_the_solve_interval(self):
        for cost in ("-1", "3", "nan"):
            self.assertIsNone(rc.parse_csv("AMGCL,m,10,20,1,2,3,4,5e-9,0,1,original-v1,1,"+cost+"\n"))

    def test_old_rows_remain_readable_without_becoming_new_measurements(self):
        row = rc.parse_csv("AMGCL,m,10,20,1,2,3,4,5e-9,0,1\n")
        self.assertNotIn("stop_contract",row)

    def test_roundtrip_precision_keeps_a_one_ulp_rejection(self):
        import math
        over = math.nextafter(1e-8, math.inf)
        row = rc.parse_csv(f"AMGCL,m,10,20,1,2,3,4,9e-9,0,1,0,-1,3,1,{over:.17e},original-v1,1,0.25\n")
        status, _ = rc.classify(row, 1e-8)
        self.assertEqual(status, "not_converged")

class CellReuseTest(unittest.TestCase):
    SHA = "a" * 40

    def cell(self, **overrides):
        record = {"cell": {"solver": "apxchol_v1", "device": "cpu", "matrix_id": "grid"},
                  "status": "complete",
                  "metrics": {"stop_contract": "original-v1", "total_s": 1.0,
                              "execution_route": "cpu"},
                  "provenance": {"git_sha": self.SHA}}
        record.update(overrides)
        return record

    def test_current_cell_is_reused(self):
        self.assertTrue(rc.cell_is_current(self.cell(), self.SHA))

    def test_cmg_cells_follow_the_competitor_contract(self):
        cell = self.cell(cell={"solver": "cmg", "device": "cpu", "matrix_id": "grid"})
        self.assertTrue(rc.cell_is_current(cell, self.SHA))
        cell["metrics"].pop("stop_contract")
        self.assertFalse(rc.cell_meets_contract(cell))

    def test_other_revision_is_rerun(self):
        self.assertFalse(rc.cell_is_current(self.cell(), "b" * 40))
        self.assertFalse(rc.cell_is_current(self.cell(), ""))
        # Meeting the contract alone does not match the selected source revision.
        self.assertTrue(rc.cell_meets_contract(self.cell()))

    def test_old_stop_contract_is_rejected(self):
        old = self.cell(metrics={"total_s": 1.0, "execution_route": "cpu"})
        self.assertFalse(rc.cell_meets_contract(old))
        self.assertFalse(rc.cell_is_current(old, self.SHA))

    def test_route_mismatch_is_rejected(self):
        mismatch = self.cell(metrics={"stop_contract": "original-v1", "total_s": 1.0,
                                      "execution_route": "gpu"})
        self.assertFalse(rc.cell_meets_contract(mismatch))

    def test_metal_cells_need_a_metal_route(self):
        identity = {"solver": "apxchol_v1", "device": "metal", "matrix_id": "grid"}
        for route, ok in (("metal", True), ("gpu", False), ("cpu", False)):
            cell = self.cell(cell=identity, metrics={"stop_contract": "original-v1",
                                                     "total_s": 1.0, "execution_route": route})
            self.assertEqual(rc.cell_meets_contract(cell), ok)

    def test_missing_route_cannot_hide_behind_missing_total(self):
        cell = self.cell(metrics={"stop_contract": "original-v1", "setup_s": 1})
        self.assertFalse(rc.cell_meets_contract(cell))

    def test_source_provenance_is_required_for_historical_rendering(self):
        for provenance in (None, {}, {"git_sha": ""}, {"git_sha": " "},
                           {"git_sha": True}, {"git_sha": []}):
            with self.subTest(provenance=provenance):
                self.assertFalse(rc.cell_meets_contract(self.cell(provenance=provenance)))

    def test_standalone_runners_share_revision_and_timeout_checks(self):
        with tempfile.TemporaryDirectory() as store, \
             mock.patch.object(rc, "CELLS", store), \
             mock.patch.object(rc, "git_sha", return_value=self.SHA):
            for solver in ("parac", "parac_physics"):
                with self.subTest(solver=solver):
                    key = ("audit", "m", solver, "", 4, "cpu")
                    path = rc.emit_cell(
                        *key[:4], "complete", {"stop_contract": "original-v1", "total_s": 1},
                        *key[4:], {"git_sha": self.SHA})
                    cell = json.loads(Path(path).read_text())
                    self.assertTrue(rc.cell_done(*key))
                    cell["provenance"]["git_sha"] = "b" * 40
                    Path(path).write_text(json.dumps(cell))
                    self.assertFalse(rc.cell_done(*key))
                    cell["provenance"]["git_sha"] = self.SHA
                    cell.update(status="timeout", metrics={})
                    Path(path).write_text(json.dumps(cell))
                    self.assertFalse(rc.cell_done(*key))
                    cell["timeout_cap_s"] = 60
                    Path(path).write_text(json.dumps(cell))
                    self.assertTrue(rc.cell_done(*key))
                    self.assertFalse(rc.cell_done(*key, rerun_timeouts=True))

    def test_timeout_needs_a_recorded_cap(self):
        self.assertFalse(rc.cell_meets_contract(self.cell(status="timeout", metrics={})))
        self.assertTrue(rc.cell_meets_contract(
            self.cell(status="timeout", metrics={}, timeout_cap_s=600)))

    def test_malformed_records_are_rejected(self):
        for record in (None, [], {"status": "complete"}, {"cell": "x", "status": "complete"}):
            self.assertFalse(rc.cell_meets_contract(record))
            self.assertFalse(rc.cell_is_current(record, self.SHA))
        for field in ("provenance", "metrics"):
            for value in (None, [], ["x"], "x", 1, True):
                with self.subTest(field=field, value=value):
                    self.assertFalse(rc.cell_is_current(self.cell(**{field: value}), self.SHA))

    def test_malformed_status_is_rejected(self):
        for status in (None, [], {}, 0, True):
            with self.subTest(status=status):
                cell = self.cell(status=status)
                self.assertFalse(rc.cell_meets_contract(cell))
                self.assertFalse(rc.cell_is_current(cell, self.SHA))


if __name__ == "__main__":
    unittest.main()
