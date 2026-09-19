#!/usr/bin/env python3
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

if __name__ == "__main__":
    unittest.main()
