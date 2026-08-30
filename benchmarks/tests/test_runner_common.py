#!/usr/bin/env python3
import os
from types import SimpleNamespace
import unittest
from unittest import mock

from benchmarks import runner_common as rc


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


if __name__ == "__main__":
    unittest.main()
