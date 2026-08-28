#!/usr/bin/env python3
import os
import unittest
from unittest import mock

from benchmarks import runner_common as rc


class AffinitySpecTest(unittest.TestCase):
    def test_uses_cpus_granted_to_packed_rank(self):
        with mock.patch.object(os, "sched_getaffinity",
                               return_value=set(range(72, 144))), \
             mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("APXCHOL_BENCH_CPUSET", None)
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


if __name__ == "__main__":
    unittest.main()
