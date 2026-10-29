import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import campaign_extract as ce  # noqa: E402

SHA = "1c4abb35f1cf08276bc9dc65534dbfb6d2b66059"
ENV = ("role=gpu job=1\n0, NVIDIA GeForce RTX 4090, 24564 MiB, 580.178.04\n"
       "1, NVIDIA GeForce RTX 4090, 24564 MiB, 580.178.04\n"
       "Model name:                              AMD EPYC 9554 64-Core Processor\n"
       "Socket(s):                               2\n")


def write(path, text, mode="w"):
    with open(path, mode) as f:
        f.write(text)


def cell(root, matrix, solver, device="cpu", status="complete", rel=5e-9, git_sha="", stderr="", cap=None):
    os.makedirs(os.path.join(root, "fam"), exist_ok=True)
    record = {"schema": 2, "status": status,
              "cell": {"family": "fam", "matrix_id": matrix, "solver": solver, "config": "",
                       "device": device, "threads": 64},
              "matrix_meta": {"kind": "graph", "n": 4, "nnz": 10},
              "metrics": {"setup_s": 1.0, "solve_s": 0.5, "total_s": 1.5, "iters": 7, "rel_res": rel,
                          "stderr_tail": stderr},
              **({"timeout_cap_s": cap} if cap else {}),
              "provenance": {"git_sha": git_sha, "compiler": "clang", "compiler_version": "19.1.3"}}
    with open(os.path.join(root, "fam", f"{matrix}__{solver}__{device}.json"), "w") as f:
        json.dump(record, f)


class Extract(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def test_denominator_lists_missing_and_grades_tolerance(self):
        cell(self.store, "a", "x")
        cell(self.store, "a", "y", rel=2e-8)
        cell(self.store, "b", "x", status="failed", rel=None,
             stderr="BUILD_META x\nradix_sort: failed on 1st step\nCommand exited with non-zero status 2\n")
        cell(self.store, "b", "y", status="timeout", rel=None, cap=1800.0)
        r = ce.extract(self.store, ["a", "b", "c"], SHA)
        self.assertEqual((r["planned_cells"], r["present_cells"], r["meets_tolerance"]), (6, 4, 1))
        self.assertEqual(sorted((m["matrix"], m["solver"]) for m in r["missing"]), [("c", "x"), ("c", "y")])
        self.assertEqual(r["status_counts"], {"complete": 2, "failed": 1, "timeout": 1})
        failed, timeout = (next(c for c in r["cells"] if c["status"] == s) for s in ("failed", "timeout"))
        self.assertEqual(failed["reason"], "radix_sort: failed on 1st step")
        self.assertEqual(timeout["process_cap_s"], 1800.0)

    def test_rejects_foreign_source_unlisted_matrix_and_short_commit(self):
        cell(self.store, "a", "x", git_sha="0" * 40)
        with self.assertRaisesRegex(ValueError, "built from"):
            ce.extract(self.store, ["a"], SHA)
        with self.assertRaisesRegex(ValueError, "40-hex"):
            ce.extract(self.store, ["a"], SHA[:12])
        os.remove(os.path.join(self.store, "fam", "a__x__cpu.json"))
        cell(self.store, "z", "x")
        with self.assertRaisesRegex(ValueError, "unlisted"):
            ce.extract(self.store, ["a"], SHA)

    def test_hardware_must_match_across_roles(self):
        a, b = (os.path.join(self.store, n) for n in ("env-a.txt", "env-b.txt"))
        write(a, ENV)
        self.assertEqual(ce.read_env([a])["gpus"][0]["count"], 2)
        write(b, ENV.replace("9554", "9654"))
        with self.assertRaisesRegex(ValueError, "different hardware"):
            ce.read_env([a, b])

    def test_hash_names_are_relative_and_unique(self):
        p = os.path.join(self.store, "bin.sha256")
        write(p, f"{'a' * 64}  /r/build-cpu/benchmark\n{'b' * 64}  /r/build-cuda/benchmark\n")
        self.assertEqual(ce.read_hashes(p), {"build-cpu/benchmark": "a" * 64, "build-cuda/benchmark": "b" * 64})
        write(p, f"{'c' * 64}  /r/build-cpu/benchmark\n", "a")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            ce.read_hashes(p)


if __name__ == "__main__":
    unittest.main()
