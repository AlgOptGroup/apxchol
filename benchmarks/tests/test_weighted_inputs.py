import copy
import json
from pathlib import Path
import shlex
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
from scipy import io, sparse

from benchmarks import weighted_inputs as wi

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import runner_common as rc
import sweep_fair
import cmg_matlab_runner
import parac_runner
from types import SimpleNamespace


class WeightedInputsTest(unittest.TestCase):
    def setUp(self):
        for module in (sweep_fair, cmg_matlab_runner, parac_runner):
            threads = mock.patch.object(module, "THREADS", 1)
            threads.start()
            self.addCleanup(threads.stop)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        # Two triangles plus an isolated vertex exercise the old root-0 gap.
        edges = [(0, 1), (0, 2), (1, 2), (3, 4), (3, 5), (4, 5)]
        u, v = np.array(edges).T
        a = sparse.coo_matrix((np.ones(len(edges)), (u, v)), shape=(7, 7))
        self.a = wi.support(a)
        self.source = self.root / "input with spaces.mtx"
        io.mmwrite(self.source, self.a, symmetry="symmetric")
        self.registry = dict(rc.MATRICES)
        self.addCleanup(self.restore_registry)

    def restore_registry(self):
        rc.MATRICES.clear()
        rc.MATRICES.update(self.registry)

    def generate(self, name="out", seed=42):
        out = self.root / name
        manifest = wi.generate(self.source, out, "fixture", seed)
        return out, manifest

    def test_forest_covers_every_component_and_preserves_isolated_vertex(self):
        parent, components = wi.spanning_forest(self.a)
        np.testing.assert_array_equal(parent, [-1, 0, 0, -1, 3, 3, -1])
        self.assertEqual(components, 3)

    def test_all_four_models_preserve_support_and_expected_ranges(self):
        out, manifest = self.generate()
        self.assertEqual(len(manifest["matrices"]), 4)
        parent, _ = wi.spanning_forest(self.a)
        for row in manifest["matrices"]:
            a = sparse.csr_matrix(io.mmread(out / row["path"]))
            np.testing.assert_array_equal(a.toarray() != 0, self.a.toarray() != 0)
            np.testing.assert_array_equal(a.toarray(), a.T.toarray())
            self.assertEqual(row["weight_model"]["forest_edges"], 4)
            self.assertEqual(row["weight_model"]["components"], 3)
            self.assertEqual(row["kind"], "graph")
            model = row["weight_model"]["name"]
            if model == "unit":
                np.testing.assert_array_equal(a.data, 1)
            else:
                lo, hi = (.1, 10) if model == "log2" else (.001, 1000)
                self.assertTrue(np.all((a.data >= lo) & (a.data <= hi)))
            if model == "backbone":
                for v, u in enumerate(parent):
                    if u >= 0:
                        self.assertTrue(.1 <= a[u, v] <= 10)

    def test_same_seed_same_bytes_and_new_seed_changes_random_models(self):
        _, a = self.generate("first")
        _, b = self.generate("second")
        _, c = self.generate("third", seed=43)
        self.assertEqual(a, b)
        self.assertEqual(a["matrices"][0]["sha256"], c["matrices"][0]["sha256"])
        for x, y in zip(a["matrices"][1:], c["matrices"][1:]):
            self.assertNotEqual(x["sha256"], y["sha256"])

    def test_refuses_existing_output_and_invalid_support(self):
        out, _ = self.generate()
        before = (out / "manifest.json").read_bytes()
        with self.assertRaises(FileExistsError):
            wi.generate(self.source, out, "fixture", 43)
        self.assertEqual(before, (out / "manifest.json").read_bytes())
        for a in [np.ones((2, 3)), np.array([[0, np.nan], [0, 0]]), np.eye(2)*1j]:
            with self.assertRaises(ValueError):
                wi.support(a)

    def test_manifest_loading_carries_identity_and_quotes_paths(self):
        out, manifest = self.generate("output with spaces")
        ids = rc.load_matrix_manifest(out / "manifest.json")
        self.assertEqual(len(ids), 4)
        for mid, row in zip(ids, manifest["matrices"]):
            meta = rc.matrix_meta_for(mid)
            self.assertEqual(meta["input_sha256"], row["sha256"])
            self.assertEqual(meta["weight_model"], row["weight_model"])
            args = shlex.split(rc.matrix_args("mtx", rc.MATRICES[mid]["spec"]))
            self.assertEqual(args, ["--mtx", str(out / row["path"]), "--kind", "graph"])
        self.assertEqual(len(list(sweep_fair.selected_matrices({"weighted"}))), 4)

    def test_tampered_input_does_not_partially_register(self):
        out, manifest = self.generate()
        with (out / manifest["matrices"][-1]["path"]).open("ab") as f:
            f.write(b"% altered\n")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            rc.load_matrix_manifest(out / "manifest.json")
        self.assertEqual(rc.MATRICES, self.registry)

    def test_duplicate_ids_and_inconsistent_declarations_are_rejected(self):
        out, manifest = self.generate()
        cases = []
        bad = copy.deepcopy(manifest);bad["matrices"][-1]["id"] = bad["matrices"][0]["id"];cases.append(bad)
        bad = copy.deepcopy(manifest);bad["matrices"][0]["id"] = "grid_500";cases.append(bad)
        bad = copy.deepcopy(manifest);bad["matrices"][0]["kind"] = "operator";cases.append(bad)
        bad = copy.deepcopy(manifest);bad["matrices"][0]["class"] = "sddm";cases.append(bad)
        bad = copy.deepcopy(manifest);bad["matrices"][0]["family"] = "../escape";cases.append(bad)
        bad = copy.deepcopy(manifest);bad["matrices"][0]["n"] = 8;cases.append(bad)
        for bad in cases:
            (out / "bad.json").write_text(json.dumps(bad))
            with self.assertRaises(ValueError):
                rc.load_matrix_manifest(out / "bad.json")
            self.assertEqual(rc.MATRICES, self.registry)

    def test_resume_rejects_different_manifest_under_same_cell_id(self):
        out, _ = self.generate()
        mid = rc.load_matrix_manifest(out / "manifest.json")[0]
        cell = {"schema": 2, "status": "complete", "cell": {"family": "weighted",
            "matrix_id": mid, "solver": "amgcl", "config": "", "threads": 1,
            "device": "cpu"}, "matrix_meta": rc.matrix_meta_for(mid)}
        with mock.patch.object(rc, "CELLS", str(self.root / "cells")):
            path = Path(rc.cell_path("weighted", mid, "amgcl", "", 1, "cpu"))
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(cell))
            self.assertIsNotNone(sweep_fair._stored_cell("weighted", mid, "amgcl", "", 1, "cpu"))
            cell["matrix_meta"]["input_manifest_sha256"] = "0" * 64
            path.write_text(json.dumps(cell))
            self.assertIsNone(sweep_fair._stored_cell("weighted", mid, "amgcl", "", 1, "cpu"))

    def test_external_dump_caches_follow_manifest_identity(self):
        out, _ = self.generate()
        mid = rc.load_matrix_manifest(out / "manifest.json")[0]

        def dump(command, **kwargs):
            args = shlex.split(command)
            path = Path(args[args.index("--dump-mtx") + 1])
            path.write_text("%%MatrixMarket matrix coordinate real symmetric\n1 1 1\n1 1 1\n")
            return SimpleNamespace(returncode=0, stdout="", stderr="APX dump setup time: 0.01")

        for module, field in [(sweep_fair, "DUMP"), (cmg_matlab_runner, "DUMP_DIR"),
                              (parac_runner, "DUMP_CPU")]:
            directory = self.root / module.__name__
            with mock.patch.object(module, field, str(directory)), \
                 mock.patch.object(parac_runner if module is parac_runner else rc,
                                   "sh", side_effect=dump) as run:
                def call():
                    if module is sweep_fair:
                        return module.dump_mtx(mid)
                    if module is cmg_matlab_runner:
                        return module._dump(mid)
                    return module._dump(mid, str(directory), "benchmark", 10)[0]
                rc.MATRICES[mid]["input_manifest_sha256"] = "1" * 64
                first = call()
                self.assertEqual(call(), first)
                rc.MATRICES[mid]["input_manifest_sha256"] = "2" * 64
                second = call()
                self.assertNotEqual(first, second)
                self.assertEqual(run.call_count, 2)
                self.assertTrue(Path(first).exists())
                self.assertTrue(Path(second).exists())

    def test_parac_rejects_failed_export_even_with_timing_and_partial_file(self):
        def fail(command, **kwargs):
            args = shlex.split(command)
            self.assertEqual(args[0], "/path with spaces/benchmark")
            Path(args[args.index("--dump-mtx") + 1]).write_text("partial")
            return SimpleNamespace(returncode=2, stdout="",
                                   stderr="APX dump setup time: 0.01")

        with mock.patch.object(parac_runner, "sh", side_effect=fail):
            result = parac_runner._dump("grid_500", str(self.root / "dump with spaces"),
                                       "/path with spaces/benchmark", 10)
        self.assertIsNone(result[0])
        self.assertEqual(list((self.root / "dump with spaces").iterdir()), [])


if __name__ == "__main__":
    unittest.main()
