#!/usr/bin/env python3
import json
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

BENCH = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCH))

import fair_charts as cpu  # noqa: E402
import gpu_charts as gpu  # noqa: E402
import runner_common as rc  # noqa: E402
import stale_cells  # noqa: E402
import sweep_fair  # noqa: E402


def record(status, total, *, threads=16, cap=None, solver="amgcl", config=""):
    out = {
        "cell": {"family": "audit", "matrix_id": "m", "solver": solver,
                 "config": config, "threads": threads, "device": "cpu"},
        "metrics": {} if total is None else {"total_s": total},
        "status": status,
    }
    if cap is not None:
        out["timeout_cap_s"] = cap
    return out


class SeriesRuleTest(unittest.TestCase):
    def test_label_maps_are_injective(self):
        rc.require_injective_labels(cpu.LABELS, "CPU")
        rc.require_injective_labels(gpu.LABELS, "GPU")
        with self.assertRaises(RuntimeError):
            rc.require_injective_labels({("a", "x"): "same", ("a", "y"): "same"},
                                        "broken")

    def test_thread_choice_is_declared_not_status_ranked(self):
        complete_t1 = record("complete", 1.0, threads=1)
        timeout_t16 = record("timeout", None, threads=16, cap=60)
        self.assertIs(cpu._pick([complete_t1, timeout_t16], "audit", "m", "AMGCL"),
                      timeout_t16)

    def test_duplicate_same_series_is_rejected_not_ranked(self):
        with self.assertRaises(RuntimeError):
            cpu._pick([record("complete", 1.0), record("timeout", None, cap=60)],
                      "audit", "m", "AMGCL")

    def test_legacy_gpu_csv_is_rejected(self):
        with tempfile.NamedTemporaryFile(suffix=".csv") as handle:
            with self.assertRaises(ValueError):
                gpu.load(handle.name)


class CellSchemaTest(unittest.TestCase):
    def test_timeout_requires_and_persists_exact_cap(self):
        with tempfile.TemporaryDirectory() as store, mock.patch.object(rc, "CELLS", store):
            with self.assertRaises(ValueError):
                rc.emit_cell("audit", "m", "solver", "", "timeout", {}, 16, "cpu", {})
            path = rc.emit_cell("audit", "m", "solver", "", "timeout", {}, 16,
                                "cpu", {}, timeout_cap_s=73.5)
            with open(path) as handle:
                cell = json.load(handle)
            self.assertEqual(cell["schema"], 2)
            self.assertEqual(cell["timeout_cap_s"], 73.5)
            self.assertEqual(rc.timeout_cap(cell), 73.5)

    def test_non_timeout_does_not_accept_cap(self):
        with tempfile.TemporaryDirectory() as store, mock.patch.object(rc, "CELLS", store):
            with self.assertRaises(ValueError):
                rc.emit_cell("audit", "m", "solver", "", "complete", {}, 16, "cpu",
                             {}, timeout_cap_s=60)

    def test_old_timeout_is_stale_and_not_terminal_for_sweep(self):
        old = record("timeout", None)
        self.assertTrue(stale_cells.timeout_cap_is_stale(old))
        with tempfile.TemporaryDirectory() as store, mock.patch.object(rc, "CELLS", store):
            path = pathlib.Path(rc.cell_path("audit", "m", "amgcl", "", 16, "cpu"))
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(old))
            with mock.patch.object(sweep_fair, "DEVICE", "cpu"):
                self.assertFalse(sweep_fair.cell_done("audit", "m", "amgcl", ""))

    def test_new_timeout_is_terminal_for_sweep(self):
        new = record("timeout", None, cap=60)
        with tempfile.TemporaryDirectory() as store, mock.patch.object(rc, "CELLS", store):
            path = pathlib.Path(rc.cell_path("audit", "m", "amgcl", "", 16, "cpu"))
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(new))
            with mock.patch.object(sweep_fair, "DEVICE", "cpu"):
                self.assertTrue(sweep_fair.cell_done("audit", "m", "amgcl", ""))


class CapReferenceTest(unittest.TestCase):
    def test_gpu_cap_reference_uses_declared_default_not_fastest_selector(self):
        with tempfile.TemporaryDirectory() as store, mock.patch.object(rc, "CELLS", store):
            base = dict(family="audit", mid="m", solver="apxchol_v1", status="complete",
                        threads=16, device="gpu", prov={})
            rc.emit_cell(config=sweep_fair.APX_DEFAULT_CONFIG,
                         metrics={"total_s": 10.0}, **base)
            rc.emit_cell(config="luby+tree[vec_pool]", metrics={"total_s": 1.0}, **base)
            self.assertEqual(sweep_fair.gpu_apx_total("audit", "m"), 10.0)


if __name__ == "__main__":
    unittest.main()
