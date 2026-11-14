"""Render checks for docs/performance.py (needs matplotlib and NumPy, from docs/requirements.txt)."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

HAVE_PLOTTING = all(importlib.util.find_spec(m) for m in ("matplotlib", "numpy"))


def cell(solver, config, setup, solve, adapt=None, matrix="grid_500"):
    return {"matrix": matrix, "class": "grids", "n": 250000, "nnz": 1248000, "solver": solver, "config": config,
            "variant": "", "device": "cpu", "threads": 64, "status": "complete", "setup_s": setup,
            "adapt_s": adapt, "solve_s": solve, "iters": 10, "rel_res": 1e-9, "cap_s": None, "max_rss_mb": None,
            "meets_tolerance": True}


@unittest.skipUnless(HAVE_PLOTTING, "matplotlib and NumPy are installed with docs/requirements.txt")
class AdaptedTimesTest(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location("performance", Path(__file__).parents[1] / "performance.py")
        self.p = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.p)
        self.data = {"tolerance": 1e-8, "cells": [
            cell("apxchol_v1", "bg+tree[vec_pool_aos]", 1.0, 1.0, 0.0),
            cell("amgcl", "", 2.0, 2.0, 0.5),
            cell("rchol", "", 3.0, 3.0, 0.0)]}

    def test_adapted(self):
        self.assertTrue(self.p.adapted({"adapt_s": 0.5}))
        for value in (0.0, None):
            self.assertFalse(self.p.adapted({"adapt_s": value}))
        self.assertFalse(self.p.adapted(None))

    def test_times_table_stars_only_adapted_cells(self):
        table = self.p.times_table(self.p.Campaign(self.data), "cpu")
        row = next(line for line in table.splitlines() if line.startswith("| grid_500"))
        self.assertEqual(row.split(" | ")[1:], ["1", "1", "2\\*", "2", "3", "3 |"])
        self.assertTrue(table.rstrip().endswith(self.p.ADAPTED_NOTE))

    def test_no_note_without_adaptation(self):
        for c in self.data["cells"]:
            c["adapt_s"] = None
        table = self.p.times_table(self.p.Campaign(self.data), "cpu")
        self.assertNotIn("\\*", table)
        self.assertNotIn(self.p.ADAPTED_NOTE, table)

    def test_heatmaps_star_adapted_cells(self):
        with tempfile.TemporaryDirectory() as out:
            run = self.p.Campaign(self.data)
            self.p.family_grid(run, "test", "cpu", Path(out))
            self.p.variant_grid(run, "test", "cpu", Path(out))
            for name in ("test-cpu-families.svg", "test-cpu-variants.svg"):
                svg = (Path(out) / name).read_text()
                self.assertIn(">2*<", svg, name)
                self.assertNotIn(">3*<", svg, name)


if __name__ == "__main__":
    unittest.main()
