import csv
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import campaign_drift as cd  # noqa: E402


def cur(matrix, status="complete", iters=10, total=2.0, solver="amgcl"):
    return {"matrix": matrix, "solver": solver, "config": "", "device": "cpu", "status": status,
            "iters": iters, "setup_s": total / 2, "solve_s": total / 2, "total_s": total}


def ref(matrix, status="complete", iters=10, total=1.0, solver="amgcl"):
    return {"matrix": matrix, "solver": solver, "config": "", "device": "cpu", "status": status,
            "iters": str(iters), "setup_s": str(total / 2), "solve_s": str(total / 2),
            "total_s": str(total)}


SERIES = [{"campaign": {"store": "prod", "solver": "amgcl", "config": "", "device": "cpu"},
           "reference": {"solver": "amgcl", "config": "", "device": "cpu"}, "iteration_bound": 1.25}]


class CampaignDriftTest(unittest.TestCase):
    def test_flags_status_iterations_and_time_outliers(self):
        campaign = {"prod": [cur("a"), cur("b"), cur("c", iters=20), cur("d", total=20.0),
                             cur("e", status="oom"), cur("f", status="failed")]}
        reference = [ref(m) for m in "abcde"] + [ref("f", status="failed"), ref("g")]
        rows, summary = cd.compare(campaign, reference, SERIES)
        flags = {r["matrix"]: r["flags"] for r in rows}
        self.assertEqual(flags["a"], [])
        self.assertEqual(flags["c"], ["iterations"])
        self.assertIn("total_time", flags["d"])            # 10x the series' 2x median ratio
        self.assertEqual(flags["e"], ["status_change"])
        self.assertEqual(flags["f"], [])                   # both incomplete: recorded only
        self.assertEqual(summary["reference_cells_in_series"], 7)
        self.assertEqual(summary["checked"], 6)
        self.assertEqual(summary["not_checked"], [{"series": "prod:amgcl/-/cpu", "matrix": "g",
                                                   "reference_status": "complete"}])

    def test_stores_keep_same_solver_series_apart(self):
        series = SERIES + [{"campaign": {"store": "shared", "solver": "amgcl", "config": "",
                                         "device": "cpu"},
                            "reference": {"solver": "amgcl", "config": "", "device": "cpu"}}]
        rows, _ = cd.compare({"prod": [cur("a")], "shared": [cur("a", iters=40)]}, [ref("a")], series)
        self.assertEqual({r["series"]: r["flags"] for r in rows},
                         {"prod:amgcl/-/cpu": [], "shared:amgcl/-/cpu": ["iterations"]})

    def test_rejects_duplicate_reference_cells(self):
        with self.assertRaises(ValueError):
            cd.compare({"prod": []}, [ref("a"), ref("a")], SERIES)

    def test_command_line_writes_rows_and_summary(self):
        with tempfile.TemporaryDirectory() as d:
            paths = {k: os.path.join(d, k) for k in ("e.json", "r.csv", "s.json", "o.csv", "sum.json")}
            with open(paths["e.json"], "w") as f:
                json.dump({"cells": [cur("a")]}, f)
            with open(paths["r.csv"], "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(ref("a")))
                w.writeheader()
                w.writerow(ref("a"))
            with open(paths["s.json"], "w") as f:
                json.dump(SERIES, f)
            cd.main(["--campaign", f"prod={paths['e.json']}", "--reference", paths["r.csv"],
                     "--series", paths["s.json"], "-o", paths["o.csv"], "--summary", paths["sum.json"]])
            with open(paths["o.csv"]) as f:
                out = list(csv.DictReader(f))
            with open(paths["sum.json"]) as f:
                summary = json.load(f)
        self.assertEqual(len(out), 1)
        self.assertAlmostEqual(float(out[0]["total_s_ratio"]), 2.0)
        self.assertEqual(summary["checked"], 1)


if __name__ == "__main__":
    unittest.main()
