"""Check the compact benchmark data that the Performance pages are rendered from."""
import csv
import json
from pathlib import Path
import unittest

DATA = Path(__file__).resolve().parents[2] / "docs/performance"
COLUMNS = ["matrix", "class", "n", "nnz", "solver", "config", "variant", "device", "threads", "status",
           "setup_s", "solve_s", "iters", "rel_res", "cap_s", "max_rss_mb"]
STATUSES = {"complete", "exceeded_cap", "failed", "oom", "timeout", "not_converged", "n/a", "missing"}
CLASSES = {"grids", "meshes", "ipm", "social"}
DEVICES = {"cpu", "gpu", "metal"}
HEADER = {"campaign", "date", "hardware", "toolchains", "notes", "tolerance", "source_commits", "cap_rule",
          "files", "columns", "full_extract"}


def headers():
    return [p for p in sorted(DATA.glob("*-2*.json")) if p.name.split("-")[0] in ("euler", "daint", "mac")]


class PerformanceDataTest(unittest.TestCase):
    def test_every_campaign_has_a_header_and_its_files(self):
        self.assertTrue(headers())
        for path in headers():
            header = json.loads(path.read_text())
            self.assertEqual(set(header), HEADER, path.name)
            self.assertEqual(header["campaign"], path.stem)
            self.assertIn(f"{path.stem}.csv", header["files"])
            for name in header["files"]:
                self.assertTrue((DATA / name).is_file(), name)
            self.assertTrue(header["full_extract"]["repository"].startswith("https://github.com/"))
            self.assertRegex(header["full_extract"]["sha256"], r"^[0-9a-f]{64}$")

    def test_rows(self):
        for path in headers():
            header = json.loads(path.read_text())
            for name in header["files"]:
                if not name.endswith(".csv"):
                    continue
                with open(DATA / name, newline="") as handle:
                    reader = csv.DictReader(handle)
                    self.assertEqual(reader.fieldnames, COLUMNS, name)
                    seen = set()
                    for row in reader:
                        where = f"{name}: {row['matrix']} {row['solver']} {row['config']} {row['variant']}"
                        self.assertIn(row["status"], STATUSES, where)
                        self.assertIn(row["class"], CLASSES, where)
                        self.assertIn(row["device"], DEVICES, where)
                        self.assertGreater(int(row["n"]), 0, where)
                        self.assertGreater(int(row["nnz"]), 0, where)
                        key = (row["matrix"], row["solver"], row["config"], row["variant"], row["device"],
                               row["threads"])
                        self.assertNotIn(key, seen, where)
                        seen.add(key)
                        if row["status"] == "complete":
                            for column in ("setup_s", "solve_s", "rel_res"):
                                self.assertGreaterEqual(float(row[column]), 0, where)
                            int(row["iters"])
                        if row["status"] == "exceeded_cap":
                            self.assertGreater(float(row["cap_s"]), 0, where)


if __name__ == "__main__":
    unittest.main()
