"""Check the compact benchmark data that the Performance pages are rendered from."""
import csv
import hashlib
import json
from pathlib import Path
import unittest

DATA = Path(__file__).resolve().parents[2] / "docs/performance"
COLUMNS = ["matrix", "class", "n", "nnz", "solver", "config", "variant", "device", "threads", "status",
           "setup_s", "adapt_s", "solve_s", "iters", "rel_res", "cap_s", "max_rss_mb"]
STATUSES = {"complete", "exceeded_cap", "failed", "oom", "timeout", "not_converged", "n/a", "missing"}
CLASSES = {"grids", "meshes", "ipm", "social"}
DEVICES = {"cpu", "gpu", "metal"}
HEADER = {"campaign", "date", "hardware", "toolchains", "notes", "tolerance", "source_commits", "cap_rule",
          "files", "columns", "full_extract"}

SCALING_HEADER = {"kind", "campaign", "date", "files", "tolerance", "profile", "host", "flags",
                  "cost_contract", "columns", "notes", "provenance"}


def headers():
    return [p for p in sorted(DATA.glob("*-2*.json")) if p.name.split("-")[0] in ("euler", "leomed", "daint", "mac")]


class PerformanceDataTest(unittest.TestCase):
    def test_every_campaign_has_a_header_and_its_files(self):
        self.assertTrue(headers())
        for path in headers():
            header = json.loads(path.read_text())
            scaling = header.get("kind") in ("scaling", "records")
            self.assertEqual(set(header), SCALING_HEADER if scaling else HEADER, path.name)
            self.assertEqual(header["campaign"], path.stem)
            self.assertIn(f"{path.stem}.csv", header["files"])
            for name in header["files"]:
                self.assertTrue((DATA / name).is_file(), name)
            if scaling:
                provenance = header["provenance"]
                required = ("source_archive_sha256", "dependency_manifest_sha256", "source_annotations_sha256", "source_cell_audit_sha256", "csv_sha256") if provenance["job"] in (21174947, 21174948) else ("csv_sha256",)
                self.assertTrue(any(key in provenance for key in ("source_annotations_sha256", "source_receipts_sha256")))
                for key in required:
                    self.assertRegex(provenance[key], r"^[0-9a-f]{64}$")
                content = (DATA / header["files"][0]).read_bytes()
                self.assertEqual(hashlib.sha256(content).hexdigest(), provenance["csv_sha256"])
                self.assertEqual(len(list(csv.DictReader(content.decode().splitlines()))), provenance["rows"])
            else:
                self.assertTrue(header["full_extract"]["repository"].startswith("https://github.com/"))
                self.assertRegex(header["full_extract"]["sha256"], r"^[0-9a-f]{64}$")

    def test_every_csv_belongs_to_a_campaign(self):
        listed = {name for path in headers() for name in json.loads(path.read_text())["files"]}
        self.assertEqual({p.name for p in DATA.glob("*.csv")} - listed, set())

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
                        for dimension in ("n", "nnz"):
                            if row[dimension] or row["status"] == "complete" or header.get("kind") != "records":
                                self.assertGreater(int(row[dimension]), 0, where)
                        key = (row["matrix"], row["solver"], row["config"], row["variant"], row["device"],
                               row["threads"])
                        self.assertNotIn(key, seen, where)
                        seen.add(key)
                        if row["status"] == "complete":
                            for column in ("setup_s", "solve_s", "rel_res"):
                                self.assertGreaterEqual(float(row[column]), 0, where)
                            self.assertLessEqual(float(row["rel_res"]), float(header["tolerance"]), where)
                            self.assertGreaterEqual(int(row["iters"]), 0, where)
                        if row["adapt_s"]:
                            self.assertGreaterEqual(float(row["adapt_s"]), 0, where)
                            self.assertLessEqual(float(row["adapt_s"]), float(row["setup_s"]), where)
                        if row["status"] == "exceeded_cap":
                            self.assertGreater(float(row["cap_s"]), 0, where)


if __name__ == "__main__":
    unittest.main()
