#!/usr/bin/env python3
"""Compare a campaign's extracted cells with a reference campaign's results.csv.

  campaign_drift.py --campaign LABEL=extract.json [...] --reference results.csv \
      --series series.json -o drift.csv [--summary drift.json]

Each extract comes from campaign_extract.py; LABEL names its store, so series
that share solver names (e.g. two ParAC RHS variants) stay apart. series.json
lists the pairs to compare:

  [{"campaign": {"store": "prod", "solver": "amgcl", "config": "", "device": "cpu"},
    "reference": {"solver": "amgcl", "config": "", "device": "cpu"},
    "iteration_bound": 1.25}, ...]

Per matrix, a pair is flagged when exactly one side completed, when the
iteration ratio leaves [1/bound, bound], or when a setup/solve/total ratio is
more than TIME_FACTOR away from that series' median ratio (the per-series
median absorbs the hardware difference). Both sides incomplete is recorded,
not flagged. The summary states the reference denominator and lists every
reference cell without a campaign counterpart.
"""
import argparse
import csv
import json
import math
import statistics

TIME_FACTOR = 2.0
TIMES = ("setup_s", "solve_s", "total_s")


def _number(value):
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) and v > 0 else None


def compare(campaign_cells, reference_rows, series):
    """Return (rows, summary). campaign_cells maps store label -> extract cells."""
    by_campaign = {}
    for store, cells in campaign_cells.items():
        for c in cells:
            by_campaign[(store, c["solver"], c["config"], c["device"], c["matrix"])] = c
    by_reference = {}
    for r in reference_rows:
        key = (r["solver"], r["config"], r["device"], r["matrix"])
        if key in by_reference:
            raise ValueError(f"duplicate reference cell {key}")
        by_reference[key] = r
    rows, unpaired = [], []
    for pair in series:
        e, d = pair["campaign"], pair["reference"]
        label = f"{e['store']}:{e['solver']}/{e['config'] or '-'}/{e['device']}"
        bound = float(pair.get("iteration_bound", 1.25))
        matrices = sorted({k[3] for k in by_reference if k[:3] == (d["solver"], d["config"], d["device"])})
        pair_rows = []
        for mat in matrices:
            ref = by_reference[(d["solver"], d["config"], d["device"], mat)]
            cur = by_campaign.get((e["store"], e["solver"], e["config"], e["device"], mat))
            if cur is None:
                unpaired.append({"series": label, "matrix": mat, "reference_status": ref["status"]})
                continue
            row = {"series": label, "matrix": mat, "campaign_status": cur["status"],
                   "reference_status": ref["status"], "flags": []}
            done = (cur["status"] == "complete", ref["status"] == "complete")
            if done[0] != done[1]:
                row["flags"].append("status_change")
            if all(done):
                it_c, it_r = _number(cur.get("iters")), _number(ref.get("iters"))
                if it_c and it_r:
                    row["iteration_ratio"] = it_c / it_r
                    if not 1 / bound <= row["iteration_ratio"] <= bound:
                        row["flags"].append("iterations")
                for t in TIMES:
                    a, b = _number(cur.get(t)), _number(ref.get(t))
                    if a and b:
                        row[f"{t}_ratio"] = a / b
            pair_rows.append(row)
        for t in TIMES:
            ratios = [r[f"{t}_ratio"] for r in pair_rows if f"{t}_ratio" in r]
            if not ratios:
                continue
            median = statistics.median(ratios)
            for r in pair_rows:
                if f"{t}_ratio" in r:
                    r[f"{t}_vs_series_median"] = r[f"{t}_ratio"] / median
                    if abs(math.log(r[f"{t}_vs_series_median"])) > math.log(TIME_FACTOR):
                        r["flags"].append(t.replace("_s", "_time"))
        rows.extend(pair_rows)
    denominator = sum(1 for k in by_reference
                      if any(k[:3] == (p["reference"]["solver"], p["reference"]["config"],
                                       p["reference"]["device"]) for p in series))
    summary = {"reference_cells_in_series": denominator, "checked": len(rows),
               "flagged": sum(1 for r in rows if r["flags"]),
               "flag_counts": {f: sum(f in r["flags"] for r in rows)
                               for f in sorted({f for r in rows for f in r["flags"]})},
               "time_factor": TIME_FACTOR, "not_checked": unpaired}
    return rows, summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--campaign", action="append", required=True, metavar="LABEL=extract.json")
    ap.add_argument("--reference", required=True, help="reference results.csv")
    ap.add_argument("--series", required=True, help="series pairing JSON")
    ap.add_argument("-o", "--output", required=True, help="per-pair CSV")
    ap.add_argument("--summary", help="summary JSON")
    a = ap.parse_args(argv)
    cells = {}
    for item in a.campaign:
        label, _, path = item.partition("=")
        if not label or not path or label in cells:
            ap.error(f"bad or repeated --campaign {item!r}")
        with open(path) as f:
            cells[label] = json.load(f)["cells"]
    with open(a.reference, newline="") as f:
        reference = list(csv.DictReader(f))
    with open(a.series) as f:
        series = json.load(f)
    rows, summary = compare(cells, reference, series)
    fields = ["series", "matrix", "campaign_status", "reference_status", "flags", "iteration_ratio",
              *[f"{t}_{s}" for t in TIMES for s in ("ratio", "vs_series_median")]]
    with open(a.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({**r, "flags": ";".join(r["flags"])})
    if a.summary:
        with open(a.summary, "w") as f:
            json.dump(summary, f, indent=1)
    print(f"checked {summary['checked']}/{summary['reference_cells_in_series']} reference cells, "
          f"{summary['flagged']} flagged, {len(summary['not_checked'])} without a campaign cell")


if __name__ == "__main__":
    main()
