#!/usr/bin/env python3
import csv
import json
import math
import pathlib
import sys


EXPECTED = (
    "iter0010", "iter0020", "iter0030", "iter0040",
    "grid_500", "G3_circuit", "thermal2", "com-Amazon",
)


def number(row, key):
    return float(row[key])


def geometric_mean(values):
    return math.exp(sum(math.log(value) for value in values) / len(values))


def main():
    campaign = pathlib.Path(sys.argv[1])
    seeds = tuple((campaign / "SEEDS").read_text().split())
    if len(seeds) != 5:
        raise RuntimeError(f"expected five seeds, found {len(seeds)}")
    expected_records = len(EXPECTED) * len(seeds) * 3
    pair_rows = []
    records = 0
    converged = 0
    exact_baselines = 0

    for label in EXPECTED:
        with (campaign / "results" / label / "run.tsv").open(newline="") as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))
        if len(rows) != 3 * len(seeds):
            raise RuntimeError(f"{label}: wrong record count {len(rows)}")
        for row in rows:
            records += 1
            converged += number(row, "true_residual") <= 1.0e-8
        for seed in seeds:
            selected = [row for row in rows if row["seed"] == seed]
            arms = {row["arm"]: row for row in selected}
            if set(arms) != {"gks_before", "bkz26", "gks_after"}:
                raise RuntimeError(f"{label} seed {seed}: wrong arms")
            before = arms["gks_before"]
            after = arms["gks_after"]
            signature_keys = ("raw_nnz", "stored_nnz", "iters", "true_residual")
            if any(before[key] != after[key] for key in signature_keys):
                raise RuntimeError(f"{label} seed {seed}: GKS bracket mismatch")
            exact_baselines += 1
            candidate = arms["bkz26"]
            setup_base = math.sqrt(number(before, "setup_s") * number(after, "setup_s"))
            solve_base = math.sqrt(number(before, "solve_s") * number(after, "solve_s"))
            total_before = number(before, "setup_s") + number(before, "solve_s")
            total_after = number(after, "setup_s") + number(after, "solve_s")
            total_base = math.sqrt(total_before * total_after)
            pair_rows.append({
                "matrix": label,
                "seed": seed,
                "iters_gks": before["iters"],
                "iters_bkz26": candidate["iters"],
                "iteration_delta": int(candidate["iters"]) - int(before["iters"]),
                "raw_nnz_ratio": number(candidate, "raw_nnz") / number(before, "raw_nnz"),
                "stored_nnz_ratio": number(candidate, "stored_nnz") / number(before, "stored_nnz"),
                "setup_ratio": number(candidate, "setup_s") / setup_base,
                "solve_ratio": number(candidate, "solve_s") / solve_base,
                "total_ratio": (number(candidate, "setup_s") + number(candidate, "solve_s")) / total_base,
                "true_residual": candidate["true_residual"],
            })

    with (campaign / "pairs.tsv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(pair_rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(pair_rows)

    aggregates = []
    scopes = [(matrix, [row for row in pair_rows if row["matrix"] == matrix])
              for matrix in EXPECTED]
    scopes += [
        ("IPM", [row for row in pair_rows if row["matrix"].startswith("iter")]),
        ("controls", [row for row in pair_rows if not row["matrix"].startswith("iter")]),
        ("all", pair_rows),
    ]
    for scope, selected in scopes:
        deltas = [int(row["iteration_delta"]) for row in selected]
        aggregates.append({
            "scope": scope,
            "pairs": len(selected),
            "mean_iters_gks": sum(int(row["iters_gks"]) for row in selected) / len(selected),
            "mean_iters_bkz26": sum(int(row["iters_bkz26"]) for row in selected) / len(selected),
            "iteration_delta_sum": sum(deltas),
            "iteration_delta_min": min(deltas),
            "iteration_delta_max": max(deltas),
            "raw_nnz_geomean": geometric_mean([float(row["raw_nnz_ratio"]) for row in selected]),
            "stored_nnz_geomean": geometric_mean([float(row["stored_nnz_ratio"]) for row in selected]),
            "setup_geomean": geometric_mean([float(row["setup_ratio"]) for row in selected]),
            "solve_geomean": geometric_mean([float(row["solve_ratio"]) for row in selected]),
            "total_geomean": geometric_mean([float(row["total_ratio"]) for row in selected]),
        })
    with (campaign / "aggregate.tsv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(aggregates[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(aggregates)
    (campaign / "aggregate.json").write_text(json.dumps(aggregates, indent=2) + "\n")

    summary = (
        f"checked matrices {len(EXPECTED)}/{len(EXPECTED)}; seeds {len(seeds)}/{len(seeds)}; "
        f"records {records}/{expected_records}; exact GKS brackets "
        f"{exact_baselines}/{len(EXPECTED) * len(seeds)}; "
        f"converged {converged}/{expected_records}\n"
    )
    (campaign / "summary.txt").write_text(summary)
    print(summary, end="")


if __name__ == "__main__":
    main()
