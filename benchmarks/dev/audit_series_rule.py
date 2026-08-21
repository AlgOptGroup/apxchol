#!/usr/bin/env python3
"""Read-only audit of chart series identity and status handling.

Usage:
  PYTHONPATH=benchmarks python3 benchmarks/dev/audit_series_rule.py [cell-store]

The audit never edits cells. Schema-1 timeout cells are reported as stale, but are
not a chart-safety failure: chart code displays them only as unbounded ``Timeout``.
``stale_cells.py`` is the tool that schedules those cells for regeneration.
"""
import collections
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import fair_charts as cpu  # noqa: E402
import gpu_charts as gpu  # noqa: E402
from runner_common import require_injective_labels, timeout_cap  # noqa: E402


def main(root):
    paths = sorted(glob.glob(f"{root}/**/*.json", recursive=True))
    cells, bad = [], 0
    for path in paths:
        try:
            cells.append(json.load(open(path)))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"UNPARSEABLE {path}: {exc}")
            bad += 1

    n_cpu = sum(c.get("cell", {}).get("device", "cpu") != "gpu" for c in cells)
    print(f"N = {len(paths)} files; parsed {len(cells)}/{len(paths)} "
          f"({n_cpu} CPU / {len(cells) - n_cpu} GPU)")

    print("\n[1] one series per (solver, configuration)")
    for name, mapping in (("CPU", cpu.LABELS), ("GPU", gpu.LABELS)):
        try:
            require_injective_labels(mapping, name)
            print(f"  ok {name}: {len(mapping)} configurations -> "
                  f"{len(set(mapping.values()))} series")
        except RuntimeError as exc:
            print(f"  FAIL {exc}")
            bad += 1

    print("\n[2] no hidden duplicate draw within a chart series")
    groups = collections.defaultdict(list)
    for cell in cells:
        ident = cell.get("cell", {})
        device = ident.get("device", "cpu")
        mapping = gpu.LABELS if device == "gpu" else cpu.LABELS
        label = mapping.get((ident.get("solver"), ident.get("config", "")))
        if label:
            key = (device, ident.get("family"), ident.get("matrix_id"), label,
                   ident.get("threads"))
            groups[key].append(cell)
    duplicates = {key: value for key, value in groups.items() if len(value) != 1}
    if duplicates:
        bad += 1
        print(f"  FAIL {len(duplicates)}/{len(groups)} series identities are duplicated")
        for key, value in sorted(duplicates.items()):
            print(f"    {key}: statuses={[v.get('status') for v in value]}")
    else:
        print(f"  ok checked {len(groups)}/{len(groups)} charted series identities")

    # Executable regression guard: status and timing may not break a duplicate tie.
    synthetic = []
    for status, total in (("complete", 1.0), ("timeout", None)):
        synthetic.append({
            "cell": {"family": "audit", "matrix_id": "m", "solver": "amgcl",
                     "config": "", "threads": 16},
            "metrics": {"total_s": total}, "status": status,
        })
    try:
        cpu._pick(synthetic, "audit", "m", "AMGCL")
    except RuntimeError:
        print("  ok duplicate mixed-status candidates are rejected, not ranked")
    else:
        print("  FAIL mixed-status candidates were silently selected")
        bad += 1

    timeouts = [cell for cell in cells if cell.get("status") == "timeout"]
    stale = [cell for cell in timeouts if timeout_cap(cell) is None]
    print("\n[3] timeout-cap provenance")
    print(f"  checked {len(timeouts)}/{len(timeouts)} timeout cells: "
          f"{len(timeouts) - len(stale)} bounded, {len(stale)} stale/unbounded")
    if stale:
        print("  note: stale timeout cells are rendered as 'Timeout' without a number; "
              "stale_cells.py schedules them for rerun")

    print("\nFAIL" if bad else "\nPASS")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "results/cells"))
