"""Extract a public, provenance-bound summary from a sweep_fair cell store.

The planned denominator is every listed matrix times every series observed in
the store. Missing cells are listed, never dropped; a complete cell whose
retained original-system residual exceeds the tolerance does not meet it.
"""
import argparse
import collections
import glob
import json
import os
import re
import sys

METRICS = ("setup_s", "solve_s", "total_s", "iters", "rel_res", "max_repeat_rel_res",
           "cuda_init_s", "max_rss_mb", "retained_repeats", "execution_route")
TOOLCHAIN = ("compiler", "compiler_version", "openmp_runtime", "arch_flags",
             "node_index_bits", "edge_index_bits", "hypre_cuda")


def read_hashes(path):
    """sha256sum output -> {path relative to the entries' common prefix: digest}."""
    entries = []
    with open(path) as f:
        lines = f.read().splitlines()
    for line in lines:
        digest, name = line.split(maxsplit=1)
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError(f"{path}: bad entry {line.strip()!r}")
        entries.append((name.strip().lstrip("*"), digest))
    names = [n for n, _ in entries]
    root = os.path.commonpath(names) if len(names) > 1 else os.path.dirname(names[0])
    out = {os.path.relpath(n, root) if root else n: d for n, d in entries}
    if len(out) != len(entries):
        raise ValueError(f"{path}: duplicate entry")
    return out


def read_env(paths):
    """Hardware lines written by run_role.sh; every role must report the same."""
    seen = set()
    for path in paths:
        with open(path) as f:
            text = f.read()
        cpu = re.search(r"Model name:\s*(.+)", text)
        sockets = re.search(r"Socket\(s\):\s*(\d+)", text)
        gpus = re.findall(r"^\d+, ([^,]+), (\d+) MiB, ([\d.]+)$", text, re.M)
        if not cpu or not sockets:
            raise ValueError(f"{path}: no CPU model or socket count")
        seen.add((cpu.group(1).strip(), int(sockets.group(1)),
                  tuple(sorted(collections.Counter((g[0], int(g[1]), g[2]) for g in gpus).items()))))
    if len(seen) != 1:
        raise ValueError(f"roles ran on different hardware: {sorted(seen)}")
    cpu, sockets, gpus = seen.pop()
    return {"cpu": cpu, "sockets": sockets,
            "gpus": [{"name": g[0], "memory_mib": g[1], "driver": g[2], "count": n} for g, n in gpus]}


def extract(store, matrix_ids, source_commit, tolerance=1e-8):
    if not re.fullmatch(r"[0-9a-f]{40}", source_commit):
        raise ValueError("source commit must be a full 40-hex object id")
    cells, toolchains = {}, []
    for path in sorted(glob.glob(os.path.join(store, "*", "*.json"))):
        with open(path) as f:
            c = json.load(f)
        key = c["cell"]
        series = (key["solver"], key["config"], key["device"], key["threads"])
        ident = (key["matrix_id"], series)
        if ident in cells:
            raise ValueError(f"duplicate cell {ident}")
        prov = c.get("provenance") or {}
        if prov.get("git_sha") and prov["git_sha"] != source_commit:
            raise ValueError(f"{path}: built from {prov['git_sha']}, not {source_commit}")
        toolchain = dict((k, str(prov[k])) for k in TOOLCHAIN if prov.get(k) not in (None, ""))
        if toolchain not in toolchains:
            toolchains.append(toolchain)
        m = c.get("metrics") or {}
        rel = [v for v in (m.get("rel_res"), m.get("max_repeat_rel_res")) if isinstance(v, (int, float))]
        meta = c.get("matrix_meta") or {}
        cells[ident] = {
            "family": key["family"], "matrix": key["matrix_id"], "kind": meta.get("kind"),
            "class": meta.get("class"), "n": meta.get("n"), "nnz": meta.get("nnz"),
            "solver": key["solver"], "config": key["config"], "device": key["device"],
            "threads": key["threads"], "toolchain": toolchains.index(toolchain), "status": c.get("status"),
            "meets_tolerance": c.get("status") == "complete" and bool(rel) and max(rel) <= tolerance,
            **{k: m.get(k) for k in METRICS if m.get(k) is not None},
        }
        if c.get("timeout_cap_s") is not None:
            cells[ident]["process_cap_s"] = c["timeout_cap_s"]
        if c.get("status") != "complete":
            reason = [ln for ln in str(m.get("stderr_tail") or "").splitlines()
                      if ln.strip() and not ln.startswith(("BUILD_META", "MATRIX_META", "APXRSS", "Command exited"))]
            if reason:
                cells[ident]["reason"] = reason[-1].strip()
    unknown = sorted({mat for mat, _ in cells} - set(matrix_ids))
    if unknown:
        raise ValueError(f"cells for unlisted matrices: {unknown}")
    series = sorted({s for _, s in cells})
    planned = [(mat, s) for mat in matrix_ids for s in series]
    missing = [{"matrix": mat, "solver": s[0], "config": s[1], "device": s[2], "threads": s[3]}
               for mat, s in planned if (mat, s) not in cells]
    records = [cells[p] for p in planned if p in cells]
    return {
        "planned_cells": len(planned), "present_cells": len(records),
        "meets_tolerance": sum(r["meets_tolerance"] for r in records),
        "status_counts": dict(collections.Counter(r["status"] for r in records)),
        "tolerance": tolerance, "source_commit": source_commit,
        "toolchains": toolchains,
        "missing": missing, "cells": records,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("store", help="cell store: STORE/<family>/<cell>.json")
    ap.add_argument("--matrices", required=True, help="comma-separated planned matrix ids")
    ap.add_argument("--source-commit", required=True)
    ap.add_argument("--campaign", required=True)
    ap.add_argument("--date", required=True, help="YYYY-MM-DD of the measurements")
    ap.add_argument("--env", nargs="+", required=True, help="run_role.sh env-*.txt files")
    ap.add_argument("--binaries", required=True, help="sha256sum list of the measured binaries")
    ap.add_argument("--matrix-hashes", required=True, help="sha256sum list of the matrix files")
    ap.add_argument("--planned", type=int, help="expected denominator; mismatch is an error")
    ap.add_argument("-o", "--output", required=True)
    a = ap.parse_args(argv)
    ids = [m for m in a.matrices.split(",") if m]
    if len(set(ids)) != len(ids):
        sys.exit("duplicate matrix id")
    report = extract(a.store, ids, a.source_commit)
    if a.planned is not None and report["planned_cells"] != a.planned:
        sys.exit(f"planned {report['planned_cells']} cells, expected {a.planned}")
    report = {"schema_version": 1, "campaign": a.campaign, "date": a.date,
              "hardware": read_env(a.env), "binaries": read_hashes(a.binaries),
              "matrix_files": read_hashes(a.matrix_hashes), **report}
    with open(a.output, "w") as f:
        json.dump(report, f, indent=1, sort_keys=False)
        f.write("\n")
    print(f"{a.output}: {report['present_cells']}/{report['planned_cells']} cells present, "
          f"{report['meets_tolerance']} within {report['tolerance']:g}, {len(report['missing'])} missing")


if __name__ == "__main__":
    main()
