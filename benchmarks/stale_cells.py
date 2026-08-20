#!/usr/bin/env python3
"""Which stored cells a code change invalidated -- so a re-sweep refills only those.

Every cell records the `git_sha` it was produced at (see PROV in sweep_fair.py), and
the sweep is resume-safe: a cell with a terminal status is skipped. Those two facts
together are the whole mechanism for a limited re-run:

    python3 benchmarks/stale_cells.py            # report, change nothing
    python3 benchmarks/stale_cells.py --delete   # drop the stale cells
    python3 benchmarks/sweep_fair.py ...         # refills exactly the gaps

A cell is stale when a rule below applies to it AND the commit that rule names is
NOT an ancestor of the cell's git_sha -- i.e. the cell was produced before the fix.
Add a rule whenever a change alters what a solver measures or which matrix it solves;
that is cheaper than re-running the suite and safer than remembering by hand.

The report always states the denominator first: total cells, then the filter.
"""
import argparse, collections, json, pathlib, subprocess, sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import runner_common as rc

HYPRE = {"hypre_boomeramg", "hypre_boomeramg_gpu"}
RCHOL = {"rchol", "rchol_par"}

# (commit, one-line reason, predicate over (solver, matrix_id, kind))
RULES = [
    ("6a677bd", "BoomerAMG's hierarchy was built twice inside the setup timer",
     lambda s, m, k: s in HYPRE),
    ("6a677bd", "RCHOL was driven by a PCG we wrote instead of the one it ships",
     lambda s, m, k: s in RCHOL),
    ("9889d01", "operator files were read through the graph reader (diagonal discarded)",
     lambda s, m, k: k == "operator"),
]


def kind_of_matrix():
    """matrix_id -> declared kind, from the registry (grids are graphs by construction)."""
    kind = {g[0]: "graph" for g in rc.GRIDS}
    for name in dir(rc):
        v = getattr(rc, name)
        if isinstance(v, list) and v and isinstance(v[0], tuple) \
           and len(v[0]) == 4 and v[0][3] in rc.KINDS:
            for e in v:
                kind.setdefault(e[0], e[3])
    return kind


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store", default="results/cells", help="cell store to examine")
    ap.add_argument("--delete", action="store_true",
                    help="remove the stale cells (they are in git; restore with git checkout)")
    args = ap.parse_args()

    contains = {}

    def sha_contains(sha, commit):
        """True if `commit` is an ancestor of `sha` -- the cell already has the fix."""
        key = (sha, commit)
        if key not in contains:
            contains[key] = subprocess.run(
                ["git", "merge-base", "--is-ancestor", commit, sha],
                capture_output=True).returncode == 0
        return contains[key]

    kind = kind_of_matrix()
    cells = sorted(pathlib.Path(args.store).rglob("*.json"))
    if not cells:
        sys.exit(f"no cells under {args.store}")

    stale, why = {}, collections.Counter()
    unknown = collections.Counter()
    for p in cells:
        d = json.loads(p.read_text())
        c = d["cell"]
        s, m = c["solver"], c["matrix_id"]
        k = (d.get("matrix_meta") or {}).get("kind") or kind.get(m)
        if k is None:
            unknown[m] += 1
        sha = (d.get("provenance") or {}).get("git_sha", "")
        for commit, reason, pred in RULES:
            if pred(s, m, k) and not (sha and sha_contains(sha, commit)):
                stale.setdefault(p, []).append(reason)
                why[f"{commit}  {reason}"] += 1

    n = len(cells)
    print(f"store {args.store}: {n} cells total")
    print(f"stale: {len(stale)} ({100 * len(stale) / n:.1f}%)   reusable: {n - len(stale)}")
    if unknown:
        print(f"WARNING unresolved kind (treated as not-operator): {dict(unknown)}")
    print("\nby rule (a cell can match more than one):")
    for k2, v in why.most_common():
        print(f"  {v:4d}  {k2}")
    print("\nstale by solver:")
    for k2, v in collections.Counter(
            json.loads(p.read_text())["cell"]["solver"] for p in stale).most_common():
        print(f"  {v:4d}  {k2}")

    if args.delete:
        for p in stale:
            p.unlink()
        print(f"\ndeleted {len(stale)} cells; re-run the sweep to refill exactly these")
    else:
        print("\n(reporting only -- pass --delete to drop them)")


if __name__ == "__main__":
    main()
