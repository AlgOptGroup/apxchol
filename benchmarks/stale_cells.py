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

# Matrices the deleted `is_laplacian_operator` ratio heuristic MISCLASSIFIED.
# All four IPM normal equations carry a uniform +1e-6 diagonal shift, so their row
# sums never vanish and only max|diag| moves; the ratio max|rowsum|/max|diag| slid
# under the 1e-10 threshold for three of the four (measured 2026-08-21):
#     iter0010  4.396623e-10  SDDM, correct -- but only 4.4x from flipping
#     iter0020  9.988164e-12  called LAPLACIAN, WRONG
#     iter0030  8.242149e-12  called LAPLACIAN, WRONG
#     iter0040  1.116573e-11  called LAPLACIAN, WRONG
# Every other registry matrix was classified correctly, so only these three flip.
MISSNIFFED = {"iter0020", "iter0030", "iter0040"}

# Solvers that build their OWN right-hand side, outside the benchmark binary, from
# the dumped matrix -- so the flag never reached them and their cells on the
# missniffed matrices stand. The dump itself is unaffected: --dump-mtx writes L
# and exits (benchmark.cpp) before make_rhs is ever called.
#   cmg   bench_cmg.m:44 `is_singular = ~as_operator` -- for a kind=operator matrix
#         it already skipped the mean-centring and the pin, i.e. it was already on
#         the post-fix system.
#   parac ParAC generates its own zero-sum RHS in graph mode and uses its own
#         physics entry point on an operator; neither consults our flag.
# ac/ac2 are NOT here: bench_laplacians.jl carried its own copy of the same ratio
# test, so they would have been missniffed too. (No ac/ac2 cell exists on these
# three matrices today, so the exclusion is moot in this store -- but the rule has
# to stay correct if one is ever produced from a pre-fix checkout.)
OWN_RHS = {"cmg", "parac", "parac_graph", "parac_physics"}

# (commit, one-line reason, predicate over (solver, matrix_id, kind))
RULES = [
    ("6a677bd", "BoomerAMG's hierarchy was built twice inside the setup timer",
     lambda s, m, k: s in HYPRE),
    ("6a677bd", "RCHOL was driven by a PCG we wrote instead of the one it ships",
     lambda s, m, k: s in RCHOL),
    ("9889d01", "operator files were read through the graph reader (diagonal discarded)",
     lambda s, m, k: k == "operator"),
    # This is a RE-RUN rule, not a re-grade: the same flag drives center_if_laplacian
    # inside make_rhs, so mean-centring came off the RIGHT-HAND SIDE as well as off
    # the scoring. The system solved was different (measured on iter0040: sum(b)
    # 1.694e-14 -> 1.815e-10), which is why it hits EVERY solver on these matrices
    # and not just the ones that were being pinned. Re-scoring stored cells cannot
    # recover it; they have to be produced again.
    ("2c95f53", "Laplacian-vs-SDDM was sniffed from a row-sum ratio; iter0020/0030/0040 "
                "were pinned and their RHS mean-centred as if singular",
     lambda s, m, k: m in MISSNIFFED and s not in OWN_RHS),
]


def kind_of_matrix():
    """matrix_id -> declared kind, straight from the registry.

    kind_of() raises on an undeclared or unknown matrix, which is the point: a
    cell whose matrix is no longer in the registry must not be silently graded as
    'not an operator'. Cells for retired matrices fall through to `unknown` below
    via their own stored matrix_meta.
    """
    return {mid: rc.kind_of(mid) for mid in rc.MATRICES}


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
