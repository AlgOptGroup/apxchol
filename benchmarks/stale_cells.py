#!/usr/bin/env python3
"""Which stored cells a code change invalidated -- so a re-sweep refills only those.

Every cell records the `git_sha` it was produced at (see PROV in sweep_fair.py), and
the sweep is resume-safe. Together they let the fair sweep perform a limited,
non-destructive re-run:

    python3 benchmarks/stale_cells.py            # report, change nothing
    python3 benchmarks/sweep_fair.py ...         # replaces stale cells as runs finish

``--delete`` remains an optional, recoverable cleanup tool. It is not required
before a fair sweep: the runner uses the predicate in this module and leaves an
old cell in place until its replacement run has returned.

A cell is stale when a rule below applies to it AND the commit that rule names is
NOT an ancestor of the cell's git_sha -- i.e. the cell was produced before the fix.
Independently, schema-1 timeout cells are stale because they do not persist the cap
needed to interpret the outcome numerically. Add a rule whenever a change alters what
a solver measures or which matrix it solves; that is cheaper than re-running the suite
and safer than remembering by hand.

The report always states the denominator first: total cells, then the filter.
"""
import argparse, collections, functools, json, pathlib, subprocess, sys

try:
    from . import runner_common as rc
except ImportError:  # Executed as ``python3 benchmarks/stale_cells.py``.
    import runner_common as rc

HYPRE = {"hypre_boomeramg", "hypre_boomeramg_gpu"}
RCHOL = {"rchol", "rchol_par"}
PARAC = {"parac", "parac_graph", "parac_physics"}
IN_PROCESS_SOLVERS = {"apxchol_v1", "amgcl", "amgcl_cuda", *HYPRE, *RCHOL}
# Complete component census with the current binary, 2026-08-30: checked all 8/8
# file-backed graph inputs plus the only file-backed Laplacian operator. Only
# these two graph Laplacians are disconnected; generated grids are connected.
DISCONNECTED_LAPLACIANS = {"kron_g500-logn16", "as-Skitter"}
# Solvers that do not read the matrix from us at all: they consume the .mtx we DUMP.
# A change to what we dump invalidates their cells even when their own code and the
# solver-side harness are untouched, which is why they need a rule of their own --
# nothing else in this file would notice.
DUMP_CONSUMERS = {"parac", "parac_graph", "parac_physics", "cmg", "ac", "ac2"}

# c915ad5 establishes the canonical timing boundary for every affected external
# or in-process competitor. CPU AMGCL is intentionally absent: it already timed
# its pin/conversion/constructor and returned host x; only its CUDA adapter was
# undercounted. Hypre/RCHOL include their now-timed result retrieval/unpermutation.
TIMING_REPAIRED = {
    "amgcl_cuda",
    "hypre_boomeramg", "hypre_boomeramg_gpu",
    "rchol", "rchol_par",
    "ac", "ac2", "cmg",
    "parac", "parac_graph", "parac_physics",
}

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

# (commit, one-line reason, predicate over (solver, matrix_id, kind, device))
RULES = [
    ("6a677bd", "BoomerAMG's hierarchy was built twice inside the setup timer",
     lambda s, m, k, d: s in HYPRE),
    ("6a677bd", "RCHOL was driven by a PCG we wrote instead of the one it ships",
     lambda s, m, k, d: s in RCHOL),
    ("9889d01", "operator files were read through the graph reader (diagonal discarded)",
     lambda s, m, k, d: k == "operator"),
    ("3bda0c9", "the dumped .mtx these solvers consume changed (--pin-dump dropped; "
                "external solvers ground natively). 3bda0c9 also contains 9889d01, so "
                "requiring it covers the operator-interpretation change to the dump too",
     lambda s, m, k, d: s in DUMP_CONSUMERS),
    # This is a RE-RUN rule, not a re-grade: the same flag drives center_if_laplacian
    # inside make_rhs, so mean-centring came off the RIGHT-HAND SIDE as well as off
    # the scoring. The system solved was different (measured on iter0040: sum(b)
    # 1.694e-14 -> 1.815e-10), which is why it hits EVERY solver on these matrices
    # and not just the ones that were being pinned. Re-scoring stored cells cannot
    # recover it; they have to be produced again.
    ("2c95f53", "Laplacian-vs-SDDM was sniffed from a row-sum ratio; iter0020/0030/0040 "
                "were pinned and their RHS mean-centred as if singular",
     lambda s, m, k, d: m in MISSNIFFED and s not in OWN_RHS),
    ("c915ad5", "competitor timing omitted mandatory conversion/setup/result work or "
                "included benchmark-only validation/cleanup",
     lambda s, m, k, d: s in TIMING_REPAIRED),
    ("5a18d14", "the CUDA primary-context creation was lazily charged inside the first "
                "GPU solver instead of reported once as shared process initialization",
     lambda s, m, k, d: d == "gpu"),
    ("a4938af", "apxchol's mandatory operator scan and optional M-matrix lumping were "
                "outside its setup timer",
     lambda s, m, k, d: s == "apxchol_v1"),
    ("a4938af", "HYPRE_Init was omitted and disconnected-component setup/scoring was not "
                "accounted coherently",
     lambda s, m, k, d: s in HYPRE),
    ("a4938af", "ParAC omitted complete producer/serialization work, used incoherent "
                "repetition medians, or handled components through an obsolete route",
     lambda s, m, k, d: s in PARAC),
    ("a4938af", "disconnected Laplacian RHS projection and residual grading used one "
                "global mean instead of one constant null vector per component",
     lambda s, m, k, d: s in IN_PROCESS_SOLVERS and m in DISCONNECTED_LAPLACIANS),
]


def matching_rules(solver, matrix_id, kind, device):
    """Rules whose semantic scope contains one cell, independent of its git SHA."""
    return [(commit, reason) for commit, reason, pred in RULES
            if pred(solver, matrix_id, kind, device)]

TIMEOUT_CAP_REASON = ("timeout outcome predates schema 2 and does not record the "
                      "actual wall-clock cap used")

def timeout_cap_is_stale(cell):
    """Schema-1 timeout outcomes cannot support a numerical lower bound."""
    return cell.get("status") == "timeout" and rc.timeout_cap(cell) is None


def kind_of_matrix():
    """matrix_id -> declared kind, straight from the registry.

    kind_of() raises on an undeclared or unknown matrix, which is the point: a
    cell whose matrix is no longer in the registry must not be silently graded as
    'not an operator'. Cells for retired matrices fall through to `unknown` below
    via their own stored matrix_meta.
    """
    return {mid: rc.kind_of(mid) for mid in rc.MATRICES}


@functools.cache
def sha_contains(sha, commit):
    """Whether the recorded revision contains one stale-rule fix.

    Missing and unknown revisions fail closed: a scoped record is reusable only
    when Git can prove that it contains the required semantic change.
    """
    if not sha:
        return False
    return subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, sha],
        cwd=rc.ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    ).returncode == 0


def stale_reasons(cell, *, sha_contains_fn=None, kinds=None):
    """Return every current invalidation rule matched by one stored cell.

    The result is ``(rule, reason)`` pairs so reporting and resume decisions use
    exactly the same predicate. ``sha_contains_fn`` is injectable for tests and
    other read-only consumers; production defaults to the cached Git query above.
    """
    if not isinstance(cell, dict):
        return ()
    check = sha_contains if sha_contains_fn is None else sha_contains_fn
    reasons = []
    if timeout_cap_is_stale(cell):
        reasons.append(("schema-2", TIMEOUT_CAP_REASON))

    identity = cell.get("cell")
    if not isinstance(identity, dict):
        return tuple(reasons)
    solver = identity.get("solver", "")
    matrix_id = identity.get("matrix_id", "")
    device = identity.get("device", "")
    matrix_meta = cell.get("matrix_meta")
    kind = matrix_meta.get("kind") if isinstance(matrix_meta, dict) else None
    if kind is None:
        registered = kind_of_matrix() if kinds is None else kinds
        kind = registered.get(matrix_id)
    provenance = cell.get("provenance")
    sha = provenance.get("git_sha", "") if isinstance(provenance, dict) else ""
    for commit, reason in matching_rules(solver, matrix_id, kind, device):
        if not check(sha, commit):
            reasons.append((commit, reason))
    return tuple(reasons)


def cell_is_stale(cell, *, sha_contains_fn=None, kinds=None):
    """Authoritative stale predicate shared by reporting and fair-sweep resume."""
    return bool(stale_reasons(cell, sha_contains_fn=sha_contains_fn, kinds=kinds))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store", default="results/cells", help="cell store to examine")
    ap.add_argument("--delete", action="store_true",
                    help="optionally remove stale cells (not needed before sweep_fair.py; "
                         "tracked cells are recoverable from Git)")
    args = ap.parse_args()

    kind = kind_of_matrix()
    cells = sorted(pathlib.Path(args.store).rglob("*.json"))
    if not cells:
        sys.exit(f"no cells under {args.store}")

    stale, why = {}, collections.Counter()
    unknown = collections.Counter()
    for p in cells:
        d = json.loads(p.read_text())
        c = d["cell"]
        m = c["matrix_id"]
        k = (d.get("matrix_meta") or {}).get("kind") or kind.get(m)
        if k is None:
            unknown[m] += 1
        for rule, reason in stale_reasons(d, kinds=kind):
            stale.setdefault(p, []).append(reason)
            why[f"{rule}  {reason}"] += 1

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
