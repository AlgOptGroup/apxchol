#!/usr/bin/env python3
"""Score a ParAC solution against the PUBLISHED matrix, independently of ParAC.

ParAC prints a relative residual, but it is computed against the operator its
driver ended up with after reordering, trimming and (for the physics path)
augmentation. Whether that equals the system the benchmark reports on is exactly
the question, so this tool recomputes it from raw vectors:

  * `--xb`   a binary dump of (int64 n, double x[n], double b[n]) written by an
             instrumented ParAC driver;
  * `--perm` the permutation p written by parac_reorder_amd.jl, with the file the
             driver read equal to G[p, p] for the reorder input G;
  * `--target` the matrix the answer must satisfy, in the benchmark's own index
             space: the pure L for a graph, the published operator for a kind=
             operator file.

Modes:
  perm       len(x) == len(p): map x, b straight back through p.
  perm_trim  len(x) == len(p) - 1: the driver trimmed the LAST row/column of the
             file it read, so pad x and b with a zero before mapping back. Use
             this when the reordered file was NOT augmented — the trimmed node is
             a real degree of freedom and the padding exposes the damage.

The augmented physics path needs no special mode: the file the driver read is
(n+1)x(n+1), the trim removes the appended ground node, and what is left is
G[p, p] itself, so len(x) == len(p) and `perm` applies.
"""
import argparse
import numpy as np
import scipy.io
import scipy.sparse as sp


def read_xb(path):
    with open(path, "rb") as f:
        n = int(np.fromfile(f, dtype=np.int64, count=1)[0])
        x = np.fromfile(f, dtype=np.float64, count=n)
        b = np.fromfile(f, dtype=np.float64, count=n)
    if x.size != n or b.size != n:
        raise ValueError(f"{path}: truncated (n={n}, got {x.size}/{b.size})")
    return x, b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, help="matrix the answer must satisfy (.mtx)")
    ap.add_argument("--xb", required=True, help="binary (n, x, b) dump")
    ap.add_argument("--perm", help="permutation file from parac_reorder_amd.jl")
    ap.add_argument("--mode", default="perm", choices=("perm", "perm_trim", "none"))
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    A = sp.csr_array(scipy.io.mmread(args.target))
    x, b = read_xb(args.xb)

    if args.mode == "none":
        xo, bo = x, b
    else:
        p = np.loadtxt(args.perm, dtype=np.int64) - 1        # to 0-based
        if args.mode == "perm_trim":
            x = np.append(x, 0.0)
            b = np.append(b, 0.0)
        if x.size != p.size:
            raise ValueError(f"length mismatch: |x|={x.size} |perm|={p.size} "
                             f"(wrong --mode?)")
        xo = np.empty_like(x); bo = np.empty_like(b)
        xo[p] = x; bo[p] = b

    if xo.size != A.shape[0]:
        raise ValueError(f"length mismatch: |x|={xo.size} n(target)={A.shape[0]}")

    r = bo - A @ xo
    nb = np.linalg.norm(bo)
    rel = np.linalg.norm(r) / nb
    print(f"{args.label + ' ' if args.label else ''}"
          f"target={args.target.split('/')[-1]} n={A.shape[0]} "
          f"||b||={nb:.6e} ||b-Ax||/||b||={rel:.6e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
