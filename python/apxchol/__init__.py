"""apxchol — CPU approximate-Cholesky preconditioner for Laplacian/SDDM systems.

Public API:
    apxchol.factorize(A, **opts)  ->  Solver       (alias: apxchol.solver)
    apxchol.solve(A, b, tol=None, maxiter=500, *, rtol=None, **opts)
                                     ->  SolveResult
    Solver.solve(b, tol=None, maxiter=500, *, rtol=None, x0=None, out=None)
                                     ->  SolveResult
    Solver.apply(r)                  ->  numpy.ndarray   (M^{-1} r)
    Solver.aslinearoperator()        ->  scipy LinearOperator
    Solver.aspreconditioner()        ->  alias of aslinearoperator()
    Solver.P / Solver.chol() / Solver.L / Solver.D   ->  factor export
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator

from . import _apxchol

__all__ = [
    "factorize", "solver", "solve", "Solver", "SolveResult", "__version__",
]

DEFAULT_TOL = 1e-8
DEFAULT_MAXITER = 500

try:
    from importlib.metadata import version as _pkg_version

    __version__ = _pkg_version("apxchol")
except Exception:                      # not installed as a distribution
    __version__ = "0.2.2+local"


@dataclass(frozen=True)
class SolveResult:
    x: np.ndarray
    iters: int
    residual: float
    converged: bool


def _to_csc(A):
    if not sp.issparse(A):
        raise ValueError("A must be a scipy.sparse matrix")
    if A.shape[0] != A.shape[1]:
        raise ValueError(f"A must be square, got shape {A.shape}")
    A = sp.csc_matrix(A)
    if not A.has_canonical_format:
        # csc_matrix() shares the arrays of an already-CSC input, so
        # canonicalize a copy — the caller's matrix must not be mutated.
        A = A.copy()
        A.sum_duplicates()  # consolidate any unmerged (i,j) entries before assembly
        A.sort_indices()
    return A


class Solver:
    """Reusable approximate-Cholesky solver: factor built once, solve many b.

    Options (all keyword-only) are documented on :func:`factorize`.
    """

    def __init__(self, A, *, seed=42, partitioner="block_greedy",
                 storage="vec_pool", keep_factor=True, **advanced):
        csc = _to_csc(A)
        self._n = int(csc.shape[0])
        self._nnz_A = int(csc.nnz)
        self._keep_factor = bool(keep_factor)
        data = np.ascontiguousarray(csc.data, dtype=np.float64)
        options = dict(seed=int(seed), partitioner=str(partitioner),
                       storage=str(storage), keep_factor=self._keep_factor)
        options.update(advanced)
        self._impl = _apxchol.Solver(csc.indptr, csc.indices, data, self._n, options)
        self._chol = None            # lazily built scipy factor (permuted space)
        self._P = None
        self._LD = None

    @property
    def shape(self):
        return (self._n, self._n)

    @property
    def sddm(self) -> bool:
        return self._impl.sddm()

    def __repr__(self) -> str:
        try:
            fac = f"factor_nnz={self.factor_nnz}, fill_ratio={self.fill_ratio:.2f}"
            if not self._keep_factor:
                fac += ", values=released"     # export unavailable
        except Exception:
            fac = "factor=released"
        return f"<apxchol.Solver n={self._n}, sddm={self.sddm}, {fac}>"

    # ── solving ──────────────────────────────────────────────────────────────

    def solve(self, b, tol=None, maxiter=DEFAULT_MAXITER, *, rtol=None,
              x0=None, out=None) -> SolveResult:
        """PCG-solve A x = b to relative residual `rtol` (`tol` is an alias).

        `x0` is an optional initial guess (an exact one returns iters == 0).
        `out`, if given, must be a writable C-contiguous float64 array of
        length n: the solution is written into it (no per-solve allocation)
        and returned as `SolveResult.x`.
        """
        b = np.ascontiguousarray(b, dtype=np.float64).ravel()
        if b.shape[0] != self._n:
            raise ValueError(f"b has length {b.shape[0]}, expected {self._n}")
        if x0 is not None:
            x0 = np.ascontiguousarray(x0, dtype=np.float64).ravel()
            if x0.shape[0] != self._n:
                raise ValueError(f"x0 has length {x0.shape[0]}, expected {self._n}")
        # rtol is the canonical (SciPy) name; tol is kept as a legacy alias.
        eff = rtol if rtol is not None else (tol if tol is not None else DEFAULT_TOL)
        res = self._impl.solve(b, float(eff), int(maxiter), x0, out)
        return SolveResult(
            x=res["x"], iters=int(res["iters"]),
            residual=float(res["residual"]), converged=bool(res["converged"]),
        )

    def apply(self, r) -> np.ndarray:
        r = np.ascontiguousarray(r, dtype=np.float64).ravel()
        if r.shape[0] != self._n:
            raise ValueError(f"r has length {r.shape[0]}, expected {self._n}")
        return self._impl.apply(r)

    def aslinearoperator(self) -> LinearOperator:
        return LinearOperator(shape=self.shape, matvec=self.apply, dtype=np.float64)

    def aspreconditioner(self) -> LinearOperator:
        """Alias of :meth:`aslinearoperator` (M= argument of scipy's Krylov solvers)."""
        return self.aslinearoperator()

    # ── factor export ────────────────────────────────────────────────────────
    # Convention: P[original_vertex] = position in the elimination order, so
    # p_tilde = np.argsort(P) is the ORIGINAL index of the k-th eliminated
    # vertex, and A_perm = A[p_tilde][:, p_tilde] is A in permuted space.

    @property
    def P(self) -> np.ndarray:
        """Elimination permutation: P[original_vertex] = position in the order.

        Always available (the permutation survives even `keep_factor=False`).
        The returned array is read-only; `.copy()` it if you need to modify.
        """
        if self._P is None:
            self._P = self._impl.perm()
            self._P.setflags(write=False)
        return self._P

    def chol(self):
        """Lower-triangular factor G (sqrt-diagonal included) in PERMUTED space.

        Returns a `scipy.sparse.csc_matrix` (float64) with
        `A[p][:, p] ≈ G @ G.T`, `p = np.argsort(self.P)`. G is an APPROXIMATE
        (randomly sampled) factor, so the identity holds only approximately.
        For a pure Laplacian (`sddm == False`) it holds on the rank-(n−k)
        subspace, where k is the number of connected components: the last
        eliminated column of each component carries a placeholder diagonal.
        Requires `keep_factor=True` (the default).

        A fresh matrix is returned on every call — mutating it does not
        affect the solver.
        """
        if self._chol is None:
            indptr, indices, data = self._impl.factor_csc()
            self._chol = sp.csc_matrix((data, indices, indptr),
                                       shape=(self._n, self._n))
        return self._chol.copy()

    def _ld(self):
        if self._LD is None:
            G = self.chol()
            d = G.diagonal()
            # Each column's diagonal is stored first, so dividing column j by
            # d[j] turns G into the unit-lower L of the L·D·L^T form.
            inv = np.where(d != 0.0, 1.0 / np.where(d != 0.0, d, 1.0), 1.0)
            G.data *= np.repeat(inv, np.diff(G.indptr))
            self._LD = (G, d ** 2)
        return self._LD

    @property
    def L(self):
        """Unit-lower-triangular factor in permuted space (G with unit diagonal).

        `A[p][:, p] ≈ L @ diags(D) @ L.T` with `p = np.argsort(self.P)`; the
        same Laplacian caveat as :meth:`chol` applies. A fresh matrix is
        returned on every access.
        """
        return self._ld()[0].copy()

    @property
    def D(self) -> np.ndarray:
        """Diagonal of the L·D·L^T form, i.e. `diag(chol())**2` (fresh copy)."""
        return self._ld()[1].copy()

    @property
    def factor_nnz(self) -> int:
        """nnz of the lower-triangular factor G (diagonal included)."""
        return int(self._impl.factor_nnz())

    @property
    def fill_ratio(self) -> float:
        """Fill relative to the input: `(2 * factor_nnz - n) / nnz(A)`.

        Symmetric convention: the numerator counts the factor G's pattern
        reflected to a full symmetric one (the diagonal once), matching
        `nnz(A)` for a full symmetric A.
        """
        return (2 * self.factor_nnz - self._n) / self._nnz_A


def factorize(A, *, seed=42, partitioner="block_greedy", storage="vec_pool",
              keep_factor=True, **advanced) -> Solver:
    """Build the reusable approximate-Cholesky factor of A.

    Parameters
    ----------
    A : scipy.sparse matrix
        Square graph Laplacian (singular, rank n−1) or SDDM matrix; the two
        cases are auto-detected.
    seed : int
        RNG seed for the randomized clique sampling (factorization is
        deterministic per seed at one thread only).
    partitioner : str
        Independent-set selector: "block_greedy" (default), "luby",
        "baumann_kyng", "rootset". Unknown names raise at factorization time.
    storage : str
        Graph backend: "vec_pool" (default), "forward_star", "vec", "bstr".
    keep_factor : bool
        Keep the factor's row/value arrays alive so `chol()`/`L`/`D` can be
        exported. Costs one extra factor-sized copy in memory (~8 bytes per
        factor nonzero in the default fp32 wheels); with `keep_factor=False`
        those exports raise, while `P`, `factor_nnz` and `fill_ratio` remain
        available. Pass `keep_factor=False` for the leanest factor-once /
        solve-many footprint.
    **advanced
        Passed straight through to the core options: `degree_quantile`,
        `degree_multiplier`, `degree_tiebreak`, `exact_clique_max_degree`,
        `residual_peel` ("natural" | "min_degree" | "bk_serial"),
        `stagnation_window`. Unknown keys raise `ValueError`. Note:
        `degree_multiplier` only takes effect when `degree_quantile=0`
        (the quantile cap, default 0.2, replaces it).
    """
    return Solver(A, seed=seed, partitioner=partitioner, storage=storage,
                  keep_factor=keep_factor, **advanced)


# `solver` is the conversational alias for `factorize` (factor once, reuse).
solver = factorize


def solve(A, b, tol=None, maxiter=DEFAULT_MAXITER, *, rtol=None,
          **opts) -> SolveResult:
    """One-shot convenience: factorize A, then solve A x = b.

    The factor is discarded afterwards, so this passes `keep_factor=False`
    (no factor export) unless overridden.
    """
    opts.setdefault("keep_factor", False)
    return factorize(A, **opts).solve(b, tol=tol, maxiter=maxiter, rtol=rtol)
