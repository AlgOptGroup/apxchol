"""apxchol — approximate-Cholesky preconditioner for Laplacian/SDDM systems.

Public API:
    apxchol.factorize(A, **opts)  ->  Solver
    apxchol.solve(A, b, rtol=None, maxiter=None, **opts)
                                     ->  SolveResult
    apxchol.as_operator(M)           ->  M, or its graph Laplacian L = D - M
    Solver.solve(b, rtol=None, maxiter=None, *, x0=None, out=None)
                                     ->  SolveResult
    Solver.apply(r)                  ->  numpy.ndarray   (M^{-1} r)
    Solver.from_csc(data, indices, indptr, **opts) -> Solver
    Solver.save(path) / Solver.load(path, **opts) -> portable archive / Solver
    Solver.lumped                    ->  int  (positive off-diagonals lumped)
    Solver.aslinearoperator()        ->  scipy LinearOperator
    Solver.P / Solver.chol() / Solver.L / Solver.D   ->  factor export
"""
from __future__ import annotations

from dataclasses import dataclass
from operator import index

import numpy as np

from . import _apxchol

__all__ = [
    "factorize", "solve", "as_operator",
    "Solver", "SolveResult", "__version__",
]

DEFAULT_TOL = _apxchol.DEFAULT_TOL
DEFAULT_MAXITER = _apxchol.DEFAULT_MAXITER

__version__ = _apxchol.__version__


@dataclass(frozen=True)
class SolveResult:
    """Solution and convergence information for the original system.

    ``residual`` is checked against the original matrix and RHS. For a
    matrix RHS, ``iters``, ``residual`` and ``converged`` contain one entry
    per column; for a vector RHS they are scalars.
    """

    x: np.ndarray
    iters: int | np.ndarray
    residual: float | np.ndarray
    converged: bool | np.ndarray


def _solver_options(*, backend="auto", num_threads=0, seed=None, partitioner=None,
                    keep_factor=True, factor_storage_dtype=None, **advanced):
    options = dict(backend=backend, num_threads=index(num_threads),
                   keep_factor=bool(keep_factor))
    if seed is not None:
        options["seed"] = index(seed)
    if partitioner is not None:
        options["partitioner"] = partitioner
    options.update(advanced)
    if "exact_clique_max_degree" in options:
        options["exact_clique_max_degree"] = index(options["exact_clique_max_degree"])
    if factor_storage_dtype is not None:
        options["factor_storage_dtype"] = np.dtype(factor_storage_dtype).name
    return options


def _scipy_sparse():
    try:
        import scipy.sparse as sp
    except ModuleNotFoundError as exc:
        if exc.name != "scipy":
            raise
        raise ImportError("Install apxchol[scipy] to use this operation.") from None
    return sp


def as_operator(M):
    """Return `M` as an operator for :func:`factorize` and :func:`solve`.

    Like :func:`numpy.asarray`, a valid input is returned unchanged:

    - If `M` passes the solver's operator check (a symmetric Laplacian or
      SDDM matrix, possibly nearly SDDM), `M` itself is returned.
    - Otherwise, if `M` has no negative off-diagonal entry, it is a graph
      adjacency matrix and its Laplacian ``L = D - M`` is returned, where
      ``D`` holds the row sums of `M`. A diagonal entry (self-loop) adds to
      ``D`` and is subtracted again, so it cancels. Boolean (pattern) input
      has unit weights.
    - Otherwise a ``ValueError`` explains why `M` is neither.

    The CLI's ``--convert-input`` applies the same rule.

    Parameters
    ----------
    M : scipy.sparse matrix or array
        Square, real matrix in any sparse format. `M` is not modified;
        duplicate entries are summed.

    Returns
    -------
    scipy.sparse matrix
        `M`, or its float64 CSC Laplacian.

    Examples
    --------
    >>> M = scipy.io.mmread("com-Amazon.mtx")              # adjacency
    >>> res = apxchol.solve(apxchol.as_operator(M), b)
    """
    sp = _scipy_sparse()
    csc = _to_csc(M)
    laplacian = _apxchol.as_operator(csc.indptr, csc.indices, csc.data)
    if laplacian is None:
        return M
    indptr, indices, data = laplacian
    return sp.csc_matrix((data, indices, indptr), shape=csc.shape)


def _to_csc(A):
    sp = _scipy_sparse()
    if not sp.issparse(A):
        raise ValueError("A must be a scipy.sparse matrix")
    if A.shape[0] != A.shape[1]:
        raise ValueError(f"A must be square, got shape {A.shape}")
    A = sp.csc_matrix(A)
    if np.iscomplexobj(A.data):
        raise ValueError("A must be real; apxchol solves real symmetric systems")
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

    def __init__(self, A, **opts):
        csc = _to_csc(A)
        self._init_csc(csc.data, csc.indices, csc.indptr, **opts)

    @classmethod
    def from_csc(cls, data, indices, indptr, **opts):
        """Build from NumPy-compatible CSC arrays, without requiring SciPy.

        The operator is square with ``len(indptr) - 1`` rows. Indices are
        zero-based integers. Unsorted entries and duplicates are accepted;
        the native importer sorts and sums them without changing the input.
        Options are the same as :func:`factorize`.
        """
        self = cls.__new__(cls)
        self._init_csc(data, indices, indptr, **opts)
        return self

    def save(self, path):
        """Save the operator and retained factor to a portable binary file.

        Requires ``keep_factor=True``. Existing destinations are rejected.
        The file contains no RHS, solution or device-specific workspace.
        """
        self._impl.save(path)

    @classmethod
    def load(cls, path, **opts):
        """Restore a solver from an archive, without requiring SciPy.

        Options use the constructor's defaults. The stored factor is adopted,
        not factorized again; backend, storage and preparation settings apply.
        ``backend="auto"`` uses CPU for adoption; ``"gpu"`` requests upload.
        Tolerance, iteration budget and warm starts remain solve arguments.
        """
        options = _solver_options(**opts)
        self = cls.__new__(cls)
        self._initialize(_apxchol.Solver.load(path, options), options["keep_factor"])
        return self

    def _init_csc(self, data, indices, indptr, **opts):
        data, indices, indptr = map(np.asarray, (data, indices, indptr))
        if np.iscomplexobj(data):
            raise ValueError("A must be real; apxchol solves real symmetric systems")
        if indices.dtype.kind not in "iu" or indptr.dtype.kind not in "iu":
            raise ValueError("indices and indptr must have integer dtype")
        options = _solver_options(**opts)
        self._initialize(_apxchol.Solver(indptr, indices, data, options),
                         options["keep_factor"])

    def _initialize(self, impl, keep_factor):
        self._impl = impl
        self._n = int(impl.rows())
        self._nnz_A = int(impl.operator_nnz())
        self._keep_factor = bool(keep_factor)
        self._chol = None            # lazily built scipy factor (permuted space)
        self._P = None
        self._LD = None

    @property
    def shape(self):
        """Shape of the square operator, as ``(n, n)``."""
        return (self._n, self._n)

    @property
    def backend(self) -> str:
        """Backend selected at construction: ``"cpu"`` or ``"gpu"``."""
        return self._impl.backend()

    @property
    def sddm(self) -> bool:
        """Whether the solver uses SDDM rather than Laplacian handling."""
        return self._impl.sddm()

    @property
    def factor_storage_dtype(self) -> np.dtype:
        """Resolved triangular-solve storage dtype; CPU arithmetic remains float64."""
        return np.dtype(self._impl.factor_storage_dtype())

    @property
    def lumped(self) -> int:
        """Number of positive off-diagonal entries moved during preconditioning.

        Zero for a Laplacian or SDDM matrix. Each positive pair is moved to
        the diagonal of the preconditioner and counts as two entries. PCG
        and the reported residual still use the original, unchanged matrix.
        """
        return self._impl.lumped()

    def __repr__(self) -> str:
        try:
            fac = f"factor_nnz={self.factor_nnz}, fill_ratio={self.fill_ratio:.2f}"
            if not self._keep_factor:
                fac += ", values=released"     # export unavailable
        except Exception:
            fac = "factor=released"
        return f"<apxchol.Solver n={self._n}, sddm={self.sddm}, {fac}>"

    # ── solving ──────────────────────────────────────────────────────────────

    def _rhs(self, value, name):
        if np.iscomplexobj(value):
            raise ValueError(f"{name} must be real; apxchol solves real symmetric systems")
        value = np.asarray(value, dtype=np.float64)
        if value.ndim not in (1, 2) or value.shape[0] != self._n:
            raise ValueError(f"{name} must have shape ({self._n},) or ({self._n}, nrhs)")
        if not np.isfinite(value).all():
            raise ValueError(f"{name} must be finite")
        return value

    def solve(self, b, rtol=None, maxiter=None, *,
              x0=None, out=None) -> SolveResult:
        """Solve ``A @ x = b`` with PCG, reusing this solver's factor.

        ``b`` has shape ``(n,)`` or ``(n, nrhs)``. Matrix inputs use the native
        batched solver and return one convergence result per column. Omitted
        ``rtol`` and ``maxiter`` use the compiled native defaults.

        ``x0`` is an initial solution estimate with the RHS shape; an
        already-converged guess takes zero iterations. ``out`` must have
        that shape and be writable, C- or F-contiguous float64. It is returned
        as :attr:`SolveResult.x`.

        The final residual is always checked against the original matrix
        and RHS. Inspect :attr:`SolveResult.converged` for success.
        """
        b = self._rhs(b, "b")
        if x0 is not None:
            x0 = self._rhs(x0, "x0")
            if x0.shape != b.shape:
                raise ValueError("x0 must have the same shape as b")
        rtol = None if rtol is None else float(rtol)
        maxiter = None if maxiter is None else index(maxiter)
        res = self._impl.solve(b, rtol, maxiter, x0, out)
        return SolveResult(**res)

    def apply(self, r) -> np.ndarray:
        """Apply the preconditioner to a vector or RHS matrix, preserving shape."""
        return self._impl.apply(self._rhs(r, "r"))

    def aslinearoperator(self):
        """Return a SciPy ``LinearOperator`` for another iterative solver.

        That solver controls convergence. With a GPU-backed factor, each
        application exchanges vectors with the host; :meth:`solve` keeps
        the GPU iteration on the device.
        """
        _scipy_sparse()
        from scipy.sparse.linalg import LinearOperator
        return LinearOperator(
            shape=self.shape, matvec=self.apply, matmat=self.apply, dtype=np.float64)

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
        """Return a copy of the FP32 lower-triangular factor in permuted order.

        The result is a ``scipy.sparse.csc_matrix`` with
        ``A[p][:, p] ≈ G @ G.T``, where ``p = np.argsort(self.P)``.
        Export requires ``keep_factor=True`` (the default), regardless of
        backend or internal factor storage dtype.

        For a Laplacian with ``k`` connected components, the factor represents
        the rank-``n-k`` subspace. Each component's last eliminated column
        carries a placeholder diagonal.
        """
        if self._chol is None:
            sp = _scipy_sparse()
            indptr, indices, data = self._impl.factor_csc()
            self._chol = sp.csc_matrix((data, indices, indptr), shape=(self._n, self._n))
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


def factorize(A, *, backend="auto", num_threads=0, seed=None, partitioner=None,
              keep_factor=True, factor_storage_dtype=None, **advanced) -> Solver:
    """Build the reusable approximate-Cholesky factor of A.

    Parameters
    ----------
    A : scipy.sparse matrix or array
        The square, real symmetric matrix in ``A @ x = b``, typically a graph
        Laplacian or SDDM matrix. Nonempty rows need positive diagonals.
        Supply both triangles; sparse inputs are converted to sorted,
        duplicate-free CSC without modifying the caller's matrix.

        Use :func:`as_operator` to convert adjacency input first. A
        Laplacian RHS must sum to zero within each connected component.

        Positive off-diagonal pairs are lumped onto the diagonal for
        preconditioning, subject to `lump_max_mass` (see :attr:`Solver.lumped`).
        PCG and residual checks still use the original `A`.
    backend : {"auto", "cpu", "gpu"}
        Auto may prefer CPU for small systems when it supports the requested storage;
        otherwise it checks GPU compatibility. Read the resolved choice from
        ``Solver.backend``. An explicit GPU request fails when the
        installed build, hardware or options cannot support it.
    num_threads : int
        CPU thread ceiling for setup and every solve/apply. Positive values
        override the calling OpenMP task's setting; zero inherits it. Small
        workloads may use fewer threads. Each operation restores the setting.
    seed : int or None
        RNG seed for the randomized clique sampling (factorization is
        deterministic per seed at one thread only). None uses the C++ default.
    partitioner : str or None
        Independent-set selector: "block_greedy", "priority_greedy", or
        "baumann_kyng". None uses the C++ default. Unknown names raise at
        factorization time.
    keep_factor : bool
        Retain an extra host factor copy for ``save()``, ``chol()``, ``L``
        and ``D`` export; enabled by default. ``False`` saves this memory while still
        allowing repeated solves, ``P``, ``factor_nnz`` and ``fill_ratio``.
    factor_storage_dtype : numpy dtype, dtype name, or None
        ``np.float16`` or ``np.float32`` selects triangular-solve storage;
        dtype names also work. ``None`` uses the backend default: FP32 on
        CPU/Metal, FP16 on CUDA.

        FP16 stores scaled off-diagonals with FP32 diagonals and scales. It
        does not change the exported factor's FP32 type or CPU arithmetic's
        FP64 type. Unsupported FP16 requests fail; FP64 storage is unavailable.
    **advanced
        Passed straight through to the core options: `degree_quantile`,
        `degree_multiplier`, `degree_tiebreak`, `exact_clique_max_degree`,
        `residual_peel` ("natural" | "min_degree" | "bk_serial"),
        `cpu_schedule` ("auto" | "levels"),
        `drop_tolerance` (in [0, 1]; 0 disables dropping),
        `regularization` (nonnegative diagonal fraction; 0 disables),
        `lump_max_mass` (fraction in [0, 1]; 0 rejects positive off-diagonals),
        and `verbose` (default False).

    Notes
    -----
    Unknown option keys raise ``ValueError``.

    ``degree_multiplier`` only takes effect when ``degree_quantile=0``.
    Omitting the quantile uses the backend's default. Metal requires a
    quantile in ``(0, 1)`` and includes all vertices tied at its degree cutoff.
    """
    return Solver(A, backend=backend, num_threads=num_threads, seed=seed, partitioner=partitioner,
                  keep_factor=keep_factor, factor_storage_dtype=factor_storage_dtype,
                  **advanced)


def solve(A, b, rtol=None, maxiter=None, *, x0=None, out=None,
          **opts) -> SolveResult:
    """One-shot convenience: factorize A, then solve A x = b.

    ``x0`` and ``out`` have the same meaning as in :meth:`Solver.solve`.
    The factor is discarded afterwards, so this passes `keep_factor=False`
    (no factor export) unless overridden.
    """
    opts.setdefault("keep_factor", False)
    return factorize(A, **opts).solve(b, rtol=rtol, maxiter=maxiter, x0=x0, out=out)
