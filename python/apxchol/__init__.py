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
    Solver.setup_seconds             ->  float  (wall-clock setup time)
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
    per column; for a vector RHS they are scalars. ``solve_seconds`` is the
    wall-clock time of the native solve, for all columns together; compare
    it with :attr:`Solver.setup_seconds` to judge whether reusing a factor
    pays off.
    """

    x: np.ndarray
    iters: int | np.ndarray
    residual: float | np.ndarray
    converged: bool | np.ndarray
    solve_seconds: float


def _solver_options(*, backend="auto", num_threads=0, seed=None, partitioner=None,
                    factor_storage_dtype=None, **advanced):
    options = dict(backend=backend, num_threads=index(num_threads))
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
    A call made while another thread uses the same solver raises
    ``RuntimeError``; give each thread its own ``copy.copy(solver)``.
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

    def __copy__(self):
        """Return a solver that shares this factor and operator.

        The copy has its own workspace, so another thread can solve with it
        while this solver is in use. GPU-backed solvers cannot be copied.
        """
        copy = type(self).__new__(type(self))
        copy.__dict__.update(self.__dict__)
        copy._initialize(self._impl.__copy__())
        return copy

    def save(self, path):
        """Save the operator and the installed factor to a portable binary file.

        The file holds the preconditioner actually used, in its storage dtype.
        Existing destinations are rejected. The file contains no RHS,
        solution or device-specific workspace.
        """
        self._impl.save(path)

    @classmethod
    def load(cls, path, **opts):
        """Restore a solver from an archive, without requiring SciPy.

        Options use the constructor's defaults. The stored factor is
        reinstalled exactly, not factorized again: its storage dtype and
        dropping come from the file, so ``factor_storage_dtype`` and
        ``drop_tolerance`` have no effect. ``backend`` and ``num_threads``
        apply; ``backend="auto"`` uses CPU, ``"gpu"`` requests upload.
        ``lump_max_mass`` still bounds the archived operator, so pass it again
        if the solver was built with a non-default value. Tolerance, iteration
        budget and warm starts remain solve arguments.
        """
        options = _solver_options(**opts)
        self = cls.__new__(cls)
        self._initialize(_apxchol.Solver.load(path, options))
        return self

    def _init_csc(self, data, indices, indptr, **opts):
        data, indices, indptr = map(np.asarray, (data, indices, indptr))
        if np.iscomplexobj(data):
            raise ValueError("A must be real; apxchol solves real symmetric systems")
        if indices.dtype.kind not in "iu" or indptr.dtype.kind not in "iu":
            raise ValueError("indices and indptr must have integer dtype")
        options = _solver_options(**opts)
        self._initialize(_apxchol.Solver(indptr, indices, data, options))

    def _initialize(self, impl):
        self._impl = impl
        self._n = int(impl.rows())
        self._nnz_A = int(impl.operator_nnz())
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
    def setup_seconds(self) -> float:
        """Wall-clock seconds of the native setup.

        Covers factorization and backend preparation, or for :meth:`load`
        reading and adopting the archive. Copies keep the original's value.
        """
        return self._impl.setup_seconds()

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
        fac = f"factor_nnz={self.factor_nnz}, fill_ratio={self.fill_ratio:.2f}"
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
        rtol = DEFAULT_TOL if rtol is None else float(rtol)
        maxiter = DEFAULT_MAXITER if maxiter is None else index(maxiter)
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
    # The exported factor is the preconditioner actually used: after dropping
    # and storage rounding, M = P^T G G^T P on the permuted space.
    # Convention: P[original_vertex] = position in the elimination order, so
    # p_tilde = np.argsort(P) is the ORIGINAL index of the k-th eliminated
    # vertex, and A_perm = A[p_tilde][:, p_tilde] is A in permuted space.

    @property
    def P(self) -> np.ndarray:
        """Elimination permutation: P[original_vertex] = position in the order.

        The permutation of the preconditioner actually used. The returned
        array is read-only; `.copy()` it if you need to modify.
        """
        if self._P is None:
            self._P = self._impl.perm()
            self._P.setflags(write=False)
        return self._P

    def chol(self):
        """Return a copy of the lower-triangular factor G of the preconditioner actually used.

        The result is a float64 ``scipy.sparse.csc_matrix`` in permuted order
        with ``M[p][:, p] ≈ G @ G.T``, where ``p = np.argsort(self.P)`` and
        ``M`` is the operator the solver factorized: ``A`` after any lumping
        or regularization. It is the installed factor after dropping and
        storage rounding; with float16 storage, G = L S holds each binary16
        off-diagonal times its column scale. A GPU-backed solver downloads it
        on the first export and caches it.

        For a Laplacian, the last row and column of G are empty. With ``k``
        connected components, the factor represents the rank-``n-k``
        subspace, and each other component's last eliminated column carries a
        placeholder diagonal.
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
        """nnz of the installed factor G returned by :meth:`chol` (diagonal included)."""
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
              factor_storage_dtype=None, **advanced) -> Solver:
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
    factor_storage_dtype : numpy dtype, dtype name, or None
        ``np.float16`` or ``np.float32`` selects triangular-solve storage;
        dtype names also work. ``None`` uses the backend default: FP32 on
        CPU/Metal, FP16 on CUDA.

        FP16 stores scaled off-diagonals with FP32 diagonals and scales;
        :meth:`Solver.chol` exports that rounded factor. CPU arithmetic
        stays FP64. Unsupported FP16 requests fail; FP64 storage is unavailable.
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
                  factor_storage_dtype=factor_storage_dtype, **advanced)


def solve(A, b, rtol=None, maxiter=None, *, x0=None, out=None,
          **opts) -> SolveResult:
    """One-shot convenience: factorize A, then solve A x = b.

    ``x0`` and ``out`` have the same meaning as in :meth:`Solver.solve`.
    The factor is discarded afterwards.
    """
    return factorize(A, **opts).solve(b, rtol=rtol, maxiter=maxiter, x0=x0, out=out)
