import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import cg, norm as spnorm, spsolve_triangular
import pytest

import apxchol


def grid2d_laplacian(m):
    """2-D m×m grid graph Laplacian (singular, rank n-1), as scipy CSR."""
    n = m * m
    rows, cols = [], []
    for i in range(m):
        for j in range(m):
            v = i * m + j
            if j + 1 < m:
                rows += [v, v + 1]; cols += [v + 1, v]
            if i + 1 < m:
                rows += [v, v + m]; cols += [v + m, v]
    A = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n)).tocsr()
    deg = np.asarray(A.sum(axis=1)).ravel()
    L = sp.diags(deg) - A
    return L.tocsr()


def grid2d_sddm(m, shift=0.5):
    """Grid Laplacian + shift·I: full-rank SDDM (positive row sums)."""
    L = grid2d_laplacian(m)
    return (L + shift * sp.eye(L.shape[0])).tocsr()


def random_sddm(n, seed=0):
    """Small SPD diagonally-dominant (SDDM) matrix."""
    rng = np.random.default_rng(seed)
    M = sp.random(n, n, density=0.1, random_state=rng)
    A = (M + M.T).tocsr()
    A.setdiag(0.0)
    A.eliminate_zeros()
    offdiag = np.asarray(np.abs(A).sum(axis=1)).ravel()
    L = sp.diags(offdiag + 1.0) - A      # row-strict diagonal dominance => SDDM
    return L.tocsr()


def test_laplacian_solve_converges():
    L = grid2d_laplacian(50)
    rng = np.random.default_rng(1)
    b = rng.standard_normal(L.shape[0])
    b -= b.mean()                        # consistent RHS for a singular Laplacian
    res = apxchol.solve(L, b, tol=1e-8, maxiter=500)
    assert res.converged
    assert res.residual <= 1e-8
    # x recovered up to an additive constant (null space = constants)
    Lx = L @ res.x
    assert np.linalg.norm(Lx - b) / np.linalg.norm(b) <= 1e-6


def test_sddm_solve_converges():
    L = random_sddm(400, seed=2)
    rng = np.random.default_rng(3)
    b = rng.standard_normal(L.shape[0])
    res = apxchol.solve(L, b, tol=1e-8, maxiter=500)
    assert res.converged
    assert res.residual <= 1e-8
    assert np.linalg.norm(L @ res.x - b) / np.linalg.norm(b) <= 1e-7


def test_factor_reused_across_many_b():
    L = grid2d_laplacian(40)
    solver = apxchol.solver(L)           # factor built ONCE
    rng = np.random.default_rng(4)
    for _ in range(5):
        b = rng.standard_normal(L.shape[0]); b -= b.mean()
        res = solver.solve(b, tol=1e-8, maxiter=500)
        assert res.converged
        assert res.residual <= 1e-8


def test_aslinearoperator_helps_scipy_cg():
    L = grid2d_laplacian(40)
    rng = np.random.default_rng(5)
    b = rng.standard_normal(L.shape[0]); b -= b.mean()
    M = apxchol.solver(L).aspreconditioner()
    _, info_pre = cg(L, b, M=M, rtol=1e-6, maxiter=2000)
    _, info_un = cg(L, b, rtol=1e-6, maxiter=50)     # unpreconditioned, capped low
    # Preconditioned converges (info==0); the capped unpreconditioned run does not.
    assert info_pre == 0
    assert info_un != 0


def test_apply_shape_and_finiteness():
    L = grid2d_laplacian(20)
    solver = apxchol.solver(L)
    r = np.random.default_rng(6).standard_normal(L.shape[0])
    z = solver.apply(r)
    assert z.shape == (L.shape[0],)
    assert np.all(np.isfinite(z))
    assert np.linalg.norm(z) > 0.0     # non-trivial preconditioner action


def test_non_square_raises():
    A = sp.csr_matrix(np.ones((3, 4)))
    with pytest.raises(ValueError):
        apxchol.solver(A)


def test_b_length_mismatch_raises():
    L = grid2d_laplacian(10)
    solver = apxchol.solver(L)
    with pytest.raises(ValueError):
        solver.solve(np.ones(L.shape[0] + 1))


def csc_with_duplicates(A):
    """Non-canonical CSC: every entry split into two duplicate halves."""
    A = sp.csc_matrix(A)
    indptr, indices, data = [0], [], []
    for j in range(A.shape[1]):
        for k in range(A.indptr[j], A.indptr[j + 1]):
            indices += [A.indices[k], A.indices[k]]
            data += [A.data[k] / 2.0, A.data[k] / 2.0]
        indptr.append(len(indices))
    B = sp.csc_matrix((np.array(data), np.array(indices), np.array(indptr)),
                      shape=A.shape)
    assert not B.has_canonical_format
    return B


def test_input_matrix_is_not_mutated():
    """Canonicalization must happen on a copy, never on the caller's matrix."""
    A = csc_with_duplicates(grid2d_sddm(10))
    nnz, data, indices, indptr = (A.nnz, A.data.copy(), A.indices.copy(),
                                  A.indptr.copy())
    slv = apxchol.factorize(A)
    assert A.nnz == nnz
    assert np.array_equal(A.data, data)
    assert np.array_equal(A.indices, indices)
    assert np.array_equal(A.indptr, indptr)
    # the duplicates were still consolidated for the factorization itself
    b = np.random.default_rng(17).standard_normal(A.shape[0])
    res = slv.solve(b, rtol=1e-8)
    assert res.converged
    assert np.linalg.norm(A @ res.x - b) / np.linalg.norm(b) < 1e-6


# ── factor export ────────────────────────────────────────────────────────────

def test_factor_export_algebra_sddm():
    """G G^T reproduces the permuted matrix; L D L^T reproduces G G^T exactly."""
    A = grid2d_sddm(30)
    slv = apxchol.factorize(A, seed=7)
    assert slv.sddm
    p = np.argsort(slv.P)                 # original index of the k-th eliminated vertex
    A_perm = A.tocsr()[p][:, p]
    GGt = (slv.chol() @ slv.chol().T).tocsc()
    # G is an APPROXIMATE (sampled) factor — loose bound only.
    assert spnorm(A_perm - GGt) / spnorm(A_perm) < 0.5
    # L D L^T == G G^T is an exact algebraic identity (up to fp round-off).
    LDLt = (slv.L @ sp.diags(slv.D) @ slv.L.T).tocsc()
    assert spnorm(LDLt - GGt) / spnorm(GGt) < 1e-6


def test_factor_export_algebra_laplacian():
    """Same identities for a singular Laplacian, plus the placeholder column."""
    A = grid2d_laplacian(20)
    n = A.shape[0]
    slv = apxchol.factorize(A, seed=7)
    assert not slv.sddm
    G = slv.chol()
    assert G.shape == (n, n)
    assert slv.L.shape == (n, n)
    assert slv.D.shape == (n,)

    # One connected component => exactly one placeholder column, the last one
    # eliminated: a bare diagonal 1.0 standing in for the dropped rank.
    Gc = G.tocsc()
    single = np.flatnonzero(np.diff(Gc.indptr) == 1)
    assert single.tolist() == [n - 1]
    assert Gc[n - 1, n - 1] == 1.0
    assert slv.D[n - 1] == 1.0

    # The G G^T identity holds on the rank-(n−1) subspace: check the leading
    # block, which excludes the placeholder's row/column.
    p = np.argsort(slv.P)
    A_perm = A.tocsr()[p][:, p]
    GGt = (G @ G.T).tocsc()
    lead = slice(0, n - 1)
    assert (spnorm(A_perm[lead, lead] - GGt[lead, lead])
            / spnorm(A_perm[lead, lead])) < 0.5
    # L D L^T == G G^T stays exact in the singular case too.
    LDLt = (slv.L @ sp.diags(slv.D) @ slv.L.T).tocsc()
    assert spnorm(LDLt - GGt) / spnorm(GGt) < 1e-6


def test_placeholder_column_per_component():
    """A k-component Laplacian carries k bare-diagonal placeholder columns."""
    A = sp.block_diag((grid2d_laplacian(8), grid2d_laplacian(6)), format="csr")
    G = apxchol.factorize(A, seed=7).chol().tocsc()
    single = np.flatnonzero(np.diff(G.indptr) == 1)
    assert single.shape[0] == 2                       # one per component
    assert single[-1] == A.shape[0] - 1               # last eliminated overall
    assert np.allclose(G.diagonal()[single], 1.0)


def test_apply_matches_explicit_factor_solve():
    """apply(r) == P^T G^{-T} G^{-1} P r, assembled from the exported factor."""
    A = grid2d_sddm(20)
    slv = apxchol.factorize(A, seed=11)
    assert slv.sddm
    P = slv.P
    p = np.argsort(P)
    G = slv.chol().tocsr()
    r = np.random.default_rng(12).standard_normal(A.shape[0])

    y = r[p]                                            # to permuted space
    w = spsolve_triangular(G, y, lower=True)
    zp = spsolve_triangular(G.T.tocsr(), w, lower=False)
    z_ref = zp[P]                                       # back to original order

    z = slv.apply(r)
    # fp32 factor values in the SpTRSV + a different traversal order.
    assert np.allclose(z, z_ref, rtol=1e-4, atol=1e-4 * np.abs(z_ref).max())


def test_keep_factor_false_disables_export():
    A = grid2d_sddm(15)
    slv = apxchol.factorize(A, keep_factor=False)
    with pytest.raises(RuntimeError):
        slv.L
    with pytest.raises(RuntimeError):
        slv.chol()
    assert slv.factor_nnz > 0            # column pointers survive the release
    # solving still works with the factor values released
    b = np.random.default_rng(13).standard_normal(A.shape[0])
    assert slv.solve(b).converged


def test_fill_ratio_and_version():
    A = grid2d_sddm(20)
    slv = apxchol.factorize(A)
    assert slv.fill_ratio > 0.0
    assert slv.factor_nnz > A.shape[0]
    assert isinstance(apxchol.__version__, str) and apxchol.__version__
    assert "Solver" in repr(slv)


# ── options ──────────────────────────────────────────────────────────────────

def test_alternate_partitioner_converges():
    L = grid2d_laplacian(30)
    slv = apxchol.factorize(L, partitioner="luby", seed=3)
    rng = np.random.default_rng(14)
    b = rng.standard_normal(L.shape[0]); b -= b.mean()
    res = slv.solve(b, rtol=1e-8, maxiter=500)
    assert res.converged


def test_unknown_option_raises():
    A = grid2d_sddm(10)
    with pytest.raises(ValueError):
        apxchol.factorize(A, no_such_option=1)


def test_x0_warm_start_needs_no_iterations():
    A = grid2d_sddm(20)
    slv = apxchol.factorize(A)
    b = np.random.default_rng(15).standard_normal(A.shape[0])
    res = slv.solve(b, rtol=1e-10)
    assert res.converged
    warm = slv.solve(b, rtol=1e-8, x0=res.x)
    assert warm.iters == 0
    assert warm.converged


def test_rtol_takes_precedence_over_tol():
    L = grid2d_laplacian(20)
    slv = apxchol.factorize(L)
    rng = np.random.default_rng(16)
    b = rng.standard_normal(L.shape[0]); b -= b.mean()
    loose = slv.solve(b, tol=1e-12, rtol=1e-3)
    tight = slv.solve(b, rtol=1e-10)
    assert loose.iters < tight.iters
    assert loose.residual <= 1e-3


def test_solve_into_out_buffer():
    A = grid2d_sddm(12)
    n = A.shape[0]
    b = np.ones(n)
    s = apxchol.factorize(A, keep_factor=False)
    x = np.zeros(n)
    res = s.solve(b, out=x)
    assert res.x is x                      # solution written into caller memory
    assert res.converged
    assert np.linalg.norm(A @ x - b) / np.linalg.norm(b) < 1e-6

    with pytest.raises(ValueError):
        s.solve(b, out=np.zeros(3))
    with pytest.raises(ValueError):
        s.solve(b, out=np.zeros(n, dtype=np.float32))
    with pytest.raises(ValueError):
        ro = np.zeros(n)
        ro.setflags(write=False)
        s.solve(b, out=ro)
