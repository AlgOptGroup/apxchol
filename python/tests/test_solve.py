import inspect
import platform
import warnings
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import cg, norm as spnorm, spsolve_triangular
import pytest

import apxchol


def test_graph_storage_is_not_a_public_option():
    assert "storage" not in inspect.signature(apxchol.factorize).parameters
    assert "storage" not in inspect.signature(apxchol.Solver).parameters


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


def test_default_graph_storage_solves_original_system():
    L = grid2d_sddm(5)
    b = np.linspace(-1.0, 1.0, L.shape[0])
    result = apxchol.solve(L, b, rtol=1e-8, maxiter=200)
    assert result.converged
    assert np.linalg.norm(L @ result.x - b) / np.linalg.norm(b) <= 1e-7


def test_retired_storage_is_rejected():
    with pytest.raises(ValueError, match="unknown option 'storage'"):
        apxchol.factorize(grid2d_sddm(3), storage="vec_pool_aos")


def test_laplacian_solve_converges():
    L = grid2d_laplacian(50)
    rng = np.random.default_rng(1)
    b = rng.standard_normal(L.shape[0])
    b -= b.mean()                        # consistent RHS for a singular Laplacian
    res = apxchol.solve(L, b, rtol=1e-8, maxiter=500)
    assert res.converged
    assert res.residual <= 1e-8
    # x recovered up to an additive constant (null space = constants)
    Lx = L @ res.x
    assert np.linalg.norm(Lx - b) / np.linalg.norm(b) <= 1e-6


def test_sddm_solve_converges():
    L = random_sddm(400, seed=2)
    rng = np.random.default_rng(3)
    b = rng.standard_normal(L.shape[0])
    res = apxchol.solve(L, b, rtol=1e-8, maxiter=500)
    assert res.converged
    assert res.residual <= 1e-8
    assert np.linalg.norm(L @ res.x - b) / np.linalg.norm(b) <= 1e-7


@pytest.mark.parametrize("dtype", [np.float16, np.float32])
def test_storage_dtype_preserves_original_system_accuracy(dtype):
    A = random_sddm(80, seed=9)
    b = np.random.default_rng(10).standard_normal(A.shape[0])
    try:
        solver = apxchol.factorize(A, backend="cpu", factor_storage_dtype=dtype)
    except RuntimeError as exc:
        assert dtype == np.float16 and str(exc) == "FP16 factor storage requires AVX/F16C or Apple ARM64"
        assert (platform.system(), platform.machine()) != ("Darwin", "arm64"), str(exc)
        pytest.skip("CPU lacks FP16 conversion support")
    result = solver.solve(b)
    assert result.converged
    assert np.linalg.norm(A @ result.x - b) / np.linalg.norm(b) < 1e-8
    assert solver.factor_storage_dtype == dtype
    assert solver.chol().dtype == np.float32


def test_factor_reused_across_many_b():
    L = grid2d_laplacian(40)
    solver = apxchol.factorize(L)        # factor built ONCE
    rng = np.random.default_rng(4)
    for _ in range(5):
        b = rng.standard_normal(L.shape[0]); b -= b.mean()
        res = solver.solve(b, rtol=1e-8, maxiter=500)
        assert res.converged
        assert res.residual <= 1e-8


def test_threads_sharing_one_solver_match_serial_calls():
    from concurrent.futures import ThreadPoolExecutor
    L = grid2d_laplacian(60)
    solver = apxchol.factorize(L, backend="cpu", num_threads=1)
    rhs = [np.sin(np.arange(L.shape[0]) * (k + 1)) for k in range(4)]
    rhs = [b - b.mean() for b in rhs]
    batch = np.column_stack(rhs[:2])
    expected = [(solver.solve(b).x, solver.apply(b)) for b in rhs]
    expected_batch = solver.solve(batch).x

    def work(k):
        return solver.solve(rhs[k]).x, solver.apply(rhs[k]), solver.solve(batch).x

    with ThreadPoolExecutor(max_workers=4) as pool:
        for _ in range(3):
            for k, (x, z, xb) in enumerate(pool.map(work, range(4))):
                np.testing.assert_array_equal(x, expected[k][0])
                np.testing.assert_array_equal(z, expected[k][1])
                np.testing.assert_array_equal(xb, expected_batch)


def test_aslinearoperator_helps_scipy_cg():
    L = grid2d_laplacian(40)
    rng = np.random.default_rng(5)
    b = rng.standard_normal(L.shape[0]); b -= b.mean()
    M = apxchol.factorize(L).aslinearoperator()
    _, info_pre = cg(L, b, M=M, rtol=1e-6, maxiter=2000)
    _, info_un = cg(L, b, rtol=1e-6, maxiter=50)     # unpreconditioned, capped low
    # Preconditioned converges (info==0); the capped unpreconditioned run does not.
    assert info_pre == 0
    assert info_un != 0


def test_apply_shape_and_finiteness():
    L = grid2d_laplacian(20)
    solver = apxchol.factorize(L)
    r = np.random.default_rng(6).standard_normal(L.shape[0])
    z = solver.apply(r)
    assert z.shape == (L.shape[0],)
    assert np.all(np.isfinite(z))
    assert np.linalg.norm(z) > 0.0     # non-trivial preconditioner action


def test_apply_vector_owns_result_after_solver_destruction():
    slv = apxchol.factorize(grid2d_sddm(7), backend="cpu")
    r = np.random.default_rng(6).standard_normal(slv.shape[0])
    z = slv.apply(r)
    expected = z.copy()
    slv.apply(-r)
    r.fill(0)
    del slv
    assert z.dtype == np.float64 and z.flags.writeable
    np.testing.assert_array_equal(z, expected)
    z[0] += 1
    expected[0] += 1
    np.testing.assert_array_equal(z, expected)


def test_non_square_raises():
    A = sp.csr_matrix(np.ones((3, 4)))
    with pytest.raises(ValueError):
        apxchol.factorize(A)


def test_b_length_mismatch_raises():
    L = grid2d_laplacian(10)
    solver = apxchol.factorize(L)
    with pytest.raises(ValueError):
        solver.solve(np.ones(L.shape[0] + 1))


@pytest.mark.parametrize("entry", ["Solver", "factorize", "solve"])
@pytest.mark.parametrize("imaginary", [0.0, 0.5])
def test_complex_operator_is_rejected_before_cast(entry, imaginary):
    # For nonzero imaginary parts, discarding them changes this Hermitian system.
    A = sp.csr_matrix([[2.0, -1.0 + imaginary * 1j],
                       [-1.0 - imaginary * 1j, 2.0]])
    call = getattr(apxchol, entry)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match="A must be real"):
            if entry == "solve":
                call(A, np.array([1.0, 2.0]))
            else:
                call(A)


@pytest.mark.parametrize("route, argument", [
    ("one_shot", "b"), ("reusable", "b"), ("reusable", "x0"),
])
@pytest.mark.parametrize("imaginary", [0.0, 0.5])
def test_complex_solve_input_is_rejected_before_cast(route, argument, imaginary):
    A = sp.csr_matrix([[2.0, -1.0], [-1.0, 2.0]])
    vector = np.array([1.0 + imaginary * 1j, 2.0 - imaginary * 1j])
    b = vector if argument == "b" else A @ vector.real
    kwargs = {"x0": vector} if argument == "x0" else {}
    slv = apxchol.factorize(A) if route == "reusable" else None
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match=f"{argument} must be real"):
            if route == "one_shot":
                apxchol.solve(A, b)
            else:
                # With x0, stripping its imaginary part would yield an exact guess.
                slv.solve(b, **kwargs)


@pytest.mark.parametrize("entry", ["apply", "aslinearoperator"])
@pytest.mark.parametrize("imaginary", [0.0, 0.5])
def test_complex_apply_input_is_rejected_before_cast(entry, imaginary):
    slv = apxchol.factorize(sp.csr_matrix([[2.0, -1.0], [-1.0, 2.0]]))
    r = np.array([1.0 + imaginary * 1j, 2.0 - imaginary * 1j])
    apply = slv.apply if entry == "apply" else getattr(slv, entry)().matvec
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match="r must be real"):
            apply(r)


@pytest.mark.parametrize("dtype", [np.int64, np.float32, np.float64])
def test_supported_real_dtypes_preserve_solve_and_apply(dtype):
    A = sp.csr_matrix([[4, -1], [-1, 3]], dtype=dtype)
    b = np.array([1, 2], dtype=dtype)
    x0 = np.array([0, 1], dtype=dtype)
    slv = apxchol.factorize(A)
    for result in (slv.solve(b, x0=x0, rtol=1e-10),
                   apxchol.solve(A, b, rtol=1e-10)):
        assert result.converged
        assert result.x.dtype == np.float64
        assert np.linalg.norm(A @ result.x - b) / np.linalg.norm(b) <= 1e-9

    expected = slv.apply(b.astype(np.float64))
    for result in (slv.apply(b), slv.aslinearoperator() @ b):
        assert result.dtype == np.float64
        assert np.all(np.isfinite(result))
        np.testing.assert_allclose(result, expected, rtol=1e-12, atol=0.0)


# ── operator contract: the class is asserted, not sniffed at ─────────────────


def grid2d_adjacency(m):
    """2-D m×m grid ADJACENCY matrix (no diagonal, positive off-diagonals)."""
    L = grid2d_laplacian(m)
    return (sp.diags(L.diagonal()) - L).tocsr()


def test_adjacency_input_raises_with_a_diagnosis():
    """An adjacency matrix used to factor to zero fill and report

    `converged=False` with no explanation. It is now refused by the
    positive-diagonal condition of the operator contract -- a hard PSD
    violation -- with the explicit conversion named."""
    A = grid2d_adjacency(10)
    with pytest.raises(ValueError) as exc:
        apxchol.factorize(A)
    msg = str(exc.value)
    # (a) which condition failed, with counts
    assert "Condition FAILED: positive diagonal" in msg
    assert f"{A.shape[0]} of {A.shape[0]} non-empty rows" in msg
    # (b) the class apxchol is defined on
    assert "apxchol solves symmetric SDDM / Laplacian operators" in msg
    # (c) the one-line fix, naming the helper
    assert "ADJACENCY matrix" in msg
    assert "apxchol.as_operator(M)" in msg


def test_every_public_entry_rejects_an_adjacency_matrix():
    A = grid2d_adjacency(8)
    b = np.ones(A.shape[0])
    for call in (lambda: apxchol.factorize(A),
                 lambda: apxchol.Solver(A),
                 lambda: apxchol.solve(A, b)):
        with pytest.raises(ValueError, match="Condition FAILED: positive diagonal"):
            call()


def test_a_row_without_a_positive_diagonal_is_refused_as_such():
    """The diagonal condition is about PSD, not about spotting adjacency

    matrices: one bad row is enough, and the message must NOT then suggest a
    conversion that would not help."""
    A = grid2d_sddm(10).tolil()
    A[3, 3] = -1.0
    with pytest.raises(ValueError) as exc:
        apxchol.factorize(A.tocsc())
    msg = str(exc.value)
    assert "Condition FAILED: positive diagonal" in msg
    assert "first at row 3" in msg
    assert "ADJACENCY" not in msg


def test_positive_offdiagonals_on_a_positive_diagonal_are_lumped_not_rejected():
    """A nearly-SDDM SPD operator (mixed-sign FEM/structural matrices look

    like this) is repaired by M-matrix lumping, not refused -- and the
    preconditioner-only transform leaves the caller's matrix alone."""
    A = grid2d_sddm(10).tolil()
    A[0, 3] = +0.25              # a positive off-diagonal on a positive diagonal
    A[3, 0] = +0.25
    A = A.tocsc()
    before = A.copy()
    assert (A.diagonal() > 0).all()

    s = apxchol.factorize(A)
    assert s.lumped == 2         # both triangles of the one pair
    # the caller's matrix is untouched
    assert (A != before).nnz == 0

    b = np.ones(A.shape[0])
    res = s.solve(b, rtol=1e-8, maxiter=500)
    assert res.converged
    # ... and the residual is against the TRUE operator, not the lumped one
    assert np.linalg.norm(b - A @ res.x) / np.linalg.norm(b) < 1e-7


def test_a_pure_laplacian_reports_no_lumping():
    s = apxchol.factorize(grid2d_laplacian(10))
    assert s.lumped == 0


def test_mixed_sign_operator_with_positive_diagonal_still_solves():
    """The class the narrow rule protects: positive diagonal, mixed off-signs."""
    A = grid2d_sddm(20).tolil()
    for i in range(0, 100, 7):   # sprinkle positive off-diagonals
        A[i, i + 1] = +0.1
        A[i + 1, i] = +0.1
    A = A.tocsc()
    rng = np.random.default_rng(11)
    b = rng.standard_normal(A.shape[0])
    res = apxchol.solve(A, b, rtol=1e-8, maxiter=500)
    assert res.converged
    assert np.linalg.norm(A @ res.x - b) / np.linalg.norm(b) < 1e-6


# ── apxchol.as_operator: the rule shared with the CLI's --convert-input ─────


def grid2d_sddm_matrix(m):
    return (grid2d_laplacian(m) + 0.1 * sp.eye(m * m)).tocsc()


def degree_minus(A):
    """D - A with D the row sums of A, self-loops included: the textbook form."""
    return sp.diags(np.asarray(A.sum(axis=1)).ravel()) - A


@pytest.mark.parametrize("make", [grid2d_laplacian, grid2d_sddm_matrix])
def test_as_operator_returns_a_valid_operator_unchanged(make):
    L = make(8)
    assert apxchol.as_operator(L) is L


def test_as_operator_converts_an_adjacency_matrix():
    A = grid2d_adjacency(12)
    L = apxchol.as_operator(A)
    assert sp.issparse(L) and L.format == "csc" and L.dtype == np.float64
    assert abs(L - degree_minus(A)).max() == 0.0


def test_as_operator_round_trips_an_adjacency_matrix_into_a_solve():
    L = apxchol.as_operator(grid2d_adjacency(30))
    rng = np.random.default_rng(23)
    b = rng.standard_normal(L.shape[0]); b -= b.mean()
    res = apxchol.solve(L, b, rtol=1e-8, maxiter=500)
    assert res.converged
    assert np.linalg.norm(L @ res.x - b) / np.linalg.norm(b) <= 1e-6


def test_as_operator_cancels_self_loops():
    A = grid2d_adjacency(6).tolil()
    A[0, 0] = 5.0
    A[4, 4] = 2.0
    A = A.tocsc()
    L = apxchol.as_operator(A)
    assert abs(L - degree_minus(A)).max() == 0.0
    assert L[0, 0] == pytest.approx(2.0)   # degree of a grid corner, not 2 + 5
    assert abs(L.sum(axis=1)).max() == pytest.approx(0.0)


def test_as_operator_converts_a_boolean_pattern():
    A = grid2d_adjacency(6).tolil()
    A[0, 0] = 1.0                          # pattern files may carry self-loops
    A = A.tocsc()
    assert abs(apxchol.as_operator(A.astype(bool)) - degree_minus(A)).max() == 0.0


def test_as_operator_raises_for_mixed_signs():
    M = sp.csc_matrix(np.array([[0.0, 1.0, 0.0], [1.0, 0.0, -1.0], [0.0, -1.0, 0.0]]))
    with pytest.raises(ValueError, match="not a graph adjacency matrix either"):
        apxchol.as_operator(M)
    negative = sp.csc_matrix(np.array([[0.0, -2.0], [-2.0, 0.0]]))
    with pytest.raises(ValueError, match="Condition FAILED: positive diagonal"):
        apxchol.as_operator(negative)


def test_as_operator_keeps_isolated_vertices_as_zero_rows():
    A = sp.csc_matrix((5, 5))              # no edges at all
    A = A.tolil(); A[1, 2] = 1.0; A[2, 1] = 1.0; A = A.tocsc()
    L = apxchol.as_operator(A)
    assert L.shape == (5, 5)
    assert np.array_equal(L.diagonal(), np.array([0.0, 1.0, 1.0, 0.0, 0.0]))
    assert L[0].nnz == 0


def test_as_operator_handles_explicit_zeros_and_duplicates():
    # explicit stored zero on an edge slot, plus an unmerged duplicate pair
    data = np.array([0.0, 1.5, 1.5, 0.0])
    rows = np.array([1, 2, 1, 2])
    cols = np.array([0, 1, 2, 1])
    A = sp.coo_matrix((data, (rows, cols)), shape=(3, 3)).tocsc()
    A = A + sp.coo_matrix((np.array([0.5, 0.5]), (np.array([1, 2]),
                                                  np.array([2, 1]))),
                          shape=(3, 3)).tocsc()
    L = apxchol.as_operator(A)
    # vertex 0 is isolated (its only entry was an explicit zero)
    assert L.diagonal()[0] == 0.0
    assert L.diagonal()[1] == pytest.approx(2.0)
    assert L[1, 2] == pytest.approx(-2.0)
    assert abs(L.sum(axis=1)).max() == pytest.approx(0.0)


def test_as_operator_accepts_a_weighted_symmetric_adjacency_in_any_format():
    rng = np.random.default_rng(5)
    M = sp.random(60, 60, density=0.08, random_state=rng, format="csr")
    A = (M + M.T).tocsr()
    A.setdiag(0.0); A.eliminate_zeros()
    ref = apxchol.as_operator(A)
    for fmt in ("csc", "coo", "lil", "dok", "csr"):
        L = apxchol.as_operator(A.asformat(fmt))
        assert abs(L - ref).max() == 0.0
    assert abs(ref.sum(axis=1)).max() == pytest.approx(0.0)


def test_as_operator_does_not_mutate_its_input():
    A = sp.csc_matrix(grid2d_adjacency(8))
    nnz, data, indices, indptr = (A.nnz, A.data.copy(), A.indices.copy(),
                                  A.indptr.copy())
    apxchol.as_operator(A)
    assert A.nnz == nnz
    assert np.array_equal(A.data, data)
    assert np.array_equal(A.indices, indices)
    assert np.array_equal(A.indptr, indptr)


def test_as_operator_rejects_bad_input():
    with pytest.raises(ValueError):
        apxchol.as_operator(np.ones((3, 3)))             # dense
    with pytest.raises(ValueError):
        apxchol.as_operator(sp.csc_matrix(np.ones((3, 4))))   # non-square
    with pytest.raises(ValueError):
        apxchol.as_operator(sp.csc_matrix(np.eye(3) * 1j))    # complex


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
    assert G.dtype == slv.L.dtype == slv.D.dtype == np.float32
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


def test_export_supports_float32_triangular_solve():
    G = apxchol.factorize(random_sddm(30, seed=5)).chol()
    exact = np.linspace(0.5, 1.5, G.shape[0], dtype=np.float32)
    result = spsolve_triangular(G, G @ exact, lower=True)
    assert result.dtype == np.float32
    np.testing.assert_allclose(result, exact, rtol=1e-6)


@pytest.mark.parametrize("field, values", [
    ("indptr", []), ("indptr", [0, 5, 4]), ("indptr", [1, 2, 4]),
    ("indptr", [0, -1, 4]), ("indptr", [0, 2, 5]),
    ("indices", [0, 2, 0, 1]), ("indices", [0, -1, 0, 1]),
    ("indices", [0, 1, 0]),
])
@pytest.mark.parametrize("dtype", [np.int32, np.int64])
def test_binding_rejects_invalid_csc_bounds(field, values, dtype):
    from apxchol import _apxchol

    A = sp.csc_matrix([[2.0, -1.0], [-1.0, 2.0]])
    setattr(A, field, np.asarray(values, dtype=dtype))
    with pytest.raises(ValueError):
        _apxchol.Solver(A.indptr, A.indices, A.data)


@pytest.mark.parametrize("offset_dtype", [np.int32, np.int64])
@pytest.mark.parametrize("index_dtype", [np.int32, np.int64])
@pytest.mark.parametrize("layout", ["sorted", "unsorted", "duplicates", "strided"])
def test_binding_imports_csc_layouts(offset_dtype, index_dtype, layout):
    from apxchol import _apxchol

    A = np.array([[4.0, -1.0], [-1.0, 3.0]])
    indptr, indices, data = [0, 2, 4], [0, 1, 0, 1], [4.0, -1.0, -1.0, 3.0]
    if layout == "unsorted":
        indices, data = [1, 0, 1, 0], [-1.0, 4.0, 3.0, -1.0]
    elif layout == "duplicates":
        indptr, indices = [0, 4, 8], [0, 0, 1, 1, 0, 0, 1, 1]
        data = [2.0, 2.0, -0.5, -0.5, -0.5, -0.5, 1.5, 1.5]
    inputs = (np.array(indptr, dtype=offset_dtype),
              np.array(indices, dtype=index_dtype), np.array(data))
    if layout == "strided":
        inputs = tuple(np.repeat(value, 2)[::2] for value in inputs)
    before = tuple(value.copy() for value in inputs)
    solver = _apxchol.Solver(*inputs, {"keep_factor": True,
                                     "factor_storage_dtype": "float32"})
    b = np.array([1.0, 2.0])
    result = solver.solve(b, 1e-12, 20)
    np.testing.assert_allclose(A @ result["x"], b, rtol=1e-11, atol=1e-12)
    assert result["converged"]
    assert solver.factor_storage_dtype() == "float32"
    assert solver.factor_csc()[2].dtype == np.float32
    for actual, original in zip(inputs, before):
        np.testing.assert_array_equal(actual, original)


@pytest.mark.parametrize("value", [-1.0, 0.0, np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("batched", [False, True])
def test_invalid_tolerance_rejected(value, batched):
    A = sp.csc_matrix([[4.0, -1.0], [-1.0, 3.0]])
    solver = apxchol.factorize(A)
    b = np.ones((2, 2)) if batched else np.ones(2)
    with pytest.raises(ValueError, match="tol"):
        solver.solve(b, rtol=value)


@pytest.mark.parametrize("value", [1.5, np.float64(2), "2"])
def test_iteration_limit_requires_integer(value):
    solver = apxchol.factorize(sp.eye(2, format="csc"))
    with pytest.raises(TypeError):
        solver.solve(np.ones(2), maxiter=value)


@pytest.mark.parametrize("name", ["seed", "num_threads", "exact_clique_max_degree"])
def test_integer_setup_options_do_not_truncate(name):
    with pytest.raises(TypeError):
        apxchol.factorize(sp.eye(2, format="csc"), **{name: 1.5})


def test_omitted_solver_defaults_match_compiled_defaults():
    from apxchol import _apxchol

    solver = apxchol.factorize(grid2d_sddm(8), num_threads=1)
    b = np.ones(solver.shape[0])
    implicit = solver.solve(b)
    explicit = solver.solve(b, rtol=_apxchol.DEFAULT_TOL, maxiter=np.int64(_apxchol.DEFAULT_MAXITER))
    np.testing.assert_array_equal(implicit.x, explicit.x)
    assert (implicit.iters, implicit.residual, implicit.converged) == (explicit.iters, explicit.residual, explicit.converged)


@pytest.mark.parametrize("construct", [apxchol.Solver, apxchol.factorize])
@pytest.mark.parametrize("options", [{}, {"seed": np.int64(7), "partitioner": "priority_greedy"}])
def test_factor_settings_match_native_options(construct, options):
    A = grid2d_sddm(8).tocsc()
    solver = construct(A, backend="cpu", num_threads=1, **options)
    native = apxchol._apxchol.Solver(A.indptr, A.indices, A.data,
                                   dict(backend="cpu", num_threads=1, keep_factor=True, **options))
    G = solver.chol()
    np.testing.assert_array_equal(solver.P, native.perm())
    for actual, expected in zip((G.indptr, G.indices, G.data), native.factor_csc()):
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("cpu_schedule", ["auto", "levels"])
@pytest.mark.parametrize("drop_tolerance", [0, 1])
def test_explicit_factor_settings(cpu_schedule, drop_tolerance):
    A = grid2d_sddm(6)
    solver = apxchol.factorize(A, backend="cpu", num_threads=1, cpu_schedule=cpu_schedule,
                               drop_tolerance=drop_tolerance, regularization=0, lump_max_mass=0.25,
                               verbose=False)
    b = np.ones(A.shape[0])
    result = solver.solve(b)
    assert result.converged
    assert np.linalg.norm(A @ result.x - b) / np.linalg.norm(b) < 1e-8


@pytest.mark.parametrize("option,value", [("cpu_schedule", "unknown"),
                                         ("regularization", -1), ("lump_max_mass", -0.1),
                                         ("lump_max_mass", 1.1)] +
                         [("drop_tolerance", value) for value in
                          [-1, np.nextafter(1.0, np.inf), 2, np.nan, np.inf, -np.inf]])
def test_invalid_factor_settings_rejected(option, value):
    with pytest.raises(ValueError):
        apxchol.factorize(sp.eye(2, format="csc"), **{option: value})


def test_factor_settings_are_per_instance():
    A = sp.csc_matrix([[1.0, -1.0], [-1.0, 1.0]])
    b = np.array([1.0, -1.0])
    plain = apxchol.factorize(A, backend="cpu", num_threads=1, regularization=0,
                             lump_max_mass=0, cpu_schedule="auto")
    before = plain.solve(b)
    regularized = apxchol.factorize(A, backend="cpu", num_threads=1,
                                   regularization=0.25, cpu_schedule="levels")
    assert not plain.sddm and regularized.sddm
    assert regularized.solve(b).converged
    after = plain.solve(b)
    np.testing.assert_array_equal(after.x, before.x)
    assert after.residual == before.residual


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
    slv = apxchol.factorize(A, seed=11, factor_storage_dtype=np.float32)
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
    slv = apxchol.factorize(L, partitioner="priority_greedy", seed=3)
    rng = np.random.default_rng(14)
    b = rng.standard_normal(L.shape[0]); b -= b.mean()
    res = slv.solve(b, rtol=1e-8, maxiter=500)
    assert res.converged


def test_unknown_option_raises():
    A = grid2d_sddm(10)
    with pytest.raises(ValueError):
        apxchol.factorize(A, no_such_option=1)


def test_removed_stagnation_option_is_rejected():
    with pytest.raises(ValueError, match="unknown option 'stagnation_window'"):
        apxchol.factorize(sp.eye(2, format="csc"), stagnation_window=0)


def test_x0_warm_start_needs_no_iterations():
    A = grid2d_sddm(20)
    slv = apxchol.factorize(A)
    b = np.random.default_rng(15).standard_normal(A.shape[0])
    res = slv.solve(b, rtol=1e-10)
    assert res.converged
    warm = slv.solve(b, rtol=1e-8, x0=res.x)
    assert warm.iters == 0
    assert warm.converged


def test_removed_tolerance_alias_is_rejected():
    A = grid2d_sddm(3)
    solver = apxchol.factorize(A)
    with pytest.raises(TypeError, match="tol"):
        solver.solve(np.ones(A.shape[0]), **{"tol": 1e-8})
    with pytest.raises(ValueError, match="unknown option 'tol'"):
        apxchol.solve(A, np.ones(A.shape[0]), **{"tol": 1e-8})


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


@pytest.mark.parametrize("num_threads", [0, 1, 2])
def test_thread_budget(num_threads):
    A = sp.diags([-np.ones(31), 3 * np.ones(32), -np.ones(31)], [-1, 0, 1], format="csc")
    b = np.linspace(1, 2, 32)
    slv = apxchol.factorize(A, backend="cpu", num_threads=num_threads)
    for _ in range(2):
        result = slv.solve(b)
        assert np.linalg.norm(A @ result.x - b) / np.linalg.norm(b) < 1e-8
        assert np.isfinite(slv.apply(b)).all()
        B = np.column_stack([b, -b])
        result = slv.solve(B)
        assert np.linalg.norm(A @ result.x - B) / np.linalg.norm(B) < 1e-8
        assert np.isfinite(slv.apply(B)).all()


def test_negative_thread_budget():
    with pytest.raises(ValueError, match="num_threads must be nonnegative"):
        apxchol.factorize(sp.eye(4, format="csc"), num_threads=-1)


def test_archive_restores_scipy_helpers(tmp_path):
    A = grid2d_sddm(5)
    original = apxchol.factorize(A, backend="cpu", seed=2)
    path = tmp_path / "scipy.apxchol"
    original.save(path)
    restored = apxchol.Solver.load(path)
    original_factor = original.chol()
    factor = restored.chol()
    np.testing.assert_array_equal(restored.P, original.P)
    np.testing.assert_array_equal(factor.indptr, original_factor.indptr)
    np.testing.assert_array_equal(factor.indices, original_factor.indices)
    np.testing.assert_array_equal(factor.data, original_factor.data)
    rhs = np.arange(A.shape[0], dtype=float)
    np.testing.assert_allclose(restored.aslinearoperator() @ rhs, original.apply(rhs))
    result = restored.solve(rhs)
    assert result.converged
    assert np.linalg.norm(A @ result.x - rhs) / np.linalg.norm(rhs) <= 1e-8
