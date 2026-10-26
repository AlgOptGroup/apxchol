import numpy as np
import pytest
import scipy.sparse as sp

import apxchol
from test_solve import grid2d_laplacian, grid2d_sddm


@pytest.fixture(params=[np.float16, np.float32])
def solver(request):
    A = grid2d_sddm(7)
    try:
        instance = apxchol.factorize(A, backend="cpu", factor_storage_dtype=request.param)
    except RuntimeError as exc:
        if request.param == np.float16 and str(exc) == "FP16 factor storage requires AVX/F16C or Apple ARM64":
            pytest.skip("CPU lacks FP16 conversion support")
        raise
    return A, instance


def rhs(A, width=9):
    return A @ np.random.default_rng(19).standard_normal((A.shape[0], width))


@pytest.mark.parametrize("layout", ["C", "F", "strided", "reversed", "broadcast"])
def test_matrix_solve_and_apply_reuse_scalar_contract(solver, layout):
    A, instance = solver
    B = rhs(A)
    B[:, 1] = 0
    if layout == "F":
        B = np.asfortranarray(B)
    elif layout == "strided":
        B = np.repeat(np.repeat(B, 2, axis=0), 2, axis=1)[::2, ::2]
    elif layout == "reversed":
        B = B[::-1, ::-1]
    elif layout == "broadcast":
        B = np.broadcast_to(B[:, :1], B.shape)
    before = B.copy()
    result = instance.solve(B, rtol=1e-9)
    assert result.x.shape == B.shape
    assert result.x.dtype == np.float64
    assert result.iters.shape == result.residual.shape == result.converged.shape == (B.shape[1],)
    assert result.converged.dtype == np.bool_
    assert result.converged.all()
    norm = np.linalg.norm(B, axis=0)
    error = np.linalg.norm(A @ result.x - B, axis=0)
    assert (error <= 1e-9 * norm + 1e-12).all()
    expected = np.column_stack([instance.apply(column) for column in B.T])
    np.testing.assert_allclose(instance.apply(B), expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(instance.aslinearoperator().matmat(B), expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(B, before)
    assert instance.backend == "cpu"


@pytest.mark.parametrize("width", [0, 1])
def test_empty_and_single_rhs_matrices_preserve_rank(solver, width):
    A, instance = solver
    B = rhs(A, width)
    result = instance.solve(B)
    assert result.x.shape == B.shape
    assert result.iters.shape == result.residual.shape == result.converged.shape == (width,)
    assert result.converged.all()
    assert instance.apply(B).shape == B.shape


@pytest.mark.parametrize("order", ["C", "F"])
def test_matrix_warm_start_and_output(solver, order):
    A, instance = solver
    exact = np.random.default_rng(20).standard_normal((A.shape[0], 9))
    B = np.array(A @ exact, order=order)
    B[:, 1] = 0
    exact[:, 1] = 0
    guess = np.zeros_like(B)
    guess[:, ::2] = exact[:, ::2]
    out = np.empty(B.shape, order=order)
    result = instance.solve(B, x0=guess, out=out, rtol=1e-9)
    assert result.x is out
    assert result.converged.all()
    assert (result.iters[::2] == 0).all()
    assert result.iters[1] == 0
    warm = instance.solve(B, x0=result.x, rtol=1e-8)
    assert warm.converged.all()
    assert (warm.iters == 0).all()
    limited = instance.solve(B, x0=guess, maxiter=1, rtol=1e-14)
    assert (limited.iters <= 1).all()
    assert limited.converged[::2].all()
    assert not limited.converged[3::2].all()


def test_broadcast_warm_start(solver):
    A, instance = solver
    B = rhs(A)
    result = instance.solve(B, x0=np.broadcast_to(0.0, B.shape))
    assert result.converged.all()
    assert (np.linalg.norm(A @ result.x - B, axis=0) / np.linalg.norm(B, axis=0) <= 1e-8).all()


@pytest.mark.parametrize("aliased", ["b", "x0"])
def test_output_overlap_cannot_overwrite_later_panels(solver, aliased):
    A, instance = solver
    out = rhs(A)
    if aliased == "b":
        B, guess = out[:, ::-1], None
    else:
        B, guess = rhs(A), out[:, ::-1]
    original = B.copy()
    result = instance.solve(B, x0=guess, out=out, rtol=1e-9)
    assert result.x is out
    assert result.converged.all()
    np.testing.assert_allclose(A @ result.x, original, rtol=1e-7, atol=1e-8)


def test_disconnected_laplacian_matrix_rhs():
    A = sp.block_diag([grid2d_laplacian(4), grid2d_laplacian(3)], format="csc")
    instance = apxchol.factorize(A, backend="cpu")
    B = rhs(A, 5)
    B[:, 0] = 0
    result = instance.solve(B, rtol=1e-9)
    assert result.converged.all()
    np.testing.assert_allclose(A @ result.x, B, rtol=1e-7, atol=1e-8)
    expected = np.column_stack([instance.apply(column) for column in B.T])
    # An intervening solve must not change preconditioner application.
    instance.solve(B, rtol=1e-9)
    np.testing.assert_allclose(instance.apply(B), expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("value", [np.ones((48, 2)), np.ones((49, 2, 1)), 1.0,
                                   np.ones((49, 2), dtype=complex)])
def test_matrix_input_shape_and_complex_rejected(solver, value):
    _, instance = solver
    with pytest.raises(ValueError):
        instance.solve(value)
    with pytest.raises(ValueError):
        instance.apply(value)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_matrix_input_rejected(solver, value):
    A, instance = solver
    B = rhs(A, 2)
    B[-1, -1] = value
    with pytest.raises(ValueError, match="b must be finite"):
        instance.solve(B)
    with pytest.raises(ValueError, match="r must be finite"):
        instance.apply(B)
    with pytest.raises(ValueError, match="x0 must be finite"):
        instance.solve(rhs(A, 2), x0=B)


def test_x0_and_out_must_match_matrix_shape(solver):
    A, instance = solver
    B = rhs(A, 2)
    for shape in ((A.shape[0],), (A.shape[0], 1), (A.shape[0], 3)):
        with pytest.raises(ValueError):
            instance.solve(B, x0=np.zeros(shape))
        with pytest.raises(ValueError):
            instance.solve(B, out=np.zeros(shape))
    for dtype in (np.float32, np.complex128, np.int64, ">f8"):
        with pytest.raises(ValueError):
            instance.solve(B, out=np.zeros_like(B, dtype=dtype))
    out = np.zeros_like(B)
    out.setflags(write=False)
    with pytest.raises(ValueError):
        instance.solve(B, out=out)
    with pytest.raises(ValueError):
        instance.solve(B, out=np.zeros((A.shape[0], 4))[:, ::2])


def test_one_shot_matrix_solve_and_vector_result_types(solver):
    A, instance = solver
    B = rhs(A, 2).astype(np.float32)
    result = apxchol.solve(A, B, backend="cpu")
    assert result.x.shape == B.shape and result.converged.all()
    vector = instance.solve(B[:, 0])
    assert isinstance(vector.iters, int)
    assert isinstance(vector.residual, float)
    assert isinstance(vector.converged, bool)
    assert vector.x.shape == (A.shape[0],)


@pytest.mark.parametrize("warm", [False, True])
def test_matrix_rhs_with_mixed_extreme_scales(solver, warm):
    A, instance = solver
    scales = np.array([1.0, 1e-200, 1e200, 1e-160, 1e160, 0.0, -1.0, -1e-200, -1e200])
    exact = np.random.default_rng(42).standard_normal((A.shape[0], len(scales)))
    B = (A @ exact) * scales
    result = instance.solve(B, x0=0.25 * exact * scales if warm else None, rtol=1e-9)
    assert result.converged.all()
    assert np.isfinite(result.x).all()
    nonzero = scales != 0
    normalized_error = (A @ result.x - B)[:, nonzero] / scales[nonzero]
    normalized_rhs = B[:, nonzero] / scales[nonzero]
    assert (np.linalg.norm(normalized_error, axis=0) / np.linalg.norm(normalized_rhs, axis=0) < 1e-9).all()
    assert (result.x[:, ~nonzero] == 0).all()
