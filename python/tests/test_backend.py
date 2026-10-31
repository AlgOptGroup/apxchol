import os
import subprocess
import sys

import numpy as np
import pytest
import scipy.sparse as sp

import apxchol


@pytest.fixture(scope="module")
def operator_matrix():
    return sp.diags([-np.ones(64), np.full(65, 4.0), -np.ones(64)],
                    [-1, 0, 1], format="csc")


@pytest.fixture(scope="module")
def auto_solver(operator_matrix):
    return apxchol.factorize(operator_matrix, factor_storage=np.float32)


@pytest.fixture(scope="module")
def gpu_solver(operator_matrix):
    try:
        solver = apxchol.factorize(operator_matrix, backend="gpu",
                                   factor_storage=np.float32)
    except (ValueError, RuntimeError) as exc:
        unavailable = (
            "GPU route requested but GPU support is not built",
            "GPU factorization requires an available CUDA device",
            "apxchol: the Metal backend is unavailable: no Metal device",
            "GPU route requires 32-bit nodes, block_greedy, vec_pool_aos, "
            "natural residual peeling and supported sampler options; request CPU explicitly",
        )
        if (os.environ.get("APXCHOL_TEST_REQUIRE_GPU") == "1"
                or str(exc) not in unavailable):
            raise
        return None
    assert solver.backend == "gpu"
    return solver


def test_backend_selection_is_read_only(operator_matrix, auto_solver, gpu_solver):
    cpu = apxchol.factorize(operator_matrix, backend="cpu")
    assert cpu.backend == "cpu"
    assert auto_solver.backend in ("cpu", "gpu")
    with pytest.raises(AttributeError):
        auto_solver.backend = "cpu"


def test_scalar_automatic_solver_uses_cpu():
    matrix = sp.csc_matrix([[4.0]])
    solver = apxchol.factorize(matrix)
    assert solver.backend == "cpu"
    assert solver.factor_storage == np.float32
    rhs = np.ones(1)
    result = solver.solve(rhs)
    assert result.converged
    assert np.linalg.norm(matrix @ result.x - rhs) / np.linalg.norm(rhs) <= 1e-8


@pytest.mark.parametrize("backend", ["invalid", "CUDA", None])
def test_invalid_backend_is_rejected(operator_matrix, backend):
    with pytest.raises(ValueError, match="backend"):
        apxchol.factorize(operator_matrix, backend=backend)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_vectors_are_rejected(auto_solver, value):
    vector = np.ones(auto_solver.shape[0])
    vector[0] = value
    with pytest.raises(ValueError, match="b must be finite"):
        auto_solver.solve(vector)
    with pytest.raises(ValueError, match="x0 must be finite"):
        auto_solver.solve(np.ones_like(vector), x0=vector)
    with pytest.raises(ValueError, match="r must be finite"):
        auto_solver.apply(vector)


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
@pytest.mark.parametrize("dtype", [np.float16, np.float32])
def test_backend_reuses_factor_and_supports_export_warm_start_and_output(
        operator_matrix, gpu_solver, backend, dtype):
    if backend == "gpu" and gpu_solver is None:
        pytest.skip("No usable GPU backend in this installation")
    try:
        solver = apxchol.factorize(operator_matrix, backend=backend,
                                  factor_storage=dtype)
    except RuntimeError as exc:
        if (backend == "cpu" and dtype == np.float16
                and str(exc) == "FP16 factor storage requires AVX/F16C or Apple ARM64"):
            pytest.skip("CPU lacks FP16 conversion support")
        raise
    assert solver.backend == backend
    assert solver.factor_storage == dtype
    factor = solver.chol()
    assert factor.dtype == np.float64
    assert factor.nnz == solver.factor_nnz
    np.testing.assert_array_equal(np.sort(solver.P), np.arange(operator_matrix.shape[0]))

    exact = np.linspace(-1.0, 1.0, operator_matrix.shape[0])
    rhs = operator_matrix @ exact
    out = np.empty_like(rhs)
    result = solver.solve(rhs, rtol=1e-8, out=out, x0=np.zeros_like(rhs))
    assert result.x is out
    assert result.converged
    assert np.linalg.norm(operator_matrix @ result.x - rhs) / np.linalg.norm(rhs) <= 1e-8

    warm = solver.solve(rhs, rtol=1e-8, x0=exact)
    assert warm.converged and warm.iterations == 0
    np.testing.assert_allclose(operator_matrix @ warm.x, rhs, rtol=1e-12, atol=1e-12)
    rhs2 = operator_matrix @ np.cos(np.arange(rhs.size))
    reused = solver.solve(rhs2, rtol=1e-8)
    assert reused.converged
    assert np.linalg.norm(operator_matrix @ reused.x - rhs2) / np.linalg.norm(rhs2) <= 1e-8
    assert solver.backend == backend
    assert np.isfinite(solver.apply(rhs)).all()


@pytest.mark.skipif(sys.platform != "linux", reason="CUDA device visibility is a Linux-wheel test")
def test_hidden_cuda_device_selects_cpu_and_explicit_gpu_fails():
    script = """
import apxchol
import scipy.sparse as sp
# Exercise device availability above the automatic small-system guards.
A = sp.diags([-1., 4., -1.], [-1, 0, 1], shape=(32769, 32769), format='csc')
assert apxchol.factorize(A).backend == 'cpu'
try:
    apxchol.factorize(A, backend='gpu')
except (ValueError, RuntimeError):
    pass
else:
    raise AssertionError('explicit GPU succeeded with no visible CUDA device')
"""
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    subprocess.run([sys.executable, "-B", "-c", script], env=environment,
                   check=True, timeout=60)


@pytest.mark.parametrize("dtype", [np.float16, np.float32])
def test_gpu_matrix_rhs_preserves_backend(operator_matrix, gpu_solver, dtype):
    if gpu_solver is None:
        pytest.skip("No usable GPU backend in this installation")
    solver = apxchol.factorize(operator_matrix, backend="gpu", factor_storage=dtype)
    exact = np.random.default_rng(5).standard_normal((operator_matrix.shape[0], 3))
    exact[:, 0] = 0
    rhs = operator_matrix @ exact
    out = np.empty_like(rhs, order="F")
    result = solver.solve(rhs, x0=exact, out=out)
    assert result.x is out and result.converged.all()
    assert (result.iterations == 0).all()
    np.testing.assert_allclose(operator_matrix @ result.x, rhs, rtol=1e-12, atol=1e-12)
    cold = solver.solve(rhs)
    assert cold.converged.all()
    expected = np.column_stack([solver.apply(column) for column in rhs.T])
    np.testing.assert_allclose(solver.apply(rhs), expected, rtol=1e-12, atol=1e-12)
    assert solver.backend == "gpu"
