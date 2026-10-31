"""Core array operations require NumPy; SciPy is an optional adapter."""
import subprocess
import sys

import numpy as np
import pytest

import apxchol


A = np.array([[4., -1., 0.], [-1., 4., -1.], [0., -1., 4.]])
DATA = np.array([4., -1., -1., 4., -1., -1., 4.])
INDICES = np.array([0, 1, 0, 1, 2, 1, 2], dtype=np.int32)
INDPTR = np.array([0, 2, 5, 7], dtype=np.int32)


def check_solve_and_apply(solver, width):
    exact = np.arange(3 * width, dtype=float).reshape(3, width) + 1
    if width == 1:
        exact = exact[:, 0]
    rhs = A @ exact
    out = np.empty_like(rhs)
    result = solver.solve(rhs, out=out)
    assert result.x is out and np.all(result.converged)
    np.testing.assert_allclose(A @ result.x, rhs, rtol=1e-8, atol=1e-12)
    assert np.isfinite(solver.apply(rhs)).all()
    warm = solver.solve(rhs, x0=exact)
    assert np.all(warm.converged) and np.all(np.asarray(warm.iters) == 0)
    assert solver.shape == A.shape and solver.P.shape == (3,)


@pytest.mark.parametrize("width", [1, 4, 16])
@pytest.mark.parametrize("index_dtype", [np.int32, np.int64])
def test_numpy_csc_solve_apply_and_warm_start(width, index_dtype):
    solver = apxchol.Solver.from_csc(DATA, INDICES.astype(index_dtype),
                                    INDPTR.astype(index_dtype), backend="cpu")
    check_solve_and_apply(solver, width)


def test_unsorted_duplicates_use_shared_canonical_import():
    data, indices, indptr = [], [], [0]
    for column in range(3):
        for row in range(2, -1, -1):
            if A[row, column]:
                data.extend([A[row, column] / 2] * 2)
                indices.extend([row] * 2)
        indptr.append(len(data))
    raw = apxchol.Solver.from_csc(data, indices, indptr, backend="cpu", seed=1)
    canonical = apxchol.Solver.from_csc(DATA, INDICES, INDPTR, backend="cpu", seed=1)
    assert raw.factor_nnz == canonical.factor_nnz
    np.testing.assert_array_equal(raw.P, canonical.P)
    np.testing.assert_allclose(raw.solve(A @ np.ones(3)).x, np.ones(3))


def test_csc_input_is_copied_and_can_be_strided():
    data, indices, indptr = [np.repeat(a, 2)[::2] for a in (DATA, INDICES, INDPTR)]
    before = [a.copy() for a in (data, indices, indptr)]
    solver = apxchol.Solver.from_csc(data, indices, indptr, backend="cpu")
    for actual, expected in zip((data, indices, indptr), before):
        np.testing.assert_array_equal(actual, expected)
        actual.fill(0)
    np.testing.assert_allclose(solver.solve(A @ np.ones(3)).x, np.ones(3))


@pytest.mark.parametrize("which", ["indices", "indptr"])
@pytest.mark.parametrize("dtype", [np.float64, np.bool_])
def test_csc_rejects_noninteger_indices(which, dtype):
    arrays = dict(data=DATA, indices=INDICES, indptr=INDPTR)
    arrays[which] = arrays[which].astype(dtype)
    with pytest.raises(ValueError, match="integer dtype"):
        apxchol.Solver.from_csc(**arrays)


@pytest.mark.parametrize("data,indices,indptr", [
    (DATA, INDICES, [1, 2, 5, 7]),
    (DATA, INDICES, [0, 5, 2, 7]),
    (DATA, INDICES, [0, 2, 5, 8]),
    (DATA, [0, 1, 0, 1, 3, 1, 2], INDPTR),
    (DATA, [0, 1, 0, 1, -1, 1, 2], INDPTR),
    (DATA.reshape(1, -1), INDICES, INDPTR),
    (DATA.astype(complex), INDICES, INDPTR),
])
def test_invalid_csc_is_rejected(data, indices, indptr):
    with pytest.raises(ValueError):
        apxchol.Solver.from_csc(data, indices, indptr)


def test_scipy_is_optional_and_helpers_report_the_extra():
    probe = r'''
import importlib.abc
import sys
class BlockScipy(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "scipy" or fullname.startswith("scipy."):
            raise ModuleNotFoundError("blocked for this test", name="scipy")
sys.meta_path.insert(0, BlockScipy())
import numpy as np
import apxchol
solver = apxchol.Solver.from_csc([4.], [0], [0, 1], backend="cpu")
assert solver.solve([8.]).x[0] == 2
assert np.isfinite(solver.apply([1.])).all()
for operation in (lambda: apxchol.factorize(None), lambda: apxchol.as_operator(None),
                  solver.chol, solver.aslinearoperator):
    try:
        operation()
    except ImportError as exc:
        assert "apxchol[scipy]" in str(exc)
    else:
        raise AssertionError("SciPy-only helper succeeded without SciPy")
assert not any(n == "scipy" or n.startswith("scipy.") for n in sys.modules)
'''
    subprocess.run([sys.executable, "-I", "-B", "-c", probe], check=True, timeout=30)


def gpu_solver(storage):
    import os
    try:
        solver = apxchol.Solver.from_csc(DATA, INDICES, INDPTR, backend="gpu",
                                        factor_storage_dtype=storage)
    except (ValueError, RuntimeError) as exc:
        unavailable = (
            "GPU route requested but GPU support is not built",
            "GPU factorization requires an available CUDA device",
            "apxchol::metal_solver: the Metal backend is unavailable: no Metal device",
        )
        if os.environ.get("APXCHOL_TEST_REQUIRE_GPU") == "1" or str(exc) not in unavailable:
            raise
        pytest.skip("No usable GPU backend in this installation")
    assert solver.backend == "gpu"
    return solver


@pytest.mark.parametrize("storage", [np.float16, np.float32])
def test_numpy_gpu_solve_and_apply_when_available(storage):
    solver = gpu_solver(storage)
    for width in (1, 4, 16):
        check_solve_and_apply(solver, width)


@pytest.mark.skipif(sys.platform != "linux", reason="CUDA visibility is a Linux-wheel test")
def test_numpy_hidden_cuda_device_keeps_cpu_available():
    import os
    probe = r'''
import apxchol
import numpy as np
data = [4., -1., -1., 4., -1., -1., 4.]
indices, indptr = [0, 1, 0, 1, 2, 1, 2], [0, 2, 5, 7]
solver = apxchol.Solver.from_csc(data, indices, indptr)
assert solver.backend == "cpu"
np.testing.assert_allclose(solver.solve([3., 2., 3.]).x, np.ones(3))
try:
    apxchol.Solver.from_csc(data, indices, indptr, backend="gpu")
except (ValueError, RuntimeError):
    pass
else:
    raise AssertionError("explicit GPU succeeded without a visible CUDA device")
'''
    subprocess.run([sys.executable, "-I", "-B", "-c", probe], check=True, timeout=30,
                   env=dict(os.environ, CUDA_VISIBLE_DEVICES=""))


def test_missing_scipy_does_not_export_factor(monkeypatch):
    from unittest.mock import Mock
    solver = apxchol.Solver.__new__(apxchol.Solver)
    solver._impl = Mock()
    solver._impl.factor_csc.return_value = ([], [], [])
    solver._chol = None
    def missing_scipy():
        raise ImportError("Install apxchol[scipy] to use this operation.")
    monkeypatch.setattr(apxchol, "_scipy_sparse", missing_scipy)
    with pytest.raises(ImportError, match=r"apxchol\[scipy\]"):
        solver.chol()
    solver._impl.factor_csc.assert_not_called()
