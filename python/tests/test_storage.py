import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

import apxchol


def csc_arrays():
    # [[2, -1], [-1, 2]]
    return (np.array([2.0, -1.0, -1.0, 2.0]),
            np.array([0, 1, 0, 1]), np.array([0, 2, 4]))


@pytest.mark.parametrize("dtype", [np.float32, "float32", np.dtype("float32"), ">f4"])
@pytest.mark.parametrize("keep_factor", [False, True])
def test_explicit_fp32_overrides_environment(monkeypatch, dtype, keep_factor):
    monkeypatch.setenv("APXCHOL_SPTRSV_FP16", "1")
    solver = apxchol.factorize(csc_arrays(), factor_storage_dtype=dtype,
                              keep_factor=keep_factor)
    assert solver.factor_storage_dtype == np.dtype("float32")
    result = solver.solve([1.0, 0.0])
    assert result.converged
    np.testing.assert_allclose(result.x, [2 / 3, 1 / 3], rtol=1e-8)
    assert os.environ["APXCHOL_SPTRSV_FP16"] == "1"


@pytest.mark.parametrize("dtype", [np.float16, "float16", np.dtype("float16")])
def test_fp16_and_fp32_instances_coexist(monkeypatch, dtype):
    monkeypatch.setenv("APXCHOL_SPTRSV_FP16", "0")
    full = apxchol.factorize(csc_arrays(), factor_storage_dtype=np.float32)
    try:
        half = apxchol.factorize(csc_arrays(), factor_storage_dtype=dtype)
    except RuntimeError as exc:
        assert "FP16 factor storage requires an x86 CPU with AVX/F16C support" in str(exc)
        pytest.skip("CPU lacks FP16 conversion support")
    assert half.factor_storage_dtype == np.dtype("float16")
    assert os.environ["APXCHOL_SPTRSV_FP16"] == "0"
    monkeypatch.setenv("APXCHOL_SPTRSV_FP16", "1")
    assert full.factor_storage_dtype == np.dtype("float32")
    for solver in (half, full):
        result = solver.solve([1.0, 0.0])
        assert result.x.dtype == np.float64
        assert result.converged
        np.testing.assert_allclose(result.x, [2 / 3, 1 / 3], rtol=1e-8)


def test_default_storage_uses_environment(monkeypatch):
    monkeypatch.delenv("APXCHOL_SPTRSV_FP16", raising=False)
    assert apxchol.factorize(csc_arrays()).factor_storage_dtype == np.float32
    monkeypatch.setenv("APXCHOL_SPTRSV_FP16", "0")
    assert apxchol.factorize(csc_arrays()).factor_storage_dtype == np.float32
    monkeypatch.setenv("APXCHOL_SPTRSV_FP16", "1")
    try:
        solver = apxchol.factorize(csc_arrays())
    except RuntimeError as exc:
        assert "AVX/F16C support" in str(exc)
    else:
        assert solver.factor_storage_dtype == np.float16


@pytest.mark.parametrize("dtype", [np.float64, np.int16, np.complex64])
def test_unsupported_storage_dtype_is_rejected(dtype):
    with pytest.raises(ValueError, match="factor_storage_dtype must be float16 or float32"):
        apxchol.factorize(csc_arrays(), factor_storage_dtype=dtype)


@pytest.mark.parametrize("data, indices, indptr, message", [
    ([2], [0], [], "integer dtype|indptr length"),
    ([2], [0], [1, 1], "indptr must start"),
    ([2], [0], [0, 2], "indptr must start"),
    ([2, 2], [0, 1], [0, 3, 2], "indptr must start"),
    ([2, 2], [0, 1], [0, -1, 2], "indptr must start"),
    ([2], [], [0, 1], "integer dtype|indptr must start"),
    ([2, 2], [0], [0, 2], "indptr must start"),
    ([2], [-1], [0, 1], "row index out of range"),
    ([2], [1], [0, 1], "row index out of range"),
    ([2], [0.5], [0, 1], "integer dtype"),
    ([2], [0], [0.0, 1.0], "integer dtype"),
    ([[2]], [0], [0, 1], "one-dimensional"),
    ([2], [[0]], [0, 1], "one-dimensional"),
    ([2], [0], [[0, 1]], "one-dimensional"),
    ([2 + 0j], [0], [0, 1], "A must be real"),
    (["2"], [0], [0, 1], "real numeric dtype"),
])
def test_invalid_csc_is_rejected(data, indices, indptr, message):
    with pytest.raises(ValueError, match=message):
        apxchol.factorize((data, indices, indptr))


def test_unsorted_duplicate_csc_matches_canonical_input():
    arrays = (np.array([-1.0, 1.0, 1.0, 2.0, -1.0]),
              np.array([1, 0, 0, 1, 0]), np.array([0, 3, 5]))
    before = tuple(a.copy() for a in arrays)
    result = apxchol.solve(arrays, [1.0, 0.0], factor_storage_dtype="float32")
    reference = apxchol.solve(csc_arrays(), [1.0, 0.0], factor_storage_dtype="float32")
    np.testing.assert_array_equal(result.x, reference.x)
    for actual, expected in zip(arrays, before):
        np.testing.assert_array_equal(actual, expected)


def test_numpy_only_install():
    code = '''
import importlib.abc
import sys
class NoScipy(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "scipy" or fullname.startswith("scipy."):
            raise ModuleNotFoundError("SciPy blocked for test", name="scipy")
sys.meta_path.insert(0, NoScipy())
import numpy as np
import apxchol
A = ([2., -1., -1., 2.], [0, 1, 0, 1], [0, 2, 4])
s = apxchol.factorize(A, factor_storage_dtype=np.float32)
assert s.factor_storage_dtype == np.float32
np.testing.assert_allclose(s.solve([1., 0.]).x, [2/3, 1/3])
assert np.isfinite(s.apply([1., 0.])).all()
assert s.P.shape == (2,)
assert "scipy" not in sys.modules
for operation in (s.chol, s.aslinearoperator, lambda: apxchol.laplacian(None)):
    try:
        operation()
    except ImportError as exc:
        assert "apxchol[scipy]" in str(exc)
    else:
        raise AssertionError("SciPy integration should require SciPy")
'''
    env = dict(os.environ, PYTHONPATH=str(Path(apxchol.__file__).parent.parent))
    subprocess.run([sys.executable, "-c", code], env=env, check=True, timeout=30)
