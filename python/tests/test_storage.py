import platform
import numpy as np
import pytest
import scipy.sparse as sp

import apxchol


def operator_matrix():
    return sp.csc_matrix([[2.0, -1.0], [-1.0, 2.0]])


@pytest.mark.parametrize("dtype", [np.float32, "float32", np.dtype("float32"), ">f4"])
def test_explicit_fp32(dtype):
    solver = apxchol.factorize(operator_matrix(), backend="cpu", factor_storage=dtype)
    assert solver.factor_storage == np.dtype("float32")
    result = solver.solve([1.0, 0.0])
    assert result.converged
    np.testing.assert_allclose(result.x, [2 / 3, 1 / 3], rtol=1e-8)


@pytest.mark.parametrize("dtype", [np.float16, "float16", np.dtype("float16")])
def test_fp16_and_fp32_instances_coexist(dtype):
    full = apxchol.factorize(operator_matrix(), backend="cpu", factor_storage=np.float32)
    try:
        half = apxchol.factorize(operator_matrix(), backend="cpu", factor_storage=dtype)
    except RuntimeError as exc:
        assert str(exc) == "FP16 factor storage requires AVX/F16C or Apple ARM64"
        assert (platform.system(), platform.machine()) != ("Darwin", "arm64"), str(exc)
        pytest.skip("CPU lacks FP16 conversion support")
    assert half.factor_storage == np.dtype("float16")
    assert full.factor_storage == np.dtype("float32")
    for solver in (half, full):
        result = solver.solve([1.0, 0.0])
        assert result.x.dtype == np.float64
        assert result.converged
        np.testing.assert_allclose(result.x, [2 / 3, 1 / 3], rtol=1e-8)


@pytest.mark.parametrize("setting", [None, "", "auto", "float16", "float32", "invalid"])
def test_default_storage_ignores_environment(monkeypatch, setting):
    if setting is None:
        monkeypatch.delenv("APXCHOL_FACTOR_STORAGE", raising=False)
    else:
        monkeypatch.setenv("APXCHOL_FACTOR_STORAGE", setting)
    assert apxchol.factorize(operator_matrix(), backend="cpu").factor_storage == np.float32


@pytest.mark.parametrize("dtype", [np.float64, np.int16, np.complex64])
def test_unsupported_storage_dtype_is_rejected(dtype):
    with pytest.raises(ValueError, match="factor storage must be auto, float16 or float32"):
        apxchol.factorize(operator_matrix(), backend="cpu", factor_storage=dtype)


def test_auto_storage_is_the_default():
    solver = apxchol.factorize(operator_matrix(), backend="cpu", factor_storage="auto")
    assert solver.factor_storage == np.float32
