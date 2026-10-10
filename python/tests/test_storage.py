import os
import numpy as np
import pytest
import scipy.sparse as sp

import apxchol


@pytest.fixture(autouse=True)
def clear_storage_environment(monkeypatch):
    monkeypatch.delenv("APXCHOL_FACTOR_STORAGE", raising=False)


def operator_matrix():
    return sp.csc_matrix([[2.0, -1.0], [-1.0, 2.0]])


@pytest.mark.parametrize("dtype", [np.float32, "float32", np.dtype("float32"), ">f4"])
@pytest.mark.parametrize("keep_factor", [False, True])
def test_explicit_fp32_overrides_environment(monkeypatch, dtype, keep_factor):
    monkeypatch.setenv("APXCHOL_FACTOR_STORAGE", "float16")
    solver = apxchol.factorize(operator_matrix(), factor_storage_dtype=dtype,
                              keep_factor=keep_factor)
    assert solver.factor_storage_dtype == np.dtype("float32")
    result = solver.solve([1.0, 0.0])
    assert result.converged
    np.testing.assert_allclose(result.x, [2 / 3, 1 / 3], rtol=1e-8)
    assert os.environ["APXCHOL_FACTOR_STORAGE"] == "float16"


@pytest.mark.parametrize("dtype", [np.float16, "float16", np.dtype("float16")])
def test_fp16_and_fp32_instances_coexist(monkeypatch, dtype):
    monkeypatch.setenv("APXCHOL_FACTOR_STORAGE", "float32")
    full = apxchol.factorize(operator_matrix(), factor_storage_dtype=np.float32)
    try:
        half = apxchol.factorize(operator_matrix(), factor_storage_dtype=dtype)
    except RuntimeError as exc:
        assert "FP16 factor storage requires an x86 CPU with AVX/F16C support" in str(exc)
        pytest.skip("CPU lacks FP16 conversion support")
    assert half.factor_storage_dtype == np.dtype("float16")
    assert os.environ["APXCHOL_FACTOR_STORAGE"] == "float32"
    monkeypatch.setenv("APXCHOL_FACTOR_STORAGE", "float16")
    assert full.factor_storage_dtype == np.dtype("float32")
    for solver in (half, full):
        result = solver.solve([1.0, 0.0])
        assert result.x.dtype == np.float64
        assert result.converged
        np.testing.assert_allclose(result.x, [2 / 3, 1 / 3], rtol=1e-8)


@pytest.mark.parametrize("setting", [None, "", "auto", "float32"])
def test_default_storage_uses_environment(monkeypatch, setting):
    if setting is not None:
        monkeypatch.setenv("APXCHOL_FACTOR_STORAGE", setting)
    assert apxchol.factorize(operator_matrix()).factor_storage_dtype == np.float32


@pytest.mark.parametrize("dtype", [np.float64, np.int16, np.complex64])
def test_unsupported_storage_dtype_is_rejected(dtype):
    with pytest.raises(ValueError, match="factor storage must be auto, float16 or float32"):
        apxchol.factorize(operator_matrix(), factor_storage_dtype=dtype)


def test_invalid_environment_fails_unless_overridden(monkeypatch):
    monkeypatch.setenv("APXCHOL_FACTOR_STORAGE", "float64")
    with pytest.raises(ValueError, match="factor storage must be"):
        apxchol.factorize(operator_matrix())
    solver = apxchol.factorize(operator_matrix(), factor_storage_dtype="float32")
    assert solver.factor_storage_dtype == np.float32
