"""Portable solver archives work without SciPy or repeated factorization."""
import subprocess
import sys

import numpy as np
import pytest

import apxchol
from test_numpy import DATA, INDICES, INDPTR, check_solve_and_apply, gpu_solver


def owner(**options):
    return apxchol.Solver.from_csc(DATA, INDICES, INDPTR, backend="cpu", **options)


def test_numpy_archive_round_trip(tmp_path):
    original = owner(seed=7)
    path = tmp_path / "factor-λ.npz"
    original.save(path)
    restored = apxchol.Solver.load(path)
    assert restored.backend == "cpu"
    assert restored.shape == original.shape
    assert restored.factor_nnz == original.factor_nnz
    assert restored.fill_ratio == original.fill_ratio
    np.testing.assert_array_equal(restored.P, original.P)
    for width in (1, 4, 16):
        check_solve_and_apply(restored, width)
    second = tmp_path / "again.npz"
    restored.save(second)
    with np.load(path, allow_pickle=False) as before, np.load(second, allow_pickle=False) as after:
        assert set(before.files) == set(after.files)
        for key in before.files:
            np.testing.assert_array_equal(before[key], after[key])


@pytest.mark.parametrize("loaded", [False, True])
def test_archive_save_requires_retained_factor(tmp_path, loaded):
    if loaded:
        path = tmp_path / "source.npz"
        owner().save(path)
        solver = apxchol.Solver.load(path, keep_factor=False)
    else:
        solver = owner(keep_factor=False)
    check_solve_and_apply(solver, 4)
    destination = tmp_path / "released.npz"
    with pytest.raises(ValueError):
        solver.save(destination)
    assert not destination.exists()


def test_archive_does_not_replace_existing_file(tmp_path):
    destination = tmp_path / "existing.npz"
    sentinel = b"existing file must remain intact"
    destination.write_bytes(sentinel)
    with pytest.raises(RuntimeError, match="already exists"):
        owner().save(destination)
    assert destination.read_bytes() == sentinel


def test_archive_null_path_and_invalid_options(tmp_path):
    prefix = tmp_path / "prefix.npz"
    invalid_path = str(prefix) + "\0suffix"
    with pytest.raises((ValueError, TypeError)):
        owner().save(invalid_path)
    assert not prefix.exists()
    with pytest.raises((ValueError, TypeError)):
        apxchol.Solver.load(invalid_path)
    with pytest.raises(ValueError, match="num_threads"):
        apxchol.Solver.load(prefix, num_threads=-1)
    with pytest.raises(ValueError, match="unknown option"):
        apxchol.Solver.load(prefix, unknown_option=True)
    with pytest.raises(RuntimeError):
        apxchol.Solver.load(prefix)
    prefix.write_bytes(b"not an archive")
    with pytest.raises(RuntimeError):
        apxchol.Solver.load(prefix)


def test_archive_round_trip_without_importing_scipy(tmp_path):
    probe = r'''
import importlib.abc
import sys
class BlockScipy(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "scipy" or fullname.startswith("scipy."):
            raise ModuleNotFoundError("blocked for this test", name="scipy")
sys.meta_path.insert(0, BlockScipy())
import apxchol
solver = apxchol.Solver.from_csc([4.], [0], [0, 1], backend="cpu")
solver.save(sys.argv[1])
restored = apxchol.Solver.load(sys.argv[1])
assert restored.solve([8.]).x[0] == 2
assert not any(n == "scipy" or n.startswith("scipy.") for n in sys.modules)
'''
    subprocess.run([sys.executable, "-I", "-B", "-c", probe, str(tmp_path / "isolated.npz")],
                   check=True, timeout=30)


@pytest.mark.parametrize("storage", [np.float16, np.float32])
def test_archive_gpu_cpu_handoff(tmp_path, storage):
    resident = gpu_solver(storage)
    gpu_path = tmp_path / "gpu.npz"
    resident.save(gpu_path)
    cpu_path = tmp_path / "cpu.npz"
    owner().save(cpu_path)
    for path in (cpu_path, gpu_path):
        for backend in ("cpu", "gpu"):
            restored = apxchol.Solver.load(path, backend=backend,
                                          factor_storage_dtype=np.float32)
            assert restored.backend == backend
            for width in (1, 4, 16):
                check_solve_and_apply(restored, width)
