"""Relocate an archive and check the fixed path5 SDDM fixture on each requested route."""
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

archive, matrix = (Path(p).resolve() for p in sys.argv[1:3])
with tempfile.TemporaryDirectory(prefix="apxchol relocated ") as directory:
    work = Path(directory)
    subprocess.run(["tar", "-xf", archive, "-C", work], check=True)
    exe, = work.glob("*/bin/apxchol")
    shutil.copyfile(matrix, work / "A.mtx")
    (work / "b.mtx").write_text("%%MatrixMarket matrix array real general\n5 1\n1\n2\n3\n4\n5\n")
    subprocess.run([exe, "--help"], cwd=work, check=True, timeout=30)
    version = subprocess.check_output([exe, "--version"], cwd=work, text=True, timeout=30)
    print(version, end="")
    assert version.startswith(f"apxchol {archive.name.split('-')[1]} ")
    if sys.platform == "linux":
        assert str(exe.parent / "../lib/libomp.so") in subprocess.check_output(["ldd", exe], text=True)
    for backend in sys.argv[3:]:
        result = subprocess.run([exe, work / "A.mtx", "--rhs", work / "b.mtx", "--backend", backend,
                                 "--threads", "2", "--tol", "1e-8", "--output", work / "x.mtx"],
                                cwd=work, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=60,
                                env=dict(os.environ, OMP_NUM_THREADS="2", MTL_DEBUG_LAYER="1", MTL_SHADER_VALIDATION="1"))
        print(result.stdout, end="", flush=True)
        result.check_returncode()
        assert f"backend:    {backend}" in result.stdout
        rows = [line for line in (work / "x.mtx").read_text().splitlines() if line and not line.startswith("%")]
        assert rows[0].split() == ["5", "1"]
        x = list(map(float, rows[1:]))
        assert len(x) == 5 and all(map(math.isfinite, x))
        diagonal = [1.5, 2.5, 2.5, 2.5, 1.5]  # tests/fixtures/path5_sddm.mtx
        residual = [diagonal[i] * x[i] - (x[i - 1] if i else 0) - (x[i + 1] if i < 4 else 0) - (i + 1) for i in range(5)]
        relative = math.sqrt(math.fsum(r * r for r in residual) / 55)
        assert relative <= 1e-8, (backend, relative)
        print(f"{backend}: independent relative residual {relative:.3g}")
