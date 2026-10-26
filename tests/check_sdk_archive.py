"""Extract one SDK archive and build/run its real CMake consumer."""
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile

archive, fixture = (Path(p).resolve() for p in sys.argv[1:3])
expected = Path(str(archive) + ".sha256").read_text().split()[0]
assert hashlib.sha256(archive.read_bytes()).hexdigest() == expected
work = Path(tempfile.mkdtemp(prefix="apxchol SDK "))
print(f"SDK consumer work: {work}", flush=True)
with tarfile.open(archive) as package:
    package.extractall(work, filter="data")
prefix, = work.iterdir()
assert (prefix / "lib/cmake/apxchol/apxcholConfig.cmake").is_file()
build = work / "consumer"
env = dict(os.environ, OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1")
for name in ("LD_LIBRARY_PATH", "LD_PRELOAD", "DYLD_LIBRARY_PATH", "DYLD_INSERT_LIBRARIES"):
    env.pop(name, None)
subprocess.run(["cmake", "-S", fixture, "-B", build, "-DCMAKE_BUILD_TYPE=Release",
                f"-Dapxchol_DIR={prefix / 'lib/cmake/apxchol'}",
                "-DCMAKE_DISABLE_FIND_PACKAGE_Eigen3=ON",
                "-DCMAKE_FIND_USE_PACKAGE_REGISTRY=OFF",
                "-DCMAKE_FIND_USE_SYSTEM_PACKAGE_REGISTRY=OFF"],
               env=env, check=True, timeout=120)
subprocess.run(["cmake", "--build", build, "--parallel", "2"], env=env, check=True, timeout=120)
for route in sys.argv[3:] or ["cpu"]:
    subprocess.run([build / "apxchol_installed_consumer", route], cwd=work,
                   env=dict(env, MTL_DEBUG_LAYER="1", MTL_SHADER_VALIDATION="1"),
                   check=True, timeout=60)
