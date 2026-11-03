"""Extract one SDK archive and build/run its real CMake consumer."""
import hashlib
import os
from pathlib import Path
import re
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
version = re.search(r'#define APXCHOL_VERSION "([0-9.]+)"',
                    (prefix / "include/apxchol/version.h").read_text()).group(1)
soversion = ".".join(version.split(".")[:2])
lib = prefix / "lib"
if sys.platform == "darwin":
    chain = [f"libapxchol_core.{v}.dylib" for v in (version, soversion)] + ["libapxchol_core.dylib"]
    out = subprocess.run(["otool", "-D", lib / chain[0]], capture_output=True, text=True,
                         check=True).stdout
    assert out.split()[-1] == f"@rpath/{chain[1]}", out
else:
    chain = [f"libapxchol_core.so.{v}" for v in (version, soversion)] + ["libapxchol_core.so"]
    out = subprocess.run(["objdump", "-p", lib / chain[0]], capture_output=True, text=True,
                         check=True).stdout
    assert re.search(rf"SONAME\s+{re.escape(chain[1])}$", out, re.M), out
assert (lib / chain[0]).is_file() and not (lib / chain[0]).is_symlink()
for link, target in zip(chain[1:], chain):
    assert os.readlink(lib / link) == target, (link, os.readlink(lib / link))
build = work / "consumer"
env = dict(os.environ, OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1")
for name in ("LD_LIBRARY_PATH", "LD_PRELOAD", "DYLD_LIBRARY_PATH", "DYLD_INSERT_LIBRARIES"):
    env.pop(name, None)
subprocess.run(["cmake", "-S", fixture, "-B", build, "-DCMAKE_BUILD_TYPE=Release",
                f"-Dapxchol_DIR={prefix / 'lib/cmake/apxchol'}",
                "-DCMAKE_FIND_USE_PACKAGE_REGISTRY=OFF",
                "-DCMAKE_FIND_USE_SYSTEM_PACKAGE_REGISTRY=OFF"],
               env=env, check=True, timeout=120)
subprocess.run(["cmake", "--build", build, "--parallel", "2"], env=env, check=True, timeout=120)
for route in sys.argv[3:] or ["cpu"]:
    subprocess.run([build / "apxchol_installed_consumer", route], cwd=work,
                   env=dict(env, MTL_DEBUG_LAYER="1", MTL_SHADER_VALIDATION="1"),
                   check=True, timeout=60)
