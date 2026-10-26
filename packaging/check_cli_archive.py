#!/usr/bin/env python3
"""Check an extracted CLI archive and independent original-system residuals."""
import hashlib
import json
import math
import os
import pathlib
import platform
import re
import subprocess
import sys
import tarfile
import tempfile


def run(command, **kwargs):
    return subprocess.run([str(x) for x in command], check=True, text=True, capture_output=True, timeout=60, **kwargs)


def main(archive, source=None, build=None):
    archive = pathlib.Path(archive).resolve()
    with tempfile.TemporaryDirectory(prefix="apxchol relocated ") as temporary:
        work = pathlib.Path(temporary)
        with tarfile.open(archive) as tar:
            members = tar.getmembers()
            assert all(not m.issym() and not m.islnk() and not pathlib.PurePosixPath(m.name).is_absolute()
                       and ".." not in pathlib.PurePosixPath(m.name).parts for m in members)
            tar.extractall(work)
        root, = work.iterdir()
        manifest = json.loads((root / "MANIFEST.json").read_text())
        files = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in root.rglob("*") if p.is_file() and p.name != "MANIFEST.json"}
        assert files == manifest
        info = json.loads((root / "BUILD.json").read_text())
        exe = root / "bin/apxchol"
        prefix = []
        if platform.system() == "Darwin":
            assert source and build, "Pass source and build directories for the sandbox negative control"
            omp = pathlib.Path(run(["brew", "--prefix", "libomp"]).stdout.strip()).resolve()
            forbidden = [pathlib.Path(source).resolve(), pathlib.Path(build).resolve(), omp]
            policy = "(version 1) (allow default) " + " ".join(
                "(deny file-read* (subpath " + json.dumps(str(p)) + "))" for p in forbidden)
            prefix = ["/usr/bin/sandbox-exec", "-p", policy]
            negative = "import pathlib,sys\nfor p in sys.argv[1:]:\n try: pathlib.Path(p).read_bytes()\n except PermissionError: continue\n raise AssertionError(p)\n"
            run(prefix + [sys.executable, "-c", negative, forbidden[0] / "CMakeLists.txt",
                          forbidden[1] / "CMakeCache.txt", omp / "lib/libomp.dylib"])
            for path in (exe, root / "lib/libomp.dylib"):
                run(["codesign", "--verify", "--strict", path])
        else:
            # CI runs this check in a fresh base container with only the archive/checker mounted.
            linked = run(["ldd", exe]).stdout
            assert "not found" not in linked and re.search(r"libomp\.so[^ ]* => " + re.escape(str(root)), linked), linked
        help_text = run(prefix + [exe, "--help"]).stdout
        version_text = run(prefix + [exe, "--version"]).stdout
        assert "--backend" in help_text and f"apxchol {info['version']} " in version_text
        side = 21
        n = side * side
        edges = []
        for row in range(side):
            for column in range(side):
                i = row * side + column
                if row:
                    edges.append((i, i - side, 0.7 + (i % 11) / 13))
                if column:
                    edges.append((i, i - 1, 0.5 + (i % 7) / 9))
        diagonal = [0.25 + (i % 5) / 40 for i in range(n)]
        for i, j, weight in edges:
            diagonal[i] += weight
            diagonal[j] += weight
        b = [math.sin(i * 0.17) + math.cos(i * 0.11) for i in range(n)]
        with (work / "A.mtx").open("w") as f:
            f.write(f"%%MatrixMarket matrix coordinate real symmetric\n{n} {n} {n + len(edges)}\n")
            for i, value in enumerate(diagonal):
                f.write(f"{i + 1} {i + 1} {value:.17g}\n")
            for i, j, weight in edges:
                f.write(f"{i + 1} {j + 1} {-weight:.17g}\n")
        (work / "b.mtx").write_text(f"%%MatrixMarket matrix array real general\n{n} 1\n" + "".join(f"{value:.17g}\n" for value in b))
        residuals = {}
        for backend in (["cpu", "gpu"] if info["backend"] == "METAL" else ["cpu"]):
            env = dict(os.environ, OMP_NUM_THREADS="2", DYLD_PRINT_LIBRARIES="1")
            if backend == "gpu":
                env.update(MTL_DEBUG_LAYER="1", MTL_SHADER_VALIDATION="1")
            trace_prefix = prefix + ["/usr/bin/env", "DYLD_PRINT_LIBRARIES=1"] if prefix else []
            result = run(trace_prefix + [exe, work / "A.mtx", "--input-kind", "laplacian", "--rhs", work / "b.mtx",
                         "--backend", backend, "--threads", "2", "--tol", "1e-9", "--maxiter", "1000",
                         "--no-timings", "--output", work / "x.mtx"], env=env)
            assert re.search(r"backend:\s+" + backend, result.stdout), result.stdout
            if platform.system() == "Darwin":
                assert str(root / "lib/libomp.dylib") in result.stderr or str(root / "bin/../lib/libomp.dylib") in result.stderr
                assert str(omp) not in result.stderr
            rows = [line for line in (work / "x.mtx").read_text().splitlines() if line and not line.startswith("%")]
            assert list(map(int, rows[0].split())) == [n, 1]
            x = list(map(float, rows[1:]))
            assert len(x) == n and all(map(math.isfinite, x))
            ax = [d * value for d, value in zip(diagonal, x)]
            for i, j, weight in edges:
                ax[i] -= weight * x[j]
                ax[j] -= weight * x[i]
            residual = math.sqrt(math.fsum((a - rhs) ** 2 for a, rhs in zip(ax, b)) / math.fsum(rhs * rhs for rhs in b))
            assert residual <= 1e-9, (backend, residual)
            residuals[backend] = residual
        print(json.dumps({"archive": archive.name, "manifest_files": len(manifest), "checks_passed": 2 + len(residuals),
                          "checks_total": 2 + len(residuals), "independent_residuals": residuals,
                          "host": platform.platform(), "source_build_runtime_read_denials": bool(prefix)}, indent=2))


if __name__ == "__main__":
    if len(sys.argv) not in (2, 4):
        raise SystemExit("usage: python packaging/check_cli_archive.py ARCHIVE [SOURCE_DIRECTORY BUILD_DIRECTORY]")
    main(*sys.argv[1:])
