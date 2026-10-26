#!/usr/bin/env python3
"""Package a portable static CPU/Linux or Metal/macOS CLI build."""
import hashlib
import json
import pathlib
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile


def run(*args):
    return subprocess.check_output([str(x) for x in args], text=True).strip()


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def copy_notices(paths, destination):
    destination.mkdir(parents=True, exist_ok=True)
    assert paths, f"No license files found for {destination.name}"
    for path in paths:
        shutil.copy2(path, destination / path.name)


def main(build, output):
    build, output = pathlib.Path(build).resolve(), pathlib.Path(output).resolve()
    cache = dict(line.split("=", 1) for line in (build / "CMakeCache.txt").read_text().splitlines()
                 if "=" in line and not line.startswith(("#", "//")))
    cache = {key.split(":", 1)[0]: value for key, value in cache.items()}
    source = pathlib.Path(cache["CMAKE_HOME_DIRECTORY"])
    linux = platform.system() == "Linux"
    assert platform.system() in ("Linux", "Darwin")
    assert cache["APXCHOL_NATIVE_ARCH"] == cache["BUILD_SHARED_LIBS"] == "OFF"
    assert cache["CMAKE_DISABLE_FIND_PACKAGE_Eigen3"] == cache["CMAKE_DISABLE_FIND_PACKAGE_spdlog"] == "ON"
    assert cache["APXCHOL_DEVICE"] == ("CPU" if linux else "METAL")
    assert platform.machine() == ("x86_64" if linux else "arm64")
    version = re.search(r"set\(APXCHOL_VERSION ([0-9.]+)\)",
                        (source / "cmake/apxchol_version.cmake").read_text())[1]
    name = f"apxchol-{version}-" + ("linux-x86_64-cpu" if linux else "macos-arm64-metal")
    output.mkdir(parents=True, exist_ok=True)
    archive = output / (name + ".tar.gz")
    assert not archive.exists(), f"Refusing to replace {archive}"
    omp, = {pathlib.Path(value).resolve() for key, value in cache.items()
            if key.startswith("OpenMP_") and key.endswith("_LIBRARY") and value and "omp" in pathlib.Path(value).name}
    info = {"version": version, "commit": run("git", "-C", source, "rev-parse", "HEAD"),
            "source": "https://github.com/AlgOptGroup/apxchol", "backend": cache["APXCHOL_DEVICE"], "compiler": run(cache["CMAKE_CXX_COMPILER"], "--version").splitlines()[0],
            "platform": name.split(version + "-", 1)[1], "dependencies": {}, "required_symbol_versions": {}}
    with tempfile.TemporaryDirectory(prefix="apxchol-cli-") as work:
        root = pathlib.Path(work) / name
        run("cmake", "--install", build, "--component", "cli", "--prefix", root)
        exe = root / "bin/apxchol"
        licenses = root / "share/licenses"
        copy_notices([source / "LICENSE"], licenses / "apxchol")
        deps = {dep: pathlib.Path(cache.get("FETCHCONTENT_SOURCE_DIR_" + dep.upper()) or build / "_deps" / (dep + "-src"))
                for dep in ("eigen", "cli11", "spdlog", "fast_matrix_market")}
        notices = {"Eigen": list(deps["eigen"].glob("COPYING.*")), "CLI11": [deps["cli11"] / "LICENSE"],
                   "spdlog": [deps["spdlog"] / "LICENSE"], "fmt": [deps["spdlog"] / "include/spdlog/fmt/bundled/fmt.license.rst"],
                   "fast_matrix_market": [deps["fast_matrix_market"] / "LICENSE.txt"]}
        for dep in ("fast_float", "dragonbox", "ryu"):
            notices[dep] = list((deps["fast_matrix_market"] / "dependencies" / dep).glob("LICENSE*"))
        for dep, paths in notices.items():
            copy_notices(paths, licenses / dep)
        eigen_header = (deps["eigen"] / "Eigen/src/Core/util/Macros.h").read_text()
        eigen_version = ".".join(re.search(r"#define EIGEN_" + part + r"_VERSION (\d+)", eigen_header)[1]
                                 for part in ("WORLD", "MAJOR", "MINOR"))
        notice = (source / "python/Eigen-NOTICE.txt.in").read_text().replace("@Eigen3_VERSION@", eigen_version)
        (licenses / "Eigen/NOTICE").write_text(notice.replace("wheel", "archive").replace("Wheels", "Archives"))
        info["dependencies"]["Eigen"] = {"version": eigen_version,
            "sha256": re.search(r"URL_HASH SHA256=([0-9a-f]+)", (source / "cmake/apxchol_eigen.cmake").read_text())[1],
            "source": f"https://gitlab.com/libeigen/eigen/-/archive/{eigen_version}/eigen-{eigen_version}.tar.gz"}
        for dep in ("cli11", "spdlog", "fast_matrix_market"):
            assert pathlib.Path(run("git", "-C", deps[dep], "rev-parse", "--show-toplevel")).resolve() == deps[dep].resolve(), dep
            assert not run("git", "-C", deps[dep], "status", "--porcelain"), f"Modified dependency source: {dep}"
            info["dependencies"][dep] = {"source": run("git", "-C", deps[dep], "remote", "get-url", "origin"),
                                           "commit": run("git", "-C", deps[dep], "rev-parse", "HEAD")}
        info["dependencies"]["vendored"] = {"fmt": "spdlog/include/spdlog/fmt/bundled",
            "fast_float": "fast_matrix_market/dependencies/fast_float", "dragonbox": "fast_matrix_market/dependencies/dragonbox",
            "ryu": "fast_matrix_market/dependencies/ryu"}
        (root / "lib").mkdir()
        if linux:
            dynamic = run("readelf", "-d", exe)
            needed = re.findall(r"\(NEEDED\).*?\[(.*?)\]", dynamic)
            runtime, = [lib for lib in needed if re.fullmatch(r"libomp\.so(?:\.\d+)*", lib)]
            assert "$ORIGIN/../lib" in dynamic, "Configure CMAKE_INSTALL_RPATH=$ORIGIN/../lib"
            library = root / "lib" / runtime
            shutil.copy2(omp, library)
            allowed = {runtime, "libpthread.so.0", "libstdc++.so.6", "libm.so.6", "libgcc_s.so.1", "libc.so.6",
                       "librt.so.1", "libdl.so.2", "ld-linux-x86-64.so.2"}
            for path in (exe, library):
                assert set(re.findall(r"\(NEEDED\).*?\[(.*?)\]", run("readelf", "-d", path))) <= allowed
                versions = run("readelf", "--version-info", path)
                requirements = {}
                for family, value in re.findall(r"\b(GLIBCXX|GLIBC|CXXABI|GCC)_([0-9.]+)\b", versions):
                    if family not in requirements or tuple(map(int, value.split("."))) > tuple(map(int, requirements[family].split("."))):
                        requirements[family] = value
                info["required_symbol_versions"][str(path.relative_to(root))] = requirements
                glibc = [tuple(map(int, v.split("."))) for v in re.findall(r"GLIBC_([0-9.]+)", versions)]
                assert max(glibc) <= (2, 28), f"{path.name} requires glibc newer than 2.28"
            rpm = run("rpm", "-qf", "--qf", "%{NAME}\n%{VERSION}-%{RELEASE}\n%{SOURCERPM}\n%{LICENSE}\n", omp).splitlines()
            info["dependencies"]["OpenMP"] = dict(zip(("package", "version", "source_rpm", "license"), rpm))
            rpm_files = [pathlib.Path(p) for p in run("rpm", "-ql", rpm[0]).splitlines() if "/licenses/" in p]
            copy_notices([p for p in rpm_files if p.is_file()], licenses / "OpenMP")
            # The manylinux toolchain links small GCC/glibc compatibility objects.
            for dep in ("gcc", "glibc"):
                copy_notices(list(pathlib.Path("/usr/share/licenses", dep).glob("*")), licenses / dep)
            runtime_sources = ["libstdc++_nonshared.a", "libgcc.a"]
            info["dependencies"]["compiler_runtime_sources"] = [run("rpm", "-qf", "--qf", "%{SOURCERPM}", path)
                for lib in runtime_sources if (path := pathlib.Path(run(cache["CMAKE_CXX_COMPILER"], "-print-file-name=" + lib))).is_file()]
            info["dependencies"]["glibc_source"] = run("rpm", "-q", "--qf", "%{SOURCERPM}", "glibc")
            run("strip", "--strip-unneeded", exe)
        else:
            library = root / "lib/libomp.dylib"
            shutil.copy2(omp, library)
            formula = json.loads(run("brew", "info", "--json=v2", "libomp"))["formulae"][0]
            installed_version = omp.parent.parent.name
            assert installed_version in [x["version"] for x in formula["installed"]]
            upstream = installed_version.split("_")[0]
            assert upstream == formula["versions"]["stable"], "Refresh libomp before packaging an outdated formula"
            license_path = licenses / "OpenMP/LICENSE.TXT"
            license_path.parent.mkdir(parents=True)
            candidates = [p for p in (omp.parent.parent / "share").glob("**/LICENSE*") if p.is_file()]
            if candidates:
                shutil.copy2(candidates[0], license_path)
            else:
                run("curl", "--fail", "--location", "--silent", "--show-error", "--max-time", "60",
                    f"https://raw.githubusercontent.com/llvm/llvm-project/llvmorg-{upstream}/openmp/LICENSE.TXT", "--output", license_path)
            info["dependencies"]["OpenMP"] = {"version": installed_version, "formula": formula,
                "source": formula["urls"]["stable"], "license_sha256": digest(license_path)}
            for dep in dylibs(exe):
                if pathlib.PurePosixPath(dep).name == "libomp.dylib":
                    run("install_name_tool", "-change", dep, "@executable_path/../lib/libomp.dylib", exe)
                else:
                    assert system_dylib(dep), dep
            run("install_name_tool", "-id", "@rpath/libomp.dylib", library)
            assert all(system_dylib(dep) for dep in dylibs(library)[1:])
            run("strip", "-x", exe)
            for path in (library, exe):
                assert run("lipo", "-archs", path) == "arm64"
                load = run("otool", "-l", path)
                minimum = re.findall(r"\bminos\s+(\d+(?:\.\d+){0,2})", load)
                assert minimum and all(tuple((list(map(int, v.split('.'))) + [0, 0])[:3]) <= (15, 0, 0) for v in minimum)
                run("codesign", "--force", "--sign", "-", path)
                run("codesign", "--verify", "--strict", path)
        info["bundled_runtime_sha256"] = digest(library)
        (root / "BUILD.json").write_text(json.dumps(info, indent=2) + "\n")
        (root / "NOTICE.txt").write_text("This product includes software developed by ETH Zürich and the apxchol contributors.\n"
            "Dependency licenses are in share/licenses; BUILD.json identifies their sources and versions.\n")
        requirements = info["required_symbol_versions"].get("bin/apxchol", {})
        requirement = ("Linux x86-64-v2, glibc 2.28+, system libstdc++.so.6 "
            f"(GLIBCXX_{requirements.get('GLIBCXX')}, CXXABI_{requirements.get('CXXABI')}) and libgcc_s.so.1 "
            f"(GCC_{requirements.get('GCC')}).") if linux else (
            "Apple silicon, macOS 15+. CPU and Metal backends. Ad-hoc signed; not notarized with an Apple developer identity.")
        (root / "README.txt").write_text(f"apxchol {version}\n{requirement}\n\n"
            "Extract the whole directory and keep bin/ and lib/ together. No Python installation is needed.\n"
            "  bin/apxchol --help\n  bin/apxchol matrix.mtx --rhs rhs.mtx --backend cpu -o solution.mtx\n"
            "The RHS is one MatrixMarket vector. OpenMP is included; system libraries remain external.\n")
        manifest = {str(p.relative_to(root)): digest(p) for p in sorted(root.rglob("*")) if p.is_file()}
        (root / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
        with tarfile.open(archive, "w:gz") as tar:
            tar.add(root, arcname=name)
    archive.with_suffix(archive.suffix + ".sha256").write_text(f"{digest(archive)}  {archive.name}\n")
    print(archive)


def dylibs(path):
    return [line.strip().split(" (compatibility version", 1)[0] for line in run("otool", "-L", path).splitlines()[1:]]


def system_dylib(path):
    return path.startswith(("/usr/lib/", "/System/Library/"))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: python packaging/cli.py BUILD_DIRECTORY OUTPUT_DIRECTORY")
    main(*sys.argv[1:])
