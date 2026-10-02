#!/usr/bin/env python3
from __future__ import annotations

import re
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path


def command(*args: str | Path) -> str:
    return subprocess.check_output([str(arg) for arg in args], text=True).strip()


# libc++ std::pmr (used by the factorization workspace) ships with macOS 14.
MINIMUM_MACOS = "14.0"


def require_minimum_macos_arm64(path: Path) -> None:
    architectures = command("lipo", "-archs", path).split()
    if architectures != ["arm64"]:
        raise SystemExit(f"{path.name}: expected arm64, found {architectures}")
    build = command("vtool", "-show-build", path)
    match = re.search(r"^\s*minos\s+(\S+)$", build, re.MULTILINE)
    if match is None or match.group(1) != MINIMUM_MACOS:
        raise SystemExit(f"{path.name}: expected minimum macOS {MINIMUM_MACOS}\n{build}")


def check_wheel(wheel: Path) -> None:
    # The platform tag must promise no older macOS than the binaries need, or
    # pip installs the wheel where the std::pmr symbols are missing.
    tag = "macosx_" + MINIMUM_MACOS.replace(".", "_") + "_arm64"
    if not wheel.name.endswith(f"-{tag}.whl"):
        raise SystemExit(f"{wheel.name}: expected the platform tag {tag}")
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        extensions = [
            name
            for name in names
            if name.startswith("apxchol/_apxchol") and name.endswith(".so")
        ]
        runtimes = [name for name in names if Path(name).name == "libomp.dylib"]
        project_licenses = [
            name for name in names if name.endswith(".dist-info/licenses/LICENSE")
        ]
        runtime_licenses = [
            name
            for name in names
            if name.endswith(".dist-info/licenses/LICENSE.libomp.txt")
        ]
        if len(extensions) != 1:
            raise SystemExit(f"{wheel.name}: expected one extension, found {extensions}")
        if runtimes != ["apxchol/.dylibs/libomp.dylib"]:
            raise SystemExit(f"{wheel.name}: unexpected libomp entries: {runtimes}")
        if len(project_licenses) != 1 or len(runtime_licenses) != 1:
            raise SystemExit(f"{wheel.name}: required license files are missing")

        with tempfile.TemporaryDirectory(prefix="apxchol-wheel-check-") as directory:
            archive.extractall(directory)
            root = Path(directory)
            extension = root / extensions[0]
            runtime = root / runtimes[0]
            require_minimum_macos_arm64(extension)
            require_minimum_macos_arm64(runtime)

            dependencies = command("otool", "-L", extension)
            libomp_dependencies = [
                line.strip().split(" ", 1)[0]
                for line in dependencies.splitlines()[1:]
                if "libomp" in line.lower()
            ]
            expected = "@loader_path/.dylibs/libomp.dylib"
            if libomp_dependencies != [expected]:
                raise SystemExit(
                    f"{wheel.name}: unexpected OpenMP dependency: "
                    f"{libomp_dependencies}"
                )

            linked = "\n".join(
                (
                    "\n".join(dependencies.splitlines()[1:]),
                    "\n".join(command("otool", "-L", runtime).splitlines()[1:]),
                    "\n".join(command("otool", "-l", extension).splitlines()[1:]),
                    "\n".join(command("otool", "-l", runtime).splitlines()[1:]),
                )
            ).lower()
            forbidden = (
                "/opt/homebrew",
                "/usr/local",
                "/private/tmp",
                "/tmp/",
                "/private/var/folders",
                "/var/folders",
                "/users/runner/work",
            )
            leaked = [entry for entry in forbidden if entry in linked]
            if leaked:
                raise SystemExit(
                    f"{wheel.name}: nonportable dependency paths remain: {leaked}"
                )

    print(f"validated {wheel}")


if len(sys.argv) < 2:
    raise SystemExit(f"usage: {Path(sys.argv[0]).name} WHEEL [WHEEL ...]")
for argument in sys.argv[1:]:
    check_wheel(Path(argument))
