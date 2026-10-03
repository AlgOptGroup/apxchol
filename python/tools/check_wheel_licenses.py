#!/usr/bin/env python3
"""Check the notices accompanying the OpenMP runtime in each platform wheel."""
import sys
import zipfile
from pathlib import Path


def check(wheel: Path) -> None:
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        runtime = "libomp"
        if not any(Path(name).name.startswith(runtime) for name in names):
            raise SystemExit(f"{wheel.name}: bundled {runtime} is missing")
        for license_name in ("LICENSE", f"LICENSE.{runtime}.txt"):
            matches = [name for name in names
                       if name.endswith(f".dist-info/licenses/{license_name}")]
            if len(matches) != 1:
                raise SystemExit(f"{wheel.name}: expected one {license_name}")
            expected = (Path(__file__).resolve().parents[1] / license_name).read_bytes()
            if archive.read(matches[0]) != expected:
                raise SystemExit(f"{wheel.name}: {license_name} content differs")
    print(f"validated runtime notices: {wheel.name}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("usage: check_wheel_licenses.py WHEEL [WHEEL ...]")
    for argument in sys.argv[1:]:
        check(Path(argument))
