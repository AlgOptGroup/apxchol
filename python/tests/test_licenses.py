from importlib.metadata import distribution

import pytest


def test_wheel_ships_a_license_for_each_bundled_component():
    package = distribution("apxchol")
    expression = package.metadata["License-Expression"]
    for license in ("BSD-3-Clause", "MPL-2.0", "Apache-2.0 WITH LLVM-exception",
                    "Zlib", "LicenseRef-NVIDIA-CUDA-EULA"):
        assert license in expression
    paths = [path.as_posix() for path in package.files or []]
    licenses = {path.split(".dist-info/licenses/", 1)[1]
                for path in paths if ".dist-info/licenses/" in path}
    if not licenses:
        pytest.skip("installed without license files (not from a wheel)")
    assert {"LICENSE", "eigen/NOTICE", "pybind11/LICENSE", "zlib/LICENSE"} <= licenses
    # Wheel repair vendors the OpenMP runtime next to the extension.
    if any("libomp" in path.rsplit("/", 1)[-1] for path in paths):
        assert "LICENSE.libomp.txt" in licenses
