from importlib.metadata import distribution
import json

import pytest


def test_wheel_ships_a_license_for_each_bundled_component():
    package = distribution("apxchol")
    expression = package.metadata["License-Expression"]
    for license in ("BSD-3-Clause", "MPL-2.0", "Apache-2.0 WITH LLVM-exception"):
        assert license in expression
    direct_url = package.read_text("direct_url.json")
    if direct_url and json.loads(direct_url).get("dir_info", {}).get("editable"):
        pytest.skip("an editable install carries no wheel license files")
    files = package.files or []
    paths = [path.as_posix() for path in files]
    licenses = {path.split(".dist-info/licenses/", 1)[1]
                for path in paths if ".dist-info/licenses/" in path}
    assert {"LICENSE", "eigen/NOTICE", "pybind11/LICENSE"} <= licenses
    # Wheel repair vendors the OpenMP runtime next to the extension.
    if any("libomp" in path.rsplit("/", 1)[-1] for path in paths):
        assert "LICENSE.libomp.txt" in licenses
    # The CUDA runtime is linked statically into the extension; only those
    # wheels declare the NVIDIA term and ship the CUDA and CCCL texts.
    extension, = [f for f in files if f.name.startswith("_apxchol") and f.parent.name == "apxchol"]
    cuda = b"cudaErrorInsufficientDriver" in extension.locate().read_bytes()
    assert ("LicenseRef-NVIDIA-CUDA-EULA" in expression) == cuda
    assert ("LICENSE.cuda.txt" in licenses) == cuda
    assert ("LICENSE.cccl.txt" in licenses) == cuda
