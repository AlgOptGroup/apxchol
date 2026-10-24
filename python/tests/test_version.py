from importlib.metadata import version

import apxchol


def test_native_version_matches_package_metadata():
    assert apxchol.__version__ == version("apxchol")
