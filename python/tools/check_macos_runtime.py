#!/usr/bin/env python3
from __future__ import annotations

import ctypes
from pathlib import Path

import apxchol  # noqa: F401 - importing loads the extension and its runtime


def loaded_images() -> list[Path]:
    dyld = ctypes.CDLL(None)
    dyld._dyld_image_count.restype = ctypes.c_uint32
    dyld._dyld_get_image_name.argtypes = [ctypes.c_uint32]
    dyld._dyld_get_image_name.restype = ctypes.c_char_p
    return [
        Path(dyld._dyld_get_image_name(index).decode()).resolve()
        for index in range(dyld._dyld_image_count())
    ]


runtimes = sorted(
    {
        path
        for path in loaded_images()
        if path.name.lower().startswith(("libomp", "libiomp", "libgomp"))
    }
)
if len(runtimes) != 1:
    raise SystemExit(f"expected one loaded libomp runtime, found {runtimes}")

runtime = runtimes[0]
expected_suffix = Path("apxchol/.dylibs/libomp.dylib")
if not runtime.as_posix().endswith(expected_suffix.as_posix()):
    raise SystemExit(f"libomp was not loaded from the wheel: {runtime}")

print(runtime)
