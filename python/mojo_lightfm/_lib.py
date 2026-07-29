"""ctypes bridge to the single Mojo shared library."""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIB_PATH = os.environ.get("MOJO_LIGHTFM_LIB") or os.path.join(
    ROOT, "dist", "libmojo-lightfm.so"
)

I = ctypes.c_int64
D = ctypes.c_double

_SIGNATURES = {
    "mlfm_representations": ([I] * 9, None),
    "mlfm_predict_dense": ([I] * 10, None),
    "mlfm_predict": ([I] * 17, None),
    "mlfm_score_matrix": ([I] * 9, None),
    "mlfm_fit_epoch": ([I] * 30 + [D] * 5 + [I] * 5, I),
}


class BuildError(RuntimeError):
    pass


def build(force: bool = False) -> str:
    source = os.path.join(ROOT, "src", "lightfm.mojo")
    if (
        not force
        and os.path.exists(LIB_PATH)
        and os.path.getmtime(LIB_PATH) >= os.path.getmtime(source)
    ):
        return LIB_PATH
    mojo = shutil.which("mojo")
    if mojo is None:
        raise BuildError("mojo executable not found; run `pixi run build`")
    os.makedirs(os.path.dirname(LIB_PATH), exist_ok=True)
    process = subprocess.run(
        [
            mojo,
            "build",
            "--emit",
            "shared-lib",
            source,
            "-o",
            LIB_PATH,
        ],
        capture_output=True,
        text=True,
        timeout=1800,
    )
    if process.returncode or not os.path.exists(LIB_PATH):
        raise BuildError((process.stderr or process.stdout).strip()[:4000])
    return LIB_PATH


_library: ctypes.CDLL | None = None


def lib() -> ctypes.CDLL:
    global _library
    if _library is None:
        _library = ctypes.CDLL(build())
        for name, (argtypes, restype) in _SIGNATURES.items():
            function = getattr(_library, name)
            function.argtypes = argtypes
            function.restype = restype
    return _library


def addr(array: np.ndarray) -> int:
    """Return an address only for arrays that satisfy the native ABI."""
    if not isinstance(array, np.ndarray):
        raise TypeError("native buffers must be NumPy arrays")
    if array.dtype not in (np.dtype(np.float32), np.dtype(np.int32)):
        raise TypeError(
            f"native buffers must use float32 or int32, received {array.dtype}"
        )
    if not array.flags.c_contiguous:
        raise ValueError("native buffers must be C-contiguous")
    address = int(array.ctypes.data)
    if address == 0:
        raise ValueError("native buffers must have a non-null address")
    return address
