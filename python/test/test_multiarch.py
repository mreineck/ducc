import os
from pathlib import Path
import subprocess
import sys

import ducc0
import numpy as np
import pytest


_NUMERICS_SCRIPT = r"""
import sys
import ducc0
import numpy as np

path, expected_level, expected_max, *expected_available = sys.argv[1:]
info = ducc0.misc.cpu_info()
assert info["multiarch"] is True, info
assert info["selected_level"] == expected_level, info
assert info["max_level"] == expected_max, info
assert info["available_levels"] == expected_available, info
assert expected_level in expected_available, info

x = np.arange(13, dtype=np.float64)
z = np.sin(0.25*x) + 1j*np.cos(0.5*x)
fft32 = ducc0.fft.c2c(z.astype(np.complex64), forward=True, inorm=1)
fft64 = ducc0.fft.c2c(z.astype(np.complex128), forward=True, inorm=1)

angles = np.array([[0.2, 0.3], [1.2, 2.4], [2.7, 5.2]], dtype=np.float64)
healpix = ducc0.healpix.Healpix_Base(8, "RING").ang2pix(angles)

alm = np.array([[1.0, 0.2, -0.3, 0.1+0.2j, -0.2+0.15j, 0.3-0.05j]],
               dtype=np.complex128)
sht = ducc0.sht.synthesis_2d(
    alm=alm, lmax=2, mmax=2, spin=0, ntheta=4, nphi=6,
    nthreads=1, geometry="CC")
np.savez(path, fft32=fft32, fft64=fft64, healpix=healpix, sht=sht)
"""


def _level_number(level):
    if level == "x86-64":
        return 1
    prefix = "x86-64-v"
    assert level.startswith(prefix), level
    return int(level[len(prefix):])


def _run_level(path, level, available):
    env = os.environ.copy()
    cap = _level_number(level)
    max_level = "x86-64" if cap == 1 else f"x86-64-v{cap}"
    env["DUCC0_MAX_PSABI_LEVEL"] = str(cap)
    subprocess.run(
        [sys.executable, "-c", _NUMERICS_SCRIPT, str(path), level,
         max_level, *available],
        check=True,
        env=env,
        cwd=Path(__file__).resolve().parents[2],
    )


def _assert_numerics(result, baseline, level):
    np.testing.assert_array_equal(result["healpix"], baseline["healpix"])
    np.testing.assert_allclose(result["fft32"], baseline["fft32"],
                               rtol=5e-7, atol=5e-7)
    np.testing.assert_allclose(result["fft64"], baseline["fft64"],
                               rtol=2e-14, atol=2e-14)
    np.testing.assert_allclose(result["sht"], baseline["sht"],
                               rtol=1e-12, atol=1e-12,
                               err_msg=f"SHT mismatch at {level}")


def test_multiarch_dispatch_matches_x86_64(tmp_path):
    info = ducc0.misc.cpu_info()
    print(f"CPU info: {info}", flush=True)
    if not info["multiarch"]:
        pytest.skip("not a multiarch build")
    assert set(info) == {
        "architecture", "multiarch", "compiled_levels", "available_levels",
        "max_level", "selected_level",
    }
    assert info["architecture"] == "x86-64"
    compiled = info["compiled_levels"]
    assert compiled == ["x86-64", "x86-64-v3", "x86-64-v4"]
    available = info["available_levels"]
    assert available
    assert available == compiled[:len(available)]
    assert available[0] == "x86-64"

    max_number = _level_number(info["max_level"])
    expected_normal = next(level for level in reversed(available)
                           if _level_number(level) <= max_number)
    assert info["selected_level"] == expected_normal
    assert info["selected_level"] in available

    baseline = None
    for level in available:
        path = tmp_path / f"{level}.npz"
        print(f"Testing {level} ... ", end="", flush=True)
        _run_level(path, level, available)
        result = np.load(path)
        if baseline is None:
            assert level == "x86-64"
            baseline = result
        else:
            _assert_numerics(result, baseline, level)
        print("PASS", flush=True)
