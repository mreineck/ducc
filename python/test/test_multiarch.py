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

path, expected_level, expected_cap = sys.argv[1:]
info = ducc0.misc.cpu_info()
assert info["selected_level"] == expected_level, info
assert info["selection_cap"] == expected_cap, info

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


def _run_level(path, cap, expected_level):
    env = os.environ.copy()
    env["DUCC0_MAX_PSABI_LEVEL"] = str(cap)
    subprocess.run(
        [sys.executable, "-c", _NUMERICS_SCRIPT, str(path), expected_level,
         "x86-64" if cap == 1 else f"x86-64-v{cap}"],
        check=True,
        env=env,
        cwd=Path(__file__).resolve().parents[2],
    )


def test_multiarch_dispatch_matches_x86_64(tmp_path):
    cpu_info = getattr(ducc0.misc, "cpu_info", None)
    if cpu_info is None:
        pytest.skip("this build has no multiarch dispatch introspection")

    info = cpu_info()
    assert info["architecture"] == "x86-64"
    compiled = info["compiled_levels"]
    compiled_numbers = [_level_number(level) for level in compiled]
    assert compiled_numbers == sorted(set(compiled_numbers))
    assert 1 in compiled_numbers

    usable_number = _level_number(info["usable_level"])
    selection_limit = min(usable_number, _level_number(info["selection_cap"]))
    expected_normal = next(level for level in reversed(compiled)
                           if _level_number(level) <= selection_limit)
    assert info["selected_level"] == expected_normal
    eligible = [level for level in compiled
                if _level_number(level) <= usable_number]
    assert eligible

    baseline_path = tmp_path / "x86-64.npz"
    _run_level(baseline_path, 1, "x86-64")
    baseline = np.load(baseline_path)

    paths = {"x86-64": baseline}
    for level in eligible:
        if level == "x86-64":
            continue
        path = tmp_path / f"{level}.npz"
        cap = _level_number(level)
        _run_level(path, cap, level)
        paths[level] = np.load(path)

    for level, result in paths.items():
        np.testing.assert_array_equal(result["healpix"], baseline["healpix"])
        np.testing.assert_allclose(result["fft32"], baseline["fft32"],
                                   rtol=5e-7, atol=5e-7)
        np.testing.assert_allclose(result["fft64"], baseline["fft64"],
                                   rtol=2e-14, atol=2e-14)
        np.testing.assert_allclose(result["sht"], baseline["sht"],
                                   rtol=1e-12, atol=1e-12,
                                   err_msg=f"SHT mismatch at {level}")

    if max(compiled_numbers) > usable_number:
        highest_usable = eligible[-1]
        path = tmp_path / "cap-above-host.npz"
        _run_level(path, max(compiled_numbers), highest_usable)
        result = np.load(path)
        np.testing.assert_array_equal(result["healpix"], baseline["healpix"])
        np.testing.assert_allclose(result["fft32"], baseline["fft32"],
                                   rtol=5e-7, atol=5e-7)
        np.testing.assert_allclose(result["fft64"], baseline["fft64"],
                                   rtol=2e-14, atol=2e-14)
        np.testing.assert_allclose(result["sht"], baseline["sht"],
                                   rtol=1e-12, atol=1e-12)
