# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# Copyright(C) 2020 Max-Planck-Society


import ducc0.fft as fft
# import pyfftw
import numpy as np
import pytest
from numpy.testing import assert_
import platform


pmp = pytest.mark.parametrize

shapes1D = ((10,), (127,))
shapes2D = ((128, 128), (128, 129),
            (1, 129), (2, 127), (3, 127), (6, 127),
            (129, 1), (127, 2), (127, 3), (127, 6))
shapes3D = ((32, 17, 39),)
shapes = shapes1D+shapes2D+shapes3D
len1D = range(1, 2048)


def _l2error(a, b):
    return np.sqrt(np.sum(np.abs(a-b)**2)/np.sum(np.abs(a)**2))


def _assert_close(a, b, epsilon):
    err = _l2error(a, b)
    if (err >= epsilon):
        print("Error: {} > {}".format(err, epsilon))
    assert_(err < epsilon)


def fftn(a, axes=None, inorm=0, out=None, nthreads=1):
    return fft.c2c(a, axes=axes, forward=True, inorm=inorm,
                   out=out, nthreads=nthreads)


def ifftn(a, axes=None, inorm=0, out=None, nthreads=1):
    return fft.c2c(a, axes=axes, forward=False, inorm=inorm,
                   out=out, nthreads=nthreads)


def rfftn(a, axes=None, inorm=0, nthreads=1):
    return fft.r2c(a, axes=axes, forward=True, inorm=inorm,
                   nthreads=nthreads)


def irfftn(a, axes=None, lastsize=0, inorm=0, nthreads=1):
    return fft.c2r(a, axes=axes, lastsize=lastsize, forward=False,
                   inorm=inorm, nthreads=nthreads)


def rfft_scipy(a, axis, inorm=0, out=None, nthreads=1):
    return fft.r2r_fftpack(a, axes=(axis,), real2hermitian=True,
                           forward=True, inorm=inorm, out=out,
                           nthreads=nthreads)


def irfft_scipy(a, axis, inorm=0, out=None, nthreads=1):
    return fft.r2r_fftpack(a, axes=(axis,), real2hermitian=False,
                           forward=False, inorm=inorm, out=out,
                           nthreads=nthreads)


def hc2c(inp, otype):
    n = inp.shape[0]
    n2 = (n-1)//2
    out = np.zeros_like(inp, dtype=otype)
    out[0] = inp[0]
    if n % 2 == 0:
        out[n//2] = inp[-1]
    out[1:n2+1] = inp[1:1+2*n2:2] + 1j*inp[2:2+2*n2:2]
    out[-1:-n2-1:-1] = inp[1:1+2*n2:2] - 1j*inp[2:2+2*n2:2]
    return out


tol = {np.float32: 6e-7, np.float64: 1.5e-15, np.longfloat: 1e-18}
ctype = {np.float32: np.complex64,
         np.float64: np.complex128,
         np.longfloat: np.longcomplex}


on_windows = ("microsoft" in platform.uname()[3].lower() or
              platform.system() == "Windows")
true_long_double = (np.longfloat != np.float64 and not on_windows)
dtypes = [np.float32, np.float64]
if true_long_double:
    dtypes += [np.longfloat]


@pmp("len", len1D)
@pmp("inorm", [0, 1, 2])
@pmp("dtype", dtypes)
def test1D(len, inorm, dtype):
    rng = np.random.default_rng(42)
    a = rng.random(len)-0.5 + 1j*rng.random(len)-0.5j
    a = a.astype(ctype[dtype])
    eps = tol[dtype]
    assert_(_l2error(a, ifftn(fftn(a, inorm=inorm), inorm=2-inorm)) < eps)
    assert_(_l2error(a.real, ifftn(fftn(a.real, inorm=inorm), inorm=2-inorm))
            < eps)
    assert_(_l2error(a.real, fftn(ifftn(a.real, inorm=inorm), inorm=2-inorm))
            < eps)
    assert_(_l2error(a.real, irfftn(rfftn(a.real, inorm=inorm),
                                    inorm=2-inorm, lastsize=len)) < eps)
    tmp = a.copy()
    assert_(ifftn(fftn(tmp, out=tmp, inorm=inorm), out=tmp, inorm=2-inorm)
            is tmp)
    assert_(_l2error(tmp, a) < eps)


@pmp("shp", shapes)
@pmp("nthreads", (0, 1, 2))
@pmp("inorm", [0, 1, 2])
def test_fftn(shp, nthreads, inorm):
    rng = np.random.default_rng(42)
    a = rng.random(shp)-0.5 + 1j*rng.random(shp)-0.5j
    assert_(_l2error(a, ifftn(fftn(a, nthreads=nthreads, inorm=inorm),
                              nthreads=nthreads, inorm=2-inorm)) < 1e-15)
    a = a.astype(np.complex64)
    assert_(_l2error(a, ifftn(fftn(a, nthreads=nthreads, inorm=inorm),
                              nthreads=nthreads, inorm=2-inorm)) < 5e-7)


@pmp("shp", shapes2D)
@pmp("axes", ((0,), (1,), (0, 1), (1, 0)))
@pmp("inorm", [0, 1, 2])
def test_fftn2D(shp, axes, inorm):
    rng = np.random.default_rng(42)
    a = rng.random(shp)-0.5 + 1j*rng.random(shp)-0.5j
    assert_(_l2error(a, ifftn(fftn(a, axes=axes, inorm=inorm),
                              axes=axes, inorm=2-inorm)) < 1e-15)
    a = a.astype(np.complex64)
    assert_(_l2error(a, ifftn(fftn(a, axes=axes, inorm=inorm),
                              axes=axes, inorm=2-inorm)) < 5e-7)


@pmp("shp", shapes)
def test_rfftn(shp):
    rng = np.random.default_rng(42)
    a = rng.random(shp)-0.5
    tmp1 = rfftn(a)
    tmp2 = fftn(a)
    part = tuple(slice(0, tmp1.shape[i]) for i in range(tmp1.ndim))
    assert_(_l2error(tmp1, tmp2[part]) < 1e-15)
    a = a.astype(np.float32)
    tmp1 = rfftn(a)
    tmp2 = fftn(a)
    part = tuple(slice(0, tmp1.shape[i]) for i in range(tmp1.ndim))
    assert_(_l2error(tmp1, tmp2[part]) < 5e-7)


# @pmp("shp", shapes)
# def test_rfft_scipy(shp):
#     for i in range(len(shp)):
#         a = rng.random(shp)-0.5
#         assert_(_l2error(pyfftw.interfaces.scipy_fftpack.rfft(a, axis=i),
#                          rfft_scipy(a, axis=i)) < 1e-15)
#         assert_(_l2error(pyfftw.interfaces.scipy_fftpack.irfft(a, axis=i),
#                          irfft_scipy(a, axis=i, inorm=2)) < 1e-15)


@pmp("shp", shapes2D)
@pmp("axes", ((0,), (1,), (0, 1), (1, 0)))
def test_rfftn2D(shp, axes):
    rng = np.random.default_rng(42)
    a = rng.random(shp)-0.5
    tmp1 = rfftn(a, axes=axes)
    tmp2 = fftn(a, axes=axes)
    part = tuple(slice(0, tmp1.shape[i]) for i in range(tmp1.ndim))
    assert_(_l2error(tmp1, tmp2[part]) < 1e-15)
    a = a.astype(np.float32)
    tmp1 = rfftn(a, axes=axes)
    tmp2 = fftn(a, axes=axes)
    part = tuple(slice(0, tmp1.shape[i]) for i in range(tmp1.ndim))
    assert_(_l2error(tmp1, tmp2[part]) < 5e-7)


@pmp("shp", shapes)
def test_identity(shp):
    rng = np.random.default_rng(42)
    a = rng.random(shp)-0.5 + 1j*rng.random(shp)-0.5j
    assert_(_l2error(ifftn(fftn(a), inorm=2), a) < 1.5e-15)
    assert_(_l2error(ifftn(fftn(a.real), inorm=2), a.real) < 1.5e-15)
    assert_(_l2error(fftn(ifftn(a.real), inorm=2), a.real) < 1.5e-15)
    tmp = a.copy()
    assert_(ifftn(fftn(tmp, out=tmp), inorm=2, out=tmp) is tmp)
    assert_(_l2error(tmp, a) < 1.5e-15)
    a = a.astype(np.complex64)
    assert_(_l2error(ifftn(fftn(a), inorm=2), a) < 6e-7)


@pmp("shp", shapes)
def test_identity_r(shp):
    rng = np.random.default_rng(42)
    a = rng.random(shp)-0.5
    b = a.astype(np.float32)
    for ax in range(a.ndim):
        n = a.shape[ax]
        assert_(_l2error(irfftn(rfftn(a, (ax,)), (ax,), lastsize=n, inorm=2),
                         a) < 1e-15)
        assert_(_l2error(irfftn(rfftn(b, (ax,)), (ax,), lastsize=n, inorm=2),
                         b) < 5e-7)


@pmp("shp", shapes)
def test_identity_r2(shp):
    rng = np.random.default_rng(42)
    a = rng.random(shp)-0.5 + 1j*rng.random(shp)-0.5j
    a = rfftn(irfftn(a))
    assert_(_l2error(rfftn(irfftn(a), inorm=2), a) < 1e-15)


@pmp("shp", shapes2D+shapes3D)
def test_genuine_hartley(shp):
    rng = np.random.default_rng(42)
    a = rng.random(shp)-0.5
    v1 = fft.genuine_hartley(a)
    v2 = fftn(a.astype(np.complex128))
    v2 = v2.real+v2.imag
    assert_(_l2error(v1, v2) < 1e-15)


@pmp("shp", shapes)
def test_hartley_identity(shp):
    rng = np.random.default_rng(42)
    a = rng.random(shp)-0.5
    v1 = fft.separable_hartley(fft.separable_hartley(a))/a.size
    assert_(_l2error(a, v1) < 1e-15)


@pmp("shp", shapes)
def test_genuine_hartley_identity(shp):
    rng = np.random.default_rng(42)
    a = rng.random(shp)-0.5
    v1 = fft.genuine_hartley(fft.genuine_hartley(a))/a.size
    assert_(_l2error(a, v1) < 1e-15)
    v1 = a.copy()
    assert_(fft.genuine_hartley(
        fft.genuine_hartley(v1, out=v1), inorm=2, out=v1) is v1)
    assert_(_l2error(a, v1) < 1e-15)


@pmp("shp", shapes2D+shapes3D)
@pmp("axes", ((0,), (1,), (0, 1), (1, 0)))
def test_genuine_hartley_2D(shp, axes):
    rng = np.random.default_rng(42)
    a = rng.random(shp)-0.5
    assert_(_l2error(fft.genuine_hartley(fft.genuine_hartley(
        a, axes=axes), axes=axes, inorm=2), a) < 1e-15)


@pmp("len", len1D)
@pmp("inorm", [0, 1])  # inorm==2 not needed, tested via inverse
@pmp("type", [1, 2, 3, 4])
@pmp("dtype", dtypes)
def testdcst1D(len, inorm, type, dtype):
    rng = np.random.default_rng(42)
    a = (rng.random(len)-0.5).astype(dtype)
    eps = tol[dtype]
    itp = (0, 1, 3, 2, 4)
    itype = itp[type]
    if type != 1 or len > 1:  # there are no length-1 type 1 DCTs
        _assert_close(a, fft.dct(fft.dct(a, inorm=inorm, type=type),
                      inorm=2-inorm, type=itype), eps)
    _assert_close(a, fft.dst(fft.dst(a, inorm=inorm, type=type), inorm=2-inorm,
                  type=itype), eps)


@pmp("len", (3, 4, 5, 6, 7, 8, 9, 10))
@pmp("dtype", dtypes)
def testfftpack_extra(len, dtype):
    rng = np.random.default_rng(42)
    a = (rng.random(len)-0.5).astype(dtype)
    eps = tol[dtype]
    ref = fft.c2c(a, forward=False)
    test = fft.r2r_fftpack(a, (0,), real2hermitian=True, forward=False)
    testc = hc2c(test, ctype[dtype])
    _assert_close(ref, testc, eps)
    ref = fft.c2c(ref, forward=True)
    test = fft.r2r_fftpack(test, (0,), real2hermitian=False, forward=True)
    _assert_close(ref, test, eps)
