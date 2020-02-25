import pypocketfft
# import pyfftw
import numpy as np
import pytest
from numpy.testing import assert_

pmp = pytest.mark.parametrize

shapes1D = ((10,), (127,))
shapes2D = ((128, 128), (128, 129), (1, 129), (129, 1))
shapes3D = ((32, 17, 39),)
shapes = shapes1D+shapes2D+shapes3D
len1D = range(1, 2048)


def _l2error(a, b):
    return np.sqrt(np.sum(np.abs(a-b)**2)/np.sum(np.abs(a)**2))


def _assert_close(a, b, epsilon):
    err = _l2error(a, b)
    if (err >= epsilon):
        print("Error: {} > {}".format(err, epsilon))
    assert_(err<epsilon)


def fftn(a, axes=None, inorm=0, out=None, nthreads=1):
    return pypocketfft.c2c(a, axes=axes, forward=True, inorm=inorm,
                           out=out, nthreads=nthreads)


def ifftn(a, axes=None, inorm=0, out=None, nthreads=1):
    return pypocketfft.c2c(a, axes=axes, forward=False, inorm=inorm,
                           out=out, nthreads=nthreads)


def rfftn(a, axes=None, inorm=0, nthreads=1):
    return pypocketfft.r2c(a, axes=axes, forward=True, inorm=inorm,
                           nthreads=nthreads)


def irfftn(a, axes=None, lastsize=0, inorm=0, nthreads=1):
    return pypocketfft.c2r(a, axes=axes, lastsize=lastsize, forward=False,
                           inorm=inorm, nthreads=nthreads)


def rfft_scipy(a, axis, inorm=0, out=None, nthreads=1):
    return pypocketfft.r2r_fftpack(a, axes=(axis,), real2hermitian=True,
                                   forward=True, inorm=inorm, out=out,
                                   nthreads=nthreads)


def irfft_scipy(a, axis, inorm=0, out=None, nthreads=1):
    return pypocketfft.r2r_fftpack(a, axes=(axis,), real2hermitian=False,
                                   forward=False, inorm=inorm, out=out,
                                   nthreads=nthreads)

tol = {np.float32: 6e-7, np.float64: 1.5e-15, np.longfloat: 1e-18}
ctype = {np.float32: np.complex64, np.float64: np.complex128, np.longfloat: np.longcomplex}

import platform
on_wsl = "microsoft" in platform.uname()[3].lower()
true_long_double = (np.longfloat != np.float64 and not on_wsl)
dtypes = [np.float32, np.float64,
          pytest.param(np.longfloat, marks=pytest.mark.xfail(
              not true_long_double,
              reason="Long double doesn't offer more precision"))]


@pmp("len", len1D)
@pmp("inorm", [0, 1, 2])
@pmp("dtype", dtypes)
def test1D(len, inorm, dtype):
    a = np.random.rand(len)-0.5 + 1j*np.random.rand(len)-0.5j
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
    a = np.random.rand(*shp)-0.5 + 1j*np.random.rand(*shp)-0.5j
    assert_(_l2error(a, ifftn(fftn(a, nthreads=nthreads, inorm=inorm),
                              nthreads=nthreads, inorm=2-inorm)) < 1e-15)
    a = a.astype(np.complex64)
    assert_(_l2error(a, ifftn(fftn(a, nthreads=nthreads, inorm=inorm),
                              nthreads=nthreads, inorm=2-inorm)) < 5e-7)


@pmp("shp", shapes2D)
@pmp("axes", ((0,), (1,), (0, 1), (1, 0)))
@pmp("inorm", [0, 1, 2])
def test_fftn2D(shp, axes, inorm):
    a = np.random.rand(*shp)-0.5 + 1j*np.random.rand(*shp)-0.5j
    assert_(_l2error(a, ifftn(fftn(a, axes=axes, inorm=inorm),
                              axes=axes, inorm=2-inorm)) < 1e-15)
    a = a.astype(np.complex64)
    assert_(_l2error(a, ifftn(fftn(a, axes=axes, inorm=inorm),
                              axes=axes, inorm=2-inorm)) < 5e-7)


@pmp("shp", shapes)
def test_rfftn(shp):
    a = np.random.rand(*shp)-0.5
    tmp1 = rfftn(a)
    tmp2 = fftn(a)
    part = tuple(slice(0,tmp1.shape[i]) for i in range(tmp1.ndim))
    assert_(_l2error(tmp1, tmp2[part]) < 1e-15)
    a = a.astype(np.float32)
    tmp1 = rfftn(a)
    tmp2 = fftn(a)
    part = tuple(slice(0,tmp1.shape[i]) for i in range(tmp1.ndim))
    assert_(_l2error(tmp1, tmp2[part]) < 5e-7)


# @pmp("shp", shapes)
# def test_rfft_scipy(shp):
#     for i in range(len(shp)):
#         a = np.random.rand(*shp)-0.5
#         assert_(_l2error(pyfftw.interfaces.scipy_fftpack.rfft(a, axis=i),
#                          rfft_scipy(a, axis=i)) < 1e-15)
#         assert_(_l2error(pyfftw.interfaces.scipy_fftpack.irfft(a, axis=i),
#                          irfft_scipy(a, axis=i, inorm=2)) < 1e-15)


@pmp("shp", shapes2D)
@pmp("axes", ((0,), (1,), (0, 1), (1, 0)))
def test_rfftn2D(shp, axes):
    a = np.random.rand(*shp)-0.5
    tmp1 = rfftn(a,axes=axes)
    tmp2 = fftn(a,axes=axes)
    part = tuple(slice(0,tmp1.shape[i]) for i in range(tmp1.ndim))
    assert_(_l2error(tmp1, tmp2[part]) < 1e-15)
    a = a.astype(np.float32)
    tmp1 = rfftn(a,axes=axes)
    tmp2 = fftn(a,axes=axes)
    part = tuple(slice(0,tmp1.shape[i]) for i in range(tmp1.ndim))
    assert_(_l2error(tmp1, tmp2[part]) < 5e-7)


@pmp("shp", shapes)
def test_identity(shp):
    a = np.random.rand(*shp)-0.5 + 1j*np.random.rand(*shp)-0.5j
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
    a = np.random.rand(*shp)-0.5
    b = a.astype(np.float32)
    for ax in range(a.ndim):
        n = a.shape[ax]
        assert_(_l2error(irfftn(rfftn(a, (ax,)), (ax,), lastsize=n, inorm=2),
                         a) < 1e-15)
        assert_(_l2error(irfftn(rfftn(b, (ax,)), (ax,), lastsize=n, inorm=2),
                         b) < 5e-7)


@pmp("shp", shapes)
def test_identity_r2(shp):
    a = np.random.rand(*shp)-0.5 + 1j*np.random.rand(*shp)-0.5j
    a = rfftn(irfftn(a))
    assert_(_l2error(rfftn(irfftn(a), inorm=2), a) < 1e-15)


@pmp("shp", shapes2D+shapes3D)
def test_genuine_hartley(shp):
    a = np.random.rand(*shp)-0.5
    v1 = pypocketfft.genuine_hartley(a)
    v2 = fftn(a.astype(np.complex128))
    v2 = v2.real+v2.imag
    assert_(_l2error(v1, v2) < 1e-15)


@pmp("shp", shapes)
def test_hartley_identity(shp):
    a = np.random.rand(*shp)-0.5
    v1 = pypocketfft.separable_hartley(pypocketfft.separable_hartley(a))/a.size
    assert_(_l2error(a, v1) < 1e-15)


@pmp("shp", shapes)
def test_genuine_hartley_identity(shp):
    a = np.random.rand(*shp)-0.5
    v1 = pypocketfft.genuine_hartley(pypocketfft.genuine_hartley(a))/a.size
    assert_(_l2error(a, v1) < 1e-15)
    v1 = a.copy()
    assert_(pypocketfft.genuine_hartley(
        pypocketfft.genuine_hartley(v1, out=v1), inorm=2, out=v1) is v1)
    assert_(_l2error(a, v1) < 1e-15)


@pmp("shp", shapes2D+shapes3D)
@pmp("axes", ((0,), (1,), (0, 1), (1, 0)))
def test_genuine_hartley_2D(shp, axes):
    a = np.random.rand(*shp)-0.5
    assert_(_l2error(pypocketfft.genuine_hartley(pypocketfft.genuine_hartley(
        a, axes=axes), axes=axes, inorm=2), a) < 1e-15)


@pmp("len", len1D)
@pmp("inorm", [0, 1])  # inorm==2 not needed, tested via inverse
@pmp("type", [1, 2, 3, 4])
@pmp("dtype", dtypes)
def testdcst1D(len, inorm, type, dtype):
    a = (np.random.rand(len)-0.5).astype(dtype)
    eps = tol[dtype]
    itp = (0, 1, 3, 2, 4)
    itype = itp[type]
    if type != 1 or len > 1:  # there are no length-1 type 1 DCTs
        _assert_close(a, pypocketfft.dct(pypocketfft.dct(a, inorm=inorm, type=type), inorm=2-inorm, type=itype), eps)
    _assert_close(a, pypocketfft.dst(pypocketfft.dst(a, inorm=inorm, type=type), inorm=2-inorm, type=itype), eps)
