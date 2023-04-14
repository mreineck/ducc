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
# Copyright(C) 2020-2021 Max-Planck-Society


import numpy as np
import pytest
from numpy.testing import assert_allclose
import ducc0

pmp = pytest.mark.parametrize


def _assert_close(a, b, epsilon):
    if isinstance(a, np.ndarray):
        err = ducc0.misc.l2error(a, b)
    else:
        err = np.abs(a-b)/np.maximum(np.abs(a),np.abs(b))
    if (err >= epsilon):
        print("Error: {} > {}".format(err, epsilon))
    assert_allclose(err, 0, atol=epsilon)


def nalm(lmax, mmax):
    return ((mmax+1)*(mmax+2))//2 + (mmax+1)*(lmax-mmax)


def random_alm(rng, lmax, mmax, ncomp):
    res = rng.uniform(-1., 1., (ncomp, nalm(lmax, mmax))) \
     + 1j*rng.uniform(-1., 1., (ncomp, nalm(lmax, mmax)))
    # make a_lm with m==0 real-valued
    res[:, 0:lmax+1].imag = 0.
    return res


def convolve(alm1, alm2, lmax, nthreads=1):
    ntheta, nphi = lmax+1, ducc0.fft.good_size(2*lmax+1, True)
    tmap = ducc0.sht.experimental.synthesis_2d(
        alm=alm1.reshape((1,-1)), ntheta=ntheta, nphi=nphi, lmax=lmax,
        geometry="GL", spin=0, nthreads=nthreads)
    tmap *= ducc0.sht.experimental.synthesis_2d(
        alm=alm2.reshape((1,-1)), ntheta=ntheta, nphi=nphi, lmax=lmax,
        geometry="GL", spin=0, nthreads=nthreads)
    res = ducc0.sht.experimental.analysis_2d(
        map=tmap, lmax=0, spin=0, geometry="GL", nthreads=nthreads)
    return np.sqrt(4*np.pi)*res[0,0]


def compress_alm(alm, lmax):
    res = np.empty(2*len(alm)-lmax-1, dtype=np.float64)
    res[0:lmax+1] = alm[0:lmax+1].real
    res[lmax+1::2] = np.sqrt(2)*alm[lmax+1:].real
    res[lmax+2::2] = np.sqrt(2)*alm[lmax+1:].imag
    return res


def myalmdot(a1, a2, lmax):
    return ducc0.misc.vdot(compress_alm(a1, lmax), compress_alm(a2, lmax))


@pmp("lkmax", [(13, 13), (2, 1), (30, 15), (35, 2)])
def test_against_convolution(lkmax):
    lmax, kmax = lkmax
    rng = np.random.default_rng(42)
    slm = random_alm(rng, lmax, lmax, 1)[0, :]
    blm = random_alm(rng, lmax, kmax, 1)[0, :]

    conv = ducc0.totalconvolve.ConvolverPlan(lmax, kmax,
                                             epsilon=1.5e-13, nthreads=2)
    nptg = 50
    ptg = np.zeros((nptg, 3))
    ptg[:, 0] = rng.uniform(0, np.pi, nptg)
    ptg[:, 1] = rng.uniform(0, 2*np.pi, nptg)
    ptg[:, 2] = rng.uniform(0, 2*np.pi, nptg)

    cube = np.empty((conv.Npsi(), conv.Ntheta(), conv.Nphi()))
    conv.getPlane(slm, blm, 0, cube[0:1])
    for mbeam in range(1, kmax+1):
        conv.getPlane(slm, blm, mbeam, cube[2*mbeam-1:2*mbeam+1])
    conv.prepPsi(cube)
    res1 = np.empty(ptg.shape[0])
    conv.interpol(cube, 0, 0, ptg[:, 0], ptg[:, 1], ptg[:, 2], res1)

    blm2 = np.zeros((nalm(lmax, lmax),))+0j
    blm2[0:blm.shape[0]] = blm
    res2 = np.zeros((nptg,))
    for i in range(nptg):
        rbeam = ducc0.sht.rotate_alm(blm2, lmax, ptg[i, 2], ptg[i, 0], ptg[i, 1])
        res2[i] = convolve(slm, rbeam, lmax).real
    _assert_close(res1, res2, 1.5e-13)


@pmp("lkmax", [(13, 13), (2, 1), (30, 15), (35, 2)])
@pmp("ncomp", [1, 3])
@pmp("separate", [True, False])
def test_against_convolution_2(lkmax, ncomp, separate):
    lmax, kmax = lkmax
    rng = np.random.default_rng(42)
    slm = random_alm(rng, lmax, lmax, ncomp)
    blm = random_alm(rng, lmax, kmax, ncomp)

    inter = ducc0.totalconvolve.Interpolator(slm, blm, separate, lmax, kmax,
                                             epsilon=1.5e-13, nthreads=2)
    nptg = 50
    ptg = np.zeros((nptg, 3))
    ptg[:, 0] = rng.uniform(0, np.pi, nptg)
    ptg[:, 1] = rng.uniform(0, 2*np.pi, nptg)
    ptg[:, 2] = rng.uniform(-np.pi, np.pi, nptg)

    res1 = inter.interpol(ptg)

    blm2 = np.zeros((ncomp, nalm(lmax, lmax)))+0j
    blm2[:, 0:blm.shape[1]] = blm
    res2 = np.zeros((ncomp, nptg))
    for c in range(ncomp):
        for i in range(nptg):
            rbeam = ducc0.sht.rotate_alm(blm2[c, :], lmax,
                                         ptg[i, 2], ptg[i, 0], ptg[i, 1])
            res2[c, i] = convolve(slm[c, :], rbeam, lmax).real
    if separate:
        _assert_close(res1, res2, 1.5e-13)
    else:
        _assert_close(res1[0, :], np.sum(res2, axis=0), 1.5e-13)


@pmp("lkmax", [(13, 13), (2, 1), (30, 15), (35, 2)])
@pmp("ncomp", [1, 3])
@pmp("separate", [True, False])
def test_against_convolution_2f(lkmax, ncomp, separate):
    lmax, kmax = lkmax
    rng = np.random.default_rng(42)
    slm = random_alm(rng, lmax, lmax, ncomp).astype("c8")
    blm = random_alm(rng, lmax, kmax, ncomp).astype("c8")

    inter = ducc0.totalconvolve.Interpolator_f(slm, blm, separate, lmax, kmax,
                                               epsilon=4e-6, nthreads=2)
    nptg = 50
    ptg = np.zeros((nptg, 3))
    ptg[:, 0] = rng.uniform(0, np.pi, nptg)
    ptg[:, 1] = rng.uniform(0, 2*np.pi, nptg)
    ptg[:, 2] = rng.uniform(-np.pi, np.pi, nptg)
    ptg = ptg.astype("f4")

    res1 = inter.interpol(ptg)

    blm2 = np.zeros((ncomp, nalm(lmax, lmax)))+0j
    blm2[:, 0:blm.shape[1]] = blm
    res2 = np.zeros((ncomp, nptg))
    for c in range(ncomp):
        for i in range(nptg):
            rbeam = ducc0.sht.rotate_alm(blm2[c, :], lmax,
                                         ptg[i, 2], ptg[i, 0], ptg[i, 1])
            res2[c, i] = convolve(slm[c, :], rbeam, lmax).real
    if separate:
        _assert_close(res1, res2, 4e-6)
    else:
        _assert_close(res1[0, :], np.sum(res2, axis=0), 4e-6)


@pmp("lkmax", [(13, 13), (20, 0), (2, 1), (30, 15), (35, 2)])
def test_adjointness(lkmax):
    lmax, kmax = lkmax
    rng = np.random.default_rng(42)
    slm = random_alm(rng, lmax, lmax, 1)[0, :]
    blm = random_alm(rng, lmax, kmax, 1)[0, :]
    nptg = 50
    ptg = rng.uniform(0., 1., nptg*3).reshape(nptg, 3)
    ptg[:, 0] *= np.pi
    ptg[:, 1] *= 2*np.pi
    ptg[:, 2] *= 2*np.pi
    conv = ducc0.totalconvolve.ConvolverPlan(lmax, kmax,
                                             epsilon=4e-6, nthreads=2)

    cube = np.empty((conv.Npsi(), conv.Ntheta(), conv.Nphi()))
    conv.getPlane(slm, blm, 0, cube[0:1])
    for mbeam in range(1, kmax+1):
        conv.getPlane(slm, blm, mbeam, cube[2*mbeam-1:2*mbeam+1])
    conv.prepPsi(cube)
    inter1 = np.empty(ptg.shape[0])
    conv.interpol(cube, 0, 0, ptg[:, 0], ptg[:, 1], ptg[:, 2], inter1)

    fake = rng.uniform(-0.5, 0.5, (ptg.shape[0],))
    cube2 = cube*0.
    conv.deinterpol(cube2, 0, 0, ptg[:, 0], ptg[:, 1], ptg[:, 2], fake)
    bla = slm*0.
    conv.deprepPsi(cube2)
    conv.updateSlm(bla, blm, 0, cube2[0:1])
    for mbeam in range(1, kmax+1):
        conv.updateSlm(bla, blm, mbeam, cube2[2*mbeam-1:2*mbeam+1])

    v1 = myalmdot(slm, bla, lmax)
    v2 = ducc0.misc.vdot(fake, inter1)
    _assert_close(v1, v2, 1e-13)


@pmp("lkmax", [(13, 13), (2, 1), (30, 15), (35, 2)])
@pmp("ncomp", [1, 3])
@pmp("separate", [True, False])
@pmp("single", [True, False])
def test_adjointness2(lkmax, ncomp, separate, single):
    lmax, kmax = lkmax
    rng = np.random.default_rng(42)
    slm = random_alm(rng, lmax, lmax, ncomp)
    blm = random_alm(rng, lmax, kmax, ncomp)
    nptg = 50
    ptg = rng.uniform(0., 1., nptg*3).reshape(nptg, 3)
    ptg[:, 0] *= np.pi
    ptg[:, 1] *= 2*np.pi
    ptg[:, 2] *= 2*np.pi
    if single:
        slm = slm.astype("c8")
        blm = blm.astype("c8")
        ptg = ptg.astype("f4")
        foo = ducc0.totalconvolve.Interpolator_f(slm, blm, separate, lmax, kmax,
                                                 epsilon=1e-5, nthreads=2)
    else:
        foo = ducc0.totalconvolve.Interpolator(slm, blm, separate, lmax, kmax,
                                               epsilon=1e-6, nthreads=2)
    inter1 = foo.interpol(ptg).astype("f8")
    ncomp2 = inter1.shape[0]
    fake = rng.uniform(-0.5, 0.5, (ncomp2, ptg.shape[0]))
    if single:
        fake = fake.astype("f4")
        foo2 = ducc0.totalconvolve.Interpolator_f(lmax, kmax, ncomp2, epsilon=1e-5,
                                                  nthreads=2)
    else:
        foo2 = ducc0.totalconvolve.Interpolator(lmax, kmax, ncomp2, epsilon=1e-6,
                                                nthreads=2)
    foo2.deinterpol(ptg.reshape((-1, 3)), fake)
    bla = foo2.getSlm(blm).astype("c16")
    v1 = np.sum([myalmdot(slm[c, :], bla[c, :], lmax)
                 for c in range(ncomp)])
    v2 = np.sum([ducc0.misc.vdot(fake[c, :], inter1[c, :]) for c in range(ncomp2)])
    epsilon = 1e-4 if single else 1e-12
    _assert_close(v1, v2, epsilon)
