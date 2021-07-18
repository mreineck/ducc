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


import ducc0.sht as sht
import ducc0
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_

pmp = pytest.mark.parametrize


def nalm(lmax, mmax):
    return ((mmax+1)*(mmax+2))//2 + (mmax+1)*(lmax-mmax)


def random_alm(lmax, mmax, spin, ncomp, rng):
    res = rng.uniform(-1., 1., (ncomp, nalm(lmax, mmax))) \
     + 1j*rng.uniform(-1., 1., (ncomp, nalm(lmax, mmax)))
    # make a_lm with m==0 real-valued
    res[:, 0:lmax+1].imag = 0.
    ofs=0
    for s in range(spin):
        res[:, ofs:ofs+spin-s] = 0.
        ofs += lmax+1-s
    return res


def compress_alm(alm, lmax):
    res = np.empty(2*len(alm)-lmax-1, dtype=np.float64)
    res[0:lmax+1] = alm[0:lmax+1].real
    res[lmax+1::2] = np.sqrt(2)*alm[lmax+1:].real
    res[lmax+2::2] = np.sqrt(2)*alm[lmax+1:].imag
    return res


def myalmdot(a1, a2, lmax):
    return ducc0.misc.vdot(compress_alm(a1, lmax), compress_alm((a2), lmax))


@pmp('params', [(511, 511, 512, 1024),
                (511, 2, 512, 5),
                (511, 0, 512, 1)])
def test_GL(params):
    lmax, mmax, nlat, nlon = params
    job = sht.sharpjob_d()
    nalm = ((mmax+1)*(mmax+2))//2 + (mmax+1)*(lmax-mmax)
    nalm_r = nalm*2-lmax-1
    rng = np.random.default_rng(np.random.SeedSequence(42))
    alm_r = rng.uniform(-1., 1., nalm_r)
    alm = np.empty(nalm, dtype=np.complex128)
    alm[0:lmax+1] = alm_r[0:lmax+1]
    alm[lmax+1:] = np.sqrt(0.5)*(alm_r[lmax+1::2] + 1j*alm_r[lmax+2::2])

    job.set_triangular_alm_info(lmax, mmax)
    job.set_gauss_geometry(nlat, nlon)
    alm2 = job.map2alm(job.alm2map(alm))
    assert_allclose(alm, alm2)


@pmp('params', [(511, 511, 1024, 1024),
                (511, 2, 1024, 5),
                (511, 0, 1024, 1)])
def test_fejer1(params):
    lmax, mmax, nlat, nlon = params
    job = sht.sharpjob_d()
    nalm = ((mmax+1)*(mmax+2))//2 + (mmax+1)*(lmax-mmax)
    nalm_r = nalm*2-lmax-1
    rng = np.random.default_rng(np.random.SeedSequence(42))
    alm_r = rng.uniform(-1., 1., nalm_r)
    alm = np.empty(nalm, dtype=np.complex128)
    alm[0:lmax+1] = alm_r[0:lmax+1]
    alm[lmax+1:] = np.sqrt(0.5)*(alm_r[lmax+1::2] + 1j*alm_r[lmax+2::2])

    job.set_triangular_alm_info(lmax, mmax)
    job.set_fejer1_geometry(nlat, nlon)
    alm2 = job.map2alm(job.alm2map(alm))
    assert_allclose(alm, alm2)


@pmp('params', [(511, 511, 1025, 1024),
                (511, 2, 1025, 5),
                (511, 0, 1025, 1)])
def test_cc(params):
    lmax, mmax, nlat, nlon = params
    job = sht.sharpjob_d()
    nalm = ((mmax+1)*(mmax+2))//2 + (mmax+1)*(lmax-mmax)
    nalm_r = nalm*2-lmax-1
    rng = np.random.default_rng(np.random.SeedSequence(42))
    alm_r = rng.uniform(-1., 1., nalm_r)
    alm = np.empty(nalm, dtype=np.complex128)
    alm[0:lmax+1] = alm_r[0:lmax+1]
    alm[lmax+1:] = np.sqrt(0.5)*(alm_r[lmax+1::2] + 1j*alm_r[lmax+2::2])

    job.set_triangular_alm_info(lmax, mmax)
    job.set_cc_geometry(nlat, nlon)
    alm2 = job.map2alm(job.alm2map(alm))
    assert_allclose(alm, alm2)


@pmp('params', [(511, 511, 1025, 1024),
                (511, 2, 1025, 5),
                (511, 0, 1025, 1)])
def test_fejer2(params):
    lmax, mmax, nlat, nlon = params
    job = sht.sharpjob_d()
    nalm = ((mmax+1)*(mmax+2))//2 + (mmax+1)*(lmax-mmax)
    nalm_r = nalm*2-lmax-1
    rng = np.random.default_rng(np.random.SeedSequence(42))
    alm_r = rng.uniform(-1., 1., nalm_r)
    alm = np.empty(nalm, dtype=np.complex128)
    alm[0:lmax+1] = alm_r[0:lmax+1]
    alm[lmax+1:] = np.sqrt(0.5)*(alm_r[lmax+1::2] + 1j*alm_r[lmax+2::2])

    job.set_triangular_alm_info(lmax, mmax)
    job.set_fejer2_geometry(nlat, nlon)
    alm2 = job.map2alm(job.alm2map(alm))
    assert_allclose(alm, alm2)


@pmp('params', [(511, 511, 1024, 1024),
                (511, 2, 1024, 5),
                (511, 0, 1024, 1)])
def test_dh(params):
    lmax, mmax, nlat, nlon = params
    job = sht.sharpjob_d()
    nalm = ((mmax+1)*(mmax+2))//2 + (mmax+1)*(lmax-mmax)
    nalm_r = nalm*2-lmax-1
    rng = np.random.default_rng(np.random.SeedSequence(42))
    alm_r = rng.uniform(-1., 1., nalm_r)
    alm = np.empty(nalm, dtype=np.complex128)
    alm[0:lmax+1] = alm_r[0:lmax+1]
    alm[lmax+1:] = np.sqrt(0.5)*(alm_r[lmax+1::2] + 1j*alm_r[lmax+2::2])

    job.set_triangular_alm_info(lmax, mmax)
    job.set_dh_geometry(nlat, nlon)
    alm2 = job.map2alm(job.alm2map(alm))
    assert_allclose(alm, alm2)


@pmp('geometry', ("CC", "F1", "MW", "MWflip", "GL", "DH", "F2"))
@pmp('spin', (0, 1, 2))
@pmp('nthreads', (1, 4))
@pmp('lmax', (2, 5, 11, 32, 600))
def test_2d_roundtrip(lmax, geometry, spin, nthreads):
    rng = np.random.default_rng(np.random.SeedSequence(42))
    ncomp = 1 if spin == 0 else 2

    nrings = lmax+1
    if geometry=="CC":
        nrings = lmax+2
    elif geometry=="DH":
        nrings = 2*lmax+2
    elif geometry=="F2":
        nrings = 2*lmax+1
    nphi=2*lmax+2

    alm = random_alm(lmax, lmax, spin, ncomp, rng)
    map = ducc0.sht.experimental.synthesis_2d(alm=alm, lmax=lmax, spin=spin, ntheta=nrings, nphi=nphi, nthreads=nthreads, geometry=geometry)
    alm2 = ducc0.sht.experimental.analysis_2d(map=map, lmax=lmax, spin=spin, geometry=geometry, nthreads=nthreads)
    assert_(ducc0.misc.l2error(alm2,alm)<1e-12)

    map = ducc0.sht.experimental.adjoint_analysis_2d(alm=alm, lmax=lmax, spin=spin, ntheta=nrings, nphi=nphi, nthreads=nthreads, geometry=geometry)
    alm2 = ducc0.sht.experimental.adjoint_synthesis_2d(map=map, lmax=lmax, spin=spin, geometry=geometry, nthreads=nthreads)
    assert_(ducc0.misc.l2error(alm2,alm)<1e-12)


@pmp('geometry', ("CC", "F1", "MW", "MWflip", "GL", "DH", "F2"))
@pmp('spin', (0, 1, 2))
@pmp('nthreads', (1, 4))
@pmp('lmax', (2, 5, 11, 32, 600))
def test_2d_adjoint(lmax, geometry, spin, nthreads):
    rng = np.random.default_rng(48)

    mmax = lmax
    ncomp = 1 if spin == 0 else 2

    nrings = lmax+1
    if geometry=="CC":
        nrings = lmax+2
    elif geometry=="DH":
        nrings = 2*lmax+2
    elif geometry=="F2":
        nrings = 2*lmax+1
    nphi=2*lmax+2

    alm0 = random_alm(lmax, mmax, spin, ncomp, rng)
    map0 = rng.uniform(0., 1., (alm0.shape[0], nrings, nphi))

    # test adjointness between synthesis and adjoint_synthesis
    map1 = ducc0.sht.experimental.synthesis_2d(alm=alm0, lmax=lmax, spin=spin, ntheta=nrings, nphi=nphi, nthreads=nthreads, geometry=geometry)
    v2 = np.sum([ducc0.misc.vdot(map0[i], map1[i]) for i in range(ncomp)])
    del map1
    alm1 = ducc0.sht.experimental.adjoint_synthesis_2d(lmax=lmax, spin=spin, map=map0, nthreads=nthreads, geometry=geometry)
    v1 = np.sum([myalmdot(alm0[i], alm1[i], lmax) for i in range(ncomp)])
    assert_(np.abs((v1-v2)/v1)<1e-10)

    # test adjointness between analysis and adjoint_analysis

    # naive version
    # I think this will only work if we somehow consider the whole torus
    # map1 = ducc0.sht.experimental.adjoint_analysis_2d(alm=alm0, lmax=lmax, spin=spin, ntheta=nrings, nphi=nphi, nthreads=nthreads, geometry=geometry)
    # v2 = np.sum([ducc0.misc.vdot(map0[i], map1[i]) for i in range(ncomp)])
    # del map1
    # alm1 = ducc0.sht.experimental.analysis_2d(lmax=lmax, spin=spin, map=map0, nthreads=nthreads, geometry=geometry)
    # v1 = np.sum([myalmdot(alm0[i], alm1[i], lmax) for i in range(ncomp)])
    # assert_(np.abs((v1-v2)/v1)<1e-12)

    # # create a band limited "map0"; so far the code only works for these maps
    # almx = random_alm(lmax, mmax, spin, ncomp, rng)
    # map0 = ducc0.sht.experimental.synthesis_2d(alm=almx, lmax=lmax, spin=spin, ntheta=nrings, nphi=nphi, nthreads=nthreads, geometry=geometry)
    # del almx

    # map1 = ducc0.sht.experimental.adjoint_analysis_2d(alm=alm0, lmax=lmax, spin=spin, ntheta=nrings, nphi=nphi, nthreads=nthreads, geometry=geometry)
    # v2 = np.sum([ducc0.misc.vdot(map0[i], map1[i]) for i in range(ncomp)])
    # del map1
    # alm1 = ducc0.sht.experimental.analysis_2d(lmax=lmax, spin=spin, map=map0, nthreads=nthreads, geometry=geometry)
    # v1 = np.sum([myalmdot(alm0[i], alm1[i], lmax) for i in range(ncomp)])
    # assert_(np.abs((v1-v2)/v1)<1e-10)

    # alternative version of the test taken from SSHT (test_forward_adjoint)
    map1 = ducc0.sht.experimental.synthesis_2d(alm=alm0, lmax=lmax, spin=spin, ntheta=nrings, nphi=nphi, nthreads=nthreads, geometry=geometry)
    alm1 = random_alm(lmax, mmax, spin, ncomp, rng)
    map0 = ducc0.sht.experimental.adjoint_analysis_2d(alm=alm1, lmax=lmax, spin=spin, ntheta=nrings, nphi=nphi, nthreads=nthreads, geometry=geometry)
    v2 = np.sum([ducc0.misc.vdot(map0[i], map1[i]) for i in range(ncomp)])
    del map1
    v1 = np.sum([myalmdot(alm0[i], alm1[i], lmax) for i in range(ncomp)])
    assert_(np.abs((v1-v2)/v1)<1e-10)
