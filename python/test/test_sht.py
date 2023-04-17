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
# Copyright(C) 2020-2023 Max-Planck-Society


import ducc0
import numpy as np
import pytest
from numpy.testing import assert_allclose

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


@pmp('geometry', ("CC", "F1", "MW", "MWflip", "GL", "DH", "F2"))
@pmp('spin', (0, 1, 2))
@pmp('nthreads', (1, 4))
@pmp('lmmax', ((2, 0), (2, 2), (5, 5), (11, 10), (11, 11), (32, 32), (200, 200)))
def test_2d_roundtrip(lmmax, geometry, spin, nthreads):
    rng = np.random.default_rng(np.random.SeedSequence(42))
    ncomp = 1 if spin == 0 else 2
    lmax, mmax = lmmax

    nrings = lmax+1
    if geometry=="CC":
        nrings = lmax+2
    elif geometry=="DH":
        nrings = 2*lmax+2
    elif geometry=="F2":
        nrings = 2*lmax+1
    nphi=2*mmax+2

    alm = random_alm(lmax, mmax, spin, ncomp, rng)
    map = ducc0.sht.experimental.synthesis_2d(alm=alm, lmax=lmax, mmax=mmax, spin=spin, ntheta=nrings, nphi=nphi, nthreads=nthreads, geometry=geometry)
    alm2 = ducc0.sht.experimental.analysis_2d(map=map, lmax=lmax, mmax=mmax, spin=spin, geometry=geometry, nthreads=nthreads)
    assert_allclose(ducc0.misc.l2error(alm2,alm), 0, atol=1e-12)

    map = ducc0.sht.experimental.adjoint_analysis_2d(alm=alm, lmax=lmax, mmax=mmax, spin=spin, ntheta=nrings, nphi=nphi, nthreads=nthreads, geometry=geometry)
    alm2 = ducc0.sht.experimental.adjoint_synthesis_2d(map=map, lmax=lmax, mmax=mmax, spin=spin, geometry=geometry, nthreads=nthreads)
    assert_allclose(ducc0.misc.l2error(alm2,alm), 0, atol=1e-12)

    job = ducc0.sht.sharpjob_d()
    job.set_triangular_alm_info(lmax, mmax)
    if geometry=="CC":
        job.set_cc_geometry(nrings, nphi)
    elif geometry=="DH":
        job.set_dh_geometry(nrings, nphi)
    elif geometry=="F1":
        job.set_fejer1_geometry(nrings, nphi)
    elif geometry=="F2":
        job.set_fejer2_geometry(nrings, nphi)
    elif geometry=="MW":
        job.set_mw_geometry(nrings, nphi)
    elif geometry=="GL":
        job.set_gauss_geometry(nrings, nphi)
    else:
        return

    if spin == 0:
        alm2 = job.map2alm(job.alm2map(alm[0])).reshape((1,-1))
    else:
        alm2 = job.map2alm_spin(job.alm2map_spin(alm, spin), spin)
    assert_allclose(alm, alm2)


@pmp('geometry', ("CC", "F1", "MW", "MWflip", "GL", "DH", "F2"))
@pmp('spin', (0, 1, 2))
@pmp('nthreads', (1, 4))
@pmp('lmmax', ((2, 0), (2, 2), (5, 5), (11, 10), (11, 11), (32, 32), (200, 200)))
def test_2d_adjoint(lmmax, geometry, spin, nthreads):
    rng = np.random.default_rng(48)

    lmax, mmax = lmmax
    ncomp = 1 if spin == 0 else 2

    nrings = lmax+1
    if geometry=="CC":
        nrings = lmax+2
    elif geometry=="DH":
        nrings = 2*lmax+2
    elif geometry=="F2":
        nrings = 2*lmax+1
    nphi=2*mmax+2

    alm0 = random_alm(lmax, mmax, spin, ncomp, rng)
    map0 = rng.uniform(0., 1., (alm0.shape[0], nrings, nphi))

    # test adjointness between synthesis and adjoint_synthesis
    map1 = ducc0.sht.experimental.synthesis_2d(alm=alm0, lmax=lmax, mmax=mmax, spin=spin, ntheta=nrings, nphi=nphi, nthreads=nthreads, geometry=geometry)
    v2 = np.sum([ducc0.misc.vdot(map0[i], map1[i]) for i in range(ncomp)])
    del map1
    alm1 = ducc0.sht.experimental.adjoint_synthesis_2d(lmax=lmax, mmax=mmax, spin=spin, map=map0, nthreads=nthreads, geometry=geometry)
    v1 = np.sum([myalmdot(alm0[i], alm1[i], lmax) for i in range(ncomp)])
    assert_allclose(v1, v2, rtol=1e-10)

    if spin > 0:
        # test adjointness between synthesis and adjoint_synthesis (gradient only)
        map1 = ducc0.sht.experimental.synthesis_2d(alm=alm0[:1], lmax=lmax, mmax=mmax, spin=spin, ntheta=nrings, nphi=nphi, nthreads=nthreads, geometry=geometry, mode="GRAD_ONLY")
        v2 = np.sum([ducc0.misc.vdot(map0[i], map1[i]) for i in range(ncomp)])
        # compare grad-only alm2map with full alm2map where curl is set to zero
        almx = alm0.copy()
        almx[1] = 0
        map1b = ducc0.sht.experimental.synthesis_2d(alm=almx, lmax=lmax, mmax=mmax, spin=spin, ntheta=nrings, nphi=nphi, nthreads=nthreads, geometry=geometry)
        assert_allclose(ducc0.misc.l2error(map1, map1b), 0, atol=1e-10)
        del almx, map1
        alm1 = ducc0.sht.experimental.adjoint_synthesis_2d(lmax=lmax, mmax=mmax, spin=spin, map=map0, nthreads=nthreads, geometry=geometry, mode="GRAD_ONLY")
        v1 = np.sum([myalmdot(alm0[i], alm1[i], lmax) for i in range(1)])
        assert_allclose(np.abs((v1-v2)/v1), 0, atol=1e-10)

    # test adjointness between analysis and adjoint_analysis

    # naive version
    # I think this will only work if we somehow consider the whole torus
    # map1 = ducc0.sht.experimental.adjoint_analysis_2d(alm=alm0, lmax=lmax, spin=spin, ntheta=nrings, nphi=nphi, nthreads=nthreads, geometry=geometry)
    # v2 = np.sum([ducc0.misc.vdot(map0[i], map1[i]) for i in range(ncomp)])
    # del map1
    # alm1 = ducc0.sht.experimental.analysis_2d(lmax=lmax, spin=spin, map=map0, nthreads=nthreads, geometry=geometry)
    # v1 = np.sum([myalmdot(alm0[i], alm1[i], lmax) for i in range(ncomp)])
    # assert_allclose(v1, v2, rtol=1e-12)

    # # create a band limited "map0"; so far the code only works for these maps
    # almx = random_alm(lmax, mmax, spin, ncomp, rng)
    # map0 = ducc0.sht.experimental.synthesis_2d(alm=almx, lmax=lmax, spin=spin, ntheta=nrings, nphi=nphi, nthreads=nthreads, geometry=geometry)
    # del almx

    # map1 = ducc0.sht.experimental.adjoint_analysis_2d(alm=alm0, lmax=lmax, spin=spin, ntheta=nrings, nphi=nphi, nthreads=nthreads, geometry=geometry)
    # v2 = np.sum([ducc0.misc.vdot(map0[i], map1[i]) for i in range(ncomp)])
    # del map1
    # alm1 = ducc0.sht.experimental.analysis_2d(lmax=lmax, spin=spin, map=map0, nthreads=nthreads, geometry=geometry)
    # v1 = np.sum([myalmdot(alm0[i], alm1[i], lmax) for i in range(ncomp)])
    # assert_allclose(v1, v2, rtol=1e-10)

    # alternative version of the test taken from SSHT (test_forward_adjoint)
    map1 = ducc0.sht.experimental.synthesis_2d(alm=alm0, lmax=lmax, mmax=mmax, spin=spin, ntheta=nrings, nphi=nphi, nthreads=nthreads, geometry=geometry)
    alm1 = random_alm(lmax, mmax, spin, ncomp, rng)
    map0 = ducc0.sht.experimental.adjoint_analysis_2d(alm=alm1, lmax=lmax, mmax=mmax, spin=spin, ntheta=nrings, nphi=nphi, nthreads=nthreads, geometry=geometry)
    v2 = np.sum([ducc0.misc.vdot(map0[i], map1[i]) for i in range(ncomp)])
    del map1
    v1 = np.sum([myalmdot(alm0[i], alm1[i], lmax) for i in range(ncomp)])
    assert_allclose(v1, v2, rtol=1e-10)


@pmp('spin', (0, 1, 2))
@pmp('nthreads', (1, 4))
@pmp('lmax', (2, 5, 32, 256))
@pmp('mmaxhalf', (False, True))
@pmp('nside', (2, 5, 128))
@pmp('theta_interpol', (False, True))
def test_healpix_adjoint(lmax, nside, spin, mmaxhalf, theta_interpol, nthreads):
    rng = np.random.default_rng(48)

    mmax = lmax//2 if mmaxhalf else lmax
    ncomp = 1 if spin == 0 else 2

    alm0 = random_alm(lmax, mmax, spin, ncomp, rng)
    map0 = rng.uniform(0., 1., (alm0.shape[0], 12*nside**2))

    base = ducc0.healpix.Healpix_Base(nside, "RING")
    geom = base.sht_info()

    # test adjointness between synthesis and adjoint_synthesis
    map1 = ducc0.sht.experimental.synthesis(alm=alm0, lmax=lmax, spin=spin, nthreads=nthreads, mmax=mmax, theta_interpol=theta_interpol, **geom)
    v2 = np.sum([ducc0.misc.vdot(map0[i], map1[i]) for i in range(ncomp)])
    del map1
    alm1 = ducc0.sht.experimental.adjoint_synthesis(lmax=lmax, spin=spin, map=map0, nthreads=nthreads, mmax=mmax, theta_interpol=theta_interpol, **geom)
    v1 = np.sum([myalmdot(alm0[i], alm1[i], lmax) for i in range(ncomp)])
    assert_allclose(v1, v2, rtol=1e-10)

    if spin > 0:
        map1 = ducc0.sht.experimental.synthesis(alm=alm0[:1], lmax=lmax, spin=spin, nthreads=nthreads, mmax=mmax, mode="GRAD_ONLY", theta_interpol=theta_interpol, **geom)
        v2 = np.sum([ducc0.misc.vdot(map0[i], map1[i]) for i in range(ncomp)])
        del map1
        alm1 = ducc0.sht.experimental.adjoint_synthesis(lmax=lmax, spin=spin, map=map0, nthreads=nthreads, mmax=mmax, mode="GRAD_ONLY", theta_interpol=theta_interpol, **geom)
        v1 = np.sum([myalmdot(alm0[i], alm1[i], lmax) for i in range(1)])
        assert_allclose(np.abs((v1-v2)/v1), 0, atol=1e-10)


@pmp("lmax", tuple(range(0,70,3)))
@pmp("nthreads", (0,1,2))
def test_rotation(lmax, nthreads):
    rng = np.random.default_rng(42)
    phi, theta, psi = rng.uniform(-2*np.pi, 2*np.pi, (3,))

    alm = random_alm(lmax, lmax, 0, 1, rng)[0,:]
    alm2 = ducc0.sht.rotate_alm(alm, lmax, phi, theta, psi, nthreads)
    alm2 = ducc0.sht.rotate_alm(alm2, lmax, -psi, -theta, -phi, nthreads)
    assert_allclose(ducc0.misc.l2error(alm,alm2), 0, atol=1e-12)
    alm = alm.astype(np.complex64)
    alm2 = ducc0.sht.rotate_alm(alm, lmax, phi, theta, psi, nthreads)
    alm2 = ducc0.sht.rotate_alm(alm2, lmax, -psi, -theta, -phi, nthreads)
    assert_allclose(ducc0.misc.l2error(alm,alm2), 0, atol=1e-6)


@pmp('spin', (0, 2))
@pmp('nthreads', (1, 4))
@pmp('nside', (32, 64))
@pmp('ntrans', (1, 8))
def test_healpix_inverse(nside, spin, ntrans, nthreads):
    rng = np.random.default_rng(48)

    lmax = int(2.5*nside)
    mmax = lmax
    ncomp = 1 if spin == 0 else 2

    alm0 = random_alm(lmax, mmax, spin, ntrans*ncomp, rng)
    alm0 = alm0.reshape((ntrans, ncomp, -1))

    base = ducc0.healpix.Healpix_Base(nside, "RING")
    geom = base.sht_info()

    map = ducc0.sht.experimental.synthesis(alm=alm0, lmax=lmax, spin=spin, nthreads=nthreads, **geom)
    alm1 = ducc0.sht.experimental.pseudo_analysis(map=map, lmax=lmax, spin=spin, nthreads=nthreads, epsilon=1e-8, maxiter=20, **geom)[0]

    assert_allclose(ducc0.misc.l2error(alm0,alm1), 0, atol=1e-6)


@pmp('spin', (0, 1, 2))
@pmp('nthreads', (1, 4))
@pmp('npix', (33, 200))
@pmp('lmmax', ((10, 8), (100,100)))
def test_adjointness_general(lmmax, npix, spin, nthreads):
    rng = np.random.default_rng(48)

    lmax, mmax = lmmax
    epsilon = 1e-5
    ncomp = 1 if spin == 0 else 2
    slm1 = random_alm(lmax, mmax, spin, ncomp, rng)
    loc = rng.uniform(0., 1., (npix,2))
    loc[:, 0] *= np.pi
    loc[:, 1] *= 2*np.pi
    points2 = rng.uniform(-0.5, 0.5, (loc.shape[0],ncomp)).T
    points1 = ducc0.sht.experimental.synthesis_general(lmax=lmax, mmax=mmax, alm=slm1, loc=loc, spin=spin, epsilon=epsilon, nthreads=nthreads)
    slm2 = ducc0.sht.experimental.adjoint_synthesis_general(lmax=lmax, mmax=mmax, map=points2, loc=loc, spin=spin, epsilon=epsilon, nthreads=nthreads)
    v1 = np.sum([myalmdot(slm1[c, :], slm2[c, :], lmax)
                for c in range(ncomp)])
    v2 = ducc0.misc.vdot(points2.real, points1.real) + ducc0.misc.vdot(points2.imag, points1.imag) 
    assert_allclose(v1, v2, rtol=1e-9)

    if spin > 0:
        points1 = ducc0.sht.experimental.synthesis_general(lmax=lmax, mmax=mmax, alm=slm1[:1], loc=loc, spin=spin, epsilon=epsilon, nthreads=nthreads, mode="GRAD_ONLY")
        slm2 = ducc0.sht.experimental.adjoint_synthesis_general(lmax=lmax, mmax=mmax, map=points2, loc=loc, spin=spin, epsilon=epsilon, nthreads=nthreads, mode="GRAD_ONLY")
        v1 = np.sum([myalmdot(slm1[c, :], slm2[c, :], lmax)
                    for c in range(1)])
        v2 = ducc0.misc.vdot(points2.real, points1.real) + ducc0.misc.vdot(points2.imag, points1.imag) 
        assert_allclose(v1, v2, rtol=1e-9)

@pmp('spin', (0, 1, 2))
@pmp('nthreads', (1, 4))
@pmp('nside', (32, ))
@pmp('lmmax', ((10, 8), (100,100)))
def test_synthesis_general(lmmax, nside, spin, nthreads):
    rng = np.random.default_rng(48)

    lmax, mmax = lmmax
    epsilon = 1e-7
    ncomp = 1 if spin == 0 else 2
    slm = random_alm(lmax, mmax, spin, ncomp, rng)
    base = ducc0.healpix.Healpix_Base(nside, "RING")
    geom = base.sht_info()
    loc = base.pix2ang(pix=np.arange(base.npix()), nthreads=nthreads)
    v1 = ducc0.sht.experimental.synthesis_general(lmax=lmax, mmax=mmax, alm=slm, loc=loc, spin=spin, epsilon=epsilon, nthreads=nthreads)
    v2 = ducc0.sht.experimental.synthesis(alm=slm, lmax=lmax, mmax=mmax, spin=spin, nthreads=nthreads, **geom)
    assert_allclose(ducc0.misc.l2error(v1,v2), 0, atol=epsilon)
