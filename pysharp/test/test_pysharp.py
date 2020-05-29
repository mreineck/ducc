import ducc_0_1.pysharp as pysharp
import numpy as np
import math
import pytest
from numpy.testing import assert_equal, assert_allclose

pmp = pytest.mark.parametrize

@pmp('params', [(511, 511, 512, 1024),
                (511, 2, 512, 5),
                (511, 0, 512, 1)])
def test_GL(params):
    lmax, mmax, nlat, nlon = params
    job = pysharp.sharpjob_d()
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
    job = pysharp.sharpjob_d()
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

@pmp('params', [(511, 511, 1024, 1024),
                (511, 2, 1024, 5),
                (511, 0, 1024, 1)])
def test_dh(params):
    lmax, mmax, nlat, nlon = params
    job = pysharp.sharpjob_d()
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
