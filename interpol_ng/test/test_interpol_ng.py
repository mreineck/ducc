import numpy as np
import pytest
from numpy.testing import assert_
import interpol_ng
import pysharp

pmp = pytest.mark.parametrize


def _l2error(a, b):
    return np.sqrt(np.sum(np.abs(a-b)**2)/np.sum(np.abs(a)**2))


def _assert_close(a, b, epsilon):
    err = _l2error(a, b)
    if (err >= epsilon):
        print("Error: {} > {}".format(err, epsilon))
    assert_(err<epsilon)


def nalm(lmax, mmax):
    return ((mmax+1)*(mmax+2))//2 + (mmax+1)*(lmax-mmax)


def random_alm(lmax, mmax):
    res = np.random.uniform(-1., 1., nalm(lmax, mmax)) \
     + 1j*np.random.uniform(-1., 1., nalm(lmax, mmax))
    # make a_lm with m==0 real-valued
    res[0:lmax+1].imag = 0.
    return res


def convolve(alm1, alm2, lmax):
    job = pysharp.sharpjob_d()
    job.set_triangular_alm_info(lmax, lmax)
    job.set_gauss_geometry(lmax+1, 2*lmax+1)
    map = job.alm2map(alm1)*job.alm2map(alm2)
    job.set_triangular_alm_info(0,0)
    return job.map2alm(map)[0]*np.sqrt(4*np.pi)


@pmp("lkmax", [(43,43),(2,1),(30,15),(512,2)])
def test_against_convolution(lkmax):
    lmax, kmax = lkmax
    slmT = random_alm(lmax, lmax)
    blmT = random_alm(lmax, kmax)

    inter = interpol_ng.PyInterpolator(slmT, blmT, lmax, kmax, epsilon=1e-8,
                                       nthreads=2)
    nptg = 50
    ptg = np.zeros((nptg,3))
    ptg[:,0] = np.random.uniform(0, np.pi, nptg)
    ptg[:,1] = np.random.uniform(0, 2*np.pi, nptg)
    ptg[:,2] = np.random.uniform(-np.pi, np.pi, nptg)

    res1 = inter.interpol(ptg)

    res2 = np.zeros(nptg)
    blmT2 = np.zeros(nalm(lmax,lmax))+0j
    blmT2[0:blmT.shape[0]] = blmT
    for i in range(nptg):
        rbeam=interpol_ng.rotate_alm(blmT2, lmax, ptg[i,2],ptg[i,0],ptg[i,1])
        res2[i] = convolve(slmT, rbeam, lmax).real
    _assert_close(res1, res2, 1e-7)
