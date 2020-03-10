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


def compress_alm(alm,lmax):
    res = np.empty(2*len(alm)-lmax-1, dtype=np.float64)
    res[0:lmax+1] = alm[0:lmax+1].real
    res[lmax+1::2] = np.sqrt(2)*alm[lmax+1:].real
    res[lmax+2::2] = np.sqrt(2)*alm[lmax+1:].imag
    return res


def myalmdot(a1,a2,lmax,mmax,spin):
    return np.vdot(compress_alm(a1,lmax),compress_alm(np.conj(a2),lmax))


@pmp("lkmax", [(43,43),(2,1),(30,15),(125,2)])
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

@pmp("lkmax", [(43,43),(2,1),(30,15),(125,2)])
def test_adjointness(lkmax):
    lmax, kmax = lkmax
    slmT = random_alm(lmax, lmax)
    blmT = random_alm(lmax, kmax)
    nptg=100000
    ptg=np.random.uniform(0.,1.,nptg*3).reshape(nptg,3)
    ptg[:,0]*=np.pi
    ptg[:,1]*=2*np.pi
    ptg[:,2]*=2*np.pi
    foo = interpol_ng.PyInterpolator(slmT,blmT,lmax, kmax, epsilon=1e-6, nthreads=2)
    inter1=foo.interpol(ptg)
    fake = np.random.uniform(0.,1., ptg.shape[0])
    foo2 = interpol_ng.PyInterpolator(lmax, kmax, epsilon=1e-6, nthreads=2)
    foo2.deinterpol(ptg.reshape((-1,3)), fake)
    bla=foo2.getSlm(blmT)
    _assert_close(myalmdot(slmT, bla, lmax, lmax, 0), np.vdot(fake,inter1), 1e-12)
