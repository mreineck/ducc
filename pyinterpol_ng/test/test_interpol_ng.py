import numpy as np
import pytest
from numpy.testing import assert_
import cxxbase1.pyinterpol_ng as pyinterpol_ng
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


def random_alm(lmax, mmax, ncomp):
    res = np.random.uniform(-1., 1., (nalm(lmax, mmax), ncomp)) \
     + 1j*np.random.uniform(-1., 1., (nalm(lmax, mmax), ncomp))
    # make a_lm with m==0 real-valued
    res[0:lmax+1,:].imag = 0.
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


@pmp("lkmax", [(13,13),(2,1),(30,15),(35,2)])
@pmp("ncomp", [1, 3])
@pmp("separate", [True, False])
def test_against_convolution(lkmax, ncomp, separate):
    lmax, kmax = lkmax
    slm = random_alm(lmax, lmax, ncomp)
    blm = random_alm(lmax, kmax, ncomp)

    inter = pyinterpol_ng.PyInterpolator(slm, blm, separate, lmax, kmax,
                                         epsilon=1e-8, nthreads=2)
    nptg = 50
    ptg = np.zeros((nptg,3))
    ptg[:,0] = np.random.uniform(0, np.pi, nptg)
    ptg[:,1] = np.random.uniform(0, 2*np.pi, nptg)
    ptg[:,2] = np.random.uniform(-np.pi, np.pi, nptg)

    res1 = inter.interpol(ptg)

    blm2 = np.zeros((nalm(lmax,lmax), ncomp))+0j
    blm2[0:blm.shape[0],:] = blm
    res2 = np.zeros((nptg, ncomp))
    for c in range(ncomp):
        for i in range(nptg):
            rbeam=pyinterpol_ng.rotate_alm(blm2[:,c], lmax, ptg[i,2],ptg[i,0],ptg[i,1])
            res2[i,c] = convolve(slm[:,c], rbeam, lmax).real
    if separate:
        _assert_close(res1, res2, 1e-7)
    else:
        _assert_close(res1[:,0], np.sum(res2,axis=1), 1e-7)

@pmp("lkmax", [(13,13),(2,1),(30,15),(35,2)])
@pmp("ncomp", [1, 3])
@pmp("separate", [True, False])
def test_adjointness(lkmax, ncomp, separate):
    lmax, kmax = lkmax
    slm = random_alm(lmax, lmax, ncomp)
    blm = random_alm(lmax, kmax, ncomp)
    nptg=100000
    ptg=np.random.uniform(0.,1.,nptg*3).reshape(nptg,3)
    ptg[:,0]*=np.pi
    ptg[:,1]*=2*np.pi
    ptg[:,2]*=2*np.pi
    foo = pyinterpol_ng.PyInterpolator(slm,blm,separate,lmax, kmax, epsilon=1e-6, nthreads=2)
    inter1=foo.interpol(ptg)
    ncomp2 = inter1.shape[1]
    fake = np.random.uniform(0.,1., (ptg.shape[0],ncomp2))
    foo2 = pyinterpol_ng.PyInterpolator(lmax, kmax, ncomp2, epsilon=1e-6, nthreads=2)
    foo2.deinterpol(ptg.reshape((-1,3)), fake)
    bla=foo2.getSlm(blm)
    v1 = np.sum([myalmdot(slm[:,c], bla[:,c], lmax, lmax, 0) for c in range(ncomp)])
    v2 = np.sum([np.vdot(fake[:,c],inter1[:,c]) for c in range(ncomp2)])
    _assert_close(v1, v2, 1e-12)
