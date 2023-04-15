import ducc0
import numpy as np
import pytest
from mueller_convolver import MuellerConvolver


pmp = pytest.mark.parametrize


def nalm(lmax, mmax):
    return ((mmax+1)*(mmax+2))//2 + (mmax+1)*(lmax-mmax)


def random_alm(lmax, mmax, ncomp, rng):
    res = rng.uniform(-1., 1., (ncomp, nalm(lmax, mmax))) \
     + 1j*rng.uniform(-1., 1., (ncomp, nalm(lmax, mmax)))
    # make a_lm with m==0 real-valued
    res[:, 0:lmax+1].imag = 0.
    return res


# This compares Mueller matrices which are simply scaled identity matrices
# against the standard convolver
@pmp("lkmax", [(13, 9), (5, 1), (30, 15), (35, 2), (58, 0)])
@pmp("ncomp", [1, 3, 4])
@pmp("fct", [1. , -1., np.pi])
def test_trivial_mueller_matrix(fct, lkmax, ncomp):
    rng = np.random.default_rng(41)

    lmax, kmax = lkmax
    nptg = 100
    epsilon = 1e-4
    nthreads = 0

    slm = random_alm(lmax, lmax, ncomp, rng)
    blm = random_alm(lmax, kmax, ncomp, rng)

    ptg = np.empty((nptg,3))
    ptg[:, 0] = rng.random(nptg)*np.pi
    ptg[:, 1] = rng.random(nptg)*2*np.pi
    ptg[:, 2] = rng.random(nptg)*2*np.pi
    alpha = rng.random(nptg)*2*np.pi

    mueller = np.identity(4)*fct

    fullconv = MuellerConvolver(lmax=lmax, kmax=kmax, slm=slm, blm=blm,
                                mueller=mueller,
                                single_precision=False, epsilon=epsilon,
                                nthreads=nthreads)
    sig = fullconv.signal(ptg=ptg, alpha=alpha)
    ref_conv = ducc0.totalconvolve.Interpolator(slm, blm, False, lmax, kmax,
                                                epsilon=epsilon,
                                                nthreads=nthreads)
    ref_sig = ref_conv.interpol(ptg)[0]*fct
    np.testing.assert_allclose(sig,ref_sig,atol=1e-3)
