import numpy as np
import ducc0
from time import time


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


def synthesis_spin1_via_spin0(alm, lmax, mmax, ntheta, nphi, nthreads=1):
    t0 = time()
    almx = ducc0.sht.experimental.spin1to0(alm, lmax, mmax, nthreads)
    print("spin1to0:", time()-t0)
    t0 = time()
    res = np.vstack([ducc0.sht.synthesis_2d(alm=ax.reshape((1,-1)), lmax=lmax+1, mmax=mmax, spin=0, geometry="GL", ntheta=ntheta, nphi=nphi, nthreads=nthreads) for ax in almx])
    print("time for spin-0 SHTs:", time()-t0)
    sintheta = np.sin(ducc0.misc.GL_thetas(ntheta)).reshape((1,-1,1))
    res /= -sintheta
    return res


def analysis_spin1_via_spin0(map, lmax, mmax, nthreads=1):
    ntheta, nphi = map.shape[1:]
    sintheta = np.sin(ducc0.misc.GL_thetas(ntheta))
    wgt = ducc0.misc.GL_weights(ntheta, nphi)
    map2 = map*(wgt/sintheta).reshape((1,-1,1))
    t0 = time()
    res = np.vstack([ducc0.sht.adjoint_synthesis_2d(map=mp.reshape((1,ntheta,nphi)), lmax=lmax+1, mmax=mmax, spin=0, geometry="GL", nthreads=nthreads) for mp in map2])
    print("time for spin-0 SHTs:", time()-t0)
    t0 = time()
    res = ducc0.sht.experimental.spin0to1(res, lmax+1, mmax, nthreads)
    print("spin0to1:", time()-t0)
    return res


def compare_spin1 (lmax, mmax, nthreads=1):
    print("lmax =",lmax, "mmax =", mmax)
    rng = np.random.default_rng(42)
    alm_ref = random_alm(lmax, mmax, 1, 2, rng)

    ntheta, nphi= lmax+1, 2*mmax+1
    t0 = time()
    map_ref = ducc0.sht.synthesis_2d(alm=alm_ref, lmax=lmax, mmax=mmax, spin=1, geometry="GL", ntheta=ntheta, nphi=nphi, nthreads=nthreads)
    print("time for spin-1 SHT:", time()-t0)

    map_test = synthesis_spin1_via_spin0(alm_ref, lmax, mmax, ntheta, nphi, nthreads=nthreads)
    print(ducc0.misc.l2error(map_ref, map_test))

    alm_test = analysis_spin1_via_spin0(map_ref, lmax, mmax, nthreads=nthreads)
    print(ducc0.misc.l2error(alm_ref,alm_test))


ducc0.misc.preallocate_memory(8)
compare_spin1(4,4)
compare_spin1(2047,2047, nthreads=8)
for i in range(20):
    lmax = np.random.randint(2,5000)
    mmax = np.random.randint(2,lmax)
    compare_spin1(lmax, mmax)
