import ducc0
import numpy as np
from time import time

rng = np.random.default_rng(48)


def nalm(lmax, mmax):
    return ((mmax+1)*(mmax+2))//2 + (mmax+1)*(lmax-mmax)


def random_alm(lmax, mmax, spin, ncomp):
    res = rng.uniform(-1., 1., (ncomp, nalm(lmax, mmax))) \
     + 1j*rng.uniform(-1., 1., (ncomp, nalm(lmax, mmax)))
    # make a_lm with m==0 real-valued
    res[:, 0:lmax+1].imag = 0.
    ofs=0
    for s in range(spin):
        res[:, ofs:ofs+spin-s] = 0.
        ofs += lmax+1-s
    return res


def test(lmax, geometry, spin, nthreads=1):
    print("testing lmax={}, spin={}, nthreads={}, geometry={}".format(lmax,spin,nthreads,geometry))
    ncomp = 1 if spin == 0 else 2

    nrings = lmax+1
    if geometry=="CC":
        nrings = lmax+2
    elif geometry=="DH":
        nrings = 2*lmax+2
    elif geometry=="F2":
        nrings = 2*lmax+1
    nphi=2*lmax+2

    alm = random_alm(lmax, lmax, spin, ncomp)
    map = np.empty((alm.shape[0], nrings,nphi))
    alm2 = np.empty((map.shape[0],nalm(lmax,lmax)),dtype=np.complex128)
    t0=time()
    map = ducc0.sht.experimental.synthesis_2d(alm=alm, lmax=lmax, spin=spin, map=map, nthreads=nthreads, geometry=geometry)
    alm2 = ducc0.sht.experimental.analysis_2d(alm=alm2, map=map, lmax=lmax, spin=spin, geometry=geometry, nthreads=nthreads)
    print(time()-t0)
    print("L2 error after full round-trip:", ducc0.misc.l2error(alm2,alm))
    print("L_inf error after full round-trip:", np.max(np.abs(alm2-alm)))

nthr=8
for l0 in [4095]:
    for geometry in ["CC", "F1", "MW", "MWflip", "GL", "DH", "F2"]:
        for spin in [0,1,2]:
            test(lmax=l0, spin=spin, nthreads=nthr, geometry=geometry)
