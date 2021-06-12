import ducc0
import numpy as np
from time import time

rng = np.random.default_rng(48)


def nalm(lmax, mmax):
    return ((mmax+1)*(mmax+2))//2 + (mmax+1)*(lmax-mmax)


def _l2error(a, b):
    return np.sqrt(np.sum(np.abs(a-b)**2)/np.sum(np.abs(a)**2))


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


def synthesize(alm, lmax, spin, nrings, nphi, nthreads, geometry):
    return ducc0.sht.experimental.synthesis_2d(alm=alm, lmax=lmax, spin=spin, map=np.empty((alm.shape[0], nrings,nphi)), nthreads=nthreads, geometry=geometry)

def analyze(map, lmax, spin, nthreads, geometry):
    return ducc0.sht.experimental.analysis_2d(alm=np.empty((map.shape[0],nalm(lmax,lmax)),dtype=np.complex128), map=map, lmax=lmax, spin=spin, geometry=geometry, nthreads=nthreads)

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

    alm1 = random_alm(lmax, lmax, spin, ncomp)
    t0=time()
    blub = synthesize(alm1,lmax,spin,nrings,2*lmax+1, nthreads, geometry)
    blub2 = analyze(blub,lmax,spin, nthreads, geometry)
    print(time()-t0)
    print("L2 error after full round-trip:", _l2error(blub2,alm1))

nthr=16
for l0 in [4096]:
    for geometry in ["CC", "F1", "MW", "MWflip", "GL", "DH", "F2"]:
        for spin in [0,1,2]:
            test(lmax=l0, spin=spin, nthreads=nthr, geometry=geometry)
