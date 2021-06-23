import ducc0
import numpy as np
import random

rng = np.random.default_rng(48)


def nalm(lmax, mmax):
    return ((mmax+1)*(mmax+2))//2 + (mmax+1)*(lmax-mmax)


def compress_alm(alm, lmax):
    res = np.empty(2*len(alm)-lmax-1, dtype=np.float64)
    res[0:lmax+1] = alm[0:lmax+1].real
    res[lmax+1::2] = np.sqrt(2)*alm[lmax+1:].real
    res[lmax+2::2] = np.sqrt(2)*alm[lmax+1:].imag
    return res


def myalmdot(a1, a2, lmax):
    return np.vdot(compress_alm(a1, lmax).astype(np.float64), compress_alm((a2), lmax).astype(np.float64))


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


def test_random_analysis_2d(lmax_max, nthreads_max):
    geometries = ["CC", "F1", "MW", "MWflip", "GL", "DH", "F2"]
    geometry = random.choice(geometries)
    lmax = random.randint(0,lmax_max)
    mmax = random.randint(0,lmax)
    spin = random.randint(0,lmax)
    if random.randint(0,1) == 0:
        spin=0

    nrings = lmax+1
    if geometry=="CC":
        nrings = lmax+2
    elif geometry=="DH":
        nrings = 2*lmax+2
    elif geometry=="F2":
        nrings = 2*lmax+1
    nrings += random.randint(0,nrings)
    nphi = 2*lmax+1
    nphi += random.randint(0,nphi)
    nthreads = random.randint(1, nthreads_max)

    print("testing analysis: lmax={}, mmax={}, spin={}, nthreads={}, geometry={}, nrings={}, nphi={}".format(lmax,mmax,spin,nthreads,geometry,nrings, nphi))
    ncomp = 1 if spin == 0 else 2
    alm = random_alm(lmax, mmax, spin, ncomp)
    map = ducc0.sht.experimental.synthesis_2d(alm=alm, lmax=lmax, mmax=mmax, spin=spin, ntheta=nrings, nphi=nphi, nthreads=nthreads, geometry=geometry)
    alm2 = ducc0.sht.experimental.analysis_2d(map=map, lmax=lmax, mmax=mmax, spin=spin, geometry=geometry, nthreads=nthreads)
    err = ducc0.misc.l2error(alm2,alm)
    if err>1e-11:
        print("AAAAARGH: L2 error after full round-trip:", err)
        raise RuntimeError


def test_random_adjointness_2d(lmax_max, nthreads_max):
    geometries = ["CC", "F1", "MW", "MWflip", "GL", "DH", "F2"]
    geometry = random.choice(geometries)
    lmax = random.randint(0,lmax_max)
    spin = random.randint(0,lmax)
    if random.randint(0,1) == 0:
        spin=0

    nrings = random.randint(1, 3*lmax)
    nphi = random.randint(1, 3*lmax)
    nthreads = random.randint(1, nthreads_max)

    print("testing adjointness: lmax={}, spin={}, nthreads={}, geometry={}, nrings={}, nphi={}".format(lmax,spin,nthreads,geometry,nrings, nphi))
    ncomp = 1 if spin == 0 else 2
    alm0 = random_alm(lmax, lmax, spin, ncomp)
    map0 = np.random.uniform(0., 1., (alm0.shape[0], nrings,nphi))
    map1 = ducc0.sht.experimental.synthesis_2d(alm=alm0, lmax=lmax, spin=spin, ntheta=nrings, nphi=nphi, nthreads=nthreads, geometry=geometry)
    alm1 = ducc0.sht.experimental.adjoint_synthesis_2d(lmax=lmax, spin=spin, map=map0, nthreads=nthreads, geometry=geometry)
    v1 = np.sum([myalmdot(alm0[i], alm1[i], lmax) for i in range(ncomp)])
    v2 = np.sum([np.vdot(map0[i], map1[i]) for i in range(ncomp)])
    err = np.abs(v1-v2)/np.maximum(np.abs(v1), np.abs(v2))
    if err>1e-11:
        print("AAAAARGH: adjointness error:", err)
        raise RuntimeError


while True:
    test_random_analysis_2d(2047, 8)
    test_random_adjointness_2d(2047, 8)
