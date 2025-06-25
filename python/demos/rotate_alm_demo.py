import ducc0
import numpy as np
import healpy as hp
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


def resize_alm(alm, lmax_in, mmax_in, lmax_out, mmax_out):
    res = np.zeros((alm.shape[0], nalm(lmax_out, mmax_out)), dtype=alm.dtype)
    ofs_in, ofs_out = 0, 0
    for m in range(min(mmax_in, mmax_out)+1):
        num = min(lmax_in, lmax_out) + 1 - m
        res[:, ofs_out:ofs_out+num] = alm[:, ofs_in:ofs_in+num]
        ofs_in += lmax_in+1-m
        ofs_out += lmax_out+1-m
    return res


def rotstress(rng, nthreads):
    phi, theta, psi = rng.uniform(-2*np.pi, 2*np.pi, (3,))
    lmax = rng.integers(500,1000+1)
    mmax_in = rng.integers(0,lmax+1)
    mmax_out = rng.integers(0,lmax+1)
    spin = rng.integers(0,5)
    ncomp = 1 if spin==0 else 2

    alm = random_alm(lmax, mmax_in, 0, ncomp, rng)
    t0=time()
    alm2 = ducc0.sht.rotate_alm(alm, lmax, phi, theta, psi, nthreads,
                                mmax_in=mmax_in, mmax_out=mmax_out)
    trot1 = time()-t0
    almfull = resize_alm(alm, lmax, mmax_in, lmax, lmax)
    t0=time()
    alm3 = ducc0.sht.rotate_alm(almfull, lmax, phi, theta, psi, nthreads,
                                mmax_in=lmax, mmax_out=lmax)
    trot2 = time()-t0
    alm4 = resize_alm(alm3, lmax, lmax, lmax, mmax_out)
    alm_hp = almfull.copy()
    t0=time()
    hp.rotate_alm(alm_hp, phi, theta, psi, lmax=lmax, mmax=lmax)
    trot3 = time()-t0
    print(lmax, mmax_in, mmax_out, trot3/trot1, trot3/trot2)
    alm_hp_clipped = resize_alm(alm_hp, lmax, lmax, lmax, mmax_out)
    if ducc0.misc.l2error(alm3,alm_hp) > 1e-12:
       raise RuntimeError("oops2 " + str(ducc0.misc.l2error(alm3,alm_hp)))
    if ducc0.misc.l2error(alm2,alm_hp_clipped) > 1e-12:
       raise RuntimeError("oops3 " + str(ducc0.misc.l2error(alm2,alm_hp_clipped)))


rng = np.random.default_rng(42)
for i in range(10000):
    rotstress(rng,8)
