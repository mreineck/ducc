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


def synthesize_equidistant(alm, lmax, spin, ntheta, nphi,
                           ring_on_north_pole, ring_on_south_pole, nthreads):
    # go to CC grid wih minimal number of rings
    ntheta0 = lmax+2
    theta0 = np.arange(ntheta0)*np.pi/(ntheta0-1)
    leg = ducc0.sht.experimental.alm2leg(alm=alm, theta=theta0, lmax=lmax,
                                         spin=spin, nthreads=nthreads)
    # resample in theta direction to the desired grid
    leg = ducc0.sht.experimental.resample_from_CC(leg, ntheta,
        ring_on_north_pole, ring_on_south_pole, spin, nthreads)
    return ducc0.sht.experimental.leg2map(
        leg=leg,
        nphi=np.full(ntheta,np.uint64(nphi)),
        phi0=np.zeros(ntheta,dtype=np.float64),
        ringstart=nphi*np.arange(ntheta,dtype=np.uint64),
        nthreads=nthreads).reshape((alm.shape[0],ntheta,nphi))

def synthesize_direct(alm, lmax, spin, theta, nphi, nthreads):
    ntheta = theta.shape[0]
    leg = ducc0.sht.experimental.alm2leg(alm=alm, theta=theta, lmax=lmax,
                                         spin=spin, nthreads=nthreads)
    return ducc0.sht.experimental.leg2map(
        leg=leg,
        nphi=np.full(ntheta,np.uint64(nphi)),
        phi0=np.zeros(ntheta,dtype=np.float64),
        ringstart=nphi*np.arange(ntheta,dtype=np.uint64),
        nthreads=nthreads).reshape((alm.shape[0],ntheta,nphi))

def analyze_equidistant(map, lmax, spin,
                        ring_on_north_pole, ring_on_south_pole, nthreads):
    leg = ducc0.sht.experimental.map2leg(
        map=map.reshape((map.shape[0], -1)),
        nphi=np.full(map.shape[1],np.uint64(map.shape[2])),
        phi0=np.zeros(map.shape[1],dtype=np.float64),
        ringstart=map.shape[2]*np.arange(map.shape[1],dtype=np.uint64),
        mmax=lmax,
        nthreads=nthreads)
    # resample in theta to obtain a compatible CC grid which already is
    # correcly set up for anaylsis
    ntheta0=leg.shape[1]
    leg = ducc0.sht.experimental.resample_to_prepared_CC(leg, ring_on_north_pole, ring_on_south_pole, spin,1)
    theta = np.arange(leg.shape[1])*2*np.pi/(2*leg.shape[1]-2)
    return ducc0.sht.experimental.leg2alm(leg=leg, theta=theta, lmax=lmax, spin=spin, nthreads=nthreads) / map.shape[2]

def test(lmax, npi, spi, spin, nthreads=1):
    print("testing lmax={}, spin={}, nthreads={}, npi={}, spi={}".format(lmax,spin,nthreads,npi,spi))
    ncomp = 1 if spin == 0 else 2

    nrings = lmax+1
    if npi and spi:
        nrings = nrings+1
    nrings_full = 2*nrings-npi-spi

    theta=(2*np.pi/nrings_full)*(np.arange(nrings)+0.5*(1-npi))
    alm1 = random_alm(lmax, lmax, spin, ncomp)
    blub = synthesize_equidistant(alm1,lmax,spin,nrings,2*lmax+2, npi, spi, nthreads)
    blubx = synthesize_direct(alm1,lmax,spin,theta,2*lmax+2, nthreads)
    print("L2 error compared to direct synthesis:", _l2error(blub, blubx))
    blub2 = analyze_equidistant(blub,lmax,spin, npi, spi, nthreads)
    print("L2 error after full round-trip:", _l2error(blub2,alm1))

l0 = 1023
nthr=4
for l0 in [10, 1023]:
    for npi in [True, False]:
        for spi in [True, False]:
            for spin in [0,1,2]:
                test(lmax=l0, npi=npi, spi=spi, spin=spin, nthreads=nthr)
