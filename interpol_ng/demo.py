# Elementary demo for pysharp interface using a Gauss-Legendre grid
# I'm not sure I have a perfect equivalent for the DH grid(s) at the moment,
# since they apparently do not include the South Pole. The Clenshaw-Curtis
# and Fejer quadrature rules are very similar (see the documentation in
# sharp_geomhelpers.h). An exact analogon to DH can be added easily, I expect.

import interpol_ng
import numpy as np
import pysharp
import time
np.random.seed(42)

lmax=2047
mmax=lmax
kmax=2

def nalm(lmax, mmax):
    return ((mmax+1)*(mmax+2))//2 + (mmax+1)*(lmax-mmax)


def idx(l,m):
    return (m*((2*lmax+1)-m))//2 + l


def deltabeam(lmax,kmax):
    beam=np.zeros(nalm(lmax, kmax))+0j
    for l in range(lmax+1):
        beam[l] = np.sqrt((2*l+1.)/(4*np.pi))
    return beam


# get random a_lm
slmT = np.random.uniform(-1., 1., nalm(lmax, mmax)) + 1j*np.random.uniform(-1., 1., nalm(lmax,mmax))
# make a_lm with m==0 real-valued
slmT[0:lmax+1].imag = 0.

# build beam a_lm (pencil beam for now)
blmT = deltabeam(lmax,kmax)

t0=time.time()
foo = interpol_ng.PyInterpolator(slmT,blmT,lmax, kmax, 1e-5)
print("setup: ",time.time()-t0)

# evaluate total convolution on a sufficiently resolved Gauss-Legendre grid
nth = lmax+1
nph = 2*mmax+1
ptg = np.zeros((nth,nph,3))
th, wgt = np.polynomial.legendre.leggauss(nth)
th = np.arccos(th)
for it in range(nth):
    for ip in range(nph):
        ptg[it,ip,0] = th[it]
        ptg[it,ip,1] = 2*np.pi*ip/nph
        ptg[it,ip,2] = 0
t0=time.time()
bar=foo.interpol(ptg.reshape((-1,3))).reshape((nth,nph))
print("interpol: ", time.time()-t0)

# get a_lm back from interpolated array
job = pysharp.sharpjob_d()
job.set_triangular_alm_info(lmax, mmax)
job.set_gauss_geometry(nth, nph)

alm2 = job.map2alm(bar.reshape((-1,)))

import matplotlib.pyplot as plt
plt.plot(np.abs(alm2/slmT))
plt.show()
