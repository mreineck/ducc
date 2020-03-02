import interpol_ng
import numpy as np
import pysharp
import time

np.random.seed(42)

def nalm(lmax, mmax):
    return ((mmax+1)*(mmax+2))//2 + (mmax+1)*(lmax-mmax)


def deltabeam(lmax,kmax):
    beam=np.zeros(nalm(lmax, kmax))+0j
    for l in range(lmax+1):
        beam[l] = np.sqrt((2*l+1.)/(4*np.pi))
    return beam


lmax=2047
mmax=lmax
kmax=2  # doesn't make any sense for the beam we are using, but just for demonstration purposes ...


# get random sky a_lm
# the a_lm arrays follow the same conventions as those in healpy
slmT =      np.random.uniform(-1., 1., nalm(lmax, mmax)) \
       + 1j*np.random.uniform(-1., 1., nalm(lmax, mmax))
# make a_lm with m==0 real-valued
slmT[0:lmax+1].imag = 0.

# build beam a_lm (pencil beam for now)
blmT = deltabeam(lmax,kmax)

t0=time.time()
foo = interpol_ng.PyInterpolator(slmT,blmT,lmax, kmax, epsilon=1e-6, nthreads=2)
print("setup time: ",time.time()-t0)

# evaluate total convolution on a sufficiently resolved Gauss-Legendre grid
nth = lmax+1
nph = 2*mmax+1
ptg = np.zeros((nth,nph,3))
th, _ = np.polynomial.legendre.leggauss(nth)
th = np.arccos(-th)
ptg[:,:,0] = th.reshape((-1,1))
ptg[:,:,1] = (2*np.pi*np.arange(nph)/nph).reshape((1,-1))
ptg[:,:,2] = 0
t0=time.time()
bar=foo.interpol(ptg.reshape((-1,3))).reshape((nth,nph))
print("interpolation time: ", time.time()-t0)

# get a_lm back from interpolated array
job = pysharp.sharpjob_d()
job.set_triangular_alm_info(lmax, mmax)
job.set_gauss_geometry(nth, nph)
alm2 = job.map2alm(bar.reshape((-1,)))

#compare with original a_lm
import matplotlib.pyplot as plt
plt.plot(np.abs(alm2-slmT))
plt.suptitle("Deviations between original and reconstructed a_lm")
plt.show()
