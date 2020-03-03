import interpol_ng
import numpy as np
import pysharp
import time
import matplotlib.pyplot as plt

np.random.seed(48)

def nalm(lmax, mmax):
    return ((mmax+1)*(mmax+2))//2 + (mmax+1)*(lmax-mmax)


def random_alm(lmax, mmax):
    res = np.random.uniform(-1., 1., nalm(lmax, mmax)) \
     + 1j*np.random.uniform(-1., 1., nalm(lmax, mmax))
    # make a_lm with m==0 real-valued
    res[0:lmax+1].imag = 0.
    return res


def deltabeam(lmax,kmax):
    beam=np.zeros(nalm(lmax, kmax))+0j
    for l in range(lmax+1):
        beam[l] = np.sqrt((2*l+1.)/(4*np.pi))
    return beam

def convolve(alm1, alm2, lmax):
    job = pysharp.sharpjob_d()
    job.set_triangular_alm_info(lmax, lmax)
    job.set_gauss_geometry(lmax+1, 2*lmax+1)
    map = job.alm2map(alm1)*job.alm2map(alm2)
    job.set_triangular_alm_info(0,0)
    return job.map2alm(map)[0]*np.sqrt(4*np.pi)

lmax=10
mmax=lmax
kmax=lmax


# get random sky a_lm
# the a_lm arrays follow the same conventions as those in healpy
slmT = random_alm(lmax, mmax)

# build beam a_lm
blmT = random_alm(lmax, mmax)

t0=time.time()
# build interpolator object for slmT and blmT
foo = interpol_ng.PyInterpolator(slmT,blmT,lmax, kmax, epsilon=1e-6, nthreads=2)
print("setup time: ",time.time()-t0)
nth = 2*lmax+1
nph = 2*mmax+1

ptg = np.zeros((nth,nph,3))
ptg[:,:,0] = (np.pi*np.arange(nth)/(nth-1)).reshape((-1,1))
ptg[:,:,1] = (2*np.pi*np.arange(nph)/nph).reshape((1,-1))
ptg[:,:,2] = np.pi*0.2
t0=time.time()
# do the actual interpolation
bar=foo.interpol(ptg.reshape((-1,3))).reshape((nth,nph))
print("interpolation time: ", time.time()-t0)
plt.subplot(2,2,1)
plt.imshow(bar.reshape((nth,nph)))
bar2 = np.zeros((nth,nph))
for ith in range(nth):
    for iph in range(nph):
        rbeam=interpol_ng.rotate_alm(blmT, lmax, ptg[ith,iph,2],ptg[ith,iph,0],ptg[ith,iph,1])
        bar2[ith,iph] = convolve(slmT, rbeam, lmax).real
plt.subplot(2,2,2)
plt.imshow(bar2)
plt.subplot(2,2,3)
plt.imshow((bar2-bar.reshape((nth,nph))))
plt.show()
