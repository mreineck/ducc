import interpol_ng
import numpy as np
import pysharp
import time
import matplotlib.pyplot as plt

np.random.seed(20)

def nalm(lmax, mmax):
    return ((mmax+1)*(mmax+2))//2 + (mmax+1)*(lmax-mmax)


def random_alm(lmax, mmax):
    res = np.random.uniform(-1., 1., nalm(lmax, mmax)) \
     + 1j*np.random.uniform(-1., 1., nalm(lmax, mmax))
    # make a_lm with m==0 real-valued
    res[0:lmax+1].imag = 0.
    return res

def compress_alm(alm,lmax):
    res = np.empty(2*len(alm)-lmax-1, dtype=np.float64)
    res[0:lmax+1] = alm[0:lmax+1].real
    res[lmax+1::2] = np.sqrt(2)*alm[lmax+1:].real
    res[lmax+2::2] = np.sqrt(2)*alm[lmax+1:].imag
    return res


def myalmdot(a1,a2,lmax,mmax,spin):
    return np.vdot(compress_alm(a1,lmax),compress_alm(a2,lmax))
def theta_extend(arr, spin):
    nth, nph = arr.shape
    arr2 = np.zeros(((nth-1)*2,nph))
    arr2[0:nth,:] = arr
    arr2[nth:,:] = np.roll(arr[nth-2:0:-1,:],nph//2,axis=1)
    if spin&1:
        arr2 = -arr2
    return arr2

def mydot(a1,a2,spin):
    return np.vdot(theta_extend(a1,spin),theta_extend(a2,spin))

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

lmax=40
mmax=lmax
kmax=10


# get random sky a_lm
# the a_lm arrays follow the same conventions as those in healpy
slmT = random_alm(lmax, mmax)

# build beam a_lm
blmT = random_alm(lmax, kmax)

t0=time.time()
# build interpolator object for slmT and blmT
foo = interpol_ng.PyInterpolator(slmT,blmT,lmax, kmax, epsilon=1e-6, nthreads=1)
print("setup time: ",time.time()-t0)
nth = 2*lmax+1
nph = 2*mmax+1
#exit()
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
blmTfull = np.zeros(slmT.size)+0j
blmTfull[0:blmT.size] = blmT
for ith in range(nth):
    rbeamth=interpol_ng.rotate_alm(blmTfull, lmax, ptg[ith,0,2],ptg[ith,0,0],0)
    for iph in range(nph):
        rbeam=interpol_ng.rotate_alm(rbeamth, lmax, 0, 0, ptg[ith,iph,1])
        bar2[ith,iph] = convolve(slmT, rbeam, lmax).real
plt.subplot(2,2,2)
plt.imshow(bar2)
plt.subplot(2,2,3)
plt.imshow((bar2-bar.reshape((nth,nph))))
plt.show()

fake = np.random.uniform(-1.,1., bar.size)
foo2 = interpol_ng.PyInterpolator(lmax, kmax, epsilon=1e-6, nthreads=2)
foo2.deinterpol(ptg.reshape((-1,3)), fake)
bla=foo2.getSlm(blmT)
print(myalmdot(slmT, bla, lmax, lmax, 0))
print(np.vdot(fake,bar))
