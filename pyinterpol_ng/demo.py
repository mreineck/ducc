import pyinterpol_ng
import numpy as np
import pysharp
import time
import matplotlib.pyplot as plt

np.random.seed(48)

def nalm(lmax, mmax):
    return ((mmax+1)*(mmax+2))//2 + (mmax+1)*(lmax-mmax)


def random_alm(lmax, mmax, ncomp):
    res = np.random.uniform(-1., 1., (nalm(lmax, mmax), ncomp)) \
     + 1j*np.random.uniform(-1., 1., (nalm(lmax, mmax), ncomp))
    # make a_lm with m==0 real-valued
    res[0:lmax+1,:].imag = 0.
    return res


def compress_alm(alm,lmax):
    res = np.empty(2*len(alm)-lmax-1, dtype=np.float64)
    res[0:lmax+1] = alm[0:lmax+1].real
    res[lmax+1::2] = np.sqrt(2)*alm[lmax+1:].real
    res[lmax+2::2] = np.sqrt(2)*alm[lmax+1:].imag
    return res


def myalmdot(a1,a2,lmax,mmax,spin):
    return np.vdot(compress_alm(a1,lmax),compress_alm(np.conj(a2),lmax))


def convolve(alm1, alm2, lmax):
    job = pysharp.sharpjob_d()
    job.set_triangular_alm_info(lmax, lmax)
    job.set_gauss_geometry(lmax+1, 2*lmax+1)
    map = job.alm2map(alm1)*job.alm2map(alm2)
    job.set_triangular_alm_info(0,0)
    return job.map2alm(map)[0]*np.sqrt(4*np.pi)


lmax=60
kmax=13
ncomp=1
separate=True
ncomp2 = ncomp if separate else 1

# get random sky a_lm
# the a_lm arrays follow the same conventions as those in healpy
slm = random_alm(lmax, lmax, ncomp)

# build beam a_lm
blm = random_alm(lmax, kmax, ncomp)


t0=time.time()
# build interpolator object for slm and blm
foo = pyinterpol_ng.PyInterpolator(slm,blm,separate,lmax, kmax, epsilon=1e-4, nthreads=2)
print("setup time: ",time.time()-t0)
nth = lmax+1
nph = 2*lmax+1


# compute a convolved map at a fixed psi and compare it to a map convolved
# "by hand"

ptg = np.zeros((nth,nph,3))
ptg[:,:,0] = (np.pi*(0.5+np.arange(nth))/nth).reshape((-1,1))
ptg[:,:,1] = (2*np.pi*(0.5+np.arange(nph))/nph).reshape((1,-1))
ptg[:,:,2] = np.pi*0.2
t0=time.time()
# do the actual interpolation
bar=foo.interpol(ptg.reshape((-1,3))).reshape((nth,nph,ncomp2))
print("interpolation time: ", time.time()-t0)
plt.subplot(2,2,1)
plt.imshow(bar[:,:,0])
bar2 = np.zeros((nth,nph))
blmfull = np.zeros(slm.shape)+0j
blmfull[0:blm.shape[0],:] = blm
for ith in range(nth):
    rbeamth=pyinterpol_ng.rotate_alm(blmfull[:,0], lmax, ptg[ith,0,2],ptg[ith,0,0],0)
    for iph in range(nph):
        rbeam=pyinterpol_ng.rotate_alm(rbeamth, lmax, 0, 0, ptg[ith,iph,1])
        bar2[ith,iph] = convolve(slm[:,0], rbeam, lmax).real
plt.subplot(2,2,2)
plt.imshow(bar2)
plt.subplot(2,2,3)
plt.imshow(bar2-bar[:,:,0])
plt.show()


ptg=np.random.uniform(0.,1.,3*1000000).reshape(1000000,3)
ptg[:,0]*=np.pi
ptg[:,1]*=2*np.pi
ptg[:,2]*=2*np.pi
#foo = pyinterpol_ng.PyInterpolator(slm,blm,separate,lmax, kmax, epsilon=1e-6, nthreads=2)
t0=time.time()
bar=foo.interpol(ptg)
print("interpolation time: ", time.time()-t0)
fake = np.random.uniform(0.,1., (ptg.shape[0],ncomp2))
foo2 = pyinterpol_ng.PyInterpolator(lmax, kmax, ncomp2, epsilon=1e-4, nthreads=2)
t0=time.time()
foo2.deinterpol(ptg.reshape((-1,3)), fake)
print("deinterpolation time: ", time.time()-t0)
t0=time.time()
bla=foo2.getSlm(blm)
print("getSlm time: ", time.time()-t0)
v1 = np.sum([myalmdot(slm[:,i], bla[:,i] , lmax, lmax, 0) for i in range(ncomp)])
v2 = np.sum([np.vdot(fake[:,i],bar[:,i]) for i in range(ncomp2)])
print(v1/v2-1.)
