import interpol_ng
import numpy as np
import pysharp
import time
import matplotlib.pyplot as plt

np.random.seed(20909)

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
def mydot(a1,a2,spin):
    return np.vdot(theta_extend(a1,spin),theta_extend(a2,spin))

lmax=512
mmax=lmax
kmax=12


# get random sky a_lm
# the a_lm arrays follow the same conventions as those in healpy
slmT = random_alm(lmax, mmax)

# build beam a_lm
blmT = random_alm(lmax, kmax)
ptg = np.array([[1.2,0.01,1.],[1.4,0.7,2.]])
foo = interpol_ng.PyInterpolator(slmT,blmT,lmax, kmax, epsilon=1e-6, nthreads=1)
bar=foo.interpol(ptg)
print(foo.Nphi(),foo.Nphi0())
fake = np.random.uniform(0.,1., ptg.shape[0])
foo2 = interpol_ng.PyInterpolator(lmax, kmax, epsilon=1e-6, nthreads=2)
foo2.deinterpol(ptg.reshape((-1,3)), fake)
bla=foo2.getSlm(blmT)
print(myalmdot(slmT, np.conj(bla), lmax, lmax, 0))
print(np.vdot(fake,bar))
print(myalmdot(slmT, np.conj(bla), lmax, lmax, 0)/np.vdot(fake,bar))
