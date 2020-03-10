import interpol_ng
import numpy as np
import pypocketfft as ppf
import matplotlib.pyplot as plt
import pysharp

np.random.seed(42)
lmax=1000
mmax=lmax
kmax=10


def nalm(lmax, mmax):
    return ((mmax+1)*(mmax+2))//2 + (mmax+1)*(lmax-mmax)


def random_alm(lmax, mmax):
    res = np.random.uniform(-1., 1., nalm(lmax, mmax)) \
     + 1j*np.random.uniform(-1., 1., nalm(lmax, mmax))
    # make a_lm with m==0 real-valued
    res[0:lmax+1].imag = 0.
    return res

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


def compress_alm(alm,lmax):
    res = np.empty(2*len(alm)-lmax-1, dtype=np.float64)
    res[0:lmax+1] = alm[0:lmax+1].real
    res[lmax+1::2] = np.sqrt(2)*alm[lmax+1:].real
    res[lmax+2::2] = np.sqrt(2)*alm[lmax+1:].imag
    return res


def myalmdot(a1,a2,lmax,mmax,spin):
    return np.vdot(compress_alm(a1,lmax),compress_alm(a2,lmax))
spin=0
inter = interpol_ng.PyInterpolator(lmax, kmax, epsilon=1e-6, nthreads=1)
nphi0, ntheta0, nphi, ntheta = inter.Nphi0(), inter.Ntheta0(), inter.Nphi(), inter.Ntheta()

alm0 = random_alm(lmax, mmax)
job = pysharp.sharpjob_d()
job.set_triangular_alm_info(lmax, lmax)
job.set_cc_geometry(ntheta0, nphi0)
in0 = job.alm2map(alm0).reshape((ntheta0,nphi0))
out0=inter.test_correct(in0,spin)
print(out0.shape,out0.strides)
job.set_cc_geometry(ntheta, nphi)
out0x=np.copy(out0)
out0x[1:-1,:]*=2
aout0=job.alm2map_adjoint(out0x.reshape((-1,)))

alm1 = random_alm(lmax, mmax)
in1 = job.alm2map(alm1).reshape((ntheta,nphi))
out1=inter.test_decorrect(in1,spin)
print(out1.shape,out1.strides)
job.set_cc_geometry(ntheta0, nphi0)
out1x=np.copy(out1)
out1x[1:-1,:]*=2
aout1=job.alm2map_adjoint(out1x.reshape((-1,)))

print(myalmdot(alm0,aout1,lmax,lmax,spin)/myalmdot(alm1,aout0,lmax,lmax,spin))
print(nphi0, ntheta0, nphi, ntheta)
