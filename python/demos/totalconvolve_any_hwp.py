import ducc0
import numpy as np
import time
from math import pi
from astropy.io import fits

rng = np.random.default_rng(41)


def nalm(lmax, mmax):
    return ((mmax+1)*(mmax+2))//2 + (mmax+1)*(lmax-mmax)


def random_alm(lmax, mmax, ncomp):
    res = rng.uniform(-1., 1., (ncomp, nalm(lmax, mmax))) \
     + 1j*rng.uniform(-1., 1., (ncomp, nalm(lmax, mmax)))
    # make a_lm with m==0 real-valued
    res[:, 0:lmax+1].imag = 0.
    return res

# Very simple class to store a_lm that allow negative m values
class AlmPM:
    def __init__(self, lmax, mmax):
        if lmax < 0 or mmax < 0 or lmax < mmax:
            raise ValueError("bad parameters")
        self._lmax, self._mmax = lmax, mmax
        self._data = np.zeros((2*mmax+1, lmax+1), dtype=np.complex128)

    def __getitem__(self, lm):
        l, m = lm

        if l < 0 or l > self._lmax: # or abs(m) > l:
            print(l,m)
            raise ValueError("out of bounds read access")
        # if we are asked for elements outside our m range, return 0
        if m < -self._mmax or m > self._mmax:
            return 0.+0j
        return self._data[m+self._mmax, l]

    def __setitem__(self, lm, val):
        l, m = lm
        if l < 0 or l > self._lmax or abs(m) > l or  m < -self._mmax or m > self._mmax:
            print(l,m)
            raise ValueError("out of bounds write access")
        self._data[m+self._mmax, l] = val


# Adri 2020 A25/A35
def m_hwp_to_C(m_hwp):
    T = np.zeros((4,4),dtype=np.complex128)
    T[0,0] = T[3,3] = 1.
    T[1,1] = T[2,1] = 1./np.sqrt(2.)
    T[1,2] = 1j/np.sqrt(2.)
    T[2,2] = -1j/np.sqrt(2.)
    C = T.dot(m_hwp.dot(np.conj(T.T)))
    return C


def hwp_tc_prep (blm, m_hwp, lmax, mmax):
    ncomp = blm.shape[0]

    # convert input blm to T/P/P*/V blm
    blm2 = [AlmPM(lmax, mmax+4) for _ in range(4)]
    idx = 0
    for m in range(mmax+1):
        for l in range(m, lmax+1):
            # T component
            blm2[0][l, m] = blm[0, idx]
            blm2[0][l,-m] = np.conj(blm[0, idx]) * (-1)**m
            # V component
            if ncomp > 3:
                blm2[3][l, m] = blm[3, idx]
                blm2[3][l,-m] = np.conj(blm[3, idx]) * (-1)**m
            # E/B components
            if ncomp > 2:
                # Adri's notes [10]
                blm2[1][l,m] = -(blm[1,idx] + 1j*blm[2,idx]) # spin +2
                # Adri's notes [9]
                blm2[2][l,m] = -(blm[1,idx] - 1j*blm[2,idx]) # spin -2
                # negative m
                # Adri's notes [2]
                blm2[1][l,-m] = np.conj(blm2[2][l,m]) * (-1)**m
                blm2[2][l,-m] = np.conj(blm2[1][l,m]) * (-1)**m
            idx += 1

    C = m_hwp_to_C(m_hwp)

    # compute the blm for the full beam+HWP system at HWP angles 0, 2pi/10, ...
    sqrt2 = np.sqrt(2.)
    nbeam = 5
    blm_eff = [[AlmPM(lmax, mmax+4) for _ in range(4)] for _ in range(nbeam)]
    for ibeam in range(nbeam):
        alpha = ibeam*np.pi/nbeam
        e2ia = np.exp(2*1j*alpha)
        e2iac = np.exp(-2*1j*alpha)
        e4ia = np.exp(4*1j*alpha)
        e4iac = np.exp(-4*1j*alpha)
        for m in range(-mmax-4, mmax+4+1):
            for l in range(abs(m), lmax+1):
                # T component, Marta notes [4a]
                blm_eff[ibeam][0][l, m] = C[0,0]*blm2[0][l,m]
                blm_eff[ibeam][0][l, m] += C[3,0]*blm2[3][l,m]
                blm_eff[ibeam][0][l, m] += 1./sqrt2*(C[1,0]*blm2[2][l,m+2]*e2ia + C[2,0]*blm2[1][l,m-2]*e2iac)
                # V component, Marta notes [4d]
                blm_eff[ibeam][3][l, m] = C[0,3]*blm2[0][l,m] \
                                        + C[3,3]*blm2[3][l,m] \
                                        + 1./sqrt2*(C[1,3]*blm2[2][l,m+2]*e2ia + C[2,3]*blm2[1][l,m-2]*e2iac)
                # E/B components, Marta notes [4b,c]
                blm_eff[ibeam][1][l, m] = sqrt2*e2iac*(C[0,1]*blm2[0][l,m+2] + C[3,1]*blm2[3][l,m+2])
                blm_eff[ibeam][1][l, m] += C[2,1]*e4iac*blm2[2][l,m+4]
                blm_eff[ibeam][1][l, m] += C[1,1]*blm2[1][l,m]
                blm_eff[ibeam][2][l, m] = sqrt2*e2ia*(C[0,2]*blm2[0][l,m-2] + C[3,2]*blm2[3][l,m-2])
                blm_eff[ibeam][2][l, m] += C[1,2]*e4ia*blm2[1][l,m-4]
                blm_eff[ibeam][2][l, m] += C[2,2]*blm2[2][l,m]

    # back to original format
    inc = 4
    res = np.zeros((nbeam, ncomp, nalm(lmax, mmax+inc)), dtype=np.complex128)

    for ibeam in range(nbeam):
        idx = 0
        for m in range(mmax+inc+1):
            for l in range(m, lmax+1):
                # T component
                res[ibeam, 0, idx] = blm_eff[ibeam][0][l, m]
                # V component
                if ncomp > 3:
                    res[ibeam, 3, idx] = blm_eff[ibeam][3][l, m]
                # E/B components
                if ncomp > 2:
                    # Adri's notes [10]
                    res[ibeam, 1, idx] = -0.5*(blm_eff[ibeam][1][l, m]+blm_eff[ibeam][2][l, m])
                    res[ibeam, 2, idx] = 0.5j*(blm_eff[ibeam][1][l, m]-blm_eff[ibeam][2][l, m])
                idx += 1
    return res


def pseudo_fft(inp):
    out = np.zeros((5, inp.shape[1], inp.shape[2]), dtype=np.complex128)
    out[0] = 0.2*(inp[0]+inp[1]+inp[2]+inp[3]+inp[4])
    c1, s1 = np.cos(2*np.pi/5), np.sin(-2*np.pi/5)
    c2, s2 = np.cos(4*np.pi/5), np.sin(-4*np.pi/5)
    out[1] = 0.4*(inp[0] + c1*(inp[1]+inp[4]) + c2*(inp[2]+inp[3]))
    out[2] = 0.4*(s1*(inp[1]-inp[4]) + s2*(inp[2]-inp[3]))
    out[3] = 0.4*(inp[0] + c2*(inp[1]+inp[4]) + c1*(inp[2]+inp[3]))
    out[4] = 0.4*(s2*(inp[1]-inp[4]) - s1*(inp[2]-inp[3]))
    return out


class Convolver:
    def __init__ (self, lmax, kmax, slm, blm, hwp):
        self._slm = slm
        self._orig_blm = blm
        self._lmax = lmax
        self._kmax = kmax
        tmp = hwp_tc_prep (blm, hwp, lmax, kmax)
        self._blm = pseudo_fft(tmp)
        for i in range(5):
            print(i, np.sum(np.abs(self._blm[i])))

    def signal_without_HWP(self, ptg):
        inter = ducc0.totalconvolve.Interpolator(self._slm, self._orig_blm,
            separate=False, lmax=self._lmax, kmax=self._kmax, epsilon=1e-4,
            nthreads=1)
        return inter.interpol(ptg)[0]

    def signal_with_ideal_HWP(self, ptg, alpha):
        slm_EB = np.empty((2, nalm(self._lmax,self._lmax)),dtype=np.complex128)
        slm_EB[0, :] = self._slm[1, :]
        slm_EB[1, :] = self._slm[2, :]

        blm_EB = np.empty((2, nalm(self._lmax,self._kmax)),dtype=np.complex128)
        blm_EB[0, :] = self._orig_blm[1, :]
        blm_EB[1, :] = self._orig_blm[2, :]

        blm_BE = np.empty((2, nalm(self._lmax,self._kmax)),dtype=np.complex128)
        blm_BE[0, :] = -self._orig_blm[2, :]
        blm_BE[1, :] =  self._orig_blm[1, :]

        inter_TT = ducc0.totalconvolve.Interpolator(self._slm[0:1,:],
            self._orig_blm[0:1,:], False, self._lmax, self._kmax,
            epsilon=epsilon, ofactor=ofactor, nthreads=nthreads)
        inter_EE_BB = ducc0.totalconvolve.Interpolator(slm_EB, blm_EB, False,
            self._lmax, self._kmax, epsilon=epsilon, ofactor=ofactor,
            nthreads=nthreads)
        inter_EB_BE = ducc0.totalconvolve.Interpolator(slm_EB, blm_BE, False,
            self._lmax, self._kmax, epsilon=epsilon, ofactor=ofactor,
            nthreads=nthreads)

        cos_alpha = np.cos(4*alpha)
        sin_alpha = np.sin(4*alpha)

        bar_TT = inter_TT.interpol(ptg)
        bar_EE_BB = inter_EE_BB.interpol(ptg)
        bar_EB_BE = inter_EB_BE.interpol(ptg)

        signal = bar_TT[0, :] + cos_alpha*bar_EE_BB[0, :] + sin_alpha*bar_EB_BE[0, :]
        return signal

    def signal(self, ptg, alpha):
        inter = ducc0.totalconvolve.Interpolator(self._slm,
            self._blm[0], False, self._lmax, self._kmax+4,
            epsilon=epsilon, ofactor=ofactor, nthreads=nthreads)
        res = inter.interpol(ptg)[0]
        inter = ducc0.totalconvolve.Interpolator(self._slm,
            self._blm[1], False, self._lmax, self._kmax+4,
            epsilon=epsilon, ofactor=ofactor, nthreads=nthreads)
        res += np.cos(2*alpha)*inter.interpol(ptg)[0]
        inter = ducc0.totalconvolve.Interpolator(self._slm,
            self._blm[2], False, self._lmax, self._kmax+4,
            epsilon=epsilon, ofactor=ofactor, nthreads=nthreads)
        res += np.sin(2*alpha)*inter.interpol(ptg)[0]
        inter = ducc0.totalconvolve.Interpolator(self._slm,
            self._blm[3], False, self._lmax, self._kmax+4,
            epsilon=epsilon, ofactor=ofactor, nthreads=nthreads)
        res += np.cos(4*alpha)*inter.interpol(ptg)[0]
        inter = ducc0.totalconvolve.Interpolator(self._slm,
            self._blm[4], False, self._lmax, self._kmax+4,
            epsilon=epsilon, ofactor=ofactor, nthreads=nthreads)
        res += np.sin(4*alpha)*inter.interpol(ptg)[0]

        return res


lmax = 256  # band limit
kmax = 13  # maximum beam azimuthal moment
nptg = 100
epsilon = 1e-4  # desired accuracy
ofactor = 1.5  # oversampling factor: for tuning tradeoff between CPU and memory usage
nthreads = 0  # use as many threads as available

# get random sky a_lm
# the a_lm arrays follow the same conventions as those in healpy

slm = random_alm(lmax, lmax, 3)
#slm[:,1:] = 0
blm = random_alm(lmax, kmax, 3)
#blm[1:] = 0

# produce random pointings (i.e. theta, phi, psi triples)

ptg = np.empty((nptg,3))

ptg[:, 0] = 0.2#rng.uniform(0., 1., nptg)*np.pi
ptg[:, 1] = 0.3#rng.uniform(0., 1., nptg)*2*np.pi
ptg[:, 2] = 0.5#rng.uniform(0., 1., nptg)*2*np.pi

# Mueller matrix
hwp = rng.random((4,4))-0.5
hwp = np.identity(4)
hwp[2,2] = hwp[3,3] = -1
phi0 = 0
omega = 88*2*pi/60
f_samp = 19.1
alpha = phi0+omega*np.arange(nptg)/f_samp
alpha = np.arange(nptg)/nptg*2*np.pi+0.4
conv = Convolver(lmax, kmax, slm, blm, hwp)
sig = conv.signal(ptg, alpha)
sig_nohwp = conv.signal_without_HWP(ptg)
sig_idealhwp = conv.signal_with_ideal_HWP(ptg, alpha)

import matplotlib.pyplot as plt
plt.plot(sig)
plt.plot(sig_nohwp)
plt.plot(sig_idealhwp)
plt.show()
