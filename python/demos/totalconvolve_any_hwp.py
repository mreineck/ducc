import ducc0
import numpy as np


def nalm(lmax, mmax):
    return ((mmax+1)*(mmax+2))//2 + (mmax+1)*(lmax-mmax)


# Adri 2020 A25/A35
def mueller_to_C(mueller):
    T = np.zeros((4,4),dtype=np.complex128)
    T[0,0] = T[3,3] = 1.
    T[1,1] = T[2,1] = 1./np.sqrt(2.)
    T[1,2] = 1j/np.sqrt(2.)
    T[2,2] = -1j/np.sqrt(2.)
    C = T.dot(mueller.dot(np.conj(T.T)))
    return C


def truncate_blm(inp, lmax, kmax, epsilon=1e-10):
    limit = epsilon * np.max(np.abs(inp))
    ncomp = inp.shape[1]
    out = []
    for i in range(len(inp)):
        maxk = -1
        idx = 0
        for k in range(kmax+1):
            if np.max(np.abs(inp[i,:,idx:idx+lmax+1-k])) > limit:
                maxk = k
            idx += lmax+1-k
#        print("component",i,"maxk=",maxk)
        if maxk == -1:
            out.append(None)
        else:
            out.append((inp[i,:,:nalm(lmax, maxk)].copy(), maxk))
    return out


class MuellerConvolver:
    """Class for computing convolutions between arbitrary beams and skies in the
    presence of an optical element with arbitrary Mueller matrix in front of the
    detector.

    Most of the expressions in this code are derived from
    Duivenvoorden et al. 2021, MNRAS 502, 4526
    (https://arxiv.org/abs/2012.10437)

    Parameters
    ----------
    lmax : int
        maximum l moment of the provided sky and beam a_lm
    kmax : int
        maximum m moment of the provided beam a_lm
    slm : numpy.ndarray((n_comp, n_slm), dtype=complex)
        input sky a_lm
        ncomp can be 1, 3, or 4, for T, TEB, TEBV components, respectively.
        The components have the a_lm format used by healpy
    blm : numpy.ndarray((n_comp, n_blm), dtype=complex)
        input beam a_lm
        ncomp can be 1, 3, or 4, for T, TEB, TEBV components, respectively.
        The components have the a_lm format used by healpy
    mueller : np.ndarray((4,4), dtype=np.float64)
        Mueller matrix of the optical elemen in front of the detector
    single_precision : bool
        if True, store internal data in single precision, else double precision
    epsilon : float
        desired accuracy for the interpolation; a typical value is 1e-4
    ofactor : float
        oversampling factor to be used for the interpolation grids.
        Should be in the range [1.2; 2], a typical value is 1.5
        Increasing this factor makes (adjoint) convolution slower and
        increases memory consumption, but speeds up interpolation/deinterpolation.
    nthreads : int
        the number of threads to use for computation
    """

    # Very simple class to store a_lm that allow negative m values
    class AlmPM:
        def __init__(self, lmax, mmax):
            if lmax < 0 or mmax < 0 or lmax < mmax:
                raise ValueError("bad parameters")
            self._lmax, self._mmax = lmax, mmax
            self._data = np.zeros((2*mmax+1, lmax+1), dtype=np.complex128)

        def __getitem__(self, lm):
            l, m = lm

            if isinstance(l, slice):
                if l.step is not None or l.start < 0 or l.stop-1 > self._lmax:
                    print(l,m)
                    raise ValueError("out of bounds read access")
            else:
                if l < 0 or l > self._lmax: # or abs(m) > l:
                    print(l,m)
                    raise ValueError("out of bounds read access")
            # if we are asked for elements outside our m range, return 0
            if m < -self._mmax or m > self._mmax:
                return 0.+0j
            return self._data[m+self._mmax, l]

        def __setitem__(self, lm, val):
            l, m = lm
            if isinstance(l, slice):
                if l.step is not None or l.start < 0 or l.stop-1 > self._lmax:
                    print(l,m)
                    raise ValueError("out of bounds write access")
            else:
                if l < 0 or l > self._lmax or abs(m) > l or  m < -self._mmax or m > self._mmax:
                    print(l,m)
                    raise ValueError("out of bounds write access")
            self._data[m+self._mmax, l] = val


    def mueller_tc_prep (self, blm, mueller, lmax, mmax):
        ncomp = blm.shape[0]

        # convert input blm to T/P/P*/V blm
        blm2 = [self.AlmPM(lmax, mmax+4) for _ in range(4)]
        idx = 0
        for m in range(mmax+1):
            sign = (-1)**m
            lrange = slice(m, lmax+1)
            idxrange = slice(idx, idx+lmax+1-m)
            # T component
            blm2[0][lrange, m] = blm[0, idxrange]
            blm2[0][lrange,-m] = np.conj(blm[0, idxrange]) * sign
            # V component
            if ncomp > 3:
                blm2[3][lrange, m] = blm[3, idxrange]
                blm2[3][lrange,-m] = np.conj(blm[3, idxrange]) * sign
            # E/B components
            if ncomp > 2:
                # Adri's notes [10]
                blm2[1][lrange,m] = -(blm[1,idxrange] + 1j*blm[2,idxrange]) # spin +2
                # Adri's notes [9]
                blm2[2][lrange,m] = -(blm[1,idxrange] - 1j*blm[2,idxrange]) # spin -2
                # negative m
                # Adri's notes [2]
                blm2[1][lrange,-m] = np.conj(blm2[2][lrange,m]) * sign
                blm2[2][lrange,-m] = np.conj(blm2[1][lrange,m]) * sign
            idx += lmax+1-m

        C = mueller_to_C(mueller)

        # compute the blm for the full beam+Mueller matrix system at angles
        # n*pi/5 for n in [0; 5[
        sqrt2 = np.sqrt(2.)
        nbeam = 5
        inc = 4
        res = np.zeros((nbeam, ncomp, nalm(lmax, mmax+inc)), dtype=self._ctype)
        blm_eff = [self.AlmPM(lmax, mmax+4) for _ in range(4)]

        for ibeam in range(nbeam):
            alpha = ibeam*np.pi/nbeam
            e2ia = np.exp(2*1j*alpha)
            e2iac = np.exp(-2*1j*alpha)
            e4ia = np.exp(4*1j*alpha)
            e4iac = np.exp(-4*1j*alpha)
            for m in range(-mmax-4, mmax+4+1):
                lrange = slice(abs(m), lmax+1)
                # T component, Marta notes [4a]
                blm_eff[0][lrange, m] = \
                      C[0,0]*blm2[0][lrange,m] \
                    + C[3,0]*blm2[3][lrange,m] \
                    + 1./sqrt2*(C[1,0]*blm2[2][lrange,m+2]*e2ia \
                              + C[2,0]*blm2[1][lrange,m-2]*e2iac)
                # V component, Marta notes [4d]
                blm_eff[3][lrange, m] = \
                      C[0,3]*blm2[0][lrange,m] \
                    + C[3,3]*blm2[3][lrange,m] \
                    + 1./sqrt2*(C[1,3]*blm2[2][lrange,m+2]*e2ia \
                              + C[2,3]*blm2[1][lrange,m-2]*e2iac)
                # E/B components, Marta notes [4b,c]
                blm_eff[1][lrange, m] = \
                      sqrt2*e2iac*(C[0,1]*blm2[0][lrange,m+2] \
                                 + C[3,1]*blm2[3][lrange,m+2]) \
                    + C[2,1]*e4iac*blm2[2][lrange,m+4] \
                    + C[1,1]*blm2[1][lrange,m]
                blm_eff[2][lrange, m] = \
                      sqrt2*e2ia*(C[0,2]*blm2[0][lrange,m-2] \
                                + C[3,2]*blm2[3][lrange,m-2]) \
                    + C[1,2]*e4ia*blm2[1][lrange,m-4] \
                    + C[2,2]*blm2[2][lrange,m]

            # back to original TEBV b_lm format
            idx = 0
            for m in range(mmax+inc+1):
                lrange = slice(m, lmax+1)
                idxrange = slice(idx, idx+lmax+1-m)
                # T component
                res[ibeam, 0, idxrange] = blm_eff[0][lrange, m]
                # V component
                if ncomp > 3:
                    res[ibeam, 3, idxrange] = blm_eff[3][lrange, m]
                # E/B components
                if ncomp > 2:
                    # Adri's notes [10]
                    res[ibeam, 1, idxrange] = -0.5*(blm_eff[1][lrange, m] \
                                                   +blm_eff[2][lrange, m])
                    res[ibeam, 2, idxrange] = 0.5j*(blm_eff[1][lrange, m] \
                                                   -blm_eff[2][lrange, m])
                idx += lmax+1-m

        return res

    # "Fourier transform" the blm at different alpha to obtain
    # blm(alpha) = out[0] + cos(2*alpha)*out[1] + sin(2*alpha)*out[2]
    #                     + cos(4*alpha)*out[3] + sin(4*alpha)*out[4]
    def pseudo_fft(self, inp):
        out = np.zeros((5, inp.shape[1], inp.shape[2]), dtype=self._ctype)
        out[0] = 0.2*(inp[0]+inp[1]+inp[2]+inp[3]+inp[4])
        # FIXME: I'm not absolutely sure about the sign of the angles yet
        c1, s1 = np.cos(2*np.pi/5), np.sin(2*np.pi/5)
        c2, s2 = np.cos(4*np.pi/5), np.sin(4*np.pi/5)
        out[1] = 0.4*(inp[0] + c1*(inp[1]+inp[4]) + c2*(inp[2]+inp[3]))
        out[2] = 0.4*(s1*(inp[1]-inp[4]) + s2*(inp[2]-inp[3]))
        out[3] = 0.4*(inp[0] + c2*(inp[1]+inp[4]) + c1*(inp[2]+inp[3]))
        out[4] = 0.4*(s2*(inp[1]-inp[4]) - s1*(inp[2]-inp[3]))
        # Alternative way via real FFT
        # out2 = inp.copy()
        # out2 = out2.view(np.float64)
        # out2 = ducc0.fft.r2r_fftpack(out2,real2hermitian=True,forward=False,axes=(0,),out=out2)
        # out2[0] *=0.2
        # out2[1:] *=0.4
        # out2 = out2.view(np.complex128)
        # print(np.max(np.abs(out-out2)))
        return out

    def __init__ (self, lmax, kmax, slm, blm, mueller, single_precision=True,
                  epsilon=1e-4, ofactor=1.5, nthreads=1):
        self._ftype = np.float32 if single_precision else np.float64
        self._ctype = np.complex64 if single_precision else np.complex128
        self._slm = slm.astype(self._ctype)
        self._lmax = lmax
        self._kmax = kmax
        tmp = self.mueller_tc_prep (blm, mueller, lmax, kmax)
        tmp = self.pseudo_fft(tmp)

        # construct the five interpolators for the individual components
        # All sets of blm are checked up to which kmax they contain significant
        # coefficients, and the interpolator is chosen accordingly
        tmp = truncate_blm(tmp, self._lmax, self._kmax+4)
        self._inter = []
        intertype = ducc0.totalconvolve.Interpolator_f \
            if self._ctype == np.complex64 else ducc0.totalconvolve.Interpolator
        for i in range(5):
            if tmp[i] is not None:  # component is not zero
                self._inter.append(intertype(self._slm,
                                   tmp[i][0], False, self._lmax, tmp[i][1],
                                   epsilon=epsilon, ofactor=ofactor,
                                   nthreads=nthreads))
            else:  # we can ignore this component entirely
                self._inter.append(None)

    def signal(self, ptg, alpha):
        """Computes the convolved signal for a set of pointings and HWP angles.

        Parameters
        ----------
        ptg : numpy.ndarray((nptg, 3), dtype=float)
            the input pointings in radians, in (theta, phi, psi) order
        alpha : numpy.ndarray((nptg,), dtype=np.float)
            the HWP angles in radians

        Returns
        -------
        signal : numpy.ndarray((nptg,), dtype=np.float)
            the signal measured by the detector
        """
        ptg = ptg.astype(self._ftype)
        alpha = alpha.astype(self._ftype)
        if self._inter[0] is not None:
            res = self._inter[0].interpol(ptg)[0]
        else:
            res = np.zeros(ptg.shape[0], dtype=self._ftype)
        if self._inter[1] is not None:
            res += np.cos(2*alpha)*self._inter[1].interpol(ptg)[0]
        if self._inter[2] is not None:
            res += np.sin(2*alpha)*self._inter[2].interpol(ptg)[0]
        if self._inter[3] is not None:
            res += np.cos(4*alpha)*self._inter[3].interpol(ptg)[0]
        if self._inter[4] is not None:
            res += np.sin(4*alpha)*self._inter[4].interpol(ptg)[0]
        return res


# demo application

def main():
    rng = np.random.default_rng(41)

    def random_alm(lmax, mmax, ncomp):
        res = rng.uniform(-1., 1., (ncomp, nalm(lmax, mmax))) \
         + 1j*rng.uniform(-1., 1., (ncomp, nalm(lmax, mmax)))
        # make a_lm with m==0 real-valued
        res[:, 0:lmax+1].imag = 0.
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
    blm = random_alm(lmax, kmax, 3)

    # produce pointings (i.e. theta, phi, psi triples)
    ptg = np.empty((nptg,3))

    # for this test, we keep (theta, phi, psi) fixed and rotate alpha through 2pi
    ptg[:, 0] = 0.2
    ptg[:, 1] = 0.3
    ptg[:, 2] = 0.5
    alpha = np.arange(nptg)/nptg*2*np.pi

    # We use an idealized HWP Mueller matrix
    mueller = np.identity(4)
    mueller[2,2] = mueller[3,3] = -1
    mueller = rng.random((4,4))-0.5

    fullconv = MuellerConvolver(lmax, kmax, slm, blm, mueller,
                                single_precision=True, epsilon=epsilon,
                                ofactor=ofactor, nthreads=nthreads)
    sig = fullconv.signal(ptg, alpha)

    import matplotlib.pyplot as plt
    plt.plot(sig)
    plt.show()


if __name__ == '__main__':
    main()
