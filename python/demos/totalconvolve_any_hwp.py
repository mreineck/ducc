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


# Class for computing convolutions between arbitrary beams and skies in the
# presence of an optical element with arbitrary Mueller matrix in from of the
# detector.
class MuellerConvolver:

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


    def mueller_tc_prep (self, blm, mueller, lmax, mmax):
        ncomp = blm.shape[0]

        # convert input blm to T/P/P*/V blm
        blm2 = [self.AlmPM(lmax, mmax+4) for _ in range(4)]
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

        C = mueller_to_C(mueller)

        # compute the blm for the full beam+Mueller matrix system at angles
        # n*pi/5 for n in [0; 5[
        sqrt2 = np.sqrt(2.)
        nbeam = 5
        inc = 4
        res = np.zeros((nbeam, ncomp, nalm(lmax, mmax+inc)), dtype=np.complex128)
        blm_eff = [self.AlmPM(lmax, mmax+4) for _ in range(4)]
        for ibeam in range(nbeam):
            alpha = ibeam*np.pi/nbeam
            e2ia = np.exp(2*1j*alpha)
            e2iac = np.exp(-2*1j*alpha)
            e4ia = np.exp(4*1j*alpha)
            e4iac = np.exp(-4*1j*alpha)
            for m in range(-mmax-4, mmax+4+1):
                for l in range(abs(m), lmax+1):
                    # T component, Marta notes [4a]
                    blm_eff[0][l, m] = \
                          C[0,0]*blm2[0][l,m] \
                        + C[3,0]*blm2[3][l,m] \
                        + 1./sqrt2*(C[1,0]*blm2[2][l,m+2]*e2ia \
                                  + C[2,0]*blm2[1][l,m-2]*e2iac)
                    # V component, Marta notes [4d]
                    blm_eff[3][l, m] = \
                          C[0,3]*blm2[0][l,m] \
                        + C[3,3]*blm2[3][l,m] \
                        + 1./sqrt2*(C[1,3]*blm2[2][l,m+2]*e2ia \
                                  + C[2,3]*blm2[1][l,m-2]*e2iac)
                    # E/B components, Marta notes [4b,c]
                    blm_eff[1][l, m] = \
                          sqrt2*e2iac*(C[0,1]*blm2[0][l,m+2] \
                                   + C[3,1]*blm2[3][l,m+2]) \
                        + C[2,1]*e4iac*blm2[2][l,m+4] \
                        + C[1,1]*blm2[1][l,m]
                    blm_eff[2][l, m] = \
                          sqrt2*e2ia*(C[0,2]*blm2[0][l,m-2] \
                                    + C[3,2]*blm2[3][l,m-2]) \
                        + C[1,2]*e4ia*blm2[1][l,m-4] \
                        + C[2,2]*blm2[2][l,m]

            # back to original TEBV b_lm format
            idx = 0
            for m in range(mmax+inc+1):
                for l in range(m, lmax+1):
                    # T component
                    res[ibeam, 0, idx] = blm_eff[0][l, m]
                    # V component
                    if ncomp > 3:
                        res[ibeam, 3, idx] = blm_eff[3][l, m]
                    # E/B components
                    if ncomp > 2:
                        # Adri's notes [10]
                        res[ibeam, 1, idx] = -0.5*(blm_eff[1][l, m] \
                                                  +blm_eff[2][l, m])
                        res[ibeam, 2, idx] = 0.5j*(blm_eff[1][l, m] \
                                                  -blm_eff[2][l, m])
                    idx += 1
        return res

    # "Fourier transform" the blm at different alpha to obtain
    # blm(alpha) = out[0] + cos(2*alpha)*out[1] + sin(2*alpha)*out[2]
    #                     + cos(4*alpha)*out[3] + sin(4*alpha)*out[4]
    def pseudo_fft(self, inp):
        out = np.zeros((5, inp.shape[1], inp.shape[2]), dtype=np.complex128)
        out[0] = 0.2*(inp[0]+inp[1]+inp[2]+inp[3]+inp[4])
        # FIXME: I'm not absolutely sure about the sign of the angles yet
        c1, s1 = np.cos(2*np.pi/5), np.sin(2*np.pi/5)
        c2, s2 = np.cos(4*np.pi/5), np.sin(4*np.pi/5)
        out[1] = 0.4*(inp[0] + c1*(inp[1]+inp[4]) + c2*(inp[2]+inp[3]))
        out[2] = 0.4*(s1*(inp[1]-inp[4]) + s2*(inp[2]-inp[3]))
        out[3] = 0.4*(inp[0] + c2*(inp[1]+inp[4]) + c1*(inp[2]+inp[3]))
        out[4] = 0.4*(s2*(inp[1]-inp[4]) - s1*(inp[2]-inp[3]))
        return out

    def __init__ (self, lmax, kmax, slm, blm, mueller, epsilon=1e-4,
                  ofactor=1.5, nthreads=1):
        self._slm = slm
        self._lmax = lmax
        self._kmax = kmax
        tmp = self.mueller_tc_prep (blm, mueller, lmax, kmax)
        tmp = self.pseudo_fft(tmp)

        # construct the five interpolators for the individual components
        # With some enhancements in the C++ code I could put this into a single
        # interpolator for better performance, but for now this should do.
        # If the blm for any interpolator are very small in comparison to the
        # others, the interpolator is replaced by a `None` object.
        maxval = [np.max(np.abs(x)) for x in tmp]
        maxmax = max(maxval)
        self._inter = []
        for i in range(5):
            if maxval[i] > 1e-10*maxmax:  # component is not zero
                self._inter.append(ducc0.totalconvolve.Interpolator(slm,
                                   tmp[i], False, self._lmax, self._kmax+4,
                                   epsilon=epsilon, ofactor=ofactor,
                                   nthreads=nthreads))
            else:  # we can ignore this component
#                print ("component",i,"vanishes")
                self._inter.append(None)

    def signal(self, ptg, alpha):
        res = self._inter[0].interpol(ptg)[0]
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
#   mueller = rng.random((4,4))-0.5

    fullconv = MuellerConvolver(lmax, kmax, slm, blm, mueller, epsilon,
                                ofactor, nthreads)
    sig = fullconv.signal(ptg, alpha)

    import matplotlib.pyplot as plt
    plt.plot(sig)
    plt.show()


if __name__ == '__main__':
    main()
