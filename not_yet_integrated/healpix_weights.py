import numpy as np
import ducc0
from scipy.sparse.linalg import LinearOperator, lsmr

def ringweights(nside, lmax, epsilon, maxiter):
    """Computes Healpix ring weights.
    
    Parameters
    ----------
    nside: int
        nside parameter to use.
    lmax: int
        lmax to use. Must be even.
    epsilon: float
    maxiter: int
        maximum number of iterations for the conjugate gradient solver
    
    Returns
    -------
    numpy.ndarray(4*nside-1,), dtype=numpy.float64)
        the ring weights
    """
    ginfo = ducc0.healpix.Healpix_Base(nside, "RING").sht_info()
    theta = ginfo["theta"][:2*nside]
    nphi = np.ones(2*nside, dtype=np.uint64)
    phi0 = np.zeros(2*nside)
    ringstart = np.arange(2*nside, dtype=np.uint64)

    def S(alm):
        alm2 = np.zeros((1,lmax+1), dtype=np.complex128)
        alm2[0,::2] = alm
        return ducc0.sht.synthesis(alm=alm2, lmax=lmax, mmax=0, spin=0,
            theta=theta, nphi=nphi, ringstart=ringstart, phi0=phi0, nthreads=1)[0]
    def ST(map):
        alm2 = ducc0.sht.adjoint_synthesis(map=map.reshape((1,-1)), lmax=lmax,
            mmax=0, spin=0, theta=theta, nphi=nphi, ringstart=ringstart,
            phi0=phi0, nthreads=1)
        return alm2[0,::2].real.copy()

    # initialize RHS
    nir = ginfo["nphi"][:2*nside].astype(np.float64)*2
    nir[-1] /= 2
    rhs = -ST(nir)
    rhs[0]+=12*nside**2/np.sqrt(4*np.pi)
    op = LinearOperator(matvec=ST, rmatvec=S, shape=(lmax//2+1, 2*nside))
    res = lsmr(A=op, b=rhs, atol=epsilon, btol=epsilon, maxiter=maxiter)
    if res[1] != 1:
        raise RuntimeError("iteration did not converge to a solution")

    # mirror the result to get full ring weight vector
    wgt = np.empty(4*nside-1)
    wgt[:2*nside] = res[0]/nir+1.
    wgt[2*nside:] = wgt[2*nside-2::-1]
    return wgt


def _pixselect(nside):
    nfullwgt = ((3*nside+1)*(nside+1))//4
    idx = np.empty(nfullwgt, dtype=int)
    idx2 = np.empty(12*nside**2, dtype=int)
    fct = np.empty(nfullwgt,dtype=np.float64)
    ipix,ofs = 0, 0
    for iring in range(2*nside):
        fsymm = 2 if iring<2*nside-1 else 1
        shifted = (iring<nside-1) or ((iring+nside)%2 == 1)
        qpix = min(nside,iring+1)
        odd = qpix%2 == 1
        wpix=((qpix+1)>>1) + (0 if (odd or shifted) else 1)
        idx[ofs:ofs+wpix]=np.arange(ipix, ipix+wpix)
        idx2[ipix:ipix+wpix]=np.arange(ofs, ofs+wpix)
        t1 = ipix if shifted else ipix+1
        idx2[ipix+qpix-1:ipix+wpix-1:-1] = idx2[t1:t1+qpix-wpix]
        # remaining three quarters of ring
        idx2[ipix+  qpix:ipix+2*qpix] = idx2[ipix:ipix+qpix]
        idx2[ipix+2*qpix:ipix+4*qpix] = idx2[ipix:ipix+2*qpix]
        # mirror
        idx2[12*nside**2-ipix-4*qpix:12*nside**2-ipix] = idx2[ipix:ipix+4*qpix]
        xfct = np.ones(wpix)*16
        if iring == 2*nside-1:  # equator
            xfct /= 2
        if not shifted:
            xfct[0] /= 2
        if shifted == odd:
            xfct[-1] /= 2
        fct[ofs:ofs+wpix] = xfct
        ipix += 4*qpix
        ofs += wpix
    fct = np.sqrt(fct)
    return idx, fct, idx2

def pixelweights(nside, lmax, mmax, epsilon, maxiter, nthreads=1, guess=None):
    """Computes Healpix pixel weights.
    
    Parameters
    ----------
    nside: int
        nside parameter to use.
    lmax: int
        lmax to use. Must be even.
    mmax: int
        mmax to use
    epsilon: float
    maxiter: int
        maximum number of iterations for the conjugate gradient solver
    nthreads: int=1
        number of threads to use
    guess: numpy.ndarray(12*nside**2, dtype=numpy.float64)
        initial guess to use
   
    Returns
    -------
    numpy.ndarray(12*nside**2,), dtype=numpy.float64)
        the pixel weights
    """
    ginfo = ducc0.healpix.Healpix_Base(nside, "RING").sht_info()
    for thing in ["theta", "nphi", "phi0", "ringstart"]:
       ginfo[thing] = ginfo[thing][:2*nside]
    theta = ginfo["theta"]
    nphi = ginfo["nphi"]
    phi0 = ginfo["phi0"]
    ringstart = ginfo["ringstart"]
    mmod=4
    mval = np.arange(0,mmax+1,mmod,dtype=np.int64)
    nm = mval.shape[0]
    mstart = np.zeros(nm, dtype=np.int64)
    alm_lut = np.zeros(((lmax+1)*(lmax+2))//2, dtype=int)
    m0limit = len(range(0,lmax+1,2))
    cnt, ofs = 0, 0
    for m in range(0,mmax+1,mmod):
        mstart[m//mmod] = ofs-m
        for l in range(m,lmax+1):
            if l%2==0:
                alm_lut[cnt] = ofs
                cnt += 1
            ofs += 1
    nalm = ofs
    alm_lut= alm_lut[:cnt].copy()
    npix = 6*nside**2+2*nside
    rf = np.full(2*nside,2.)
    rf[-1] = 1.
    idx, fct, idx2 = _pixselect(nside)

    def compress_alm(alm):
        res = alm[alm_lut].real.copy()
        res[m0limit:] *= np.sqrt(2)
        return res
    def expand_alm(alm):
        res = np.zeros(nalm, dtype=np.complex128)
        res[alm_lut] = alm
        res[lmax+1:] /= np.sqrt(2.)
        return res
    def compress_map(map):
        res = map[idx]
        res *= fct
        return res
    def expand_map(map):
        res = (map/fct)[idx2[:npix]]
        return res
    def expand_map_full(map):
        return (map/fct)[idx2]
    def S(alm):
        leg = np.zeros((1,2*nside,mmax+1),dtype=np.complex128)
        ducc0.sht.alm2leg(alm=expand_alm(alm).reshape((1,-1)), lmax=lmax,
            mval=mval, mstart=mstart, spin=0, theta=theta, nthreads=nthreads,
            leg=leg[:,:,::mmod])
        map = ducc0.sht.leg2map(leg=leg, nphi=nphi, phi0=phi0, 
            ringstart=ringstart, nthreads=nthreads)
        return compress_map(map.reshape((-1,)))
    def ST(map):
        leg = ducc0.sht.map2leg(map=expand_map(map).reshape((1,-1)), nphi=nphi,
            phi0=phi0, ringstart=ringstart, nthreads=nthreads, mmax=mmax,
            ringfactor=rf)
        alm2 = ducc0.sht.leg2alm(leg=leg[:,:,::mmod], lmax=lmax, mval=mval,
            mstart=mstart, spin=0, theta=theta, nthreads=nthreads)
        return compress_alm(alm2.reshape((-1,)))

    rhs = -ST(compress_map(np.ones(npix)))
    rhs[0] += 12*nside**2/np.sqrt(4*np.pi)
    op = LinearOperator(matvec=ST, rmatvec=S, shape=(rhs.shape[0],
        ((3*nside+1)*(nside+1))//4))
    res = lsmr(A=op, b=rhs, atol=epsilon, btol=epsilon, maxiter=maxiter,
               x0=compress_map(guess) if guess is not None else None)
    if res[1] != 1:
        raise RuntimeError("iteration did not converge to a solution")
    return expand_map_full(res[0])+1.

# iterative map2alm using LSMR for iteration and starting from a user-supplied
# first guess.
def experimental(map, lmax, mmax, guess, tol=1e-14, maxiter=20, nthreads=1):
    nside = hp.npix2nside(map.shape[1])
    ginfo = ducc0.healpix.Healpix_Base(nside, "RING").sht_info()
    res = ducc0.sht.pseudo_analysis(map=map,alm=guess.copy(),lmax=lmax, mmax=mmax, spin=0, **ginfo, maxiter=maxiter, epsilon=tol, alm_contains_initial_guess=True, nthreads=nthreads)
    istop = res[1]
    niter = res[2]
    normres = res[3]
    return (res[0][0], normres/np.linalg.norm(map), niter, istop<7)
