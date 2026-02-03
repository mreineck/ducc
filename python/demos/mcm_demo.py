import numpy as np
from time import time

# We have to set nthreads before importing pspy, otherwise the
# environment variable change will have no effect.
nthreads=8
import os
os.environ["OMP_NUM_THREADS"]=str(nthreads)

# This must happen after setting OMP_NUM_THREADS!
from pspy._mcm_fortran import mcm_compute as mcm_fortran
import ducc0

def format_toepliz_fortran2(coupling, l_toep, l_exact, lmax):
    """Take a matrix and apply the toepliz appoximation (fortran)

    Parameters
    ----------

    coupling: array
      consist of an array where the upper part is the exact matrix and
      the lower part is the diagonal. We will feed the off diagonal
      of the lower part using the measurement of the correlation from the exact computatio
    l_toep: integer
      the l at which we start the approx
    l_exact: integer
      the l until which we do the exact computation
    lmax: integer
      the maximum multipole of the array
    """
    toepliz_array = np.zeros(coupling.shape)
    mcm_fortran.toepliz_array_fortran2(toepliz_array.T, coupling.T, l_toep, l_exact)
    toepliz_array[coupling != 0] = coupling[coupling != 0]
    return toepliz_array

# This routine is more complicated than mcm00_ducc, since a few multiplication
# steps are carried out in Python in pspy, and since the array indices are
# a bit different. Overall this should not have noticeable impact on performance
# at higher lmax.
def mcm00_pspy(spec, lmax):
    nspec = spec.shape[0]
    lrange_spec = np.arange(spec.shape[1])
    res=np.zeros((nspec, lmax+1, lmax+1))
    mcmtmp = np.zeros((lmax+1, lmax+1))
    for i in range(nspec):
        mcmtmp[()] = 0
        wcl = spec[i]*(2*lrange_spec+1)
        mcm_fortran.calc_coupling_spin0(wcl, coupling=mcmtmp.T, l_exact=l_exact, l_band=dl_band, l_toeplitz=l_toeplitz)
        if l_exact < lmax:
           mcmtmp = format_toepliz_fortran2(mcmtmp, l_toeplitz, l_exact, lmax)
        mcm_fortran.fill_upper(mcmtmp.T)
        res[i, 2:, 2:] = mcmtmp[:-2,:-2]
    return res

def mcm02_pspy(spec, lmax):
    nspec = spec.shape[0]
    lrange_spec = np.arange(spec.shape[2])
    res=np.zeros((nspec, 5, lmax+1, lmax+1))
    mcmtmp = np.zeros((5, lmax+1, lmax+1))
    for i in range(nspec):
        mcmtmp[()] = 0
        wcl = spec[i]*((2*lrange_spec+1).reshape((1,-1)))
        mcm_fortran.calc_coupling_spin0and2(wcl[0], wcl[1], wcl[2], wcl[3], coupling=mcmtmp.T, l_exact=l_exact, l_band=dl_band, l_toeplitz=l_toeplitz)
        for j in range(5):
            if l_exact < lmax:
                mcmtmp[j] = format_toepliz_fortran2(mcmtmp[j], l_toeplitz, l_exact, lmax)
            mcm_fortran.fill_upper(mcmtmp[j].T)
        res[i, :, 2:, 2:] = mcmtmp[:,:-2,:-2]
    return res

def mcm00_ducc(spec, l1, l2):
    out= np.zeros((spec.shape[0],l1+1,l2+1),dtype=np.float32)
    ducc0.misc.experimental.coupling_matrix_rect(spec, optype=(0,)*spec.shape[0], nthreads=nthreads, res=out, l_exact=l_exact, dl_band=dl_band, l_toeplitz=l_toeplitz)
    return out

def mcm02_ducc(spec, l1, l2):
    nspec = spec.shape[0]
    out= np.zeros((nspec*5,l1+1,l2+1),dtype=np.float32)
    spec = spec.reshape((nspec*4, spec.shape[2]))
    optype = (0,1,1,4)*nspec
    ducc0.misc.experimental.coupling_matrix_rect(spec, optype, nthreads=nthreads, res=out, l_exact=l_exact, dl_band=dl_band, l_toeplitz=l_toeplitz)
    return out

def mcmpm_ducc(spec, l1, l2):
    out= np.empty((2*spec.shape[0],l1+1, l2+1),dtype=np.float32)
    ducc0.misc.experimental.coupling_matrix_rect(spec[:,3,:], lmax, (4,)*spec.shape[0], nthreads=nthreads, res=out, l_exact=l_exact, dl_band=dl_band, l_toeplitz=l_toeplitz)
    return out

# lmax up to which the MCM will be computed
l1=1000
l2=700
lmax=max(l1,l2)

l_exact=100
dl_band=200
l_toeplitz=170

# number of spectra to process simultaneously

nspec=5

print()
print("Mode coupling matrix computation comparison")
print(f"nspec={nspec}, lmax={lmax}, nthreads={nthreads}")

# we generate the spectra up to 2*lmax+1 to use all Wigner 3j symbols
# but this could also be lower.
spec = np.random.normal(size=(nspec, 4, 2*lmax+1))
spec = np.random.uniform(0.1,1.,size=(nspec, 4, 2*lmax+1))

print()
print("Spin 0 case:")

t0=time()
pspy = mcm00_pspy(spec[:,0,:], lmax)
print(f"pspy time: {time()-t0}s")

t0=time()
duccsq = mcm00_ducc(spec[:,0,:], l1, l2)
print(f"ducc square time (single precision): {time()-t0}s")

# compare the results
print(f"L2 error between pspy and ducc solutions: {ducc0.misc.l2error(pspy[:,2:l1+1,2:l2+1],4*np.pi*duccsq[:,2:,2:])}")
print()
print("Spin 0and2 case:")

t0=time()
pspy = mcm02_pspy(spec, lmax)
print(f"pspy time: {time()-t0}s")
pspy = np.where(np.isnan(pspy), 0, pspy)

t0=time()
duccsq = mcm02_ducc(spec, l1, l2)
print(f"ducc square time (single precision): {time()-t0}s")
duccsq = np.where(np.isnan(duccsq), 0, duccsq)
pspy2 = pspy[:,:,2:l1+1,2:l2+1].reshape((-1,l1-1,l2-1))
# compare the results
print(f"L2 error between pspy and ducc solutions: {ducc0.misc.l2error(pspy2,4*np.pi*duccsq[:,2:,2:])}")
