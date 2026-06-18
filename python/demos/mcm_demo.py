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

def format_toeplitz_fortran2(coupling, l_toep, l_exact, lmax):
    toeplitz_array = np.zeros(coupling.shape)
    mcm_fortran.toepliz_array_fortran2(toeplitz_array.T, coupling.T, l_toep, l_exact)
    toeplitz_array[coupling != 0] = coupling[coupling != 0]
    return toeplitz_array

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
           mcmtmp = format_toeplitz_fortran2(mcmtmp, l_toeplitz, l_exact, lmax)
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
                mcmtmp[j] = format_toeplitz_fortran2(mcmtmp[j], l_toeplitz, l_exact, lmax)
            mcm_fortran.fill_upper(mcmtmp[j].T)
        res[i, :, 2:, 2:] = mcmtmp[:,:-2,:-2]
    return res

def mcm00_ducc(spec, l1, l2):
    out= np.empty((spec.shape[0],l1+1,l2+1),dtype=np.float64)
    ducc0.misc.experimental.coupling_matrix_rect(spec, optype=(0,)*spec.shape[0], nthreads=nthreads, res=out, l_exact=l_exact, dl_band=dl_band, l_toeplitz=l_toeplitz)
    return out
def mcm00_ducc_new(spec, l1, l2):
    out= np.empty((spec.shape[0],l1+1,l2+1),dtype=np.float64)
    ducc0.misc.experimental.coupling_matrix_rect_new(spec, optype=(0,)*spec.shape[0], nthreads=nthreads, res=out, l_exact=l_exact, dl_band=dl_band, l_toeplitz=l_toeplitz)
    return out
def mcmpp_ducc(spec, l1, l2):
    out= np.empty((spec.shape[0],l1+1,l2+1),dtype=np.float64)
    ducc0.misc.experimental.coupling_matrix_rect(spec, optype=(2,)*spec.shape[0], nthreads=nthreads, res=out, l_exact=l_exact, dl_band=dl_band, l_toeplitz=l_toeplitz)
    return out
def mcmpp_ducc_new(spec, l1, l2):
    out= np.empty((spec.shape[0],l1+1,l2+1),dtype=np.float64)
    ducc0.misc.experimental.coupling_matrix_rect_new(spec, optype=(2,)*spec.shape[0], nthreads=nthreads, res=out, l_exact=l_exact, dl_band=dl_band, l_toeplitz=l_toeplitz)
    return out
def mcm02_ducc(spec, l1, l2):
    out= np.empty((spec.shape[0],l1+1,l2+1),dtype=np.float64)
    ducc0.misc.experimental.coupling_matrix_rect(spec, optype=(1,)*spec.shape[0], nthreads=nthreads, res=out, l_exact=l_exact, dl_band=dl_band, l_toeplitz=l_toeplitz)
    return out
def mcm02_ducc_new(spec, l1, l2):
    out= np.empty((spec.shape[0],l1+1,l2+1),dtype=np.float64)
    ducc0.misc.experimental.coupling_matrix_rect_new(spec, optype=(1,)*spec.shape[0], nthreads=nthreads, res=out, l_exact=l_exact, dl_band=dl_band, l_toeplitz=l_toeplitz)
    return out
def mcmEB_ducc(spec, l1, l2):
    out= np.empty((spec.shape[0],l1+1,l2+1),dtype=np.float64)
    ducc0.misc.experimental.coupling_matrix_rect(spec, optype=(3,)*spec.shape[0], nthreads=nthreads, res=out, l_exact=l_exact, dl_band=dl_band, l_toeplitz=l_toeplitz)
    return out
def mcmEB_ducc_new(spec, l1, l2):
    out= np.empty((spec.shape[0],l1+1,l2+1),dtype=np.float64)
    ducc0.misc.experimental.coupling_matrix_rect_new(spec, optype=(3,)*spec.shape[0], nthreads=nthreads, res=out, l_exact=l_exact, dl_band=dl_band, l_toeplitz=l_toeplitz)
    return out
def mcmall_ducc(spec, l1, l2):
    out= np.empty((spec.shape[0],l1+1,l2+1),dtype=np.float64)
    ducc0.misc.experimental.coupling_matrix_rect(spec, optype=(0,1,2,3,)*(spec.shape[0]//4), nthreads=nthreads, res=out, l_exact=l_exact, dl_band=dl_band, l_toeplitz=l_toeplitz)
    return out
def mcmall_ducc_new(spec, l1, l2):
    out= np.empty((spec.shape[0],l1+1,l2+1),dtype=np.float64)
    ducc0.misc.experimental.coupling_matrix_rect_new(spec, optype=(0,1,2,3)*(spec.shape[0]//4), nthreads=nthreads, res=out, l_exact=l_exact, dl_band=dl_band, l_toeplitz=l_toeplitz)
    return out

# lmax up to which the MCM will be computed
l1=5000
l2=5000
lmax=max(l1,l2)

l_exact=-1
dl_band=200
l_toeplitz=170

# number of spectra to process simultaneously

nspec=1

print()
print("Mode coupling matrix computation comparison")
print(f"nspec={nspec}, lmax={lmax}, nthreads={nthreads}")

# we generate the spectra up to 2*lmax+1 to use all Wigner 3j symbols
# but this could also be lower.
spec = np.random.normal(size=(nspec, 4, 2*lmax+1))
spec = np.random.uniform(0.1,1.,size=(nspec, 4, 2*lmax+1))
#spec[3:] = 0
print()
print("TT case:")

t0=time()
duccsq = mcm00_ducc(spec[:,0,:], l1, l2)
print(f"ducc square time: {time()-t0}s")
t0=time()
duccnewsq = mcm00_ducc_new(spec[:,0,:], l1, l2)
print(f"ducc square new time: {time()-t0}s")
print(ducc0.misc.l2error(duccsq,duccnewsq))
print()
print("EE case:")

t0=time()
duccsq = mcmpp_ducc(spec[:,0,:], l1, l2)
print(f"ducc square time: {time()-t0}s")
t0=time()
duccnewsq = mcmpp_ducc_new(spec[:,0,:], l1, l2)
print(f"ducc square new time: {time()-t0}s")
print(ducc0.misc.l2error(duccsq,duccnewsq))

print()
print("TE case:")

t0=time()
duccsq = mcm02_ducc(spec[:,0,:], l1, l2)
print(f"ducc square time: {time()-t0}s")
t0=time()
duccnewsq = mcm02_ducc_new(spec[:,0,:], l1, l2)
print(f"ducc square new time: {time()-t0}s")
print(ducc0.misc.l2error(duccsq,duccnewsq))

print()
print("EB case:")

t0=time()
duccsq = mcmEB_ducc(spec[:,0,:], l1, l2)
print(f"ducc square time: {time()-t0}s")
t0=time()
duccnewsq = mcmEB_ducc_new(spec[:,0,:], l1, l2)
print(f"ducc square new time: {time()-t0}s")
print(ducc0.misc.l2error(duccsq,duccnewsq))
import matplotlib.pyplot as plt
#print(duccsq)
#print(duccnewsq)
#plt.imshow((duccsq-duccnewsq)[0])
#plt.show()

print()
print("all case:")

t0=time()
duccsq = mcmall_ducc(spec[:,0,:], l1, l2)
print(f"ducc square time: {time()-t0}s")
t0=time()
duccnewsq = mcmall_ducc_new(spec[:,0,:], l1, l2)
print(f"ducc square new time: {time()-t0}s")
print(ducc0.misc.l2error(duccsq,duccnewsq))
import matplotlib.pyplot as plt
#print(duccsq)
#print(duccnewsq)
#plt.imshow((duccsq-duccnewsq)[0])
#plt.show()

exit()
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
