import numpy as np
from time import time

# We have to set nthreads before importing pspy, otherwise the
# environment variable change will have no effect.
nthreads=8
import os
os.environ["OMP_NUM_THREADS"]=str(nthreads)

# This must happen after setting OMP_NUM_THREADS!
from pspy.mcm_fortran.mcm_fortran import mcm_compute as mcm_fortran
import ducc0

def tri2full(tri, lmax):
    res = np.zeros((tri.shape[0], tri.shape[1], lmax+1, lmax+1))
    lfac = 2.*np.arange(lmax+1) + 1.
    for l1 in range(lmax+1):
        startidx = l1*(lmax+1) - (l1*(l1+1))//2
        res[:,:,l1,l1:] = lfac[l1:] * tri[:,:, startidx+l1:startidx+lmax+1]
        res[:,:,l1:,l1] = (2*l1+1) * tri[:,:, startidx+l1:startidx+lmax+1]
    return res
    

# This routine is more complicated than mcm00_ducc, since a few multiplication
# steps are carried out in Python in pspy, and since the array indices are
# a bit different. Overall this should not have noticeable impact on performance
# at higher lmax. 
def mcm00_pspy(spec, lmax):
    nspec = spec.shape[0]
    lrange_spec = np.arange(spec.shape[1])
    res=np.zeros((nspec, lmax+1, lmax+1))
    mcmtmp = np.empty((lmax+1, lmax+1))
    for i in range(nspec):
        wcl = spec[i]*(2*lrange_spec+1)
        mcm_fortran.calc_coupling_spin0(wcl, lmax+1, lmax+1, lmax+1, mcmtmp.T)
        mcm_fortran.fill_upper(mcmtmp.T)
        mcmtmp *= (np.arange(2, lmax+3)*2+1.)/(4*np.pi)
        res[i, 2:, 2:] = mcmtmp[:-2,:-2]
    return res

def mcm02_pspy(spec, lmax):
    nspec = spec.shape[0]
    lrange_spec = np.arange(spec.shape[2])
    res=np.zeros((nspec, 5, lmax+1, lmax+1))
    mcmtmp = np.empty((5, lmax+1, lmax+1))
    for i in range(nspec):
        wcl = spec[i]*((2*lrange_spec+1).reshape((1,-1)))
        mcm_fortran.calc_coupling_spin0and2(wcl[0], wcl[1], wcl[2], wcl[3], lmax+1, lmax+1, lmax+1, mcmtmp.T)
        for j in range(5):
            mcm_fortran.fill_upper(mcmtmp[j].T)
        mcmtmp *= (np.arange(2, lmax+3)*2+1.)/(4*np.pi)
        res[i, :, 2:, 2:] = mcmtmp[:,:-2,:-2]
    return res

def mcm02_pure_pspy(spec, lmax):
    nspec = spec.shape[0]
    lrange_spec = np.arange(spec.shape[2])
    res=np.zeros((nspec, 5, lmax+1, lmax+1))
    mcmtmp = np.empty((5, lmax+1, lmax+1))
    for i in range(nspec):
        wcl = spec[i]*((2*lrange_spec+1).reshape((1,-1)))
        mcm_fortran.calc_mcm_spin0and2_pure(wcl[0], wcl[1], wcl[2], wcl[3], mcmtmp.T)
        mcmtmp *= (np.arange(2, lmax+3)*2+1.)/(4*np.pi)
        res[i, :, 2:, 2:] = mcmtmp[:,:-2,:-2]
    return res


def mcm00_ducc_square(spec, lmax):
    return ducc0.misc.experimental.coupling_matrix_spin0(spec, lmax, nthreads=nthreads)

def mcm00_ducc_tri(spec, lmax):
    out= np.empty((spec.shape[0],1,((lmax+1)*(lmax+2))//2),dtype=np.float32)
    ducc0.misc.experimental.coupling_matrix_spin0and2_tri(spec.reshape((spec.shape[0],1,spec.shape[1])), lmax, (0,0,0,0), (0,-1,-1,-1,-1), nthreads=nthreads, res=out)
    return out

def mcm02_ducc_square(spec, lmax):
    return ducc0.misc.experimental.coupling_matrix_spin0and2(spec, lmax, nthreads=nthreads)
def mcm02_ducc_tri(spec, lmax):
    out= np.empty((spec.shape[0],5,((lmax+1)*(lmax+2))//2),dtype=np.float32)
    ducc0.misc.experimental.coupling_matrix_spin0and2_tri(spec[:,:,:], lmax, (0,1,2,3), (0,1,2,3,4), nthreads=nthreads, res=out)
    return out

def mcm02_pure_ducc(spec, lmax):
    res = np.zeros((nspec, 5, lmax+1, lmax+1))
    return ducc0.misc.experimental.coupling_matrix_spin0and2_pure(spec, lmax, nthreads=nthreads, res=res)

# lmax up to which the MCM will be computed
lmax=1000
# number of spectra to process simultaneously
    
nspec=10

print()
print("Mode coupling matrix computation comparison")
print(f"nspec={nspec}, lmax={lmax}, nthreads={nthreads}")

# we generate the spectra up to 2*lmax+1 to use all Wigner 3j symbols
# but this could also be lower.
spec = np.random.normal(size=(nspec, 4, 2*lmax+1))

print()
print("Spin 0 case:")

t0=time()
pspy = mcm00_pspy(spec[:,0,:], lmax)
print(f"pspy time: {time()-t0}s")

t0=time()
ducc_1 = mcm00_ducc_square(spec[:,0,:], lmax)
print(f"ducc square time: {time()-t0}s")
t0=time()
ducc_2 = mcm00_ducc_tri(spec[:,0,:], lmax)
print(f"ducc triangular time (single precision): {time()-t0}s")

# compare the results
print(f"L2 error between pspy and square ducc solutions: {ducc0.misc.l2error(pspy[:,2:,2:],ducc_1[:,2:,2:])}")
print(f"L2 error between ducc solutions: {ducc0.misc.l2error(ducc_1, tri2full(ducc_2,lmax)[:,0,:,:])}")

print()
print("Spin 0and2 case:")

t0=time()
pspy = mcm02_pspy(spec, lmax)
print(f"pspy time: {time()-t0}s")

t0=time()
ducc_1 = mcm02_ducc_square(spec, lmax)
print(f"ducc square time: {time()-t0}s")

t0=time()
ducc_2 = mcm02_ducc_tri(spec, lmax)
print(f"ducc triangular time (single precision): {time()-t0}s")

# compare the results
print(f"L2 error between pspy and square ducc solutions: {ducc0.misc.l2error(pspy[:,:,2:,2:],ducc_1[:,:,2:,2:])}")
print(f"L2 error between ducc solutions: {ducc0.misc.l2error(ducc_1,tri2full(ducc_2,lmax))}")

print()
print("Spin 0and2_pure case:")

t0=time()
pspy = mcm02_pure_pspy(spec, lmax)
print(f"pspy time: {time()-t0}s")

t0=time()
ducc_1 = mcm02_pure_ducc(spec, lmax)
print(f"ducc square time: {time()-t0}s")

# compare the results
print(f"L2 error between pspy and square ducc solutions: {ducc0.misc.l2error(pspy[:,:,2:,2:],ducc_1[:,:,2:,2:])}")
