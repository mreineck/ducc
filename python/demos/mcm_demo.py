import numpy as np
from time import time
import ducc0

def mcm_ducc(spec, optype, l1, l2):
    nmap = 0
    for op in optype:
        nmap += 2 if op==4 else 1
    out= np.empty((nmap,l1+1,l2+1),dtype=np.float64)
    ducc0.misc.experimental.coupling_matrix_rect(spec, optype=optype, nthreads=nthreads, res=out, l_exact=l_exact, dl_band=dl_band, l_toeplitz=l_toeplitz)
    return out
def mcm_ducc_new(spec, optype, l1, l2):
    nmap = 0
    for op in optype:
        nmap += 2 if op==4 else 1
    out= np.empty((nmap,l1+1,l2+1),dtype=np.float64)
    ducc0.misc.experimental.coupling_matrix_rect_new(spec, optype=optype, nthreads=nthreads, res=out, l_exact=l_exact, dl_band=dl_band, l_toeplitz=l_toeplitz)
    return out

nthreads=8

# lmax up to which the MCM will be computed
l1=5000
l2=5000
lmax=max(l1,l2)

l_exact=-1
dl_band=200
l_toeplitz=170

print("Mode coupling matrix computation comparison")
print(f"lmax={lmax}, nthreads={nthreads}")

# pick one depending on your preferred naming convention
opname = ("TT", "TE", "EE", "EB", "EE/EB")
opname = ("00", "02", "++", "--", "++/--")

for ops in ((0,), (2,), (1,), (3,), (4,), (0,1,2,3,4), (0,0,0,0,0,0,0,0,0,0), (0,1,4,0,1,4,0,1,4,0,1,4,0,1,4)):
    print()
    opstring = tuple(opname[i] for i in ops)
    print(f"{opstring} case:")

    # we generate the spectra up to 2*lmax+1 to use all Wigner 3j symbols
    # but this could also be lower.
    spec = np.random.normal(size=(len(ops), 2*lmax+1))
    t0=time()
    duccsq = mcm_ducc(spec[:len(ops)], ops, l1, l2)
    print(f"ducc square time: {time()-t0}s")
    t0=time()
    duccnewsq = mcm_ducc_new(spec[:len(ops)], ops, l1, l2)
    print(f"ducc square new time: {time()-t0}s")
    print("L2 error:", ducc0.misc.l2error(duccsq,duccnewsq))
