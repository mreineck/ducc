import ducc0
import numpy as np
from time import time

def nalm(lmax, mmax):
    return ((mmax+1)*(mmax+2))//2 + (mmax+1)*(lmax-mmax)

def random_alm(lmax, mmax, spin, ncomp, rng):
    res = rng.uniform(-1., 1., (ncomp, nalm(lmax, mmax))) \
     + 1j*rng.uniform(-1., 1., (ncomp, nalm(lmax, mmax)))
    # make a_lm with m==0 real-valued
    res[:, 0:lmax+1].imag = 0.
    ofs=0
    for s in range(spin):
        res[:, ofs:ofs+spin-s] = 0.
        ofs += lmax+1-s
    return res

# Computes the locations of a Fibonacci lattice with `npoints` points
def fiblat(npoints):
    phi = (1.+np.sqrt(5.))/2.
    res =np.empty((npoints,2), dtype=np.float64)
    rng = np.arange(npoints, dtype=np.float64)
    res[:,0] = np.arccos(1.-(2/npoints*rng))
    res[:,1] = np.mod(rng/phi,1.)*(2*np.pi)
    return res

lmax = 1187  # This seems to be almost the band limit of the lattice
npoints = 2000000
mmax = lmax
spin = 2
epsilon = 1e-7
nthreads = 8
maxiter = 20

rng = np.random.default_rng(42)
# create random sky ...
alm = random_alm(lmax, mmax, spin, 1 if spin==0 else 2, rng)

loc = fiblat(npoints)
maps = ducc0.sht.experimental.synthesis_general(alm=alm, loc=loc, lmax=lmax, mmax=mmax, spin=spin, epsilon=1e-12, nthreads=nthreads)
#maps =np.random.uniform(0.,1.,maps.shape)
t0=time()
res=ducc0.sht.experimental.pseudo_analysis_general(map=maps, loc=loc, lmax=lmax, mmax=mmax, spin=spin, epsilon=epsilon, nthreads=nthreads, maxiter=maxiter)
dt = time()-t0

print(f"Spherical harmonic analysis on a Fibonacci lattice with {npoints} points.")
print(f"lmax={lmax}, mmax={mmax}, spin={spin}, epsilon={epsilon}")
print(f"Analysis needed {dt:.2f} seconds on {nthreads} threads to do {res[2]} iterations.")
if res[1] == 7:
    print("Maximum number of iterations reached")
elif res[1] == 1:
    print("Approximate solution fond")
elif res[1] == 2:
    print("Least-squares approximation found")
elif res[1] == 3:
    print("Matrix condition number too high")
else:
    print("unclear error")
maps2 = ducc0.sht.experimental.synthesis_general(alm=res[0], loc=loc, lmax=lmax, mmax=mmax, spin=spin, epsilon=1e-12, nthreads=nthreads)
print(f"L2 error between result map and input map: {ducc0.misc.l2error(maps,maps2):.5g}")
print(f"L2 error between result a_lm and input a_lm: {ducc0.misc.l2error(alm,res[0]):.5g}")
