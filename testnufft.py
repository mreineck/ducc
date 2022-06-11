import ducc0
import numpy as np
import finufft
from time import time

# number of nonuniform points
M = 10000000

nthreads=8
epsilon=1e-5
forward=False

# the nonuniform points
coord = 2 * np.pi * np.random.uniform(size=2*M).reshape((-1,2))

N1, N2 = 4096,4096

f = (np.random.standard_normal(size=(N1, N2))
     + 1J * np.random.standard_normal(size=(N1, N2)))
f.imag=0
# calculate the 2D type 2 transform
c = finufft.nufft2d2(coord[:,0], coord[:,1], f, nthreads=nthreads,eps=epsilon)
t0=time()
c = finufft.nufft2d2(coord[:,0], coord[:,1], f, nthreads=nthreads,eps=epsilon,debug=1,isign=-1 if forward else 1)
print(time()-t0)
t0=time()
d = finufft.nufft2d1(coord[:,0], coord[:,1], c, (N1, N2), nthreads=nthreads,eps=epsilon,debug=1,isign=-1 if forward else 1)
print(time()-t0)

bla = ducc0.nufft.u2nu(grid=f.astype(np.complex64), coord=coord.astype(np.float64), forward=forward, epsilon=epsilon, nthreads=nthreads)
res = np.empty((N1, N2), dtype=np.complex64)
print (bla.shape, bla.dtype)
blub = ducc0.nufft.nu2u(points=bla, coord=coord.astype(np.float64), forward=forward, epsilon=epsilon, nthreads=nthreads, out=res)

print("L2 error:", ducc0.misc.l2error(c,bla))
print("L2 error:", ducc0.misc.l2error(d, blub))
