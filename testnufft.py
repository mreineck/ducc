import ducc0
import numpy as np
import finufft
from time import time

# number of nonuniform points
M = 10000000

nthreads=2
epsilon=1e-8
forward=False

# the nonuniform points
coord = 2 * np.pi * np.random.uniform(size=M)

N1 = 255

f = (np.random.standard_normal(size=(N1,))
     + 1J * np.random.standard_normal(size=(N1,)))

# calculate the 3D type 2 transform
c = finufft.nufft1d2(coord, f, nthreads=nthreads,eps=epsilon)
t0=time()
c = finufft.nufft1d2(coord, f, nthreads=nthreads,eps=epsilon,debug=1,isign=-1 if forward else 1)
print(time()-t0)
t0=time()
d = finufft.nufft1d1(coord, c, (N1,), nthreads=nthreads,eps=epsilon,debug=1,isign=-1 if forward else 1)
print(time()-t0)

bla = ducc0.nufft.u2nu(grid=f, coord=coord, forward=forward, epsilon=epsilon, nthreads=nthreads, verbosity=1)
res = np.empty((N1,), dtype=np.complex128)
print (bla.shape, bla.dtype)
blub = ducc0.nufft.nu2u(points=bla, coord=coord, forward=forward, epsilon=epsilon, nthreads=nthreads, out=res, verbosity=1)

print("L2 error:", ducc0.misc.l2error(c,bla))
print("L2 error:", ducc0.misc.l2error(d, blub))


coord = 2 * np.pi * np.random.uniform(size=3*M).reshape((-1,3))

N1, N2,N3 = 255,480, 370

f = (np.random.standard_normal(size=(N1, N2, N3))
     + 1J * np.random.standard_normal(size=(N1, N2, N3)))

# calculate the 3D type 2 transform
c = finufft.nufft3d2(coord[:,0], coord[:,1], coord[:,2], f, nthreads=nthreads,eps=epsilon)
t0=time()
c = finufft.nufft3d2(coord[:,0], coord[:,1], coord[:,2], f, nthreads=nthreads,eps=epsilon,debug=1,isign=-1 if forward else 1)
print(time()-t0)
t0=time()
d = finufft.nufft3d1(coord[:,0], coord[:,1], coord[:,2], c, (N1, N2,N3), nthreads=nthreads,eps=epsilon,debug=1,isign=-1 if forward else 1)
print(time()-t0)

bla = ducc0.nufft.u2nu(grid=f, coord=coord, forward=forward, epsilon=epsilon, nthreads=nthreads, verbosity=1)
res = np.empty((N1, N2, N3), dtype=np.complex128)
print (bla.shape, bla.dtype)
blub = ducc0.nufft.nu2u(points=bla, coord=coord, forward=forward, epsilon=epsilon, nthreads=nthreads, out=res, verbosity=1)

print("L2 error:", ducc0.misc.l2error(c,bla))
print("L2 error:", ducc0.misc.l2error(d, blub))
