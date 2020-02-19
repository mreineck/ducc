import pysharp
import numpy as np
from numpy.testing import assert_allclose
from time import time

def _l2error(a, b):
    return np.sqrt(np.sum(np.abs(a-b)**2)/np.sum(np.abs(a)**2))

lmax = 1023
mmax = lmax

print("Generating spherical harmonic coefficients up to {}".format(lmax))

# number of required a_lm coefficients
nalm = ((mmax+1)*(mmax+2))//2 + (mmax+1)*(lmax-mmax)
# get random a_lm
alm = np.random.uniform(-1., 1., nalm) + 1j*np.random.uniform(-1., 1., nalm)
# make a_lm with m==0 real-valued
alm[0:lmax+1].imag = 0.

job = pysharp.sharpjob_d()
# describe the a_lm array to the job
job.set_triangular_alm_info(lmax, mmax)

nlon = 2*mmax+2
nlat = lmax+1

# lmax+1 iso-latitude rings, first ring at 0.5*pi/nrings 
print("Converting them to Fejer1 grid with {}x{} points".format(nlat, nlon))
job.set_fejer1_geometry(nlat, nlon)

t0=time()
map = job.alm2map(alm)
print("time for map synthesis: {}s".format(time()-t0))
nlat2 = 2*lmax+3
t0=time()
map2 = pysharp.upsample_to_cc(map.reshape((nlat,nlon)), nlat2, False, False)
print("time for upsampling: {}s".format(time()-t0))
job.set_cc_geometry(nlat2, nlon)
t0=time()
alm2 = job.map2alm(map2.reshape((-1,)))
print("time for map analysis: {}s".format(time()-t0))

# make sure input was recovered accurately
print("L2 error: ", _l2error(alm,alm2))

nlon = 2*mmax+2
nlat = lmax+2

# lmax+2 iso-latitude rings, first ring at north pole
print("Converting them to Clenshaw-Curtis grid with {}x{} points".format(nlat, nlon))
job.set_cc_geometry(nlat, nlon)

t0=time()
map = job.alm2map(alm)
print("time for map synthesis: {}s".format(time()-t0))
nlat2 = 2*lmax+3
t0=time()
map2 = pysharp.upsample_to_cc(map.reshape((nlat,nlon)), nlat2, True, True)
print("time for upsampling: {}s".format(time()-t0))
job.set_cc_geometry(nlat2, nlon)
t0=time()
alm2 = job.map2alm(map2.reshape((-1,)))
print("time for map analysis: {}s".format(time()-t0))

# make sure input was recovered accurately
print("L2 error: ", _l2error(alm,alm2))


nlon = 2*mmax+2
nlat = lmax+1

# pixelisation according to https://arxiv.org/abs/1110.6298
print("Converting them to McEwen-Wiaux grid with {}x{} points".format(nlat, nlon))
job.set_mw_geometry(nlat, nlon)

t0=time()
map = job.alm2map(alm)
print("time for map synthesis: {}s".format(time()-t0))
nlat2 = 2*lmax+3
t0=time()
map2 = pysharp.upsample_to_cc(map.reshape((nlat,nlon)), nlat2, False, True)
print("time for upsampling: {}s".format(time()-t0))
job.set_cc_geometry(nlat2, nlon)
t0=time()
alm2 = job.map2alm(map2.reshape((-1,)))
print("time for map analysis: {}s".format(time()-t0))

# make sure input was recovered accurately
print("L2 error: ", _l2error(alm,alm2))
