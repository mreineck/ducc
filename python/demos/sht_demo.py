# Elementary demo for the ducc0.sht interface using a Gauss-Legendre grid
# I'm not sure I have a perfect equivalent for the DH grid(s) at the moment,
# since they apparently do not include the South Pole. The Clenshaw-Curtis
# and Fejer quadrature rules are very similar (see the documentation in
# sharp_geomhelpers.h). An exact analogon to DH can be added easily, I expect.

import ducc0.sht as sht
import numpy as np
from time import time


def _l2error(a, b):
    return np.sqrt(np.sum(np.abs(a-b)**2)/np.sum(np.abs(a)**2))


# set maximum multipole moment
lmax = 2047
# maximum m. For SHTOOLS this is alway equal to lmax, if I understand correctly.
mmax = lmax

# Number of pixels per ring. Must be >=2*lmax+1, but I'm choosing a larger
# number for which the FFT is faster.
nlon = 4096

# create an object which will do the SHT work
job = sht.sharpjob_d()

# create a set of spherical harmonic coefficients to transform
# Libsharp works exclusively on real-valued maps. The corresponding harmonic
# coefficients are termed a_lm; they are complex numbers with 0<=m<=lmax and
# m<=l<=lmax.
# Symmetry: a_l,-m = (-1)**m*conj(a_l,m).
# The symmetry implies that all coefficients with m==0 are purely real-valued.
# The a_lm are stored in a 1D complex-valued array, in the following order:
# a_(0,0), a(1,0), ..., a_(lmax,0), a(1,1), a(2,1), ... a(lmax,1), ..., a(lmax, mmax)

# number of required a_lm coefficients
nalm = ((mmax+1)*(mmax+2))//2 + (mmax+1)*(lmax-mmax)
# get random a_lm
rng = np.random.default_rng(42)
alm = rng.uniform(-1., 1., nalm) + 1j*rng.uniform(-1., 1., nalm)
# make a_lm with m==0 real-valued
alm[0:lmax+1].imag = 0.

# describe the a_lm array to the job
job.set_triangular_alm_info(lmax, mmax)


print("testing Gauss-Legendre grid")

# Number of iso-latitude rings required for Gauss-Legendre grid
nlat = lmax+1

# describe the Gauss-Legendre geometry to the job
job.set_gauss_geometry(nlat, nlon)

# go from a_lm to map
t0 = time()
map = job.alm2map(alm)
print("time for map synthesis: {}s".format(time()-t0))

# map is a 1D real-valued array with (nlat*nlon) entries. It can be reshaped
# to (nlat, nlon) for plotting.
# Libsharp woks on "1D" maps because it apso supports pixelizations that varying
# number of pixels on each iso-latitude ring, which cannot be represented by 2D
# arrays (e.g. Healpix)

t0 = time()
alm2 = job.map2alm(map)
print("time for map analysis: {}s".format(time()-t0))

# make sure input was recovered accurately
print("L2 error: ", _l2error(alm, alm2))


print("testing Driscoll-Healy grid")

# Number of iso-latitude rings required for Driscoll-Healy grid
nlat = 2*lmax+2

# describe the Gauss-Legendre geometry to the job
job.set_dh_geometry(nlat, nlon)

# go from a_lm to map
t0 = time()
map = job.alm2map(alm)
print("time for map synthesis: {}s".format(time()-t0))

# map is a 1D real-valued array with (nlat*nlon) entries. It can be reshaped
# to (nlat, nlon) for plotting.
# Libsharp woks on "1D" maps because it apso supports pixelizations that varying
# number of pixels on each iso-latitude ring, which cannot be represented by 2D
# arrays (e.g. Healpix)

t0 = time()
alm2 = job.map2alm(map)
print("time for map analysis: {}s".format(time()-t0))

# make sure input was recovered accurately
print("L2 error: ", _l2error(alm, alm2))
