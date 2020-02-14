# Elementary demo for pysharp interface using a Gauss-Legendre grid
# I'm not sure I have a perfect equivalent for the DH grid(s) at the moment,
# since they apparently do not include the South Pole. The Clenshaw-Curtis
# and Fejer quadrature rules are very similar (see the documentation in
# sharp_geomhelpers.h). An exact analogon to DH can be added easily, I expect.

import pysharp
import numpy as np
from numpy.testing import assert_allclose

# set maximum multipole moment
lmax = 4095
# maximum m. For SHTOOLS this is alway equal to lmax, if I understand correctly.
mmax = lmax

# Number of iso-latitude rings required for Gauss-Legendre grid
nlat = lmax+1

# Number of pixels per ring. Must be >=2*lmax+1, but I'm choosing a larger
# number for which the FFT is faster.
nlon = 8192

# create an object which will do the SHT work
job = pysharp.sharpjob_d()

# create a set of spherical harmonic coefficients to transform
# Libsharp works exclusively on real-valued maps. The corresponding harmonic
# coefficients are termed a_lm; they are complex numbers with 0<=m<=lmax and
# m<=l<=lmax.
# Symmetry: a_l,-m = (-1)**m*conj(a_l,m).
# The symmetry implies that all coefficients with m=0 are purely real-valued.

# number of required a_lm coefficients
nalm = ((mmax+1)*(mmax+2))//2 + (mmax+1)*(lmax-mmax)
# number of real-valued random numbers to draw
nalm_r = nalm*2-lmax-1
# get random numbers
alm_r = np.random.uniform(-1., 1., nalm_r)

# create the complex-valued a_lm array
alm = np.empty(nalm, dtype=np.complex128)
alm[0:lmax+1] = alm_r[0:lmax+1]
alm[lmax+1:] = np.sqrt(0.5)*(alm_r[lmax+1::2] + 1j*alm_r[lmax+2::2])

# describe the a_lm array to the job
job.set_triangular_alm_info(lmax, mmax)

# describe the Gauss-Legendre geometry to the job
job.set_Gauss_geometry(nlat, nlon)

# go from a_lm to map and back
alm2 = job.map2alm(job.alm2map(alm))

# make sure input was recovered accurately
assert_allclose(alm, alm2)
