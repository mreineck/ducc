# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# Copyright(C) 2020-2021 Max-Planck-Society

# Elementary demo for the ducc0.sht interface using a Gauss-Legendre grid and
# a Clenshaw-Curtis grid.


import ducc0
import numpy as np
from time import time


# just run on one thread
nthreads = 1

# set maximum multipole moment
lmax = 2047
# maximum m.
mmax = lmax

# Number of pixels per ring. Must be >=2*lmax+1, but I'm choosing a larger
# number for which the FFT is faster.
nlon = 2*lmax+2

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
# add an extra leading dimension to the a_lm. This is necessary since for
# transforms with spin!=0 two a_lm sets are required instead of one.
alm = alm.reshape((1,-1))

print("testing Gauss-Legendre grid with lmax+1 rings")

# Number of iso-latitude rings required for Gauss-Legendre grid
nlat = lmax+1

# go from a_lm to map
t0 = time()
map = ducc0.sht.experimental.synthesis_2d(
    alm=alm, ntheta=nlat, nphi=nlon, lmax=lmax, mmax=mmax, spin=0,
    geometry="GL", nthreads=nthreads)
print("time for map synthesis: {}s".format(time()-t0))

# transform back to a_lm

t0 = time()
alm2 = ducc0.sht.experimental.analysis_2d(
    map=map, lmax=lmax, mmax=mmax, spin=0, geometry="GL", nthreads=nthreads)
print("time for map analysis: {}s".format(time()-t0))

# make sure input was recovered accurately
print("L2 error: ", ducc0.misc.l2error(alm, alm2))


print("testing synthesis/analysis on a Clenshaw-Curtis grid with lmax+2 rings")
print("For 'standard' Clenshaw-Curtis quadrature 2*lmax+2 rings would be needed,")
print("but ducc.sht supports advanced analysis techniques which lower this limit.")
# Number of iso-latitude rings required.
nlat = lmax+2

# go from a_lm to map
t0 = time()
map = ducc0.sht.experimental.synthesis_2d(
    alm=alm, ntheta=nlat, nphi=nlon, lmax=lmax, mmax=mmax, spin=0,
    geometry="CC", nthreads=nthreads)
print("time for map synthesis: {}s".format(time()-t0))

t0 = time()
alm2 = ducc0.sht.experimental.analysis_2d(
    map=map, lmax=lmax, mmax=mmax, spin=0, geometry="CC", nthreads=nthreads)
print("time for map analysis: {}s".format(time()-t0))

# make sure input was recovered accurately
print("L2 error: ", ducc0.misc.l2error(alm, alm2))
