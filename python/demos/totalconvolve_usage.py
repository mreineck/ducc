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
# Copyright(C) 2020 Max-Planck-Society


# Short usage demo for the interpol_ng module

import ducc0.totalconvolve as totalconvolve
import numpy as np


# establish a random number generator
rng = np.random.default_rng(np.random.SeedSequence(42))


def nalm(lmax, mmax):
    return ((mmax+1)*(mmax+2))//2 + (mmax+1)*(lmax-mmax)


def random_alm(lmax, mmax, ncomp):
    res = rng.uniform(-1., 1., (nalm(lmax, mmax), ncomp)) \
     + 1j*rng.uniform(-1., 1., (nalm(lmax, mmax), ncomp))
    # make a_lm with m==0 real-valued
    res[0:lmax+1, :].imag = 0.
    return res


# simulation parameters
lmax = 1024  # highest l and m moment for sky, highest l moment for beam
kmax = 13  # highest m moment for beam
ncomp = 3  # T, E and B

# get random sky a_lm
# the a_lm arrays follow the same conventions as those in healpy
slm = random_alm(lmax, lmax, ncomp)

# build beam a_lm
blm = random_alm(lmax, kmax, ncomp)

# build random pointings
npnt = 1000000  # play with this to measure performance
ptg = rng.uniform(0., 1., (npnt, 3))
ptg[:, 0] *= np.pi   # theta
ptg[:, 1] *= 2*np.pi  # phi
ptg[:, 2] *= 2*np.pi  # psi

# build the interpolator
# For a "classic" CMB experiment we want to combine the T, E and B signals,
# so we set `separate` to `False`

print("classic interpolator setup...")
inter_classic = totalconvolve.Interpolator(
    slm, blm, separate=False, lmax=lmax, kmax=kmax, epsilon=1e-4, nthreads=2)
print("...done")

# get interpolated values
print("interpolating...")
res = inter_classic.interpol(ptg)
print("...done")

# res is an array of shape (nptg, 1).
# If we had specified `separate=True`, it would be of shape(nptg, 3).
print(res.shape)

# Since the interpolator object holds large data structures, it should be
# deleted once it is no longer needed
del inter_classic

# Now the same thing for an experiment with HWP. In this case we need the
# interpolated T, E and B signals separate.
separate = True

print("HWP interpolator setup...")
inter_hwp = totalconvolve.Interpolator(
    slm, blm, separate=True, lmax=lmax, kmax=kmax, epsilon=1e-4, nthreads=2)
print("...done")

# get interpolated values
print("interpolating...")
res = inter_hwp.interpol(ptg)
print("...done")

# now res has shape(nptg,3)
print(res.shape)
