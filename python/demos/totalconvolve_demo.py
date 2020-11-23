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


import ducc0.totalconvolve as totalconvolve
import numpy as np
import time

rng = np.random.default_rng(48)


def nalm(lmax, mmax):
    return ((mmax+1)*(mmax+2))//2 + (mmax+1)*(lmax-mmax)


def random_alm(lmax, mmax, ncomp):
    res = rng.uniform(-1., 1., (nalm(lmax, mmax), ncomp)) \
     + 1j*rng.uniform(-1., 1., (nalm(lmax, mmax), ncomp))
    # make a_lm with m==0 real-valued
    res[0:lmax+1, :].imag = 0.
    return res


def compress_alm(alm, lmax):
    res = np.empty(2*len(alm)-lmax-1, dtype=np.float64)
    res[0:lmax+1] = alm[0:lmax+1].real
    res[lmax+1::2] = np.sqrt(2)*alm[lmax+1:].real
    res[lmax+2::2] = np.sqrt(2)*alm[lmax+1:].imag
    return res


def myalmdot(a1, a2, lmax, mmax, spin):
    return np.vdot(compress_alm(a1, lmax), compress_alm(np.conj(a2), lmax))


lmax = 1024
kmax = 13
ncomp = 1
separate = True
nptg = 50000000
epsilon = 1e-4
ofactor = 1.5
nthreads = 0  # use as many threads as available

ncomp2 = ncomp if separate else 1

# get random sky a_lm
# the a_lm arrays follow the same conventions as those in healpy
slm = random_alm(lmax, lmax, ncomp)

# build beam a_lm
blm = random_alm(lmax, kmax, ncomp)


t0 = time.time()
# build interpolator object for slm and blm
foo = totalconvolve.Interpolator(slm, blm, separate, lmax, kmax,
                                 epsilon=epsilon, ofactor=ofactor,
                                 nthreads=nthreads)
t1 = time.time()-t0

print("Convolving sky and beam with lmax=mmax={}, kmax={}".format(lmax, kmax))
print("Interpolation taking place with a maximum error of {}\n"
      "and an oversampling factor of {}".format(epsilon, ofactor))
# supp = foo.support()
# print("(resulting in a kernel support size of {}x{})".format(supp, supp))
if ncomp == 1:
    print("One component")
else:
    print("{} components, which are {}coadded".format(ncomp, "not " if separate else ""))

print("\nDouble precision convolution/interpolation:")
print("preparation of interpolation grid: {}s".format(t1))
t0 = time.time()
nth = lmax+1
nph = 2*lmax+1

ptg = rng.uniform(0., 1., 3*nptg).reshape(nptg, 3)
ptg[:, 0] *= np.pi
ptg[:, 1] *= 2*np.pi
ptg[:, 2] *= 2*np.pi

t0 = time.time()
bar = foo.interpol(ptg)
del foo
print("Interpolating {} random angle triplets: {}s".format(nptg, time.time()-t0))
t0 = time.time()
fake = rng.uniform(0., 1., (ptg.shape[0], ncomp2))
foo2 = totalconvolve.Interpolator(lmax, kmax, ncomp2, epsilon=epsilon, ofactor=ofactor, nthreads=nthreads)
t0 = time.time()
foo2.deinterpol(ptg.reshape((-1, 3)), fake)
print("Adjoint interpolation: {}s".format(time.time()-t0))
t0 = time.time()
bla = foo2.getSlm(blm)
del foo2
print("Computing s_lm: {}s".format(time.time()-t0))
v1 = np.sum([myalmdot(slm[:, i], bla[:, i], lmax, lmax, 0) for i in range(ncomp)])
v2 = np.sum([np.vdot(fake[:, i], bar[:, i]) for i in range(ncomp2)])
print("Adjointness error: {}".format(v1/v2-1.))

# build interpolator object for slm and blm
t0 = time.time()
foo_f = totalconvolve.Interpolator_f(slm.astype(np.complex64), blm.astype(np.complex64), separate, lmax, kmax, epsilon=epsilon, ofactor=ofactor, nthreads=nthreads)
print("\nSingle precision convolution/interpolation:")
print("preparation of interpolation grid: {}s".format(time.time()-t0))

ptgf = ptg.astype(np.float32)
del ptg
fake_f = fake.astype(np.float32)
del fake
t0 = time.time()
bar_f = foo_f.interpol(ptgf)
del foo_f
print("Interpolating {} random angle triplets: {}s".format(nptg, time.time()-t0))
foo2_f = totalconvolve.Interpolator_f(lmax, kmax, ncomp2, epsilon=epsilon, ofactor=ofactor, nthreads=nthreads)
t0 = time.time()
foo2_f.deinterpol(ptgf.reshape((-1, 3)), fake_f)
print("Adjoint interpolation: {}s".format(time.time()-t0))
t0 = time.time()
bla_f = foo2_f.getSlm(blm.astype(np.complex64))
del foo2_f
print("Computing s_lm: {}s".format(time.time()-t0))
v1 = np.sum([myalmdot(slm[:, i], bla_f[:, i], lmax, lmax, 0) for i in range(ncomp)])
v2 = np.sum([np.vdot(fake_f[:, i], bar_f[:, i]) for i in range(ncomp2)])
print("Adjointness error: {}".format(v1/v2-1.))
