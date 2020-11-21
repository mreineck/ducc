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
import ducc0.sht as sht
import ducc0.misc as misc
import time
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)


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


def convolve(alm1, alm2, lmax):
    job = sht.sharpjob_d()
    job.set_triangular_alm_info(lmax, lmax)
    job.set_gauss_geometry(lmax+1, 2*lmax+1)
    map = job.alm2map(alm1)*job.alm2map(alm2)
    job.set_triangular_alm_info(0, 0)
    return job.map2alm(map)[0]*np.sqrt(4*np.pi)


lmax = 50
kmax = 13
ncomp = 1

# get random sky a_lm
# the a_lm arrays follow the same conventions as those in healpy
slm = random_alm(lmax, lmax, ncomp)

# build beam a_lm
blm = random_alm(lmax, kmax, ncomp)


t0 = time.time()

print("start")
plan = totalconvolve.ConvolverPlan(lmax=lmax, kmax=kmax, sigma=1.5, epsilon=1e-4, nthreads=2)
cube = np.empty((2*kmax+1, plan.Ntheta(), plan.Nphi()), dtype=np.float64)
#slmf = slm.astype(np.complex64)
#blmf = blm.astype(np.complex64)
cube[()]=0
print (np.sum(cube))
plan.getPlane(slm[:,0], blm[:,0], 0, cube[0])
for mbeam in range(1,kmax+1):
    plan.getPlane(slm[:,0], blm[:,0], mbeam, cube[2*mbeam-1], cube[2*mbeam])
print (np.sum(cube))
cube2 = np.empty((plan.Npsi(), plan.Ntheta(), plan.Nphi()), dtype=np.float64)
print("beep")
cube2[:2*kmax+1,:,:] = cube
cube2[1:,:,:]*=0.5   # We still multiply these values by 2 for the old approach
#cube2[2::2,:,:]*=-1
print("beep")
plan.prepPsi(cube2)
print("end")

print("setup time: ", time.time()-t0)
nth = (lmax+1)
nph = (2*lmax+1)


ptg = np.zeros((nth, nph, 3))
ptg[:, :, 0] = (np.pi*(0.5+np.arange(nth))/nth).reshape((-1, 1))
ptg[:, :, 1] = (2*np.pi*(0.5+np.arange(nph))/nph).reshape((1, -1))
ptg[:, :, 2] = np.pi*0.0
ptgbla = ptg.reshape((-1, 3)).astype(np.float64)

res = np.empty(ptgbla.shape[0], dtype=np.float64)
t0 = time.time()
plan.interpol(cube, 0, 0, ptgbla[:,0], ptgbla[:,1], ptgbla[:,2], res)
print("interpolation time: ", time.time()-t0)
res=res.reshape((nth, nph, 1))
print(np.min(res), np.max(res))

res2 = np.zeros(ptgbla.shape[0], dtype=np.float64)
t0 = time.time()
print(np.min(cube2), np.max(cube2))
plan.interpol3(cube2, 0, 0, ptgbla[:,0], ptgbla[:,1], ptgbla[:,2], res2)
print("interpolation2 time: ", time.time()-t0)
res2=res2.reshape((nth, nph, 1))
print(np.min(res2/res), np.max(res2/res))
print(2*kmax+1, cube2.shape[0])

plt.subplot(2, 2, 1)
plt.imshow(res[:, :, 0])
bar2 = np.zeros((nth, nph))
blmfull = np.zeros(slm.shape)+0j
blmfull[0:blm.shape[0], :] = blm
for ith in range(nth):
    rbeamth = misc.rotate_alm(blmfull[:, 0], lmax, ptg[ith, 0, 2], ptg[ith, 0, 0], 0)
    for iph in range(nph):
        rbeam = misc.rotate_alm(rbeamth, lmax, 0, 0, ptg[ith, iph, 1])
        bar2[ith, iph] = convolve(slm[:, 0], rbeam, lmax).real
plt.subplot(2, 2, 2)
plt.imshow(bar2)
plt.subplot(2, 2, 3)
plt.imshow((res-res2)[:,:,0])
# plt.subplot(2, 2, 3)
# plt.imshow(bar[:, :, 0]-res[:,:,0])
# print(np.max(np.abs(bar[:, :, 0]-res[:,:,0]))/np.max(np.abs(bar[:, :, 0])))
plt.subplot(2, 2, 4)
plt.imshow(bar2-res[:, :, 0])
print(np.max(np.abs(bar2-res[:,:,0]))/np.max(np.abs(bar2)))
plt.show()
