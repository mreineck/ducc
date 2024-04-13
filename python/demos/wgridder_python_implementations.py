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
# Author: Philipp Arras

from time import time

import numpy as np

import numba.cuda as cuda
import numba.types as types
import cupyx.scipy.fft as cufft
import cupy as cp

import ducc0.wgridder as wg
import math
import scipy.fft
from numba import njit
from scipy.special import p_roots

speedoflight = 299792458.
THREADSPERBLOCK = 32
# TODO Figure out if and where to enable fastmath https://numba.readthedocs.io/en/stable/cuda/fastmath.html
gpu_kwargs = {
    "debug": False,
    "fastmath": False
}
precision = "double"

# TODO Remove this for performance debugging
from numba.core.errors import NumbaWarning
import warnings
warnings.simplefilter('ignore', category=NumbaWarning)



def mymod(arr):
    return arr-np.floor(arr)-0.5


def init(uvw, freq, nxdirty, nydirty, pixsizex, pixsizey, epsilon, do_wgridding, ms=None):
    ofac = 2
    x, y = np.meshgrid(*[-ss/2 + np.arange(ss) for ss in (nxdirty, nydirty)], indexing='ij')
    x *= pixsizex
    y *= pixsizey
    eps = x**2+y**2
    nm1 = -eps/(np.sqrt(1.-eps)+1.)
    ng = ofac*nxdirty, ofac*nydirty
    supp = int(np.ceil(np.log10(1/epsilon*(3 if do_wgridding else 2)))) + 1
    kernel = es_kernel(supp, 2.3*supp)
    uvw = np.transpose((uvw[..., None]*freq/speedoflight), (0, 2, 1)).reshape(-1, 3)
    conjind = uvw[:, 2] < 0
    uvw[conjind] *= -1
    u, v, w = uvw.T
    if do_wgridding:
        wmin, wmax = np.min(w), np.max(w)
        dw = 1/ofac/np.max(np.abs(nm1))/2
        nwplanes = int(np.ceil((wmax-wmin)/dw+supp)) if do_wgridding else 1
        w0 = (wmin+wmax)/2 - dw*(nwplanes-1)/2
    else:
        nwplanes, w0, dw = 1, None, None
    gridcoord = [np.linspace(-0.5, 0.5, nn, endpoint=False) for nn in ng]
    slc0, slc1 = slice(nxdirty//2, nxdirty*3//2), slice(nydirty//2, nydirty*3//2)
    u *= pixsizex
    v *= pixsizey
    if ms is not None:
        ms = ms.flatten()
        ms[conjind] = ms[conjind].conjugate()
        return u, v, w, w0, dw, nwplanes, nm1, kernel, gridcoord, slc0, slc1, ng, ms
    return u, v, w, w0, dw, nwplanes, nm1, kernel, gridcoord, slc0, slc1, ng, conjind


class Kernel:
    def __init__(self, supp, func):
        self._func = func
        self._supp = supp

    def ft(self, x):
        x = x*self._supp*np.pi
        nroots = 2*self._supp
        if self._supp % 2 == 0:
            nroots += 1
        q, weights = p_roots(nroots)
        ind = q > 0
        weights = weights[ind]
        q = q[ind]
        kq = np.outer(x, q) if len(x.shape) == 1 else np.einsum('ij,k->ijk', x, q)
        arr = np.sum(weights*self._raw(q)*np.cos(kq), axis=-1)
        return self._supp*arr

    def __call__(self, x):
        return self._raw(x/self._supp*2)

    def _raw(self, x):
        ind = np.logical_and(x <= 1, x >= -1)
        res = np.zeros_like(x)
        res[ind] = self._func(x[ind])
        return res


def es_kernel(supp, beta):
    return Kernel(supp, lambda x: np.exp(beta*(pow((1-x)*(1+x), 0.5) - 1)))


@cuda.jit(**gpu_kwargs)
def _apply_wkernel_on_ms(ms, w, ii, w0, dw, supp, outms):
    assert ms.shape == w.shape

    pos = cuda.grid(1)
    if pos < ms.shape[0]:
        for jj in range(ms.shape[1]):
            x = ii-(w[pos, jj]-w0)/dw
            outms[pos, jj] = ms[pos, jj] * __gk_wkernel(x, supp)


# TODO Can I pass "supp" as constant value?
@cuda.jit(device=True, **gpu_kwargs)
def __gk_wkernel(x, supp):
    # TODO @Martin rewrite as arithmetic expression (is this possible given
    # that the expression below shall not be executed?)
    x = 2*x/supp
    if abs(x) <= 1:
        return math.exp(2.3*supp*(math.pow((1-x)*(1+x), 0.5) - 1))
    return 0.


@cuda.jit(**gpu_kwargs)
def _ms2grid_gpu_supp5(u, v, ms, ng, grid_real, grid_imag):
    supp = 5

    xkernel = cuda.local.array(supp, dtype=types.double)
    ykernel = cuda.local.array(supp, dtype=types.double)

    pos = cuda.grid(1)
    if pos >= u.shape[0]:
        return

    for ifreq in range(u.shape[1]):
        if ms[pos, ifreq] == 0:
            continue

        ratposx = (u[pos, ifreq]*ng[0]) % ng[0]
        ratposy = (v[pos, ifreq]*ng[1]) % ng[1]
        xle = int(round(ratposx))-supp//2
        yle = int(round(ratposy))-supp//2
        dx = xle-ratposx
        dy = yle-ratposy
        for i in range(supp):
            knl = lambda x: math.exp(2.3*supp*(math.pow((1-x)*(1+x), 0.5) - 1))
            fct = 2./supp
            xkernel[i] = knl((i+dx)*fct)
            ykernel[i] = knl((i+dy)*fct)
        if xle+supp <= ng[0] and yle+supp <= ng[1]:
            for xx in range(supp):
                foo = ms[pos, ifreq]*xkernel[xx]
                myxpos = xle+xx
                for yy in range(supp):
                    val = foo*ykernel[yy]*ng[0]*ng[1]
                    # TODO Highly inperformant due to global atomics
                    cuda.atomic.add(grid_real, (myxpos, yle+yy), val.real)
                    cuda.atomic.add(grid_imag, (myxpos, yle+yy), val.imag)
        else:
            for xx in range(supp):
                foo = ms[pos, ifreq]*xkernel[xx]
                myxpos = (xle+xx) % ng[0]
                for yy in range(supp):
                    val = foo*ykernel[yy]*ng[0]*ng[1]
                    myypos = (yle+yy) % ng[1]
                    cuda.atomic.add(grid_real, (myxpos, myypos), val.real)
                    cuda.atomic.add(grid_imag, (myxpos, myypos), val.imag)


def ms2dirty_numba_gpu(uvw, freq, ms, nxdirty, nydirty, pixsizex, pixsizey, epsilon, do_wgridding):

    # TODO FFT Tuning guide: https://docs.cupy.dev/en/stable/reference/scipy_fft.html#code-compatibility-features

    ofac = 2
    x, y = np.meshgrid(*[-ss/2 + np.arange(ss) for ss in (nxdirty, nydirty)], indexing='ij')
    x *= pixsizex
    y *= pixsizey
    eps = x**2+y**2
    nm1 = -eps/(np.sqrt(1.-eps)+1.)
    ng = ofac*nxdirty, ofac*nydirty
    supp = int(np.ceil(np.log10(1/epsilon*(3 if do_wgridding else 2)))) + 1
    kernel = es_kernel(supp, 2.3*supp)
    uvw = np.transpose((uvw[..., None]*freq/speedoflight), (0, 2, 1)).reshape(-1, 3)
    conjind = uvw[:, 2] < 0
    uvw[conjind] *= -1
    u, v, w = uvw.T
    if do_wgridding:
        wmin, wmax = np.min(w), np.max(w)
        dw = 1/ofac/np.max(np.abs(nm1))/2
        nwplanes = int(np.ceil((wmax-wmin)/dw+supp)) if do_wgridding else 1
        w0 = (wmin+wmax)/2 - dw*(nwplanes-1)/2
    else:
        nwplanes, w0, dw = 1, None, None
    gridcoord = [np.linspace(-0.5, 0.5, nn, endpoint=False) for nn in ng]
    slc0, slc1 = slice(nxdirty//2, nxdirty*3//2), slice(nydirty//2, nydirty*3//2)
    u *= pixsizex
    v *= pixsizey
    ms = ms.flatten()  # TODO do not flatten ms...
    ms[conjind] = ms[conjind].conjugate()

    im = cp.zeros((nxdirty, nydirty))
    nm1 = cp.array(nm1)  # TODO Generate nm1 on device

    # TODO do not flatten, do not compute outer product of uvw and freq
    ms = ms[:, None]
    u = np.ascontiguousarray(u[:, None])
    v = np.ascontiguousarray(v[:, None])
    w = np.ascontiguousarray(w[:, None])
    conjind = conjind[:, None]

    nfreq = len(freq)
    ms = ms.reshape(-1, nfreq)
    u = u.reshape(-1, nfreq)
    v = v.reshape(-1, nfreq)
    w = w.reshape(-1, nfreq)
    conjind = conjind

    myms = ms.copy()

    u = cuda.to_device(u)
    v = cuda.to_device(v)
    w = cuda.to_device(w)
    ms = cuda.to_device(ms)

    blockspergrid = (w.shape[0] + (THREADSPERBLOCK - 1)) // THREADSPERBLOCK

    if ms.dtype == np.complex128:
        # grid_dtype = types.float64
        grid_dtype = np.float64
    elif ms.dtype == np.complex64:
        # grid_dtype = types.float32
        grid_dtype = np.float32
    else:
        raise RuntimeError()

    for ii in range(nwplanes):
        if do_wgridding:
            # TODO Get rid of myms and directly write to the grid
            myms = cuda.device_array(ms.shape, dtype=ms.dtype)
            _apply_wkernel_on_ms[blockspergrid, THREADSPERBLOCK](ms, w, ii, w0, dw, supp, myms)
        else:
            myms = ms

        # TODO Allocate grid outside of loop and zero it within the ms2grid kernel
        grid2_real = cuda.to_device(np.zeros(ng, dtype=grid_dtype))
        grid2_imag = cuda.to_device(np.zeros(ng, dtype=grid_dtype))
        # TODO Can we pass compile-time constants somehow differently?
        assert supp == 5
        _ms2grid_gpu_supp5[blockspergrid, THREADSPERBLOCK](u, v, myms, ng, grid2_real, grid2_imag)

        loopim = cufft.ifft2(cp.array(grid2_real) + 1j*cp.array(grid2_imag))

        # TODO Merge all the following operations into one kernel
        loopim = cufft.fftshift(cp.array(loopim))
        loopim = loopim[slc0, slc1]
        if do_wgridding:
            fac = -2j*np.pi*(w0+ii*dw)
            loopim *= cp.exp(fac*nm1)
        im += loopim.real

    post_correction1 = cp.array(1/kernel.ft(gridcoord[0][slc0])[:, None])
    post_correction2 = cp.array(1/kernel.ft(gridcoord[1][slc1]))
    im *= post_correction1
    im *= post_correction2

    if do_wgridding:
        # TODO Write post_correction3 as proper GPU kernel
        x = nm1*dw*supp*np.pi
        nroots = 2*supp
        if supp % 2 == 0:
            nroots += 1
        q, weights = p_roots(nroots)
        ind = q > 0
        weights = cp.array(weights[ind])
        q = cp.array(q[ind])
        assert len(x.shape) != 1
        kq = cp.einsum('ij,k->ijk', x, q)
        kernel = lambda x: cp.exp(2.3*supp*(pow((1-x)*(1+x), 0.5) - 1))
        post_correction3 = supp*cp.sum(weights*kernel(q)*cp.cos(kq), axis=-1)
        im /= (nm1+1)*post_correction3

    return cp.asnumpy(im)


# Interface adapters
def ms2dirty_ducc(uvw, freq, ms, nxdirty, nydirty, pixsizex, pixsizey, epsilon, do_wgridding):
    return wg.ms2dirty(uvw, freq, ms, None, nxdirty, nydirty, pixsizex, pixsizey, 0, 0, epsilon, do_wgridding, nthreads=8)

def dirty2ms_ducc(uvw, freq, dirty, pixsizex, pixsizey, epsilon, do_wgridding):
    return wg.dirty2ms(uvw, freq, dirty, None, pixsizex, pixsizey, 0, 0, epsilon, do_wgridding, nthread=8)
# End interface adapters


def main():
    fov = 5
    nxdirty, nydirty = 1024, 1024
    nrow, nchan = 100000, 10
    rng = np.random.default_rng(42)
    pixsizex = fov*np.pi/180/nxdirty
    pixsizey = fov*np.pi/180/nydirty*1.1
    f0 = 1e9
    freq = f0 + np.arange(nchan)*(f0/nchan)
    uvw = (rng.random((nrow, 3))-0.5)/(pixsizex*f0/speedoflight)
    uvw[:, 2] /= 20
    ms = rng.random((nrow, nchan))-0.5 + 1j*(rng.random((nrow, nchan))-0.5)
    epsilon = 1e-3
    do_wgridding = True
    nvis = nrow*nchan

    dirty0 = None

    if precision == "single":
        ms = ms.astype(np.complex64)
    elif precision == "double":
        pass
    else:
        raise RuntimeError()

    # Compiling...
    ms2dirty_numba_gpu(uvw, freq, ms, nxdirty, nydirty, pixsizex, pixsizey, epsilon, do_wgridding)

    for f in (ms2dirty_ducc, ms2dirty_numba_gpu):
        t0 = time()
        dirty = f(uvw, freq, ms, nxdirty, nydirty, pixsizex, pixsizey, epsilon, do_wgridding)
        t1 = time()-t0
        print(f'Wall time {f.__name__} {t1:.2f} s ({nvis/t1:.0f} vis/s)')
        if dirty0 is not None:
            err = np.max(np.abs(dirty-dirty0)) / np.max(np.abs(dirty0))
            if err > epsilon:
                raise AssertionError(f"Implementation not accurate: err={err}. Should be: epsilon={epsilon}\n{dirty}\n\n{dirty0}")
            assert np.max(np.abs(dirty-dirty0)) / np.max(np.abs(dirty0)) < epsilon
        else:
            dirty0 = dirty

    # TODO
    # ms0 = None
    # for f in (dirty2ms_ducc, ms2dirty_numba_gpu):
    #     t0 = time()
    #     ms = f(uvw, freq, dirty, pixsizex, pixsizey, epsilon, do_wgridding)
    #     t1 = time()-t0
    #     print(f'Wall time {f.__name__} {t1:.2f} s ({nvis/t1:.0f} vis/s)')
    #     if ms0 is not None:
    #         print(np.max(np.abs(ms-ms0)) / np.max(np.abs(ms0)))
    #     else:
    #         ms0 = ms


if __name__ == '__main__':
    main()
