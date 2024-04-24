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
# Copyright(C) 2024 Philipp Arras
# Author: Philipp Arras

from time import time

import numpy as np

import numba.cuda as cuda
import numba.types as types

import ducc0.wgridder as wg
import math
import scipy.fft
from numba import njit
from scipy.special import p_roots
from time import time

# TODO @Martin: wgridder says supp=4, my implementation uses supp=5. Why?


speedoflight = 299792458.

TILESIZE = 16
THREADSPERBLOCK = 32
THREADSPERBLOCK2d = 8, 8

MS2GRID_VISPERTHREADBLOCK = 512

# NOTE that the configuration of the run changes if DEBUG_ON_CPU (e.g., fewer visibilities)
DEBUG_ON_CPU = False

# TODO Figure out if and where to enable fastmath https://numba.readthedocs.io/en/stable/cuda/fastmath.html
gpu_kwargs = {
    "debug": False,
    "fastmath": False
}
precision = "single"


if DEBUG_ON_CPU:
    import scipy.fft as cufft
    import numpy as cp
else:
    import cupyx.scipy.fft as cufft
    import cupy as cp

# TODO Remove this for performance debugging
from numba.core.errors import NumbaWarning
import warnings
warnings.simplefilter('ignore', category=NumbaWarning)

cuda.profile_stop()


class Timer:
    def __init__(self, name):
        self.reset()
        self._name = str(name)

    def push(self, s):
        assert self._t0 is None
        self._t0 = time(), s

    def pop(self):
        self.synchronize()
        t1 = time()
        t0, s = self._t0
        if s not in self._dct:
            self._dct[s] = 0.
        self._dct[s] += t1 - t0
        self._t0 = None

    def poppush(self, s):
        self.pop()
        self.push(s)

    def reset(self):
        self._dct = {}
        self._t0 = None
        self._globalstart = time()

    def synchronize(self):
        if not DEBUG_ON_CPU:
            cuda.synchronize()

    def report(self):
        if self._t0 is not None:
            self.pop()
        self.synchronize()

        total = time() - self._globalstart
        not_allocated = total - sum(self._dct.values())

        values = list(self._dct.values()) + [not_allocated]
        keys = list(self._dct.keys()) + ["<not_allocated>"]

        idx = np.argsort(values)[::-1]
        keys = [keys[ii] for ii in idx]
        values = [values[ii] for ii in idx]

        print()
        title = f"Timer report: {self._name}"
        print(title)
        print("^"*len(title))
        for kk, vv in zip(keys, values):
            print(f"{vv:.3f} s: {kk}")
        print(len(title)*"-")
        print(f"{total:.3f} s: Total")
        print()
        self.reset()


def kernelconfig(*nthreads):
    assert all(isinstance(xx, (int, np.int64)) and xx > 0 for xx in nthreads)

    if len(nthreads) == 1:
        return int(math.ceil(nthreads[0] / THREADSPERBLOCK)), THREADSPERBLOCK

    elif len(nthreads) == 2:
        nthreads = np.array(nthreads, dtype=int)
        return tuple((np.ceil(nthreads / np.array(THREADSPERBLOCK2d))).astype(int)), THREADSPERBLOCK2d

    raise NotImplementedError()


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


@cuda.jit(**gpu_kwargs)
def _apply_wkernel_on_ms(ms, w, ii, w0, dw, supp, outms):
    assert ms.shape == w.shape

    pos = cuda.grid(1)
    if pos < ms.shape[0]:
        x = ii-(w[pos]-w0)/dw
        outms[pos] = ms[pos] * __gk_wkernel(x, supp)


# TODO Can I pass "supp" as constant value?
@cuda.jit(device=True, **gpu_kwargs)
def __gk_wkernel(x, supp):
    # TODO @Martin rewrite as arithmetic expression (is this possible given
    # that the expression below shall not be executed?)
    x = 2*x/supp
    if abs(x) <= 1:
        return math.exp(2.3*supp*(math.sqrt((1-x)*(1+x)) - 1))
    return 0.


@cuda.jit(device=True, **gpu_kwargs)
def __gk_wkernel_no_bound_checks(x, supp):
    x = 2*x/supp
    return math.exp(2.3*supp*(math.sqrt((1-x)*(1+x)) - 1))


@cuda.jit(**gpu_kwargs)
def _2dzero_gpu(arr):
    x, y = cuda.grid(2)
    if x < arr.shape[0] or y < arr.shape[1]:
        arr[x, y] = 0

# TODO Somehow tell numba that this is a compile-time constant
supp = 5
shared_grid_size = TILESIZE+2*supp

@cuda.jit(device=True, **gpu_kwargs)
def _distribute_1d_in_TB(ndata):
    local_tid = cuda.threadIdx.x
    nthreads = min(ndata, cuda.blockDim.x)
    i0 = int(math.ceil((local_tid + 0) / nthreads * ndata))
    i1 = int(math.ceil((local_tid + 1) / nthreads * ndata))
    return i0, i1

@cuda.jit(**gpu_kwargs)
def _ms2grid_gpu_supp5_tiles(
        u, v, ms,
        TB_to_xtile, TB_to_ytile, TB_to_idx_start, TB_to_nvis_in_TB,
        ng, grid_real, grid_imag):

    local_tid = cuda.threadIdx.x  # Index of thread in current TB
    TB = cuda.blockIdx.x

    if local_tid >= TB_to_nvis_in_TB[TB]:
        return

    # Initialize shared grid
    # TODO Try single precision here
    dtype = types.double
    shp = (shared_grid_size, shared_grid_size)
    shared_grid_real = cuda.shared.array(shp, dtype=dtype)
    shared_grid_imag = cuda.shared.array(shp, dtype=dtype)
    if local_tid < shared_grid_size:
        for xx in range(*_distribute_1d_in_TB(shared_grid_size)):
            for yy in range(shared_grid_size):
                shared_grid_real[xx, yy] = 0.
                shared_grid_imag[xx, yy] = 0.
    cuda.syncthreads()

    xtile = TB_to_xtile[TB]
    ytile = TB_to_ytile[TB]

    tile_dx = TILESIZE * xtile - supp//2
    tile_dy = TILESIZE * ytile - supp//2

    idx0 = TB_to_idx_start[TB]
    # TODO Can we use global_tid here somehow?
    data_idx = idx0 + local_tid

    dd = ms[data_idx]
    ratposx = u[data_idx]
    ratposy = v[data_idx]

    # Subpixel offset
    dx = int(round(ratposx)) - supp//2 - ratposx
    dy = int(round(ratposy)) - supp//2 - ratposy

    xle = int(round(ratposx)) - supp//2 - tile_dx
    yle = int(round(ratposy)) - supp//2 - tile_dy

    # @Martin: It makes no difference if the loop is this or the other way around
    ykernel = cuda.local.array(supp, dtype=dtype)
    for j in range(supp):
        ykernel[j] = __gk_wkernel_no_bound_checks(j+dy, supp)
    for i in range(supp):
        xkernel = __gk_wkernel_no_bound_checks(i+dx, supp)
        myxpos = xle+i
        for j in range(supp):
            # TODO Optimize shared memory accesses (bank conflicts are bad)
            val = dd * xkernel * ykernel[j]
            cuda.atomic.add(shared_grid_real, (myxpos, yle+j), val.real)
            cuda.atomic.add(shared_grid_imag, (myxpos, yle+j), val.imag)

    cuda.syncthreads()

    # Write local grid atomically to global grid
    # TODO Split the atomic write operation into a separate kernel and write the shared_grid to global memory here?
    # TODO Distribute this across threads

    if local_tid < shared_grid_size:
        for xx in range(*_distribute_1d_in_TB(shared_grid_size)):
            xpos = (tile_dx + xx) % ng[0]
            for yy in range(shared_grid_size):
                ypos = (tile_dy + yy) % ng[1]
                cuda.atomic.add(grid_real, (xpos, ypos), shared_grid_real[xx, yy])
                cuda.atomic.add(grid_imag, (xpos, ypos), shared_grid_imag[xx, yy])


def bucketsort(buckets, n_buckets):
    if len(buckets) == 0:
        return np.array([], int), np.zeros(n_buckets+1, int)

    idx = cp.argsort(buckets)
    if DEBUG_ON_CPU:
        assert min(buckets) >= 0
        assert max(buckets) < n_buckets
        host_buckets = buckets
    else:
        host_buckets = buckets.get()

    # TODO Compute histogram on GPU
    offsets, _ = np.histogram(host_buckets, bins=np.array(list(np.arange(n_buckets))+ [np.infty]))
    offsets = np.cumsum([0] + list(offsets))

    if DEBUG_ON_CPU:
        assert n_buckets + 1 == len(offsets)
        assert offsets[-1] == len(idx)
        assert offsets[0] == 0
        for tx in range(len(offsets) - 1):
            start, stop = offsets[tx], offsets[tx+1]
            assert len(np.unique(buckets[idx[start:stop]])) in [0, 1]

    return idx, offsets


def apply_periodicity_on_uv(u, v, ng):
    # Alternative ideas by Martin
    # u=u/ng[0] + 100;  u = (u-int(u))
    # xng0 = 1./ng[0]; u= u*xng0 + 100; ...

    u = (cp.array(u) * ng[0]) % ng[0]
    v = (cp.array(v) * ng[1]) % ng[1]
    return u, v


def sort_into_tiles(u, v, ng):
    # Resources for how to implement bucket sort on GPU
    # https://developer.nvidia.com/gpugems/gpugems2/part-vi-simulation-and-numerical-algorithms/chapter-46-improved-gpu-sorting
    # https://stackoverflow.com/questions/16781995/efficient-bucket-sort-on-gpu

    if DEBUG_ON_CPU:
        assert u.ndim == v.ndim == 2
        assert u.shape == v.shape
        assert isinstance(ng, tuple) and len(ng) == 2 and all(isinstance(nn, int) for nn in ng)

        assert ng[0] >= TILESIZE
        assert ng[1] >= TILESIZE

    ntile_u = int(math.ceil(ng[0] / TILESIZE))
    ntile_v = int(math.ceil(ng[1] / TILESIZE))

    start_tile_idx = np.empty((ntile_u, ntile_v), dtype=int)
    stop_tile_idx = np.empty_like(start_tile_idx)

    u = u.ravel()
    v = v.ravel()

    u_tiles = (u / TILESIZE).astype(int)
    v_tiles = (v / TILESIZE).astype(int)

    if DEBUG_ON_CPU:
        assert min(u_tiles) >= 0
        assert max(u_tiles) < ntile_u
        assert min(v_tiles) >= 0
        assert max(v_tiles) < ntile_v

        # print(min(u_tiles), max(u_tiles), ntile_u)
        # print(min(u_tiles), max(v_tiles), ntile_v)

    idx_sorting, offsets_u = bucketsort(u_tiles, ntile_u)

    if DEBUG_ON_CPU:
        for tx in range(len(offsets_u)-1):
            start = offsets_u[tx]
            stop = offsets_u[tx+1]
            idx = idx_sorting[start:stop]
            uu = u[idx]

            if len(uu) > 0:
                if not (uu.min() >= tx*TILESIZE):
                    print("Min fail", tx, uu, tx*TILESIZE)
                if not (uu.max() < (tx+1)*TILESIZE):
                    print("Max fail", tx, uu, (tx+1)*TILESIZE)

    # Bring v coordinates into u-sorted order
    v_tiles = v_tiles[idx_sorting]

    # For each u-bucket, perform bucket sort in v-direction
    for itile_u in range(len(offsets_u) - 1):
        istart, iend = offsets_u[itile_u], offsets_u[itile_u+1]
        idx_sorting_v, offsets_v = bucketsort(v_tiles[istart:iend], ntile_v)

        # Overwrite sorting indices for current u-tile-column
        idx_sorting[istart:iend] = idx_sorting[istart:iend][idx_sorting_v]
        v_tiles[istart:iend] = v_tiles[istart:iend][idx_sorting_v]

        start_tile_idx[itile_u] = offsets_u[itile_u] + offsets_v[:-1]

    # Stop index of tiles equals start index of next tile (with special case at the end)
    stop_tile_idx.ravel()[:-1] = start_tile_idx.ravel()[1:]
    stop_tile_idx.ravel()[-1] = u.size

    if DEBUG_ON_CPU:
        tile_occupancy = stop_tile_idx - start_tile_idx
        # print("Tile occupancy")
        # print(tile_occupancy)

    return idx_sorting, start_tile_idx, stop_tile_idx


def ms2dirty_numba_gpu_verbose(uvw, freq, ms, nxdirty, nydirty, pixsizex, pixsizey, epsilon, do_wgridding):
    return ms2dirty_numba_gpu(uvw, freq, ms, nxdirty, nydirty, pixsizex, pixsizey, epsilon, do_wgridding, verbose=True)

def ms2dirty_numba_gpu(uvw, freq, ms, nxdirty, nydirty, pixsizex, pixsizey, epsilon, do_wgridding, verbose=False):

    timer = Timer("numba_gpu")

    # TODO FFT Tuning guide: https://docs.cupy.dev/en/stable/reference/scipy_fft.html#code-compatibility-features

    timer.push("Preparation")  # -----------------------------------------------

    ofac = 2
    x, y = np.meshgrid(*[-ss/2 + np.arange(ss) for ss in (nxdirty, nydirty)], indexing='ij')
    x *= pixsizex
    y *= pixsizey
    eps = x**2+y**2
    nm1 = -eps/(np.sqrt(1.-eps)+1.)
    ng = int(ofac*nxdirty), int(ofac*nydirty)
    supp = int(np.ceil(np.log10(1/epsilon*(3 if do_wgridding else 2)))) + 1
    # TODO Compute outer product of uv and freq on GPU
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

    timer.poppush("Copy to device")  # -----------------------------------------------

    # TODO do not flatten, do not compute outer product of uvw and freq
    ms = ms[:, None]
    u = np.ascontiguousarray(u[:, None])
    v = np.ascontiguousarray(v[:, None])
    w = np.ascontiguousarray(w[:, None])
    conjind = conjind[:, None]

    if precision == "single":
        u = u.astype(np.float32)
        v = v.astype(np.float32)
        w = w.astype(np.float32)

    nfreq = len(freq)
    ms = ms.reshape(-1, nfreq)
    u = u.reshape(-1, nfreq)
    v = v.reshape(-1, nfreq)
    w = w.reshape(-1, nfreq)
    conjind = conjind

    timer.poppush("Preparation: Create tiles")  # -----------------------------------------------

    # tile_start, tile_stop are refer to indices of sorting_1d_idx
    u, v = apply_periodicity_on_uv(u, v, ng)

    sorting_1d_idx, tile_start, tile_stop = sort_into_tiles(u, v, ng)

    if DEBUG_ON_CPU:
        assert u.min() >= 0
        assert u.max() < ng[0]
        assert v.min() >= 0
        assert v.max() < ng[1]
        for tx in range(tile_start.shape[0]):
            for ty in range(tile_start.shape[1]):
                idx = sorting_1d_idx[tile_start[tx, ty]:tile_stop[tx, ty]]
                uu = u.ravel()[idx]
                vv = v.ravel()[idx]
                # print("tx", tx, "ty", ty) # , tile_start.shape)
                if len(uu) > 0:
                    assert uu.min() >= tx*TILESIZE
                    if not (uu.max() < (tx+1)*TILESIZE):
                        raise AssertionError(uu.max(), (tx+1)*TILESIZE, (tx, ty), ng[0])
                    assert uu.max() < (tx+1)*TILESIZE
                    if not vv.min() >= ty*TILESIZE:
                        print("err3", vv.min(), ty*TILESIZE, (tx, ty), ng[1])
                    if not vv.max() < (ty+1)*TILESIZE:
                        print("err4", vv.max(), (ty+1)*TILESIZE, (tx, ty), ng[1])

    timer.poppush("Copy to device")  # -----------------------------------------------

    im = cp.zeros((nxdirty, nydirty))
    nm1 = cp.array(nm1)  # TODO Generate nm1 on device

    u = cuda.to_device(u)
    v = cuda.to_device(v)
    w = cuda.to_device(w)
    ms = cuda.to_device(ms)

    if ms.dtype == np.complex128:
        # grid_dtype = types.float64
        grid_dtype = np.float64
    elif ms.dtype == np.complex64:
        # grid_dtype = types.float32
        grid_dtype = np.float32
    else:
        raise RuntimeError()

    timer.pop()

    grid_real = cuda.device_array(ng, dtype=grid_dtype)
    grid_imag = cuda.device_array(ng, dtype=grid_dtype)
    nxtiles, nytiles = tile_start.shape

    # Idea sort u, v, w, vis once and get rid of sorting_1d_idx? Compatible with wgridding?
    timer.push("u, v, w, ms ravel and sort")
    u = cuda.to_device(cp.array(u).ravel()[sorting_1d_idx])
    v = cuda.to_device(cp.array(v).ravel()[sorting_1d_idx])
    w = cuda.to_device(cp.array(w).ravel()[sorting_1d_idx])
    ms = cuda.to_device(cp.array(ms).ravel()[sorting_1d_idx])

    sorting_1d_idx, tile_start, tile_stop = sort_into_tiles(cp.array(u), cp.array(v), ng)
    assert list(np.unique(np.diff(sorting_1d_idx))) == [1]


    timer.poppush("Preparation: More work on tile indices")

    TB_to_xtile = []
    TB_to_ytile = []
    TB_to_nvis_in_TB = []
    TB_to_idx_start = []

    nxtiles, nytiles = tile_start.shape
    for xtile in range(nxtiles):
        for ytile in range(nytiles):
            idx0 = tile_start[xtile, ytile]
            idx1 = tile_stop[xtile, ytile]
            nvis_in_tile = idx1 - idx0

            # Skip empty tiles
            if nvis_in_tile == 0:
                continue

            n_TBs_needed = int(math.ceil(nvis_in_tile / MS2GRID_VISPERTHREADBLOCK))
            TB_to_xtile.extend(n_TBs_needed*[xtile])
            TB_to_ytile.extend(n_TBs_needed*[ytile])

            TB_to_idx_start.extend(list(idx0 + MS2GRID_VISPERTHREADBLOCK*ii for ii in range(n_TBs_needed)))

            lst_nvis_in_TB = n_TBs_needed*[MS2GRID_VISPERTHREADBLOCK]
            lst_nvis_in_TB[-1] -= sum(lst_nvis_in_TB) - nvis_in_tile
            assert sum(lst_nvis_in_TB) == nvis_in_tile
            TB_to_nvis_in_TB.extend(lst_nvis_in_TB)

    n_TBs = len(TB_to_xtile)
    TB_to_xtile = cp.array(TB_to_xtile)
    TB_to_ytile = cp.array(TB_to_ytile)
    TB_to_idx_start = cp.array(TB_to_idx_start)
    TB_to_nvis_in_TB = cp.array(TB_to_nvis_in_TB)

    timer.pop()




    for ii in range(nwplanes):
        # print(f"wplane #{ii}")
        if do_wgridding:
            timer.push("Apply wkernel")
            # TODO Get rid of myms and directly write to the grid
            myms = cuda.device_array(ms.shape, dtype=ms.dtype)
            nblocks, nthreadsperblock = kernelconfig(myms.shape[0])
            _apply_wkernel_on_ms[nblocks, nthreadsperblock](ms, w, ii, w0, dw, supp, myms)
            timer.pop()
        else:
            myms = ms

        timer.push("zero grid")

        if DEBUG_ON_CPU:
            grid_real *= 0.
            grid_imag *= 0.
        else:
            # TODO apply by tiles already here?
            nblocks, nthreadsperblock = kernelconfig(*ng)
            _2dzero_gpu[nblocks, nthreadsperblock](grid_real)
            _2dzero_gpu[nblocks, nthreadsperblock](grid_imag)


        ########################################################################

        timer.poppush("ms2grid")
        # TODO: Split kernels into groups of different sizes and launch them together respectively

        # TODO Can we pass compile-time constants somehow differently?
        assert supp == 5

        # TODO Think about if all this indexing logic can be computed in the kernel
        cuda.profile_start()
        _ms2grid_gpu_supp5_tiles[n_TBs, MS2GRID_VISPERTHREADBLOCK](
            u, v, ms,
            TB_to_xtile, TB_to_ytile, TB_to_idx_start, TB_to_nvis_in_TB,
            ng, grid_real, grid_imag)
        cuda.profile_stop()

        ########################################################################



        timer.poppush("FFT (incl. conversion 2xreal -> complex)")

        loopim = cufft.ifft2(cp.array(grid_real) + 1j*cp.array(grid_imag)) * ng[0] * ng[1]

        timer.poppush("Grid correction")
        # TODO Merge all the following operations into one kernel
        loopim = cufft.fftshift(cp.array(loopim))
        loopim = loopim[slc0, slc1]
        if do_wgridding:
            fac = -2j*np.pi*(w0+ii*dw)
            loopim *= cp.exp(fac*nm1)
        timer.poppush("Grid accumulation")
        im += loopim.real
        timer.pop()

    timer.push("Correct uv-kernel")
    krl = Kernel(supp, lambda x: np.exp(2.3*supp*(np.sqrt((1-x)*(1+x)) - 1)))
    post_correction1 = cp.array(1/krl.ft(gridcoord[0][slc0])[:, None])
    post_correction2 = cp.array(1/krl.ft(gridcoord[1][slc1]))
    im *= post_correction1
    im *= post_correction2

    if do_wgridding:
        timer.poppush("Correct w-kernel")
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

    timer.poppush("Copy result to host")
    res = im if DEBUG_ON_CPU else cp.asnumpy(im)
    if verbose and not DEBUG_ON_CPU:
        timer.report()
    return res


# Interface adapters
def ms2dirty_ducc(uvw, freq, ms, nxdirty, nydirty, pixsizex, pixsizey, epsilon, do_wgridding):
    return wg.ms2dirty(uvw, freq, ms, None, nxdirty, nydirty, pixsizex, pixsizey, 0, 0, epsilon, do_wgridding, nthreads=8, verbosity=2)

def dirty2ms_ducc(uvw, freq, dirty, pixsizex, pixsizey, epsilon, do_wgridding):
    return wg.dirty2ms(uvw, freq, dirty, None, pixsizex, pixsizey, 0, 0, epsilon, do_wgridding, nthread=8)
# End interface adapters


def main():
    fov = 5
    if DEBUG_ON_CPU:
        nxdirty, nydirty = 100, 120
        nrow, nchan = 20, 2
    else:
        nxdirty, nydirty = 1024, 1024
        nrow, nchan = 100000, 50
    rng = np.random.default_rng(42)
    pixsizex = fov*np.pi/180/nxdirty
    pixsizey = fov*np.pi/180/nydirty*1.1
    f0 = 1e9
    freq = f0 + np.arange(nchan)*(f0/nchan)
    uvw = (rng.random((nrow, 3))-0.5)/(pixsizex*f0/speedoflight)
    uvw[:, 0] *= 1.5
    uvw[:, 2] /= 20
    ms = rng.random((nrow, nchan))-0.5 + 1j*(rng.random((nrow, nchan))-0.5)
    epsilon = 1e-3
    do_wgridding = True
    do_wgridding = False
    nvis = nrow*nchan

    dirty0 = None

    if precision == "single":
        ms = ms.astype(np.complex64)
    elif precision == "double":
        pass
    else:
        raise RuntimeError()

    # Compiling...
    if not DEBUG_ON_CPU:
        ms2dirty_numba_gpu(uvw, freq, ms, nxdirty, nydirty, pixsizex, pixsizey, epsilon, do_wgridding)

    for f in (ms2dirty_ducc, ms2dirty_numba_gpu_verbose):
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
