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

import ducc0.wgridder as wg
import ducc0.wgridder.experimental as wg_future
import numpy as np
import scipy.fft
from numba import njit
from scipy.special.orthogonal import p_roots

speedoflight = 299792458.


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


def ms2dirty_dft(uvw, freq, ms, nxdirty, nydirty, pixsizex, pixsizey, epsilon, do_wgridding):
    shp = nxdirty, nydirty
    u, v, w, w0, dw, nwplanes, nm1, kernel, gridcoord, slc0, slc1, ng, conjind = init(uvw, freq, shp[0], shp[1], pixsizex, pixsizey, epsilon, do_wgridding)
    x, y = np.meshgrid(*[-ss/2 + np.arange(ss) for ss in shp], indexing='ij')
    x *= pixsizex
    y *= pixsizey
    res = np.zeros(shp)
    for row in range(ms.shape[0]):
        for chan in range(ms.shape[1]):
            phase = freq[chan]/speedoflight*(x*uvw[row, 0]+y*uvw[row, 1]-uvw[row, 2]*nm1)
            res += (ms[row, chan]*np.exp(2j*np.pi*phase)).real
    if do_wgridding:
        return res/(nm1+1)
    return res


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


def dirty2ms_python_slow(uvw, freq, dirty, pixsizex, pixsizey, epsilon, do_wgridding):
    u, v, w, w0, dw, nwplanes, nm1, kernel, gridcoord, slc0, slc1, ng, conjind = init(uvw, freq, dirty.shape[0], dirty.shape[1], pixsizex, pixsizey, epsilon, do_wgridding)
    im = np.zeros(ng)
    im[slc0, slc1] = dirty / ((nm1+1)*kernel.ft(nm1*dw) if do_wgridding else 1)
    im /= kernel.ft(gridcoord[0])[:, None]
    im /= kernel.ft(gridcoord[1])
    ms = np.zeros(len(u), dtype=np.complex128)
    wscreen = np.zeros(im.shape, dtype=np.complex128)
    for ii in range(nwplanes):
        wscreen[slc0, slc1] = np.exp(2j*np.pi*nm1*(w0+ii*dw)) if do_wgridding else 1
        grid = scipy.fft.fft2(np.fft.fftshift(im*wscreen))
        for jj, (uu, vv, ww) in enumerate(zip(u, v, w)):
            wfactor = kernel(ii-(ww-w0)/dw) if do_wgridding else 1
            ms[jj] += wfactor*np.sum(grid*np.outer(kernel(mymod(gridcoord[0]-uu)*ng[0]),
                                                   kernel(mymod(gridcoord[1]-vv)*ng[1])))
    ms[conjind] = ms[conjind].conjugate()
    return ms.reshape(-1, len(freq))


def ms2dirty_python_slow(uvw, freq, ms, nxdirty, nydirty, pixsizex, pixsizey, epsilon, do_wgridding):
    u, v, w, w0, dw, nwplanes, nm1, kernel, gridcoord, slc0, slc1, ng, ms = init(uvw, freq, nxdirty, nydirty, pixsizex, pixsizey, epsilon, do_wgridding, ms)
    im = np.zeros((nxdirty, nydirty))
    for ii in range(nwplanes):
        grid = np.zeros(ng, dtype=ms.dtype)
        for uu, vv, vis in zip(u, v, ms*kernel(ii-(w-w0)/dw) if do_wgridding else ms):
            grid += vis*np.outer(kernel(mymod(gridcoord[0]-uu)*ng[0]),
                                 kernel(mymod(gridcoord[1]-vv)*ng[1]))
        loopim = np.fft.fftshift(scipy.fft.ifft2(grid)*np.prod(ng))
        loopim = loopim[slc0, slc1]
        if do_wgridding:
            loopim *= np.exp(-2j*np.pi*nm1*(w0+ii*dw))
        im += loopim.real
    im /= kernel.ft(gridcoord[0][slc0])[:, None]
    im /= kernel.ft(gridcoord[1][slc1])
    if do_wgridding:
        im /= (nm1+1)*kernel.ft(nm1*dw)
    return im


def ms2dirty_python_fast(uvw, freq, ms, nxdirty, nydirty, pixsizex, pixsizey, epsilon, do_wgridding):
    u, v, w, w0, dw, nwplanes, nm1, kernel, gridcoord, slc0, slc1, ng, ms = init(uvw, freq, nxdirty, nydirty, pixsizex, pixsizey, epsilon, do_wgridding, ms)
    im = np.zeros((nxdirty, nydirty))
    supp = kernel._supp
    for ii in range(nwplanes):
        grid = np.zeros(ng, dtype=ms.dtype)
        for uu, vv, vis in zip(u, v, ms*kernel(ii-(w-w0)/dw) if do_wgridding else ms):
            if vis == 0:
                continue
            ratposx = (uu*ng[0]) % ng[0]
            ratposy = (vv*ng[1]) % ng[1]
            xle = int(np.round(ratposx))-supp//2
            yle = int(np.round(ratposy))-supp//2
            pos = np.arange(0, supp)
            xkernel = kernel(pos-ratposx+xle)
            ykernel = kernel(pos-ratposy+yle)
            for xx in range(supp):
                foo = vis*xkernel[xx]
                myxpos = (xle+xx) % ng[0]
                for yy in range(supp):
                    myypos = (yle+yy) % ng[1]
                    grid[myxpos, myypos] += foo*ykernel[yy]
        loopim = np.fft.fftshift(scipy.fft.ifft2(grid)*np.prod(ng))
        loopim = loopim[slc0, slc1]
        if do_wgridding:
            loopim *= np.exp(-2j*np.pi*nm1*(w0+ii*dw))
        im += loopim.real
    im /= kernel.ft(gridcoord[0][slc0])[:, None]
    im /= kernel.ft(gridcoord[1][slc1])
    if do_wgridding:
        im /= (nm1+1)*kernel.ft(nm1*dw)
    return im


def dirty2ms_python_fast(uvw, freq, dirty, pixsizex, pixsizey, epsilon, do_wgridding):
    u, v, w, w0, dw, nwplanes, nm1, kernel, gridcoord, slc0, slc1, ng, conjind = init(uvw, freq, dirty.shape[0], dirty.shape[1], pixsizex, pixsizey, epsilon, do_wgridding)
    supp = kernel._supp
    im = np.zeros(ng)
    im[slc0, slc1] = dirty / ((nm1+1)*kernel.ft(nm1*dw) if do_wgridding else 1)
    im /= kernel.ft(gridcoord[0])[:, None]
    im /= kernel.ft(gridcoord[1])
    ms = np.zeros(len(u), dtype=np.complex128)
    if do_wgridding:
        wscreen = np.zeros(im.shape, dtype=np.complex128)
    for ii in range(nwplanes):
        if do_wgridding:
            wscreen[slc0, slc1] = np.exp(2j*np.pi*nm1*(w0+ii*dw))
            loopim = im*wscreen
        else:
            loopim = im
        grid = scipy.fft.fft2(np.fft.fftshift(loopim))
        for jj, (uu, vv, ww) in enumerate(zip(u, v, w)):
            if do_wgridding:
                arg = ii-(ww-w0)/dw
                if abs(arg) > supp/2:
                    continue
                wfactor = kernel(arg)
            else:
                wfactor = 1.
            ratposx = (uu*ng[0]) % ng[0]
            ratposy = (vv*ng[1]) % ng[1]
            xle = int(np.round(ratposx))-supp//2
            yle = int(np.round(ratposy))-supp//2
            pos = np.arange(0, supp)
            xkernel = kernel(pos-ratposx+xle)
            ykernel = kernel(pos-ratposy+yle)
            if xle+supp > ng[0] or xle < 0:
                inds = (pos+xle) % ng[0]
                mygrid = grid[inds]
            else:
                mygrid = grid[xle:xle+supp]
            if yle+supp > ng[1] or yle < 0:
                inds = (pos+yle) % ng[1]
                mygrid = mygrid[:, inds]
            else:
                mygrid = mygrid[:, yle:yle+supp]
            assert mygrid.shape == (supp, supp)
            mygrid = mygrid*np.outer(xkernel, ykernel)
            myvis = np.sum(mygrid)
            ms[jj] += myvis*wfactor
    ms[conjind] = ms[conjind].conjugate()
    return ms.reshape(-1, len(freq))


def ms2dirty_numba(uvw, freq, ms, nxdirty, nydirty, pixsizex, pixsizey, epsilon, do_wgridding):
    u, v, w, w0, dw, nwplanes, nm1, kernel, gridcoord, slc0, slc1, ng, ms = init(uvw, freq, nxdirty, nydirty, pixsizex, pixsizey, epsilon, do_wgridding, ms)
    im = np.zeros((nxdirty, nydirty))
    for ii in range(nwplanes):
        myms = ms*kernel(ii-(w-w0)/dw) if do_wgridding else ms
        grid = _ms2dirty_inner_loop(ii, kernel._supp, u, v, w, w0, dw, ng, myms)
        loopim = np.fft.fftshift(scipy.fft.ifft2(grid)*np.prod(ng))
        loopim = loopim[slc0, slc1]
        if do_wgridding:
            loopim *= np.exp(-2j*np.pi*nm1*(w0+ii*dw))
        im += loopim.real
    im /= kernel.ft(gridcoord[0][slc0])[:, None]
    im /= kernel.ft(gridcoord[1][slc1])
    if do_wgridding:
        im /= (nm1+1)*kernel.ft(nm1*dw)
    return im


@njit
def _ms2dirty_inner_loop(ii, supp, u, v, w, w0, dw, ng, myms):
    grid = np.zeros(ng, dtype=myms.dtype)
    kernel = lambda x: np.exp(2.3*supp*(np.sqrt((1-x)*(1+x)) - 1))
    fct = 2./supp
    xkernel = np.empty(supp)
    ykernel = np.empty(supp)
    for uu, vv, vis in zip(u, v, myms):
        if vis == 0:
            continue
        ratposx = (uu*ng[0]) % ng[0]
        ratposy = (vv*ng[1]) % ng[1]
        xle = int(np.round(ratposx))-supp//2
        yle = int(np.round(ratposy))-supp//2
        dx = xle-ratposx
        dy = yle-ratposy
        for i in range(supp):
            xkernel[i] = kernel((i+dx)*fct)
            ykernel[i] = kernel((i+dy)*fct)
        if xle+supp <= ng[0] and yle+supp <= ng[1]:
            for xx in range(supp):
                foo = vis*xkernel[xx]
                myxpos = xle+xx
                for yy in range(supp):
                    grid[myxpos, yle+yy] += foo*ykernel[yy]
        else:
            for xx in range(supp):
                foo = vis*xkernel[xx]
                myxpos = (xle+xx) % ng[0]
                for yy in range(supp):
                    myypos = (yle+yy) % ng[1]
                    grid[myxpos, myypos] += foo*ykernel[yy]
    return grid


# Interface adapters
def ms2dirty_ducc1(uvw, freq, ms, nxdirty, nydirty, pixsizex, pixsizey, epsilon, do_wgridding):
    return wg.ms2dirty(uvw, freq, ms, None, nxdirty, nydirty, pixsizex, pixsizey, 0, 0, epsilon, do_wgridding)

def ms2dirty_ducc2(uvw, freq, ms, nxdirty, nydirty, pixsizex, pixsizey, epsilon, do_wgridding):
    return wg_future.vis2dirty(uvw=uvw, freq=freq, vis=ms, npix_x=nxdirty, npix_y=nydirty, pixsize_x=pixsizex, pixsize_y=pixsizey, epsilon=epsilon, do_wgridding=do_wgridding)

def dirty2ms_ducc1(uvw, freq, dirty, pixsizex, pixsizey, epsilon, do_wgridding):
    return wg.dirty2ms(uvw, freq, dirty, None, pixsizex, pixsizey, 0, 0, epsilon, do_wgridding)

def dirty2ms_ducc2(uvw, freq, dirty, pixsizex, pixsizey, epsilon, do_wgridding):
    return wg_future.dirty2vis(uvw=uvw, freq=freq, dirty=dirty, pixsize_x=pixsizex, pixsize_y=pixsizey, epsilon=epsilon, do_wgridding=do_wgridding)
# End interface adapters


def main():
    fov = 5
    nxdirty, nydirty = 512, 512
    nrow, nchan = 100, 2
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
    for f in (ms2dirty_dft, ms2dirty_python_slow, ms2dirty_python_fast, ms2dirty_numba, ms2dirty_ducc1, ms2dirty_ducc2):
        t0 = time()
        dirty = f(uvw, freq, ms, nxdirty, nydirty, pixsizex, pixsizey, epsilon, do_wgridding)
        t1 = time()-t0
        print(f'Wall time {f.__name__} {t1:.2f} s ({nvis/t1:.0f} vis/s)')
        if dirty0 is not None:
            print(np.max(np.abs(dirty-dirty0)) / np.max(np.abs(dirty0)))
        else:
            dirty0 = dirty

    ms0 = None
    for f in (dirty2ms_python_slow, dirty2ms_python_fast, dirty2ms_ducc1, dirty2ms_ducc2):
        t0 = time()
        ms = f(uvw, freq, dirty, pixsizex, pixsizey, epsilon, do_wgridding)
        t1 = time()-t0
        print(f'Wall time {f.__name__} {t1:.2f} s ({nvis/t1:.0f} vis/s)')
        if ms0 is not None:
            print(np.max(np.abs(ms-ms0)) / np.max(np.abs(ms0)))
        else:
            ms0 = ms


if __name__ == '__main__':
    print("""Disclaimer:
Note that the following values for vis/s are unusually small
since in the demo only few visibilities are used. In typical
real-world scenarios these values are several orders of magnitude
larger (except for "*dft" and "*_python_slow").\n""")
    main()
