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

from itertools import product

import ducc0.wgridder as ng
try:
    import finufft
    have_finufft = True
except ImportError:
    have_finufft = False
import numpy as np
import pytest
from numpy.testing import assert_allclose

pmp = pytest.mark.parametrize
SPEEDOFLIGHT = 299792458.


# attempt to write a more accurate version of numpy.vdot()
def my_vdot(a, b):
    import math
    if (np.issubdtype(a.dtype, np.complexfloating)
            or np.issubdtype(b.dtype, np.complexfloating)):
        tmp = (np.conj(a)*b).reshape((-1,))
        return math.fsum(tmp.real)+1j*math.fsum(tmp.imag)
    else:
        tmp = (a*b).reshape((-1,))
        return math.fsum(tmp)


def _l2error(a, b):
    return np.sqrt(np.sum(np.abs(a-b)**2)/np.maximum(np.sum(np.abs(a)**2),
                                                     np.sum(np.abs(b)**2)))


def explicit_gridder(uvw, freq, ms, wgt, nxdirty, nydirty, xpixsize, ypixsize,
                     apply_w, mask):
    x, y = np.meshgrid(*[-ss/2 + np.arange(ss) for ss in [nxdirty, nydirty]],
                       indexing='ij')
    x *= xpixsize
    y *= ypixsize
    res = np.zeros((nxdirty, nydirty))
    eps = x**2+y**2
    if apply_w:
        nm1 = -eps/(np.sqrt(1.-eps)+1.)
        n = nm1+1
    else:
        nm1 = 0.
        n = 1.
    for row in range(ms.shape[0]):
        for chan in range(ms.shape[1]):
            if mask is not None and mask[row, chan] == 0:
                continue
            phase = (freq[chan]/SPEEDOFLIGHT *
                     (x*uvw[row, 0] + y*uvw[row, 1] - uvw[row, 2]*nm1))
            if wgt is None:
                res += (ms[row, chan]*np.exp(2j*np.pi*phase)).real
            else:
                res += (ms[row, chan]*wgt[row, chan]
                        * np.exp(2j*np.pi*phase)).real
    return res/n


def with_finufft(uvw, freq, ms, wgt, nxdirty, nydirty, xpixsize, ypixsize,
                 mask, epsilon):
    u = np.outer(uvw[:, 0], freq)*(xpixsize/SPEEDOFLIGHT)*2*np.pi
    v = np.outer(uvw[:, 1], freq)*(ypixsize/SPEEDOFLIGHT)*2*np.pi
    if wgt is not None:
        ms = ms*wgt
    if mask is not None:
        ms = ms*mask
    eps = epsilon/10  # Apparently finufft measures epsilon differently
    # Plan on the fly
    res0 = finufft.nufft2d1(u.ravel(), v.ravel(), ms.ravel(), (nxdirty,
                            nydirty), eps=eps).real
    # Plan beforehand
    plan = finufft.Plan(1, (nxdirty, nydirty), eps=eps)
    plan.setpts(u.ravel(), v.ravel())
    res1 = plan.execute(ms.ravel()).real
    np.testing.assert_allclose(res0, res1)
    return res0


def vis2dirty_with_faceting(nfacets_x, nfacets_y, npix_x, npix_y, **kwargs):
    if npix_x % nfacets_x != 0:
        raise ValueError("nfacets_x needs to be divisor of npix_x.")
    if npix_y % nfacets_y != 0:
        raise ValueError("nfacets_y needs to be divisor of npix_y.")
    nx = npix_x // nfacets_x
    ny = npix_y // nfacets_y
    dtype = np.float32 if kwargs["vis"].dtype == np.complex64 else np.float64
    res = np.zeros((npix_x, npix_y), dtype)
    for xx, yy in product(range(nfacets_x), range(nfacets_y)):
        cx = ((0.5+xx)/nfacets_x - 0.5) * kwargs["pixsize_x"]*npix_x
        cy = ((0.5+yy)/nfacets_y - 0.5) * kwargs["pixsize_y"]*npix_y
        im = ng.experimental.vis2dirty(**kwargs,
                                       center_x=cx, center_y=cy,
                                       npix_x=nx, npix_y=ny)
        res[nx*xx:nx*(xx+1), ny*yy:ny*(yy+1)] = im
    return res


def dirty2vis_with_faceting(nfacets_x, nfacets_y, dirty, **kwargs):
    npix_x, npix_y = dirty.shape
    if npix_x % nfacets_x != 0:
        raise ValueError("nfacets_x needs to be divisor of npix_x.")
    if npix_y % nfacets_y != 0:
        raise ValueError("nfacets_y needs to be divisor of npix_y.")
    nx = npix_x // nfacets_x
    ny = npix_y // nfacets_y
    vis = None
    for xx, yy in product(range(nfacets_x), range(nfacets_y)):
        cx = ((0.5+xx)/nfacets_x - 0.5) * kwargs["pixsize_x"]*npix_x
        cy = ((0.5+yy)/nfacets_y - 0.5) * kwargs["pixsize_y"]*npix_y
        facet = dirty[nx*xx:nx*(xx+1), ny*yy:ny*(yy+1)]
        foo = ng.experimental.dirty2vis(**kwargs, dirty=facet,
                                        center_x=cx, center_y=cy)
        if vis is None:
            vis = foo
        else:
            vis += foo
    return vis


@pmp('nx', [(30, 3), (128, 2)])
@pmp('ny', [(128, 2), (250, 5)])
@pmp("nrow", (1, 2, 27))
@pmp("nchan", (1, 5))
@pmp("epsilon", (1e-1, 1e-3, 3e-5, 2e-13))
@pmp("singleprec", (True, False))
@pmp("wstacking", (True, False))
@pmp("use_wgt", (True, False))
@pmp("use_mask", (False, True))
@pmp("nthreads", (1, 2, 7))
def test_adjointness_ms2dirty(nx, ny, nrow, nchan, epsilon,
                              singleprec, wstacking, use_wgt, nthreads,
                              use_mask):
    (nxdirty, nxfacets), (nydirty, nyfacets) = nx, ny
    if singleprec and epsilon < 1e-6:
        pytest.skip()
    rng = np.random.default_rng(42)
    pixsizex = np.pi/180/60/nxdirty*0.2398
    pixsizey = np.pi/180/60/nxdirty
    f0 = 1e9
    freq = f0 + np.arange(nchan)*(f0/nchan)
    uvw = (rng.random((nrow, 3))-0.5)/(pixsizey*f0/SPEEDOFLIGHT)
    ms = rng.random((nrow, nchan))-0.5 + 1j*(rng.random((nrow, nchan))-0.5)
    wgt = rng.uniform(0.9, 1.1, (nrow, nchan)) if use_wgt else None
    mask = (rng.uniform(0, 1, (nrow, nchan)) > 0.5).astype(np.uint8) \
        if use_mask else None
    dirty = rng.random((nxdirty, nydirty))-0.5
    nu = nv = 0
    if singleprec:
        ms = ms.astype("c8")
        dirty = dirty.astype("f4")
        if wgt is not None:
            wgt = wgt.astype("f4")

    def check(d2, m2):
        ref = max(my_vdot(ms, ms).real, my_vdot(m2, m2).real,
                  my_vdot(dirty, dirty).real, my_vdot(d2, d2).real)
        tol = 3e-5*ref if singleprec else 2e-13*ref
        assert_allclose(my_vdot(ms, m2).real, my_vdot(d2, dirty), rtol=tol)

    dirty2 = ng.ms2dirty(uvw, freq, ms, wgt, nxdirty, nydirty, pixsizex,
                         pixsizey, nu, nv, epsilon, wstacking, nthreads, 0,
                         mask).astype("f8")
    ms2 = ng.dirty2ms(uvw, freq, dirty, wgt, pixsizex, pixsizey, nu, nv,
                      epsilon, wstacking, nthreads+1, 0, mask).astype("c16")
    check(dirty2, ms2)


    dirty2 = vis2dirty_with_faceting(nxfacets, nyfacets, uvw=uvw, freq=freq,
                                     vis=ms, wgt=wgt, npix_x=nxdirty,
                                     npix_y=nydirty, pixsize_x=pixsizex,
                                     pixsize_y=pixsizey, epsilon=epsilon,
                                     do_wgridding=wstacking, nthreads=nthreads,
                                     mask=mask).astype("f8")
    ms2 = dirty2vis_with_faceting(nxfacets, nyfacets, uvw=uvw, freq=freq,
                                  dirty=dirty, wgt=wgt, pixsize_x=pixsizex,
                                  pixsize_y=pixsizey, epsilon=epsilon,
                                  do_wgridding=wstacking, nthreads=nthreads+1,
                                  mask=mask).astype("c16")
    check(dirty2, ms2)


@pmp('nx', [(16, 2), (64, 4)])
@pmp('ny', [(64, 2)])
@pmp("nrow", (1, 2, 27))
@pmp("nchan", (1, 5))
@pmp("epsilon", (1e-2, 1e-3, 1e-4, 1e-7))
@pmp("singleprec", (False,))
@pmp("wstacking", (False, True))
@pmp("use_wgt", (True,))
@pmp("use_mask", (True,))
@pmp("nthreads", (1, 2, 7))
@pmp("fov", (0.001, 0.01, 0.1, 1., 20.))
def test_ms2dirty_against_wdft2(nx, ny, nrow, nchan, epsilon,
                                singleprec, wstacking, use_wgt, use_mask, fov,
                                nthreads):
    (nxdirty, nxfacets), (nydirty, nyfacets) = nx, ny
    if singleprec and epsilon < 1e-6:
        pytest.skip()
    rng = np.random.default_rng(42)
    pixsizex = fov*np.pi/180/nxdirty
    pixsizey = fov*np.pi/180/nydirty*1.1
    f0 = 1e9
    freq = f0 + np.arange(nchan)*(f0/nchan)
    uvw = (rng.random((nrow, 3))-0.5)/(pixsizex*f0/SPEEDOFLIGHT)
    ms = rng.random((nrow, nchan))-0.5 + 1j*(rng.random((nrow, nchan))-0.5)
    wgt = rng.uniform(0.9, 1.1, (nrow, 1)) if use_wgt else None
    mask = (rng.uniform(0, 1, (nrow, nchan)) > 0.5).astype(np.uint8) \
        if use_mask else None
    wgt = np.broadcast_to(wgt, (nrow, nchan)) if use_wgt else None
    nu = nv = 0
    if singleprec:
        ms = ms.astype("c8")
        if wgt is not None:
            wgt = wgt.astype("f4")
    dirty = ng.ms2dirty(uvw, freq, ms, wgt, nxdirty, nydirty, pixsizex,
                        pixsizey, nu, nv, epsilon, wstacking, nthreads,
                        0, mask).astype("f8")
    ref = explicit_gridder(uvw, freq, ms, wgt, nxdirty, nydirty, pixsizex,
                           pixsizey, wstacking, mask)
    assert_allclose(_l2error(dirty, ref), 0, atol=epsilon)

    dirty2 = vis2dirty_with_faceting(nxfacets, nyfacets, uvw=uvw, freq=freq,
                                     vis=ms, wgt=wgt, npix_x=nxdirty,
                                     npix_y=nydirty, pixsize_x=pixsizex,
                                     pixsize_y=pixsizey, epsilon=epsilon,
                                     do_wgridding=wstacking, nthreads=nthreads,
                                     mask=mask).astype("f8")
    assert_allclose(_l2error(dirty2, ref), 0, atol=epsilon)

    if wstacking or (not have_finufft):
        return
    dirty = with_finufft(uvw, freq, ms, wgt, nxdirty, nydirty, pixsizex,
                         pixsizey, mask, epsilon)
    assert_allclose(_l2error(dirty, ref), 0, atol=epsilon)
