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
# Copyright(C) 2020-2023 Max-Planck-Society

from itertools import product

import ducc0.wgridder as ng
import ducc0
from ducc0.misc import vdot
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
    return np.where(n>0, res/n, 0.)

def explicit_degridder(uvw, freq, dirty, wgt, xpixsize, ypixsize,
                     apply_w, mask):
    nxdirty, nydirty = dirty.shape
    x, y = np.meshgrid(*[-ss/2 + np.arange(ss) for ss in [nxdirty, nydirty]],
                       indexing='ij')
    x *= xpixsize
    y *= ypixsize
    res = np.zeros((uvw.shape[0], freq.shape[0]), dtype=np.complex128)
    eps = x**2+y**2
    if apply_w:
        nm1 = -eps/(np.sqrt(1.-eps)+1.)
        n = nm1+1
    else:
        nm1 = 0.
        n = 1.
    dirty2 = np.where(n>0, dirty/n, 0.)
    for row in range(uvw.shape[0]):
        for chan in range(freq.shape[0]):
            if mask is not None and mask[row, chan] == 0:
                continue
            phase = (-freq[chan]/SPEEDOFLIGHT *
                     (x*uvw[row, 0] + y*uvw[row, 1] - uvw[row, 2]*nm1))
            if wgt is None:
                res[row, chan] = np.sum(dirty2*np.exp(2j*np.pi*phase))
            else:
                res[row, chan] = np.sum((dirty2*wgt[row, chan]
                        * np.exp(2j*np.pi*phase)))
    return res


def with_finufft(uvw, freq, ms, wgt, nxdirty, nydirty, xpixsize, ypixsize,
                 mask, epsilon):
    u = np.fmod(np.outer(uvw[:, 0], freq)*(xpixsize/SPEEDOFLIGHT), 1.)*2*np.pi
    v = np.fmod(np.outer(uvw[:, 1], freq)*(ypixsize/SPEEDOFLIGHT), 1.)*2*np.pi
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


def vis2dirty_with_faceting(*, nfacets_x=1, nfacets_y=1, npix_x, npix_y,
                            center_x=0., center_y=0., dirty=None, **kwargs):
    import ducc0.wgridder.experimental as wgridder
    if dirty is None:
        dtype = np.float32 if kwargs["vis"].dtype == np.complex64 else np.float64
        dirty = np.zeros((npix_x, npix_y), dtype)
    else:
        if npix_x != 0 and npix_x != dirty.shape[0]:
            raise ValueError("bad npix_x")
        if npix_y != 0 and npix_y != dirty.shape[1]:
            raise ValueError("bad npix_y")
        npix_x, npix_y = dirty.shape

    istep = (npix_x+nfacets_x-1) // nfacets_x
    istep += istep % 2  # make even
    jstep = (npix_y+nfacets_y-1) // nfacets_y
    jstep += jstep % 2  # make even

    tdirty = None
    for i in range(nfacets_x):
        cx = center_x + kwargs["pixsize_x"]*0.5*((2*i+1)*istep-npix_x)
        imax = min((i+1)*istep, dirty.shape[0])
        for j in range(nfacets_y):
            cy = center_y + kwargs["pixsize_y"]*0.5*((2*j+1)*jstep-npix_y)
            jmax = min((j+1)*jstep, dirty.shape[1])
            if imax == (i+1)*istep and jmax == (j+1)*jstep:
                _ = wgridder.vis2dirty(**kwargs,
                                       center_x=cx, center_y=cy,
                                       npix_x=istep, npix_y=jstep,
                                       dirty=dirty[i*istep:imax, j*jstep:jmax])
            else:
                tdirty = wgridder.vis2dirty(
                    **kwargs, center_x=cx, center_y=cy,
                    npix_x=istep, npix_y=jstep, dirty=tdirty)
                dirty[i*istep:imax, j*jstep:jmax] = tdirty[:imax-i*istep, :jmax-j*jstep]
    return dirty


def dirty2vis_with_faceting(*, nfacets_x=1, nfacets_y=1, center_x=0., center_y=0.,
                            dirty, vis=None, **kwargs):
    import ducc0.wgridder.experimental as wgridder
    npix_x, npix_y = dirty.shape
    istep = (npix_x+nfacets_x-1) // nfacets_x
    istep += istep % 2  # make even
    jstep = (npix_y+nfacets_y-1) // nfacets_y
    jstep += jstep % 2  # make even

    tvis = None
    for i in range(nfacets_x):
        cx = center_x + kwargs["pixsize_x"]*0.5*((2*i+1)*istep-npix_x)
        imax = min((i+1)*istep, dirty.shape[0])
        for j in range(nfacets_y):
            cy = center_y + kwargs["pixsize_y"]*0.5*((2*j+1)*jstep-npix_y)
            jmax = min((j+1)*jstep, dirty.shape[1])
            if imax == (i+1)*istep and jmax == (j+1)*jstep:
                tdirty = dirty[i*istep:imax, j*jstep:jmax]
            else:
                tdirty = np.zeros((istep, jstep), dtype=dirty.dtype)
                tdirty[:imax-i*istep, :jmax-j*jstep] = dirty[i*istep:imax, j*jstep:jmax]

            if i == 0 and j == 0:
                vis = wgridder.dirty2vis(**kwargs, dirty=tdirty,
                                         center_x=cx, center_y=cy, vis=vis)
            else:
                tvis = wgridder.dirty2vis(**kwargs, dirty=tdirty,
                                          center_x=cx, center_y=cy, vis=tvis)
                vis += tvis
    return vis


@pmp('nx', [(2, 2), (32, 3), (128, 2)])
@pmp('ny', [(2, 2), (128, 2), (250, 5)])
@pmp("nrow", (1, 2, 27))
@pmp("nchan", (1, 5))
@pmp("epsilon", (1e-1, 3e-5, 2e-13))
@pmp("singleprec", (True, False))
@pmp("wstacking", (True, False))
@pmp("use_wgt", (True, False))
@pmp("use_mask", (False, True))
@pmp("nthreads", (1, 2))
@pmp("gpu", (False, True) if ng.experimental.sycl_active() else (False,))
def test_adjointness_ms2dirty(nx, ny, nrow, nchan, epsilon,
                              singleprec, wstacking, use_wgt, nthreads,
                              use_mask, gpu):
    (nxdirty, nxfacets), (nydirty, nyfacets) = nx, ny
    if singleprec and epsilon < 1e-6:
        pytest.skip()
    rng = np.random.default_rng(43)
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
        ref = max(vdot(ms, ms).real, vdot(m2, m2).real,
                  vdot(dirty, dirty).real, vdot(d2, d2).real)
        tol = 3e-5*ref if singleprec else 2e-13*ref
        assert_allclose(vdot(ms, m2).real, vdot(d2, dirty), rtol=3*tol)

    dirty2 = ng.ms2dirty(uvw, freq, ms, wgt, nxdirty, nydirty, pixsizex,
                         pixsizey, nu, nv, epsilon, wstacking, nthreads, 0,
                         mask).astype("f8")
    ms2 = ng.dirty2ms(uvw, freq, dirty, wgt, pixsizex, pixsizey, nu, nv,
                      epsilon, wstacking, nthreads, 0, mask).astype("c16")
    check(dirty2, ms2)

    dirty2 = vis2dirty_with_faceting(nfacets_x=nxfacets, nfacets_y=nyfacets,
                                     uvw=uvw, freq=freq,
                                     vis=ms, wgt=wgt, npix_x=nxdirty,
                                     npix_y=nydirty, pixsize_x=pixsizex,
                                     pixsize_y=pixsizey, epsilon=epsilon,
                                     do_wgridding=wstacking, nthreads=nthreads,
                                     mask=mask, gpu=gpu).astype("f8")
    ms2 = dirty2vis_with_faceting(nfacets_x=nxfacets, nfacets_y=nyfacets,
                                  uvw=uvw, freq=freq,
                                  dirty=dirty, wgt=wgt, pixsize_x=pixsizex,
                                  pixsize_y=pixsizey, epsilon=epsilon,
                                  do_wgridding=wstacking, nthreads=nthreads,
                                  mask=mask, gpu=gpu).astype("c16")
    check(dirty2, ms2)


@pmp('nx', [(2, 2), (6, 2), (18, 2), (66, 4)])
@pmp('ny', [(2, 2), (64, 2)])
@pmp("nrow", (1, 2, 27))
@pmp("nchan", (1, 5))
@pmp("epsilon", (1e-2, 1e-4, 1e-7))
@pmp("singleprec", (False,))
@pmp("wstacking", (False, True))
@pmp("use_wgt", (True,))
@pmp("use_mask", (True,))
@pmp("nthreads", (1, 2))
@pmp("fov", (0.001, 0.1, 20.))
@pmp("gpu", (False, True) if ng.experimental.sycl_active() else (False,))
def test_ms2dirty_against_wdft2(nx, ny, nrow, nchan, epsilon,
                                singleprec, wstacking, use_wgt, use_mask, fov,
                                nthreads, gpu):
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
    wgt = rng.uniform(0.9, 1.1, (nrow, nchan)) if use_wgt else None
    mask = (rng.uniform(0, 1, (nrow, nchan)) > 0.5).astype(np.uint8) \
        if use_mask else None
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
    assert_allclose(ducc0.misc.l2error(dirty, ref), 0, atol=epsilon)

    dirty2 = vis2dirty_with_faceting(nfacets_x=nxfacets, nfacets_y=nyfacets,
                                     uvw=uvw, freq=freq,
                                     vis=ms, wgt=wgt, npix_x=nxdirty,
                                     npix_y=nydirty, pixsize_x=pixsizex,
                                     pixsize_y=pixsizey, epsilon=epsilon,
                                     do_wgridding=wstacking, nthreads=nthreads,
                                     mask=mask,gpu=gpu).astype("f8")
    assert_allclose(ducc0.misc.l2error(dirty2, ref), 0, atol=epsilon)

    if wstacking or (not have_finufft):
        return
    dirty = with_finufft(uvw, freq, ms, wgt, nxdirty, nydirty, pixsizex,
                         pixsizey, mask, epsilon)
    assert_allclose(ducc0.misc.l2error(dirty, ref), 0, atol=epsilon)


@pmp('nxdirty', [2, 16, 64])
@pmp('nydirty', [2, 64])
@pmp("nrow", (1, 100))
@pmp("nchan", (1, 7))
@pmp("epsilon", list(10.**np.linspace(-2., -12., 20)))
@pmp("singleprec", (False,))
@pmp("wstacking", (True,))
@pmp("use_wgt", (True,))
@pmp("nthreads", (1, 10))
@pmp("fov", (10.,))
def test_ms2dirty_against_wdft3(nxdirty, nydirty, nrow, nchan, epsilon,
                                singleprec, wstacking, use_wgt, fov, nthreads):
    if singleprec and epsilon < 5e-5:
        return
    rng = np.random.default_rng(42)
    pixsizex = fov*np.pi/180/nxdirty
    pixsizey = fov*np.pi/180/nydirty*1.1
    speedoflight, f0 = 299792458., 1e9
    freq = f0 + np.arange(nchan)*(f0/nchan)
    uvw = (rng.random((nrow, 3))-0.5)/(pixsizex*f0/speedoflight)
    ms = rng.random((nrow, nchan))-0.5 + 1j*(rng.random((nrow, nchan))-0.5)
    wgt = rng.random((nrow, 1)) if use_wgt else None
    wgt = np.broadcast_to(wgt, (nrow, nchan)) if use_wgt else None
    if singleprec:
        ms = ms.astype("c8")
        if wgt is not None:
            wgt = wgt.astype("f4")
    try:
        dirty = ng.ms2dirty(
            uvw, freq, ms, wgt, nxdirty, nydirty, pixsizex,
            pixsizey, 0, 0, epsilon, wstacking, nthreads, 0).astype("f8")
    except:
        # no matching kernel was found
        pytest.skip()
    ref = explicit_gridder(uvw, freq, ms, wgt, nxdirty, nydirty, pixsizex,
                           pixsizey, wstacking, None)
    assert_allclose(ducc0.misc.l2error(dirty, ref), 0, atol=2*epsilon)
    x1 = explicit_degridder(uvw, freq, ref, wgt, pixsizex,
                           pixsizey, wstacking, None)
    x2 = ng.dirty2ms(uvw, freq, ref, wgt, pixsizex, pixsizey, 0, 0,
                      epsilon, wstacking, nthreads, 0).astype("c16")
    assert_allclose(ducc0.misc.l2error(x1,x2), 0, atol=epsilon)


@pmp('nx', [(2, 2), (30, 3), (128, 2)])
@pmp('ny', [(2, 2), (128, 2), (250, 5)])
@pmp("nrow", (1, 2, 27))
@pmp("nchan", (1, 5))
@pmp("epsilon", (1e-1, 3e-5, 2e-13))
@pmp("singleprec", (True, False))
@pmp("wstacking", (True, False))
@pmp("use_wgt", (True, False))
@pmp("use_mask", (False, True))
@pmp("nthreads", (1, 2))
def test_adjointness_ms2dirty_complex(nx, ny, nrow, nchan, epsilon,
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
    dirty = dirty +  1j*(rng.random((nxdirty, nydirty))-0.5)
    nu = nv = 0
    if singleprec:
        ms = ms.astype("c8")
        dirty = dirty.astype("c8")
        if wgt is not None:
            wgt = wgt.astype("f4")

    def check(d2, m2):
        ref = max(vdot(ms, ms).real, vdot(m2, m2).real,
                  vdot(dirty, dirty).real, vdot(d2, d2).real)
        tol = 3e-5*ref if singleprec else 2e-13*ref
        assert_allclose(vdot(ms, m2), vdot(d2, dirty), rtol=tol)

    dirty2 = ng.ms2dirty(uvw, freq, ms, wgt, nxdirty, nydirty, pixsizex,
                         pixsizey, nu, nv, epsilon, wstacking, nthreads, 0,
                         mask).astype("f8") \
            +1j * ng.ms2dirty(uvw, freq, -1j*ms, wgt, nxdirty, nydirty, pixsizex,
                         pixsizey, nu, nv, epsilon, wstacking, nthreads, 0,
                         mask).astype("f8")
    ms2 = ng.dirty2ms(uvw, freq, dirty.real, wgt, pixsizex, pixsizey, nu, nv,
                       epsilon, wstacking, nthreads, 0, mask).astype("c16") \
          +1j*ng.dirty2ms(uvw, freq, dirty.imag, wgt, pixsizex, pixsizey, nu, nv,
                       epsilon, wstacking, nthreads, 0, mask).astype("c16")
    check(dirty2, ms2)
